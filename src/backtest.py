#!/usr/bin/env python
"""
backtest.py - バックテストモジュール

機能:
- 訓練済みモデルからシグナルを生成
- vectorbtpro によるバックテスト実行
- パフォーマンス指標の計算と保存
"""

import os
import sys
import yaml
import json
import pickle
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union

# vectorbtpro - インストールチェック
try:
    import vectorbtpro as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    logging.warning("vectorbtpro がインストールされていません。バックテスト機能は無効です。")

# データ処理とモデル
import lightgbm as lgb
import catboost as ctb
from sklearn.preprocessing import StandardScaler

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'backtest.log'), mode='a')
    ]
)
logger = logging.getLogger('backtest')

# プロジェクトルートディレクトリ
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = ROOT_DIR / 'config'
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'
REPORTS_DIR = ROOT_DIR / 'reports'

# ログディレクトリ作成
os.makedirs(ROOT_DIR / 'logs', exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# SimplePortfolio class - vectorbtproの代替として使用
class SimplePortfolio:
    """シンプルなポートフォリオクラス (vectorbtpro代替用)"""
    
    def __init__(self, df, stats):
        """初期化
        
        Args:
            df: バックテスト結果のデータフレーム
            stats: 統計指標
        """
        self.df = df
        self._stats = stats
    
    def plot(self):
        """リターン曲線をプロット"""
        plt.figure(figsize=(12, 6))
        self.df['cumulative_returns'].plot()
        plt.title('Cumulative Returns')
        plt.grid(True, alpha=0.3)
        return plt.gcf()
    
    def plot_drawdowns(self):
        """ドローダウン曲線をプロット"""
        plt.figure(figsize=(12, 6))
        running_max = self.df['cumulative_returns'].cummax()
        drawdown = (self.df['cumulative_returns'] / running_max - 1) * 100
        drawdown.plot()
        plt.title('Drawdowns (%)')
        plt.grid(True, alpha=0.3)
        return plt.gcf()
    
    def stats(self):
        """統計指標を返す"""
        return self._stats
    
    def save(self, path):
        """ポートフォリオオブジェクトを保存"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

class BacktestRunner:
    """バックテストクラス"""
    
    def __init__(self, config_dir: Path = CONFIG_DIR):
        """初期化
        
        Args:
            config_dir: 設定ファイルディレクトリ
        """
        self.config_dir = config_dir
        self.intervals = self._load_intervals()
        self.fee_config = self._load_fee_config()
        self.mtf_config = self._load_mtf_config()
        self.model_dir = MODELS_DIR
        self.report_dir = REPORTS_DIR
        
        # VectorBT チェック
        if not VBT_AVAILABLE:
            logger.warning("vectorbtpro がインストールされていません。バックテスト機能は限定的になります。")
    
    def _load_intervals(self) -> List[str]:
        """時間枠設定を読み込む
        
        Returns:
            List[str]: 時間枠リスト
        """
        try:
            with open(self.config_dir / 'intervals.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config['intervals']
        except Exception as e:
            logger.error(f"時間枠設定の読み込みに失敗しました: {e}")
            raise
    
    def _load_fee_config(self) -> Dict:
        """手数料設定を読み込む
        
        Returns:
            Dict: 手数料設定
        """
        try:
            with open(self.config_dir / 'fees.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"手数料設定の読み込みに失敗しました: {e}")
            raise
    
    def _load_mtf_config(self) -> Dict:
        """MTF設定を読み込む
        
        Returns:
            Dict: MTF設定
        """
        try:
            mtf_path = self.config_dir / 'mtf.yaml'
            if not mtf_path.exists():
                logger.warning(f"MTF設定ファイルが見つかりません: {mtf_path}")
                return {"use_trend_filter": False}
                
            with open(mtf_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"MTF設定の読み込みに失敗しました: {e}")
            return {"use_trend_filter": False}
    
    def load_data(self, interval: str) -> pd.DataFrame:
        """特徴量を読み込む
        
        Args:
            interval: 時間枠
            
        Returns:
            pd.DataFrame: データフレーム
        """
        logger.info(f"データ読み込み開始: {interval}")
        
        # マージ済みデータを読み込む (price_close が必要)
        merged_path = DATA_DIR / 'features' / interval / 'merged.parquet'
        if not os.path.exists(merged_path):
            raise FileNotFoundError(f"マージ済みデータファイルが見つかりません: {merged_path}")
        
        df = pd.read_parquet(merged_path)
        logger.info(f"マージ済みデータ読み込み完了: {merged_path}, 形状: {df.shape}")
        
        # 特徴量読み込み
        features_path = DATA_DIR / 'features' / interval / 'X.parquet'
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"特徴量ファイルが見つかりません: {features_path}")
        
        X = pd.read_parquet(features_path)
        logger.info(f"特徴量読み込み完了: {features_path}, 形状: {X.shape}")
        
        # インデックスが同じかチェック
        if not X.index.equals(df.index):
            logger.warning("特徴量とデータのインデックスが一致しません。共通部分のみ使用します。")
            common_idx = X.index.intersection(df.index)
            df = df.loc[common_idx]
            X = X.loc[common_idx]
            logger.info(f"インデックス調整後: df形状: {df.shape}, X形状: {X.shape}")
        
        # 特徴量を本体データフレームに追加 (一度に追加して断片化を防ぐ)
        # Use pandas.concat instead of iteratively adding columns to avoid fragmentation
        df = pd.concat([df, X], axis=1)
        
        # 必要な価格データの確認
        required_cols = ['price_close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"必要な価格カラムが見つかりません: {required_cols}")
        
        # 欠損値の確認と処理
        if df[required_cols].isna().any().any():
            logger.warning(f"価格データに欠損値があります。補完します。")
            df[required_cols] = df[required_cols].ffill()
        
        return df
    
    def load_model(self, interval: str, model_type: str = 'lightgbm', model_date: Optional[str] = None) -> Tuple[Any, List[str]]:
        """モデルを読み込む
        
        Args:
            interval: 時間枠
            model_type: モデル種類 ('lightgbm' または 'catboost')
            model_date: モデル日付 (省略時は最新)
            
        Returns:
            Tuple[Any, List[str]]: モデルと特徴量名のリスト
        """
        model_dir = self.model_dir / interval / model_type
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"モデルディレクトリが見つかりません: {model_dir}")
        
        # モデルファイルを検索
        if model_type == 'lightgbm':
            model_files = list(model_dir.glob("model_fold1_*.txt"))
        elif model_type == 'catboost':
            model_files = list(model_dir.glob("model_fold1_*.cbm"))
        else:
            raise ValueError(f"サポートされていないモデル種類です: {model_type}")
        
        if not model_files:
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_dir}")
        
        # モデル日付が指定されていない場合は最新を使用
        if model_date is None:
            model_file = sorted(model_files)[-1]
        else:
            matching_files = [f for f in model_files if model_date in f.name]
            if not matching_files:
                raise FileNotFoundError(f"指定日付 {model_date} のモデルが見つかりません")
            model_file = matching_files[0]
        
        logger.info(f"モデルファイル読み込み: {model_file}")
        
        # モデル情報ファイルの検索
        timestamp = model_file.stem.split('_')[-1]
        model_info_file = self.model_dir / f"model_info_{timestamp}.json"
        
        if not os.path.exists(model_info_file):
            logger.warning(f"モデル情報ファイルが見つかりません: {model_info_file}")
            
            # 特徴量リストをデータから抽出
            feature_cols = self.load_data(interval).columns.tolist()
            feature_cols = [col for col in feature_cols if col.endswith('_zscore')]
        else:
            # モデル情報から特徴量リストを取得
            with open(model_info_file, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            feature_cols = model_info.get('feature_cols', [])
        
        # モデル読み込み
        if model_type == 'lightgbm':
            model = lgb.Booster(model_file=str(model_file))
        elif model_type == 'catboost':
            if 'Classification' in model_file.stem:
                model = ctb.CatBoostClassifier()
            else:
                model = ctb.CatBoostRegressor()
            model.load_model(str(model_file))
        
        return model, feature_cols
    
    def generate_signals(self, df: pd.DataFrame, model: Any, feature_cols: List[str],
                        threshold: float = 0.55) -> pd.DataFrame:
        """モデルからシグナルを生成
        
        Args:
            df: 入力データフレーム
            model: モデル
            feature_cols: モデルで使用する特徴量
            threshold: シグナル閾値
            
        Returns:
            pd.DataFrame: シグナル付きデータフレーム
        """
        logger.info(f"シグナル生成開始: 特徴量数={len(feature_cols)}, 閾値={threshold}")
        
        # データフレームをコピー
        result = df.copy()
        
        # 日足トレンドフィルター設定の確認
        use_trend_filter = self.mtf_config.get('use_trend_filter', False)
        if use_trend_filter:
            logger.info("日足トレンドフィルターが有効です")
            # 日足トレンドデータの読み込み
            try:
                trend_path = DATA_DIR / 'features' / '1d' / 'trend_flag.parquet'
                if not os.path.exists(trend_path):
                    logger.warning(f"日足トレンドフラグファイルが見つかりません: {trend_path}")
                    logger.warning("日足トレンドフィルターは使用できません")
                else:
                    # トレンドフラグ読み込み
                    trend_flag = pd.read_parquet(trend_path)
                    
                    # インデックスのタイムゾーンを合わせる
                    # まず両方のデータフレームのタイムゾーン情報をログ
                    logger.info(f"シグナルインデックスのタイムゾーン: {getattr(result.index, 'tz', None)}")
                    logger.info(f"トレンドフラグインデックスのタイムゾーン: {getattr(trend_flag.index, 'tz', None)}")
                    
                    # 両方ともタイムゾーンを削除する方法で統一
                    if hasattr(result.index, 'tz_localize') and result.index.tz is not None:
                        result.index = result.index.tz_localize(None)
                    
                    if hasattr(trend_flag.index, 'tz_localize') and trend_flag.index.tz is not None:
                        trend_flag.index = trend_flag.index.tz_localize(None)
                    
                    # シグナルのインデックスに合わせてリサンプリング (直前の値を転送)
                    trend_flag = trend_flag.reindex(result.index, method='ffill')
                    
                    # トレンドフラグをデータフレームに追加
                    result['trend_flag'] = trend_flag
                    
                    logger.info(f"日足トレンドフラグ読み込み完了: {len(trend_flag)} レコード")
                    sum_value = trend_flag['trend_flag'].sum()
                    len_value = len(trend_flag)
                    mean_pct = (sum_value / len_value) * 100 if len_value > 0 else 0
                    logger.info(f"上昇トレンド期間: {sum_value}/{len_value} ({mean_pct:.1f}%)")
            except Exception as e:
                logger.error(f"日足トレンドフラグ読み込みエラー: {str(e)}")
                logger.warning("日足トレンドフィルターは使用できません")
                use_trend_filter = False
        
        # 特徴量の抽出と前処理
        features = []
        for col in feature_cols:
            if col in result.columns:
                features.append(col)
            else:
                logger.warning(f"特徴量 {col} がデータフレームに見つかりません")
        
        if not features:
            raise ValueError("有効な特徴量がデータフレームに見つかりません")
        
        # 特徴量マトリックスの作成
        X = result[features].values
        
        # モデルで予測
        if isinstance(model, lgb.Booster):
            # LightGBM
            raw_pred = model.predict(X)
            
            # 出力次元数を確認 (多クラスの場合は形状が (n_samples, n_classes) になる)
            if len(raw_pred.shape) > 1 and raw_pred.shape[1] == 3:
                logger.info("3クラス分類モデルを検出しました。確率を適切に変換します。")
                # -1, 0, 1 のそれぞれに対応する確率
                neg_proba = raw_pred[:, 0]  # -1 クラスの確率
                neutral_proba = raw_pred[:, 1]  # 0 クラスの確率
                pos_proba = raw_pred[:, 2]  # 1 クラスの確率
                
                # 正規化された「方向」確率を計算：-1 から 1 の範囲
                # (pos_proba - neg_proba) は、正の方向と負の方向の確率の差
                y_pred_proba = pos_proba - neg_proba
                
                # データフレームに詳細な確率を追加
                result['neg_proba'] = neg_proba
                result['neutral_proba'] = neutral_proba
                result['pos_proba'] = pos_proba
            else:
                # バイナリ分類の場合
                y_pred_proba = raw_pred
        elif isinstance(model, (ctb.CatBoostClassifier, ctb.CatBoostRegressor)):
            # CatBoost
            if isinstance(model, ctb.CatBoostClassifier):
                raw_pred = model.predict_proba(X)
                if raw_pred.shape[1] == 3:  # 3クラス分類
                    neg_proba = raw_pred[:, 0]  # -1 クラス
                    neutral_proba = raw_pred[:, 1]  # 0 クラス
                    pos_proba = raw_pred[:, 2]  # 1 クラス
                    y_pred_proba = pos_proba - neg_proba
                    
                    # データフレームに詳細な確率を追加
                    result['neg_proba'] = neg_proba
                    result['neutral_proba'] = neutral_proba
                    result['pos_proba'] = pos_proba
                else:
                    y_pred_proba = raw_pred[:, 1]
            else:
                y_pred_proba = model.predict(X)
        else:
            raise TypeError(f"サポートされていないモデル種類です: {type(model)}")
        
        # 確率をデータフレームに追加
        result['pred_proba'] = y_pred_proba
        
        # シグナル生成: 閾値以上ならロング、(1-閾値)以下ならショート
        result['raw_signal'] = np.where(
            result['pred_proba'] > threshold, 1,     # ロングシグナル
            np.where(
                result['pred_proba'] < (1 - threshold), -1,   # ショートシグナル
                0   # ニュートラル
            )
        )
        
        # 日足トレンドフィルターの適用
        if use_trend_filter and 'trend_flag' in result.columns:
            logger.info("日足トレンドフィルターを適用します")
            
            # 前日のトレンドフラグが1のときのみロングエントリー
            entries = (result['raw_signal'] == 1) & (result['trend_flag'].shift(1) == 1)
            
            # シグナルが-1の場合か、トレンドフラグが0の場合にイグジット
            exits = (result['raw_signal'] == -1) | (result['trend_flag'].shift(1) == 0)
            
            # フィルタリングしたシグナルを作成
            result['signal'] = np.where(entries, 1, np.where(exits, -1, 0))
            
            # フィルタリング前後の統計
            before_counts = result['raw_signal'].value_counts()
            after_counts = result['signal'].value_counts()
            
            logger.info("フィルタリング前のシグナル:")
            for signal_value, count in before_counts.items():
                signal_name = {1: 'ロング', -1: 'ショート', 0: 'ニュートラル'}.get(signal_value, str(signal_value))
                logger.info(f"  {signal_name}シグナル: {count} ({count/len(result):.2%})")
                
            logger.info("フィルタリング後のシグナル:")
            for signal_value, count in after_counts.items():
                signal_name = {1: 'ロング', -1: 'ショート', 0: 'ニュートラル'}.get(signal_value, str(signal_value))
                logger.info(f"  {signal_name}シグナル: {count} ({count/len(result):.2%})")
        else:
            # フィルタリングなしの場合はraw_signalをそのまま使用
            result['signal'] = result['raw_signal']
        
        # 集計
        signal_counts = result['signal'].value_counts()
        total_signals = len(result)
        
        for signal_value, count in signal_counts.items():
            signal_name = {1: 'ロング', -1: 'ショート', 0: 'ニュートラル'}.get(signal_value, str(signal_value))
            logger.info(f"{signal_name}シグナル: {count} ({count/total_signals:.2%})")
        
        return result
    
    def run_backtest(self, df: pd.DataFrame, fees: Optional[Dict] = None) -> Tuple[Any, pd.DataFrame]:
        """バックテストを実行
        
        Args:
            df: シグナル付きデータフレーム
            fees: 手数料設定 (省略時は fees.yaml から読み込み)
            
        Returns:
            Tuple[Any, pd.DataFrame]: パフォーマンス結果
        """
        logger.info("バックテスト実行開始")
        
        # 手数料設定
        if fees is None:
            fees_config = self.fee_config
            fees = {
                'taker_fee': fees_config['futures']['taker_fee'],
                'maker_fee': fees_config['futures']['maker_fee'],
                'slippage': fees_config['futures']['slippage']
            }
        
        # コスト計算に使用する手数料（通常はテイカー手数料）
        trade_fee = fees['taker_fee']
        slippage = fees['slippage']
        
        # 総コスト
        total_fee = trade_fee + slippage
        
        logger.info(f"バックテスト設定: 手数料={trade_fee:.6f}, スリッページ={slippage:.6f}, 総コスト={total_fee:.6f}")
        
        # 価格とシグナルの抽出
        price = df['price_close']
        signal = df['signal']
        
        # VectorBTが利用可能な場合は、その機能を使用
        if VBT_AVAILABLE:
            # シグナルからエントリー・イグジットを生成
            entries = signal == 1    # ロングエントリー
            exits = signal == -1     # ショートエントリー = ロングイグジット
            
            # バックテスト実行
            pf = vbt.Portfolio.from_signals(
                price,
                entries,
                exits,
                fees=total_fee,
                slippage=0.0,   # 既にスリッページを含んでいるので重複させない
                init_cash=100_000,
                freq='infer'
            )
            
            # パフォーマンス統計
            stats = pf.stats()
            
            # キーメトリクスをログ出力
            logger.info(f"総リターン: {stats['Total Return [%]']}%")
            logger.info(f"年間リターン: {stats['Annual Return [%]']}%")
            logger.info(f"Sharpe比率: {stats['Sharpe Ratio']}")
            logger.info(f"Sortino比率: {stats['Sortino Ratio']}")
            logger.info(f"CAGR: {stats['CAGR [%]']}%")
            logger.info(f"最大ドローダウン: {stats['Max Drawdown [%]']}%")
            
            # MAR比率（CAGR / 最大ドローダウン）の計算
            cagr = stats['CAGR [%]'] / 100  # パーセントから小数に変換
            max_dd = stats['Max Drawdown [%]'] / 100  # パーセントから小数に変換
            mar_ratio = abs(cagr / max_dd) if max_dd != 0 else float('inf')
            logger.info(f"MAR比率 (CAGR/MaxDD): {mar_ratio:.4f}")
            
            # 合格基準の確認
            meets_criteria = (
                stats['Sharpe Ratio'] > 1.0 and
                mar_ratio >= 0.5 and
                stats['Max Drawdown [%]'] < 25.0
            )
            
            logger.info(f"合格基準を満たしている: {meets_criteria}")
            logger.info(f"- Sharpe > 1: {'✓' if stats['Sharpe Ratio'] > 1.0 else '✗'} ({stats['Sharpe Ratio']:.2f})")
            logger.info(f"- MAR ≥ 0.5: {'✓' if mar_ratio >= 0.5 else '✗'} ({mar_ratio:.2f})")
            logger.info(f"- MaxDD < 25%: {'✓' if stats['Max Drawdown [%]'] < 25.0 else '✗'} ({stats['Max Drawdown [%]']:.2f}%)")
            
            return pf, stats
        
        else:
            # シンプルなバックテスト実装（vectorbtproなしの場合）
            logger.info("vectorbtproが使用できないため、シンプルなバックテストを実行します")
            
            # 基本的なバックテスト計算
            df_backtest = pd.DataFrame({
                'price': price,
                'signal': signal
            })
            
            # ポジションを計算 (0, 1, -1)
            # Use ffill() instead of fillna(method='ffill') to avoid deprecation warning
            position_series = signal.replace(0, np.nan)
            df_backtest['position'] = position_series.ffill().fillna(0)
            
            # リターン計算
            df_backtest['returns'] = df_backtest['price'].pct_change()
            
            # 戦略リターン計算
            df_backtest['strategy_returns'] = df_backtest['position'].shift(1) * df_backtest['returns']
            
            # 手数料を考慮
            df_backtest['position_change'] = df_backtest['position'].diff().fillna(0)
            df_backtest['fees'] = np.abs(df_backtest['position_change']) * total_fee
            df_backtest['strategy_returns_after_fees'] = df_backtest['strategy_returns'] - df_backtest['fees']
            
            # 累積リターン
            df_backtest['cumulative_returns'] = (1 + df_backtest['strategy_returns_after_fees']).cumprod()
            
            # 簡易統計の計算
            total_return = df_backtest['cumulative_returns'].iloc[-1] - 1
            
            # 日次リターンに変換して年率リターン計算
            daily_returns = df_backtest['strategy_returns_after_fees'].resample('D').sum()
            annual_return = daily_returns.mean() * 252
            annual_volatility = daily_returns.std() * np.sqrt(252)
            sharpe = annual_return / annual_volatility if annual_volatility != 0 else 0
            
            # ドローダウン計算
            cumulative_returns = df_backtest['cumulative_returns']
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max - 1) * 100
            max_drawdown = drawdown.min()
            
            # 結果をSeriesとして返す
            stats = pd.Series({
                'Total Return [%]': total_return * 100,
                'Annual Return [%]': annual_return * 100,
                'Sharpe Ratio': sharpe,
                'Max Drawdown [%]': max_drawdown,
                'CAGR [%]': ((1 + total_return) ** (252 / len(df_backtest)) - 1) * 100,
            })
            
            # キーメトリクスをログ出力
            logger.info(f"総リターン: {stats['Total Return [%]']:.2f}%")
            logger.info(f"年間リターン: {stats['Annual Return [%]']:.2f}%")
            logger.info(f"Sharpe比率: {stats['Sharpe Ratio']:.2f}")
            logger.info(f"CAGR: {stats['CAGR [%]']:.2f}%")
            logger.info(f"最大ドローダウン: {stats['Max Drawdown [%]']:.2f}%")
            
            # MAR比率 (CAGR / MaxDD)
            cagr = stats['CAGR [%]'] / 100
            max_dd = abs(stats['Max Drawdown [%]']) / 100
            mar_ratio = abs(cagr / max_dd) if max_dd != 0 else float('inf')
            stats['MAR Ratio'] = mar_ratio
            logger.info(f"MAR比率 (CAGR/MaxDD): {mar_ratio:.4f}")
            
            # 合格基準の確認
            meets_criteria = (
                stats['Sharpe Ratio'] > 1.0 and
                mar_ratio >= 0.5 and
                stats['Max Drawdown [%]'] > -25.0
            )
            stats['Meets Criteria'] = meets_criteria
            
            logger.info(f"合格基準を満たしている: {meets_criteria}")
            logger.info(f"- Sharpe > 1: {'✓' if stats['Sharpe Ratio'] > 1.0 else '✗'} ({stats['Sharpe Ratio']:.2f})")
            logger.info(f"- MAR ≥ 0.5: {'✓' if mar_ratio >= 0.5 else '✗'} ({mar_ratio:.2f})")
            logger.info(f"- MaxDD < 25%: {'✓' if stats['Max Drawdown [%]'] > -25.0 else '✗'} ({stats['Max Drawdown [%]']:.2f}%)")
            
            # SimplePortfolioインスタンスを作成
            pf = SimplePortfolio(df_backtest, stats)
            
            return pf, stats
    
    def save_backtest_results(self, interval: str, portfolio: Any, stats: pd.Series, 
                             df: pd.DataFrame, model_type: str) -> Dict:
        """バックテスト結果を保存
        
        Args:
            interval: 時間枠
            portfolio: バックテストのポートフォリオオブジェクト
            stats: 統計指標
            df: シグナル付きデータフレーム
            model_type: モデル種類
            
        Returns:
            Dict: 保存先情報
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.report_dir / interval / model_type / 'backtest' / timestamp
        
        # ディレクトリ作成
        os.makedirs(report_dir, exist_ok=True)
        
        logger.info(f"バックテスト結果を保存: {report_dir}")
        
        # 統計を JSON 形式で保存
        stats_dict = stats.to_dict()
        
        # MAR比率の追加
        cagr = stats['CAGR [%]'] / 100  # パーセントから小数に変換
        max_dd = stats['Max Drawdown [%]'] / 100  # パーセントから小数に変換
        mar_ratio = abs(cagr / max_dd) if max_dd != 0 else float('inf')
        stats_dict['MAR Ratio'] = mar_ratio
        
        # 合格基準の追加
        stats_dict['Meets Criteria'] = (
            stats['Sharpe Ratio'] > 1.0 and
            mar_ratio >= 0.5 and
            stats['Max Drawdown [%]'] < 25.0
        )
        
        with open(report_dir / 'stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2, default=str)
        
        # ポートフォリオを保存
        portfolio.save(report_dir / 'portfolio.pkl')
        
        # シグナルと価格データを保存
        signals_df = df[['price_close', 'pred_proba', 'signal']].copy()
        signals_df.to_parquet(report_dir / 'signals.parquet')
        
        # チャートを生成
        self._plot_backtest_results(portfolio, signals_df, report_dir)
        
        return {
            'report_dir': str(report_dir),
            'stats_file': str(report_dir / 'stats.json'),
            'portfolio_file': str(report_dir / 'portfolio.pkl'),
            'signals_file': str(report_dir / 'signals.parquet'),
            'timestamp': timestamp
        }
    
    def _plot_backtest_results(self, portfolio: Any, signals_df: pd.DataFrame, 
                              report_dir: Path) -> None:
        """バックテスト結果をプロット
        
        Args:
            portfolio: バックテストのポートフォリオオブジェクト
            signals_df: シグナルデータフレーム
            report_dir: 結果保存ディレクトリ
        """
        # リターンチャート
        plt.figure(figsize=(12, 6))
        portfolio.plot()
        plt.title('Cumulative Returns')
        plt.tight_layout()
        plt.savefig(report_dir / 'returns.png')
        plt.close()
        
        # ドローダウンチャート
        plt.figure(figsize=(12, 6))
        portfolio.plot_drawdowns()
        plt.title('Drawdowns')
        plt.tight_layout()
        plt.savefig(report_dir / 'drawdowns.png')
        plt.close()
        
        # 価格とシグナル
        plt.figure(figsize=(12, 8))
        
        # 上のサブプロット: 価格
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(signals_df.index, signals_df['price_close'], label='Price')
        
        # エントリーポイントのマーク
        long_entries = signals_df[signals_df['signal'] == 1]
        short_entries = signals_df[signals_df['signal'] == -1]
        
        ax1.scatter(long_entries.index, long_entries['price_close'], 
                   marker='^', color='green', s=50, label='Long')
        ax1.scatter(short_entries.index, short_entries['price_close'], 
                   marker='v', color='red', s=50, label='Short')
        
        ax1.set_title('Price and Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 下のサブプロット: 予測確率
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(signals_df.index, signals_df['pred_proba'], label='Prediction Probability')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # シグナル閾値のライン
        threshold = 0.55  # デフォルト閾値
        ax2.axhline(y=threshold, color='green', linestyle='--', alpha=0.5, label=f'Long Threshold ({threshold})')
        ax2.axhline(y=1-threshold, color='red', linestyle='--', alpha=0.5, label=f'Short Threshold ({1-threshold})')
        
        ax2.set_title('Prediction Probability')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(report_dir / 'signals.png')
        plt.close()
    
    def run_all_backtests(self, model_type: str = 'lightgbm', threshold: float = 0.55,
                         model_date: Optional[str] = None) -> Dict:
        """すべての時間枠でバックテストを実行
        
        Args:
            model_type: モデル種類 ('lightgbm' または 'catboost')
            threshold: シグナル閾値
            model_date: モデル日付 (省略時は最新)
            
        Returns:
            Dict: バックテスト結果
        """
        results = {}
        
        for interval in self.intervals:
            try:
                logger.info(f"時間枠 {interval} のバックテスト開始")
                
                # データ読み込み
                df = self.load_data(interval)
                
                # モデル読み込み
                model, feature_cols = self.load_model(interval, model_type, model_date)
                
                # シグナル生成
                df_with_signals = self.generate_signals(df, model, feature_cols, threshold)
                
                # バックテスト実行
                portfolio, stats = self.run_backtest(df_with_signals)
                
                # 結果保存
                save_info = self.save_backtest_results(interval, portfolio, stats, df_with_signals, model_type)
                
                results[interval] = {
                    'portfolio': portfolio,
                    'stats': stats.to_dict(),
                    'save_info': save_info
                }
                
                logger.info(f"時間枠 {interval} のバックテスト完了")
                
            except Exception as e:
                logger.error(f"時間枠 {interval} のバックテスト中にエラー: {str(e)}")
                continue
        
        return results

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='バックテスト実行')
    parser.add_argument('--interval', type=str, help='時間枠 (例: 15m, 2h)')
    parser.add_argument('--model_type', type=str, default='lightgbm', 
                       choices=['lightgbm', 'catboost'], help='モデル種類')
    parser.add_argument('--threshold', type=float, default=0.55, help='シグナル閾値')
    parser.add_argument('--model_date', type=str, help='モデル日付 (例: 20250503)')
    parser.add_argument('--all', action='store_true', help='すべての時間枠でバックテスト')
    return parser.parse_args()

def main():
    """メイン関数"""
    try:
        # ログディレクトリ作成
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'), exist_ok=True)
        
        args = parse_args()
        
        # vectorbtpro チェック
        if not VBT_AVAILABLE:
            logger.warning("vectorbtpro がインストールされていません。シンプルなバックテスト実装を使用します。")
        
        runner = BacktestRunner()
        
        if args.all:
            # すべての時間枠でバックテスト
            results = runner.run_all_backtests(args.model_type, args.threshold, args.model_date)
        elif args.interval:
            # 単一時間枠でバックテスト
            df = runner.load_data(args.interval)
            model, feature_cols = runner.load_model(args.interval, args.model_type, args.model_date)
            df_with_signals = runner.generate_signals(df, model, feature_cols, args.threshold)
            portfolio, stats = runner.run_backtest(df_with_signals)
            runner.save_backtest_results(args.interval, portfolio, stats, df_with_signals, args.model_type)
        else:
            # デフォルトでは最初の時間枠を使用
            interval = runner.intervals[0]
            logger.info(f"時間枠が指定されていないため、デフォルト値 {interval} を使用します")
            df = runner.load_data(interval)
            model, feature_cols = runner.load_model(interval, args.model_type, args.model_date)
            df_with_signals = runner.generate_signals(df, model, feature_cols, args.threshold)
            portfolio, stats = runner.run_backtest(df_with_signals)
            runner.save_backtest_results(interval, portfolio, stats, df_with_signals, args.model_type)
        
        return 0
    
    except Exception as e:
        logger.error(f"実行エラー: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
