#!/usr/bin/env python
"""
feature.py - 特徴量生成モジュール

機能:
- 各データソース (price, oi, funding, etc.) から特徴量を生成
- 各特徴量を rolling Z-score で正規化
- 各時間枠で特徴量を計算し保存
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'feature.log'), mode='a')
    ]
)
logger = logging.getLogger('feature')

# プロジェクトルートディレクトリ
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = ROOT_DIR / 'config'
DATA_DIR = ROOT_DIR / 'data'

# ログディレクトリ作成
os.makedirs(ROOT_DIR / 'logs', exist_ok=True)

class FeatureBuilder:
    """特徴量生成クラス"""
    
    def __init__(self, config_dir: Path = CONFIG_DIR):
        """初期化
        
        Args:
            config_dir: 設定ファイルディレクトリ
        """
        self.config_dir = config_dir
        self.intervals = self._load_intervals()
    
    def _load_intervals(self) -> List[str]:
        """時間枠設定を読み込む
        
        Returns:
            List[str]: 時間枠リスト
        """
        try:
            with open(self.config_dir / 'intervals.yaml', 'r') as f:
                config = yaml.safe_load(f)
            return config['intervals']
        except Exception as e:
            logger.error(f"時間枠設定の読み込みに失敗しました: {e}")
            raise
    
    def load_merged_data(self, interval: str) -> pd.DataFrame:
        """マージされたデータを読み込む
        
        Args:
            interval: 時間枠
            
        Returns:
            pd.DataFrame: 読み込んだデータフレーム
        """
        merged_path = DATA_DIR / 'features' / interval / 'merged.parquet'
        
        # マージされたデータがない場合は生成
        if not os.path.exists(merged_path):
            logger.info(f"マージされたデータが見つかりません。生成します: {interval}")
            self.merge_interval(interval)
        
        return pd.read_parquet(merged_path)
    
    def merge_interval(self, interval: str) -> pd.DataFrame:
        """時間枠ごとのデータをマージ
        
        Args:
            interval: 時間枠
            
        Returns:
            pd.DataFrame: マージされたデータフレーム
        """
        logger.info(f"データマージ開始: {interval}")
        
        # 生データディレクトリ
        raw_dir = DATA_DIR / 'raw' / interval
        
        # 生データファイル一覧
        endpoint_files = [f for f in os.listdir(raw_dir) if f.endswith('.parquet')]
        
        if not endpoint_files:
            raise FileNotFoundError(f"時間枠 {interval} の生データが見つかりません")
        
        # 各エンドポイントのデータを読み込む
        dfs = {}
        for filename in endpoint_files:
            endpoint = filename.split('.')[0]  # 拡張子を除いたファイル名
            file_path = raw_dir / filename
            logger.info(f"読み込み: {file_path}")
            
            try:
                dfs[endpoint] = pd.read_parquet(file_path)
            except Exception as e:
                logger.error(f"ファイル読み込みエラー: {file_path}, {str(e)}")
                continue
        
        if not dfs:
            raise ValueError(f"時間枠 {interval} の読み込み可能なデータがありません")
        
        # データをマージ
        merged = pd.concat(dfs.values(), axis=1)
        
        # インデックスでソート
        merged = merged.sort_index()
        
        # 前方補完で欠損値を埋める
        merged = merged.ffill()
        
        # 残りの欠損値を削除
        merged = merged.dropna()
        
        # 特徴量ディレクトリに保存
        features_dir = DATA_DIR / 'features' / interval
        os.makedirs(features_dir, exist_ok=True)
        
        output_path = features_dir / 'merged.parquet'
        merged.to_parquet(output_path)
        
        logger.info(f"マージ完了: {output_path}, レコード数: {len(merged)}")
        
        return merged
    
    def zscore_normalize(self, series: pd.Series, window: int = 100) -> pd.Series:
        """時系列を rolling Z-score で正規化
        
        Args:
            series: 入力時系列
            window: ローリングウィンドウサイズ
            
        Returns:
            pd.Series: 正規化された時系列
        """
        # 無限大や欠損値を含む行を事前処理
        clean_series = series.replace([np.inf, -np.inf], np.nan)
        
        # 移動平均と標準偏差を計算
        mean = clean_series.rolling(window=window, min_periods=window//2).mean()
        std = clean_series.rolling(window=window, min_periods=window//2).std()
        
        # 0除算を防ぐ
        std = std.replace(0, np.nan)
        
        # Z-score 計算
        z_score = (clean_series - mean) / std
        
        # 外れ値を切り詰め (-3 から 3 の範囲に)
        z_score = z_score.clip(-3, 3)
        
        # 欠損値を 0 で補完
        z_score = z_score.fillna(0)
        
        return z_score
    
    def build_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """価格関連の特徴量を生成
        
        Args:
            df: 入力データフレーム
            
        Returns:
            pd.DataFrame: 特徴量を追加したデータフレーム
        """
        price_cols = [col for col in df.columns if col.startswith('price_')]
        
        if not price_cols or 'price_close' not in df.columns:
            logger.warning("価格データが見つかりません")
            return df
        
        # 基本的な技術指標
        
        # リターン
        for period in [1, 3, 6, 12, 24]:
            df[f'price_return_{period}'] = df['price_close'].pct_change(period)
        
        # ボラティリティ
        for window in [12, 24, 48]:
            df[f'price_volatility_{window}'] = df['price_close'].pct_change().rolling(window=window).std()
        
        # 移動平均
        for window in [8, 24, 48, 96]:
            df[f'price_ma_{window}'] = df['price_close'].rolling(window=window).mean()
            
            # 移動平均との乖離率
            df[f'price_ma_deviation_{window}'] = (df['price_close'] / df[f'price_ma_{window}'] - 1)
        
        # RSI
        for window in [14, 28]:
            delta = df['price_close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            df[f'price_rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # ATR (Average True Range)
        if all(col in df.columns for col in ['price_high', 'price_low']):
            for window in [14, 28]:
                tr1 = df['price_high'] - df['price_low']
                tr2 = (df['price_high'] - df['price_close'].shift()).abs()
                tr3 = (df['price_low'] - df['price_close'].shift()).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df[f'price_atr_{window}'] = tr.rolling(window=window).mean()
                
                # ATR 比率 (ATR / 価格)
                df[f'price_atr_ratio_{window}'] = df[f'price_atr_{window}'] / df['price_close']
        
        # MACD
        ema12 = df['price_close'].ewm(span=12, adjust=False).mean()
        ema26 = df['price_close'].ewm(span=26, adjust=False).mean()
        df['price_macd_line'] = ema12 - ema26
        df['price_macd_signal'] = df['price_macd_line'].ewm(span=9, adjust=False).mean()
        df['price_macd_hist'] = df['price_macd_line'] - df['price_macd_signal']
        
        # 全特徴量を Z-score 正規化
        feature_cols = [col for col in df.columns if col.startswith('price_') and col not in price_cols]
        
        for col in feature_cols:
            df[f'{col}_zscore'] = self.zscore_normalize(df[col])
        
        logger.info(f"価格特徴量生成完了: {len(feature_cols)} 特徴量")
        return df
    
    def build_oi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """オープンインタレスト関連の特徴量を生成
        
        Args:
            df: 入力データフレーム
            
        Returns:
            pd.DataFrame: 特徴量を追加したデータフレーム
        """
        oi_cols = [col for col in df.columns if col.startswith('oi_')]
        
        if not oi_cols:
            logger.warning("オープンインタレストデータが見つかりません")
            return df
        
        # メインの OI 列を特定
        main_oi_col = None
        for col in ['oi_total', 'oi_value', 'oi_c']:
            if col in df.columns:
                main_oi_col = col
                break
        
        if main_oi_col is None:
            logger.warning("メインのオープンインタレスト列が見つかりません")
            return df
        
        # OI 変化率
        for period in [1, 3, 6, 12, 24]:
            df[f'oi_change_{period}'] = df[main_oi_col].pct_change(period)
        
        # OI ボラティリティ
        for window in [12, 24, 48]:
            df[f'oi_volatility_{window}'] = df[main_oi_col].pct_change().rolling(window=window).std()
        
        # OI モメンタム
        for window in [12, 24, 48]:
            df[f'oi_momentum_{window}'] = (df[main_oi_col].diff(window) / df[main_oi_col].shift(window))
        
        # OI と価格の比率 (OIが増加し価格も上昇→強気、OIが減少し価格上昇→弱気)
        if 'price_close' in df.columns:
            df['oi_price_ratio'] = df[main_oi_col] / df['price_close']
            
            # OI 変化と価格変化の関係
            df['oi_price_correlation'] = (np.sign(df[main_oi_col].pct_change()) == 
                                         np.sign(df['price_close'].pct_change())).astype(float)
            
            # OI と価格のダイバージェンス
            for window in [12, 24]:
                oi_ma = df[main_oi_col].rolling(window=window).mean()
                price_ma = df['price_close'].rolling(window=window).mean()
                
                df[f'oi_price_divergence_{window}'] = (
                    (df[main_oi_col] > oi_ma) & (df['price_close'] < price_ma) |
                    (df[main_oi_col] < oi_ma) & (df['price_close'] > price_ma)
                ).astype(float)
        
        # 全特徴量を Z-score 正規化
        feature_cols = [col for col in df.columns if col.startswith('oi_') and col not in oi_cols]
        
        for col in feature_cols:
            df[f'{col}_zscore'] = self.zscore_normalize(df[col])
        
        logger.info(f"オープンインタレスト特徴量生成完了: {len(feature_cols)} 特徴量")
        return df
    
    def build_funding_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ファンディングレート関連の特徴量を生成
        
        Args:
            df: 入力データフレーム
            
        Returns:
            pd.DataFrame: 特徴量を追加したデータフレーム
        """
        funding_cols = [col for col in df.columns if col.startswith('funding_')]
        
        if not funding_cols:
            logger.warning("ファンディングレートデータが見つかりません")
            return df
        
        # メインのファンディングレート列を特定
        main_funding_col = None
        for col in ['funding_rate', 'funding_c', 'funding_value']:
            if col in df.columns:
                main_funding_col = col
                break
        
        if main_funding_col is None:
            logger.warning("メインのファンディングレート列が見つかりません")
            return df
        
        # 累積ファンディングレート
        for window in [6, 12, 24, 48]:
            df[f'funding_cum_{window}'] = df[main_funding_col].rolling(window=window).sum()
        
        # ファンディングレートの移動平均
        for window in [6, 12, 24]:
            df[f'funding_ma_{window}'] = df[main_funding_col].rolling(window=window).mean()
        
        # ファンディングレートの標準偏差
        for window in [12, 24, 48]:
            df[f'funding_std_{window}'] = df[main_funding_col].rolling(window=window).std()
        
        # ファンディングレートの絶対値
        df['funding_abs'] = df[main_funding_col].abs()
        
        # ファンディングレートの符号
        df['funding_sign'] = np.sign(df[main_funding_col])
        
        # ファンディングレートの継続性 (同じ符号が何期間続いているか)
        df['funding_streak'] = df['funding_sign'].groupby(
            (df['funding_sign'] != df['funding_sign'].shift()).cumsum()
        ).cumcount() + 1
        
        # ファンディングレートの極値からの乖離
        for window in [24, 48, 96]:
            funding_max = df[main_funding_col].rolling(window=window).max()
            funding_min = df[main_funding_col].rolling(window=window).min()
            funding_range = funding_max - funding_min
            
            # レンジ内の相対位置 (0～1)
            df[f'funding_relative_pos_{window}'] = (df[main_funding_col] - funding_min) / funding_range.replace(0, np.finfo(float).eps)
        
        # 価格との関係
        if 'price_close' in df.columns:
            # ファンディングレートと価格変化の相関
            for window in [12, 24, 48]:
                price_change = df['price_close'].pct_change()
                df[f'funding_price_corr_{window}'] = df[main_funding_col].rolling(window=window).corr(price_change)
        
        # 全特徴量を Z-score 正規化
        feature_cols = [col for col in df.columns if col.startswith('funding_') and col not in funding_cols]
        
        for col in feature_cols:
            df[f'{col}_zscore'] = self.zscore_normalize(df[col])
        
        logger.info(f"ファンディングレート特徴量生成完了: {len(feature_cols)} 特徴量")
        return df
    
    def build_liq_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """清算関連の特徴量を生成
        
        Args:
            df: 入力データフレーム
            
        Returns:
            pd.DataFrame: 特徴量を追加したデータフレーム
        """
        liq_cols = [col for col in df.columns if col.startswith('liq_')]
        
        if not liq_cols:
            logger.warning("清算データが見つかりません")
            return df
        
        # メインの清算列を特定
        long_liq_col = None
        short_liq_col = None
        total_liq_col = None
        
        for col in liq_cols:
            if 'long' in col.lower():
                long_liq_col = col
            elif 'short' in col.lower():
                short_liq_col = col
            elif 'total' in col.lower():
                total_liq_col = col
        
        # 総清算額がない場合は長期・短期清算額から計算
        if total_liq_col is None and long_liq_col is not None and short_liq_col is not None:
            df['liq_total'] = df[long_liq_col] + df[short_liq_col]
            total_liq_col = 'liq_total'
        
        if total_liq_col is None:
            logger.warning("メインの清算列が見つかりません")
            return df
        
        # 清算量の移動平均
        for window in [6, 12, 24]:
            df[f'liq_ma_{window}'] = df[total_liq_col].rolling(window=window).mean()
        
        # 清算量のボラティリティ
        for window in [12, 24, 48]:
            df[f'liq_volatility_{window}'] = df[total_liq_col].rolling(window=window).std()
        
        # 長期/短期清算比率
        if long_liq_col is not None and short_liq_col is not None:
            df['liq_long_short_ratio'] = df[long_liq_col] / df[short_liq_col].replace(0, np.finfo(float).eps)
            
            # 清算の偏り (正 = 主に長期が清算、負 = 主に短期が清算)
            df['liq_skew'] = (df[long_liq_col] - df[short_liq_col]) / (df[long_liq_col] + df[short_liq_col]).replace(0, np.finfo(float).eps)
        
        # 清算スパイク検出
        for window in [12, 24, 48]:
            # 移動平均ベースの異常値
            liq_ma = df[total_liq_col].rolling(window=window).mean()
            liq_std = df[total_liq_col].rolling(window=window).std()
            
            df[f'liq_spike_{window}'] = (df[total_liq_col] > liq_ma + 2 * liq_std).astype(float)
        
        # 価格との関係
        if 'price_close' in df.columns:
            # 価格変化率別の清算量
            price_change = df['price_close'].pct_change()
            
            df['liq_on_up_move'] = df[total_liq_col] * (price_change > 0).astype(float)
            df['liq_on_down_move'] = df[total_liq_col] * (price_change < 0).astype(float)
        
        # 全特徴量を Z-score 正規化
        feature_cols = [col for col in df.columns if col.startswith('liq_') and col not in liq_cols]
        
        for col in feature_cols:
            df[f'{col}_zscore'] = self.zscore_normalize(df[col])
        
        logger.info(f"清算特徴量生成完了: {len(feature_cols)} 特徴量")
        return df
    
    def build_lsr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ロングショート比率関連の特徴量を生成
        
        Args:
            df: 入力データフレーム
            
        Returns:
            pd.DataFrame: 特徴量を追加したデータフレーム
        """
        lsr_cols = [col for col in df.columns if col.startswith('lsr_')]
        
        if not lsr_cols:
            logger.warning("ロングショート比率データが見つかりません")
            return df
        
        # メインのロングショート比率列を特定
        main_lsr_col = None
        for col in ['lsr_ratio', 'lsr_value', 'lsr_longShortRatio']:
            if col in df.columns:
                main_lsr_col = col
                break
        
        if main_lsr_col is None:
            logger.warning("メインのロングショート比率列が見つかりません")
            return df
        
        # LSR の移動平均
        for window in [6, 12, 24]:
            df[f'lsr_ma_{window}'] = df[main_lsr_col].rolling(window=window).mean()
        
        # LSR のボラティリティ
        for window in [12, 24, 48]:
            df[f'lsr_volatility_{window}'] = df[main_lsr_col].rolling(window=window).std()
        
        # LSR の変化率
        for period in [1, 3, 6, 12]:
            df[f'lsr_change_{period}'] = df[main_lsr_col].pct_change(period)
        
        # LSR 極値からの乖離
        for window in [24, 48, 96]:
            lsr_max = df[main_lsr_col].rolling(window=window).max()
            lsr_min = df[main_lsr_col].rolling(window=window).min()
            lsr_range = lsr_max - lsr_min
            
            # レンジ内の相対位置 (0～1)
            df[f'lsr_relative_pos_{window}'] = (df[main_lsr_col] - lsr_min) / lsr_range.replace(0, np.finfo(float).eps)
        
        # 価格との関係
        if 'price_close' in df.columns:
            # LSR と価格変化の相関
            for window in [12, 24, 48]:
                price_change = df['price_close'].pct_change()
                df[f'lsr_price_corr_{window}'] = df[main_lsr_col].rolling(window=window).corr(price_change)
        
        # 全特徴量を Z-score 正規化
        feature_cols = [col for col in df.columns if col.startswith('lsr_') and col not in lsr_cols]
        
        for col in feature_cols:
            df[f'{col}_zscore'] = self.zscore_normalize(df[col])
        
        logger.info(f"ロングショート比率特徴量生成完了: {len(feature_cols)} 特徴量")
        return df
    
    def build_taker_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """テイカー出来高関連の特徴量を生成
        
        Args:
            df: 入力データフレーム
            
        Returns:
            pd.DataFrame: 特徴量を追加したデータフレーム
        """
        taker_cols = [col for col in df.columns if col.startswith('taker_')]
        
        if not taker_cols:
            logger.warning("テイカー出来高データが見つかりません")
            return df
        
        # 買い/売りのテイカー出来高列を特定
        buy_col = None
        sell_col = None
        
        for col in taker_cols:
            if 'buy' in col.lower():
                buy_col = col
            elif 'sell' in col.lower():
                sell_col = col
        
        if buy_col is None or sell_col is None:
            logger.warning("買い/売りのテイカー出来高列が見つかりません")
            return df
        
        # 総テイカー出来高
        df['taker_total'] = df[buy_col] + df[sell_col]
        
        # 買い/売り比率
        df['taker_buy_sell_ratio'] = df[buy_col] / df[sell_col].replace(0, np.finfo(float).eps)
        
        # テイカー買い圧力 (-1 to 1)
        df['taker_buy_pressure'] = (df[buy_col] - df[sell_col]) / (df[buy_col] + df[sell_col])
        
        # テイカー出来高の移動平均
        for window in [6, 12, 24]:
            df[f'taker_total_ma_{window}'] = df['taker_total'].rolling(window=window).mean()
            df[f'taker_buy_ma_{window}'] = df[buy_col].rolling(window=window).mean()
            df[f'taker_sell_ma_{window}'] = df[sell_col].rolling(window=window).mean()
            df[f'taker_pressure_ma_{window}'] = df['taker_buy_pressure'].rolling(window=window).mean()
        
        # テイカー出来高のボラティリティ
        for window in [12, 24, 48]:
            df[f'taker_volatility_{window}'] = df['taker_total'].rolling(window=window).std()
        
        # テイカー出来高の傾き
        for window in [6, 12, 24]:
            df[f'taker_slope_{window}'] = df['taker_total'].diff(window) / window
        
        # 価格との関係
        if 'price_close' in df.columns:
            # テイカー圧力と価格変化の相関
            for window in [12, 24, 48]:
                price_change = df['price_close'].pct_change()
                df[f'taker_price_corr_{window}'] = df['taker_buy_pressure'].rolling(window=window).corr(price_change)
        
        # 全特徴量を Z-score 正規化
        feature_cols = [col for col in df.columns if col.startswith('taker_') and col not in taker_cols]
        
        for col in feature_cols:
            df[f'{col}_zscore'] = self.zscore_normalize(df[col])
        
        logger.info(f"テイカー出来高特徴量生成完了: {len(feature_cols)} 特徴量")
        return df
    
    def build_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """オーダーブック関連の特徴量を生成
        
        Args:
            df: 入力データフレーム
            
        Returns:
            pd.DataFrame: 特徴量を追加したデータフレーム
        """
        ob_cols = [col for col in df.columns if col.startswith('orderbook_')]
        
        if not ob_cols:
            logger.warning("オーダーブックデータが見つかりません")
            return df
        
        # 買い/売りの注文量列を特定
        bid_col = None
        ask_col = None
        
        for col in ob_cols:
            if 'bid' in col.lower():
                bid_col = col
            elif 'ask' in col.lower():
                ask_col = col
        
        if bid_col is None or ask_col is None:
            logger.warning("買い/売りの注文量列が見つかりません")
            return df
        
        # 買い/売り比率
        df['orderbook_bid_ask_ratio'] = df[bid_col] / df[ask_col].replace(0, np.finfo(float).eps)
        
        # オーダーブックの傾き (-1 to 1)
        df['orderbook_skew'] = (df[bid_col] - df[ask_col]) / (df[bid_col] + df[ask_col])
        
        # オーダーブックの移動平均
        for window in [6, 12, 24]:
            df[f'orderbook_bid_ma_{window}'] = df[bid_col].rolling(window=window).mean()
            df[f'orderbook_ask_ma_{window}'] = df[ask_col].rolling(window=window).mean()
            df[f'orderbook_ratio_ma_{window}'] = df['orderbook_bid_ask_ratio'].rolling(window=window).mean()
        
        # オーダーブックのボラティリティ
        for window in [12, 24, 48]:
            df[f'orderbook_volatility_{window}'] = df['orderbook_bid_ask_ratio'].rolling(window=window).std()
        
        # オーダーブックの変化率
        for period in [1, 3, 6]:
            df[f'orderbook_bid_change_{period}'] = df[bid_col].pct_change(period)
            df[f'orderbook_ask_change_{period}'] = df[ask_col].pct_change(period)
            df[f'orderbook_ratio_change_{period}'] = df['orderbook_bid_ask_ratio'].pct_change(period)
        
        # 価格との関係
        if 'price_close' in df.columns:
            # オーダーブックの偏りと価格変化の相関
            for window in [12, 24, 48]:
                price_change = df['price_close'].pct_change()
                df[f'orderbook_price_corr_{window}'] = df['orderbook_skew'].rolling(window=window).corr(price_change)
        
        # 全特徴量を Z-score 正規化
        feature_cols = [col for col in df.columns if col.startswith('orderbook_') and col not in ob_cols]
        
        for col in feature_cols:
            df[f'{col}_zscore'] = self.zscore_normalize(df[col])
        
        logger.info(f"オーダーブック特徴量生成完了: {len(feature_cols)} 特徴量")
        return df
    
    def build_premium_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """プレミアム指数関連の特徴量を生成
        
        Args:
            df: 入力データフレーム
            
        Returns:
            pd.DataFrame: 特徴量を追加したデータフレーム
        """
        premium_cols = [col for col in df.columns if col.startswith('premium_')]
        
        if not premium_cols:
            logger.warning("プレミアム指数データが見つかりません")
            return df
        
        # メインのプレミアム指数列を特定
        main_premium_col = None
        for col in ['premium_index', 'premium_value', 'premium_premiumIndex']:
            if col in df.columns:
                main_premium_col = col
                break
        
        if main_premium_col is None:
            logger.warning("メインのプレミアム指数列が見つかりません")
            return df
        
        # プレミアム指数の移動平均
        for window in [6, 12, 24]:
            df[f'premium_ma_{window}'] = df[main_premium_col].rolling(window=window).mean()
        
        # プレミアム指数のボラティリティ
        for window in [12, 24, 48]:
            df[f'premium_volatility_{window}'] = df[main_premium_col].rolling(window=window).std()
        
        # プレミアム指数の変化率
        for period in [1, 3, 6, 12]:
            df[f'premium_change_{period}'] = df[main_premium_col].pct_change(period)
        
        # プレミアム指数の絶対値
        df['premium_abs'] = df[main_premium_col].abs()
        
        # プレミアム指数の符号
        df['premium_sign'] = np.sign(df[main_premium_col])
        
        # プレミアム指数の継続性 (同じ符号が何期間続いているか)
        df['premium_streak'] = df['premium_sign'].groupby(
            (df['premium_sign'] != df['premium_sign'].shift()).cumsum()
        ).cumcount() + 1
        
        # プレミアム指数の極値からの乖離
        for window in [24, 48, 96]:
            premium_max = df[main_premium_col].rolling(window=window).max()
            premium_min = df[main_premium_col].rolling(window=window).min()
            premium_range = premium_max - premium_min
            
            # レンジ内の相対位置 (0～1)
            df[f'premium_relative_pos_{window}'] = (df[main_premium_col] - premium_min) / premium_range.replace(0, np.finfo(float).eps)
        
        # 価格との関係
        if 'price_close' in df.columns:
            # プレミアム指数と価格変化の相関
            for window in [12, 24, 48]:
                price_change = df['price_close'].pct_change()
                df[f'premium_price_corr_{window}'] = df[main_premium_col].rolling(window=window).corr(price_change)
        
        # 全特徴量を Z-score 正規化
        feature_cols = [col for col in df.columns if col.startswith('premium_') and col not in premium_cols]
        
        for col in feature_cols:
            df[f'{col}_zscore'] = self.zscore_normalize(df[col])
        
        logger.info(f"プレミアム指数特徴量生成完了: {len(feature_cols)} 特徴量")
        return df
    
    def build_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """クロスデータセット特徴量を生成
        
        Args:
            df: 入力データフレーム
            
        Returns:
            pd.DataFrame: 特徴量を追加したデータフレーム
        """
        # 価格と OI の関係
        if 'price_close' in df.columns and any(col.startswith('oi_') for col in df.columns):
            # OI メインカラムを取得
            oi_col = None
            for col in ['oi_total', 'oi_value', 'oi_c']:
                if col in df.columns:
                    oi_col = col
                    break
            
            if oi_col is not None:
                # OI/価格比率の変化率
                df['cross_oi_price_ratio'] = df[oi_col] / df['price_close']
                df['cross_oi_price_ratio_change'] = df['cross_oi_price_ratio'].pct_change()
                
                # OI 増加・価格上昇 = 強いトレンド確認
                # OI 減少・価格上昇 = 弱いトレンド
                df['cross_oi_price_agreement'] = np.sign(df[oi_col].pct_change()) == np.sign(df['price_close'].pct_change())
                
                # OI と価格のダイバージェンス
                for window in [12, 24]:
                    df[f'cross_oi_price_divergence_{window}'] = (
                        # OI 上昇、価格下落 または OI 下落、価格上昇
                        (df[oi_col].rolling(window).mean().pct_change() > 0) & 
                        (df['price_close'].rolling(window).mean().pct_change() < 0) |
                        (df[oi_col].rolling(window).mean().pct_change() < 0) & 
                        (df['price_close'].rolling(window).mean().pct_change() > 0)
                    ).astype(float)
        
        # 価格とファンディングレートの関係
        if 'price_close' in df.columns and any(col.startswith('funding_') for col in df.columns):
            # ファンディングレートメインカラムを取得
            funding_col = None
            for col in ['funding_rate', 'funding_c', 'funding_value']:
                if col in df.columns:
                    funding_col = col
                    break
            
            if funding_col is not None:
                # ファンディングレートと価格変化の相関
                for window in [12, 24, 48]:
                    price_pct = df['price_close'].pct_change()
                    df[f'cross_funding_price_corr_{window}'] = df[funding_col].rolling(window).corr(price_pct)
                
                # ファンディングレートの乖離と価格変化
                # 正: ファンディングが高すぎる (買い過剰)
                # 負: ファンディングが低すぎる (売り過剰)
                for window in [24, 48]:
                    funding_ma = df[funding_col].rolling(window).mean()
                    funding_std = df[funding_col].rolling(window).std()
                    
                    df[f'cross_funding_deviation_{window}'] = (df[funding_col] - funding_ma) / funding_std.replace(0, np.finfo(float).eps)
        
        # テイカー買い圧力とオーダーブックの関係
        if any(col.startswith('taker_') for col in df.columns) and any(col.startswith('orderbook_') for col in df.columns):
            # 関連するカラムを確認
            if all(col in df.columns for col in ['taker_buy_pressure', 'orderbook_skew']):
                # テイカーとオーダーブックの一致度
                # 正: 両方とも同じサイドに偏っている (強い方向性)
                # 負: 異なるサイドに偏っている (弱い方向性)
                df['cross_taker_ob_agreement'] = np.sign(df['taker_buy_pressure']) == np.sign(df['orderbook_skew'])
                
                # テイカーとオーダーブックの相関
                for window in [12, 24]:
                    df[f'cross_taker_ob_corr_{window}'] = df['taker_buy_pressure'].rolling(window).corr(df['orderbook_skew'])
        
        # ロングショート比率とファンディングレートの関係
        if any(col.startswith('lsr_') for col in df.columns) and any(col.startswith('funding_') for col in df.columns):
            # 関連するカラムを特定
            lsr_col = None
            for col in ['lsr_ratio', 'lsr_value', 'lsr_longShortRatio']:
                if col in df.columns:
                    lsr_col = col
                    break
            
            funding_col = None
            for col in ['funding_rate', 'funding_c', 'funding_value']:
                if col in df.columns:
                    funding_col = col
                    break
            
            if lsr_col is not None and funding_col is not None:
                # LSR とファンディングレートの相関
                for window in [12, 24, 48]:
                    df[f'cross_lsr_funding_corr_{window}'] = df[lsr_col].rolling(window).corr(df[funding_col])
                
                # LSR とファンディングのサイン一致度
                df['cross_lsr_funding_agreement'] = np.sign(df[lsr_col] - 1) == np.sign(df[funding_col])
        
        # 全特徴量を Z-score 正規化
        cross_cols = [col for col in df.columns if col.startswith('cross_')]
        
        for col in cross_cols:
            df[f'{col}_zscore'] = self.zscore_normalize(df[col])
        
        logger.info(f"クロスデータセット特徴量生成完了: {len(cross_cols)} 特徴量")
        return df
    
    def build_features(self, interval: str) -> pd.DataFrame:
        """特徴量を生成して保存する
        
        Args:
            interval: 時間枠
            
        Returns:
            pd.DataFrame: 特徴量データフレーム
        """
        logger.info(f"特徴量生成開始: {interval}")
        
        # マージされたデータを読み込む
        try:
            df = self.load_merged_data(interval)
        except Exception as e:
            logger.error(f"マージデータの読み込みに失敗しました: {interval}, {str(e)}")
            raise
        
        # 時系列の順序を確認
        df = df.sort_index()
        
        # 各タイプの特徴量を生成
        try:
            # 価格特徴量
            df = self.build_price_features(df)
            
            # オープンインタレスト特徴量
            df = self.build_oi_features(df)
            
            # ファンディングレート特徴量
            df = self.build_funding_features(df)
            
            # 清算特徴量
            df = self.build_liq_features(df)
            
            # ロングショート比率特徴量
            df = self.build_lsr_features(df)
            
            # テイカー出来高特徴量
            df = self.build_taker_features(df)
            
            # オーダーブック特徴量
            df = self.build_orderbook_features(df)
            
            # プレミアム指数特徴量
            df = self.build_premium_features(df)
            
            # クロスデータセット特徴量
            df = self.build_cross_features(df)
            
        except Exception as e:
            logger.error(f"特徴量生成でエラーが発生しました: {str(e)}")
            raise
        
        # Z-score の特徴量だけを抽出
        zscore_features = [col for col in df.columns if col.endswith('_zscore')]
        X = df[zscore_features].copy()
        
        # 欠損値を 0 で埋める
        X = X.fillna(0)
        
        # 特徴量ディレクトリに保存
        features_dir = DATA_DIR / 'features' / interval
        os.makedirs(features_dir, exist_ok=True)
        
        output_path = features_dir / 'X.parquet'
        X.to_parquet(output_path)
        
        logger.info(f"特徴量生成完了: {output_path}, レコード数: {len(X)}, 特徴量数: {len(zscore_features)}")
        
        return X

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='特徴量生成')
    parser.add_argument('--interval', type=str, help='時間枠 (例: 15m, 2h)')
    return parser.parse_args()

def main():
    """メイン関数"""
    try:
        # ログディレクトリ作成
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'), exist_ok=True)
        
        args = parse_args()
        builder = FeatureBuilder()
        
        if args.interval:
            # 単一時間枠の処理
            builder.build_features(args.interval)
        else:
            # 全時間枠の処理
            for interval in builder.intervals:
                builder.build_features(interval)
        
        return 0
    
    except Exception as e:
        logger.error(f"実行エラー: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
