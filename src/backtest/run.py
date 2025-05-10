#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
バックテスト実行モジュール
-------------------

MTFロジックを使用した仮想通貨トレーディングシステムのバックテストを実行するコアモジュール。
各時間枠（1d, 2h, 15m）のデータを活用して売買判断を行います。

バックテストルール：
- vectorbt, `upon_op="Next"`, `fees=0.00055`, `slippage=0.0001`
- 15分足でシグナル生成、ポジションサイズはβ = proba×μ / VaR95
- 大局トレンドと需給はshift(1)で参照
"""

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Union
import vectorbt as vbt

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_type: str) -> Dict[str, Any]:
    """
    設定ファイルを読み込む
    
    Parameters:
    -----------
    config_type : str
        設定ファイルのタイプ ('intervals', 'fees', 'model', 'mtf')
        
    Returns:
    --------
    dict
        設定情報
    """
    config_path = os.path.join(project_root, 'config', f'{config_type}.yaml')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.warning(f"設定ファイルが見つかりません: {config_path}")
        # デフォルト設定を返す
        if config_type == 'fees':
            return {'trading_fee': 0.00055, 'slippage': 0.0001}
        elif config_type == 'intervals':
            return {'day': '1d', 'hour': '2h', 'minute': '15m'}
        return {}

def generate_signals(data: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    バックテスト用のシグナルを生成
    
    Parameters:
    -----------
    data : pd.DataFrame
        入力データフレーム
    interval : str
        バックテストの時間枠 (15m/2h/1d)
        
    Returns:
    --------
    pd.DataFrame
        シグナルを含むデータフレーム
    """
    logger.info(f"{interval}のシグナル生成中...")
    
    # データのコピーを作成
    df = data.copy()
    
    # サンプルロジック用の特徴量が存在するか確認
    if 'trend_flag' not in df.columns and interval == '15m':
        logger.warning("トレンドフラグが見つかりません。サンプルロジックに切り替えます。")
        # シンプルな移動平均クロスオーバー戦略
        df['sma_short'] = df['close'].rolling(20).mean()
        df['sma_long'] = df['close'].rolling(50).mean()
        df['signal'] = np.where(df['sma_short'] > df['sma_long'], 1, -1)
        return df
    
    # MTFロジックに基づくシグナル生成
    if interval == '15m':
        # 15分足のロジック - 二段モデル（実際のモデルは実装により異なる）
        if 'trend_flag' in df.columns:
            # トレンドフラグに基づくフィルタリング
            trend_filter = df['trend_flag'].fillna(0).astype(int)
            
            # サンプルロジック: RSI + トレンドフラグ
            df['rsi'] = calculate_rsi(df['close'], 14)
            long_condition = (df['rsi'] < 30) & (trend_filter == 1)
            short_condition = (df['rsi'] > 70) & (trend_filter == 0)
            
            # シグナル生成（1: ロング, -1: ショート, 0: ノーポジション）
            df['signal_direction'] = 0
            df.loc[long_condition, 'signal_direction'] = 1
            df.loc[short_condition, 'signal_direction'] = -1
            
            # レグレッションによるμ予測（サンプル）
            df['mu_pred'] = np.random.normal(0, 0.01, size=len(df))
            df.loc[df['signal_direction'] == 1, 'mu_pred'] = abs(df.loc[df['signal_direction'] == 1, 'mu_pred'])
            df.loc[df['signal_direction'] == -1, 'mu_pred'] = -abs(df.loc[df['signal_direction'] == -1, 'mu_pred'])
            
            # 確率値（LightGBMの出力を想定）
            df['proba'] = 0.5
            df.loc[df['signal_direction'] == 1, 'proba'] = 0.5 + np.random.uniform(0, 0.4, size=len(df.loc[df['signal_direction'] == 1]))
            df.loc[df['signal_direction'] == -1, 'proba'] = 0.5 - np.random.uniform(0, 0.4, size=len(df.loc[df['signal_direction'] == -1]))
            
            # VaR計算（簡易版）
            df['var95'] = calculate_rolling_var(df['close'].pct_change(), 95, 20)
            
            # ポジションサイズβを計算 (β = proba×μ / VaR95)
            df['beta'] = (df['proba'] - 0.5) * 2 * df['mu_pred'] / df['var95'].clip(lower=0.001)
            
            # シグナル生成
            df['signal'] = df['beta'].clip(lower=-1, upper=1)
        else:
            # 簡易的なサンプルロジック
            logger.warning("MTFデータが不足しているため、単純なロジックに切り替えます")
            df['rsi'] = calculate_rsi(df['close'], 14)
            df['signal'] = np.where(df['rsi'] < 30, 1, np.where(df['rsi'] > 70, -1, 0))
    
    elif interval == '2h':
        # 2時間足のロジック - Coinglassデータを活用した需給指標ベース戦略
        logger.info("2h Coinglassデータベース戦略を適用します")
        
        # 基本テクニカル指標
        df['rsi'] = calculate_rsi(df['close'], 14)
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ema200'] = df['close'].ewm(span=200).mean()
        df['atr'] = calculate_atr(df, 14)
        
        # Coinglassデータ列を特定
        funding_cols = [col for col in df.columns if 'funding_' in col]
        oi_cols = [col for col in df.columns if 'oi_' in col]
        liq_cols = [col for col in df.columns if 'liq_' in col or 'liquidation' in col]
        lsr_cols = [col for col in df.columns if 'lsr_' in col or 'longShortRatio' in col]
        premium_cols = [col for col in df.columns if 'premium_' in col]
        
        # 需給指標の集計変数を初期化
        supply_demand_score = np.zeros(len(df))
        
        # 1. ファンディングレート指標
        if funding_cols:
            funding_col = next((col for col in funding_cols if col.endswith('_c') or 'rate' in col), None)
            if funding_col:
                logger.info(f"ファンディングレート指標を使用: {funding_col}")
                # ファンディングレートの正規化
                df['funding_z'] = (df[funding_col] - df[funding_col].rolling(30).mean()) / df[funding_col].rolling(30).std()
                # 負のファンディングはロング有利、正のファンディングはショート有利
                supply_demand_score -= df['funding_z'].fillna(0).clip(-3, 3) / 3
        
        # 2. 未決済建玉（OI）指標
        if oi_cols:
            oi_col = next((col for col in oi_cols if col.endswith('_c')), None)
            if oi_col:
                logger.info(f"OI指標を使用: {oi_col}")
                # OI変化率
                df['oi_change'] = df[oi_col].pct_change(5)
                # OIの急増はトレンド転換の可能性
                df['oi_z'] = (df['oi_change'] - df['oi_change'].rolling(30).mean()) / df['oi_change'].rolling(30).std()
                # 価格上昇 + OI増加 = 強気、価格下落 + OI増加 = 弱気
                oi_signal = np.sign(df['close'].pct_change(5)) * np.sign(df['oi_change'])
                supply_demand_score += oi_signal.fillna(0) * 0.5
        
        # 3. 清算データ指標
        if len(liq_cols) >= 2:
            long_liq_col = next((col for col in liq_cols if 'long' in col.lower()), None)
            short_liq_col = next((col for col in liq_cols if 'short' in col.lower()), None)
            if long_liq_col and short_liq_col:
                logger.info(f"清算データ指標を使用: {long_liq_col}, {short_liq_col}")
                # 清算比率（ロング/ショート）
                df['liq_ratio'] = df[long_liq_col] / (df[short_liq_col] + 1e-10)
                # 比率の対数を取って正規化
                df['liq_ratio_log'] = np.log(df['liq_ratio'].clip(0.1, 10))
                df['liq_z'] = (df['liq_ratio_log'] - df['liq_ratio_log'].rolling(30).mean()) / df['liq_ratio_log'].rolling(30).std()
                # ショート清算増加はロング有利、ロング清算増加はショート有利
                supply_demand_score -= df['liq_z'].fillna(0).clip(-3, 3) / 3
        
        # 4. ロングショート比率指標
        if lsr_cols:
            lsr_col = next((col for col in lsr_cols if 'ratio' in col.lower()), None)
            if lsr_col:
                logger.info(f"LSR指標を使用: {lsr_col}")
                # LSRの正規化（1.0が均衡）
                df['lsr_norm'] = df[lsr_col] - 1.0
                df['lsr_z'] = (df['lsr_norm'] - df['lsr_norm'].rolling(30).mean()) / df['lsr_norm'].rolling(30).std()
                # 逆張り：LSRが高すぎるとショート、低すぎるとロング
                supply_demand_score -= df['lsr_z'].fillna(0).clip(-3, 3) / 3
        
        # 5. プレミアム指標
        if premium_cols:
            premium_col = next((col for col in premium_cols if 'rate' in col.lower()), None)
            if premium_col:
                logger.info(f"プレミアム指標を使用: {premium_col}")
                # プレミアムの正規化
                df['premium_z'] = (df[premium_col] - df[premium_col].rolling(30).mean()) / df[premium_col].rolling(30).std()
                # プレミアムが高いとショート有利、低いとロング有利
                supply_demand_score -= df['premium_z'].fillna(0).clip(-3, 3) / 3
        
        # 需給指標とテクニカル指標の組み合わせ
        # 1. トレンド判断（EMA200）
        df['trend'] = np.where(df['close'] > df['ema200'], 1, -1)
        
        # 2. モメンタム判断（RSI）
        df['momentum'] = np.where(df['rsi'] < 30, 1, np.where(df['rsi'] > 70, -1, 0))
        
        # 3. 統合シグナル計算
        # - トレンド：30%
        # - モメンタム：20%
        # - 需給指標：50%
        df['signal_raw'] = (0.3 * df['trend'] + 0.2 * df['momentum'] + 0.5 * supply_demand_score)
        
        # シグナルのスムージング
        df['signal_smooth'] = df['signal_raw'].rolling(3).mean().fillna(df['signal_raw'])
        
        # ポジションサイズを決定（-1.0～1.0）
        df['signal'] = df['signal_smooth'].clip(-1.0, 1.0)
        
        # トレード数を適切に抑制するためのフィルター
        df['signal_change'] = df['signal'].diff().abs()
        # シグナル変化が小さい場合は前の値を維持
        mask = df['signal_change'] < 0.3
        df.loc[mask, 'signal'] = df['signal'].shift(1)
    
    elif interval == '1d':
        # 日足のロジック - 大局トレンド判断
        df['ema200'] = df['close'].ewm(span=200).mean()
        df['signal'] = np.where(df['close'] > df['ema200'], 1, -1)
    
    logger.info(f"{interval}のシグナル生成完了")
    return df

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI（相対力指数）を計算
    
    Parameters:
    -----------
    series : pd.Series
        価格の時系列
    period : int
        RSIの期間
        
    Returns:
    --------
    pd.Series
        RSI値
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, 1e-10)  # ゼロ除算防止
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR（Average True Range）を計算
    
    Parameters:
    -----------
    df : pd.DataFrame
        価格のデータフレーム（'high', 'low', 'close'列を含む）
    period : int
        ATRの期間
        
    Returns:
    --------
    pd.Series
        ATR値
    """
    # True Range計算
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    
    # 3つの中から最大値を取得
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # ATR計算（単純移動平均）
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_rolling_var(returns: pd.Series, percentile: int = 95, window: int = 20) -> pd.Series:
    """
    ローリングVaR（Value at Risk）を計算
    
    Parameters:
    -----------
    returns : pd.Series
        リターンの時系列
    percentile : int
        パーセンタイル（例: 95）
    window : int
        ローリングウィンドウサイズ
        
    Returns:
    --------
    pd.Series
        VaR値
    """
    def roll_var(x):
        return np.percentile(x, 100 - percentile)
    
    var = returns.rolling(window=window).apply(roll_var, raw=True)
    return var.abs()  # 絶対値を取る

def run_backtest(data: pd.DataFrame = None, interval: str = '2h', symbol: str = 'BTC-USD', days: int = 360) -> Dict[str, Any]:
    """
    バックテストを実行（単一時間枠）

    Parameters:
    -----------
    data : pd.DataFrame, optional
        入力データフレーム（提供されない場合は自動取得）
    interval : str
        バックテストの時間枠 (デフォルト: '2h')
    symbol : str
        取得対象のシンボル（データが提供されない場合に使用）
    days : int
        取得する日数（データが提供されない場合に使用、デフォルト: 360日）

    Returns:
    --------
    dict
        バックテスト結果
    """
    logger.info(f"{interval}のバックテスト実行中...")

    # 設定読み込み
    fees_config = load_config('fees')
    trading_fee = fees_config.get('trading_fee', 0.00055)
    slippage = fees_config.get('slippage', 0.0001)

    # データが提供されていない場合は取得
    if data is None:
        from src.data.sync import DataSync
        api_key_file = os.path.join(project_root, 'config', 'api_keys.yaml')
        data_sync = DataSync(api_key_file=api_key_file)
        data = data_sync.fetch_single_timeframe_data(symbol=symbol, interval=interval, limit=days)
        logger.info(f"{len(data)}行のデータを取得しました（{days}日分）")

    # シグナル生成
    df_signals = generate_signals(data, interval)

    # 必要な列を取得
    price = df_signals['close'].astype(float)
    signal = df_signals['signal'].fillna(0).astype(float)
    
    # 不適切な値をチェック (NaN, Inf)
    price = price.replace([np.inf, -np.inf], np.nan).dropna()
    signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 両方のシリーズが同じインデックスを持つようにする
    common_idx = price.index.intersection(signal.index)
    price = price.loc[common_idx]
    signal = signal.loc[common_idx]
    
    # データが空でないことを確認
    if len(price) == 0 or len(signal) == 0:
        logger.error("データが空です。バックテストを実行できません。")
        return {
            'portfolio': None,
            'price': price,
            'signal': signal,
            'df': df_signals,
            'interval': interval
        }

    # vectorbtでバックテスト実行
    # fees：取引手数料
    # slippage：スリッページ
    portfolio = None
    try:
        # 新しいvectorbtバージョン用
        try:
            # 基本的なシグナルベースのバックテスト
            portfolio = vbt.Portfolio.from_signals(
                close=price,             # 終値（必須パラメータ）
                entries=signal > 0,
                exits=signal < 0,
                direction='both',        # ロングとショートの両方
                size=np.abs(signal),     # ポジションサイズ
                fees=trading_fee,
                slippage=slippage,
                freq=interval,
                init_cash=10000,         # 初期資金
                conflict_mode='exit'     # シグナル衝突時の処理
            )
        except Exception as e1:
            logger.warning(f"高度なバックテスト実行で例外が発生しました: {e1}。シンプルなパラメータに切り替えます。")
            # シンプルなパラメータでリトライ
            portfolio = vbt.Portfolio.from_signals(
                close=price,             # 終値（必須パラメータ）
                entries=signal > 0,
                exits=signal < 0,
                size=np.abs(signal),     # ポジションサイズ
                fees=trading_fee,
                init_cash=10000          # 初期資金
            )
    except Exception as e:
        logger.warning(f"VectorBTでのバックテスト実行に失敗しました: {e}。手動バックテストに切り替えます。")
        
        # 手動バックテスト実装（シンプルな実装）
        init_cash = 10000
        cash = init_cash
        position = 0
        positions = []
        equity = []
        trades = []
        
        # バックテストループ
        for i in range(1, len(price)):
            prev_price = price.iloc[i-1]
            curr_price = price.iloc[i]
            curr_signal = signal.iloc[i]
            
            # シグナルに基づいたポジション変更
            if curr_signal > 0 and position <= 0:  # ロングエントリー
                # 前のショートポジションをクローズ
                if position < 0:
                    profit = -position * (prev_price - curr_price) - abs(position * prev_price * trading_fee)
                    cash += abs(position) * curr_price + profit
                    trades.append({
                        'entry_time': price.index[i-1],
                        'exit_time': price.index[i],
                        'type': 'short',
                        'entry_price': prev_price,
                        'exit_price': curr_price,
                        'size': abs(position),
                        'pnl': profit
                    })
                
                # 新しいロングポジション
                size = (cash * abs(curr_signal)) / curr_price
                cost = size * curr_price * (1 + trading_fee)
                if cost <= cash:
                    position = size
                    cash -= cost
            
            elif curr_signal < 0 and position >= 0:  # ショートエントリー
                # 前のロングポジションをクローズ
                if position > 0:
                    profit = position * (curr_price - prev_price) - position * curr_price * trading_fee
                    cash += position * curr_price + profit
                    trades.append({
                        'entry_time': price.index[i-1],
                        'exit_time': price.index[i],
                        'type': 'long',
                        'entry_price': prev_price,
                        'exit_price': curr_price,
                        'size': position,
                        'pnl': profit
                    })
                
                # 新しいショートポジション
                size = (cash * abs(curr_signal)) / curr_price
                cost = size * curr_price * (1 + trading_fee)
                if cost <= cash:
                    position = -size
                    cash -= cost
            
            # 評価額計算
            if position > 0:  # ロングポジション
                equity_val = cash + position * curr_price
            elif position < 0:  # ショートポジション
                equity_val = cash - position * curr_price
            else:  # ノーポジション
                equity_val = cash
            
            positions.append(position)
            equity.append(equity_val)
        
        # バックテスト結果の集計
        equity_series = pd.Series(equity, index=price.index[1:])
        final_equity = equity[-1] if equity else init_cash
        total_return = (final_equity - init_cash) / init_cash
        
        # シンプルなポートフォリオオブジェクト（VectorBT互換）
        class SimplePortfolio:
            def __init__(self, equity, positions, trades, total_return):
                self.equity = equity
                self.positions = positions
                self.trades = trades
                self.total_return_value = total_return
                self.init_cash = 10000
                self.wrapper = type('wrapper', (), {'index': equity.index})
                
                # 取引履歴をDataFrameに変換（VectorBT互換）
                self.trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
                
                # フリーク用のクラスを作成
                class MockFreq:
                    def __init__(self, wrapper):
                        self.wrapper = wrapper
                        
                    def get(self):
                        if interval == '2h':
                            return '2h'
                        elif interval == '15m':
                            return '15min'
                        elif interval == '1d':
                            return '1d'
                        return None
                
                self.freq = MockFreq(self.wrapper)
            
            # VectorBT互換API
            def total_return(self):
                return self.total_return_value
            
            def max_drawdown(self):
                return self._calculate_max_drawdown()
            
            def sharpe_ratio(self, risk_free=0.0, year_freq=None):
                return self._calculate_sharpe_ratio(year_freq)
            
            def sortino_ratio(self, risk_free=0.0, year_freq=None):
                return self._calculate_sortino_ratio(year_freq)
            
            def calmar_ratio(self, risk_free=0.0, year_freq=None):
                max_dd = self._calculate_max_drawdown()
                return self.total_return_value / abs(max_dd) if abs(max_dd) > 0 else 0
                
            def drawdown(self):
                equity_values = np.array(self.equity)
                result = np.zeros_like(equity_values, dtype=float)
                peak = equity_values[0]
                
                for i, value in enumerate(equity_values):
                    if value > peak:
                        peak = value
                    if peak > 0:
                        result[i] = (value - peak) / peak
                    
                return pd.Series(result, index=self.equity.index)
            
            @property
            def value(self):
                return self.equity
                
            def stats(self):
                return {
                    'total_return': self.total_return_value,
                    'max_drawdown': self._calculate_max_drawdown(),
                    'win_rate': self._calculate_win_rate(),
                    'sharpe_ratio': self._calculate_sharpe_ratio()
                }
                
            def _calculate_max_drawdown(self):
                # シンプルな最大ドローダウン計算
                equity_arr = np.array(self.equity)
                max_dd = 0
                peak = equity_arr[0]
                
                for eq in equity_arr:
                    if eq > peak:
                        peak = eq
                    dd = (peak - eq) / peak
                    if dd > max_dd:
                        max_dd = dd
                
                return max_dd
                
            def _calculate_win_rate(self):
                if not self.trades:
                    return 0
                    
                wins = sum(1 for t in self.trades if t['pnl'] > 0)
                return wins / len(self.trades) if self.trades else 0
                
            def _calculate_sharpe_ratio(self, year_freq=None):
                if len(self.equity) < 2:
                    return 0
                
                # リターン計算    
                returns = np.diff(self.equity) / self.equity[:-1]
                
                # 年率調整係数
                if year_freq is None:
                    if interval == '15m':
                        year_freq = 365 * 24 * 4  # 15分足は1日に4*24回
                    elif interval == '2h':
                        year_freq = 365 * 12  # 2時間足は1日に12回
                    else:  # 1d
                        year_freq = 365  # 日足は年間365日
                
                # 年率調整したシャープレシオ
                annual_factor = np.sqrt(year_freq)
                mean_return = np.mean(returns)
                std_return = np.std(returns, ddof=1)
                
                if std_return > 0:
                    return mean_return / std_return * annual_factor
                return 0
                
            def _calculate_sortino_ratio(self, year_freq=None):
                if len(self.equity) < 2:
                    return 0
                
                # リターン計算    
                returns = np.diff(self.equity) / self.equity[:-1]
                
                # 年率調整係数
                if year_freq is None:
                    if interval == '15m':
                        year_freq = 365 * 24 * 4
                    elif interval == '2h':
                        year_freq = 365 * 12
                    else:  # 1d
                        year_freq = 365
                
                # ソルティノレシオ計算（下方リスクのみ）
                annual_factor = np.sqrt(year_freq)
                mean_return = np.mean(returns)
                # 下方リスク（負のリターンのみ）の標準偏差
                downside_returns = returns[returns < 0]
                downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else 0.0001
                
                if downside_std > 0:
                    return mean_return / downside_std * annual_factor
                return 0
                
            # VectorBT互換のための追加プロパティ
            @property
            def trades(self):
                # trades属性のAPI構造をVectorBT風にエミュレート
                class TradesAccessor:
                    def __init__(self, trades_data, trades_df):
                        self.data = trades_data
                        self.df = trades_df
                        
                    def __len__(self):
                        return len(self.data)
                        
                    @property
                    def win_rate(self):
                        if not self.data:
                            return 0
                        wins = sum(1 for t in self.data if t['pnl'] > 0)
                        return wins / len(self.data)
                        
                    @property
                    def duration(self):
                        class Duration:
                            def __init__(self, trades):
                                self.trades = trades
                                
                            def mean(self):
                                if not self.trades:
                                    return 0
                                # 実際の取引期間の平均を計算
                                # 簡略化のため1取引=1期間とする
                                return 1
                                
                        return Duration(self.data)
                    
                    def stats(self):
                        return {
                            'win_rate': self.win_rate,
                            'count': len(self.data)
                        }
                        
                    def __getitem__(self, key):
                        # データフレームインデックス操作をエミュレート
                        if isinstance(key, str):
                            if key == 'pnl':
                                return pd.Series([t['pnl'] for t in self.data])
                        return self.df[key]
                        
                    @property
                    def winning(self):
                        class WinningTrades:
                            def __init__(self, trades):
                                self.pnl = pd.Series([t['pnl'] for t in trades if t['pnl'] > 0])
                                
                            def sum(self):
                                return self.pnl.sum()
                                
                            def mean(self):
                                return self.pnl.mean() if len(self.pnl) > 0 else 0
                                
                        return WinningTrades(self.data)
                        
                    @property
                    def losing(self):
                        class LosingTrades:
                            def __init__(self, trades):
                                self.pnl = pd.Series([t['pnl'] for t in trades if t['pnl'] <= 0])
                                
                            def sum(self):
                                return self.pnl.sum()
                                
                            def mean(self):
                                return self.pnl.mean() if len(self.pnl) > 0 else 0
                                
                        return LosingTrades(self.data)
                
                # インスタンスを返す
                return TradesAccessor(self.trades_df.to_dict('records') if not self.trades_df.empty else self.trades, self.trades_df)
            
            @property
            def returns(self):
                # 累積リターンからの変化率計算
                if len(self.equity) <= 1:
                    return pd.Series([], index=self.equity.index)
                return self.equity.pct_change().fillna(0)
        
        portfolio = SimplePortfolio(
            equity=equity_series,
            positions=positions,
            trades=trades,
            total_return=total_return
        )

    # VectorBTのポートフォリオ結果をカスタム辞書に変換
    result = {
        'portfolio': portfolio,
        'price': price,
        'signal': signal,
        'df': df_signals,
        'interval': interval
    }

    logger.info(f"{interval}のバックテスト完了")

    return result
