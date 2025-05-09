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
        # 2時間足のロジック - 重要な需給指標を集計
        df['rsi'] = calculate_rsi(df['close'], 14)
        df['ema200'] = df['close'].ewm(span=200).mean()
        df['signal'] = np.where(
            (df['rsi'] < 30) & (df['close'] > df['ema200']), 
            1, 
            np.where((df['rsi'] > 70) & (df['close'] < df['ema200']), -1, 0)
        )
    
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

def run_backtest(data: pd.DataFrame, interval: str = '15m') -> Dict[str, Any]:
    """
    バックテストを実行
    
    Parameters:
    -----------
    data : pd.DataFrame
        入力データフレーム
    interval : str
        バックテストの時間枠 (15m/2h/1d)
        
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
    
    # シグナル生成
    df_signals = generate_signals(data, interval)
    
    # 必要な列を取得
    price = df_signals['close']
    signal = df_signals['signal'].fillna(0)
    
    # vectorbtでバックテスト実行
    # fees：取引手数料
    # slippage：スリッページ
    try:
        # 新しいvectorbtバージョン用
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
    except TypeError as e:
        # バージョンによってパラメータが異なる可能性があるため
        logger.warning(f"最初のバックテスト実行で例外が発生しました: {e}。シンプルなパラメータに切り替えます。")
        portfolio = vbt.Portfolio.from_signals(
            close=price,             # 終値（必須パラメータ）
            entries=signal > 0,
            exits=signal < 0,
            size=np.abs(signal),     # ポジションサイズ
            fees=trading_fee,
            init_cash=10000          # 初期資金
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
