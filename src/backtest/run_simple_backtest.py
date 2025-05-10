#!/usr/bin/env python
"""
run_simple_backtest.py - シンプルなバックテスト実行モジュール

機能:
- ベーシックなバックテストの実行
- ポートフォリオオブジェクトの生成
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

# 内部モジュール
from src.backtest.portfolio import SimplePortfolio, BacktestRunner

# ロギング設定
logger = logging.getLogger(__name__)

def run_simple_backtest(interval: str, start_date: Optional[str] = None, 
                        end_date: Optional[str] = None, 
                        use_sample: bool = False) -> SimplePortfolio:
    """
    シンプルなバックテストを実行する

    Parameters:
    -----------
    interval : str
        バックテストの時間枠 (例: '15m', '1h', '2h')
    start_date : str, optional
        開始日 (YYYY-MM-DD形式)
    end_date : str, optional
        終了日 (YYYY-MM-DD形式)
    use_sample : bool, optional
        サンプルデータを使用するかどうか

    Returns:
    --------
    SimplePortfolio
        バックテスト結果のポートフォリオオブジェクト
    """
    logger.info(f"シンプルバックテスト開始: {interval}")

    # BacktestRunner インスタンスを作成
    runner = BacktestRunner()
    
    try:
        # データ読み込み
        df = runner.load_data(interval)
        
        # 日付フィルタリング
        if start_date:
            start_date_dt = pd.to_datetime(start_date)
            df = df[df.index >= start_date_dt]
        if end_date:
            end_date_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_date_dt]
        
        logger.info(f"データ範囲: {df.index.min()} から {df.index.max()}, レコード数: {len(df)}")
        
        # モデル読み込み
        model, feature_cols = runner.load_model(interval)
        
        # シグナル生成
        signals_df = runner.generate_signals(df, model, feature_cols)
        
        # バックテスト実行
        portfolio, stats = runner.run_backtest(signals_df)
        
        return portfolio
        
    except Exception as e:
        logger.error(f"バックテスト実行エラー: {str(e)}")
        logger.exception(e)
        
        # エラー時は空のポートフォリオを返す
        empty_df = pd.DataFrame({
            'cumulative_returns': pd.Series([1.0]),
            'drawdown': pd.Series([0.0])
        })
        empty_stats = pd.Series({
            'Total Return [%]': 0.0,
            'Annual Return [%]': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown [%]': 0.0
        })
        return SimplePortfolio(empty_df, empty_stats)
