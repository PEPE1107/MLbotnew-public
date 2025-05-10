#!/usr/bin/env python
"""
show_backtest_results.py - バックテスト結果表示モジュール

機能:
- バックテスト結果の統計表示
- パフォーマンス指標の表示
- JSONへの保存
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 内部モジュール
from src.backtest.portfolio import SimplePortfolio

# ロギング設定
logger = logging.getLogger(__name__)

def show_backtest_results(portfolio: SimplePortfolio, interval: str = "15m") -> None:
    """
    バックテスト結果を表示する

    Parameters:
    -----------
    portfolio : SimplePortfolio
        バックテスト結果のポートフォリオオブジェクト
    interval : str, optional
        時間枠 (例: '15m', '1h', '2h')
    """
    if portfolio is None:
        logger.warning("ポートフォリオが None です")
        return

    # 統計情報を取得
    stats = portfolio.stats()
    
    logger.info(f"{interval}時間足バックテスト結果:")
    logger.info("=" * 40)
    
    # 主要メトリクスを表示
    for key, value in stats.items():
        logger.info(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    # 追加メトリクスがあれば計算
    if hasattr(portfolio, 'df') and 'strategy_returns' in portfolio.df.columns:
        # 月次リターン集計
        df = portfolio.df
        monthly_returns = df['strategy_returns'].resample('M').sum()
        positive_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns)
        win_rate = positive_months / total_months if total_months > 0 else 0
        
        logger.info(f"月次勝率: {win_rate:.2%} ({positive_months}/{total_months})")
    
    # 統計情報をJSONとして保存
    save_stats_to_json(stats, interval)
    
    # グラフを表示
    plot_returns(portfolio)
    plot_drawdowns(portfolio)
    
    logger.info("=" * 40)

def save_stats_to_json(stats: dict, interval: str = "15m") -> None:
    """
    統計情報をJSONとして保存

    Parameters:
    -----------
    stats : dict
        統計情報
    interval : str, optional
        時間枠
    """
    # プロジェクトルートディレクトリ
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    reports_dir = root_dir / 'reports' / interval
    
    # レポートディレクトリが存在しなければ作成
    os.makedirs(reports_dir, exist_ok=True)
    
    # 現在の統計情報を保存
    stats_file = reports_dir / 'stats.json'
    
    # 古い統計情報をバックアップ
    if os.path.exists(stats_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = reports_dir / f"stats_{timestamp}.json"
        with open(stats_file, 'r') as f:
            old_stats = json.load(f)
        with open(backup_file, 'w') as f:
            json.dump(old_stats, f, indent=2)
    
    # 新しい統計情報を保存
    with open(stats_file, 'w') as f:
        # PandasのSeriesやNumPy型を通常の型に変換
        stats_dict = {}
        for key, value in stats.items():
            if isinstance(value, (np.integer, np.floating)):
                stats_dict[key] = float(value)
            else:
                stats_dict[key] = value
        json.dump(stats_dict, f, indent=2)
        
    logger.info(f"統計情報を保存しました: {stats_file}")

def plot_returns(portfolio: SimplePortfolio) -> None:
    """
    リターン曲線をプロット

    Parameters:
    -----------
    portfolio : SimplePortfolio
        バックテスト結果のポートフォリオオブジェクト
    """
    if not hasattr(portfolio, 'plot'):
        logger.warning("ポートフォリオにplotメソッドがありません")
        return
    
    try:
        fig = portfolio.plot()
        plt.title('Cumulative Returns')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"リターン曲線のプロットに失敗しました: {e}")

def plot_drawdowns(portfolio: SimplePortfolio) -> None:
    """
    ドローダウン曲線をプロット

    Parameters:
    -----------
    portfolio : SimplePortfolio
        バックテスト結果のポートフォリオオブジェクト
    """
    if not hasattr(portfolio, 'plot_drawdowns'):
        logger.warning("ポートフォリオにplot_drawdownsメソッドがありません")
        return
    
    try:
        fig = portfolio.plot_drawdowns()
        plt.title('Drawdowns (%)')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"ドローダウン曲線のプロットに失敗しました: {e}")
