#!/usr/bin/env python
"""
plot_backtest_results.py - バックテスト結果プロットモジュール

機能:
- リターン曲線のプロット
- ドローダウン曲線のプロット
- 月次リターンヒートマップ
- サマリーテーブル
"""

import os
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union

# 内部モジュール
from src.backtest.portfolio import SimplePortfolio

# ロギング設定
logger = logging.getLogger(__name__)

def plot_returns(portfolio: SimplePortfolio, save_path: Optional[str] = None) -> plt.Figure:
    """
    リターン曲線をプロット

    Parameters:
    -----------
    portfolio : SimplePortfolio
        バックテスト結果のポートフォリオオブジェクト
    save_path : str, optional
        保存先パス
    
    Returns:
    --------
    plt.Figure
        プロット図オブジェクト
    """
    if not hasattr(portfolio, 'df') or portfolio.df is None:
        logger.warning("ポートフォリオにdfがありません")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'cumulative_returns' in portfolio.df.columns:
        portfolio.df['cumulative_returns'].plot(ax=ax)
        
        # 開始値と終了値をテキスト表示
        start_value = portfolio.df['cumulative_returns'].iloc[0]
        end_value = portfolio.df['cumulative_returns'].iloc[-1]
        
        # プロット開始位置と終了位置に値をテキスト表示
        ax.text(portfolio.df.index[0], start_value, f'{start_value:.2f}',
                verticalalignment='bottom', horizontalalignment='left')
        ax.text(portfolio.df.index[-1], end_value, f'{end_value:.2f}',
                verticalalignment='bottom', horizontalalignment='right')
    else:
        logger.warning("cumulative_returns列が見つかりません")
    
    ax.set_title('Cumulative Returns')
    ax.set_ylabel('Returns')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig

def plot_drawdowns(portfolio: SimplePortfolio, save_path: Optional[str] = None) -> plt.Figure:
    """
    ドローダウン曲線をプロット

    Parameters:
    -----------
    portfolio : SimplePortfolio
        バックテスト結果のポートフォリオオブジェクト
    save_path : str, optional
        保存先パス
    
    Returns:
    --------
    plt.Figure
        プロット図オブジェクト
    """
    if not hasattr(portfolio, 'df') or portfolio.df is None:
        logger.warning("ポートフォリオにdfがありません")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'cumulative_returns' in portfolio.df.columns:
        # ドローダウン計算
        running_max = portfolio.df['cumulative_returns'].cummax()
        drawdown = (portfolio.df['cumulative_returns'] / running_max - 1) * 100
        drawdown.plot(ax=ax)
        
        # 最大ドローダウン位置にマーク
        min_idx = drawdown.idxmin()
        min_value = drawdown.min()
        ax.plot(min_idx, min_value, 'ro')
        ax.text(min_idx, min_value, f'{min_value:.2f}%',
                verticalalignment='top', horizontalalignment='right')
    else:
        logger.warning("cumulative_returns列が見つかりません")
    
    ax.set_title('Drawdowns (%)')
    ax.set_ylabel('Drawdown %')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig

def plot_monthly_returns(portfolio: SimplePortfolio, save_path: Optional[str] = None) -> plt.Figure:
    """
    月次リターンヒートマップをプロット

    Parameters:
    -----------
    portfolio : SimplePortfolio
        バックテスト結果のポートフォリオオブジェクト
    save_path : str, optional
        保存先パス
    
    Returns:
    --------
    plt.Figure
        プロット図オブジェクト
    """
    if not hasattr(portfolio, 'df') or portfolio.df is None:
        logger.warning("ポートフォリオにdfがありません")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if 'strategy_returns' in portfolio.df.columns:
        # 戦略リターンを月次で集計
        returns = portfolio.df['strategy_returns'].copy()
        returns.index = pd.to_datetime(returns.index)
        
        # 月次リターン集計
        monthly_returns = returns.resample('M').sum() * 100  # パーセント表示
        
        # 年と月でピボット
        monthly_returns.index = monthly_returns.index.to_period('M')
        monthly_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).sum()
        monthly_pivot = monthly_pivot.unstack(level=1)
        
        # ヒートマップで表示
        sns.heatmap(monthly_pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, linewidths=0.5, cbar_kws={"label": "リターン %"})
        
        ax.set_title('Monthly Returns (%)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        # 月の名前を設定
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_names, rotation=45)
    else:
        logger.warning("strategy_returns列が見つかりません")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig

def plot_summary_table(portfolio: SimplePortfolio, interval: str = "15m",
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    サマリーテーブルをプロット

    Parameters:
    -----------
    portfolio : SimplePortfolio
        バックテスト結果のポートフォリオオブジェクト
    interval : str
        時間枠
    save_path : str, optional
        保存先パス
    
    Returns:
    --------
    plt.Figure
        プロット図オブジェクト
    """
    stats = portfolio.stats().copy()
    
    # テーブルデータの準備
    table_data = []
    
    # 基本情報
    table_data.append(["時間枠", interval])
    table_data.append(["バックテスト期間", f"{portfolio.df.index[0].strftime('%Y-%m-%d')} - {portfolio.df.index[-1].strftime('%Y-%m-%d')}"])
    
    # パフォーマンス指標
    if "Total Return [%]" in stats:
        table_data.append(["総リターン", f"{stats['Total Return [%]']:.2f}%"])
    if "Annual Return [%]" in stats:
        table_data.append(["年間リターン", f"{stats['Annual Return [%]']:.2f}%"])
    if "Sharpe Ratio" in stats:
        table_data.append(["シャープレシオ", f"{stats['Sharpe Ratio']:.2f}"])
    if "Max Drawdown [%]" in stats:
        table_data.append(["最大ドローダウン", f"{stats['Max Drawdown [%]']:.2f}%"])
    
    # トレード統計
    if hasattr(portfolio, 'df') and 'signal' in portfolio.df.columns:
        # シグナルからポジション変化を計算
        signals = portfolio.df['signal'].fillna(0)
        position_changes = signals.diff().fillna(0).abs()
        num_trades = position_changes[position_changes > 0].count()
        
        # Win率計算 (signal=1のときの次期リターンが正ならWin)
        if 'strategy_returns' in portfolio.df.columns:
            wins = ((signals == 1) & (portfolio.df['strategy_returns'].shift(-1) > 0)).sum()
            losses = ((signals == 1) & (portfolio.df['strategy_returns'].shift(-1) <= 0)).sum()
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            table_data.append(["トレード回数", f"{num_trades}"])
            table_data.append(["勝率", f"{win_rate:.2%} ({wins}/{wins+losses})"])
    
    # figureの準備
    fig, ax = plt.subplots(figsize=(8, len(table_data) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # カラーマップの設定
    colors = [['#f0f0f0', '#f0f0f0']] * len(table_data)
    
    # テーブルの作成
    table = ax.table(
        cellText=table_data,
        colWidths=[0.4, 0.6],
        cellLoc='left',
        loc='center',
        cellColours=colors
    )
    
    # テーブルスタイル調整
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # タイトル設定
    ax.set_title(f'バックテスト結果サマリー ({interval})', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig

def generate_all_plots(portfolio: SimplePortfolio, interval: str = "15m",
                      save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
    """
    すべてのプロットを生成

    Parameters:
    -----------
    portfolio : SimplePortfolio
        バックテスト結果のポートフォリオオブジェクト
    interval : str
        時間枠
    save_dir : str, optional
        保存先ディレクトリ
    
    Returns:
    --------
    Dict[str, plt.Figure]
        プロット図オブジェクトの辞書
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    plots = {}
    
    # リターン曲線
    returns_path = os.path.join(save_dir, f"{interval}_returns.png") if save_dir else None
    plots['returns'] = plot_returns(portfolio, returns_path)
    
    # ドローダウン曲線
    drawdowns_path = os.path.join(save_dir, f"{interval}_drawdowns.png") if save_dir else None
    plots['drawdowns'] = plot_drawdowns(portfolio, drawdowns_path)
    
    # 月次リターン
    monthly_returns_path = os.path.join(save_dir, f"{interval}_monthly_returns.png") if save_dir else None
    plots['monthly_returns'] = plot_monthly_returns(portfolio, monthly_returns_path)
    
    # サマリーテーブル
    summary_table_path = os.path.join(save_dir, f"{interval}_summary_table.png") if save_dir else None
    plots['summary_table'] = plot_summary_table(portfolio, interval, summary_table_path)
    
    # シグナル分布
    if hasattr(portfolio, 'df') and 'signal' in portfolio.df.columns and 'pred_proba' in portfolio.df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # シグナルごとに色分け
        colors = {1: 'green', -1: 'red', 0: 'blue'}
        
        for signal, group in portfolio.df.groupby('signal'):
            ax.hist(group['pred_proba'], bins=50, alpha=0.5, 
                   label=f"Signal={signal}", color=colors.get(signal, 'gray'))
        
        ax.set_title('Signal Distribution')
        ax.set_xlabel('Prediction Probability')
        ax.set_ylabel('Count')
        ax.legend()
        plt.tight_layout()
        
        if save_dir:
            signal_dist_path = os.path.join(save_dir, f"{interval}_signal_distribution.png")
            plt.savefig(signal_dist_path, dpi=100, bbox_inches='tight')
            plots['signal_distribution'] = fig
        else:
            plots['signal_distribution'] = fig
    
    logger.info(f"{len(plots)}個のプロットを生成しました" +
               (f", 保存先: {save_dir}" if save_dir else ""))
    
    return plots

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("このスクリプトは直接実行用ではありません。")
