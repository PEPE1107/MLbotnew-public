#!/usr/bin/env python
"""
plot_backtest_results.py - バックテスト結果プロットモジュール

機能:
- リターン曲線のプロット (BTCの市場価格を追加)
- ドローダウン曲線のプロット
- 月次リターンヒートマップ
- サマリーテーブル
"""

import os
import logging
import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from typing import Dict, List, Optional, Any, Union, Tuple

# 内部モジュール
import sys
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.append(sys_path)
    
from backtest.portfolio import SimplePortfolio

# ロギング設定
logger = logging.getLogger(__name__)

def plot_returns_with_btc(portfolio: SimplePortfolio, save_path: Optional[str] = None) -> plt.Figure:
    """
    リターン曲線をBTC価格と一緒にプロット

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
    
    # BTCの価格データを取得
    try:
        # ポートフォリオのインデックスから日付範囲を取得
        if hasattr(portfolio.df.index, 'min') and hasattr(portfolio.df.index, 'max'):
            start_date = portfolio.df.index.min()
            end_date = portfolio.df.index.max()
            
            # インデックスがdatetime型でない場合は変換をスキップ
            btc_price = None
            if 'price_close' in portfolio.df.columns:
                # 既に価格データがある場合はそれを使用
                btc_price = portfolio.df['price_close']
                logger.info(f"既存の価格データを使用 (price_close): {len(btc_price)}行")
            elif 'price' in portfolio.df.columns:
                # price列がある場合はそれを使用
                btc_price = portfolio.df['price']
                logger.info(f"既存の価格データを使用 (price): {len(btc_price)}行")
        else:
            btc_price = None
            logger.warning("ポートフォリオのインデックスからは日付範囲を取得できません")
    except Exception as e:
        logger.error(f"BTCの価格データ取得中にエラー発生: {str(e)}")
        btc_price = None
    
    # グラフ作成
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # リターン曲線プロット
    if 'cumulative_returns' in portfolio.df.columns:
        portfolio.df['cumulative_returns'].plot(ax=ax1, color='blue', label='Strategy Returns')
        
        # 開始値と終了値をテキスト表示
        start_value = portfolio.df['cumulative_returns'].iloc[0]
        end_value = portfolio.df['cumulative_returns'].iloc[-1]
        
        # プロット開始位置と終了位置に値をテキスト表示
        ax1.text(portfolio.df.index[0], start_value, f'{start_value:.2f}',
                verticalalignment='bottom', horizontalalignment='left')
        ax1.text(portfolio.df.index[-1], end_value, f'{end_value:.2f}',
                verticalalignment='bottom', horizontalalignment='right')
    else:
        logger.warning("cumulative_returns列が見つかりません")
    
    ax1.set_ylabel('Strategy Returns', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    # BTC価格のプロット（価格データがある場合）
    if btc_price is not None and len(btc_price) > 0:
        ax2 = ax1.twinx()  # 二次Y軸を作成
        btc_price.plot(ax=ax2, color='red', alpha=0.7, label='BTC Price')
        ax2.set_ylabel('BTC Price (USD)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 凡例の設定
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')
    
    ax1.set_title('Cumulative Returns with BTC Price')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig

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
        
        # インデックスがdatetimeでない場合は変換をスキップ
        if not pd.api.types.is_datetime64_any_dtype(returns.index):
            logger.warning("インデックスがdatetimeタイプではないため、月次リターンヒートマップを作成できません")
            return None
            
        # 月次リターン集計
        monthly_returns = returns.resample('M').sum() * 100  # パーセント表示
        
        try:
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
        except Exception as e:
            logger.error(f"月次リターンヒートマップ作成中にエラー発生: {str(e)}")
            return None
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
    
    # バックテスト期間の表示（インデックスの型によって処理を分ける）
    if hasattr(portfolio, 'df') and len(portfolio.df) > 0:
        start_idx = portfolio.df.index[0]
        end_idx = portfolio.df.index[-1]
        
        # datetime型の場合はフォーマット
        if pd.api.types.is_datetime64_any_dtype(portfolio.df.index):
            start_str = start_idx.strftime('%Y-%m-%d')
            end_str = end_idx.strftime('%Y-%m-%d')
        else:
            start_str = str(start_idx)
            end_str = str(end_idx)
            
        table_data.append(["バックテスト期間", f"{start_str} - {end_str}"])
    
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
    
    # リターン曲線 (通常版)
    returns_path = os.path.join(save_dir, f"{interval}_returns.png") if save_dir else None
    plots['returns'] = plot_returns(portfolio, returns_path)
    
    # リターン曲線 (BTC価格付き)
    returns_btc_path = os.path.join(save_dir, f"{interval}_returns_with_btc.png") if save_dir else None
    plots['returns_with_btc'] = plot_returns_with_btc(portfolio, returns_btc_path)
    
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

def load_portfolio_and_generate_plots(interval: str, timestamp: Optional[str] = None) -> Tuple[SimplePortfolio, Dict[str, plt.Figure]]:
    """
    ポートフォリオをロードしてプロットを生成する

    Parameters:
    -----------
    interval : str
        時間枠
    timestamp : str, optional
        タイムスタンプディレクトリ (省略時は最新)
        
    Returns:
    --------
    Tuple[SimplePortfolio, Dict[str, plt.Figure]]
        ポートフォリオとプロット図のタプル
    """
    # プロジェクトルートディレクトリ
    root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    reports_dir = root_dir / 'reports'
    
    # レポートディレクトリ
    interval_dir = reports_dir / interval
    
    if not interval_dir.exists():
        logger.error(f"レポートディレクトリが見つかりません: {interval_dir}")
        return None, {}
    
    # タイムスタンプディレクトリを検索
    if timestamp:
        target_dir = interval_dir / timestamp
        if not target_dir.exists():
            logger.error(f"指定されたタイムスタンプディレクトリが見つかりません: {target_dir}")
            return None, {}
    else:
        # 最新のタイムスタンプディレクトリを探す
        timestamp_dirs = [d for d in interval_dir.iterdir() if d.is_dir()]
        if not timestamp_dirs:
            # タイムスタンプディレクトリがない場合は、直接ポートフォリオファイルを探す
            portfolio_file = interval_dir / "backtest_portfolio.pkl"
        else:
            # 最新のディレクトリを使用
            target_dir = sorted(timestamp_dirs)[-1]
            portfolio_file = target_dir / "backtest_portfolio.pkl"
    
    # ポートフォリオファイル探索
    if 'portfolio_file' not in locals():
        # まずはタイムスタンプディレクトリでポートフォリオファイルを探す
        portfolio_files = list(target_dir.glob("*portfolio*.pkl"))
        if portfolio_files:
            portfolio_file = portfolio_files[0]
        else:
            # 見つからない場合は親ディレクトリで探す
            portfolio_files = list(interval_dir.glob("*portfolio*.pkl"))
            if portfolio_files:
                portfolio_file = portfolio_files[0]
            else:
                logger.error("ポートフォリオファイルが見つかりません")
                return None, {}
    
    # ポートフォリオをロード
    try:
        with open(portfolio_file, 'rb') as f:
            portfolio = pickle.load(f)
        logger.info(f"ポートフォリオをロード: {portfolio_file}")
    except Exception as e:
        logger.error(f"ポートフォリオのロード中にエラー発生: {str(e)}")
        return None, {}
    
    # 保存先ディレクトリの設定
    save_dir = interval_dir / "plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # プロット生成
    plots = generate_all_plots(portfolio, interval, str(save_dir))
    
    return portfolio, plots

if __name__ == "__main__":
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # コマンドライン引数解析
    import argparse
    parser = argparse.ArgumentParser(description='バックテスト結果プロット生成')
    parser.add_argument('--interval', '-i', type=str, required=True, help='時間枠 (例: 15m, 2h)')
    parser.add_argument('--timestamp', '-t', type=str, help='タイムスタンプ (省略時は最新)')
    args = parser.parse_args()
    
    # プロット生成実行
    portfolio, plots = load_portfolio_and_generate_plots(args.interval, args.timestamp)
    
    if portfolio is not None:
        logger.info(f"{args.interval}の{len(plots)}個のプロットを生成しました")
    else:
        logger.error("プロット生成に失敗しました")
