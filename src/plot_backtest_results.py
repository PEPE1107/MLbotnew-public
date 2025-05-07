#!/usr/bin/env python
"""
plot_backtest_results.py - バックテスト結果をグラフ化して保存するスクリプト

機能:
- リターン曲線のグラフ化
- ドローダウンのグラフ化
- 月次/週次パフォーマンスのヒートマップ
- 主要指標のサマリーテーブル
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tabulate import tabulate
from datetime import datetime

# アプリケーションルートへのパスを設定
app_home = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(app_home))

# src ディレクトリをインポートパスに追加
src_path = app_home / "src"
sys.path.append(str(src_path))

# バックテストモジュールからSimplePortfolioクラスをインポート
from backtest import SimplePortfolio

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('plot_results')

def load_portfolio(file_path):
    """ポートフォリオオブジェクトを読み込む"""
    try:
        with open(file_path, 'rb') as f:
            portfolio = pickle.load(f)
        return portfolio
    except Exception as e:
        logger.error(f"ポートフォリオ読み込みエラー: {e}")
        return None

def setup_plot_style():
    """プロットのスタイル設定"""
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 12
    
    # フォントを日本語対応に設定（利用可能なフォントがあれば使用）
    try:
        plt.rcParams['font.family'] = 'Yu Gothic'
    except:
        try:
            plt.rcParams['font.family'] = 'MS Gothic'
        except:
            pass  # フォール バック: デフォルトフォントを使用

def plot_returns(portfolio, interval, output_dir):
    """リターン曲線をプロット"""
    plt.figure(figsize=(14, 8))
    
    # リターン曲線
    ax = portfolio.df['cumulative_returns'].plot(color='#1f77b4', linewidth=2)
    
    # グラフの設定
    plt.title(f'{interval} 時間枠 累積リターン', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylabel('累積リターン (1 = 元本)', fontsize=14)
    
    # X軸の日付フォーマットを設定
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # 水平線を追加（元本を表す）
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)
    
    # 実績をテキストで表示
    stats = portfolio.stats()
    annual_return = stats['Annual Return [%]']
    sharpe = stats['Sharpe Ratio']
    max_dd = stats['Max Drawdown [%]']
    
    text_info = f"年間リターン: {annual_return:.2f}%\nシャープレシオ: {sharpe:.2f}\n最大DD: {max_dd:.2f}%"
    plt.annotate(text_info, xy=(0.02, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                 fontsize=12, verticalalignment='top')
    
    # 保存先ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # プロットを保存
    plot_path = os.path.join(output_dir, f'{interval}_returns.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"リターンプロット保存: {plot_path}")
    plt.close()

def plot_drawdowns(portfolio, interval, output_dir):
    """ドローダウンをプロット"""
    plt.figure(figsize=(14, 8))
    
    # ドローダウン計算
    running_max = portfolio.df['cumulative_returns'].cummax()
    drawdown = (portfolio.df['cumulative_returns'] / running_max - 1) * 100
    
    # ドローダウンプロット
    drawdown.plot(color='#d62728', linewidth=2)
    
    # グラフの設定
    plt.title(f'{interval} 時間枠 ドローダウン', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylabel('ドローダウン (%)', fontsize=14)
    
    # 最大ドローダウンをマーク
    min_idx = drawdown.idxmin()
    min_value = drawdown.min()
    plt.plot(min_idx, min_value, 'ro')
    plt.annotate(f'最大DD: {min_value:.2f}%', 
                xy=(min_idx, min_value),
                xytext=(min_idx, min_value*0.8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # 保存
    plot_path = os.path.join(output_dir, f'{interval}_drawdowns.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"ドローダウンプロット保存: {plot_path}")
    plt.close()

def plot_monthly_returns(portfolio, interval, output_dir):
    """月次リターンのヒートマップをプロット"""
    # データフレームを取得
    df = portfolio.df.copy()
    
    # 日次リターンに変換
    if 'strategy_returns' in df.columns:
        returns = df['strategy_returns']
    else:
        returns = df['cumulative_returns'].pct_change().fillna(0)
    
    # 日付インデックスを確認、必要に応じて変換
    if not isinstance(returns.index, pd.DatetimeIndex):
        # 数値インデックスの場合は、適当な開始日から日付を生成
        start_date = pd.Timestamp('2020-01-01')
        dates = pd.date_range(start=start_date, periods=len(returns))
        returns.index = dates
    
    # 月次リターン表を作成
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1).to_frame()
    monthly_returns.columns = ['return']
    
    # 年と月の列を追加
    monthly_returns['year'] = monthly_returns.index.year
    monthly_returns['month'] = monthly_returns.index.month
    
    # ピボットテーブル作成
    pivot_table = monthly_returns.pivot_table(index='year', columns='month', values='return')
    
    # 月の名前に変換
    month_names = {1: '1月', 2: '2月', 3: '3月', 4: '4月', 5: '5月', 6: '6月',
                   7: '7月', 8: '8月', 9: '9月', 10: '10月', 11: '11月', 12: '12月'}
    pivot_table.columns = [month_names[m] for m in pivot_table.columns]
    
    # パーセントに変換
    pivot_table = pivot_table * 100
    
    # ヒートマップ作成
    plt.figure(figsize=(14, 8))
    
    # カスタムカラーマップ (緑=利益、赤=損失)
    cmap = LinearSegmentedColormap.from_list('rg', ["#d62728", "#FFFFFF", "#2ca02c"], N=256)
    
    ax = sns.heatmap(pivot_table, annot=True, cmap=cmap, center=0, fmt='.1f',
                    linewidths=0.5, cbar_kws={'label': 'リターン (%)'})
    
    plt.title(f'{interval} 時間枠 月次リターン (%)', fontsize=16)
    plt.ylabel('年', fontsize=14)
    
    # 保存
    plot_path = os.path.join(output_dir, f'{interval}_monthly_returns.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"月次リターンヒートマップ保存: {plot_path}")
    plt.close()

def plot_summary_table(portfolio, interval, output_dir):
    """主要指標のサマリーテーブルを画像として保存"""
    plt.figure(figsize=(10, 6))
    
    # Canvas をアックス無しでプロット (表のみ)
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # 統計データ取得
    stats = portfolio.stats()
    
    # MAR比率（CAGR / 最大ドローダウン）の計算
    annual_return = stats.get('Annual Return [%]', 0) / 100  # パーセントから小数に変換
    max_dd = abs(stats.get('Max Drawdown [%]', 0) / 100)  # パーセントから小数に変換、絶対値を取る
    mar_ratio = annual_return / max_dd if max_dd != 0 else float('inf')
    
    # 表示する指標とその値を選択
    summary_data = {
        '指標': ['年間リターン', 'シャープレシオ', 'ソルティノレシオ', '最大ドローダウン', 'MAR比率', 
                'Win率', '平均勝ちトレード', '平均負けトレード', 'プロフィットファクター'],
        '値': [
            f"{stats.get('Annual Return [%]', 0):.2f}%",
            f"{stats.get('Sharpe Ratio', 0):.2f}",
            f"{stats.get('Sortino Ratio', 0):.2f}" if 'Sortino Ratio' in stats else "N/A",
            f"{stats.get('Max Drawdown [%]', 0):.2f}%",
            f"{mar_ratio:.2f}",
            f"{stats.get('Win Rate [%]', 0):.2f}%" if 'Win Rate [%]' in stats else "N/A",
            f"{stats.get('Avg Winning Trade [%]', 0):.2f}%" if 'Avg Winning Trade [%]' in stats else "N/A",
            f"{stats.get('Avg Losing Trade [%]', 0):.2f}%" if 'Avg Losing Trade [%]' in stats else "N/A",
            f"{stats.get('Profit Factor', 0):.2f}" if 'Profit Factor' in stats else "N/A"
        ]
    }
    
    # データフレーム作成
    summary_df = pd.DataFrame(summary_data)
    
    # テーブルとして表示
    table = plt.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.4]
    )
    
    # テーブルのスタイル設定
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # タイトル
    plt.title(f'{interval} 時間枠 パフォーマンスサマリー', fontsize=16, y=0.8)
    
    # 保存
    plot_path = os.path.join(output_dir, f'{interval}_summary_table.png')
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    logger.info(f"サマリーテーブル保存: {plot_path}")
    plt.close()

def plot_signals_distribution(portfolio, interval, output_dir):
    """シグナルの分布を円グラフでプロット (可能な場合)"""
    # ポートフォリオデータフレーム取得
    df = portfolio.df.copy()
    
    # シグナルカラムが存在するか確認
    if 'signal' in df.columns:
        plt.figure(figsize=(10, 8))
        
        # シグナルの分布を計算
        signal_counts = df['signal'].value_counts()
        
        # シグナルを分かりやすいラベルに変換
        signal_map = {1: 'ロング', -1: 'ショート', 0: 'ニュートラル'}
        signal_counts.index = [signal_map.get(s, str(s)) for s in signal_counts.index]
        
        # 色の設定
        colors = ['#2ca02c', '#d62728', '#7f7f7f']
        
        # 円グラフ作成
        plt.pie(signal_counts, labels=signal_counts.index, autopct='%1.1f%%',
                startangle=90, colors=colors, explode=[0.05] * len(signal_counts))
        
        plt.axis('equal')  # 円を円形に保つ
        plt.title(f'{interval} 時間枠 シグナル分布', fontsize=16)
        
        # 保存
        plot_path = os.path.join(output_dir, f'{interval}_signal_distribution.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        logger.info(f"シグナル分布プロット保存: {plot_path}")
        plt.close()

def generate_all_plots(portfolio, interval, output_dir):
    """すべてのプロットを生成"""
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # リターン曲線
        plot_returns(portfolio, interval, output_dir)
        
        # ドローダウン
        plot_drawdowns(portfolio, interval, output_dir)
        
        # 月次リターンヒートマップ
        try:
            plot_monthly_returns(portfolio, interval, output_dir)
        except Exception as e:
            logger.error(f"月次リターンプロット生成エラー: {e}")
        
        # サマリーテーブル
        plot_summary_table(portfolio, interval, output_dir)
        
        # シグナル分布 (可能な場合)
        try:
            plot_signals_distribution(portfolio, interval, output_dir)
        except Exception as e:
            logger.error(f"シグナル分布プロット生成エラー: {e}")
            
    except Exception as e:
        logger.error(f"プロット生成エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='バックテスト結果をグラフ化して保存するツール')
    parser.add_argument('--interval', '-i', type=str, help='時間枠 (15m, 1h, 2h など)')
    parser.add_argument('--file', '-f', type=str, help='特定のファイルを指定 (省略時は最新)')
    parser.add_argument('--output-dir', '-o', type=str, default=str(app_home / 'reports' / 'plots'),
                        help='プロットの保存先ディレクトリ')
    parser.add_argument('--all', '-a', action='store_true', help='すべての時間枠の結果を処理')
    args = parser.parse_args()
    
    # プロットのスタイル設定
    setup_plot_style()
    
    if args.all:
        # すべての時間枠を処理
        intervals = ['15m', '2h']
        for interval in intervals:
            logger.info(f"時間枠 {interval} の処理開始")
            process_interval(interval, args.file, args.output_dir)
    else:
        # 指定された時間枠のみ処理
        if args.interval is None:
            logger.error("エラー: --interval引数または--all引数を指定してください。")
            parser.print_help()
            return 1
        process_interval(args.interval, args.file, args.output_dir)
    
    return 0

def process_interval(interval, file=None, output_dir=None):
    """指定された時間枠のバックテスト結果を処理"""
    # レポートディレクトリ
    reports_dir = app_home / 'reports' / interval
    
    if not os.path.exists(reports_dir):
        logger.error(f"レポートディレクトリが見つかりません: {reports_dir}")
        return
    
    # バックテスト結果ファイルを検索
    if file:
        file_path = os.path.join(reports_dir, file)
        if not os.path.exists(file_path):
            logger.error(f"指定されたファイルが見つかりません: {file_path}")
            return
        portfolio_files = [file_path]
    else:
        portfolio_files = sorted(list(reports_dir.glob("backtest_portfolio_*.pkl")))
        if not portfolio_files:
            logger.error(f"バックテスト結果ファイルが見つかりません: {reports_dir}")
            return
        # 最新のファイルを使用
        file_path = portfolio_files[-1]
    
    logger.info(f"バックテスト結果ファイル: {file_path}")
    
    # ポートフォリオを読み込む
    portfolio = load_portfolio(file_path)
    if portfolio is None:
        return
    
    # プロットを生成して保存
    if output_dir is None:
        output_dir = app_home / 'reports' / 'plots'
    
    generate_all_plots(portfolio, interval, output_dir)
    logger.info(f"時間枠 {interval} のプロット生成完了")

if __name__ == "__main__":
    sys.exit(main())
