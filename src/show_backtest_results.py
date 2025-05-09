#!/usr/bin/env python
"""
show_backtest_results.py - バックテスト結果表示スクリプト

保存されたバックテスト結果を読み込んで表示します。
"""

import os
import sys
import pickle
import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tabulate import tabulate

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
logger = logging.getLogger('show_results')

def load_portfolio(file_path, is_dir=False):
    """ポートフォリオオブジェクトを読み込む"""
    try:
        if is_dir:
            # 日付ディレクトリの場合は、内部のbacktest_portfolio.pklを読み込む
            portfolio_path = Path(file_path) / "backtest_portfolio.pkl"
            if not portfolio_path.exists():
                # バックワードコンパティビリティのために慣例的な名前も確認
                alt_files = list(Path(file_path).glob("backtest_portfolio_*.pkl"))
                if alt_files:
                    portfolio_path = alt_files[0]
                else:
                    raise FileNotFoundError(f"ポートフォリオファイルが見つかりません: {file_path}")
        else:
            portfolio_path = file_path
            
        with open(portfolio_path, 'rb') as f:
            portfolio = pickle.load(f)
        return portfolio
    except Exception as e:
        logger.error(f"ポートフォリオ読み込みエラー: {e}")
        return None
        
def load_stats_from_json(file_path, is_dir=False):
    """JSONからバックテスト統計情報を読み込む"""
    try:
        if is_dir:
            # 日付ディレクトリの場合は、内部のstats.jsonを読み込む
            stats_path = Path(file_path) / "stats.json"
            if not stats_path.exists():
                logger.warning(f"統計情報ファイルが見つかりません: {stats_path}")
                return None
        else:
            # 直接JSONファイルを指定された場合
            stats_path = file_path
            
        # JSONファイルを読み込んでSeriesに変換
        with open(stats_path, 'r', encoding='utf-8') as f:
            import json
            stats_dict = json.load(f)
            stats = pd.Series(stats_dict)
        return stats
    except Exception as e:
        logger.error(f"統計情報読み込みエラー: {e}")
        return None

def show_stats(portfolio, interval):
    """統計情報を表示"""
    if portfolio is None:
        return
    
    stats = portfolio.stats()
    
    # データフレームに変換して表示
    stats_df = pd.DataFrame({'指標': stats.index, f'{interval}': stats.values})
    
    # 表として整形して表示
    print("\n=== バックテスト結果 ===")
    print(f"時間枠: {interval}")
    print(tabulate(stats_df, headers='keys', tablefmt='psql', showindex=False))

    # 特に重要な指標を個別表示
    if 'Total Return [%]' in stats:
        print(f"\n総リターン: {stats['Total Return [%]']:.2f}%")
    if 'Annual Return [%]' in stats:
        print(f"年間リターン: {stats['Annual Return [%]']:.2f}%")
    if 'Sharpe Ratio' in stats:
        print(f"シャープレシオ: {stats['Sharpe Ratio']:.2f}")
    if 'Max Drawdown [%]' in stats:
        print(f"最大ドローダウン: {stats['Max Drawdown [%]']:.2f}%")
        
    # MAR比率（CAGR / 最大ドローダウン）の計算
    if 'Annual Return [%]' in stats and 'Max Drawdown [%]' in stats:
        annual_return = stats['Annual Return [%]'] / 100  # パーセントから小数に変換
        max_dd = abs(stats['Max Drawdown [%]'] / 100)  # パーセントから小数に変換、絶対値を取る
        mar_ratio = annual_return / max_dd if max_dd != 0 else float('inf')
        print(f"MAR比率 (年間リターン/最大DD): {mar_ratio:.2f}")

    # 合格基準の確認
    if 'Sharpe Ratio' in stats and 'Max Drawdown [%]' in stats:
        sharpe = stats['Sharpe Ratio']
        max_dd = stats['Max Drawdown [%]']
        
        # MAR比率の再計算（上で計算したものを使用）
        meets_criteria = (
            sharpe > 1.0 and
            mar_ratio >= 0.5 and
            abs(max_dd) < 25.0
        )
        
        print("\n=== 合格基準評価 ===")
        print(f"合格基準を満たしている: {'はい' if meets_criteria else 'いいえ'}")
        print(f"- Sharpe > 1: {'✓' if sharpe > 1.0 else '✗'} ({sharpe:.2f})")
        print(f"- MAR ≥ 0.5: {'✓' if mar_ratio >= 0.5 else '✗'} ({mar_ratio:.2f})")
        print(f"- MaxDD < 25%: {'✓' if abs(max_dd) < 25.0 else '✗'} ({abs(max_dd):.2f}%)")

def show_stats_from_series(stats, interval):
    """Seriesから統計情報を表示"""
    # データフレームに変換して表示
    stats_df = pd.DataFrame({'指標': stats.index, f'{interval}': stats.values})
    
    # 表として整形して表示
    print("\n=== バックテスト結果 ===")
    print(f"時間枠: {interval}")
    print(tabulate(stats_df, headers='keys', tablefmt='psql', showindex=False))

    # 特に重要な指標を個別表示
    if 'Total Return [%]' in stats:
        print(f"\n総リターン: {stats['Total Return [%]']:.2f}%")
    if 'Annual Return [%]' in stats:
        print(f"年間リターン: {stats['Annual Return [%]']:.2f}%")
    if 'Sharpe Ratio' in stats:
        print(f"シャープレシオ: {stats['Sharpe Ratio']:.2f}")
    if 'Max Drawdown [%]' in stats:
        print(f"最大ドローダウン: {stats['Max Drawdown [%]']:.2f}%")
        
    # MAR比率（CAGR / 最大ドローダウン）の計算
    if 'Annual Return [%]' in stats and 'Max Drawdown [%]' in stats:
        annual_return = stats['Annual Return [%]'] / 100  # パーセントから小数に変換
        max_dd = abs(stats['Max Drawdown [%]'] / 100)  # パーセントから小数に変換、絶対値を取る
        mar_ratio = annual_return / max_dd if max_dd != 0 else float('inf')
        print(f"MAR比率 (年間リターン/最大DD): {mar_ratio:.2f}")

    # 合格基準の確認
    if 'Sharpe Ratio' in stats and 'Max Drawdown [%]' in stats:
        sharpe = stats['Sharpe Ratio']
        max_dd = stats['Max Drawdown [%]']
        
        # MAR比率の再計算（上で計算したものを使用）
        meets_criteria = (
            sharpe > 1.0 and
            mar_ratio >= 0.5 and
            abs(max_dd) < 25.0
        )
        
        print("\n=== 合格基準評価 ===")
        print(f"合格基準を満たしている: {'はい' if meets_criteria else 'いいえ'}")
        print(f"- Sharpe > 1: {'✓' if sharpe > 1.0 else '✗'} ({sharpe:.2f})")
        print(f"- MAR ≥ 0.5: {'✓' if mar_ratio >= 0.5 else '✗'} ({mar_ratio:.2f})")
        print(f"- MaxDD < 25%: {'✓' if abs(max_dd) < 25.0 else '✗'} ({abs(max_dd):.2f}%)")

def save_plot(portfolio, interval, output_dir):
    """プロットを生成して保存"""
    if portfolio is None or not hasattr(portfolio, 'plot'):
        return
    
    # リターン曲線をプロット
    try:
        plt.figure(figsize=(12, 6))
        portfolio.df['cumulative_returns'].plot()
        plt.title(f'{interval} Cumulative Returns')
        plt.grid(True, alpha=0.3)
        
        # 保存先ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        # プロットを保存
        plot_path = os.path.join(output_dir, f'{interval}_returns.png')
        plt.savefig(plot_path)
        logger.info(f"プロット保存: {plot_path}")
        
        # ドローダウンプロット
        plt.figure(figsize=(12, 6))
        running_max = portfolio.df['cumulative_returns'].cummax()
        drawdown = (portfolio.df['cumulative_returns'] / running_max - 1) * 100
        drawdown.plot()
        plt.title(f'{interval} Drawdowns (%)')
        plt.grid(True, alpha=0.3)
        
        # ドローダウンプロットを保存
        dd_plot_path = os.path.join(output_dir, f'{interval}_drawdowns.png')
        plt.savefig(dd_plot_path)
        logger.info(f"ドローダウンプロット保存: {dd_plot_path}")
        
    except Exception as e:
        logger.error(f"プロット生成エラー: {e}")

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='バックテスト結果表示ツール')
    parser.add_argument('--interval', '-i', type=str, help='時間枠 (15m, 1h, 2h など)')
    parser.add_argument('--file', '-f', type=str, help='特定のファイルまたは日付ディレクトリを指定 (省略時は最新)')
    parser.add_argument('--plot', '-p', action='store_true', help='リターンとドローダウンの図を生成して保存')
    parser.add_argument('--all', '-a', action='store_true', help='すべての時間枠の結果を表示')
    args = parser.parse_args()
    
    if args.all:
        # すべての時間枠を処理
        intervals = ['15m', '2h']
        for interval in intervals:
            process_interval(interval, args.file, args.plot)
    else:
        # 指定された時間枠のみ処理
        if args.interval is None:
            print("エラー: --interval引数または--all引数を指定してください。")
            parser.print_help()
            return 1
        process_interval(args.interval, args.file, args.plot)
    
    return 0

def process_interval(interval, file=None, plot=False):
    """指定された時間枠のバックテスト結果を処理"""
    # レポートディレクトリ
    reports_dir = app_home / 'reports' / interval
    
    if not os.path.exists(reports_dir):
        logger.error(f"レポートディレクトリが見つかりません: {reports_dir}")
        return
    
    # バックテスト結果ディレクトリまたはファイルを検索
    if file:
        file_path = os.path.join(reports_dir, file)
        if not os.path.exists(file_path):
            logger.error(f"指定されたファイル/ディレクトリが見つかりません: {file_path}")
            return
        
        # ファイルの種類を判定
        if os.path.isdir(file_path):
            # ディレクトリの場合
            is_dir = True
        else:
            # ファイルの場合
            is_dir = False
    else:
        # 最新のタイムスタンプディレクトリを探す
        timestamp_dirs = sorted([d for d in reports_dir.glob("*") if d.is_dir() and d.name[0].isdigit()])
        if timestamp_dirs:
            file_path = timestamp_dirs[-1]
            is_dir = True
        else:
            # 旧形式のファイルを探す
            portfolio_files = sorted(list(reports_dir.glob("backtest_portfolio_*.pkl")))
            if not portfolio_files:
                logger.error(f"バックテスト結果が見つかりません: {reports_dir}")
                return
            file_path = portfolio_files[-1]
            is_dir = False
    
    logger.info(f"バックテスト結果: {file_path}")
    
    # 統計情報を読み込む方法を決定
    stats = None
    portfolio = None
    
    # まずJSONからの読み込みを試みる
    if is_dir:
        stats = load_stats_from_json(file_path, is_dir=True)
    elif file_path.name.endswith('.json'):
        stats = load_stats_from_json(file_path)
    
    # 統計情報が読み込めなかった場合、またはpklファイルが指定された場合はポートフォリオを読み込む
    if stats is None or not is_dir:
        portfolio = load_portfolio(file_path, is_dir=is_dir)
        if portfolio is None and stats is None:
            logger.error("統計情報もポートフォリオも読み込めませんでした")
            return
    
    # 統計情報を表示
    if stats is not None:
        show_stats_from_series(stats, interval)
    elif portfolio is not None:
        show_stats(portfolio, interval)
    
    # プロット生成（オプション）
    if plot and portfolio:
        output_dir = app_home / 'reports' / 'plots'
        save_plot(portfolio, interval, output_dir)
    elif plot and is_dir:
        # 画像ファイルをチェック
        returns_file = Path(file_path) / "returns.png"
        drawdowns_file = Path(file_path) / "drawdowns.png"
        if returns_file.exists():
            logger.info(f"リターングラフ: {returns_file}")
        if drawdowns_file.exists():
            logger.info(f"ドローダウングラフ: {drawdowns_file}")

if __name__ == "__main__":
    sys.exit(main())
