#!/usr/bin/env python
"""
run_backtest.py - バックテスト実行スクリプト

機能:
- 指定された時間枠のモデルに対してバックテストを実行
- バックテスト結果の保存とプロット
- パフォーマンス指標の計算と出力
"""

import os
import sys
import json
import argparse
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# プロジェクトのルートディレクトリをパスに追加
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT_DIR))

from src.backtest import BacktestRunner

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(ROOT_DIR, 'logs', 'run_backtest.log'), mode='a')
    ]
)
logger = logging.getLogger('run_backtest')

# ディレクトリ設定
REPORTS_DIR = ROOT_DIR / 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)


def run_backtest(interval: str, model_type: str = 'lightgbm', 
                model_date: Optional[str] = None, threshold: float = 0.55,
                trend_filter: bool = False) -> Dict[str, Any]:
    """バックテストを実行する

    Args:
        interval: 時間枠 ('15m', '2h' など)
        model_type: モデル種類 ('lightgbm' または 'catboost')
        model_date: モデル日付 (省略時は最新)
        threshold: シグナル閾値 (0.5～1.0)
        trend_filter: 日足トレンドフィルター適用

    Returns:
        Dict[str, Any]: バックテスト結果
    """
    logger.info(f"バックテスト開始: 時間枠={interval}, モデル={model_type}, 閾値={threshold}")
    
    # BacktestRunner インスタンス作成
    runner = BacktestRunner()
    
    # トレンドフィルター設定
    runner.mtf_config = {"use_trend_filter": trend_filter}
    if trend_filter:
        logger.info("日足トレンドフィルター有効")
    
    try:
        # バックテスト実行
        portfolio, stats = runner.run_interval_backtest(
            interval, model_type, model_date, threshold
        )
        
        # 結果ディレクトリ作成
        report_dir = REPORTS_DIR / interval
        os.makedirs(report_dir, exist_ok=True)
        
        # タイムスタンプ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 統計情報の保存
        stats_file = report_dir / f"stats_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            if hasattr(stats, 'to_dict'):
                # pd.Series の場合
                json.dump(stats.to_dict(), f, indent=2)
            else:
                # 辞書の場合
                json.dump(stats, f, indent=2)
        
        logger.info(f"統計情報を保存しました: {stats_file}")
        
        # グラフの描画と保存
        if hasattr(portfolio, 'plot'):
            # リターン曲線
            plt.figure(figsize=(12, 6))
            fig = portfolio.plot()
            plot_file = report_dir / f"returns_{timestamp}.png"
            fig.savefig(plot_file)
            logger.info(f"リターン曲線を保存しました: {plot_file}")
            plt.close(fig)
            
            # ドローダウン曲線
            plt.figure(figsize=(12, 6))
            fig = portfolio.plot_drawdowns()
            dd_file = report_dir / f"drawdowns_{timestamp}.png"
            fig.savefig(dd_file)
            logger.info(f"ドローダウン曲線を保存しました: {dd_file}")
            plt.close(fig)
        
        # キー指標の表示
        logger.info(f"バックテスト結果 ({interval}):")
        
        # stats から主要統計情報を抽出して表示
        if isinstance(stats, dict):
            # 辞書形式
            stats_dict = stats
        else:
            # pd.Series または他の形式
            stats_dict = stats.to_dict() if hasattr(stats, 'to_dict') else {}
        
        # 主要指標の表示
        metrics_to_display = [
            'Total Return [%]', 'Annual Return [%]', 'Sharpe Ratio', 
            'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown [%]'
        ]
        
        for metric in metrics_to_display:
            if metric in stats_dict:
                logger.info(f"  {metric}: {stats_dict[metric]}")
        
        # バックテスト結果を返す
        return {
            'portfolio': portfolio,
            'stats': stats
        }
    
    except Exception as e:
        logger.error(f"バックテスト実行エラー: {str(e)}")
        logger.exception(e)
        return {
            'error': str(e)
        }


def main():
    """コマンドライン実行用のメイン関数"""
    parser = argparse.ArgumentParser(description='バックテスト実行ツール')
    parser.add_argument('--interval', '-i', type=str, required=True, 
                        help='時間枠 (例: 15m, 2h)')
    parser.add_argument('--model-type', '-m', type=str, default='lightgbm',
                        choices=['lightgbm', 'catboost'], help='モデル種類')
    parser.add_argument('--model-date', '-d', type=str, help='モデル日付 (省略時は最新)')
    parser.add_argument('--threshold', '-t', type=float, default=0.55,
                        help='シグナル閾値 (0.5～1.0)')
    parser.add_argument('--trend-filter', '-f', action='store_true',
                        help='日足トレンドフィルター適用')
    args = parser.parse_args()
    
    # シグナル閾値のバリデーション
    if not 0.5 <= args.threshold <= 1.0:
        logger.warning(f"シグナル閾値が範囲外です: {args.threshold}、0.55 を使用します")
        threshold = 0.55
    else:
        threshold = args.threshold
    
    # バックテスト実行
    result = run_backtest(
        args.interval, 
        args.model_type, 
        args.model_date, 
        threshold,
        args.trend_filter
    )
    
    if 'error' in result:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
