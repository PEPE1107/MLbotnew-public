#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
バックテスト実行スクリプト
-----------------------

MLbotnewのバックテストを実行し、結果をタイムスタンプ付きディレクトリに保存します。
また、最新結果への簡易アクセス用にシンボリックリンクも作成します。

使用方法:
    python scripts/run_backtest.py [--interval INTERVAL] [--quick] [--no-plot]
    
オプション:
    --interval: バックテストの時間枠 (15m/2h/1d)（デフォルト: 15m）
    --quick: クイックモード（CIなど高速実行用）
    --no-plot: プロット生成をスキップ
"""

import os
import sys
import argparse
import logging
import datetime
import json
import shutil
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# モジュールインポート
from src.backtest.run import run_backtest
from src.backtest.utils import generate_stats, plot_backtest_with_price

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_result_directory(interval: str) -> Tuple[str, str]:
    """
    タイムスタンプ付きの結果ディレクトリを作成
    
    Parameters:
    -----------
    interval : str
        バックテストの時間枠 (15m/2h/1d)
        
    Returns:
    --------
    tuple
        (結果ディレクトリパス, タイムスタンプ)
    """
    # 現在時刻からタイムスタンプを生成
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 時間枠ディレクトリが存在しない場合は作成
    interval_dir = os.path.join(project_root, 'reports', interval)
    os.makedirs(interval_dir, exist_ok=True)
    
    # タイムスタンプ付きディレクトリを作成
    result_dir = os.path.join(interval_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    
    logger.info(f"結果ディレクトリを作成しました: {result_dir}")
    return result_dir, timestamp

def update_latest_symlinks(result_dir: str, interval: str) -> None:
    """
    最新結果への簡易アクセス用にファイルをコピー
    
    Parameters:
    -----------
    result_dir : str
        タイムスタンプ付き結果ディレクトリパス
    interval : str
        バックテストの時間枠 (15m/2h/1d)
    """
    interval_dir = os.path.join(project_root, 'reports', interval)
    
    # ソースファイルパス
    stats_src = os.path.join(result_dir, 'stats.json')
    html_src = os.path.join(result_dir, 'bt_with_price.html')
    
    # 1. 互換性のため従来の場所にもコピー
    stats_dst = os.path.join(interval_dir, 'stats.json')
    html_dst = os.path.join(interval_dir, 'bt_with_price.html')
    
    if os.path.exists(stats_src):
        shutil.copy2(stats_src, stats_dst)
        logger.info(f"統計情報をコピーしました: {stats_dst}")
    
    if os.path.exists(html_src):
        shutil.copy2(html_src, html_dst)
        logger.info(f"HTMLレポートをコピーしました: {html_dst}")
    
    # 2. latest ディレクトリを作成してそこにもコピー（新機能）
    latest_dir = os.path.join(interval_dir, 'latest')
    os.makedirs(latest_dir, exist_ok=True)
    
    stats_latest = os.path.join(latest_dir, 'stats.json')
    html_latest = os.path.join(latest_dir, 'bt_with_price.html')
    
    if os.path.exists(stats_src):
        shutil.copy2(stats_src, stats_latest)
        logger.info(f"最新ディレクトリに統計情報をコピーしました: {stats_latest}")
    
    if os.path.exists(html_src):
        shutil.copy2(html_src, html_latest)
        logger.info(f"最新ディレクトリにHTMLレポートをコピーしました: {html_latest}")


def main():
    parser = argparse.ArgumentParser(description='バックテスト実行スクリプト')
    parser.add_argument('--interval', type=str, default='15m', choices=['15m', '2h', '1d'],
                       help='バックテストの時間枠 (デフォルト: 15m)')
    parser.add_argument('--quick', action='store_true',
                       help='クイックモード（CIなど高速実行用）')
    parser.add_argument('--no-plot', action='store_true',
                       help='プロット生成をスキップ')
    
    args = parser.parse_args()
    interval = args.interval
    quick_mode = args.quick
    skip_plot = args.no_plot
    
    logger.info(f"バックテスト開始: 時間枠={interval}, クイックモード={quick_mode}")
    
    # 結果ディレクトリを作成
    result_dir, timestamp = create_result_directory(interval)
    
    try:
        # 実データをAPIから読み込む
        logger.info("実データをAPIから読み込みます...")
        from src.data.sync import DataSync
        
        # データ同期クラスの初期化
        api_key_file = os.path.join(project_root, 'config', 'api_keys.yaml')
        if not os.path.exists(api_key_file):
            logger.warning("API鍵ファイルが見つかりません。デフォルト設定で続行します。")
            api_key_file = None
        
        data_sync = DataSync(api_key_file=api_key_file)
        
        # ターゲット時間枠の実データを取得してMTFデータ結合
        data = data_sync.get_combined_data(symbol='BTC-USD', target_interval=interval)
        
        if data is None or data.empty:
            raise ValueError("実データの取得に失敗しました")
        
        logger.info(f"実データを読み込みました: {len(data)}行")
        
        # クイックモードの場合はデータを減らす
        if quick_mode and len(data) > 1000:
            logger.info(f"クイックモード: データを削減します {len(data)} → 1000行")
            data = data.iloc[-1000:]
        
        # バックテストを実行
        logger.info(f"バックテスト実行中...")
        backtest_result = run_backtest(data, interval=interval)
        
        # 統計情報を生成
        stats = generate_stats(backtest_result)
        
        # HTMLプロットを生成（オプション）
        if not skip_plot:
            logger.info(f"バックテスト結果をプロット中...")
            plot_backtest_with_price(backtest_result, data, result_dir)
        
        # 統計情報をJSONとして保存
        stats_path = os.path.join(result_dir, 'stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"統計情報を保存しました: {stats_path}")
        
        # 最新結果へのシンボリックリンク更新
        update_latest_symlinks(result_dir, interval)
        
        # 結果を表示
        logger.info("バックテスト完了")
        logger.info(f"シャープレシオ: {stats.get('sharpe_ratio', 'N/A'):.2f}")
        logger.info(f"最大ドローダウン: {stats.get('max_drawdown_pct', 'N/A'):.2f}%")
        logger.info(f"リターン: {stats.get('total_return_pct', 'N/A'):.2f}%")
        logger.info(f"結果ディレクトリ: {result_dir}")
        
        return 0
    
    except Exception as e:
        logger.error(f"バックテスト実行中にエラーが発生しました: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
