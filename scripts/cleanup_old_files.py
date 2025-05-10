#!/usr/bin/env python
"""
cleanup_old_files.py - 古いバックテスト結果ファイルを削除するスクリプト

機能:
- 最新の結果以外の古いバックテストファイルを削除
- 最新の結果のみを保持
"""

import os
import glob
import shutil
import logging
from pathlib import Path
from datetime import datetime

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('cleanup')

# アプリケーションルートへのパス設定
app_home = Path(os.path.dirname(os.path.abspath(__file__)))
reports_dir = app_home / 'reports'

def find_latest_timestamp_dir(directory):
    """指定されたディレクトリ内で最新のタイムスタンプディレクトリを見つける"""
    dirs = glob.glob(os.path.join(directory, '2*_*'))
    if not dirs:
        return None
    # 最新のタイムスタンプディレクトリを探す（名前でソート）
    return sorted(dirs)[-1]

def cleanup_plots_directory():
    """reportsディレクトリのplotsフォルダをクリーンアップ"""
    plots_dir = reports_dir / 'plots'
    if not plots_dir.exists():
        logger.info(f"プロットディレクトリが見つかりません: {plots_dir}")
        return
    
    # ルートのPNGファイルを削除
    root_png_files = glob.glob(os.path.join(plots_dir, '*.png'))
    for file in root_png_files:
        logger.info(f"古いPNGファイルを削除: {file}")
        os.remove(file)
    
    # 時間枠ごとにクリーンアップ
    intervals = ['15m', '2h']
    for interval in intervals:
        interval_dir = plots_dir / interval
        if not interval_dir.exists():
            continue
        
        # 最新のタイムスタンプディレクトリを見つける
        latest_dir = find_latest_timestamp_dir(interval_dir)
        if not latest_dir:
            continue
        
        logger.info(f"保持する最新のディレクトリ: {latest_dir}")
        
        # 最新以外のディレクトリを削除
        all_dirs = glob.glob(os.path.join(interval_dir, '2*_*'))
        for dir_path in all_dirs:
            if dir_path != latest_dir:
                logger.info(f"古いタイムスタンプディレクトリを削除: {dir_path}")
                shutil.rmtree(dir_path)

def cleanup_backtest_results(interval):
    """指定された時間枠のバックテスト結果をクリーンアップ"""
    interval_dir = reports_dir / interval
    if not interval_dir.exists():
        logger.info(f"レポートディレクトリが見つかりません: {interval_dir}")
        return
    
    # stats.jsonを保持
    stats_file = interval_dir / 'stats.json'
    if stats_file.exists():
        logger.info(f"statsファイルを保持: {stats_file}")
    
    # 最新のportfolioファイルのみを保持
    portfolio_files = sorted(list(interval_dir.glob('backtest_portfolio_*.pkl')))
    if portfolio_files:
        latest_portfolio = portfolio_files[-1]
        logger.info(f"保持する最新のポートフォリオファイル: {latest_portfolio}")
        
        # 最新以外のportfolioファイルを削除
        for file in portfolio_files:
            if file != latest_portfolio:
                logger.info(f"古いポートフォリオファイルを削除: {file}")
                os.remove(file)
    
    # 最新のstats JSONファイルのみを保持（stats.jsonは触れない）
    stats_files = sorted(list(interval_dir.glob('stats_*.json')))
    if stats_files:
        latest_stats = stats_files[-1]
        logger.info(f"保持する最新のstatsファイル: {latest_stats}")
        
        # 最新以外のstatsファイルを削除
        for file in stats_files:
            if file != latest_stats:
                logger.info(f"古いstatsファイルを削除: {file}")
                os.remove(file)
    
    # 最新のPNGファイルのみを保持
    for pattern in ['returns_*.png', 'drawdowns_*.png']:
        png_files = sorted(list(interval_dir.glob(pattern)))
        if png_files:
            latest_png = png_files[-1]
            logger.info(f"保持する最新の{pattern}ファイル: {latest_png}")
            
            # 最新以外のPNGファイルを削除
            for file in png_files:
                if file != latest_png and not file.name.endswith('_new.png'):
                    logger.info(f"古いPNGファイルを削除: {file}")
                    os.remove(file)
    
    # 最新のタイムスタンプディレクトリのみを保持
    latest_dir = find_latest_timestamp_dir(interval_dir)
    if latest_dir:
        logger.info(f"保持する最新のディレクトリ: {latest_dir}")
        
        # 最新以外のディレクトリを削除 (lightgbmディレクトリは触れない)
        all_dirs = glob.glob(os.path.join(interval_dir, '2*_*'))
        for dir_path in all_dirs:
            if dir_path != latest_dir:
                logger.info(f"古いタイムスタンプディレクトリを削除: {dir_path}")
                shutil.rmtree(dir_path)

def main():
    """メイン関数"""
    logger.info("古いバックテスト結果ファイルのクリーンアップを開始します")
    
    # 時間枠ごとにクリーンアップ
    cleanup_backtest_results('15m')
    cleanup_backtest_results('2h')
    
    # プロットディレクトリのクリーンアップ
    cleanup_plots_directory()
    
    logger.info("クリーンアップ完了")

if __name__ == "__main__":
    main()
