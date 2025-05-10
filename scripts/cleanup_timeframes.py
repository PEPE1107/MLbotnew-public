#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
不要な時間枠データ削除スクリプト
--------------------------

15分足と日足のデータを削除し、2時間足のデータのみを残します。
2時間足のCoinglassデータのみを使用するように変更するためのスクリプトです。
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_timeframe_data(keep_timeframe: str = '2h', remove_config: bool = False):
    """
    指定した時間枠以外のデータを削除する
    
    Parameters:
    -----------
    keep_timeframe : str
        残す時間枠（デフォルト: '2h'）
    remove_config : bool
        設定ファイルも更新するかどうか
    """
    # データディレクトリ
    data_dir = os.path.join(project_root, 'data')
    
    if os.path.exists(data_dir):
        # データディレクトリ内の時間枠ごとのディレクトリをチェック
        for interval_dir in os.listdir(data_dir):
            interval_path = os.path.join(data_dir, interval_dir)
            
            # ディレクトリかどうか確認
            if os.path.isdir(interval_path):
                # rawとfeaturesサブディレクトリをチェック
                for subdir in ['raw', 'features']:
                    subdir_path = os.path.join(interval_path, subdir)
                    
                    if os.path.exists(subdir_path):
                        # 時間枠サブディレクトリをチェック
                        for tf_dir in os.listdir(subdir_path):
                            tf_path = os.path.join(subdir_path, tf_dir)
                            
                            if os.path.isdir(tf_path) and tf_dir != keep_timeframe:
                                # 残す時間枠以外は削除
                                logger.info(f"削除中: {tf_path}")
                                shutil.rmtree(tf_path)
    
    # レポートディレクトリ
    reports_dir = os.path.join(project_root, 'reports')
    
    if os.path.exists(reports_dir):
        # レポートディレクトリ内の時間枠ごとのディレクトリをチェック
        for interval_dir in os.listdir(reports_dir):
            interval_path = os.path.join(reports_dir, interval_dir)
            
            # ディレクトリかどうか確認 & 残す時間枠以外をチェック
            if os.path.isdir(interval_path) and interval_dir != keep_timeframe and interval_dir != 'backtest_results_20250508' and interval_dir != 'data_check' and interval_dir != 'plots':
                # 残す時間枠以外は削除
                logger.info(f"削除中: {interval_path}")
                shutil.rmtree(interval_path)
    
    # モデルディレクトリ
    models_dir = os.path.join(project_root, 'models')
    
    if os.path.exists(models_dir):
        # モデルディレクトリ内の時間枠ごとのディレクトリをチェック
        for interval_dir in os.listdir(models_dir):
            interval_path = os.path.join(models_dir, interval_dir)
            
            # ディレクトリかどうか確認 & 残す時間枠以外をチェック
            if os.path.isdir(interval_path) and interval_dir != keep_timeframe:
                # 残す時間枠以外は削除
                logger.info(f"削除中: {interval_path}")
                shutil.rmtree(interval_path)
    
    # 設定ファイルの更新（オプション）
    if remove_config:
        # intervals.yamlを更新
        intervals_config_path = os.path.join(project_root, 'config', 'intervals.yaml')
        if os.path.exists(intervals_config_path):
            try:
                import yaml
                
                with open(intervals_config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # バックテスト間隔を2hのみに設定
                config['backtest_intervals'] = [keep_timeframe]
                
                # 時間枠設定を更新
                config['minute'] = keep_timeframe
                config['hour'] = keep_timeframe
                config['day'] = keep_timeframe
                
                with open(intervals_config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                logger.info(f"設定ファイルを更新しました: {intervals_config_path}")
            except Exception as e:
                logger.error(f"設定ファイルの更新に失敗しました: {e}")

def main():
    parser = argparse.ArgumentParser(description='不要な時間枠データ削除スクリプト')
    parser.add_argument('--keep', type=str, default='2h',
                       help='残す時間枠（デフォルト: 2h）')
    parser.add_argument('--update-config', action='store_true',
                       help='設定ファイルも更新するかどうか')
    
    args = parser.parse_args()
    
    logger.info(f"===== 不要な時間枠データの削除を開始します =====")
    logger.info(f"残す時間枠: {args.keep}")
    
    clean_timeframe_data(args.keep, args.update_config)
    
    logger.info(f"===== 不要な時間枠データの削除が完了しました =====")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
