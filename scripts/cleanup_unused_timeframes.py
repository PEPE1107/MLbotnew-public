#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cleanup_unused_timeframes.py - 15分足と日足のデータを削除

MLbotnewプロジェクトを2時間足のCoinglassデータのみを使用するように再構成するため、
15分足と日足のデータ、レポート、特徴量ファイルを削除します。
"""

import os
import sys
import shutil
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

def remove_directory(path):
    """
    ディレクトリを削除する
    
    Args:
        path: 削除するディレクトリのパス
    """
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            logger.info(f"ディレクトリを削除しました: {path}")
        except Exception as e:
            logger.error(f"ディレクトリの削除に失敗しました: {path}, エラー: {e}")
    else:
        logger.info(f"ディレクトリが存在しません: {path}")

def main():
    """
    メイン関数
    """
    logger.info("不要な時間枠データの削除を開始します...")
    
    # 削除する時間枠
    timeframes_to_remove = ['15m', '1d']
    
    # 主要ディレクトリ
    data_dir = os.path.join(project_root, 'data')
    features_dir = os.path.join(project_root, 'features')
    reports_dir = os.path.join(project_root, 'reports')
    models_dir = os.path.join(project_root, 'models')
    
    for timeframe in timeframes_to_remove:
        logger.info(f"時間枠 {timeframe} のデータを削除しています...")
        
        # データディレクトリ
        remove_directory(os.path.join(data_dir, 'raw', timeframe))
        remove_directory(os.path.join(data_dir, timeframe))
        
        # 特徴量ディレクトリ
        remove_directory(os.path.join(features_dir, timeframe))
        
        # レポートディレクトリ
        remove_directory(os.path.join(reports_dir, timeframe))
        
        # モデルディレクトリ
        remove_directory(os.path.join(models_dir, timeframe))
        
        logger.info(f"時間枠 {timeframe} のデータ削除が完了しました")
    
    logger.info("不要な時間枠データの削除が完了しました")
    logger.info("MLbotnewプロジェクトは2時間足(2h)のCoinglassデータのみを使用するように再構成されました")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
