#!/usr/bin/env python
"""
clean_sample_data.py - 不要なサンプルデータを削除するスクリプト

機能:
- 生データ (raw) ディレクトリのサンプルデータを削除
- 特徴量 (features) ディレクトリのサンプルデータを削除
- サンプルデータの削除前に確認を要求
"""

import os
import sys
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('clean_data')

# プロジェクトルートディレクトリ
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / 'data'

def is_sample_data(file_path: Path) -> bool:
    """ファイルがサンプルデータかどうかを判定
    
    サンプルデータは以下の一つ以上の特徴を持っています:
    - 2013年の価格データが60,000ドル前後である
    - 特定の乱数シード(42-49)で生成されたデータパターン
    
    Args:
        file_path: 判定するファイルのパス
        
    Returns:
        bool: サンプルデータの場合True
    """
    try:
        import pandas as pd
        
        # parquetファイルでない場合はスキップ
        if not file_path.suffix == '.parquet':
            return False
            
        # ファイルを読み込む
        df = pd.read_parquet(file_path)
        
        # 以下の基準でサンプルデータを判定
        # 1. priceデータで2013-2018年のデータが60,000ドル前後の場合
        if 'price_open' in df.columns and len(df) > 0:
            if df.index.min().year < 2020:
                earliest_price = df.iloc[0]['price_open'] if 'price_open' in df.columns else None
                # 2015年以前のBTC価格が5万ドル以上なら明らかにサンプルデータ
                if earliest_price is not None and df.index.min().year < 2020 and earliest_price > 50000:
                    return True
                    
        # 2. 各種レートに見られる一貫したパターン (サンプルデータ生成時の乱数シードの影響)
        # これは高度な判定が必要で、ここでは省略
        
        # 現状では詳細なパターン分析は実装せず、日付ベースの判定のみ実施
        return False
        
    except Exception as e:
        logger.warning(f"ファイル分析中にエラー {file_path}: {e}")
        return False

def find_sample_data() -> List[Tuple[Path, bool]]:
    """サンプルデータと思われるファイルのリストを取得
    
    Returns:
        List[Tuple[Path, bool]]: (ファイルパス, 確実にサンプルか)のリスト
    """
    sample_files = []
    
    # rawデータを確認
    for interval_dir in ['15m', '2h', '1d']:
        raw_dir = DATA_DIR / 'raw' / interval_dir
        if raw_dir.exists():
            for file_path in raw_dir.glob('*.parquet'):
                is_sample = is_sample_data(file_path)
                sample_files.append((file_path, is_sample))
    
    # 特徴量データを確認
    for interval_dir in ['15m', '2h', '1d']:
        features_dir = DATA_DIR / 'features' / interval_dir
        if features_dir.exists():
            for file_path in features_dir.glob('*.parquet'):
                is_sample = is_sample_data(file_path)
                sample_files.append((file_path, is_sample))
    
    return sample_files

def backup_data(backup_dir: Path = None):
    """データをバックアップ
    
    Args:
        backup_dir: バックアップ先ディレクトリ(省略時は自動生成)
    """
    # バックアップ先が指定されていない場合は日時からディレクトリ名を生成
    if backup_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = ROOT_DIR / f"data_backup_{timestamp}"
    
    # データディレクトリが存在する場合のみバックアップ
    if DATA_DIR.exists():
        logger.info(f"データをバックアップ中: {DATA_DIR} -> {backup_dir}")
        shutil.copytree(DATA_DIR, backup_dir)
        logger.info(f"バックアップ完了: {backup_dir}")
    else:
        logger.warning(f"データディレクトリが存在しません: {DATA_DIR}")

def clean_data(confirmed: bool = False, backup: bool = True):
    """サンプルデータを削除
    
    Args:
        confirmed: 確認なしで削除するかどうか
        backup: バックアップを作成するかどうか
    """
    # サンプルデータを検出
    sample_files = find_sample_data()
    
    # サンプルファイル一覧を表示
    if sample_files:
        logger.info(f"{len(sample_files)} 個のデータファイルが見つかりました")
        
        confirmed_samples = [f for f, is_confirmed in sample_files if is_confirmed]
        if confirmed_samples:
            logger.info(f"{len(confirmed_samples)} 個のファイルはサンプルデータの可能性が高いです:")
            for file_path, _ in confirmed_samples:
                logger.info(f"  - {file_path.relative_to(ROOT_DIR)}")
                
        # バックアップするかどうか確認
        if backup and not confirmed and sample_files:
            logger.info("処理を続行する前にデータをバックアップします")
            backup_data()
            
        # 削除確認
        if not confirmed:
            response = input("これらのファイルを削除しますか？(y/n): ")
            confirmed = response.lower() in ['y', 'yes']
            
        # 削除実行
        if confirmed:
            for file_path, _ in sample_files:
                try:
                    logger.info(f"削除中: {file_path.relative_to(ROOT_DIR)}")
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"削除中にエラー {file_path}: {e}")
            logger.info("サンプルデータの削除が完了しました")
        else:
            logger.info("操作はキャンセルされました")
    else:
        logger.info("サンプルデータは見つかりませんでした")

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='サンプルデータクリーナー')
    parser.add_argument('-y', '--yes', action='store_true', 
                        help='確認なしで削除を実行')
    parser.add_argument('--no-backup', action='store_true',
                        help='バックアップを作成しない')
    parser.add_argument('--clean-all', action='store_true',
                        help='すべてのデータを削除(サンプルかどうかにかかわらず)')
    return parser.parse_args()

def clean_all_data(confirmed: bool = False):
    """すべてのデータを削除
    
    Args:
        confirmed: 確認なしで削除するかどうか
    """
    if DATA_DIR.exists():
        logger.warning("注意: すべてのデータを削除します")
        
        if not confirmed:
            response = input("本当にすべてのデータを削除しますか？この操作は元に戻せません (y/n): ")
            confirmed = response.lower() in ['y', 'yes']
            
        if confirmed:
            logger.info(f"削除中: {DATA_DIR}")
            shutil.rmtree(DATA_DIR)
            logger.info(f"すべてのデータが削除されました")
            
            # 空の必須ディレクトリを再作成
            os.makedirs(DATA_DIR, exist_ok=True)
            intervals = ['15m', '2h', '1d']
            for interval in intervals:
                os.makedirs(DATA_DIR / 'raw' / interval, exist_ok=True)
                os.makedirs(DATA_DIR / 'features' / interval, exist_ok=True)
            logger.info("空のデータディレクトリ構造が作成されました")
        else:
            logger.info("操作はキャンセルされました")

def main():
    """メイン関数"""
    args = parse_args()
    
    try:
        if args.clean_all:
            # すべてのデータを削除
            clean_all_data(confirmed=args.yes)
        else:
            # サンプルデータのみ削除
            clean_data(confirmed=args.yes, backup=not args.no_backup)
            
        logger.info("正常に完了しました")
        return 0
        
    except Exception as e:
        logger.error(f"実行中にエラーが発生しました: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
