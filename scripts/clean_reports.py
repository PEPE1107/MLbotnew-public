#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
レポートファイル整理スクリプト
------------------------

古いバックテスト結果を整理し、最新のレポートのみを保持します。
- 直近N件のみを保持
- 古いレポートはLFS参照ポインタのみ残してS3にアーカイブ

使用方法:
    python scripts/clean_reports.py [--keep 5] [--intervals 15m,2h,1d] [--archive]
"""

import os
import sys
import re
import glob
import shutil
import logging
import argparse
import subprocess
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_report_dirs(reports_path: str, interval: str) -> List[Tuple[str, datetime.datetime]]:
    """
    レポートディレクトリと日時情報を取得
    
    Parameters:
    -----------
    reports_path : str
        レポートディレクトリのパス
    interval : str
        時間枠
    
    Returns:
    --------
    List[Tuple[str, datetime.datetime]]
        (ディレクトリパス, 日時)のリスト
    """
    interval_path = os.path.join(reports_path, interval)
    
    if not os.path.exists(interval_path):
        logger.warning(f"時間枠ディレクトリが見つかりません: {interval_path}")
        return []
    
    # レポートディレクトリのリストを作成
    report_dirs = []
    for item in os.listdir(interval_path):
        dir_path = os.path.join(interval_path, item)
        
        # ディレクトリでない場合はスキップ
        if not os.path.isdir(dir_path):
            continue
        
        # 日時情報を取得（ディレクトリ名から）
        try:
            # runidフォーマット (例: run_20250501_120000)
            # または日付フォーマット (例: 20250501)
            date_match = re.search(r'(\d{8})', item)
            if date_match:
                date_str = date_match.group(1)
                
                # 時間情報も含まれている場合
                time_match = re.search(r'_(\d{6})', item)
                if time_match:
                    time_str = time_match.group(1)
                    dt = datetime.datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                else:
                    dt = datetime.datetime.strptime(date_str, "%Y%m%d")
                
                report_dirs.append((dir_path, dt))
            else:
                # stats.jsonファイルの更新日時を使用
                stats_file = os.path.join(dir_path, 'stats.json')
                if os.path.exists(stats_file):
                    mtime = os.path.getmtime(stats_file)
                    dt = datetime.datetime.fromtimestamp(mtime)
                    report_dirs.append((dir_path, dt))
        except Exception as e:
            logger.warning(f"ディレクトリ {dir_path} の日時情報を取得できませんでした: {e}")
    
    # 日時の新しい順にソート
    report_dirs.sort(key=lambda x: x[1], reverse=True)
    
    return report_dirs

def is_lfs_file(file_path: str) -> bool:
    """
    ファイルがGit LFSで管理されているかチェック
    
    Parameters:
    -----------
    file_path : str
        チェックするファイルパス
    
    Returns:
    --------
    bool
        LFSファイルかどうか
    """
    try:
        # ファイルの先頭数バイトを読み込んでLFSマーカーをチェック
        with open(file_path, 'rb') as f:
            header = f.read(100)
        
        # LFSファイルの特徴的なマーカー
        return b'version https://git-lfs.github.com/spec/' in header
    except Exception:
        return False

def get_lfs_managed_extensions() -> Set[str]:
    """
    Git LFSで管理されているファイル拡張子を取得
    
    Returns:
    --------
    Set[str]
        LFS管理対象の拡張子セット
    """
    try:
        # .gitattributesファイルを探す
        git_attr_path = os.path.join(project_root, '.gitattributes')
        if not os.path.exists(git_attr_path):
            return {'.pkl', '.html', '.parquet', '.h5', '.csv'}  # デフォルト値
        
        # .gitattributesファイルを解析
        extensions = set()
        with open(git_attr_path, 'r', encoding='utf-8') as f:
            for line in f:
                # LFS追跡行を検出
                if 'filter=lfs' in line:
                    parts = line.strip().split()
                    if parts:
                        pattern = parts[0].strip()
                        # *.拡張子 パターンを検出
                        if pattern.startswith('*'):
                            ext = pattern[1:]  # '*'を削除
                            extensions.add(ext)
        
        return extensions if extensions else {'.pkl', '.html', '.parquet', '.h5', '.csv'}
    except Exception as e:
        logger.warning(f".gitattributes解析エラー: {e}")
        return {'.pkl', '.html', '.parquet', '.h5', '.csv'}

def simulate_s3_archive(source_path: str, destination_key: str, bucket_name: str = 'mlbot-archive') -> bool:
    """
    S3アーカイブ処理をシミュレート（実際の環境では実装を差し替え）
    
    Parameters:
    -----------
    source_path : str
        アーカイブするファイルパス
    destination_key : str
        S3内の保存先キー
    bucket_name : str
        S3バケット名
    
    Returns:
    --------
    bool
        成功したかどうか
    """
    # 実際のS3アップロード処理を実装する場合はここを修正
    # 現在はシミュレーションのみ
    logger.info(f"S3アーカイブ（シミュレーション）: {source_path} → s3://{bucket_name}/{destination_key}")
    return True

def archive_report_dir(dir_path: str, interval: str, s3_archive: bool = False) -> bool:
    """
    レポートディレクトリをアーカイブ処理
    
    Parameters:
    -----------
    dir_path : str
        アーカイブするディレクトリパス
    interval : str
        時間枠
    s3_archive : bool
        S3アーカイブを実行するかどうか
    
    Returns:
    --------
    bool
        成功したかどうか
    """
    logger.info(f"ディレクトリをアーカイブします: {dir_path}")
    
    # LFS管理対象の拡張子リスト
    lfs_extensions = get_lfs_managed_extensions()
    
    # ディレクトリ内のすべてのファイルを処理
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            # stats.jsonファイルはGit管理対象として残す
            if file == 'stats.json':
                logger.debug(f"Git管理対象として保持: {file_path}")
                continue
            
            # 拡張子を取得
            _, ext = os.path.splitext(file)
            
            # LFS管理対象の場合、ポインタのみ残してコンテンツを削除
            if ext in lfs_extensions or is_lfs_file(file_path):
                rel_path = os.path.relpath(file_path, project_root)
                
                # S3アーカイブが有効な場合
                if s3_archive:
                    # S3にアーカイブ
                    s3_key = f"reports/{interval}/{os.path.basename(dir_path)}/{file}"
                    simulate_s3_archive(file_path, s3_key)
                
                # ファイルサイズをログ
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"LFSファイル処理: {rel_path} ({size_mb:.2f} MB)")
                
                # Git LFSポインタの場合は何もしない（削除しない）
                if is_lfs_file(file_path):
                    logger.debug(f"LFSポインタ保持: {file_path}")
                else:
                    # 通常ファイルの場合は削除（git cleanupで処理される想定）
                    try:
                        os.remove(file_path)
                        logger.debug(f"ファイル削除: {file_path}")
                    except Exception as e:
                        logger.warning(f"ファイル削除エラー {file_path}: {e}")
    
    return True

def run_git_lfs_prune():
    """
    Git LFS pruneコマンドを実行して未使用のLFSオブジェクトを削除
    """
    try:
        logger.info("Git LFS pruneを実行...")
        result = subprocess.run(
            ['git', 'lfs', 'prune'],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Git LFS prune完了: {result.stdout}")
        else:
            logger.warning(f"Git LFS pruneエラー: {result.stderr}")
    except Exception as e:
        logger.error(f"Git LFS prune実行エラー: {e}")

def main():
    parser = argparse.ArgumentParser(description='レポートファイル整理スクリプト')
    parser.add_argument('--keep', type=int, default=5,
                       help='保持するレポート数（デフォルト: 5）')
    parser.add_argument('--intervals', type=str, default='15m,2h,1d',
                       help='処理対象の時間枠（カンマ区切り、デフォルト: 15m,2h,1d）')
    parser.add_argument('--archive', action='store_true',
                       help='古いレポートをS3にアーカイブするかどうか')
    parser.add_argument('--dry-run', action='store_true',
                       help='ドライラン実行（実際の削除・アーカイブは行わない）')
    
    args = parser.parse_args()
    
    # 実行モードを表示
    if args.dry_run:
        logger.info("ドライラン実行（実際の削除・アーカイブは行いません）")
    
    # レポートディレクトリパス
    reports_path = os.path.join(project_root, 'reports')
    
    if not os.path.exists(reports_path):
        logger.error(f"レポートディレクトリが見つかりません: {reports_path}")
        return 1
    
    # 対象の時間枠リスト
    intervals = [interval.strip() for interval in args.intervals.split(',')]
    
    # 各時間枠ごとに処理
    for interval in intervals:
        logger.info(f"時間枠の処理開始: {interval}")
        
        # レポートディレクトリを取得
        report_dirs = get_report_dirs(reports_path, interval)
        
        if not report_dirs:
            logger.info(f"時間枠 {interval} にはレポートが見つかりませんでした")
            continue
        
        # 古いレポートを処理（最新のkeep件以外）
        if len(report_dirs) > args.keep:
            old_dirs = report_dirs[args.keep:]
            logger.info(f"時間枠 {interval} には {len(report_dirs)} 件のレポートがあり、{len(old_dirs)} 件を処理します")
            
            for dir_path, dt in old_dirs:
                logger.info(f"古いレポートを処理: {dir_path} ({dt.strftime('%Y-%m-%d %H:%M:%S')})")
                
                if not args.dry_run:
                    # アーカイブ処理
                    archive_report_dir(dir_path, interval, args.archive)
        else:
            logger.info(f"時間枠 {interval} には {len(report_dirs)} 件のレポートがあり、すべて保持範囲内です")
    
    # Git LFS pruneを実行（オプション）
    if not args.dry_run:
        run_git_lfs_prune()
    
    logger.info("レポート整理完了")
    return 0

if __name__ == "__main__":
    sys.exit(main())
