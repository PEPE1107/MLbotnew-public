#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
バックテスト実行スクリプト
-----------------------

指定した時間枠のバックテストを実行し、結果をレポートディレクトリに保存します。
通常モードとクイックモード（CI用）を提供します。

使用方法:
    python scripts/run_backtest.py --interval 15m [--quick] [--start 2023-01-01] [--end 2023-12-31]
"""

import os
import sys
import argparse
import logging
import json
import datetime
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 内部モジュールのインポート
try:
    from src.config_loader import ConfigLoader
    from src.backtest.run import run_backtest
    from src.backtest.utils import calculate_statistics, generate_html_report
except ImportError as e:
    logger.error(f"モジュールのインポートに失敗しました: {e}")
    sys.exit(1)

def parse_date(date_str: str) -> datetime.datetime:
    """
    日付文字列をdatetimeオブジェクトに変換
    
    Parameters:
    -----------
    date_str : str
        日付文字列 (YYYY-MM-DD)
    
    Returns:
    --------
    datetime.datetime
        変換後のdatetimeオブジェクト
    """
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        logger.error(f"無効な日付形式です: {date_str} (正しい形式: YYYY-MM-DD)")
        sys.exit(1)

def generate_runid() -> str:
    """
    一意の実行IDを生成
    
    Returns:
    --------
    str
        生成された実行ID (run_YYYYMMDD_HHMMSS_xxxxx)
    """
    now = datetime.datetime.now()
    date_part = now.strftime("%Y%m%d_%H%M%S")
    random_part = str(uuid.uuid4())[:8]
    return f"run_{date_part}_{random_part}"

def create_report_dir(interval: str, runid: Optional[str] = None) -> str:
    """
    レポートディレクトリを作成
    
    Parameters:
    -----------
    interval : str
        時間枠
    runid : Optional[str]
        実行ID（未指定の場合は自動生成）
    
    Returns:
    --------
    str
        作成されたディレクトリパス
    """
    # 実行IDを取得または生成
    runid = runid or generate_runid()
    
    # レポートディレクトリを作成
    reports_dir = os.path.join(project_root, 'reports')
    interval_dir = os.path.join(reports_dir, interval)
    report_dir = os.path.join(interval_dir, runid)
    
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(interval_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    logger.info(f"レポートディレクトリを作成しました: {report_dir}")
    
    return report_dir

def load_config(interval: str) -> Dict[str, Any]:
    """
    各種設定を読み込む
    
    Parameters:
    -----------
    interval : str
        時間枠
    
    Returns:
    --------
    Dict[str, Any]
        設定内容
    """
    config_loader = ConfigLoader()
    
    # 基本設定を読み込む
    intervals_config = config_loader.load('intervals')
    fees_config = config_loader.load('fees')
    model_config = config_loader.load('model')
    mtf_config = config_loader.load('mtf')
    
    # 特定の時間枠の設定を抽出
    interval_config = config_loader.get_interval_config(interval)
    
    # バックテスト設定を構築
    bt_config = {
        'interval': interval,
        'interval_config': interval_config,
        'fees': fees_config.get('fees', {}).get('default', 0.00055),
        'slippage': fees_config.get('slippage', {}).get('default', 0.0001),
        'upon_op': fees_config.get('trading_options', {}).get('backtest', {}).get('upon_op', 'Next'),
        'model': model_config.get('models', {}),
        'mtf': mtf_config,
    }
    
    return bt_config

def validate_environment() -> bool:
    """
    実行環境をチェック
    
    Returns:
    --------
    bool
        環境が有効かどうか
    """
    # 必要なディレクトリが存在するか確認
    required_dirs = [
        os.path.join(project_root, 'data'),
        os.path.join(project_root, 'features'),
        os.path.join(project_root, 'models'),
        os.path.join(project_root, 'reports')
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logger.warning(f"必要なディレクトリが見つかりません: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"ディレクトリを作成しました: {dir_path}")
    
    # 設定ファイルが存在するか確認
    required_configs = ['intervals.yaml', 'fees.yaml', 'model.yaml', 'mtf.yaml']
    config_dir = os.path.join(project_root, 'config')
    
    for config_file in required_configs:
        config_path = os.path.join(config_dir, config_file)
        if not os.path.exists(config_path):
            logger.error(f"必要な設定ファイルが見つかりません: {config_path}")
            return False
    
    return True

def save_stats(stats: Dict[str, Any], report_dir: str) -> str:
    """
    統計情報をJSONファイルに保存
    
    Parameters:
    -----------
    stats : Dict[str, Any]
        統計情報
    report_dir : str
        レポートディレクトリパス
    
    Returns:
    --------
    str
        保存されたファイルパス
    """
    stats_file = os.path.join(report_dir, 'stats.json')
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"統計情報を保存しました: {stats_file}")
    
    return stats_file

def main():
    parser = argparse.ArgumentParser(description='バックテスト実行スクリプト')
    parser.add_argument('--interval', type=str, default='15m',
                      help='バックテスト時間枠 (例: 15m, 2h, 1d)')
    parser.add_argument('--quick', action='store_true',
                      help='クイックモードで実行（CI用、短期間のテスト）')
    parser.add_argument('--start', type=str,
                      help='開始日（YYYY-MM-DD形式）')
    parser.add_argument('--end', type=str,
                      help='終了日（YYYY-MM-DD形式）')
    parser.add_argument('--runid', type=str,
                      help='レポート用の実行ID（未指定時は自動生成）')
    
    args = parser.parse_args()
    
    # 環境チェック
    if not validate_environment():
        logger.error("実行環境が不正です。必要なファイルとディレクトリを確認してください。")
        return 1
    
    # 設定を読み込む
    config = load_config(args.interval)
    
    # 日付範囲を設定
    if args.quick:
        # クイックモード: 直近1ヶ月のみ
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=30)
    else:
        # 通常モード: 指定された範囲またはデフォルト
        end_date = parse_date(args.end) if args.end else datetime.datetime.now()
        if args.start:
            start_date = parse_date(args.start)
        else:
            # 時間枠に応じたデフォルト期間を設定
            days_lookup = {
                '15m': 30,    # 15分足: 30日
                '2h': 90,     # 2時間足: 90日
                '1d': 365     # 日足: 365日
            }
            days = days_lookup.get(args.interval, 30)
            start_date = end_date - datetime.timedelta(days=days)
    
    # 日付範囲をログ出力
    logger.info(f"バックテスト期間: {start_date.strftime('%Y-%m-%d')} から {end_date.strftime('%Y-%m-%d')}")
    
    # レポートディレクトリを作成
    report_dir = create_report_dir(args.interval, args.runid)
    
    # バックテストを実行
    logger.info(f"バックテストを開始します: {args.interval}")
    try:
        # 日数を計算
        days = (end_date - start_date).days
        
        result = run_backtest(
            interval=args.interval,
            symbol='BTC-USD',
            days=days
        )
        
        # 統計情報を計算
        stats = calculate_statistics(result)
        
        # 主要な統計情報をログに出力
        logger.info(f"バックテスト完了: {args.interval}")
        logger.info(f"  シャープレシオ: {stats['sharpe_ratio']:.2f}")
        logger.info(f"  最大ドローダウン: {stats['max_drawdown']:.2%}")
        logger.info(f"  トータルリターン: {stats['total_return']:.2%}")
        logger.info(f"  勝率: {stats['win_rate']:.2%}")
        
        # 統計情報をファイルに保存
        stats_file = save_stats(stats, report_dir)
        
        # HTMLレポートを生成
        html_file = os.path.join(report_dir, 'bt_with_price.html')
        generate_html_report(result, html_file)
        logger.info(f"HTMLレポートを生成しました: {html_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"バックテスト実行中にエラーが発生しました: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
