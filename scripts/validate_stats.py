#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
バックテスト結果検証スクリプト
----------------------------

reports/{interval}/stats.jsonファイルを検証し、
パフォーマンス指標が許容範囲内かどうかを確認します。

使用方法:
    python scripts/validate_stats.py [--path reports/15m/stats.json] [--strict]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

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
except ImportError as e:
    logger.error(f"モジュールのインポートに失敗しました: {e}")
    sys.exit(1)

# 検証基準（デフォルト値）
DEFAULT_VALIDATION_CRITERIA = {
    "sharpe_ratio": {
        "min": 0.5,  # 最小許容シャープレシオ
        "max": 5.0   # 最大許容シャープレシオ
    },
    "max_drawdown": {
        "max": 0.25  # 最大許容ドローダウン (25%)
    },
    "total_return": {
        "min": 0.0   # 最小許容リターン (0%)
    },
    "win_rate": {
        "min": 0.45  # 最小許容勝率 (45%)
    },
    "calmar_ratio": {
        "min": 0.3   # 最小許容カルマーレシオ
    }
}

def load_stats_file(stats_path: str) -> Dict[str, Any]:
    """
    statsファイルを読み込む
    
    Parameters:
    -----------
    stats_path : str
        統計ファイルのパス
        
    Returns:
    --------
    Dict[str, Any]
        統計データ
    """
    try:
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        return stats
    except FileNotFoundError:
        logger.error(f"統計ファイルが見つかりません: {stats_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"統計ファイルの形式が不正です: {stats_path}")
        return {}
    except Exception as e:
        logger.error(f"統計ファイルの読み込みに失敗しました: {e}")
        return {}

def load_validation_criteria() -> Dict[str, Dict[str, float]]:
    """
    検証基準を読み込む
    
    Returns:
    --------
    Dict[str, Dict[str, float]]
        検証基準
    """
    # 設定からモデルの検証基準を読み込む
    config_loader = ConfigLoader()
    model_config = config_loader.load('model')
    
    # リスク管理セクションがあれば、そこから検証基準を取得
    if 'risk' in model_config and 'guardrails' in model_config['risk']:
        guardrails = model_config['risk']['guardrails']
        
        # デフォルト値をベースに設定値で上書き
        criteria = DEFAULT_VALIDATION_CRITERIA.copy()
        
        # モデル設定から値を取得して上書き
        if 'max_drawdown' in guardrails:
            criteria['max_drawdown']['max'] = guardrails['max_drawdown']
        
        return criteria
    
    # 設定が見つからない場合はデフォルト値を返す
    return DEFAULT_VALIDATION_CRITERIA

def validate_stats(stats: Dict[str, Any], criteria: Dict[str, Dict[str, float]], strict: bool = False) -> Tuple[bool, List[str]]:
    """
    統計データを検証する
    
    Parameters:
    -----------
    stats : Dict[str, Any]
        統計データ
    criteria : Dict[str, Dict[str, float]]
        検証基準
    strict : bool
        厳格モード（Trueの場合、すべての基準を満たす必要がある）
        
    Returns:
    --------
    Tuple[bool, List[str]]
        (検証結果, エラーメッセージのリスト)
    """
    errors = []
    
    # 基本的な検証（必須フィールドの存在確認）
    required_fields = ['sharpe_ratio', 'max_drawdown', 'total_return']
    for field in required_fields:
        if field not in stats:
            errors.append(f"必須フィールドがありません: {field}")
    
    if errors and strict:
        return False, errors
    
    # 各指標の検証
    # 1. シャープレシオ
    if 'sharpe_ratio' in stats and 'sharpe_ratio' in criteria:
        sharpe = stats['sharpe_ratio']
        sharpe_criteria = criteria['sharpe_ratio']
        
        if 'min' in sharpe_criteria and sharpe < sharpe_criteria['min']:
            errors.append(f"シャープレシオが最小値を下回っています: {sharpe:.2f} < {sharpe_criteria['min']:.2f}")
        
        if 'max' in sharpe_criteria and sharpe > sharpe_criteria['max']:
            errors.append(f"シャープレシオが最大値を上回っています: {sharpe:.2f} > {sharpe_criteria['max']:.2f}")
    
    # 2. 最大ドローダウン
    if 'max_drawdown' in stats and 'max_drawdown' in criteria:
        # 値の絶対値を取得（正負の符号を無視）
        drawdown = abs(stats['max_drawdown'])
        drawdown_criteria = criteria['max_drawdown']
        
        if 'max' in drawdown_criteria and drawdown > drawdown_criteria['max']:
            errors.append(f"最大ドローダウンが許容値を超えています: {drawdown:.2%} > {drawdown_criteria['max']:.2%}")
    
    # 3. トータルリターン
    if 'total_return' in stats and 'total_return' in criteria:
        total_return = stats['total_return']
        return_criteria = criteria['total_return']
        
        if 'min' in return_criteria and total_return < return_criteria['min']:
            errors.append(f"トータルリターンが最小値を下回っています: {total_return:.2%} < {return_criteria['min']:.2%}")
    
    # 4. 勝率
    if 'win_rate' in stats and 'win_rate' in criteria:
        win_rate = stats['win_rate']
        win_rate_criteria = criteria['win_rate']
        
        if 'min' in win_rate_criteria and win_rate < win_rate_criteria['min']:
            errors.append(f"勝率が最小値を下回っています: {win_rate:.2%} < {win_rate_criteria['min']:.2%}")
    
    # 5. カルマーレシオ（年間リターン / 最大ドローダウン）
    if 'calmar_ratio' in stats and 'calmar_ratio' in criteria:
        calmar = stats['calmar_ratio']
        calmar_criteria = criteria['calmar_ratio']
        
        if 'min' in calmar_criteria and calmar < calmar_criteria['min']:
            errors.append(f"カルマーレシオが最小値を下回っています: {calmar:.2f} < {calmar_criteria['min']:.2f}")
    
    # 検証結果
    is_valid = len(errors) == 0
    
    return is_valid, errors

def main():
    parser = argparse.ArgumentParser(description='バックテスト結果検証スクリプト')
    parser.add_argument('--path', type=str, default=None,
                       help='検証する統計ファイルパス（例: reports/15m/stats.json）')
    parser.add_argument('--strict', action='store_true',
                       help='厳格モード（すべての基準を満たす必要がある）')
    
    args = parser.parse_args()
    
    # 統計ファイルパスの特定
    if args.path:
        stats_path = args.path
    else:
        # パスが指定されていない場合、最新の統計ファイルを探す
        reports_dir = os.path.join(project_root, 'reports')
        
        if not os.path.exists(reports_dir) or not os.path.isdir(reports_dir):
            logger.error(f"レポートディレクトリが見つかりません: {reports_dir}")
            return 1
        
        # 全ての時間枠ディレクトリを探索
        stats_files = []
        for interval_dir in os.listdir(reports_dir):
            interval_path = os.path.join(reports_dir, interval_dir)
            
            # ディレクトリでない場合はスキップ
            if not os.path.isdir(interval_path):
                continue
            
            # stats.jsonファイルを探す
            stats_file = os.path.join(interval_path, 'stats.json')
            if os.path.exists(stats_file) and os.path.isfile(stats_file):
                stats_files.append(stats_file)
        
        if not stats_files:
            logger.error("検証する統計ファイルが見つかりません")
            return 1
        
        # 最新の統計ファイルを選択
        stats_path = max(stats_files, key=os.path.getmtime)
    
    logger.info(f"統計ファイルを検証します: {stats_path}")
    
    # 統計ファイルの読み込み
    stats = load_stats_file(stats_path)
    
    if not stats:
        logger.error("統計データを読み込めませんでした")
        return 1
    
    # 検証基準の読み込み
    criteria = load_validation_criteria()
    
    # 検証実行
    is_valid, errors = validate_stats(stats, criteria, args.strict)
    
    # 結果出力
    if is_valid:
        logger.info("検証成功: すべての基準を満たしています")
        
        # 主要指標のログ出力
        logger.info(f"シャープレシオ: {stats.get('sharpe_ratio', 'N/A'):.2f}")
        logger.info(f"最大ドローダウン: {stats.get('max_drawdown', 'N/A'):.2%}")
        logger.info(f"トータルリターン: {stats.get('total_return', 'N/A'):.2%}")
        logger.info(f"勝率: {stats.get('win_rate', 'N/A'):.2%}")
        
        return 0
    else:
        logger.error("検証失敗: 以下の基準を満たしていません")
        for error in errors:
            logger.error(f"  - {error}")
        
        # 主要指標のログ出力
        logger.info(f"シャープレシオ: {stats.get('sharpe_ratio', 'N/A')}")
        logger.info(f"最大ドローダウン: {stats.get('max_drawdown', 'N/A')}")
        logger.info(f"トータルリターン: {stats.get('total_return', 'N/A')}")
        logger.info(f"勝率: {stats.get('win_rate', 'N/A')}")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
