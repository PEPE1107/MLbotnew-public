#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Coinglassデータ確認スクリプト
-----------------------

Coinglassから取得したデータが適切に保存されているか確認するためのスクリプト。
特に2h時間枠で360日分（limit=4320）のデータがあるかを検証します。
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import glob

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
    from src.data.coinglass import CoinglassClient
    from src.config_loader import ConfigLoader
except ImportError as e:
    logger.error(f"モジュールのインポートに失敗しました: {e}")
    sys.exit(1)

def load_api_keys() -> Dict[str, str]:
    """
    APIキーを読み込む
    
    Returns:
    --------
    dict
        APIキー情報
    """
    config_path = os.path.join(project_root, 'config', 'api_keys.yaml')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"APIキーファイルの読み込みに失敗しました: {e}")
        return {}

def check_data_directory(interval: str = '2h') -> Dict[str, Any]:
    """
    データディレクトリ内のCoinglassデータを確認
    
    Parameters:
    -----------
    interval : str
        確認する時間枠
        
    Returns:
    --------
    dict
        確認結果の統計情報
    """
    logger.info(f"{interval}時間枠のCoinglassデータを確認しています...")
    
    # データディレクトリのパス
    raw_dir = os.path.join(project_root, 'data', 'raw', interval)
    
    if not os.path.exists(raw_dir):
        logger.warning(f"データディレクトリが存在しません: {raw_dir}")
        return {
            'interval': interval,
            'exists': False,
            'files': [],
            'days_coverage': 0,
            'records': {}
        }
    
    # ディレクトリ内のParquetファイルを検索
    parquet_files = glob.glob(os.path.join(raw_dir, '*.parquet'))
    
    result = {
        'interval': interval,
        'exists': True,
        'files': [],
        'days_coverage': 0,
        'records': {}
    }
    
    for file_path in parquet_files:
        file_name = os.path.basename(file_path)
        try:
            # ファイルの読み込み
            df = pd.read_parquet(file_path)
            
            # データ範囲の確認
            if not df.empty:
                start_date = df.index.min()
                end_date = df.index.max()
                days_coverage = (end_date - start_date).days
                record_count = len(df)
                
                file_info = {
                    'file': file_name,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'days_coverage': days_coverage,
                    'records': record_count
                }
                
                result['files'].append(file_info)
                result['records'][file_name] = record_count
                
                # 合計日数を更新（最大値を採用）
                result['days_coverage'] = max(result['days_coverage'], days_coverage)
            else:
                logger.warning(f"ファイルが空です: {file_name}")
                result['files'].append({
                    'file': file_name,
                    'warning': 'ファイルが空です'
                })
        except Exception as e:
            logger.error(f"ファイル読み込みエラー: {file_name}, エラー: {e}")
            result['files'].append({
                'file': file_name,
                'error': str(e)
            })
    
    return result

def check_endpoint_coverage(interval: str = '2h', expected_days: int = 360) -> Dict[str, Any]:
    """
    各エンドポイントのデータカバレッジを確認
    
    Parameters:
    -----------
    interval : str
        確認する時間枠
    expected_days : int
        期待されるデータ日数
        
    Returns:
    --------
    dict
        エンドポイントごとのカバレッジ情報
    """
    logger.info(f"{interval}時間枠のエンドポイントカバレッジを確認しています...")
    
    # データディレクトリのパス
    raw_dir = os.path.join(project_root, 'data', 'raw', interval)
    
    if not os.path.exists(raw_dir):
        logger.warning(f"データディレクトリが存在しません: {raw_dir}")
        return {
            'interval': interval,
            'exists': False,
            'coverage': {}
        }
    
    # エンドポイント一覧
    endpoints = ['price', 'oi', 'funding', 'liq', 'lsr', 'taker', 'orderbook', 'premium']
    
    result = {
        'interval': interval,
        'exists': True,
        'coverage': {}
    }
    
    for endpoint in endpoints:
        file_path = os.path.join(raw_dir, f"{endpoint}.parquet")
        
        if os.path.exists(file_path):
            try:
                # ファイルの読み込み
                df = pd.read_parquet(file_path)
                
                # データ範囲の確認
                if not df.empty:
                    start_date = df.index.min()
                    end_date = df.index.max()
                    days_coverage = (end_date - start_date).days
                    record_count = len(df)
                    
                    # 期待値との比較
                    coverage_ratio = min(days_coverage / expected_days * 100, 100) if expected_days > 0 else 0
                    
                    # 2h時間枠の期待レコード数（1日8レコード×日数の85%以上あるか）
                    expected_records = expected_days * 12
                    records_ratio = min(record_count / expected_records * 100, 100) if expected_records > 0 else 0
                    
                    result['coverage'][endpoint] = {
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': end_date.strftime('%Y-%m-%d'),
                        'days_coverage': days_coverage,
                        'records': record_count,
                        'coverage_ratio': round(coverage_ratio, 1),
                        'records_ratio': round(records_ratio, 1),
                        'status': 'OK' if records_ratio >= 85 else 'WARNING'
                    }
                else:
                    logger.warning(f"エンドポイントのファイルが空です: {endpoint}")
                    result['coverage'][endpoint] = {
                        'warning': 'ファイルが空です',
                        'status': 'ERROR'
                    }
            except Exception as e:
                logger.error(f"ファイル読み込みエラー: {endpoint}, エラー: {e}")
                result['coverage'][endpoint] = {
                    'error': str(e),
                    'status': 'ERROR'
                }
        else:
            logger.warning(f"エンドポイントのファイルが存在しません: {endpoint}")
            result['coverage'][endpoint] = {
                'warning': 'ファイルが存在しません',
                'status': 'ERROR'
            }
    
    return result

def check_data_quality(interval: str = '2h') -> Dict[str, Any]:
    """
    データ品質を確認
    
    Parameters:
    -----------
    interval : str
        確認する時間枠
        
    Returns:
    --------
    dict
        データ品質情報
    """
    logger.info(f"{interval}時間枠のデータ品質を確認しています...")
    
    # データディレクトリのパス
    raw_dir = os.path.join(project_root, 'data', 'raw', interval)
    
    if not os.path.exists(raw_dir):
        logger.warning(f"データディレクトリが存在しません: {raw_dir}")
        return {
            'interval': interval,
            'exists': False,
            'quality': {}
        }
    
    # エンドポイント一覧
    endpoints = ['price', 'oi', 'funding', 'liq', 'lsr', 'taker', 'orderbook', 'premium']
    
    result = {
        'interval': interval,
        'exists': True,
        'quality': {}
    }
    
    for endpoint in endpoints:
        file_path = os.path.join(raw_dir, f"{endpoint}.parquet")
        
        if os.path.exists(file_path):
            try:
                # ファイルの読み込み
                df = pd.read_parquet(file_path)
                
                # データ品質確認
                if not df.empty:
                    # 欠損値のチェック
                    null_counts = df.isnull().sum()
                    null_ratio = (null_counts / len(df)).mean() * 100
                    
                    # 重複インデックスのチェック
                    duplicate_count = df.index.duplicated().sum()
                    duplicate_ratio = duplicate_count / len(df) * 100 if len(df) > 0 else 0
                    
                    # インデックスの連続性チェック
                    if interval == '2h':
                        expected_freq = '2H'
                    elif interval == '15m':
                        expected_freq = '15min'
                    elif interval == '1d':
                        expected_freq = '1D'
                    else:
                        expected_freq = None
                    
                    continuity_score = 0
                    gaps_count = 0
                    
                    if expected_freq:
                        # 理想的な連続インデックスを生成
                        ideal_index = pd.date_range(
                            start=df.index.min(),
                            end=df.index.max(),
                            freq=expected_freq
                        )
                        
                        # 実際のインデックスとの差異を計算
                        missing_points = len(ideal_index) - len(df)
                        continuity_score = max(0, 100 - (missing_points / len(ideal_index) * 100)) if len(ideal_index) > 0 else 0
                        
                        # ギャップの特定
                        expected_index_set = set(ideal_index)
                        actual_index_set = set(df.index)
                        missing_indices = expected_index_set - actual_index_set
                        gaps_count = len(missing_indices)
                    
                    result['quality'][endpoint] = {
                        'records': len(df),
                        'null_ratio': round(null_ratio, 1),
                        'duplicate_ratio': round(duplicate_ratio, 1),
                        'continuity_score': round(continuity_score, 1),
                        'gaps_count': gaps_count,
                        'status': 'OK' if null_ratio < 5 and duplicate_ratio < 1 and continuity_score > 90 else 'WARNING'
                    }
                else:
                    logger.warning(f"エンドポイントのファイルが空です: {endpoint}")
                    result['quality'][endpoint] = {
                        'warning': 'ファイルが空です',
                        'status': 'ERROR'
                    }
            except Exception as e:
                logger.error(f"ファイル読み込みエラー: {endpoint}, エラー: {e}")
                result['quality'][endpoint] = {
                    'error': str(e),
                    'status': 'ERROR'
                }
        else:
            logger.warning(f"エンドポイントのファイルが存在しません: {endpoint}")
            result['quality'][endpoint] = {
                'warning': 'ファイルが存在しません',
                'status': 'ERROR'
            }
    
    return result

def plot_data_coverage(results: Dict[str, Any], output_dir: str = None) -> str:
    """
    データカバレッジのプロット作成
    
    Parameters:
    -----------
    results : dict
        確認結果
    output_dir : str, optional
        出力ディレクトリ
        
    Returns:
    --------
    str
        保存したファイルのパス
    """
    if not output_dir:
        output_dir = os.path.join(project_root, 'reports', 'data_check')
        os.makedirs(output_dir, exist_ok=True)
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 各エンドポイントをプロット
    coverage = results.get('coverage', {})
    endpoints = list(coverage.keys())
    x_pos = np.arange(len(endpoints))
    
    coverage_values = []
    records_values = []
    colors = []
    
    for endpoint in endpoints:
        endpoint_info = coverage.get(endpoint, {})
        coverage_ratio = endpoint_info.get('coverage_ratio', 0)
        records_ratio = endpoint_info.get('records_ratio', 0)
        status = endpoint_info.get('status', 'ERROR')
        
        coverage_values.append(coverage_ratio)
        records_values.append(records_ratio)
        
        if status == 'OK':
            colors.append('green')
        elif status == 'WARNING':
            colors.append('orange')
        else:
            colors.append('red')
    
    # 日数カバレッジのバーチャート
    ax.bar(x_pos, coverage_values, width=0.4, align='edge', color=colors, alpha=0.7, label='日数カバレッジ (%)')
    
    # レコード数比率のバーチャート
    ax.bar(x_pos + 0.4, records_values, width=0.4, align='edge', color=colors, alpha=0.4, label='レコード数比率 (%)')
    
    # 目標ライン
    ax.axhline(y=85, color='r', linestyle='--', label='目標 85%')
    
    # グラフの設定
    ax.set_xlabel('エンドポイント')
    ax.set_ylabel('カバレッジ / 比率 (%)')
    ax.set_title(f'Coinglass データカバレッジ ({results.get("interval", "")})')
    ax.set_xticks(x_pos + 0.4)
    ax.set_xticklabels(endpoints, rotation=45)
    ax.legend()
    
    # グリッド追加
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # グラフの余白調整
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(output_dir, f'coinglass_coverage_{results.get("interval", "")}.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def print_summary_report(results: Dict[str, Any]) -> None:
    """
    サマリーレポートの出力
    
    Parameters:
    -----------
    results : dict
        確認結果
    """
    print("\n" + "="*80)
    print(f"Coinglass データ確認サマリー ({results.get('interval', '')})")
    print("="*80)
    
    # カバレッジ情報
    coverage = results.get('coverage', {})
    
    print("\n【日数カバレッジ】")
    print(f"目標日数: {results.get('expected_days', 360)}日")
    
    status_count = {'OK': 0, 'WARNING': 0, 'ERROR': 0}
    
    for endpoint, info in coverage.items():
        status = info.get('status', 'ERROR')
        status_count[status] += 1
        
        status_symbol = 'OK' if status == 'OK' else ('!!' if status == 'WARNING' else 'NG')
        
        if 'warning' in info or 'error' in info:
            message = info.get('warning', info.get('error', 'Unknown error'))
            print(f"  [{status_symbol}] {endpoint}: {message}")
        else:
            coverage_str = f"{info.get('days_coverage', 0)}日 ({info.get('coverage_ratio', 0)}%)"
            records_str = f"{info.get('records', 0)}件 ({info.get('records_ratio', 0)}%)"
            print(f"  [{status_symbol}] {endpoint}: {coverage_str}, {records_str}")
    
    # 品質情報
    quality = results.get('quality', {})
    
    print("\n【データ品質】")
    for endpoint, info in quality.items():
        status = info.get('status', 'ERROR')
        
        status_symbol = 'OK' if status == 'OK' else ('!!' if status == 'WARNING' else 'NG')
        
        if 'warning' in info or 'error' in info:
            message = info.get('warning', info.get('error', 'Unknown error'))
            print(f"  [{status_symbol}] {endpoint}: {message}")
        else:
            null_str = f"欠損率: {info.get('null_ratio', 0)}%"
            dup_str = f"重複率: {info.get('duplicate_ratio', 0)}%"
            con_str = f"連続性: {info.get('continuity_score', 0)}%"
            gaps_str = f"ギャップ: {info.get('gaps_count', 0)}件"
            print(f"  [{status_symbol}] {endpoint}: {null_str}, {dup_str}, {con_str}, {gaps_str}")
    
    # 総合評価
    print("\n【総合評価】")
    if status_count['ERROR'] > 0:
        print(f"  [NG] 不合格: {status_count['ERROR']}件のエラーがあります")
    elif status_count['WARNING'] > 0:
        print(f"  [!!] 条件付き合格: {status_count['WARNING']}件の警告があります")
    else:
        print(f"  [OK] 合格: すべてのエンドポイントが基準を満たしています")
    
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Coinglassデータ確認ツール')
    parser.add_argument('--interval', type=str, default='2h',
                        help='確認する時間枠 (例: 15m, 2h, 1d)')
    parser.add_argument('--days', type=int, default=360,
                        help='期待されるデータ日数')
    parser.add_argument('--output', type=str, default=None,
                        help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    logger.info(f"Coinglassデータ確認を開始: 時間枠={args.interval}, 期待日数={args.days}日")
    
    # 出力ディレクトリの設定
    output_dir = args.output
    if not output_dir:
        output_dir = os.path.join(project_root, 'reports', 'data_check')
        os.makedirs(output_dir, exist_ok=True)
    
    # カバレッジ確認
    coverage_results = check_endpoint_coverage(args.interval, args.days)
    coverage_results['expected_days'] = args.days
    
    # データ品質確認
    quality_results = check_data_quality(args.interval)
    
    # 結果の統合
    results = {
        'interval': args.interval,
        'expected_days': args.days,
        'coverage': coverage_results.get('coverage', {}),
        'quality': quality_results.get('quality', {})
    }
    
    # レポート出力
    print_summary_report(results)
    
    # グラフ作成
    plot_path = plot_data_coverage(results, output_dir)
    logger.info(f"カバレッジグラフを保存しました: {plot_path}")
    
    # 結果をJSONとして保存
    import json
    json_path = os.path.join(output_dir, f'coinglass_check_{args.interval}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"検証結果をJSONとして保存しました: {json_path}")
    
    # 成功・失敗の判定
    status_ok = all(info.get('status', 'ERROR') == 'OK' for info in results.get('coverage', {}).values())
    
    if status_ok:
        logger.info("すべてのエンドポイントで十分なデータが確認されました")
        return 0
    else:
        logger.warning("一部のエンドポイントでデータが不足しています")
        return 1

if __name__ == "__main__":
    sys.exit(main())
