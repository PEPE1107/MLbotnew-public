#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Coinglassデータ取得スクリプト
------------------------

BTC/USDTの価格データとCoinglassの各種需給指標データを取得し、
データディレクトリに保存するスクリプト。2時間足の360日分のデータを取得。

使用方法:
    python scripts/fetch_all.py [--days 360] [--output data/]
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

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
    from src.data.coinglass_new import CoinglassClient
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
        APIキーの辞書
    """
    api_keys_path = os.path.join(project_root, 'config', 'api_keys.yaml')
    
    if not os.path.exists(api_keys_path):
        logger.warning(f"APIキーファイルが見つかりません: {api_keys_path}")
        return {}
    
    try:
        with open(api_keys_path, 'r', encoding='utf-8') as f:
            api_keys = yaml.safe_load(f)
        return api_keys or {}
    except Exception as e:
        logger.error(f"APIキーファイルの読み込みに失敗しました: {e}")
        return {}

def fetch_coinglass_data(days: int, output_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Coinglassデータを取得
    
    Parameters:
    -----------
    days : int
        取得日数 (最大360日)
    output_dir : str
        出力ディレクトリ
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        取得したデータフレームの辞書
    """
    # 固定の時間枠
    interval = '2h'
    
    # 時間枠に応じてlimitを計算
    # 2h間隔なので、1日に12データポイント
    limit = min(days * 12, 4320)  # 最大4320ポイント (360日分)
    
    logger.info(f"{interval}のCoinglassデータを{days}日分取得しています (limit: {limit})...")
    
    # APIキーの読み込み
    api_keys = load_api_keys()
    coinglass_key = api_keys.get('coinglass', "5a2ac6bc211648ee96f894307c4dc6af")  # デフォルトキーも設定
    
    # Coinglass クライアント初期化
    client = CoinglassClient(api_key=coinglass_key)
    
    # 出力ディレクトリの作成
    data_dir = os.path.join(output_dir, interval)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'features'), exist_ok=True)
    
    # すべてのデータタイプを取得
    try:
        logger.info(f"すべてのCoinglassデータタイプを取得しています...")
        
        # 各データタイプを取得
        results = {}
        
        # 1. 価格データ
        logger.info("価格OHLC取得中...")
        results['price'] = client.fetch_price_ohlc(interval=interval, limit=limit)
        
        # 2. ファンディングレート
        logger.info("ファンディングレート取得中...")
        results['funding'] = client.fetch_funding_rate(interval=interval, limit=limit)
        
        # 3. オープンインタレスト
        logger.info("オープンインタレスト取得中...")
        results['oi'] = client.fetch_open_interest(interval=interval, limit=limit)
        
        # 4. 清算データ
        logger.info("清算データ取得中...")
        results['liquidation'] = client.fetch_liquidation(interval=interval, limit=limit)
        
        # 5. ロングショート比率
        logger.info("ロングショート比率取得中...")
        results['lsr'] = client.fetch_long_short_ratio(interval=interval, limit=limit)
        
        # 6. Takerバイセル
        logger.info("Takerバイセルボリューム取得中...")
        results['taker'] = client.fetch_taker_buy_sell(interval=interval, limit=limit)
        
        # 7. オーダーブック
        logger.info("オーダーブックデータ取得中...")
        results['orderbook'] = client.fetch_orderbook(interval=interval, limit=limit)
        
        # 8. プレミアム指数
        logger.info("Coinbaseプレミアム指数取得中...")
        results['premium'] = client.fetch_premium_index(interval=interval, limit=limit)
        
        # 各データを個別に保存
        for name, df in results.items():
            if not df.empty:
                # ファイル名の生成
                today = datetime.datetime.now().strftime('%Y%m%d')
                filename = f"{name}_{interval}_{today}.parquet"
                output_path = os.path.join(data_dir, 'raw', filename)
                
                # データ保存
                df.to_parquet(output_path)
                logger.info(f"{name}データを保存しました: {output_path} (レコード数: {len(df)})")
            else:
                logger.warning(f"{name}データは空でした")
        
        # すべてのデータを統合
        logger.info("すべてのデータを統合中...")
        merged_df = client.merge_all_data(results)
        
        # マージしたデータを保存
        merged_filename = f"merged_{interval}_{today}.parquet"
        merged_path = os.path.join(data_dir, 'features', merged_filename)
        merged_df.to_parquet(merged_path)
        logger.info(f"統合データを保存しました: {merged_path} (レコード数: {len(merged_df)})")
        
        return results
    
    except Exception as e:
        logger.error(f"Coinglassデータの取得中にエラーが発生しました: {e}", exc_info=True)
        return {}

def main():
    parser = argparse.ArgumentParser(description='Coinglassデータ取得スクリプト')
    parser.add_argument('--days', type=int, default=360,
                       help='取得日数（デフォルト: 360日、最大360日）')
    parser.add_argument('--output', type=str, default='data',
                       help='出力ディレクトリ（デフォルト: data/）')
    
    args = parser.parse_args()
    
    # 日数を360日に制限
    days = min(args.days, 360)
    
    # 相対パスを絶対パスに変換
    output_dir = os.path.join(project_root, args.output)
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 固定の時間枠
    interval = '2h'
    
    logger.info(f"===== {interval} のデータ取得開始（{days}日分） =====")
    
    # Coinglassデータ取得
    results = fetch_coinglass_data(days, output_dir)
    
    if results:
        logger.info(f"===== {interval} のデータ取得完了 =====")
        logger.info(f"取得したデータタイプ: {', '.join(results.keys())}")
        
        # 各データタイプのレコード数
        for name, df in results.items():
            logger.info(f"  - {name}: {len(df)}レコード")
    else:
        logger.error(f"===== {interval} のデータ取得に失敗しました =====")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
