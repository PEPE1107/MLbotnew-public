#!/usr/bin/env python
"""
generate_sample_data.py - Coinglass繝・・繧ｿ縺ｮ莉｣譖ｿ繧ｵ繝ｳ繝励Ν繝・・繧ｿ逕滓・繧ｹ繧ｯ繝ｪ繝励ヨ

讖溯・:
- 蜷・お繝ｳ繝峨・繧､繝ｳ繝・(price, oi, funding, liq, lsr, taker, orderbook, premium) 縺ｮ讓｡謫ｬ繝・・繧ｿ逕滓・
- 謖・ｮ壹＆繧後◆譎る俣譫 (15m, 2h) 蜷代￠縺ｮ繝・・繧ｿ逕滓・
- parquet蠖｢蠑上〒繝・・繧ｿ菫晏ｭ・

注意: このスクリプトはテストやデバッグ目的でのみ使用してください
      実際のトレーディングやモデル学習には実データを使用することを推奨します
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import argparse
import yaml

# デフォルト設定値
USE_REAL_DATA = False
MAX_SAMPLES = 4320

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('sample_data')

# プロジェクトルートディレクトリ
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / 'data'

# サンプルデータ生成用定数
INTERVALS = ["15m", "2h", "1d"]
ENDPOINTS = [
    'price', 'oi', 'funding', 'liq', 'lsr', 'taker', 'orderbook', 'premium'
]
NUM_SAMPLES = 4320  # 同じく4320サンプルを生成

def ensure_directories():
    """データ保存用ディレクトリを確認・作成"""
    for interval in INTERVALS:
        # 生データディレクトリ
        raw_dir = DATA_DIR / 'raw' / interval
        os.makedirs(raw_dir, exist_ok=True)
        
        # 特徴量データディレクトリ
        features_dir = DATA_DIR / 'features' / interval
        os.makedirs(features_dir, exist_ok=True)
    
    logger.info("ディレクトリ準備完了")

def generate_timestamps(interval, num_samples):
    """指定された時間枠でタイムスタンプを生成"""
    end_time = datetime.now()
    timestamps = []
    
    if interval == "15m":
        for i in range(num_samples):
            timestamps.append(end_time - timedelta(minutes=15 * i))
    elif interval == "2h":
        for i in range(num_samples):
            timestamps.append(end_time - timedelta(hours=2 * i))
    elif interval == "1d":
        for i in range(num_samples):
            timestamps.append(end_time - timedelta(days=1 * i))
    
    return sorted(timestamps)

def generate_price_data(interval, num_samples):
    """価格データの生成"""
    timestamps = generate_timestamps(interval, num_samples)
    
    # 初期値設定
    initial_price = 60000  # BTC初期価格
    volatility = 0.002  # ボラティリティ
    
    # 乱数シード設定（再現性のため）
    np.random.seed(42)
    
    # ランダムウォークでBTC価格をシミュレーション
    changes = np.random.normal(0, volatility, num_samples)
    prices = [initial_price]
    for i in range(1, num_samples):
        # 前の価格に対して変動率を適用
        next_price = prices[-1] * (1 + changes[i])
        prices.append(next_price)
    
    # DataFrameを作成
    df = pd.DataFrame({
        'price_timestamp': timestamps,
        'price_open': prices,
        'price_high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
        'price_low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'price_close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'price_volume': np.random.uniform(100, 1000, num_samples) * 1000000,
    })
    
    # タイムスタンプをインデックスに設定
    df['timestamp'] = df['price_timestamp']
    df = df.set_index('timestamp')
    
    return df

def generate_oi_data(interval, num_samples):
    """オープンインタレストデータの生成"""
    timestamps = generate_timestamps(interval, num_samples)
    
    # 初期値設定
    initial_oi = 2000000000  # 20億ドル
    
    # 乱数シード設定
    np.random.seed(43)
    
    # OIの変動をシミュレーション
    changes = np.random.normal(0, 0.01, num_samples)
    oi_values = [initial_oi]
    for i in range(1, num_samples):
        next_oi = oi_values[-1] * (1 + changes[i])
        oi_values.append(next_oi)
    
    # DataFrameを作成
    df = pd.DataFrame({
        'oi_timestamp': timestamps,
        'oi_value': oi_values,
        'oi_exchange_binance': [v * 0.3 for v in oi_values],
        'oi_exchange_bybit': [v * 0.25 for v in oi_values],
        'oi_exchange_okx': [v * 0.2 for v in oi_values],
        'oi_exchange_deribit': [v * 0.15 for v in oi_values],
        'oi_exchange_bitmex': [v * 0.1 for v in oi_values],
    })
    
    # タイムスタンプをインデックスに設定
    df['timestamp'] = df['oi_timestamp']
    df = df.set_index('timestamp')
    
    return df

def generate_funding_data(interval, num_samples):
    """ファンディングレートデータの生成"""
    timestamps = generate_timestamps(interval, num_samples)
    
    # 乱数シード設定
    np.random.seed(44)
    
    # ファンディングレートをシミュレーション
    funding_rates = np.random.normal(0, 0.0008, num_samples)  # 平均0、標準偏差0.08%
    
    # DataFrameを作成
    df = pd.DataFrame({
        'funding_timestamp': timestamps,
        'funding_rate_binance': funding_rates,
        'funding_rate_bybit': funding_rates + np.random.normal(0, 0.0001, num_samples),
        'funding_rate_okx': funding_rates + np.random.normal(0, 0.0001, num_samples),
        'funding_rate_deribit': funding_rates + np.random.normal(0, 0.0001, num_samples),
        'funding_rate_bitmex': funding_rates + np.random.normal(0, 0.0001, num_samples),
        'funding_rate_avg': funding_rates,
    })
    
    # タイムスタンプをインデックスに設定
    df['timestamp'] = df['funding_timestamp']
    df = df.set_index('timestamp')
    
    return df

def generate_liquidation_data(interval, num_samples):
    """清算データの生成"""
    timestamps = generate_timestamps(interval, num_samples)
    
    # 乱数シード設定
    np.random.seed(45)
    
    # 清算量をシミュレーション - 価格変動が大きい時に増加
    base_liq = np.random.exponential(5000000, num_samples)  # 基本清算量
    
    # DataFrameを作成
    df = pd.DataFrame({
        'liq_timestamp': timestamps,
        'liq_total': base_liq,
        'liq_long': base_liq * np.random.uniform(0.3, 0.7, num_samples),
        'liq_short': base_liq * np.random.uniform(0.3, 0.7, num_samples),
        'liq_exchange_binance': base_liq * np.random.uniform(0.2, 0.4, num_samples),
        'liq_exchange_bybit': base_liq * np.random.uniform(0.15, 0.3, num_samples),
        'liq_exchange_okx': base_liq * np.random.uniform(0.1, 0.2, num_samples),
    })
    
    # タイムスタンプをインデックスに設定
    df['timestamp'] = df['liq_timestamp']
    df = df.set_index('timestamp')
    
    return df

def generate_lsr_data(interval, num_samples):
    """ロングショート比率データの生成"""
    timestamps = generate_timestamps(interval, num_samples)
    
    # 乱数シード設定
    np.random.seed(46)
    
    # LSRをシミュレーション
    base_lsr = np.random.normal(1.0, 0.2, num_samples)  # 平均1.0、標準偏差0.2
    
    # DataFrameを作成
    df = pd.DataFrame({
        'lsr_timestamp': timestamps,
        'lsr_ratio': base_lsr,
        'lsr_long_pct': 50 + (base_lsr - 1) * 25,  # LSR 1.0 = 50/50, 変動に応じて調整
        'lsr_short_pct': 50 - (base_lsr - 1) * 25,
        'lsr_exchange_binance': base_lsr + np.random.normal(0, 0.05, num_samples),
        'lsr_exchange_bybit': base_lsr + np.random.normal(0, 0.05, num_samples),
        'lsr_exchange_okx': base_lsr + np.random.normal(0, 0.05, num_samples),
    })
    
    # タイムスタンプをインデックスに設定
    df['timestamp'] = df['lsr_timestamp']
    df = df.set_index('timestamp')
    
    return df

def generate_taker_data(interval, num_samples):
    """テイカー出来高データの生成"""
    timestamps = generate_timestamps(interval, num_samples)
    
    # 乱数シード設定
    np.random.seed(47)
    
    # テイカー比率をシミュレーション
    base_taker_ratio = np.random.normal(0.0, 0.3, num_samples)  # 平均0、標準偏差0.3
    
    # DataFrameを作成
    df = pd.DataFrame({
        'taker_timestamp': timestamps,
        'taker_buy_ratio': 50 + base_taker_ratio * 10,  # 比率を-50～+50%の範囲に
        'taker_sell_ratio': 50 - base_taker_ratio * 10,
        'taker_exchange_binance': 50 + np.random.normal(0, 5, num_samples),
        'taker_exchange_bybit': 50 + np.random.normal(0, 5, num_samples),
        'taker_exchange_okx': 50 + np.random.normal(0, 5, num_samples),
    })
    
    # タイムスタンプをインデックスに設定
    df['timestamp'] = df['taker_timestamp']
    df = df.set_index('timestamp')
    
    return df

def generate_orderbook_data(interval, num_samples):
    """オーダーブックデータの生成"""
    timestamps = generate_timestamps(interval, num_samples)
    
    # 乱数シード設定
    np.random.seed(48)
    
    # オーダーブック比率をシミュレーション
    base_ob_ratio = np.random.normal(0.0, 0.2, num_samples)  # 平均0、標準偏差0.2
    
    # DataFrameを作成
    df = pd.DataFrame({
        'orderbook_timestamp': timestamps,
        'orderbook_bid_ratio': 50 + base_ob_ratio * 15,  # 比率を-50～+50%の範囲に
        'orderbook_ask_ratio': 50 - base_ob_ratio * 15,
        'orderbook_bid_ask_ratio': 1 + base_ob_ratio * 0.3,
        'orderbook_exchange_binance': 1 + np.random.normal(0, 0.1, num_samples),
        'orderbook_exchange_bybit': 1 + np.random.normal(0, 0.1, num_samples),
        'orderbook_exchange_okx': 1 + np.random.normal(0, 0.1, num_samples),
    })
    
    # タイムスタンプをインデックスに設定
    df['timestamp'] = df['orderbook_timestamp']
    df = df.set_index('timestamp')
    
    return df

def generate_premium_data(interval, num_samples):
    """プレミアム指数データの生成"""
    timestamps = generate_timestamps(interval, num_samples)
    
    # 乱数シード設定
    np.random.seed(49)
    
    # プレミアム指数をシミュレーション
    base_premium = np.random.normal(0.0, 0.001, num_samples)  # 平均0、標準偏差0.1%
    
    # DataFrameを作成
    df = pd.DataFrame({
        'premium_timestamp': timestamps,
        'premium_index': base_premium,
        'premium_exchange_binance': base_premium + np.random.normal(0, 0.0005, num_samples),
        'premium_exchange_bybit': base_premium + np.random.normal(0, 0.0005, num_samples),
        'premium_exchange_okx': base_premium + np.random.normal(0, 0.0005, num_samples),
        'premium_exchange_deribit': base_premium + np.random.normal(0, 0.0005, num_samples),
        'premium_exchange_bitmex': base_premium + np.random.normal(0, 0.0005, num_samples),
    })
    
    # タイムスタンプをインデックスに設定
    df['timestamp'] = df['premium_timestamp']
    df = df.set_index('timestamp')
    
    return df

def generate_and_save_all_data():
    """すべてのエンドポイントとインターバルのデータを生成して保存"""
    ensure_directories()
    
    # 各インターバルごとにデータを生成
    for interval in INTERVALS:
        logger.info(f"インターバル {interval} のデータ生成開始")
        
        # 各エンドポイントのデータを生成して保存
        endpoint_generators = {
            'price': generate_price_data,
            'oi': generate_oi_data,
            'funding': generate_funding_data,
            'liq': generate_liquidation_data,
            'lsr': generate_lsr_data,
            'taker': generate_taker_data,
            'orderbook': generate_orderbook_data,
            'premium': generate_premium_data,
        }
        
        for endpoint, generator in endpoint_generators.items():
            try:
                logger.info(f"エンドポイント {endpoint} のデータ生成")
                df = generator(interval, NUM_SAMPLES)
                
                # ファイルに保存
                output_path = DATA_DIR / 'raw' / interval / f"{endpoint}.parquet"
                df.to_parquet(output_path)
                
                logger.info(f"保存完了: {output_path}, レコード数: {len(df)}")
                
            except Exception as e:
                logger.error(f"エンドポイント {endpoint} のデータ生成中にエラー: {str(e)}")
        
        logger.info(f"インターバル {interval} のデータ生成完了")

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='サンプルデータ生成ツール')
    parser.add_argument('--force', action='store_true', 
                       help='システム設定を無視して強制的にサンプルデータを生成')
    return parser.parse_args()

def main():
    """メイン関数"""
    args = parse_args()
    
    if USE_REAL_DATA and not args.force:
        logger.error("""
        ======================================================================
        エラー: システム設定で実データの使用が指定されています
        
        サンプルデータを生成するには以下のいずれかを実行してください:
        
        1. config/system.yamlの設定を変更:
           use_real_data: false
           
        2. --force オプションを付けて実行:
           python src/generate_sample_data.py --force
        
        警告: サンプルデータはテスト目的専用です。
              本番環境や実際のトレーディングには使用しないでください。
        ======================================================================
        """)
        return 1
    
    if args.force:
        logger.warning("--force オプションが指定されました。システム設定を無視してサンプルデータを生成します。")
    
    logger.info("サンプルデータ生成開始")
    generate_and_save_all_data()
    logger.info("サンプルデータ生成完了")
    return 0

if __name__ == "__main__":
    sys.exit(main())
