#!/usr/bin/env python
"""
daily_trend.py - 日足トレンドフィルター

機能:
- 日足価格データから、200日移動平均(EMA)に基づいたトレンドフラグを生成
- 上昇トレンド(1)と下降トレンド(0)を判定
- 結果を parquet 形式で保存
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'daily_trend.log'), mode='a')
    ]
)
logger = logging.getLogger('daily_trend')

# プロジェクトルートディレクトリ
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / 'data'

def daily_filter():
    """日足価格データからトレンドフィルターを生成
    
    200日EMAを計算し、価格がEMAより上か下かに基づいてトレンドフラグを作成します。
    フラグは1（上昇トレンド）または0（下降トレンド）の値を取ります。
    """
    try:
        # 日足価格データ読み込み
        price_path = DATA_DIR / 'raw' / '1d' / 'price.parquet'
        
        if not price_path.exists():
            logger.error(f"日足価格データが見つかりません: {price_path}")
            return
            
        df = pd.read_parquet(price_path)
        
        # 'price_close'列が存在するか確認（prefixが付いている可能性があるため）
        close_col = [col for col in df.columns if 'close' in col.lower()]
        if not close_col:
            logger.error(f"価格データに終値(close)列が見つかりません: {df.columns}")
            return
            
        # 終値列を使用
        close_col = close_col[0]
        
        # 200日EMA計算
        ema200 = df[close_col].ewm(span=200).mean()
        
        # トレンドフラグ生成 (1=上昇, 0=下降)
        trend_flag = ((df[close_col] > ema200).astype(int)).shift(1)
        
        # 時間インデックスのローカライズを解除（Noneと置換）
        if hasattr(trend_flag.index, 'tz_localize'):
            trend_flag.index = trend_flag.index.tz_localize(None)
        
        # 出力ディレクトリ確認・作成
        output_dir = DATA_DIR / 'features' / '1d'
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存
        output_path = output_dir / 'trend_flag.parquet'
        # Series を DataFrame に変換
        trend_df = pd.DataFrame({'trend_flag': trend_flag})
        trend_df.to_parquet(output_path)
        
        logger.info(f"トレンドフラグを生成しました: {output_path}, レコード数: {len(trend_flag)}")
        logger.info(f"上昇トレンド期間: {trend_flag.sum()}/{len(trend_flag)} ({trend_flag.mean()*100:.1f}%)")
        
        return trend_flag
        
    except Exception as e:
        logger.error(f"トレンドフラグ生成エラー: {str(e)}", exc_info=True)
        return None

def main():
    """メイン関数"""
    try:
        # ログディレクトリ作成
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'), exist_ok=True)
        
        # トレンドフィルター実行
        daily_filter()
        
        return 0
    
    except Exception as e:
        logger.error(f"実行エラー: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
