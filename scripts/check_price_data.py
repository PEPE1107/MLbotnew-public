#!/usr/bin/env python
"""
parquetファイルのBTC価格データを確認するスクリプト
"""

import os
import pandas as pd
from pathlib import Path

# プロジェクトのルートディレクトリ
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_price_data(interval: str):
    """
    指定された時間枠の価格データを確認します。
    """
    try:
        file_path = ROOT_DIR / f"data/raw/{interval}/price.parquet"
        print(f"ファイルパス: {file_path}")
        
        if not file_path.exists():
            print(f"エラー: {file_path} が見つかりません")
            return
        
        # parquetファイルを読み込み
        df = pd.read_parquet(file_path)
        
        # カラム名を表示
        print(f"カラム: {df.columns.tolist()}")
        
        # 基本統計量を表示
        print("\n基本統計量:")
        print(df.describe())
        
        # 先頭と末尾のデータを表示
        print("\n先頭5行:")
        print(df.head())
        
        print("\n末尾5行:")
        print(df.tail())
        
        # インデックスの情報を表示
        print("\nインデックス情報:")
        print(f"タイプ: {type(df.index)}")
        print(f"開始: {df.index.min()}")
        print(f"終了: {df.index.max()}")
        print(f"期間: {df.index.max() - df.index.min()}")
        print(f"データ数: {len(df)}")
        
        # 実際のBTC価格を表示（最新の価格が実際の価格かを確認）
        print("\n最新のBTC価格:")
        latest_price = df.iloc[-1]["price_close"]
        print(f"最終日時: {df.index[-1]}")
        print(f"価格: {latest_price}")
        
        # データが本物かサンプルデータかを推測
        print("\nデータタイプ分析:")
        # 価格の分散が小さすぎる場合はサンプルデータの可能性
        price_std = df["price_close"].std()
        price_mean = df["price_close"].mean()
        cv = price_std / price_mean  # 変動係数
        
        if cv < 0.05:  # 変動係数が5%未満の場合
            print("警告: 価格の変動が小さすぎます。サンプルデータの可能性があります。")
        else:
            print("価格変動は自然です。実際のデータの可能性が高いです。")
        
        # 価格範囲をチェック（BTCの合理的な範囲）
        min_price = df["price_close"].min()
        max_price = df["price_close"].max()
        print(f"価格範囲: {min_price} - {max_price}")
        
        if min_price < 1000 or max_price > 100000:
            print("警告: 価格範囲が現実的でない可能性があります。")
        
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BTC価格データを確認します")
    parser.add_argument("--interval", "-i", default="15m", help="確認する時間枠 ('15m', '2h' など)")
    
    args = parser.parse_args()
    check_price_data(args.interval)
