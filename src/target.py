#!/usr/bin/env python
"""
target.py - 目的変数生成モジュール

機能:
- ATR (Average True Range) ベースの予測目標を生成
- 将来リターンが ATR の一定割合を超えた場合にシグナルを生成
- 各時間枠で目的変数を計算し保存
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'target.log'), mode='a')
    ]
)
logger = logging.getLogger('target')

# プロジェクトルートディレクトリ
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = ROOT_DIR / 'config'
DATA_DIR = ROOT_DIR / 'data'

# ログディレクトリ作成
os.makedirs(ROOT_DIR / 'logs', exist_ok=True)

class TargetGenerator:
    """目的変数生成クラス"""
    
    def __init__(self, config_dir: Path = CONFIG_DIR):
        """初期化
        
        Args:
            config_dir: 設定ファイルディレクトリ
        """
        self.config_dir = config_dir
        self.intervals = self._load_intervals()
    
    def _load_intervals(self) -> List[str]:
        """時間枠設定を読み込む
        
        Returns:
            List[str]: 時間枠リスト
        """
        try:
            with open(self.config_dir / 'intervals.yaml', 'r') as f:
                config = yaml.safe_load(f)
            return config['intervals']
        except Exception as e:
            logger.error(f"時間枠設定の読み込みに失敗しました: {e}")
            raise
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range (ATR) を計算
        
        Args:
            high: 高値
            low: 低値
            close: 終値
            period: 期間
            
        Returns:
            pd.Series: ATR 値
        """
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr
    
    def make_target(self, interval: str, k: int = 3, thr: float = 0.15) -> pd.Series:
        """ATR ベースの目的変数を生成
        
        Args:
            interval: 時間枠
            k: 将来リターンの期間
            thr: シグナル発生の ATR 倍率 (閾値)
            
        Returns:
            pd.Series: 目的変数 (1=ロング, -1=ショート, 0=ニュートラル)
        """
        logger.info(f"目的変数生成開始: {interval}, k={k}, thr={thr}")
        
        # マージされたデータを読み込む
        merged_path = DATA_DIR / 'features' / interval / 'merged.parquet'
        
        if not os.path.exists(merged_path):
            raise FileNotFoundError(f"マージされたデータが見つかりません: {merged_path}")
        
        df = pd.read_parquet(merged_path)
        
        # 必要なカラムの確認
        price_cols = [col for col in df.columns if col.startswith('price_')]
        required_cols = ['price_close']
        high_col = next((col for col in price_cols if 'high' in col), None)
        low_col = next((col for col in price_cols if 'low' in col), None)
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"必要なカラムが見つかりません: {required_cols}")
        
        # 高値・安値がない場合は終値から生成
        if high_col is None:
            logger.warning("高値データが見つかりません。終値から生成します")
            df['price_high'] = df['price_close'] * 1.005  # 終値 + 0.5%
            high_col = 'price_high'
        
        if low_col is None:
            logger.warning("安値データが見つかりません。終値から生成します")
            df['price_low'] = df['price_close'] * 0.995   # 終値 - 0.5%
            low_col = 'price_low'
        
        # ATR 計算
        atr = self.calculate_atr(df[high_col], df[low_col], df['price_close'], 14)
        
        # 将来リターン計算
        future_return = df['price_close'].pct_change(k).shift(-k)
        
        # ATR 比率による閾値
        atr_ratio = atr / df['price_close']
        threshold = thr * atr_ratio
        
        # 目的変数生成
        y = np.where(
            future_return > threshold, 1,      # ロングシグナル
            np.where(
                future_return < -threshold, -1,  # ショートシグナル
                0                               # ニュートラル
            )
        )
        
        # Series に変換
        target = pd.Series(y, index=df.index, name='y')
        
        # カウント集計
        long_count = (target == 1).sum()
        short_count = (target == -1).sum()
        neutral_count = (target == 0).sum()
        total_count = len(target)
        
        logger.info(f"目的変数分布: ロング={long_count}({long_count/total_count:.2%}), " +
                   f"ショート={short_count}({short_count/total_count:.2%}), " +
                   f"ニュートラル={neutral_count}({neutral_count/total_count:.2%}), " +
                   f"合計={total_count}")
        
        # 保存
        output_path = DATA_DIR / 'features' / interval / f'y_k{k}_thr{thr:.2f}.parquet'
        target.to_frame().to_parquet(output_path)
        
        logger.info(f"目的変数生成完了: {output_path}")
        
        return target
    
    def make_all_targets(self, k_values: List[int] = [3, 6, 12], thr_values: List[float] = [0.15, 0.25]) -> Dict:
        """複数のパラメータで目的変数を生成
        
        Args:
            k_values: 将来リターン期間のリスト
            thr_values: 閾値のリスト
            
        Returns:
            Dict: 生成した目的変数の辞書
        """
        results = {}
        
        for interval in self.intervals:
            logger.info(f"時間枠 {interval} の目的変数生成開始")
            interval_results = {}
            
            for k in k_values:
                for thr in thr_values:
                    key = f"k{k}_thr{thr:.2f}"
                    try:
                        target = self.make_target(interval, k, thr)
                        interval_results[key] = target
                    except Exception as e:
                        logger.error(f"目的変数生成エラー: {interval}, k={k}, thr={thr}, エラー: {str(e)}")
                        continue
            
            results[interval] = interval_results
            logger.info(f"時間枠 {interval} の目的変数生成完了")
        
        return results

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='目的変数生成')
    parser.add_argument('--interval', type=str, help='時間枠 (例: 15m, 2h)')
    parser.add_argument('--k', type=int, default=3, help='将来リターンの期間')
    parser.add_argument('--thr', type=float, default=0.15, help='シグナル発生の ATR 倍率 (閾値)')
    parser.add_argument('--all', action='store_true', help='全時間枠・全パラメータで生成')
    return parser.parse_args()

def main():
    """メイン関数"""
    try:
        # ログディレクトリ作成
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'), exist_ok=True)
        
        args = parse_args()
        generator = TargetGenerator()
        
        if args.all:
            # 全時間枠・全パラメータで生成
            generator.make_all_targets()
        elif args.interval:
            # 単一時間枠・指定パラメータで生成
            generator.make_target(args.interval, args.k, args.thr)
        else:
            # 全時間枠・指定パラメータで生成
            for interval in generator.intervals:
                generator.make_target(interval, args.k, args.thr)
        
        return 0
    
    except Exception as e:
        logger.error(f"実行エラー: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
