#!/usr/bin/env python
"""
run_simple_backtest.py - シンプルなバックテスト実行スクリプト

特徴量を直接データから抽出して、バックテストを実行します。
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path

# アプリケーションルートへのパスを設定
app_home = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(app_home))

# src ディレクトリをインポートパスに追加
src_path = app_home / "src"
sys.path.append(str(src_path))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('simple_backtest')

def main():
    """メイン関数"""
    import argparse
    from backtest import BacktestRunner

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='シンプルなバックテスト実行スクリプト')
    parser.add_argument('--interval', '-i', type=str, default='15m',
                        help='時間枠 (デフォルト: 15m)')
    parser.add_argument('--model-type', '-m', type=str, default='lightgbm',
                        choices=['lightgbm', 'catboost'], help='モデル種類')
    parser.add_argument('--threshold', '-t', type=float, default=0.55,
                        help='シグナル閾値 (0.5 〜 1.0)')
    args = parser.parse_args()

    # BacktestRunner インスタンスの作成
    runner = BacktestRunner()

    try:
        # 指定された時間枠のデータを読み込む
        logger.info(f"時間枠 {args.interval} のデータを読み込み中...")
        df = runner.load_data(args.interval)
        logger.info(f"データ読み込み完了: {len(df)} レコード")
        
        # モデルの読み込み
        logger.info(f"{args.model_type} モデルを読み込み中...")
        model_dir = runner.model_dir / args.interval / args.model_type
        model_files = list(model_dir.glob("model_fold1_*.txt"))
        if not model_files:
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_dir}")
        
        model_file = sorted(model_files)[-1]  # 最新のモデルを使用
        logger.info(f"モデルファイル: {model_file}")
        
        # 特徴量を抽出
        logger.info("データから特徴量を抽出しています...")
        feature_cols = [col for col in df.columns if col.endswith('_zscore')]
        logger.info(f"抽出された特徴量: {len(feature_cols)}個")
        
        if args.model_type == 'lightgbm':
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(model_file))
        else:
            import catboost as ctb
            if 'Classification' in model_file.stem:
                model = ctb.CatBoostClassifier()
            else:
                model = ctb.CatBoostRegressor()
            model.load_model(str(model_file))
        
        # シグナル生成
        logger.info("シグナル生成中...")
        signals_df = runner.generate_signals(df, model, feature_cols, args.threshold)
        
        # バックテスト実行
        logger.info("バックテスト実行中...")
        portfolio, stats = runner.run_backtest(signals_df)
        
        # 結果表示
        logger.info(f"バックテスト結果 ({args.interval}):")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
            
        return 0
    
    except Exception as e:
        logger.error(f"バックテスト失敗: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
