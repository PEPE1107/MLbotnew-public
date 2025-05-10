#!/usr/bin/env python
"""
バックテスト結果のポートフォリオ情報を確認するスクリプト
"""
import os
import sys
import pickle
from pathlib import Path

# アプリケーションルートへのパス設定
app_home = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(app_home))
sys.path.append(str(app_home / 'src'))

# バックテストモジュールからSimplePortfolioクラスをインポート
from src.backtest import SimplePortfolio

def main():
    # 最新のポートフォリオファイルをロード
    app_home = Path(os.path.dirname(os.path.abspath(__file__)))
    portfolio_file = app_home / 'reports' / '15m' / 'backtest_portfolio_20250507_221845.pkl'
    
    print(f"Loading portfolio file: {portfolio_file}")
    
    with open(portfolio_file, 'rb') as f:
        portfolio = pickle.load(f)
    
    # 統計情報を表示
    stats = portfolio.stats()
    print("\nPortfolio Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # グラフ生成用の情報を表示
    if hasattr(portfolio, 'df'):
        print("\nDataFrame Info:")
        print(f"  Shape: {portfolio.df.shape}")
        print(f"  Columns: {portfolio.df.columns.tolist()}")
        
        # 最初と最後の行を表示
        print("\nFirst 3 rows:")
        print(portfolio.df.head(3))
        
        print("\nLast 3 rows:")
        print(portfolio.df.tail(3))
        
        # 累積リターンの最終値
        if 'cumulative_returns' in portfolio.df.columns:
            final_return = portfolio.df['cumulative_returns'].iloc[-1]
            print(f"\nFinal Cumulative Return: {final_return:.4f}")
            print(f"Total Return: {(final_return - 1) * 100:.2f}%")

if __name__ == "__main__":
    main()
