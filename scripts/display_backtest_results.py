#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
バックテスト結果表示スクリプト
--------------------

バックテスト結果のstats.jsonファイルを読み込み、見やすい形式で表示します。
特徴量の重要度、ロング/ショート比率など、詳細な統計情報を出力します。
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def load_stats(stats_file: str) -> Dict[str, Any]:
    """統計情報ファイルを読み込む"""
    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        return stats
    except Exception as e:
        print(f"エラー：ファイル読み込みに失敗しました - {e}")
        return {}

def print_separator(char="=", length=80):
    """区切り線を表示"""
    print(char * length)

def print_header(title: str):
    """セクションヘッダーを表示"""
    print_separator()
    print(f" {title} ".center(78, "-"))
    print_separator()

def print_basic_stats(stats: Dict[str, Any]):
    """基本的な統計情報を表示"""
    print_header("バックテスト基本情報")
    
    print(f"時間枠:         {stats.get('interval', 'N/A')}")
    print(f"開始日:         {stats.get('start_date', 'N/A')}")
    print(f"終了日:         {stats.get('end_date', 'N/A')}")
    print(f"期間:           {stats.get('days_in_backtest', 0)}日")
    print(f"初期資金:       ${stats.get('initial_capital', 0):,.2f}")
    print(f"最終資金:       ${stats.get('final_capital', 0):,.2f}")
    print()
    
    print(f"総リターン:     {stats.get('total_return_pct', 0):.2f}%")
    print(f"年率リターン:   {stats.get('yearly_return_pct', 0):.2f}%")
    print(f"最大ドローダウン: {stats.get('max_drawdown_pct', 0):.2f}%")
    print(f"カルマー比率:   {stats.get('calmar_ratio', 0):.4f}")
    print(f"シャープレシオ: {stats.get('sharpe_ratio', 0):.4f}")
    print(f"ソルティノレシオ: {stats.get('sortino_ratio', 0):.4f}")
    
    # リカバリーファクターを表示（追加された統計）
    if 'recovery_factor' in stats:
        print(f"リカバリーファクター: {stats.get('recovery_factor', 0):.4f}")
    
    # アルサー指数を表示（追加された統計）
    if 'ulcer_index' in stats:
        print(f"アルサー指数:   {stats.get('ulcer_index', 0):.4f}")

def print_trade_stats(stats: Dict[str, Any]):
    """トレード統計を表示"""
    print_header("トレード統計")
    
    print(f"総トレード数:   {stats.get('total_trades', 0)}回")
    print(f"勝率:           {stats.get('win_rate_pct', 0):.2f}%")
    print(f"平均利益:       {stats.get('avg_win_pct', 0):.2f}%")
    print(f"平均損失:       {stats.get('avg_loss_pct', 0):.2f}%")
    print(f"ペイオフ比率:   {stats.get('payoff_ratio', 0):.2f}")
    print(f"プロフィットファクター: {stats.get('profit_factor', 0):.2f}")
    print(f"平均トレード期間: {stats.get('avg_trade_duration', 0):.1f}バー")
    
    # ロング/ショート統計（追加された統計）
    print("\n【ロング/ショート詳細】")
    if 'long_trades' in stats and 'short_trades' in stats:
        print(f"ロングトレード: {stats.get('long_trades', 0)}回 ({stats.get('long_ratio', 0)*100:.1f}%)")
        print(f"ショートトレード: {stats.get('short_trades', 0)}回 ({stats.get('short_ratio', 0)*100:.1f}%)")
        print(f"ロングPnL:      {stats.get('long_pnl_pct', 0):.2f}%")
        print(f"ショートPnL:    {stats.get('short_pnl_pct', 0):.2f}%")
        print(f"ロング勝率:     {stats.get('long_win_rate', 0)*100:.2f}%")
        print(f"ショート勝率:   {stats.get('short_win_rate', 0)*100:.2f}%")
        print(f"方向バイアス:   {stats.get('direction_bias', 'NEUTRAL')}")

def print_drawdown_stats(stats: Dict[str, Any]):
    """ドローダウン統計を表示"""
    if any(key in stats for key in ['avg_drawdown_duration_bars', 'max_drawdown_duration_bars']):
        print_header("ドローダウン統計")
        
        print(f"最大ドローダウン: {stats.get('max_drawdown_pct', 0):.2f}%")
        
        if 'avg_drawdown_duration_bars' in stats:
            print(f"平均DD期間(バー): {stats.get('avg_drawdown_duration_bars', 0):.1f}バー")
        
        if 'max_drawdown_duration_bars' in stats:
            print(f"最長DD期間(バー): {stats.get('max_drawdown_duration_bars', 0):.1f}バー")
        
        if 'avg_drawdown_duration_days' in stats:
            print(f"平均DD期間(日): {stats.get('avg_drawdown_duration_days', 0):.1f}日")
        
        if 'max_drawdown_duration_days' in stats:
            print(f"最長DD期間(日): {stats.get('max_drawdown_duration_days', 0):.1f}日")

def print_feature_importance(stats: Dict[str, Any]):
    """特徴量の重要度を表示"""
    if 'feature_importance' in stats and stats['feature_importance']:
        print_header("特徴量の重要度 (上位10件)")
        
        for feature, importance in stats['feature_importance'].items():
            print(f"{feature}: {importance:.4f}")
    
    if 'feature_groups' in stats and stats['feature_groups']:
        print("\n【特徴量グループ】")
        
        for group, importance in stats['feature_groups'].items():
            print(f"{group}: {importance:.4f}")

def plot_monthly_returns(stats: Dict[str, Any], output_dir: Optional[str] = None):
    """月別リターンをプロット"""
    if 'monthly_returns' in stats and stats['monthly_returns']:
        monthly_returns = stats['monthly_returns']
        
        if not monthly_returns:
            return None
        
        # 日付でソート
        sorted_months = sorted(monthly_returns.keys())
        returns = [monthly_returns[month] for month in sorted_months]
        
        # 色の設定
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        # プロット作成
        plt.figure(figsize=(12, 6))
        bars = plt.bar(sorted_months, returns, color=colors)
        
        # 軸ラベル設定
        plt.xlabel('Month')
        plt.ylabel('Return (%)')
        plt.title('Monthly Returns')
        
        # x軸のラベルを調整
        if len(sorted_months) > 12:
            plt.xticks(rotation=90)
        
        # 目盛線の追加
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 余白の調整
        plt.tight_layout()
        
        # 0%の水平線を追加
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 保存先がある場合は保存
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'monthly_returns.png')
            plt.savefig(output_path)
            print(f"\n月別リターングラフを保存しました: {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            plt.close()
    
    return None

def print_monthly_stats(stats: Dict[str, Any]):
    """月別統計を表示"""
    print_header("月別パフォーマンス")
    
    print(f"月次プラス: {stats.get('positive_months', 0)}ヶ月")
    print(f"月次マイナス: {stats.get('negative_months', 0)}ヶ月")
    print(f"平均月次リターン: {stats.get('avg_monthly_return_pct', 0):.2f}%")
    
    # 月別リターンの詳細（追加された統計）
    if 'monthly_returns' in stats and stats['monthly_returns']:
        print("\n【月別リターン詳細】")
        
        monthly_returns = stats['monthly_returns']
        sorted_months = sorted(monthly_returns.keys())
        
        # テーブルヘッダー
        print(f"{'月':^10}|{'リターン':^10}")
        print("-" * 21)
        
        # 月別データ
        for month in sorted_months:
            ret = monthly_returns[month]
            print(f"{month:^10}|{ret:^10.2f}%")

def generate_summary_for_readme(stats: Dict[str, Any]) -> str:
    """README用のサマリーテキストを生成"""
    summary = f"""
## バックテスト結果サマリー ({stats.get('interval', 'N/A')})

- **期間**: {stats.get('start_date', 'N/A')} ～ {stats.get('end_date', 'N/A')} ({stats.get('days_in_backtest', 0)}日間)
- **総リターン**: {stats.get('total_return_pct', 0):.2f}%
- **年率リターン**: {stats.get('yearly_return_pct', 0):.2f}%
- **最大ドローダウン**: {stats.get('max_drawdown_pct', 0):.2f}%
- **シャープレシオ**: {stats.get('sharpe_ratio', 0):.4f}
- **総トレード数**: {stats.get('total_trades', 0)}回
- **勝率**: {stats.get('win_rate_pct', 0):.2f}%

### ロング/ショート分析
- ロング比率: {stats.get('long_ratio', 0)*100:.1f}% ({stats.get('long_trades', 0)}回)
- ショート比率: {stats.get('short_ratio', 0)*100:.1f}% ({stats.get('short_trades', 0)}回)
- ロングPnL: {stats.get('long_pnl_pct', 0):.2f}%
- ショートPnL: {stats.get('short_pnl_pct', 0):.2f}%
- 方向バイアス: {stats.get('direction_bias', 'NEUTRAL')}

### 主要特徴量グループ
"""
    
    # 特徴量グループの追加
    if 'feature_groups' in stats and stats['feature_groups']:
        for group, importance in stats['feature_groups'].items():
            summary += f"- {group}: {importance:.4f}\n"
    else:
        summary += "- (特徴量グループ情報なし)\n"
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='バックテスト結果表示ツール')
    parser.add_argument('--stats', type=str, required=True,
                        help='stats.jsonファイルのパス')
    parser.add_argument('--plot', action='store_true',
                        help='月別リターンをプロットするかどうか')
    parser.add_argument('--output', type=str, default=None,
                        help='出力ディレクトリ（グラフ保存用）')
    parser.add_argument('--readme', action='store_true',
                        help='README用のサマリーテキストを生成')
    
    args = parser.parse_args()
    
    # 統計ファイルの読み込み
    stats = load_stats(args.stats)
    
    if not stats:
        print("エラー：統計情報が読み込めませんでした。")
        return 1
    
    # README用サマリーのみの場合
    if args.readme:
        summary = generate_summary_for_readme(stats)
        print(summary)
        
        if args.output:
            readme_path = os.path.join(args.output, 'README.md')
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"\nREADMEを保存しました: {readme_path}")
        
        return 0
    
    # 通常の詳細表示
    print_basic_stats(stats)
    print()
    print_trade_stats(stats)
    print()
    print_drawdown_stats(stats)
    print()
    print_monthly_stats(stats)
    print()
    print_feature_importance(stats)
    
    # 月別リターンのプロット
    if args.plot:
        plot_monthly_returns(stats, args.output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
