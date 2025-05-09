#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
バックテスト結果処理ユーティリティ
------------------------

バックテスト結果の統計情報計算、プロット作成、レポート生成などの
共通ユーティリティ関数を提供します。
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Union
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_stats(backtest_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    バックテスト結果から統計情報を生成
    
    Parameters:
    -----------
    backtest_result : dict
        run_backtest()の戻り値
        
    Returns:
    --------
    dict
        統計情報辞書
    """
    logger.info("バックテスト結果の統計情報を計算中...")
    
    # VectorBTのポートフォリオオブジェクトを取得
    portfolio = backtest_result['portfolio']
    interval = backtest_result['interval']
    
    # 年間取引日数を設定（時間枠ごとに異なる）
    if interval == '15m':
        annual_trading_days = 365 * 24 * 4  # 15分足は1日に4*24回
    elif interval == '2h':
        annual_trading_days = 365 * 12  # 2時間足は1日に12回
    else:  # 1d
        annual_trading_days = 365  # 日足は年間365日
    
    # トレード統計
    trade_stats = portfolio.trades.stats()
    trade_duration = portfolio.trades.duration.mean() if len(portfolio.trades) > 0 else 0
    
    # 基本的なリターン統計
    total_return = portfolio.total_return()
    total_return_pct = portfolio.total_return() * 100
    
    # ドローダウン統計
    drawdown = portfolio.drawdown()
    max_drawdown = portfolio.max_drawdown()
    max_drawdown_pct = portfolio.max_drawdown() * 100
    
    # リスク調整済みリターン - APIの互換性に対応
    try:
        # 新しいバージョン用
        sharpe_ratio = portfolio.sharpe_ratio(year_freq=annual_trading_days)
        sortino_ratio = portfolio.sortino_ratio(year_freq=annual_trading_days)
        calmar_ratio = portfolio.calmar_ratio(year_freq=annual_trading_days)
    except TypeError as e:
        # 古いバージョン用
        try:
            sharpe_ratio = portfolio.sharpe_ratio(risk_free=0.0, year_freq=annual_trading_days)
            sortino_ratio = portfolio.sortino_ratio(risk_free=0.0, year_freq=annual_trading_days)
            calmar_ratio = portfolio.calmar_ratio(year_freq=annual_trading_days)
        except TypeError:
            # フォールバック - 基本的なパラメータのみ
            logger.warning("リスク調整済みリターンの計算でフォールバックを使用します")
            sharpe_ratio = portfolio.sharpe_ratio()
            sortino_ratio = portfolio.sortino_ratio()
            calmar_ratio = portfolio.calmar_ratio()
    
    # トレード詳細
    total_trades = len(portfolio.trades)
    if total_trades > 0:
        # win_rateがメソッドかプロパティかを確認
        try:
            win_rate = portfolio.trades.win_rate
            # オブジェクトかどうかをチェック
            if callable(win_rate):
                win_rate = win_rate()  # メソッドとして呼び出す
        except:
            # フォールバック - 勝ちトレード数 / 全トレード数
            win_trades = len(portfolio.trades[portfolio.trades['pnl'] > 0])
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            
        win_rate_pct = win_rate * 100
        
        # 平均利益と平均損失
        try:
            avg_win = portfolio.trades.winning.pnl.mean() / portfolio.init_cash
            avg_loss = portfolio.trades.losing.pnl.mean() / portfolio.init_cash
        except:
            # 手動で計算
            winning_trades = portfolio.trades[portfolio.trades['pnl'] > 0]
            losing_trades = portfolio.trades[portfolio.trades['pnl'] <= 0]
            avg_win = winning_trades['pnl'].mean() / portfolio.init_cash if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() / portfolio.init_cash if len(losing_trades) > 0 else 0
            
        avg_win_pct = avg_win * 100
        avg_loss_pct = avg_loss * 100
        
        # 勝ちトレードと負けトレードの比率
        if avg_loss != 0:
            payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            payoff_ratio = float('inf')
        
        # プロフィットファクター
        try:
            gross_profit = portfolio.trades.winning.pnl.sum()
            gross_loss = abs(portfolio.trades.losing.pnl.sum())
        except:
            # 手動で計算
            winning_trades = portfolio.trades[portfolio.trades['pnl'] > 0]
            losing_trades = portfolio.trades[portfolio.trades['pnl'] <= 0]
            gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1  # ゼロ除算防止
        
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    else:
        win_rate = 0
        win_rate_pct = 0
        avg_win = 0
        avg_loss = 0
        avg_win_pct = 0
        avg_loss_pct = 0
        payoff_ratio = 0
        profit_factor = 0
    
    # 月次リターン - APIの互換性に対応
    try:
        # returnsがメソッドかプロパティかをチェック
        returns = portfolio.returns
        if callable(returns):
            returns = returns()  # メソッドとして呼び出す
        
        # 月次リターン計算(ME: Month End)
        monthly_returns = returns.resample('ME').apply(
            lambda x: (1 + x).prod() - 1
        )
        positive_months = len(monthly_returns[monthly_returns > 0])
        negative_months = len(monthly_returns[monthly_returns < 0])
        
        # 平均月次リターン（パーセント）
        avg_monthly_return = monthly_returns.mean() * 100 if len(monthly_returns) > 0 else 0
    except Exception as e:
        # フォールバック - 月次データなしの場合
        logger.warning(f"月次リターン計算でエラーが発生: {e}")
        positive_months = 0
        negative_months = 0
        avg_monthly_return = 0
    
    # 最終資産（初期資金からの成長額）
    try:
        # valueがメソッドかプロパティかをチェック
        value = portfolio.value
        if callable(value):
            value = value()  # メソッドとして呼び出す
        final_value = value.iloc[-1]
    except Exception as e:
        logger.warning(f"最終資産の取得でエラーが発生: {e}")
        # フォールバック - 初期資産からの変化率を使用
        final_value = portfolio.init_cash * (1 + portfolio.total_return())
        
    initial_value = portfolio.init_cash
    
    # 年率リターン
    days_in_backtest = (portfolio.wrapper.index[-1] - portfolio.wrapper.index[0]).days
    if days_in_backtest > 0:
        yearly_return = ((final_value / initial_value) ** (365 / days_in_backtest) - 1) * 100
    else:
        yearly_return = 0
    
    # 統計情報辞書
    stats = {
        'interval': interval,
        'total_return': float(total_return),
        'total_return_pct': float(total_return_pct),
        'yearly_return_pct': float(yearly_return),
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        'calmar_ratio': float(calmar_ratio),
        'max_drawdown': float(max_drawdown),
        'max_drawdown_pct': float(max_drawdown_pct),
        'total_trades': int(total_trades),
        'win_rate': float(win_rate),
        'win_rate_pct': float(win_rate_pct),
        'avg_win_pct': float(avg_win_pct),
        'avg_loss_pct': float(avg_loss_pct),
        'payoff_ratio': float(payoff_ratio),
        'profit_factor': float(profit_factor),
        'positive_months': int(positive_months),
        'negative_months': int(negative_months),
        'avg_monthly_return_pct': float(avg_monthly_return),
        'avg_trade_duration': float(trade_duration) if isinstance(trade_duration, (int, float)) else 0,
        'start_date': portfolio.wrapper.index[0].strftime('%Y-%m-%d'),
        'end_date': portfolio.wrapper.index[-1].strftime('%Y-%m-%d'),
        'days_in_backtest': int(days_in_backtest),
        'initial_capital': float(initial_value),
        'final_capital': float(final_value)
    }
    
    logger.info("統計情報計算完了")
    
    # NANとInfinityをフィルタリング 
    cleaned_stats = {}
    for k, v in stats.items():
        try:
            if pd.isna(v) or (isinstance(v, (int, float)) and np.isinf(v)):
                cleaned_stats[k] = 0
            else:
                cleaned_stats[k] = v
        except:
            # どうしても処理できない値の場合は0にフォールバック
            cleaned_stats[k] = 0
    return cleaned_stats

def plot_backtest_with_price(backtest_result: Dict[str, Any], price_data: pd.DataFrame, 
                           result_dir: str) -> str:
    """
    バックテスト結果と価格をプロットし、HTMLファイルとして保存
    
    Parameters:
    -----------
    backtest_result : dict
        run_backtest()の戻り値
    price_data : pd.DataFrame
        価格データを含むデータフレーム
    result_dir : str
        結果を保存するディレクトリパス
        
    Returns:
    --------
    str
        保存したHTMLファイルのパス
    """
    logger.info("バックテスト結果をプロット中...")
    
    # VectorBTのポートフォリオオブジェクトを取得
    portfolio = backtest_result['portfolio']
    interval = backtest_result['interval']
    
    # プロット用のデータを準備
    try:
        # valueがメソッドかプロパティかをチェック
        value = portfolio.value
        if callable(value):
            equity_curve = value().copy()
        else:
            equity_curve = value.copy()
    except Exception as e:
        logger.warning(f"プロット用資産曲線の取得でエラーが発生: {e}")
        # フォールバック - 簡易的な資産曲線生成
        idx = portfolio.wrapper.index
        equity_curve = pd.Series(
            index=idx,
            data=portfolio.init_cash * (1 + portfolio.total_return() * np.linspace(0, 1, len(idx)))
        )
    
    price_series = price_data['close'].copy()
    
    # 同じインデックスに揃える
    start_date = max(equity_curve.index[0], price_series.index[0])
    end_date = min(equity_curve.index[-1], price_series.index[-1])
    equity_curve = equity_curve.loc[start_date:end_date]
    price_series = price_series.loc[start_date:end_date]
    
    # 統計情報を取得
    stats = generate_stats(backtest_result)
    stats_str = [
        f"Total Return: {stats['total_return_pct']:.2f}%",
        f"Yearly Return: {stats['yearly_return_pct']:.2f}%",
        f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}",
        f"Max Drawdown: {stats['max_drawdown_pct']:.2f}%",
        f"Win Rate: {stats['win_rate_pct']:.2f}%",
    ]
    
    # Plotlyのサブプロットを作成（2行1列）
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"Portfolio Value & BTC Price ({interval})", "Drawdown"))
    
    # ポートフォリオ価値のプロット（左軸）
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index, 
            y=equity_curve, 
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # BTCの価格を正規化してプロット（右軸）
    # 価格を初期資金と同じスケールに調整
    initial_capital = portfolio.init_cash
    price_normalized = price_series / price_series.iloc[0] * initial_capital
    
    fig.add_trace(
        go.Scatter(
            x=price_normalized.index, 
            y=price_normalized, 
            name='BTC Price (normalized)',
            line=dict(color='gray', width=1, dash='dot'),
            yaxis='y2'
        ),
        row=1, col=1
    )
    
    # ドローダウンのプロット
    drawdown = portfolio.drawdown()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, 
            y=drawdown * 100, # パーセント表示
            name='Drawdown %',
            fill='tozeroy',
            line=dict(color='red', width=1)
        ),
        row=2, col=1
    )
    
    # レイアウト設定
    fig.update_layout(
        title=f"MLbotnew Backtest Results - {interval}",
        height=800,
        width=1200,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified',
        # 統計情報を注釈として追加
        annotations=[
            dict(
                x=0.01,
                y=0.05,
                xref="paper",
                yref="paper",
                text="<br>".join(stats_str),
                showarrow=False,
                font=dict(family="Arial", size=12),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                align="left"
            )
        ]
    )
    
    # Y軸のタイトル設定
    fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
    fig.update_yaxes(title_text="BTC Price", overlaying="y", side="right", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    
    # X軸のタイトル
    fig.update_xaxes(title_text=f"Date ({interval})", row=2, col=1)
    
    # HTMLファイルとして保存
    html_path = os.path.join(result_dir, 'bt_with_price.html')
    fig.write_html(html_path, full_html=True, include_plotlyjs='cdn')
    
    logger.info(f"プロットを保存しました: {html_path}")
    
    return html_path
