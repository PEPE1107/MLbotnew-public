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
    
    # ロングとショートのトレード統計
    try:
        # ポジションタイプごとのトレード数の計算
        if hasattr(portfolio.trades, 'position_idx'):
            # VectorBT 新バージョン
            long_trades = portfolio.trades[portfolio.trades.position_idx == 0]
            short_trades = portfolio.trades[portfolio.trades.position_idx == 1]
        elif 'direction' in portfolio.trades.columns:
            # 方向情報がある場合（旧バージョンや拡張版）
            long_trades = portfolio.trades[portfolio.trades['direction'] > 0]
            short_trades = portfolio.trades[portfolio.trades['direction'] < 0]
        else:
            # フォールバック - シグナルに基づいて予測
            df = backtest_result.get('df', pd.DataFrame())
            if 'signal' in df.columns:
                # シグナルの符号に基づいて集計
                long_days = (df['signal'] > 0).sum()
                short_days = (df['signal'] < 0).sum()
                # おおよそのトレード回数を推定
                long_trades = pd.DataFrame({'pnl': [0] * (total_trades * long_days // (long_days + short_days + 1))})
                short_trades = pd.DataFrame({'pnl': [0] * (total_trades * short_days // (long_days + short_days + 1))})
            else:
                # 詳細情報が取得できない場合
                long_trades = pd.DataFrame({'pnl': [0]})
                short_trades = pd.DataFrame({'pnl': [0]})
        
        # トレード統計の追加
        long_count = len(long_trades)
        short_count = len(short_trades)
        total_with_direction = long_count + short_count
        
        # ロングとショートの比率
        if total_with_direction > 0:
            long_ratio = long_count / total_with_direction
            short_ratio = short_count / total_with_direction
        else:
            long_ratio = 0.5
            short_ratio = 0.5
        
        # 各方向のパフォーマンス
        long_pnl = long_trades['pnl'].sum() / portfolio.init_cash if long_count > 0 else 0
        short_pnl = short_trades['pnl'].sum() / portfolio.init_cash if short_count > 0 else 0
        
        # 方向ごとの勝率
        long_wins = (long_trades['pnl'] > 0).sum() if long_count > 0 else 0
        short_wins = (short_trades['pnl'] > 0).sum() if short_count > 0 else 0
        
        long_win_rate = long_wins / long_count if long_count > 0 else 0
        short_win_rate = short_wins / short_count if short_count > 0 else 0
        
        # 統計に追加
        stats.update({
            'long_trades': int(long_count),
            'short_trades': int(short_count),
            'long_ratio': float(long_ratio),
            'short_ratio': float(short_ratio),
            'long_pnl_pct': float(long_pnl * 100),
            'short_pnl_pct': float(short_pnl * 100),
            'long_win_rate': float(long_win_rate),
            'short_win_rate': float(short_win_rate),
            'direction_bias': 'LONG' if long_ratio > 0.6 else ('SHORT' if short_ratio > 0.6 else 'NEUTRAL')
        })
    except Exception as e:
        logger.warning(f"ロング/ショート統計の計算でエラーが発生: {e}")
        stats.update({
            'long_trades': 0,
            'short_trades': 0,
            'long_ratio': 0.5,
            'short_ratio': 0.5,
            'direction_bias': 'UNKNOWN'
        })
    
    # 特徴量の重要度（入力データに基づく簡易分析）
    try:
        df = backtest_result.get('df', pd.DataFrame())
        signal = backtest_result.get('signal', pd.Series())
        
        if not df.empty and not signal.empty:
            # 特徴量の候補を取得（価格・ボリューム・テクニカル指標）
            feature_columns = []
            
            # 基本OHLCV列
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            feature_columns.extend([col for col in ohlcv_columns if col in df.columns])
            
            # テクニカル指標
            technical_patterns = ['rsi', 'ema', 'sma', 'macd', 'atr', 'cci', 'trend', 'momentum']
            for col in df.columns:
                if any(pattern in col.lower() for pattern in technical_patterns):
                    feature_columns.append(col)
            
            # Coinglassデータ列
            for prefix in ['funding_', 'oi_', 'liq_', 'lsr_', 'premium_']:
                for col in df.columns:
                    if col.startswith(prefix):
                        feature_columns.append(col)
            
            # 特徴量に対するシグナルの相関係数を計算
            feature_importance = {}
            for feature in feature_columns:
                if feature in df.columns:
                    # 一部のカラムは型変換が必要
                    try:
                        feature_data = pd.to_numeric(df[feature], errors='coerce')
                        if not feature_data.empty and not signal.empty:
                            # 有効なデータポイントのみで相関を計算
                            valid_indices = ~(feature_data.isna() | signal.isna())
                            if valid_indices.sum() > 10:  # 十分なデータポイントがある場合
                                corr = feature_data[valid_indices].corr(signal[valid_indices])
                                if not np.isnan(corr):
                                    feature_importance[feature] = abs(corr)  # 絶対値で重要度を評価
                    except Exception as e:
                        logger.debug(f"特徴量 '{feature}' の相関計算でエラー: {e}")
            
            # 重要度でソート
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            # 上位10件を取得
            top_features = {k: float(v) for k, v in list(sorted_importance.items())[:10]}
            
            # 特徴量グループごとの集計（Coinglassデータ種別ごとなど）
            grouped_importance = {}
            
            # OHLCVグループ
            ohlcv_imp = sum(v for k, v in sorted_importance.items() if any(k.endswith(suffix) for suffix in ['_o', '_h', '_l', '_c', '_v']) or k in ohlcv_columns)
            if ohlcv_imp > 0:
                grouped_importance['price_data'] = float(ohlcv_imp)
            
            # 需給指標グループ
            for prefix, group_name in [
                ('funding_', 'funding_rates'),
                ('oi_', 'open_interest'),
                ('liq_', 'liquidations'),
                ('lsr_', 'long_short_ratio'),
                ('premium_', 'premium')
            ]:
                group_imp = sum(v for k, v in sorted_importance.items() if k.startswith(prefix))
                if group_imp > 0:
                    grouped_importance[group_name] = float(group_imp)
            
            # テクニカル指標グループ
            tech_imp = sum(v for k, v in sorted_importance.items() 
                         if any(pattern in k.lower() for pattern in ['rsi', 'ema', 'sma', 'macd', 'atr']) 
                         and not any(k.startswith(prefix) for prefix in ['funding_', 'oi_', 'liq_', 'lsr_', 'premium_']))
            if tech_imp > 0:
                grouped_importance['technical_indicators'] = float(tech_imp)
            
            # トレンド/モメンタム指標
            trend_imp = sum(v for k, v in sorted_importance.items() if 'trend' in k.lower() or 'momentum' in k.lower())
            if trend_imp > 0:
                grouped_importance['trend_momentum'] = float(trend_imp)
            
            # 統計に追加
            stats.update({
                'feature_importance': top_features,
                'feature_groups': grouped_importance
            })
    except Exception as e:
        logger.warning(f"特徴量重要度の計算でエラーが発生: {e}")
        stats.update({
            'feature_importance': {},
            'feature_groups': {}
        })
    
    # 追加の統計情報
    try:
        # ドローダウン期間の分析
        dd_series = drawdown
        
        # 平均ドローダウン期間（バー数）
        dd_periods = []
        is_in_dd = False
        current_dd_start = 0
        
        for i, dd_value in enumerate(dd_series):
            if dd_value < 0 and not is_in_dd:
                # ドローダウン開始
                is_in_dd = True
                current_dd_start = i
            elif dd_value == 0 and is_in_dd:
                # ドローダウン終了
                is_in_dd = False
                dd_periods.append(i - current_dd_start)
        
        # 最後のドローダウンが続いている場合
        if is_in_dd:
            dd_periods.append(len(dd_series) - current_dd_start)
        
        avg_dd_bars = np.mean(dd_periods) if dd_periods else 0
        max_dd_bars = np.max(dd_periods) if dd_periods else 0
        
        # 時間枠に基づく期間の単位変換（バー数→日数）
        bars_per_day = 1  # デフォルト（日足）
        if interval == '15m':
            bars_per_day = 24 * 4  # 15分足は1日に4*24回
        elif interval == '2h':
            bars_per_day = 12  # 2時間足は1日に12回
        
        avg_dd_days = avg_dd_bars / bars_per_day
        max_dd_days = max_dd_bars / bars_per_day
        
        # 月別リターン
        if hasattr(portfolio, 'returns') and not callable(portfolio.returns):
            monthly_returns_by_date = {}
            try:
                monthly_returns_series = portfolio.returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
                for date, value in monthly_returns_series.items():
                    if not pd.isna(value):
                        monthly_returns_by_date[date.strftime('%Y-%m')] = float(value * 100)  # パーセント表示
            except Exception as e:
                logger.debug(f"月別リターンの計算でエラー: {e}")
        else:
            monthly_returns_by_date = {}
        
        # 追加統計
        stats.update({
            'avg_drawdown_duration_bars': float(avg_dd_bars),
            'max_drawdown_duration_bars': float(max_dd_bars),
            'avg_drawdown_duration_days': float(avg_dd_days),
            'max_drawdown_duration_days': float(max_dd_days),
            'monthly_returns': monthly_returns_by_date,
            'ulcer_index': float(np.sqrt(np.mean(np.square(dd_series.fillna(0))))),
            'recovery_factor': float(total_return_pct / abs(max_drawdown_pct)) if max_drawdown_pct != 0 else 0
        })
    except Exception as e:
        logger.warning(f"追加統計の計算でエラーが発生: {e}")
    
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
