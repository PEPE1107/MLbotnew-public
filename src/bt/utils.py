#!/usr/bin/env python
"""
utils.py - バックテスト関連ユーティリティ

機能:
- 価格シリーズ取得
- バックテストプロットに価格ライン追加
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# ロギング設定
logger = logging.getLogger(__name__)

# プロジェクトのルートディレクトリ
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def get_price_series(interval: str) -> pd.Series:
    """
    指定された時間枠の価格シリーズを取得します。

    Args:
        interval: 時間枠 ('15m', '2h', '1d' など)

    Returns:
        pd.Series: 価格シリーズ
    """
    fn = ROOT_DIR / f"data/raw/{interval}/price.parquet"
    
    if not fn.exists():
        error_msg = f"価格データファイルが見つかりません: {fn}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # 生のparquetファイルを読み込む
        df = pd.read_parquet(fn)
        
        # データフレームの構造を確認
        logger.info(f"読み込みデータフレーム構造: {df.columns.tolist()}")
        logger.info(f"データ期間: {df.index.min()} ~ {df.index.max()}")
        
        # 'price_close' カラムが存在するか確認
        if 'price_close' not in df.columns:
            available_cols = ', '.join(df.columns.tolist())
            error_msg = f"'price_close' カラムが見つかりません。利用可能なカラム: {available_cols}"
            logger.error(error_msg)
            raise KeyError(error_msg)
        
        # price_closeをシリーズに変換し名前を設定
        ser = df["price_close"].rename("BTC Spot Price")
        
        # インデックスが日時型になっていない場合は変換
        if not isinstance(ser.index, pd.DatetimeIndex):
            ser.index = pd.to_datetime(ser.index)
        
        # 重複インデックスを削除（後のデータを保持）
        if ser.index.has_duplicates:
            logger.warning(f"重複するインデックスを検出: {sum(ser.index.duplicated())}件")
            ser = ser[~ser.index.duplicated(keep='last')]
        
        # 欠損値を削除
        if ser.isna().any():
            logger.warning(f"欠損値を検出: {ser.isna().sum()}件")
            ser = ser.dropna()
        
        # ソート
        ser = ser.sort_index()
        
        # 詳細なログ出力
        logger.info(f"価格データを読み込みました: {interval} - {len(ser)}件")
        logger.info(f"価格期間: {ser.index.min()} ~ {ser.index.max()}")
        logger.info(f"価格範囲: {ser.min():.2f} - {ser.max():.2f} USD")
        
        return ser
        
    except Exception as e:
        error_msg = f"価格データの読み込みエラー ({interval}): {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def add_price_trace(fig, interval: str, run_id: str = None):
    """
    バックテスト結果プロットに価格ラインを追加します。

    Args:
        fig: plotlyまたはmatplotlibのfigureオブジェクト
        interval: 時間枠 ('15m', '2h', '1d' など)
        run_id: 実行ID (省略時は現在時刻を使用)

    Returns:
        fig: 更新されたfigureオブジェクト
    """
    try:
        # 価格シリーズ取得
        price_ser = get_price_series(interval)
        
        # 適切なリサンプリング
        if price_ser.index.inferred_freq is None:
            logger.info(f"時間枠 {interval} の適切なリサンプリングを適用")
            
            if interval == '15m':
                price_ser = price_ser.resample('15min').last()
            elif interval == '2h':
                price_ser = price_ser.resample('2h').last()
            elif interval == '1d':
                price_ser = price_ser.resample('D').last()
            else:
                # その他の時間枠の場合
                match = re.match(r'(\d+)([mhd])', interval)
                if match:
                    num, unit = match.groups()
                    if unit == 'm':
                        price_ser = price_ser.resample(f'{num}min').last()
                    elif unit == 'h':
                        price_ser = price_ser.resample(f'{num}H').last()
                    elif unit == 'd':
                        price_ser = price_ser.resample(f'{num}D').last()
                else:
                    logger.warning(f"未知の時間枠フォーマット: {interval}")
                    price_ser = price_ser.resample(interval).last()
        
        # リサンプリング後の欠損値を処理
        if price_ser.isna().any():
            logger.warning(f"リサンプリング後の欠損値: {price_ser.isna().sum()}件")
            price_ser = price_ser.fillna(method='ffill')
            
        # リサンプリング結果のログ
        logger.info(f"リサンプリング後のデータ: {len(price_ser)}件")
        logger.info(f"価格期間: {price_ser.index.min()} ~ {price_ser.index.max()}")
    except Exception as e:
        logger.error(f"価格データ取得エラー: {str(e)}")
        raise ValueError(f"価格データ取得に失敗しました: {str(e)}") from e
    
    # figureオブジェクトの種類をチェック
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure as mplFigure
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        has_plotly = True
    except ImportError:
        has_plotly = False
        logger.warning("plotlyがインストールされていません。HTMLプロットは生成されません。")
    
    # レポートディレクトリの設定
    if run_id:
        output_dir = ROOT_DIR / f"reports/{interval}/{run_id}"
    else:
        output_dir = ROOT_DIR / f"reports/{interval}"
    os.makedirs(output_dir, exist_ok=True)
    
    # matplotlibのFigureの場合
    if isinstance(fig, mplFigure):
        # 現在のaxesを取得
        ax1 = fig.axes[0]
        
        # 右側のY軸を追加
        ax2 = ax1.twinx()
        
        # 価格ラインのプロット
        ax2.plot(price_ser.index, price_ser, 'gray', linestyle=':', linewidth=1, label='BTC Price')
        
        # 軸ラベル設定
        ax2.set_ylabel('Price (USD)')
        
        # 凡例の設定
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # PNGとして保存
        plot_file = output_dir / "bt_with_price.png"
        fig.savefig(plot_file)
        logger.info(f"PNGプロット保存: {plot_file}")
        
        if not has_plotly:
            return fig
            
        # HTMLとして出力するための対応
        # 実データを使用したHTMLプロット生成
        html_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # エクイティ曲線データの取得
        equity_data = None
        for line in ax1.get_lines():
            if "equity" in line.get_label().lower() or len(ax1.get_lines()) == 1:
                # エクイティラインを検出またはデフォルトで最初のライン
                equity_data = {
                    'x': line.get_xdata(),
                    'y': line.get_ydata()
                }
                break
        
        # エクイティデータがある場合はプロット
        if equity_data:
            # 日付がdatetimeの場合は文字列に変換
            x_data = equity_data['x']
            if hasattr(x_data, 'tolist'):
                x_data = x_data.tolist()
            
            html_fig.add_trace(
                go.Scatter(
                    x=x_data, 
                    y=equity_data['y'], 
                    name="Equity", 
                    line=dict(color="blue", width=2)
                ),
                secondary_y=False
            )
        else:
            # 実データが取得できない場合はダミーデータを使用
            logger.warning("エクイティデータが取得できないため、ダミーデータを使用します")
            html_fig.add_trace(
                go.Scatter(
                    x=price_ser.index, 
                    y=[1.0 + i*0.01 for i in range(len(price_ser))], 
                    name="Equity (Placeholder)", 
                    line=dict(color="blue", width=2)
                ),
                secondary_y=False
            )
        
        # 価格データのプロット
        html_fig.add_trace(
            go.Scatter(
                x=price_ser.index, 
                y=price_ser.values, 
                name="BTC Price", 
                line=dict(width=1, dash="dot", color="gray")
            ),
            secondary_y=True
        )
        
        # プロット範囲の調整
        y_min = price_ser.min() * 0.98
        y_max = price_ser.max() * 1.02
        
        # レイアウト設定
        html_fig.update_layout(
            title_text=f"Backtest Results with BTC Price ({interval})",
            xaxis_title="Date",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        html_fig.update_yaxes(title_text="Equity", secondary_y=False)
        html_fig.update_yaxes(
            title_text="Price (USD)", 
            secondary_y=True,
            range=[y_min, y_max]
        )
        
        # HTMLとして保存
        html_file = output_dir / "bt_with_price.html"
        html_fig.write_html(str(html_file))
        logger.info(f"HTMLプロット保存: {html_file}")
        
    # plotlyのFigureの場合
    elif has_plotly and hasattr(fig, 'add_trace'):
        # 価格ラインを追加
        fig.add_trace(
            go.Scatter(
                x=price_ser.index, 
                y=price_ser.values,
                name="BTC Price", 
                mode="lines",
                line=dict(width=1, dash="dot", color="gray")
            ),
            secondary_y=True
        )
        
        # プロット範囲の調整
        y_min = price_ser.min() * 0.98
        y_max = price_ser.max() * 1.02
        
        # 右側Y軸の設定
        fig.update_layout(
            title_text=f"Backtest Results with BTC Price ({interval})",
            xaxis_title="Date",
            yaxis_title="Equity",
            yaxis2=dict(
                title="Price (USD)",
                range=[y_min, y_max]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # HTMLとして保存
        html_file = output_dir / "bt_with_price.html"
        fig.write_html(str(html_file))
        logger.info(f"HTMLプロット保存: {html_file}")
    
    # その他のタイプの場合（非対応）
    else:
        logger.warning(f"未対応のFigureタイプです: {type(fig)}。価格ラインは追加されません。")
    
    return fig
