#!/usr/bin/env python
"""
feature_functions.py - 特徴量関数モジュール

機能:
- 特徴量正規化
- 価格関連特徴量
- OI (Open Interest)関連特徴量
- 資金調達率関連特徴量
- 清算関連特徴量
- ロング/ショート比関連特徴量
- テイカー関連特徴量
- オーダーブック関連特徴量
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple


def zscore_normalize(series: pd.Series, window: int = 100) -> pd.Series:
    """Z-score正規化を適用
    
    Args:
        series: 対象の系列データ
        window: 移動平均ウィンドウサイズ
        
    Returns:
        pd.Series: 正規化されたシリーズ
    """
    # 移動平均と標準偏差を計算
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    
    # ゼロ割防止
    std = std.replace(0, 1)
    
    # Z-score計算
    normalized = (series - mean) / std
    
    # 極端な値をクリッピング (-3から3の範囲に)
    normalized = normalized.clip(-3, 3)
    
    return normalized


def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """価格データから特徴量を構築
    
    Args:
        df: 価格データを含むデータフレーム
        
    Returns:
        pd.DataFrame: 特徴量が追加されたデータフレーム
    """
    result = df.copy()
    
    # 価格列の存在を確認
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in result.columns for col in required_cols):
        # プレフィックス付きの列名をチェック
        prefixed_cols = [f"price_{col}" for col in required_cols]
        if all(col in result.columns for col in prefixed_cols):
            # プレフィックス付きの列名があれば変数を更新
            required_cols = prefixed_cols
        else:
            # 必要な価格データがない場合は元のデータを返す
            return result
    
    # 列名を標準化（プレフィックスがある場合は保持）
    prefix = "price_" if required_cols[0].startswith("price_") else ""
    price_open = f"{prefix}open"
    price_high = f"{prefix}high"
    price_low = f"{prefix}low"
    price_close = f"{prefix}close"
    
    # リターン計算
    result[f'{prefix}returns'] = result[price_close].pct_change()
    result[f'{prefix}returns_next'] = result[f'{prefix}returns'].shift(-1)  # 次の期間のリターン（ラベル用）
    
    # ボラティリティ計算
    for window in [12, 24, 48]:  # 3時間、6時間、12時間（15分足の場合）
        result[f'{prefix}volatility_{window}'] = result[f'{prefix}returns'].rolling(window).std()
    
    # ATR計算
    high_low = result[price_high] - result[price_low]
    high_close = (result[price_high] - result[price_close].shift()).abs()
    low_close = (result[price_low] - result[price_close].shift()).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    for window in [7, 14, 21]:
        result[f'{prefix}atr_{window}'] = true_range.rolling(window).mean()
    
    # 移動平均計算
    for window in [5, 10, 20, 50, 100, 200]:
        # 単純移動平均 (SMA)
        result[f'{prefix}sma_{window}'] = result[price_close].rolling(window=window).mean()
        
        # 指数移動平均 (EMA)
        result[f'{prefix}ema_{window}'] = result[price_close].ewm(span=window, adjust=False).mean()
        
        # 移動平均からの乖離率
        result[f'{prefix}sma_{window}_dist'] = (result[price_close] / result[f'{prefix}sma_{window}'] - 1) * 100
        result[f'{prefix}ema_{window}_dist'] = (result[price_close] / result[f'{prefix}ema_{window}'] - 1) * 100
    
    # 各時間枠のボリンジャーバンド
    for window in [20, 50]:
        mid = result[price_close].rolling(window=window).mean()
        std = result[price_close].rolling(window=window).std()
        
        result[f'{prefix}bb_upper_{window}'] = mid + (std * 2)
        result[f'{prefix}bb_lower_{window}'] = mid - (std * 2)
        result[f'{prefix}bb_width_{window}'] = (result[f'{prefix}bb_upper_{window}'] - result[f'{prefix}bb_lower_{window}']) / mid
        
        # %B指標（価格のボリンジャーバンド内での相対位置、0～1）
        result[f'{prefix}bb_pctb_{window}'] = (result[price_close] - result[f'{prefix}bb_lower_{window}']) / (result[f'{prefix}bb_upper_{window}'] - result[f'{prefix}bb_lower_{window}'])
    
    # RSI計算
    for window in [7, 14, 21]:
        # 上昇幅と下落幅を計算
        delta = result[price_close].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 平均上昇幅と平均下落幅を計算
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # RSI計算
        rs = avg_gain / avg_loss.replace(0, 1e-9)  # ゼロ割防止
        result[f'{prefix}rsi_{window}'] = 100 - (100 / (1 + rs))
    
    # MACD計算
    ema_12 = result[price_close].ewm(span=12, adjust=False).mean()
    ema_26 = result[price_close].ewm(span=26, adjust=False).mean()
    result[f'{prefix}macd'] = ema_12 - ema_26
    result[f'{prefix}macd_signal'] = result[f'{prefix}macd'].ewm(span=9, adjust=False).mean()
    result[f'{prefix}macd_hist'] = result[f'{prefix}macd'] - result[f'{prefix}macd_signal']
    
    # 価格モメンタム
    for window in [5, 10, 20]:
        result[f'{prefix}momentum_{window}'] = result[price_close] / result[price_close].shift(window) - 1
    
    return result


def build_oi_features(df: pd.DataFrame) -> pd.DataFrame:
    """OI（未決済約定）関連の特徴量を構築
    
    Args:
        df: OIデータを含むデータフレーム
        
    Returns:
        pd.DataFrame: 特徴量が追加されたデータフレーム
    """
    result = df.copy()
    
    # OI列の存在を確認
    oi_cols = [col for col in df.columns if 'open_interest' in col.lower() or 'oi' in col.lower()]
    if not oi_cols:
        # OIデータがない場合は元のデータを返す
        return result
    
    # 設定するOI列名
    oi_col = oi_cols[0]
    price_col = 'close' if 'close' in df.columns else 'price_close' if 'price_close' in df.columns else None
    
    # OI変化率
    result[f'{oi_col}_change'] = result[oi_col].diff()
    result[f'{oi_col}_change_pct'] = result[oi_col].pct_change() * 100
    
    # OI移動平均
    for window in [12, 24, 48, 96]:  # 3h, 6h, 12h, 24h (15分足基準)
        result[f'{oi_col}_ma_{window}'] = result[oi_col].rolling(window=window).mean()
        result[f'{oi_col}_ma_{window}_dist'] = (result[oi_col] / result[f'{oi_col}_ma_{window}'] - 1) * 100
    
    # OI累積変化
    for window in [4, 8, 16, 32]:
        result[f'{oi_col}_cum_change_{window}'] = result[f'{oi_col}_change'].rolling(window=window).sum()
        result[f'{oi_col}_cum_change_pct_{window}'] = result[f'{oi_col}_change_pct'].rolling(window=window).sum()
    
    # 価格に対するOIの比率（価格列がある場合）
    if price_col:
        result[f'{oi_col}_price_ratio'] = result[oi_col] / result[price_col]
        
        # OI変化と価格変化の関係性
        result[f'{oi_col}_price_divergence'] = result[f'{oi_col}_change_pct'] - result[price_col].pct_change() * 100
        
        # 価格変動時のOI反応（符号付き）
        price_change = result[price_col].pct_change()
        result[f'{oi_col}_with_price_up'] = result[f'{oi_col}_change'] * (price_change > 0)
        result[f'{oi_col}_with_price_down'] = result[f'{oi_col}_change'] * (price_change < 0)
    
    return result


def build_funding_features(df: pd.DataFrame) -> pd.DataFrame:
    """資金調達率関連の特徴量を構築
    
    Args:
        df: 資金調達率データを含むデータフレーム
        
    Returns:
        pd.DataFrame: 特徴量が追加されたデータフレーム
    """
    result = df.copy()
    
    # 資金調達率列の存在を確認
    funding_cols = [col for col in df.columns if 'funding' in col.lower() and 'rate' in col.lower()]
    if not funding_cols:
        # 資金調達率データがない場合は元のデータを返す
        return result
    
    # 設定する資金調達率列名
    funding_col = funding_cols[0]
    
    # 資金調達率の累積 (8h, 24h, 48h, 96h)
    # 資金調達率は通常8時間ごとに発生するため、累積値が重要
    for window in [4, 12, 24, 48]:  # 2h足なら4=8h, 12=24h
        result[f'{funding_col}_cum_{window}'] = result[funding_col].rolling(window).sum()
    
    # 資金調達率の正負の分離
    result[f'{funding_col}_positive'] = result[funding_col].clip(lower=0)
    result[f'{funding_col}_negative'] = result[funding_col].clip(upper=0)
    
    # 資金調達率の移動平均
    for window in [12, 24, 48]:
        result[f'{funding_col}_ma_{window}'] = result[funding_col].rolling(window=window).mean()
    
    # 資金調達率のボラティリティ
    for window in [24, 48]:
        result[f'{funding_col}_std_{window}'] = result[funding_col].rolling(window=window).std()
    
    # 資金調達率の乖離（現在値と移動平均の差）
    for window in [12, 24, 48]:
        result[f'{funding_col}_deviation_{window}'] = result[funding_col] - result[f'{funding_col}_ma_{window}']
    
    # 資金調達率の方向（正/負の持続期間）
    result[f'{funding_col}_sign'] = np.sign(result[funding_col])
    result[f'{funding_col}_sign_change'] = result[f'{funding_col}_sign'].diff().fillna(0) != 0
    
    # 資金調達率の偏り（正の回数 - 負の回数）/ 全体
    for window in [24, 48]:
        pos_count = (result[funding_col] > 0).rolling(window=window).sum()
        neg_count = (result[funding_col] < 0).rolling(window=window).sum()
        result[f'{funding_col}_bias_{window}'] = (pos_count - neg_count) / window
    
    return result


def build_liq_features(df: pd.DataFrame) -> pd.DataFrame:
    """清算関連の特徴量を構築
    
    Args:
        df: 清算データを含むデータフレーム
        
    Returns:
        pd.DataFrame: 特徴量が追加されたデータフレーム
    """
    result = df.copy()
    
    # 清算列の存在を確認
    liq_cols = [col for col in df.columns if 'liquidat' in col.lower() or 'liq' in col.lower()]
    if not liq_cols:
        # 清算データがない場合は元のデータを返す
        return result
    
    # 設定する清算列名
    liq_col = liq_cols[0]
    price_col = 'close' if 'close' in df.columns else 'price_close' if 'price_close' in df.columns else None
    
    # ロング/ショート清算の分離（列が存在する場合）
    liq_long_col = next((col for col in df.columns if 'long' in col.lower() and 'liquidat' in col.lower()), None)
    liq_short_col = next((col for col in df.columns if 'short' in col.lower() and 'liquidat' in col.lower()), None)
    
    # 清算の移動合計
    for window in [4, 8, 16, 24]:  # 1h, 2h, 4h, 6h (15分足基準)
        result[f'{liq_col}_sum_{window}'] = result[liq_col].rolling(window=window).sum()
        
        # ロング/ショート清算（列が存在する場合）
        if liq_long_col:
            result[f'{liq_long_col}_sum_{window}'] = result[liq_long_col].rolling(window=window).sum()
        if liq_short_col:
            result[f'{liq_short_col}_sum_{window}'] = result[liq_short_col].rolling(window=window).sum()
    
    # 清算の偏り（ロング/ショート清算の比） - 両方の列がある場合
    if liq_long_col and liq_short_col:
        for window in [8, 16, 24]:
            long_sum = result[liq_long_col].rolling(window=window).sum()
            short_sum = result[liq_short_col].rolling(window=window).sum()
            
            # ゼロ除算を防ぐ
            long_sum = long_sum.replace(0, 1e-9)
            short_sum = short_sum.replace(0, 1e-9)
            
            result[f'liq_ratio_{window}'] = long_sum / short_sum
            result[f'liq_bias_{window}'] = (long_sum - short_sum) / (long_sum + short_sum)
    
    # 清算の急増検出（Z-score）
    for window in [24, 48, 96]:  # 6h, 12h, 24h (15分足基準)
        result[f'{liq_col}_zscore_{window}'] = zscore_normalize(result[liq_col], window)
    
    # 価格に対する清算量の比率（価格列がある場合）
    if price_col:
        for window in [8, 16, 24]:
            liq_sum = result[liq_col].rolling(window=window).sum()
            result[f'{liq_col}_price_ratio_{window}'] = liq_sum / result[price_col]
    
    return result


def build_lsr_features(df: pd.DataFrame) -> pd.DataFrame:
    """ロング/ショート比関連の特徴量を構築
    
    Args:
        df: LSRデータを含むデータフレーム
        
    Returns:
        pd.DataFrame: 特徴量が追加されたデータフレーム
    """
    result = df.copy()
    
    # LSR列の存在を確認
    lsr_cols = [col for col in df.columns if 'long_short' in col.lower() or 'lsr' in col.lower() or 'ls_ratio' in col.lower()]
    if not lsr_cols:
        # LSRデータがない場合は元のデータを返す
        return result
    
    # 設定するLSR列名
    lsr_col = lsr_cols[0]
    
    # 対数変換（ロングとショートを対称にする）
    result[f'{lsr_col}_log'] = np.log(result[lsr_col])
    
    # 中立値からの乖離（1.0が中立）
    result[f'{lsr_col}_deviation'] = result[lsr_col] - 1.0
    
    # LSRの移動平均
    for window in [8, 16, 32, 64]:
        result[f'{lsr_col}_ma_{window}'] = result[lsr_col].rolling(window=window).mean()
        result[f'{lsr_col}_ma_{window}_dist'] = (result[lsr_col] / result[f'{lsr_col}_ma_{window}'] - 1) * 100
    
    # LSRの変化率
    result[f'{lsr_col}_change'] = result[lsr_col].diff()
    result[f'{lsr_col}_change_pct'] = result[lsr_col].pct_change() * 100
    
    # LSRのボラティリティ
    for window in [16, 32, 64]:
        result[f'{lsr_col}_std_{window}'] = result[lsr_col].rolling(window=window).std()
        result[f'{lsr_col}_zscore_{window}'] = zscore_normalize(result[lsr_col], window)
    
    # LSRの極値（高値・安値からの位置）
    for window in [24, 48, 96]:
        lsr_max = result[lsr_col].rolling(window=window).max()
        lsr_min = result[lsr_col].rolling(window=window).min()
        result[f'{lsr_col}_range_pos_{window}'] = (result[lsr_col] - lsr_min) / (lsr_max - lsr_min).replace(0, 1)
    
    # LSRの傾き
    for window in [8, 16, 32]:
        result[f'{lsr_col}_slope_{window}'] = (result[lsr_col] - result[lsr_col].shift(window)) / window
    
    # LSRの加速度
    result[f'{lsr_col}_acceleration'] = result[f'{lsr_col}_change'].diff()
    
    return result


def build_taker_features(df: pd.DataFrame) -> pd.DataFrame:
    """テイカー関連の特徴量を構築
    
    Args:
        df: テイカー量データを含むデータフレーム
        
    Returns:
        pd.DataFrame: 特徴量が追加されたデータフレーム
    """
    result = df.copy()
    
    # テイカー列の存在を確認（買いと売りの両方があるか）
    taker_buy_cols = [col for col in df.columns if 'taker' in col.lower() and 'buy' in col.lower()]
    taker_sell_cols = [col for col in df.columns if 'taker' in col.lower() and 'sell' in col.lower()]
    
    if not (taker_buy_cols and taker_sell_cols):
        # テイカーデータがない場合は元のデータを返す
        return result
    
    # 設定するテイカー列名
    taker_buy_col = taker_buy_cols[0]
    taker_sell_col = taker_sell_cols[0]
    
    # テイカーバイ/セルの合計と比率
    result['taker_total'] = result[taker_buy_col] + result[taker_sell_col]
    result['taker_buy_ratio'] = result[taker_buy_col] / result['taker_total'].replace(0, 1)
    result['taker_sell_ratio'] = result[taker_sell_col] / result['taker_total'].replace(0, 1)
    
    # テイカーバイ/セルの偏り
    result['taker_bias'] = (result[taker_buy_col] - result[taker_sell_col]) / result['taker_total'].replace(0, 1)
    
    # テイカー移動平均
    for window in [12, 24, 48]:
        # 総量の移動平均
        result[f'taker_total_ma_{window}'] = result['taker_total'].rolling(window=window).mean()
        
        # バイ/セル比のバイアス
        buy_sum = result[taker_buy_col].rolling(window=window).sum()
        sell_sum = result[taker_sell_col].rolling(window=window).sum()
        result[f'taker_bias_{window}'] = (buy_sum - sell_sum) / (buy_sum + sell_sum).replace(0, 1)
    
    # テイカー量の急増検出（Z-score）
    for window in [24, 48, 96]:
        result[f'taker_buy_zscore_{window}'] = zscore_normalize(result[taker_buy_col], window)
        result[f'taker_sell_zscore_{window}'] = zscore_normalize(result[taker_sell_col], window)
        result[f'taker_total_zscore_{window}'] = zscore_normalize(result['taker_total'], window)
    
    # テイカー量の変化率
    result['taker_buy_change'] = result[taker_buy_col].pct_change()
    result['taker_sell_change'] = result[taker_sell_col].pct_change()
    result['taker_total_change'] = result['taker_total'].pct_change()
    
    # テイカー量の累積変化
    for window in [8, 16, 24]:
        result[f'taker_buy_cum_{window}'] = result['taker_buy_change'].rolling(window).sum()
        result[f'taker_sell_cum_{window}'] = result['taker_sell_change'].rolling(window).sum()
        result[f'taker_total_cum_{window}'] = result['taker_total_change'].rolling(window).sum()
    
    return result


def build_orderbook_features(df: pd.DataFrame) -> pd.DataFrame:
    """オーダーブック関連の特徴量を構築
    
    Args:
        df: オーダーブックデータを含むデータフレーム
        
    Returns:
        pd.DataFrame: 特徴量が追加されたデータフレーム
    """
    result = df.copy()
    
    # オーダーブック列の存在を確認（板の厚さや情報）
    bid_cols = [col for col in df.columns if 'bid' in col.lower()]
    ask_cols = [col for col in df.columns if 'ask' in col.lower()]
    
    if not (bid_cols and ask_cols):
        # オーダーブックデータがない場合は元のデータを返す
        return result
    
    # 代表的な板厚指標の列を選択
    bid_depth_col = next((col for col in bid_cols if 'depth' in col.lower()), bid_cols[0])
    ask_depth_col = next((col for col in ask_cols if 'depth' in col.lower()), ask_cols[0])
    
    # 板の厚さの合計と比率
    result['ob_total_depth'] = result[bid_depth_col] + result[ask_depth_col]
    result['ob_bid_ratio'] = result[bid_depth_col] / result['ob_total_depth'].replace(0, 1)
    result['ob_ask_ratio'] = result[ask_depth_col] / result['ob_total_depth'].replace(0, 1)
    
    # 板の厚さの非対称性
    result['ob_imbalance'] = (result[bid_depth_col] - result[ask_depth_col]) / result['ob_total_depth'].replace(0, 1)
    
    # 板の厚さの移動平均
    for window in [12, 24, 48]:
        result[f'ob_bid_ma_{window}'] = result[bid_depth_col].rolling(window=window).mean()
        result[f'ob_ask_ma_{window}'] = result[ask_depth_col].rolling(window=window).mean()
        result[f'ob_total_ma_{window}'] = result['ob_total_depth'].rolling(window=window).mean()
        
        # 現在の板厚と移動平均の比率
        result[f'ob_bid_ma_ratio_{window}'] = result[bid_depth_col] / result[f'ob_bid_ma_{window}'].replace(0, 1)
        result[f'ob_ask_ma_ratio_{window}'] = result[ask_depth_col] / result[f'ob_ask_ma_{window}'].replace(0, 1)
        
        # 板の非対称性の移動平均
        result[f'ob_imbalance_ma_{window}'] = result['ob_imbalance'].rolling(window=window).mean()
    
    # 板の厚さ変化率
    result['ob_bid_change'] = result[bid_depth_col].pct_change()
    result['ob_ask_change'] = result[ask_depth_col].pct_change()
    
    # 急増検出（Z-score）
    for window in [24, 48, 96]:
        result[f'ob_bid_zscore_{window}'] = zscore_normalize(result[bid_depth_col], window)
        result[f'ob_ask_zscore_{window}'] = zscore_normalize(result[ask_depth_col], window)
        result[f'ob_imbalance_zscore_{window}'] = zscore_normalize(result['ob_imbalance'], window)
    
    return result
