#!/usr/bin/env python
"""
builders.py - Feature generation module

Features:
- Generate features from various data sources (price, oi, funding, etc.)
- Normalize features using rolling Z-score
- Calculate features for each timeframe and save them
"""

import os
import sys
import yaml
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats

# Project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIG_DIR = ROOT_DIR / 'config'
DATA_DIR = ROOT_DIR / 'data'

# Ensure log directory exists
os.makedirs(ROOT_DIR / 'logs', exist_ok=True)

# Logger setup
logger = logging.getLogger('features.builders')


def zscore_normalize(series: pd.Series, window: int = 100) -> pd.Series:
    """Normalize a time series using rolling Z-score

    Args:
        series: Input time series
        window: Rolling window size

    Returns:
        pd.Series: Normalized time series
    """
    # Pre-process rows with infinities or NaNs
    clean_series = series.replace([np.inf, -np.inf], np.nan)

    # Calculate moving average and standard deviation
    mean = clean_series.rolling(window=window, min_periods=window//2).mean()
    std = clean_series.rolling(window=window, min_periods=window//2).std()

    # Prevent division by zero
    std = std.replace(0, np.nan)

    # Calculate Z-score
    z_score = (clean_series - mean) / std

    # Clip outliers (-3 to 3 range)
    z_score = z_score.clip(-3, 3)

    # Fill NaNs with 0
    z_score = z_score.fillna(0)

    return z_score


def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate price-related features

    Args:
        df: Input dataframe

    Returns:
        pd.DataFrame: Dataframe with added features
    """
    price_cols = [col for col in df.columns if col.startswith('price_')]

    if not price_cols or 'price_close' not in df.columns:
        logger.warning("Price data not found")
        return df

    # Basic technical indicators

    # Returns
    for period in [1, 3, 6, 12, 24]:
        df[f'price_return_{period}'] = df['price_close'].pct_change(period)

    # Volatility
    for window in [12, 24, 48]:
        df[f'price_volatility_{window}'] = df['price_close'].pct_change().rolling(window=window).std()

    # Moving averages
    for window in [8, 24, 48, 96]:
        df[f'price_ma_{window}'] = df['price_close'].rolling(window=window).mean()

        # Moving average deviation
        df[f'price_ma_deviation_{window}'] = (df['price_close'] / df[f'price_ma_{window}'] - 1)

    # RSI
    for window in [14, 28]:
        delta = df['price_close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        df[f'price_rsi_{window}'] = 100 - (100 / (1 + rs))

    # ATR (Average True Range)
    if all(col in df.columns for col in ['price_high', 'price_low']):
        for window in [14, 28]:
            tr1 = df['price_high'] - df['price_low']
            tr2 = (df['price_high'] - df['price_close'].shift()).abs()
            tr3 = (df['price_low'] - df['price_close'].shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df[f'price_atr_{window}'] = tr.rolling(window=window).mean()

            # ATR ratio (ATR / price)
            df[f'price_atr_ratio_{window}'] = df[f'price_atr_{window}'] / df['price_close']

    # MACD
    ema12 = df['price_close'].ewm(span=12, adjust=False).mean()
    ema26 = df['price_close'].ewm(span=26, adjust=False).mean()
    df['price_macd_line'] = ema12 - ema26
    df['price_macd_signal'] = df['price_macd_line'].ewm(span=9, adjust=False).mean()
    df['price_macd_hist'] = df['price_macd_line'] - df['price_macd_signal']

    # Normalize all features with Z-score
    feature_cols = [col for col in df.columns if col.startswith('price_') and col not in price_cols]

    for col in feature_cols:
        df[f'{col}_zscore'] = zscore_normalize(df[col])

    logger.info(f"Price feature generation complete: {len(feature_cols)} features")
    return df


def build_oi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate open interest related features

    Args:
        df: Input dataframe

    Returns:
        pd.DataFrame: Dataframe with added features
    """
    oi_cols = [col for col in df.columns if col.startswith('oi_')]

    if not oi_cols:
        logger.warning("Open interest data not found")
        return df

    # Identify main OI column
    main_oi_col = None
    for col in ['oi_total', 'oi_value', 'oi_c']:
        if col in df.columns:
            main_oi_col = col
            break

    if main_oi_col is None:
        logger.warning("Main open interest column not found")
        return df

    # OI change rate
    for period in [1, 3, 6, 12, 24]:
        df[f'oi_change_{period}'] = df[main_oi_col].pct_change(period)

    # OI volatility
    for window in [12, 24, 48]:
        df[f'oi_volatility_{window}'] = df[main_oi_col].pct_change().rolling(window=window).std()

    # OI momentum
    for window in [12, 24, 48]:
        df[f'oi_momentum_{window}'] = (df[main_oi_col].diff(window) / df[main_oi_col].shift(window))

    # OI and price ratio
    if 'price_close' in df.columns:
        df['oi_price_ratio'] = df[main_oi_col] / df['price_close']

        # OI change and price change relationship
        df['oi_price_correlation'] = (np.sign(df[main_oi_col].pct_change()) ==
                                     np.sign(df['price_close'].pct_change())).astype(float)

        # OI and price divergence
        for window in [12, 24]:
            oi_ma = df[main_oi_col].rolling(window=window).mean()
            price_ma = df['price_close'].rolling(window=window).mean()

            df[f'oi_price_divergence_{window}'] = (
                (df[main_oi_col] > oi_ma) & (df['price_close'] < price_ma) |
                (df[main_oi_col] < oi_ma) & (df['price_close'] > price_ma)
            ).astype(float)

    # Normalize all features with Z-score
    feature_cols = [col for col in df.columns if col.startswith('oi_') and col not in oi_cols]

    for col in feature_cols:
        df[f'{col}_zscore'] = zscore_normalize(df[col])

    logger.info(f"Open interest feature generation complete: {len(feature_cols)} features")
    return df


def build_funding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate funding rate related features

    Args:
        df: Input dataframe

    Returns:
        pd.DataFrame: Dataframe with added features
    """
    funding_cols = [col for col in df.columns if col.startswith('funding_')]

    if not funding_cols:
        logger.warning("Funding rate data not found")
        return df

    # Identify main funding rate column
    main_funding_col = None
    for col in ['funding_rate', 'funding_c', 'funding_value']:
        if col in df.columns:
            main_funding_col = col
            break

    if main_funding_col is None:
        logger.warning("Main funding rate column not found")
        return df

    # Cumulative funding rate
    for window in [6, 12, 24, 48]:
        df[f'funding_cum_{window}'] = df[main_funding_col].rolling(window=window).sum()

    # Funding rate moving average
    for window in [6, 12, 24]:
        df[f'funding_ma_{window}'] = df[main_funding_col].rolling(window=window).mean()

    # Funding rate standard deviation
    for window in [12, 24, 48]:
        df[f'funding_std_{window}'] = df[main_funding_col].rolling(window=window).std()

    # Funding rate absolute value
    df['funding_abs'] = df[main_funding_col].abs()

    # Funding rate sign
    df['funding_sign'] = np.sign(df[main_funding_col])

    # Funding rate streak (how many periods with the same sign)
    df['funding_streak'] = df['funding_sign'].groupby(
        (df['funding_sign'] != df['funding_sign'].shift()).cumsum()
    ).cumcount() + 1

    # Funding rate deviation from extremes
    for window in [24, 48, 96]:
        funding_max = df[main_funding_col].rolling(window=window).max()
        funding_min = df[main_funding_col].rolling(window=window).min()
        funding_range = funding_max - funding_min

        # Relative position in range (0-1)
        df[f'funding_relative_pos_{window}'] = (df[main_funding_col] - funding_min) / funding_range.replace(0, np.finfo(float).eps)

    # Price relationship
    if 'price_close' in df.columns:
        # Funding rate and price change correlation
        for window in [12, 24, 48]:
            price_change = df['price_close'].pct_change()
            df[f'funding_price_corr_{window}'] = df[main_funding_col].rolling(window=window).corr(price_change)

    # Normalize all features with Z-score
    feature_cols = [col for col in df.columns if col.startswith('funding_') and col not in funding_cols]

    for col in feature_cols:
        df[f'{col}_zscore'] = zscore_normalize(df[col])

    logger.info(f"Funding rate feature generation complete: {len(feature_cols)} features")
    return df


def build_liq_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate liquidation related features

    Args:
        df: Input dataframe

    Returns:
        pd.DataFrame: Dataframe with added features
    """
    liq_cols = [col for col in df.columns if col.startswith('liq_')]

    if not liq_cols:
        logger.warning("Liquidation data not found")
        return df

    # Identify main liquidation columns
    long_liq_col = None
    short_liq_col = None
    total_liq_col = None

    for col in liq_cols:
        if 'long' in col.lower():
            long_liq_col = col
        elif 'short' in col.lower():
            short_liq_col = col
        elif 'total' in col.lower():
            total_liq_col = col

    # Calculate total liquidation from long/short if needed
    if total_liq_col is None and long_liq_col is not None and short_liq_col is not None:
        df['liq_total'] = df[long_liq_col] + df[short_liq_col]
        total_liq_col = 'liq_total'

    if total_liq_col is None:
        logger.warning("Main liquidation column not found")
        return df

    # Liquidation moving average
    for window in [6, 12, 24]:
        df[f'liq_ma_{window}'] = df[total_liq_col].rolling(window=window).mean()

    # Liquidation volatility
    for window in [12, 24, 48]:
        df[f'liq_volatility_{window}'] = df[total_liq_col].rolling(window=window).std()

    # Long/short liquidation ratio
    if long_liq_col is not None and short_liq_col is not None:
        df['liq_long_short_ratio'] = df[long_liq_col] / df[short_liq_col].replace(0, np.finfo(float).eps)

        # Liquidation skew (positive = mainly longs liquidated, negative = mainly shorts liquidated)
        df['liq_skew'] = (df[long_liq_col] - df[short_liq_col]) / (df[long_liq_col] + df[short_liq_col]).replace(0, np.finfo(float).eps)

    # Liquidation spike detection
    for window in [12, 24, 48]:
        # Moving average based anomaly
        liq_ma = df[total_liq_col].rolling(window=window).mean()
        liq_std = df[total_liq_col].rolling(window=window).std()

        df[f'liq_spike_{window}'] = (df[total_liq_col] > liq_ma + 2 * liq_std).astype(float)

    # Price relationship
    if 'price_close' in df.columns:
        # Liquidation by price change
        price_change = df['price_close'].pct_change()

        df['liq_on_up_move'] = df[total_liq_col] * (price_change > 0).astype(float)
        df['liq_on_down_move'] = df[total_liq_col] * (price_change < 0).astype(float)

    # Normalize all features with Z-score
    feature_cols = [col for col in df.columns if col.startswith('liq_') and col not in liq_cols]

    for col in feature_cols:
        df[f'{col}_zscore'] = zscore_normalize(df[col])

    logger.info(f"Liquidation feature generation complete: {len(feature_cols)} features")
    return df


def build_lsr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate long/short ratio related features

    Args:
        df: Input dataframe

    Returns:
        pd.DataFrame: Dataframe with added features
    """
    lsr_cols = [col for col in df.columns if col.startswith('lsr_')]

    if not lsr_cols:
        logger.warning("Long/short ratio data not found")
        return df

    # Identify main long/short ratio column
    main_lsr_col = None
    for col in ['lsr_ratio', 'lsr_value', 'lsr_longShortRatio']:
        if col in df.columns:
            main_lsr_col = col
            break

    if main_lsr_col is None:
        logger.warning("Main long/short ratio column not found")
        return df

    # LSR moving average
    for window in [6, 12, 24]:
        df[f'lsr_ma_{window}'] = df[main_lsr_col].rolling(window=window).mean()

    # LSR volatility
    for window in [12, 24, 48]:
        df[f'lsr_volatility_{window}'] = df[main_lsr_col].rolling(window=window).std()

    # LSR change rate
    for period in [1, 3, 6, 12]:
        df[f'lsr_change_{period}'] = df[main_lsr_col].pct_change(period)

    # LSR deviation from extremes
    for window in [24, 48, 96]:
        lsr_max = df[main_lsr_col].rolling(window=window).max()
        lsr_min = df[main_lsr_col].rolling(window=window).min()
        lsr_range = lsr_max - lsr_min

        # Relative position in range (0-1)
        df[f'lsr_relative_pos_{window}'] = (df[main_lsr_col] - lsr_min) / lsr_range.replace(0, np.finfo(float).eps)

    # Price relationship
    if 'price_close' in df.columns:
        # LSR and price change correlation
        for window in [12, 24, 48]:
            price_change = df['price_close'].pct_change()
            df[f'lsr_price_corr_{window}'] = df[main_lsr_col].rolling(window=window).corr(price_change)

    # Normalize all features with Z-score
    feature_cols = [col for col in df.columns if col.startswith('lsr_') and col not in lsr_cols]

    for col in feature_cols:
        df[f'{col}_zscore'] = zscore_normalize(df[col])

    logger.info(f"Long/short ratio feature generation complete: {len(feature_cols)} features")
    return df


def build_taker_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate taker volume related features

    Args:
        df: Input dataframe

    Returns:
        pd.DataFrame: Dataframe with added features
    """
    taker_cols = [col for col in df.columns if col.startswith('taker_')]

    if not taker_cols:
        logger.warning("Taker volume data not found")
        return df

    # Identify buy/sell taker volume columns
    buy_col = None
    sell_col = None

    for col in taker_cols:
        if 'buy' in col.lower():
            buy_col = col
        elif 'sell' in col.lower():
            sell_col = col

    if buy_col is None or sell_col is None:
        logger.warning("Buy/sell taker volume columns not found")
        return df

    # Total taker volume
    df['taker_total'] = df[buy_col] + df[sell_col]

    # Buy/sell ratio
    df['taker_buy_sell_ratio'] = df[buy_col] / df[sell_col].replace(0, np.finfo(float).eps)

    # Taker buy pressure (-1 to 1)
    df['taker_buy_pressure'] = (df[buy_col] - df[sell_col]) / (df[buy_col] + df[sell_col])

    # Taker volume moving average
    for window in [6, 12, 24]:
        df[f'taker_total_ma_{window}'] = df['taker_total'].rolling(window=window).mean()
        df[f'taker_buy_ma_{window}'] = df[buy_col].rolling(window=window).mean()
        df[f'taker_sell_ma_{window}'] = df[sell_col].rolling(window=window).mean()
        df[f'taker_pressure_ma_{window}'] = df['taker_buy_pressure'].rolling(window=window).mean()

    # Taker volume volatility
    for window in [12, 24, 48]:
        df[f'taker_volatility_{window}'] = df['taker_total'].rolling(window=window).std()

    # Taker volume slope
    for window in [6, 12, 24]:
        df[f'taker_slope_{window}'] = df['taker_total'].diff(window) / window

    # Price relationship
    if 'price_close' in df.columns:
        # Taker pressure and price change correlation
        for window in [12, 24, 48]:
            price_change = df['price_close'].pct_change()
            df[f'taker_price_corr_{window}'] = df['taker_buy_pressure'].rolling(window=window).corr(price_change)

    # Normalize all features with Z-score
    feature_cols = [col for col in df.columns if col.startswith('taker_') and col not in taker_cols]

    for col in feature_cols:
        df[f'{col}_zscore'] = zscore_normalize(df[col])

    logger.info(f"Taker volume feature generation complete: {len(feature_cols)} features")
    return df


def build_orderbook_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate orderbook related features

    Args:
        df: Input dataframe

    Returns:
        pd.DataFrame: Dataframe with added features
    """
    ob_cols = [col for col in df.columns if col.startswith('orderbook_')]

    if not ob_cols:
        logger.warning("Orderbook data not found")
        return df

    # Identify bid/ask order volume columns
    bid_col = None
    ask_col = None

    for col in ob_cols:
        if 'bid' in col.lower():
            bid_col = col
        elif 'ask' in col.lower():
            ask_col = col

    if bid_col is None or ask_col is None:
        logger.warning("Bid/ask order volume columns not found")
        return df

    # Bid/ask ratio
    df['orderbook_bid_ask_ratio'] = df[bid_col] / df[ask_col].replace(0, np.finfo(float).eps)

    # Orderbook skew (-1 to 1)
    df['orderbook_skew'] = (df[bid_col] - df[ask_col]) / (df[bid_col] + df[ask_col])

    # Orderbook moving average
    for window in [6, 12, 24]:
        df[f'orderbook_bid_ma_{window}'] = df[bid_col].rolling(window=window).mean()
        df[f'orderbook_ask_ma_{window}'] = df[ask_col].rolling(window=window).mean()
        df[f'orderbook_ratio_ma_{window}'] = df['orderbook_bid_ask_ratio'].rolling(window=window).mean()

    # Orderbook volatility
    for window in [12, 24, 48]:
        df[f'orderbook_volatility_{window}'] = df['orderbook_bid_ask_ratio'].rolling(window=window).std()

    # Orderbook pressure dynamics
    for window in [6, 12, 24]:
        df[f'orderbook_pressure_change_{window}'] = df['orderbook_skew'].diff(window)

    # Extreme imbalance detection
    for window in [24, 48]:
        skew_mean = df['orderbook_skew'].rolling(window=window).mean()
        skew_std = df['orderbook_skew'].rolling(window=window).std()
        
        df[f'orderbook_extreme_bullish_{window}'] = (df['orderbook_skew'] > skew_mean + 2 * skew_std).astype(float)
        df[f'orderbook_extreme_bearish_{window}'] = (df['orderbook_skew'] < skew_mean - 2 * skew_std).astype(float)

    # Price relationship
    if 'price_close' in df.columns:
        # Orderbook skew and price change correlation
        for window in [12, 24, 48]:
            price_change = df['price_close'].pct_change()
            df[f'orderbook_price_corr_{window}'] = df['orderbook_skew'].rolling(window=window).corr(price_change)

    # Normalize all features with Z-score
    feature_cols = [col for col in df.columns if col.startswith('orderbook_') and col not in ob_cols]

    for col in feature_cols:
        df[f'{col}_zscore'] = zscore_normalize(df[col])

    logger.info(f"Orderbook feature generation complete: {len(feature_cols)} features")
    return df


def build_premium_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate premium index related features

    Args:
        df: Input dataframe

    Returns:
        pd.DataFrame: Dataframe with added features
    """
    premium_cols = [col for col in df.columns if col.startswith('premium_')]

    if not premium_cols:
        logger.warning("Premium index data not found")
        return df

    # Identify main premium index column
    main_premium_col = None
    for col in ['premium_index', 'premium_value', 'premium_mark_index']:
        if col in df.columns:
            main_premium_col = col
            break

    if main_premium_col is None:
        logger.warning("Main premium index column not found")
        return df

    # Premium moving average
    for window in [6, 12, 24]:
        df[f'premium_ma_{window}'] = df[main_premium_col].rolling(window=window).mean()

    # Premium volatility
    for window in [12, 24, 48]:
        df[f'premium_volatility_{window}'] = df[main_premium_col].rolling(window=window).std()

    # Premium absolute value
    df['premium_abs'] = df[main_premium_col].abs()

    # Premium sign
    df['premium_sign'] = np.sign(df[main_premium_col])

    # Premium streak (how many periods with the same sign)
    df['premium_streak'] = df['premium_sign'].groupby(
        (df['premium_sign'] != df['premium_sign'].shift()).cumsum()
    ).cumcount() + 1

    # Premium deviation from extremes
    for window in [24, 48, 96]:
        premium_max = df[main_premium_col].rolling(window=window).max()
        premium_min = df[main_premium_col].rolling(window=window).min()
        premium_range = premium_max - premium_min

        # Relative position in range (0-1)
        df[f'premium_relative_pos_{window}'] = (df[main_premium_col] - premium_min) / premium_range.replace(0, np.finfo(float).eps)

    # Price relationship
    if 'price_close' in df.columns:
        # Premium and price change correlation
        for window in [12, 24, 48]:
            price_change = df['price_close'].pct_change()
            df[f'premium_price_corr_{window}'] = df[main_premium_col].rolling(window=window).corr(price_change)

    # Normalize all features with Z-score
    feature_cols = [col for col in df.columns if col.startswith('premium_') and col not in premium_cols]

    for col in feature_cols:
        df[f'{col}_zscore'] = zscore_normalize(df[col])

    logger.info(f"Premium index feature generation complete: {len(feature_cols)} features")
    return df


class FeatureBuilder:
    """Feature generation class"""

    def __init__(self, config_dir: Path = CONFIG_DIR):
        """Initialize

        Args:
            config_dir: Configuration directory path
        """
        self.config_dir = config_dir
        self.intervals = self._load_intervals()

    def _load_intervals(self) -> List[str]:
        """Load timeframe configuration

        Returns:
            List[str]: List of timeframes
        """
        try:
            with open(self.config_dir / 'intervals.yaml', 'r') as f:
                config = yaml.safe_load(f)
            return config['intervals']
        except Exception as e:
            logger.error(f"Failed to load interval configuration: {e}")
            raise

    def load_merged_data(self, interval: str) -> pd.DataFrame:
        """Load merged data

        Args:
            interval: Timeframe

        Returns:
            pd.DataFrame: Loaded dataframe
        """
        merged_path = DATA_DIR / 'features' / interval / 'merged.parquet'

        # Generate merged data if it doesn't exist
        if not os.path.exists(merged_path):
            logger.info(f"Merged data not found. Generating: {interval}")
            self.merge_interval(interval)

        return pd.read_parquet(merged_path)

    def merge_interval(self, interval: str) -> pd.DataFrame:
        """Merge data for timeframe

        Args:
            interval: Timeframe

        Returns:
            pd.DataFrame: Merged dataframe
        """
        logger.info(f"Starting data merge: {interval}")

        # Raw data directory
        raw_dir = DATA_DIR / 'raw' / interval

        # List raw data files
        endpoint_files = [f for f in os.listdir(raw_dir) if f.endswith('.parquet')]

        if not endpoint_files:
            raise FileNotFoundError(f"Raw data for timeframe {interval} not found")

        # Load data from each endpoint
        dfs = {}
        for filename in endpoint_files:
            endpoint = filename.split('.')[0]  # Filename without extension
            file_path = raw_dir / filename
            logger.info(f"Loading: {file_path}")

            try:
                dfs[endpoint] = pd.read_parquet(file_path)
            except Exception as e:
                logger.error(f"File loading error: {file_path}, {str(e)}")
                continue

        if not dfs:
            raise ValueError(f"No loadable data for timeframe {interval}")

        # Merge data
        merged = pd.concat(dfs.values(), axis=1)

        # Sort by index
        merged = merged.sort_index()

        # Shift 2h data to avoid forward dependency
        if interval == '2h':
            merged = merged.shift(1)  # Use only already completed 2h bars at 15m point
            logger.info("Shifted 2h data to avoid forward dependency")

        # Forward fill missing values
        merged = merged.ffill()

        # Remove remaining missing values
        merged = merged.dropna()

        # Save to features directory
        features_dir = DATA_DIR / 'features' / interval
        os.makedirs(features_dir, exist_ok=True)

        output_path = features_dir / 'merged.parquet'
        merged.to_parquet(output_path)

        logger.info(f"Merge complete: {output_path}, record count: {len(merged)}")

        return merged

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all features on the dataframe

        Args:
            df: Input dataframe

        Returns:
            pd.DataFrame: Dataframe with all features added
        """
        logger.info("Starting feature generation")
        
        # Apply all feature builders
        df = build_price_features(df)
        df = build_oi_features(df)
        df = build_funding_features(df)
        df = build_liq_features(df)
        df = build_lsr_features(df)
        df = build_taker_features(df)
        df = build_orderbook_features(df)
        df = build_premium_features(df)
        
        logger.info("Feature generation complete")
        return df
        
    def process_interval(self, interval: str) -> pd.DataFrame:
        """Process data for a specific timeframe
        
        Args:
            interval: Timeframe to process
            
        Returns:
            pd.DataFrame: Processed dataframe with all features
        """
        logger.info(f"Processing features for timeframe: {interval}")
        
        # Load merged data
        df = self.load_merged_data(interval)
        
        # Build features
        df = self.build_features(df)
        
        # Save processed features
        features_dir = DATA_DIR / 'features' / interval
        os.makedirs(features_dir, exist_ok=True)
        
        output_path = features_dir / 'features.parquet'
        df.to_parquet(output_path)
        
        logger.info(f"Feature processing complete: {output_path}, feature count: {len(df.columns)}")
        
        return df
        
    def process_all_intervals(self):
        """Process all timeframes"""
        logger.info("Processing features for all timeframes")
        
        for interval in self.intervals:
            self.process_interval(interval)
            
        logger.info("All timeframes processed")
