#!/usr/bin/env python
"""
run_real_backtest.py - Run 2h Coinglass data backtest

Features:
- Download 360 days of 2h data from Coinglass API (all data sources)
- Generate comprehensive features using Coinglass metrics
- Run 2h backtest with machine learning model
- Save results and performance metrics

This script is optimized for the 2h timeframe only, as specified in the project requirements.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(project_root))

# Import project modules
from src.data.coinglass_new import CoinglassClient
from src.backtest.portfolio import BacktestRunner
from src.config_loader import CG_API, SLACK_WEBHOOK

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(project_root, 'logs', 'real_backtest.log'), mode='a')
    ]
)
logger = logging.getLogger('real_backtest')

# Directory setup
DATA_DIR = project_root / 'data'
REPORTS_DIR = project_root / 'reports'

def calculate_zscore(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Calculate z-score normalization for numeric columns
    
    Args:
        df: Input dataframe
        window: Rolling window size for z-score calculation
        
    Returns:
        DataFrame with added z-score columns
    """
    logger.info(f"Calculating z-scores with window={window}")
    
    result_df = df.copy()
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Process each numeric column
    for col in numeric_cols:
        # Skip columns that are already zscore or don't need normalization
        if '_zscore' in col or col in ['timestamp']:
            continue
            
        try:
            # Rolling mean and std
            rolling_mean = df[col].rolling(window=window).mean()
            rolling_std = df[col].rolling(window=window).std()
            
            # Handle zero std (avoid division by zero)
            rolling_std = rolling_std.replace(0, np.nan)
            
            # Calculate z-score
            zscore = (df[col] - rolling_mean) / rolling_std
            
            # Add to result dataframe
            zscore_col = f"{col}_zscore"
            result_df[zscore_col] = zscore
            
            # Fill NaN values (first window-1 elements)
            result_df[zscore_col] = result_df[zscore_col].fillna(0)
            
            logger.debug(f"Calculated z-score for {col}")
        except Exception as e:
            logger.warning(f"Failed to calculate z-score for {col}: {e}")
    
    logger.info(f"Added {len(result_df.filter(like='_zscore').columns)} z-score columns")
    return result_df

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate returns and additional price-based features
    
    Args:
        df: Input dataframe with price data
        
    Returns:
        DataFrame with added return-based features
    """
    result_df = df.copy()
    
    # Ensure we have price close
    if 'close' in df.columns:
        logger.info("Calculating return-based features")
        
        # Rename close to price_close for compatibility
        result_df['price_close'] = df['close']
        
        # Calculate returns
        result_df['returns'] = df['close'].pct_change()
        result_df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Calculate volatility (rolling standard deviation of returns)
        result_df['volatility_14'] = result_df['returns'].rolling(14).std()
        result_df['volatility_30'] = result_df['returns'].rolling(30).std()
        
        # Calculate price momentum
        result_df['momentum_12'] = result_df['close'] / result_df['close'].shift(12) - 1
        result_df['momentum_24'] = result_df['close'] / result_df['close'].shift(24) - 1
        
        # Fill NaN values
        for col in ['returns', 'log_returns', 'volatility_14', 'volatility_30', 'momentum_12', 'momentum_24']:
            result_df[col] = result_df[col].fillna(0)
        
        logger.info("Added return-based features")
    else:
        logger.warning("Price 'close' column not found in dataframe, skipping return calculations")
    
    return result_df

def add_funding_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features based on funding rate
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with added funding-based features
    """
    result_df = df.copy()
    
    # Check if funding rate columns exist
    if 'funding_close' in df.columns:
        logger.info("Adding funding rate features")
        
        # Moving averages of funding rate
        result_df['funding_ma_8'] = df['funding_close'].rolling(8).mean()
        result_df['funding_ma_24'] = df['funding_close'].rolling(24).mean()
        
        # Funding rate divergence
        result_df['funding_divergence'] = df['funding_close'] - result_df['funding_ma_24']
        
        # Fill NaN values
        for col in ['funding_ma_8', 'funding_ma_24', 'funding_divergence']:
            result_df[col] = result_df[col].fillna(0)
            
        logger.info("Added funding rate features")
    else:
        logger.warning("Funding rate columns not found, skipping funding features")
    
    return result_df

def add_oi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features based on open interest
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with added OI-based features
    """
    result_df = df.copy()
    
    # Check if OI columns exist
    if 'oi_close' in df.columns:
        logger.info("Adding open interest features")
        
        # OI change
        result_df['oi_change'] = df['oi_close'].pct_change()
        
        # OI momentum
        result_df['oi_momentum_12'] = df['oi_close'] / df['oi_close'].shift(12) - 1
        result_df['oi_momentum_24'] = df['oi_close'] / df['oi_close'].shift(24) - 1
        
        # OI and price divergence (if price_close exists)
        if 'price_close' in result_df.columns:
            # Normalize both series
            price_norm = (result_df['price_close'] / result_df['price_close'].iloc[0])
            oi_norm = (result_df['oi_close'] / result_df['oi_close'].iloc[0])
            
            # Calculate divergence
            result_df['oi_price_divergence'] = oi_norm - price_norm
        
        # Fill NaN values
        for col in ['oi_change', 'oi_momentum_12', 'oi_momentum_24']:
            result_df[col] = result_df[col].fillna(0)
            
        if 'oi_price_divergence' in result_df.columns:
            result_df['oi_price_divergence'] = result_df['oi_price_divergence'].fillna(0)
            
        logger.info("Added open interest features")
    else:
        logger.warning("Open interest columns not found, skipping OI features")
    
    return result_df

def add_liquidation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features based on liquidation data
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with added liquidation-based features
    """
    result_df = df.copy()
    
    # Check if liquidation columns exist
    if 'long_liquidation' in df.columns and 'short_liquidation' in df.columns:
        logger.info("Adding liquidation features")
        
        # Rolling sum of liquidations
        result_df['liq_long_12h'] = df['long_liquidation'].rolling(12).sum()
        result_df['liq_short_12h'] = df['short_liquidation'].rolling(12).sum()
        result_df['liq_total_12h'] = df['long_liquidation'].rolling(12).sum() + df['short_liquidation'].rolling(12).sum()
        
        # Liquidation imbalance
        result_df['liq_imbalance'] = np.where(
            result_df['liq_total_12h'] > 0,
            (result_df['liq_long_12h'] - result_df['liq_short_12h']) / result_df['liq_total_12h'],
            0
        )
        
        # Fill NaN values
        for col in ['liq_long_12h', 'liq_short_12h', 'liq_total_12h', 'liq_imbalance']:
            result_df[col] = result_df[col].fillna(0)
            
        logger.info("Added liquidation features")
    else:
        logger.warning("Liquidation columns not found, skipping liquidation features")
    
    return result_df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare all features for the model
    
    Args:
        df: Input raw dataframe
        
    Returns:
        DataFrame with all features
    """
    logger.info("Preparing all features")
    
    # Apply each feature engineering function
    df = calculate_returns(df)
    df = add_funding_features(df)
    df = add_oi_features(df)
    df = add_liquidation_features(df)
    
    # Finally, calculate z-scores for all features
    df = calculate_zscore(df)
    
    # Create a feature dataframe with only z-score columns
    features_df = df.filter(like='_zscore')
    
    logger.info(f"Feature preparation complete, created {len(features_df.columns)} features")
    
    # Save the features separately
    return df

def download_and_process_data(interval: str, limit: int = 4320) -> pd.DataFrame:
    """
    Download and process data for a specific interval
    
    Args:
        interval: Timeframe (e.g., '15m', '2h')
        limit: Number of data points to fetch
        
    Returns:
        Processed DataFrame with all features
    """
    logger.info(f"Downloading and processing data for interval: {interval}")
    
    # Initialize Coinglass client
    client = CoinglassClient(api_key=CG_API, slack_webhook=SLACK_WEBHOOK)
    
    # Download data
    logger.info(f"Downloading data from Coinglass API")
    merged_df = client.download_and_save_data(interval, limit)
    
    # Prepare features
    logger.info(f"Preparing features")
    processed_df = prepare_features(merged_df)
    
    # Save processed data
    features_path = DATA_DIR / 'features' / interval / 'X.parquet'
    os.makedirs(features_path.parent, exist_ok=True)
    
    # Extract only z-score features for X.parquet
    features_df = processed_df.filter(like='_zscore')
    features_df.to_parquet(features_path)
    logger.info(f"Saved features to {features_path}")
    
    # Save the full processed dataframe for reference
    processed_path = DATA_DIR / 'features' / interval / 'processed.parquet'
    processed_df.to_parquet(processed_path)
    logger.info(f"Saved processed data to {processed_path}")
    
    return processed_df

def run_backtest(interval: str):
    """
    Run backtest for a specific interval
    
    Args:
        interval: Timeframe (e.g., '15m', '2h')
    """
    logger.info(f"Running backtest for interval: {interval}")
    
    # Create BacktestRunner instance
    runner = BacktestRunner()
    
    try:
        # Run the backtest
        logger.info(f"Starting backtest execution")
        portfolio, stats = runner.run_interval_backtest(interval)
        
        # Log key metrics
        logger.info(f"Backtest results for {interval}:")
        logger.info(f"Total Return: {stats['Total Return [%]']}%")
        logger.info(f"Annual Return: {stats['Annual Return [%]']}%")
        logger.info(f"Sharpe Ratio: {stats['Sharpe Ratio']}")
        logger.info(f"Max Drawdown: {stats['Max Drawdown [%]']}%")
        
        logger.info(f"Backtest completed successfully")
        return portfolio, stats
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run backtest with real data')
    parser.add_argument('--interval', '-i', type=str, help='Timeframe (e.g., 15m, 2h)')
    parser.add_argument('--limit', '-l', type=int, default=4320, help='Number of data points to fetch')
    parser.add_argument('--skip-download', '-s', action='store_true', help='Skip data download and use existing data')
    parser.add_argument('--all-intervals', '-a', action='store_true', help='Run for all configured intervals')
    
    args = parser.parse_args()
    
    try:
        # Create necessary directories
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        if args.all_intervals:
            # Load intervals from config
            with open(project_root / 'config' / 'intervals.yaml', 'r') as f:
                import yaml
                intervals = yaml.safe_load(f)['intervals']
                
            logger.info(f"Running for all intervals: {intervals}")
            
            for interval in intervals:
                try:
                    logger.info(f"Processing interval: {interval}")
                    
                    # Download data if not skipped
                    if not args.skip_download:
                        download_and_process_data(interval, args.limit)
                    
                    # Run backtest
                    run_backtest(interval)
                    
                except Exception as e:
                    logger.error(f"Error processing interval {interval}: {str(e)}")
            
            logger.info("All intervals completed")
            
        elif args.interval:
            # Download data if not skipped
            if not args.skip_download:
                download_and_process_data(args.interval, args.limit)
            
            # Run backtest
            run_backtest(args.interval)
            
        else:
            logger.error("No interval specified. Use --interval or --all-intervals")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
