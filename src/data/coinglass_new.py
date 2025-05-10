#!/usr/bin/env python
"""
coinglass_new.py - Coinglass API client module (Updated for new API format)

Features:
- Fetch BTC data from Coinglass API using the latest endpoint formats
- Support multiple data types (price, funding, open interest, liquidation, etc.)
- Support multiple timeframes (configured in intervals.yaml)
- Automatic retry with tenacity
- Save data in parquet format
"""

import os
import sys
import time
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, List, Optional, Union, Any, Tuple
import slack_sdk

# Project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIG_DIR = ROOT_DIR / 'config'
DATA_DIR = ROOT_DIR / 'data'

# Ensure log directory exists
os.makedirs(ROOT_DIR / 'logs', exist_ok=True)

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(ROOT_DIR, 'logs', 'coinglass.log'), mode='a')
    ]
)
logger = logging.getLogger('data.coinglass')

# API key - can be configured via env or config file
DEFAULT_API_KEY = "5a2ac6bc211648ee96f894307c4dc6af"  # Default key from examples
API_KEY = os.environ.get("COINGLASS_API_KEY", DEFAULT_API_KEY)

class CoinglassClient:
    """Client for fetching data from Coinglass API"""
    
    def __init__(self, api_key: str = API_KEY, slack_webhook: Optional[str] = None, config_dir: Path = CONFIG_DIR):
        """Initialize the client

        Args:
            api_key: Coinglass API key
            slack_webhook: Optional Slack webhook URL for notifications
            config_dir: Configuration directory path
        """
        self.config_dir = config_dir
        self.api_key = api_key
        self.intervals = self._load_intervals()
        self.slack_webhook = slack_webhook
        self.slack_client = self._setup_slack_client() if self.slack_webhook else None
        
        # Setup data directories
        self._setup_data_directories()
        
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
    
    def _setup_slack_client(self) -> Optional[slack_sdk.WebhookClient]:
        """Set up Slack client

        Returns:
            Optional[slack_sdk.WebhookClient]: Slack webhook client
        """
        if self.slack_webhook:
            return slack_sdk.WebhookClient(self.slack_webhook)
        return None
    
    def _setup_data_directories(self):
        """Set up data directories"""
        for interval in self.intervals:
            # Raw data directory for each interval
            raw_dir = DATA_DIR / 'raw' / interval
            os.makedirs(raw_dir, exist_ok=True)
            
            # Feature data directory for each interval
            features_dir = DATA_DIR / 'features' / interval
            os.makedirs(features_dir, exist_ok=True)
    
    def send_slack_message(self, message: str, emoji: str = ":chart_with_upwards_trend:"):
        """Send message to Slack

        Args:
            message: Message to send
            emoji: Icon to attach to the message
        """
        if self.slack_client:
            try:
                response = self.slack_client.send(
                    text=f"{emoji} {message}"
                )
                if not response.status_code == 200:
                    logger.warning(f"Failed to send Slack notification: {response.status_code}, {response.body}")
            except Exception as e:
                logger.error(f"Error sending Slack notification: {e}")
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError))
    )
    def fetch_request(self, url: str, headers: Dict[str, str], params: Optional[Dict[str, Any]] = None) -> Dict:
        """Make a request to the Coinglass API

        Args:
            url: API endpoint URL
            headers: Request headers including API key
            params: Optional query parameters

        Returns:
            Dict: JSON response 
        """
        try:
            logger.info(f"API request: {url}, parameters: {params}")
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code != 200:
                error_msg = f"API error: {url}, status code: {response.status_code}, response: {response.text}"
                logger.error(error_msg)
                self.send_slack_message(f"API error: {url}, status code: {response.status_code}", emoji=":x:")
                response.raise_for_status()
                
            data = response.json()
            
            # Basic validation of response
            if data.get('code') != "0" or 'data' not in data or not data['data']:
                error_msg = f"Empty or error response: {url}, response: {data}"
                logger.error(error_msg)
                self.send_slack_message(f"Empty or error response: {url}", emoji=":warning:")
                raise ValueError(error_msg)
                
            return data
        
        except Exception as e:
            error_msg = f"API request error: {url}, error: {str(e)}"
            logger.error(error_msg)
            self.send_slack_message(f"API request error: {url}, {str(e)}", emoji=":x:")
            raise
    
    def fetch_price_ohlc(self, interval: str, limit: int = 4320) -> pd.DataFrame:
        """Fetch price OHLC data

        Args:
            interval: Timeframe (e.g., '15m', '2h')
            limit: Number of bars to fetch (max 4320)

        Returns:
            pd.DataFrame: Processed OHLC dataframe
        """
        url = "https://open-api-v3.coinglass.com/api/price/ohlc-history"
        headers = {"CG-API-KEY": self.api_key}
        params = {
            "exchange": "Binance",
            "symbol": "BTCUSDT",
            "type": "futures",
            "interval": interval,
            "limit": limit
        }
        
        try:
            data = self.fetch_request(url, headers, params)
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            
            # Process columns:
            # Expected format: [timestamp, open, high, low, close, volume]
            if len(df.columns) >= 6:  # Should have at least 6 columns
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                
                # Ensure numeric columns
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                # Set timestamp as index
                df = df.set_index('timestamp').sort_index()
                
                return df
            else:
                logger.error(f"Unexpected data format for price OHLC: {df.columns}")
                raise ValueError(f"Unexpected data format for price OHLC")
                
        except Exception as e:
            logger.error(f"Error fetching price OHLC: {str(e)}")
            raise
    
    def fetch_funding_rate(self, interval: str, limit: int = 4320) -> pd.DataFrame:
        """Fetch funding rate history

        Args:
            interval: Timeframe (e.g., '15m', '2h')
            limit: Number of data points to fetch (max 4320)

        Returns:
            pd.DataFrame: Processed funding rate dataframe
        """
        url = "https://open-api-v3.coinglass.com/api/futures/fundingRate/oi-weight-ohlc-history"
        headers = {
            "accept": "application/json",
            "CG-API-KEY": self.api_key
        }
        params = {
            "symbol": "BTC",
            "interval": interval,
            "limit": limit
        }
        
        try:
            data = self.fetch_request(url, headers, params)
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            
            # Rename and process columns
            df = df.rename(columns={
                't': 'timestamp',
                'o': 'funding_open',
                'h': 'funding_high',
                'l': 'funding_low',
                'c': 'funding_close'
            })
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Ensure numeric columns
            for col in ['funding_open', 'funding_high', 'funding_low', 'funding_close']:
                df[col] = pd.to_numeric(df[col])
            
            # Set timestamp as index
            df = df.set_index('timestamp').sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching funding rate: {str(e)}")
            raise
    
    def fetch_open_interest(self, interval: str, limit: int = 4320) -> pd.DataFrame:
        """Fetch open interest history

        Args:
            interval: Timeframe (e.g., '15m', '2h')
            limit: Number of data points to fetch (max 4320)

        Returns:
            pd.DataFrame: Processed open interest dataframe
        """
        url = "https://open-api-v3.coinglass.com/api/futures/openInterest/ohlc-aggregated-history"
        headers = {
            "accept": "application/json",
            "CG-API-KEY": self.api_key
        }
        params = {
            "symbol": "BTC",
            "interval": interval,
            "limit": limit
        }
        
        try:
            data = self.fetch_request(url, headers, params)
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            
            # Rename columns
            df = df.rename(columns={
                't': 'timestamp',
                'o': 'oi_open',
                'h': 'oi_high',
                'l': 'oi_low',
                'c': 'oi_close'
            })
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Ensure numeric columns
            for col in ['oi_open', 'oi_high', 'oi_low', 'oi_close']:
                df[col] = pd.to_numeric(df[col])
            
            # Set timestamp as index
            df = df.set_index('timestamp').sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching open interest: {str(e)}")
            raise
    
    def fetch_liquidation(self, interval: str, limit: int = 4320) -> pd.DataFrame:
        """Fetch liquidation data

        Args:
            interval: Timeframe (e.g., '15m', '2h')
            limit: Number of data points to fetch (max 4320)

        Returns:
            pd.DataFrame: Processed liquidation dataframe
        """
        url = "https://open-api-v3.coinglass.com/api/futures/liquidation/v3/aggregated-history"
        headers = {
            "accept": "application/json",
            "CG-API-KEY": self.api_key
        }
        params = {
            "exchanges": "ALL",
            "symbol": "BTC",
            "interval": interval,
            "limit": limit
        }
        
        try:
            data = self.fetch_request(url, headers, params)
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            
            # Rename columns
            df = df.rename(columns={
                't': 'timestamp',
                'longLiquidationUsd': 'long_liquidation',
                'shortLiquidationUsd': 'short_liquidation'
            })
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Ensure numeric columns
            for col in ['long_liquidation', 'short_liquidation']:
                df[col] = pd.to_numeric(df[col])
            
            # Add total liquidation
            df['total_liquidation'] = df['long_liquidation'] + df['short_liquidation']
            
            # Add long/short liquidation ratio
            df['long_short_liq_ratio'] = np.where(
                df['short_liquidation'] > 0,
                df['long_liquidation'] / df['short_liquidation'],
                0
            )
            
            # Set timestamp as index
            df = df.set_index('timestamp').sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching liquidation data: {str(e)}")
            raise
    
    def fetch_long_short_ratio(self, interval: str, limit: int = 4320) -> pd.DataFrame:
        """Fetch long/short ratio

        Args:
            interval: Timeframe (e.g., '15m', '2h') - note: API uses h2 format for 2h
            limit: Number of data points to fetch (max 4320)

        Returns:
            pd.DataFrame: Processed long/short ratio dataframe
        """
        # Convert standard interval format to the API format
        api_interval = interval
        if interval == '2h':
            api_interval = 'h2'
        elif interval == '4h':
            api_interval = 'h4'
        
        url = "https://open-api-v3.coinglass.com/api/futures/topLongShortAccountRatio/history"
        headers = {
            "accept": "application/json",
            "CG-API-KEY": self.api_key
        }
        params = {
            "exchange": "Binance",
            "symbol": "BTCUSDT",
            "interval": api_interval,
            "limit": limit
        }
        
        try:
            data = self.fetch_request(url, headers, params)
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            
            # Rename columns
            df = df.rename(columns={
                'time': 'timestamp',
                'longAccount': 'long_account',
                'shortAccount': 'short_account',
                'longShortRatio': 'long_short_ratio'
            })
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Ensure numeric columns
            for col in ['long_account', 'short_account', 'long_short_ratio']:
                df[col] = pd.to_numeric(df[col])
            
            # Set timestamp as index
            df = df.set_index('timestamp').sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching long/short ratio: {str(e)}")
            raise
    
    def fetch_taker_buy_sell(self, interval: str, limit: int = 4320) -> pd.DataFrame:
        """Fetch taker buy/sell volume

        Args:
            interval: Timeframe (e.g., '15m', '2h') - note: API uses h2 format for 2h
            limit: Number of data points to fetch (max 4320)

        Returns:
            pd.DataFrame: Processed taker buy/sell volume dataframe
        """
        # Convert standard interval format to the API format
        api_interval = interval
        if interval == '2h':
            api_interval = 'h2'
        elif interval == '4h':
            api_interval = 'h4'
        
        url = "https://open-api-v3.coinglass.com/api/spot/takerBuySellVolume/history"
        headers = {
            "accept": "application/json",
            "CG-API-KEY": self.api_key
        }
        params = {
            "exchange": "Binance",
            "symbol": "BTCUSDT",
            "interval": api_interval,
            "limit": limit
        }
        
        try:
            data = self.fetch_request(url, headers, params)
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            
            # Rename columns
            df = df.rename(columns={
                'time': 'timestamp',
                'buy': 'taker_buy_volume',
                'sell': 'taker_sell_volume'
            })
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Ensure numeric columns are numeric
            for col in ['taker_buy_volume', 'taker_sell_volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Calculate buy/sell ratio
            df['taker_buy_sell_ratio'] = np.where(
                df['taker_sell_volume'] > 0,
                df['taker_buy_volume'] / df['taker_sell_volume'],
                0
            )
            
            # Set timestamp as index
            df = df.set_index('timestamp').sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching taker buy/sell volume: {str(e)}")
            raise
    
    def fetch_orderbook(self, interval: str, limit: int = 4320) -> pd.DataFrame:
        """Fetch aggregated orderbook data

        Args:
            interval: Timeframe (e.g., '15m', '2h')
            limit: Number of data points to fetch

        Returns:
            pd.DataFrame: Processed orderbook dataframe
        """
        # For 2h interval, API requires h1 format
        api_interval = interval
        if interval == '2h':
            api_interval = 'h1'
        
        url = "https://open-api-v3.coinglass.com/api/spot/orderbook/aggregated-history"
        headers = {
            "accept": "application/json",
            "CG-API-KEY": self.api_key
        }
        params = {
            "exchanges": "Binance",
            "symbol": "BTC",
            "interval": api_interval,
            "limit": limit
        }
        
        try:
            data = self.fetch_request(url, headers, params)
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            
            # Rename columns
            df = df.rename(columns={
                'time': 'timestamp',
                'bidsUsd': 'bids_usd',
                'bidsAmount': 'bids_amount',
                'asksUsd': 'asks_usd',
                'asksAmount': 'asks_amount'
            })
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Ensure numeric columns
            for col in ['bids_usd', 'bids_amount', 'asks_usd', 'asks_amount']:
                df[col] = pd.to_numeric(df[col])
            
            # Calculate bid/ask ratio
            df['bid_ask_ratio'] = np.where(
                df['asks_usd'] > 0,
                df['bids_usd'] / df['asks_usd'],
                0
            )
            
            # Set timestamp as index
            df = df.set_index('timestamp').sort_index()
            
            # If 2h was requested but API needs h1, resample to 2h
            if interval == '2h' and api_interval == 'h1':
                df = df.resample('2H').agg({
                    'bids_usd': 'mean',
                    'bids_amount': 'mean',
                    'asks_usd': 'mean',
                    'asks_amount': 'mean',
                    'bid_ask_ratio': 'mean'
                })
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching orderbook data: {str(e)}")
            raise
    
    def fetch_premium_index(self, interval: str, limit: int = 4320) -> pd.DataFrame:
        """Fetch Coinbase premium index

        Args:
            interval: Timeframe (e.g., '15m', '2h')
            limit: Number of data points to fetch

        Returns:
            pd.DataFrame: Processed premium index dataframe
        """
        url = "https://open-api-v3.coinglass.com/api/coinbase-premium-index"
        headers = {
            "accept": "application/json",
            "CG-API-KEY": self.api_key
        }
        params = {
            "interval": interval,
            "limit": limit
        }
        
        try:
            data = self.fetch_request(url, headers, params)
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            
            # Rename columns
            df = df.rename(columns={
                'time': 'timestamp',
                'premium': 'premium_usd',
                'premiumRate': 'premium_rate'
            })
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Ensure numeric columns
            for col in ['premium_usd', 'premium_rate']:
                df[col] = pd.to_numeric(df[col])
            
            # Set timestamp as index
            df = df.set_index('timestamp').sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching premium index: {str(e)}")
            raise
    
    def fetch_all_data(self, interval: str, limit: int = 4320) -> Dict[str, pd.DataFrame]:
        """Fetch all data types for a given interval

        Args:
            interval: Timeframe (e.g., '15m', '2h')
            limit: Number of data points to fetch

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with all dataframes
        """
        logger.info(f"Fetching all data for interval: {interval}, limit: {limit}")
        
        results = {}
        
        # Define all data types to fetch
        data_types = [
            ('price', self.fetch_price_ohlc),
            ('funding', self.fetch_funding_rate),
            ('oi', self.fetch_open_interest),
            ('liq', self.fetch_liquidation),
            ('lsr', self.fetch_long_short_ratio),
            ('taker', self.fetch_taker_buy_sell),
            ('orderbook', self.fetch_orderbook),
            ('premium', self.fetch_premium_index)
        ]
        
        # Fetch each data type
        for name, fetch_func in data_types:
            try:
                logger.info(f"Fetching {name} data for {interval}")
                df = fetch_func(interval, limit)
                results[name] = df
                logger.info(f"Successfully fetched {name} data: {len(df)} records")
            except Exception as e:
                logger.error(f"Failed to fetch {name} data: {str(e)}")
                # Continue with other data types even if one fails
                self.send_slack_message(f"Failed to fetch {name} data for {interval}: {str(e)}", emoji=":warning:")
        
        return results
    
    def merge_all_data(self, dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all dataframes into a single dataframe

        Args:
            dfs: Dictionary of dataframes

        Returns:
            pd.DataFrame: Merged dataframe
        """
        if not dfs:
            raise ValueError("No dataframes to merge")
        
        # Start with the price dataframe as the base
        if 'price' in dfs:
            base_df = dfs['price'].copy()
            logger.info(f"Using price dataframe as base: {len(base_df)} records")
        else:
            # Use the first available dataframe as base
            key = list(dfs.keys())[0]
            base_df = dfs[key].copy()
            logger.info(f"Price dataframe not found, using {key} as base: {len(base_df)} records")
        
        # Add a prefix to all columns except timestamp
        base_columns = base_df.columns.tolist()
        
        # Prepare the merged dataframe
        merged_df = base_df.copy()
        
        # Merge each dataframe
        for name, df in dfs.items():
            if name == 'price':
                continue  # Already used as base
                
            try:
                # Add prefix to column names
                df_prefixed = df.add_prefix(f"{name}_")
                
                # Merge on index (timestamp)
                merged_df = merged_df.join(df_prefixed, how='outer')
                logger.info(f"Merged {name} dataframe: {len(df)} records")
            except Exception as e:
                logger.error(f"Error merging {name} dataframe: {str(e)}")
        
        # Forward fill NaN values to handle different timestamps
        merged_df = merged_df.fillna(method='ffill')
        
        # Backward fill any remaining NaNs at the beginning
        merged_df = merged_df.fillna(method='bfill')
        
        logger.info(f"Final merged dataframe: {len(merged_df)} records, {len(merged_df.columns)} columns")
        
        return merged_df
    
    def download_and_save_data(self, interval: str, limit: int = 4320) -> pd.DataFrame:
        """Download all data for an interval and save to parquet files

        Args:
            interval: Timeframe (e.g., '15m', '2h')
            limit: Number of data points to fetch

        Returns:
            pd.DataFrame: Merged dataframe
        """
        # Fetch all data
        all_data = self.fetch_all_data(interval, limit)
        
        # Save each individual dataframe
        for name, df in all_data.items():
            try:
                output_path = DATA_DIR / 'raw' / interval / f"{name}.parquet"
                os.makedirs(output_path.parent, exist_ok=True)
                df.to_parquet(output_path)
                logger.info(f"Saved {name} data to {output_path}")
            except Exception as e:
                logger.error(f"Error saving {name} data: {str(e)}")
        
        # Merge all data
        merged_df = self.merge_all_data(all_data)
        
        # Save merged data
        try:
            output_path = DATA_DIR / 'features' / interval / 'merged.parquet'
            os.makedirs(output_path.parent, exist_ok=True)
            merged_df.to_parquet(output_path)
            logger.info(f"Saved merged data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving merged data: {str(e)}")
        
        return merged_df
    
    def download_all_intervals(self, limit: int = 4320):
        """Download data for all configured intervals

        Args:
            limit: Number of data points to fetch per interval
        """
        logger.info(f"Starting download for all intervals: {self.intervals}")
        self.send_slack_message(f"Starting download for all intervals: {self.intervals}", emoji=":rocket:")
        
        for interval in self.intervals:
            try:
                logger.info(f"Processing interval: {interval}")
                self.download_and_save_data(interval, limit)
                logger.info(f"Completed interval: {interval}")
            except Exception as e:
                logger.error(f"Error processing interval {interval}: {str(e)}")
                self.send_slack_message(f"Error processing interval {interval}: {str(e)}", emoji=":x:")
        
        logger.info("All intervals completed")
        self.send_slack_message("All intervals completed", emoji=":white_check_mark:")

# Execute if run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download data from Coinglass API')
    parser.add_argument('--interval', '-i', type=str, help='Specific interval to download')
    parser.add_argument('--limit', '-l', type=int, default=4320, help='Number of data points to fetch')
    parser.add_argument('--api-key', '-k', type=str, help='Coinglass API key')
    
    args = parser.parse_args()
    
    # Use provided API key or default
    api_key = args.api_key if args.api_key else API_KEY
    
    client = CoinglassClient(api_key=api_key)
    
    if args.interval:
        # Download specific interval
        client.download_and_save_data(args.interval, args.limit)
    else:
        # Download all intervals
        client.download_all_intervals(args.limit)
