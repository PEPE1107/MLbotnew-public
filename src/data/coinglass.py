#!/usr/bin/env python
"""
coinglass.py - Coinglass API client module

Features:
- Fetch BTC data from Coinglass API
- Support multiple endpoints (price, open interest, funding, liquidation, etc.)
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
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, List, Optional, Union, Any
import slack_sdk

# Project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIG_DIR = ROOT_DIR / 'config'
DATA_DIR = ROOT_DIR / 'data'

# Ensure log directory exists
os.makedirs(ROOT_DIR / 'logs', exist_ok=True)

# Logger setup
logger = logging.getLogger('data.coinglass')

# Coinglass API endpoint definitions
ENDPOINTS = {
    'price': 'price/ohlc-history',                                # Price data
    'oi': 'futures/openInterest/ohlc-aggregated-history',         # Open Interest
    'funding': 'futures/fundingRate/oi-weight-ohlc-history',      # Funding rates
    'liq': 'futures/liquidation/v3/aggregated-history',           # Liquidation data
    'lsr': 'futures/topLongShortAccountRatio/history',            # Long/Short ratio
    'taker': 'spot/takerBuySellVolume/history',                   # Taker volume
    'orderbook': 'spot/orderbook/aggregated-history',             # Orderbook
    'premium': 'coinbase-premium-index'                           # Premium index
}


class CoinglassClient:
    """Client for fetching data from Coinglass API"""
    
    def __init__(self, api_key: str, slack_webhook: Optional[str] = None, config_dir: Path = CONFIG_DIR):
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
            with open(self.config_dir / 'intervals.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            # Look for backtest_intervals key, fallback to ['2h']
            return config.get('backtest_intervals', ['2h'])
        except Exception as e:
            logger.error(f"Failed to load interval configuration: {e}")
            # Return default value in case of error
            return ['2h']
    
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
            # Raw data directory
            raw_dir = DATA_DIR / 'raw' / interval
            os.makedirs(raw_dir, exist_ok=True)
            
            # Feature data directory
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
    def fetch(self, endpoint: str, params: Dict[str, Any], key: str = None) -> pd.DataFrame:
        """Fetch data from Coinglass API

        Args:
            endpoint: Endpoint name
            params: Request parameters
            key: Endpoint key (uses endpoint name if not provided)

        Returns:
            pd.DataFrame: Fetched dataframe
        """
        if key is None:
            key = endpoint
            
        url = f"https://open-api-v3.coinglass.com/api/{ENDPOINTS[key]}"
        headers = {
            "accept": "application/json",
            "CG-API-KEY": self.api_key
        }
        
        try:
            logger.info(f"API request: {key}, parameters: {params}")
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code != 200:
                error_msg = f"API error: {key}, status code: {response.status_code}, response: {response.text}"
                logger.error(error_msg)
                self.send_slack_message(f"API error: {key}, status code: {response.status_code}", emoji=":x:")
                response.raise_for_status()
                
            data = response.json()
            
            # Process according to data structure
            if 'data' not in data or not data['data']:
                error_msg = f"Empty response: {key}, response: {data}"
                logger.error(error_msg)
                self.send_slack_message(f"Empty response: {key}", emoji=":warning:")
                raise ValueError(error_msg)
                
            # Extract data by endpoint
            df = self._parse_response(data, key)
            
            # Convert timestamps to datetime and set index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp').sort_index()
            elif 'ts' in df.columns:
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                df = df.set_index('ts').sort_index()
                
            return df
        
        except Exception as e:
            error_msg = f"Data fetch error: {key}, error: {str(e)}"
            logger.error(error_msg)
            self.send_slack_message(f"Data fetch error: {key}, {str(e)}", emoji=":x:")
            raise
    
    def _parse_response(self, response: Dict, endpoint: str) -> pd.DataFrame:
        """Parse response into DataFrame

        Args:
            response: API response
            endpoint: Endpoint name

        Returns:
            pd.DataFrame: Converted dataframe
        """
        data = response.get('data', {})
        
        if endpoint == 'price':
            # Process price data
            return pd.DataFrame(data)
        
        elif endpoint == 'oi':
            # Process open interest data
            return pd.DataFrame(data)
        
        elif endpoint == 'funding':
            # Process funding rate data
            filtered_data = []
            for item in data:
                if item.get('symbol') == 'BTC':
                    filtered_data.append(item)
            return pd.DataFrame(filtered_data)
        
        elif endpoint == 'taker':
            # Takerデータの処理 - 文字列を数値に変換
            processed_data = []
            for item in data:
                # 文字列から数値への変換を確実に行う
                processed_item = {}
                for key, value in item.items():
                    if key in ['buy', 'sell'] and isinstance(value, str):
                        try:
                            processed_item[key] = float(value)
                        except (ValueError, TypeError):
                            processed_item[key] = 0.0  # 変換できない場合は0をセット
                    else:
                        processed_item[key] = value
                processed_data.append(processed_item)
            return pd.DataFrame(processed_data)
        
        elif endpoint in ['liq', 'lsr', 'orderbook', 'premium']:
            # Process other endpoints (generic)
            return pd.DataFrame(data)
        
        # Default processing
        return pd.DataFrame(data)
    
    def fetch_price(self, symbol: str, interval: str, limit: int = 4320,
                   start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch price data

        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            interval: Timeframe (e.g., '15m', '2h')
            limit: Number of bars to fetch
            start_time: Start time (uses limit if not provided)
            end_time: End time (uses current time if not provided)

        Returns:
            pd.DataFrame: Price dataframe
        """
        if end_time is None:
            end_time = datetime.now()
            
        if start_time is None:
            # Calculate period from interval
            if interval.endswith('m'):
                minutes = int(interval[:-1])
                start_time = end_time - timedelta(minutes=minutes * limit)
            elif interval.endswith('h'):
                hours = int(interval[:-1])
                start_time = end_time - timedelta(hours=hours * limit)
            elif interval.endswith('d'):
                days = int(interval[:-1])
                start_time = end_time - timedelta(days=days * limit)
            else:
                raise ValueError(f"Unsupported interval: {interval}")
        
        # Convert to Unix timestamp (milliseconds)
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        # Set parameters
        params = {
            'symbol': symbol,
            'interval': interval,
            'from': start_ts,
            'to': end_ts,
            'limit': limit
        }
        
        return self.fetch('price', params)
    
    def fetch_open_interest(self, symbol: str, interval: str, limit: int = 4320,
                           start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch open interest data

        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            interval: Timeframe (e.g., '15m', '2h')
            limit: Number of bars to fetch
            start_time: Start time (uses limit if not provided)
            end_time: End time (uses current time if not provided)

        Returns:
            pd.DataFrame: Open interest dataframe
        """
        if end_time is None:
            end_time = datetime.now()
            
        if start_time is None:
            # Calculate period from interval
            if interval.endswith('m'):
                minutes = int(interval[:-1])
                start_time = end_time - timedelta(minutes=minutes * limit)
            elif interval.endswith('h'):
                hours = int(interval[:-1])
                start_time = end_time - timedelta(hours=hours * limit)
            elif interval.endswith('d'):
                days = int(interval[:-1])
                start_time = end_time - timedelta(days=days * limit)
            else:
                raise ValueError(f"Unsupported interval: {interval}")
        
        # Convert to Unix timestamp (milliseconds)
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        # Set parameters
        params = {
            'symbol': symbol,
            'interval': interval,
            'from': start_ts,
            'to': end_ts,
            'limit': limit
        }
        
        return self.fetch('oi', params)
    
    def fetch_funding_rates(self, symbol: str, interval: str, limit: int = 4320,
                           start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch funding rate data

        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            interval: Timeframe (e.g., '15m', '2h')
            limit: Number of bars to fetch
            start_time: Start time (uses limit if not provided)
            end_time: End time (uses current time if not provided)

        Returns:
            pd.DataFrame: Funding rate dataframe
        """
        if end_time is None:
            end_time = datetime.now()
            
        if start_time is None:
            # Calculate period from interval
            if interval.endswith('m'):
                minutes = int(interval[:-1])
                start_time = end_time - timedelta(minutes=minutes * limit)
            elif interval.endswith('h'):
                hours = int(interval[:-1])
                start_time = end_time - timedelta(hours=hours * limit)
            elif interval.endswith('d'):
                days = int(interval[:-1])
                start_time = end_time - timedelta(days=days * limit)
            else:
                raise ValueError(f"Unsupported interval: {interval}")
        
        # Convert to Unix timestamp (milliseconds)
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        # Set parameters
        params = {
            'symbol': symbol,
            'interval': interval,
            'from': start_ts,
            'to': end_ts,
            'limit': limit
        }
        
        return self.fetch('funding', params)
    
    def download_data(self, interval: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, limit: int = 4320):
        """Download data for specified timeframe

        Args:
            interval: Timeframe
            start_time: Start time (calculates from limit if not provided)
            end_time: End time (uses current time if not provided)
            limit: Number of bars to fetch
        """
        if end_time is None:
            end_time = datetime.now()
            
        if start_time is None:
            # Calculate period from interval
            if interval.endswith('m'):
                minutes = int(interval[:-1])
                start_time = end_time - timedelta(minutes=minutes * limit)
            elif interval.endswith('h'):
                hours = int(interval[:-1])
                start_time = end_time - timedelta(hours=hours * limit)
            elif interval.endswith('d'):
                days = int(interval[:-1])
                start_time = end_time - timedelta(days=days * limit)
            else:
                raise ValueError(f"Unsupported interval: {interval}")
        
        logger.info(f"Starting data download: {interval}, from {start_time} to {end_time}")
        self.send_slack_message(f"Starting data download: {interval}", emoji=":rocket:")
        
        # 時間足の特殊処理 (h2, d1形式への変換)
        h_format = interval.replace('2h', 'h2').replace('4h', 'h4').replace('1d', 'd1')
        
        # エンドポイント別パラメータ設定
        endpoint_params = {
            'price': {
                'exchange': 'Binance',
                'symbol': 'BTCUSDT',
                'type': 'futures',
                'interval': interval,
                'limit': limit
            },
            'oi': {
                'symbol': 'BTC',
                'interval': interval,
                'limit': limit
            },
            'funding': {
                'symbol': 'BTC',
                'interval': interval,
                'limit': limit
            },
            'liq': {
                'exchanges': 'ALL',
                'symbol': 'BTC',
                'interval': interval,
                'limit': limit
            },
            'lsr': {
                'exchange': 'Binance',
                'symbol': 'BTCUSDT',
                'interval': h_format,
                'limit': limit
            },
            'taker': {
                'exchange': 'Binance',
                'symbol': 'BTCUSDT',
                'interval': h_format,
                'limit': limit
            },
            'orderbook': {
                'exchanges': 'Binance',
                'symbol': 'BTC',
                'interval': interval.replace('2h', 'h1'),
                'limit': limit
            },
            'premium': {
                'interval': interval,
                'limit': limit
            }
        }
        
        # Fetch from all endpoints
        for endpoint in ENDPOINTS:
            try:
                logger.info(f"Processing endpoint: {endpoint}, timeframe: {interval}")
                
                # エンドポイント別のパラメータを取得
                params = endpoint_params.get(endpoint, {})
                
                df = self.fetch(endpoint, params)
                
                # Add endpoint prefix to column names
                df = df.add_prefix(f"{endpoint}_")
                
                # Save to file
                output_path = DATA_DIR / 'raw' / interval / f"{endpoint}.parquet"
                df.to_parquet(output_path)
                
                logger.info(f"Save complete: {output_path}, record count: {len(df)}")
                
            except Exception as e:
                error_msg = f"Endpoint fetch error: {endpoint}, timeframe: {interval}, error: {str(e)}"
                logger.error(error_msg)
                self.send_slack_message(error_msg, emoji=":x:")
        
        success_msg = f"Data download complete: {interval}, period: {start_time} to {end_time}"
        logger.info(success_msg)
        self.send_slack_message(success_msg, emoji=":white_check_mark:")
    
    def download_all_intervals(self):
        """Download data for all timeframes"""
        for interval in self.intervals:
            self.download_data(interval)
