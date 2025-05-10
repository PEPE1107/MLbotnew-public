#!/usr/bin/env python
"""
coinglass_api.py - Coinglass API wrapper for MLbotnew

このモジュールは、Coinglass APIからデータを取得するためのクラスを提供します。
fetch_all.pyスクリプトとの互換性を確保するためのラッパーです。
"""

import os
import sys
import time
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Logger setup
logger = logging.getLogger('data.coinglass_api')

class CoinglassAPI:
    """
    CoinglassデータをAPIから取得するためのクライアントクラス
    """
    
    def __init__(self, api_key: str):
        """
        初期化

        Args:
            api_key: Coinglass API キー
        """
        self.api_key = api_key
        self.base_url = "https://open-api-v3.coinglass.com/api"
        self.headers = {
            "accept": "application/json",
            "CG-API-KEY": self.api_key
        }
        
    def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict:
        """
        APIリクエストを実行する

        Args:
            endpoint: APIエンドポイント
            params: リクエストパラメータ

        Returns:
            Dict: レスポンスデータ
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            logger.info(f"API request: {endpoint}, parameters: {params}")
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                error_msg = f"API error: {endpoint}, status code: {response.status_code}, response: {response.text}"
                logger.error(error_msg)
                response.raise_for_status()
                
            data = response.json()
            
            if 'data' not in data or not data['data']:
                error_msg = f"Empty response: {endpoint}, response: {data}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            return data
            
        except Exception as e:
            error_msg = f"Data fetch error: {endpoint}, error: {str(e)}"
            logger.error(error_msg)
            raise
    
    def _convert_time_to_timestamp(self, time_obj: datetime) -> int:
        """
        日時オブジェクトをUnixタイムスタンプに変換する

        Args:
            time_obj: 日時オブジェクト

        Returns:
            int: Unixタイムスタンプ（秒）
        """
        return int(time_obj.timestamp())
    
    def get_price_data(self, exchange: str, symbol: str, interval: str, 
                      start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, 
                      limit: int = 4320) -> pd.DataFrame:
        """
        価格データを取得する

        Args:
            exchange: 取引所名（例: 'Binance'）
            symbol: 通貨ペア（例: 'BTCUSDT'）
            interval: 時間間隔（例: '2h'）
            start_date: 開始日時
            end_date: 終了日時
            limit: データ取得上限

        Returns:
            pd.DataFrame: 価格データ
        """
        endpoint = "price/ohlc-history"
        
        params = {
            "exchange": exchange,
            "symbol": symbol,
            "type": "futures",
            "interval": interval,
            "limit": limit
        }
        
        if start_date:
            params["from"] = self._convert_time_to_timestamp(start_date) * 1000
        
        if end_date:
            params["to"] = self._convert_time_to_timestamp(end_date) * 1000
            
        try:
            response = self._request(endpoint, params)
            
            # レスポンスデータをDataFrameに変換
            price_data = pd.DataFrame(response['data'])
            
            if not price_data.empty:
                # カラム名の設定
                price_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                
                # タイムスタンプをdatetimeに変換
                price_data['timestamp'] = pd.to_datetime(price_data['timestamp'], unit='s')
                price_data = price_data.set_index('timestamp')
                
            return price_data
            
        except Exception as e:
            logger.error(f"Failed to fetch price data: {e}")
            return pd.DataFrame()
    
    def get_funding_rates(self, symbol: str, interval: str,
                         start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                         limit: int = 4320) -> pd.DataFrame:
        """
        ファンディングレートデータを取得する

        Args:
            symbol: 暗号資産シンボル（例: 'BTC'）
            interval: 時間間隔（例: '2h'）
            start_date: 開始日時
            end_date: 終了日時
            limit: データ取得上限

        Returns:
            pd.DataFrame: ファンディングレートデータ
        """
        endpoint = "futures/fundingRate/oi-weight-ohlc-history"
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_date:
            params["from"] = self._convert_time_to_timestamp(start_date) * 1000
        
        if end_date:
            params["to"] = self._convert_time_to_timestamp(end_date) * 1000
        
        try:
            response = self._request(endpoint, params)
            
            # レスポンスデータをDataFrameに変換
            funding_data = pd.DataFrame(response['data'])
            
            if not funding_data.empty:
                # カラム名のリネーム
                funding_data = funding_data.rename(columns={
                    't': 'timestamp',
                    'o': 'funding_rate_open',
                    'h': 'funding_rate_high',
                    'l': 'funding_rate_low',
                    'c': 'funding_rate_close'
                })
                
                # タイムスタンプをdatetimeに変換
                funding_data['timestamp'] = pd.to_datetime(funding_data['timestamp'], unit='s')
                funding_data = funding_data.set_index('timestamp')
                
            return funding_data
            
        except Exception as e:
            logger.error(f"Failed to fetch funding rate data: {e}")
            return pd.DataFrame()
    
    def get_open_interest(self, symbol: str, interval: str,
                         start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                         limit: int = 4320) -> pd.DataFrame:
        """
        オープンインタレストデータを取得する

        Args:
            symbol: 暗号資産シンボル（例: 'BTC'）
            interval: 時間間隔（例: '2h'）
            start_date: 開始日時
            end_date: 終了日時
            limit: データ取得上限

        Returns:
            pd.DataFrame: オープンインタレストデータ
        """
        endpoint = "futures/openInterest/ohlc-aggregated-history"
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_date:
            params["from"] = self._convert_time_to_timestamp(start_date) * 1000
        
        if end_date:
            params["to"] = self._convert_time_to_timestamp(end_date) * 1000
        
        try:
            response = self._request(endpoint, params)
            
            # レスポンスデータをDataFrameに変換
            oi_data = pd.DataFrame(response['data'])
            
            if not oi_data.empty:
                # カラム名のリネーム
                oi_data = oi_data.rename(columns={
                    't': 'timestamp',
                    'o': 'oi_open',
                    'h': 'oi_high',
                    'l': 'oi_low',
                    'c': 'oi_close'
                })
                
                # タイムスタンプをdatetimeに変換
                oi_data['timestamp'] = pd.to_datetime(oi_data['timestamp'], unit='s')
                oi_data = oi_data.set_index('timestamp')
                
            return oi_data
            
        except Exception as e:
            logger.error(f"Failed to fetch open interest data: {e}")
            return pd.DataFrame()
    
    def get_liquidations(self, symbol: str, interval: str,
                       start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                       limit: int = 4320) -> pd.DataFrame:
        """
        清算データを取得する

        Args:
            symbol: 暗号資産シンボル（例: 'BTC'）
            interval: 時間間隔（例: '2h'）
            start_date: 開始日時
            end_date: 終了日時
            limit: データ取得上限

        Returns:
            pd.DataFrame: 清算データ
        """
        endpoint = "futures/liquidation/v3/aggregated-history"
        
        params = {
            "exchanges": "ALL",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_date:
            params["from"] = self._convert_time_to_timestamp(start_date) * 1000
        
        if end_date:
            params["to"] = self._convert_time_to_timestamp(end_date) * 1000
        
        try:
            response = self._request(endpoint, params)
            
            # レスポンスデータをDataFrameに変換
            liq_data = pd.DataFrame(response['data'])
            
            if not liq_data.empty:
                # タイムスタンプカラム名のリネーム
                liq_data = liq_data.rename(columns={'t': 'timestamp'})
                
                # タイムスタンプをdatetimeに変換
                liq_data['timestamp'] = pd.to_datetime(liq_data['timestamp'], unit='s')
                liq_data = liq_data.set_index('timestamp')
                
            return liq_data
            
        except Exception as e:
            logger.error(f"Failed to fetch liquidation data: {e}")
            return pd.DataFrame()
    
    def get_long_short_ratio(self, symbol: str, interval: str,
                           start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                           limit: int = 4320) -> pd.DataFrame:
        """
        ロングショート比率データを取得する

        Args:
            symbol: 暗号資産シンボル（例: 'BTC'）
            interval: 時間間隔（例: '2h'）
            start_date: 開始日時
            end_date: 終了日時
            limit: データ取得上限

        Returns:
            pd.DataFrame: ロングショート比率データ
        """
        endpoint = "futures/topLongShortAccountRatio/history"
        
        # インターバル形式変換 (2h → h2)
        if interval.endswith('h'):
            hours = interval[:-1]
            interval_param = f"h{hours}"
        elif interval.endswith('m'):
            minutes = interval[:-1]
            interval_param = f"m{minutes}"
        elif interval.endswith('d'):
            days = interval[:-1]
            interval_param = f"d{days}"
        else:
            interval_param = interval
        
        params = {
            "exchange": "Binance",
            "symbol": "BTCUSDT",
            "interval": interval_param,
            "limit": limit
        }
        
        if start_date:
            params["from"] = self._convert_time_to_timestamp(start_date) * 1000
        
        if end_date:
            params["to"] = self._convert_time_to_timestamp(end_date) * 1000
        
        try:
            response = self._request(endpoint, params)
            
            # レスポンスデータをDataFrameに変換
            lsr_data = pd.DataFrame(response['data'])
            
            if not lsr_data.empty:
                # タイムスタンプカラム名のリネーム
                lsr_data = lsr_data.rename(columns={'time': 'timestamp'})
                
                # タイムスタンプをdatetimeに変換
                lsr_data['timestamp'] = pd.to_datetime(lsr_data['timestamp'], unit='s')
                lsr_data = lsr_data.set_index('timestamp')
                
            return lsr_data
            
        except Exception as e:
            logger.error(f"Failed to fetch long-short ratio data: {e}")
            return pd.DataFrame()

    def get_taker_buy_sell(self, symbol: str, interval: str,
                          start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                          limit: int = 4320) -> pd.DataFrame:
        """
        Taker Buy/Sellデータを取得する

        Args:
            symbol: 暗号資産シンボル（例: 'BTCUSDT'）
            interval: 時間間隔（例: '2h'）
            start_date: 開始日時
            end_date: 終了日時
            limit: データ取得上限

        Returns:
            pd.DataFrame: Taker Buy/Sellデータ
        """
        endpoint = "spot/takerBuySellVolume/history"
        
        # インターバル形式変換 (2h → h2)
        if interval.endswith('h'):
            hours = interval[:-1]
            interval_param = f"h{hours}"
        elif interval.endswith('m'):
            minutes = interval[:-1]
            interval_param = f"m{minutes}"
        elif interval.endswith('d'):
            days = interval[:-1]
            interval_param = f"d{days}"
        else:
            interval_param = interval
        
        params = {
            "exchange": "Binance",
            "symbol": symbol,
            "interval": interval_param,
            "limit": limit
        }
        
        if start_date:
            params["from"] = self._convert_time_to_timestamp(start_date) * 1000
        
        if end_date:
            params["to"] = self._convert_time_to_timestamp(end_date) * 1000
        
        try:
            response = self._request(endpoint, params)
            
            # レスポンスデータをDataFrameに変換
            taker_data = pd.DataFrame(response['data'])
            
            if not taker_data.empty:
                # タイムスタンプカラム名のリネーム
                taker_data = taker_data.rename(columns={'time': 'timestamp'})
                
                # 数値型への変換
                for col in ['buy', 'sell']:
                    if col in taker_data.columns:
                        taker_data[col] = pd.to_numeric(taker_data[col], errors='coerce')
                
                # タイムスタンプをdatetimeに変換
                taker_data['timestamp'] = pd.to_datetime(taker_data['timestamp'], unit='s')
                taker_data = taker_data.set_index('timestamp')
                
            return taker_data
            
        except Exception as e:
            logger.error(f"Failed to fetch taker buy/sell data: {e}")
            return pd.DataFrame()

    def get_orderbook(self, symbol: str, interval: str,
                    start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                    limit: int = 4320) -> pd.DataFrame:
        """
        注文板データを取得する

        Args:
            symbol: 暗号資産シンボル（例: 'BTC'）
            interval: 時間間隔（例: '2h'）
            start_date: 開始日時
            end_date: 終了日時
            limit: データ取得上限

        Returns:
            pd.DataFrame: 注文板データ
        """
        endpoint = "spot/orderbook/aggregated-history"
        
        # 1hへの変換（APIの制約）
        interval_param = "h1"
        
        params = {
            "exchanges": "Binance",
            "symbol": symbol,
            "interval": interval_param,
            "limit": limit
        }
        
        if start_date:
            params["from"] = self._convert_time_to_timestamp(start_date) * 1000
        
        if end_date:
            params["to"] = self._convert_time_to_timestamp(end_date) * 1000
        
        try:
            response = self._request(endpoint, params)
            
            # レスポンスデータをDataFrameに変換
            orderbook_data = pd.DataFrame(response['data'])
            
            if not orderbook_data.empty:
                # タイムスタンプカラム名のリネーム
                orderbook_data = orderbook_data.rename(columns={'time': 'timestamp'})
                
                # タイムスタンプをdatetimeに変換
                orderbook_data['timestamp'] = pd.to_datetime(orderbook_data['timestamp'], unit='s')
                orderbook_data = orderbook_data.set_index('timestamp')
                
            return orderbook_data
            
        except Exception as e:
            logger.error(f"Failed to fetch orderbook data: {e}")
            return pd.DataFrame()

    def get_premium_index(self, interval: str,
                         start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                         limit: int = 4320) -> pd.DataFrame:
        """
        Coinbase Premium Indexデータを取得する

        Args:
            interval: 時間間隔（例: '2h'）
            start_date: 開始日時
            end_date: 終了日時
            limit: データ取得上限

        Returns:
            pd.DataFrame: Premium Indexデータ
        """
        endpoint = "coinbase-premium-index"
        
        params = {
            "interval": interval,
            "limit": limit
        }
        
        if start_date:
            params["from"] = self._convert_time_to_timestamp(start_date) * 1000
        
        if end_date:
            params["to"] = self._convert_time_to_timestamp(end_date) * 1000
        
        try:
            response = self._request(endpoint, params)
            
            # レスポンスデータをDataFrameに変換
            premium_data = pd.DataFrame(response['data'])
            
            if not premium_data.empty:
                # タイムスタンプカラム名のリネーム
                premium_data = premium_data.rename(columns={'time': 'timestamp'})
                
                # タイムスタンプをdatetimeに変換
                premium_data['timestamp'] = pd.to_datetime(premium_data['timestamp'], unit='s')
                premium_data = premium_data.set_index('timestamp')
                
            return premium_data
            
        except Exception as e:
            logger.error(f"Failed to fetch premium index data: {e}")
            return pd.DataFrame()
