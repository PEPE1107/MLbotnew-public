#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
データ同期モジュール
--------------

外部APIから最新のBTC価格データ、OIデータなどを取得し、
ローカルのデータディレクトリに保存します。
"""

import os
import sys
import time
import json
import yaml
import logging
import datetime
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.data.coinglass import CoinglassClient, ENDPOINTS as COINGLASS_ENDPOINTS

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataSync:
    """
    マルチタイムフレーム戦略用のデータ同期クラス
    外部APIからBTC価格データ、OI、ファンディングレートなどを取得
    """
    
    def __init__(self, api_key_file: str = None, use_cache: bool = True, use_coinglass: bool = True):
        """
        初期化
        
        Parameters:
        -----------
        api_key_file : str
            API鍵を含むYAMLファイルのパス
        use_cache : bool
            キャッシュを使用するかどうか
        use_coinglass : bool
            Coinglassデータを使用するかどうか
        """
        self.data_dir = os.path.join(project_root, 'data')
        self.use_cache = use_cache
        self.use_coinglass = use_coinglass
        
        # データディレクトリが存在しない場合は作成
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # API鍵を読み込む（オプション）
        self.api_keys = {}
        if api_key_file:
            try:
                with open(api_key_file, 'r', encoding='utf-8') as f:
                    self.api_keys = yaml.safe_load(f)
                logger.info("API鍵を読み込みました")
            except Exception as e:
                logger.warning(f"API鍵の読み込みに失敗しました: {e}")
        
        # Coinglassクライアントの初期化
        self.coinglass_client = None
        if use_coinglass and 'coinglass' in self.api_keys:
            self.coinglass_client = CoinglassClient(self.api_keys['coinglass'])
            logger.info("Coinglassクライアントを初期化しました")
        elif use_coinglass:
            logger.warning("CoinglassのAPIキーが設定されていません。API_KEYSファイルを確認してください。")
        
        # 時間枠設定を読み込む
        self.intervals_config = self._load_config('intervals')
        
    def _load_config(self, config_type: str) -> Dict[str, Any]:
        """
        設定ファイルを読み込む
        
        Parameters:
        -----------
        config_type : str
            設定ファイルのタイプ ('intervals', 'fees', 'model', 'mtf')
            
        Returns:
        --------
        dict
            設定情報
        """
        config_path = os.path.join(project_root, 'config', f'{config_type}.yaml')
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"設定ファイルが見つかりません: {config_path}")
            # デフォルト設定を返す
            if config_type == 'intervals':
                return {'day': '1d', 'hour': '2h', 'minute': '15m'}
            return {}
    
    def get_cache_path(self, symbol: str, interval: str) -> str:
        """
        キャッシュファイルのパスを取得
        
        Parameters:
        -----------
        symbol : str
            取得対象のシンボル（例: 'BTC-USD'）
        interval : str
            時間枠（例: '1d', '2h', '15m'）
            
        Returns:
        --------
        str
            キャッシュファイルのパス
        """
        # シンボル名から不正な文字を削除（ファイル名に使用できない文字を除去）
        safe_symbol = ''.join(c for c in symbol if c.isalnum() or c in ['-', '_'])
        
        # キャッシュディレクトリ
        cache_dir = os.path.join(self.data_dir, 'ohlcv')
        os.makedirs(cache_dir, exist_ok=True)
        
        return os.path.join(cache_dir, f"{safe_symbol}_{interval}.csv")
    
    def fetch_price_data(self, symbol: str = 'BTC-USD', interval: str = '1d', 
                       limit: int = 1000) -> pd.DataFrame:
        """
        価格データを取得
        
        Parameters:
        -----------
        symbol : str
            取得対象のシンボル（例: 'BTC-USD'）
        interval : str
            時間枠（例: '1d', '2h', '15m'）
        limit : int
            取得する最大行数
            
        Returns:
        --------
        pd.DataFrame
            OHLCV価格データ
        """
        cache_path = self.get_cache_path(symbol, interval)
        
        # キャッシュが有効かつファイルが存在する場合は読み込み
        if self.use_cache and os.path.exists(cache_path):
            try:
                cache_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                
                # キャッシュが新しい場合（24時間以内）は使用
                cache_mtime = os.path.getmtime(cache_path)
                if time.time() - cache_mtime < 86400:  # 24時間 = 86400秒
                    logger.info(f"キャッシュからデータを読み込みました: {cache_path}")
                    return cache_df
                
                # キャッシュが古い場合は差分更新
                logger.info(f"キャッシュは古いため更新します: {cache_path}")
                new_data = self._fetch_data_from_api(symbol, interval, limit)
                
                if new_data is not None and not new_data.empty:
                    # 新しいデータと古いデータをマージ
                    merged_df = pd.concat([cache_df, new_data])
                    merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
                    merged_df = merged_df.sort_index()
                    
                    # キャッシュを更新
                    merged_df.to_csv(cache_path)
                    logger.info(f"キャッシュを更新しました: {cache_path}")
                    return merged_df
            
            except Exception as e:
                logger.warning(f"キャッシュの読み込みに失敗しました: {e}")
        
        # キャッシュがない場合はAPIから取得
        data = self._fetch_data_from_api(symbol, interval, limit)
        
        if data is not None and not data.empty:
            # キャッシュを保存
            data.to_csv(cache_path)
            logger.info(f"新しいデータをキャッシュに保存しました: {cache_path}")
        
        return data
    
    def _fetch_data_from_api(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """
        APIから価格データを取得
        
        Parameters:
        -----------
        symbol : str
            取得対象のシンボル（例: 'BTC-USD'）
        interval : str
            時間枠（例: '1d', '2h', '15m'）
        limit : int
            取得する最大行数
            
        Returns:
        --------
        pd.DataFrame
            OHLCV価格データ
        """
        # 複数のAPIから取得を試みる（バックアップ）
        try:
            return self._fetch_from_binance(symbol, interval, limit)
        except Exception as e:
            logger.warning(f"Binance APIからの取得に失敗しました: {e}")
            
            try:
                return self._fetch_from_coinapi(symbol, interval, limit)
            except Exception as e:
                logger.warning(f"CoinAPI からの取得に失敗しました: {e}")
                
                try:
                    return self._fetch_from_yahoo_finance(symbol, interval, limit)
                except Exception as e:
                    logger.error(f"全APIからの取得に失敗しました: {e}")
                    raise ValueError(f"実データの取得に失敗しました。API設定を確認してください: {e}")
    
    def _fetch_from_binance(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """
        Binance APIから価格データを取得
        
        Parameters:
        -----------
        symbol : str
            取得対象のシンボル（例: 'BTCUSDT'）
        interval : str
            時間枠（例: '1d', '2h', '15m'）
        limit : int
            取得する最大行数
            
        Returns:
        --------
        pd.DataFrame
            OHLCV価格データ
        """
        # Binance APIに合わせてシンボルを変換
        binance_symbol = symbol.replace('-', '').upper()
        if not binance_symbol.endswith('USDT'):
            binance_symbol = binance_symbol.replace('USD', 'USDT')
        
        # Binance APIの時間枠フォーマットに合わせて変換
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
        }
        
        binance_interval = interval_map.get(interval, '1d')
        
        # APIエンドポイント
        url = f"https://api.binance.com/api/v3/klines"
        
        # リクエストパラメータ
        params = {
            'symbol': binance_symbol,
            'interval': binance_interval,
            'limit': limit
        }
        
        # APIリクエスト
        response = requests.get(url, params=params)
        
        # レスポンスチェック
        if response.status_code != 200:
            raise Exception(f"Binance API error: {response.status_code} - {response.text}")
        
        # データの解析
        data = response.json()
        
        # DataFrameに変換
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # データ型を変換
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # タイムスタンプをインデックスに設定（ミリ秒→秒）
        df.index = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop(['timestamp', 'close_time', 'ignore'], axis=1)
        
        logger.info(f"Binance APIからデータを取得しました: {len(df)}行")
        return df
    
    def _fetch_from_coinapi(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """
        CoinAPI.io から価格データを取得
        
        Parameters:
        -----------
        symbol : str
            取得対象のシンボル（例: 'BTC-USD'）
        interval : str
            時間枠（例: '1d', '2h', '15m'）
        limit : int
            取得する最大行数
            
        Returns:
        --------
        pd.DataFrame
            OHLCV価格データ
        """
        # APIキーがない場合は例外発生
        if 'coinapi' not in self.api_keys:
            raise Exception("CoinAPI APIキーが設定されていません")
        
        # CoinAPI.io の時間枠フォーマットに変換
        interval_map = {
            '1m': '1MIN', '5m': '5MIN', '15m': '15MIN', '30m': '30MIN',
            '1h': '1HRS', '2h': '2HRS', '4h': '4HRS', '6h': '6HRS', '8h': '8HRS', '12h': '12HRS',
            '1d': '1DAY', '3d': '3DAY', '1w': '1WEK', '1M': '1MTH'
        }
        
        coinapi_interval = interval_map.get(interval, '1DAY')
        
        # APIエンドポイント
        url = f"https://rest.coinapi.io/v1/ohlcv/{symbol}/latest"
        
        # リクエストヘッダー（APIキー）
        headers = {
            'X-CoinAPI-Key': self.api_keys['coinapi']
        }
        
        # リクエストパラメータ
        params = {
            'period_id': coinapi_interval,
            'limit': limit
        }
        
        # APIリクエスト
        response = requests.get(url, headers=headers, params=params)
        
        # レスポンスチェック
        if response.status_code != 200:
            raise Exception(f"CoinAPI error: {response.status_code} - {response.text}")
        
        # データの解析
        data = response.json()
        
        # DataFrameに変換
        df = pd.DataFrame(data)
        
        # カラム名を標準化
        df = df.rename(columns={
            'price_open': 'open',
            'price_high': 'high',
            'price_low': 'low',
            'price_close': 'close',
            'volume_traded': 'volume',
            'time_period_start': 'timestamp'
        })
        
        # タイムスタンプをインデックスに設定
        df.index = pd.to_datetime(df['timestamp'])
        df = df.drop(['timestamp', 'time_period_end', 'time_open', 'time_close'], axis=1, errors='ignore')
        
        logger.info(f"CoinAPI からデータを取得しました: {len(df)}行")
        return df
    
    def _fetch_from_yahoo_finance(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """
        Yahoo Finance から価格データを取得
        
        Parameters:
        -----------
        symbol : str
            取得対象のシンボル（例: 'BTC-USD'）
        interval : str
            時間枠（例: '1d', '2h', '15m'）
        limit : int
            取得する最大行数
            
        Returns:
        --------
        pd.DataFrame
            OHLCV価格データ
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinanceがインストールされていません。pip install yfinance を実行してください。")
        
        # Yahoo Financeの時間枠フォーマットに変換
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1d', '5d': '5d', '1wk': '1wk', '1mo': '1mo', '3mo': '3mo'
        }
        
        yahoo_interval = interval_map.get(interval, '1d')
        
        # 期間を計算
        end_date = datetime.datetime.now()
        
        if interval in ['1m', '5m', '15m', '30m']:
            # データ期間の制限があるため、最近のデータのみ取得
            period = "7d"
        elif interval in ['1h', '2h', '4h']:
            period = "60d"
        else:
            period = "max"
        
        # Yahoo Finance からデータを取得
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=yahoo_interval)
        
        # 列名を標準化
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # 大文字列名を小文字に
        df.columns = df.columns.str.lower()
        
        # 不要な列を削除
        df = df.drop(['dividends', 'stock splits'], axis=1, errors='ignore')
        
        # 最新のデータに制限
        if len(df) > limit:
            df = df.iloc[-limit:]
        
        logger.info(f"Yahoo Finance からデータを取得しました: {len(df)}行")
        return df
    
    def _generate_sample_data(self, interval: str, limit: int) -> pd.DataFrame:
        """
        サンプルデータを生成（API取得に失敗した場合のフォールバック）
        
        Parameters:
        -----------
        interval : str
            時間枠（例: '1d', '2h', '15m'）
        limit : int
            生成する行数
            
        Returns:
        --------
        pd.DataFrame
            サンプルOHLCVデータ
        """
        logger.warning(f"実データ取得に失敗したためサンプルデータを生成します: {interval}")
        
        # 現在時刻からインデックスを生成
        end_date = datetime.datetime.now()
        
        # 時間枠に応じた頻度を設定
        if interval == '15m':
            freq = '15min'
        elif interval == '2h':
            freq = '2h'
        else:  # 1d
            freq = 'D'
        
        # インデックスを作成（現在時刻から過去に向かって）
        periods = min(limit, 1000)  # 最大1000行に制限
        index = pd.date_range(end=end_date, periods=periods, freq=freq)[::-1]
        
        # ランダムウォークでBTC価格を生成
        np.random.seed(42)  # 再現性のため
        returns = np.random.normal(0.0001, 0.01, size=len(index))
        price = 100 * (1 + returns).cumprod()
        
        # トレンド要素を追加
        trend = np.sin(np.linspace(0, 4 * np.pi, len(index))) * 10
        price = price + trend
        
        # データフレーム作成
        df = pd.DataFrame(index=index)
        df['close'] = price
        df['open'] = price * (1 + np.random.normal(0, 0.003, size=len(index)))
        df['high'] = np.maximum(df['open'], df['close']) * (1 + abs(np.random.normal(0, 0.005, size=len(index))))
        df['low'] = np.minimum(df['open'], df['close']) * (1 - abs(np.random.normal(0, 0.005, size=len(index))))
        df['volume'] = abs(np.random.normal(1000000, 500000, size=len(index))) * (1 + returns * 10) ** 2
        
        logger.info(f"サンプルデータを生成しました: {len(df)}行")
        return df
    
    def _convert_interval_format(self, interval: str, endpoint_type: str = 'default') -> str:
        """
        時間枠フォーマットを変換（Coinglass API用）
        
        Parameters:
        -----------
        interval : str
            元の時間枠形式 (例: '15m', '2h', '1d')
        endpoint_type : str
            エンドポイントのタイプ ('lsr'など特殊フォーマットが必要なエンドポイント用)
            
        Returns:
        --------
        str
            変換後の時間枠形式
        """
        # LSRエンドポイント用の特殊フォーマット (h2, m15など)
        if endpoint_type == 'lsr':
            if interval == '15m':
                return 'm15'
            elif interval == '2h':
                return 'h2'
            elif interval == '1d':
                return 'd1'
            # そのまま返す
            return interval
            
        # 標準フォーマット (2h, 15mなど)
        return interval
    
    def fetch_funding_rate(self, symbol: str = 'BTC-USD', interval: str = '8h', 
                          limit: int = 90) -> pd.DataFrame:
        """
        ファンディングレートデータを取得
        
        Parameters:
        -----------
        symbol : str
            取得対象のシンボル（例: 'BTC-USD'）
        interval : str
            時間枠（例: '8h'）
        limit : int
            取得する最大行数
            
        Returns:
        --------
        pd.DataFrame
            ファンディングレートデータ
        """
        # Coinglassクライアントが利用可能ならそれを使用
        if self.use_coinglass and self.coinglass_client:
            # シンボル名をBTCの形式に変換
            coinglass_symbol = symbol.split('-')[0]
            
            try:
                # Coinglassからファンディングレートを取得
                funding_params = {
                    'symbol': coinglass_symbol,
                    'interval': self._convert_interval_format(interval),
                    'limit': limit
                }
                
                funding_df = self.coinglass_client.fetch('funding', funding_params)
                
                # カラム名を標準化
                if 'funding_o' in funding_df.columns:
                    funding_df = funding_df.rename(columns={
                        'funding_o': 'funding_rate_open',
                        'funding_h': 'funding_rate_high',
                        'funding_l': 'funding_rate_low',
                        'funding_c': 'funding_rate'
                    })
                
                logger.info(f"Coinglassからファンディングレートデータを取得しました: {len(funding_df)}行")
                return funding_df
                
            except Exception as e:
                logger.warning(f"Coinglassからのファンディングレート取得に失敗しました: {e}")
        
        # Coinglassが使えない場合はシミュレーションデータを生成
        price_data = self.fetch_price_data(symbol, interval, limit)
        
        # 価格データに基づいてファンディングレートをシミュレート
        df = pd.DataFrame(index=price_data.index)
        df['funding_rate'] = np.sin(np.linspace(0, 8 * np.pi, len(df))) * 0.001
        
        logger.info(f"ファンディングレートデータを生成しました: {len(df)}行")
        return df
    
    def fetch_oi_data(self, symbol: str = 'BTC-USD', interval: str = '2h', 
                    limit: int = 90) -> pd.DataFrame:
        """
        未決済建玉データを取得
        
        Parameters:
        -----------
        symbol : str
            取得対象のシンボル（例: 'BTC-USD'）
        interval : str
            時間枠（例: '2h'）
        limit : int
            取得する最大行数
            
        Returns:
        --------
        pd.DataFrame
            未決済建玉データ
        """
        # Coinglassクライアントが利用可能ならそれを使用
        if self.use_coinglass and self.coinglass_client:
            # シンボル名をBTCの形式に変換
            coinglass_symbol = symbol.split('-')[0]
            
            try:
                # Coinglassから未決済建玉データを取得
                oi_params = {
                    'symbol': coinglass_symbol,
                    'interval': self._convert_interval_format(interval),
                    'limit': limit
                }
                
                oi_df = self.coinglass_client.fetch('oi', oi_params)
                
                # カラム名を標準化
                if 'oi_o' in oi_df.columns:
                    oi_df = oi_df.rename(columns={
                        'oi_o': 'open_interest_open',
                        'oi_h': 'open_interest_high',
                        'oi_l': 'open_interest_low',
                        'oi_c': 'open_interest'
                    })
                
                # 追加の需給データを取得
                try:
                    # 清算データ
                    liq_params = {
                        'exchanges': 'ALL',
                        'symbol': coinglass_symbol,
                        'interval': self._convert_interval_format(interval),
                        'limit': limit
                    }
                    liq_df = self.coinglass_client.fetch('liq', liq_params)
                    
                    # ロングショート比率
                    ls_params = {
                        'exchange': 'Binance',
                        'symbol': f"{coinglass_symbol}USDT",
                        'interval': self._convert_interval_format(interval, 'lsr'),
                        'limit': limit
                    }
                    ls_df = self.coinglass_client.fetch('lsr', ls_params)
                    
                    # テイカー買い/売りボリューム
                    taker_params = {
                        'exchange': 'Binance',
                        'symbol': f"{coinglass_symbol}USDT",
                        'interval': self._convert_interval_format(interval, 'lsr'),
                        'limit': limit
                    }
                    taker_df = self.coinglass_client.fetch('taker', taker_params)
                    
                    # プレミアム指数
                    premium_params = {
                        'interval': self._convert_interval_format(interval),
                        'limit': limit
                    }
                    premium_df = self.coinglass_client.fetch('premium', premium_params)
                    
                    # マージ処理
                    if liq_df is not None and not liq_df.empty:
                        if not oi_df.empty:
                            oi_df = oi_df.join(liq_df, how='left')
                        else:
                            oi_df = liq_df
                    
                    if ls_df is not None and not ls_df.empty:
                        if not oi_df.empty:
                            oi_df = oi_df.join(ls_df, how='left')
                        else:
                            oi_df = ls_df
                    
                    if taker_df is not None and not taker_df.empty:
                        if not oi_df.empty:
                            oi_df = oi_df.join(taker_df, how='left')
                        else:
                            oi_df = taker_df
                    
                    if premium_df is not None and not premium_df.empty:
                        if not oi_df.empty:
                            oi_df = oi_df.join(premium_df, how='left')
                        else:
                            oi_df = premium_df
                
                except Exception as e:
                    logger.warning(f"追加の需給データ取得に失敗しました: {e}")
                
                if not oi_df.empty:
                    logger.info(f"Coinglassから未決済建玉データを取得しました: {len(oi_df)}行")
                    return oi_df
                
            except Exception as e:
                logger.warning(f"Coinglassからの未決済建玉データ取得に失敗しました: {e}")
        
        # Coinglassが使えない場合はシミュレーションデータを生成
        price_data = self.fetch_price_data(symbol, interval, limit)
        
        # 価格データに基づいてOIをシミュレート
        df = pd.DataFrame(index=price_data.index)
        
        # 価格変動に基づく基本的なトレンド
        price_returns = price_data['close'].pct_change().fillna(0)
        trend = np.cumsum(price_returns)
        
        # OI変化率
        df['oi_change'] = np.random.normal(0, 0.02, size=len(df))
        
        # 清算量 - 価格の急激な変動時に増加
        df['liquidation'] = abs(np.random.exponential(1, size=len(df))) * (price_returns.abs() * 10) ** 2
        
        # ロングショート比率 - 価格トレンドに若干相関
        df['long_short_ratio'] = 1 + np.sin(np.linspace(0, 6 * np.pi, len(df))) * 0.3
        
        # プレミアム指数 - ファンディングに相関
        df['premium_index'] = np.random.normal(0, 0.001, size=len(df))
        
        logger.info(f"OIデータを生成しました: {len(df)}行")
        return df
    
    def fetch_mtf_data(self, symbol: str = 'BTC-USD') -> Dict[str, pd.DataFrame]:
        """
        マルチタイムフレームデータを取得
        
        Parameters:
        -----------
        symbol : str
            取得対象のシンボル（例: 'BTC-USD'）
            
        Returns:
        --------
        dict
            各時間枠のデータフレーム辞書
        """
        logger.info("MTFデータ取得を開始します")
        
        # 設定から時間枠を取得
        minute_frame = self.intervals_config.get('minute', '15m')
        hour_frame = self.intervals_config.get('hour', '2h')
        day_frame = self.intervals_config.get('day', '1d')
        
        # データ取得範囲設定
        lookback = self.intervals_config.get('data_lookback', {})
        day_limit = lookback.get('1d', 365)
        hour_limit = lookback.get('2h', 90)
        minute_limit = lookback.get('15m', 30)
        
        # 各時間枠のデータを取得
        data = {}
        
        # 1. 日足データ（トレンド判断用）
        day_data = self.fetch_price_data(symbol, day_frame, day_limit)
        
        # 日足からトレンドフラグを計算
        day_data['ema200'] = day_data['close'].ewm(span=200).mean()
        day_data['trend_flag'] = (day_data['close'] > day_data['ema200']).astype(int)
        day_data['trend_flag'] = day_data['trend_flag'].shift(1)  # shift(1)を適用
        data[day_frame] = day_data
        
        # 2. 時間足データ（需給指標用）
        hour_data = self.fetch_price_data(symbol, hour_frame, hour_limit)
        
        # 需給指標を追加（別のAPI呼び出しで補完も可能）
        oi_data = self.fetch_oi_data(symbol, hour_frame, hour_limit)
        funding_data = self.fetch_funding_rate(symbol, '8h', hour_limit)
        
        # 需給データをマージ
        hour_data = hour_data.merge(oi_data, left_index=True, right_index=True, how='left')
        hour_data = hour_data.merge(funding_data, left_index=True, right_index=True, how='left')
        
        # shift(1)を需給指標に適用
        cols_to_shift = ['funding_rate', 'oi_change', 'liquidation', 'long_short_ratio', 'premium_index']
        for col in cols_to_shift:
            if col in hour_data.columns:
                hour_data[col] = hour_data[col].shift(1)
        
        data[hour_frame] = hour_data
        
        # 3. 15分足データ（エントリー用）
        minute_data = self.fetch_price_data(symbol, minute_frame, minute_limit * 24 * 4)  # 15分足の日数→行数変換
        data[minute_frame] = minute_data
        
        # データ読み込み確認
        for interval, df in data.items():
            logger.info(f"{interval}データ: {len(df)}行")
        
        return data
    
    def sync_all_data(self, symbol: str = 'BTC-USD', force_update: bool = False) -> Dict[str, pd.DataFrame]:
        """
        全時間枠のデータを同期（キャッシュ更新も含む）
        
        Parameters:
        -----------
        symbol : str
            取得対象のシンボル（例: 'BTC-USD'）
        force_update : bool
            強制的にAPI取得を行うかどうか
            
        Returns:
        --------
        dict
            各時間枠のデータフレーム辞書
        """
        # キャッシュ使用設定を一時的に上書き
        original_cache_setting = self.use_cache
        if force_update:
            self.use_cache = False
        
        try:
            data = self.fetch_mtf_data(symbol)
            
            # データフォルダに保存
            for interval, df in data.items():
                cache_path = self.get_cache_path(symbol, interval)
                df.to_csv(cache_path)
                logger.info(f"データを保存しました: {cache_path}")
            
            return data
        
        finally:
            # 元のキャッシュ設定に戻す
            self.use_cache = original_cache_setting
    
    def get_combined_data(self, symbol: str = 'BTC-USD', target_interval: str = '15m') -> pd.DataFrame:
        """
        異なる時間枠のデータを結合（MTFデータをターゲット時間枠に統合）
        
        Parameters:
        -----------
        symbol : str
            取得対象のシンボル（例: 'BTC-USD'）
        target_interval : str
            ターゲット時間枠（例: '15m'）
            
        Returns:
        --------
        pd.DataFrame
            結合されたデータフレーム
        """
        logger.info(f"MTFデータ結合を開始: ターゲット時間枠 = {target_interval}")
        
        # MTFデータを取得
        mtf_data = self.fetch_mtf_data(symbol)
        
        # ターゲット時間枠のデータをベースとする
        if target_interval not in mtf_data:
            raise ValueError(f"指定されたターゲット時間枠のデータがありません: {target_interval}")
        
        base_df = mtf_data[target_interval].copy()
        logger.info(f"ベースデータフレーム: {len(base_df)}行")
        
        # MTF設定を読み込む
        mtf_config = self._load_config('mtf')
        
        # 各時間枠からデータを転送
        if 'transfers' in mtf_config:
            for transfer_name, transfer_config in mtf_config['transfers'].items():
                source_interval = transfer_config.get('source')
                if source_interval not in mtf_data:
                    logger.warning(f"ソース時間枠のデータがありません: {source_interval}")
                    continue
                
                # ソースデータ
                source_df = mtf_data[source_interval]
                
                # 転送する列
                columns = transfer_config.get('columns', [])
                for column_config in columns:
                    column_name = column_config.get('name')
                    if column_name not in source_df.columns:
                        logger.warning(f"ソースデータに列が見つかりません: {column_name}")
                        continue
                    
                    # 変換操作（必要に応じてshift）
                    transform = column_config.get('transform')
                    source_series = source_df[column_name]
                    
                    if transform and 'shift' in transform:
                        import re
                        shift_match = re.search(r'shift\((\d+)\)', transform)
                        if shift_match:
                            shift_periods = int(shift_match.group(1))
                            source_series = source_df[column_name].shift(shift_periods)
                    
                    # 時間枠変換（リサンプル）
                    resampled_series = self._resample_to_target(source_series, source_interval, 
                                                             target_interval, 
                                                             method=column_config.get('fill_method', 'ffill'))
                    
                    # ベースデータフレームに追加
                    base_df[column_name] = resampled_series
        
        logger.info(f"MTFデータ結合完了: {len(base_df)}列, {len(base_df)}行")
        return base_df
    
    def _resample_to_target(self, series: pd.Series, source_interval: str, 
                           target_interval: str, method: str = 'ffill') -> pd.Series:
        """
        ソース時間枠からターゲット時間枠にデータをリサンプリング
        
        Parameters:
        -----------
        series : pd.Series
            リサンプリングするデータ系列
        source_interval : str
            ソース時間枠
        target_interval : str
            ターゲット時間枠
        method : str
            欠損値の補完方法（'ffill', 'bfill', 'nearest'等）
            
        Returns:
        --------
        pd.Series
            リサンプリングされたデータ系列
        """
        # リサンプリングのルール
        # 大きな時間枠→小さな時間枠への変換
        if source_interval in ['1d', '2h'] and target_interval == '15m':
            # まずダウンサンプリング
            resampled = series.resample(target_interval[:-1] + 'min').asfreq()
            
            # 欠損値を補完
            if method == 'ffill':
                resampled = resampled.fillna(method='ffill')
            elif method == 'bfill':
                resampled = resampled.fillna(method='bfill')
            elif method == 'nearest':
                resampled = resampled.fillna(method='nearest')
            
            return resampled
        
        # 小さな時間枠→大きな時間枠への変換
        elif source_interval == '15m' and target_interval in ['2h', '1d']:
            # アップサンプリング - 最初の値を使用
            if target_interval == '2h':
                rule = '2h'
            else:  # 1d
                rule = 'D'
            
            return series.resample(rule).first()
        
        # 同じ時間枠の場合はそのまま返す
        elif source_interval == target_interval:
            return series
        
        # その他の場合はエラー
        else:
            logger.warning(f"未サポートのリサンプリング: {source_interval} → {target_interval}")
            return pd.Series(index=series.index)
            
    def fetch_single_timeframe_data(self, symbol: str = 'BTC-USD', interval: str = '2h', limit: int = 360) -> pd.DataFrame:
        """
        指定した単一時間枠のデータを取得（MTFではなく2h時間枠のみを使用）
        
        Parameters:
        -----------
        symbol : str
            取得対象のシンボル（例: 'BTC-USD'）
        interval : str 
            時間枠（デフォルト: '2h'）
        limit : int
            取得する日数（デフォルト: 360日分 = 4320レコード）
            
        Returns:
        --------
        pd.DataFrame
            取得したデータ
        """
        logger.info(f"単一時間枠データ取得を開始: {interval}")
        
        # 時間枠ごとのレコード数計算
        records_per_day = {
            '15m': 4 * 24,  # 15分足は1日に96レコード
            '2h': 12,       # 2時間足は1日に12レコード
            '1d': 1         # 日足は1日に1レコード
        }
        
        # 日数からレコード数に変換
        records = limit * records_per_day.get(interval, 12)
        
        # 価格データを取得
        data = self.fetch_price_data(symbol, interval, records)
        
        # 需給指標データを追加
        if interval == '2h':
            # OIデータを取得
            oi_data = self.fetch_oi_data(symbol, interval, records)
            
            # ファンディングレートを取得
            funding_data = self.fetch_funding_rate(symbol, '8h', records)
            
            # データをマージ
            data = data.merge(oi_data, left_index=True, right_index=True, how='left')
            data = data.merge(funding_data, left_index=True, right_index=True, how='left')
            
            # 需給指標にshift(1)を適用（ラグを考慮）
            cols_to_shift = ['funding_rate', 'oi_change', 'liquidation', 'long_short_ratio', 'premium_index']
            for col in cols_to_shift:
                if col in data.columns:
                    data[col] = data[col].shift(1)
            
            logger.info(f"需給指標を追加し、shift(1)を適用しました")
        
        logger.info(f"{interval}データ取得完了: {len(data)}行 x {len(data.columns)}列")
        return data
        
    def sync_single_timeframe(self, symbol: str = 'BTC-USD', interval: str = '2h', force_update: bool = False) -> pd.DataFrame:
        """
        単一時間枠のデータを同期（キャッシュ更新も含む）
        
        Parameters:
        -----------
        symbol : str
            取得対象のシンボル（例: 'BTC-USD'）
        interval : str
            時間枠（デフォルト: '2h'）
        force_update : bool
            強制的にAPI取得を行うかどうか
            
        Returns:
        --------
        pd.DataFrame
            取得したデータ
        """
        # キャッシュ使用設定を一時的に上書き
        original_cache_setting = self.use_cache
        if force_update:
            self.use_cache = False
        
        try:
            data = self.fetch_single_timeframe_data(symbol, interval)
            
            # データフォルダに保存
            cache_path = self.get_cache_path(symbol, interval)
            data.to_csv(cache_path)
            logger.info(f"データを保存しました: {cache_path}")
            
            return data
        
        finally:
            # 元のキャッシュ設定に戻す
            self.use_cache = original_cache_setting
