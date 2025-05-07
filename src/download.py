#!/usr/bin/env python
"""
download.py - Coinglass API データ取得モジュール

機能:
- Coinglass API から BTC データを取得
- 複数エンドポイント (price, oi, funding, liq, lsr, taker, orderbook, premium) をサポート
- 複数時間枠 (intervals.yaml で設定) に対応
- tenacity による自動リトライ
- データを parquet 形式で保存
"""

import os
import sys
import time
import yaml
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import requests
import pyarrow
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import slack_sdk
from typing import Dict, List, Optional, Union, Any
from config_loader import CG_API, SLACK_WEBHOOK, USE_REAL_DATA, MAX_SAMPLES

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'download.log'), mode='a')
    ]
)
logger = logging.getLogger('download')

# プロジェクトルートディレクトリ
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = ROOT_DIR / 'config'
DATA_DIR = ROOT_DIR / 'data'

# ログディレクトリ作成
os.makedirs(ROOT_DIR / 'logs', exist_ok=True)

# Coinglass API エンドポイント定義
ENDPOINTS = {
    'price': 'futures/btc/market/candles', # 価格データ
    'oi': 'futures/btc/openInterest/history', # オープンインタレスト
    'funding': 'futures/funding-rates/history', # ファンディングレート
    'liq': 'futures/liquidation/history', # 清算データ
    'lsr': 'futures/longShortRatio', # ロングショート比率
    'taker': 'futures/takerVolume', # テイカー出来高
    'orderbook': 'futures/orderBook/ratio', # オーダーブック
    'premium': 'futures/premiumIndex' # プレミアム指数
}

class CoinglassDownloader:
    """Coinglass API からデータをダウンロードするクラス"""
    
    def __init__(self, config_dir: Path = CONFIG_DIR):
        """初期化

        Args:
            config_dir: 設定ファイルディレクトリ
        """
        self.config_dir = config_dir
        self.api_key = self._load_api_key()
        self.intervals = self._load_intervals()
        self.slack_webhook = self._load_slack_webhook()
        self.slack_client = self._setup_slack_client() if self.slack_webhook else None
        
        # データディレクトリの確認・作成
        self._setup_data_directories()
        
    def _load_api_key(self) -> str:
        """API キーを読み込む

        Returns:
            str: Coinglass API キー
        """
        try:
            return CG_API
        except Exception as e:
            logger.error(f"API キーの読み込みに失敗しました: {e}")
            raise
    
    def _load_intervals(self) -> List[str]:
        """時間枠設定を読み込む

        Returns:
            List[str]: 時間枠リスト
        """
        try:
            with open(self.config_dir / 'intervals.yaml', 'r') as f:
                config = yaml.safe_load(f)
            return config['intervals']
        except Exception as e:
            logger.error(f"時間枠設定の読み込みに失敗しました: {e}")
            raise
    
    def _load_slack_webhook(self) -> Optional[str]:
        """Slack Webhook URL を読み込む

        Returns:
            Optional[str]: Slack Webhook URL
        """
        try:
            return SLACK_WEBHOOK
        except Exception as e:
            logger.error(f"Slack Webhook の読み込みに失敗しました: {e}")
            return None
    
    def _setup_slack_client(self) -> Optional[slack_sdk.WebhookClient]:
        """Slack クライアントを設定

        Returns:
            Optional[slack_sdk.WebhookClient]: Slack Webhook クライアント
        """
        if self.slack_webhook:
            return slack_sdk.WebhookClient(self.slack_webhook)
        return None
    
    def _setup_data_directories(self):
        """データディレクトリを設定"""
        for interval in self.intervals:
            # 生データディレクトリ
            raw_dir = DATA_DIR / 'raw' / interval
            os.makedirs(raw_dir, exist_ok=True)
            
            # 特徴量データディレクトリ
            features_dir = DATA_DIR / 'features' / interval
            os.makedirs(features_dir, exist_ok=True)
    
    def send_slack_message(self, message: str, emoji: str = ":chart_with_upwards_trend:"):
        """Slack にメッセージを送信

        Args:
            message: 送信するメッセージ
            emoji: メッセージに付けるアイコン
        """
        if self.slack_client:
            try:
                response = self.slack_client.send(
                    text=f"{emoji} {message}"
                )
                if not response.status_code == 200:
                    logger.warning(f"Slack 通知の送信に失敗: {response.status_code}, {response.body}")
            except Exception as e:
                logger.error(f"Slack 通知の送信でエラー: {e}")
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError))
    )
    def fetch(self, endpoint: str, params: Dict[str, Any], key: str = None) -> pd.DataFrame:
        """Coinglass API からデータをフェッチする

        Args:
            endpoint: エンドポイント名
            params: リクエストパラメータ
            key: エンドポイントキー (省略時はエンドポイント名)

        Returns:
            pd.DataFrame: 取得したデータフレーム
        """
        if key is None:
            key = endpoint
            
        url = f"https://api.coinglass.com/api/v3/{ENDPOINTS[key]}"
        headers = {
            "accept": "application/json",
            "coinglassSecret": self.api_key
        }
        
        try:
            logger.info(f"API リクエスト: {key}, パラメータ: {params}")
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code != 200:
                error_msg = f"API エラー: {key}, ステータスコード: {response.status_code}, レスポンス: {response.text}"
                logger.error(error_msg)
                self.send_slack_message(f"API エラー: {key}, ステータスコード: {response.status_code}", emoji=":x:")
                response.raise_for_status()
                
            data = response.json()
            
            # データ構造に応じて処理
            if 'data' not in data or not data['data']:
                error_msg = f"空のレスポンス: {key}, レスポンス: {data}"
                logger.error(error_msg)
                self.send_slack_message(f"空のレスポンス: {key}", emoji=":warning:")
                raise ValueError(error_msg)
                
            # エンドポイント別のデータ抽出
            df = self._parse_response(data, key)
            
            # 必要に応じてUTC->JSTへの変換と前処理
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp').sort_index()
            elif 'ts' in df.columns:
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                df = df.set_index('ts').sort_index()
                
            return df
        
        except Exception as e:
            error_msg = f"データ取得エラー: {key}, エラー: {str(e)}"
            logger.error(error_msg)
            self.send_slack_message(f"データ取得エラー: {key}, {str(e)}", emoji=":x:")
            raise
    
    def _parse_response(self, response: Dict, endpoint: str) -> pd.DataFrame:
        """レスポンスを解析してDataFrameに変換

        Args:
            response: API レスポンス
            endpoint: エンドポイント名

        Returns:
            pd.DataFrame: 変換したデータフレーム
        """
        data = response.get('data', {})
        
        if endpoint == 'price':
            # 価格データの処理
            return pd.DataFrame(data)
        
        elif endpoint == 'oi':
            # オープンインタレストの処理
            return pd.DataFrame(data)
        
        elif endpoint == 'funding':
            # ファンディングレートの処理
            filtered_data = []
            for item in data:
                if item.get('symbol') == 'BTC':
                    filtered_data.append(item)
            return pd.DataFrame(filtered_data)
        
        elif endpoint in ['liq', 'lsr', 'taker', 'orderbook', 'premium']:
            # その他のエンドポイント処理 (汎用)
            return pd.DataFrame(data)
        
        # デフォルト処理
        return pd.DataFrame(data)
    
    def download_data(self, interval: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, limit: int = 4320):
        """指定された時間枠でデータをダウンロード

        Args:
            interval: 時間枠
            start_time: 開始時刻 (省略時は現在時刻から limit バーを計算)
            end_time: 終了時刻 (省略時は現在時刻)
            limit: 取得するバー数
        """
        if end_time is None:
            end_time = datetime.now()
            
        if start_time is None:
            # interval から期間を計算
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
                raise ValueError(f"サポートされていない時間枠: {interval}")
                
        # Unix タイムスタンプに変換 (ミリ秒)
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        logger.info(f"データ取得開始: {interval}, {start_time} から {end_time}")
        self.send_slack_message(f"データ取得開始: {interval}", emoji=":rocket:")
        
        # 全エンドポイントからデータを取得
        for endpoint in ENDPOINTS:
            try:
                logger.info(f"エンドポイント処理: {endpoint}, 時間枠: {interval}")
                
                # エンドポイント別のパラメータ設定
                params = {
                    'symbol': 'BTC',
                    'interval': interval,
                    'from': start_ts,
                    'to': end_ts,
                    'limit': limit
                }
                
                df = self.fetch(endpoint, params)
                
                # 列名にエンドポイントのプレフィックスを追加
                df = df.add_prefix(f"{endpoint}_")
                
                # ファイルに保存
                output_path = DATA_DIR / 'raw' / interval / f"{endpoint}.parquet"
                df.to_parquet(output_path)
                
                logger.info(f"保存完了: {output_path}, レコード数: {len(df)}")
                
            except Exception as e:
                error_msg = f"エンドポイント取得エラー: {endpoint}, 時間枠: {interval}, エラー: {str(e)}"
                logger.error(error_msg)
                self.send_slack_message(error_msg, emoji=":x:")
        
        success_msg = f"データ取得完了: {interval}, 期間: {start_time} から {end_time}"
        logger.info(success_msg)
        self.send_slack_message(success_msg, emoji=":white_check_mark:")
    
    def download_all_intervals(self):
        """すべての時間枠でデータをダウンロード"""
        for interval in self.intervals:
            self.download_data(interval)

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='Coinglass データダウンローダー')
    parser.add_argument('--interval', type=str, help='時間枠 (例: 15m, 2h)')
    parser.add_argument('--start', type=str, help='開始時刻 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end', type=str, help='終了時刻 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--limit', type=int, default=4320, help='取得するバー数')
    return parser.parse_args()

def parse_advanced_args():
    """拡張コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='Coinglass データダウンローダー')
    parser.add_argument('--interval', type=str, help='時間枠 (例: 15m, 2h)')
    parser.add_argument('--start', type=str, help='開始時刻 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end', type=str, help='終了時刻 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--limit', type=int, default=MAX_SAMPLES, help='取得するバー数')
    parser.add_argument('--allow-sample', action='store_true', 
                        help='API接続エラー時にサンプルデータの使用を許可')
    return parser.parse_args()

def main():
    """メイン関数"""
    try:
        # ログディレクトリ作成
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'), exist_ok=True)
        
        # 実データを使用するか確認
        if not USE_REAL_DATA:
            logger.warning("""
            =====================================================================
            警告: システム設定でサンプルデータの使用が指定されています。
            config/system.yamlの設定を変更してください:
            
            data:
              use_real_data: true
              
            実データを使用せずサンプルデータでのみ処理を続行します。
            =====================================================================
            """)
            
            # サンプルデータ生成スクリプトを実行
            import subprocess
            subprocess.run([sys.executable, os.path.join(ROOT_DIR, "src", "generate_sample_data.py"), "--force"])
            return 0
        
        # 拡張引数解析
        args = parse_advanced_args()
        
        try:
            downloader = CoinglassDownloader()
            
            if args.interval:
                # 単一時間枠の処理
                start_time = None
                end_time = None
                
                if args.start:
                    start_time = datetime.strptime(args.start, '%Y-%m-%d %H:%M:%S')
                if args.end:
                    end_time = datetime.strptime(args.end, '%Y-%m-%d %H:%M:%S')
                    
                downloader.download_data(
                    interval=args.interval,
                    start_time=start_time,
                    end_time=end_time,
                    limit=args.limit
                )
            else:
                # 全時間枠の処理
                downloader.download_all_intervals()
        
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.error(f"APIへの接続エラー: {str(e)}")
            logger.error("Coinglassサーバーに接続できません。ネットワーク接続を確認してください。")
            
            if args.allow_sample:
                logger.warning("""
                =====================================================================
                警告: API接続エラーのため、サンプルデータを生成します。
                サンプルデータはテスト目的でのみ使用し、
                実際のトレーディングには使用しないでください。
                =====================================================================
                """)
                
                # サンプルデータ生成スクリプトを実行
                import subprocess
                subprocess.run([sys.executable, os.path.join(ROOT_DIR, "src", "generate_sample_data.py"), "--force"])
                return 0
            else:
                logger.error("""
                =====================================================================
                エラー: API接続エラーが発生しました。
                
                サンプルデータを使用するには、--allow-sample オプションを付けて再実行してください:
                python src/download.py --allow-sample
                
                または、以下の方法でサンプルデータを直接生成できます:
                python src/generate_sample_data.py --force
                
                警告: サンプルデータはテスト目的専用です。
                      本番環境や実際のトレーディングには使用しないでください。
                =====================================================================
                """)
                return 1
        
        return 0
    
    except Exception as e:
        logger.error(f"実行エラー: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
