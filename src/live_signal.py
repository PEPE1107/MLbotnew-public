#!/usr/bin/env python
"""
live_signal.py - リアルタイムシグナル生成モジュール

機能:
- FastAPI + asyncio による常駐 microservice 
- Coinglass API からリアルタイムデータ取得
- 訓練済みモデルによる予測
- S3 / ファイルへのシグナル出力
- Prometheus メトリクスの提供
"""

import os
import sys
import yaml
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

# FastAPI
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Prometheus メトリクス
import prometheus_client
from prometheus_client import Counter, Gauge, Histogram

# データ取得とモデル
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import lightgbm as lgb
import catboost as ctb

# S3 (オプション)
try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    logging.warning("boto3 がインストールされていません。S3 出力機能は無効です。")

# Slack
try:
    from slack_sdk.webhook import WebhookClient
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    logging.warning("slack_sdk がインストールされていません。Slack 通知機能は無効です。")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'live_signal.log'), mode='a')
    ]
)
logger = logging.getLogger('live_signal')

# プロジェクトルートディレクトリ
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = ROOT_DIR / 'config'
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'
SIGNALS_DIR = ROOT_DIR / 'signals'

# ログとシグナルディレクトリの作成
os.makedirs(ROOT_DIR / 'logs', exist_ok=True)
os.makedirs(SIGNALS_DIR, exist_ok=True)

# Prometheus メトリクス
prometheus_client.start_http_server(9090)  # デフォルトポート
SIGNAL_COUNTER = Counter('mlbot_signals_total', 'Total number of signals generated', ['signal_type'])
POSITION_GAUGE = Gauge('mlbot_position_size', 'Current position size (-1 to 1)')
PREDICTION_GAUGE = Gauge('mlbot_prediction_probability', 'Current prediction probability (0 to 1)')
LATENCY_HISTOGRAM = Histogram('mlbot_api_latency_seconds', 'API request latency in seconds')
ERROR_COUNTER = Counter('mlbot_errors_total', 'Total number of errors', ['error_type'])

# FastAPI アプリケーション
app = FastAPI(
    title="ML Signal Bot",
    description="Coinglass データからBTCのトレードシグナルを生成するサービス",
    version="1.0.0"
)

# シグナルレスポンスモデル
class SignalResponse(BaseModel):
    timestamp: float
    position: float
    prediction: float
    confidence: float
    interval: str
    signal_type: str

# グローバル変数
running = False
config = {}
models = {}
feature_processors = {}
slack_client = None

class LiveSignalGenerator:
    """リアルタイムシグナル生成クラス"""
    
    def __init__(self, config_dir: Path = CONFIG_DIR):
        """初期化
        
        Args:
            config_dir: 設定ファイルディレクトリ
        """
        self.config_dir = config_dir
        self.api_config = self._load_api_config()
        self.intervals = self._load_intervals()
        self.signals_dir = SIGNALS_DIR
        
        # Slackクライアント設定
        self.setup_slack_client()
        
        # 最新のシグナル状態（インターバルごと）
        self.current_signals = {interval: None for interval in self.intervals}
        
        # S3クライアント設定（必要に応じて）
        self.s3_client = None
        if S3_AVAILABLE and 's3_bucket' in self.api_config:
            self.setup_s3_client()
    
    def _load_api_config(self) -> Dict:
        """API設定を読み込む
        
        Returns:
            Dict: API設定
        """
        try:
            with open(self.config_dir / 'api_keys.yaml', 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"API設定の読み込みに失敗しました: {e}")
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
    
    def setup_slack_client(self):
        """Slack Webhook クライアントを設定"""
        if SLACK_AVAILABLE and 'slack_webhook' in self.api_config:
            webhook_url = self.api_config['slack_webhook']
            if webhook_url:
                self.slack_client = WebhookClient(webhook_url)
                logger.info("Slack Webhook クライアント初期化完了")
            else:
                self.slack_client = None
        else:
            self.slack_client = None
    
    def setup_s3_client(self):
        """S3 クライアントを設定 (オプション)"""
        if not S3_AVAILABLE:
            return
        
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.api_config.get('aws_access_key_id'),
                aws_secret_access_key=self.api_config.get('aws_secret_access_key'),
                region_name=self.api_config.get('aws_region', 'us-east-1')
            )
            logger.info("S3 クライアント初期化完了")
        except Exception as e:
            logger.error(f"S3 クライアント初期化に失敗: {e}")
            self.s3_client = None
    
    def send_slack_notification(self, message: str, emoji: str = ":chart_with_upwards_trend:"):
        """Slack に通知を送信
        
        Args:
            message: 送信するメッセージ
            emoji: 使用する絵文字
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
                ERROR_COUNTER.labels(error_type='slack_notification').inc()
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(requests.exceptions.RequestException)
    )
    async def fetch_latest_data(self, interval: str) -> pd.DataFrame:
        """最新のデータを取得
        
        Args:
            interval: 時間枠
            
        Returns:
            pd.DataFrame: 最新データのデータフレーム
        """
        start_time = time.time()
        logger.info(f"時間枠 {interval} の最新データ取得開始")
        
        # エンドポイントと API キー
        api_key = self.api_config['cg_api']
        
        # 計測開始
        with LATENCY_HISTOGRAM.time():
            # 各エンドポイントからデータ取得
            data_frames = {}
            
            for endpoint, path in {
                'price': 'futures/btc/market/candles',
                'oi': 'futures/btc/openInterest/history',
                'funding': 'futures/funding-rates/history',
                'liq': 'futures/liquidation/history',
                'lsr': 'futures/longShortRatio',
                'taker': 'futures/takerVolume',
                'orderbook': 'futures/orderBook/ratio',
                'premium': 'futures/premiumIndex'
            }.items():
                try:
                    url = f"https://api.coinglass.com/api/v3/{path}"
                    headers = {
                        "accept": "application/json",
                        "coinglassSecret": api_key
                    }
                    
                    # 現在時刻から1日前までのデータを取得
                    now = int(time.time() * 1000)
                    one_day_ago = int((time.time() - 86400) * 1000)
                    
                    params = {
                        'symbol': 'BTC',
                        'interval': interval,
                        'from': one_day_ago,
                        'to': now,
                        'limit': 100  # 最大100件まで
                    }
                    
                    response = await asyncio.to_thread(
                        requests.get, url, headers=headers, params=params
                    )
                    
                    if response.status_code != 200:
                        error_msg = f"API エラー: {endpoint}, ステータスコード: {response.status_code}"
                        logger.error(error_msg)
                        ERROR_COUNTER.labels(error_type='api_error').inc()
                        continue
                        
                    data = response.json()
                    
                    if 'data' not in data or not data['data']:
                        logger.warning(f"エンドポイント {endpoint} からデータを取得できませんでした")
                        continue
                    
                    # DataFrame に変換
                    df = pd.DataFrame(data['data'])
                    
                    # タイムスタンプをインデックスに設定
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df = df.set_index('timestamp')
                    elif 'ts' in df.columns:
                        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                        df = df.set_index('ts')
                    
                    # 列名にプレフィックスを追加
                    df = df.add_prefix(f"{endpoint}_")
                    
                    data_frames[endpoint] = df
                    
                except Exception as e:
                    logger.error(f"エンドポイント {endpoint} の取得中にエラー: {e}")
                    ERROR_COUNTER.labels(error_type='data_fetch').inc()
        
        # データフレームを結合
        if not data_frames:
            raise ValueError("どのエンドポイントからもデータを取得できませんでした")
        
        merged = pd.concat(data_frames.values(), axis=1)
        
        # インデックスでソート
        merged = merged.sort_index()
        
        # 前方補完で欠損値を埋める
        merged = merged.ffill()
        
        # 最新のレコードを使用
        latest_data = merged.iloc[-1:].copy()
        
        elapsed = time.time() - start_time
        logger.info(f"時間枠 {interval} の最新データ取得完了 (所要時間: {elapsed:.2f}秒)")
        
        return latest_data
    
    def load_model(self, interval: str, model_type: str = 'lightgbm') -> Tuple[Any, List[str]]:
        """モデルを読み込む
        
        Args:
            interval: 時間枠
            model_type: モデル種類 ('lightgbm' または 'catboost')
            
        Returns:
            Tuple[Any, List[str]]: モデルと特徴量のリスト
        """
        logger.info(f"時間枠 {interval} の {model_type} モデル読み込み開始")
        
        model_dir = MODELS_DIR / interval / model_type
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"モデルディレクトリが見つかりません: {model_dir}")
        
        # 最新のモデルファイルを検索
        if model_type == 'lightgbm':
            model_files = list(model_dir.glob("model_fold1_*.txt"))
        elif model_type == 'catboost':
            model_files = list(model_dir.glob("model_fold1_*.cbm"))
        else:
            raise ValueError(f"サポートされていないモデル種類です: {model_type}")
        
        if not model_files:
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_dir}")
        
        # 最新のモデルを使用
        model_file = sorted(model_files)[-1]
        logger.info(f"モデルファイル読み込み: {model_file}")
        
        # モデル情報ファイルの検索
        timestamp = model_file.stem.split('_')[-1]
        model_info_file = MODELS_DIR / f"model_info_{timestamp}.json"
        
        # 特徴量リスト
        if os.path.exists(model_info_file):
            with open(model_info_file, 'r') as f:
                model_info = json.load(f)
            feature_cols = model_info.get('feature_cols', [])
        else:
            logger.warning(f"モデル情報ファイルが見つかりません: {model_info_file}")
            # デフォルトで zscore で終わる特徴量を使用
            feature_cols = []
        
        # モデル読み込み
        if model_type == 'lightgbm':
            model = lgb.Booster(model_file=str(model_file))
        elif model_type == 'catboost':
            if 'Classification' in model_file.stem:
                model = ctb.CatBoostClassifier()
            else:
                model = ctb.CatBoostRegressor()
            model.load_model(str(model_file))
        
        logger.info(f"モデル読み込み完了: 特徴量数={len(feature_cols)}")
        
        return model, feature_cols
    
    def load_feature_processor(self, interval: str) -> Dict:
        """特徴量処理のパラメータを読み込む
        
        Args:
            interval: 時間枠
            
        Returns:
            Dict: 特徴量処理のパラメータ
        """
        # TODO: 実際の特徴量処理パラメータを読み込む実装
        # ここでは簡易的な実装
        return {
            'zscore_window': 100
        }
    
    def process_features(self, df: pd.DataFrame, feature_processor: Dict) -> pd.DataFrame:
        """特徴量を処理
        
        Args:
            df: 入力データフレーム
            feature_processor: 特徴量処理のパラメータ
            
        Returns:
            pd.DataFrame: 処理済みデータフレーム
        """
        # TODO: feature.py の実装に合わせた特徴量処理を実装
        # ここでは簡易的な実装
        processed = df.copy()
        
        # 末尾が _zscore の特徴量がすでにある場合はそのまま使用
        zscore_features = [col for col in processed.columns if col.endswith('_zscore')]
        
        # 既存特徴量がない場合は、単純なZ正規化を適用
        if not zscore_features and len(df) > 0:
            for col in df.columns:
                if col.startswith(('price_', 'oi_', 'funding_', 'liq_', 'lsr_', 'taker_', 'orderbook_', 'premium_')):
                    processed[f"{col}_zscore"] = 0  # シングルレコードの場合はゼロに設定
        
        return processed
    
    def predict(self, df: pd.DataFrame, model: Any, feature_cols: List[str], 
               threshold: float = 0.55) -> Dict:
        """モデルで予測
        
        Args:
            df: 処理済みデータフレーム
            model: モデル
            feature_cols: 特徴量リスト
            threshold: シグナル閾値
            
        Returns:
            Dict: 予測結果
        """
        # 特徴量の抽出
        features = []
        for col in feature_cols:
            if col in df.columns:
                features.append(col)
            else:
                logger.warning(f"特徴量 {col} がデータフレームに見つかりません")
        
        if not features:
            logger.error("有効な特徴量がデータフレームに見つかりません")
            return {
                'position': 0,
                'prediction': 0.5,
                'confidence': 0,
                'signal_type': 'neutral'
            }
        
        # 特徴量マトリックスの作成
        X = df[features].values
        
        # モデルで予測
        if isinstance(model, lgb.Booster):
            # LightGBM
            prediction = model.predict(X)[0]
        elif isinstance(model, (ctb.CatBoostClassifier, ctb.CatBoostRegressor)):
            # CatBoost
            if isinstance(model, ctb.CatBoostClassifier):
                prediction = model.predict_proba(X)[0, 1]
            else:
                prediction = model.predict(X)[0]
        else:
            raise TypeError(f"サポートされていないモデル種類です: {type(model)}")
        
        # 位置サイズ計算: 閾値以上ならロング、(1-閾値)以下ならショート
        if prediction > threshold:
            # ロングシグナル: 0〜1
            confidence = min(1.0, (prediction - threshold) / (1 - threshold) * 2)
            position = confidence
            signal_type = 'long'
        elif prediction < (1 - threshold):
            # ショートシグナル: -1〜0
            confidence = min(1.0, ((1 - threshold) - prediction) / (1 - threshold) * 2)
            position = -confidence
            signal_type = 'short'
        else:
            # ニュートラル: 0
            confidence = 0
            position = 0
            signal_type = 'neutral'
        
        result = {
            'position': float(position),
            'prediction': float(prediction),
            'confidence': float(confidence),
            'signal_type': signal_type
        }
        
        return result
    
    def save_signal(self, interval: str, signal: Dict):
        """シグナルを保存
        
        Args:
            interval: 時間枠
            signal: シグナル情報
        """
        # タイムスタンプの追加
        signal['timestamp'] = time.time()
        signal['interval'] = interval
        
        # JSONファイルとして保存
        signal_file = self.signals_dir / f"btc_{interval}.json"
        
        with open(signal_file, 'w') as f:
            json.dump(signal, f, indent=2)
        
        # S3に保存 (設定されている場合)
        if self.s3_client and 's3_bucket' in self.api_config:
            try:
                bucket = self.api_config['s3_bucket']
                key = f"signals/btc_{interval}.json"
                
                self.s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=json.dumps(signal),
                    ContentType='application/json'
                )
                logger.info(f"シグナルをS3に保存: s3://{bucket}/{key}")
            except Exception as e:
                logger.error(f"S3へのシグナル保存中にエラー: {e}")
                ERROR_COUNTER.labels(error_type='s3_upload').inc()
        
        # Prometheusメトリクスの更新
        SIGNAL_COUNTER.labels(signal_type=signal['signal_type']).inc()
        POSITION_GAUGE.set(signal['position'])
        PREDICTION_GAUGE.set(signal['prediction'])
        
        # シグナル変更の通知
        prev_signal = self.current_signals[interval]
        if prev_signal is None or prev_signal['signal_type'] != signal['signal_type']:
            direction = {
                'long': 'ロング',
                'short': 'ショート',
                'neutral': 'ニュートラル'
            }.get(signal['signal_type'], signal['signal_type'])
            
            emoji = {
                'long': ':chart_with_upwards_trend:',
                'short': ':chart_with_downwards_trend:',
                'neutral': ':scales:'
            }.get(signal['signal_type'], ':chart_with_upwards_trend:')
            
            message = (
                f"BTC-{interval} シグナル変更: {direction}\n"
                f"位置サイズ: {signal['position']:.2f}\n"
                f"予測確率: {signal['prediction']:.4f}\n"
                f"確信度: {signal['confidence']:.2f}"
            )
            
            self.send_slack_notification(message, emoji)
        
        # 現在のシグナルを更新
        self.current_signals[interval] = signal

    async def generate_signal(self, interval: str, model_type: str = 'lightgbm', threshold: float = 0.55) -> Dict:
        """シグナルを生成
        
        Args:
            interval: 時間枠
            model_type: モデル種類
            threshold: シグナル閾値
            
        Returns:
            Dict: 生成したシグナル
        """
        try:
            # モデルと特徴量プロセッサを読み込み（未読み込みの場合）
            model_key = f"{interval}_{model_type}"
            
            if model_key not in models:
                model, feature_cols = self.load_model(interval, model_type)
                models[model_key] = (model, feature_cols)
            else:
                model, feature_cols = models[model_key]
            
            if interval not in feature_processors:
                feature_processors[interval] = self.load_feature_processor(interval)
            
            # 最新データの取得
            latest_data = await self.fetch_latest_data(interval)
            
            # 特徴量処理
            processed_data = self.process_features(latest_data, feature_processors[interval])
            
            # 予測
            signal = self.predict(processed_data, model, feature_cols, threshold)
            
            # シグナル保存
            self.save_signal(interval, signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"シグナル生成中にエラー: {e}", exc_info=True)
            ERROR_COUNTER.labels(error_type='signal_generation').inc()
            raise

async def signal_generator_task():
    """シグナル生成タスク（バックグラウンドで実行）"""
    global running, config
    
    generator = LiveSignalGenerator()
    intervals = generator.intervals
    
    # デフォルト設定
    model_type = config.get('model_type', 'lightgbm')
    threshold = config.get('threshold', 0.55)
    interval_seconds = config.get('interval_seconds', {})
    
    running = True
    logger.info(f"シグナル生成タスク開始: インターバル={intervals}, モデル={model_type}")
    
    # 各時間枠の最終更新時刻
    last_updated = {interval: 0 for interval in intervals}
    
    # 時間枠ごとの更新間隔（秒）
    update_intervals = {}
    for interval in intervals:
        if interval in interval_seconds:
            update_intervals[interval] = interval_seconds[interval]
        elif interval.endswith('m'):
            # 分単位のインターバル: 更新間隔はインターバルの1/3
            minutes = int(interval[:-1])
            update_intervals[interval] = minutes * 60 // 3
        elif interval.endswith('h'):
            # 時間単位のインターバル: 更新間隔はインターバルの1/6
            hours = int(interval[:-1])
            update_intervals[interval] = hours * 3600 // 6
        else:
            # デフォルト: 15分間隔
            update_intervals[interval] = 15 * 60
    
    while running:
        for interval in intervals:
            try:
                current_time = time.time()
                
                # 更新間隔を確認
                if current_time - last_updated[interval] > update_intervals[interval]:
                    logger.info(f"時間枠 {interval} のシグナル更新開始")
                    
                    # シグナル生成
                    await generator.generate_signal(interval, model_type, threshold)
                    
                    # 最終更新時刻を更新
                    last_updated[interval] = current_time
                    
                    logger.info(f"時間枠 {interval} のシグナル更新完了")
            
            except Exception as e:
                logger.error(f"時間枠 {interval} のシグナル更新中にエラー: {e}")
                ERROR_COUNTER.labels(error_type='update_cycle').inc()
        
        # 短い間隔で更新状況を確認
        await asyncio.sleep(10)

# API ルート

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"status": "active", "service": "ML Signal Bot"}

@app.get("/status")
async def status():
    """サービスステータス"""
    global running
    return {
        "running": running,
        "config": config,
        "models_loaded": list(models.keys()),
        "intervals": list(feature_processors.keys()) if feature_processors else []
    }

@app.post("/start")
async def start_service(background_tasks: BackgroundTasks, model_type: str = 'lightgbm', threshold: float = 0.55):
    """サービス開始"""
    global running, config
    
    if running:
        return {"status": "already_running"}
    
    config = {
        "model_type": model_type,
        "threshold": threshold
    }
    
    # バックグラウンドタスクの開始
    background_tasks.add_task(signal_generator_task)
    
    return {"status": "started", "config": config}

@app.post("/stop")
async def stop_service():
    """サービス停止"""
    global running
    
    if not running:
        return {"status": "not_running"}
    
    running = False
    return {"status": "stopping"}

@app.get("/signals/{interval}")
async def get_signal(interval: str):
    """特定時間枠のシグナルを取得"""
    generator = LiveSignalGenerator()
    
    if interval not in generator.intervals:
        raise HTTPException(status_code=404, detail=f"時間枠 {interval} が見つかりません")
    
    signal_file = SIGNALS_DIR / f"btc_{interval}.json"
    
    if not os.path.exists(signal_file):
        raise HTTPException(status_code=404, detail=f"時間枠 {interval} のシグナルが見つかりません")
    
    with open(signal_file, 'r') as f:
        signal = json.load(f)
    
    return signal

@app.post("/generate/{interval}")
async def generate_signal_endpoint(interval: str, model_type: str = 'lightgbm', threshold: float = 0.55):
    """指定時間枠のシグナルを生成"""
    generator = LiveSignalGenerator()
    
    if interval not in generator.intervals:
        raise HTTPException(status_code=404, detail=f"時間枠 {interval} が見つかりません")
    
    try:
        signal = await generator.generate_signal(interval, model_type, threshold)
        return signal
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """メイン関数"""
    # ログディレクトリの作成
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'), exist_ok=True)
    
    # サービス開始
    uvicorn.run(
        "live_signal:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

if __name__ == "__main__":
    main()
