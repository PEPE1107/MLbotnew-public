#!/usr/bin/env python
"""
execution.py - 注文・約定実行モジュール

機能:
- Bybit REST/WebSocket API により自動売買を実行
- シグナルサービスからポジションサイズを取得
- ポジション管理とリスク制限
- 約定状況のモニタリングとログ記録
"""

import os
import sys
import yaml
import json
import time
import logging
import argparse
import asyncio
import signal
from datetime import datetime, timedelta
from pathlib import Path
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple, Any, Union
import traceback
import uuid

# データ処理ライブラリ
import pandas as pd
import numpy as np

# Bybit API
try:
    from pybit.unified_trading import HTTP, WebSocket
    BYBIT_AVAILABLE = True
except ImportError:
    BYBIT_AVAILABLE = False
    logging.warning("pybit がインストールされていません。Bybit API は無効です。")

# Prometheus メトリクス
try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client がインストールされていません。メトリクス出力機能は無効です。")

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
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'execution.log'), mode='a')
    ]
)
logger = logging.getLogger('execution')

# プロジェクトルートディレクトリ
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = ROOT_DIR / 'config'
SIGNALS_DIR = ROOT_DIR / 'signals'

# ログディレクトリ作成
os.makedirs(ROOT_DIR / 'logs', exist_ok=True)
os.makedirs(SIGNALS_DIR, exist_ok=True)

# Prometheus メトリクス（有効な場合）
if PROMETHEUS_AVAILABLE:
    # メトリクス定義
    TRADE_COUNTER = Counter('mlbot_trades_total', 'Total number of executed trades', ['direction', 'status'])
    POSITION_GAUGE = Gauge('mlbot_position_size', 'Current position size in BTC')
    POSITION_VALUE_GAUGE = Gauge('mlbot_position_value_usd', 'Current position value in USD')
    NAV_GAUGE = Gauge('mlbot_nav_usd', 'Net Asset Value in USD')
    PNL_GAUGE = Gauge('mlbot_pnl_usd', 'Unrealized PnL in USD')
    DAILY_PNL_GAUGE = Gauge('mlbot_daily_pnl_usd', 'Daily PnL in USD')
    TRADE_LATENCY = Histogram('mlbot_trade_latency_seconds', 'Trade execution latency in seconds')
    ORDER_LATENCY = Histogram('mlbot_order_latency_seconds', 'Order placement latency in seconds')
    ERROR_COUNTER = Counter('mlbot_errors_total', 'Total number of errors', ['error_type'])

class ExecutionEngine:
    """注文・約定実行エンジン"""
    
    def __init__(self, config_dir: Path = CONFIG_DIR, test_mode: bool = False):
        """初期化
        
        Args:
            config_dir: 設定ファイルディレクトリ
            test_mode: True の場合はテストネット使用
        """
        self.config_dir = config_dir
        self.test_mode = test_mode
        self.api_config = self._load_api_config()
        self.fees_config = self._load_fees_config()
        self.signals_dir = SIGNALS_DIR
        
        # Bybit API クライアント
        self.http_client = None
        self.ws_client = None
        
        if BYBIT_AVAILABLE:
            self._setup_bybit_client()
        else:
            raise ImportError("pybit がインストールされていないため、Bybit API を使用できません")
        
        # Slack クライアント
        self.slack_client = None
        if SLACK_AVAILABLE and 'slack_webhook' in self.api_config:
            self._setup_slack_client()
        
        # Prometheus HTTP サーバー
        if PROMETHEUS_AVAILABLE:
            try:
                start_http_server(9091)  # ポート9091でメトリクスを提供
                logger.info("Prometheus メトリックサーバーを開始しました (ポート 9091)")
            except Exception as e:
                logger.error(f"Prometheus メトリックサーバーの開始に失敗しました: {e}")
        
        # トレード状態
        self.current_position = 0.0  # 現在のポジションサイズ（BTC）
        self.position_value = 0.0   # ポジション価値（USD）
        self.nav = 100000.0         # 純資産価値（USD）- デフォルト: 10万ドル
        self.unrealized_pnl = 0.0   # 未実現損益（USD）
        self.daily_pnl = 0.0        # 日次損益（USD）
        self.daily_pnl_start = 0.0  # 日次損益の開始値
        self.day_start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # リスク管理パラメータ
        self.max_position_pct = 0.10  # NAVに対する最大ポジションサイズ（10%）
        self.max_daily_loss_pct = 0.03  # NAVに対する最大日次損失（3%）
        
        # 注文リスト
        self.orders = {}
        
        # 最新の信号
        self.latest_signal = {
            'timestamp': 0,
            'position': 0.0,
            'prediction': 0.5,
            'confidence': 0.0,
            'interval': '',
            'signal_type': 'neutral'
        }
        
        # 更新フラグ
        self.running = False
        self.update_account_info()
        
        logger.info(f"ExecutionEngine 初期化完了 (テストモード: {test_mode})")
    
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
    
    def _load_fees_config(self) -> Dict:
        """手数料設定を読み込む
        
        Returns:
            Dict: 手数料設定
        """
        try:
            with open(self.config_dir / 'fees.yaml', 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"手数料設定の読み込みに失敗しました: {e}")
            raise
    
    def _setup_bybit_client(self):
        """Bybit API クライアントを設定"""
        try:
            api_key = self.api_config.get('bybit', {}).get('key')
            api_secret = self.api_config.get('bybit', {}).get('secret')
            
            if not api_key or not api_secret:
                raise ValueError("Bybit API キーまたはシークレットが見つかりません")
            
            # テストネットかメインネットかを設定
            testnet = self.test_mode
            
            # HTTP クライアント
            self.http_client = HTTP(
                testnet=testnet,
                api_key=api_key,
                api_secret=api_secret
            )
            
            # WebSocket クライアント
            self.ws_client = WebSocket(
                testnet=testnet,
                api_key=api_key,
                api_secret=api_secret,
                channel_type="private"
            )
            
            # WebSocket ハンドラの登録
            self._register_ws_handlers()
            
            logger.info(f"Bybit API クライアント初期化完了 (テストネット: {testnet})")
            
        except Exception as e:
            logger.error(f"Bybit API クライアント初期化に失敗しました: {e}")
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(error_type='api_init').inc()
            raise
    
    def _setup_slack_client(self):
        """Slack Webhook クライアントを設定"""
        if SLACK_AVAILABLE and 'slack_webhook' in self.api_config:
            webhook_url = self.api_config['slack_webhook']
            if webhook_url:
                self.slack_client = WebhookClient(webhook_url)
                logger.info("Slack Webhook クライアント初期化完了")
            else:
                self.slack_client = None
    
    def _register_ws_handlers(self):
        """WebSocket イベントハンドラを登録"""
        # ポジション更新ハンドラ
        self.ws_client.position_stream(self._position_handler)
        
        # 約定ハンドラ
        self.ws_client.execution_stream(self._execution_handler)
        
        # 注文ハンドラ
        self.ws_client.order_stream(self._order_handler)
        
        # ウォレット更新ハンドラ
        self.ws_client.wallet_stream(self._wallet_handler)
        
        logger.info("WebSocket ハンドラ登録完了")
    
    def _position_handler(self, message):
        """ポジション更新イベントのハンドラ"""
        try:
            if 'data' in message and message['data']:
                for pos_data in message['data']:
                    if pos_data.get('symbol') == 'BTCUSDT':
                        size = float(pos_data.get('size', 0))
                        side = pos_data.get('side', 'None')
                        position_value = float(pos_data.get('positionValue', 0))
                        unrealized_pnl = float(pos_data.get('unrealisedPnl', 0))
                        
                        # ポジションサイズの符号をサイドに合わせる
                        if side == 'Sell':
                            size = -size
                        
                        self.current_position = size
                        self.position_value = position_value
                        self.unrealized_pnl = unrealized_pnl
                        
                        # プロメテウスメトリクス更新
                        if PROMETHEUS_AVAILABLE:
                            POSITION_GAUGE.set(size)
                            POSITION_VALUE_GAUGE.set(position_value)
                            PNL_GAUGE.set(unrealized_pnl)
                        
                        logger.info(f"ポジション更新: サイズ={size} BTC, 価値={position_value} USD, 未実現PnL={unrealized_pnl} USD")
        except Exception as e:
            logger.error(f"ポジション更新処理中にエラー: {e}")
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(error_type='ws_position').inc()
    
    def _execution_handler(self, message):
        """約定イベントのハンドラ"""
        try:
            if 'data' in message and message['data']:
                for exec_data in message['data']:
                    if exec_data.get('symbol') == 'BTCUSDT':
                        order_id = exec_data.get('orderId')
                        order_type = exec_data.get('orderType')
                        side = exec_data.get('side')
                        price = float(exec_data.get('execPrice', 0))
                        qty = float(exec_data.get('execQty', 0))
                        fee = float(exec_data.get('execFee', 0))
                        
                        # 約定ログ
                        logger.info(f"約定: ID={order_id}, タイプ={order_type}, サイド={side}, 価格={price}, 数量={qty}, 手数料={fee}")
                        
                        # Slack通知
                        self.send_slack_notification(
                            f"【約定】{side} {qty} BTC @ {price} USD (手数料: {fee} USD)",
                            emoji=":white_check_mark:"
                        )
                        
                        # プロメテウスメトリクス更新
                        if PROMETHEUS_AVAILABLE:
                            TRADE_COUNTER.labels(direction=side.lower(), status='executed').inc()
                        
                        # 注文リストから削除
                        if order_id in self.orders:
                            self.orders.pop(order_id)
        except Exception as e:
            logger.error(f"約定処理中にエラー: {e}")
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(error_type='ws_execution').inc()
    
    def _order_handler(self, message):
        """注文イベントのハンドラ"""
        try:
            if 'data' in message and message['data']:
                for order_data in message['data']:
                    if order_data.get('symbol') == 'BTCUSDT':
                        order_id = order_data.get('orderId')
                        order_status = order_data.get('orderStatus')
                        
                        # 注文ステータス更新
                        logger.info(f"注文ステータス: ID={order_id}, ステータス={order_status}")
                        
                        # 注文完了/キャンセル時の処理
                        if order_status in ['Filled', 'Cancelled', 'Rejected']:
                            if order_id in self.orders:
                                order_info = self.orders.get(order_id)
                                # 注文の終了時間を記録
                                end_time = time.time()
                                if 'create_time' in order_info:
                                    latency = end_time - order_info['create_time']
                                    if PROMETHEUS_AVAILABLE:
                                        ORDER_LATENCY.observe(latency)
                                    logger.info(f"注文完了までの時間: {latency:.2f}秒")
                                
                                # 注文失敗時の通知
                                if order_status in ['Cancelled', 'Rejected']:
                                    self.send_slack_notification(
                                        f"【注文失敗】ID={order_id}, ステータス={order_status}",
                                        emoji=":x:"
                                    )
                                    if PROMETHEUS_AVAILABLE:
                                        TRADE_COUNTER.labels(
                                            direction=order_info.get('side', 'unknown').lower(), 
                                            status='failed'
                                        ).inc()
        except Exception as e:
            logger.error(f"注文処理中にエラー: {e}")
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(error_type='ws_order').inc()
    
    def _wallet_handler(self, message):
        """ウォレット更新イベントのハンドラ"""
        try:
            if 'data' in message and message['data']:
                for wallet_data in message['data']:
                    if wallet_data.get('accountType') == 'UNIFIED':
                        # アカウント情報の更新
                        balance = float(wallet_data.get('totalWalletBalance', 0))
                        equity = float(wallet_data.get('totalEquity', 0))
                        
                        # NAV更新
                        self.nav = equity
                        
                        # プロメテウスメトリクス更新
                        if PROMETHEUS_AVAILABLE:
                            NAV_GAUGE.set(equity)
                        
                        # 日次PnL更新
                        current_time = datetime.now()
                        if current_time.date() > self.day_start_time.date():
                            # 日付が変わった場合、日次PnLをリセット
                            self.day_start_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                            self.daily_pnl_start = equity
                            self.daily_pnl = 0
                        else:
                            # 日次PnL計算
                            if self.daily_pnl_start == 0:
                                self.daily_pnl_start = equity
                            self.daily_pnl = equity - self.daily_pnl_start
                        
                        # プロメテウスメトリクス更新
                        if PROMETHEUS_AVAILABLE:
                            DAILY_PNL_GAUGE.set(self.daily_pnl)
                        
                        logger.info(f"ウォレット更新: 残高={balance} USD, 純資産={equity} USD, 日次PnL={self.daily_pnl} USD")
        except Exception as e:
            logger.error(f"ウォレット処理中にエラー: {e}")
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(error_type='ws_wallet').inc()
    
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
                if PROMETHEUS_AVAILABLE:
                    ERROR_COUNTER.labels(error_type='slack_notification').inc()
    
    def update_account_info(self) -> Dict:
        """アカウント情報を更新
        
        Returns:
            Dict: アカウント情報
        """
        try:
            # アカウント情報の取得
            account_info = self.http_client.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
            
            if 'result' in account_info and 'list' in account_info['result']:
                for account in account_info['result']['list']:
                    # USDTの残高を確認
                    for coin in account['coin']:
                        if coin['coin'] == 'USDT':
                            wallet_balance = float(coin['walletBalance'])
                            equity = float(coin['equity'])
                            
                            # NAV更新
                            self.nav = equity
                            
                            # プロメテウスメトリクス更新
                            if PROMETHEUS_AVAILABLE:
                                NAV_GAUGE.set(equity)
                            
                            logger.info(f"アカウント情報更新: 残高={wallet_balance} USDT, 純資産={equity} USDT")
                            
                            return {
                                'wallet_balance': wallet_balance,
                                'equity': equity
                            }
            
            logger.warning("アカウント情報の取得に失敗しました")
            return {}
            
        except Exception as e:
            logger.error(f"アカウント情報更新中にエラー: {e}")
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(error_type='account_update').inc()
            return {}
    
    def update_position_info(self) -> Dict:
        """ポジション情報を更新
        
        Returns:
            Dict: ポジション情報
        """
        try:
            # ポジション情報の取得
            position_info = self.http_client.get_positions(
                category="linear",
                symbol="BTCUSDT"
            )
            
            if 'result' in position_info and 'list' in position_info['result']:
                positions = position_info['result']['list']
                if positions:
                    pos = positions[0]
                    size = float(pos['size'])
                    side = pos['side']
                    position_value = float(pos['positionValue'])
                    unrealized_pnl = float(pos['unrealisedPnl'])
                    
                    # ポジションサイズの符号をサイドに合わせる
                    if side == 'Sell':
                        size = -size
                    
                    self.current_position = size
                    self.position_value = position_value
                    self.unrealized_pnl = unrealized_pnl
                    
                    # プロメテウスメトリクス更新
                    if PROMETHEUS_AVAILABLE:
                        POSITION_GAUGE.set(size)
                        POSITION_VALUE_GAUGE.set(position_value)
                        PNL_GAUGE.set(unrealized_pnl)
                    
                    logger.info(f"ポジション情報更新: サイズ={size} BTC, 価値={position_value} USD, 未実現PnL={unrealized_pnl} USD")
                    
                    return {
                        'size': size,
                        'side': side,
                        'position_value': position_value,
                        'unrealized_pnl': unrealized_pnl
                    }
                else:
                    # ポジションなし
                    self.current_position = 0.0
                    self.position_value = 0.0
                    self.unrealized_pnl = 0.0
                    
                    # プロメテウスメトリクス更新
                    if PROMETHEUS_AVAILABLE:
                        POSITION_GAUGE.set(0)
                        POSITION_VALUE_GAUGE.set(0)
                        PNL_GAUGE.set(0)
                    
                    logger.info("ポジションなし")
                    
                    return {
                        'size': 0.0,
                        'side': 'None',
                        'position_value': 0.0,
                        'unrealized_pnl': 0.0
                    }
            
            logger.warning("ポジション情報の取得に失敗しました")
            return {}
            
        except Exception as e:
            logger.error(f"ポジション情報更新中にエラー: {e}")
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(error_type='position_update').inc()
            return {}
    
    def get_market_price(self) -> float:
        """現在の市場価格を取得
        
        Returns:
            float: BTC/USDT の現在価格
        """
        try:
            # ティッカー情報の取得
            ticker = self.http_client.get_tickers(
                category="linear",
                symbol="BTCUSDT"
            )
            
            if 'result' in ticker and 'list' in ticker['result']:
                tickers = ticker['result']['list']
                if tickers:
                    last_price = float(tickers[0]['lastPrice'])
                    return last_price
            
            logger.warning("市場価格の取得に失敗しました")
            return 0.0
            
        except Exception as e:
            logger.error(f"市場価格取得中にエラー: {e}")
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(error_type='price_update').inc()
            return 0.0
    
    def calculate_order_quantity(self, target_position: float) -> float:
        """注文数量を計算
        
        Args:
            target_position: 目標ポジションサイズ (-1.0〜1.0)
            
        Returns:
            float: 注文数量 (BTC)
        """
        # アカウント情報の更新
        account_info = self.update_account_info()
        
        # ポジション情報の更新
        position_info = self.update_position_info()
        
        # 現在の市場価格
        market_price = self.get_market_price()
        
        if not market_price or not account_info:
            logger.error("注文数量の計算に必要な情報が不足しています")
            return 0.0
        
        # NAV（純資産価値）
        nav = account_info.get('equity', self.nav)
        
        # 最大ポジションサイズ（BTC）
        max_position_btc = (nav * self.max_position_pct) / market_price
        
        # 目標ポジションサイズ（BTC）
        target_position_btc = max_position_btc * target_position
        
        # 現在のポジションサイズ（BTC）
        current_position_btc = self.current_position
        
        # 注文数量 = 目標ポジション - 現在のポジション
        order_quantity = target_position_btc - current_position_btc
        
        # 小数点以下6桁に丸める（Bybitの最小数量に合わせる）
        order_quantity = round(order_quantity, 6)
        
        logger.info(f"注文数量計算: 目標={target_position_btc} BTC, 現在={current_position_btc} BTC, 注文={order_quantity} BTC")
        
        return order_quantity
    
    def place_market_order(self, quantity: float) -> Dict:
        """マーケット注文を発注
        
        Args:
            quantity: 注文数量（正: 買い、負: 売り）
            
        Returns:
            Dict: 注文結果
        """
        if abs(quantity) < 0.000001:
            logger.info("注文数量が小さすぎるため、注文をスキップします")
            return {}
        
        # 注文の方向を決定
        side = "Buy" if quantity > 0 else "Sell"
        abs_quantity = abs(quantity)
        
        try:
            # 計測開始
            start_time = time.time()
            
            # 注文の実行
            order_result = self.http_client.place_order(
                category="linear",
                symbol="BTCUSDT",
                side=side,
                orderType="Market",
                qty=str(abs_quantity),
                timeInForce="IOC",  # Immediate or Cancel
                reduceOnly=False,
                closeOnTrigger=False
            )
            
            # 計測終了
            end_time = time.time()
            latency = end_time - start_time
            
            # プロメテウスメトリクス更新
            if PROMETHEUS_AVAILABLE:
                TRADE_LATENCY.observe(latency)
                TRADE_COUNTER.labels(direction=side.lower(), status='placed').inc()
            
            logger.info(f"マーケット注文: {side} {abs_quantity} BTC (レイテンシ: {latency:.3f}秒)")
            
            # 注文ID
            if 'result' in order_result and 'orderId' in order_result['result']:
                order_id = order_result['result']['orderId']
                # 注文情報を保存
                self.orders[order_id] = {
                    'create_time': start_time,
                    'side': side,
                    'quantity': abs_quantity,
                    'type': 'Market'
                }
                
                # Slack通知
                self.send_slack_notification(
                    f"【注文】{side} {abs_quantity} BTC @ Market",
                    emoji=":rocket:"
                )
            
            return order_result
            
        except Exception as e:
            logger.error(f"マーケット注文中にエラー: {e}", exc_info=True)
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(error_type='order_placement').inc()
            
            # Slack通知
            self.send_slack_notification(
                f"【注文エラー】{side} {abs_quantity} BTC @ Market\nエラー: {str(e)}",
                emoji=":x:"
            )
            
            return {}
    
    def check_risk_limits(self) -> bool:
        """リスク制限をチェック
        
        Returns:
            bool: リスク制限内の場合は True
        """
        # アカウント情報を更新
        account_info = self.update_account_info()
        
        if not account_info:
            logger.warning("アカウント情報の取得に失敗したため、リスクチェックをスキップします")
            return True
        
        # 日次PnLの確認
        if self.daily_pnl < -self.nav * self.max_daily_loss_pct:
            logger.warning(f"日次損失制限に達しました: {self.daily_pnl} USD (最大許容損失: {-self.nav * self.max_daily_loss_pct} USD)")
            
            # Slack通知
            self.send_slack_notification(
                f"⚠️ リスク警告 ⚠️\n日次損失制限に達しました: {self.daily_pnl:.2f} USD\n最大許容損失: {-self.nav * self.max_daily_loss_pct:.2f} USD\nポジションを閉じます",
                emoji=":warning:"
            )
            
            # ポジションをクローズ
            if self.current_position != 0:
                self.place_market_order(-self.current_position)
            
            return False
        
        # ポジションサイズの確認
        if abs(self.current_position) > 0 and self.nav > 0:
            position_ratio = abs(self.position_value) / self.nav
            if position_ratio > self.max_position_pct * 1.1:  # 10%マージンを追加
                logger.warning(f"ポジションサイズが制限を超えています: {position_ratio:.2%} (最大: {self.max_position_pct:.2%})")
                
                # Slack通知
                self.send_slack_notification(
                    f"⚠️ リスク警告 ⚠️\nポジションサイズが制限を超えています: {position_ratio:.2%}\n最大許容サイズ: {self.max_position_pct:.2%}\nポジションを調整します",
                    emoji=":warning:"
                )
                
                # ポジションを調整
                target_size = math.copysign(self.max_position_pct * self.nav / self.get_market_price(), self.current_position)
                order_quantity = target_size - self.current_position
                self.place_market_order(order_quantity)
                
                return False
        
        return True
    
    def get_latest_signal(self, interval: str = '2h') -> Dict:
        """最新のシグナルを取得
        
        Args:
            interval: 時間枠
            
        Returns:
            Dict: シグナル情報
        """
        try:
            # シグナルファイルのパス
            signal_file = self.signals_dir / f"btc_{interval}.json"
            
            if not os.path.exists(signal_file):
                logger.warning(f"シグナルファイルが見つかりません: {signal_file}")
                return {}
            
            # ファイルの最終更新時刻を確認
            file_mtime = os.path.getmtime(signal_file)
            current_time = time.time()
            
            # 最終更新から5分以上経過している場合は古いと判断
            if current_time - file_mtime > 300:
                logger.warning(f"シグナルファイルが古いです: {signal_file} (最終更新: {datetime.fromtimestamp(file_mtime)})")
            
            # シグナルファイルを読み込む
            with open(signal_file, 'r') as f:
                signal = json.load(f)
            
            return signal
            
        except Exception as e:
            logger.error(f"シグナル取得中にエラー: {e}")
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(error_type='signal_read').inc()
            return {}
    
    def process_signal(self, signal: Dict) -> bool:
        """シグナルを処理
        
        Args:
            signal: シグナル情報
            
        Returns:
            bool: 処理成功の場合は True
        """
        if not signal or 'position' not in signal:
            logger.warning("有効なシグナルがありません")
            return False
        
        # シグナルのタイムスタンプをチェック
        if 'timestamp' in signal:
            signal_time = signal['timestamp']
            current_time = time.time()
            
            # シグナルが古すぎる場合（10分以上前）
            if current_time - signal_time > 600:
                logger.warning(f"シグナルが古すぎます: {datetime.fromtimestamp(signal_time)}")
                return False
        
        # 最新のシグナルと同じ場合はスキップ
        if (self.latest_signal.get('position') == signal.get('position') and
            self.latest_signal.get('signal_type') == signal.get('signal_type')):
            return False
        
        # シグナル情報をログに記録
        logger.info(f"新しいシグナル: ポジション={signal['position']}, タイプ={signal.get('signal_type', 'unknown')}")
        
        # リスク制限をチェック
        if not self.check_risk_limits():
            logger.warning("リスク制限のため、シグナルの処理をスキップします")
            return False
        
        # 目標ポジションサイズ
        target_position = float(signal['position'])
        
        # 注文数量を計算
        order_quantity = self.calculate_order_quantity(target_position)
        
        # 注文が必要な場合
        if abs(order_quantity) > 0.000001:
            # マーケット注文を発注
            order_result = self.place_market_order(order_quantity)
            
            if order_result and 'result' in order_result and 'orderId' in order_result['result']:
                # 最新のシグナルを更新
                self.latest_signal = signal.copy()
                return True
        else:
            logger.info("注文数量が小さすぎるため、注文をスキップします")
            # 最新のシグナルを更新
            self.latest_signal = signal.copy()
            return True
        
        return False
    
    async def execution_loop(self, interval: str = '2h', check_interval: int = 60):
        """実行ループ
        
        Args:
            interval: 時間枠
            check_interval: チェック間隔（秒）
        """
        self.running = True
        logger.info(f"実行ループ開始: 時間枠={interval}, チェック間隔={check_interval}秒")
        
        # Slack通知
        self.send_slack_notification(
            f"🚀 ML Bot 実行開始\n時間枠: {interval}\nテストモード: {self.test_mode}",
            emoji=":robot_face:"
        )
        
        last_signal_check = 0
        last_account_check = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # 定期的なシグナルチェック
                if current_time - last_signal_check > check_interval:
                    # 最新のシグナルを取得
                    signal = self.get_latest_signal(interval)
                    
                    if signal:
                        # シグナル処理
                        self.process_signal(signal)
                    
                    last_signal_check = current_time
                
                # 定期的なアカウント情報更新 (5分ごと)
                if current_time - last_account_check > 300:
                    self.update_account_info()
                    self.update_position_info()
                    last_account_check = current_time
                
                # 待機
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"実行ループでエラーが発生しました: {e}", exc_info=True)
                if PROMETHEUS_AVAILABLE:
                    ERROR_COUNTER.labels(error_type='execution_loop').inc()
                
                # 再試行のための短い待機
                await asyncio.sleep(5)
        
        logger.info("実行ループ終了")
    
    def stop(self):
        """実行を停止"""
        self.running = False
        logger.info("停止リクエストを受け付けました")

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='ML Bot 売買実行エンジン')
    parser.add_argument('--test', action='store_true', help='テストモード（テストネット使用）')
    parser.add_argument('--interval', type=str, default='2h', help='使用する時間枠 (例: 15m, 2h)')
    parser.add_argument('--check-interval', type=int, default=60, help='シグナルチェック間隔（秒）')
    parser.add_argument('--max-position', type=float, default=0.1, help='NAVに対する最大ポジションサイズ（割合）')
    parser.add_argument('--max-daily-loss', type=float, default=0.03, help='NAVに対する最大日次損失（割合）')
    return parser.parse_args()

async def main():
    """メイン関数"""
    try:
        # 引数の解析
        args = parse_args()
        
        # 実行エンジンの初期化
        engine = ExecutionEngine(test_mode=args.test)
        
        # リスク管理パラメータを設定
        engine.max_position_pct = args.max_position
        engine.max_daily_loss_pct = args.max_daily_loss
        
        # シグナルハンドラの設定
        def signal_handler(sig, frame):
            print('Ctrl+C が押されました。終了します...')
            engine.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 実行ループの開始
        await engine.execution_loop(
            interval=args.interval,
            check_interval=args.check_interval
        )
        
        return 0
    
    except Exception as e:
        logger.error(f"実行エラー: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    import math  # 必要なライブラリの追加インポート
    asyncio.run(main())
