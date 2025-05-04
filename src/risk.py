#!/usr/bin/env python
"""
risk.py - リスク計測と監視モジュール

機能:
- ポジションサイズとPnLの監視
- リスクメトリクスの計算と記録
- Prometheus エクスポーター (メトリクスの提供)
- Grafana ダッシュボード用 JSON の生成
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
from typing import Dict, List, Optional, Tuple, Any, Union
import traceback
import socket
import threading

# データ処理ライブラリ
import pandas as pd
import numpy as np

# Bybit API (ポジション情報の取得)
try:
    from pybit.unified_trading import HTTP
    BYBIT_AVAILABLE = True
except ImportError:
    BYBIT_AVAILABLE = False
    logging.warning("pybit がインストールされていません。Bybit API は無効です。")

# Prometheus メトリクス
try:
    import prometheus_client
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, 
        start_http_server, push_to_gateway
    )
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
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'risk.log'), mode='a')
    ]
)
logger = logging.getLogger('risk')

# プロジェクトルートディレクトリ
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = ROOT_DIR / 'config'
REPORTS_DIR = ROOT_DIR / 'reports'

# ログディレクトリ作成
os.makedirs(ROOT_DIR / 'logs', exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Prometheus メトリクス定義
if PROMETHEUS_AVAILABLE:
    # 基本メトリクス
    POSITION_GAUGE = Gauge('mlbot_position_size', 'Current position size in BTC')
    POSITION_VALUE_GAUGE = Gauge('mlbot_position_value_usd', 'Current position value in USD')
    NAV_GAUGE = Gauge('mlbot_nav_usd', 'Net Asset Value in USD')
    PNL_GAUGE = Gauge('mlbot_pnl_usd', 'Unrealized PnL in USD')
    DAILY_PNL_GAUGE = Gauge('mlbot_daily_pnl_usd', 'Daily PnL in USD')
    
    # リスクメトリクス
    POSITION_PCT_GAUGE = Gauge('mlbot_position_pct', 'Position size as percentage of NAV')
    DRAWDOWN_GAUGE = Gauge('mlbot_drawdown_pct', 'Current drawdown percentage')
    VAR_GAUGE = Gauge('mlbot_var_usd', 'Value at Risk (95% confidence) in USD')
    EXPECTED_SHORTFALL_GAUGE = Gauge('mlbot_expected_shortfall_usd', 'Expected Shortfall (95%) in USD')
    SHARPE_GAUGE = Gauge('mlbot_sharpe_ratio', 'Sharpe ratio (15-day rolling)')
    SORTINO_GAUGE = Gauge('mlbot_sortino_ratio', 'Sortino ratio (15-day rolling)')
    
    # パフォーマンスメトリクス
    LATENCY_GAUGE = Gauge('mlbot_latency_ms', 'API request latency in milliseconds')
    LATENCY_HISTOGRAM = Histogram('mlbot_request_latency_ms', 'API request latency histogram', 
                                 buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000])
    ERROR_COUNTER = Counter('mlbot_errors_total', 'Total number of errors', ['error_type'])
    HEARTBEAT_GAUGE = Gauge('mlbot_heartbeat_timestamp', 'Last heartbeat timestamp')

class RiskMonitor:
    """リスク監視クラス"""
    
    def __init__(self, config_dir: Path = CONFIG_DIR, test_mode: bool = False):
        """初期化
        
        Args:
            config_dir: 設定ファイルディレクトリ
            test_mode: True の場合はテストネット使用
        """
        self.config_dir = config_dir
        self.test_mode = test_mode
        self.api_config = self._load_api_config()
        self.reports_dir = REPORTS_DIR
        
        # Bybit API クライアント
        self.http_client = None
        
        if BYBIT_AVAILABLE:
            self._setup_bybit_client()
        
        # Slack クライアント
        self.slack_client = None
        if SLACK_AVAILABLE and 'slack_webhook' in self.api_config:
            self._setup_slack_client()
        
        # リスク管理パラメータ
        self.max_position_pct = 0.10  # NAVに対する最大ポジションサイズ（10%）
        self.max_daily_loss_pct = 0.03  # NAVに対する最大日次損失（3%）
        self.max_drawdown_pct = 0.25  # 最大許容ドローダウン（25%）
        
        # トレード状態
        self.current_position = 0.0  # 現在のポジションサイズ（BTC）
        self.position_value = 0.0   # ポジション価値（USD）
        self.nav = 100000.0         # 純資産価値（USD）- デフォルト: 10万ドル
        self.nav_peak = self.nav    # NAVのピーク値
        self.unrealized_pnl = 0.0   # 未実現損益（USD）
        self.daily_pnl = 0.0        # 日次損益（USD）
        self.daily_pnl_start = 0.0  # 日次損益の開始値
        self.drawdown = 0.0         # 現在のドローダウン（%）
        self.day_start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # リスク計測用の履歴
        self.nav_history = []       # NAV履歴
        self.daily_returns = []     # 日次リターン履歴
        
        # 更新フラグ
        self.running = False
        self.update_account_info()
        
        # Prometheus HTTP サーバー
        if PROMETHEUS_AVAILABLE:
            try:
                start_http_server(9092)  # ポート9092でメトリクスを提供
                logger.info("Prometheus メトリックサーバーを開始しました (ポート 9092)")
            except Exception as e:
                logger.error(f"Prometheus メトリックサーバーの開始に失敗しました: {e}")
        
        logger.info(f"RiskMonitor 初期化完了 (テストモード: {test_mode})")
    
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
        if not BYBIT_AVAILABLE or not self.http_client:
            logger.warning("Bybit API が利用できないため、アカウント情報を更新できません")
            return {}
        
        try:
            start_time = time.time()
            
            # アカウント情報の取得
            account_info = self.http_client.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
            
            # レイテンシ計測
            latency_ms = (time.time() - start_time) * 1000
            if PROMETHEUS_AVAILABLE:
                LATENCY_GAUGE.set(latency_ms)
                LATENCY_HISTOGRAM.observe(latency_ms)
            
            if 'result' in account_info and 'list' in account_info['result']:
                for account in account_info['result']['list']:
                    # USDTの残高を確認
                    for coin in account['coin']:
                        if coin['coin'] == 'USDT':
                            wallet_balance = float(coin['walletBalance'])
                            equity = float(coin['equity'])
                            
                            # NAV更新
                            self.nav = equity
                            
                            # ピークNAV更新
                            if equity > self.nav_peak:
                                self.nav_peak = equity
                            
                            # ドローダウン計算
                            if self.nav_peak > 0:
                                self.drawdown = (self.nav_peak - self.nav) / self.nav_peak * 100
                            else:
                                self.drawdown = 0.0
                            
                            # 日次PnL更新
                            current_time = datetime.now()
                            if current_time.date() > self.day_start_time.date():
                                # 日付が変わった場合、日次リターンを記録して日次PnLをリセット
                                if self.daily_pnl_start > 0:
                                    daily_return = (self.nav - self.daily_pnl_start) / self.daily_pnl_start
                                    self.daily_returns.append(daily_return)
                                    # 直近15日分のみ保持
                                    if len(self.daily_returns) > 15:
                                        self.daily_returns = self.daily_returns[-15:]
                                
                                self.day_start_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                                self.daily_pnl_start = equity
                                self.daily_pnl = 0
                            else:
                                # 日次PnL計算
                                if self.daily_pnl_start == 0:
                                    self.daily_pnl_start = equity
                                self.daily_pnl = equity - self.daily_pnl_start
                            
                            # NAV履歴を更新 (一日に一度)
                            if not self.nav_history or (current_time - datetime.fromtimestamp(self.nav_history[-1][0])).days >= 1:
                                self.nav_history.append((time.time(), equity))
                                # 直近30日分のみ保持
                                if len(self.nav_history) > 30:
                                    self.nav_history = self.nav_history[-30:]
                            
                            # Prometheusメトリクス更新
                            if PROMETHEUS_AVAILABLE:
                                NAV_GAUGE.set(equity)
                                DAILY_PNL_GAUGE.set(self.daily_pnl)
                                DRAWDOWN_GAUGE.set(self.drawdown)
                                # リスクメトリクスも更新
                                self.update_risk_metrics()
                            
                            logger.info(f"アカウント情報更新: 残高={wallet_balance} USDT, 純資産={equity} USDT, 日次PnL={self.daily_pnl} USD, DD={self.drawdown:.2f}%")
                            
                            return {
                                'wallet_balance': wallet_balance,
                                'equity': equity,
                                'drawdown': self.drawdown,
                                'daily_pnl': self.daily_pnl
                            }
            
            logger.warning("アカウント情報の取得に失敗しました")
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(error_type='account_fetch').inc()
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
        if not BYBIT_AVAILABLE or not self.http_client:
            logger.warning("Bybit API が利用できないため、ポジション情報を更新できません")
            return {}
        
        try:
            start_time = time.time()
            
            # ポジション情報の取得
            position_info = self.http_client.get_positions(
                category="linear",
                symbol="BTCUSDT"
            )
            
            # レイテンシ計測
            latency_ms = (time.time() - start_time) * 1000
            if PROMETHEUS_AVAILABLE:
                LATENCY_GAUGE.set(latency_ms)
                LATENCY_HISTOGRAM.observe(latency_ms)
            
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
                    
                    # ポジションの対NAV比率の計算
                    position_pct = 0.0
                    if self.nav > 0:
                        position_pct = abs(position_value) / self.nav * 100
                    
                    # Prometheusメトリクス更新
                    if PROMETHEUS_AVAILABLE:
                        POSITION_GAUGE.set(size)
                        POSITION_VALUE_GAUGE.set(position_value)
                        PNL_GAUGE.set(unrealized_pnl)
                        POSITION_PCT_GAUGE.set(position_pct)
                    
                    logger.info(f"ポジション情報更新: サイズ={size} BTC ({position_pct:.2f}%), 価値={position_value} USD, 未実現PnL={unrealized_pnl} USD")
                    
                    return {
                        'size': size,
                        'side': side,
                        'position_value': position_value,
                        'position_pct': position_pct,
                        'unrealized_pnl': unrealized_pnl
                    }
                else:
                    # ポジションなし
                    self.current_position = 0.0
                    self.position_value = 0.0
                    self.unrealized_pnl = 0.0
                    
                    # Prometheusメトリクス更新
                    if PROMETHEUS_AVAILABLE:
                        POSITION_GAUGE.set(0)
                        POSITION_VALUE_GAUGE.set(0)
                        PNL_GAUGE.set(0)
                        POSITION_PCT_GAUGE.set(0)
                    
                    logger.info("ポジションなし")
                    
                    return {
                        'size': 0.0,
                        'side': 'None',
                        'position_value': 0.0,
                        'position_pct': 0.0,
                        'unrealized_pnl': 0.0
                    }
            
            logger.warning("ポジション情報の取得に失敗しました")
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(error_type='position_fetch').inc()
            return {}
            
        except Exception as e:
            logger.error(f"ポジション情報更新中にエラー: {e}")
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(error_type='position_update').inc()
            return {}
    
    def update_risk_metrics(self):
        """リスクメトリクスを更新"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            # シャープレシオとソルティノレシオの計算 (直近15日間のデータを使用)
            if len(self.daily_returns) >= 5:
                # シャープレシオ計算
                returns_array = np.array(self.daily_returns)
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                
                risk_free_rate = 0.03 / 365  # 年利3%の日次換算
                
                if std_return > 0:
                    sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(252)  # 年間換算
                    SHARPE_GAUGE.set(sharpe_ratio)
                
                # ソルティノレシオ計算
                negative_returns = returns_array[returns_array < 0]
                if len(negative_returns) > 0:
                    downside_deviation = np.std(negative_returns)
                    if downside_deviation > 0:
                        sortino_ratio = (mean_return - risk_free_rate) / downside_deviation * np.sqrt(252)
                        SORTINO_GAUGE.set(sortino_ratio)
            
            # VaR (Value at Risk) の計算 (95%信頼区間)
            if len(self.daily_returns) >= 10:
                returns_array = np.array(self.daily_returns)
                var_95 = np.percentile(returns_array, 5)  # 下位5%
                var_usd = var_95 * self.nav
                VAR_GAUGE.set(-var_usd)  # 負の値で保存（損失額）
                
                # 期待ショートフォール (Expected Shortfall) の計算
                returns_below_var = returns_array[returns_array <= var_95]
                if len(returns_below_var) > 0:
                    expected_shortfall = np.mean(returns_below_var)
                    es_usd = expected_shortfall * self.nav
                    EXPECTED_SHORTFALL_GAUGE.set(-es_usd)  # 負の値で保存（損失額）
        
        except Exception as e:
            logger.error(f"リスクメトリクス更新中にエラー: {e}")
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(error_type='risk_metrics').inc()
    
    def check_risk_limits(self) -> bool:
        """リスク制限をチェック
        
        Returns:
            bool: リスク制限内の場合は True
        """
        warnings = []
        
        # 日次PnLの確認
        if self.daily_pnl < -self.nav * self.max_daily_loss_pct:
            warning = f"日次損失制限に達しました: {self.daily_pnl:.2f} USD (最大許容損失: {-self.nav * self.max_daily_loss_pct:.2f} USD)"
            warnings.append(warning)
            logger.warning(warning)
        
        # ドローダウンの確認
        if self.drawdown > self.max_drawdown_pct * 100:
            warning = f"最大ドローダウン制限に達しました: {self.drawdown:.2f}% (最大許容ドローダウン: {self.max_drawdown_pct * 100:.2f}%)"
            warnings.append(warning)
            logger.warning(warning)
        
        # ポジションサイズの確認
        position_pct = 0.0
        if self.nav > 0:
            position_pct = abs(self.position_value) / self.nav * 100
            
        if position_pct > self.max_position_pct * 100 * 1.1:  # 10%マージンを追加
            warning = f"ポジションサイズ制限を超えています: {position_pct:.2f}% (最大許容サイズ: {self.max_position_pct * 100:.2f}%)"
            warnings.append(warning)
            logger.warning(warning)
        
        # 警告がある場合は Slack 通知
        if warnings and self.slack_client:
            message = "⚠️ リスク警告 ⚠️\n" + "\n".join(warnings)
            self.send_slack_notification(message, emoji=":warning:")
            return False
        
        return True
    
    def generate_risk_report(self) -> Dict:
        """リスクレポートを生成
        
        Returns:
            Dict: リスクレポート
        """
        # アカウント情報とポジション情報を更新
        account_info = self.update_account_info()
        position_info = self.update_position_info()
        
        # リスクメトリクスを更新
        self.update_risk_metrics()
        
        # レポート作成
        report = {
            'timestamp': datetime.now().isoformat(),
            'nav': self.nav,
            'drawdown_pct': self.drawdown,
            'daily_pnl': self.daily_pnl,
            'position': {
                'size': self.current_position,
                'value_usd': self.position_value,
                'pct_of_nav': position_info.get('position_pct', 0.0),
                'unrealized_pnl': self.unrealized_pnl
            },
            'risk_metrics': {
                'max_position_pct': self.max_position_pct * 100,
                'max_daily_loss_pct': self.max_daily_loss_pct * 100,
                'max_drawdown_pct': self.max_drawdown_pct * 100
            },
            'status': {
                'limits_exceeded': not self.check_risk_limits()
            }
        }
        
        # 特定のリスクメトリクスが計算されている場合は追加
        if PROMETHEUS_AVAILABLE:
            for metric, gauge in [
                ('sharpe_ratio', SHARPE_GAUGE),
                ('sortino_ratio', SORTINO_GAUGE),
                ('var_95_usd', VAR_GAUGE),
                ('expected_shortfall_usd', EXPECTED_SHORTFALL_GAUGE)
            ]:
                try:
                    value = gauge._value.get()
                    if value is not None:
                        if 'advanced_metrics' not in report:
                            report['advanced_metrics'] = {}
                        report['advanced_metrics'][metric] = value
                except:
                    pass
        
        return report
    
    def save_risk_report(self, report: Dict):
        """リスクレポートを保存
        
        Args:
            report: リスクレポート
        """
        try:
            # 日付を含むファイル名
            date_str = datetime.now().strftime("%Y%m%d")
            report_file = self.reports_dir / f"risk_report_{date_str}.json"
            
            # 既存レポートがあれば読み込んで追記
            if os.path.exists(report_file):
                with open(report_file, 'r') as f:
                    try:
                        existing_reports = json.load(f)
                    except json.JSONDecodeError:
                        existing_reports = []
                
                if not isinstance(existing_reports, list):
                    existing_reports = [existing_reports]
            else:
                existing_reports = []
            
            # 新しいレポートを追加
            existing_reports.append(report)
            
            # 保存
            with open(report_file, 'w') as f:
                json.dump(existing_reports, f, indent=2)
            
            logger.info(f"リスクレポートを保存しました: {report_file}")
            
        except Exception as e:
            logger.error(f"リスクレポート保存中にエラー: {e}")
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(error_type='report_save').inc()
    
    def generate_grafana_dashboard(self) -> Dict:
        """Grafana ダッシュボード設定を生成
        
        Returns:
            Dict: Grafana ダッシュボード設定
        """
        # ダッシュボード設定
        dashboard = {
            "annotations": {
                "list": [
                    {
                        "builtIn": 1,
                        "datasource": "-- Grafana --",
                        "enable": True,
                        "hide": True,
                        "iconColor": "rgba(0, 211, 255, 1)",
                        "name": "Annotations & Alerts",
                        "type": "dashboard"
                    }
                ]
            },
            "editable": True,
            "gnetId": None,
            "graphTooltip": 0,
            "id": None,
            "links": [],
            "panels": [
                # NAV パネル
                {
                    "aliasColors": {},
                    "bars": False,
                    "dashLength": 10,
                    "dashes": False,
                    "datasource": "Prometheus",
                    "fieldConfig": {
                        "defaults": {},
                        "overrides": []
                    },
                    "fill": 1,
                    "fillGradient": 0,
                    "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 0,
                        "y": 0
                    },
                    "hiddenSeries": False,
                    "id": 1,
                    "legend": {
                        "avg": False,
                        "current": True,
                        "max": True,
                        "min": True,
                        "show": True,
                        "total": False,
                        "values": True
                    },
                    "lines": True,
                    "linewidth": 1,
                    "nullPointMode": "null",
                    "options": {
                        "alertThreshold": True
                    },
                    "percentage": False,
                    "pointradius": 2,
                    "points": False,
                    "renderer": "flot",
                    "seriesOverrides": [],
                    "spaceLength": 10,
                    "stack": False,
                    "steppedLine": False,
                    "targets": [
                        {
                            "expr": "mlbot_nav_usd",
                            "interval": "",
                            "legendFormat": "NAV",
                            "refId": "A"
                        }
                    ],
                    "thresholds": [],
                    "timeFrom": None,
                    "timeRegions": [],
                    "timeShift": None,
                    "title": "資産推移 (USD)",
                    "tooltip": {
                        "shared": True,
                        "sort": 0,
                        "value_type": "individual"
                    },
                    "type": "graph",
                    "xaxis": {
                        "buckets": None,
                        "mode": "time",
                        "name": None,
                        "show": True,
                        "values": []
                    },
                    "yaxes": [
                        {
                            "format": "currencyUSD",
                            "label": None,
                            "logBase": 1,
                            "max": None,
                            "min": None,
                            "show": True
                        },
                        {
                            "format": "short",
                            "label": None,
                            "logBase": 1,
                            "max": None,
                            "min": None,
                            "show": True
                        }
                    ],
                    "yaxis": {
                        "align": False,
                        "alignLevel": None
                    }
                },
                # ポジションパネル
                {
                    "aliasColors": {},
                    "bars": False,
                    "dashLength": 10,
                    "dashes": False,
                    "datasource": "Prometheus",
                    "fieldConfig": {
                        "defaults": {},
                        "overrides": []
                    },
                    "fill": 1,
                    "fillGradient": 0,
                    "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 12,
                        "y": 0
                    },
                    "hiddenSeries": False,
                    "id": 2,
                    "legend": {
                        "avg": False,
                        "current": True,
                        "max": True,
                        "min": True,
                        "show": True,
                        "total": False,
                        "values": True
                    },
                    "lines": True,
                    "linewidth": 1,
                    "nullPointMode": "null",
                    "options": {
                        "alertThreshold": True
                    },
                    "percentage": False,
                    "pointradius": 2,
                    "points": False,
                    "renderer": "flot",
                    "seriesOverrides": [],
                    "spaceLength": 10,
                    "stack": False,
                    "steppedLine": False,
                    "targets": [
                        {
                            "expr": "mlbot_position_size",
                            "interval": "",
                            "legendFormat": "Position (BTC)",
                            "refId": "A"
                        },
                        {
                            "expr": "mlbot_position_pct",
                            "interval": "",
                            "legendFormat": "Position (%)",
                            "refId": "B"
                        }
                    ],
                    "thresholds": [],
                    "timeFrom": None,
                    "timeRegions": [],
                    "timeShift": None,
                    "title": "ポジション",
                    "tooltip": {
                        "shared": True,
                        "sort": 0,
                        "value_type": "individual"
                    },
                    "type": "graph",
                    "xaxis": {
                        "buckets": None,
                        "mode": "time",
                        "name": None,
                        "show": True,
                        "values": []
                    },
                    "yaxes": [
                        {
                            "format": "short",
                            "label": None,
                            "logBase": 1,
                            "max": None,
                            "min": None,
                            "show": True
                        },
                        {
                            "format": "percent",
                            "label": None,
                            "logBase": 1,
                            "max": None,
                            "min": None,
                            "show": True
                        }
                    ],
                    "yaxis": {
                        "align": False,
                        "alignLevel": None
                    }
                },
                # PnL パネル
                {
                    "aliasColors": {},
                    "bars": False,
                    "dashLength": 10,
                    "dashes": False,
                    "datasource": "Prometheus",
                    "fieldConfig": {
                        "defaults": {},
                        "overrides": []
                    },
                    "fill": 1,
                    "fillGradient": 0,
                    "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 0,
                        "y": 8
                    },
                    "hiddenSeries": False,
                    "id": 3,
                    "legend": {
                        "avg": False,
                        "current": True,
                        "max": True,
                        "min": True,
                        "show": True,
                        "total": False,
                        "values": True
                    },
                    "lines": True,
                    "linewidth": 1,
                    "nullPointMode": "null",
                    "options": {
                        "alertThreshold": True
                    },
                    "percentage": False,
                    "pointradius": 2,
                    "points": False,
                    "renderer": "flot",
                    "seriesOverrides": [],
                    "spaceLength": 10,
                    "stack": False,
                    "steppedLine": False,
                    "targets": [
                        {
                            "expr": "mlbot_pnl_usd",
                            "interval": "",
                            "legendFormat": "未実現 P&L",
                            "refId": "A"
                        },
                        {
                            "expr": "mlbot_daily_pnl_usd",
                            "interval": "",
                            "legendFormat": "日次 P&L",
                            "refId": "B"
                        }
                    ],
                    "thresholds": [],
                    "timeFrom": None,
                    "timeRegions": [],
                    "timeShift": None,
                    "title": "損益 (USD)",
                    "tooltip": {
                        "shared": True,
                        "sort": 0,
                        "value_type": "individual"
                    },
                    "type": "graph",
                    "xaxis": {
                        "buckets": None,
                        "mode": "time",
                        "name": None,
                        "show": True,
                        "values": []
                    },
                    "yaxes": [
                        {
                            "format": "currencyUSD",
                            "label": None,
                            "logBase": 1,
                            "max": None,
                            "min": None,
                            "show": True
                        },
                        {
                            "format": "short",
                            "label": None,
                            "logBase": 1,
                            "max": None,
                            "min": None,
                            "show": True
                        }
                    ],
                    "yaxis": {
                        "align": False,
                        "alignLevel": None
                    }
                },
                # リスクメトリクスパネル
                {
                    "aliasColors": {},
                    "bars": False,
                    "dashLength": 10,
                    "dashes": False,
                    "datasource": "Prometheus",
                    "fieldConfig": {
                        "defaults": {},
                        "overrides": []
                    },
                    "fill": 1,
                    "fillGradient": 0,
                    "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 12,
                        "y": 8
                    },
                    "hiddenSeries": False,
                    "id": 4,
                    "legend": {
                        "avg": False,
                        "current": True,
                        "max": True,
                        "min": True,
                        "show": True,
                        "total": False,
                        "values": True
                    },
                    "lines": True,
                    "linewidth": 1,
                    "nullPointMode": "null",
                    "options": {
                        "alertThreshold": True
                    },
                    "percentage": False,
                    "pointradius": 2,
                    "points": False,
                    "renderer": "flot",
                    "seriesOverrides": [],
                    "spaceLength": 10,
                    "stack": False,
                    "steppedLine": False,
                    "targets": [
                        {
                            "expr": "mlbot_drawdown_pct",
                            "interval": "",
                            "legendFormat": "ドローダウン (%)",
                            "refId": "A"
                        },
                        {
                            "expr": "mlbot_sharpe_ratio",
                            "interval": "",
                            "legendFormat": "シャープレシオ",
                            "refId": "B"
                        }
                    ],
                    "thresholds": [],
                    "timeFrom": None,
                    "timeRegions": [],
                    "timeShift": None,
                    "title": "リスクメトリクス",
                    "tooltip": {
                        "shared": True,
                        "sort": 0,
                        "value_type": "individual"
                    },
                    "type": "graph",
                    "xaxis": {
                        "buckets": None,
                        "mode": "time",
                        "name": None,
                        "show": True,
                        "values": []
                    },
                    "yaxes": [
                        {
                            "format": "percent",
                            "label": None,
                            "logBase": 1,
                            "max": None,
                            "min": None,
                            "show": True
                        },
                        {
                            "format": "short",
                            "label": None,
                            "logBase": 1,
                            "max": None,
                            "min": None,
                            "show": True
                        }
                    ],
                    "yaxis": {
                        "align": False,
                        "alignLevel": None
                    }
                }
            ],
            "refresh": "10s",
            "schemaVersion": 27,
            "style": "dark",
            "tags": ["mlbot", "trading"],
            "time": {
                "from": "now-24h",
                "to": "now"
            },
            "timepicker": {
                "refresh_intervals": [
                    "5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"
                ]
            },
            "timezone": "browser",
            "title": "ML Bot モニタリングダッシュボード",
            "uid": "mlbot-risk",
            "version": 1
        }
        
        return dashboard
    
    def save_grafana_dashboard(self, dashboard: Dict):
        """Grafana ダッシュボード設定を保存
        
        Args:
            dashboard: ダッシュボード設定
        """
        try:
            # 保存先ファイル
            dashboard_file = self.reports_dir / 'grafana_dashboard.json'
            
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard, f, indent=2)
            
            logger.info(f"Grafana ダッシュボード設定を保存しました: {dashboard_file}")
            
        except Exception as e:
            logger.error(f"Grafana ダッシュボード保存中にエラー: {e}")
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(error_type='dashboard_save').inc()
    
    async def monitoring_loop(self, interval_seconds: int = 60):
        """モニタリングループ
        
        Args:
            interval_seconds: チェック間隔（秒）
        """
        self.running = True
        logger.info(f"モニタリングループ開始: チェック間隔={interval_seconds}秒")
        
        # Grafana ダッシュボード生成と保存
        dashboard = self.generate_grafana_dashboard()
        self.save_grafana_dashboard(dashboard)
        
        # Slack通知
        self.send_slack_notification(
            "🔍 リスクモニタリング開始\nGrafana ダッシュボードを生成しました",
            emoji=":bar_chart:"
        )
        
        report_interval = 6 * 60 * 60  # 6時間ごとにレポート保存
        last_report_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # アカウント情報とポジション情報を更新
                self.update_account_info()
                self.update_position_info()
                
                # リスク制限をチェック
                self.check_risk_limits()
                
                # ハートビートの更新
                if PROMETHEUS_AVAILABLE:
                    HEARTBEAT_GAUGE.set(current_time)
                
                # 定期的なレポート保存
                if current_time - last_report_time > report_interval:
                    report = self.generate_risk_report()
                    self.save_risk_report(report)
                    last_report_time = current_time
                
                # 待機
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"モニタリングループでエラーが発生しました: {e}", exc_info=True)
                if PROMETHEUS_AVAILABLE:
                    ERROR_COUNTER.labels(error_type='monitoring_loop').inc()
                
                # エラー時は短時間待機して再試行
                await asyncio.sleep(5)
        
        logger.info("モニタリングループ終了")
    
    def stop(self):
        """監視を停止"""
        self.running = False
        logger.info("停止リクエストを受け付けました")

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='ML Bot リスク監視')
    parser.add_argument('--test', action='store_true', help='テストモード（テストネット使用）')
    parser.add_argument('--interval', type=int, default=60, help='チェック間隔（秒）')
    parser.add_argument('--max-position', type=float, default=0.1, help='NAVに対する最大ポジションサイズ（割合）')
    parser.add_argument('--max-daily-loss', type=float, default=0.03, help='NAVに対する最大日次損失（割合）')
    parser.add_argument('--max-drawdown', type=float, default=0.25, help='最大許容ドローダウン（割合）')
    return parser.parse_args()

async def main():
    """メイン関数"""
    try:
        # 引数の解析
        args = parse_args()
        
        # リスクモニターの初期化
        monitor = RiskMonitor(test_mode=args.test)
        
        # リスク管理パラメータを設定
        monitor.max_position_pct = args.max_position
        monitor.max_daily_loss_pct = args.max_daily_loss
        monitor.max_drawdown_pct = args.max_drawdown
        
        # シグナルハンドラの設定
        def signal_handler(sig, frame):
            print('Ctrl+C が押されました。終了します...')
            monitor.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # モニタリングループの開始
        await monitor.monitoring_loop(interval_seconds=args.interval)
        
        return 0
    
    except Exception as e:
        logger.error(f"実行エラー: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    asyncio.run(main())
