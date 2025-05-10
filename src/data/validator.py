#!/usr/bin/env python
"""
validator.py - データバリデーションモジュール

機能:
- 取得データの品質チェック
- 0値・NaN・UTC timezone ズレ・連続欠損の検出
- Slack と Prometheus へのアラート送信
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import yaml
import slack_sdk
from prometheus_client import Gauge, push_to_gateway

# プロジェクトのルートディレクトリをパスに追加
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, str(ROOT_DIR))

# 設定ディレクトリ
CONFIG_DIR = ROOT_DIR / 'config'
DATA_DIR = ROOT_DIR / 'data'

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(ROOT_DIR, 'logs', 'validator.log'), mode='a')
    ]
)
logger = logging.getLogger('data.validator')

# Prometheusメトリクス定義
PROMETHEUS_METRICS = {
    'cg_up': Gauge('cg_up', 'Coinglass API availability'),
    'cg_missing_rows': Gauge('cg_missing_rows', 'Number of missing rows in data', ['interval', 'endpoint']),
    'cg_zero_values': Gauge('cg_zero_values', 'Number of zero values in data', ['interval', 'endpoint']),
    'cg_nan_values': Gauge('cg_nan_values', 'Number of NaN values in data', ['interval', 'endpoint']),
    'cg_utc_drift': Gauge('cg_utc_drift', 'UTC timezone drift in minutes', ['interval', 'endpoint']),
    'cg_rate_limit_remaining': Gauge('cg_rate_limit_remaining', 'Remaining API rate limit')
}


class DataValidator:
    """データバリデーションクラス"""
    
    def __init__(self, config_dir: Path = CONFIG_DIR, data_dir: Path = DATA_DIR,
                slack_webhook: Optional[str] = None, prometheus_gateway: Optional[str] = None):
        """初期化

        Args:
            config_dir: 設定ディレクトリのパス
            data_dir: データディレクトリのパス
            slack_webhook: Slack webhook URL (任意)
            prometheus_gateway: Prometheus push gateway URL (任意)
        """
        self.config_dir = config_dir
        self.data_dir = data_dir
        self.slack_webhook = slack_webhook
        self.prometheus_gateway = prometheus_gateway
        self.slack_client = self._setup_slack_client() if slack_webhook else None
        
        # 設定読み込み
        self.intervals = self._load_intervals()
        self.endpoints = {
            'price': 'Price data',
            'oi': 'Open Interest',
            'funding': 'Funding rates',
            'liq': 'Liquidation data',
            'lsr': 'Long/Short ratio',
            'taker': 'Taker volume',
            'orderbook': 'Orderbook',
            'premium': 'Premium index'
        }
        
        logger.info(f"データバリデータ初期化: intervals={self.intervals}")
    
    def _load_intervals(self) -> List[str]:
        """時間枠設定読み込み

        Returns:
            List[str]: 時間枠リスト
        """
        try:
            with open(self.config_dir / 'intervals.yaml', 'r') as f:
                config = yaml.safe_load(f)
            return config['intervals']
        except Exception as e:
            logger.error(f"時間枠設定読み込みエラー: {e}")
            return ['15m', '2h', '1d']  # デフォルト値
    
    def _setup_slack_client(self) -> Optional[slack_sdk.WebhookClient]:
        """Slackクライアント設定

        Returns:
            Optional[slack_sdk.WebhookClient]: Slack webhookクライアント
        """
        if self.slack_webhook:
            return slack_sdk.WebhookClient(self.slack_webhook)
        return None
    
    def send_slack_alert(self, message: str, emoji: str = ":warning:"):
        """Slackアラート送信

        Args:
            message: 送信するメッセージ
            emoji: アイコン
        """
        if self.slack_client:
            try:
                response = self.slack_client.send(
                    text=f"{emoji} {message}"
                )
                if not response.status_code == 200:
                    logger.warning(f"Slack通知送信失敗: {response.status_code}, {response.body}")
            except Exception as e:
                logger.error(f"Slack通知エラー: {e}")
    
    def push_to_prometheus(self, job_name: str = 'data_validator'):
        """Prometheusメトリクス送信

        Args:
            job_name: ジョブ名
        """
        if not self.prometheus_gateway:
            return
            
        try:
            from prometheus_client import push_to_gateway, CollectorRegistry, REGISTRY
            
            # メトリクスを送信
            push_to_gateway(self.prometheus_gateway, job=job_name, registry=REGISTRY)
            logger.info(f"Prometheusメトリクス送信完了: {job_name}")
        except Exception as e:
            logger.error(f"Prometheusメトリクス送信エラー: {e}")
    
    def validate_dataframe(self, df: pd.DataFrame, interval: str, endpoint: str) -> Dict[str, Any]:
        """データフレーム検証

        Args:
            df: 検証するDataFrame
            interval: 時間枠
            endpoint: エンドポイント名

        Returns:
            Dict[str, Any]: 検証結果
        """
        result = {
            'missing_rows': 0,
            'zero_values': 0,
            'nan_values': 0,
            'utc_drift': 0,
            'issues': []
        }
        
        # エラーチェック
        if df is None or df.empty:
            result['issues'].append(f"Empty dataframe: {endpoint}, {interval}")
            result['missing_rows'] = 100  # 100%欠損とみなす
            return result
        
        # インデックスがタイムスタンプかどうかを確認
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                # タイムスタンプ列がある場合はそれをインデックスに設定
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp')
                elif 'ts' in df.columns:
                    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                    df = df.set_index('ts')
                else:
                    result['issues'].append(f"No timestamp column: {endpoint}, {interval}")
                    return result
            except Exception as e:
                result['issues'].append(f"Failed to convert timestamp: {endpoint}, {interval}, {e}")
                return result
        
        # 欠損行チェック
        expected_freq = None
        if interval == '15m':
            expected_freq = '15Min'
        elif interval == '2h':
            expected_freq = '2H'
        elif interval == '1d':
            expected_freq = 'D'
            
        if expected_freq:
            # 期待される完全なインデックス範囲を作成
            full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)
            missing_idx = full_idx.difference(df.index)
            result['missing_rows'] = len(missing_idx)
            
            if len(missing_idx) > 0:
                missing_percent = (len(missing_idx) / len(full_idx)) * 100
                if missing_percent > 5:  # 5%以上の欠損は警告
                    result['issues'].append(f"Missing rows: {endpoint}, {interval}, {missing_percent:.1f}% ({len(missing_idx)} rows)")
        
        # 0値チェック
        for col in df.columns:
            # 非数値列はスキップ
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                zero_percent = (zero_count / len(df)) * 100
                result['zero_values'] += zero_count
                
                if zero_percent > 10:  # 10%以上の0値は警告
                    result['issues'].append(f"Zero values: {endpoint}, {interval}, column {col}, {zero_percent:.1f}% ({zero_count} values)")
        
        # NaN値チェック
        for col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                nan_percent = (nan_count / len(df)) * 100
                result['nan_values'] += nan_count
                
                if nan_percent > 5:  # 5%以上のNaN値は警告
                    result['issues'].append(f"NaN values: {endpoint}, {interval}, column {col}, {nan_percent:.1f}% ({nan_count} values)")
        
        # UTCタイムゾーンズレチェック
        # タイムスタンプの分単位の値が期待値と一致するか確認
        expected_minutes = set()
        if interval == '15m':
            expected_minutes = {0, 15, 30, 45}
        elif interval == '2h':
            expected_minutes = {0}
        elif interval == '1d':
            expected_minutes = {0}
            
        if expected_minutes:
            actual_minutes = {ts.minute for ts in df.index}
            unexpected_minutes = actual_minutes - expected_minutes
            
            if unexpected_minutes:
                result['utc_drift'] = 1  # ズレあり
                result['issues'].append(f"UTC drift: {endpoint}, {interval}, unexpected minutes: {unexpected_minutes}")
        
        return result
    
    def validate_file(self, file_path: Path, interval: str, endpoint: str) -> Dict[str, Any]:
        """ファイル検証

        Args:
            file_path: 検証するファイルパス
            interval: 時間枠
            endpoint: エンドポイント名

        Returns:
            Dict[str, Any]: 検証結果
        """
        result = {
            'file_exists': False,
            'file_size': 0,
            'valid': False,
            'validation': None,
            'issues': []
        }
        
        # ファイル存在チェック
        if not file_path.exists():
            result['issues'].append(f"File not found: {file_path}")
            return result
            
        result['file_exists'] = True
        result['file_size'] = file_path.stat().st_size
        
        # サイズが小さすぎる場合
        if result['file_size'] < 100:
            result['issues'].append(f"File too small: {file_path}, {result['file_size']} bytes")
            return result
        
        # ファイル読み込み
        try:
            df = pd.read_parquet(file_path)
            validation_result = self.validate_dataframe(df, interval, endpoint)
            result['validation'] = validation_result
            result['issues'].extend(validation_result['issues'])
            result['valid'] = len(validation_result['issues']) == 0
            
            # Prometheusメトリクス更新
            PROMETHEUS_METRICS['cg_missing_rows'].labels(interval=interval, endpoint=endpoint).set(validation_result['missing_rows'])
            PROMETHEUS_METRICS['cg_zero_values'].labels(interval=interval, endpoint=endpoint).set(validation_result['zero_values'])
            PROMETHEUS_METRICS['cg_nan_values'].labels(interval=interval, endpoint=endpoint).set(validation_result['nan_values'])
            PROMETHEUS_METRICS['cg_utc_drift'].labels(interval=interval, endpoint=endpoint).set(validation_result['utc_drift'])
            
        except Exception as e:
            result['issues'].append(f"File read error: {file_path}, {str(e)}")
            
        return result
    
    def validate_data_dir(self, interval: str) -> Dict[str, Any]:
        """時間枠ディレクトリ内の全データ検証

        Args:
            interval: 時間枠

        Returns:
            Dict[str, Any]: 検証結果
        """
        results = {
            'interval': interval,
            'endpoints': {},
            'issues': [],
            'summary': {
                'total_files': 0,
                'valid_files': 0,
                'invalid_files': 0,
                'missing_files': 0
            }
        }
        
        # インターバルディレクトリ確認
        data_dir = self.data_dir / 'raw' / interval
        if not data_dir.exists():
            results['issues'].append(f"Interval directory not found: {data_dir}")
            results['summary']['missing_files'] = len(self.endpoints)
            return results
        
        # 各エンドポイントのファイル検証
        for endpoint in self.endpoints:
            file_path = data_dir / f"{endpoint}.parquet"
            results['summary']['total_files'] += 1
            
            validation_result = self.validate_file(file_path, interval, endpoint)
            results['endpoints'][endpoint] = validation_result
            
            if validation_result['file_exists']:
                if validation_result['valid']:
                    results['summary']['valid_files'] += 1
                else:
                    results['summary']['invalid_files'] += 1
                    results['issues'].extend(validation_result['issues'])
            else:
                results['summary']['missing_files'] += 1
                results['issues'].append(f"Missing file: {endpoint}.parquet")
        
        return results
    
    def validate_all_data(self) -> Dict[str, Any]:
        """すべての時間枠の全データ検証

        Returns:
            Dict[str, Any]: 検証結果
        """
        results = {
            'intervals': {},
            'issues': [],
            'summary': {
                'total_intervals': len(self.intervals),
                'valid_intervals': 0,
                'invalid_intervals': 0,
                'total_files': 0,
                'valid_files': 0,
                'invalid_files': 0,
                'missing_files': 0
            }
        }
        
        # APIアップフラグをデフォルトで設定
        PROMETHEUS_METRICS['cg_up'].set(1)
        
        # 各時間枠の検証
        for interval in self.intervals:
            interval_result = self.validate_data_dir(interval)
            results['intervals'][interval] = interval_result
            
            # サマリー集計
            results['summary']['total_files'] += interval_result['summary']['total_files']
            results['summary']['valid_files'] += interval_result['summary']['valid_files']
            results['summary']['invalid_files'] += interval_result['summary']['invalid_files']
            results['summary']['missing_files'] += interval_result['summary']['missing_files']
            
            if len(interval_result['issues']) == 0:
                results['summary']['valid_intervals'] += 1
            else:
                results['summary']['invalid_intervals'] += 1
                results['issues'].extend(interval_result['issues'])
        
        # 重大な問題があればSlack通知とPrometheusメトリクス更新
        if results['issues']:
            # 上位5件のみ表示（多すぎると読みにくい）
            top_issues = results['issues'][:5]
            
            # 残りの件数
            remaining = len(results['issues']) - len(top_issues)
            
            # Slack通知
            message = f"*データ検証結果 - 問題あり*\n" + \
                      f"時間枠: {results['summary']['valid_intervals']}/{results['summary']['total_intervals']} OK\n" + \
                      f"ファイル: {results['summary']['valid_files']}/{results['summary']['total_files']} OK, " + \
                      f"{results['summary']['missing_files']} missing, " + \
                      f"{results['summary']['invalid_files']} invalid\n\n" + \
                      "*問題点 (上位5件)*:\n" + \
                      "\n".join([f"• {issue}" for issue in top_issues])
                      
            if remaining > 0:
                message += f"\n\n...他 {remaining} 件の問題"
                
            self.send_slack_alert(message)
            
            # 深刻な問題がある場合はAPIダウンとみなす
            if results['summary']['missing_files'] > 0.5 * results['summary']['total_files']:
                PROMETHEUS_METRICS['cg_up'].set(0)
        
        # Prometheusメトリクス送信
        self.push_to_prometheus()
        
        return results


def main():
    # 環境変数から設定を読み込み
    slack_webhook = os.environ.get('SLACK_WEBHOOK')
    prometheus_gateway = os.environ.get('PROMETHEUS_GATEWAY', 'localhost:9091')
    
    # バリデータインスタンス作成
    validator = DataValidator(slack_webhook=slack_webhook, prometheus_gateway=prometheus_gateway)
    
    # データ検証実行
    results = validator.validate_all_data()
    
    # 結果ログ出力
    logger.info(f"データ検証完了: " + 
               f"時間枠 {results['summary']['valid_intervals']}/{results['summary']['total_intervals']} OK, " +
               f"ファイル {results['summary']['valid_files']}/{results['summary']['total_files']} OK")
    
    if results['issues']:
        logger.warning(f"検出された問題: {len(results['issues'])} 件")
        for issue in results['issues'][:10]:  # 最初の10件のみ表示
            logger.warning(f"  - {issue}")
        if len(results['issues']) > 10:
            logger.warning(f"  ... 他 {len(results['issues']) - 10} 件")


if __name__ == "__main__":
    main()
