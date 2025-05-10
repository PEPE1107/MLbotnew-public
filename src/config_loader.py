#!/usr/bin/env python
"""
config_loader.py - 設定ファイル管理モジュール

機能:
- YAMLファイルからの設定読み込み
- 設定の検証
- 設定の提供

使用例:
```
from src.config_loader import ConfigLoader
config = ConfigLoader()
fees_config = config.load('fees')
```
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# ロギング設定
logger = logging.getLogger(__name__)

# プロジェクトのルートディレクトリを特定
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
CONFIG_DIR = ROOT_DIR / 'config'


class ConfigLoader:
    """設定ファイル管理クラス"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """初期化
        
        Args:
            config_dir: 設定ファイルディレクトリ。未指定時はデフォルトを使用
        """
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
        self.config_cache = {}  # 読み込んだ設定のキャッシュ
        
        if not self.config_dir.exists():
            logger.warning(f"設定ディレクトリが存在しません: {self.config_dir}")
            os.makedirs(self.config_dir, exist_ok=True)
    
    def load(self, config_name: str, reload: bool = False) -> Dict[str, Any]:
        """設定ファイルを読み込む
        
        Args:
            config_name: 設定名（ファイル名から拡張子を除いたもの）
            reload: キャッシュを無視して再読み込みするかどうか
            
        Returns:
            Dict[str, Any]: 設定内容
        """
        # キャッシュから返す
        if config_name in self.config_cache and not reload:
            return self.config_cache[config_name]
        
        # ファイルパスを構築
        config_file = self.config_dir / f"{config_name}.yaml"
        
        # ファイルの存在チェック
        if not config_file.exists():
            logger.error(f"設定ファイルが見つかりません: {config_file}")
            return {}
        
        # YAMLファイルを読み込む
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            if config is None:
                logger.warning(f"空の設定ファイル: {config_file}")
                config = {}
                
            # 設定を検証
            self._validate_config(config_name, config)
            
            # キャッシュに保存
            self.config_cache[config_name] = config
            return config
            
        except Exception as e:
            logger.error(f"設定ファイル {config_file} の読み込みに失敗: {str(e)}")
            return {}
    
    def _validate_config(self, config_name: str, config: Dict[str, Any]) -> None:
        """設定内容を検証する
        
        Args:
            config_name: 設定名
            config: 設定内容
        """
        # 設定ファイルに応じた検証
        if config_name == 'fees':
            self._validate_fees_config(config)
        elif config_name == 'intervals':
            self._validate_intervals_config(config)
        elif config_name == 'model':
            self._validate_model_config(config)
        
        # 追加の検証はここに追加
    
    def _validate_fees_config(self, config: Dict[str, Any]) -> None:
        """手数料設定を検証する
        
        Args:
            config: 手数料設定
        """
        if 'fees' not in config:
            logger.warning("fees.yaml: 'fees'セクションが見つかりません")
        elif 'default' not in config['fees']:
            logger.warning("fees.yaml: 'fees.default'値が見つかりません")
        
        if 'slippage' not in config:
            logger.warning("fees.yaml: 'slippage'セクションが見つかりません")
        elif 'default' not in config['slippage']:
            logger.warning("fees.yaml: 'slippage.default'値が見つかりません")
    
    def _validate_intervals_config(self, config: Dict[str, Any]) -> None:
        """時間枠設定を検証する
        
        Args:
            config: 時間枠設定
        """
        if 'intervals' not in config:
            logger.warning("intervals.yaml: 'intervals'セクションが見つかりません")
        else:
            # 各時間枠の必須フィールドをチェック
            required_fields = ['name', 'minutes']
            for i, interval in enumerate(config['intervals']):
                for field in required_fields:
                    if field not in interval:
                        logger.warning(f"intervals.yaml: intervals[{i}]に'{field}'が見つかりません")
    
    def _validate_model_config(self, config: Dict[str, Any]) -> None:
        """モデル設定を検証する
        
        Args:
            config: モデル設定
        """
        if 'models' not in config:
            logger.warning("model.yaml: 'models'セクションが見つかりません")
        else:
            # 各モデルタイプの必須フィールドをチェック
            model_types = ['classifier', 'regressor']
            for model_type in model_types:
                if model_type not in config['models']:
                    logger.warning(f"model.yaml: 'models.{model_type}'セクションが見つかりません")
                elif 'type' not in config['models'][model_type]:
                    logger.warning(f"model.yaml: 'models.{model_type}.type'が見つかりません")
    
    def get_interval_config(self, interval_name: str) -> Dict[str, Any]:
        """指定した時間枠の設定を取得する
        
        Args:
            interval_name: 時間枠名 (例: '15m', '2h', '1d')
            
        Returns:
            Dict[str, Any]: 時間枠設定（見つからない場合は空辞書）
        """
        intervals_config = self.load('intervals')
        
        if 'intervals' not in intervals_config:
            return {}
        
        # 名前が一致する時間枠設定を探す
        for interval in intervals_config['intervals']:
            if interval.get('name') == interval_name:
                return interval
        
        return {}
    
    def save(self, config_name: str, config_data: Dict[str, Any]) -> bool:
        """設定をファイルに保存する
        
        Args:
            config_name: 設定名（ファイル名から拡張子を除いたもの）
            config_data: 保存する設定データ
            
        Returns:
            bool: 保存成功かどうか
        """
        config_file = self.config_dir / f"{config_name}.yaml"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            # キャッシュを更新
            self.config_cache[config_name] = config_data
            return True
            
        except Exception as e:
            logger.error(f"設定ファイル {config_file} の保存に失敗: {str(e)}")
            return False


# 便利な関数
def get_config(config_name: str, reload: bool = False) -> Dict[str, Any]:
    """設定を取得する便利関数
    
    Args:
        config_name: 設定名
        reload: 再読み込みするかどうか
        
    Returns:
        Dict[str, Any]: 設定内容
    """
    config_loader = ConfigLoader()
    return config_loader.load(config_name, reload)
