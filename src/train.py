#!/usr/bin/env python
"""
train.py - モデル学習モジュール

機能:
- 各時間枠の特徴量と目的変数を読み込み
- Walk-forward 分割による時系列交差検証
- LightGBM, CatBoost, Transformer による予測モデル構築
- Optuna による最適化
- 複数時間枠のモデルを組み合わせたメタモデル
"""

import os
import sys
import yaml
import json
import pickle
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from typing import Dict, List, Optional, Tuple, Any, Union

# 機械学習モデル
import lightgbm as lgb
import catboost as ctb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import TimeSeriesSplit

# PyTorch (Transformer モデル用)
try:
    import torch
    import pytorch_forecasting
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch または pytorch_forecasting が見つかりません。Transformer モデルは無効になります。")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'train.log'), mode='a')
    ]
)
logger = logging.getLogger('train')

# プロジェクトルートディレクトリ
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = ROOT_DIR / 'config'
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'
REPORTS_DIR = ROOT_DIR / 'reports'

# ログディレクトリ作成
os.makedirs(ROOT_DIR / 'logs', exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 乱数シード固定
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(RANDOM_SEED)

class ModelTrainer:
    """モデル学習クラス"""
    
    def __init__(self, config_dir: Path = CONFIG_DIR):
        """初期化
        
        Args:
            config_dir: 設定ファイルディレクトリ
        """
        self.config_dir = config_dir
        self.intervals = self._load_intervals()
        self.model_dir = MODELS_DIR
        self.report_dir = REPORTS_DIR
        
        # デフォルト設定
        self.n_splits = 6  # Walk-forward 分割数
        self.train_months = 6  # 訓練期間（月）
        self.test_months = 1  # テスト期間（月）
        self.target_params = {
            'k': 3,  # 将来リターン期間
            'thr': 0.15  # シグナル閾値
        }
        
        # モデル種類
        self.model_types = ['lightgbm', 'catboost']
        if TORCH_AVAILABLE:
            self.model_types.append('transformer')
    
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
    
    def load_data(self, interval: str) -> Tuple[pd.DataFrame, pd.Series]:
        """特徴量と目的変数を読み込む
        
        Args:
            interval: 時間枠
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: 特徴量と目的変数
        """
        logger.info(f"データ読み込み開始: {interval}")
        
        # 特徴量読み込み
        features_path = DATA_DIR / 'features' / interval / 'X.parquet'
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"特徴量ファイルが見つかりません: {features_path}")
        
        X = pd.read_parquet(features_path)
        logger.info(f"特徴量読み込み完了: {features_path}, 形状: {X.shape}")
        
        # 目的変数読み込み
        target_path = DATA_DIR / 'features' / interval / f'y_k{self.target_params["k"]}_thr{self.target_params["thr"]:.2f}.parquet'
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"目的変数ファイルが見つかりません: {target_path}")
        
        y_df = pd.read_parquet(target_path)
        logger.info(f"目的変数読み込み完了: {target_path}, 形状: {y_df.shape}")
        
        # DataFrameをSeriesに変換 (y列を取得)
        y = y_df['y'] if 'y' in y_df.columns else y_df.iloc[:, 0]
        
        # インデックスが同じかチェック
        if not X.index.equals(y.index):
            logger.warning("特徴量と目的変数のインデックスが一致しません。共通部分のみ使用します。")
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            logger.info(f"インデックス調整後: X形状: {X.shape}, y形状: {y.shape}")
        
        # 欠損値の確認と処理
        if X.isna().any().any():
            logger.warning(f"特徴量に欠損値があります: {X.isna().sum().sum()} 個")
            X = X.fillna(0)
        
        if y.isna().any().any() if isinstance(y, pd.DataFrame) else y.isna().any():
            logger.warning(f"目的変数に欠損値があります: {y.isna().sum()} 個")
            # 欠損値のある行を削除
            valid_idx = ~y.isna()
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
            logger.info(f"欠損値削除後: X形状: {X.shape}, y形状: {y.shape}")
        
        return X, y
    
    def create_walk_forward_splits(self, X: pd.DataFrame, y: pd.Series, 
                                  n_splits: int = None, train_months: int = None, 
                                  test_months: int = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Walk-forward 分割による時系列交差検証セットを作成
        
        Args:
            X: 特徴量
            y: 目的変数
            n_splits: 分割数
            train_months: 訓練期間（月）
            test_months: テスト期間（月）
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: 訓練・テスト用インデックスのリスト
        """
        if n_splits is None:
            n_splits = self.n_splits
        if train_months is None:
            train_months = self.train_months
        if test_months is None:
            test_months = self.test_months
        
        logger.info(f"Walk-forward 分割作成: n_splits={n_splits}, train_months={train_months}, test_months={test_months}")
        
        # 日時インデックスに変換
        if not isinstance(X.index, pd.DatetimeIndex):
            logger.warning("インデックスが DatetimeIndex ではありません。変換します。")
            try:
                X.index = pd.to_datetime(X.index)
                y.index = pd.to_datetime(y.index)
            except Exception as e:
                logger.error(f"日時インデックスへの変換に失敗しました: {e}")
                raise
        
        # データの時間間隔を推定
        if len(X) > 1:
            # 連続するインデックス間の平均時間差（秒）
            avg_interval_seconds = (X.index[-1] - X.index[0]).total_seconds() / (len(X) - 1)
            # 時間単位に変換
            hours_per_bar = avg_interval_seconds / 3600
            logger.info(f"データ間隔: 約 {hours_per_bar:.2f} 時間/バー")
        else:
            logger.warning("データが1行しかないため、間隔を推定できません。")
            hours_per_bar = 1  # デフォルト値
        
        # 月ごとのバー数を推定
        # 実際のデータサイズに基づいて、より現実的な値に調整
        total_bars = len(X)
        bars_per_month = min(int((30 * 24) / hours_per_bar), total_bars // (train_months + test_months))
        
        # テストサイズが全体の20%を超えないようにする
        train_size = min(train_months * bars_per_month, int(total_bars * 0.5))
        test_size = min(test_months * bars_per_month, int(total_bars * 0.15))
        
        logger.info(f"推定バー数: 月あたり {bars_per_month} バー, 訓練 {train_size} バー, テスト {test_size} バー")
        logger.info(f"データ総数: {total_bars} バー")
        
        # 分割数も調整
        adjusted_n_splits = min(n_splits, total_bars // (train_size + test_size))
        # TimeSeriesSplit には少なくとも 2 分割が必要
        adjusted_n_splits = max(2, adjusted_n_splits)
        if adjusted_n_splits < n_splits:
            logger.warning(f"データサイズに基づいて分割数を {n_splits} から {adjusted_n_splits} に調整しました")
            n_splits = adjusted_n_splits
        
        # テストサイズが0より大きいことを確認
        if test_size <= 0:
            test_size = max(1, int(total_bars * 0.1))
            logger.warning(f"テストサイズが小さすぎるため、{test_size} に調整しました")
        
        # タイムシリーズ分割オブジェクト
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=0)
        
        # 最小訓練サイズの設定（データ量が少ない場合に調整）
        min_train_size = min(train_size, len(X) // (n_splits + 1))
        
        # 分割の生成と検証
        splits = []
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # 最小訓練サイズを確保
            if len(train_idx) < min_train_size and i > 0:
                # 前の分割のテスト期間を含める
                prev_split = splits[-1]
                prev_test_idx = prev_split[1]
                train_idx = np.concatenate([train_idx, prev_test_idx])
            
            # 訓練・テスト期間の確認
            train_start = X.index[train_idx[0]]
            train_end = X.index[train_idx[-1]]
            test_start = X.index[test_idx[0]]
            test_end = X.index[test_idx[-1]]
            
            train_days = (train_end - train_start).days
            test_days = (test_end - test_start).days
            
            logger.info(f"分割 {i+1}/{n_splits}: "
                       f"訓練 {train_start.date()} - {train_end.date()} ({train_days} 日, {len(train_idx)} バー), "
                       f"テスト {test_start.date()} - {test_end.date()} ({test_days} 日, {len(test_idx)} バー)")
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
                      params: Optional[Dict] = None) -> Tuple[lgb.Booster, Dict]:
        """LightGBM モデルを訓練
        
        Args:
            X_train: 訓練特徴量
            y_train: 訓練目的変数
            X_val: 検証特徴量
            y_val: 検証目的変数
            params: モデルパラメータ
            
        Returns:
            Tuple[lgb.Booster, Dict]: 訓練済みモデルとメトリクス
        """
        # ラベルの種類を確認
        unique_labels = np.unique(y_train)
        
        # ラベルのマッピング (-1, 0, 1) → (0, 1, 2)
        if set(unique_labels).issubset({-1, 0, 1}) and len(unique_labels) > 2:
            logger.info("3値ラベル (-1, 0, 1) を検出しました。LightGBM用に (0, 1, 2) に変換します。")
            label_map = {-1: 0, 0: 1, 1: 2}
            y_train_mapped = np.array([label_map[y] for y in y_train])
            y_val_mapped = np.array([label_map[y] for y in y_val])
            reverse_map = {0: -1, 1: 0, 2: 1}
            is_binary = False
            num_class = 3
        else:
            # 通常のバイナリ分類か判定
            is_binary = set(unique_labels).issubset({0, 1})
            y_train_mapped = y_train
            y_val_mapped = y_val
            num_class = 1 if is_binary else len(unique_labels)
        
        # デフォルトパラメータ
        default_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary' if is_binary else 'multiclass',
            'num_class': 1 if is_binary else len(np.unique(y_train)),
            'metric': 'binary_logloss' if is_binary else 'multi_logloss',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': RANDOM_SEED
        }
        
        if params is not None:
            default_params.update(params)
        
        # データセット作成
        train_data = lgb.Dataset(X_train, label=y_train_mapped)
        val_data = lgb.Dataset(X_val, label=y_val_mapped, reference=train_data)
        
        # モデル訓練
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100, show_stdv=True)
        ]
        
        model = lgb.train(
            default_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            num_boost_round=1000,
            callbacks=callbacks
        )
        
        # 検証セットで予測
        if is_binary:
            y_pred_proba = model.predict(X_val)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # 評価メトリクス
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='binary'),
                'recall': recall_score(y_val, y_pred, average='binary'),
                'f1': f1_score(y_val, y_pred, average='binary'),
                'mcc': matthews_corrcoef(y_val, y_pred),
                'auc': roc_auc_score(y_val, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
            }
        else:
            y_pred_proba = model.predict(X_val)
            y_pred = y_pred_proba.argmax(axis=1)
            
            # 3値分類の場合 (-1, 0, 1)、予測値を元の値に戻す
            if set(np.unique(y_train)).issubset({-1, 0, 1}) and len(np.unique(y_train)) > 2:
                # 予測を元のラベルにマッピング (0, 1, 2) → (-1, 0, 1)
                reverse_map = {0: -1, 1: 0, 2: 1}
                y_pred_original = np.array([reverse_map[y] for y in y_pred])
                
                # 評価メトリクス (元のラベルで評価)
                metrics = {
                    'accuracy': accuracy_score(y_val, y_pred_original),
                    'precision': precision_score(y_val, y_pred_original, average='macro'),
                    'recall': recall_score(y_val, y_pred_original, average='macro'),
                    'f1': f1_score(y_val, y_pred_original, average='macro'),
                    'confusion_matrix': confusion_matrix(y_val, y_pred_original).tolist()
                }
            else:
                # 通常のマルチクラス評価
                metrics = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred, average='macro'),
                    'recall': recall_score(y_val, y_pred, average='macro'),
                    'f1': f1_score(y_val, y_pred, average='macro'),
                    'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
                }
        
        return model, metrics
    
    def train_catboost(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
                      params: Optional[Dict] = None) -> Tuple[Union[ctb.CatBoostClassifier, ctb.CatBoostRegressor], Dict]:
        """CatBoost モデルを訓練
        
        Args:
            X_train: 訓練特徴量
            y_train: 訓練目的変数
            X_val: 検証特徴量
            y_val: 検証目的変数
            params: モデルパラメータ
            
        Returns:
            Tuple[Union[ctb.CatBoostClassifier, ctb.CatBoostRegressor], Dict]: 訓練済みモデルとメトリクス
        """
        # 分類か回帰か判定
        unique_values = np.unique(y_train)
        is_binary = set(unique_values).issubset({0, 1})
        is_multiclass = len(unique_values) > 2
        is_regression = not is_binary and not is_multiclass
        
        # デフォルトパラメータ
        default_params = {
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': RANDOM_SEED,
            'verbose': 100
        }
        
        if params is not None:
            default_params.update(params)
        
        # モデル作成
        if is_regression:
            model = ctb.CatBoostRegressor(**default_params)
            eval_metric = 'RMSE'
        elif is_binary:
            model = ctb.CatBoostClassifier(loss_function='Logloss', **default_params)
            eval_metric = 'AUC'
        else:  # multiclass
            model = ctb.CatBoostClassifier(loss_function='MultiClass', **default_params)
            eval_metric = 'Accuracy'
        
        # モデル訓練
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=100
        )
        
        # 検証セットで予測
        if is_regression:
            y_pred = model.predict(X_val)
            
            # 評価メトリクス
            metrics = {
                'rmse': mean_squared_error(y_val, y_pred, squared=False),
                'mae': mean_absolute_error(y_val, y_pred),
                'r2': r2_score(y_val, y_pred)
            }
        elif is_binary:
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # 評価メトリクス
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='binary'),
                'recall': recall_score(y_val, y_pred, average='binary'),
                'f1': f1_score(y_val, y_pred, average='binary'),
                'mcc': matthews_corrcoef(y_val, y_pred),
                'auc': roc_auc_score(y_val, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
            }
        else:  # multiclass
            y_pred_proba = model.predict_proba(X_val)
            y_pred = y_pred_proba.argmax(axis=1)
            
            # 評価メトリクス
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='macro'),
                'recall': recall_score(y_val, y_pred, average='macro'),
                'f1': f1_score(y_val, y_pred, average='macro'),
                'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
            }
        
        return model, metrics
    
    def train_transformer(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
                        feature_names: List[str], params: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """Temporal Fusion Transformer モデルを訓練
        
        Args:
            X_train: 訓練特徴量
            y_train: 訓練目的変数
            X_val: 検証特徴量
            y_val: 検証目的変数
            feature_names: 特徴量名のリスト
            params: モデルパラメータ
            
        Returns:
            Tuple[Any, Dict]: 訓練済みモデルとメトリクス
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch または pytorch_forecasting がインストールされていません。")
        
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import SMAPE, MAE
        from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
        from pytorch_lightning import Trainer
        import pytorch_lightning as pl
        
        # PyTorch Lightningの設定
        pl.seed_everything(RANDOM_SEED)
        
        # デフォルトパラメータ
        default_params = {
            'max_encoder_length': 24,
            'max_prediction_length': 1,
            'hidden_size': 64,
            'attention_head_size': 4,
            'dropout': 0.1,
            'hidden_continuous_size': 32,
            'learning_rate': 0.001,
            'batch_size': 128,
            'max_epochs': 100
        }
        
        if params is not None:
            default_params.update(params)
        
        # タイムシリーズデータセット用にデータ整形
        max_encoder_length = default_params['max_encoder_length']
        max_prediction_length = default_params['max_prediction_length']
        
        # インデックスがDateTime型であることを確認
        train_dates = pd.to_datetime(X_train.index)
        val_dates = pd.to_datetime(X_val.index)
        
        # 訓練データの準備
        train_df = pd.DataFrame(X_train, index=train_dates, columns=feature_names)
        train_df['target'] = y_train
        train_df['time_idx'] = np.arange(len(train_df))
        train_df['series'] = 'BTC'  # 単一時系列の場合
        
        # 検証データの準備
        val_df = pd.DataFrame(X_val, index=val_dates, columns=feature_names)
        val_df['target'] = y_val
        val_df['time_idx'] = np.arange(len(train_df), len(train_df) + len(val_df))
        val_df['series'] = 'BTC'
        
        # 結合データ
        data = pd.concat([train_df, val_df])
        
        # TimeSeriesDataSetの作成
        training_cutoff = train_df['time_idx'].max()
        
        # 特徴量リストの作成
        static_categoricals = []
        static_reals = []
        time_varying_known_categoricals = []
        time_varying_known_reals = []
        time_varying_unknown_categoricals = []
        time_varying_unknown_reals = feature_names
        
        # データセット作成
        training = TimeSeriesDataSet(
            data=data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="target",
            group_ids=["series"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(
                groups=["series"], 
                transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        # 検証データセットを同じ正規化で作成
        validation = TimeSeriesDataSet.from_dataset(
            training, 
            data, 
            min_prediction_idx=training_cutoff + 1,
            stop_randomization=True
        )
        
        # データローダー作成
        batch_size = default_params['batch_size']
        train_dataloader = training.to_dataloader(
            train=True, 
            batch_size=batch_size, 
            num_workers=0
        )
        val_dataloader = validation.to_dataloader(
            train=False, 
            batch_size=batch_size * 10, 
            num_workers=0
        )
        
        # モデルの作成
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=default_params['learning_rate'],
            hidden_size=default_params['hidden_size'],
            attention_head_size=default_params['attention_head_size'],
            dropout=default_params['dropout'],
            hidden_continuous_size=default_params['hidden_continuous_size'],
            loss=SMAPE(),
            optimizer="adamw",
            reduce_on_plateau_patience=5,
        )
        
        # 訓練
        early_stop_callback = EarlyStopping(
            monitor="val_loss", 
            min_delta=1e-4, 
            patience=10, 
            verbose=False, 
            mode="min"
        )
        lr_logger = LearningRateMonitor()
        
        trainer = Trainer(
            max_epochs=default_params['max_epochs'],
            accelerator="auto",
            gradient_clip_val=0.1,
            limit_train_batches=50,
            callbacks=[early_stop_callback, lr_logger],
            enable_checkpointing=True,
            logger=False
        )
        
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        # 検証セットで予測
        predictions = tft.predict(val_dataloader)
        actuals = torch.cat([y for x, y in iter(val_dataloader)])
        
        # numpy 配列に変換
        predictions_np = predictions.numpy()
        actuals_np = actuals.numpy()
        
        # 評価メトリクス
        metrics = {
            'mae': mean_absolute_error(actuals_np, predictions_np),
            'rmse': mean_squared_error(actuals_np, predictions_np, squared=False),
            'smape': np.mean(2 * np.abs(predictions_np - actuals_np) / (np.abs(predictions_np) + np.abs(actuals_np)))
        }
        
        return tft, metrics
    
    def optimize_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                         n_trials: int = 50) -> Dict:
        """LightGBM のハイパーパラメータを最適化
        
        Args:
            X_train: 訓練特徴量
            y_train: 訓練目的変数
            X_val: 検証特徴量
            y_val: 検証目的変数
            n_trials: 最適化の試行回数
            
        Returns:
            Dict: 最適化されたパラメータ
        """
        # バイナリ分類かどうか判定
        is_binary = set(np.unique(y_train)).issubset({0, 1})
        
        def objective(trial):
            params = {
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                'objective': 'binary' if is_binary else 'multiclass',
                'num_class': 1 if is_binary else len(np.unique(y_train)),
                'metric': 'binary_logloss' if is_binary else 'multi_logloss',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'random_state': RANDOM_SEED,
                'verbose': -1
            }
            
            # データセット作成
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # モデル訓練
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100, show_stdv=False)
            ]
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=callbacks
            )
            
            # メトリクスの評価
            if is_binary:
                y_pred_proba = model.predict(X_val)
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                # F1スコアとMCCの組み合わせを目標値として使用
                f1 = f1_score(y_val, y_pred, average='binary')
                mcc = matthews_corrcoef(y_val, y_pred)
                objective_score = (f1 + mcc) / 2
            else:
                y_pred_proba = model.predict(X_val)
                y_pred = y_pred_proba.argmax(axis=1)
                
                # マルチクラスの場合はF1 (macro) を使用
                objective_score = f1_score(y_val, y_pred, average='macro')
            
            return objective_score
        
        # 最適化実行
        logger.info(f"LightGBM ハイパーパラメータ最適化開始: {n_trials} 試行")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # 最適パラメータ
        best_params = study.best_params
        best_score = study.best_value
        
        # バイナリ分類用のパラメータを追加
        best_params['objective'] = 'binary' if is_binary else 'multiclass'
        best_params['num_class'] = 1 if is_binary else len(np.unique(y_train))
        best_params['random_state'] = RANDOM_SEED
        
        logger.info(f"最適化完了 - 最高スコア: {best_score:.4f}")
        logger.info(f"最適パラメータ: {best_params}")
        
        return best_params
    
    def optimize_catboost(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                         n_trials: int = 50) -> Dict:
        """CatBoost のハイパーパラメータを最適化
        
        Args:
            X_train: 訓練特徴量
            y_train: 訓練目的変数
            X_val: 検証特徴量
            y_val: 検証目的変数
            n_trials: 最適化の試行回数
            
        Returns:
            Dict: 最適化されたパラメータ
        """
        # 分類か回帰か判定
        unique_values = np.unique(y_train)
        is_binary = set(unique_values).issubset({0, 1})
        is_multiclass = len(unique_values) > 2
        is_regression = not is_binary and not is_multiclass
        
        def objective(trial):
            if is_regression:
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
                    'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'random_seed': RANDOM_SEED,
                    'verbose': 0
                }
                
                model = ctb.CatBoostRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=0
                )
                
                y_pred = model.predict(X_val)
                rmse = mean_squared_error(y_val, y_pred, squared=False)
                return -rmse  # 最小化なので負の値
                
            else:  # 分類
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
                    'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'random_seed': RANDOM_SEED,
                    'verbose': 0
                }
                
                if is_binary:
                    model = ctb.CatBoostClassifier(loss_function='Logloss', **params)
                else:  # multiclass
                    model = ctb.CatBoostClassifier(loss_function='MultiClass', **params)
                
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=0
                )
                
                if is_binary:
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    
                    # F1スコアとMCCの組み合わせを目標値として使用
                    f1 = f1_score(y_val, y_pred, average='binary')
                    mcc = matthews_corrcoef(y_val, y_pred)
                    objective_score = (f1 + mcc) / 2
                else:
                    y_pred_proba = model.predict_proba(X_val)
                    y_pred = y_pred_proba.argmax(axis=1)
                    
                    # マルチクラスの場合はF1 (macro) を使用
                    objective_score = f1_score(y_val, y_pred, average='macro')
                
                return objective_score
        
        # 最適化実行
        logger.info(f"CatBoost ハイパーパラメータ最適化開始: {n_trials} 試行")
        
        study = optuna.create_study(direction='maximize' if not is_regression else 'minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # 最適パラメータ
        best_params = study.best_params
        best_score = study.best_value
        
        # モデル種別に応じたパラメータを追加
        if is_regression:
            best_params['loss_function'] = 'RMSE'
        elif is_binary:
            best_params['loss_function'] = 'Logloss'
        else:  # multiclass
            best_params['loss_function'] = 'MultiClass'
        
        best_params['random_seed'] = RANDOM_SEED
        
        if is_regression:
            logger.info(f"最適化完了 - 最高スコア (RMSE): {-best_score:.4f}")
        else:
            logger.info(f"最適化完了 - 最高スコア: {best_score:.4f}")
        
        logger.info(f"最適パラメータ: {best_params}")
        
        return best_params
    
    def train_models_for_interval(self, interval: str, model_type: str = 'lightgbm', 
                                 optimize: bool = True, n_trials: int = 50) -> Dict:
        """特定時間枠のモデルを訓練
        
        Args:
            interval: 時間枠
            model_type: モデル種類 ('lightgbm', 'catboost', 'transformer')
            optimize: ハイパーパラメータ最適化を行うか
            n_trials: 最適化の試行回数
            
        Returns:
            Dict: 訓練結果
        """
        logger.info(f"時間枠 {interval} の {model_type} モデル訓練開始")
        
        # データ読み込み
        X, y = self.load_data(interval)
        
        # データ分割
        splits = self.create_walk_forward_splits(X, y)
        
        # モデル、メトリクス、特徴量重要度を保存
        fold_models = []
        fold_metrics = []
        fold_importances = []
        feature_names = X.columns.tolist()
        
        # 各分割で訓練
        for i, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"分割 {i+1}/{len(splits)} の訓練開始")
            
            # データ抽出
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            
            # NumPy 配列に変換
            X_train_np = X_train.values
            y_train_np = y_train.values
            X_test_np = X_test.values
            y_test_np = y_test.values
            
            # モデル訓練
            params = None
            
            # ハイパーパラメータ最適化
            if optimize:
                logger.info(f"ハイパーパラメータ最適化中...")
                
                # 訓練データを訓練/検証に分割
                train_size = int(len(X_train) * 0.8)
                X_train_opt, y_train_opt = X_train_np[:train_size], y_train_np[:train_size]
                X_val_opt, y_val_opt = X_train_np[train_size:], y_train_np[train_size:]
                
                if model_type == 'lightgbm':
                    params = self.optimize_lightgbm(X_train_opt, y_train_opt, X_val_opt, y_val_opt, n_trials)
                elif model_type == 'catboost':
                    params = self.optimize_catboost(X_train_opt, y_train_opt, X_val_opt, y_val_opt, n_trials)
            
            # モデル訓練
            if model_type == 'lightgbm':
                model, metrics = self.train_lightgbm(X_train_np, y_train_np, X_test_np, y_test_np, params)
                
                # 特徴量重要度
                importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importance(importance_type='gain')
                }).sort_values('importance', ascending=False)
                
                fold_importances.append(importance)
                
            elif model_type == 'catboost':
                model, metrics = self.train_catboost(X_train_np, y_train_np, X_test_np, y_test_np, params)
                
                # 特徴量重要度
                importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fold_importances.append(importance)
                
            elif model_type == 'transformer':
                # Transformer モデルは特別な処理が必要
                if TORCH_AVAILABLE:
                    model, metrics = self.train_transformer(X_train_np, y_train_np, X_test_np, y_test_np, feature_names, params)
                    
                    # 特徴量重要度 (Transformer では利用不可の場合がある)
                    importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': np.ones(len(feature_names))  # ダミー値
                    })
                    
                    fold_importances.append(importance)
                else:
                    logger.error("PyTorch が利用できないため Transformer モデルをスキップします")
                    continue
            
            fold_models.append(model)
            fold_metrics.append(metrics)
            
            # 主要メトリクスのログ出力
            if 'accuracy' in metrics:
                logger.info(f"分割 {i+1} の精度: {metrics['accuracy']:.4f}")
            if 'f1' in metrics:
                logger.info(f"分割 {i+1} の F1 スコア: {metrics['f1']:.4f}")
            if 'rmse' in metrics:
                logger.info(f"分割 {i+1} の RMSE: {metrics['rmse']:.4f}")
        
        # 全分割の結果集計
        combined_metrics = {}
        metric_keys = fold_metrics[0].keys() if fold_metrics else []
        
        for key in metric_keys:
            if key != 'confusion_matrix':
                values = [m[key] for m in fold_metrics]
                combined_metrics[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'values': [float(v) for v in values]
                }
        
        # 特徴量重要度の集計
        combined_importance = None
        if fold_importances:
            # すべての分割の重要度を集計
            all_importances = []
            for imp in fold_importances:
                imp_dict = dict(zip(imp['feature'], imp['importance']))
                all_importances.append(imp_dict)
            
            # 各特徴量の平均重要度を計算
            importance_df = pd.DataFrame(all_importances).fillna(0)
            mean_importance = importance_df.mean().sort_values(ascending=False)
            
            combined_importance = pd.DataFrame({
                'feature': mean_importance.index,
                'importance': mean_importance.values
            })
        
        # 結果保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.model_dir / interval / model_type
        report_dir = self.report_dir / interval / model_type
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        
        # メトリクス保存
        with open(report_dir / f"metrics_{timestamp}.json", 'w') as f:
            json.dump(combined_metrics, f, indent=4)
        
        # 特徴量重要度保存
        if combined_importance is not None:
            combined_importance.to_csv(report_dir / f"importance_{timestamp}.csv", index=False)
            
            # 重要度プロット
            plt.figure(figsize=(10, 8))
            top_n = min(20, len(combined_importance))
            combined_importance.head(top_n).plot.barh(x='feature', y='importance')
            plt.title(f'Top {top_n} Features')
            plt.tight_layout()
            plt.savefig(report_dir / f"importance_{timestamp}.png")
            plt.close()
        
        # モデル保存
        for i, model in enumerate(fold_models):
            if model_type == 'lightgbm':
                model.save_model(str(model_dir / f"model_fold{i+1}_{timestamp}.txt"))
            elif model_type == 'catboost':
                model.save_model(str(model_dir / f"model_fold{i+1}_{timestamp}.cbm"))
            elif model_type == 'transformer' and TORCH_AVAILABLE:
                # PyTorch モデルの保存
                torch.save(model.state_dict(), str(model_dir / f"model_fold{i+1}_{timestamp}.pt"))
        
        # 最終結果ログ
        logger.info(f"時間枠 {interval} の {model_type} モデル訓練完了")
        
        for key in combined_metrics:
            logger.info(f"平均 {key}: {combined_metrics[key]['mean']:.4f} ± {combined_metrics[key]['std']:.4f}")
        
        return {
            'models': fold_models,
            'metrics': combined_metrics,
            'importance': combined_importance,
            'timestamp': timestamp
        }
    
    def train_and_combine_models(self, model_type: str = 'lightgbm', optimize: bool = True, n_trials: int = 50) -> Dict:
        """すべての時間枠でモデルを訓練し、メタモデルを作成
        
        Args:
            model_type: モデル種類 ('lightgbm', 'catboost', 'transformer')
            optimize: ハイパーパラメータ最適化を行うか
            n_trials: 最適化の試行回数
            
        Returns:
            Dict: 訓練結果
        """
        logger.info(f"すべての時間枠での {model_type} モデル訓練開始")
        
        # 各時間枠のモデル訓練
        interval_results = {}
        
        for interval in self.intervals:
            try:
                result = self.train_models_for_interval(interval, model_type, optimize, n_trials)
                interval_results[interval] = result
            except Exception as e:
                logger.error(f"時間枠 {interval} の訓練中にエラー: {str(e)}")
                continue
        
        # メタモデル作成は Week 5 以降の実装
        
        return interval_results

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='ML モデル訓練')
    parser.add_argument('--interval', type=str, help='時間枠 (例: 15m, 2h)')
    parser.add_argument('--model_type', type=str, default='lightgbm', choices=['lightgbm', 'catboost', 'transformer'],
                       help='モデル種類')
    parser.add_argument('--optimize', action='store_true', help='ハイパーパラメータ最適化を行う')
    parser.add_argument('--trials', type=int, default=50, help='最適化の試行回数')
    parser.add_argument('--target_k', type=int, default=3, help='目的変数の将来期間')
    parser.add_argument('--target_thr', type=float, default=0.15, help='目的変数の閾値')
    parser.add_argument('--all', action='store_true', help='すべての時間枠で訓練')
    return parser.parse_args()

def main():
    """メイン関数"""
    try:
        # ログディレクトリ作成
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'), exist_ok=True)
        
        args = parse_args()
        trainer = ModelTrainer()
        
        # 目的変数パラメータを設定
        trainer.target_params = {
            'k': args.target_k,
            'thr': args.target_thr
        }
        
        if args.all:
            # すべての時間枠で訓練
            results = trainer.train_and_combine_models(args.model_type, args.optimize, args.trials)
        elif args.interval:
            # 単一時間枠で訓練
            results = trainer.train_models_for_interval(args.interval, args.model_type, args.optimize, args.trials)
        else:
            # デフォルトでは最初の時間枠を使用
            interval = trainer.intervals[0]
            logger.info(f"時間枠が指定されていないため、デフォルト値 {interval} を使用します")
            results = trainer.train_models_for_interval(interval, args.model_type, args.optimize, args.trials)
        
        return 0
    
    except Exception as e:
        logger.error(f"実行エラー: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
