"""
特徴量ビルダーモジュール
---------------------

異なる時間枠のデータから特徴量を構築するためのモジュール。
MTFロジックに基づいた特徴量エンジニアリングを行います。
"""

import pandas as pd
import numpy as np
import talib
import warnings
import re
from typing import Dict, List, Optional, Union, Tuple

from .resampler import TimeFrameResampler

class FeatureBuilder:
    """
    MTFロジックに基づいた特徴量を構築するクラス
    """
    
    def __init__(self, intervals_config: Dict, mtf_config: Dict):
        """
        初期化
        
        Parameters:
        -----------
        intervals_config : dict
            時間枠の設定情報の辞書
        mtf_config : dict
            MTFロジックの設定情報の辞書
        """
        self.intervals_config = intervals_config
        self.mtf_config = mtf_config
        self.resampler = TimeFrameResampler(intervals_config)
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基本的な特徴量を構築
        
        Parameters:
        -----------
        df : pandas.DataFrame
            元データフレーム（OHLCV）
            
        Returns:
        --------
        pandas.DataFrame
            特徴量が追加されたデータフレーム
        """
        # データのコピーを作成
        features = df.copy()
        
        # 基本的な価格データの列名を統一
        self._standardize_column_names(features)
        
        # 基本テクニカル指標の追加
        features = self._add_basic_indicators(features)
        
        # トレンド指標の追加
        features = self._add_trend_indicators(features)
        
        # ボラティリティ指標の追加
        features = self._add_volatility_indicators(features)
        
        # モメンタム指標の追加
        features = self._add_momentum_indicators(features)
        
        # ボリューム関連指標の追加
        if 'volume' in features.columns:
            features = self._add_volume_indicators(features)
        
        # Coinglassデータからの特徴量追加
        coinglass_cols = [col for col in features.columns if any(col.startswith(prefix) for prefix in 
                         ['oi_', 'funding_', 'liq_', 'lsr_', 'taker_', 'premium_', 'orderbook_'])]
        
        if coinglass_cols:
            features = self._build_coinglass_features(features)
        
        # NaNの除去
        features = features.dropna()
        
        return features
    
    def build_mtf_features(self, base_df: pd.DataFrame, *higher_dfs) -> pd.DataFrame:
        """
        複数の時間枠からMTF特徴量を構築
        
        Parameters:
        -----------
        base_df : pandas.DataFrame
            ベースとなる時間枠のデータフレーム
        *higher_dfs : tuple of pandas.DataFrame
            より上位の時間枠のデータフレーム
            
        Returns:
        --------
        pandas.DataFrame
            MTF特徴量が追加されたデータフレーム
        """
        # ベース時間枠の特徴量を構築
        features = self.build_features(base_df)
        
        # より上位の時間枠からの特徴量を追加
        for i, higher_df in enumerate(higher_dfs):
            # 上位時間枠の特徴量を構築
            higher_features = self.build_features(higher_df)
            
            # 上位時間枠のプレフィックスを設定
            prefix = f"h{i+1}_"
            
            # 上位時間枠の特徴量をリサンプリングしてベース時間枠に合わせる
            # まず、上位時間枠と下位時間枠のインターバルを特定
            if 'interval' in higher_df.attrs:
                higher_interval = higher_df.attrs['interval']
            else:
                # インターバルが明示的に指定されていない場合は推定
                # この例では、単純に1日、2時間、15分などの順番を前提とする
                if i == 0:
                    higher_interval = '1d'
                elif i == 1:
                    higher_interval = '2h'
                else:
                    higher_interval = '4h'
            
            if 'interval' in base_df.attrs:
                base_interval = base_df.attrs['interval']
            else:
                # ベースインターバルも同様に推定
                base_interval = '15m'
            
            # リサンプリング
            resampled_higher = self.resampler.resample(higher_features, higher_interval, base_interval)
            
            # カラム名にプレフィックスを追加して結合
            for col in resampled_higher.columns:
                if col not in ['open', 'high', 'low', 'close', 'volume']:  # 基本価格データは除外
                    features[prefix + col] = resampled_higher[col]
        
        # MTFロジックに基づく統合特徴量の追加
        features = self._add_mtf_combination_features(features)
        
        # NaNの除去
        features = features.dropna()
        
        return features
    
    def audit_features(self, df: pd.DataFrame, lookback_window: int = 10) -> List[str]:
        """
        特徴量の未来漏洩をチェック
        
        Parameters:
        -----------
        df : pandas.DataFrame
            チェック対象のデータフレーム
        lookback_window : int
            相関チェックのためのルックバックウィンドウ
            
        Returns:
        --------
        list
            未来漏洩の可能性があるカラムのリスト
        """
        return self.resampler.audit_features(df, lookback_window)
    
    def generate_quick_signals(self, features: pd.DataFrame) -> pd.Series:
        """
        クイックモード用の単純なシグナルを生成
        
        Parameters:
        -----------
        features : pandas.DataFrame
            特徴量データフレーム
            
        Returns:
        --------
        pandas.Series
            シグナル系列
        """
        # トレンドフラグ（1日足からのEMA200）
        trend_flag = None
        for col in features.columns:
            if 'trend_flag' in col:
                trend_flag = features[col]
                break
        
        # トレンドフラグがない場合はSMAを使用
        if trend_flag is None:
            sma200 = features['close'].rolling(200).mean()
            trend_flag = (features['close'] > sma200).astype(int)
        
        # RSIの計算（過買い・過売り判断用）
        rsi = None
        for col in features.columns:
            if 'rsi' in col.lower():
                rsi = features[col]
                break
        
        # RSIがない場合は計算
        if rsi is None:
            try:
                rsi = talib.RSI(features['close'], timeperiod=14)
            except:
                # タリブが使用できない場合は自前で計算
                delta = features['close'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
        
        # シグナル生成
        signals = pd.Series(0.0, index=features.index)
        
        # トレンドが上昇中（1）かつRSIが30以下でロング
        long_condition = (trend_flag == 1) & (rsi < 30)
        signals[long_condition] = 1.0
        
        # トレンドが下降中（0）かつRSIが70以上でショート
        short_condition = (trend_flag == 0) & (rsi > 70)
        signals[short_condition] = -1.0
        
        return signals
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        列名を標準化
        
        Parameters:
        -----------
        df : pandas.DataFrame
            列名を標準化するデータフレーム
            
        Returns:
        --------
        pandas.DataFrame
            列名が標準化されたデータフレーム
        """
        column_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Date': 'date',
            'Timestamp': 'timestamp',
            'Time': 'time'
        }
        
        df.rename(columns=column_map, inplace=True, errors='ignore')
        return df
    
    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基本的なテクニカル指標を追加
        
        Parameters:
        -----------
        df : pandas.DataFrame
            特徴量を追加するデータフレーム
            
        Returns:
        --------
        pandas.DataFrame
            特徴量が追加されたデータフレーム
        """
        # 移動平均
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # 指数移動平均
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # 価格とSMAの差
        df['close_sma_10_dist'] = (df['close'] - df['sma_10']) / df['sma_10']
        df['close_sma_20_dist'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        # 価格変化率
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        
        # その他の指標はTaLibで計算
        try:
            # MACD
            macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
            
            # RSI
            df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
            
            # ストキャスティクス
            df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'], 
                                                  fastk_period=14, slowk_period=3, slowk_matype=0, 
                                                  slowd_period=3, slowd_matype=0)
        except:
            # TaLibがインストールされていない場合は警告
            warnings.warn("TaLibが使用できないため、一部の指標は計算されません。")
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        トレンド関連の指標を追加
        
        Parameters:
        -----------
        df : pandas.DataFrame
            特徴量を追加するデータフレーム
            
        Returns:
        --------
        pandas.DataFrame
            特徴量が追加されたデータフレーム
        """
        # EMA傾斜（トレンドの強さを示す）
        df['ema_10_slope'] = df['ema_10'].pct_change(5)
        df['ema_20_slope'] = df['ema_20'].pct_change(10)
        
        # MA間のクロス（クロスの兆候を数値化）
        df['ema_10_20_ratio'] = df['ema_10'] / df['ema_20']
        df['ema_20_50_ratio'] = df['ema_20'] / df['ema_50']
        
        # トレンドフラグ（EMA200を基準としたトレンド判定）
        df['trend_flag'] = (df['close'] > df['ema_200']).astype(int)
        
        # 価格の高値/安値更新状況
        df['high_20d'] = df['high'].rolling(20).max()
        df['low_20d'] = df['low'].rolling(20).min()
        df['close_to_high_20d'] = (df['close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'])
        
        try:
            # ADX (Average Directional Index) - トレンドの強さを示す
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Aroon - トレンドの方向性と強さを示す
            df['aroon_up'], df['aroon_down'] = talib.AROON(df['high'], df['low'], timeperiod=14)
            df['aroon_osc'] = df['aroon_up'] - df['aroon_down']
            
            # Parabolic SAR - トレンド転換点を示す
            df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
            df['sar_dist'] = (df['close'] - df['sar']) / df['close']
        except:
            warnings.warn("TaLibが使用できないため、ADX等の指標は計算されません。")
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ボラティリティ関連の指標を追加
        
        Parameters:
        -----------
        df : pandas.DataFrame
            特徴量を追加するデータフレーム
            
        Returns:
        --------
        pandas.DataFrame
            特徴量が追加されたデータフレーム
        """
        # ロールボラティリティ（標準偏差）
        df['volatility_10'] = df['returns_1'].rolling(10).std()
        df['volatility_20'] = df['returns_1'].rolling(20).std()
        
        # True Range
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr_14'] = df['tr'].rolling(14).mean()
        
        # ボラティリティ比率
        df['vol_ratio_20_10'] = df['volatility_20'] / df['volatility_10']
        
        # 価格レンジに対する終値位置
        df['close_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        try:
            # ボリンジャーバンド
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            
            # バンド幅とバンド内での位置
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        except:
            warnings.warn("TaLibが使用できないため、ボリンジャーバンド等の指標は計算されません。")
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        モメンタム関連の指標を追加
        
        Parameters:
        -----------
        df : pandas.DataFrame
            特徴量を追加するデータフレーム
            
        Returns:
        --------
        pandas.DataFrame
            特徴量が追加されたデータフレーム
        """
        # モメンタム（単純な過去Nの変化率）
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        try:
            # ROC (Rate of Change)
            df['roc_10'] = talib.ROC(df['close'], timeperiod=10)
            
            # CCI (Commodity Channel Index)
            df['cci_14'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Williams %R
            df['willr_14'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        except:
            warnings.warn("TaLibが使用できないため、ROC等の指標は計算されません。")
        
        return df
    
    def _build_coinglass_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Coinglassデータから特徴量を生成
        
        Parameters:
        -----------
        df : pandas.DataFrame
            CoinglassデータをDf
            
        Returns:
        --------
        pandas.DataFrame
            特徴量が追加されたデータフレーム
        """
        if df is None or df.empty:
            return df
            
        # Coinglassのデータ列を特定
        prefix_pattern = re.compile(r'^(oi|funding|liq|lsr|taker|premium|orderbook)_')
        coinglass_columns = [col for col in df.columns if prefix_pattern.match(col)]
        
        if not coinglass_columns:
            return df  # Coinglassデータが見つからない場合はそのまま返す
        
        # 基本特徴量
        # ファンディングレート特徴量
        funding_cols = [col for col in df.columns if col.startswith('funding_')]
        if funding_cols:
            funding_rate_col = next((col for col in funding_cols if col.endswith('_c') or col.endswith('rate')), None)
            if funding_rate_col:
                # ファンディングレートの変化率
                df['funding_rate_change'] = df[funding_rate_col].pct_change()
                # ファンディングレートの移動平均
                df['funding_rate_ma7'] = df[funding_rate_col].rolling(7).mean()
                # ファンディングレートの偏差
                df['funding_rate_dev'] = df[funding_rate_col] - df['funding_rate_ma7']
                # ファンディングレートの絶対値（方向性に関係なく大きさを見る）
                df['funding_rate_abs'] = df[funding_rate_col].abs()
                # ファンディングレートの正負の継続期間
                df['funding_rate_sign'] = np.sign(df[funding_rate_col])
                df['funding_streak'] = df['funding_rate_sign'].groupby(
                    (df['funding_rate_sign'] != df['funding_rate_sign'].shift(1)).cumsum()).cumcount() + 1
        
        # オープンインタレスト特徴量
        oi_cols = [col for col in df.columns if col.startswith('oi_')]
        if oi_cols:
            oi_col = next((col for col in oi_cols if col.endswith('_c')), None)
            if oi_col:
                # OIの変化率
                df['oi_change'] = df[oi_col].pct_change()
                # OIの移動平均
                df['oi_ma7'] = df[oi_col].rolling(7).mean()
                # OIの相対的な位置
                df['oi_rel_position'] = df[oi_col] / df['oi_ma7'] - 1
                # OIの加速度（変化率の変化率）
                df['oi_acceleration'] = df['oi_change'].pct_change()
                # OIのボラティリティ
                df['oi_volatility'] = df['oi_change'].rolling(7).std()
        
        # 清算データ特徴量
        liq_cols = [col for col in df.columns if col.startswith('liq_')]
        if any('longLiquidationUsd' in col for col in liq_cols) and any('shortLiquidationUsd' in col for col in liq_cols):
            long_liq_col = next(col for col in liq_cols if 'longLiquidationUsd' in col)
            short_liq_col = next(col for col in liq_cols if 'shortLiquidationUsd' in col)
            
            # 総清算量
            df['total_liquidation'] = df[long_liq_col] + df[short_liq_col]
            # 清算比率（ロング/ショート）
            df['liquidation_ratio'] = df[long_liq_col] / (df[short_liq_col] + 1e-10)
            # 清算の移動平均
            df['liquidation_ma7'] = df['total_liquidation'].rolling(7).mean()
            # 清算のボラティリティ
            df['liquidation_volatility'] = df['total_liquidation'].rolling(7).std()
            # 清算のZ値（標準化）
            df['liquidation_zscore'] = (df['total_liquidation'] - df['liquidation_ma7']) / (df['liquidation_volatility'] + 1e-10)
            
            # 価格変動との相関
            if 'returns_1' in df.columns:
                # 清算と価格変動の関係性
                df['liquidation_price_correlation'] = df['total_liquidation'] * df['returns_1'].shift(-1)
                df['liquidation_signal'] = np.sign(df['liquidation_ratio'] - 1) * (df['liquidation_zscore'] > 2).astype(int)
        
        # ロング/ショート比率特徴量
        lsr_cols = [col for col in df.columns if col.startswith('lsr_')]
        if any('longShortRatio' in col for col in lsr_cols):
            lsr_col = next(col for col in lsr_cols if 'longShortRatio' in col)
            # LSRの移動平均
            df['lsr_ma7'] = df[lsr_col].rolling(7).mean()
            # LSRの偏差
            df['lsr_dev'] = df[lsr_col] - df['lsr_ma7']
            # LSRのボラティリティ
            df['lsr_volatility'] = df[lsr_col].rolling(7).std()
            # LSRのZ値（標準化）
            df['lsr_zscore'] = (df[lsr_col] - df['lsr_ma7']) / (df['lsr_volatility'] + 1e-10)
            # LSRの閾値シグナル（極端な値をシグナルとする）
            df['lsr_extreme'] = ((df['lsr_zscore'] > 2) | (df['lsr_zscore'] < -2)).astype(int)
            
            # LSRが1より大きいか小さいか（1 = ロング優勢、-1 = ショート優勢）
            df['lsr_direction'] = np.sign(df[lsr_col] - 1)
        
        # Taker Buy/Sell特徴量
        taker_cols = [col for col in df.columns if col.startswith('taker_')]
        if any('buy' in col for col in taker_cols) and any('sell' in col for col in taker_cols):
            buy_col = next(col for col in taker_cols if 'buy' in col.lower())
            sell_col = next(col for col in taker_cols if 'sell' in col.lower())
            
            # 買い/売り比率
            df['taker_buy_sell_ratio'] = df[buy_col] / (df[sell_col] + 1e-10)
            # 買い-売り差分
            df['taker_buy_sell_diff'] = df[buy_col] - df[sell_col]
            # 比率の移動平均
            df['taker_ratio_ma7'] = df['taker_buy_sell_ratio'].rolling(7).mean()
            # 差分の移動平均
            df['taker_diff_ma7'] = df['taker_buy_sell_diff'].rolling(7).mean()
            # 買い/売り圧力（正規化）
            total_volume = df[buy_col] + df[sell_col]
            df['taker_buy_pressure'] = (df[buy_col] / (total_volume + 1e-10) - 0.5) * 2  # -1 to 1
        
        # プレミアム指標特徴量
        premium_cols = [col for col in df.columns if col.startswith('premium_')]
        if any('premiumRate' in col for col in premium_cols):
            premium_col = next(col for col in premium_cols if 'premiumRate' in col)
            
            # プレミアムの移動平均
            df['premium_ma7'] = df[premium_col].rolling(7).mean()
            # プレミアムの偏差
            df['premium_dev'] = df[premium_col] - df['premium_ma7']
            # プレミアムのボラティリティ
            df['premium_volatility'] = df[premium_col].rolling(7).std()
            # プレミアムのZ値（標準化）
            df['premium_zscore'] = (df[premium_col] - df['premium_ma7']) / (df['premium_volatility'] + 1e-10)
            # 極端なプレミアム
            df['premium_extreme'] = ((df['premium_zscore'] > 2) | (df['premium_zscore'] < -2)).astype(int)
        
        # 複合特徴量（各指標の相互作用）
        # 1. 資金調達率とLSRの関係
        if all(col in df.columns for col in ['funding_rate_dev', 'lsr_dev']):
            df['funding_lsr_correlation'] = df['funding_rate_dev'] * df['lsr_dev']
            # 両方が負、両方が正、または混合シグナル
            df['funding_lsr_agreement'] = np.sign(df['funding_rate_dev']) * np.sign(df['lsr_dev'])
        
        # 2. 清算とボラティリティの関係
        if all(col in df.columns for col in ['total_liquidation', 'volatility_20']):
            df['liquidation_vol_ratio'] = df['total_liquidation'] / (df['volatility_20'] + 1e-10)
        
        # 3. 総合的な需給指標
        demand_supply_features = []
        if 'funding_rate_sign' in df.columns:
            demand_supply_features.append(-1 * df['funding_rate_sign'])  # 逆転（正のファンディングは売り圧力）
        if 'lsr_direction' in df.columns:
            demand_supply_features.append(df['lsr_direction'])
        if 'taker_buy_pressure' in df.columns:
            demand_supply_features.append(df['taker_buy_pressure'])
        if 'premium_dev' in df.columns:
            demand_supply_features.append(np.sign(df['premium_dev']))
            
        if demand_supply_features:
            df['demand_supply_indicator'] = pd.concat(demand_supply_features, axis=1).mean(axis=1)
            # 強さも考慮した指標
            df['demand_supply_strength'] = df['demand_supply_indicator'].abs()
        
        # NaN値を前方補完
        df.fillna(method='ffill', inplace=True)
        # 残りのNaN値を0で埋める
        df.fillna(0, inplace=True)
        
        return df
        
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ボリューム関連の指標を追加
        
        Parameters:
        -----------
        df : pandas.DataFrame
            特徴量を追加するデータフレーム
            
        Returns:
        --------
        pandas.DataFrame
            特徴量が追加されたデータフレーム
        """
        # ボリューム変化率
        df['volume_change'] = df['volume'].pct_change()
        
        # 移動平均ボリューム
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        
        # ボリューム比率
        df['volume_ratio_10'] = df['volume'] / df['volume_sma_10']
        
        # 価格上昇/下降時のボリューム
        up_volume = df['volume'].copy()
        up_volume[df['returns_1'] <= 0] = 0
        down_volume = df['volume'].copy()
        down_volume[df['returns_1'] > 0] = 0
        
        df['up_volume_10'] = up_volume.rolling(10).mean()
        df['down_volume_10'] = down_volume.rolling(10).mean()
        
        # ボリューム加重平均価格 (VWAP)
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        try:
            # MFI (Money Flow Index)
            df['mfi_14'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
            
            # On-Balance Volume (OBV)
            df['obv'] = talib.OBV(df['close'], df['volume'])
            
            # Chaikin A/D Line
            df['ad_line'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
            
            # Chaikin A/D Oscillator
            df['ad_osc'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)
        except:
            warnings.warn("TaLibが使用できないため、MFI等の指標は計算されません。")
        
        return df
    
    def _add_mtf_combination_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        複数の時間枠からの特徴量を組み合わせた特徴量を追加
        
        Parameters:
        -----------
        df : pandas.DataFrame
            特徴量を追加するデータフレーム
            
        Returns:
        --------
        pandas.DataFrame
            特徴量が追加されたデータフレーム
        """
        # 列名のプレフィックスを取得（上位時間枠の識別）
        prefixes = set()
        for col in df.columns:
            parts = col.split('_', 1)
            if len(parts) > 1 and parts[0].startswith('h'):
                prefixes.add(parts[0])
        
        # MTF組み合わせ特徴量の作成
        for prefix in prefixes:
            # トレンドフラグの統合（各時間枠のトレンドフラグを組み合わせ）
            if f'{prefix}_trend_flag' in df.columns and 'trend_flag' in df.columns:
                # 両方上昇トレンドならより強いシグナル
                df[f'mtf_{prefix}_trend_alignment'] = df[f'{prefix}_trend_flag'] + df['trend_flag']
            
            # RSIの差（上位時間枠と下位時間枠のRSIの差）
            if f'{prefix}_rsi_14' in df.columns and 'rsi_14' in df.columns:
                df[f'mtf_{prefix}_rsi_diff'] = df[f'{prefix}_rsi_14'] - df['rsi_14']
            
            # MACDの差（上位時間枠と下位時間枠のMACDの差）
            if f'{prefix}_macd' in df.columns and 'macd' in df.columns:
                df[f'mtf_{prefix}_macd_diff'] = df[f'{prefix}_macd'] - df['macd']
            
            # Coinglassデータの統合
            # ファンディングレート
            funding_cols = [col for col in df.columns if col.startswith(f'{prefix}_') and 
                            ('funding_rate' in col or 'funding_c' in col)]
            if funding_cols:
                funding_col = funding_cols[0]
                df[f'mtf_{prefix}_funding_signal'] = -1 * (df[funding_col] / 0.0001)  # 正規化
            
            # ロングショート比率
            lsr_cols = [col for col in df.columns if col.startswith(f'{prefix}_') and 
                        ('longShortRatio' in col or 'lsr_' in col)]
            if lsr_cols:
                lsr_col = lsr_cols[0]
                lsr = df[lsr_col]
                df[f'mtf_{prefix}_lsr_signal'] = (lsr - 1.0) / 0.1  # LSR 1.0を基準に正規化
            
            # プレミアム指数
            premium_cols = [col for col in df.columns if col.startswith(f'{prefix}_') and 'premium' in col]
            if premium_cols:
                premium_col = premium_cols[0]
                df[f'mtf_{prefix}_premium_signal'] = -1 * (df[premium_col] / 0.001)  # 正規化
            
            # 清算データ
            liq_cols = [col for col in df.columns if col.startswith(f'{prefix}_') and 
                         ('liquidation' in col or 'liq_' in col)]
            if liq_cols:
                # 複数の清算カラムがある場合
                if len(liq_cols) > 1:
                    long_liq_col = next((col for col in liq_cols if 'long' in col.lower()), None)
                    short_liq_col = next((col for col in liq_cols if 'short' in col.lower()), None)
                    if long_liq_col and short_liq_col:
                        # ロング/ショート清算比率のシグナル
                        liq_ratio = df[long_liq_col] / (df[short_liq_col] + 1e-10)
                        df[f'mtf_{prefix}_liq_signal'] = (liq_ratio - 1.0) * 0.1  # 正規化
                else:
                    # 単一の清算カラムのみの場合
                    liq_col = liq_cols[0]
                    df[f'mtf_{prefix}_liq_signal'] = df[liq_col] / df[liq_col].rolling(10).mean() - 1
            
            # OIデータ
            oi_cols = [col for col in df.columns if col.startswith(f'{prefix}_') and 'oi_' in col]
            if oi_cols:
                oi_col = next((col for col in oi_cols if col.endswith('_c')), None)
                if oi_col:
                    # OI変化率シグナル
                    df[f'mtf_{prefix}_oi_change'] = df[oi_col].pct_change()
                    df[f'mtf_{prefix}_oi_signal'] = df[f'mtf_{prefix}_oi_change'] / df[f'mtf_{prefix}_oi_change'].rolling(10).std()
        
        # MTFトレンド強度指標（全時間枠でのトレンド一致度を計算）
        trend_flags = [col for col in df.columns if 'trend_flag' in col]
        if trend_flags:
            # 全時間枠のトレンドフラグの平均（0～1）
            df['mtf_trend_strength'] = df[trend_flags].mean(axis=1)
            
            # トレンド一致/不一致フラグ（すべての時間枠で同じトレンド方向か）
            if len(trend_flags) > 1:
                # すべての時間枠で上昇トレンドなら1、下降トレンドなら-1、不一致なら0
                df['trend_all_up'] = df[trend_flags].all(axis=1).astype(int)
                df['trend_all_down'] = (~df[trend_flags].any(axis=1)).astype(int)
                df['mtf_trend_agreement'] = df['trend_all_up'] - df['trend_all_down']
        
        # ボラティリティクラスタリング
        vol_cols = [col for col in df.columns if 'volatility' in col]
        if len(vol_cols) > 1:
            # 最も短い時間枠と最も長い時間枠のボラティリティの比率
            shortest_vol = sorted(vol_cols, key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 999)[0]
            longest_vol = sorted(vol_cols, key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)[-1]
            df['mtf_vol_ratio'] = df[shortest_vol] / (df[longest_vol] + 1e-10)
            
            # ボラティリティ拡大/縮小トレンド
            df['vol_expanding'] = (df['mtf_vol_ratio'] > df['mtf_vol_ratio'].shift(1)).astype(int)
        
        # Coinglassデータ統合
        # ファンディングレート
        funding_signal_cols = [col for col in df.columns if 'funding_signal' in col and col.startswith('mtf_')]
        if funding_signal_cols:
            df['mtf_funding_avg_signal'] = df[funding_signal_cols].mean(axis=1)
        
        # ロングショート比率
        lsr_signal_cols = [col for col in df.columns if 'lsr_signal' in col and col.startswith('mtf_')]
        if lsr_signal_cols:
            df['mtf_lsr_avg_signal'] = df[lsr_signal_cols].mean(axis=1)
        
        # プレミアム指数
        premium_signal_cols = [col for col in df.columns if 'premium_signal' in col and col.startswith('mtf_')]
        if premium_signal_cols:
            df['mtf_premium_avg_signal'] = df[premium_signal_cols].mean(axis=1)
        
        # 清算データ
        liq_signal_cols = [col for col in df.columns if 'liq_signal' in col and col.startswith('mtf_')]
        if liq_signal_cols:
            df['mtf_liq_avg_signal'] = df[liq_signal_cols].mean(axis=1)
        
        # OIデータ
        oi_signal_cols = [col for col in df.columns if 'oi_signal' in col and col.startswith('mtf_')]
        if oi_signal_cols:
            df['mtf_oi_avg_signal'] = df[oi_signal_cols].mean(axis=1)
        
        # 複合シグナル
        # 1. 基本シグナル（すべてのMTFシグナルの平均）
        mtf_signal_cols = [col for col in df.columns if col.startswith('mtf_') and col.endswith('_signal')]
        if mtf_signal_cols:
            df['mtf_combined_signal'] = df[mtf_signal_cols].mean(axis=1)
            
            # 2. トレンドの方向に合わせて調整
            if 'mtf_trend_strength' in df.columns:
                # トレンド強度が0.5より大きい場合は上昇トレンド、小さい場合は下降トレンド
                trend_direction = (df['mtf_trend_strength'] - 0.5) * 2  # -1～1に正規化
                df['mtf_adjusted_signal'] = df['mtf_combined_signal'] * trend_direction
            
            # 3. トレンドと需給の組み合わせ（需給が強くトレンドも一致している場合により強いシグナル）
            if 'demand_supply_indicator' in df.columns and 'mtf_trend_agreement' in df.columns:
                # 需給とトレンドが一致するか（符号が同じか）
                supply_demand_trend_agreement = np.sign(df['demand_supply_indicator']) * np.sign(df['mtf_trend_agreement'])
                
                # 一致する場合は強化、不一致の場合は弱化
                df['mtf_trend_supply_alignment'] = supply_demand_trend_agreement * df['demand_supply_strength']
                
                # 最終的な複合シグナル
                df['mtf_final_signal'] = df['mtf_adjusted_signal'] * (1 + 0.5 * df['mtf_trend_supply_alignment'])
        
        return df
