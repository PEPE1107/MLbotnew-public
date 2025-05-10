"""
時間枠リサンプリングモジュール
-------------------------

異なる時間枠間のデータのリサンプリングを行うためのモジュール。
MTFロジックにおいて、上位時間枠の判断を下位時間枠に適用するために使用。
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Union, Optional

class TimeFrameResampler:
    """
    異なる時間枠間のデータをリサンプリングするクラス
    """
    
    def __init__(self, intervals_config: Dict):
        """
        初期化
        
        Parameters:
        -----------
        intervals_config : dict
            時間枠の設定情報の辞書
        """
        self.intervals_config = intervals_config
        self.freq_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1H',
            '2h': '2H',
            '4h': '4H',
            '6h': '6H',
            '12h': '12H',
            '1d': '1D',
            '3d': '3D',
            '1w': '1W'
        }
    
    def resample(self, df: pd.DataFrame, source_interval: str, target_interval: str, 
                auto_shift: bool = True, shift_periods: int = 1) -> pd.DataFrame:
        """
        上位時間枠から下位時間枠へのリサンプリング(ダウンサンプリング)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            リサンプリングするデータフレーム
        source_interval : str
            ソースの時間枠 (例: '1d')
        target_interval : str
            ターゲットの時間枠 (例: '15m')
        auto_shift : bool
            需給指標に自動的にシフトを適用するかどうか
        shift_periods : int
            シフトする期間数（auto_shift=Trueの場合に使用）
            
        Returns:
        --------
        pandas.DataFrame
            リサンプリングされたデータフレーム
        """
        # 入力チェック
        if source_interval not in self.freq_map or target_interval not in self.freq_map:
            raise ValueError(f"不明な時間枠: {source_interval} または {target_interval}")
        
        # 上位→下位の時間枠への変換のみサポート
        source_minutes = self._to_minutes(source_interval)
        target_minutes = self._to_minutes(target_interval)
        
        if source_minutes < target_minutes:
            raise ValueError(f"上位時間枠({source_interval})から下位時間枠({target_interval})へのリサンプリングのみサポートしています")
        
        # データフレームに時間インデックスがあるか確認
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("データフレームにDatetimeIndexが必要です")
        
        # MTFルール適用：自動シフト処理
        if auto_shift and source_interval in ['1d', '2h', '4h', '6h']:
            # シフト対象となる需給指標のパターン
            supply_demand_patterns = [
                'funding_', 'oi_', 'liq_', 'liquidation_', 'lsr_', 'longShortRatio', 
                'premium_', 'taker_', 'orderbook_'
            ]
            
            # 自動シフト適用
            df_to_resample = df.copy()
            
            for col in df.columns:
                # 需給指標かどうかをチェック
                is_supply_demand = any(pattern in col for pattern in supply_demand_patterns)
                
                # 需給指標にはシフトを適用し、その他の指標はそのまま
                if is_supply_demand:
                    df_to_resample[col] = df[col].shift(shift_periods)
                    print(f"[MTF] '{col}' に {shift_periods} 期間のシフトを適用しました（{source_interval}→{target_interval}）")
        else:
            df_to_resample = df
        
        # カラム名の保存
        columns = df_to_resample.columns.tolist()
        
        # 下位時間枠のインデックスを生成
        target_freq = self.freq_map[target_interval]
        start_time = df_to_resample.index.min()
        end_time = df_to_resample.index.max()
        
        target_index = pd.date_range(start=start_time, end=end_time, freq=target_freq)
        
        # 空のデータフレームを作成
        resampled_df = pd.DataFrame(index=target_index, columns=columns)
        
        # 前方補完（上位時間枠の値を下位時間枠に引き継ぐ）
        for col in columns:
            # NaNで初期化
            resampled_df[col] = np.nan
            
            # 上位時間枠の各行を下位時間枠にマッピング
            for idx, val in df_to_resample[col].items():
                # 該当する下位時間枠の行を特定
                target_rows = resampled_df.index.searchsorted(idx)
                if target_rows < len(resampled_df):
                    # 値を設定
                    resampled_df.iloc[target_rows, resampled_df.columns.get_loc(col)] = val
            
            # 前方補完
            resampled_df[col] = resampled_df[col].ffill()
        
        return resampled_df
    
    def upsample(self, df: pd.DataFrame, source_interval: str, target_interval: str, agg_methods: Optional[Dict] = None) -> pd.DataFrame:
        """
        下位時間枠から上位時間枠へのリサンプリング(アップサンプリング)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            リサンプリングするデータフレーム
        source_interval : str
            ソースの時間枠 (例: '15m')
        target_interval : str
            ターゲットの時間枠 (例: '1d')
        agg_methods : dict, optional
            カラムごとの集計方法の辞書。指定がない場合はデフォルト設定を使用
            例: {'close': 'last', 'open': 'first', 'high': 'max', 'low': 'min', 'volume': 'sum'}
            
        Returns:
        --------
        pandas.DataFrame
            リサンプリングされたデータフレーム
        """
        # 入力チェック
        if source_interval not in self.freq_map or target_interval not in self.freq_map:
            raise ValueError(f"不明な時間枠: {source_interval} または {target_interval}")
        
        # 下位→上位の時間枠への変換のみサポート
        source_minutes = self._to_minutes(source_interval)
        target_minutes = self._to_minutes(target_interval)
        
        if source_minutes > target_minutes:
            raise ValueError(f"下位時間枠({source_interval})から上位時間枠({target_interval})へのリサンプリングのみサポートしています")
        
        # データフレームに時間インデックスがあるか確認
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("データフレームにDatetimeIndexが必要です")
        
        # デフォルトの集計方法
        default_agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum',
            'trades': 'sum'
        }
        
        # 集計方法の設定
        if agg_methods is None:
            agg_methods = {}
            
            # デフォルト設定を適用
            for col in df.columns:
                if col.lower() in default_agg:
                    agg_methods[col] = default_agg[col.lower()]
                else:
                    # その他の列は最後の値を使用
                    agg_methods[col] = 'last'
        
        # リサンプリング実行
        target_freq = self.freq_map[target_interval]
        resampled = df.resample(target_freq).agg(agg_methods)
        
        return resampled
    
    def apply_shifts(self, df: pd.DataFrame, interval: str, shift_rules: Dict) -> pd.DataFrame:
        """
        未来漏洩防止のためのshift適用
        
        Parameters:
        -----------
        df : pandas.DataFrame
            シフトを適用するデータフレーム
        interval : str
            時間枠 (例: '15m')
        shift_rules : dict
            シフトルールの辞書
            例: {'global': 1, 'exceptions': {'volume': 0}}
            
        Returns:
        --------
        pandas.DataFrame
            シフトが適用されたデータフレームのコピー
        """
        # 入力チェック
        if 'global' not in shift_rules:
            raise ValueError("shift_rulesにはglobalキーが必要です")
        
        # データのコピーを作成
        df_shifted = df.copy()
        
        # グローバルシフト値を取得
        global_shift = shift_rules['global']
        
        # 例外リストを取得（指定がなければ空の辞書）
        exceptions = shift_rules.get('exceptions', {})
        
        # グローバルシフトを適用
        for col in df.columns:
            # 例外リストにあるか確認
            if col in exceptions:
                shift_val = exceptions[col]
            else:
                shift_val = global_shift
            
            # シフトを適用
            if shift_val != 0:
                df_shifted[col] = df[col].shift(shift_val)
                
                # シフト適用ログ
                print(f"[{interval}] カラム '{col}' に {shift_val} 期間のシフトを適用しました")
        
        return df_shifted
    
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
        suspicious_cols = []
        
        # 価格参照用カラム（基本的に終値を使用）
        price_col = None
        for candidate in ['close', 'Close', 'price', 'Price']:
            if candidate in df.columns:
                price_col = candidate
                break
        
        if price_col is None:
            warnings.warn("価格カラムが見つかりません。未来漏洩チェックをスキップします。")
            return suspicious_cols
        
        # 未来リターン計算（理論上はこれと特徴量の相関がある場合、未来漏洩の可能性がある）
        future_return = df[price_col].pct_change().shift(-1)
        
        # shift(1)が適用されている必要がある特殊なパターン
        requires_shift_patterns = [
            'funding_', 'oi_', 'liq_', 'liquidation_', 'lsr_', 'longShortRatio', 
            'premium_', 'taker_', 'orderbook_'
        ]
        
        # 各特徴量と未来リターンの相関をチェック
        for col in df.columns:
            if col == price_col:
                continue
                
            # 異常に高い相関がないかチェック
            rolling_corr = df[col].rolling(window=lookback_window).corr(future_return)
            mean_corr = rolling_corr.tail(lookback_window).abs().mean()
            
            # 需給指標かどうかをチェック
            is_supply_demand = any(pattern in col for pattern in requires_shift_patterns)
            
            # 需給指標の場合は厳しくチェック
            if is_supply_demand:
                # 需給指標は修正が必要なので厳しく
                threshold = 0.3
            else:
                # その他の特徴量は一般的な閾値
                threshold = 0.5
            
            if mean_corr > threshold:
                if is_supply_demand:
                    warnings.warn(f"警告: 需給指標 '{col}' は未来リターンと高い相関 ({mean_corr:.2f}) があります。shift(1)を適用する必要があります。")
                else:
                    warnings.warn(f"警告: カラム '{col}' は未来リターンと高い相関 ({mean_corr:.2f}) があります。未来漏洩の可能性があります。")
                suspicious_cols.append(col)
        
        return suspicious_cols
    
    def _to_minutes(self, interval: str) -> int:
        """
        時間枠を分単位に変換
        
        Parameters:
        -----------
        interval : str
            時間枠 (例: '15m', '1h', '1d')
            
        Returns:
        --------
        int
            分単位の時間
        """
        unit = interval[-1]
        value = int(interval[:-1])
        
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 24 * 60
        elif unit == 'w':
            return value * 7 * 24 * 60
        else:
            raise ValueError(f"不明な時間単位: {unit}")
