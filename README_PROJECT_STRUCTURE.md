# MLbot プロジェクト構造ガイドライン

このドキュメントは、MLbotプロジェクトの推奨ディレクトリ構造と命名規則を定義します。プロジェクトの一貫性を保ち、コードの管理を容易にするために、すべての開発者はこのガイドラインに従ってください。

## ディレクトリ構造

```
MLbotnew/
│
├── config/                 # 設定ファイル
│   ├── api_keys.yaml       # API認証情報（非公開）
│   ├── fees.yaml           # 取引手数料設定
│   ├── intervals.yaml      # 時間枠設定
│   ├── mtf.yaml            # マルチタイムフレーム設定
│   └── system.yaml         # システム全体の設定
│
├── data/                   # 生データ保存ディレクトリ
│   └── raw/                # 未処理の生データ
│
├── docs/                   # ドキュメント
│   └── architecture/       # アーキテクチャ図など
│
├── models/                 # 学習済みモデルの保存先
│   └── model_info_*.json   # モデルのメタ情報
│
├── notebooks/              # Jupyter notebooks（探索的分析用）
│
├── prometheus/             # Prometheusモニタリング設定
│
├── reports/                # バックテスト結果報告
│   ├── 15m/                # 15分足の結果
│   │   ├── stats.json      # パフォーマンス統計
│   │   └── [日付]_[時刻]/   # タイムスタンプ付き結果ディレクトリ
│   │
│   ├── 2h/                 # 2時間足の結果
│   │   └── ...
│   │
│   └── plots/              # グラフ保存ディレクトリ
│       ├── 15m/            # 15分足のグラフ
│       └── 2h/             # 2時間足のグラフ
│
├── scripts/                # 実行スクリプト
│   ├── check_price_data.py # 価格データ確認スクリプト
│   └── run_backtest.py     # バックテスト実行スクリプト
│
├── signals/                # 生成されたシグナルデータ
│
├── src/                    # ソースコード
│   ├── backtest/           # バックテスト関連モジュール
│   ├── bt/                 # バックテストユーティリティ
│   ├── data/               # データ処理モジュール
│   │   ├── __init__.py
│   │   └── coinglass.py    # Coinglass APIデータ取得
│   │
│   ├── features/           # 特徴量エンジニアリング
│   │   ├── __init__.py
│   │   └── builders.py     # 特徴量構築クラス
│   │
│   ├── live/               # ライブトレード関連
│   │
│   ├── models/             # モデル定義モジュール
│   │
│   ├── backtest.py         # バックテストメインモジュール
│   ├── config_loader.py    # 設定ロードユーティリティ
│   ├── daily_trend.py      # 日足トレンド計算
│   ├── download.py         # データダウンロード
│   ├── execution.py        # 注文執行ロジック
│   ├── feature.py          # 特徴量生成メインモジュール
│   ├── live_signal.py      # ライブシグナル生成
│   ├── plot_backtest_results.py  # バックテスト結果プロット
│   ├── risk.py             # リスク管理ロジック
│   ├── show_backtest_results.py  # バックテスト結果表示
│   ├── target.py           # モデルターゲット定義
│   └── train.py            # モデルトレーニング
│
├── tests/                  # テストコード
│
├── tools/                  # 開発用ツール
│   └── quick_push.ps1      # クイックプッシュスクリプト
│
├── .env                    # 環境変数
├── .gitattributes          # Git属性設定
├── .gitignore              # Git無視設定
├── cleanup_old_files.py    # 古いファイル削除スクリプト
├── docker-compose.yml      # Dockerコンポーズ設定
├── Dockerfile              # Dockerビルド設定
├── README.md               # プロジェクト説明
├── requirements.txt        # 依存パッケージリスト
└── run_filter_repo.ps1     # Gitフィルタースクリプト
```

## プロジェクト改善ガイドライン

### 1. モジュール分割

- 単一目的の原則: 各Pythonモジュールは単一の目的/機能に焦点を当て、適切に分割する
- 大きなファイルは複数の小さなモジュールに分割し、適切なサブディレクトリに配置する

### 2. 命名規則

- ファイル名: スネークケース（例: `feature_extraction.py`）
- クラス名: パスカルケース（例: `FeatureBuilder`）
- 関数・変数: スネークケース（例: `calculate_returns`）
- 定数: 大文字のスネークケース（例: `MAX_RETRY_COUNT`）

### 3. インポート順序

```python
# 標準ライブラリ
import os
import sys
import json

# サードパーティライブラリ
import numpy as np
import pandas as pd

# アプリケーション内部モジュール
from src.data import loader
from src.features import builders
```

### 4. バックテスト結果管理

- `reports/`ディレクトリで時間枠ごとにサブディレクトリを分ける
- タイムスタンプ付きのディレクトリを使用して実行ごとに結果を区別する
- 最新の結果のみを保持し、古いものは`cleanup_old_files.py`で定期的に削除する

### 5. コード移行ガイドライン

- コードを適切なディレクトリに移動させる
- すでに`src`ディレクトリにあるファイルは、機能に応じてサブディレクトリに移動する
- ルートにある実行ファイルは`scripts`ディレクトリに移動する

## 新しいモジュール追加時のルール

1. 適切なディレクトリを選択する（迷ったら上記の構造を参照）
2. 命名規則に従ってファイル名を決定する
3. モジュールのドキュメント文字列（docstring）で目的を明確に説明する
4. 関連するテストを`tests`ディレクトリに追加する

この構造に従うことで、プロジェクトの保守性と拡張性が向上し、新しい開発者もスムーズにプロジェクトに参加できるようになります。
