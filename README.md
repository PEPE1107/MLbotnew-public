# ML Bot - Coinglass データを用いた BTC 自動売買

## 概要

Coinglass API のデータを利用して Bitcoin の自動売買を行う機械学習ベースのトレーディングボットです。市場データの取得、特徴量生成、ML モデルの学習、バックテスト、シグナル生成、取引実行までを一貫して行います。

## 機能

- **データ取得**: Coinglass API からの価格・OI・資金調達率・ロングショート比などのデータ収集
- **特徴量生成**: 時系列データからの特徴量抽出とZ正規化
- **モデル学習**: LightGBM, CatBoost, Transformer モデルによる予測
- **バックテスト**: vectorbtpro を用いた戦略性能の評価
- **シグナル生成**: リアルタイム市場データに基づく取引シグナルの生成
- **取引実行**: Bybit API を用いた取引の自動執行
- **リスク管理**: ポジションサイズ制限、損失制限などのリスク管理
- **モニタリング**: Prometheus + Grafana によるパフォーマンス監視
- **日足トレンドフィルター**: 上位時間枠（日足）のトレンドを用いてエントリーをフィルタリング

## システム構成

```
bot/
├── config/            # 設定ファイル
│   ├── api_keys.yaml  # API キー (Coinglass, Bybit, etc.)
│   ├── intervals.yaml # 時間枠設定
│   ├── fees.yaml      # 手数料・スリッページ設定
│   ├── system.yaml    # システム全体設定（データソース設定など）
│   └── mtf.yaml       # マルチタイムフレーム設定
├── data/              # データストレージ
│   ├── raw/           # API 生データ (parquet)
│   └── features/      # 前処理済みデータ
├── models/            # 学習済みモデル
├── reports/           # バックテスト結果・レポート
├── signals/           # 生成されたシグナル
├── src/               # ソースコード
│   ├── download.py    # データ取得モジュール
│   ├── feature.py     # 特徴量生成モジュール
│   ├── target.py      # 目的変数生成モジュール
│   ├── train.py       # モデル学習モジュール
│   ├── backtest.py    # バックテストモジュール
│   ├── live_signal.py # シグナル生成サービス
│   ├── execution.py   # 取引実行モジュール
│   ├── risk.py        # リスク監視モジュール
│   ├── generate_sample_data.py # サンプルデータ生成ユーティリティ
│   ├── clean_sample_data.py    # サンプルデータクリーンアップユーティリティ
│   └── config_loader.py        # 設定ロードユーティリティ
├── prometheus/        # Prometheus 設定
├── logs/              # ログファイル
├── Dockerfile         # Docker イメージ定義
├── docker-compose.yml # Docker Compose 設定
└── requirements.txt   # Python 依存関係
```

## 前提条件

- Docker と Docker Compose がインストール済み
- Coinglass API キー
- Bybit API キー（実取引時）

## セットアップ

1. リポジトリをクローン:
   ```
   git clone https://github.com/yourusername/mlbot.git
   cd mlbot
   ```

2. 設定ファイルの準備:
   ```
   # .env ファイルの作成
   CG_API_KEY=YOUR_COINGLASS_KEY
   BYBIT_API_KEY=YOUR_BYBIT_API_KEY
   BYBIT_API_SECRET=YOUR_BYBIT_API_SECRET
   SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR_WEBHOOK_URL
   
   # config/system.yaml の作成
   data:
     use_real_data: true  # 実データを使用（サンプルデータは使用しない）
     max_samples_per_interval: 4320
     log_level: INFO
   ```

3. Docker イメージのビルドと起動:
   ```
   docker-compose build
   docker-compose up -d
   ```

## 使用方法

### データ収集

データ収集のみ実行:

```bash
docker-compose up downloader
```

または特定の時間枠のみ実行:

```bash
docker-compose run downloader python src/download.py --interval 2h
```

API接続エラー時にサンプルデータを許可する場合:

```bash
docker-compose run downloader python src/download.py --interval 2h --allow-sample
```

### サンプルデータ管理

サンプルデータは開発・テスト目的のみに使用し、実際のトレーディングには使用しないでください。

サンプルデータの削除:

```bash
python src/clean_sample_data.py
```

サンプルデータの強制生成:

```bash
python src/generate_sample_data.py --force
```

### システム設定

`config/system.yaml` で以下の設定が可能:

```yaml
# データソース設定
data:
  # 実データを使用するか（true）、サンプルデータを使用するか（false）
  use_real_data: true
  
  # 各時間足ごとの最大サンプル数
  max_samples_per_interval: 4320
  
  # データ操作のログレベル
  log_level: INFO
```

### 特徴量生成

```bash
docker-compose run downloader python src/feature.py --interval 2h
```

### モデル学習

```bash
docker-compose run downloader python src/train.py --interval 2h --model_type lightgbm --optimize
```

### バックテスト

```bash
docker-compose run downloader python src/backtest.py --interval 2h --model_type lightgbm
```

### 本番環境

全サービスを起動:

```bash
docker-compose up -d
```

モニタリングダッシュボード:
- Grafana: http://localhost:3000 (初期認証: admin/admin)
- Prometheus: http://localhost:9093

## 開発ロードマップ (8週間)

| 週 | マイルストーン                     | 成果物                               |
| - | ----------------------------------- | ------------------------------------ |
| 1 | データ取得 & Storage 完成           | download.py, データパイプライン      |
| 2 | feature.py + target.py v1           | 特徴量生成パイプライン              |
| 3 | LightGBM + WF-CV → Sharpe>1         | 基本モデルと交差検証結果            |
| 4 | vectorbt バックテスト pass          | バックテスト結果と統計              |
| 5 | Multi-TF stacking + Optuna          | マルチタイムフレームメタモデル      |
| 6 | RL-bandit size 最適化               | ポジションサイジング最適化          |
| 7 | Testnet paper-trade & 監視          | テストネット検証結果                |
| 8 | Mainnet 小規模運用開始、監視安定化   | 本番システム稼働                    |

## リスク管理

- NAV（純資産価値）に対して最大 10% のポジションサイズ制限
- 日次損失制限 -3% NAV で撤退
- 最大許容ドローダウン 25%

## 学習メトリクス

バックテスト・学習の合格基準:
- Sharpe > 1.0 (リスク調整後リターン)
- MAR ≥ 0.5 (CAGR / MaxDD)
- MaxDD < 25 % (最大ドローダウン)

## モデル方式

1. LightGBMClassifier: 方向予測 (上昇/下降/ニュートラル)
2. CatBoostRegressor: ボラティリティ調整リターン予測
3. Temporal Fusion Transformer: シーケンシャル予測 (深層学習)
4. メタモデル: 複数時間枠・複数モデルの予測を組み合わせ

## 日足トレンドフィルター

日足の指標を使用して下位時間足（例: 15分足、1時間足）のトレードシグナルをフィルタリングする機能を実装:

- `mtf.yaml` 設定ファイルにより制御可能:
  ```yaml
  use_trend_filter: true    # 有効/無効の切り替え
  trend_filter:
    indicator: "ema"        # 使用する指標 (ema, sma)
    period: 200             # 期間 (例: 200日移動平均線)
    condition: "price_above" # 条件 (価格が移動平均線より上/下)
  ```

- 動作:
  - 上昇トレンド時のみロングポジションを許可
  - 下降トレンドに転換した場合は速やかにポジションを閉じる
  - バックテスト結果はより安定した収益曲線となる傾向

## データソース

システムは以下の2つのデータソースをサポートしています:

### 1. Coinglass API (実データ)
- 実際のトレーディングには実データの使用を推奨
- 時間枠: 15分、2時間、日足
- 取得データ: 価格、OI、ファンディングレート、清算量、L/S比、テイカー比率、オーダーブック比率、プレミアム指数
- 各時間枠ごとに最大4,320サンプル（15m=約45日、2h=約1年、1d=約12年）

### 2. サンプルデータ
- 開発・テスト目的でのみ使用
- API接続エラー時のフォールバックオプション
- サンプルデータは実データと同じ構造で生成され、時間枠ごとに4,320サンプル
- コマンド `python src/download.py --allow-sample` で許可可能

## ライセンス

MIT

## 謝辞

- Coinglass API for market data
- Bybit API for trading execution
- VectorBT Pro for backtesting capabilities
