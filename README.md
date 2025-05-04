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

## システム構成

```
bot/
├── config/            # 設定ファイル
│   ├── api_keys.yaml  # API キー (Coinglass, Bybit, etc.)
│   ├── intervals.yaml # 時間枠設定
│   └── fees.yaml      # 手数料・スリッページ設定
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
│   └── risk.py        # リスク監視モジュール
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
   # config/api_keys.yaml 作成
   cg_api: "YOUR_COINGLASS_KEY"
   bybit:
     key: "YOUR_BYBIT_API_KEY"
     secret: "YOUR_BYBIT_API_SECRET"
   slack_webhook: "https://hooks.slack.com/services/YOUR_WEBHOOK_URL"
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

## ライセンス

MIT

## 謝辞

- Coinglass API for market data
- Bybit API for trading execution
- VectorBT Pro for backtesting capabilities
