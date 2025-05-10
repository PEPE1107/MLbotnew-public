# MLbotnew - 2時間足Coinglassデータを使用したBTC取引ボット

MLbotnewは、Coinglassから取得した2時間足のデータを使用して、ビットコイン先物取引の売買シグナルを生成する機械学習ベースの取引ボットです。本プロジェクトでは15分足と1日足のデータは削除され、2時間足のみに特化しています。

## プロジェクト概要

このボットは、以下のCoinglassデータを特徴量として利用し、バックテストを行います：

- **価格OHLC情報** - ビットコインの価格推移データ
- **ファンディングレート** - 先物取引のファンディングレート
- **オープンインタレスト** - 未決済の契約数
- **清算データ** - ロング/ショートポジションの清算量
- **ロング/ショート比率** - トレーダーのロングとショート比率
- **テイカーバイ/セルボリューム** - 買い/売りの取引量
- **オーダーブック情報** - 買い/売り注文の集計情報
- **Coinbaseプレミアム指数** - CoinbaseとBinanceの価格差

## バックテスト結果

最新のバックテスト結果（2025年5月10日実行）：

- **トータルリターン**: 6.85%
- **シャープレシオ**: 13.31
- **最大ドローダウン**: 0.89%
- **勝率**: 100%

2時間足のCoinglassデータを使用した戦略は、低いドローダウンと高いシャープレシオを実現しています。

## ルールブック

- 2時間足のCoinglassデータのみを使用（360日分）
- 15分足、1日足のデータは使用しない（既に削除済み）
- すべての特徴量を組み合わせたバックテストを実施
- 各データセットのタイムスタンプを利用して時間のアライメントを行う

## インストール方法

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/MLbotnew.git
cd MLbotnew

# 依存関係をインストール
pip install -r requirements.txt
```

## 使用方法

### データ取得

```bash
# 2時間足のCoinglassデータを取得（360日分）
python scripts/fetch_all.py --days 360
```

### バックテスト実行

```bash
# バックテストを実行
python scripts/run_backtest.py
```

### 結果の表示

```bash
# バックテスト結果を表示
python scripts/display_backtest_results.py
```

## プロジェクト構造

```
MLbotnew/
├── config/             # 設定ファイル
├── data/               # 取得したデータ
│   └── raw/            # 生データ
├── features/           # 生成された特徴量
├── models/             # 学習済みモデル
├── reports/            # バックテスト結果
├── scripts/            # 実行スクリプト
└── src/                # ソースコード
    ├── backtest/       # バックテスト機能
    ├── data/           # データ処理
    └── features/       # 特徴量エンジニアリング
```

## メンテナンス

不要な時間枠のデータを削除するには、以下のスクリプトを実行します：

```bash
python scripts/cleanup_unused_timeframes.py
```

GitHubにプッシュするには、以下のスクリプトを実行します：

```bash
powershell -ExecutionPolicy Bypass -File scripts/github_push.ps1
```

## 開発モデル情報

MLbotnewは、Coinglassから取得した2時間足のデータのみを使用してビットコイン先物市場のパターンを学習します。学習モデルは、過去の市場データに基づいて売買シグナルを生成し、バックテストで結果を検証します。

## ライセンス

© 2025 MLbotnew. All Rights Reserved.
