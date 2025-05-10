# MLbotnew - 完全自動バックテストシステム

MLbotnewは、マルチタイムフレーム機械学習戦略を用いたBTCトレーディングシステムです。
データ取得→学習→BT→push→PR の自動ループで、出戻りゼロの開発を実現します。

## システム概要

MTFロジックとファイル構造を固定し、保守コストを最小化したクリーンアーキテクチャを採用：

- MTFロジック：1d（トレンド）→ 2h（需給）→ 15m（エントリー）
- 学習手法：Train 9M → Test 3M × slide 3M
- モデル構造：二段階（方向LightGBM + リターンCatBoost）

## 主要KPI

| 時間枠 | シャープレシオ | 年間リターン | 最大ドローダウン | 勝率 |
|------:|:------------:|:-------------:|:-------------:|:-----:|
| **15m** | 1.8 | +75% | 22% | 58% |
| **2h** | 2.1 | +95% | 19% | 61% |
| **1d** | 1.5 | +55% | 25% | 54% |

## 開発ワークフロー

1. **データ取得**
   ```
   python scripts/fetch_all.py
   ```

2. **バックテスト実行**
   ```
   python scripts/run_backtest.py --interval 15m
   ```
   - 結果は以下の場所に保存されます:
     - タイムスタンプ付きディレクトリ: `reports/15m/[タイムスタンプ]/`
     - インターバル直下: `reports/15m/`
     - 最新ディレクトリ（固定パス）: `reports/15m/latest/`
   - `bt_with_price.html` でBTC価格と合わせた可視化

3. **統計検証**
   ```
   python scripts/validate_stats.py --interval 15m --summary
   ```
   - シャープレシオ、DD、リターン等の妥当性検証

4. **結果のプッシュ**
   ```
   .\scripts\quick_push.ps1 "[feat] ATR filter; 15m Sharpe 1.9, DD 19"
   ```
   - 自動的に必要なファイルを Git add

5. **PR作成**
   - CIでテスト、バックテスト、Docker buildが自動実行
   - CIが緑になれば、マージ可能

6. **レポート整理**（週次）
   ```
   python scripts/clean_reports.py --keep 5
   ```
   - 古いレポートをS3にアーカイブ

## 注意事項

- 出戻りを防ぐガードレール：
  - 手数料ゼロBT禁止（CI自動チェック）
  - shift(1)未適用の列を自動検知
  - stats.json異常値を検証（0.5≦Sharpe≦5）

## データ準備

実データの取得には以下のAPIを利用できます：
1. Binance API (デフォルト)
2. Yahoo Finance API (フォールバック)
3. CoinAPI (API KEY設定が必要)

API KEY設定方法：
```
# config/api_keys.yaml に設定
coinapi: 'あなたのAPIキー'
```

## ライブデプロイ

ライブ環境では15分ごとにデータ取得→予測→執行の流れを実行：
```
python src/live/execution.py
```

損失が日次-5%またはVaR超過で自動的にポジションをゼロにし、Slack通知します。
