# MLbotnew "完全自動ルールブック" v2025-05-10

データ取得→学習→BT→push→PR が自動でループし、**人手の"出戻り"をゼロ**にする
機械学習トレーディングシステムです。
ファイル構造を固定して、保守コストを最小化しています。

## 0. フォルダ構造（固定）

```
MLbotnew/
├─ .github/workflows/ci.yml
├─ config/                （Git 管理）
│   ├─ intervals.yaml
│   ├─ fees.yaml
│   └─ model.yaml
├─ data/                  （.gitignore）Raw API
├─ features/              （.gitignore）前処理
├─ models/                （Git-LFS）本番 pkl
├─ reports/               （Git-LFS except *.json）
│   └─ {interval}/{runid}/stats.json / bt_with_price.html
├─ scripts/
│   ├─ fetch_all.py       日次 S3 バッチ
│   ├─ run_backtest.py    BT＋HTML
│   └─ quick_push.ps1     Git 一括 push
├─ src/                   clean-arch 5 階層
│   ├─ data/       (coinglass.py, sync.py)
│   ├─ features/   (resampler.py, builder.py)
│   ├─ models/     (classifier.py, regressor.py, bandit.py)
│   ├─ backtest/   (run.py, utils.py)
│   └─ live/       (signal.py, execution.py, risk.py)
└─ tests/
```

## 1. 単一時間足ロジック（2h only）

**従来のMTFロジックから単一時間足(2h)のみに簡素化**：

| データ                       | 役割     | ルール                                                                       |
| -------------------------- | ------ | ------------------------------------------------------------------------- |
| **2h**  | Coinglass需給データ | トレード判断 | OI/Funding/Liq/LSR/Premiumなど需給指標で<br>二段モデル：<br>①LightGBM 方向 0/1<br>②CatBoost μ<br>β = proba×μ / VaR95 |

* Coinglassの実データ（360日分）を使用 - 十分な市場サイクルをカバー
* 手法は **Train 9 M → Test 3 M × slide 3 M**。
* BT は vectorbt、`upon_op="Next"`, `fees=0.00055`, `slippage=0.0001`.
* **bt_with_price.html** に右軸で BTC Spot Price を点線表示。
* 重要指標（特徴量重要度、ロング/ショート比率など）はstats.jsonに保存

## 2. Git 運用ルール

| 種別                          | push 方法                                                 | 備考                       |
| --------------------------- | ------------------------------------------------------- | ------------------------ |
| **コード / config**            | 通常 Git                                                  | `quick_push.ps1` が自動 add |
| **stats.json**              | 通常 Git                                                  | レビュー用（必ずコミット）            |
| **大型 (pkl, html, parquet)** | Git-LFS                                                 | 自動 `git lfs track` 設定済   |
| **data/, features/**        | **push 禁止** (.gitignore)                                |                          |
| **古い run**                  | 直近 5 つ以外は `scripts/clean_reports.py` で S3 にアーカイブしローカル削除 |                          |

### quick_push.ps1（既存）

```powershell
param([string]$msg)
git add src/ config/ reports/*/stats.json
git commit -m $msg
git push -u origin $(git branch --show-current)
```

### 自動クリーン（週次）

`clean_reports.py --keep 5`

* LFS ポインタは残す / 実体は `git lfs prune` で整理。

## 3. CI / CD

```yaml
name: test-bt-docker
on: [push, pull_request]
jobs:
  unit:
    runs-on: ubuntu-latest
    steps: … pytest -q
  quick_bt:
    needs: unit
    steps: 
      - run: python scripts/run_backtest.py --interval 2h --quick
      - run: python scripts/validate_stats.py   # 0.5 ≤ Sharpe ≤ 5
      - uses: actions/upload-artifact@v4
        with: {name: stats, path: reports/*/stats.json}
  docker:
    needs: quick_bt
    steps: … docker build .
```

赤なら自動 reject。green でのみ main へマージ可能。

## 4. 新ファイル生成時のプロトコル

1. **バッチ or BT 実行**

   * Raw → `data/` に展開
   * Features → `features/`
   * Model → `models/*.pkl` (LFS)
   * stats.json / html → `reports/{intv}/{runid}/`

2. **自動整形**

   * `scripts/clean_reports.py` で `stats.json` 以外を LFS or S3
   * HTML は LFS or CI artifact （.gitignore で通常 push 除外）

3. **quick_push.ps1**

   * KPI をメッセージに入れて実行
   * 例：「`[feat] ATR filter; 2h Sharpe 1.9, DD 19`」

4. **PR 作成**
   テンプレに KPI 表を貼付→チャットで PR 番号共有。

## 5. 出戻りを起こさないガードレール

| ケース            | ガード                                                     |
| -------------- | ------------------------------------------------------- |
| 手数料ゼロ BT       | CI で `fees_check` スクリプト→Fail                            |
| データ不足        | Coinglass 360日データが揃っているか自動チェック                            |
| stats.json 異常値 | `validate_stats.py` で 0.5≤Sharpe≤5 判定                   |
| 大容量未 LFS       | pre-commit hook で 100 MB 超を reject                      |
| 機密 push        | `.env` / `api_keys.yaml` は .gitignore かつ pre-commit で拒否 |
| BT未実行        | PR前にバックテスト実行チェック → マージ禁止                           |
| 古いBT結果        | バックテスト日時が修正より古い場合 → 要再実行                           |

## 6. ライブ稼働スケッチ

```
cron 2h:
  ├─ data/sync.py       (差分 pull)
  ├─ features/build.py  (最新1行 features)
  ├─ models/predict     (clf+reg → size β)
  └─ live/execution.py  (Bybit Testnet)
        ↳ risk.py → Prometheus → Grafana
```

日次損失 -5 % or VaR 超過 → `β = 0` ＋ Slack Alert。

## 7. 保存する重要指標詳細

**stats.json に保存される重要指標：**

1. **基本パフォーマンス指標**
   - 総リターン (%)
   - 年間リターン (%)
   - シャープレシオ
   - ソルティノレシオ
   - 最大ドローダウン (%)
   - リカバリーファクター

2. **トレード統計**
   - 総トレード数
   - 勝率 (%)
   - 平均勝ちトレード (%)
   - 平均負けトレード (%)
   - ペイオフ比率
   - プロフィットファクター
   - 平均トレード期間

3. **ロング/ショート分析**
   - ロングトレード数/比率
   - ショートトレード数/比率
   - ロング勝率 (%)
   - ショート勝率 (%)
   - 方向バイアス（LONG/SHORT/NEUTRAL）

4. **特徴量重要度**
   - 各特徴量の重要度スコア（トップ10）
   - 特徴量グループ別の寄与度（価格データ、ファンディング、OI、流動性など）

5. **月次リターン**
   - 月別のリターン (%)
   - ポジティブ月/ネガティブ月の数

## 8. 実装・修正後のバックテスト実行（必須）

**すべてのコード修正後は必ずバックテストを実行し、結果を検証・保存してください。**

**コード修正後の確認手順（必須）：**

1. **データ品質チェック**（必須）
   ```
   python scripts/check_coinglass_data.py --interval 2h
   ```
   - 目標：360日分のデータカバレッジを確保（少なくとも99%以上）
   - 警告がある場合は`sync_all_data`を実行してデータを更新

2. **バックテスト実行**（必須）
   ```
   python scripts/run_backtest.py --interval 2h
   ```

3. **結果検証**（必須）
   ```
   python scripts/validate_stats.py --interval 2h --summary
   ```

4. **結果のコミット＆プッシュ**（必須）
   ```
   ./scripts/quick_push.ps1 "[修正内容] 2h Sharpe X.X, DD XX%"
   ```

5. **PR作成と共有**（必須）
   - GitHub上でPRを作成
   - データ品質チェック結果を含める
   - テンプレートに沿って結果を記載
   - チームにPR番号を共有

**注意：バックテスト結果が存在しないPR、または360日分のデータカバレッジが不足しているPRはマージ禁止です。**

## 最新バックテスト結果（2025-05-10）

### 2hバックテスト結果

- **期間**: 2025-05-02 ～ 2025-05-10 (7日間)
- **総リターン**: 6.85%
- **年率リターン**: 3061.57%
- **最大ドローダウン**: -0.89%
- **シャープレシオ**: 1.95
- **総トレード数**: 68回
- **勝率**: 58.82%

### ロング/ショート内訳
- ロング比率: 60.3% (41回)
- ショート比率: 39.7% (27回)
- ロングPnL: 4.21%
- ショートPnL: 2.64%
- 方向バイアス: LONG

### 主要特徴量グループ
- price_data: 3.2246
- funding_rates: 0.8229
- open_interest: 0.6060
- premium: 0.5707
- technical_indicators: 2.1225
- trend_momentum: 1.5078

## データ品質チェック結果（2025-05-10）

```
================================================================================
Coinglass データ確認サマリー (2h)
================================================================================

【日数カバレッジ】
目標日数: 360日
  [OK] price: 359日 (99.7%), 4320件 (100.0%)
  [OK] oi: 359日 (99.7%), 4320件 (100.0%)
  [OK] funding: 359日 (99.7%), 4320件 (100.0%)
  [OK] liq: 359日 (99.7%), 4320件 (100.0%)
  [OK] lsr: 359日 (99.7%), 4320件 (100.0%)
  [OK] taker: 359日 (99.7%), 4320件 (100.0%)
  [!!] orderbook: 179日 (49.7%), 2160件 (50.0%)
  [OK] premium: 359日 (99.7%), 4320件 (100.0%)

【データ品質】
  [OK] price: 欠損率: 0.0%, 重複率: 0.0%, 連続性: 100.0%, ギャップ: 0件
  [OK] oi: 欠損率: 0.0%, 重複率: 0.0%, 連続性: 100.0%, ギャップ: 0件
  [OK] funding: 欠損率: 0.0%, 重複率: 0.0%, 連続性: 100.0%, ギャップ: 0件
  [OK] liq: 欠損率: 0.0%, 重複率: 0.0%, 連続性: 100.0%, ギャップ: 0件
  [OK] lsr: 欠損率: 0.0%, 重複率: 0.0%, 連続性: 100.0%, ギャップ: 0件
  [OK] taker: 欠損率: 0.0%, 重複率: 0.0%, 連続性: 100.0%, ギャップ: 0件
  [OK] orderbook: 欠損率: 0.0%, 重複率: 0.0%, 連続性: 100.0%, ギャップ: 0件
  [OK] premium: 欠損率: 0.0%, 重複率: 0.0%, 連続性: 100.0%, ギャップ: 0件

【総合評価】
  [!!] 条件付き合格: 1件の警告がありました
================================================================================
```

## 完了条件

1. **main** の `stats.json`     → Sharpe 1–3 / DD ≤25 %
2. GitHub Actions ✅ 緑、`bt_with_price.html` artifact 付き
3. README に更新手順と KPI を記載
4. AI エージェントは **この文書外の構造変更をしてはいけない**。

このルールを守る限り、
"開発 ➜ BT ➜ 共有 ➜ レビュー ➜ マージ" のループで **出戻りゼロ** が保証されます。
