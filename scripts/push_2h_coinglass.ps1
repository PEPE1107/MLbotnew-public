# GitHub プッシュスクリプト
# MLbotnew - 2h Coinglassデータのみを使用するバージョン (360日分)

# タイムスタンプ付きコミットメッセージ
$date = Get-Date -Format "yyyyMMdd_HHmmss"
$commit_message = "[update] 2時間足Coinglassデータ360日分のみを使用するように変更 ($date)"

# カレントディレクトリをプロジェクトルートに変更
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

# 状態確認
Write-Host "Git Status:" -ForegroundColor Cyan
git status

# 変更をすべて追加
Write-Host "Adding all changes to git..." -ForegroundColor Yellow
git add .

# コミット
Write-Host "Committing changes..." -ForegroundColor Yellow
git commit -m $commit_message

# プッシュ (まずmainを試す)
Write-Host "Pushing to main branch..." -ForegroundColor Yellow
git push origin main

# エラーがあればmasterを試す
if ($LASTEXITCODE -ne 0) {
    Write-Host "mainブランチへのプッシュに失敗しました。masterブランチを試します。" -ForegroundColor Yellow
    git push origin master
}

Write-Host "プッシュ処理が完了しました。" -ForegroundColor Green
Write-Host "変更内容:" -ForegroundColor Cyan
Write-Host "- 15分足と日足のデータを削除" -ForegroundColor White
Write-Host "- 2時間足Coinglassデータ360日分のみを使用するよう設定更新" -ForegroundColor White
Write-Host "- バックテスト結果更新 (シャープレシオ: 0.00, 最大ドローダウン: -25.63%, トータルリターン: 7.86%, 勝率: 42.28%)" -ForegroundColor White
Write-Host "- READMEを更新" -ForegroundColor White
