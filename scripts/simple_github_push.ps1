# GitHub簡易プッシュスクリプト
# MLbotnew - 2h Coinglassデータのみを使用するように変更

# タイムスタンプ付きコミットメッセージ
$date = Get-Date -Format "yyyyMMdd_HHmmss"
$commit_message = "[refactor] 2時間足Coinglassデータのみを使用するように変更 ($date)"

# カレントディレクトリをプロジェクトルートに変更
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

# 状態確認
git status

# 変更をすべて追加
git add .

# コミット
git commit -m $commit_message

# プッシュ (まずmainを試す)
git push origin main

# エラーがあればmasterを試す
if ($LASTEXITCODE -ne 0) {
    Write-Host "mainブランチへのプッシュに失敗しました。masterブランチを試します。" -ForegroundColor Yellow
    git push origin master
}

Write-Host "プッシュ処理が完了しました。" -ForegroundColor Green
