# Quick Push Script for MLbotnew
# Usage: ./tools/quick_push.ps1 "[feat] ATR filter; 15m Sharpe 1.9, DD 19"

param([string]$msg)

# 引数チェック
if ([string]::IsNullOrEmpty($msg)) {
    Write-Host "エラー: コミットメッセージを指定してください" -ForegroundColor Red
    Write-Host "使用例: ./tools/quick_push.ps1 `"[feat] ATR filter; 15m Sharpe 1.9, DD 19`"" -ForegroundColor Yellow
    exit 1
}

# KPIが含まれているかチェック
if ($msg -notmatch "Sharpe|シャープ|DD|ドローダウン") {
    Write-Host "警告: メッセージにKPI情報が含まれていません (Sharpe/DD)" -ForegroundColor Yellow
    $continue = Read-Host "続行しますか？ [y/N]"
    if ($continue -ne "y") {
        exit 0
    }
}

Write-Host "以下のファイルを追加します:" -ForegroundColor Cyan
Write-Host "- src/ (コードファイル)" -ForegroundColor Green
Write-Host "- config/ (設定ファイル)" -ForegroundColor Green
Write-Host "- reports/*/stats.json (統計レポート)" -ForegroundColor Green

# Git add
git add src/ config/ reports/*/stats.json

# 変更されたファイルを表示
Write-Host "`n変更されたファイル:" -ForegroundColor Cyan
git status --short

# コミット確認
$confirmation = Read-Host "`nこれらの変更を`"$msg`"というメッセージでコミットしますか？ [y/N]"
if ($confirmation -ne "y") {
    Write-Host "コミットをキャンセルしました" -ForegroundColor Yellow
    exit 0
}

# コミット
git commit -m $msg
if ($LASTEXITCODE -ne 0) {
    Write-Host "コミットに失敗しました" -ForegroundColor Red
    exit 1
}

# 現在のブランチを取得
$branch = git branch --show-current
Write-Host "`n現在のブランチ: $branch" -ForegroundColor Cyan

# プッシュ確認
$push = Read-Host "変更を origin/$branch にプッシュしますか？ [y/N]"
if ($push -ne "y") {
    Write-Host "プッシュをキャンセルしました。ローカルにのみコミットされました。" -ForegroundColor Yellow
    exit 0
}

# プッシュ
git push -u origin $branch
if ($LASTEXITCODE -ne 0) {
    Write-Host "プッシュに失敗しました" -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ 変更が正常にプッシュされました！" -ForegroundColor Green
Write-Host "ブランチ: $branch" -ForegroundColor Cyan
Write-Host "メッセージ: $msg" -ForegroundColor Cyan

# PR作成の案内
Write-Host "`nGitHubでPRを作成する場合:" -ForegroundColor Magenta
Write-Host "https://github.com/your-org/MLbotnew/compare/$branch`?expand=1" -ForegroundColor Blue
