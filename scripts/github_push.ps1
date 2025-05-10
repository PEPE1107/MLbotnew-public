# GitHub自動プッシュスクリプト
# MLbotnew - 2h Coinglassデータで構築されたBTC取引ボット

$date = Get-Date -Format "yyyyMMdd_HHmmss"
$commit_message = "[refactor] 2時間足Coinglassデータのみを使用するように変更 ($date)"

# カレントディレクトリをプロジェクトルートに変更
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

# GitHubリポジトリが存在するか確認
if (-not (Test-Path ".git")) {
    Write-Host "GitHubリポジトリが見つかりません。初期化します..." -ForegroundColor Yellow
    
    # Gitリポジトリを初期化
    git init
    
    # 最初のコミット
    git add .
    git commit -m "Initial commit: 2h Coinglass only BTC trading bot"
    
    # ユーザーにURLの入力を求める
    $repo_url = Read-Host "GitHubリポジトリのURLを入力してください (例: https://github.com/username/MLbotnew.git)"
    
    # リモートを追加
    git remote add origin $repo_url
} 
else {
    Write-Host "既存のGitHubリポジトリが見つかりました。変更をプッシュします..." -ForegroundColor Green
}

# 状態確認
git status

# 変更をすべて追加
git add .

# コミット
git commit -m $commit_message

# プッシュ
Write-Host "変更をGitHubにプッシュしています..." -ForegroundColor Cyan
git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "プッシュが成功しました！" -ForegroundColor Green
    Write-Host "GitHub上でMLbotnewプロジェクトが更新されました。" -ForegroundColor Green
} 
else {
    Write-Host "プッシュ中にエラーが発生しました。" -ForegroundColor Red
    Write-Host "エラーコード: $LASTEXITCODE" -ForegroundColor Red
    
    # mainブランチが存在しない場合の対応
    $choice = Read-Host "mainブランチがない場合は、masterブランチにプッシュしますか？ (y/n)"
    if ($choice -eq "y") {
        git push origin master
        if ($LASTEXITCODE -eq 0) {
            Write-Host "masterブランチへのプッシュが成功しました！" -ForegroundColor Green
        } 
        else {
            Write-Host "masterブランチへのプッシュも失敗しました。手動で解決してください。" -ForegroundColor Red
        }
    }
}

Write-Host "完了！" -ForegroundColor Green
