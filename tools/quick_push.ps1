#!/usr/bin/env pwsh
# 差分コミットと高速プッシュスクリプト

# コミットメッセージを取得
param(
    [Parameter(Mandatory=$true)]
    [string]$Message
)

# 作業ディレクトリをプロジェクトルートに変更
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
cd $rootDir

# ステータス表示
Write-Host "🔍 Git変更確認..." -ForegroundColor Cyan
git status -s

# 差分ステージング
Write-Host "➕ Git変更をステージング..." -ForegroundColor Cyan
git add .

# コミット
Write-Host "✅ コミット中: $Message" -ForegroundColor Green
git commit -m "$Message"

# プッシュ
Write-Host "🚀 変更をプッシュ中..." -ForegroundColor Yellow
git push

Write-Host "✨ 完了" -ForegroundColor Magenta
