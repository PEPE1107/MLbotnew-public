#!/usr/bin/env pwsh
<#
.SYNOPSIS
    MLbotnew用の一括Git pushスクリプト

.DESCRIPTION
    所定のファイルを自動的にステージング、コミット、プッシュします。
    - src/ 配下のソースコード
    - config/ 配下の設定ファイル
    - reports/*/stats.json の統計情報

.PARAMETER msg
    コミットメッセージ（必須）

.EXAMPLE
    ./scripts/quick_push.ps1 "[feat] ATR filter; 15m Sharpe 1.9, DD 19"

.NOTES
    このスクリプトは、開発→BT→共有→レビュー→マージの
    効率的なサイクルを実現するために設計されています。
#>

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$msg
)

# エラー処理設定
$ErrorActionPreference = "Stop"

# プロジェクトルートの確認
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath

# カレントディレクトリをプロジェクトルートに変更
Push-Location $projectRoot

try {
    # ブランチ名を取得
    $currentBranch = git branch --show-current
    
    if (-not $currentBranch) {
        Write-Error "現在のブランチを取得できませんでした。detachedステートでないか確認してください。"
        exit 1
    }
    
    Write-Host "【INFO】現在のブランチ: $currentBranch" -ForegroundColor Cyan
    
    # ファイルの追加（自動でステージング）
    Write-Host "【INFO】ファイルをステージングします..." -ForegroundColor Cyan
    git add src/ config/ reports/*/stats.json
    
    # 変更があるか確認
    $status = git status --porcelain
    
    if (-not $status) {
        Write-Host "【INFO】コミットする変更がありません。" -ForegroundColor Yellow
        exit 0
    }
    
    # 変更ファイル一覧を表示
    Write-Host "【INFO】以下のファイルがコミットされます:" -ForegroundColor Cyan
    git status --short
    
    # コミット
    Write-Host "【INFO】変更をコミットします: $msg" -ForegroundColor Cyan
    git commit -m $msg
    
    # プッシュ
    Write-Host "【INFO】変更をリモートにプッシュします..." -ForegroundColor Cyan
    git push -u origin $currentBranch
    
    # 完了メッセージ
    Write-Host "【SUCCESS】プッシュ完了: $currentBranch" -ForegroundColor Green
    Write-Host "  コミットメッセージ: $msg" -ForegroundColor Green
    
} catch {
    Write-Host "【ERROR】エラーが発生しました: $_" -ForegroundColor Red
    exit 1
} finally {
    # カレントディレクトリを元に戻す
    Pop-Location
}
