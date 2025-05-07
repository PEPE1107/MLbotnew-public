$env:PYTHONPATH="$env:APPDATA\Python\Python313\site-packages"
cd C:/Users/hiros/Desktop/MLbotnew

# 実行前の警告と確認
Write-Host "警告: このスクリプトはGitリポジトリの履歴を書き換えます。" -ForegroundColor Red
Write-Host "リモートリポジトリにプッシュ済みの場合、強制プッシュが必要になります。" -ForegroundColor Red
Write-Host "チーム開発環境では注意して使用してください。" -ForegroundColor Red
Write-Host ""

$confirm = Read-Host "実行しますか？ (y/n)"
if ($confirm -ne "y") {
    Write-Host "中止しました。" -ForegroundColor Yellow
    exit
}

# 機密ファイルと生成されたサンプルデータを履歴から削除
Write-Host "Git履歴から機密ファイルとサンプルデータを削除しています..." -ForegroundColor Cyan

# 以下のパスを含むコミットを除外 (機密ファイルとサンプルデータ)
python -m git_filter_repo --path config/api_keys.yaml --path "data/" --path ".env" --invert-paths --force

Write-Host "完了しました。" -ForegroundColor Green
Write-Host "注意: 既存のクローンで作業を続けるには 'git pull --rebase' が必要です。" -ForegroundColor Yellow
