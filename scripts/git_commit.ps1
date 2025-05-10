# Very simple git commit and push script

# Change to project root directory
cd $(Split-Path -Parent $PSScriptRoot)

# Add all changes
git add .

# Commit with timestamp
git commit -m "Update to 2h Coinglass data only $(Get-Date -Format 'yyyyMMdd_HHmmss')"

# Push to repository
git push
