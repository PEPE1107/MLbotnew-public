$env:PYTHONPATH="$env:APPDATA\Python\Python313\site-packages"
cd C:/Users/hiros/Desktop/MLbotnew
python -m git_filter_repo --path config/api_keys.yaml --invert-paths --force
