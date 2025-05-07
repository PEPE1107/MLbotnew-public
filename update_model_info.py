import pandas as pd
import json

df = pd.read_parquet("data/features/15m/X.parquet")
data = {
    "model_type": "lightgbm",
    "date": "20250503",
    "feature_cols": df.columns.tolist()
}

with open("models/model_info_124041.json", "w") as f:
    json.dump(data, f, indent=2)

