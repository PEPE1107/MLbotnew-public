import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# プロジェクトルートディレクトリ
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = ROOT_DIR / 'config'

# Load environment variables from .env file
load_dotenv()

# Coinglass API key
CG_API = os.getenv("CG_API_KEY")

# Bybit API credentials
BYBIT_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET = os.getenv("BYBIT_API_SECRET")

# Slack webhook
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")

# システム設定の読み込み
def load_system_config():
    """システム設定をロードする"""
    try:
        with open(CONFIG_DIR / 'system.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"システム設定の読み込みに失敗しました: {e}")
        # デフォルト設定を返す
        return {
            "data": {
                "use_real_data": True,
                "max_samples_per_interval": 4320,
                "log_level": "INFO"
            }
        }

# システム設定を読み込む
SYSTEM_CONFIG = load_system_config()

# データソース設定
USE_REAL_DATA = SYSTEM_CONFIG["data"]["use_real_data"]
MAX_SAMPLES = SYSTEM_CONFIG["data"]["max_samples_per_interval"]
DATA_LOG_LEVEL = SYSTEM_CONFIG["data"]["log_level"]
