import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Coinglass API key
CG_API = os.getenv("CG_API_KEY")

# Bybit API credentials
BYBIT_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET = os.getenv("BYBIT_API_SECRET")

# Slack webhook
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")
