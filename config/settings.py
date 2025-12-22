"""
Configuracoes do EliBotCD
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega .env do diretorio raiz do projeto
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

# Credenciais FX Open
WEB_API_ID = os.getenv("FX_WEB_API_ID", "")
WEB_API_KEY = os.getenv("FX_WEB_API_KEY", "")
WEB_API_SECRET = os.getenv("FX_WEB_API_SECRET", "")

# URLs WebSocket
WS_FEED_URL = os.getenv("FX_WS_FEED_URL", "wss://marginalttdemowebapi.fxopen.net/feed")
WS_TRADE_URL = os.getenv("FX_WS_TRADE_URL", "wss://marginalttdemowebapi.fxopen.net/trade")

# Simbolo padrao
SYMBOL = os.getenv("FX_SYMBOL", "EURUSD")
