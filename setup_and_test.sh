#!/bin/bash
# Script de instalação e teste do EliBotHFT

echo "=========================================="
echo "  EliBotHFT - Setup e Teste"
echo "=========================================="

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Diretório do projeto
PROJECT_DIR="$(dirname "$0")"
cd "$PROJECT_DIR"

echo ""
echo "[1/5] Verificando Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}[OK]${NC} $PYTHON_VERSION"
else
    echo -e "${RED}[ERRO]${NC} Python3 não encontrado!"
    exit 1
fi

echo ""
echo "[2/5] Instalando dependências essenciais via pip..."
python3 -m pip install --upgrade pip --quiet
python3 -m pip install \
    pydantic pydantic-settings python-dotenv colorlog \
    numpy pandas websockets aiohttp redis ntplib \
    scikit-learn joblib \
    --quiet 2>&1 | tail -3

echo -e "${GREEN}[OK]${NC} Dependências básicas instaladas"

echo ""
echo "[3/5] Testando imports básicos..."

python3 << 'PYTHON_TEST'
import sys
errors = []

# Teste 1: Config
try:
    from config import CONFIG, RISK_PARAMS
    print("[OK] config imports")
except Exception as e:
    errors.append(f"config: {e}")
    print(f"[ERRO] config: {e}")

# Teste 2: Logger
try:
    from utils.logger import setup_logger
    print("[OK] utils.logger imports")
except Exception as e:
    errors.append(f"utils.logger: {e}")
    print(f"[ERRO] utils.logger: {e}")

# Teste 3: API
try:
    from api import TickTraderFeed, TickTraderTrade
    print("[OK] api imports")
except Exception as e:
    errors.append(f"api: {e}")
    print(f"[ERRO] api: {e}")

if errors:
    print(f"\n[AVISO] {len(errors)} erro(s) encontrado(s)")
    sys.exit(1)
else:
    print("\n[OK] Todos os imports básicos funcionando!")
    sys.exit(0)
PYTHON_TEST

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}[AVISO]${NC} Alguns imports falharam. Verifique os erros acima."
fi

echo ""
echo "[4/5] Verificando configurações..."

python3 << 'PYTHON_CONFIG'
from config.settings import CONFIG

print(f"  Login: {CONFIG.LOGIN}")
print(f"  Server: {CONFIG.SERVER}")
print(f"  Symbol: {CONFIG.SYMBOL}")
print(f"  WS Feed URL: {CONFIG.WS_FEED_URL}")
print(f"  Token ID: {CONFIG.WEB_API_TOKEN_ID[:20]}...")

if not CONFIG.LOGIN or not CONFIG.WEB_API_TOKEN_ID:
    print("\n[AVISO] Credenciais não configuradas no .env!")
    exit(1)
else:
    print("\n[OK] Configurações carregadas!")
PYTHON_CONFIG

echo ""
echo "[5/5] Instruções para instalação completa..."
echo ""
echo "Para instalar TODAS as dependências (incluindo TA-Lib):"
echo ""
echo "  # Instalar TA-Lib (biblioteca C):"
echo "  wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
echo "  tar -xzf ta-lib-0.4.0-src.tar.gz"
echo "  cd ta-lib && ./configure --prefix=/usr && make && sudo make install"
echo ""
echo "  # Depois instalar o wrapper Python:"
echo "  python3 -m pip install TA-Lib"
echo ""
echo "  # Instalar todas as dependências do requirements.txt:"
echo "  python3 -m pip install -r requirements.txt"
echo ""
echo "=========================================="
echo "  Para executar o bot:"
echo "  python3 main.py --mode paper"
echo "=========================================="
