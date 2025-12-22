# backtest/historical_data.py
"""
Cliente para download de dados históricos da API TickTrader
"""
import asyncio
import websockets
import json
import hmac
import hashlib
import base64
import time
import ssl
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config.settings import CONFIG
from utils.logger import setup_logger

logger = setup_logger("historical_data")


class HistoricalDataClient:
    """Cliente para obter dados históricos da API TickTrader"""

    def __init__(self):
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.connected: bool = False
        self.responses: Dict[str, Any] = {}
        self.request_id: int = 0
        self.data_dir = Path("data/historical")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_next_id(self) -> str:
        self.request_id += 1
        return f"hist_{self.request_id}"

    def _generate_signature(self, timestamp: str) -> str:
        """Gera assinatura HMAC-SHA256 em Base64"""
        message = f"{timestamp}{CONFIG.WEB_API_TOKEN_ID}{CONFIG.WEB_API_TOKEN_KEY}"
        signature = base64.b64encode(
            hmac.new(
                CONFIG.WEB_API_TOKEN_SECRET.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        return signature

    async def connect(self) -> bool:
        """Conecta e autentica no WebSocket"""
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            self.ws = await websockets.connect(
                CONFIG.WS_FEED_URL,
                ssl=ssl_context,
                ping_interval=30,
                ping_timeout=10
            )

            # Autenticar
            timestamp = str(int(time.time() * 1000))
            auth_msg = {
                "Id": self._get_next_id(),
                "Request": "Login",
                "Params": {
                    "AuthType": "HMAC",
                    "WebApiId": CONFIG.WEB_API_TOKEN_ID,
                    "WebApiKey": CONFIG.WEB_API_TOKEN_KEY,
                    "Timestamp": timestamp,
                    "Signature": self._generate_signature(timestamp)
                }
            }

            await self.ws.send(json.dumps(auth_msg))
            response = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            data = json.loads(response)

            if data.get('Result', {}).get('Info') == 'ok':
                self.connected = True
                logger.info("Conectado ao Feed WebSocket para dados históricos")
                return True
            else:
                logger.error(f"Falha na autenticação: {data}")
                return False

        except Exception as e:
            logger.error(f"Erro ao conectar: {e}")
            return False

    async def disconnect(self):
        """Desconecta do WebSocket"""
        if self.ws:
            await self.ws.close()
        self.connected = False
        logger.info("Desconectado do Feed WebSocket")

    async def _send_and_wait(self, message: Dict, timeout: float = 30.0) -> Optional[Dict]:
        """Envia mensagem e aguarda resposta"""
        if not self.ws or not self.connected:
            return None

        msg_id = message['Id']
        await self.ws.send(json.dumps(message))

        try:
            while True:
                response = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
                data = json.loads(response)

                if data.get('Id') == msg_id:
                    return data

        except asyncio.TimeoutError:
            logger.warning(f"Timeout aguardando resposta para {msg_id}")
            return None

    async def get_available_symbols(self) -> List[str]:
        """Obtém lista de símbolos disponíveis para dados históricos"""
        msg = {
            "Id": self._get_next_id(),
            "Request": "QuoteHistorySymbols",
            "Params": {}
        }

        response = await self._send_and_wait(msg)
        if response and 'Result' in response:
            return response['Result']
        return []

    async def get_available_periodicities(self) -> List[str]:
        """Obtém lista de periodicidades disponíveis"""
        msg = {
            "Id": self._get_next_id(),
            "Request": "QuoteHistoryPeriodicities",
            "Params": {}
        }

        response = await self._send_and_wait(msg)
        if response and 'Result' in response:
            return response['Result']
        return []

    async def get_bars(self, symbol: str, periodicity: str,
                       count: int = 1000,
                       end_time: Optional[datetime] = None,
                       price_type: str = "Bid") -> pd.DataFrame:
        """
        Obtém barras históricas

        Args:
            symbol: Símbolo (ex: "EURUSD")
            periodicity: Periodicidade (ex: "M1", "M5", "H1", "D1")
            count: Número de barras (negativo para barras anteriores)
            end_time: Timestamp final (default: agora)
            price_type: Tipo de preço ("Bid" ou "Ask")

        Returns:
            DataFrame com colunas: timestamp, open, high, low, close, volume
        """
        if end_time is None:
            end_time = datetime.now(timezone.utc)

        timestamp_ms = int(end_time.timestamp() * 1000)

        msg = {
            "Id": self._get_next_id(),
            "Request": "QuoteHistoryBars",
            "Params": {
                "Symbol": symbol,
                "Periodicity": periodicity,
                "PriceType": price_type,
                "Timestamp": timestamp_ms,
                "Count": -abs(count)  # Negativo para barras anteriores
            }
        }

        response = await self._send_and_wait(msg, timeout=60.0)

        if response and 'Result' in response and not response.get('Error'):
            result = response['Result']
            # A API retorna { Symbol, Bars: [...] }
            bars = result.get('Bars', []) if isinstance(result, dict) else result
            if isinstance(bars, list) and len(bars) > 0:
                df = pd.DataFrame(bars)

                # Renomear colunas se necessário
                column_map = {
                    'Timestamp': 'timestamp',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }
                df = df.rename(columns=column_map)

                # Converter timestamp para datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                    df = df.sort_values('timestamp').reset_index(drop=True)

                logger.info(f"Obtidas {len(df)} barras de {symbol} {periodicity}")
                return df

        error = response.get('Error') if response else "Sem resposta"
        logger.error(f"Erro ao obter barras: {error}")
        return pd.DataFrame()

    async def download_historical_data(self, symbol: str, periodicity: str,
                                       days: int = 30) -> pd.DataFrame:
        """
        Baixa dados históricos para um período específico

        Args:
            symbol: Símbolo
            periodicity: Periodicidade
            days: Número de dias de dados

        Returns:
            DataFrame completo com todos os dados
        """
        all_bars = []
        end_time = datetime.now(timezone.utc)

        # Calcular quantas barras por request
        bars_per_request = 1000

        # Calcular período de cada barra em minutos
        period_minutes = {
            'S1': 1/60, 'S10': 10/60,
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240,
            'D1': 1440, 'W1': 10080, 'MN1': 43200
        }.get(periodicity, 1)

        # Total de barras necessárias
        total_bars_needed = int((days * 24 * 60) / period_minutes)
        bars_downloaded = 0

        logger.info(f"Baixando ~{total_bars_needed} barras de {symbol} {periodicity} ({days} dias)")

        while bars_downloaded < total_bars_needed:
            df = await self.get_bars(symbol, periodicity, bars_per_request, end_time)

            if df.empty:
                break

            all_bars.append(df)
            bars_downloaded += len(df)

            # Próximo request começa antes da primeira barra obtida
            if 'timestamp' in df.columns:
                end_time = df['timestamp'].min() - timedelta(minutes=period_minutes)

            logger.info(f"  Progresso: {bars_downloaded}/{total_bars_needed} barras")

            await asyncio.sleep(0.5)  # Rate limiting

        if all_bars:
            result = pd.concat(all_bars, ignore_index=True)
            result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

            # Salvar em arquivo
            filename = self.data_dir / f"{symbol}_{periodicity}_{days}d.parquet"
            result.to_parquet(filename)
            logger.info(f"Dados salvos em {filename}")

            return result

        return pd.DataFrame()

    def load_cached_data(self, symbol: str, periodicity: str, days: int) -> Optional[pd.DataFrame]:
        """Carrega dados do cache se disponível"""
        filename = self.data_dir / f"{symbol}_{periodicity}_{days}d.parquet"
        if filename.exists():
            df = pd.read_parquet(filename)
            logger.info(f"Dados carregados do cache: {filename}")
            return df
        return None


async def main():
    """Teste do cliente de dados históricos"""
    client = HistoricalDataClient()

    if await client.connect():
        # Testar download de dados
        df = await client.get_bars("EURUSD", "M1", count=100)
        print(f"\nBarras M1 obtidas: {len(df)}")
        if not df.empty:
            print(df.head())
            print(df.tail())

        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
