"""
Módulo para buscar dados históricos da FX Open
"""
import aiohttp
import hmac
import hashlib
import base64
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional

from config import settings


class HistoricalData:
    """Busca dados históricos via REST API da FX Open"""

    def __init__(self):
        self.base_url = "https://marginalttdemowebapi.fxopen.net"

    def _generate_signature(self, timestamp: str) -> str:
        """Gera assinatura HMAC"""
        message = f"{timestamp}{settings.WEB_API_ID}{settings.WEB_API_KEY}"
        signature = base64.b64encode(
            hmac.new(
                settings.WEB_API_SECRET.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        return signature

    def _get_headers(self) -> dict:
        """Gera headers de autenticação"""
        timestamp = str(int(time.time() * 1000))
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Auth": "HMAC",
            "WebApiId": settings.WEB_API_ID,
            "WebApiKey": settings.WEB_API_KEY,
            "Timestamp": timestamp,
            "Signature": self._generate_signature(timestamp)
        }

    async def get_bars(
        self,
        symbol: str = "EURUSD",
        periodicity: str = "M1",  # M1, M5, M15, M30, H1, H4, D1
        bars_count: int = 500
    ) -> Optional[pd.DataFrame]:
        """
        Busca barras/candles históricos

        Periodicidade:
        - S1, S10: segundos
        - M1, M5, M15, M30: minutos
        - H1, H4: horas
        - D1, W1, MN1: dia, semana, mês
        """
        try:
            # Calcula timestamp de início (X barras atrás)
            now = datetime.now(timezone.utc)

            # Endpoint para quotehistory
            url = f"{self.base_url}/api/v2/quotehistory/{symbol}/{periodicity}/bars/ask"
            params = {
                "count": bars_count,
                "timestamp": int(now.timestamp() * 1000)
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    ssl=False
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        if 'Bars' in data and len(data['Bars']) > 0:
                            bars = data['Bars']

                            df = pd.DataFrame(bars)
                            df['timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
                            df = df.rename(columns={
                                'Open': 'open',
                                'High': 'high',
                                'Low': 'low',
                                'Close': 'close',
                                'Volume': 'volume'
                            })
                            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                            df = df.sort_values('timestamp').reset_index(drop=True)

                            return df
                        else:
                            print(f"Sem dados de barras: {data}")
                            return None
                    else:
                        text = await response.text()
                        print(f"Erro na API ({response.status}): {text}")
                        return None

        except Exception as e:
            print(f"Erro ao buscar dados históricos: {e}")
            return None

    async def get_ticks(
        self,
        symbol: str = "EURUSD",
        count: int = 1000
    ) -> Optional[pd.DataFrame]:
        """Busca ticks históricos"""
        try:
            now = datetime.now(timezone.utc)
            url = f"{self.base_url}/api/v2/quotehistory/{symbol}/ticks"
            params = {
                "count": count,
                "timestamp": int(now.timestamp() * 1000)
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    ssl=False
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        if 'Ticks' in data and len(data['Ticks']) > 0:
                            ticks = data['Ticks']

                            df = pd.DataFrame(ticks)
                            df['timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)

                            # Extrai bid/ask
                            df['bid'] = df['BestBid'].apply(lambda x: x.get('Price', 0) if x else 0)
                            df['ask'] = df['BestAsk'].apply(lambda x: x.get('Price', 0) if x else 0)
                            df['mid'] = (df['bid'] + df['ask']) / 2

                            df = df[['timestamp', 'bid', 'ask', 'mid']]
                            df = df.sort_values('timestamp').reset_index(drop=True)

                            return df
                        else:
                            print(f"Sem dados de ticks")
                            return None
                    else:
                        text = await response.text()
                        print(f"Erro na API ({response.status}): {text}")
                        return None

        except Exception as e:
            print(f"Erro ao buscar ticks históricos: {e}")
            return None
