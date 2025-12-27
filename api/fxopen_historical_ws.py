#!/usr/bin/env python3
"""
================================================================================
FXOPEN HISTORICAL DATA CLIENT - WebSocket
Cliente para Download de Dados Historicos REAIS via WebSocket
================================================================================

Este modulo baixa dados historicos REAIS do mercado Forex via WebSocket.
IMPORTANTE: Todos os dados sao reais - nenhuma simulacao e permitida.

Baseado no protocolo QuoteHistoryBars da FXOpen API.
"""

import asyncio
import websockets
import json
import hmac
import hashlib
import base64
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict
from dataclasses import dataclass

# Configuracoes WebSocket
WS_FEED_URL = "wss://marginalttdemowebapi.fxopen.net/feed"
DEFAULT_TOKEN_ID = "0473113a-f96d-4576-bd1b-507e71ec3d4f"
DEFAULT_TOKEN_KEY = "EGqeZPpJQSW2BjCb"
DEFAULT_TOKEN_SECRET = "YdafQEND2Fnrc5JGryX6ZPCJ5pf9rmyHnAk6wTDjWGddcRjWtxw369YhKzkBzPkM"


@dataclass
class Bar:
    """
    Representa uma barra/candle OHLCV com dados de BID e ASK

    Campos principais (BID por padrão para compatibilidade):
        open, high, low, close, volume

    Campos de spread real (opcionais):
        bid_open, bid_high, bid_low, bid_close
        ask_open, ask_high, ask_low, ask_close
        spread_open, spread_close (em pips)
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Dados de BID (opcionais - para spread real)
    bid_open: Optional[float] = None
    bid_high: Optional[float] = None
    bid_low: Optional[float] = None
    bid_close: Optional[float] = None

    # Dados de ASK (opcionais - para spread real)
    ask_open: Optional[float] = None
    ask_high: Optional[float] = None
    ask_low: Optional[float] = None
    ask_close: Optional[float] = None

    @property
    def has_spread_data(self) -> bool:
        """Verifica se tem dados de spread real"""
        return self.bid_close is not None and self.ask_close is not None

    @property
    def spread_pips(self) -> float:
        """Retorna o spread real em pips (baseado no close)"""
        if self.has_spread_data:
            return (self.ask_close - self.bid_close) * 10000
        return 0.0

    @property
    def spread_open_pips(self) -> float:
        """Retorna o spread real no open em pips"""
        if self.bid_open is not None and self.ask_open is not None:
            return (self.ask_open - self.bid_open) * 10000
        return 0.0

    @property
    def mid_open(self) -> float:
        """Preço médio no open"""
        if self.bid_open is not None and self.ask_open is not None:
            return (self.bid_open + self.ask_open) / 2
        return self.open

    @property
    def mid_close(self) -> float:
        """Preço médio no close"""
        if self.bid_close is not None and self.ask_close is not None:
            return (self.bid_close + self.ask_close) / 2
        return self.close

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    def __str__(self) -> str:
        spread_str = f" Spread:{self.spread_pips:.1f}p" if self.has_spread_data else ""
        return (f"{self.timestamp.strftime('%Y-%m-%d %H:%M')} | "
                f"O:{self.open:.5f} H:{self.high:.5f} L:{self.low:.5f} C:{self.close:.5f} V:{self.volume:.0f}{spread_str}")


class FXOpenHistoricalClient:
    """
    Cliente WebSocket para download de dados historicos REAIS da FXOpen API

    IMPORTANTE: Este cliente baixa APENAS dados REAIS do mercado.
    Nenhuma simulacao ou dados sinteticos sao utilizados.
    """

    # Mapeamento de periodicidade para segundos
    PERIODICITY_SECONDS = {
        'S1': 1,
        'S10': 10,
        'M1': 60,
        'M5': 300,
        'M15': 900,
        'M30': 1800,
        'H1': 3600,
        'H4': 14400,
        'D1': 86400,
        'W1': 604800,
        'MN1': 2592000
    }

    def __init__(self,
                 token_id: str = DEFAULT_TOKEN_ID,
                 token_key: str = DEFAULT_TOKEN_KEY,
                 token_secret: str = DEFAULT_TOKEN_SECRET,
                 ws_url: str = WS_FEED_URL):
        """
        Inicializa o cliente

        Args:
            token_id: ID do token da API
            token_key: Chave do token
            token_secret: Segredo do token
            ws_url: URL do WebSocket
        """
        self.token_id = token_id
        self.token_key = token_key
        self.token_secret = token_secret
        self.ws_url = ws_url

        self._ws = None
        self._request_id = 0
        self._pending_requests: Dict[str, asyncio.Future] = {}

    def _generate_signature(self, timestamp: str) -> str:
        """Gera assinatura HMAC-SHA256 para autenticacao"""
        message = f"{timestamp}{self.token_id}{self.token_key}"
        signature = base64.b64encode(
            hmac.new(
                self.token_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        return signature

    def _next_id(self) -> str:
        """Gera proximo ID de requisicao"""
        self._request_id += 1
        return f"hist_{self._request_id}"

    async def connect(self):
        """Conecta ao WebSocket e autentica"""
        print(f"  Conectando ao WebSocket FXOpen...")

        self._ws = await websockets.connect(
            self.ws_url,
            ping_interval=30,
            ping_timeout=10,
            close_timeout=10
        )

        # Autentica
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp)

        auth_request = {
            "Id": self._next_id(),
            "Request": "Login",
            "Params": {
                "AuthType": "HMAC",
                "WebApiId": self.token_id,
                "WebApiKey": self.token_key,
                "Timestamp": timestamp,
                "Signature": signature,
                "DeviceId": "backtest-client",
                "AppSessionId": f"backtest-{int(time.time())}"
            }
        }

        await self._ws.send(json.dumps(auth_request))
        response = await self._ws.recv()
        response_data = json.loads(response)

        if response_data.get("Response") == "Login" and "Result" in response_data:
            print(f"  Autenticado com sucesso!")
            return True
        else:
            error = response_data.get("Error", {}).get("Message", "Erro desconhecido")
            print(f"  Erro na autenticacao: {error}")
            return False

    async def disconnect(self):
        """Desconecta do WebSocket"""
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def get_bars(self,
                      symbol: str,
                      periodicity: str,
                      timestamp: int,
                      count: int = -1000,
                      price_type: str = "bid") -> List[Bar]:
        """
        Obtem barras historicas via WebSocket

        Args:
            symbol: Simbolo (ex: 'EURUSD')
            periodicity: Periodicidade (M1, H1, D1, etc)
            timestamp: Timestamp em milliseconds
            count: Numero de barras (negativo = antes do timestamp)
            price_type: 'bid' ou 'ask'

        Returns:
            Lista de barras reais
        """
        request_id = self._next_id()

        request = {
            "Id": request_id,
            "Request": "QuoteHistoryBars",
            "Params": {
                "Symbol": symbol,
                "Periodicity": periodicity,
                "PriceType": price_type,
                "Timestamp": timestamp,
                "Count": count
            }
        }

        await self._ws.send(json.dumps(request))

        # Aguarda resposta
        while True:
            response = await self._ws.recv()
            data = json.loads(response)

            if data.get("Id") == request_id:
                break

        bars = []

        if "Result" in data and "Bars" in data["Result"]:
            for bar_data in data["Result"]["Bars"]:
                try:
                    ts = datetime.fromtimestamp(bar_data["Timestamp"] / 1000, tz=timezone.utc)
                    bar = Bar(
                        timestamp=ts,
                        open=float(bar_data["Open"]),
                        high=float(bar_data["High"]),
                        low=float(bar_data["Low"]),
                        close=float(bar_data["Close"]),
                        volume=float(bar_data.get("Volume", 0))
                    )
                    bars.append(bar)
                except Exception as e:
                    continue

        return bars

    async def download_range(self,
                            symbol: str,
                            periodicity: str,
                            start_date: datetime,
                            end_date: datetime,
                            price_type: str = "bid") -> List[Bar]:
        """
        Baixa barras historicas REAIS para um periodo

        IMPORTANTE: Estes sao dados REAIS do mercado Forex.

        Args:
            symbol: Simbolo (ex: 'EURUSD')
            periodicity: Periodicidade (M1, H1, D1, etc)
            start_date: Data/hora de inicio
            end_date: Data/hora de fim
            price_type: 'bid' ou 'ask'

        Returns:
            Lista de barras reais ordenadas por timestamp
        """
        # Garante timezone UTC
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        all_bars = []
        current_ts = end_ts
        batch_count = 0

        print(f"  Baixando barras de {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}...")

        while current_ts > start_ts:
            batch_count += 1
            bars = await self.get_bars(
                symbol=symbol,
                periodicity=periodicity,
                timestamp=current_ts,
                count=-1000,
                price_type=price_type
            )

            if not bars:
                break

            # Filtra barras dentro do periodo
            valid_bars = [b for b in bars if start_ts <= int(b.timestamp.timestamp() * 1000) <= end_ts]
            all_bars.extend(valid_bars)

            # Atualiza timestamp para proxima iteracao
            oldest_bar = min(bars, key=lambda b: b.timestamp)
            new_ts = int(oldest_bar.timestamp.timestamp() * 1000) - 1

            if new_ts >= current_ts:
                break

            current_ts = new_ts

            print(f"    Batch {batch_count}: {len(valid_bars)} barras (total: {len(all_bars)})")

            # Rate limiting
            await asyncio.sleep(0.1)

        # Remove duplicatas e ordena
        seen = set()
        unique_bars = []
        for bar in all_bars:
            ts_key = int(bar.timestamp.timestamp())
            if ts_key not in seen:
                seen.add(ts_key)
                unique_bars.append(bar)

        unique_bars.sort(key=lambda b: b.timestamp)

        return unique_bars


async def download_historical_data(
    symbol: str,
    periodicity: str,
    start_time: datetime,
    end_time: datetime = None,
    price_type: str = "bid"
) -> List[Bar]:
    """
    Funcao principal para download de dados historicos REAIS

    IMPORTANTE: Todos os dados sao REAIS do mercado Forex.

    Args:
        symbol: Simbolo (ex: 'EURUSD')
        periodicity: Periodicidade (M1, H1, D1, etc)
        start_time: Data/hora de inicio
        end_time: Data/hora de fim (default: agora)
        price_type: 'bid' ou 'ask'

    Returns:
        Lista de barras reais
    """
    if end_time is None:
        end_time = datetime.now(timezone.utc)

    print(f"\n{'='*60}")
    print(f"  DOWNLOAD DE DADOS HISTORICOS REAIS (FXOpen WebSocket)")
    print(f"{'='*60}")
    print(f"  Simbolo: {symbol}")
    print(f"  Periodicidade: {periodicity}")
    print(f"  Periodo: {start_time.strftime('%Y-%m-%d %H:%M')} a {end_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Tipo de preco: {price_type}")
    print(f"{'='*60}\n")

    client = FXOpenHistoricalClient()

    try:
        connected = await client.connect()
        if not connected:
            print("  ERRO: Falha na conexao")
            return []

        bars = await client.download_range(
            symbol=symbol,
            periodicity=periodicity,
            start_date=start_time,
            end_date=end_time,
            price_type=price_type
        )

        print(f"\n  Total de barras baixadas: {len(bars)}")
        if bars:
            print(f"  Primeira barra: {bars[0].timestamp}")
            print(f"  Ultima barra: {bars[-1].timestamp}")

        return bars

    finally:
        await client.disconnect()


def get_historical_data_sync(
    symbol: str,
    periodicity: str,
    start_time: datetime,
    end_time: datetime = None,
    price_type: str = "bid"
) -> List[Bar]:
    """
    Versao sincrona do download de dados historicos

    Wrapper para uso em codigo sincrono.

    Args:
        symbol: Simbolo (ex: 'EURUSD')
        periodicity: Periodicidade (M1, H1, D1, etc)
        start_time: Data/hora de inicio
        end_time: Data/hora de fim
        price_type: 'bid' ou 'ask'

    Returns:
        Lista de barras reais
    """
    return asyncio.run(download_historical_data(
        symbol=symbol,
        periodicity=periodicity,
        start_time=start_time,
        end_time=end_time,
        price_type=price_type
    ))


# ==============================================================================
# DOWNLOAD COM SPREAD REAL (BID + ASK)
# ==============================================================================

async def download_historical_data_with_spread(
    symbol: str,
    periodicity: str,
    start_time: datetime,
    end_time: datetime = None
) -> List[Bar]:
    """
    Baixa dados historicos com SPREAD REAL (BID e ASK separados)

    Esta funcao baixa tanto dados BID quanto ASK e mescla em barras
    que contem o spread real para cada momento.

    IMPORTANTE: Demora aproximadamente 2x mais que download normal.

    Args:
        symbol: Simbolo (ex: 'EURUSD')
        periodicity: Periodicidade (M1, H1, D1, etc)
        start_time: Data/hora de inicio
        end_time: Data/hora de fim (default: agora)

    Returns:
        Lista de barras com spread real (bid_*, ask_*, spread_pips)
    """
    if end_time is None:
        end_time = datetime.now(timezone.utc)

    print(f"\n{'='*70}")
    print(f"  DOWNLOAD COM SPREAD REAL (BID + ASK)")
    print(f"{'='*70}")
    print(f"  Simbolo: {symbol}")
    print(f"  Periodicidade: {periodicity}")
    print(f"  Periodo: {start_time.strftime('%Y-%m-%d %H:%M')} a {end_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    client = FXOpenHistoricalClient()

    try:
        connected = await client.connect()
        if not connected:
            print("  ERRO: Falha na conexao")
            return []

        # Baixa dados BID
        print("  [1/2] Baixando dados BID...")
        bid_bars = await client.download_range(
            symbol=symbol,
            periodicity=periodicity,
            start_date=start_time,
            end_date=end_time,
            price_type="bid"
        )
        print(f"        Barras BID: {len(bid_bars)}")

        # Baixa dados ASK
        print("  [2/2] Baixando dados ASK...")
        ask_bars = await client.download_range(
            symbol=symbol,
            periodicity=periodicity,
            start_date=start_time,
            end_date=end_time,
            price_type="ask"
        )
        print(f"        Barras ASK: {len(ask_bars)}")

        if not bid_bars or not ask_bars:
            print("  ERRO: Falha ao baixar dados")
            return []

        # Mescla BID e ASK por timestamp
        print("  Mesclando dados BID/ASK...")
        bid_dict = {int(b.timestamp.timestamp()): b for b in bid_bars}
        ask_dict = {int(a.timestamp.timestamp()): a for a in ask_bars}

        # Encontra timestamps comuns
        common_timestamps = sorted(set(bid_dict.keys()) & set(ask_dict.keys()))

        merged_bars = []
        spreads = []

        for ts in common_timestamps:
            bid = bid_dict[ts]
            ask = ask_dict[ts]

            # Cria barra com spread real
            bar = Bar(
                timestamp=bid.timestamp,
                # Campos principais = BID (padrao)
                open=bid.open,
                high=bid.high,
                low=bid.low,
                close=bid.close,
                volume=bid.volume,
                # Dados BID
                bid_open=bid.open,
                bid_high=bid.high,
                bid_low=bid.low,
                bid_close=bid.close,
                # Dados ASK
                ask_open=ask.open,
                ask_high=ask.high,
                ask_low=ask.low,
                ask_close=ask.close
            )

            merged_bars.append(bar)
            spreads.append(bar.spread_pips)

        print(f"\n  Barras mescladas: {len(merged_bars)}")

        if spreads:
            import numpy as np
            spreads_arr = np.array(spreads)
            print(f"\n  ESTATISTICAS DO SPREAD:")
            print(f"    Minimo:  {spreads_arr.min():.2f} pips")
            print(f"    Maximo:  {spreads_arr.max():.2f} pips")
            print(f"    Media:   {spreads_arr.mean():.2f} pips")
            print(f"    Mediana: {np.median(spreads_arr):.2f} pips")

        return merged_bars

    finally:
        await client.disconnect()


def get_historical_data_with_spread_sync(
    symbol: str,
    periodicity: str,
    start_time: datetime,
    end_time: datetime = None
) -> List[Bar]:
    """
    Versao sincrona do download com spread real

    Args:
        symbol: Simbolo (ex: 'EURUSD')
        periodicity: Periodicidade (M1, H1, D1, etc)
        start_time: Data/hora de inicio
        end_time: Data/hora de fim

    Returns:
        Lista de barras com spread real
    """
    return asyncio.run(download_historical_data_with_spread(
        symbol=symbol,
        periodicity=periodicity,
        start_time=start_time,
        end_time=end_time
    ))


# ==============================================================================
# TESTE
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TESTE DO CLIENTE FXOPEN WEBSOCKET")
    print("=" * 60)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=7)

    bars = get_historical_data_sync(
        symbol="EURUSD",
        periodicity="H1",
        start_time=start_time,
        end_time=end_time
    )

    if bars:
        print(f"\nPrimeiras 5 barras:")
        for bar in bars[:5]:
            print(f"  {bar}")

        print(f"\nUltimas 5 barras:")
        for bar in bars[-5:]:
            print(f"  {bar}")
    else:
        print("\nNenhuma barra baixada")
