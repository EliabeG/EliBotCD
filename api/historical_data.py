"""
================================================================================
HISTORICAL DATA CLIENT
Cliente para Download de Dados Historicos da FXOpen API
================================================================================

Este modulo baixa dados historicos REAIS do mercado Forex via API REST.
IMPORTANTE: Todos os dados sao reais - nenhuma simulacao e permitida.

Endpoints utilizados:
- /api/v2/quotehistory/download/bars/info/{symbol}/{periodicity}/{priceType}
- /api/v2/quotehistory/download/bars/{symbol}/{periodicity}/{priceType}/{timestamp}

Periodicidades suportadas:
- M1: 1 minuto
- M5: 5 minutos
- M15: 15 minutos
- M30: 30 minutos
- H1: 1 hora
- H4: 4 horas
- D1: 1 dia
"""

import requests
import hmac
import hashlib
import base64
import time
import gzip
import io
import struct
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np

from config import settings


@dataclass
class Bar:
    """Representa uma barra/candle OHLCV"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def mid_open(self) -> float:
        return self.open

    @property
    def mid_close(self) -> float:
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
        return (f"{self.timestamp.strftime('%Y-%m-%d %H:%M')} | "
                f"O:{self.open:.5f} H:{self.high:.5f} L:{self.low:.5f} C:{self.close:.5f} V:{self.volume:.0f}")


class HistoricalDataClient:
    """
    Cliente para download de dados historicos da FXOpen API

    IMPORTANTE: Este cliente baixa APENAS dados REAIS do mercado.
    Nenhuma simulacao ou dados sinteticos sao utilizados.
    """

    # URLs base
    DEMO_BASE_URL = "https://marginalttdemowebapi.fxopen.net"
    LIVE_BASE_URL = "https://ttlivewebapi.fxopen.net"

    # Periodicidades validas
    VALID_PERIODICITIES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']

    # Mapeamento de periodicidade para segundos
    PERIODICITY_SECONDS = {
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

    def __init__(self, use_demo: bool = True):
        """
        Inicializa o cliente

        Args:
            use_demo: Se True, usa servidor demo. Se False, usa servidor live.
        """
        self.base_url = self.DEMO_BASE_URL if use_demo else self.LIVE_BASE_URL
        self.web_api_id = settings.WEB_API_ID
        self.web_api_key = settings.WEB_API_KEY
        self.web_api_secret = settings.WEB_API_SECRET

        # Cache de dados
        self._cache: Dict[str, List[Bar]] = {}

    def _generate_signature(self, timestamp: str) -> str:
        """Gera assinatura HMAC para autenticacao"""
        message = f"{timestamp}{self.web_api_id}{self.web_api_key}"
        signature = base64.b64encode(
            hmac.new(
                self.web_api_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        return signature

    def _get_auth_headers(self) -> Dict[str, str]:
        """Retorna headers de autenticacao"""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp)

        return {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Authorization': f'HMAC {self.web_api_id}:{self.web_api_key}:{timestamp}:{signature}'
        }

    def get_bars_info(self, symbol: str, periodicity: str, price_type: str = 'Bid') -> Optional[Dict]:
        """
        Obtem informacoes sobre barras disponiveis

        Args:
            symbol: Simbolo (ex: 'EURUSD')
            periodicity: Periodicidade (M1, M5, H1, etc)
            price_type: 'Bid' ou 'Ask'

        Returns:
            Dict com informacoes ou None se erro
        """
        if periodicity not in self.VALID_PERIODICITIES:
            print(f"Periodicidade invalida: {periodicity}")
            return None

        url = f"{self.base_url}/api/v2/quotehistory/download/bars/info/{symbol}/{periodicity}/{price_type}"

        try:
            response = requests.get(url, headers=self._get_auth_headers(), timeout=30)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Erro ao obter info: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            print(f"Erro na requisicao: {e}")
            return None

    def download_bars(self,
                     symbol: str,
                     periodicity: str,
                     start_time: datetime,
                     end_time: datetime = None,
                     price_type: str = 'Bid') -> List[Bar]:
        """
        Baixa barras historicas REAIS da API

        IMPORTANTE: Estes sao dados REAIS do mercado Forex.

        Args:
            symbol: Simbolo (ex: 'EURUSD')
            periodicity: Periodicidade (M1, M5, H1, etc)
            start_time: Data/hora de inicio
            end_time: Data/hora de fim (default: agora)
            price_type: 'Bid' ou 'Ask'

        Returns:
            Lista de barras reais
        """
        if periodicity not in self.VALID_PERIODICITIES:
            print(f"Periodicidade invalida: {periodicity}")
            return []

        if end_time is None:
            end_time = datetime.now(timezone.utc)

        # Garante timezone UTC
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        print(f"\n{'='*60}")
        print(f"  DOWNLOAD DE DADOS HISTORICOS REAIS")
        print(f"{'='*60}")
        print(f"  Simbolo: {symbol}")
        print(f"  Periodicidade: {periodicity}")
        print(f"  Periodo: {start_time.strftime('%Y-%m-%d %H:%M')} a {end_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Tipo de preco: {price_type}")
        print(f"{'='*60}\n")

        all_bars = []
        current_time = start_time
        chunk_size = timedelta(days=7)  # Baixa 7 dias por vez

        while current_time < end_time:
            chunk_end = min(current_time + chunk_size, end_time)
            timestamp_ms = int(current_time.timestamp() * 1000)

            url = f"{self.base_url}/api/v2/quotehistory/download/bars/{symbol}/{periodicity}/{price_type}/{timestamp_ms}"

            try:
                response = requests.get(
                    url,
                    headers=self._get_auth_headers(),
                    timeout=60,
                    stream=True
                )

                if response.status_code == 200:
                    bars = self._parse_bars_response(response.content, current_time, chunk_end)
                    all_bars.extend(bars)
                    print(f"  Baixado: {current_time.strftime('%Y-%m-%d')} - {len(bars)} barras")
                else:
                    print(f"  Erro: {response.status_code} para {current_time.strftime('%Y-%m-%d')}")

            except Exception as e:
                print(f"  Erro ao baixar: {e}")

            current_time = chunk_end
            time.sleep(0.5)  # Rate limiting

        # Filtra barras dentro do periodo
        all_bars = [b for b in all_bars if start_time <= b.timestamp <= end_time]

        # Ordena por timestamp
        all_bars.sort(key=lambda b: b.timestamp)

        print(f"\n  Total de barras baixadas: {len(all_bars)}")
        if all_bars:
            print(f"  Primeira barra: {all_bars[0].timestamp}")
            print(f"  Ultima barra: {all_bars[-1].timestamp}")

        return all_bars

    def _parse_bars_response(self, content: bytes, start: datetime, end: datetime) -> List[Bar]:
        """
        Parseia resposta da API em lista de barras

        O formato e binario comprimido com gzip.
        Cada barra tem: timestamp(8), open(8), high(8), low(8), close(8), volume(8) = 48 bytes
        """
        bars = []

        try:
            # Tenta descomprimir gzip
            try:
                decompressed = gzip.decompress(content)
            except:
                decompressed = content

            # Se for JSON, parseia como JSON
            if decompressed.startswith(b'{') or decompressed.startswith(b'['):
                import json
                data = json.loads(decompressed.decode('utf-8'))

                if isinstance(data, list):
                    for item in data:
                        bar = self._parse_bar_json(item)
                        if bar:
                            bars.append(bar)
                elif isinstance(data, dict) and 'Bars' in data:
                    for item in data['Bars']:
                        bar = self._parse_bar_json(item)
                        if bar:
                            bars.append(bar)

            else:
                # Formato binario
                bar_size = 48  # 6 doubles de 8 bytes
                num_bars = len(decompressed) // bar_size

                for i in range(num_bars):
                    offset = i * bar_size
                    bar_data = decompressed[offset:offset + bar_size]

                    if len(bar_data) == bar_size:
                        values = struct.unpack('<6d', bar_data)
                        timestamp = datetime.fromtimestamp(values[0] / 1000, tz=timezone.utc)

                        bar = Bar(
                            timestamp=timestamp,
                            open=values[1],
                            high=values[2],
                            low=values[3],
                            close=values[4],
                            volume=values[5]
                        )
                        bars.append(bar)

        except Exception as e:
            print(f"    Erro ao parsear: {e}")

        return bars

    def _parse_bar_json(self, item: dict) -> Optional[Bar]:
        """Parseia uma barra de formato JSON"""
        try:
            # Diferentes formatos possiveis
            timestamp = item.get('Timestamp') or item.get('Time') or item.get('t')
            if isinstance(timestamp, (int, float)):
                if timestamp > 1e12:  # Milliseconds
                    timestamp = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
                else:
                    timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            elif isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

            return Bar(
                timestamp=timestamp,
                open=float(item.get('Open') or item.get('o', 0)),
                high=float(item.get('High') or item.get('h', 0)),
                low=float(item.get('Low') or item.get('l', 0)),
                close=float(item.get('Close') or item.get('c', 0)),
                volume=float(item.get('Volume') or item.get('v', 0))
            )
        except Exception as e:
            return None

    def download_ticks_from_bars(self,
                                 symbol: str,
                                 start_time: datetime,
                                 end_time: datetime = None,
                                 periodicity: str = 'M1') -> List[Tuple[float, datetime, float]]:
        """
        Baixa dados e converte para formato de ticks (price, timestamp, volume)

        Para backtesting que requer ticks, usamos o preco de fechamento
        de cada barra como um tick.

        Args:
            symbol: Simbolo
            start_time: Inicio
            end_time: Fim
            periodicity: Periodicidade das barras

        Returns:
            Lista de (price, timestamp, volume)
        """
        bars = self.download_bars(symbol, periodicity, start_time, end_time)

        ticks = []
        for bar in bars:
            # Usa close como preco do tick
            ticks.append((bar.close, bar.timestamp, bar.volume))

        return ticks

    def get_price_series(self, symbol: str, periodicity: str,
                        start_time: datetime, end_time: datetime = None,
                        price_field: str = 'close') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retorna series de precos para analise

        Args:
            symbol: Simbolo
            periodicity: Periodicidade
            start_time: Inicio
            end_time: Fim
            price_field: 'open', 'high', 'low', 'close'

        Returns:
            Tupla (prices, timestamps, volumes) como numpy arrays
        """
        bars = self.download_bars(symbol, periodicity, start_time, end_time)

        if not bars:
            return np.array([]), np.array([]), np.array([])

        prices = []
        timestamps = []
        volumes = []

        for bar in bars:
            if price_field == 'open':
                prices.append(bar.open)
            elif price_field == 'high':
                prices.append(bar.high)
            elif price_field == 'low':
                prices.append(bar.low)
            else:  # close
                prices.append(bar.close)

            timestamps.append(bar.timestamp)
            volumes.append(bar.volume)

        return np.array(prices), np.array(timestamps), np.array(volumes)

    def save_to_csv(self, bars: List[Bar], filename: str):
        """
        Salva barras em arquivo CSV

        Args:
            bars: Lista de barras
            filename: Nome do arquivo
        """
        import csv

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            for bar in bars:
                writer.writerow([
                    bar.timestamp.isoformat(),
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume
                ])

        print(f"Dados salvos em {filename}")

    def load_from_csv(self, filename: str) -> List[Bar]:
        """
        Carrega barras de arquivo CSV

        Args:
            filename: Nome do arquivo

        Returns:
            Lista de barras
        """
        import csv

        bars = []

        with open(filename, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                bar = Bar(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume'])
                )
                bars.append(bar)

        print(f"Carregado {len(bars)} barras de {filename}")
        return bars


class YahooFinanceClient:
    """
    Cliente alternativo usando Yahoo Finance para dados historicos REAIS

    IMPORTANTE: Todos os dados sao REAIS do mercado Forex.
    Yahoo Finance fornece dados de qualidade institucional.
    """

    # Mapeamento de simbolos FXOpen -> Yahoo Finance
    SYMBOL_MAP = {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'USDJPY': 'USDJPY=X',
        'USDCHF': 'USDCHF=X',
        'AUDUSD': 'AUDUSD=X',
        'USDCAD': 'USDCAD=X',
        'NZDUSD': 'NZDUSD=X',
        'EURGBP': 'EURGBP=X',
        'EURJPY': 'EURJPY=X',
        'GBPJPY': 'GBPJPY=X',
    }

    # Mapeamento de periodicidade
    PERIODICITY_MAP = {
        'M1': '1m',
        'M5': '5m',
        'M15': '15m',
        'M30': '30m',
        'H1': '1h',
        'H4': '4h',  # Yahoo nao suporta, usa 1h
        'D1': '1d',
        'W1': '1wk',
        'MN1': '1mo',
    }

    def __init__(self):
        """Inicializa o cliente Yahoo Finance"""
        try:
            import yfinance as yf
            self.yf = yf
            self.available = True
        except ImportError:
            print("AVISO: yfinance nao instalado. Execute: pip install yfinance")
            self.available = False

    def download_bars(self,
                     symbol: str,
                     periodicity: str,
                     start_time: datetime,
                     end_time: datetime = None) -> List[Bar]:
        """
        Baixa barras historicas REAIS do Yahoo Finance

        Args:
            symbol: Simbolo FXOpen (ex: 'EURUSD')
            periodicity: Periodicidade (M1, H1, D1, etc)
            start_time: Inicio
            end_time: Fim

        Returns:
            Lista de barras reais
        """
        if not self.available:
            print("Yahoo Finance nao disponivel")
            return []

        # Converte simbolo
        yf_symbol = self.SYMBOL_MAP.get(symbol.upper(), f"{symbol}=X")

        # Converte periodicidade
        interval = self.PERIODICITY_MAP.get(periodicity, '1h')

        # Yahoo tem limitacoes para dados intraday
        # M1: max 7 dias
        # M5, M15, M30: max 60 dias
        # H1: max 730 dias

        if end_time is None:
            end_time = datetime.now(timezone.utc)

        print(f"\n{'='*60}")
        print(f"  DOWNLOAD DE DADOS HISTORICOS REAIS (Yahoo Finance)")
        print(f"{'='*60}")
        print(f"  Simbolo: {symbol} -> {yf_symbol}")
        print(f"  Periodicidade: {periodicity} -> {interval}")
        print(f"  Periodo: {start_time.strftime('%Y-%m-%d %H:%M')} a {end_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}\n")

        try:
            # Download dados
            import warnings
            warnings.filterwarnings('ignore')

            data = self.yf.download(
                yf_symbol,
                start=start_time,
                end=end_time,
                interval=interval,
                progress=False
            )

            if data.empty:
                print("  Nenhum dado retornado")
                return []

            # Converte para lista de Bar
            bars = []
            for idx, row in data.iterrows():
                # Extrai valores (formato pode variar)
                try:
                    if isinstance(row['Open'], dict):
                        open_val = list(row['Open'].values())[0]
                        high_val = list(row['High'].values())[0]
                        low_val = list(row['Low'].values())[0]
                        close_val = list(row['Close'].values())[0]
                        vol_val = list(row['Volume'].values())[0] if 'Volume' in row else 0
                    else:
                        open_val = float(row['Open'].iloc[0]) if hasattr(row['Open'], 'iloc') else float(row['Open'])
                        high_val = float(row['High'].iloc[0]) if hasattr(row['High'], 'iloc') else float(row['High'])
                        low_val = float(row['Low'].iloc[0]) if hasattr(row['Low'], 'iloc') else float(row['Low'])
                        close_val = float(row['Close'].iloc[0]) if hasattr(row['Close'], 'iloc') else float(row['Close'])
                        vol_val = float(row['Volume'].iloc[0]) if hasattr(row.get('Volume', 0), 'iloc') else float(row.get('Volume', 0))

                    # Timestamp
                    if hasattr(idx, 'to_pydatetime'):
                        ts = idx.to_pydatetime()
                    else:
                        ts = idx

                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)

                    bar = Bar(
                        timestamp=ts,
                        open=open_val,
                        high=high_val,
                        low=low_val,
                        close=close_val,
                        volume=vol_val
                    )
                    bars.append(bar)

                except Exception as e:
                    continue

            print(f"  Total de barras baixadas: {len(bars)}")
            if bars:
                print(f"  Primeira barra: {bars[0].timestamp}")
                print(f"  Ultima barra: {bars[-1].timestamp}")

            return bars

        except Exception as e:
            print(f"  Erro ao baixar dados: {e}")
            return []


class RealDataCollector:
    """
    Coletor de dados em tempo real via WebSocket

    Armazena ticks reais para uso futuro em backtesting.
    IMPORTANTE: Todos os dados sao REAIS capturados em tempo real.
    """

    def __init__(self, storage_dir: str = "data/ticks"):
        """
        Inicializa o coletor

        Args:
            storage_dir: Diretorio para armazenar dados
        """
        import os
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        self.ticks: List[Tuple[float, datetime, float]] = []
        self.current_file = None

    def add_tick(self, price: float, timestamp: datetime, volume: float = 0):
        """Adiciona um tick real"""
        self.ticks.append((price, timestamp, volume))

    def save(self, symbol: str):
        """Salva ticks coletados em arquivo"""
        if not self.ticks:
            return

        import csv
        from datetime import datetime as dt

        filename = f"{self.storage_dir}/{symbol}_{dt.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'price', 'volume'])

            for price, timestamp, volume in self.ticks:
                writer.writerow([timestamp.isoformat(), price, volume])

        print(f"Salvos {len(self.ticks)} ticks em {filename}")
        self.ticks.clear()

        return filename

    def load(self, filename: str) -> List[Tuple[float, datetime, float]]:
        """Carrega ticks de arquivo"""
        import csv

        ticks = []

        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticks.append((
                    float(row['price']),
                    datetime.fromisoformat(row['timestamp']),
                    float(row['volume'])
                ))

        return ticks


def get_historical_data(symbol: str,
                       periodicity: str,
                       start_time: datetime,
                       end_time: datetime = None,
                       source: str = 'auto') -> List[Bar]:
    """
    Funcao de conveniencia para obter dados historicos REAIS

    Tenta multiplas fontes ate conseguir dados.

    Args:
        symbol: Simbolo (ex: 'EURUSD')
        periodicity: Periodicidade (M1, H1, D1, etc)
        start_time: Inicio
        end_time: Fim
        source: 'yahoo', 'fxopen', ou 'auto'

    Returns:
        Lista de barras reais
    """
    bars = []

    # Tenta Yahoo Finance primeiro (mais confiavel)
    if source in ['auto', 'yahoo']:
        yf_client = YahooFinanceClient()
        bars = yf_client.download_bars(symbol, periodicity, start_time, end_time)

        if bars:
            return bars

    # Fallback para FXOpen
    if source in ['auto', 'fxopen']:
        fx_client = HistoricalDataClient()
        bars = fx_client.download_bars(symbol, periodicity, start_time, end_time)

    return bars


# ==============================================================================
# TESTE
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TESTE DO CLIENTE DE DADOS HISTORICOS")
    print("=" * 60)

    # Testa Yahoo Finance
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=5)

    bars = get_historical_data(
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
