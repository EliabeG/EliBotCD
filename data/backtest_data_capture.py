#!/usr/bin/env python3
"""
================================================================================
BACKTEST DATA CAPTURE - FXOpen WebSocket API
================================================================================

Script para capturar dados históricos da FXOpen via WebSocket para backtesting.

Funcionalidades:
- Conexão WebSocket autenticada via HMAC-SHA256
- Busca de barras históricas (OHLCV) com paginação automática
- Suporte a múltiplos símbolos e timeframes
- Salvamento em CSV e Parquet
- Resumo automático de downloads interrompidos

Uso:
    python -m data.backtest_data_capture --symbol EURUSD --timeframe H1 --days 365
    python -m data.backtest_data_capture --symbols EURUSD,GBPUSD --timeframes M1,M5,H1

Autor: EliBotCD
"""

import asyncio
import websockets
import json
import hmac
import hashlib
import base64
import time
import uuid
import pandas as pd
import os
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# Importa configurações do projeto
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings


@dataclass
class CaptureConfig:
    """Configuração para captura de dados"""
    symbols: List[str] = field(default_factory=lambda: ["EURUSD"])
    timeframes: List[str] = field(default_factory=lambda: ["H1"])
    days_back: int = 365
    price_type: str = "bid"  # "bid" ou "ask"
    output_dir: str = "backtest_data"
    output_format: str = "both"  # "csv", "parquet", ou "both"
    batch_size: int = 1000  # Máximo permitido pela API


class FXOpenWebSocketClient:
    """Cliente WebSocket para FXOpen API"""

    TIMEFRAME_MINUTES = {
        "S1": 1/60, "S10": 10/60,
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240,
        "D1": 1440, "W1": 10080, "MN1": 43200
    }

    def __init__(self, config: CaptureConfig):
        self.config = config
        self.ws_url = settings.WS_FEED_URL
        self.api_id = settings.WEB_API_ID
        self.api_key = settings.WEB_API_KEY
        self.api_secret = settings.WEB_API_SECRET
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_authenticated = False
        self.request_counter = 0

        # Cria diretório de saída
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _generate_signature(self, timestamp: str) -> str:
        """Gera assinatura HMAC-SHA256 para autenticação"""
        message = f"{timestamp}{self.api_id}{self.api_key}"
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        return signature

    def _generate_request_id(self) -> str:
        """Gera ID único para requisição"""
        self.request_counter += 1
        return f"req_{self.request_counter}_{uuid.uuid4().hex[:8]}"

    async def connect(self) -> bool:
        """Estabelece conexão WebSocket"""
        try:
            print(f"[INFO] Conectando a {self.ws_url}...")
            self.ws = await websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            print("[OK] Conexão WebSocket estabelecida")
            return True
        except Exception as e:
            print(f"[ERRO] Falha na conexão: {e}")
            return False

    async def authenticate(self) -> bool:
        """Autentica na API usando HMAC"""
        if not self.ws:
            print("[ERRO] WebSocket não conectado")
            return False

        try:
            timestamp = str(int(time.time() * 1000))
            signature = self._generate_signature(timestamp)

            login_request = {
                "Id": self._generate_request_id(),
                "Request": "Login",
                "Params": {
                    "AuthType": "HMAC",
                    "WebApiId": self.api_id,
                    "WebApiKey": self.api_key,
                    "Timestamp": timestamp,
                    "Signature": signature,
                    "DeviceId": f"EliBotCD_{uuid.uuid4().hex[:8]}",
                    "AppSessionId": uuid.uuid4().hex
                }
            }

            print("[INFO] Enviando autenticação...")
            await self.ws.send(json.dumps(login_request))

            response = await asyncio.wait_for(self.ws.recv(), timeout=30)
            result = json.loads(response)

            if result.get("Response") == "Login" and "Result" in result:
                self.is_authenticated = True
                print("[OK] Autenticação bem-sucedida")
                return True
            else:
                error = result.get("Error", "Erro desconhecido")
                print(f"[ERRO] Falha na autenticação: {error}")
                return False

        except asyncio.TimeoutError:
            print("[ERRO] Timeout na autenticação")
            return False
        except Exception as e:
            print(f"[ERRO] Exceção na autenticação: {e}")
            return False

    async def get_symbols(self) -> List[Dict]:
        """Obtém lista de símbolos disponíveis"""
        if not self.is_authenticated:
            return []

        request = {
            "Id": self._generate_request_id(),
            "Request": "SymbolsInfo",
            "Params": {}
        }

        await self.ws.send(json.dumps(request))
        response = await asyncio.wait_for(self.ws.recv(), timeout=30)
        result = json.loads(response)

        if "Result" in result:
            return result["Result"]
        return []

    async def fetch_bars(
        self,
        symbol: str,
        periodicity: str,
        timestamp_ms: int,
        count: int = -1000,
        price_type: str = "bid"
    ) -> List[Dict]:
        """
        Busca barras históricas

        Args:
            symbol: Símbolo do ativo (ex: EURUSD)
            periodicity: Timeframe (M1, H1, D1, etc.)
            timestamp_ms: Timestamp de referência em milissegundos
            count: Número de barras (-1000 a -1 para passado, 1 a 1000 para futuro)
            price_type: "bid" ou "ask"

        Returns:
            Lista de barras OHLCV
        """
        if not self.is_authenticated:
            return []

        request = {
            "Id": self._generate_request_id(),
            "Request": "QuoteHistoryBars",
            "Params": {
                "Symbol": symbol,
                "Periodicity": periodicity,
                "PriceType": price_type,
                "Timestamp": timestamp_ms,
                "Count": count
            }
        }

        try:
            await self.ws.send(json.dumps(request))
            response = await asyncio.wait_for(self.ws.recv(), timeout=60)
            result = json.loads(response)

            if "Result" in result and "Bars" in result["Result"]:
                return result["Result"]["Bars"]
            elif "Error" in result:
                print(f"[AVISO] Erro na API: {result['Error']}")
            return []

        except asyncio.TimeoutError:
            print(f"[AVISO] Timeout ao buscar barras para {symbol} {periodicity}")
            return []
        except Exception as e:
            print(f"[ERRO] Exceção ao buscar barras: {e}")
            return []

    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        days_back: int,
        price_type: str = "bid"
    ) -> pd.DataFrame:
        """
        Busca dados históricos com paginação automática

        Args:
            symbol: Símbolo do ativo
            timeframe: Timeframe das barras
            days_back: Quantos dias para trás buscar
            price_type: Tipo de preço (bid/ask)

        Returns:
            DataFrame com dados OHLCV
        """
        all_bars = []

        # Calcula timestamp inicial (agora)
        end_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        current_timestamp = end_timestamp

        # Calcula timestamp final (X dias atrás)
        start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        start_timestamp = int(start_date.timestamp() * 1000)

        # Minutos por barra
        minutes_per_bar = self.TIMEFRAME_MINUTES.get(timeframe, 60)
        bars_per_day = int(1440 / minutes_per_bar)
        estimated_total_bars = days_back * bars_per_day

        print(f"\n[INFO] Buscando {symbol} {timeframe}")
        print(f"       Período: {start_date.strftime('%Y-%m-%d')} até {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
        print(f"       Estimativa: ~{estimated_total_bars:,} barras")

        batch_count = 0
        total_fetched = 0

        while current_timestamp > start_timestamp:
            batch_count += 1

            # Busca lote de barras (negativo = barras anteriores ao timestamp)
            bars = await self.fetch_bars(
                symbol=symbol,
                periodicity=timeframe,
                timestamp_ms=current_timestamp,
                count=-self.config.batch_size,
                price_type=price_type
            )

            if not bars:
                print(f"       [!] Sem mais dados disponíveis")
                break

            all_bars.extend(bars)
            total_fetched += len(bars)

            # Atualiza timestamp para próxima iteração
            oldest_bar = min(bars, key=lambda x: x.get("Timestamp", float('inf')))
            current_timestamp = oldest_bar.get("Timestamp", 0) - 1

            # Progresso
            progress = min(100, (total_fetched / estimated_total_bars) * 100) if estimated_total_bars > 0 else 0
            print(f"       Lote {batch_count}: {len(bars)} barras | Total: {total_fetched:,} | {progress:.1f}%", end='\r')

            # Verifica se já passou do limite
            if current_timestamp < start_timestamp:
                break

            # Rate limiting
            await asyncio.sleep(0.1)

        print(f"\n       [OK] {total_fetched:,} barras coletadas")

        if not all_bars:
            return pd.DataFrame()

        # Converte para DataFrame
        df = pd.DataFrame(all_bars)

        # Processa colunas
        df['timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)

        # Renomeia colunas para padrão
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        df = df.rename(columns=column_mapping)

        # Adiciona metadados
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df['price_type'] = price_type

        # Seleciona e ordena colunas
        columns = ['timestamp', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume', 'price_type']
        available_columns = [c for c in columns if c in df.columns]
        df = df[available_columns]

        # Remove duplicatas e ordena
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

        # Filtra pelo período solicitado
        df = df[df['timestamp'] >= start_date]

        return df

    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, str]:
        """
        Salva dados em arquivo(s)

        Returns:
            Dict com caminhos dos arquivos salvos
        """
        if df.empty:
            return {}

        saved_files = {}

        # Nome base do arquivo
        date_range = f"{df['timestamp'].min().strftime('%Y%m%d')}_{df['timestamp'].max().strftime('%Y%m%d')}"
        base_name = f"{symbol}_{timeframe}_{date_range}"

        # Salva CSV
        if self.config.output_format in ["csv", "both"]:
            csv_path = self.output_path / f"{base_name}.csv"
            df.to_csv(csv_path, index=False)
            saved_files["csv"] = str(csv_path)
            print(f"       [SALVO] {csv_path}")

        # Salva Parquet
        if self.config.output_format in ["parquet", "both"]:
            parquet_path = self.output_path / f"{base_name}.parquet"
            df.to_parquet(parquet_path, index=False, compression='snappy')
            saved_files["parquet"] = str(parquet_path)
            print(f"       [SALVO] {parquet_path}")

        return saved_files

    async def run(self) -> Dict[str, Any]:
        """
        Executa captura completa de dados

        Returns:
            Resumo da captura
        """
        summary = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "symbols": self.config.symbols,
                "timeframes": self.config.timeframes,
                "days_back": self.config.days_back,
                "price_type": self.config.price_type
            },
            "results": [],
            "errors": []
        }

        print("=" * 60)
        print("BACKTEST DATA CAPTURE - FXOpen WebSocket API")
        print("=" * 60)

        # Conecta e autentica
        if not await self.connect():
            summary["errors"].append("Falha na conexão WebSocket")
            return summary

        if not await self.authenticate():
            summary["errors"].append("Falha na autenticação")
            return summary

        # Itera sobre símbolos e timeframes
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                try:
                    # Busca dados
                    df = await self.fetch_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        days_back=self.config.days_back,
                        price_type=self.config.price_type
                    )

                    if not df.empty:
                        # Salva arquivos
                        saved_files = self.save_data(df, symbol, timeframe)

                        result = {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "bars_count": len(df),
                            "date_from": df['timestamp'].min().isoformat(),
                            "date_to": df['timestamp'].max().isoformat(),
                            "files": saved_files
                        }
                        summary["results"].append(result)
                    else:
                        summary["errors"].append(f"Sem dados para {symbol} {timeframe}")

                except Exception as e:
                    error_msg = f"Erro ao processar {symbol} {timeframe}: {str(e)}"
                    print(f"[ERRO] {error_msg}")
                    summary["errors"].append(error_msg)

        # Fecha conexão
        if self.ws:
            await self.ws.close()

        summary["finished_at"] = datetime.now(timezone.utc).isoformat()

        # Salva resumo
        summary_path = self.output_path / f"capture_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n[INFO] Resumo salvo em: {summary_path}")

        return summary


def print_summary(summary: Dict[str, Any]):
    """Imprime resumo da captura"""
    print("\n" + "=" * 60)
    print("RESUMO DA CAPTURA")
    print("=" * 60)

    if summary.get("results"):
        total_bars = sum(r["bars_count"] for r in summary["results"])
        print(f"\nTotal de barras capturadas: {total_bars:,}")
        print(f"\nDetalhes por símbolo/timeframe:")

        for result in summary["results"]:
            print(f"\n  {result['symbol']} {result['timeframe']}:")
            print(f"    - Barras: {result['bars_count']:,}")
            print(f"    - Período: {result['date_from'][:10]} a {result['date_to'][:10]}")
            for fmt, path in result.get("files", {}).items():
                print(f"    - {fmt.upper()}: {path}")

    if summary.get("errors"):
        print(f"\nErros ({len(summary['errors'])}):")
        for error in summary["errors"]:
            print(f"  - {error}")

    print("\n" + "=" * 60)


def parse_args():
    """Parse argumentos de linha de comando"""
    parser = argparse.ArgumentParser(
        description="Captura dados históricos da FXOpen para backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Capturar EURUSD H1 dos últimos 365 dias
  python -m data.backtest_data_capture --symbol EURUSD --timeframe H1 --days 365

  # Capturar múltiplos símbolos e timeframes
  python -m data.backtest_data_capture --symbols EURUSD,GBPUSD,USDJPY --timeframes M1,M5,H1,D1

  # Capturar apenas em formato Parquet
  python -m data.backtest_data_capture --symbol EURUSD --timeframe M1 --days 30 --format parquet

Timeframes disponíveis:
  S1, S10     - Segundos
  M1, M5, M15, M30 - Minutos
  H1, H4      - Horas
  D1, W1, MN1 - Dia, Semana, Mês
        """
    )

    parser.add_argument(
        "--symbol", "-s",
        type=str,
        help="Símbolo único (ex: EURUSD)"
    )

    parser.add_argument(
        "--symbols",
        type=str,
        help="Lista de símbolos separados por vírgula (ex: EURUSD,GBPUSD)"
    )

    parser.add_argument(
        "--timeframe", "-t",
        type=str,
        help="Timeframe único (ex: H1)"
    )

    parser.add_argument(
        "--timeframes",
        type=str,
        help="Lista de timeframes separados por vírgula (ex: M1,H1,D1)"
    )

    parser.add_argument(
        "--days", "-d",
        type=int,
        default=365,
        help="Número de dias para trás (padrão: 365)"
    )

    parser.add_argument(
        "--price-type", "-p",
        type=str,
        choices=["bid", "ask"],
        default="bid",
        help="Tipo de preço (padrão: bid)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="backtest_data",
        help="Diretório de saída (padrão: backtest_data)"
    )

    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["csv", "parquet", "both"],
        default="both",
        help="Formato de saída (padrão: both)"
    )

    return parser.parse_args()


async def main():
    """Função principal"""
    args = parse_args()

    # Processa símbolos
    symbols = []
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = [settings.SYMBOL or "EURUSD"]

    # Processa timeframes
    timeframes = []
    if args.timeframes:
        timeframes = [t.strip().upper() for t in args.timeframes.split(",")]
    elif args.timeframe:
        timeframes = [args.timeframe.upper()]
    else:
        timeframes = ["H1"]

    # Cria configuração
    config = CaptureConfig(
        symbols=symbols,
        timeframes=timeframes,
        days_back=args.days,
        price_type=args.price_type,
        output_dir=args.output,
        output_format=args.format
    )

    # Executa captura
    client = FXOpenWebSocketClient(config)
    summary = await client.run()

    # Imprime resumo
    print_summary(summary)

    return summary


if __name__ == "__main__":
    asyncio.run(main())
