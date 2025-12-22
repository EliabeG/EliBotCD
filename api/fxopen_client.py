"""
Cliente WebSocket para FX Open (TickTrader)
"""
import asyncio
import websockets
import json
import hmac
import hashlib
import base64
import time
import ssl
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

from config import settings as config


@dataclass
class Tick:
    """Dados de um tick"""
    symbol: str
    bid: float
    ask: float
    bid_volume: float
    ask_volume: float
    timestamp: datetime

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    def __str__(self) -> str:
        return (f"{self.symbol} | Bid: {self.bid:.5f} | Ask: {self.ask:.5f} | "
                f"Spread: {self.spread*10000:.1f} pips | {self.timestamp.strftime('%H:%M:%S')}")


class FXOpenClient:
    """Cliente para conectar ao WebSocket da FX Open"""

    def __init__(self):
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.connected: bool = False
        self.subscriptions: set = set()
        self.message_id: int = 0
        self.on_tick: Optional[Callable[[Tick], Any]] = None
        self._receive_task: Optional[asyncio.Task] = None

    def _generate_signature(self, timestamp: str) -> str:
        """Gera assinatura HMAC para autenticacao"""
        message = f"{timestamp}{config.WEB_API_ID}{config.WEB_API_KEY}"
        signature = base64.b64encode(
            hmac.new(
                config.WEB_API_SECRET.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        return signature

    def _next_id(self) -> str:
        """Gera proximo ID de mensagem"""
        self.message_id += 1
        return f"req_{self.message_id}"

    async def connect(self) -> bool:
        """Conecta ao WebSocket da FX Open"""
        try:
            print(f"Conectando a {config.WS_FEED_URL}...")

            # SSL context
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            self.ws = await websockets.connect(
                config.WS_FEED_URL,
                ssl=ssl_context,
                ping_interval=20,
                ping_timeout=10
            )

            self.connected = True
            print("Conectado! Autenticando...")

            # Autenticar
            await self._authenticate()

            # Iniciar processamento de mensagens
            self._receive_task = asyncio.create_task(self._process_messages())

            return True

        except Exception as e:
            print(f"Erro ao conectar: {e}")
            self.connected = False
            return False

    async def _authenticate(self):
        """Autentica no WebSocket"""
        timestamp = str(int(time.time() * 1000))

        auth_msg = {
            "Id": self._next_id(),
            "Request": "Login",
            "Params": {
                "AuthType": "HMAC",
                "WebApiId": config.WEB_API_ID,
                "WebApiKey": config.WEB_API_KEY,
                "Timestamp": timestamp,
                "Signature": self._generate_signature(timestamp)
            }
        }

        await self.ws.send(json.dumps(auth_msg))
        await asyncio.sleep(1)  # Aguardar resposta

    async def subscribe(self, symbol: str):
        """Inscreve para receber ticks de um simbolo"""
        if not self.connected:
            print("Nao conectado!")
            return

        if symbol in self.subscriptions:
            print(f"Ja inscrito em {symbol}")
            return

        msg = {
            "Id": self._next_id(),
            "Request": "FeedSubscribe",
            "Params": {
                "Subscribe": [{
                    "Symbol": symbol,
                    "BookDepth": 1
                }]
            }
        }

        await self.ws.send(json.dumps(msg))
        self.subscriptions.add(symbol)
        print(f"Inscrito em {symbol}")

    async def _process_messages(self):
        """Processa mensagens recebidas"""
        while self.connected and self.ws:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=30)
                data = json.loads(msg)

                response_type = data.get('Response') or data.get('Event')

                if response_type == 'Login':
                    result = data.get('Result', {})
                    if result.get('Info') == 'ok':
                        print("Autenticado com sucesso!")
                    else:
                        print(f"Falha na autenticacao: {data}")

                elif response_type == 'FeedSubscribe':
                    # Snapshot inicial
                    snapshots = data.get('Result', {}).get('Snapshot', [])
                    for snap in snapshots:
                        tick = self._parse_tick(snap)
                        if tick:
                            if self.on_tick:
                                await self._call_on_tick(tick)

                elif response_type in ('Tick', 'FeedTick'):
                    tick = self._parse_tick(data.get('Result', {}))
                    if tick:
                        if self.on_tick:
                            await self._call_on_tick(tick)

                elif response_type == 'Error':
                    print(f"Erro: {data}")

            except asyncio.TimeoutError:
                continue
            except websockets.ConnectionClosed:
                print("Conexao fechada")
                self.connected = False
                break
            except Exception as e:
                print(f"Erro ao processar mensagem: {e}")

    def _parse_tick(self, data: Dict) -> Optional[Tick]:
        """Converte dados em objeto Tick"""
        try:
            symbol = data.get('Symbol')
            if not symbol:
                return None

            best_bid = data.get('BestBid', {})
            best_ask = data.get('BestAsk', {})

            timestamp_ms = data.get('Timestamp', time.time() * 1000)

            return Tick(
                symbol=symbol,
                bid=float(best_bid.get('Price', 0)),
                ask=float(best_ask.get('Price', 0)),
                bid_volume=float(best_bid.get('Volume', 0)),
                ask_volume=float(best_ask.get('Volume', 0)),
                timestamp=datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            )
        except Exception as e:
            print(f"Erro ao parsear tick: {e}")
            return None

    async def _call_on_tick(self, tick: Tick):
        """Chama callback de tick"""
        if asyncio.iscoroutinefunction(self.on_tick):
            await self.on_tick(tick)
        else:
            self.on_tick(tick)

    async def disconnect(self):
        """Desconecta do WebSocket"""
        print("Desconectando...")
        self.connected = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self.ws:
            await self.ws.close()
            self.ws = None

        print("Desconectado.")


async def main():
    """Funcao principal - teste de conexao"""
    print("=" * 60)
    print("  EliBotCD - FX Open Connection Test")
    print("=" * 60)

    client = FXOpenClient()

    try:
        if await client.connect():
            await client.subscribe(config.SYMBOL)

            print(f"\nRecebendo ticks de {config.SYMBOL}...")
            print("Pressione Ctrl+C para sair\n")
            print("-" * 60)

            # Manter conexao ativa
            while client.connected:
                await asyncio.sleep(1)
        else:
            print("Falha ao conectar")

    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuario")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
