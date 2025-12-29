#!/usr/bin/env python3
"""
Cliente WebSocket para Trading FXOpen (TickTrader)
Execução de ordens em conta DEMO ou REAL
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
from enum import Enum

from config import settings as config


class OrderSide(Enum):
    BUY = "Buy"
    SELL = "Sell"


class OrderType(Enum):
    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"


@dataclass
class OrderResult:
    """Resultado de uma ordem"""
    success: bool
    order_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_volume: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: Optional[datetime] = None


class FXOpenTradeClient:
    """Cliente para executar ordens via WebSocket FXOpen"""

    def __init__(self):
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.connected: bool = False
        self.authenticated: bool = False
        self.message_id: int = 0
        self._response_futures: Dict[str, asyncio.Future] = {}
        self._receive_task: Optional[asyncio.Task] = None

        # Callbacks
        self.on_order_update: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None

        # Estado
        self.account_info: Dict = {}
        self.positions: Dict[str, Any] = {}

    def _generate_signature(self, timestamp: str) -> str:
        """Gera assinatura HMAC para autenticação"""
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
        """Gera próximo ID de mensagem"""
        self.message_id += 1
        return f"trade_{self.message_id}"

    async def connect(self) -> bool:
        """Conecta ao WebSocket de Trade"""
        try:
            print(f"[TRADE] Conectando a {config.WS_TRADE_URL}...")

            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            self.ws = await websockets.connect(
                config.WS_TRADE_URL,
                ssl=ssl_context,
                ping_interval=20,
                ping_timeout=10
            )

            self.connected = True
            print("[TRADE] Conectado! Autenticando...")

            # Autenticar
            success = await self._authenticate()
            if not success:
                print("[TRADE] Falha na autenticação!")
                return False

            # Iniciar processamento de mensagens
            self._receive_task = asyncio.create_task(self._process_messages())

            # Buscar info da conta
            await self._get_account_info()

            return True

        except Exception as e:
            print(f"[TRADE] Erro ao conectar: {e}")
            self.connected = False
            return False

    async def _authenticate(self) -> bool:
        """Autentica no WebSocket de Trade"""
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

        # Aguarda resposta
        try:
            msg = await asyncio.wait_for(self.ws.recv(), timeout=10)
            data = json.loads(msg)

            if data.get('Response') == 'Login':
                result = data.get('Result', {})
                if result.get('Info') == 'ok':
                    self.authenticated = True
                    print("[TRADE] Autenticado com sucesso!")
                    return True
                else:
                    print(f"[TRADE] Falha na autenticação: {data}")
                    return False
        except asyncio.TimeoutError:
            print("[TRADE] Timeout na autenticação")
            return False

        return False

    async def _get_account_info(self):
        """Busca informações da conta"""
        msg_id = self._next_id()

        msg = {
            "Id": msg_id,
            "Request": "Account"
        }

        await self.ws.send(json.dumps(msg))

    async def _process_messages(self):
        """Processa mensagens recebidas"""
        while self.connected and self.ws:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=30)
                data = json.loads(msg)

                msg_id = data.get('Id')
                response_type = data.get('Response') or data.get('Event')

                # Se há um future esperando esta resposta
                if msg_id and msg_id in self._response_futures:
                    self._response_futures[msg_id].set_result(data)
                    continue

                # Processar por tipo
                if response_type == 'Account':
                    self.account_info = data.get('Result', {})
                    balance = self.account_info.get('Balance', 0)
                    equity = self.account_info.get('Equity', 0)
                    margin = self.account_info.get('Margin', 0)
                    print(f"[TRADE] Conta: Balance=${balance:.2f} | Equity=${equity:.2f} | Margin=${margin:.2f}")

                elif response_type == 'Trade':
                    result = data.get('Result', {})
                    print(f"[TRADE] Ordem executada: {result}")

                elif response_type == 'OrderUpdate':
                    if self.on_order_update:
                        self.on_order_update(data.get('Result', {}))

                elif response_type == 'PositionUpdate':
                    pos = data.get('Result', {})
                    symbol = pos.get('Symbol')
                    if symbol:
                        self.positions[symbol] = pos
                    if self.on_position_update:
                        self.on_position_update(pos)

                elif response_type == 'Error':
                    print(f"[TRADE] Erro: {data}")

            except asyncio.TimeoutError:
                continue
            except websockets.ConnectionClosed:
                print("[TRADE] Conexão fechada")
                self.connected = False
                break
            except Exception as e:
                print(f"[TRADE] Erro ao processar mensagem: {e}")

    async def place_order(self,
                          symbol: str,
                          side: OrderSide,
                          volume: float,
                          order_type: OrderType = OrderType.MARKET,
                          price: Optional[float] = None,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None,
                          comment: str = "EliBotCD") -> OrderResult:
        """
        Envia uma ordem ao mercado

        Args:
            symbol: Par de moedas (ex: EURUSD)
            side: BUY ou SELL
            volume: Volume em lotes (ex: 0.01)
            order_type: MARKET, LIMIT ou STOP
            price: Preço (necessário para LIMIT/STOP)
            stop_loss: Nível de stop loss
            take_profit: Nível de take profit
            comment: Comentário da ordem

        Returns:
            OrderResult com status da ordem
        """
        if not self.connected or not self.authenticated:
            return OrderResult(
                success=False,
                error_message="Não conectado ou não autenticado"
            )

        msg_id = self._next_id()

        # Converte lotes para unidades (0.01 lote = 1000 unidades para EURUSD)
        # Volume em lotes * 100000 = unidades
        amount_units = int(volume * 100000)

        order_params = {
            "Type": order_type.value,
            "Side": side.value,
            "Symbol": symbol,
            "Amount": amount_units,
            "Comment": comment
        }

        if order_type != OrderType.MARKET and price:
            order_params["Price"] = price

        if stop_loss:
            order_params["StopLoss"] = stop_loss

        if take_profit:
            order_params["TakeProfit"] = take_profit

        msg = {
            "Id": msg_id,
            "Request": "TradeCreate",
            "Params": order_params
        }

        # Criar future para aguardar resposta
        future = asyncio.get_event_loop().create_future()
        self._response_futures[msg_id] = future

        try:
            print(f"[TRADE] Enviando ordem: {side.value} {volume} {symbol}")
            print(f"[TRADE] Request: {json.dumps(msg, indent=2)}")
            await self.ws.send(json.dumps(msg))

            # Aguardar resposta (timeout de 30s)
            response = await asyncio.wait_for(future, timeout=30)
            print(f"[TRADE] Response: {json.dumps(response, indent=2, default=str)}")

            result = response.get('Result', {})

            if response.get('Response') == 'TradeCreate' and result.get('Trade'):
                trade = result.get('Trade', {})
                order_id = trade.get('Id')
                filled_price = trade.get('Price')
                filled_volume = trade.get('Amount')

                print(f"[TRADE] ✓ Ordem executada! ID: {order_id} | Preço: {filled_price}")

                return OrderResult(
                    success=True,
                    order_id=order_id,
                    filled_price=filled_price,
                    filled_volume=filled_volume,
                    timestamp=datetime.now(timezone.utc)
                )
            else:
                error = result.get('Error') or response.get('Error', 'Erro desconhecido')
                print(f"[TRADE] ✗ Ordem rejeitada: {error}")

                return OrderResult(
                    success=False,
                    error_message=str(error)
                )

        except asyncio.TimeoutError:
            return OrderResult(
                success=False,
                error_message="Timeout aguardando confirmação"
            )
        except Exception as e:
            return OrderResult(
                success=False,
                error_message=str(e)
            )
        finally:
            del self._response_futures[msg_id]

    async def close_position(self, symbol: str, volume: Optional[float] = None) -> OrderResult:
        """
        Fecha uma posição aberta

        Args:
            symbol: Par de moedas
            volume: Volume a fechar (None = fechar tudo)

        Returns:
            OrderResult com status
        """
        if symbol not in self.positions:
            return OrderResult(
                success=False,
                error_message=f"Nenhuma posição aberta em {symbol}"
            )

        pos = self.positions[symbol]
        pos_side = pos.get('Side', 'Buy')
        pos_volume = volume or pos.get('Volume', 0)

        # Para fechar, envia ordem oposta
        close_side = OrderSide.SELL if pos_side == 'Buy' else OrderSide.BUY

        return await self.place_order(
            symbol=symbol,
            side=close_side,
            volume=pos_volume,
            order_type=OrderType.MARKET,
            comment="EliBotCD_Close"
        )

    async def get_balance(self) -> float:
        """Retorna balanço da conta"""
        return self.account_info.get('Balance', 0)

    async def get_equity(self) -> float:
        """Retorna equity da conta"""
        return self.account_info.get('Equity', 0)

    async def disconnect(self):
        """Desconecta do WebSocket"""
        print("[TRADE] Desconectando...")
        self.connected = False
        self.authenticated = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self.ws:
            await self.ws.close()
            self.ws = None

        print("[TRADE] Desconectado.")


# Teste
async def test_trade_client():
    """Testa o cliente de trade"""
    print("=" * 60)
    print("  FXOpen Trade Client Test")
    print("=" * 60)

    client = FXOpenTradeClient()

    try:
        if await client.connect():
            print("\n[TEST] Conectado com sucesso!")

            # Aguarda um pouco para receber info da conta
            await asyncio.sleep(2)

            print(f"\n[TEST] Balance: ${await client.get_balance():.2f}")
            print(f"[TEST] Equity: ${await client.get_equity():.2f}")

            # Teste: Enviar ordem de teste (MUITO PEQUENA)
            print("\n[TEST] Testando ordem BUY EURUSD 0.01 lote...")

            result = await client.place_order(
                symbol="EURUSD",
                side=OrderSide.BUY,
                volume=0.01,  # Micro lote
                stop_loss=None,  # Sem SL para teste
                take_profit=None,  # Sem TP para teste
                comment="EliBotCD_Test"
            )

            if result.success:
                print(f"[TEST] ✓ Ordem executada!")
                print(f"  Order ID: {result.order_id}")
                print(f"  Preço: {result.filled_price}")

                # Aguarda e fecha
                print("\n[TEST] Aguardando 3s e fechando posição...")
                await asyncio.sleep(3)

                close_result = await client.close_position("EURUSD")
                if close_result.success:
                    print("[TEST] ✓ Posição fechada!")
                else:
                    print(f"[TEST] ✗ Erro ao fechar: {close_result.error_message}")
            else:
                print(f"[TEST] ✗ Erro: {result.error_message}")

        else:
            print("[TEST] Falha ao conectar")

    except KeyboardInterrupt:
        print("\nInterrompido")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(test_trade_client())
