#!/usr/bin/env python3
"""
Teste de conexão WebSocket para ver formato exato das mensagens
"""
import asyncio
import websockets
import json
import ssl
import hmac
import hashlib
import base64
import time
import sys

sys.path.insert(0, '/home/azureuser/botforex/trading_bot')
from config.settings import CONFIG

async def test_websocket():
    """Testa conexão WebSocket e mostra mensagens recebidas"""
    print(f"Conectando a: {CONFIG.WS_FEED_URL}")

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        async with websockets.connect(
            CONFIG.WS_FEED_URL,
            ssl=ssl_context,
            ping_interval=20,
            ping_timeout=10
        ) as ws:
            print("Conectado! Autenticando...")

            # Autenticar
            timestamp = str(int(time.time() * 1000))
            message = f"{timestamp}{CONFIG.WEB_API_TOKEN_ID}{CONFIG.WEB_API_TOKEN_KEY}"
            signature = base64.b64encode(
                hmac.new(
                    CONFIG.WEB_API_TOKEN_SECRET.encode('utf-8'),
                    message.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')

            auth_msg = {
                "Id": "auth_1",
                "Request": "Login",
                "Params": {
                    "AuthType": "HMAC",
                    "WebApiId": CONFIG.WEB_API_TOKEN_ID,
                    "WebApiKey": CONFIG.WEB_API_TOKEN_KEY,
                    "Timestamp": timestamp,
                    "Signature": signature
                }
            }

            await ws.send(json.dumps(auth_msg))
            print("Mensagem de login enviada")

            # Receber resposta do login
            response = await asyncio.wait_for(ws.recv(), timeout=10)
            print(f"\n=== RESPOSTA LOGIN ===\n{json.dumps(json.loads(response), indent=2)}")

            # Subscrever a ticks
            subscribe_msg = {
                "Id": "sub_1",
                "Request": "FeedSubscribe",
                "Params": {
                    "Subscribe": [{
                        "Symbol": "EURUSD",
                        "BookDepth": 1
                    }]
                }
            }

            await ws.send(json.dumps(subscribe_msg))
            print("\nMensagem de subscrição enviada")

            # Receber resposta da subscrição
            response = await asyncio.wait_for(ws.recv(), timeout=10)
            print(f"\n=== RESPOSTA SUBSCRIÇÃO ===\n{json.dumps(json.loads(response), indent=2)}")

            # Escutar mensagens por 30 segundos
            print("\n=== ESCUTANDO MENSAGENS (30 segundos) ===")
            start = time.time()
            msg_count = 0

            while time.time() - start < 30:
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(response)
                    msg_count += 1

                    # Mostrar estrutura da mensagem
                    print(f"\n--- Mensagem #{msg_count} ---")
                    print(f"Chaves: {list(data.keys())}")

                    # Mostrar detalhes importantes
                    if 'Response' in data:
                        print(f"Response: {data['Response']}")
                    if 'Event' in data:
                        print(f"Event: {data['Event']}")
                    if 'Result' in data:
                        result = data['Result']
                        if isinstance(result, dict):
                            print(f"Result keys: {list(result.keys())}")
                            if 'Symbol' in result:
                                print(f"  Symbol: {result.get('Symbol')}")
                            if 'BestBid' in result:
                                print(f"  BestBid: {result.get('BestBid')}")
                            if 'BestAsk' in result:
                                print(f"  BestAsk: {result.get('BestAsk')}")
                        else:
                            print(f"Result: {result}")

                    # Mostrar JSON completo para mensagens de dados
                    if data.get('Response') in ['Tick', 'Quote', 'FeedTick'] or data.get('Event') in ['Tick', 'Quote']:
                        print(f"JSON completo:\n{json.dumps(data, indent=2)}")

                except asyncio.TimeoutError:
                    print(".", end="", flush=True)
                    continue

            print(f"\n\nTotal de mensagens recebidas: {msg_count}")

    except Exception as e:
        print(f"ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_websocket())
