#!/usr/bin/env python3
"""
Script para conectar à FX Open via WebSocket e obter ticks do EURUSD
"""
import asyncio
import sys
import os

# Adicionar o diretório pai ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.ticktrader_ws import TickTraderFeed, TickData
from config.settings import CONFIG

# Contador de ticks
tick_count = 0

async def on_tick(tick: TickData):
    """Callback executado quando um tick é recebido"""
    global tick_count
    tick_count += 1

    print(f"[{tick_count}] {tick.symbol} | "
          f"Bid: {tick.bid:.5f} | Ask: {tick.ask:.5f} | "
          f"Spread: {tick.spread*10000:.1f} pips | "
          f"Mid: {tick.mid:.5f} | "
          f"Time: {tick.timestamp.strftime('%H:%M:%S.%f')[:-3]}")

async def main():
    """Função principal"""
    print("=" * 70)
    print("  FX OPEN - EURUSD TICK STREAM")
    print("=" * 70)
    print(f"Server: {CONFIG.SERVER}")
    print(f"WebSocket Feed URL: {CONFIG.WS_FEED_URL}")
    print(f"Symbol: EURUSD")
    print("=" * 70)
    print()

    # Criar cliente de feed
    feed = TickTraderFeed()

    # Registrar callback para ticks
    feed.register_callback('tick', on_tick)

    try:
        # Conectar ao WebSocket
        print("Conectando ao WebSocket...")
        await feed.connect()

        if feed.is_connected():
            print("Conectado com sucesso!")
            print()

            # Inscrever-se no EURUSD
            print("Inscrevendo-se em EURUSD...")
            await feed.subscribe_symbol("EURUSD")
            print()
            print("Aguardando ticks... (Ctrl+C para sair)")
            print("-" * 70)

            # Manter a conexão ativa e processar ticks
            while feed.is_connected():
                await asyncio.sleep(1)

                # Verificar latência periodicamente
                latency = await feed.get_latency()
                if tick_count > 0 and tick_count % 100 == 0:
                    print(f"--- Latência atual: {latency:.0f}ms | Total ticks: {tick_count} ---")
        else:
            print("Falha na conexão!")

    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário")
    except Exception as e:
        print(f"\nErro: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Desconectar
        print("\nDesconectando...")
        await feed.disconnect()
        print(f"Total de ticks recebidos: {tick_count}")
        print("Finalizado.")

if __name__ == "__main__":
    asyncio.run(main())
