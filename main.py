#!/usr/bin/env python3
"""
EliBotCD - Bot de Trading Forex
"""
import asyncio
from api import FXOpenClient
from config import settings


async def main():
    """Funcao principal"""
    print("=" * 60)
    print("  EliBotCD - FX Open Trading Bot")
    print("=" * 60)
    print(f"Symbol: {settings.SYMBOL}")
    print(f"Feed URL: {settings.WS_FEED_URL}")
    print("=" * 60)
    print()

    client = FXOpenClient()

    try:
        if await client.connect():
            await client.subscribe(settings.SYMBOL)

            print(f"\nRecebendo ticks de {settings.SYMBOL}...")
            print("Pressione Ctrl+C para sair\n")
            print("-" * 60)

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
