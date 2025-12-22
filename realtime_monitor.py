#!/usr/bin/env python3
"""
EliBotCD - Monitor de Volatilidade em Tempo Real
Carrega dados históricos e atualiza instantaneamente com cada tick
"""
import asyncio
from datetime import datetime, timezone

from api import FXOpenClient, Tick
from indicators import RealtimeVolatility
from config import settings


class VolatilityMonitor:
    """Monitor de volatilidade em tempo real"""

    def __init__(self):
        self.client = FXOpenClient()
        self.indicator = RealtimeVolatility(
            candle_seconds=5,
            max_candles=500,
            parkinson_window=20,
            hurst_window=50,
            entropy_window=20
        )

        self.tick_count = 0
        self.last_classification = None

    async def start(self):
        """Inicia o monitor"""
        print("=" * 60)
        print("  EliBotCD - Monitor de Volatilidade em Tempo Real")
        print("=" * 60)
        print(f"Simbolo: {settings.SYMBOL}")
        print("=" * 60)
        print()

        # 1. Inicializa indicador (acumulará ticks em tempo real)
        print("[1/3] Inicializando indicador de volatilidade...")
        print("      Acumulando ticks em tempo real para cálculo instantâneo")

        # 2. Conecta ao WebSocket
        print("\n[2/3] Conectando ao feed em tempo real...")
        self.client.on_tick = self._on_tick

        if not await self.client.connect():
            print("Falha ao conectar!")
            return

        # 3. Inscreve no símbolo
        print("\n[3/3] Inscrevendo no símbolo...")
        await self.client.subscribe(settings.SYMBOL)

        print("\n" + "=" * 60)
        print("MONITORAMENTO ATIVO - Volatilidade atualiza a cada tick")
        print("=" * 60)
        print("Pressione Ctrl+C para sair\n")

        # Exibe estado inicial
        self._display_status()

        # Loop principal
        try:
            while self.client.connected:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nEncerrando...")
        finally:
            await self.client.disconnect()

    async def _on_tick(self, tick: Tick):
        """Callback para cada tick recebido"""
        self.tick_count += 1

        # Atualiza indicador instantaneamente
        self.indicator.add_tick(tick.mid, tick.timestamp)

        # Exibe atualização
        state = self.indicator.get_state()

        # Verifica se classificação mudou
        if state['classification'] != self.last_classification:
            self.last_classification = state['classification']
            self._display_alert(state)

        # Exibe status a cada 10 ticks
        if self.tick_count % 10 == 0:
            self._display_status()

    def _display_status(self):
        """Exibe status atual"""
        state = self.indicator.get_state()

        # Limpa linha e escreve
        classification = state['classification']

        # Cores ANSI
        if classification == "BAIXA":
            color = "\033[92m"  # Verde
        elif classification == "MEDIA":
            color = "\033[93m"  # Amarelo
        elif classification == "ALTA":
            color = "\033[91m"  # Vermelho
        else:
            color = "\033[0m"   # Reset

        reset = "\033[0m"

        hurst_str = "Tend" if state['hurst'] > 0.55 else "Rev" if state['hurst'] < 0.45 else "Neutro"

        print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
              f"Preço: {state['last_price']:.5f} | "
              f"Vol: {state['volatility']:.2f} pips | "
              f"H: {state['hurst']:.2f} ({hurst_str}) | "
              f"E: {state['entropy']:.2f} | "
              f"{color}>>> {classification} <<<{reset}    ", end="", flush=True)

    def _display_alert(self, state: dict):
        """Exibe alerta quando classificação muda"""
        classification = state['classification']

        print()  # Nova linha
        print()

        if classification == "BAIXA":
            print("\033[92m" + "=" * 60 + "\033[0m")
            print("\033[92m  ⚡ VOLATILIDADE MUDOU PARA: BAIXA\033[0m")
            print("\033[92m  → Mercado em consolidação / Aguardar breakout\033[0m")
            print("\033[92m" + "=" * 60 + "\033[0m")
        elif classification == "MEDIA":
            print("\033[93m" + "=" * 60 + "\033[0m")
            print("\033[93m  ⚡ VOLATILIDADE MUDOU PARA: MÉDIA\033[0m")
            print("\033[93m  → Mercado operável / Condições normais\033[0m")
            print("\033[93m" + "=" * 60 + "\033[0m")
        elif classification == "ALTA":
            print("\033[91m" + "=" * 60 + "\033[0m")
            print("\033[91m  ⚡ VOLATILIDADE MUDOU PARA: ALTA\033[0m")
            print("\033[91m  → Cautela! Mercado com alta atividade\033[0m")
            print("\033[91m" + "=" * 60 + "\033[0m")

        print()


async def main():
    monitor = VolatilityMonitor()
    await monitor.start()


if __name__ == "__main__":
    asyncio.run(main())
