#!/usr/bin/env python3
"""
================================================================================
ELIBOTCD - TRADING BOT
Sistema de Trading Automatizado baseado em Volatilidade
================================================================================

Fluxo de operacao:
1. Captura dados de cada tick via API (FXOpen WebSocket)
2. Classifica volatilidade do mercado (ALTA, MEDIA, BAIXA)
3. Roteia para estrategias apropriadas ao nivel de volatilidade
4. Gera sinais de compra/venda atraves de votacao majoritaria
5. (Opcional) Executa ordens na conta real

Estrategias:
- ALTA VOLATILIDADE (3): PRM, DTT, FIFN
- MEDIA VOLATILIDADE (10): LSQPC, H2PLO, RCTF, BPHS, KdVSH, FSIGE, HJBNES, MPSDEO, MVGKSD, FKQPIP
- BAIXA VOLATILIDADE (10): GJFCP, PRSBD, QZDLDT, RZCID, NSPPS, IPCBM, GLDMD, RDME, GMSCS, HBBP

Uso:
    python main.py [--execute] [--duration SECONDS]

    --execute: Habilita execucao real de ordens (CUIDADO!)
    --duration: Duracao do bot em segundos (0 = infinito)
"""

import asyncio
import argparse
import signal
import sys
from datetime import datetime, timezone
from typing import Optional

from config import settings
from api.fxopen_client import FXOpenClient, Tick
from strategies.orchestrator import StrategyOrchestrator
from strategies.strategy_factory import create_all_strategies, get_strategy_count
from strategies.base import Signal, SignalType


class EliBotCD:
    """
    Bot de Trading Principal

    Integra:
    - API FXOpen (dados em tempo real)
    - Classificador de volatilidade
    - 23 estrategias especializadas
    - Sistema de sinais agregados
    """

    def __init__(self,
                 symbol: str = "EURUSD",
                 min_confidence: float = 0.4,
                 signal_cooldown_ticks: int = 100,
                 execute_orders: bool = False):
        """
        Inicializa o bot

        Args:
            symbol: Par de moedas para operar
            min_confidence: Confianca minima para gerar sinal (0.0-1.0)
            signal_cooldown_ticks: Ticks de cooldown entre sinais
            execute_orders: Se True, executa ordens reais (CUIDADO!)
        """
        self.symbol = symbol
        self.execute_orders = execute_orders
        self.running = False

        # Cliente API
        self.client = FXOpenClient()

        # Orquestrador de estrategias
        self.orchestrator = StrategyOrchestrator(
            symbol=symbol,
            min_confidence=min_confidence,
            signal_cooldown_ticks=signal_cooldown_ticks
        )

        # Estatisticas
        self.start_time: Optional[datetime] = None
        self.total_signals = 0
        self.buy_signals = 0
        self.sell_signals = 0

        # Registra todas as estrategias reais
        self._register_strategies()

    def _register_strategies(self):
        """Registra todas as estrategias reais no orquestrador"""
        print("\n" + "=" * 70)
        print("  REGISTRANDO ESTRATEGIAS REAIS")
        print("=" * 70)

        strategies = create_all_strategies()

        for level, strats in strategies.items():
            print(f"\n{level} VOLATILIDADE:")
            for strategy in strats:
                self.orchestrator.register_strategy(strategy, level)

        counts = get_strategy_count()
        print(f"\nTotal: {counts['TOTAL']} estrategias registradas")
        print("=" * 70)

    async def on_tick(self, tick: Tick):
        """
        Callback para cada tick recebido

        Args:
            tick: Dados do tick
        """
        # Processa tick no orquestrador
        signal = self.orchestrator.process_tick(
            price=tick.mid,
            timestamp=tick.timestamp,
            volume=tick.bid_volume + tick.ask_volume
        )

        # Exibe status periodico
        state = self.orchestrator.get_state()
        if state.tick_count % 50 == 0:
            self._print_status(tick, state)

        # Processa sinal se houver
        if signal:
            self._process_signal(signal)

    def _print_status(self, tick: Tick, state):
        """Exibe status periodico"""
        print(f"[{state.tick_count:5d}] {tick.symbol} | "
              f"Price: {tick.mid:.5f} | "
              f"Vol: {state.current_volatility:.3f} pips | "
              f"Level: {state.volatility_level:10s} | "
              f"Hurst: {state.hurst:.2f} | "
              f"Entropy: {state.entropy:.2f} | "
              f"Signals: {self.total_signals}")

    def _process_signal(self, signal: Signal):
        """
        Processa um sinal gerado

        Args:
            signal: Sinal a processar
        """
        self.total_signals += 1

        if signal.type == SignalType.BUY:
            self.buy_signals += 1
            direction = "COMPRA"
            color_start = "\033[92m"  # Verde
        else:
            self.sell_signals += 1
            direction = "VENDA"
            color_start = "\033[91m"  # Vermelho

        color_end = "\033[0m"

        print(f"\n{'='*70}")
        print(f"{color_start}[SINAL #{self.total_signals}] {direction}{color_end}")
        print(f"  Preco: {signal.price:.5f}")
        print(f"  Confianca: {signal.confidence:.1%}")
        print(f"  Stop Loss: {signal.stop_loss:.5f}")
        print(f"  Take Profit: {signal.take_profit:.5f}")
        print(f"  Estrategia: {signal.strategy_name}")
        print(f"  Razao: {signal.reason}")
        print(f"  Timestamp: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

        if self.execute_orders:
            print(f"\n  {color_start}>>> EXECUTANDO ORDEM REAL <<<{color_end}")
            # TODO: Implementar execucao real via API de trade
            # self._execute_order(signal)
        else:
            print(f"\n  [Modo simulacao - ordem nao executada]")

        print(f"{'='*70}\n")

    async def run(self, duration_seconds: int = 0):
        """
        Executa o bot

        Args:
            duration_seconds: Duracao em segundos (0 = infinito)
        """
        self.running = True
        self.start_time = datetime.now(timezone.utc)

        print("\n" + "=" * 70)
        print("  ELIBOTCD - INICIANDO")
        print("=" * 70)
        print(f"  Simbolo: {self.symbol}")
        print(f"  Modo: {'REAL (ORDENS ATIVAS)' if self.execute_orders else 'SIMULACAO'}")
        print(f"  Duracao: {duration_seconds}s" if duration_seconds > 0 else "  Duracao: Infinita")
        print(f"  Confianca minima: {self.orchestrator.min_confidence:.0%}")
        print(f"  Cooldown: {self.orchestrator.signal_cooldown_ticks} ticks")
        print("=" * 70 + "\n")

        # Conecta a callback
        self.client.on_tick = self.on_tick

        try:
            # Conecta ao WebSocket
            if await self.client.connect():
                await self.client.subscribe(self.symbol)

                print(f"\nRecebendo ticks de {self.symbol}...")
                print("Pressione Ctrl+C para parar\n")
                print("-" * 70)

                # Loop principal
                if duration_seconds > 0:
                    await asyncio.sleep(duration_seconds)
                else:
                    while self.running:
                        await asyncio.sleep(1)
            else:
                print("Falha ao conectar!")

        except KeyboardInterrupt:
            print("\n\nInterrompido pelo usuario")
        except Exception as e:
            print(f"\nErro: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """Para o bot e exibe estatisticas finais"""
        self.running = False

        if self.client.connected:
            await self.client.disconnect()

        self._print_final_stats()

    def _print_final_stats(self):
        """Exibe estatisticas finais"""
        if not self.start_time:
            return

        duration = datetime.now(timezone.utc) - self.start_time
        stats = self.orchestrator.get_stats()
        state = self.orchestrator.get_state()

        print("\n" + "=" * 70)
        print("  ESTATISTICAS FINAIS")
        print("=" * 70)
        print(f"  Duracao: {duration}")
        print(f"  Ticks processados: {stats['ticks_processed']}")
        print(f"  Sinais gerados: {stats['signals_generated']}")
        print(f"    - Compras: {self.buy_signals}")
        print(f"    - Vendas: {self.sell_signals}")

        if stats['ticks_processed'] > 0:
            signal_rate = stats['signals_generated'] / stats['ticks_processed'] * 100
            print(f"  Taxa de sinais: {signal_rate:.2f}%")

        print(f"\n  Sinais por nivel de volatilidade:")
        for level, count in stats['signals_by_level'].items():
            print(f"    {level}: {count}")

        print(f"\n  Estado final do mercado:")
        print(f"    Preco: {state.current_price:.5f}")
        print(f"    Volatilidade: {state.current_volatility:.3f} pips")
        print(f"    Nivel: {state.volatility_level}")
        print(f"    Hurst: {state.hurst:.3f}")
        print(f"    Entropy: {state.entropy:.3f}")

        print("=" * 70 + "\n")


def signal_handler(signum, frame):
    """Handler para sinais do sistema"""
    print("\n\nRecebido sinal de interrupcao...")
    sys.exit(0)


async def main():
    """Funcao principal"""
    parser = argparse.ArgumentParser(
        description="EliBotCD - Trading Bot baseado em Volatilidade",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
    python main.py                    # Modo simulacao, infinito
    python main.py --duration 300     # Modo simulacao, 5 minutos
    python main.py --execute          # Modo real (CUIDADO!)
        """
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Habilita execucao real de ordens (CUIDADO!)"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Duracao do bot em segundos (0 = infinito)"
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default=settings.SYMBOL,
        help=f"Par de moedas (default: {settings.SYMBOL})"
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.4,
        help="Confianca minima para sinais (0.0-1.0, default: 0.4)"
    )

    parser.add_argument(
        "--cooldown",
        type=int,
        default=100,
        help="Ticks de cooldown entre sinais (default: 100)"
    )

    args = parser.parse_args()

    # Aviso para modo real
    if args.execute:
        print("\n" + "!" * 70)
        print("  ATENCAO: MODO DE EXECUCAO REAL ATIVADO!")
        print("  Ordens serao executadas na sua conta!")
        print("!" * 70)
        response = input("\nDigite 'CONFIRMAR' para continuar: ")
        if response != "CONFIRMAR":
            print("Cancelado.")
            return

    # Registra handler de sinal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Cria e executa o bot
    bot = EliBotCD(
        symbol=args.symbol,
        min_confidence=args.min_confidence,
        signal_cooldown_ticks=args.cooldown,
        execute_orders=args.execute
    )

    await bot.run(args.duration)


if __name__ == "__main__":
    asyncio.run(main())
