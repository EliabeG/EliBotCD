#!/usr/bin/env python3
"""
ELIBOTCD - Live Trading Bot (Verbose Mode)
Versão verbosa para diagnóstico
"""

import asyncio
import sys
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from api.fxopen_client import FXOpenClient, Tick
from api.fxopen_trade import FXOpenTradeClient, OrderSide, OrderType
from strategies.orchestrator import StrategyOrchestrator
from strategies.strategy_factory import create_all_strategies
from strategies.base import SignalType


class LiveTradingBotVerbose:
    def __init__(self, lot_size: float = 0.01):
        self.lot_size = lot_size
        self.running = False
        self.start_time = None

        self.feed_client = FXOpenClient()
        self.trade_client = FXOpenTradeClient()

        self.orchestrator = StrategyOrchestrator(
            symbol=settings.SYMBOL,
            min_confidence=0.4,  # Reduzido para ver mais sinais
            signal_cooldown_ticks=50  # Reduzido para ver mais sinais
        )

        self.tick_count = 0
        self.signal_count = 0
        self.last_volatility_level = None
        self.positions = {}
        self.trade_log = []

        self._register_strategies()

    def _register_strategies(self):
        print("\n" + "=" * 70)
        print("  REGISTRANDO ESTRATÉGIAS")
        print("=" * 70)

        strategies = create_all_strategies()
        total = 0

        for level, strats in strategies.items():
            for strategy in strats:
                self.orchestrator.register_strategy(strategy, level)
                total += 1

        print(f"Total: {total} estratégias registradas")
        print("=" * 70 + "\n")

    async def on_tick(self, tick: Tick):
        """Callback para cada tick - VERBOSE"""
        self.tick_count += 1

        # SEMPRE exibe os primeiros 10 ticks
        if self.tick_count <= 10:
            print(f"[TICK #{self.tick_count}] {tick.symbol} Bid:{tick.bid:.5f} Ask:{tick.ask:.5f} Mid:{tick.mid:.5f}")

        # Processa no orquestrador
        try:
            signal = self.orchestrator.process_tick(
                price=tick.mid,
                timestamp=tick.timestamp,
                volume=tick.bid_volume + tick.ask_volume
            )

            state = self.orchestrator.get_state()

            # Detecta mudança de volatilidade
            if state.volatility_level != self.last_volatility_level:
                print(f"\n{'*'*70}")
                print(f"  VOLATILIDADE: {self.last_volatility_level} -> {state.volatility_level}")
                print(f"  Vol: {state.current_volatility:.4f} pips | Hurst: {state.hurst:.3f}")
                print(f"{'*'*70}\n")
                self.last_volatility_level = state.volatility_level

            # Status a cada 20 ticks
            if self.tick_count % 20 == 0:
                elapsed = datetime.now(timezone.utc) - self.start_time
                print(f"[{elapsed}] Tick #{self.tick_count} | {tick.mid:.5f} | "
                      f"Vol:{state.current_volatility:.4f} | Level:{state.volatility_level} | "
                      f"H:{state.hurst:.2f} | E:{state.entropy:.2f} | Sinais:{self.signal_count}")

            # Processa sinal
            if signal:
                await self._process_signal(signal, tick, state)

        except Exception as e:
            print(f"[ERRO] Tick #{self.tick_count}: {e}")

    async def _process_signal(self, signal, tick, state):
        """Processa sinal gerado"""
        self.signal_count += 1

        color = '\033[92m' if signal.type == SignalType.BUY else '\033[91m'
        reset = '\033[0m'

        # Calcula SL/TP a partir de pips se não definidos
        pip_value = 0.0001  # Para EURUSD
        stop_loss = signal.stop_loss
        take_profit = signal.take_profit

        if not stop_loss and hasattr(signal, 'stop_loss_pips') and signal.stop_loss_pips:
            if signal.type == SignalType.BUY:
                stop_loss = tick.mid - (signal.stop_loss_pips * pip_value)
            else:
                stop_loss = tick.mid + (signal.stop_loss_pips * pip_value)
            print(f"  [CALC] SL calculado de {signal.stop_loss_pips} pips: {stop_loss:.5f}")

        if not take_profit and hasattr(signal, 'take_profit_pips') and signal.take_profit_pips:
            if signal.type == SignalType.BUY:
                take_profit = tick.mid + (signal.take_profit_pips * pip_value)
            else:
                take_profit = tick.mid - (signal.take_profit_pips * pip_value)
            print(f"  [CALC] TP calculado de {signal.take_profit_pips} pips: {take_profit:.5f}")

        print(f"\n{'='*70}")
        print(f"{color}[SINAL #{self.signal_count}] {signal.type.name}{reset}")
        print(f"  Preço: {signal.price:.5f}")
        print(f"  Confiança: {signal.confidence:.1%}")
        print(f"  SL: {stop_loss:.5f}" if stop_loss else "  SL: N/A")
        print(f"  TP: {take_profit:.5f}" if take_profit else "  TP: N/A")
        print(f"  Estratégia: {signal.strategy_name}")
        print(f"  Volatilidade: {state.volatility_level}")
        print(f"  Razão: {signal.reason[:100]}...")

        # Verifica se pode executar
        if tick.symbol in self.positions:
            print(f"  [BLOQUEADO] Já existe posição em {tick.symbol}")
            print(f"{'='*70}\n")
            return

        if signal.confidence < 0.5:
            print(f"  [BLOQUEADO] Confiança {signal.confidence:.1%} < 50%")
            print(f"{'='*70}\n")
            return

        if not stop_loss:
            print(f"  [BLOQUEADO] Sem stop loss")
            print(f"{'='*70}\n")
            return

        # EXECUTA ORDEM
        print(f"\n  {color}>>> EXECUTANDO ORDEM REAL <<<{reset}")

        side = OrderSide.BUY if signal.type == SignalType.BUY else OrderSide.SELL

        result = await self.trade_client.place_order(
            symbol=tick.symbol,
            side=side,
            volume=self.lot_size,
            order_type=OrderType.MARKET,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=f"EliBotCD_{signal.strategy_name[:15]}"
        )

        if result.success:
            print(f"  {color}✓ ORDEM EXECUTADA!{reset}")
            print(f"    ID: {result.order_id}")
            print(f"    Preço: {result.filled_price}")

            self.positions[tick.symbol] = {
                'side': side.value,
                'price': result.filled_price or tick.mid,
                'sl': stop_loss,
                'tp': take_profit,
                'strategy': signal.strategy_name,
                'time': datetime.now(timezone.utc).isoformat()
            }

            self.trade_log.append({
                'signal': self.signal_count,
                'type': side.value,
                'price': result.filled_price,
                'strategy': signal.strategy_name,
                'confidence': signal.confidence,
                'time': datetime.now(timezone.utc).isoformat()
            })
        else:
            print(f"  \033[91m✗ ERRO: {result.error_message}\033[0m")

        print(f"{'='*70}\n")

    async def run(self, duration_minutes: int = 0):
        """Executa o bot"""
        self.running = True
        self.start_time = datetime.now(timezone.utc)

        print("\n" + "!" * 70)
        print("  ELIBOTCD - LIVE TRADING (VERBOSE)")
        print("!" * 70)
        print(f"  Símbolo: {settings.SYMBOL}")
        print(f"  Lot: {self.lot_size}")
        print(f"  Duração: {duration_minutes} min" if duration_minutes > 0 else "  Duração: Infinita")
        print("!" * 70 + "\n")

        self.feed_client.on_tick = self.on_tick

        try:
            # Conecta Trade
            print("[1/3] Conectando Trade...")
            if not await self.trade_client.connect():
                print("FALHA Trade!")
                return
            await asyncio.sleep(2)

            # Conecta Feed
            print("[2/3] Conectando Feed...")
            if not await self.feed_client.connect():
                print("FALHA Feed!")
                return

            # Subscribe
            print("[3/3] Inscrevendo...")
            await self.feed_client.subscribe(settings.SYMBOL)

            print("\n" + "=" * 70)
            print("  BOT ATIVO - Aguardando ticks...")
            print("=" * 70 + "\n")

            if duration_minutes > 0:
                await asyncio.sleep(duration_minutes * 60)
            else:
                while self.running:
                    await asyncio.sleep(1)

        except KeyboardInterrupt:
            print("\n\nInterrompido")
        except Exception as e:
            print(f"\nErro: {e}")
        finally:
            self.running = False
            await self.feed_client.disconnect()
            await self.trade_client.disconnect()
            self._print_report()

    def _print_report(self):
        """Relatório final"""
        print("\n" + "=" * 70)
        print("  RELATÓRIO")
        print("=" * 70)
        print(f"  Ticks: {self.tick_count}")
        print(f"  Sinais: {self.signal_count}")
        print(f"  Trades: {len(self.trade_log)}")

        if self.positions:
            print(f"\n  Posições abertas:")
            for sym, pos in self.positions.items():
                print(f"    {pos['side']} {sym} @ {pos['price']}")

        if self.trade_log:
            log_file = f"/home/azureuser/EliBotCD/logs/trades_verbose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_file, 'w') as f:
                json.dump(self.trade_log, f, indent=2)
            print(f"\n  Log: {log_file}")

        print("=" * 70 + "\n")


async def main():
    # Sem confirmação para execução direta
    bot = LiveTradingBotVerbose(lot_size=0.01)
    await bot.run(duration_minutes=10)  # 10 minutos


if __name__ == "__main__":
    asyncio.run(main())
