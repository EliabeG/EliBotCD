#!/usr/bin/env python3
"""
================================================================================
ELIBOTCD - LIVE TRADING BOT
Sistema de Trading Automatizado com Execução Real
================================================================================

Este bot:
1. Conecta ao feed de dados em tempo real (WebSocket)
2. Conecta ao trade endpoint (WebSocket)
3. Processa sinais das 30 estratégias
4. EXECUTA ORDENS REAIS na conta

ATENÇÃO: Este bot executa ordens REAIS. Use com cautela!

Uso:
    python live_trading_bot.py [--lot-size 0.01] [--max-positions 3]
"""

import asyncio
import sys
import os
import signal
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from api.fxopen_client import FXOpenClient, Tick
from api.fxopen_trade import FXOpenTradeClient, OrderSide, OrderType, OrderResult
from strategies.orchestrator import StrategyOrchestrator
from strategies.strategy_factory import create_all_strategies
from strategies.base import SignalType


@dataclass
class Position:
    """Posição aberta"""
    symbol: str
    side: str
    volume: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float]
    take_profit: Optional[float]
    signal_strategy: str
    pnl: float = 0.0


@dataclass
class TradeStats:
    """Estatísticas de trading"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pips: float = 0.0
    max_drawdown: float = 0.0
    current_equity: float = 0.0
    starting_equity: float = 0.0


class LiveTradingBot:
    """
    Bot de Trading com Execução Real
    """

    def __init__(self,
                 lot_size: float = 0.01,
                 max_positions: int = 3,
                 min_confidence: float = 0.5,
                 cooldown_seconds: int = 60):
        """
        Inicializa o bot

        Args:
            lot_size: Tamanho do lote para cada trade
            max_positions: Máximo de posições simultâneas
            min_confidence: Confiança mínima para executar
            cooldown_seconds: Segundos entre trades do mesmo símbolo
        """
        self.lot_size = lot_size
        self.max_positions = max_positions
        self.min_confidence = min_confidence
        self.cooldown_seconds = cooldown_seconds

        self.running = False
        self.start_time: Optional[datetime] = None

        # Clientes
        self.feed_client = FXOpenClient()
        self.trade_client = FXOpenTradeClient()

        # Orquestrador
        self.orchestrator = StrategyOrchestrator(
            symbol=settings.SYMBOL,
            min_confidence=min_confidence,
            signal_cooldown_ticks=100
        )

        # Estado
        self.positions: Dict[str, Position] = {}
        self.last_trade_time: Dict[str, datetime] = {}
        self.stats = TradeStats()
        self.signal_count = 0

        # Log
        self.trade_log: List[Dict] = []

        # Registra estratégias
        self._register_strategies()

    def _register_strategies(self):
        """Registra todas as estratégias"""
        print("\n" + "=" * 70)
        print("  REGISTRANDO ESTRATÉGIAS PARA TRADING REAL")
        print("=" * 70)

        strategies = create_all_strategies()

        for level, strats in strategies.items():
            print(f"\n{level} VOLATILIDADE ({len(strats)} estratégias):")
            for strategy in strats:
                self.orchestrator.register_strategy(strategy, level)
                print(f"  - {strategy.name}")

        print("\n" + "=" * 70)

    async def on_tick(self, tick: Tick):
        """Callback para cada tick"""
        try:
            # Processa no orquestrador
            signal = self.orchestrator.process_tick(
                price=tick.mid,
                timestamp=tick.timestamp,
                volume=tick.bid_volume + tick.ask_volume
            )

            # Atualiza PnL das posições
            self._update_positions_pnl(tick)

            # Exibe status periódico
            state = self.orchestrator.get_state()
            if state.tick_count % 100 == 0:
                self._print_status(tick, state)

            # Processa sinal se houver
            if signal:
                await self._process_signal(signal, tick, state)

        except Exception as e:
            print(f"[ERROR] Erro ao processar tick: {e}")

    def _update_positions_pnl(self, tick: Tick):
        """Atualiza PnL das posições abertas"""
        for symbol, pos in self.positions.items():
            if symbol == tick.symbol:
                if pos.side == 'Buy':
                    pos.pnl = (tick.mid - pos.entry_price) * 10000 * pos.volume
                else:
                    pos.pnl = (pos.entry_price - tick.mid) * 10000 * pos.volume

    async def _process_signal(self, signal, tick: Tick, state):
        """Processa um sinal e decide se executa"""
        self.signal_count += 1

        # Verifica condições de entrada
        can_trade, reason = self._can_trade(signal, tick)

        if not can_trade:
            print(f"\n[SINAL #{self.signal_count}] {signal.type.name} - BLOQUEADO: {reason}")
            return

        # Executa ordem
        print(f"\n{'='*70}")
        print(f"[EXECUTANDO SINAL #{self.signal_count}]")
        print(f"{'='*70}")

        await self._execute_trade(signal, tick, state)

    def _can_trade(self, signal, tick: Tick) -> tuple:
        """Verifica se pode executar trade"""
        symbol = tick.symbol

        # Verifica máximo de posições
        if len(self.positions) >= self.max_positions:
            return False, f"Máximo de posições atingido ({self.max_positions})"

        # Verifica se já tem posição no símbolo
        if symbol in self.positions:
            return False, f"Já existe posição aberta em {symbol}"

        # Verifica cooldown
        if symbol in self.last_trade_time:
            elapsed = (datetime.now(timezone.utc) - self.last_trade_time[symbol]).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False, f"Cooldown ativo ({self.cooldown_seconds - elapsed:.0f}s restantes)"

        # Verifica confiança
        if signal.confidence < self.min_confidence:
            return False, f"Confiança insuficiente ({signal.confidence:.1%} < {self.min_confidence:.1%})"

        # Verifica stop loss
        if not signal.stop_loss:
            return False, "Sinal sem stop loss"

        return True, "OK"

    async def _execute_trade(self, signal, tick: Tick, state):
        """Executa a ordem"""
        symbol = tick.symbol
        side = OrderSide.BUY if signal.type == SignalType.BUY else OrderSide.SELL

        # Cor para output
        color = '\033[92m' if side == OrderSide.BUY else '\033[91m'
        reset = '\033[0m'

        print(f"  {color}Tipo: {side.value}{reset}")
        print(f"  Símbolo: {symbol}")
        print(f"  Volume: {self.lot_size} lote(s)")
        print(f"  Preço atual: {tick.mid:.5f}")
        print(f"  Stop Loss: {signal.stop_loss:.5f}")
        print(f"  Take Profit: {signal.take_profit:.5f}" if signal.take_profit else "  Take Profit: N/A")
        print(f"  Confiança: {signal.confidence:.1%}")
        print(f"  Estratégia: {signal.strategy_name}")
        print(f"  Volatilidade: {state.volatility_level}")

        # Envia ordem
        result = await self.trade_client.place_order(
            symbol=symbol,
            side=side,
            volume=self.lot_size,
            order_type=OrderType.MARKET,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            comment=f"EliBotCD_{signal.strategy_name[:20]}"
        )

        if result.success:
            print(f"\n  {color}✓ ORDEM EXECUTADA!{reset}")
            print(f"    Order ID: {result.order_id}")
            print(f"    Preço de execução: {result.filled_price}")

            # Registra posição
            self.positions[symbol] = Position(
                symbol=symbol,
                side=side.value,
                volume=self.lot_size,
                entry_price=result.filled_price or tick.mid,
                entry_time=datetime.now(timezone.utc),
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                signal_strategy=signal.strategy_name
            )

            # Atualiza cooldown
            self.last_trade_time[symbol] = datetime.now(timezone.utc)

            # Atualiza stats
            self.stats.total_trades += 1

            # Log
            self.trade_log.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'signal_number': self.signal_count,
                'type': 'OPEN',
                'side': side.value,
                'symbol': symbol,
                'volume': self.lot_size,
                'price': result.filled_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'strategy': signal.strategy_name,
                'confidence': signal.confidence,
                'volatility_level': state.volatility_level,
                'order_id': result.order_id
            })

        else:
            print(f"\n  \033[91m✗ ORDEM REJEITADA: {result.error_message}\033[0m")

            self.trade_log.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'signal_number': self.signal_count,
                'type': 'REJECTED',
                'side': side.value,
                'symbol': symbol,
                'error': result.error_message,
                'strategy': signal.strategy_name
            })

        print(f"{'='*70}\n")

    def _print_status(self, tick: Tick, state):
        """Exibe status periódico"""
        elapsed = datetime.now(timezone.utc) - self.start_time if self.start_time else timedelta(0)

        # Cores para volatilidade
        colors = {
            'BAIXA': '\033[92m',
            'MEDIA': '\033[93m',
            'ALTA': '\033[91m',
            'INDEFINIDO': '\033[0m'
        }
        vol_color = colors.get(state.volatility_level, '\033[0m')
        reset = '\033[0m'

        print(f"\n[{elapsed}] Tick #{state.tick_count}")
        print(f"  Preço: {tick.mid:.5f} | Spread: {tick.spread*10000:.1f} pips")
        print(f"  {vol_color}Volatilidade: {state.current_volatility:.3f} pips | Nível: {state.volatility_level}{reset}")
        print(f"  Hurst: {state.hurst:.3f} | Entropy: {state.entropy:.3f}")
        print(f"  Sinais: {self.signal_count} | Trades: {self.stats.total_trades}")
        print(f"  Posições abertas: {len(self.positions)}/{self.max_positions}")

        # Mostra posições
        if self.positions:
            print(f"  Posições:")
            for symbol, pos in self.positions.items():
                pnl_color = '\033[92m' if pos.pnl >= 0 else '\033[91m'
                print(f"    {pos.side} {pos.volume} {symbol} @ {pos.entry_price:.5f} | "
                      f"PnL: {pnl_color}{pos.pnl:+.1f} pips{reset}")

    def _print_final_report(self):
        """Exibe relatório final"""
        if not self.start_time:
            return

        elapsed = datetime.now(timezone.utc) - self.start_time

        print("\n" + "=" * 70)
        print("  RELATÓRIO FINAL DE TRADING")
        print("=" * 70)

        print(f"\n[RESUMO GERAL]")
        print(f"  Duração: {elapsed}")
        print(f"  Sinais recebidos: {self.signal_count}")
        print(f"  Trades executados: {self.stats.total_trades}")

        if self.stats.total_trades > 0:
            print(f"\n[PERFORMANCE]")
            print(f"  Winning: {self.stats.winning_trades}")
            print(f"  Losing: {self.stats.losing_trades}")
            win_rate = self.stats.winning_trades / self.stats.total_trades * 100 if self.stats.total_trades > 0 else 0
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Total PnL: {self.stats.total_pnl:+.2f} pips")

        print(f"\n[POSIÇÕES ABERTAS: {len(self.positions)}]")
        for symbol, pos in self.positions.items():
            print(f"  {pos.side} {pos.volume} {symbol} @ {pos.entry_price:.5f}")
            print(f"    SL: {pos.stop_loss} | TP: {pos.take_profit}")
            print(f"    PnL: {pos.pnl:+.1f} pips")
            print(f"    Estratégia: {pos.signal_strategy}")

        print("\n" + "=" * 70)

        # Salva log
        self._save_log()

    def _save_log(self):
        """Salva log de trades"""
        if not self.trade_log:
            return

        log_file = f"/home/azureuser/EliBotCD/logs/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        log_data = {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': datetime.now(timezone.utc).isoformat(),
            'settings': {
                'lot_size': self.lot_size,
                'max_positions': self.max_positions,
                'min_confidence': self.min_confidence,
                'cooldown_seconds': self.cooldown_seconds
            },
            'stats': {
                'total_trades': self.stats.total_trades,
                'signals_received': self.signal_count
            },
            'open_positions': [
                {
                    'symbol': p.symbol,
                    'side': p.side,
                    'volume': p.volume,
                    'entry_price': p.entry_price,
                    'pnl': p.pnl
                }
                for p in self.positions.values()
            ],
            'trades': self.trade_log
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

        print(f"\nLog salvo em: {log_file}")

    async def run(self, duration_minutes: int = 0):
        """Executa o bot"""
        self.running = True
        self.start_time = datetime.now(timezone.utc)

        print("\n" + "!" * 70)
        print("  ELIBOTCD - LIVE TRADING BOT")
        print("  >>> MODO DE EXECUÇÃO REAL <<<")
        print("!" * 70)
        print(f"  Símbolo: {settings.SYMBOL}")
        print(f"  Lot Size: {self.lot_size}")
        print(f"  Max Posições: {self.max_positions}")
        print(f"  Confiança Mínima: {self.min_confidence:.0%}")
        print(f"  Cooldown: {self.cooldown_seconds}s")
        print(f"  Duração: {duration_minutes} min" if duration_minutes > 0 else "  Duração: Infinita")
        print("!" * 70 + "\n")

        # Conecta callback de feed
        self.feed_client.on_tick = self.on_tick

        try:
            # Conecta ao trade endpoint primeiro
            print("[1/3] Conectando ao Trade endpoint...")
            if not await self.trade_client.connect():
                print("Falha ao conectar ao Trade!")
                return

            await asyncio.sleep(2)  # Aguarda info da conta

            # Conecta ao feed
            print("\n[2/3] Conectando ao Feed endpoint...")
            if not await self.feed_client.connect():
                print("Falha ao conectar ao Feed!")
                return

            # Subscribe
            print("\n[3/3] Inscrevendo no símbolo...")
            await self.feed_client.subscribe(settings.SYMBOL)

            print("\n" + "=" * 70)
            print("  BOT ATIVO - TRADING REAL")
            print("  Pressione Ctrl+C para parar")
            print("=" * 70 + "\n")

            # Loop principal
            if duration_minutes > 0:
                await asyncio.sleep(duration_minutes * 60)
            else:
                while self.running:
                    await asyncio.sleep(1)

        except KeyboardInterrupt:
            print("\n\nInterrompido pelo usuário")
        except Exception as e:
            print(f"\nErro: {e}")
        finally:
            self.running = False
            await self.feed_client.disconnect()
            await self.trade_client.disconnect()
            self._print_final_report()


async def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="EliBotCD - Live Trading Bot"
    )

    parser.add_argument(
        "--lot-size",
        type=float,
        default=0.01,
        help="Tamanho do lote (default: 0.01)"
    )

    parser.add_argument(
        "--max-positions",
        type=int,
        default=3,
        help="Máximo de posições simultâneas (default: 3)"
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Confiança mínima (default: 0.5)"
    )

    parser.add_argument(
        "--cooldown",
        type=int,
        default=60,
        help="Cooldown em segundos (default: 60)"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Duração em minutos (0 = infinito)"
    )

    args = parser.parse_args()

    # Confirmação
    print("\n" + "!" * 70)
    print("  ATENÇÃO: MODO DE TRADING REAL!")
    print("  Este bot EXECUTARÁ ORDENS na sua conta!")
    print("!" * 70)

    response = input("\nDigite 'CONFIRMAR' para continuar: ")
    if response != "CONFIRMAR":
        print("Cancelado.")
        return

    bot = LiveTradingBot(
        lot_size=args.lot_size,
        max_positions=args.max_positions,
        min_confidence=args.min_confidence,
        cooldown_seconds=args.cooldown
    )

    await bot.run(duration_minutes=args.duration)


if __name__ == "__main__":
    asyncio.run(main())
