#!/usr/bin/env python3
"""
================================================================================
BACKTEST PRM STANDALONE COM SPREAD REAL
================================================================================

Executa backtest diretamente com barras pré-baixadas para evitar re-download.
Usa lógica simplificada compatível com PRMOptimizedStrategy.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from api.fxopen_historical_ws import Bar, get_historical_data_with_spread_sync
from strategies.alta_volatilidade.prm_optimized_strategy import PRMOptimizedStrategy
from strategies.base import Signal, SignalType

# Importar custos
from config.execution_costs import SPREAD_PIPS, SLIPPAGE_PIPS, COMMISSION_PER_LOT


class PositionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    type: PositionType
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float


@dataclass
class Trade:
    type: PositionType
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    pnl_pips: float
    exit_reason: str


def run_simple_backtest(bars: List[Bar], strategy: PRMOptimizedStrategy,
                        position_size: float = 0.01,
                        use_real_spread: bool = True) -> List[Trade]:
    """
    Backtest simplificado que usa barras pré-baixadas

    Args:
        bars: Lista de barras com dados BID/ASK
        strategy: Estratégia PRM otimizada
        position_size: Tamanho em lotes
        use_real_spread: Se True, usa spread real de cada barra
    """
    trades = []
    position: Optional[Position] = None
    pending_signal: Optional[Signal] = None
    pip = 0.0001

    print(f"\n  Executando backtest em {len(bars)} barras...")

    for i, bar in enumerate(bars):
        # 1. Executar sinal pendente no OPEN desta barra
        if pending_signal and position is None:
            # Aplicar spread e slippage
            if use_real_spread and bar.has_spread_data:
                spread = bar.spread_pips * pip
            else:
                spread = SPREAD_PIPS * pip
            slippage = SLIPPAGE_PIPS * pip

            if pending_signal.type == SignalType.BUY:
                # Compra no ASK
                if bar.ask_open:
                    entry_price = bar.ask_open + slippage
                else:
                    entry_price = bar.open + spread / 2 + slippage
                pos_type = PositionType.LONG
                sl = entry_price - (strategy.stop_loss_pips * pip)
                tp = entry_price + (strategy.take_profit_pips * pip)
            else:
                # Venda no BID
                if bar.bid_open:
                    entry_price = bar.bid_open - slippage
                else:
                    entry_price = bar.open - spread / 2 - slippage
                pos_type = PositionType.SHORT
                sl = entry_price + (strategy.stop_loss_pips * pip)
                tp = entry_price - (strategy.take_profit_pips * pip)

            position = Position(
                type=pos_type,
                entry_price=entry_price,
                entry_time=bar.timestamp,
                stop_loss=sl,
                take_profit=tp
            )
            pending_signal = None

        # 2. Verificar stop/take para posição aberta
        if position:
            exit_price = None
            exit_reason = None

            if use_real_spread and bar.has_spread_data:
                bid_close = bar.bid_close if bar.bid_close else bar.close
                ask_close = bar.ask_close if bar.ask_close else bar.close
                bid_low = bar.bid_low if bar.bid_low else bar.low
                ask_high = bar.ask_high if bar.ask_high else bar.high
            else:
                bid_close = bar.close
                ask_close = bar.close
                bid_low = bar.low
                ask_high = bar.high

            if position.type == PositionType.LONG:
                # Stop loss (sai no BID)
                if bid_low <= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = "stop_loss"
                # Take profit (sai no BID)
                elif ask_high >= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "take_profit"
            else:  # SHORT
                # Stop loss (sai no ASK)
                if ask_high >= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = "stop_loss"
                # Take profit (sai no ASK)
                elif bid_low <= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "take_profit"

            if exit_price:
                # Calcular PnL
                if position.type == PositionType.LONG:
                    pnl_pips = (exit_price - position.entry_price) / pip
                else:
                    pnl_pips = (position.entry_price - exit_price) / pip

                trades.append(Trade(
                    type=position.type,
                    entry_price=position.entry_price,
                    entry_time=position.entry_time,
                    exit_price=exit_price,
                    exit_time=bar.timestamp,
                    pnl_pips=pnl_pips,
                    exit_reason=exit_reason
                ))
                position = None

        # 3. Gerar novo sinal (se não há posição)
        if position is None and pending_signal is None:
            signal = strategy.analyze(
                price=bar.close,
                timestamp=bar.timestamp,
                volume=bar.volume,
                high=bar.high,
                low=bar.low,
                open=bar.open
            )
            if signal:
                pending_signal = signal

        # Progresso
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(bars)} barras, {len(trades)} trades")

    # Fechar posição aberta no final
    if position:
        last_bar = bars[-1]
        if position.type == PositionType.LONG:
            exit_price = last_bar.bid_close if last_bar.bid_close else last_bar.close
            pnl_pips = (exit_price - position.entry_price) / pip
        else:
            exit_price = last_bar.ask_close if last_bar.ask_close else last_bar.close
            pnl_pips = (position.entry_price - exit_price) / pip

        trades.append(Trade(
            type=position.type,
            entry_price=position.entry_price,
            entry_time=position.entry_time,
            exit_price=exit_price,
            exit_time=last_bar.timestamp,
            pnl_pips=pnl_pips,
            exit_reason="end_of_data"
        ))

    return trades


def main():
    print("=" * 70)
    print("  BACKTEST PRM COM SPREAD REAL")
    print("  Usando PRMOptimizedStrategy (compatível com parâmetros otimizados)")
    print("=" * 70)

    # Período mais curto para teste rápido
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=30)  # Apenas 30 dias para teste

    print(f"\n  Período: {start_time.date()} a {end_time.date()}")
    print(f"  Par: EURUSD M5")

    # Baixar dados com spread real
    print("\n  Baixando dados com spread real...")
    bars = get_historical_data_with_spread_sync(
        symbol="EURUSD",
        periodicity="M5",
        start_time=start_time,
        end_time=end_time
    )

    if not bars:
        print("  ERRO: Falha ao baixar dados")
        return

    print(f"  Barras: {len(bars)}")

    # Estatísticas de spread
    spreads = [bar.spread_pips for bar in bars if bar.has_spread_data]
    if spreads:
        print(f"\n  Spread (pips):")
        print(f"    Min: {min(spreads):.2f}")
        print(f"    Max: {max(spreads):.2f}")
        print(f"    Média: {np.mean(spreads):.2f}")
        print(f"    Mediana: {np.median(spreads):.2f}")

    # Criar estratégia otimizada
    print("\n" + "=" * 70)
    print("  INICIALIZANDO ESTRATÉGIA")
    print("=" * 70)

    strategy = PRMOptimizedStrategy(load_optimized_config=True)

    print(f"\n  Parâmetros carregados:")
    print(f"    hmm_threshold: {strategy.hmm_threshold}")
    print(f"    lyapunov_threshold: {strategy.lyapunov_threshold}")
    print(f"    hmm_states_allowed: {strategy.hmm_states_allowed}")
    print(f"    stop_loss_pips: {strategy.stop_loss_pips}")
    print(f"    take_profit_pips: {strategy.take_profit_pips}")

    # Executar backtest
    print("\n" + "=" * 70)
    print("  EXECUTANDO BACKTEST")
    print("=" * 70)

    trades = run_simple_backtest(bars, strategy, use_real_spread=True)

    # Mostrar resultados
    print("\n" + "=" * 70)
    print("  RESULTADOS")
    print("=" * 70)

    print(f"\n  Período: {start_time.date()} a {end_time.date()}")
    print(f"  Barras: {len(bars)}")

    if not trades:
        print("\n  NENHUM TRADE GERADO!")
        print("  Isso pode indicar que a estratégia não encontrou sinais válidos")
        print("  ou que os thresholds estão muito restritivos.")
        print("\n" + "=" * 70)
        return

    # Calcular métricas
    pnls = [t.pnl_pips for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_pnl_pips = sum(pnls)

    # USD por pip (para 0.01 lotes EURUSD = $0.10/pip)
    usd_per_pip = 0.10
    total_pnl_usd = total_pnl_pips * usd_per_pip

    win_rate = len(wins) / len(trades) if trades else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss

    print(f"\n  TRADES:")
    print(f"    Total: {len(trades)}")
    print(f"    Vencedores: {len(wins)}")
    print(f"    Perdedores: {len(losses)}")
    print(f"    Win Rate: {win_rate:.1%}")

    print(f"\n  PERFORMANCE:")
    print(f"    PnL Total: ${total_pnl_usd:.2f} ({total_pnl_pips:.1f} pips)")
    print(f"    Profit Factor: {profit_factor:.2f}")
    print(f"    Avg Trade: {np.mean(pnls):.1f} pips")
    print(f"    Max Win: {max(pnls):.1f} pips")
    print(f"    Max Loss: {min(pnls):.1f} pips")

    # Calcular drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    drawdowns = peak - equity
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

    print(f"\n  RISCO:")
    print(f"    Max Drawdown: {max_dd:.1f} pips")

    # Por tipo de saída
    stops = [t for t in trades if t.exit_reason == "stop_loss"]
    takes = [t for t in trades if t.exit_reason == "take_profit"]
    print(f"\n  SAÍDAS:")
    print(f"    Stop Loss: {len(stops)} ({len(stops)/len(trades)*100:.0f}%)")
    print(f"    Take Profit: {len(takes)} ({len(takes)/len(trades)*100:.0f}%)")

    # Listar alguns trades
    print(f"\n  ÚLTIMOS 10 TRADES:")
    for trade in trades[-10:]:
        pnl_sign = "+" if trade.pnl_pips > 0 else ""
        print(f"    {trade.entry_time.strftime('%Y-%m-%d %H:%M')} | "
              f"{trade.type.name:5} | "
              f"Entry: {trade.entry_price:.5f} | "
              f"Exit: {trade.exit_price:.5f} | "
              f"PnL: {pnl_sign}{trade.pnl_pips:.1f} pips | "
              f"{trade.exit_reason}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
