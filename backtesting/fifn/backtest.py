#!/usr/bin/env python3
"""
================================================================================
BACKTEST FIFN - Fluxo de Informacao Fisher-Navier
Backtest com Dados Historicos REAIS
================================================================================

Este script executa backtest do indicador FIFN usando dados historicos REAIS.

IMPORTANTE: Nenhuma simulacao - apenas dados reais do mercado.

O indicador FIFN usa:
1. Equacao de Navier-Stokes - Fluxo de informacao
2. Numero de Reynolds - Regime turbulento/laminar
3. Fisher Information - Incerteza do sistema

Uso:
    python -m backtesting.backtest_fifn [--days DAYS] [--symbol SYMBOL]
"""

import sys
import os
import argparse
from datetime import datetime, timezone, timedelta

# Adiciona diretorio raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backtesting.common.backtest_engine import BacktestEngine, BacktestResult
from strategies.alta_volatilidade import FIFNStrategy


def run_fifn_backtest(
    symbol: str = "EURUSD",
    days: int = 30,
    periodicity: str = "H1",
    initial_capital: float = 10000.0,
    min_prices: int = 50,
    stop_loss_pips: float = 20.0,
    take_profit_pips: float = 40.0,
    verbose: bool = True
) -> BacktestResult:
    """
    Executa backtest do FIFN com dados REAIS

    Args:
        symbol: Par de moedas
        days: Dias de historico
        periodicity: Periodicidade
        initial_capital: Capital inicial
        min_prices: Minimo de precos
        stop_loss_pips: Stop loss
        take_profit_pips: Take profit
        verbose: Mostrar detalhes

    Returns:
        BacktestResult
    """
    print("\n" + "=" * 70)
    print("  BACKTEST FIFN - FLUXO DE INFORMACAO FISHER-NAVIER")
    print("  Dados Historicos REAIS do Mercado Forex")
    print("=" * 70)

    # Cria estrategia
    strategy = FIFNStrategy(
        min_prices=min_prices,
        stop_loss_pips=stop_loss_pips,
        take_profit_pips=take_profit_pips
    )

    # Configura engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        position_size=0.01,
        pip_value=0.0001,
        spread_pips=1.0,
        slippage_pips=0.5
    )

    # Define periodo
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    # Executa backtest
    result = engine.run(
        strategy=strategy,
        symbol=symbol,
        periodicity=periodicity,
        start_time=start_time,
        end_time=end_time,
        verbose=verbose
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Backtest FIFN com dados REAIS")

    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--periodicity", type=str, default="H1")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--stop-loss", type=float, default=20.0)
    parser.add_argument("--take-profit", type=float, default=40.0)

    args = parser.parse_args()

    run_fifn_backtest(
        symbol=args.symbol,
        days=args.days,
        periodicity=args.periodicity,
        initial_capital=args.capital,
        stop_loss_pips=args.stop_loss,
        take_profit_pips=args.take_profit
    )


if __name__ == "__main__":
    main()
