"""
================================================================================
BACKTESTING MODULE
Sistema de Backtesting para Estrategias de Trading
================================================================================

IMPORTANTE: Este modulo usa APENAS dados REAIS do mercado.
Nenhuma simulacao ou dados sinteticos sao permitidos.
Isso envolve dinheiro real, entao a precisao e crucial.

Componentes:
- BacktestEngine: Motor principal de backtesting
- HistoricalDataClient: Cliente de dados historicos reais
- Backtests individuais por indicador

Metricas calculadas:
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Calmar Ratio

Backtests disponiveis:
- backtest_prm.py: PRM (Protocolo Riemann-Mandelbrot)

Uso:
    python -m backtesting.backtest_prm --days 30 --symbol EURUSD
"""

from .backtest_engine import (
    BacktestEngine,
    BacktestResult,
    Trade,
    Position,
    PositionType,
    run_backtest
)

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'Trade',
    'Position',
    'PositionType',
    'run_backtest'
]
