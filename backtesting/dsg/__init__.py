"""
DSG-SingularidadeGravitacional optimizer, backtesting and debugging.

VERSÃO CORRIGIDA - SEM LOOK-AHEAD BIAS
======================================
- EMA causal (substituiu gaussian_filter1d)
- Entrada no OPEN da próxima barra
- Stop/Take em PIPS (recalculado no entry real)
- Direção baseada em barras fechadas
- Evita trades simultâneos

Módulos:
    - optimizer.py: Otimização robusta com train/test split
    - backtest.py: Backtest com dados reais
    - debug.py: Análise de distribuição de sinais
"""

from .optimizer import DSGRobustOptimizer, DSGSignal
from .backtest import run_dsg_backtest, create_dsg_strategy, generate_report

__all__ = [
    'DSGRobustOptimizer',
    'DSGSignal',
    'run_dsg_backtest',
    'create_dsg_strategy',
    'generate_report',
]
