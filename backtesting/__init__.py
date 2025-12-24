"""
================================================================================
BACKTESTING MODULE
Sistema de Backtesting para Estrategias de Trading
================================================================================

IMPORTANTE: Este modulo usa APENAS dados REAIS do mercado.
Nenhuma simulacao ou dados sinteticos sao permitidos.
Isso envolve dinheiro real, entao a precisao e crucial.

Estrutura:
    backtesting/
    ├── common/          - Componentes compartilhados
    │   ├── backtest_engine.py
    │   └── robust_optimizer.py
    ├── prm/             - PRM-RiemannMandelbrot
    ├── dtt/             - DTT-TunelamentoTopologico
    ├── fifn/            - FIFN-FisherNavier
    ├── dsg/             - DSG-SingularidadeGravitacional
    ├── odmn/            - ODMN-MalliavinnNash
    └── generic/         - SEED, STGK, PHM, RHHF, SEMA

Uso:
    python -m backtesting.prm.optimizer
    python -m backtesting.dtt.optimizer
    python -m backtesting.generic.optimizer --strategy SEED
"""

from .common.backtest_engine import (
    BacktestEngine,
    BacktestResult,
    Trade,
    Position,
    PositionType,
    run_backtest
)

from .common.robust_optimizer import (
    RobustBacktester,
    RobustResult,
    save_robust_config
)

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'Trade',
    'Position',
    'PositionType',
    'run_backtest',
    'RobustBacktester',
    'RobustResult',
    'save_robust_config'
]
