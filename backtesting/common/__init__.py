"""
Common backtesting components.
"""
from .backtest_engine import BacktestEngine, BacktestResult, Trade, Position, PositionType, run_backtest
from .robust_optimizer import RobustBacktester, RobustResult, BacktestResult as RobustBacktestResult, save_robust_config

__all__ = [
    'BacktestEngine', 'BacktestResult', 'Trade', 'Position', 'PositionType', 'run_backtest',
    'RobustBacktester', 'RobustResult', 'RobustBacktestResult', 'save_robust_config'
]
