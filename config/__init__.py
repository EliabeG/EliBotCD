# config/__init__.py
"""Módulo de configuração do Trading Bot"""
from .settings import CONFIG, REGIME_CONFIG, DATA_MANAGER_CONFIG
from .settings import RISK_PARAMS as RISK_PARAMS_CONFIG  # Config baseado em env vars
from .risk_config import RISK_LIMITS, RISK_PARAMS, RISK_SCORE_WEIGHTS, RISK_ADJUSTMENTS, POSITION_SIZING, RISK_MATRIX

__all__ = [
    'CONFIG',
    'REGIME_CONFIG',
    'DATA_MANAGER_CONFIG',
    'RISK_PARAMS_CONFIG',
    'RISK_LIMITS',
    'RISK_PARAMS',
    'RISK_SCORE_WEIGHTS',
    'RISK_ADJUSTMENTS',
    'POSITION_SIZING',
    'RISK_MATRIX'
]

# ===================================