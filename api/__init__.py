# api/__init__.py
"""Módulo de APIs e conectividade"""
from .ticktrader_ws import TickTraderFeed, TickTraderTrade, TickData, DOMSnapshot
from .ticktrader_rest import TickTraderREST

# FIXClient é opcional (requer quickfix)
try:
    from .fix_client import FIXClient
    _HAS_FIX = True
except ImportError:
    FIXClient = None  # type: ignore
    _HAS_FIX = False

__all__ = [
    'TickTraderFeed',
    'TickTraderTrade',
    'TickData',
    'DOMSnapshot',
    'TickTraderREST',
    'FIXClient'
]

# ===================================