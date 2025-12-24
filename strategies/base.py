"""
Classe Base para todas as Estratégias
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Optional


class SignalType(Enum):
    """Tipo de sinal"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """Representa um sinal de trading"""
    type: SignalType
    price: float
    timestamp: datetime
    strategy_name: str
    confidence: float  # 0.0 a 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""

    # CORREÇÃO #5: Campos para stop/take em PIPS
    # O BacktestEngine usará estes para calcular níveis reais baseados na entrada
    stop_loss_pips: Optional[float] = None
    take_profit_pips: Optional[float] = None

    def __str__(self) -> str:
        direction = "COMPRA" if self.type == SignalType.BUY else "VENDA" if self.type == SignalType.SELL else "AGUARDAR"
        return (f"[{self.strategy_name}] {direction} @ {self.price:.5f} | "
                f"Confiança: {self.confidence:.0%} | {self.reason}")


class BaseStrategy(ABC):
    """Classe base para todas as estratégias"""

    def __init__(self, name: str):
        self.name = name
        self.is_active = True
        self.last_signal: Optional[Signal] = None

    @abstractmethod
    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna um sinal se houver.

        Args:
            price: Preço atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (volatilidade, hurst, entropy, etc)

        Returns:
            Signal se houver sinal, None caso contrário
        """
        pass

    @abstractmethod
    def reset(self):
        """Reseta o estado da estratégia"""
        pass

    def get_info(self) -> dict:
        """Retorna informações da estratégia"""
        return {
            'name': self.name,
            'active': self.is_active,
            'last_signal': str(self.last_signal) if self.last_signal else None
        }
