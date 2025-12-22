"""
Adaptador de Estratégia para o Detector de Tunelamento Topológico
Integra o indicador DTT com o sistema de trading
"""
from datetime import datetime
from typing import Optional
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .dtt_tunelamento_topologico import DetectorTunelamentoTopologico


class DTTStrategy(BaseStrategy):
    """
    Estratégia baseada no Detector de Tunelamento Topológico

    Detecta "Tunnelling Events" - momentos onde a topologia do mercado
    indica que o preço está atravessando uma barreira de liquidez.
    """

    def __init__(self,
                 min_prices: int = 150,
                 stop_loss_pips: float = 20.0,
                 take_profit_pips: float = 40.0,
                 persistence_entropy_threshold: float = 0.3,
                 tunneling_probability_threshold: float = 0.15):
        """
        Inicializa a estratégia DTT

        Args:
            min_prices: Mínimo de preços necessários para análise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            persistence_entropy_threshold: Limiar de entropia para mercado não-trivial
            tunneling_probability_threshold: Limiar de probabilidade de tunelamento
        """
        super().__init__(name="DTT-TunelamentoTopologico")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de preços
        self.prices = deque(maxlen=600)

        # Indicador DTT
        self.dtt = DetectorTunelamentoTopologico(
            max_points=200,
            use_dimensionality_reduction=True,
            reduction_method='pca',
            persistence_entropy_threshold=persistence_entropy_threshold,
            tunneling_probability_threshold=tunneling_probability_threshold
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def add_price(self, price: float):
        """Adiciona um preço ao buffer"""
        self.prices.append(price)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal se houver tunnelling event

        Args:
            price: Preço atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais

        Returns:
            Signal se tunnelling detectado, None caso contrário
        """
        # Adiciona preço ao buffer
        self.add_price(price)

        # Verifica se temos dados suficientes
        if len(self.prices) < self.min_prices:
            return None

        # Cooldown para evitar sinais em sequência
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        # Converte para numpy array
        prices_array = np.array(self.prices)

        try:
            # Executa análise DTT
            result = self.dtt.analyze(prices_array)
            self.last_analysis = result

            # Verifica se há tunnelling event
            if result['trade_on']:
                # Determina direção
                direction = self._get_direction(result)

                if direction == SignalType.HOLD:
                    return None

                # Calcula níveis de stop e take profit
                pip_value = 0.0001

                if direction == SignalType.BUY:
                    stop_loss = price - (self.stop_loss_pips * pip_value)
                    take_profit = price + (self.take_profit_pips * pip_value)
                else:  # SELL
                    stop_loss = price + (self.stop_loss_pips * pip_value)
                    take_profit = price - (self.take_profit_pips * pip_value)

                # Calcula confiança
                confidence = result['signal_strength']

                # Cria sinal
                signal = Signal(
                    type=direction,
                    price=price,
                    timestamp=timestamp,
                    strategy_name=self.name,
                    confidence=confidence,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=self._generate_reason(result)
                )

                self.last_signal = signal
                self.signal_cooldown = 15  # Cooldown maior para DTT

                return signal

        except Exception as e:
            print(f"Erro na análise DTT: {e}")

        return None

    def _get_direction(self, result: dict) -> SignalType:
        """Determina a direção do trade baseado na análise"""
        direction = result['direction']

        if direction == 'LONG':
            return SignalType.BUY
        elif direction == 'SHORT':
            return SignalType.SELL
        else:
            return SignalType.HOLD

    def _generate_reason(self, result: dict) -> str:
        """Gera descrição do motivo do sinal"""
        decision = result['decision']
        entropy = result['entropy']
        tunneling = result['tunneling']

        return (f"Tunnelling Event | "
                f"Entropia: {entropy['persistence_entropy']:.3f} | "
                f"P(Tunnel): {tunneling['tunneling_probability']:.3f} | "
                f"Betti_1: {result['topology']['betti_1']}")

    def reset(self):
        """Reseta o estado da estratégia"""
        self.prices.clear()
        self.last_analysis = None
        self.last_signal = None
        self.signal_cooldown = 0

    def get_analysis_summary(self) -> Optional[dict]:
        """Retorna resumo da última análise"""
        if self.last_analysis is None:
            return None

        return {
            'trade_on': self.last_analysis['trade_on'],
            'direction': self.last_analysis['direction'],
            'signal_strength': self.last_analysis['signal_strength'],
            'persistence_entropy': self.last_analysis['entropy']['persistence_entropy'],
            'tunneling_probability': self.last_analysis['tunneling']['tunneling_probability'],
            'betti_1': self.last_analysis['topology']['betti_1'],
            'tda_backend': self.last_analysis['tda_backend']
        }
