"""
================================================================================
Adaptador de Estratégia para o Detector de Tunelamento Topológico
================================================================================

VERSÃO V2.1 - CORREÇÕES DA AUDITORIA 24/12/2025:
1. Usa módulo compartilhado para cálculo de direção (consistência)
2. Tratamento de erros com logging (não silencia erros)
3. Documentação clara sobre look-ahead

Integra o indicador DTT com o sistema de trading.
================================================================================
"""
import logging
from datetime import datetime
from typing import Optional
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .dtt_tunelamento_topologico import DetectorTunelamentoTopologico

# Importar módulo compartilhado de direção
try:
    from backtesting.common.direction_calculator import (
        calculate_direction_from_closes,
        DEFAULT_DIRECTION_LOOKBACK,
        DIRECTION_LONG,
        DIRECTION_SHORT,
        DIRECTION_NEUTRAL
    )
    USE_SHARED_DIRECTION = True
except ImportError:
    USE_SHARED_DIRECTION = False
    DEFAULT_DIRECTION_LOOKBACK = 12

# Configurar logger
logger = logging.getLogger(__name__)


class DTTStrategy(BaseStrategy):
    """
    Estratégia baseada no Detector de Tunelamento Topológico

    Detecta "Tunnelling Events" - momentos onde a topologia do mercado
    indica que o preço está atravessando uma barreira de liquidez.

    VERSÃO V2.1 - CORREÇÕES DA AUDITORIA:
    - Direção via módulo compartilhado (consistência entre componentes)
    - Tratamento de erros com logging
    - Entry no OPEN da próxima barra (via BacktestEngine)

    NOTA SOBRE DIREÇÃO:
    A direção é calculada por momentum de barras fechadas, NÃO pela
    análise topológica. O DTT serve como FILTRO (trade_on/off), não
    como gerador de direção. Isso é intencional para evitar look-ahead
    no momentum_direction calculado pelo Schrödinger.
    """

    def __init__(self,
                 min_prices: int = 150,
                 stop_loss_pips: float = 25.0,
                 take_profit_pips: float = 50.0,
                 persistence_entropy_threshold: float = 0.5,
                 tunneling_probability_threshold: float = 0.15,
                 min_signal_strength: float = 0.3,
                 direction_lookback: int = DEFAULT_DIRECTION_LOOKBACK):
        """
        Inicializa a estratégia DTT

        Args:
            min_prices: Mínimo de preços necessários para análise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            persistence_entropy_threshold: Limiar de entropia para mercado não-trivial
            tunneling_probability_threshold: Limiar de probabilidade de tunelamento
            min_signal_strength: Força mínima do sinal para disparo
            direction_lookback: Barras para calcular direção (SEM look-ahead)
        """
        super().__init__(name="DTT-TunelamentoTopologico")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.min_signal_strength = min_signal_strength
        self.direction_lookback = direction_lookback

        # Buffer de preços para cálculo de direção SEM look-ahead
        self.prices = deque(maxlen=600)
        self.closes = deque(maxlen=600)  # Apenas closes para direção

        # Indicador DTT
        self.dtt = DetectorTunelamentoTopologico(
            max_points=150,  # Reduzido para performance
            use_dimensionality_reduction=True,
            reduction_method='pca',
            persistence_entropy_threshold=persistence_entropy_threshold,
            tunneling_probability_threshold=tunneling_probability_threshold
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0
        self.error_count = 0
        self.max_errors_to_log = 10  # Limitar logging de erros repetidos

        logger.info(f"DTTStrategy inicializada: min_prices={min_prices}, "
                    f"direction_lookback={direction_lookback}, "
                    f"use_shared_direction={USE_SHARED_DIRECTION}")

    def add_price(self, price: float):
        """Adiciona um preço ao buffer"""
        self.prices.append(price)
        self.closes.append(price)  # Para cálculo de direção

    def _calculate_direction(self) -> int:
        """
        Calcula direção usando módulo compartilhado ou fallback local.

        REGRAS ANTI LOOK-AHEAD:
        - closes[-1] = barra atual (momento do sinal) - NÃO USAR
        - closes[-2] = última barra completamente fechada - USAR
        - closes[-(lookback+1)] = barra de comparação - USAR

        Returns:
            1 para LONG, -1 para SHORT, 0 para NEUTRAL
        """
        if USE_SHARED_DIRECTION:
            # Usar módulo compartilhado para consistência
            return calculate_direction_from_closes(
                list(self.closes),
                self.direction_lookback
            )
        else:
            # Fallback: cálculo local (mantido para compatibilidade)
            if len(self.closes) < self.direction_lookback + 2:
                return 0

            recent_close = self.closes[-2]
            past_close = self.closes[-(self.direction_lookback + 1)]
            trend = recent_close - past_close
            return 1 if trend > 0 else -1

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal se houver tunnelling event

        VERSÃO V2.1 - CORREÇÕES:
        - Direção via módulo compartilhado
        - Tratamento de erros com logging
        - Entry será no OPEN da próxima barra (via BacktestEngine)

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
                # Verifica força mínima do sinal
                if result['signal_strength'] < self.min_signal_strength:
                    return None

                # V2.1: Calcula direção via módulo compartilhado
                direction_num = self._calculate_direction()

                if direction_num == 0:
                    return None

                direction = SignalType.BUY if direction_num == 1 else SignalType.SELL

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

                # Cria sinal COM stop_loss_pips e take_profit_pips
                signal = Signal(
                    type=direction,
                    price=price,
                    timestamp=timestamp,
                    strategy_name=self.name,
                    confidence=confidence,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    stop_loss_pips=self.stop_loss_pips,
                    take_profit_pips=self.take_profit_pips,
                    reason=self._generate_reason(result)
                )

                self.last_signal = signal
                self.signal_cooldown = 15  # Cooldown maior para DTT

                return signal

        except Exception as e:
            # V2.1: Tratamento de erros com logging (não silencia)
            self.error_count += 1
            if self.error_count <= self.max_errors_to_log:
                logger.warning(f"DTT análise falhou [{self.error_count}]: {e}")
            elif self.error_count == self.max_errors_to_log + 1:
                logger.warning(f"DTT: Erros subsequentes serão suprimidos...")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descrição do motivo do sinal"""
        entropy = result['entropy']
        tunneling = result['tunneling']

        return (f"Tunnelling Event | "
                f"Entropia: {entropy['persistence_entropy']:.3f} | "
                f"P(Tunnel): {tunneling['tunneling_probability']:.3f} | "
                f"Betti_1: {result['topology']['betti_1']}")

    def reset(self):
        """Reseta o estado da estratégia"""
        self.prices.clear()
        self.closes.clear()
        self.last_analysis = None
        self.last_signal = None
        self.signal_cooldown = 0
        self.error_count = 0

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

    def get_error_count(self) -> int:
        """Retorna contagem de erros para diagnóstico"""
        return self.error_count
