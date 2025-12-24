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

    VERSÃO V2.0 - SEM LOOK-AHEAD BIAS:
    - Direção baseada em barras FECHADAS (não usa momentum_direction)
    - Entry no OPEN da próxima barra (via BacktestEngine)
    """

    def __init__(self,
                 min_prices: int = 150,
                 stop_loss_pips: float = 25.0,
                 take_profit_pips: float = 50.0,
                 persistence_entropy_threshold: float = 0.5,
                 tunneling_probability_threshold: float = 0.15,
                 min_signal_strength: float = 0.3,
                 direction_lookback: int = 12):
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

    def add_price(self, price: float):
        """Adiciona um preço ao buffer"""
        self.prices.append(price)
        self.closes.append(price)  # Para cálculo de direção

    def _calculate_direction_from_closes(self) -> int:
        """
        CORREÇÃO V2.0: Calcula direção baseada APENAS em barras FECHADAS (sem look-ahead)

        No contexto do BacktestEngine:
        - analyze() é chamado com bar.close da barra que ACABOU de fechar
        - Portanto, closes[-1] é a barra que acabou de fechar (momento do sinal)
        - Para evitar look-ahead, usamos closes[-2] (barra ANTERIOR à do sinal)
        - E closes[-(direction_lookback+1)] para comparação

        Isso garante que a direção é baseada apenas em informação
        que estava disponível ANTES do momento do sinal.

        Returns:
            1 para LONG, -1 para SHORT, 0 para NEUTRAL
        """
        # Precisamos de pelo menos direction_lookback + 2 barras
        # (+1 para o lookback + 1 para a barra atual que não usamos)
        if len(self.closes) < self.direction_lookback + 2:
            return 0

        # closes[-1] = barra atual (momento do sinal) - NÃO USAR
        # closes[-2] = barra anterior (já fechada) - USAR
        # closes[-(direction_lookback+1)] = N barras antes da atual
        recent_close = self.closes[-2]
        past_close = self.closes[-(self.direction_lookback + 1)]

        trend = recent_close - past_close
        return 1 if trend > 0 else -1

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal se houver tunnelling event

        VERSÃO V2.0 - SEM LOOK-AHEAD BIAS:
        - Direção é calculada usando apenas barras FECHADAS
        - Entry será no OPEN da próxima barra (via BacktestEngine)
        - stop_loss_pips e take_profit_pips são passados no Signal

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

                # CORREÇÃO V2.0: Calcula direção baseada em barras FECHADAS
                # NÃO usa result['direction'] que pode ter look-ahead
                direction_num = self._calculate_direction_from_closes()

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
                # Isso permite que o BacktestEngine recalcule os níveis
                # baseado no preço de entrada REAL (OPEN da próxima barra)
                signal = Signal(
                    type=direction,
                    price=price,
                    timestamp=timestamp,
                    strategy_name=self.name,
                    confidence=confidence,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    stop_loss_pips=self.stop_loss_pips,      # CORREÇÃO V2.0
                    take_profit_pips=self.take_profit_pips,  # CORREÇÃO V2.0
                    reason=self._generate_reason(result)
                )

                self.last_signal = signal
                self.signal_cooldown = 15  # Cooldown maior para DTT

                return signal

        except Exception as e:
            # Silenciar erros para não poluir log
            pass

        return None

    def _get_direction(self, result: dict) -> SignalType:
        """
        DEPRECATED - NÃO USAR!

        Este método usa result['direction'] que pode ter look-ahead bias.
        Use _calculate_direction_from_closes() ao invés.

        Mantido apenas para compatibilidade, não é chamado no código atual.
        """
        import warnings
        warnings.warn(
            "_get_direction() is deprecated. Use _calculate_direction_from_closes() instead.",
            DeprecationWarning,
            stacklevel=2
        )
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
        self.closes.clear()
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
