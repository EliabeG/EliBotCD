"""
Adaptador de Estratégia para o Protocolo Riemann-Mandelbrot
Integra o indicador PRM com o sistema de trading
"""
from datetime import datetime
from typing import Optional
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot


class PRMStrategy(BaseStrategy):
    """
    Estratégia baseada no Protocolo Riemann-Mandelbrot

    Detecta "Singularidades de Preço" - momentos onde todos os subsistemas
    (HMM, Lyapunov, Curvatura) concordam que há uma oportunidade de entrada.
    """

    def __init__(self,
                 min_prices: int = 100,
                 stop_loss_pips: float = 15.0,
                 take_profit_pips: float = 30.0,
                 hmm_threshold: float = 0.85,
                 lyapunov_threshold: float = 0.5,
                 curvature_threshold: float = 0.1):
        """
        Inicializa a estratégia PRM

        Args:
            min_prices: Mínimo de preços necessários para análise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            hmm_threshold: Threshold para ativação do HMM
            lyapunov_threshold: Threshold K para Lyapunov
            curvature_threshold: Threshold para aceleração da curvatura
        """
        super().__init__(name="PRM-RiemannMandelbrot")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de preços
        self.prices = deque(maxlen=500)
        self.volumes = deque(maxlen=500)

        # Indicador PRM
        self.prm = ProtocoloRiemannMandelbrot(
            n_states=3,
            hmm_threshold=hmm_threshold,
            lyapunov_threshold_k=lyapunov_threshold,
            curvature_threshold=curvature_threshold,
            lookback_window=100
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0  # Evita sinais em sequência

    def add_price(self, price: float, volume: float = None):
        """Adiciona um preço ao buffer"""
        self.prices.append(price)
        self.volumes.append(volume if volume else 1.0)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal se houver singularidade

        Args:
            price: Preço atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (volatility, hurst, entropy)

        Returns:
            Signal se singularidade detectada, None caso contrário
        """
        # Adiciona preço ao buffer
        volume = indicators.get('volume', 1.0)
        self.add_price(price, volume)

        # Verifica se temos dados suficientes
        if len(self.prices) < self.min_prices:
            return None

        # Cooldown para evitar sinais em sequência
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        # Converte para numpy arrays
        prices_array = np.array(self.prices)
        volumes_array = np.array(self.volumes)

        try:
            # Executa análise PRM
            result = self.prm.analyze(prices_array, volumes_array)
            self.last_analysis = result

            # Verifica se há singularidade
            if result['singularity_detected']:
                # Determina direção baseada na curvatura e tendência
                direction = self._determine_direction(result)

                if direction == SignalType.HOLD:
                    return None

                # Calcula níveis de stop e take profit
                pip_value = 0.0001  # Para EURUSD

                if direction == SignalType.BUY:
                    stop_loss = price - (self.stop_loss_pips * pip_value)
                    take_profit = price + (self.take_profit_pips * pip_value)
                else:  # SELL
                    stop_loss = price + (self.stop_loss_pips * pip_value)
                    take_profit = price - (self.take_profit_pips * pip_value)

                # Calcula confiança baseada nos scores
                confidence = self._calculate_confidence(result)

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
                self.signal_cooldown = 10  # Cooldown de 10 ticks

                return signal

        except Exception as e:
            # Log do erro mas continua operando
            print(f"Erro na análise PRM: {e}")

        return None

    def _determine_direction(self, result: dict) -> SignalType:
        """
        Determina a direção do trade baseado na análise

        Usa a curvatura e o estado do HMM para decidir direção
        """
        curvature_acc = result['curvature_analysis']['current_acceleration']
        hmm_state = result['hmm_analysis']['current_state']

        # Estado 1 (Alta Vol. Direcional) com curvatura positiva = BUY
        # Estado 1 com curvatura negativa = SELL
        # Estado 2 (Choque) = mais volátil, usa curvatura como guia

        if hmm_state == 1:  # Alta volatilidade direcional
            if curvature_acc > 0:
                return SignalType.BUY
            elif curvature_acc < 0:
                return SignalType.SELL
        elif hmm_state == 2:  # Choque de volatilidade
            # Em choques, seguir a curvatura com mais cautela
            if abs(curvature_acc) > self.prm.curvature_threshold * 1.5:
                if curvature_acc > 0:
                    return SignalType.BUY
                else:
                    return SignalType.SELL

        return SignalType.HOLD

    def _calculate_confidence(self, result: dict) -> float:
        """Calcula nível de confiança do sinal (0.0 a 1.0)"""
        # Componentes de confiança
        hmm_prob = result['Prob_HMM']
        lyap_score = result['Lyapunov_Score']
        curv_signal = result['Curvature_Signal']

        # Normaliza Lyapunov para 0-1 (assume range 0-0.5)
        lyap_normalized = min(lyap_score / 0.5, 1.0) if lyap_score > 0 else 0

        # Média ponderada
        confidence = (
            0.4 * hmm_prob +           # HMM tem maior peso
            0.3 * lyap_normalized +    # Lyapunov
            0.3 * curv_signal          # Curvatura
        )

        return min(max(confidence, 0.0), 1.0)

    def _generate_reason(self, result: dict) -> str:
        """Gera descrição do motivo do sinal"""
        hmm = result['hmm_analysis']
        lyap = result['lyapunov_analysis']

        state_names = ['Consolidação', 'Alta Vol. Direcional', 'Choque de Vol.']
        state_name = state_names[hmm['current_state']]

        return (f"Singularidade PRM | Estado: {state_name} "
                f"(P={hmm['current_prob']:.2f}) | "
                f"Lyap: {lyap['lyapunov_max']:.4f} ({lyap['classification']})")

    def reset(self):
        """Reseta o estado da estratégia"""
        self.prices.clear()
        self.volumes.clear()
        self.last_analysis = None
        self.last_signal = None
        self.signal_cooldown = 0
        self.prm.is_fitted = False

    def get_analysis_summary(self) -> Optional[dict]:
        """Retorna resumo da última análise"""
        if self.last_analysis is None:
            return None

        return {
            'singularity': self.last_analysis['singularity_detected'],
            'prob_hmm': self.last_analysis['Prob_HMM'],
            'lyapunov': self.last_analysis['Lyapunov_Score'],
            'curvature': self.last_analysis['Curvature_Signal'],
            'hmm_state': self.last_analysis['hmm_analysis']['current_state'],
            'lyap_class': self.last_analysis['lyapunov_analysis']['classification']
        }
