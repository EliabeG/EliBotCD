"""
Adaptador de Estratégia para o Fluxo de Informação Fisher-Navier
Integra o indicador FIFN com o sistema de trading
"""
from datetime import datetime
from typing import Optional
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .fifn_fisher_navier import FluxoInformacaoFisherNavier


class FIFNStrategy(BaseStrategy):
    """
    Estratégia baseada no Fluxo de Informação Fisher-Navier

    Usa o Número de Reynolds para identificar a "Kill Zone" (Sweet Spot)
    onde breakouts institucionais limpos ocorrem.
    """

    def __init__(self,
                 min_prices: int = 120,
                 stop_loss_pips: float = 18.0,
                 take_profit_pips: float = 36.0,
                 reynolds_sweet_low: float = 2300,
                 reynolds_sweet_high: float = 4000,
                 skewness_threshold: float = 0.5):
        """
        Inicializa a estratégia FIFN

        Args:
            min_prices: Mínimo de preços necessários para análise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            reynolds_sweet_low: Limite inferior da Kill Zone
            reynolds_sweet_high: Limite superior da Kill Zone
            skewness_threshold: Limiar de assimetria para sinal
        """
        super().__init__(name="FIFN-FisherNavier")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de preços
        self.prices = deque(maxlen=600)

        # Indicador FIFN
        self.fifn = FluxoInformacaoFisherNavier(
            window_size=50,
            kl_lookback=10,
            reynolds_sweet_low=reynolds_sweet_low,
            reynolds_sweet_high=reynolds_sweet_high,
            skewness_threshold=skewness_threshold
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def add_price(self, price: float):
        """Adiciona um preço ao buffer"""
        self.prices.append(price)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal se estiver na Kill Zone

        Args:
            price: Preço atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais

        Returns:
            Signal se na Kill Zone com direção clara, None caso contrário
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
            # Executa análise FIFN
            result = self.fifn.analyze(prices_array)
            self.last_analysis = result

            # Verifica sinal
            directional = result['directional_signal']

            if directional['signal'] != 0 and directional['in_sweet_spot']:
                # Determina direção
                if directional['signal'] == 1:
                    direction = SignalType.BUY
                else:
                    direction = SignalType.SELL

                # Calcula níveis de stop e take profit
                pip_value = 0.0001

                if direction == SignalType.BUY:
                    stop_loss = price - (self.stop_loss_pips * pip_value)
                    take_profit = price + (self.take_profit_pips * pip_value)
                else:
                    stop_loss = price + (self.stop_loss_pips * pip_value)
                    take_profit = price - (self.take_profit_pips * pip_value)

                # Calcula confiança
                confidence = directional['confidence']

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
                self.signal_cooldown = 12

                return signal

        except Exception as e:
            print(f"Erro na análise FIFN: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descrição do motivo do sinal"""
        re_class = result['reynolds_classification']
        directional = result['directional_signal']

        return (f"FIFN Kill Zone | "
                f"Re={re_class['reynolds']:.0f} ({re_class['state']}) | "
                f"Skew={directional['skewness']:.3f} | "
                f"KL={directional['kl_divergence']:.4f}")

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

        re_class = self.last_analysis['reynolds_classification']
        directional = self.last_analysis['directional_signal']

        return {
            'reynolds': re_class['reynolds'],
            'state': re_class['state'],
            'in_sweet_spot': re_class['in_sweet_spot'],
            'signal': directional['signal_name'],
            'confidence': directional['confidence'],
            'skewness': directional['skewness'],
            'kl_divergence': directional['kl_divergence'],
            'pressure_gradient': directional['pressure_gradient']
        }
