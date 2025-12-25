"""
Adaptador de Estratégia para o Fluxo de Informação Fisher-Navier
Integra o indicador FIFN com o sistema de trading

VERSAO V3.4 - CORRIGIDO AUDITORIA 25:
- Usa módulo centralizado direction_calculator.py
- Implementa stops dinâmicos baseados em Reynolds
- Exclui barra atual para evitar look-ahead
- Calcula direção baseada em barras FECHADAS (igual ao optimizer)
- Usa direção para filtrar sinais
- Suporta volumes opcionais
"""
from datetime import datetime
from typing import Optional, Tuple
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .fifn_fisher_navier import FluxoInformacaoFisherNavier

# AUDITORIA 25: Usar módulo centralizado para cálculo de direção
from backtesting.common.direction_calculator import calculate_direction_from_closes


class FIFNStrategy(BaseStrategy):
    """
    Estratégia baseada no Fluxo de Informação Fisher-Navier

    Usa o Número de Reynolds para identificar a "Kill Zone" (Sweet Spot)
    onde breakouts institucionais limpos ocorrem.

    VERSAO V3.4 - CORRIGIDO AUDITORIA 25:
    - Usa módulo centralizado direction_calculator.py
    - Implementa stops dinâmicos baseados em Reynolds
    - Sem look-ahead bias (exclui barra atual)
    - Direção baseada em barras FECHADAS
    - Consistente com optimizer.py
    """

    # AUDITORIA 25: Mínimo de barras = lookback(10) + 2 + buffer
    MIN_BARS_FOR_DIRECTION = 14

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
        self.skewness_threshold = skewness_threshold

        # Buffer de preços e volumes
        self.prices = deque(maxlen=600)
        self.volumes = deque(maxlen=600)

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

    def add_price(self, price: float, volume: float = None):
        """Adiciona um preço e volume ao buffer"""
        self.prices.append(price)
        if volume is not None:
            self.volumes.append(volume)

    def _calculate_direction(self) -> int:
        """
        Calcula direção baseada em barras FECHADAS usando módulo centralizado.

        AUDITORIA 25: Usa calculate_direction_from_closes() do módulo centralizado
        - Garante consistência com optimizer.py e outros componentes
        - Compara closes[-2] vs closes[-12] = 10 barras de diferença
        """
        if len(self.prices) < self.MIN_BARS_FOR_DIRECTION:
            return 0

        # AUDITORIA 25: Usar função centralizada para garantir consistência
        return calculate_direction_from_closes(list(self.prices))

    def _calculate_dynamic_stops(self, reynolds: float) -> Tuple[float, float]:
        """
        AUDITORIA 25: Calcula stops dinâmicos baseados no regime de volatilidade.

        O Número de Reynolds indica o regime de mercado:
        - Turbulento (Re > 4000): Alta volatilidade → stops mais largos
        - Sweet Spot (2300-4000): Volatilidade moderada → stops padrão
        - Laminar (Re < 2000): Baixa volatilidade → stops mais apertados

        Returns:
            Tuple[stop_loss_pips, take_profit_pips]
        """
        base_sl = self.stop_loss_pips
        base_tp = self.take_profit_pips

        if reynolds > 4000:  # Turbulento - stops mais largos para evitar ruído
            multiplier = 1.5
        elif reynolds > 3000:  # Transição alta
            multiplier = 1.2
        elif reynolds < 2000:  # Laminar - stops mais apertados
            multiplier = 0.8
        elif reynolds < 2300:  # Transição baixa
            multiplier = 0.9
        else:  # Sweet Spot (2300-3000)
            multiplier = 1.0

        return base_sl * multiplier, base_tp * multiplier

    def analyze(self, price: float, timestamp: datetime, volume: float = None, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal se estiver na Kill Zone

        VERSAO V3.0 - CORRIGIDO AUDITORIA 11:
        - Exclui barra atual (prices_array[:-1])
        - Calcula direção baseada em barras FECHADAS
        - Filtra sinais usando direção (igual ao optimizer)

        Args:
            price: Preço atual
            timestamp: Timestamp do tick
            volume: Volume do tick (opcional)
            **indicators: Indicadores adicionais

        Returns:
            Signal se na Kill Zone com direção clara, None caso contrário
        """
        # Adiciona preço e volume ao buffer
        self.add_price(price, volume)

        # Verifica se temos dados suficientes
        if len(self.prices) < self.min_prices:
            return None

        # Cooldown para evitar sinais em sequência
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        # AUDITORIA 11 FIX #1: Excluir barra atual para evitar look-ahead
        # A barra atual ainda está em formação, não podemos usar seu valor
        prices_array = np.array(self.prices)[:-1]  # Exclui barra atual!

        # Preparar volumes se disponíveis
        volumes_array = None
        if len(self.volumes) > 0:
            volumes_array = np.array(self.volumes)[:-1]  # Também exclui volume atual

        try:
            # Executa análise FIFN (sem a barra atual)
            result = self.fifn.analyze(prices_array, volumes_array)
            self.last_analysis = result

            # AUDITORIA 11 FIX #2: Calcular direção baseada em barras FECHADAS
            trend_direction = self._calculate_direction()

            # Verifica sinal
            directional = result['directional_signal']
            skewness = directional['skewness']
            pressure_gradient = directional['pressure_gradient']

            # AUDITORIA 11 FIX #3: Filtrar usando direção (igual ao optimizer)
            # Consistente com optimizer.py linhas 437-445
            signal_type = None

            if directional['in_sweet_spot']:
                # LONG: skewness positiva, pressão negativa, tendência ALTA
                if (skewness > self.skewness_threshold and
                    pressure_gradient < 0 and
                    trend_direction == 1):
                    signal_type = SignalType.BUY

                # SHORT: skewness negativa, pressão positiva, tendência BAIXA
                elif (skewness < -self.skewness_threshold and
                      pressure_gradient > 0 and
                      trend_direction == -1):
                    signal_type = SignalType.SELL

            if signal_type is not None:
                # AUDITORIA 25: Calcula stops dinâmicos baseados em Reynolds
                reynolds = result['Reynolds_Number']
                dynamic_sl, dynamic_tp = self._calculate_dynamic_stops(reynolds)

                pip_value = 0.0001

                if signal_type == SignalType.BUY:
                    stop_loss = price - (dynamic_sl * pip_value)
                    take_profit = price + (dynamic_tp * pip_value)
                else:
                    stop_loss = price + (dynamic_sl * pip_value)
                    take_profit = price - (dynamic_tp * pip_value)

                # Calcula confiança
                confidence = directional['confidence']

                # Cria sinal
                signal = Signal(
                    type=signal_type,
                    price=price,
                    timestamp=timestamp,
                    strategy_name=self.name,
                    confidence=confidence,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=self._generate_reason(result, trend_direction)
                )

                self.last_signal = signal
                self.signal_cooldown = 12

                return signal

        except Exception as e:
            print(f"Erro na análise FIFN: {e}")

        return None

    def _generate_reason(self, result: dict, trend_direction: int = 0) -> str:
        """Gera descrição do motivo do sinal"""
        re_class = result['reynolds_classification']
        directional = result['directional_signal']
        trend_str = "UP" if trend_direction == 1 else ("DOWN" if trend_direction == -1 else "NEUTRAL")

        return (f"FIFN Kill Zone | "
                f"Re={re_class['reynolds']:.0f} ({re_class['state']}) | "
                f"Skew={directional['skewness']:.3f} | "
                f"KL={directional['kl_divergence']:.4f} | "
                f"Trend={trend_str}")

    def reset(self):
        """Reseta o estado da estratégia"""
        self.prices.clear()
        self.volumes.clear()
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
