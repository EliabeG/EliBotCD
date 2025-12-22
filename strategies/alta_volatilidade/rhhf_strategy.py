"""
Adaptador de Estratégia para o Ressonador Hilbert-Huang Fractal
Integra o indicador RHHF com o sistema de trading
"""
from datetime import datetime
from typing import Optional
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .rhhf_ressonador_hilbert_huang import RessonadorHilbertHuangFractal


class RHHFStrategy(BaseStrategy):
    """
    Estratégia baseada no Ressonador Hilbert-Huang Fractal (RHHF)

    Usa decomposição EEMD (Ensemble Empirical Mode Decomposition) e
    Transformada de Hilbert para detectar "Chirps" - sinais de frequência
    crescente que precedem movimentos explosivos de volatilidade, similar
    ao sinal gravitacional de buracos negros colidindo.
    """

    def __init__(self,
                 min_prices: int = 150,
                 stop_loss_pips: float = 20.0,
                 take_profit_pips: float = 40.0,
                 n_ensembles: int = 50,
                 noise_amplitude: float = 0.2,
                 mirror_extension: int = 50,
                 chirp_threshold: float = 0.0,
                 fractal_threshold: float = 1.2):
        """
        Inicializa a estratégia RHHF

        Args:
            min_prices: Mínimo de preços necessários para análise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            n_ensembles: Número de repetições para EEMD
            noise_amplitude: Amplitude do ruído para EEMD
            mirror_extension: Pontos para extensão preditiva
            chirp_threshold: Limiar para detecção de chirp
            fractal_threshold: Limiar de dimensão fractal
        """
        super().__init__(name="RHHF-HilbertHuang")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de preços
        self.prices = deque(maxlen=800)

        # Indicador RHHF
        self.rhhf = RessonadorHilbertHuangFractal(
            n_ensembles=n_ensembles,
            noise_amplitude=noise_amplitude,
            mirror_extension=mirror_extension,
            use_predictive_extension=True,
            chirp_threshold=chirp_threshold,
            fractal_threshold=fractal_threshold
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def add_price(self, price: float):
        """Adiciona um preço ao buffer"""
        self.prices.append(price)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal se Chirp detectado

        Args:
            price: Preço atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais

        Returns:
            Signal se Chirp + condições atendidas, None caso contrário
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
            # Executa análise RHHF
            result = self.rhhf.analyze(prices_array)
            self.last_analysis = result

            # Verifica sinal
            signal_details = result['signal_details']

            if result['signal'] != 0:
                # Determina direção
                if result['signal'] == 1:
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

                # Confiança
                confidence = result['confidence']

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
                self.signal_cooldown = 15  # Cooldown para RHHF

                return signal

        except Exception as e:
            print(f"Erro na análise RHHF: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descrição do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"RHHF Chirp | "
                f"D_fractal={result['fractal_dimension']:.3f} | "
                f"Chirp={result['chirp_direction']} | "
                f"{reasons_str}")

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
            'signal': self.last_analysis['signal_name'],
            'confidence': self.last_analysis['confidence'],
            'chirp_detected': self.last_analysis['chirp_detected'],
            'chirp_direction': self.last_analysis['chirp_direction'],
            'fractal_dimension': self.last_analysis['fractal_dimension'],
            'fractal_trigger': self.last_analysis['fractal_trigger'],
            'n_imfs': self.last_analysis['n_imfs'],
            'current_price': self.last_analysis['current_price'],
            'cloud_value': self.last_analysis['cloud_value'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_imf_decomposition(self) -> Optional[dict]:
        """Retorna a decomposição EEMD da última análise"""
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('eemd')

    def get_chirp_data(self) -> Optional[dict]:
        """Retorna dados de detecção de chirp"""
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('chirp')

    def get_hilbert_spectrum(self) -> Optional[dict]:
        """Retorna o espectro de Hilbert"""
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('spectrum')
