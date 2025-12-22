"""
Adaptador de Estratégia para o Hilbert-Huang Phase-Lock Oscillator
Integra o indicador H2-PLO com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .h2plo_hilbert_huang import HilbertHuangPhaseLockOscillator, PhaseAnalysis


class H2PLOStrategy(BaseStrategy):
    """
    Estratégia baseada no Hilbert-Huang Phase-Lock Oscillator (H2-PLO)

    Usa Decomposição Espectral Adaptativa para identificar pontos de
    inflexão cíclica instantânea sem lag. Baseado em:
    - CEEMDAN: Decomposição em Modos Intrínsecos (IMFs)
    - Transformada de Hilbert: Fase e Frequência Instantâneas
    - Phase Locking Value (PLV): Sincronização entre ciclos
    - Gatilho Sniper: Detecção de reversões no exato momento
    """

    def __init__(self,
                 min_prices: int = 100,
                 stop_loss_pips: float = 20.0,
                 take_profit_pips: float = 40.0,
                 ceemdan_trials: int = 50,
                 fast_imf_index: int = 1,
                 slow_imf_index: int = 2,
                 plv_threshold: float = 0.3,
                 phase_peak_range: Tuple[float, float] = (1.4, 1.7),
                 phase_trough_range: Tuple[float, float] = (4.5, 4.9)):
        """
        Inicializa a estratégia H2-PLO

        Args:
            min_prices: Mínimo de preços necessários para análise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            ceemdan_trials: Número de trials do CEEMDAN
            fast_imf_index: Índice do IMF rápido (ciclo curto)
            slow_imf_index: Índice do IMF lento (ciclo médio)
            plv_threshold: Threshold mínimo de PLV para sinal válido
            phase_peak_range: Range de fase para detectar topo
            phase_trough_range: Range de fase para detectar fundo
        """
        super().__init__(name="H2PLO-PhaseOscillator")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de preços
        self.prices = deque(maxlen=600)

        # Indicador H2-PLO
        self.h2plo = HilbertHuangPhaseLockOscillator(
            ceemdan_trials=ceemdan_trials,
            fast_imf_index=fast_imf_index,
            slow_imf_index=slow_imf_index,
            plv_threshold=plv_threshold,
            phase_peak_range=phase_peak_range,
            phase_trough_range=phase_trough_range,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def add_price(self, price: float):
        """Adiciona um preço ao buffer"""
        self.prices.append(price)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em H2-PLO

        Args:
            price: Preço atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais

        Returns:
            Signal se ponto de inflexão cíclica detectado, None caso contrário
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
            # Executa análise H2-PLO
            result = self.h2plo.analyze(prices_array)
            self.last_analysis = result

            # Verifica sinal
            if result['signal'] != 0 and result['confidence'] >= 0.5:
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
                self.signal_cooldown = 15  # Cooldown para H2-PLO

                return signal

        except Exception as e:
            print(f"Erro na análise H2-PLO: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descrição do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"H2PLO Phase | "
                f"Cycle={result['cycle_position_degrees']:.1f}° | "
                f"PLV={result['plv']:.3f} | "
                f"IMFs={result['n_imfs']} | "
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
            'cycle_position': self.last_analysis['cycle_position'],
            'cycle_position_degrees': self.last_analysis['cycle_position_degrees'],
            'frequency_acceleration': self.last_analysis['frequency_acceleration'],
            'phase_fast': self.last_analysis['phase_fast'],
            'phase_slow': self.last_analysis['phase_slow'],
            'plv': self.last_analysis['plv'],
            'amplitude': self.last_analysis['amplitude'],
            'n_imfs': self.last_analysis['n_imfs'],
            'mean_frequency_fast': self.last_analysis['mean_frequency_fast'],
            'mean_frequency_slow': self.last_analysis['mean_frequency_slow'],
            'current_price': self.last_analysis['current_price'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_imf_info(self) -> Optional[dict]:
        """Retorna informações sobre os IMFs"""
        if self.last_analysis is None:
            return None
        return {
            'n_imfs': self.last_analysis['n_imfs'],
            'imf_energies': self.last_analysis['imf_energies'],
            'mean_frequency_fast': self.last_analysis['mean_frequency_fast'],
            'mean_frequency_slow': self.last_analysis['mean_frequency_slow']
        }

    def get_phase_info(self) -> Optional[dict]:
        """Retorna informações sobre a fase"""
        if self.last_analysis is None:
            return None
        return {
            'cycle_position': self.last_analysis['cycle_position'],
            'cycle_position_degrees': self.last_analysis['cycle_position_degrees'],
            'phase_fast': self.last_analysis['phase_fast'],
            'phase_slow': self.last_analysis['phase_slow'],
            'frequency_acceleration': self.last_analysis['frequency_acceleration']
        }

    def get_synchronization_info(self) -> Optional[dict]:
        """Retorna informações sobre sincronização de fase"""
        if self.last_analysis is None:
            return None
        return {
            'plv': self.last_analysis['plv'],
            'amplitude': self.last_analysis['amplitude']
        }

    def get_imfs(self) -> Optional[np.ndarray]:
        """Retorna os IMFs calculados"""
        return self.h2plo.get_imfs()

    def get_hilbert_spectrum(self) -> Optional[Dict]:
        """Retorna dados do espectro de Hilbert-Huang"""
        return self.h2plo.get_hilbert_spectrum()

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saída principal

        Returns:
            Lista com [cycle_position, plv, amplitude, freq_accel]
        """
        if self.last_analysis is None:
            return None
        return [
            self.last_analysis['cycle_position'],
            self.last_analysis['plv'],
            self.last_analysis['amplitude'],
            self.last_analysis['frequency_acceleration']
        ]

    def is_at_cycle_peak(self) -> bool:
        """
        Verifica se está em topo cíclico

        Returns:
            True se fase próxima de π/2
        """
        if self.last_analysis is None:
            return False
        cycle = self.last_analysis['cycle_position']
        return 1.4 < cycle < 1.7

    def is_at_cycle_trough(self) -> bool:
        """
        Verifica se está em fundo cíclico

        Returns:
            True se fase próxima de 3π/2
        """
        if self.last_analysis is None:
            return False
        cycle = self.last_analysis['cycle_position']
        return 4.5 < cycle < 4.9

    def is_phase_locked(self, threshold: float = 0.5) -> bool:
        """
        Verifica se os ciclos estão sincronizados

        Args:
            threshold: Limiar mínimo de PLV

        Returns:
            True se PLV >= threshold
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['plv'] >= threshold

    def get_cycle_position_degrees(self) -> Optional[float]:
        """
        Retorna a posição no ciclo em graus

        Returns:
            Posição de 0° a 360°
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['cycle_position_degrees']

    def get_plv(self) -> Optional[float]:
        """
        Retorna o Phase Locking Value

        Returns:
            PLV entre 0 e 1
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['plv']

    def get_instantaneous_frequency(self) -> Optional[float]:
        """
        Retorna a frequência instantânea do ciclo rápido

        Returns:
            Frequência média do IMF rápido
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['mean_frequency_fast']

    def is_frequency_accelerating(self) -> bool:
        """
        Verifica se a frequência está acelerando

        Returns:
            True se dω/dt > 0
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['frequency_acceleration'] > 0

    def is_frequency_decelerating(self) -> bool:
        """
        Verifica se a frequência está desacelerando

        Returns:
            True se dω/dt < 0
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['frequency_acceleration'] < 0
