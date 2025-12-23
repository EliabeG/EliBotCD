"""
Adaptador de Estrategia para o Korteweg-de Vries Soliton Hunter
Integra o indicador KdV-SH com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List, Dict
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .kdvsh_korteweg_devries import KdVSolitonHunter, SolitonType


class KdVSHStrategy(BaseStrategy):
    """
    Estrategia baseada no Korteweg-de Vries Soliton Hunter (KdV-SH)

    Usa Dinamica de Fluidos e Equacoes Diferenciais Parciais para detectar
    Ondas Solitarias (Solitons) no mercado. Um soliton e uma onda que
    mantem sua forma enquanto viaja - o movimento perfeito de swing.

    Conceitos-chave:
    - Equacao KdV: dphi/dt + d^3phi/dx^3 + 6*phi*(dphi/dx) = 0
    - Inverse Scattering Transform: Separa sinal estrutural do ruido
    - Numero de Ursell: Verifica regime KdV (media volatilidade)
    - Colisao de Solitons: Preve reflexao em barreiras de liquidez
    """

    def __init__(self,
                 min_prices: int = 100,
                 stop_loss_pips: float = 20.0,
                 take_profit_pips: float = 40.0,
                 ur_min: float = 0.3,
                 ur_max: float = 3.0,
                 n_points: int = 256,
                 eigenvalue_threshold: float = 0.001,
                 collision_horizon: int = 10):
        """
        Inicializa a estrategia KdV-SH

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            ur_min: Minimo do Numero de Ursell para regime KdV
            ur_max: Maximo do Numero de Ursell para regime KdV
            n_points: Pontos no grid para FFT (potencia de 2)
            eigenvalue_threshold: Threshold para detectar solitons
            collision_horizon: Horizonte de previsao de colisao em barras
        """
        super().__init__(name="KdVSH-SolitonHunter")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffers de precos e volumes
        self.prices = deque(maxlen=600)
        self.volumes = deque(maxlen=600)

        # Indicador KdV-SH
        self.kdvsh = KdVSolitonHunter(
            ur_min=ur_min,
            ur_max=ur_max,
            n_points=n_points,
            eigenvalue_threshold=eigenvalue_threshold,
            collision_horizon=collision_horizon,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def add_price(self, price: float, volume: float = None):
        """Adiciona um preco e volume ao buffer"""
        self.prices.append(price)
        if volume is not None:
            self.volumes.append(volume)
        else:
            # Volume sintetico baseado na variacao de preco
            if len(self.prices) > 1:
                delta = abs(price - self.prices[-2])
                synthetic_vol = delta * 50000 + np.random.rand() * 1000 + 500
                self.volumes.append(synthetic_vol)
            else:
                self.volumes.append(1000.0)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em KdV-SH

        Args:
            price: Preco atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (pode incluir 'volume')

        Returns:
            Signal se colisao de soliton detectada, None caso contrario
        """
        # Extrai volume se disponivel
        volume = indicators.get('volume', None)

        # Adiciona preco ao buffer
        self.add_price(price, volume)

        # Verifica se temos dados suficientes
        if len(self.prices) < self.min_prices:
            return None

        # Cooldown para evitar sinais em sequencia
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        # Converte para numpy arrays
        prices_array = np.array(self.prices)
        volumes_array = np.array(self.volumes)

        try:
            # Executa analise KdV-SH
            result = self.kdvsh.analyze(prices_array, volumes_array)
            self.last_analysis = result

            # Verifica sinal (ignora HIBERNATE e NEUTRAL)
            if result['signal'] != 0 and result['confidence'] >= 0.3:
                # Determina direcao
                if result['signal'] == 1:
                    direction = SignalType.BUY
                else:
                    direction = SignalType.SELL

                # Calcula niveis de stop e take profit
                pip_value = 0.0001

                if direction == SignalType.BUY:
                    stop_loss = price - (self.stop_loss_pips * pip_value)
                    take_profit = price + (self.take_profit_pips * pip_value)
                else:
                    stop_loss = price + (self.stop_loss_pips * pip_value)
                    take_profit = price - (self.take_profit_pips * pip_value)

                # Confianca
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
                self.signal_cooldown = 20  # Cooldown para KdV-SH

                return signal

        except Exception as e:
            print(f"Erro na analise KdV-SH: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        collision_info = ""
        if result['collision_predicted']:
            collision_info = f"Colisao={result['collision_type']}@{result['collision_time']:.1f}b | "

        return (f"KdVSH Soliton | "
                f"Ur={result['ursell_number']:.2f} | "
                f"N={result['n_solitons']} | "
                f"{collision_info}"
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.prices.clear()
        self.volumes.clear()
        self.kdvsh.reset()
        self.last_analysis = None
        self.last_signal = None
        self.signal_cooldown = 0

    def get_analysis_summary(self) -> Optional[dict]:
        """Retorna resumo da ultima analise"""
        if self.last_analysis is None:
            return None

        return {
            'signal': self.last_analysis['signal_name'],
            'confidence': self.last_analysis['confidence'],
            'ursell_number': self.last_analysis['ursell_number'],
            'ursell_regime': self.last_analysis.get('ursell_regime', 'N/A'),
            'n_solitons': self.last_analysis['n_solitons'],
            'dominant_soliton': self.last_analysis['dominant_soliton'],
            'collision_predicted': self.last_analysis['collision_predicted'],
            'collision_time': self.last_analysis['collision_time'],
            'collision_type': self.last_analysis['collision_type'],
            'n_supports': self.last_analysis.get('n_supports', 0),
            'n_resistances': self.last_analysis.get('n_resistances', 0),
            'predicted_reversal_price': self.last_analysis['predicted_reversal_price'],
            'current_price': self.last_analysis['current_price'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_soliton_info(self) -> Optional[dict]:
        """Retorna informacoes sobre solitons"""
        if self.last_analysis is None:
            return None
        return {
            'n_solitons': self.last_analysis['n_solitons'],
            'solitons': self.last_analysis['solitons'],
            'dominant': self.last_analysis['dominant_soliton']
        }

    def get_ursell_info(self) -> Optional[dict]:
        """Retorna informacoes sobre o Numero de Ursell"""
        if self.last_analysis is None:
            return None
        return {
            'ursell_number': self.last_analysis['ursell_number'],
            'regime': self.last_analysis.get('ursell_regime', 'N/A')
        }

    def get_collision_info(self) -> Optional[dict]:
        """Retorna informacoes sobre colisao prevista"""
        if self.last_analysis is None:
            return None
        return {
            'collision_predicted': self.last_analysis['collision_predicted'],
            'collision_time': self.last_analysis['collision_time'],
            'collision_type': self.last_analysis['collision_type'],
            'predicted_reversal_price': self.last_analysis['predicted_reversal_price']
        }

    def get_liquidity_barriers(self) -> Optional[dict]:
        """Retorna informacoes sobre barreiras de liquidez"""
        if self.last_analysis is None:
            return None
        return {
            'n_supports': self.last_analysis.get('n_supports', 0),
            'n_resistances': self.last_analysis.get('n_resistances', 0)
        }

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [ursell, n_solitons, collision, amplitude]
        """
        if self.last_analysis is None:
            return None

        dominant = self.last_analysis['dominant_soliton']
        amplitude = dominant['amplitude'] if dominant else 0.0

        return [
            self.last_analysis['ursell_number'],
            float(self.last_analysis['n_solitons']),
            1.0 if self.last_analysis['collision_predicted'] else 0.0,
            amplitude
        ]

    def is_in_kdv_regime(self) -> bool:
        """
        Verifica se o mercado esta em regime KdV

        Returns:
            True se Ur esta na faixa valida
        """
        if self.last_analysis is None:
            return False
        signal_name = self.last_analysis['signal_name']
        return signal_name not in ['HIBERNATE', 'INSUFFICIENT_DATA']

    def is_hibernating(self) -> bool:
        """
        Verifica se o indicador esta hibernando

        Returns:
            True se fora do regime KdV
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['signal_name'] == 'HIBERNATE'

    def has_solitons(self) -> bool:
        """
        Verifica se ha solitons detectados

        Returns:
            True se n_solitons > 0
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['n_solitons'] > 0

    def is_collision_imminent(self) -> bool:
        """
        Verifica se ha colisao iminente

        Returns:
            True se colisao prevista
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['collision_predicted']

    def get_ursell_number(self) -> Optional[float]:
        """
        Retorna o Numero de Ursell atual

        Returns:
            Valor do numero de Ursell
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['ursell_number']

    def get_n_solitons(self) -> Optional[int]:
        """
        Retorna o numero de solitons detectados

        Returns:
            Numero de solitons
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['n_solitons']

    def get_dominant_soliton_type(self) -> Optional[str]:
        """
        Retorna o tipo do soliton dominante

        Returns:
            BUYING_SOLITON ou SELLING_SOLITON
        """
        if self.last_analysis is None:
            return None
        dominant = self.last_analysis['dominant_soliton']
        if dominant:
            return dominant['type']
        return None

    def get_dominant_soliton_amplitude(self) -> Optional[float]:
        """
        Retorna a amplitude do soliton dominante

        Returns:
            Amplitude do soliton mais forte
        """
        if self.last_analysis is None:
            return None
        dominant = self.last_analysis['dominant_soliton']
        if dominant:
            return dominant['amplitude']
        return None

    def get_collision_time(self) -> Optional[float]:
        """
        Retorna o tempo ate a colisao prevista

        Returns:
            Tempo em barras ate a colisao
        """
        if self.last_analysis is None:
            return None
        if self.last_analysis['collision_predicted']:
            return self.last_analysis['collision_time']
        return None

    def get_predicted_reversal_price(self) -> Optional[float]:
        """
        Retorna o preco de reversao previsto

        Returns:
            Preco onde a reversao deve ocorrer
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['predicted_reversal_price']
