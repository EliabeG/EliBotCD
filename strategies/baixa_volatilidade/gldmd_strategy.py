"""
Adaptador de Estrategia para o Gravitational Lensing Dark Matter Detector
Integra o indicador GL-DMD com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .gldmd_gravitational_lensing import (
    GravitationalLensingDarkMatterDetector,
    LensType,
    MassType
)


class GLDMDStrategy(BaseStrategy):
    """
    Estrategia baseada no Gravitational Lensing & Dark Matter Detector (GL-DMD)

    Usa Relatividade Geral e Weak Lensing para detectar liquidez oculta
    (materia escura) no mercado.

    Conceitos-chave:
    - Potencial Gravitacional (Phi): Campo de forca criado por ordens massivas
    - Cisalhamento Cosmico (gamma): Distorcao estatistica causada por massa
    - Convergencia (kappa): Densidade de massa reconstruida via Kaiser-Squires
    - Materia Escura: Liquidez oculta (iceberg orders) detectada indiretamente
    - Anel de Einstein: Alinhamento perfeito - sinal de alta confianca

    Sinais:
    - EINSTEIN_RING: Trade no centro de massa com precisao matematica
    - ATTRACTIVE: Buy Wall detectado - suporte forte
    - REPULSIVE: Sell Wall detectado - resistencia forte
    - WAIT: Cisalhamento alto mas sem clareza
    """

    def __init__(self,
                 min_prices: int = 50,
                 stop_loss_pips: float = 15.0,
                 take_profit_pips: float = 30.0,
                 grid_size: int = 64,
                 G: float = 1.0,
                 smoothing_scale: float = 2.0,
                 detection_threshold: float = 2.0,
                 ring_threshold: float = 0.7):
        """
        Inicializa a estrategia GL-DMD

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            grid_size: Tamanho do grid para espaco de fase
            G: Constante gravitacional
            smoothing_scale: Escala de suavizacao para kappa
            detection_threshold: Threshold para deteccao de materia escura
            ring_threshold: Threshold para deteccao de anel de Einstein
        """
        super().__init__(name="GLDMD-Lensing")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de precos e volumes
        self.prices = deque(maxlen=500)
        self.volumes = deque(maxlen=500)

        # Indicador GL-DMD
        self.gldmd = GravitationalLensingDarkMatterDetector(
            grid_size=grid_size,
            G=G,
            smoothing_scale=smoothing_scale,
            detection_threshold=detection_threshold,
            ring_threshold=ring_threshold,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em GL-DMD

        Args:
            price: Preco atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (volume opcional)

        Returns:
            Signal se lente gravitacional detectada, None caso contrario
        """
        # Adiciona preco ao buffer
        self.prices.append(price)

        # Extrai volume se disponivel
        volume = indicators.get('volume', None)
        if volume is not None:
            self.volumes.append(volume)

        # Verifica se temos dados suficientes
        if len(self.prices) < self.min_prices:
            return None

        # Cooldown para evitar sinais em sequencia
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        # Converte para numpy arrays
        prices_array = np.array(self.prices)
        volumes_array = np.array(self.volumes) if len(self.volumes) >= len(self.prices) else None

        try:
            # Executa analise GL-DMD
            result = self.gldmd.analyze(prices_array, volumes_array)
            self.last_analysis = result

            # Verifica sinal
            if result['signal'] != 0 and result['confidence'] >= 0.3:
                # Determina direcao
                if result['signal'] == 1:
                    direction = SignalType.BUY
                else:
                    direction = SignalType.SELL

                # Calcula stop/take baseado na posicao da massa
                pip_value = 0.0001
                if direction == SignalType.BUY:
                    stop_loss = price - (self.stop_loss_pips * pip_value)
                    take_profit = price + (self.take_profit_pips * pip_value)
                else:
                    stop_loss = price + (self.stop_loss_pips * pip_value)
                    take_profit = price - (self.take_profit_pips * pip_value)

                confidence = result['confidence']

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
                self.signal_cooldown = 30  # Cooldown para GL-DMD

                return signal

        except Exception as e:
            print(f"Erro na analise GL-DMD: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"GLDMD Lensing | "
                f"Lens={result['lens_type']} | "
                f"Mass={result['mass_type']} | "
                f"kappa={result['max_convergence']:.3f} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.prices.clear()
        self.volumes.clear()
        self.gldmd.reset()
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
            'lens_type': self.last_analysis['lens_type'],
            'mass_type': self.last_analysis['mass_type'],
            'total_mass': self.last_analysis['total_mass'],
            'mass_center_price': self.last_analysis['mass_center_price'],
            'mean_shear': self.last_analysis['mean_shear'],
            'max_shear': self.last_analysis['max_shear'],
            'shear_direction': self.last_analysis['shear_direction'],
            'max_convergence': self.last_analysis['max_convergence'],
            'dark_matter_fraction': self.last_analysis['dark_matter_fraction'],
            'einstein_radius': self.last_analysis['einstein_radius'],
            'is_ring_detected': self.last_analysis['is_ring_detected'],
            'baseline_ellipticity': self.last_analysis['baseline_ellipticity'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_potential_info(self) -> Optional[dict]:
        """Retorna informacoes do potencial gravitacional"""
        if self.last_analysis is None:
            return None
        return {
            'total_mass': self.last_analysis['total_mass'],
            'mass_center_x': self.last_analysis['mass_center_x'],
            'mass_center_y': self.last_analysis['mass_center_y'],
            'mass_center_price': self.last_analysis['mass_center_price'],
            'max_deflection': self.last_analysis['max_deflection']
        }

    def get_shear_info(self) -> Optional[dict]:
        """Retorna informacoes do cisalhamento cosmico"""
        if self.last_analysis is None:
            return None
        return {
            'mean_shear': self.last_analysis['mean_shear'],
            'max_shear': self.last_analysis['max_shear'],
            'shear_direction': self.last_analysis['shear_direction'],
            'baseline_ellipticity': self.last_analysis['baseline_ellipticity']
        }

    def get_convergence_info(self) -> Optional[dict]:
        """Retorna informacoes de convergencia (kappa)"""
        if self.last_analysis is None:
            return None
        return {
            'max_convergence': self.last_analysis['max_convergence'],
            'convergence_position': self.last_analysis['convergence_position'],
            'n_peaks': self.last_analysis['n_peaks']
        }

    def get_dark_matter_info(self) -> Optional[dict]:
        """Retorna informacoes de materia escura"""
        if self.last_analysis is None:
            return None
        return {
            'dark_matter_fraction': self.last_analysis['dark_matter_fraction'],
            'dark_matter_mass': self.last_analysis['dark_matter_mass'],
            'dark_matter_position': self.last_analysis['dark_matter_position'],
            'n_dark_detections': self.last_analysis['n_dark_detections']
        }

    def get_einstein_ring_info(self) -> Optional[dict]:
        """Retorna informacoes do anel de Einstein"""
        if self.last_analysis is None:
            return None
        return {
            'is_ring_detected': self.last_analysis['is_ring_detected'],
            'einstein_radius': self.last_analysis['einstein_radius']
        }

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [max_convergence, dark_matter_fraction, mean_shear, total_mass]
        """
        if self.last_analysis is None:
            return None

        return [
            self.last_analysis['max_convergence'],
            self.last_analysis['dark_matter_fraction'],
            self.last_analysis['mean_shear'],
            self.last_analysis['total_mass']
        ]

    def is_attractive(self) -> bool:
        """
        Verifica se a lente e atrativa (Buy Wall)

        Returns:
            True se lente atrativa detectada
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['lens_type'] == 'ATTRACTIVE'

    def is_repulsive(self) -> bool:
        """
        Verifica se a lente e repulsiva (Sell Wall)

        Returns:
            True se lente repulsiva detectada
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['lens_type'] == 'REPULSIVE'

    def is_einstein_ring(self) -> bool:
        """
        Verifica se ha anel de Einstein

        Returns:
            True se anel de Einstein detectado
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['is_ring_detected']

    def is_dark_matter_detected(self) -> bool:
        """
        Verifica se ha materia escura detectada

        Returns:
            True se materia escura significativa
        """
        if self.last_analysis is None:
            return False
        return bool(self.last_analysis['dark_matter_fraction'] > 0.3)

    def is_waiting(self) -> bool:
        """
        Verifica se o indicador esta em modo espera

        Returns:
            True se aguardando melhor oportunidade
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['signal_name'] in ['WAIT', 'NEUTRAL', 'INSUFFICIENT_DATA']

    def is_high_shear(self) -> bool:
        """
        Verifica se ha cisalhamento alto

        Returns:
            True se shear medio > 0.05
        """
        if self.last_analysis is None:
            return False
        return bool(self.last_analysis['mean_shear'] > 0.05)

    def is_no_lens(self) -> bool:
        """
        Verifica se nao ha lente significativa

        Returns:
            True se sem lente detectada
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['lens_type'] == 'NO_LENS'

    def get_lens_type(self) -> Optional[str]:
        """
        Retorna o tipo de lente detectada

        Returns:
            Tipo de lente: NO_LENS, ATTRACTIVE, REPULSIVE, EINSTEIN_RING
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['lens_type']

    def get_mass_type(self) -> Optional[str]:
        """
        Retorna o tipo de massa detectada

        Returns:
            Tipo de massa: VISIBLE, DARK, MIXED
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['mass_type']

    def get_max_convergence(self) -> Optional[float]:
        """
        Retorna a convergencia maxima kappa

        Returns:
            Valor maximo de kappa
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['max_convergence']

    def get_dark_matter_fraction(self) -> Optional[float]:
        """
        Retorna a fracao de materia escura

        Returns:
            Fracao de massa que e escura
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['dark_matter_fraction']

    def get_mean_shear(self) -> Optional[float]:
        """
        Retorna o cisalhamento medio

        Returns:
            Shear medio gamma
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['mean_shear']

    def get_einstein_radius(self) -> Optional[float]:
        """
        Retorna o raio de Einstein

        Returns:
            Raio do anel de Einstein
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['einstein_radius']

    def get_total_mass(self) -> Optional[float]:
        """
        Retorna a massa total detectada

        Returns:
            Massa total no campo
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['total_mass']

    def get_mass_center_price(self) -> Optional[float]:
        """
        Retorna o preco no centro de massa

        Returns:
            Preco correspondente ao centro de massa
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['mass_center_price']

    def is_buy_wall_detected(self) -> bool:
        """
        Verifica se ha Buy Wall (massa atrativa abaixo do preco)

        Returns:
            True se Buy Wall detectado
        """
        if self.last_analysis is None:
            return False
        return (self.last_analysis['lens_type'] == 'ATTRACTIVE' and
                self.last_analysis['mass_type'] in ['DARK', 'MIXED'])

    def is_sell_wall_detected(self) -> bool:
        """
        Verifica se ha Sell Wall (massa repulsiva acima do preco)

        Returns:
            True se Sell Wall detectado
        """
        if self.last_analysis is None:
            return False
        return (self.last_analysis['lens_type'] == 'REPULSIVE' and
                self.last_analysis['mass_type'] in ['DARK', 'MIXED'])

    def is_high_confidence_setup(self) -> bool:
        """
        Verifica se e um setup de alta confianca

        Returns:
            True se anel de Einstein ou lente forte com DM
        """
        if self.last_analysis is None:
            return False

        return bool(self.last_analysis['is_ring_detected'] or
                    (self.last_analysis['max_convergence'] > 0.2 and
                     self.last_analysis['dark_matter_fraction'] > 0.5))

    def get_lensing_stats(self) -> Optional[dict]:
        """
        Retorna estatisticas completas de lensing

        Returns:
            Dict com todas as metricas de lensing
        """
        if self.last_analysis is None:
            return None

        return {
            'lens_type': self.last_analysis['lens_type'],
            'mass_type': self.last_analysis['mass_type'],
            'max_convergence': self.last_analysis['max_convergence'],
            'mean_shear': self.last_analysis['mean_shear'],
            'max_shear': self.last_analysis['max_shear'],
            'dark_matter_fraction': self.last_analysis['dark_matter_fraction'],
            'einstein_radius': self.last_analysis['einstein_radius'],
            'total_mass': self.last_analysis['total_mass'],
            'n_peaks': self.last_analysis['n_peaks'],
            'n_dark_detections': self.last_analysis['n_dark_detections']
        }
