"""
Adaptador de Estrategia para o Invasive Percolation Capillary Breakthrough Monitor
Integra o indicador IP-CBM com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .ipcbm_invasive_percolation import (
    InvasivePercolationCapillaryBreakthroughMonitor,
    FluidState,
    ClusterShape,
    FRACTAL_DIMENSION_2D
)


class IPCBMStrategy(BaseStrategy):
    """
    Estrategia baseada no Invasive Percolation & Capillary Breakthrough Monitor (IP-CBM)

    Usa Fisica de Escoamento em Meios Porosos para detectar onde e quando a
    "barragem" vai estourar em baixa volatilidade.

    Conceitos-chave:
    - Permeabilidade (k): Inversamente proporcional ao volume de ordens
    - Pressao Capilar (Pc): Resistencia a invasao do fluido
    - Percolacao Invasiva: Avanca pelo poro de menor resistencia
    - Dimensao Fractal (Df): Geometria do cluster de invasao
    - Viscous Fingering: Instabilidade de Saffman-Taylor

    Sinais:
    - BREAKTHROUGH: Finger atingiu L_crit - entrada forte
    - FINGERING iminente: Preparar para entrada
    - INSTABILIDADE: Observar formacao de finger
    - TRAPPED: Aguardar acumulo de pressao
    """

    def __init__(self,
                 min_prices: int = 50,
                 stop_loss_pips: float = 15.0,
                 take_profit_pips: float = 30.0,
                 n_price_levels: int = 50,
                 n_time_steps: int = 50,
                 base_surface_tension: float = 1.0,
                 contact_angle: float = 120.0,
                 max_invasion_steps: int = 500):
        """
        Inicializa a estrategia IP-CBM

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            n_price_levels: Numero de niveis de preco no grid
            n_time_steps: Numero de passos temporais
            base_surface_tension: gamma base (aversao ao risco)
            contact_angle: theta em graus (>90 para fluido nao-molhante)
            max_invasion_steps: Maximo de celulas a invadir
        """
        super().__init__(name="IPCBM-Percolation")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de precos e volumes
        self.prices = deque(maxlen=500)
        self.volumes = deque(maxlen=500)

        # Indicador IP-CBM
        self.ipcbm = InvasivePercolationCapillaryBreakthroughMonitor(
            n_price_levels=n_price_levels,
            n_time_steps=n_time_steps,
            base_surface_tension=base_surface_tension,
            contact_angle=contact_angle,
            max_invasion_steps=max_invasion_steps,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em IP-CBM

        Args:
            price: Preco atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (volume opcional)

        Returns:
            Signal se breakthrough detectado, None caso contrario
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
            # Executa analise IP-CBM
            result = self.ipcbm.analyze(prices_array, volumes_array)
            self.last_analysis = result

            # Verifica sinal
            if result['signal'] != 0 and result['confidence'] >= 0.3:
                # Determina direcao
                if result['signal'] == 1:
                    direction = SignalType.BUY
                else:
                    direction = SignalType.SELL

                # Usa stop/take do indicador ou calcula
                if result['stop_loss'] > 0:
                    stop_loss = result['stop_loss']
                    take_profit = result['take_profit']
                else:
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
                self.signal_cooldown = 30  # Cooldown para IP-CBM

                return signal

        except Exception as e:
            print(f"Erro na analise IP-CBM: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"IPCBM Percolation | "
                f"State={result['fluid_state']} | "
                f"Shape={result['cluster_shape']} | "
                f"Df={result['fractal_dimension']:.2f} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.prices.clear()
        self.volumes.clear()
        self.ipcbm.reset()
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
            'fluid_state': self.last_analysis['fluid_state'],
            'cluster_shape': self.last_analysis['cluster_shape'],
            'porosity': self.last_analysis['porosity'],
            'permeability_mean': self.last_analysis['permeability_mean'],
            'surface_tension': self.last_analysis['surface_tension'],
            'cluster_mass': self.last_analysis['cluster_mass'],
            'radius_of_gyration': self.last_analysis['radius_of_gyration'],
            'fractal_dimension': self.last_analysis['fractal_dimension'],
            'aspect_ratio': self.last_analysis['aspect_ratio'],
            'finger_length': self.last_analysis['finger_length'],
            'finger_direction': self.last_analysis['finger_direction'],
            'critical_length': self.last_analysis['critical_length'],
            'finger_progress': self.last_analysis['finger_progress'],
            'mobility_ratio': self.last_analysis['mobility_ratio'],
            'capillary_number': self.last_analysis['capillary_number'],
            'is_unstable': self.last_analysis['is_unstable'],
            'breakthrough_imminent': self.last_analysis['breakthrough_imminent'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_porosity_info(self) -> Optional[dict]:
        """Retorna informacoes de porosidade"""
        if self.last_analysis is None:
            return None
        return {
            'porosity': self.last_analysis['porosity'],
            'permeability_mean': self.last_analysis['permeability_mean'],
            'surface_tension': self.last_analysis['surface_tension']
        }

    def get_cluster_info(self) -> Optional[dict]:
        """Retorna informacoes do cluster de invasao"""
        if self.last_analysis is None:
            return None
        return {
            'mass': self.last_analysis['cluster_mass'],
            'radius_of_gyration': self.last_analysis['radius_of_gyration'],
            'fractal_dimension': self.last_analysis['fractal_dimension'],
            'aspect_ratio': self.last_analysis['aspect_ratio'],
            'shape': self.last_analysis['cluster_shape']
        }

    def get_finger_info(self) -> Optional[dict]:
        """Retorna informacoes do viscous finger"""
        if self.last_analysis is None:
            return None
        return {
            'length': self.last_analysis['finger_length'],
            'direction': self.last_analysis['finger_direction'],
            'critical_length': self.last_analysis['critical_length'],
            'progress': self.last_analysis['finger_progress'],
            'growth_rate': self.last_analysis['finger_growth_rate']
        }

    def get_saffman_taylor_info(self) -> Optional[dict]:
        """Retorna informacoes da analise de Saffman-Taylor"""
        if self.last_analysis is None:
            return None
        return {
            'mobility_ratio': self.last_analysis['mobility_ratio'],
            'capillary_number': self.last_analysis['capillary_number'],
            'is_unstable': self.last_analysis['is_unstable']
        }

    def get_breakthrough_info(self) -> Optional[dict]:
        """Retorna informacoes de breakthrough"""
        if self.last_analysis is None:
            return None
        return {
            'imminent': self.last_analysis['breakthrough_imminent'],
            'path_length': self.last_analysis['path_length'],
            'path_resistance': self.last_analysis['path_resistance']
        }

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [fractal_dimension, finger_progress, mobility_ratio, porosity]
        """
        if self.last_analysis is None:
            return None

        return [
            self.last_analysis['fractal_dimension'],
            self.last_analysis['finger_progress'],
            self.last_analysis['mobility_ratio'],
            self.last_analysis['porosity']
        ]

    def is_trapped(self) -> bool:
        """
        Verifica se o fluido esta travado por capilaridade

        Returns:
            True se fluido preso (baixa vol segura)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['fluid_state'] == 'TRAPPED'

    def is_advancing(self) -> bool:
        """
        Verifica se o fluido esta avancando

        Returns:
            True se fluido avancando lentamente
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['fluid_state'] == 'ADVANCING'

    def is_fingering(self) -> bool:
        """
        Verifica se ha formacao de finger

        Returns:
            True se viscous fingering ativo
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['fluid_state'] == 'FINGERING'

    def is_breakthrough(self) -> bool:
        """
        Verifica se ha breakthrough

        Returns:
            True se breakthrough ocorrendo
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['fluid_state'] == 'BREAKTHROUGH'

    def is_compact(self) -> bool:
        """
        Verifica se o cluster e compacto

        Returns:
            True se cluster esferico
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['cluster_shape'] == 'COMPACT'

    def is_unstable(self) -> bool:
        """
        Verifica se a interface e instavel (Saffman-Taylor)

        Returns:
            True se instabilidade detectada
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['is_unstable']

    def is_breakthrough_imminent(self) -> bool:
        """
        Verifica se breakthrough e iminente

        Returns:
            True se finger proximo de L_crit
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['breakthrough_imminent']

    def is_waiting(self) -> bool:
        """
        Verifica se o indicador esta em modo espera

        Returns:
            True se aguardando melhor oportunidade
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['signal_name'] in ['WAIT', 'NEUTRAL', 'INSUFFICIENT_DATA']

    def is_geometry_changed(self) -> bool:
        """
        Verifica se a geometria do mercado mudou

        Returns:
            True se Df desviou do normal
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['geometry_changed']

    def get_porosity(self) -> Optional[float]:
        """
        Retorna a porosidade phi

        Returns:
            Porosidade media do campo
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['porosity']

    def get_permeability(self) -> Optional[float]:
        """
        Retorna a permeabilidade media k

        Returns:
            Permeabilidade media
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['permeability_mean']

    def get_fractal_dimension(self) -> Optional[float]:
        """
        Retorna a dimensao fractal Df

        Returns:
            Dimensao fractal do cluster
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['fractal_dimension']

    def get_finger_length(self) -> Optional[float]:
        """
        Retorna o comprimento do finger

        Returns:
            Comprimento do maior dedo viscoso
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['finger_length']

    def get_critical_length(self) -> Optional[float]:
        """
        Retorna o comprimento critico L_crit

        Returns:
            Comprimento necessario para breakthrough
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['critical_length']

    def get_finger_progress(self) -> Optional[float]:
        """
        Retorna o progresso do finger (L/L_crit)

        Returns:
            Fracao do comprimento critico atingida
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['finger_progress']

    def get_finger_direction(self) -> Optional[str]:
        """
        Retorna a direcao do finger

        Returns:
            'UP' ou 'DOWN'
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['finger_direction']

    def get_mobility_ratio(self) -> Optional[float]:
        """
        Retorna o mobility ratio M

        Returns:
            Razao de mobilidade (viscosidades)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['mobility_ratio']

    def get_capillary_number(self) -> Optional[float]:
        """
        Retorna o capillary number Ca

        Returns:
            Numero capilar
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['capillary_number']

    def is_breakout_setup(self) -> bool:
        """
        Verifica se ha setup de breakout iminente

        Returns:
            True se fingering + breakthrough iminente
        """
        if self.last_analysis is None:
            return False

        return (self.last_analysis['fluid_state'] in ['FINGERING', 'BREAKTHROUGH'] and
                self.last_analysis['breakthrough_imminent'])

    def is_low_risk_entry(self) -> bool:
        """
        Verifica se e uma entrada de baixo risco

        Returns:
            True se stop na base do dedo e bem definido
        """
        if self.last_analysis is None:
            return False

        return (self.last_analysis['signal'] != 0 and
                self.last_analysis['confidence'] >= 0.5 and
                self.last_analysis['finger_progress'] >= 0.7)

    def get_reference_fractal_dimension(self) -> float:
        """
        Retorna a dimensao fractal de referencia

        Returns:
            Df teorico para percolacao 2D (1.89)
        """
        return FRACTAL_DIMENSION_2D

    def get_invasion_stats(self) -> Optional[dict]:
        """
        Retorna estatisticas da invasao percolativa

        Returns:
            Dict com cluster_mass, Rg, Df, aspect_ratio
        """
        if self.last_analysis is None:
            return None

        return {
            'cluster_mass': self.last_analysis['cluster_mass'],
            'radius_of_gyration': self.last_analysis['radius_of_gyration'],
            'fractal_dimension': self.last_analysis['fractal_dimension'],
            'aspect_ratio': self.last_analysis['aspect_ratio'],
            'finger_length': self.last_analysis['finger_length'],
            'critical_length': self.last_analysis['critical_length']
        }
