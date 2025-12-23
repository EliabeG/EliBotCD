"""
Adaptador de Estrategia para o Holographic AdS/CFT Bulk-Boundary Projector
Integra o indicador HBBP com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List, Dict
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .hbbp_holographic_projector import (
    HolographicAdSCFTBulkBoundaryProjector,
    HBBPSignalType,
    GeometryState,
    EntanglementPhase
)


class HBBPStrategy(BaseStrategy):
    """
    Estrategia baseada no Holographic AdS/CFT Bulk-Boundary Projector (HBBP)

    Usa correspondencia holografica e gravidade entropica para detectar
    "buracos negros" (atratores de preco massivos) antes que sejam visiveis.

    Conceitos-chave:
    - Cadeia de Spin quantica: Order book como estado |psi>
    - MERA: Renormalizacao multi-escala para separar ruido de estrutura
    - Ryu-Takayanagi: S_A = Area(gamma_A) / 4G_N
    - Forca Entropica: F_e = T nabla S (gravidade de Verlinde)
    - Geometria AdS: Buraco negro = atrator gravitacional no bulk

    Sinais:
    - BLACK_HOLE_NUCLEATION: Sinal forte! Atrator formado
    - ENTROPY_SPIKE: Entropia disparou, movimento iminente
    - HORIZON_FORMATION: Horizonte se formando, preparar
    - VACUUM: Espaco plano, preco e ruido termico
    """

    def __init__(self,
                 min_prices: int = 50,
                 stop_loss_pips: float = 15.0,
                 take_profit_pips: float = 30.0,
                 n_sites: int = 64,
                 bond_dimension: int = 16,
                 n_mera_layers: int = 4,
                 entropy_threshold: float = 1.2,
                 force_threshold: float = 0.3):
        """
        Inicializa a estrategia HBBP

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            n_sites: Numero de sites na cadeia quantica
            bond_dimension: Dimensao do bond para MPS/MERA
            n_mera_layers: Numero de camadas MERA
            entropy_threshold: Limiar para entropia
            force_threshold: Limiar para forca entropica
        """
        super().__init__(name="HBBP-Holographic")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffers
        self.price_buffer = deque(maxlen=500)
        self.volume_buffer = deque(maxlen=500)

        # Indicador HBBP
        self.hbbp = HolographicAdSCFTBulkBoundaryProjector(
            n_sites=n_sites,
            bond_dimension=bond_dimension,
            n_mera_layers=n_mera_layers,
            entropy_threshold=entropy_threshold,
            force_threshold=force_threshold,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0
        self.current_price = 0.0

    def add_tick(self, price: float, volume: float = None):
        """
        Adiciona um tick ao buffer

        Args:
            price: Preco do tick
            volume: Volume do tick (opcional)
        """
        self.price_buffer.append(price)
        if volume is not None:
            self.volume_buffer.append(volume)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em HBBP

        Args:
            price: Preco atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais

        Returns:
            Signal se buraco negro detectado, None caso contrario
        """
        self.current_price = price

        # Adiciona ao buffer
        self.add_tick(price, indicators.get('volume'))

        # Verifica se temos dados suficientes
        if len(self.price_buffer) < self.min_prices:
            return None

        # Cooldown para evitar sinais em sequencia
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        # Converte buffers para arrays
        prices = np.array(self.price_buffer)
        volumes = np.array(self.volume_buffer) if len(self.volume_buffer) > 0 else None

        try:
            # Executa analise HBBP
            result = self.hbbp.analyze(prices, volumes)
            self.last_analysis = result

            # Verifica sinal
            if result['signal'] != 0 and result['confidence'] >= 0.3:
                # Determina direcao
                if result['signal'] == 1:
                    direction = SignalType.BUY
                else:
                    direction = SignalType.SELL

                # Calcula stop/take
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
                self.signal_cooldown = 50  # Cooldown maior para HBBP

                return signal

        except Exception as e:
            print(f"Erro na analise HBBP: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        return (f"HBBP Holographic | "
                f"Geometry={result['geometry_state']} | "
                f"Anomaly={result['anomaly_type']} | "
                f"S_A={result['S_A']:.3f} | "
                f"F_e={result['entropic_force']:.3f} | "
                f"{result['reason'][:50]}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.price_buffer.clear()
        self.volume_buffer.clear()
        self.hbbp.reset()
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
            'geometry_state': self.last_analysis['geometry_state'],
            'entanglement_phase': self.last_analysis['entanglement_phase'],
            'anomaly_type': self.last_analysis['anomaly_type'],
            'S_A': self.last_analysis['S_A'],
            'is_black_hole': self.last_analysis['is_black_hole'],
            'entropic_force': self.last_analysis['entropic_force'],
            'force_direction': self.last_analysis['force_direction'],
            'reason': self.last_analysis['reason']
        }

    def get_entropy_info(self) -> Optional[dict]:
        """Retorna informacoes de entropia"""
        if self.last_analysis is None:
            return None
        return {
            'S_A': self.last_analysis['S_A'],
            'S_renyi': self.last_analysis['S_renyi'],
            'mutual_information': self.last_analysis['mutual_information'],
            'ryu_takayanagi_area': self.last_analysis['ryu_takayanagi_area']
        }

    def get_geometry_info(self) -> Optional[dict]:
        """Retorna informacoes de geometria"""
        if self.last_analysis is None:
            return None
        return {
            'geometry_state': self.last_analysis['geometry_state'],
            'ricci_scalar': self.last_analysis['ricci_scalar'],
            'geodesic_length': self.last_analysis['geodesic_length'],
            'horizon_radius': self.last_analysis['horizon_radius'],
            'is_black_hole': self.last_analysis['is_black_hole']
        }

    def get_force_info(self) -> Optional[dict]:
        """Retorna informacoes de forca entropica"""
        if self.last_analysis is None:
            return None
        return {
            'entropic_force': self.last_analysis['entropic_force'],
            'force_direction': self.last_analysis['force_direction'],
            'gradient_entropy': self.last_analysis['gradient_entropy'],
            'temperature': self.last_analysis['temperature'],
            'inertia': self.last_analysis['inertia'],
            'net_force': self.last_analysis['net_force']
        }

    def get_mera_info(self) -> Optional[dict]:
        """Retorna informacoes MERA"""
        if self.last_analysis is None:
            return None
        return {
            'n_mera_layers': self.last_analysis['n_mera_layers'],
            'long_range_entanglement': self.last_analysis['long_range_entanglement'],
            'n_sites': self.last_analysis['n_sites']
        }

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [S_A, ricci_scalar, entropic_force, long_range_entanglement]
        """
        if self.last_analysis is None:
            return None

        return [
            self.last_analysis['S_A'],
            self.last_analysis['ricci_scalar'],
            self.last_analysis['entropic_force'],
            self.last_analysis['long_range_entanglement']
        ]

    def is_vacuum(self) -> bool:
        """
        Verifica se o bulk esta em vacuo (AdS puro)

        Returns:
            True se geometria plana
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['geometry_state'] == 'FLAT'

    def is_curved(self) -> bool:
        """
        Verifica se a geometria esta curvada

        Returns:
            True se curvatura significativa
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['geometry_state'] == 'CURVED'

    def is_black_hole(self) -> bool:
        """
        Verifica se um buraco negro se formou

        Returns:
            True se buraco negro detectado
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['is_black_hole']

    def is_singularity(self) -> bool:
        """
        Verifica se ha singularidade

        Returns:
            True se singularidade (crash iminente)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['geometry_state'] == 'SINGULARITY'

    def is_area_law(self) -> bool:
        """
        Verifica se emaranhamento segue lei de area

        Returns:
            True se baixo emaranhamento (ground state)
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['entanglement_phase'] == 'AREA_LAW'

    def is_volume_law(self) -> bool:
        """
        Verifica se emaranhamento segue lei de volume

        Returns:
            True se alto emaranhamento (termico)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['entanglement_phase'] == 'VOLUME_LAW'

    def is_critical(self) -> bool:
        """
        Verifica se esta em ponto critico

        Returns:
            True se no ponto critico
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['entanglement_phase'] == 'CRITICAL'

    def is_anomaly_detected(self) -> bool:
        """
        Verifica se anomalia holografica foi detectada

        Returns:
            True se anomalia detectada
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['anomaly_detected']

    def is_black_hole_nucleation(self) -> bool:
        """
        Verifica se houve nucleacao de buraco negro

        Returns:
            True se nucleacao detectada (sinal forte!)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['anomaly_type'] == 'BLACK_HOLE_NUCLEATION'

    def is_entropy_spike(self) -> bool:
        """
        Verifica se houve spike de entropia

        Returns:
            True se spike detectado
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['anomaly_type'] == 'ENTROPY_SPIKE'

    def is_horizon_forming(self) -> bool:
        """
        Verifica se horizonte esta se formando

        Returns:
            True se horizonte se formando
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['anomaly_type'] == 'HORIZON_FORMATION'

    def is_entropy_rising(self) -> bool:
        """
        Verifica se entropia esta subindo

        Returns:
            True se entropia em alta
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['anomaly_type'] == 'ENTROPY_RISING'

    def get_geometry_state(self) -> Optional[str]:
        """
        Retorna o estado da geometria

        Returns:
            Estado: FLAT, CURVED, BLACK_HOLE, SINGULARITY
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['geometry_state']

    def get_entanglement_phase(self) -> Optional[str]:
        """
        Retorna a fase do emaranhamento

        Returns:
            Fase: AREA_LAW, VOLUME_LAW, CRITICAL, THERMAL
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['entanglement_phase']

    def get_anomaly_type(self) -> Optional[str]:
        """
        Retorna o tipo de anomalia

        Returns:
            Tipo: VACUUM, ENTROPY_RISING, HORIZON_FORMATION, ENTROPY_SPIKE, BLACK_HOLE_NUCLEATION
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['anomaly_type']

    def get_von_neumann_entropy(self) -> Optional[float]:
        """
        Retorna a entropia de von Neumann S_A

        Returns:
            Entropia de emaranhamento
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['S_A']

    def get_ryu_takayanagi_area(self) -> Optional[float]:
        """
        Retorna a area de Ryu-Takayanagi

        Returns:
            Area da superficie minima no bulk
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['ryu_takayanagi_area']

    def get_ricci_scalar(self) -> Optional[float]:
        """
        Retorna o escalar de Ricci

        Returns:
            Curvatura do bulk
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['ricci_scalar']

    def get_horizon_radius(self) -> Optional[float]:
        """
        Retorna o raio do horizonte

        Returns:
            Raio do horizonte do buraco negro (0 se nao houver)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['horizon_radius']

    def get_entropic_force(self) -> Optional[float]:
        """
        Retorna a forca entropica |F_e|

        Returns:
            Magnitude da forca entropica
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['entropic_force']

    def get_force_direction(self) -> Optional[str]:
        """
        Retorna a direcao da forca entropica

        Returns:
            UP, DOWN ou NEUTRAL
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['force_direction']

    def get_temperature(self) -> Optional[float]:
        """
        Retorna a temperatura efetiva

        Returns:
            Temperatura (analogica a Hawking)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['temperature']

    def get_long_range_entanglement(self) -> Optional[float]:
        """
        Retorna o emaranhamento de longo alcance

        Returns:
            Emaranhamento preservado apos MERA
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['long_range_entanglement']

    def get_mutual_information(self) -> Optional[float]:
        """
        Retorna a informacao mutua I(A:B)

        Returns:
            Correlacoes totais entre subsistemas
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['mutual_information']

    def get_severity(self) -> Optional[float]:
        """
        Retorna a severidade da anomalia

        Returns:
            Severidade de 0 a 1
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['severity']

    def is_high_entropy_state(self) -> bool:
        """
        Verifica se esta em estado de alta entropia

        Returns:
            True se S_A > 0.5
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['S_A'] > 0.5

    def is_force_significant(self) -> bool:
        """
        Verifica se forca entropica e significativa

        Returns:
            True se F_e > 0.5
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['entropic_force'] > 0.5

    def is_attractor_forming(self) -> bool:
        """
        Verifica se um atrator gravitacional esta se formando

        Returns:
            True se buraco negro ou entropia alta + forca significativa
        """
        if self.last_analysis is None:
            return False

        return bool(self.last_analysis['is_black_hole'] or
                   (self.last_analysis['S_A'] > 0.5 and
                    self.last_analysis['entropic_force'] > 0.3))

    def is_safe_vacuum(self) -> bool:
        """
        Verifica se e vacuo seguro (AdS puro, sem atrator)

        Returns:
            True se geometria plana + baixa entropia
        """
        if self.last_analysis is None:
            return True

        return bool(self.last_analysis['geometry_state'] == 'FLAT' and
                   self.last_analysis['S_A'] < 0.3 and
                   not self.last_analysis['is_black_hole'])

    def get_hbbp_stats(self) -> Optional[dict]:
        """
        Retorna estatisticas completas do HBBP

        Returns:
            Dict com todas as metricas
        """
        if self.last_analysis is None:
            return None

        return {
            'geometry_state': self.last_analysis['geometry_state'],
            'entanglement_phase': self.last_analysis['entanglement_phase'],
            'anomaly_type': self.last_analysis['anomaly_type'],
            'S_A': self.last_analysis['S_A'],
            'ryu_takayanagi_area': self.last_analysis['ryu_takayanagi_area'],
            'ricci_scalar': self.last_analysis['ricci_scalar'],
            'horizon_radius': self.last_analysis['horizon_radius'],
            'is_black_hole': self.last_analysis['is_black_hole'],
            'entropic_force': self.last_analysis['entropic_force'],
            'force_direction': self.last_analysis['force_direction'],
            'temperature': self.last_analysis['temperature'],
            'long_range_entanglement': self.last_analysis['long_range_entanglement'],
            'mutual_information': self.last_analysis['mutual_information'],
            'severity': self.last_analysis['severity']
        }
