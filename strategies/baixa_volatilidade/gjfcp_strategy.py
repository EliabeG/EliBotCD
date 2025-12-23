"""
Adaptador de Estrategia para o Granular Jamming Force Chain Percolator
Integra o indicador GJ-FCP com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .gjfcp_granular_jamming import (
    GranularJammingForceChainPercolator,
    JammingState,
    FailureMode
)


class GJFCPStrategy(BaseStrategy):
    """
    Estrategia baseada no Granular Jamming & Force Chain Percolator (GJ-FCP)

    Usa fisica da materia granular para detectar transicoes de volatilidade
    em baixa vol, tratando o Order Book como um empacotamento de graos.

    Conceitos-chave:
    - Tesselacao de Voronoi: Espaco de volume livre das ordens
    - Cadeias de Forca: Estruturas que seguram o preco
    - Tensor de Tensao (Weber): Tensao interna do sistema
    - Criterio de Mohr-Coulomb: Quando a estrutura vai falhar
    - Transicao de Jamming: Solido -> Liquido

    Sinais:
    - Mohr-Coulomb: Criterio de falha atingido (shear failure)
    - Buckling: Cadeias de forca flambando
    - Unjamming: Modulo G caindo rapidamente
    - Z Critico: Numero de coordenacao proximo do ponto isostatico
    """

    def __init__(self,
                 min_prices: int = 50,
                 stop_loss_pips: float = 15.0,
                 take_profit_pips: float = 30.0,
                 particles_per_bar: int = 5,
                 compactness_critical: float = 0.64,
                 z_critical: float = 4.0,
                 g_drop_threshold: float = 0.3):
        """
        Inicializa a estrategia GJ-FCP

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            particles_per_bar: Particulas geradas por barra
            compactness_critical: phi_c - densidade critica de jamming
            z_critical: Z_c - numero de coordenacao critico
            g_drop_threshold: Limiar para queda do modulo G
        """
        super().__init__(name="GJFCP-Granular")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de precos e volumes
        self.prices = deque(maxlen=400)
        self.volumes = deque(maxlen=400)

        # Indicador GJ-FCP
        self.gjfcp = GranularJammingForceChainPercolator(
            particles_per_bar=particles_per_bar,
            compactness_critical=compactness_critical,
            z_critical=z_critical,
            g_drop_threshold=g_drop_threshold,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em GJ-FCP

        Args:
            price: Preco atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (volume opcional)

        Returns:
            Signal se transicao de jamming detectada, None caso contrario
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
            # Executa analise GJ-FCP
            result = self.gjfcp.analyze(prices_array, volumes_array)
            self.last_analysis = result

            # Verifica sinal (ignora WAIT, NEUTRAL e INSUFFICIENT_DATA)
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
                self.signal_cooldown = 20  # Cooldown para GJ-FCP

                return signal

        except Exception as e:
            print(f"Erro na analise GJ-FCP: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"GJFCP Granular | "
                f"State={result['jamming_state']} | "
                f"Mode={result['failure_mode']} | "
                f"G={result['shear_modulus']:.4f} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.prices.clear()
        self.volumes.clear()
        self.gjfcp.reset()
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
            'jamming_state': self.last_analysis['jamming_state'],
            'failure_mode': self.last_analysis['failure_mode'],
            'shear_stress': self.last_analysis['shear_stress'],
            'normal_stress': self.last_analysis['normal_stress'],
            'deviatoric_stress': self.last_analysis['deviatoric_stress'],
            'shear_modulus': self.last_analysis['shear_modulus'],
            'shear_modulus_rate': self.last_analysis['shear_modulus_rate'],
            'coordination_number': self.last_analysis['coordination_number'],
            'coordination_critical': self.last_analysis['coordination_critical'],
            'compactness': self.last_analysis['compactness'],
            'compactness_critical': self.last_analysis['compactness_critical'],
            'yield_criterion': self.last_analysis['yield_criterion'],
            'friction_angle': self.last_analysis['friction_angle'],
            'price_direction': self.last_analysis['price_direction'],
            'n_particles': self.last_analysis['n_particles'],
            'max_chain_length': self.last_analysis['max_chain_length'],
            'is_anisotropic': self.last_analysis['is_anisotropic'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_stress_info(self) -> Optional[dict]:
        """Retorna informacoes de tensao"""
        if self.last_analysis is None:
            return None
        return {
            'shear': self.last_analysis['shear_stress'],
            'normal': self.last_analysis['normal_stress'],
            'deviatoric': self.last_analysis['deviatoric_stress'],
            'yield_criterion': self.last_analysis['yield_criterion']
        }

    def get_mechanical_info(self) -> Optional[dict]:
        """Retorna informacoes mecanicas"""
        if self.last_analysis is None:
            return None
        return {
            'shear_modulus': self.last_analysis['shear_modulus'],
            'shear_modulus_rate': self.last_analysis['shear_modulus_rate'],
            'coordination_number': self.last_analysis['coordination_number'],
            'coordination_critical': self.last_analysis['coordination_critical'],
            'friction_angle': self.last_analysis['friction_angle']
        }

    def get_compactness_info(self) -> Optional[dict]:
        """Retorna informacoes de compacidade"""
        if self.last_analysis is None:
            return None
        return {
            'compactness': self.last_analysis['compactness'],
            'compactness_critical': self.last_analysis['compactness_critical'],
            'n_particles': self.last_analysis['n_particles']
        }

    def get_network_info(self) -> Optional[dict]:
        """Retorna informacoes da rede de forca"""
        if self.last_analysis is None:
            return None
        return {
            'is_anisotropic': self.last_analysis['is_anisotropic'],
            'max_chain_length': self.last_analysis['max_chain_length'],
            'failure_direction': self.last_analysis['failure_direction']
        }

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [shear_modulus, coordination_number, compactness, yield_criterion]
        """
        if self.last_analysis is None:
            return None

        return [
            self.last_analysis['shear_modulus'],
            self.last_analysis['coordination_number'],
            self.last_analysis['compactness'],
            self.last_analysis['yield_criterion']
        ]

    def is_jammed(self) -> bool:
        """
        Verifica se o sistema esta no estado Jammed

        Returns:
            True se mercado esta compactado (baixa vol)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['jamming_state'] == 'JAMMED'

    def is_unjamming(self) -> bool:
        """
        Verifica se o sistema esta em transicao (Unjamming)

        Returns:
            True se mercado esta comecando a ceder
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['jamming_state'] == 'UNJAMMING'

    def is_unjammed(self) -> bool:
        """
        Verifica se o sistema esta no estado Unjammed

        Returns:
            True se mercado esta fluindo (alta vol)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['jamming_state'] == 'UNJAMMED'

    def is_critical(self) -> bool:
        """
        Verifica se o sistema esta no ponto critico

        Returns:
            True se no ponto critico de transicao
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['jamming_state'] == 'CRITICAL'

    def is_stable(self) -> bool:
        """
        Verifica se a estrutura esta estavel

        Returns:
            True se modo de falha e STABLE
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['failure_mode'] == 'STABLE'

    def is_shear_failure(self) -> bool:
        """
        Verifica se ha falha por cisalhamento

        Returns:
            True se modo de falha e SHEAR_FAILURE
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['failure_mode'] == 'SHEAR_FAILURE'

    def is_buckling(self) -> bool:
        """
        Verifica se ha flambagem das cadeias

        Returns:
            True se modo de falha e BUCKLING
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['failure_mode'] == 'BUCKLING'

    def is_avalanche(self) -> bool:
        """
        Verifica se ha avalanche (SOC)

        Returns:
            True se modo de falha e AVALANCHE
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['failure_mode'] == 'AVALANCHE'

    def is_liquefaction(self) -> bool:
        """
        Verifica se ha liquefacao do book

        Returns:
            True se modo de falha e LIQUEFACTION
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['failure_mode'] == 'LIQUEFACTION'

    def is_waiting(self) -> bool:
        """
        Verifica se o indicador esta em modo espera

        Returns:
            True se aguardando melhor oportunidade
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['signal_name'] in ['WAIT', 'NEUTRAL', 'INSUFFICIENT_DATA']

    def is_anisotropic(self) -> bool:
        """
        Verifica se a rede de forca e anisotropica

        Returns:
            True se ha direcao preferencial de forca
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['is_anisotropic']

    def get_shear_modulus(self) -> Optional[float]:
        """
        Retorna o modulo de cisalhamento G

        Returns:
            G - rigidez do sistema
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['shear_modulus']

    def get_coordination_number(self) -> Optional[float]:
        """
        Retorna o numero de coordenacao Z

        Returns:
            Z - numero medio de vizinhos
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['coordination_number']

    def get_compactness(self) -> Optional[float]:
        """
        Retorna a compacidade phi

        Returns:
            phi - densidade de empacotamento
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['compactness']

    def get_yield_criterion(self) -> Optional[float]:
        """
        Retorna o criterio de yield (Mohr-Coulomb)

        Returns:
            Distancia para falha (>0 = falhando)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['yield_criterion']

    def get_price_direction(self) -> Optional[str]:
        """
        Retorna a direcao prevista do preco

        Returns:
            'UP' ou 'DOWN'
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['price_direction']

    def get_g_history(self) -> np.ndarray:
        """Retorna historico do modulo de cisalhamento"""
        return self.gjfcp.get_g_history()

    def get_z_history(self) -> np.ndarray:
        """Retorna historico do numero de coordenacao"""
        return self.gjfcp.get_z_history()

    def get_phi_history(self) -> np.ndarray:
        """Retorna historico da compacidade"""
        return self.gjfcp.get_phi_history()

    def is_near_z_critical(self) -> bool:
        """
        Verifica se Z esta proximo do valor critico

        Returns:
            True se Z < Z_c + 0.5
        """
        if self.last_analysis is None:
            return False

        Z = self.last_analysis['coordination_number']
        Z_c = self.last_analysis['coordination_critical']

        return Z < Z_c + 0.5

    def is_g_dropping(self) -> bool:
        """
        Verifica se G esta caindo rapidamente

        Returns:
            True se dG/dt < -0.2
        """
        if self.last_analysis is None:
            return False

        return self.last_analysis['shear_modulus_rate'] < -0.2

    def is_breakout_setup(self) -> bool:
        """
        Verifica se ha setup de breakout iminente

        Returns:
            True se transicao de jamming + anisotropia
        """
        if self.last_analysis is None:
            return False

        state = self.last_analysis['jamming_state']
        anisotropic = self.last_analysis['is_anisotropic']

        return state in ['UNJAMMING', 'CRITICAL'] and anisotropic
