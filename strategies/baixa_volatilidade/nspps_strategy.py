"""
Adaptador de Estrategia para o Neuromorphic Spiking Pre-Potentiation Scanner
Integra o indicador NS-PPS com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .nspps_neuromorphic_spiking import (
    NeuromorphicSpikingPrePotentiationScanner,
    CorticalState,
    ExcitabilityState
)


class NSPPSStrategy(BaseStrategy):
    """
    Estrategia baseada no Neuromorphic Spiking Synaptic Pre-Potentiation Scanner (NS-PPS)

    Usa Redes Neurais de Espiking (SNN) de terceira geracao para detectar
    hiperexcitabilidade silenciosa em baixa volatilidade.

    Conceitos-chave:
    - Neuronio LIF: Leaky Integrate-and-Fire - simula voltagem real
    - STDP: Plasticidade Dependente do Tempo de Disparo - aprende caminhos
    - Coeficiente de Avalanche (kappa): Parametro de ramificacao
    - Indice de Kuramoto: Sincronia de fase entre neuronios
    - Estado Pre-Ictal: "Epilepsia financeira" iminente

    Sinais:
    - PRE-ICTAL + STDP UP: LONG (convulsao de alta)
    - PRE-ICTAL + STDP DOWN: SHORT (convulsao de baixa)
    - CRITICO + Hiperexcitavel: Preparar para breakout
    - SUBCRITICO: Aguardar acumulacao
    """

    def __init__(self,
                 min_prices: int = 30,
                 stop_loss_pips: float = 15.0,
                 take_profit_pips: float = 30.0,
                 n_neurons: int = 200,
                 simulation_steps: int = 40,
                 potential_threshold_ratio: float = 0.85,
                 kappa_critical: float = 0.9):
        """
        Inicializa a estrategia NS-PPS

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            n_neurons: Numero de neuronios na rede
            simulation_steps: Passos de simulacao por tick
            potential_threshold_ratio: V/V_th para considerar hiperexcitavel
            kappa_critical: Limiar do coeficiente de avalanche
        """
        super().__init__(name="NSPPS-Neuromorphic")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de precos e volumes
        self.prices = deque(maxlen=400)
        self.volumes = deque(maxlen=400)

        # Indicador NS-PPS
        self.nspps = NeuromorphicSpikingPrePotentiationScanner(
            n_neurons=n_neurons,
            simulation_steps=simulation_steps,
            potential_threshold_ratio=potential_threshold_ratio,
            kappa_critical=kappa_critical,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em NS-PPS

        Args:
            price: Preco atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (volume opcional)

        Returns:
            Signal se hiperexcitabilidade detectada, None caso contrario
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
            # Executa analise NS-PPS
            result = self.nspps.analyze(prices_array, volumes_array)
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
                self.signal_cooldown = 25  # Cooldown para NS-PPS

                return signal

        except Exception as e:
            print(f"Erro na analise NS-PPS: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"NSPPS Neuromorphic | "
                f"State={result['cortical_state']} | "
                f"Excit={result['excitability_state']} | "
                f"kappa={result['avalanche_coefficient']:.3f} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.prices.clear()
        self.volumes.clear()
        self.nspps.reset()
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
            'cortical_state': self.last_analysis['cortical_state'],
            'excitability_state': self.last_analysis['excitability_state'],
            'mean_potential': self.last_analysis['mean_potential'],
            'potential_ratio': self.last_analysis['potential_ratio'],
            'avalanche_coefficient': self.last_analysis['avalanche_coefficient'],
            'is_critical': self.last_analysis['is_critical'],
            'is_pre_ictal': self.last_analysis['is_pre_ictal'],
            'synchrony_index': self.last_analysis['synchrony_index'],
            'is_synchronized': self.last_analysis['is_synchronized'],
            'stdp_up_strength': self.last_analysis['stdp_up_strength'],
            'stdp_down_strength': self.last_analysis['stdp_down_strength'],
            'stdp_asymmetry': self.last_analysis['stdp_asymmetry'],
            'total_spikes': self.last_analysis['total_spikes'],
            'firing_rate': self.last_analysis['firing_rate'],
            'n_neurons': self.last_analysis['n_neurons'],
            'n_active': self.last_analysis['n_active'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_potential_info(self) -> Optional[dict]:
        """Retorna informacoes de potencial de membrana"""
        if self.last_analysis is None:
            return None
        return {
            'mean_potential': self.last_analysis['mean_potential'],
            'potential_ratio': self.last_analysis['potential_ratio'],
            'n_active': self.last_analysis['n_active']
        }

    def get_avalanche_info(self) -> Optional[dict]:
        """Retorna informacoes de avalanche"""
        if self.last_analysis is None:
            return None
        return {
            'avalanche_coefficient': self.last_analysis['avalanche_coefficient'],
            'is_critical': self.last_analysis['is_critical'],
            'is_pre_ictal': self.last_analysis['is_pre_ictal']
        }

    def get_synchrony_info(self) -> Optional[dict]:
        """Retorna informacoes de sincronia"""
        if self.last_analysis is None:
            return None
        return {
            'synchrony_index': self.last_analysis['synchrony_index'],
            'is_synchronized': self.last_analysis['is_synchronized']
        }

    def get_stdp_info(self) -> Optional[dict]:
        """Retorna informacoes de plasticidade STDP"""
        if self.last_analysis is None:
            return None
        return {
            'up_strength': self.last_analysis['stdp_up_strength'],
            'down_strength': self.last_analysis['stdp_down_strength'],
            'asymmetry': self.last_analysis['stdp_asymmetry']
        }

    def get_spike_info(self) -> Optional[dict]:
        """Retorna informacoes de spikes"""
        if self.last_analysis is None:
            return None
        return {
            'total_spikes': self.last_analysis['total_spikes'],
            'firing_rate': self.last_analysis['firing_rate'],
            'n_neurons': self.last_analysis['n_neurons']
        }

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [potential_ratio, kappa, synchrony, stdp_asymmetry]
        """
        if self.last_analysis is None:
            return None

        return [
            self.last_analysis['potential_ratio'],
            self.last_analysis['avalanche_coefficient'],
            self.last_analysis['synchrony_index'],
            self.last_analysis['stdp_asymmetry']
        ]

    def is_subcritical(self) -> bool:
        """
        Verifica se o sistema esta no estado Subcritico

        Returns:
            True se kappa < 1 (sinais morrem)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['cortical_state'] == 'SUBCRITICAL'

    def is_critical(self) -> bool:
        """
        Verifica se o sistema esta no estado Critico

        Returns:
            True se kappa ~ 1 (criticalidade)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['cortical_state'] == 'CRITICAL'

    def is_supercritical(self) -> bool:
        """
        Verifica se o sistema esta no estado Supercritico

        Returns:
            True se kappa > 1 (avalanche em andamento)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['cortical_state'] == 'SUPERCRITICAL'

    def is_pre_ictal(self) -> bool:
        """
        Verifica se o sistema esta no estado Pre-Ictal

        Returns:
            True se prestes a ter "convulsao financeira"
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['cortical_state'] == 'PRE_ICTAL'

    def is_quiescent(self) -> bool:
        """
        Verifica se esta em estado Quiescente

        Returns:
            True se baixa excitabilidade
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['excitability_state'] == 'QUIESCENT'

    def is_building(self) -> bool:
        """
        Verifica se esta no estado Building

        Returns:
            True se acumulando potencial
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['excitability_state'] == 'BUILDING'

    def is_hyperexcitable(self) -> bool:
        """
        Verifica se esta Hiperexcitavel

        Returns:
            True se proximo do limiar de disparo
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['excitability_state'] == 'HYPEREXCITABLE'

    def is_firing(self) -> bool:
        """
        Verifica se esta no estado Firing

        Returns:
            True se disparando ativamente
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['excitability_state'] == 'FIRING'

    def is_waiting(self) -> bool:
        """
        Verifica se o indicador esta em modo espera

        Returns:
            True se aguardando melhor oportunidade
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['signal_name'] in ['WAIT', 'NEUTRAL', 'INSUFFICIENT_DATA']

    def is_synchronized(self) -> bool:
        """
        Verifica se a rede esta sincronizada

        Returns:
            True se indice de Kuramoto alto
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['is_synchronized']

    def get_potential_ratio(self) -> Optional[float]:
        """
        Retorna a razao V/V_th

        Returns:
            Fracao do potencial em relacao ao limiar
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['potential_ratio']

    def get_avalanche_coefficient(self) -> Optional[float]:
        """
        Retorna o coeficiente de avalanche kappa

        Returns:
            kappa - parametro de ramificacao
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['avalanche_coefficient']

    def get_synchrony_index(self) -> Optional[float]:
        """
        Retorna o indice de sincronia de Kuramoto

        Returns:
            r - indice de Kuramoto [0, 1]
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['synchrony_index']

    def get_stdp_asymmetry(self) -> Optional[float]:
        """
        Retorna a assimetria STDP

        Returns:
            Positivo = UP, Negativo = DOWN
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['stdp_asymmetry']

    def get_firing_rate(self) -> Optional[float]:
        """
        Retorna a taxa de disparo

        Returns:
            Taxa em Hz
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['firing_rate']

    def get_predicted_direction(self) -> Optional[str]:
        """
        Retorna a direcao prevista do preco

        Returns:
            'UP' se STDP assimetria > 0, 'DOWN' caso contrario
        """
        if self.last_analysis is None:
            return None

        asymmetry = self.last_analysis['stdp_asymmetry']
        if asymmetry > 0.1:
            return 'UP'
        elif asymmetry < -0.1:
            return 'DOWN'
        return 'NEUTRAL'

    def is_breakout_setup(self) -> bool:
        """
        Verifica se ha setup de breakout iminente

        Returns:
            True se Pre-Ictal ou Critico + Hiperexcitavel
        """
        if self.last_analysis is None:
            return False

        state = self.last_analysis['cortical_state']
        excit = self.last_analysis['excitability_state']

        if state == 'PRE_ICTAL':
            return True

        if state == 'CRITICAL' and excit == 'HYPEREXCITABLE':
            return True

        return False

    def is_accumulating(self) -> bool:
        """
        Verifica se o mercado esta acumulando pressao

        Returns:
            True se Building ou Subcritico com sincronia alta
        """
        if self.last_analysis is None:
            return False

        excit = self.last_analysis['excitability_state']
        sync = self.last_analysis['synchrony_index']

        if excit == 'BUILDING':
            return True

        if excit == 'QUIESCENT' and sync > 0.5:
            return True

        return False

    def get_network_stats(self) -> Optional[dict]:
        """
        Retorna estatisticas da rede neural

        Returns:
            Dict com n_neurons, n_active, total_spikes, firing_rate
        """
        if self.last_analysis is None:
            return None

        return {
            'n_neurons': self.last_analysis['n_neurons'],
            'n_active': self.last_analysis['n_active'],
            'total_spikes': self.last_analysis['total_spikes'],
            'firing_rate': self.last_analysis['firing_rate']
        }
