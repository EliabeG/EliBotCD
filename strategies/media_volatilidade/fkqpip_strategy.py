"""
Adaptador de Estrategia para o Feynman-Kac Quantum Path Integral Propagator
Integra o indicador FK-QPIP com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .fkqpip_feynman_kac import (
    FeynmanKacQuantumPathIntegralPropagator,
    InterferenceType,
    TunnelingState
)


class FKQPIPStrategy(BaseStrategy):
    """
    Estrategia baseada no Feynman-Kac Quantum Path Integral Propagator (FK-QPIP)

    Usa Integrais de Caminho de Feynman para calcular a amplitude de probabilidade
    quantica do preco futuro, detectando tunelamento e interferencia.

    Conceitos-chave:
    - Lagrangiana L = T - V: Energia cinetica vs potencial do mercado
    - Integral de Caminho: Soma todas as trajetorias possiveis
    - Quantum Monte Carlo: Amostragem via Metropolis-Hastings
    - Tunelamento: Preco atravessa barreira (suporte/resistencia)
    - Interferencia: Construtiva (atrator) ou destrutiva (cancelamento)
    - Colapso da Funcao de Onda: Densidade de probabilidade futura

    Sinais:
    - Tunelamento: psi aparece do outro lado da barreira antes do preco
    - Vacuo Quantico: FADE quando preco rompe mas psi nao acompanha
    - Atrator Quantico: Interferencia construtiva cria pico de probabilidade
    """

    def __init__(self,
                 min_prices: int = 100,
                 stop_loss_pips: float = 20.0,
                 take_profit_pips: float = 40.0,
                 n_paths: int = 3000,
                 n_time_steps: int = 12,
                 hbar: float = 0.1,
                 tunneling_threshold: float = 0.15):
        """
        Inicializa a estrategia FK-QPIP

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            n_paths: Numero de caminhos para Monte Carlo
            n_time_steps: Passos de tempo no futuro
            hbar: Constante de Planck reduzida (escala quantica)
            tunneling_threshold: Limiar de probabilidade de tunelamento
        """
        super().__init__(name="FKQPIP-QED")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de precos e volumes
        self.prices = deque(maxlen=600)
        self.volumes = deque(maxlen=600)

        # Indicador FK-QPIP
        self.fkqpip = FeynmanKacQuantumPathIntegralPropagator(
            n_paths=n_paths,
            n_time_steps=n_time_steps,
            hbar=hbar,
            mass_scale=1.0,
            potential_scale=1.0,
            tunneling_threshold=tunneling_threshold,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em FK-QPIP

        Args:
            price: Preco atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (volume opcional)

        Returns:
            Signal se oportunidade quantica detectada, None caso contrario
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
            # Executa analise FK-QPIP
            result = self.fkqpip.analyze(prices_array, volumes_array)
            self.last_analysis = result

            # Verifica sinal (ignora NEUTRAL e INSUFFICIENT_DATA)
            if result['signal'] != 0 and result['confidence'] >= 0.25:
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
                self.signal_cooldown = 30  # Cooldown maior para FK-QPIP (Monte Carlo pesado)

                return signal

        except Exception as e:
            print(f"Erro na analise FK-QPIP: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"FKQPIP QED | "
                f"Interf={result['interference_type']} | "
                f"Tunel={result['tunneling_state']} | "
                f"P(up)={result['prob_up']:.1%} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.prices.clear()
        self.volumes.clear()
        self.fkqpip.reset()
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
            'interference_type': self.last_analysis['interference_type'],
            'tunneling_state': self.last_analysis['tunneling_state'],
            'quantum_attractors': self.last_analysis['quantum_attractors'],
            'attractor_probabilities': self.last_analysis['attractor_probabilities'],
            'tunneling_probability': self.last_analysis['tunneling_probability'],
            'barrier_price': self.last_analysis['barrier_price'],
            'kinetic_energy': self.last_analysis['kinetic_energy'],
            'potential_energy': self.last_analysis['potential_energy'],
            'total_action': self.last_analysis['total_action'],
            'prob_up': self.last_analysis['prob_up'],
            'prob_down': self.last_analysis['prob_down'],
            'expected_price': self.last_analysis['expected_price'],
            'n_paths_sampled': self.last_analysis['n_paths_sampled'],
            'classical_path_price': self.last_analysis['classical_path_price'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_tunneling_info(self) -> Optional[dict]:
        """Retorna informacoes de tunelamento"""
        if self.last_analysis is None:
            return None
        return {
            'state': self.last_analysis['tunneling_state'],
            'probability': self.last_analysis['tunneling_probability'],
            'barrier_price': self.last_analysis['barrier_price'],
            'barrier_height': self.last_analysis.get('barrier_height', 0.0),
            'barrier_width': self.last_analysis.get('barrier_width', 0.0)
        }

    def get_interference_info(self) -> Optional[dict]:
        """Retorna informacoes de interferencia"""
        if self.last_analysis is None:
            return None
        return {
            'type': self.last_analysis['interference_type'],
            'prob_up': self.last_analysis['prob_up'],
            'prob_down': self.last_analysis['prob_down'],
            'expected_price': self.last_analysis['expected_price']
        }

    def get_energy_info(self) -> Optional[dict]:
        """Retorna informacoes de energia"""
        if self.last_analysis is None:
            return None
        return {
            'kinetic': self.last_analysis['kinetic_energy'],
            'potential': self.last_analysis['potential_energy'],
            'action': self.last_analysis['total_action']
        }

    def get_path_info(self) -> Optional[dict]:
        """Retorna informacoes dos caminhos amostrados"""
        if self.last_analysis is None:
            return None
        return {
            'n_sampled': self.last_analysis['n_paths_sampled'],
            'classical_path_price': self.last_analysis['classical_path_price']
        }

    def get_attractors(self) -> Optional[List[dict]]:
        """Retorna lista de atratores quanticos"""
        if self.last_analysis is None:
            return None

        attractors = []
        prices = self.last_analysis['quantum_attractors']
        probs = self.last_analysis['attractor_probabilities']

        for price, prob in zip(prices, probs):
            attractors.append({
                'price': price,
                'probability': prob
            })

        return attractors

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [prob_up, prob_down, tunneling_prob, action]
        """
        if self.last_analysis is None:
            return None

        return [
            self.last_analysis['prob_up'],
            self.last_analysis['prob_down'],
            self.last_analysis['tunneling_probability'],
            self.last_analysis['total_action']
        ]

    def is_constructive_interference(self) -> bool:
        """
        Verifica se ha interferencia construtiva (atrator forte)

        Returns:
            True se caminhos convergem em um ponto
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['interference_type'] == 'CONSTRUCTIVE'

    def is_destructive_interference(self) -> bool:
        """
        Verifica se ha interferencia destrutiva

        Returns:
            True se caminhos se cancelam
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['interference_type'] == 'DESTRUCTIVE'

    def is_tunneling_likely(self) -> bool:
        """
        Verifica se tunelamento e provavel

        Returns:
            True se alta probabilidade de atravessar barreira
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['tunneling_state'] == 'TUNNELING_LIKELY'

    def is_tunneling_unlikely(self) -> bool:
        """
        Verifica se tunelamento e improvavel

        Returns:
            True se baixa probabilidade de atravessar barreira
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['tunneling_state'] == 'TUNNELING_UNLIKELY'

    def has_barrier(self) -> bool:
        """
        Verifica se ha barreira detectada

        Returns:
            True se barreira de potencial presente
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['tunneling_state'] != 'NO_BARRIER'

    def is_waiting(self) -> bool:
        """
        Verifica se o indicador esta em modo espera

        Returns:
            True se aguardando melhor oportunidade
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['signal_name'] in ['NEUTRAL', 'INSUFFICIENT_DATA']

    def is_fade_signal(self) -> bool:
        """
        Verifica se e um sinal de FADE (operar contra rompimento)

        Returns:
            True se rompimento falso detectado
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['signal_name'] == 'FADE'

    def get_prob_up(self) -> Optional[float]:
        """
        Retorna probabilidade de subir

        Returns:
            P(up) da funcao de onda
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['prob_up']

    def get_prob_down(self) -> Optional[float]:
        """
        Retorna probabilidade de descer

        Returns:
            P(down) da funcao de onda
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['prob_down']

    def get_tunneling_probability(self) -> Optional[float]:
        """
        Retorna probabilidade de tunelamento

        Returns:
            P(tunnel) - probabilidade WKB
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['tunneling_probability']

    def get_expected_price(self) -> Optional[float]:
        """
        Retorna preco esperado (expectativa da funcao de onda)

        Returns:
            E[preco] = sum(price * psi^2)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['expected_price']

    def get_action(self) -> Optional[float]:
        """
        Retorna a acao total media

        Returns:
            S - acao media dos caminhos amostrados
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['total_action']

    def get_classical_path_price(self) -> Optional[float]:
        """
        Retorna preco do caminho classico (acao minima)

        Returns:
            Preco final do caminho de menor acao
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['classical_path_price']

    def get_barrier_price(self) -> Optional[float]:
        """
        Retorna preco da barreira mais relevante

        Returns:
            Preco do suporte/resistencia mais proximo
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['barrier_price']

    def get_prob_history(self) -> tuple:
        """Retorna historico de probabilidades"""
        return self.fkqpip.get_prob_history()

    def get_tunneling_history(self) -> np.ndarray:
        """Retorna historico de tunelamento"""
        return self.fkqpip.get_tunneling_history()

    def is_quantum_setup(self) -> bool:
        """
        Verifica se ha um setup quantico ativo

        Returns:
            True se tunelamento + viés direcional forte
        """
        if self.last_analysis is None:
            return False

        tunneling_likely = self.last_analysis['tunneling_state'] == 'TUNNELING_LIKELY'
        prob_up = self.last_analysis['prob_up']
        prob_down = self.last_analysis['prob_down']

        strong_bias = abs(prob_up - prob_down) > 0.2

        return tunneling_likely and strong_bias

    def is_vacuum_trap(self) -> bool:
        """
        Verifica se ha armadilha de vacuo quantico (rompimento falso)

        Returns:
            True se interferencia destrutiva + movimento recente
        """
        if self.last_analysis is None:
            return False

        return (self.last_analysis['interference_type'] == 'DESTRUCTIVE' and
                self.last_analysis['signal_name'] == 'FADE')

    def get_dominant_attractor(self) -> Optional[dict]:
        """
        Retorna o atrator quantico dominante

        Returns:
            Dict com price e probability do atrator mais forte
        """
        if self.last_analysis is None:
            return None

        attractors = self.last_analysis['quantum_attractors']
        probs = self.last_analysis['attractor_probabilities']

        if not attractors:
            return None

        max_idx = np.argmax(probs)
        return {
            'price': attractors[max_idx],
            'probability': probs[max_idx]
        }
