"""
Adaptador de Estrategia para o Parisi Replica Symmetry Breaking Detector
Integra o indicador PRSBD com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .prsbd_parisi_replica import (
    ParisiReplicaSymmetryBreakingDetector,
    PhaseState,
    SymmetryState
)


class PRSBDStrategy(BaseStrategy):
    """
    Estrategia baseada no Parisi Replica Symmetry Breaking Detector (PRSBD)

    Usa fisica de vidros de spin para detectar transicoes de volatilidade
    antes do preco se mover, modelando a Paisagem de Energia Livre do mercado.

    Conceitos-chave:
    - Hamiltoniano de Vidro de Spin: H = -Sum J_ij S_i S_j
    - Metodo das Replicas: n copias do sistema com condicoes iniciais diferentes
    - Parametro de Edwards-Anderson (q_EA): Ordem do sistema
    - Susceptibilidade chi_SG: Detecta transicoes de fase
    - Estrutura Ultrametrica: Hierarquia de estados

    Sinais:
    - FERROMAGNETICO: Simetria quebrada, tendencia iniciando
    - CRITICO + chi divergindo: Transicao iminente
    - SPIN GLASS + chi divergindo: RSB colapsando
    """

    def __init__(self,
                 min_prices: int = 50,
                 stop_loss_pips: float = 15.0,
                 take_profit_pips: float = 30.0,
                 n_replicas: int = 25,
                 n_sweeps: int = 400,
                 T_initial: float = 4.0,
                 T_final: float = 0.1,
                 divergence_threshold: float = 2.5):
        """
        Inicializa a estrategia PRSBD

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            n_replicas: Numero de replicas a simular
            n_sweeps: Numero de sweeps de Monte Carlo
            T_initial: Temperatura inicial
            T_final: Temperatura final
            divergence_threshold: Limiar para divergencia de chi
        """
        super().__init__(name="PRSBD-SpinGlass")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de precos
        self.prices = deque(maxlen=400)

        # Indicador PRSBD
        self.prsbd = ParisiReplicaSymmetryBreakingDetector(
            n_replicas=n_replicas,
            n_sweeps=n_sweeps,
            T_initial=T_initial,
            T_final=T_final,
            divergence_threshold=divergence_threshold,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em PRSBD

        Args:
            price: Preco atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (nao utilizados)

        Returns:
            Signal se transicao de fase detectada, None caso contrario
        """
        # Adiciona preco ao buffer
        self.prices.append(price)

        # Verifica se temos dados suficientes
        if len(self.prices) < self.min_prices:
            return None

        # Cooldown para evitar sinais em sequencia
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        # Converte para numpy array
        prices_array = np.array(self.prices)

        try:
            # Executa analise PRSBD
            result = self.prsbd.analyze(prices_array)
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
                self.signal_cooldown = 25  # Cooldown para PRSBD (Monte Carlo pesado)

                return signal

        except Exception as e:
            print(f"Erro na analise PRSBD: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"PRSBD SpinGlass | "
                f"Phase={result['phase_state']} | "
                f"Sym={result['symmetry_state'][:3]} | "
                f"chi_SG={result['chi_SG']:.2f} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.prices.clear()
        self.prsbd.reset()
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
            'phase_state': self.last_analysis['phase_state'],
            'symmetry_state': self.last_analysis['symmetry_state'],
            'q_EA': self.last_analysis['q_EA'],
            'q_variance': self.last_analysis['q_variance'],
            'chi_SG': self.last_analysis['chi_SG'],
            'chi_FM': self.last_analysis['chi_FM'],
            'chi_diverging': self.last_analysis['chi_diverging'],
            'magnetization': self.last_analysis['magnetization'],
            'best_magnetization': self.last_analysis['best_magnetization'],
            'mean_energy': self.last_analysis['mean_energy'],
            'is_ultrametric': self.last_analysis['is_ultrametric'],
            'hierarchy_depth': self.last_analysis['hierarchy_depth'],
            'complexity_entropy': self.last_analysis['complexity_entropy'],
            'frustration': self.last_analysis['frustration'],
            'n_replicas': self.last_analysis['n_replicas'],
            'n_converged': self.last_analysis['n_converged'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_edwards_anderson_info(self) -> Optional[dict]:
        """Retorna informacoes de Edwards-Anderson"""
        if self.last_analysis is None:
            return None
        return {
            'q_EA': self.last_analysis['q_EA'],
            'q_variance': self.last_analysis['q_variance']
        }

    def get_susceptibility_info(self) -> Optional[dict]:
        """Retorna informacoes de susceptibilidade"""
        if self.last_analysis is None:
            return None
        return {
            'chi_SG': self.last_analysis['chi_SG'],
            'chi_FM': self.last_analysis['chi_FM'],
            'divergence_ratio': self.last_analysis['divergence_ratio'],
            'chi_diverging': self.last_analysis['chi_diverging']
        }

    def get_magnetization_info(self) -> Optional[dict]:
        """Retorna informacoes de magnetizacao"""
        if self.last_analysis is None:
            return None
        return {
            'mean': self.last_analysis['magnetization'],
            'best': self.last_analysis['best_magnetization'],
            'variance': self.last_analysis['magnetization_variance']
        }

    def get_energy_info(self) -> Optional[dict]:
        """Retorna informacoes de energia"""
        if self.last_analysis is None:
            return None
        return {
            'mean': self.last_analysis['mean_energy'],
            'variance': self.last_analysis['energy_variance']
        }

    def get_structure_info(self) -> Optional[dict]:
        """Retorna informacoes de estrutura"""
        if self.last_analysis is None:
            return None
        return {
            'is_ultrametric': self.last_analysis['is_ultrametric'],
            'hierarchy_depth': self.last_analysis['hierarchy_depth'],
            'complexity_entropy': self.last_analysis['complexity_entropy'],
            'frustration': self.last_analysis['frustration']
        }

    def get_replica_info(self) -> Optional[dict]:
        """Retorna informacoes das replicas"""
        if self.last_analysis is None:
            return None
        return {
            'n_replicas': self.last_analysis['n_replicas'],
            'n_converged': self.last_analysis['n_converged'],
            'n_spins': self.last_analysis['n_spins']
        }

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [q_EA, chi_SG, magnetization, complexity_entropy]
        """
        if self.last_analysis is None:
            return None

        return [
            self.last_analysis['q_EA'],
            self.last_analysis['chi_SG'],
            self.last_analysis['magnetization'],
            self.last_analysis['complexity_entropy']
        ]

    def is_paramagnetic(self) -> bool:
        """
        Verifica se o sistema esta na fase paramagnetica

        Returns:
            True se mercado morto (ruido puro)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['phase_state'] == 'PARAMAGNETIC'

    def is_spin_glass(self) -> bool:
        """
        Verifica se o sistema esta na fase de vidro de spin

        Returns:
            True se RSB ativo (acumulacao)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['phase_state'] == 'SPIN_GLASS'

    def is_ferromagnetic(self) -> bool:
        """
        Verifica se o sistema esta na fase ferromagnetica

        Returns:
            True se simetria quebrada (tendencia)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['phase_state'] == 'FERROMAGNETIC'

    def is_critical(self) -> bool:
        """
        Verifica se o sistema esta no ponto critico

        Returns:
            True se no ponto de transicao
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['phase_state'] == 'CRITICAL'

    def is_replica_symmetric(self) -> bool:
        """
        Verifica se ha simetria de replicas

        Returns:
            True se todas replicas iguais
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['symmetry_state'] == 'REPLICA_SYMMETRIC'

    def is_rsb(self) -> bool:
        """
        Verifica se ha quebra de simetria de replicas (RSB)

        Returns:
            True se estrutura hierarquica
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['symmetry_state'] == 'REPLICA_SYMMETRY_BREAKING'

    def is_symmetry_broken(self) -> bool:
        """
        Verifica se a simetria esta totalmente quebrada

        Returns:
            True se simetria quebrada
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['symmetry_state'] == 'SYMMETRY_BROKEN'

    def is_waiting(self) -> bool:
        """
        Verifica se o indicador esta em modo espera

        Returns:
            True se aguardando melhor oportunidade
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['signal_name'] in ['WAIT', 'NEUTRAL', 'INSUFFICIENT_DATA']

    def is_chi_diverging(self) -> bool:
        """
        Verifica se chi_SG esta divergindo

        Returns:
            True se susceptibilidade crescendo exponencialmente
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['chi_diverging']

    def is_ultrametric(self) -> bool:
        """
        Verifica se a estrutura e ultrametrica

        Returns:
            True se estrutura hierarquica detectada
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['is_ultrametric']

    def get_q_EA(self) -> Optional[float]:
        """
        Retorna o parametro de Edwards-Anderson

        Returns:
            q_EA - parametro de ordem do vidro de spin
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['q_EA']

    def get_chi_SG(self) -> Optional[float]:
        """
        Retorna a susceptibilidade de spin glass

        Returns:
            chi_SG
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['chi_SG']

    def get_magnetization(self) -> Optional[float]:
        """
        Retorna a magnetizacao media

        Returns:
            M - magnetizacao
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['magnetization']

    def get_best_magnetization(self) -> Optional[float]:
        """
        Retorna a magnetizacao da replica de menor energia

        Returns:
            M do ground state
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['best_magnetization']

    def get_frustration(self) -> Optional[float]:
        """
        Retorna o nivel de frustracao

        Returns:
            Fracao de plaquetas frustradas
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['frustration']

    def get_complexity_entropy(self) -> Optional[float]:
        """
        Retorna a entropia de complexidade de P(q)

        Returns:
            Entropia de Shannon de P(q)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['complexity_entropy']

    def get_chi_history(self) -> np.ndarray:
        """Retorna historico de susceptibilidade"""
        return self.prsbd.get_chi_history()

    def is_phase_transition_imminent(self) -> bool:
        """
        Verifica se transicao de fase e iminente

        Returns:
            True se chi divergindo + estrutura ultrametrica
        """
        if self.last_analysis is None:
            return False

        return (self.last_analysis['chi_diverging'] and
                (self.last_analysis['is_ultrametric'] or
                 self.last_analysis['phase_state'] in ['SPIN_GLASS', 'CRITICAL']))

    def is_false_breakout_likely(self) -> bool:
        """
        Verifica se rompimento falso e provavel

        Returns:
            True se RSB ativo mas preco rompeu (frustracao interna)
        """
        if self.last_analysis is None:
            return False

        return (self.last_analysis['symmetry_state'] == 'REPLICA_SYMMETRY_BREAKING' and
                self.last_analysis['is_ultrametric'] and
                not self.last_analysis['chi_diverging'])

    def get_phase_state(self) -> Optional[str]:
        """
        Retorna o estado de fase atual

        Returns:
            PARAMAGNETIC, SPIN_GLASS, FERROMAGNETIC ou CRITICAL
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['phase_state']

    def get_symmetry_state(self) -> Optional[str]:
        """
        Retorna o estado de simetria atual

        Returns:
            RS, RSB ou BROKEN
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['symmetry_state']
