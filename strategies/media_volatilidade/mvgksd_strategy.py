"""
Adaptador de Estrategia para o Multiplex Visibility Graph Kuramoto Synchronization Detector
Integra o indicador MVG-KSD com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .mvgksd_visibility_kuramoto import (
    MultiplexVisibilityKuramotoDetector,
    SyncState,
    NetworkTopology
)


class MVGKSDStrategy(BaseStrategy):
    """
    Estrategia baseada no Multiplex Visibility Graph & Kuramoto Synchronization Detector

    Transforma a serie temporal em uma Rede Complexa (Complex Network) e mede
    a Sincronizacao de Fase dos nos para prever rupturas de estabilidade critica.

    Conceitos-chave:
    - Natural Visibility Graph (NVG): Conecta candles com "linha de visao"
    - Kuramoto Model: Osciladores de fase acoplados
    - Parametro de Ordem r: Mede sincronizacao global (0=caos, 1=sync)
    - Estado Chimera: Coexistencia de grupos sync/desync (media vol ideal)
    - Entropia de Von Neumann: Complexidade estrutural do grafo
    - Centralidade de Autovetor: Identifica hubs (suportes/resistencias)

    Sinais:
    - Ruptura de Simetria: Entropia cai + Hub instavel
    - Queda Cega: No desconectado (reverter para reconectar)
    - No Folha: Visibilidade bloqueada (preco obrigado a cair)
    """

    def __init__(self,
                 min_prices: int = 100,
                 stop_loss_pips: float = 20.0,
                 take_profit_pips: float = 40.0,
                 visibility_window: int = 30,
                 kuramoto_coupling: float = 1.0,
                 kuramoto_iterations: int = 30,
                 entropy_drop_threshold: float = 0.3,
                 hub_concentration_threshold: float = 0.7):
        """
        Inicializa a estrategia MVG-KSD

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            visibility_window: Janela para visibilidade no grafo
            kuramoto_coupling: K - forca de acoplamento de Kuramoto
            kuramoto_iterations: Iteracoes para convergencia de Kuramoto
            entropy_drop_threshold: Limiar para queda de entropia
            hub_concentration_threshold: Limiar para concentracao de hubs
        """
        super().__init__(name="MVGKSD-ComplexNetwork")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de precos
        self.prices = deque(maxlen=600)

        # Indicador MVG-KSD
        self.mvgksd = MultiplexVisibilityKuramotoDetector(
            visibility_window=visibility_window,
            use_horizontal_visibility=True,
            kuramoto_coupling=kuramoto_coupling,
            kuramoto_iterations=kuramoto_iterations,
            entropy_drop_threshold=entropy_drop_threshold,
            hub_concentration_threshold=hub_concentration_threshold,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em MVG-KSD

        Args:
            price: Preco atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (nao utilizados)

        Returns:
            Signal se ruptura topologica detectada, None caso contrario
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
            # Executa analise MVG-KSD
            result = self.mvgksd.analyze(prices_array)
            self.last_analysis = result

            # Verifica sinal (ignora WAIT e NEUTRAL)
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
                self.signal_cooldown = 25  # Cooldown para MVG-KSD

                return signal

        except Exception as e:
            print(f"Erro na analise MVG-KSD: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"MVGKSD Network | "
                f"Topo={result['network_topology']} | "
                f"Sync={result['sync_state']} | "
                f"r={result['order_parameter']:.3f} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.prices.clear()
        self.mvgksd.reset()
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
            'network_topology': self.last_analysis['network_topology'],
            'sync_state': self.last_analysis['sync_state'],
            'order_parameter': self.last_analysis['order_parameter'],
            'von_neumann_entropy': self.last_analysis['von_neumann_entropy'],
            'entropy_drop': self.last_analysis['entropy_drop'],
            'hub_concentration': self.last_analysis['hub_concentration'],
            'hub_instability': self.last_analysis['hub_instability'],
            'current_node_degree': self.last_analysis['current_node_degree'],
            'is_leaf_node': self.last_analysis['is_leaf_node'],
            'lost_connectivity': self.last_analysis['lost_connectivity'],
            'n_nodes': self.last_analysis['n_nodes'],
            'n_edges': self.last_analysis['n_edges'],
            'graph_density': self.last_analysis['graph_density'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_kuramoto_info(self) -> Optional[dict]:
        """Retorna informacoes de Kuramoto"""
        if self.last_analysis is None:
            return None
        return {
            'order_parameter': self.last_analysis['order_parameter'],
            'mean_phase': self.last_analysis['mean_phase'],
            'sync_state': self.last_analysis['sync_state']
        }

    def get_spectral_info(self) -> Optional[dict]:
        """Retorna informacoes espectrais"""
        if self.last_analysis is None:
            return None
        return {
            'von_neumann_entropy': self.last_analysis['von_neumann_entropy'],
            'entropy_drop': self.last_analysis['entropy_drop'],
            'algebraic_connectivity': self.last_analysis['algebraic_connectivity'],
            'spectral_gap': self.last_analysis.get('spectral_gap', 0.0)
        }

    def get_centrality_info(self) -> Optional[dict]:
        """Retorna informacoes de centralidade"""
        if self.last_analysis is None:
            return None
        return {
            'hub_concentration': self.last_analysis['hub_concentration'],
            'hub_instability': self.last_analysis['hub_instability'],
            'max_centrality_node': self.last_analysis.get('max_centrality_node', 0)
        }

    def get_graph_info(self) -> Optional[dict]:
        """Retorna informacoes do grafo"""
        if self.last_analysis is None:
            return None
        return {
            'n_nodes': self.last_analysis['n_nodes'],
            'n_edges': self.last_analysis['n_edges'],
            'density': self.last_analysis['graph_density'],
            'topology': self.last_analysis['network_topology']
        }

    def get_node_info(self) -> Optional[dict]:
        """Retorna informacoes do no atual"""
        if self.last_analysis is None:
            return None
        return {
            'degree': self.last_analysis['current_node_degree'],
            'is_leaf': self.last_analysis['is_leaf_node'],
            'lost_connectivity': self.last_analysis['lost_connectivity']
        }

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [order_parameter, entropy, hub_concentration, degree]
        """
        if self.last_analysis is None:
            return None

        return [
            self.last_analysis['order_parameter'],
            self.last_analysis['von_neumann_entropy'],
            self.last_analysis['hub_concentration'],
            float(self.last_analysis['current_node_degree'])
        ]

    def is_chimera_state(self) -> bool:
        """
        Verifica se o mercado esta em estado Chimera (ideal para media vol)

        Returns:
            True se 0.3 < r < 0.7
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['sync_state'] == 'CHIMERA'

    def is_synchronized(self) -> bool:
        """
        Verifica se o mercado esta sincronizado (tendencia forte)

        Returns:
            True se r > 0.7
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['sync_state'] == 'SYNCHRONIZED'

    def is_incoherent(self) -> bool:
        """
        Verifica se o mercado esta incoerente (ruido)

        Returns:
            True se r < 0.3
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['sync_state'] == 'INCOHERENT'

    def is_waiting(self) -> bool:
        """
        Verifica se o indicador esta em modo espera

        Returns:
            True se aguardando melhor oportunidade
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['signal_name'] == 'WAIT'

    def is_scale_free(self) -> bool:
        """
        Verifica se a rede e Scale-Free (robusta)

        Returns:
            True se topologia Scale-Free
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['network_topology'] == 'SCALE_FREE'

    def is_small_world(self) -> bool:
        """
        Verifica se a rede e Small-World

        Returns:
            True se topologia Small-World
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['network_topology'] == 'SMALL_WORLD'

    def has_entropy_drop(self) -> bool:
        """
        Verifica se houve queda de entropia

        Returns:
            True se entropia caiu subitamente
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['entropy_drop']

    def has_hub_instability(self) -> bool:
        """
        Verifica se ha hub instavel

        Returns:
            True se hub instavel formado
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['hub_instability']

    def is_current_node_leaf(self) -> bool:
        """
        Verifica se o no atual e uma folha

        Returns:
            True se grau baixo (folha)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['is_leaf_node']

    def has_lost_connectivity(self) -> bool:
        """
        Verifica se o no atual perdeu conectividade

        Returns:
            True se desconectado do passado
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['lost_connectivity']

    def get_order_parameter(self) -> Optional[float]:
        """
        Retorna o parametro de ordem de Kuramoto

        Returns:
            r - parametro de ordem global
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['order_parameter']

    def get_von_neumann_entropy(self) -> Optional[float]:
        """
        Retorna a entropia de Von Neumann do grafo

        Returns:
            S(rho) - entropia quantica do grafo
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['von_neumann_entropy']

    def get_hub_concentration(self) -> Optional[float]:
        """
        Retorna a concentracao de hubs

        Returns:
            Gini-like concentration
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['hub_concentration']

    def get_current_node_degree(self) -> Optional[int]:
        """
        Retorna o grau do no atual

        Returns:
            Numero de conexoes do no atual
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['current_node_degree']

    def get_network_topology(self) -> Optional[str]:
        """
        Retorna a topologia da rede

        Returns:
            SCALE_FREE, SMALL_WORLD, RANDOM ou REGULAR
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['network_topology']

    def get_entropy_history(self) -> np.ndarray:
        """Retorna historico de entropia"""
        return self.mvgksd.get_entropy_history()

    def get_order_param_history(self) -> np.ndarray:
        """Retorna historico do parametro de ordem"""
        return self.mvgksd.get_order_param_history()

    def is_symmetry_breaking(self) -> bool:
        """
        Verifica se ha ruptura de simetria (setup principal)

        Returns:
            True se entropia cai + hub instavel
        """
        if self.last_analysis is None:
            return False

        return (self.last_analysis['entropy_drop'] and
                self.last_analysis['hub_instability'])
