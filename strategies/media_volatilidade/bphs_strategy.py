"""
Adaptador de Estrategia para o Betti-Persistence Homology Scanner
Integra o indicador B-PHS com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List, Dict
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .bphs_betti_persistence import BettiPersistenceHomologyScanner, TopologyRegime


class BPHSStrategy(BaseStrategy):
    """
    Estrategia baseada no Betti-Persistence Homology Scanner (B-PHS)

    Usa Topologia Algebrica para analisar a FORMA dos dados de mercado.
    Detecta ciclos persistentes via Homologia e gera sinais baseados na
    posicao do preco dentro do ciclo topologico.

    Conceitos-chave:
    - Takens' Embedding: Reconstrucao do espaco de fase
    - Vietoris-Rips Filtration: Complexo simplicial
    - Numeros de Betti: B0 (componentes), B1 (loops), B2 (vazios)
    - Persistencia: Robustez das caracteristicas topologicas
    - Santo Graal: B1 alto e persistente = mercado ciclico
    """

    def __init__(self,
                 min_prices: int = 100,
                 stop_loss_pips: float = 20.0,
                 take_profit_pips: float = 40.0,
                 embedding_dim: int = 3,
                 time_delay: Optional[int] = None,
                 filtration_steps: int = 50,
                 min_loop_persistence: float = 0.1,
                 max_entropy_threshold: float = 2.0,
                 position_threshold: float = 0.3):
        """
        Inicializa a estrategia B-PHS

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            embedding_dim: Dimensao de imersao de Takens
            time_delay: Atraso temporal (tau). Se None, calcula automaticamente.
            filtration_steps: Passos da filtracao de Vietoris-Rips
            min_loop_persistence: Persistencia minima para sinal valido
            max_entropy_threshold: Limiar de entropia para regime caotico
            position_threshold: Limiar para detectar extremos do ciclo
        """
        super().__init__(name="BPHS-BettiHomology")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de precos
        self.prices = deque(maxlen=600)

        # Indicador B-PHS
        self.bphs = BettiPersistenceHomologyScanner(
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            max_homology_dim=2,
            filtration_steps=filtration_steps,
            min_loop_persistence=min_loop_persistence,
            max_entropy_threshold=max_entropy_threshold,
            position_threshold=position_threshold,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def add_price(self, price: float):
        """Adiciona um preco ao buffer"""
        self.prices.append(price)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em B-PHS

        Args:
            price: Preco atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais

        Returns:
            Signal se extremo de ciclo topologico detectado, None caso contrario
        """
        # Adiciona preco ao buffer
        self.add_price(price)

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
            # Executa analise B-PHS
            result = self.bphs.analyze(prices_array)
            self.last_analysis = result

            # Verifica sinal (ignora BLOCKED e NEUTRAL)
            if result['signal'] != 0 and result['confidence'] >= 0.4:
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
                self.signal_cooldown = 25  # Cooldown maior para B-PHS

                return signal

        except Exception as e:
            print(f"Erro na analise B-PHS: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"BPHS Topology | "
                f"Regime={result['regime']} | "
                f"B1={result['betti_1']} | "
                f"Pers={result['max_loop_persistence']:.3f} | "
                f"Pos={result['position_in_loop']} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.prices.clear()
        self.bphs.reset()
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
            'regime': self.last_analysis['regime'],
            'betti_0': self.last_analysis['betti_0'],
            'betti_1': self.last_analysis['betti_1'],
            'betti_2': self.last_analysis['betti_2'],
            'max_loop_persistence': self.last_analysis['max_loop_persistence'],
            'topological_entropy': self.last_analysis['topological_entropy'],
            'position_in_loop': self.last_analysis['position_in_loop'],
            'distance_to_centroid': self.last_analysis['distance_to_centroid'],
            'embedding_dim': self.last_analysis['embedding_dim'],
            'time_delay': self.last_analysis['time_delay'],
            'current_price': self.last_analysis['current_price'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_betti_numbers(self) -> Optional[dict]:
        """Retorna numeros de Betti atuais"""
        if self.last_analysis is None:
            return None
        return {
            'betti_0': self.last_analysis['betti_0'],
            'betti_1': self.last_analysis['betti_1'],
            'betti_2': self.last_analysis['betti_2']
        }

    def get_topology_info(self) -> Optional[dict]:
        """Retorna informacoes topologicas"""
        if self.last_analysis is None:
            return None
        return {
            'regime': self.last_analysis['regime'],
            'max_loop_persistence': self.last_analysis['max_loop_persistence'],
            'topological_entropy': self.last_analysis['topological_entropy'],
            'position_in_loop': self.last_analysis['position_in_loop']
        }

    def get_embedding_info(self) -> Optional[dict]:
        """Retorna informacoes do embedding de Takens"""
        if self.last_analysis is None:
            return None
        return {
            'embedding_dim': self.last_analysis['embedding_dim'],
            'time_delay': self.last_analysis['time_delay'],
            'optimal_epsilon': self.last_analysis['optimal_epsilon']
        }

    def get_betti_history(self) -> List[int]:
        """Retorna historico de B1"""
        return self.bphs.get_betti_history()

    def get_entropy_history(self) -> List[float]:
        """Retorna historico de entropia topologica"""
        return self.bphs.get_entropy_history()

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [betti_1, persistence, entropy, position_code]
        """
        if self.last_analysis is None:
            return None

        # Codifica posicao
        pos_map = {"TOP": 1.0, "BOTTOM": -1.0, "MIDDLE": 0.0, "UNKNOWN": 0.0}
        pos_code = pos_map.get(self.last_analysis['position_in_loop'], 0.0)

        return [
            float(self.last_analysis['betti_1']),
            self.last_analysis['max_loop_persistence'],
            self.last_analysis['topological_entropy'],
            pos_code
        ]

    def is_cyclic_regime(self) -> bool:
        """
        Verifica se o mercado esta em regime ciclico

        Returns:
            True se B1 > 0 e persistencia significativa
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['regime'] == 'CYCLIC'

    def is_linear_regime(self) -> bool:
        """
        Verifica se o mercado esta em regime linear

        Returns:
            True se B1 = 0 (sem ciclos)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['regime'] == 'LINEAR'

    def is_chaotic_regime(self) -> bool:
        """
        Verifica se o mercado esta em regime caotico

        Returns:
            True se entropia alta
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['regime'] == 'CHAOTIC'

    def is_at_cycle_top(self) -> bool:
        """
        Verifica se esta no topo do ciclo topologico

        Returns:
            True se posicao = TOP
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['position_in_loop'] == 'TOP'

    def is_at_cycle_bottom(self) -> bool:
        """
        Verifica se esta na base do ciclo topologico

        Returns:
            True se posicao = BOTTOM
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['position_in_loop'] == 'BOTTOM'

    def get_betti_1(self) -> Optional[int]:
        """
        Retorna o numero de Betti-1 (loops) - SANTO GRAAL

        Returns:
            Numero de loops no espaco topologico
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['betti_1']

    def get_loop_persistence(self) -> Optional[float]:
        """
        Retorna a persistencia do loop dominante

        Returns:
            Persistencia maxima (vida util do loop)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['max_loop_persistence']

    def get_topological_entropy(self) -> Optional[float]:
        """
        Retorna a entropia topologica

        Returns:
            Entropia = medida de complexidade topologica
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['topological_entropy']

    def get_position_in_loop(self) -> Optional[str]:
        """
        Retorna a posicao atual no ciclo

        Returns:
            TOP, BOTTOM, MIDDLE ou UNKNOWN
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['position_in_loop']

    def has_persistent_loop(self, min_persistence: float = 0.1) -> bool:
        """
        Verifica se ha um loop persistente

        Args:
            min_persistence: Persistencia minima

        Returns:
            True se B1 > 0 e persistencia >= min_persistence
        """
        if self.last_analysis is None:
            return False
        return (self.last_analysis['betti_1'] > 0 and
                self.last_analysis['max_loop_persistence'] >= min_persistence)

    def is_topology_stable(self) -> bool:
        """
        Verifica se a topologia esta estavel

        Returns:
            True se entropia baixa e B1 consistente
        """
        if self.last_analysis is None:
            return False

        entropy = self.last_analysis['topological_entropy']
        betti_history = self.get_betti_history()

        if len(betti_history) < 3:
            return False

        # Entropia baixa
        if entropy > 1.5:
            return False

        # B1 consistente (sem grandes variacoes)
        recent = betti_history[-3:]
        return max(recent) - min(recent) <= 1
