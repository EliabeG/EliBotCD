"""
Sintetizador de Topos Grothendieck-Kolmogorov (STGK)
=====================================================
Nível de Complexidade: Medalha Fields / Fundamentos da Matemática.

Premissa Teórica:
1. Visão Categórica: O mercado não é uma série temporal. É um Feixe (Sheaf) sobre um
   espaço topológico. Dados locais (preços em 1 min) devem ser "colados" para formar
   dados globais (tendência diária). Falhas nessa colagem (obstruções cohomológicas) são
   onde o lucro reside.

2. Visão Algorítmica: A probabilidade de um movimento de preço é inversamente
   proporcional à complexidade do programa necessário para descrevê-lo (Navalha de
   Occam computacional). Buscamos padrões de Baixa Complexidade de Kolmogorov
   escondidos dentro de dados aparentemente aleatórios.

Dependências Críticas: networkx, zlib, lzma, scipy
"""

import numpy as np
import zlib
import lzma
from collections import defaultdict
from typing import Dict, List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

# Tentar importar networkx
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Tentar importar scipy para álgebra linear
try:
    from scipy.linalg import svd, null_space
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class Sheaf:
    """
    Implementação de um Feixe (Sheaf) sobre um espaço topológico.

    Um Feixe F sobre um espaço topológico X associa a cada conjunto aberto U ⊆ X
    um conjunto F(U) (as seções sobre U), junto com mapas de restrição.

    Módulo 1: A Estrutura do Topos (Definição da Categoria)
    """

    def __init__(self):
        """
        Inicializa o Feixe como um grafo direcionado acíclico (DAG).

        - Nós representam conjuntos abertos (janelas de tempo)
        - Arestas representam mapas de restrição
        """
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            self.graph = None
            self._nodes = {}
            self._edges = {}

        # F(U) - Seções sobre cada conjunto aberto
        self.sections: Dict[str, np.ndarray] = {}

        # Mapas de restrição ρ_{V,U}: F(V) → F(U) para U ⊆ V
        self.restriction_maps: Dict[Tuple[str, str], Callable] = {}

        # Estrutura de cobertura (quais abertos cobrem quais)
        self.covers: Dict[str, List[str]] = {}

        # Cache de cohomologia
        self._cohomology_cache = {}

    def add_open_set(self, name: str, section: np.ndarray):
        """
        Adiciona um conjunto aberto U com sua seção F(U).
        """
        if NETWORKX_AVAILABLE:
            self.graph.add_node(name, section=section)
        else:
            self._nodes[name] = {'section': section}

        self.sections[name] = section

    def add_restriction(self, V: str, U: str, rho: Callable = None):
        """
        Adiciona mapa de restrição ρ_{V,U}: F(V) → F(U).
        """
        if rho is None:
            rho = lambda x: x

        if NETWORKX_AVAILABLE:
            self.graph.add_edge(V, U, restriction=rho)
        else:
            self._edges[(V, U)] = {'restriction': rho}

        self.restriction_maps[(V, U)] = rho

        if V not in self.covers:
            self.covers[V] = []
        self.covers[V].append(U)

    def get_section(self, U: str) -> np.ndarray:
        """Retorna a seção F(U) sobre o conjunto aberto U."""
        return self.sections.get(U, np.array([]))

    def restrict(self, V: str, U: str, section: np.ndarray = None) -> np.ndarray:
        """Aplica o mapa de restrição ρ_{V,U} a uma seção."""
        if section is None:
            section = self.get_section(V)

        if (V, U) in self.restriction_maps:
            return self.restriction_maps[(V, U)](section)

        return section

    def check_gluing_axiom(self, U: str, covering: List[str]) -> Tuple[bool, float]:
        """
        Verifica o axioma de colagem (gluing axiom) do feixe.

        Retorna (satisfeito, erro_de_colagem)
        """
        if len(covering) < 2:
            return True, 0.0

        total_error = 0.0
        n_comparisons = 0

        for i, U_i in enumerate(covering):
            for j, U_j in enumerate(covering[i+1:], i+1):
                s_i = self.get_section(U_i)
                s_j = self.get_section(U_j)

                min_len = min(len(s_i), len(s_j))
                if min_len > 0:
                    error = np.mean(np.abs(s_i[:min_len] - s_j[:min_len]))
                    total_error += error
                    n_comparisons += 1

        avg_error = total_error / n_comparisons if n_comparisons > 0 else 0.0

        return avg_error < 0.1, avg_error


class CechCohomology:
    """
    Módulo 2: Cálculo da Cohomologia de Čech (H^n(U, F))

    - H⁰ (0-th Cohomology Group): Mede as seções globais.
    - H¹ (1st Cohomology Group): Mede as obstruções.

    H¹ = Ker(δ¹)/Im(δ⁰)
    """

    def __init__(self, sheaf: Sheaf):
        self.sheaf = sheaf
        self.eps = 1e-10
        self.H0 = None
        self.H1 = None
        self.delta0_matrix = None
        self.delta1_matrix = None

    def build_nerve(self, open_sets: List[str]) -> Dict:
        """Constrói o nervo da cobertura."""
        nerve = {
            'vertices': open_sets,
            'edges': [],
            'faces': []
        }

        n = len(open_sets)

        for i in range(n):
            for j in range(i + 1, n):
                nerve['edges'].append((open_sets[i], open_sets[j]))

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    nerve['faces'].append(
                        (open_sets[i], open_sets[j], open_sets[k])
                    )

        return nerve

    def compute_delta0(self, open_sets: List[str]) -> np.ndarray:
        """
        Computa o operador de cobordo δ⁰: C⁰ → C¹

        δ⁰(f)(U_i, U_j) = f(U_j) - f(U_i)
        """
        n = len(open_sets)
        n_edges = n * (n - 1) // 2

        delta0 = np.zeros((n_edges, n))

        edge_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                delta0[edge_idx, i] = -1
                delta0[edge_idx, j] = 1
                edge_idx += 1

        self.delta0_matrix = delta0
        return delta0

    def compute_delta1(self, open_sets: List[str]) -> np.ndarray:
        """
        Computa o operador de cobordo δ¹: C¹ → C²

        δ¹(g)(U_i, U_j, U_k) = g(U_j, U_k) - g(U_i, U_k) + g(U_i, U_j)
        """
        n = len(open_sets)
        n_edges = n * (n - 1) // 2
        n_faces = n * (n - 1) * (n - 2) // 6

        if n_faces == 0:
            self.delta1_matrix = np.zeros((1, n_edges))
            return self.delta1_matrix

        edge_to_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_to_idx[(i, j)] = idx
                idx += 1

        delta1 = np.zeros((n_faces, n_edges))

        face_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    delta1[face_idx, edge_to_idx[(j, k)]] = 1
                    delta1[face_idx, edge_to_idx[(i, k)]] = -1
                    delta1[face_idx, edge_to_idx[(i, j)]] = 1
                    face_idx += 1

        self.delta1_matrix = delta1
        return delta1

    def compute_H0(self, open_sets: List[str]) -> dict:
        """
        Computa H⁰ = Ker(δ⁰)
        """
        delta0 = self.compute_delta0(open_sets)

        if SCIPY_AVAILABLE:
            kernel = null_space(delta0)
            dim_H0 = kernel.shape[1] if kernel.size > 0 else 0
        else:
            U, S, Vh = np.linalg.svd(delta0)
            null_mask = S < self.eps
            dim_H0 = np.sum(null_mask)
            kernel = Vh[len(S):].T if len(S) < Vh.shape[0] else np.array([])

        sections = [self.sheaf.get_section(U) for U in open_sets]

        if len(sections) > 1:
            consistency = self._compute_section_consistency(sections)
        else:
            consistency = 1.0

        self.H0 = {
            'dimension': max(1, dim_H0),
            'kernel': kernel,
            'consistency': consistency,
            'unified_trend': consistency > 0.7
        }

        return self.H0

    def compute_H1(self, open_sets: List[str]) -> dict:
        """
        Computa H¹ = Ker(δ¹)/Im(δ⁰)
        """
        delta0 = self.compute_delta0(open_sets)
        delta1 = self.compute_delta1(open_sets)

        if SCIPY_AVAILABLE:
            kernel_delta1 = null_space(delta1)
            dim_ker = kernel_delta1.shape[1] if kernel_delta1.size > 0 else delta1.shape[1]
        else:
            U, S, Vh = np.linalg.svd(delta1)
            null_mask = S < self.eps
            dim_ker = np.sum(null_mask) + max(0, delta1.shape[1] - len(S))

        rank_delta0 = np.linalg.matrix_rank(delta0)

        betti_1 = max(0, dim_ker - rank_delta0)

        obstruction = self._compute_obstruction(open_sets)

        self.H1 = {
            'dimension': betti_1,
            'betti_number': betti_1,
            'ker_delta1_dim': dim_ker,
            'im_delta0_dim': rank_delta0,
            'obstruction': obstruction,
            'has_twist': betti_1 > 0 or obstruction > 0.3
        }

        return self.H1

    def _compute_section_consistency(self, sections: List[np.ndarray]) -> float:
        """Calcula a consistência entre seções."""
        if len(sections) < 2:
            return 1.0

        normalized = []
        for s in sections:
            if len(s) > 0:
                s_norm = (s - np.mean(s)) / (np.std(s) + self.eps)
                normalized.append(s_norm)

        if len(normalized) < 2:
            return 1.0

        correlations = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                min_len = min(len(normalized[i]), len(normalized[j]))
                if min_len > 1:
                    corr = np.corrcoef(normalized[i][:min_len],
                                      normalized[j][:min_len])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

        return np.mean(correlations) if correlations else 0.0

    def _compute_obstruction(self, open_sets: List[str]) -> float:
        """Calcula a obstrução real entre seções."""
        sections = [self.sheaf.get_section(U) for U in open_sets]

        if len(sections) < 2:
            return 0.0

        obstructions = []

        for i in range(len(sections) - 1):
            s_i = sections[i]
            s_j = sections[i + 1]

            if len(s_i) > 0 and len(s_j) > 0:
                boundary_diff = np.abs(s_i[-1] - s_j[0]) if len(s_i) > 0 and len(s_j) > 0 else 0

                trend_i = np.mean(np.diff(s_i)) if len(s_i) > 1 else 0
                trend_j = np.mean(np.diff(s_j)) if len(s_j) > 1 else 0
                trend_diff = np.abs(trend_i - trend_j)

                obstructions.append(boundary_diff + trend_diff)

        return np.mean(obstructions) if obstructions else 0.0


class KolmogorovComplexity:
    """
    Módulo 3: Estimativa de Complexidade de Kolmogorov (K(x))

    - K(x) baixo + movimento grande = Artificial/Manipulado (HFT)
    - K(x) alto + movimento grande = Orgânico/Sustentável
    """

    def __init__(self):
        self.eps = 1e-10
        self._complexity_cache = {}

    def price_to_binary(self, prices: np.ndarray) -> bytes:
        """Converte série de preços em sequência binária de movimentos."""
        returns = np.diff(prices)
        binary = (returns > 0).astype(np.uint8)

        binary_str = ''.join(map(str, binary))

        while len(binary_str) % 8 != 0:
            binary_str += '0'

        byte_array = bytearray()
        for i in range(0, len(binary_str), 8):
            byte_array.append(int(binary_str[i:i+8], 2))

        return bytes(byte_array)

    def estimate_K_zlib(self, data: bytes) -> float:
        """Estima K(x) usando compressão zlib."""
        if len(data) == 0:
            return 0.0

        compressed = zlib.compress(data, level=9)
        K = len(compressed) / len(data)

        return K

    def estimate_K_lzma(self, data: bytes) -> float:
        """Estima K(x) usando compressão lzma."""
        if len(data) == 0:
            return 0.0

        try:
            compressed = lzma.compress(data, preset=9)
            K = len(compressed) / len(data)
        except:
            K = self.estimate_K_zlib(data)

        return K

    def estimate_K_bdm(self, binary_sequence: np.ndarray,
                       block_size: int = 4) -> float:
        """Estima K(x) usando Block Decomposition Method (BDM)."""
        n = len(binary_sequence)
        if n == 0:
            return 0.0

        n_blocks = n // block_size
        if n_blocks == 0:
            n_blocks = 1
            block_size = n

        block_counts = defaultdict(int)

        for i in range(n_blocks):
            block = tuple(binary_sequence[i*block_size:(i+1)*block_size])
            block_counts[block] += 1

        total_K = 0.0

        for block, count in block_counts.items():
            unique_bits = len(set(block))
            block_complexity = np.log2(unique_bits + 1) + len(block) * 0.5
            total_K += block_complexity + np.log2(count + 1)

        K_normalized = total_K / n

        return K_normalized

    def compute_complexity(self, prices: np.ndarray) -> dict:
        """Computa todas as métricas de complexidade de Kolmogorov."""
        binary_data = self.price_to_binary(prices)
        binary_array = np.array([(prices[i+1] > prices[i]) for i in range(len(prices)-1)], dtype=int)

        K_zlib = self.estimate_K_zlib(binary_data)
        K_lzma = self.estimate_K_lzma(binary_data)
        K_bdm = self.estimate_K_bdm(binary_array)

        K_combined = 0.3 * K_zlib + 0.5 * K_lzma + 0.2 * K_bdm

        K_scaled = K_combined * 8
        M_solomonoff = 2 ** (-K_scaled)

        return {
            'K_zlib': K_zlib,
            'K_lzma': K_lzma,
            'K_bdm': K_bdm,
            'K_combined': K_combined,
            'K_scaled': K_scaled,
            'M_solomonoff': M_solomonoff,
            'is_algorithmic': K_combined < 0.5,
            'is_organic': K_combined > 0.7
        }

    def analyze_movement(self, prices: np.ndarray,
                         window: int = 50) -> dict:
        """Analisa se um movimento de preço é artificial ou orgânico."""
        if len(prices) < window:
            window = len(prices)

        recent_prices = prices[-window:]

        total_return = (recent_prices[-1] - recent_prices[0]) / (recent_prices[0] + self.eps)
        volatility = np.std(np.diff(recent_prices) / recent_prices[:-1])

        complexity = self.compute_complexity(recent_prices)
        K = complexity['K_combined']

        magnitude = np.abs(total_return)

        if magnitude > 0.01:
            if K < 0.4:
                classification = "ARTIFICIAL_MANIPULATED"
                sustainability = "FRAGIL"
                confidence = 1 - K
            elif K > 0.7:
                classification = "ORGANIC_SUSTAINABLE"
                sustainability = "ROBUSTO"
                confidence = K
            else:
                classification = "MIXED"
                sustainability = "INCERTO"
                confidence = 0.5
        else:
            classification = "NOISE"
            sustainability = "IRRELEVANTE"
            confidence = 0.0

        return {
            'classification': classification,
            'sustainability': sustainability,
            'confidence': confidence,
            'K': K,
            'M_solomonoff': complexity['M_solomonoff'],
            'total_return': total_return,
            'volatility': volatility,
            **complexity
        }


class ForgetfulFunctor:
    """
    Módulo 4: O Functor de Esquecimento (Gatilho de Decisão)

    Gatilho de Compra:
    1. Cohomologia: H¹ colapsa para 0
    2. Algorítmica: K(x) diminui
    3. Topos: Existe morfismo isomorfo com Bull Run histórico
    """

    def __init__(self):
        self.eps = 1e-10

        self.historical_patterns = {
            'bull_run': {
                'H1_range': (0, 0.5),
                'K_range': (0.3, 0.6),
                'trend': 'up'
            },
            'bear_market': {
                'H1_range': (0.5, 2.0),
                'K_range': (0.2, 0.5),
                'trend': 'down'
            },
            'accumulation': {
                'H1_range': (0, 0.3),
                'K_range': (0.6, 0.9),
                'trend': 'neutral'
            },
            'distribution': {
                'H1_range': (0.3, 1.0),
                'K_range': (0.4, 0.7),
                'trend': 'neutral'
            }
        }

    def check_morphism(self, current_state: dict,
                       pattern_name: str) -> Tuple[bool, float]:
        """Verifica se existe morfismo isomorfo com padrão histórico."""
        if pattern_name not in self.historical_patterns:
            return False, 0.0

        pattern = self.historical_patterns[pattern_name]

        H1 = current_state.get('H1_dimension', 0)
        K = current_state.get('K_combined', 0.5)

        H1_match = pattern['H1_range'][0] <= H1 <= pattern['H1_range'][1]
        K_match = pattern['K_range'][0] <= K <= pattern['K_range'][1]

        H1_center = (pattern['H1_range'][0] + pattern['H1_range'][1]) / 2
        K_center = (pattern['K_range'][0] + pattern['K_range'][1]) / 2

        H1_dist = np.abs(H1 - H1_center) / (pattern['H1_range'][1] - pattern['H1_range'][0] + self.eps)
        K_dist = np.abs(K - K_center) / (pattern['K_range'][1] - pattern['K_range'][0] + self.eps)

        similarity = 1 - (H1_dist + K_dist) / 2
        similarity = max(0, min(1, similarity))

        is_isomorphic = H1_match and K_match and similarity > 0.5

        return is_isomorphic, similarity

    def apply_functor(self, cohomology_H0: dict,
                      cohomology_H1: dict,
                      complexity: dict,
                      prices: np.ndarray) -> dict:
        """Aplica o Functor de Esquecimento para gerar sinal de trading."""
        H1_dim = cohomology_H1.get('dimension', 0)
        H1_obstruction = cohomology_H1.get('obstruction', 0)
        H0_consistency = cohomology_H0.get('consistency', 0)
        K = complexity.get('K_combined', 0.5)

        current_state = {
            'H1_dimension': H1_dim,
            'K_combined': K,
            'H0_consistency': H0_consistency
        }

        bull_iso, bull_sim = self.check_morphism(current_state, 'bull_run')
        bear_iso, bear_sim = self.check_morphism(current_state, 'bear_market')

        if len(prices) > 10:
            recent_trend = (prices[-1] - prices[-10]) / (prices[-10] + self.eps)
        else:
            recent_trend = 0

        # Trend como principal discriminador - espelhado para balanceamento
        buy_conditions = {
            'H1_low': H1_obstruction < 0.1,
            'K_moderate': K < 4.0,
            'bull_morphism': bull_iso,
            'trend_up': recent_trend > 0.0008,  # Trend positivo
            'positive_consistency': H0_consistency > 0
        }

        sell_conditions = {
            'H1_high': H1_obstruction >= 0.1,
            'K_moderate': K < 4.0,  # Mesma condicao
            'bear_morphism': bear_iso,
            'trend_down': recent_trend < -0.0008,  # Trend negativo
            'negative_consistency': H0_consistency <= 0
        }

        buy_score = sum(buy_conditions.values())
        sell_score = sum(sell_conditions.values())

        if buy_score >= 3 and buy_score > sell_score:
            signal = 1
            signal_name = "COMPRA (Functor -> R+)"
            confidence = buy_score / 5
            reasons = [k for k, v in buy_conditions.items() if v]
        elif sell_score >= 3 and sell_score > buy_score:
            signal = -1
            signal_name = "VENDA (Functor -> R-)"
            confidence = sell_score / 5
            reasons = [k for k, v in sell_conditions.items() if v]
        else:
            signal = 0
            signal_name = "NEUTRO"
            confidence = 0.0
            reasons = []

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'reasons': reasons,
            'buy_conditions': buy_conditions,
            'sell_conditions': sell_conditions,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'bull_similarity': bull_sim,
            'bear_similarity': bear_sim
        }


class SintetizadorToposGrothendieckKolmogorov:
    """
    Implementação completa do Sintetizador de Topos Grothendieck-Kolmogorov (STGK)

    Módulos:
    1. A Estrutura do Topos (Definição da Categoria) - Sheaf
    2. Cálculo da Cohomologia de Čech (H^n(U, F))
    3. Estimativa de Complexidade de Kolmogorov (K(x))
    4. O Functor de Esquecimento (Gatilho de Decisão)
    5. Output e Visualização

    Valor de Retorno: [Betti_Number_H1, Kolmogorov_Complexity_Index, Solomonoff_Prior]
    """

    def __init__(self,
                 timeframes: List[int] = None,
                 complexity_window: int = 100):
        """
        Inicializa o Sintetizador de Topos.

        Parâmetros:
        -----------
        timeframes : List[int]
            Lista de timeframes para análise (em minutos)
        complexity_window : int
            Janela para cálculo de complexidade de Kolmogorov
        """
        if timeframes is None:
            timeframes = [1, 5, 15, 60]
        self.timeframes = timeframes
        self.complexity_window = complexity_window

        self.sheaf = Sheaf()
        self.cohomology = None
        self.kolmogorov = KolmogorovComplexity()
        self.functor = ForgetfulFunctor()

        self._H1_history = []
        self._K_history = []

    def _resample_to_timeframe(self, prices: np.ndarray,
                                tf: int, base_tf: int = 1) -> np.ndarray:
        """Resample preços para um timeframe específico."""
        ratio = tf // base_tf
        if ratio <= 1:
            return prices

        n = len(prices)
        n_candles = n // ratio

        if n_candles == 0:
            return prices

        resampled = []
        for i in range(n_candles):
            candle = prices[i*ratio:(i+1)*ratio]
            resampled.append(candle[-1])

        return np.array(resampled)

    def _build_sheaf_structure(self, prices: np.ndarray):
        """Constrói a estrutura de Feixe sobre os diferentes timeframes."""
        self.sheaf = Sheaf()

        for tf in self.timeframes:
            section = self._resample_to_timeframe(prices, tf)
            self.sheaf.add_open_set(f"TF_{tf}M", section)

        for i in range(len(self.timeframes) - 1):
            tf_large = self.timeframes[i + 1]
            tf_small = self.timeframes[i]

            ratio = tf_large // tf_small

            def create_restriction(r):
                return lambda x: x[::r] if len(x) > r else x

            self.sheaf.add_restriction(
                f"TF_{tf_large}M",
                f"TF_{tf_small}M",
                create_restriction(ratio)
            )

        self.cohomology = CechCohomology(self.sheaf)

    def analyze(self, prices: np.ndarray) -> dict:
        """
        Execução completa do Sintetizador de Topos Grothendieck-Kolmogorov.

        Retorna: [Betti_Number_H1, Kolmogorov_Complexity_Index, Solomonoff_Prior]
        """
        n = len(prices)

        if n < 20:
            return self._empty_result()

        # 1. Construir estrutura de Feixe (Topos)
        self._build_sheaf_structure(prices)

        open_sets = [f"TF_{tf}M" for tf in self.timeframes]

        gluing_ok, gluing_error = self.sheaf.check_gluing_axiom(
            open_sets[0], open_sets
        )

        # 2. Calcular Cohomologia de Čech
        H0 = self.cohomology.compute_H0(open_sets)
        H1 = self.cohomology.compute_H1(open_sets)

        self._H1_history.append(H1['dimension'])
        if len(self._H1_history) > 50:
            self._H1_history = self._H1_history[-50:]

        H1_spike = False
        if len(self._H1_history) > 3:
            recent_H1 = self._H1_history[-3:]
            H1_spike = H1['dimension'] > 0 and recent_H1[-1] > max(recent_H1[:-1])

        # 3. Calcular Complexidade de Kolmogorov
        window = min(self.complexity_window, n)
        complexity = self.kolmogorov.compute_complexity(prices[-window:])
        movement = self.kolmogorov.analyze_movement(prices, window)

        self._K_history.append(complexity['K_combined'])
        if len(self._K_history) > 50:
            self._K_history = self._K_history[-50:]

        # 4. Aplicar Functor de Esquecimento
        signal_result = self.functor.apply_functor(H0, H1, complexity, prices)

        # 5. Verificar comutatividade do diagrama
        diagram_commutes = self._check_diagram_commutativity(prices)

        # Vetor de Retorno Principal
        output_vector = [
            H1['betti_number'],
            complexity['K_combined'],
            complexity['M_solomonoff']
        ]

        return {
            'output_vector': output_vector,
            'Betti_Number_H1': output_vector[0],
            'Kolmogorov_Complexity_Index': output_vector[1],
            'Solomonoff_Prior': output_vector[2],

            'H0': H0,
            'H1': H1,
            'H1_spike': H1_spike,
            'has_obstruction': H1['has_twist'],
            'unified_trend': H0['unified_trend'],

            'complexity': complexity,
            'movement_analysis': movement,
            'is_artificial': movement['classification'] == 'ARTIFICIAL_MANIPULATED',
            'is_organic': movement['classification'] == 'ORGANIC_SUSTAINABLE',

            'gluing_error': gluing_error,
            'diagram_commutes': diagram_commutes,

            'signal': signal_result['signal'],
            'signal_name': signal_result['signal_name'],
            'confidence': signal_result['confidence'],
            'reasons': signal_result['reasons'],

            'n_observations': n,
            'current_price': prices[-1],
            'timeframes_used': self.timeframes
        }

    def _check_diagram_commutativity(self, prices: np.ndarray) -> bool:
        """Verifica se o diagrama de timeframes é comutativo."""
        if len(self.timeframes) < 3:
            return True

        tf1, tf2, tf3 = self.timeframes[0], self.timeframes[1], self.timeframes[2]

        direct = self._resample_to_timeframe(prices, tf3, tf1)

        via_tf2 = self._resample_to_timeframe(prices, tf2, tf1)
        indirect = self._resample_to_timeframe(via_tf2, tf3, tf2)

        min_len = min(len(direct), len(indirect))
        if min_len == 0:
            return True

        diff = np.mean(np.abs(direct[:min_len] - indirect[:min_len]))

        return diff < 0.001

    def _empty_result(self) -> dict:
        """Retorna resultado vazio quando não há dados suficientes"""
        return {
            'output_vector': [0, 0.0, 0.0],
            'Betti_Number_H1': 0,
            'Kolmogorov_Complexity_Index': 0.0,
            'Solomonoff_Prior': 0.0,
            'H0': {'dimension': 0, 'consistency': 0, 'unified_trend': False},
            'H1': {'dimension': 0, 'betti_number': 0, 'obstruction': 0, 'has_twist': False},
            'H1_spike': False,
            'has_obstruction': False,
            'unified_trend': False,
            'complexity': {'K_combined': 0, 'M_solomonoff': 0},
            'movement_analysis': {'classification': 'UNKNOWN', 'sustainability': 'N/A'},
            'is_artificial': False,
            'is_organic': False,
            'gluing_error': 0,
            'diagram_commutes': True,
            'signal': 0,
            'signal_name': 'HOLD',
            'confidence': 0.0,
            'reasons': ['Dados insuficientes'],
            'n_observations': 0,
            'current_price': 0.0,
            'timeframes_used': []
        }

    def get_signal(self, prices: np.ndarray) -> int:
        """Retorna sinal simplificado."""
        result = self.analyze(prices)
        return result['signal']


def plot_stgk_analysis(prices: np.ndarray, save_path: str = None):
    """Visualização do STGK."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib não disponível")
        return None

    stgk = SintetizadorToposGrothendieckKolmogorov(
        timeframes=[1, 5, 15],
        complexity_window=50
    )
    result = stgk.analyze(prices)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.5, 1])

    time = np.arange(len(prices))

    # Plot 1: Diagrama Comutativo
    ax1 = fig.add_subplot(gs[0, 0])

    positions = {
        'TF_1M': (0.2, 0.5),
        'TF_5M': (0.5, 0.8),
        'TF_15M': (0.8, 0.5)
    }

    if not result['diagram_commutes']:
        ax1.add_patch(mpatches.Rectangle((0, 0), 1, 1, alpha=0.2, color='red'))
        ax1.text(0.5, 0.1, 'NAO COMUTA - ARBITRAGEM', ha='center',
                fontsize=12, color='red', weight='bold')
    else:
        ax1.add_patch(mpatches.Rectangle((0, 0), 1, 1, alpha=0.1, color='green'))
        ax1.text(0.5, 0.1, 'DIAGRAMA COMUTA', ha='center',
                fontsize=10, color='green')

    for name, pos in positions.items():
        circle = mpatches.Circle(pos, 0.08, color='lightblue', ec='navy', lw=2)
        ax1.add_patch(circle)
        ax1.text(pos[0], pos[1], name.replace('TF_', '').replace('M', ''),
                ha='center', va='center', fontsize=10, weight='bold')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Diagrama Comutativo da Categoria C', fontsize=12)

    # Plot 2: Histograma de Complexidade
    ax2 = fig.add_subplot(gs[0, 1])

    window_size = 20
    n_windows = len(prices) // window_size
    complexities = []

    for i in range(n_windows):
        w = prices[i*window_size:(i+1)*window_size]
        K = stgk.kolmogorov.compute_complexity(w)['K_combined']
        complexities.append(K)

    colors = ['green' if K < 0.5 else 'orange' if K < 0.7 else 'red'
              for K in complexities]

    ax2.bar(range(len(complexities)), complexities, color=colors, alpha=0.7)
    ax2.axhline(y=0.5, color='green', linestyle='--', label='Easy (K<0.5)')
    ax2.axhline(y=0.7, color='red', linestyle='--', label='Hard (K>0.7)')

    ax2.set_xlabel('Janela Temporal')
    ax2.set_ylabel('Complexidade K(x)')
    ax2.set_title('Histograma de Complexidade Algoritmica', fontsize=12)
    ax2.legend(fontsize=8)

    # Plot 3: Preço com Obstruções
    ax3 = fig.add_subplot(gs[1, 0])

    ax3.plot(time, prices, 'b-', linewidth=1.5, label='Preco')

    if result['has_obstruction']:
        ax3.axvspan(0, len(prices), alpha=0.15, color='red')
        ax3.text(0.5, 0.95, 'H1 != 0 (TORCAO DETECTADA)', transform=ax3.transAxes,
                ha='center', va='top', fontsize=11, color='red', weight='bold')
    else:
        ax3.axvspan(0, len(prices), alpha=0.1, color='green')
        ax3.text(0.5, 0.95, 'H1 = 0 (CAMINHO LIVRE)', transform=ax3.transAxes,
                ha='center', va='top', fontsize=10, color='green')

    if result['signal'] == 1:
        ax3.scatter([time[-1]], [prices[-1]], c='green', s=300, marker='^', zorder=5)
    elif result['signal'] == -1:
        ax3.scatter([time[-1]], [prices[-1]], c='red', s=300, marker='v', zorder=5)

    ax3.set_xlabel('Tempo')
    ax3.set_ylabel('Preco')
    ax3.set_title(f'Preco + Cohomologia | B1 = {result["Betti_Number_H1"]}', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Gauge de Complexidade
    ax4 = fig.add_subplot(gs[1, 1])

    theta = np.linspace(0, np.pi, 100)
    r = 1

    ax4.plot(r * np.cos(theta), r * np.sin(theta), 'k-', lw=2)

    ax4.fill_between(np.cos(theta[:33]), 0, np.sin(theta[:33]), alpha=0.3, color='red')
    ax4.fill_between(np.cos(theta[33:66]), 0, np.sin(theta[33:66]), alpha=0.3, color='yellow')
    ax4.fill_between(np.cos(theta[66:]), 0, np.sin(theta[66:]), alpha=0.3, color='green')

    K = result['Kolmogorov_Complexity_Index']
    angle = np.pi * K
    ax4.arrow(0, 0, 0.7*np.cos(angle), 0.7*np.sin(angle),
             head_width=0.1, head_length=0.1, fc='navy', ec='navy')

    ax4.text(-0.8, -0.2, 'ARTIFICIAL\n(HFT)', ha='center', fontsize=9, color='red')
    ax4.text(0, -0.2, 'MISTO', ha='center', fontsize=9, color='orange')
    ax4.text(0.8, -0.2, 'ORGANICO\n(Humanos)', ha='center', fontsize=9, color='green')

    ax4.text(0, 1.15, f'K(x) = {K:.3f}', ha='center', fontsize=11, weight='bold')

    ax4.set_xlim(-1.2, 1.2)
    ax4.set_ylim(-0.7, 1.3)
    ax4.set_aspect('equal')
    ax4.axis('off')
    ax4.set_title('Gauge de Complexidade Algoritmica', fontsize=12)

    # Plot 5: Vetor de Saída
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    output = result['output_vector']

    table_data = [
        ['Betti_Number_H1', f'{output[0]}',
         'Dimensao do grupo de cohomologia H1'],
        ['Kolmogorov_Complexity_Index', f'{output[1]:.4f}',
         'Complexidade algoritmica normalizada'],
        ['Solomonoff_Prior', f'{output[2]:.2e}',
         'Probabilidade a priori M(x)']
    ]

    table = ax5.table(cellText=table_data,
                     colLabels=['Variavel', 'Valor', 'Descricao'],
                     loc='center',
                     cellLoc='center',
                     colWidths=[0.25, 0.15, 0.5])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for i in range(3):
        table[(0, i)].set_facecolor('navy')
        table[(0, i)].set_text_props(color='white', weight='bold')

    ax5.set_title('Vetor de Retorno: [Betti_Number_H1, Kolmogorov_Complexity_Index, Solomonoff_Prior]',
                 fontsize=12, pad=20)

    summary = (
        f"STGK | Sinal: {result['signal_name']} | "
        f"H1={'OBSTRUCAO' if result['has_obstruction'] else 'LIVRE'} | "
        f"K={result['Kolmogorov_Complexity_Index']:.3f}"
    )

    color = 'green' if result['signal'] == 1 else 'red' if result['signal'] == -1 else 'lightblue'
    fig.text(0.5, 0.01, summary, fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("SINTETIZADOR DE TOPOS GROTHENDIECK-KOLMOGOROV (STGK)")
    print("Medalha Fields / Fundamentos da Matematica")
    print("=" * 70)

    np.random.seed(42)

    # Dados simulados
    organic = 1.1 + 0.0002 * np.cumsum(np.random.randn(100))
    artificial = organic[-1] + 0.0005 * np.cumsum(np.ones(50) + 0.1 * np.random.randn(50))
    twist = artificial[-1] + 0.0003 * np.cumsum(np.sign(np.sin(np.linspace(0, 4*np.pi, 50))))

    prices = np.concatenate([organic, artificial, twist])

    print(f"\nDados: {len(prices)} pontos")
    print(f"Preco: {prices[0]:.5f} -> {prices[-1]:.5f}")

    stgk = SintetizadorToposGrothendieckKolmogorov(
        timeframes=[1, 5, 15],
        complexity_window=50
    )

    print("\n" + "-" * 40)
    print("Executando analise STGK...")
    print("-" * 40)

    result = stgk.analyze(prices)

    print("\n VETOR DE RETORNO:")
    print(f"   [Betti_Number_H1, Kolmogorov_Complexity_Index, Solomonoff_Prior]")
    output = result['output_vector']
    print(f"   [{output[0]}, {output[1]:.4f}, {output[2]:.2e}]")

    print("\n COHOMOLOGIA DE CECH:")
    print(f"   H0 (Secoes Globais):")
    print(f"      Dimensao: {result['H0']['dimension']}")
    print(f"      Consistencia: {result['H0']['consistency']:.4f}")
    print(f"      Tendencia Unificada: {'SIM' if result['unified_trend'] else 'NAO'}")
    print(f"   H1 (Obstrucoes):")
    print(f"      Numero de Betti B1: {result['Betti_Number_H1']}")
    print(f"      Obstrucao: {result['H1']['obstruction']:.4f}")
    print(f"      Torcao: {'DETECTADA' if result['has_obstruction'] else 'Nenhuma'}")

    print("\n COMPLEXIDADE DE KOLMOGOROV:")
    print(f"   K(x) combinado: {result['Kolmogorov_Complexity_Index']:.4f}")
    print(f"   K_zlib: {result['complexity']['K_zlib']:.4f}")
    print(f"   K_lzma: {result['complexity']['K_lzma']:.4f}")
    print(f"   Prior de Solomonoff M(x): {result['Solomonoff_Prior']:.2e}")

    print("\n ANALISE DE MOVIMENTO:")
    mov = result['movement_analysis']
    print(f"   Classificacao: {mov['classification']}")
    print(f"   Sustentabilidade: {mov['sustainability']}")
    print(f"   Retorno Total: {mov['total_return']:.4%}")

    print("\n SINAL:")
    print(f"   {result['signal_name']}")
    print(f"   Confianca: {result['confidence']:.0%}")
    print(f"   Razoes: {', '.join(result['reasons']) if result['reasons'] else 'Nenhuma'}")

    print("\n" + "=" * 70)
    if result['signal'] == 1:
        print("COMPRA!")
        print("   H1 colapsou, caminho livre, morfismo isomorfo com Bull Run")
    elif result['signal'] == -1:
        print("VENDA!")
        print("   Obstrucao cohomologica detectada ou movimento artificial")
    elif result['has_obstruction']:
        print("OBSTRUCAO COHOMOLOGICA!")
        print("   O tecido do mercado esta se rasgando. Correcao iminente.")
    else:
        print("NEUTRO")
        print(f"   K(x) = {result['Kolmogorov_Complexity_Index']:.3f}")
    print("=" * 70)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print("\nGerando visualizacao...")
        plot_stgk_analysis(prices, '/tmp/stgk_analysis.png')
        print("Visualizacao salva: /tmp/stgk_analysis.png")
        plt.close()
    except Exception as e:
        print(f"Erro na visualizacao: {e}")

    print("\n Teste do STGK concluido com sucesso!")
