"""
================================================================================
BETTI-PERSISTENCE HOMOLOGY SCANNER (B-PHS)
Indicador de Forex baseado em Topologia Algebrica
================================================================================

Este indicador nao ve numeros, medias ou geometria curva. Ele ve a FORMA dos dados.
Utiliza a Teoria da Homologia Persistente para analisar a estrutura topologica da
nuvem de pontos do mercado em um espaco de fase multidimensional.

Em media volatilidade, o mercado forma "loops" (ciclos) estaveis. A Topologia
Algebrica e a unica ferramenta capaz de quantificar a "vida util" e a "robustez"
desses loops ANTES que eles acontecam no grafico de velas.

Arquitetura Matematica:
1. Reconstrucao do Espaco de Fase (Takens' Embedding)
2. Complexo Simplicial (Vietoris-Rips Filtration)
3. Numeros de Betti e Homologia (H_k)
4. Diagramas de Persistencia e Paisagem de Persistencia
5. Entropia Topologica Persistente

O Santo Graal e B1 (Betti-1): Loops ou Tuneis (Ciclos de mercado)
================================================================================
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
from typing import Tuple, Optional, List, Dict, Set
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

class SignalType(Enum):
    """Tipos de sinais de trading"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    BLOCKED = "BLOCKED"  # Bloqueado por crash do Betti


class TopologyRegime(Enum):
    """Regime topologico do mercado"""
    CYCLIC = "CYCLIC"           # B1 alto - mercado ciclico
    LINEAR = "LINEAR"           # B1 baixo - tendencia linear
    CHAOTIC = "CHAOTIC"         # Entropia alta - caos
    TRANSITIONAL = "TRANSITIONAL"  # Em transicao


@dataclass
class PersistencePair:
    """Par de persistencia (nascimento, morte)"""
    birth: float
    death: float
    dimension: int  # 0 = componente, 1 = loop, 2 = vazio

    @property
    def persistence(self) -> float:
        """Vida util da caracteristica topologica"""
        return self.death - self.birth if self.death != np.inf else np.inf

    @property
    def midlife(self) -> float:
        """Ponto medio da vida"""
        if self.death == np.inf:
            return self.birth + 1.0
        return (self.birth + self.death) / 2


@dataclass
class PersistenceDiagram:
    """Diagrama de Persistencia completo"""
    pairs_dim0: List[PersistencePair]  # B0 - componentes conectados
    pairs_dim1: List[PersistencePair]  # B1 - loops (SANTO GRAAL)
    pairs_dim2: List[PersistencePair]  # B2 - vazios

    def get_betti_0(self, epsilon: float) -> int:
        """Numero de componentes conectados em epsilon"""
        return sum(1 for p in self.pairs_dim0
                   if p.birth <= epsilon < p.death)

    def get_betti_1(self, epsilon: float) -> int:
        """Numero de loops em epsilon"""
        return sum(1 for p in self.pairs_dim1
                   if p.birth <= epsilon < p.death)

    def get_max_persistence_dim1(self) -> float:
        """Persistencia maxima de loops"""
        if not self.pairs_dim1:
            return 0.0
        finite_pers = [p.persistence for p in self.pairs_dim1
                       if p.persistence != np.inf]
        return max(finite_pers) if finite_pers else 0.0


@dataclass
class PersistenceLandscape:
    """Paisagem de Persistencia L_k(t)"""
    landscapes: np.ndarray  # Array [k, t] de funcoes de paisagem
    t_values: np.ndarray    # Valores de t (epsilon)
    max_k: int              # Numero de paisagens

    def get_integral(self, k: int = 0) -> float:
        """Integral da k-esima paisagem (norma L1)"""
        if k >= self.max_k:
            return 0.0
        return np.trapz(self.landscapes[k], self.t_values)


@dataclass
class TopologicalAnalysis:
    """Analise topologica completa"""
    persistence_diagram: PersistenceDiagram
    persistence_landscape: PersistenceLandscape
    betti_numbers: Dict[int, int]  # B_k no epsilon otimo
    topological_entropy: float
    dominant_loop_persistence: float
    loop_centroid: Optional[np.ndarray]
    regime: TopologyRegime


@dataclass
class BPHSSignal:
    """Sinal gerado pelo B-PHS"""
    signal_type: SignalType
    regime: TopologyRegime
    confidence: float
    betti_0: int              # Componentes conectados
    betti_1: int              # Loops (Santo Graal)
    betti_2: int              # Vazios
    max_loop_persistence: float
    topological_entropy: float
    distance_to_centroid: float
    position_in_loop: str     # "TOP", "BOTTOM", "MIDDLE"
    reason: str
    timestamp: str


# ==============================================================================
# TAKENS' EMBEDDING - RECONSTRUCAO DO ESPACO DE FASE
# ==============================================================================

class TakensEmbedding:
    """
    Reconstrucao do Espaco de Fase via Teorema de Takens

    O primeiro passo e transformar a serie temporal linear do EURUSD em uma
    nuvem de pontos tridimensional (ou n-dimensional).

    V_t = [x(t), x(t-tau), x(t-2*tau), ..., x(t-(m-1)*tau)]

    Onde:
    - m: Dimensao de imersao (Embedding Dimension). Para EURUSD em media vol, m=3 ou m=4
    - tau: Atraso temporal (Time Delay), calculado via Informacao Mutua Minima

    Isso gera uma "nuvem" de pontos que representa a dinamica do sistema.
    """

    def __init__(self,
                 embedding_dim: int = 3,
                 time_delay: Optional[int] = None,
                 auto_delay: bool = True):
        """
        Args:
            embedding_dim: Dimensao de imersao (m)
            time_delay: Atraso temporal (tau). Se None, calcula automaticamente.
            auto_delay: Se True, calcula tau via Informacao Mutua Minima
        """
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.auto_delay = auto_delay

    def _mutual_information(self, x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
        """
        Calcula Informacao Mutua entre duas series

        I(X;Y) = Sum p(x,y) log(p(x,y) / (p(x)p(y)))
        """
        # Histograma conjunto
        hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins, density=True)

        # Histogramas marginais
        p_x = np.sum(hist_2d, axis=1)
        p_y = np.sum(hist_2d, axis=0)

        # Normaliza
        p_x = p_x / (np.sum(p_x) + 1e-10)
        p_y = p_y / (np.sum(p_y) + 1e-10)
        hist_2d = hist_2d / (np.sum(hist_2d) + 1e-10)

        # Informacao mutua
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if hist_2d[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (p_x[i] * p_y[j] + 1e-10) + 1e-10)

        return mi

    def _find_optimal_delay(self, signal: np.ndarray, max_delay: int = 50) -> int:
        """
        Encontra o atraso otimo tau via primeiro minimo da Informacao Mutua
        """
        n = len(signal)
        max_delay = min(max_delay, n // 4)

        mi_values = []
        for delay in range(1, max_delay):
            x = signal[:-delay]
            y = signal[delay:]
            mi = self._mutual_information(x, y)
            mi_values.append(mi)

        # Encontra primeiro minimo local
        mi_values = np.array(mi_values)
        for i in range(1, len(mi_values) - 1):
            if mi_values[i] < mi_values[i-1] and mi_values[i] < mi_values[i+1]:
                return i + 1

        # Se nao encontrar minimo, usa o ponto de maior queda
        if len(mi_values) > 1:
            diffs = np.diff(mi_values)
            return np.argmin(diffs) + 1

        return 1

    def embed(self, signal: np.ndarray) -> np.ndarray:
        """
        Reconstroi o espaco de fase via Takens' Embedding

        Args:
            signal: Serie temporal unidimensional

        Returns:
            Matriz [n_points, embedding_dim] de vetores de atraso
        """
        n = len(signal)

        # Determina tau
        if self.auto_delay and self.time_delay is None:
            self.time_delay = self._find_optimal_delay(signal)
        elif self.time_delay is None:
            self.time_delay = 1

        tau = self.time_delay
        m = self.embedding_dim

        # Numero de pontos no espaco de fase
        n_points = n - (m - 1) * tau

        if n_points <= 0:
            raise ValueError(f"Dados insuficientes para embedding: n={n}, m={m}, tau={tau}")

        # Constroi matriz de embedding
        # V_t = [x(t), x(t-tau), x(t-2*tau), ..., x(t-(m-1)*tau)]
        embedded = np.zeros((n_points, m))

        for i in range(n_points):
            for j in range(m):
                embedded[i, j] = signal[i + j * tau]

        return embedded

    def get_current_point(self, signal: np.ndarray) -> np.ndarray:
        """Retorna o ponto atual V_t no espaco de fase"""
        embedded = self.embed(signal)
        return embedded[-1]


# ==============================================================================
# VIETORIS-RIPS FILTRATION - COMPLEXO SIMPLICIAL
# ==============================================================================

class VietorisRipsComplex:
    """
    Complexo Simplicial de Vietoris-Rips

    Para entender a forma da nuvem de pontos, conectamos pontos proximos.
    Imagine esferas de raio epsilon crescendo ao redor de cada ponto:

    - Quando duas esferas se tocam, cria-se uma aresta (1-simplex)
    - Quando tres se tocam, cria-se um triangulo (2-simplex)
    - Isso forma o Complexo de Vietoris-Rips

    Voce deve variar epsilon de 0 ate um limite e monitorar como a conectividade muda.
    Isso e a Filtracao.

    Complexidade: O(n^3) ou mais
    """

    def __init__(self, max_dimension: int = 2, max_epsilon: Optional[float] = None):
        """
        Args:
            max_dimension: Dimensao maxima dos simplices a computar
            max_epsilon: Epsilon maximo para filtracao
        """
        self.max_dimension = max_dimension
        self.max_epsilon = max_epsilon

    def _compute_distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """Computa matriz de distancias entre todos os pontos"""
        return squareform(pdist(points))

    def build_filtration(self,
                        points: np.ndarray,
                        n_steps: int = 50) -> List[Tuple[float, Set[Tuple]]]:
        """
        Constroi a filtracao de Vietoris-Rips

        Args:
            points: Nuvem de pontos [n, dim]
            n_steps: Numero de passos de epsilon

        Returns:
            Lista de (epsilon, simplices) para cada passo
        """
        n = len(points)
        dist_matrix = self._compute_distance_matrix(points)

        # Define range de epsilon
        max_dist = np.max(dist_matrix)
        if self.max_epsilon is not None:
            max_dist = min(max_dist, self.max_epsilon)

        epsilon_values = np.linspace(0, max_dist, n_steps)

        filtration = []

        for epsilon in epsilon_values:
            simplices = set()

            # 0-simplices (vertices) - sempre presentes
            for i in range(n):
                simplices.add((i,))

            # 1-simplices (arestas)
            for i in range(n):
                for j in range(i + 1, n):
                    if dist_matrix[i, j] <= epsilon:
                        simplices.add(tuple(sorted([i, j])))

            # 2-simplices (triangulos) se max_dimension >= 2
            if self.max_dimension >= 2:
                for i in range(n):
                    for j in range(i + 1, n):
                        for k in range(j + 1, n):
                            if (dist_matrix[i, j] <= epsilon and
                                dist_matrix[i, k] <= epsilon and
                                dist_matrix[j, k] <= epsilon):
                                simplices.add(tuple(sorted([i, j, k])))

            filtration.append((epsilon, simplices))

        return filtration


# ==============================================================================
# HOMOLOGIA PERSISTENTE - NUMEROS DE BETTI
# ==============================================================================

class PersistentHomology:
    """
    Calculo de Homologia Persistente

    A medida que epsilon cresce, "buracos" topologicos nascem e morrem na estrutura:

    - B0 (Betti-0): Componentes conectados (ilhas de dados)
    - B1 (Betti-1): Loops ou Tuneis (Ciclos de mercado) - SANTO GRAAL
    - B2 (Betti-2): Vazios (Cavidades tridimensionais)

    Para nos, o Santo Graal e o B1. Um valor alto e persistente de B1 significa
    que o mercado esta preso em um ciclo recorrente forte (caracteristica de
    media volatilidade).
    """

    def __init__(self, max_dimension: int = 2):
        """
        Args:
            max_dimension: Dimensao maxima de homologia a computar
        """
        self.max_dimension = max_dimension

    def _compute_betti_0(self,
                        points: np.ndarray,
                        dist_matrix: np.ndarray,
                        epsilon: float) -> Tuple[int, List[Set[int]]]:
        """
        Computa B0 (componentes conectados) via Union-Find
        """
        n = len(points)

        # Constroi grafo de adjacencia
        adj_matrix = (dist_matrix <= epsilon).astype(int)
        np.fill_diagonal(adj_matrix, 0)

        # Encontra componentes conectados
        n_components, labels = connected_components(
            csr_matrix(adj_matrix), directed=False
        )

        # Agrupa pontos por componente
        components = [set() for _ in range(n_components)]
        for i, label in enumerate(labels):
            components[label].add(i)

        return n_components, components

    def _compute_betti_1_approximate(self,
                                     points: np.ndarray,
                                     dist_matrix: np.ndarray,
                                     epsilon: float) -> int:
        """
        Computa B1 (loops) via aproximacao de Euler

        Para um complexo simplicial:
        chi = V - E + F = B0 - B1 + B2

        Em baixa dimensao e sem B2 significativo:
        B1 ~= E - V + B0
        """
        n = len(points)

        # Conta vertices (V)
        V = n

        # Conta arestas (E)
        adj_matrix = dist_matrix <= epsilon
        np.fill_diagonal(adj_matrix, False)
        E = np.sum(adj_matrix) // 2

        # Conta triangulos aproximado (F)
        F = 0
        if epsilon > 0:
            for i in range(n):
                neighbors_i = set(np.where(adj_matrix[i])[0])
                for j in neighbors_i:
                    if j > i:
                        neighbors_j = set(np.where(adj_matrix[j])[0])
                        common = neighbors_i & neighbors_j
                        F += len([k for k in common if k > j])

        # B0
        beta_0, _ = self._compute_betti_0(points, dist_matrix, epsilon)

        # Caracteristica de Euler: chi = V - E + F
        euler = V - E + F

        # B1 = B0 - chi + B2 (assumindo B2 ~= 0 para dados de mercado)
        beta_1 = max(0, beta_0 - euler)

        return beta_1

    def compute_persistence(self,
                           points: np.ndarray,
                           n_steps: int = 100) -> PersistenceDiagram:
        """
        Computa o Diagrama de Persistencia completo

        Args:
            points: Nuvem de pontos [n, dim]
            n_steps: Numero de passos de filtracao

        Returns:
            PersistenceDiagram com pares (nascimento, morte) para cada dimensao
        """
        n = len(points)
        dist_matrix = squareform(pdist(points))
        max_dist = np.max(dist_matrix)

        epsilon_values = np.linspace(0, max_dist, n_steps)

        # Rastreia nascimento e morte de caracteristicas
        pairs_dim0 = []
        pairs_dim1 = []
        pairs_dim2 = []

        # Historico de Betti
        prev_beta_0 = n  # Inicialmente, cada ponto e um componente
        prev_beta_1 = 0

        # Rastreia componentes ativos
        active_components = {i: 0.0 for i in range(n)}  # nascimento de cada componente

        for eps_idx, epsilon in enumerate(epsilon_values):
            # Computa Betti numbers
            beta_0, components = self._compute_betti_0(points, dist_matrix, epsilon)
            beta_1 = self._compute_betti_1_approximate(points, dist_matrix, epsilon)

            # Detecta mortes de componentes (fusoes) - dim 0
            if beta_0 < prev_beta_0:
                n_deaths = prev_beta_0 - beta_0
                for _ in range(n_deaths):
                    # Encontra o componente mais jovem que morreu
                    if active_components:
                        youngest = max(active_components.items(), key=lambda x: x[1])
                        birth = youngest[1]
                        pairs_dim0.append(PersistencePair(birth, epsilon, dimension=0))
                        del active_components[youngest[0]]

            # Detecta nascimento de loops - dim 1
            if beta_1 > prev_beta_1:
                n_births = beta_1 - prev_beta_1
                for _ in range(n_births):
                    pairs_dim1.append(PersistencePair(epsilon, np.inf, dimension=1))

            # Detecta morte de loops - dim 1
            elif beta_1 < prev_beta_1:
                n_deaths = prev_beta_1 - beta_1
                # Fecha os loops mais recentes
                open_loops = [p for p in pairs_dim1 if p.death == np.inf]
                for i in range(min(n_deaths, len(open_loops))):
                    # Atualiza morte do loop mais antigo aberto
                    oldest_open = min(open_loops, key=lambda x: x.birth)
                    oldest_open_idx = pairs_dim1.index(oldest_open)
                    pairs_dim1[oldest_open_idx] = PersistencePair(
                        oldest_open.birth, epsilon, dimension=1
                    )
                    open_loops.remove(oldest_open)

            prev_beta_0 = beta_0
            prev_beta_1 = beta_1

        # Adiciona o componente "infinito" (o que nunca morre)
        if active_components:
            oldest = min(active_components.items(), key=lambda x: x[1])
            pairs_dim0.append(PersistencePair(oldest[1], np.inf, dimension=0))

        return PersistenceDiagram(
            pairs_dim0=pairs_dim0,
            pairs_dim1=pairs_dim1,
            pairs_dim2=pairs_dim2
        )


# ==============================================================================
# PAISAGEM DE PERSISTENCIA
# ==============================================================================

class PersistenceLandscapeCalculator:
    """
    Calculo da Paisagem de Persistencia (Persistence Landscape)

    Em vez de usar o diagrama bruto, transformamos em uma funcao L_k(t) integravel
    (Logica de Banach space). Isso resume a "topologia dominante" do mercado.
    """

    def __init__(self, max_k: int = 5, n_points: int = 100):
        self.max_k = max_k
        self.n_points = n_points

    def _tent_function(self, t: float, birth: float, death: float) -> float:
        """Funcao tenda para um par (birth, death)"""
        if death == np.inf:
            death = birth + 2 * (t - birth) if t > birth else birth + 1
        return max(0, min(t - birth, death - t))

    def compute_landscape(self,
                         pairs: List[PersistencePair],
                         t_min: float = 0,
                         t_max: Optional[float] = None) -> PersistenceLandscape:
        """Computa a Paisagem de Persistencia"""
        if not pairs:
            t_values = np.linspace(0, 1, self.n_points)
            return PersistenceLandscape(
                landscapes=np.zeros((self.max_k, self.n_points)),
                t_values=t_values,
                max_k=self.max_k
            )

        births = [p.birth for p in pairs]
        deaths = [p.death if p.death != np.inf else p.birth + 1 for p in pairs]

        if t_max is None:
            t_max = max(deaths) * 1.1

        t_values = np.linspace(t_min, t_max, self.n_points)
        landscapes = np.zeros((self.max_k, self.n_points))

        for t_idx, t in enumerate(t_values):
            tent_values = []
            for p in pairs:
                death = p.death if p.death != np.inf else t_max
                tent = self._tent_function(t, p.birth, death)
                tent_values.append(tent)

            tent_values.sort(reverse=True)

            for k in range(min(self.max_k, len(tent_values))):
                landscapes[k, t_idx] = tent_values[k]

        return PersistenceLandscape(
            landscapes=landscapes,
            t_values=t_values,
            max_k=self.max_k
        )


# ==============================================================================
# ENTROPIA TOPOLOGICA
# ==============================================================================

class TopologicalEntropyCalculator:
    """
    Calculo da Entropia Topologica Persistente

    E = -Sum p_i log(p_i)

    Onde p_i e a persistencia relativa do ciclo i.
    """

    def compute_entropy(self, pairs: List[PersistencePair]) -> float:
        """Computa entropia topologica"""
        if not pairs:
            return 0.0

        persistences = []
        for p in pairs:
            pers = p.persistence
            if pers != np.inf and pers > 0:
                persistences.append(pers)

        if not persistences:
            return 0.0

        total = sum(persistences)
        if total == 0:
            return 0.0

        probs = [p / total for p in persistences]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)

        return entropy


# ==============================================================================
# INDICADOR B-PHS COMPLETO
# ==============================================================================

class BettiPersistenceHomologyScanner:
    """
    Betti-Persistence Homology Scanner (B-PHS)

    O Sinal de Entrada (Trading Topological Features):

    - LONG: Quando V_t esta na "base" da variedade topologica e B1 esta no maximo.
            O preco precisa SUBIR para completar a topologia do anel.

    - SHORT: Quando V_t esta no "topo" da variedade e a persistencia ainda e alta.

    - BLOCKED: Se Entropia Topologica subir subitamente ou B1 cair para zero,
               o mercado "quebrou" a estrutura ciclica. BLOQUEIA operacao.
    """

    def __init__(self,
                 embedding_dim: int = 3,
                 time_delay: Optional[int] = None,
                 max_homology_dim: int = 2,
                 filtration_steps: int = 50,
                 min_loop_persistence: float = 0.1,
                 max_entropy_threshold: float = 2.0,
                 position_threshold: float = 0.3,
                 min_data_points: int = 100):

        self.embedding_dim = embedding_dim
        self.max_homology_dim = max_homology_dim
        self.filtration_steps = filtration_steps
        self.min_loop_persistence = min_loop_persistence
        self.max_entropy_threshold = max_entropy_threshold
        self.position_threshold = position_threshold
        self.min_data_points = min_data_points

        self.takens = TakensEmbedding(
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            auto_delay=True
        )
        self.vietoris_rips = VietorisRipsComplex(max_dimension=max_homology_dim)
        self.homology = PersistentHomology(max_dimension=max_homology_dim)
        self.landscape_calc = PersistenceLandscapeCalculator()
        self.entropy_calc = TopologicalEntropyCalculator()

        self.last_embedding: Optional[np.ndarray] = None
        self.last_analysis: Optional[TopologicalAnalysis] = None
        self.entropy_history: List[float] = []
        self.betti1_history: List[int] = []

    def _find_optimal_epsilon(self, persistence_diagram: PersistenceDiagram) -> float:
        """Encontra o epsilon otimo onde B1 e maximo e estavel"""
        if not persistence_diagram.pairs_dim1:
            return 0.5

        max_persistence = 0
        optimal_eps = 0.5

        for pair in persistence_diagram.pairs_dim1:
            if pair.persistence != np.inf and pair.persistence > max_persistence:
                max_persistence = pair.persistence
                optimal_eps = pair.midlife

        return optimal_eps

    def _compute_loop_centroid(self,
                               points: np.ndarray,
                               epsilon: float) -> Optional[np.ndarray]:
        """Calcula o centroide do loop dominante"""
        dist_matrix = squareform(pdist(points))
        adj_matrix = dist_matrix <= epsilon
        connected_mask = np.any(adj_matrix, axis=1)
        connected_points = points[connected_mask]

        if len(connected_points) == 0:
            return None

        return np.mean(connected_points, axis=0)

    def _determine_position_in_loop(self,
                                    current_point: np.ndarray,
                                    centroid: np.ndarray,
                                    all_points: np.ndarray) -> Tuple[str, float]:
        """Determina a posicao do ponto atual no ciclo topologico"""
        dist_to_centroid = np.linalg.norm(current_point - centroid)
        all_distances = np.linalg.norm(all_points - centroid, axis=1)

        max_dist = np.max(all_distances)
        min_dist = np.min(all_distances)

        if max_dist == min_dist:
            return "MIDDLE", dist_to_centroid

        normalized_dist = (dist_to_centroid - min_dist) / (max_dist - min_dist)
        vertical_diff = current_point[1] - centroid[1] if len(current_point) > 1 else 0

        if normalized_dist > (1 - self.position_threshold):
            if vertical_diff > 0:
                return "TOP", dist_to_centroid
            else:
                return "BOTTOM", dist_to_centroid
        else:
            return "MIDDLE", dist_to_centroid

    def _detect_betti_crash(self) -> bool:
        """Detecta "crash" do Betti (quebra da estrutura ciclica)"""
        if len(self.entropy_history) < 3 or len(self.betti1_history) < 3:
            return False

        recent_entropy = self.entropy_history[-3:]
        entropy_increase = recent_entropy[-1] - recent_entropy[0]

        if entropy_increase > 0.5:
            return True

        recent_betti1 = self.betti1_history[-3:]
        if recent_betti1[0] > 0 and recent_betti1[-1] == 0:
            return True

        return False

    def _determine_regime(self,
                         betti_1: int,
                         entropy: float,
                         max_persistence: float) -> TopologyRegime:
        """Determina o regime topologico atual"""
        if self._detect_betti_crash():
            return TopologyRegime.LINEAR

        if entropy > self.max_entropy_threshold:
            return TopologyRegime.CHAOTIC

        if betti_1 > 0 and max_persistence > self.min_loop_persistence:
            return TopologyRegime.CYCLIC

        if betti_1 == 0:
            return TopologyRegime.LINEAR

        return TopologyRegime.TRANSITIONAL

    def analyze(self, prices: np.ndarray) -> dict:
        """Executa analise topologica completa"""
        embedded = self.takens.embed(prices)
        self.last_embedding = embedded

        persistence_diagram = self.homology.compute_persistence(
            embedded, n_steps=self.filtration_steps
        )

        landscape = self.landscape_calc.compute_landscape(
            persistence_diagram.pairs_dim1
        )

        entropy = self.entropy_calc.compute_entropy(persistence_diagram.pairs_dim1)
        optimal_eps = self._find_optimal_epsilon(persistence_diagram)

        betti_0 = persistence_diagram.get_betti_0(optimal_eps)
        betti_1 = persistence_diagram.get_betti_1(optimal_eps)

        max_loop_pers = persistence_diagram.get_max_persistence_dim1()
        centroid = self._compute_loop_centroid(embedded, optimal_eps)
        regime = self._determine_regime(betti_1, entropy, max_loop_pers)

        # Atualiza historico
        self.entropy_history.append(entropy)
        self.betti1_history.append(betti_1)

        if len(self.entropy_history) > 50:
            self.entropy_history.pop(0)
            self.betti1_history.pop(0)

        # Posicao no ciclo
        current_point = embedded[-1]

        if centroid is not None:
            position, dist_to_centroid = self._determine_position_in_loop(
                current_point, centroid, embedded
            )
        else:
            position = "UNKNOWN"
            dist_to_centroid = 0.0

        # Determina sinal
        signal = 0
        signal_name = "NEUTRAL"
        confidence = 0.0
        reasons = []

        # FILTRO DE ABORTAR
        if self._detect_betti_crash():
            signal_name = "BLOCKED"
            reasons.append("BETTI_CRASH")
            reasons.append("estrutura_ciclica_quebrada")

        elif regime == TopologyRegime.LINEAR:
            signal_name = "BLOCKED"
            reasons.append(f"REGIME_LINEAR")
            reasons.append(f"B1={betti_1}")

        elif regime == TopologyRegime.CHAOTIC:
            signal_name = "BLOCKED"
            reasons.append("REGIME_CAOTICO")
            reasons.append(f"entropy={entropy:.3f}")

        elif regime == TopologyRegime.CYCLIC:

            if position == "BOTTOM" and max_loop_pers > self.min_loop_persistence:
                signal = 1
                signal_name = "LONG"
                confidence = min(1.0, max_loop_pers * 2) * 0.7
                confidence += (1 - dist_to_centroid / (np.max([dist_to_centroid, 1]) + 1)) * 0.3
                confidence = np.clip(confidence, 0, 1)
                reasons.append("base_do_ciclo")
                reasons.append(f"B1={betti_1}")
                reasons.append(f"pers={max_loop_pers:.3f}")

            elif position == "TOP" and max_loop_pers > self.min_loop_persistence:
                signal = -1
                signal_name = "SHORT"
                confidence = min(1.0, max_loop_pers * 2) * 0.7
                confidence += (1 - dist_to_centroid / (np.max([dist_to_centroid, 1]) + 1)) * 0.3
                confidence = np.clip(confidence, 0, 1)
                reasons.append("topo_do_ciclo")
                reasons.append(f"B1={betti_1}")
                reasons.append(f"pers={max_loop_pers:.3f}")

            else:
                signal_name = "NEUTRAL"
                reasons.append(f"pos={position}")
                reasons.append("aguardando_extremo")

        else:
            signal_name = "NEUTRAL"
            reasons.append("regime_transitional")

        self.last_analysis = TopologicalAnalysis(
            persistence_diagram=persistence_diagram,
            persistence_landscape=landscape,
            betti_numbers={0: betti_0, 1: betti_1, 2: 0},
            topological_entropy=entropy,
            dominant_loop_persistence=max_loop_pers,
            loop_centroid=centroid,
            regime=regime
        )

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'regime': regime.value,
            'betti_0': betti_0,
            'betti_1': betti_1,
            'betti_2': 0,
            'max_loop_persistence': max_loop_pers,
            'topological_entropy': entropy,
            'position_in_loop': position,
            'distance_to_centroid': dist_to_centroid,
            'optimal_epsilon': optimal_eps,
            'embedding_dim': self.embedding_dim,
            'time_delay': self.takens.time_delay,
            'reasons': reasons,
            'current_price': prices[-1]
        }

    def get_persistence_diagram(self) -> Optional[PersistenceDiagram]:
        """Retorna o ultimo diagrama de persistencia"""
        if self.last_analysis is not None:
            return self.last_analysis.persistence_diagram
        return None

    def get_embedding(self) -> Optional[np.ndarray]:
        """Retorna o ultimo embedding de Takens"""
        return self.last_embedding

    def get_betti_history(self) -> List[int]:
        """Retorna historico de B1"""
        return self.betti1_history.copy()

    def get_entropy_history(self) -> List[float]:
        """Retorna historico de entropia"""
        return self.entropy_history.copy()

    def reset(self):
        """Reseta o estado do indicador"""
        self.last_embedding = None
        self.last_analysis = None
        self.entropy_history.clear()
        self.betti1_history.clear()
        self.takens.time_delay = None


# ==============================================================================
# EXEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BETTI-PERSISTENCE HOMOLOGY SCANNER (B-PHS)")
    print("Indicador baseado em Topologia Algebrica")
    print("=" * 70)

    np.random.seed(42)

    # Gera dados ciclicos
    n_points = 300
    t = np.arange(n_points)
    trend = 1.0850 + 0.00005 * t
    cycle = 0.003 * np.sin(2 * np.pi * t / 40)
    harmonic = 0.001 * np.sin(2 * np.pi * t / 20)
    noise = np.random.randn(n_points) * 0.0003
    prices = trend + cycle + harmonic + noise

    print(f"\nDados: {len(prices)} pontos")
    print(f"Preco: {prices[0]:.5f} -> {prices[-1]:.5f}")

    # Cria indicador
    indicator = BettiPersistenceHomologyScanner(
        embedding_dim=3,
        filtration_steps=50,
        min_loop_persistence=0.05,
        max_entropy_threshold=2.0,
        min_data_points=100
    )

    print("\nAnalisando topologia do mercado...")

    result = indicator.analyze(prices)

    print("\n" + "-" * 40)
    print("RESULTADO:")
    print(f"  Sinal: {result['signal_name']}")
    print(f"  Regime: {result['regime']}")
    print(f"  Confianca: {result['confidence']:.0%}")

    print("\nBETTI NUMBERS:")
    print(f"  B0 (componentes): {result['betti_0']}")
    print(f"  B1 (loops) [SANTO GRAAL]: {result['betti_1']}")
    print(f"  B2 (vazios): {result['betti_2']}")

    print("\nTOPOLOGIA:")
    print(f"  Persistencia max: {result['max_loop_persistence']:.4f}")
    print(f"  Entropia: {result['topological_entropy']:.4f}")
    print(f"  Posicao no ciclo: {result['position_in_loop']}")

    print("\nEMBEDDING:")
    print(f"  Dimensao: {result['embedding_dim']}")
    print(f"  Time delay (tau): {result['time_delay']}")

    print("\n" + "=" * 70)
    print("Teste concluido!")
    print("=" * 70)
