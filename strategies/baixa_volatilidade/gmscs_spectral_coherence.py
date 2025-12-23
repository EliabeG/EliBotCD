"""
================================================================================
GLOBAL MACRO SPECTRAL COHERENCE SCANNER (GMS-CS)
Indicador de Forex baseado em Teoria Espectral de Grafos e MST
================================================================================

Este indicador utiliza Teoria Espectral de Grafos e Topologia de Árvores
Geradoras Mínimas (MST). Ele constrói, em tempo real, a "coluna vertebral" da
estrutura de correlação de todo o mercado financeiro global.

O objetivo é detectar o Colapso Topológico. Antes de uma explosão de volatilidade
no EURUSD, a rede global se contrai (sincroniza) e a geometria da árvore muda
drasticamente.

A Matemática: O Laplaciano do Grafo de Mercado
Você não vai monitorar preços. Você vai monitorar a Conectividade Algébrica da
matriz de correlação mundial.

Por que usar Teoria dos Grafos?
1. Visão Holística: O EURUSD muitas vezes não se move por causa da Europa ou dos
   EUA, mas por causa de um margin call no Japão ou um choque no petróleo.
   Indicadores univariados (RSI, Bollinger) são cegos a isso. O GMS-CS vê o contágio.
2. Sinal Precoce: A correlação muda ANTES do preço. Os robôs de arbitragem
   estatística alinham os ativos milissegundos antes do movimento direcional
   acontecer. A contração da MST é o "carregar da mola".
3. Filtragem de Ruído Robusta: A MST, por definição, elimina as conexões fracas
   (ruído) e mantém apenas o caminho mais forte de transmissão de informação.
   É o filtro de ruído supremo.

Nota Técnica: Use networkx para manipulação de grafos e scipy.linalg.eigh para
as métricas de distância. O cálculo de autovalores é rápido o suficiente para
matrizes 50x50.

Autor: Gerado por Claude AI
Versão: 1.0.0
================================================================================
"""

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import squareform, pdist
from typing import Tuple, Optional, List, Dict, Set
from dataclasses import dataclass
from enum import Enum
from collections import deque
import warnings
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

class GMSCSSignalType(Enum):
    """Tipos de sinais de trading"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    WAIT = "WAIT"


class TopologyState(Enum):
    """Estado topológico da rede"""
    RELAXED = "RELAXED"             # Árvore espalhada, cadeia longa
    CONTRACTING = "CONTRACTING"      # Árvore se contraindo
    STAR = "STAR"                    # Topologia estrela (pré-crash)
    COLLAPSED = "COLLAPSED"          # Colapso topológico


class MarketPhase(Enum):
    """Fase do mercado baseada na topologia"""
    FRAGMENTED = "FRAGMENTED"        # Mercado fragmentado (cada um por si)
    COUPLING = "COUPLING"             # Ativos se acoplando
    SYNCHRONIZED = "SYNCHRONIZED"     # Alta sincronia (perigo!)
    DECOUPLING = "DECOUPLING"        # Desacoplando (pós-evento)


@dataclass
class Asset:
    """Representa um ativo no grafo"""
    name: str
    returns: np.ndarray
    current_price: float = 0.0


@dataclass
class CorrelationMatrix:
    """Matriz de correlação"""
    rho: np.ndarray                  # Matriz de correlação
    distance: np.ndarray             # Matriz de distância ultramétrica
    n_assets: int
    asset_names: List[str]


@dataclass
class MSTEdge:
    """Aresta da MST"""
    node_i: int
    node_j: int
    weight: float
    asset_i: str
    asset_j: str


@dataclass
class MinimumSpanningTree:
    """Árvore Geradora Mínima"""
    edges: List[MSTEdge]
    adjacency: np.ndarray            # Matriz de adjacência
    total_length: float              # Comprimento total
    normalized_length: float         # NTL - Comprimento normalizado
    n_nodes: int


@dataclass
class LaplacianSpectrum:
    """Espectro do Laplaciano"""
    laplacian: np.ndarray            # Matriz Laplaciana L = D - A
    eigenvalues: np.ndarray          # Autovalores (ordenados)
    eigenvectors: np.ndarray         # Autovetores
    fiedler_value: float             # λ₂ - Segundo menor autovalor
    fiedler_vector: np.ndarray       # Vetor de Fiedler
    algebraic_connectivity: float    # Conectividade algébrica


@dataclass
class Centrality:
    """Centralidades dos nós"""
    degree: np.ndarray               # Grau de cada nó
    eigenvector: np.ndarray          # Centralidade de autovetor
    betweenness: np.ndarray          # Centralidade de intermediação
    central_node: int                # Nó mais central
    central_asset: str               # Ativo mais central


@dataclass
class TopologyMetrics:
    """Métricas topológicas"""
    ntl: float                       # Normalized Tree Length
    fiedler: float                   # Valor de Fiedler
    diameter: int                    # Diâmetro da árvore
    max_degree: int                  # Grau máximo
    star_index: float                # Índice de "estrela" (0-1)
    chain_index: float               # Índice de "cadeia" (0-1)


# ==============================================================================
# MATRIZ DE CORRELAÇÃO E DISTÂNCIA
# ==============================================================================

class CorrelationMatrixBuilder:
    """
    1. A Matriz de Distância Ultramétrica (D)

    Pegue 50 ativos globais descorrelacionados em teoria (Forex Majors,
    Commodities, Índices, Bonds). Calcule a matriz de correlação ρ_ij.
    Converta isso em uma métrica de distância Euclidiana:

    d_ij = √(2(1 - ρ_ij))

    - Se d_ij ≈ 0: Os ativos andam juntos (Sincronia)
    - Se d_ij ≈ √2: Ortogonais
    - Se d_ij ≈ 2: Anti-correlacionados
    """

    def __init__(self, window_size: int = 200):
        """
        Args:
            window_size: Janela para cálculo de correlação
        """
        self.window_size = window_size

    def build_correlation_matrix(self,
                                 returns_dict: Dict[str, np.ndarray]) -> CorrelationMatrix:
        """
        Constrói matriz de correlação a partir dos retornos
        """
        asset_names = list(returns_dict.keys())
        n_assets = len(asset_names)

        # Matriz de retornos
        min_len = min(len(r) for r in returns_dict.values())
        returns_matrix = np.zeros((min_len, n_assets))

        for i, name in enumerate(asset_names):
            returns_matrix[:, i] = returns_dict[name][-min_len:]

        # Correlação
        rho = np.corrcoef(returns_matrix.T)

        # Garante que é simétrica e sem NaN
        rho = np.nan_to_num(rho, nan=0.0)
        rho = (rho + rho.T) / 2
        np.fill_diagonal(rho, 1.0)

        # Distância ultramétrica: d_ij = √(2(1 - ρ_ij))
        distance = np.sqrt(2 * (1 - rho))
        np.fill_diagonal(distance, 0.0)

        return CorrelationMatrix(
            rho=rho,
            distance=distance,
            n_assets=n_assets,
            asset_names=asset_names
        )


# ==============================================================================
# ÁRVORE GERADORA MÍNIMA (MST)
# ==============================================================================

class MinimumSpanningTreeBuilder:
    """
    2. Árvore Geradora Mínima (MST - Minimum Spanning Tree)

    O mercado é ruidoso. Para ver a estrutura real, precisamos filtrar o grafo
    completo mantendo apenas as conexões mais fortes que conectam todos os nós
    sem formar ciclos. Use o Algoritmo de Kruskal ou Prim para extrair a MST.
    Isso revela a hierarquia taxonômica do mercado naquele instante. Quem manda
    em quem?
    """

    def __init__(self):
        pass

    def build_mst_kruskal(self, corr_matrix: CorrelationMatrix) -> MinimumSpanningTree:
        """
        Constrói MST usando algoritmo de Kruskal

        1. Ordena todas as arestas por peso
        2. Para cada aresta, adiciona se não formar ciclo
        3. Para até ter N-1 arestas
        """
        n = corr_matrix.n_assets
        distance = corr_matrix.distance
        names = corr_matrix.asset_names

        # Lista todas as arestas com pesos
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                weight = distance[i, j]
                edges.append((weight, i, j))

        # Ordena por peso (menor primeiro)
        edges.sort(key=lambda x: x[0])

        # Union-Find para detectar ciclos
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True

        # Constrói MST
        mst_edges = []
        adjacency = np.zeros((n, n))
        total_length = 0.0

        for weight, i, j in edges:
            if union(i, j):
                mst_edges.append(MSTEdge(
                    node_i=i,
                    node_j=j,
                    weight=weight,
                    asset_i=names[i],
                    asset_j=names[j]
                ))
                adjacency[i, j] = weight
                adjacency[j, i] = weight
                total_length += weight

                if len(mst_edges) == n - 1:
                    break

        # Comprimento normalizado (NTL)
        # L(t) = (1/(N-1)) * Σ d_ij(t) para edges em MST
        ntl = total_length / (n - 1) if n > 1 else 0.0

        return MinimumSpanningTree(
            edges=mst_edges,
            adjacency=adjacency,
            total_length=total_length,
            normalized_length=ntl,
            n_nodes=n
        )


# ==============================================================================
# LAPLACIANO ESPECTRAL
# ==============================================================================

class LaplacianAnalyzer:
    """
    3. O Valor de Fiedler (Conectividade Algébrica)

    Aqui entra a Álgebra Linear Espectral. Construa a Matriz Laplaciana do
    grafo da MST:

    L = D_deg - A

    - D_deg: Matriz de graus (quantas conexões cada ativo tem)
    - A: Matriz de adjacência

    Calcule os autovalores de L. O segundo menor autovalor (λ₂) é chamado de
    Valor de Fiedler.

    - λ₂ mede a "dificuldade" de cortar o grafo em dois.
    - Em baixa volatilidade segura: λ₂ é baixo (mercado fragmentado, cada um por si)
    - Em baixa volatilidade perigosa (pré-crash): λ₂ dispara. O mercado se torna
      um "supercondutor" de risco.
    """

    def __init__(self):
        pass

    def compute_laplacian_spectrum(self,
                                   mst: MinimumSpanningTree) -> LaplacianSpectrum:
        """
        Calcula espectro do Laplaciano da MST
        """
        n = mst.n_nodes

        # Matriz de adjacência (binária para MST)
        A = (mst.adjacency > 0).astype(float)

        # Matriz de graus
        degrees = np.sum(A, axis=1)
        D = np.diag(degrees)

        # Laplaciano: L = D - A
        L = D - A

        # Autovalores e autovetores
        eigenvalues, eigenvectors = eigh(L)

        # Ordena por autovalor
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Valor de Fiedler (segundo menor autovalor)
        # O primeiro é sempre 0 para grafo conectado
        fiedler_value = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        fiedler_vector = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else np.zeros(n)

        return LaplacianSpectrum(
            laplacian=L,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            fiedler_value=fiedler_value,
            fiedler_vector=fiedler_vector,
            algebraic_connectivity=fiedler_value
        )


# ==============================================================================
# CENTRALIDADE
# ==============================================================================

class CentralityCalculator:
    """
    Passo 3: Centralidade de Autovetor

    Calcule quem é o "Nó Central" da MST. Normalmente é o SPX500 ou US10Y.
    Se o EURUSD subitamente se tornar o centro da árvore ou se conectar
    diretamente ao centro (reduzindo a "Mean Occupation Layer"), ele está
    prestes a explodir.
    """

    def __init__(self):
        pass

    def compute_centralities(self,
                            mst: MinimumSpanningTree,
                            asset_names: List[str]) -> Centrality:
        """
        Calcula várias medidas de centralidade
        """
        n = mst.n_nodes
        A = (mst.adjacency > 0).astype(float)

        # Grau
        degree = np.sum(A, axis=1)

        # Centralidade de autovetor
        # É o autovetor principal da matriz de adjacência
        eigenvalues, eigenvectors = eigh(A)
        eigenvector_centrality = np.abs(eigenvectors[:, -1])
        eigenvector_centrality /= np.max(eigenvector_centrality) + 1e-10

        # Centralidade de intermediação (simplificada para árvore)
        betweenness = self._compute_betweenness(A, n)

        # Nó mais central
        central_node = int(np.argmax(eigenvector_centrality))
        central_asset = asset_names[central_node] if central_node < len(asset_names) else "Unknown"

        return Centrality(
            degree=degree,
            eigenvector=eigenvector_centrality,
            betweenness=betweenness,
            central_node=central_node,
            central_asset=central_asset
        )

    def _compute_betweenness(self, A: np.ndarray, n: int) -> np.ndarray:
        """
        Calcula centralidade de intermediação (betweenness)
        Para uma árvore, é proporcional ao número de nós em cada lado
        """
        betweenness = np.zeros(n)

        # Para cada nó, conta quantos pares de nós passam por ele
        for node in range(n):
            # Remove o nó e conta tamanho das componentes
            A_temp = A.copy()
            A_temp[node, :] = 0
            A_temp[:, node] = 0

            # Encontra componentes conectadas
            visited = set()
            components = []

            for start in range(n):
                if start != node and start not in visited:
                    component = self._bfs(A_temp, start, visited)
                    components.append(len(component))

            # Betweenness é proporcional ao produto dos tamanhos
            if len(components) >= 2:
                total = 0
                for i in range(len(components)):
                    for j in range(i + 1, len(components)):
                        total += components[i] * components[j]
                betweenness[node] = total

        # Normaliza
        max_between = np.max(betweenness) + 1e-10
        betweenness /= max_between

        return betweenness

    def _bfs(self, A: np.ndarray, start: int, visited: Set[int]) -> Set[int]:
        """BFS para encontrar componente conectada"""
        component = set()
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)

            # Vizinhos
            neighbors = np.where(A[node] > 0)[0]
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

        return component

    def compute_distance_to_center(self,
                                   mst: MinimumSpanningTree,
                                   node: int,
                                   center: int) -> int:
        """
        Calcula distância (em hops) de um nó até o centro
        """
        if node == center:
            return 0

        n = mst.n_nodes
        A = (mst.adjacency > 0).astype(float)

        # BFS do nó até o centro
        visited = {node}
        queue = deque([(node, 0)])

        while queue:
            current, dist = queue.popleft()

            if current == center:
                return dist

            neighbors = np.where(A[current] > 0)[0]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return -1  # Não encontrado (não deveria acontecer em árvore conectada)


# ==============================================================================
# ANALISADOR DE TOPOLOGIA
# ==============================================================================

class TopologyAnalyzer:
    """
    Analisa a topologia da MST para detectar transições
    """

    def __init__(self):
        pass

    def compute_topology_metrics(self,
                                mst: MinimumSpanningTree,
                                centrality: Centrality) -> TopologyMetrics:
        """
        Calcula métricas topológicas
        """
        n = mst.n_nodes
        A = (mst.adjacency > 0).astype(float)

        # Diâmetro (maior distância entre dois nós)
        diameter = self._compute_diameter(A, n)

        # Grau máximo
        max_degree = int(np.max(centrality.degree))

        # Índice de estrela: quanto mais o grau máximo se aproxima de N-1
        star_index = max_degree / (n - 1) if n > 1 else 0.0

        # Índice de cadeia: quanto mais o diâmetro se aproxima de N-1
        chain_index = diameter / (n - 1) if n > 1 else 0.0

        return TopologyMetrics(
            ntl=mst.normalized_length,
            fiedler=0.0,  # Será preenchido depois
            diameter=diameter,
            max_degree=max_degree,
            star_index=star_index,
            chain_index=chain_index
        )

    def _compute_diameter(self, A: np.ndarray, n: int) -> int:
        """
        Calcula diâmetro da árvore (maior caminho)
        """
        max_dist = 0

        for start in range(n):
            # BFS de cada nó
            visited = {start}
            queue = deque([(start, 0)])

            while queue:
                node, dist = queue.popleft()
                max_dist = max(max_dist, dist)

                neighbors = np.where(A[node] > 0)[0]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return max_dist

    def detect_topology_state(self,
                             metrics: TopologyMetrics,
                             fiedler: float,
                             fiedler_history: List[float],
                             ntl_history: List[float]) -> TopologyState:
        """
        Detecta estado topológico atual
        """
        # Verifica tendências
        if len(ntl_history) >= 5:
            ntl_trend = np.mean(ntl_history[-3:]) - np.mean(ntl_history[-6:-3])
        else:
            ntl_trend = 0.0

        if len(fiedler_history) >= 5:
            fiedler_trend = np.mean(fiedler_history[-3:]) - np.mean(fiedler_history[-6:-3])
        else:
            fiedler_trend = 0.0

        # Classifica
        if metrics.star_index > 0.5:
            # Topologia estrela
            if fiedler > 0.5:
                return TopologyState.COLLAPSED
            else:
                return TopologyState.STAR
        elif ntl_trend < -0.05 or fiedler_trend > 0.1:
            # Árvore se contraindo
            return TopologyState.CONTRACTING
        elif metrics.chain_index > 0.7:
            return TopologyState.RELAXED
        else:
            return TopologyState.RELAXED


# ==============================================================================
# INDICADOR GMS-CS COMPLETO
# ==============================================================================

class GlobalMacroSpectralCoherenceScanner:
    """
    Global Macro Spectral Coherence Scanner (GMS-CS)

    Indicador completo que usa Teoria Espectral de Grafos para detectar
    colapso topológico no mercado global.

    A Lógica de Trading (A Contração da Árvore)
    O indicador opera a Transição de Fase Topológica.

    1. Estado Relaxado (Noise): O Comprimento da Árvore (NTL) é alto. A MST é
       espalhada, parecida com uma cadeia longa. O EURUSD está na periferia.
       Diagnóstico: A baixa volatilidade é genuína. Não há perigo.

    2. O Gatilho (Topology Crunch): O indicador detecta que o NTL começou a cair
       rapidamente (o mercado global está se encolhendo/apertando). O Valor de
       Fiedler sobe. A topologia da MST muda de "Cadeia" para "Estrela"
       (Star Graph), com todos os ativos se conectando a um único driver.

    3. SINAL: O EURUSD está parado (Baixa Vol), mas a rede global colapsou para
       um estado de alta sincronia.
       - Ação: O rompimento é iminente e será sistêmico.
       - Direção: Olhe para o "Nó Raiz" da MST (o ativo que puxou a árvore).
         Se o Yield de 10 anos subiu e puxou a árvore, venda EURUSD. Se o Ouro
         subiu e puxou a árvore, compre EURUSD. Você segue o Líder Topológico.
    """

    def __init__(self,
                 # Parâmetros de correlação
                 correlation_window: int = 200,

                 # Limiares
                 fiedler_threshold: float = 0.3,
                 ntl_contraction_threshold: float = 0.1,
                 star_threshold: float = 0.4,

                 # Geral
                 min_history: int = 10,
                 min_data_points: int = 50):
        """
        Inicializa o GMS-CS
        """
        self.correlation_window = correlation_window
        self.fiedler_threshold = fiedler_threshold
        self.ntl_contraction_threshold = ntl_contraction_threshold
        self.star_threshold = star_threshold
        self.min_history = min_history
        self.min_data_points = min_data_points

        # Componentes
        self.corr_builder = CorrelationMatrixBuilder(window_size=correlation_window)
        self.mst_builder = MinimumSpanningTreeBuilder()
        self.laplacian_analyzer = LaplacianAnalyzer()
        self.centrality_calculator = CentralityCalculator()
        self.topology_analyzer = TopologyAnalyzer()

        # Histórico
        self.fiedler_history: List[float] = []
        self.ntl_history: List[float] = []
        self.central_asset_history: List[str] = []

    def analyze(self,
               returns_dict: Dict[str, np.ndarray],
               target_asset: str = "EURUSD") -> dict:
        """
        Processa dados de múltiplos ativos e gera resultado de análise

        Args:
            returns_dict: Dicionário {nome_ativo: array_retornos}
            target_asset: Ativo alvo para o sinal (default: EURUSD)

        Returns:
            Dict com todos os resultados da análise
        """
        n_assets = len(returns_dict)

        # Validação
        if n_assets < 5:
            return self._create_empty_result("INSUFFICIENT_ASSETS")

        # Verifica se todos os ativos têm dados suficientes
        min_len = min(len(r) for r in returns_dict.values())
        if min_len < self.min_data_points:
            return self._create_empty_result("INSUFFICIENT_DATA")

        # PASSO 1: MATRIZ DE CORRELAÇÃO E DISTÂNCIA
        corr_matrix = self.corr_builder.build_correlation_matrix(returns_dict)

        mean_correlation = np.mean(corr_matrix.rho[np.triu_indices(n_assets, k=1)])
        max_correlation = np.max(corr_matrix.rho[np.triu_indices(n_assets, k=1)])

        # PASSO 2: ÁRVORE GERADORA MÍNIMA
        mst = self.mst_builder.build_mst_kruskal(corr_matrix)

        # Atualiza histórico
        self.ntl_history.append(mst.normalized_length)
        if len(self.ntl_history) > 100:
            self.ntl_history.pop(0)

        # Mudança no NTL
        if len(self.ntl_history) >= 2:
            ntl_change = self.ntl_history[-1] - self.ntl_history[-2]
        else:
            ntl_change = 0.0

        # PASSO 3: ESPECTRO LAPLACIANO
        spectrum = self.laplacian_analyzer.compute_laplacian_spectrum(mst)

        # Atualiza histórico
        self.fiedler_history.append(spectrum.fiedler_value)
        if len(self.fiedler_history) > 100:
            self.fiedler_history.pop(0)

        # Mudança no Fiedler
        if len(self.fiedler_history) >= 2:
            fiedler_change = self.fiedler_history[-1] - self.fiedler_history[-2]
        else:
            fiedler_change = 0.0

        # PASSO 4: CENTRALIDADE
        centrality = self.centrality_calculator.compute_centralities(
            mst, corr_matrix.asset_names
        )

        # Centralidade do ativo alvo
        target_idx = -1
        if target_asset in corr_matrix.asset_names:
            target_idx = corr_matrix.asset_names.index(target_asset)

        if target_idx >= 0:
            target_centrality = centrality.eigenvector[target_idx]
            target_distance = self.centrality_calculator.compute_distance_to_center(
                mst, target_idx, centrality.central_node
            )
        else:
            target_centrality = 0.0
            target_distance = -1

        # PASSO 5: MÉTRICAS TOPOLÓGICAS
        topology_metrics = self.topology_analyzer.compute_topology_metrics(mst, centrality)
        topology_metrics = TopologyMetrics(
            ntl=topology_metrics.ntl,
            fiedler=spectrum.fiedler_value,
            diameter=topology_metrics.diameter,
            max_degree=topology_metrics.max_degree,
            star_index=topology_metrics.star_index,
            chain_index=topology_metrics.chain_index
        )

        # Determina topologia
        if topology_metrics.star_index > 0.5:
            mst_topology = "STAR"
        elif topology_metrics.chain_index > 0.7:
            mst_topology = "CHAIN"
        else:
            mst_topology = "MIXED"

        # PASSO 6: DETECÇÃO DE ESTADO
        topology_state = self.topology_analyzer.detect_topology_state(
            topology_metrics,
            spectrum.fiedler_value,
            self.fiedler_history,
            self.ntl_history
        )

        # Fase do mercado
        if mean_correlation < 0.2:
            market_phase = MarketPhase.FRAGMENTED
        elif mean_correlation > 0.6 and topology_state == TopologyState.STAR:
            market_phase = MarketPhase.SYNCHRONIZED
        elif ntl_change < -self.ntl_contraction_threshold:
            market_phase = MarketPhase.COUPLING
        else:
            market_phase = MarketPhase.FRAGMENTED

        # PASSO 7: DETERMINAÇÃO DA DIREÇÃO
        central_name = centrality.central_asset
        leader_direction = "NEUTRAL"

        if central_name in returns_dict:
            central_returns = returns_dict[central_name]
            recent_return = np.mean(central_returns[-10:])

            if recent_return > 0.0001:
                leader_direction = "UP"
            elif recent_return < -0.0001:
                leader_direction = "DOWN"

        # PASSO 8: GERAÇÃO DE SINAL
        signal = 0
        signal_name = "NEUTRAL"
        confidence = 0.0
        reasons = []

        # CONDIÇÃO 1: ESTADO RELAXADO
        if topology_state == TopologyState.RELAXED:
            signal_name = "WAIT"
            reasons.append(f"RELAXADO: NTL={mst.normalized_length:.3f}")
            reasons.append("Baixa vol genuína")

        # CONDIÇÃO 2: CONTRAÇÃO
        elif topology_state == TopologyState.CONTRACTING:
            if market_phase == MarketPhase.COUPLING:
                signal_name = "WAIT"
                confidence = 0.5
                reasons.append(f"CONTRAINDO: NTL caindo ({ntl_change:+.4f})")
                reasons.append("Preparar para breakout")
            else:
                signal_name = "WAIT"
                reasons.append(f"CONTRAINDO: Líder={central_name}")

        # CONDIÇÃO 3: ESTRELA
        elif topology_state == TopologyState.STAR:
            if leader_direction == "UP":
                if central_name in ["SPX500", "USDJPY", "BTC"]:
                    signal = 1
                    signal_name = "LONG"
                    confidence = min(0.8, topology_metrics.star_index + mean_correlation)
                    reasons.append(f"ESTRELA: {central_name} UP (Risk-on)")
                elif central_name in ["XAUUSD", "US10Y"]:
                    signal = -1
                    signal_name = "SHORT"
                    confidence = min(0.8, topology_metrics.star_index + mean_correlation)
                    reasons.append(f"ESTRELA: {central_name} UP (Risk-off)")
                else:
                    signal = 1
                    signal_name = "LONG"
                    confidence = 0.6
                    reasons.append(f"ESTRELA: {central_name} UP")
            elif leader_direction == "DOWN":
                if central_name in ["SPX500", "USDJPY", "BTC"]:
                    signal = -1
                    signal_name = "SHORT"
                    confidence = min(0.8, topology_metrics.star_index + mean_correlation)
                    reasons.append(f"ESTRELA: {central_name} DOWN (Risk-off)")
                elif central_name in ["XAUUSD", "US10Y"]:
                    signal = 1
                    signal_name = "LONG"
                    confidence = min(0.8, topology_metrics.star_index + mean_correlation)
                    reasons.append(f"ESTRELA: {central_name} DOWN (Risk-on)")
                else:
                    signal = -1
                    signal_name = "SHORT"
                    confidence = 0.6
                    reasons.append(f"ESTRELA: {central_name} DOWN")
            else:
                signal_name = "WAIT"
                reasons.append(f"ESTRELA: {central_name} neutro")

        # CONDIÇÃO 4: COLAPSADO
        elif topology_state == TopologyState.COLLAPSED:
            signal_name = "NEUTRAL"
            reasons.append("COLAPSADO: Tarde para entrar")

        else:
            reasons.append(f"Estado={topology_state.value}")

        confidence = np.clip(confidence, 0, 1)

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'topology_state': topology_state.value,
            'market_phase': market_phase.value,
            'ntl': mst.normalized_length,
            'ntl_change': ntl_change,
            'mst_topology': mst_topology,
            'fiedler_value': spectrum.fiedler_value,
            'fiedler_change': fiedler_change,
            'algebraic_connectivity': spectrum.algebraic_connectivity,
            'central_asset': centrality.central_asset,
            'central_node': centrality.central_node,
            'target_centrality': target_centrality,
            'target_distance_to_center': target_distance,
            'mean_correlation': mean_correlation,
            'max_correlation': max_correlation,
            'leader_direction': leader_direction,
            'star_index': topology_metrics.star_index,
            'chain_index': topology_metrics.chain_index,
            'diameter': topology_metrics.diameter,
            'max_degree': topology_metrics.max_degree,
            'n_assets': n_assets,
            'reasons': reasons
        }

    def _create_empty_result(self, signal_name: str) -> dict:
        """Cria resultado vazio"""
        return {
            'signal': 0,
            'signal_name': signal_name,
            'confidence': 0.0,
            'topology_state': TopologyState.RELAXED.value,
            'market_phase': MarketPhase.FRAGMENTED.value,
            'ntl': 0.0,
            'ntl_change': 0.0,
            'mst_topology': "UNKNOWN",
            'fiedler_value': 0.0,
            'fiedler_change': 0.0,
            'algebraic_connectivity': 0.0,
            'central_asset': "Unknown",
            'central_node': -1,
            'target_centrality': 0.0,
            'target_distance_to_center': -1,
            'mean_correlation': 0.0,
            'max_correlation': 0.0,
            'leader_direction': "NEUTRAL",
            'star_index': 0.0,
            'chain_index': 0.0,
            'diameter': 0,
            'max_degree': 0,
            'n_assets': 0,
            'reasons': [signal_name]
        }

    def reset(self):
        """Reseta o indicador"""
        self.fiedler_history.clear()
        self.ntl_history.clear()
        self.central_asset_history.clear()


# ==============================================================================
# DEMONSTRAÇÃO
# ==============================================================================

def generate_multi_asset_data(n_points: int = 200, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Gera dados sintéticos de múltiplos ativos com correlações estruturadas
    """
    np.random.seed(seed)

    # Ativos
    assets = [
        "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
        "XAUUSD", "SPX500", "DAX", "US10Y", "OIL", "BTC"
    ]

    n_assets = len(assets)

    # Fator comum (mercado)
    market_factor = np.cumsum(np.random.randn(n_points) * 0.001)

    # Retornos com correlação estruturada
    returns_dict = {}

    # Betas (sensibilidade ao mercado)
    betas = {
        "EURUSD": 0.3, "USDJPY": -0.4, "GBPUSD": 0.35, "AUDUSD": 0.5, "USDCAD": -0.3,
        "XAUUSD": -0.2, "SPX500": 0.8, "DAX": 0.7, "US10Y": -0.3, "OIL": 0.4, "BTC": 0.6
    }

    for asset in assets:
        beta = betas[asset]
        idiosyncratic = np.random.randn(n_points) * 0.0005
        returns = beta * np.diff(np.concatenate([[0], market_factor])) + idiosyncratic
        returns_dict[asset] = returns

    return returns_dict


def main():
    """Demonstração do indicador GMS-CS"""
    print("=" * 70)
    print("GLOBAL MACRO SPECTRAL COHERENCE SCANNER (GMS-CS)")
    print("Indicador baseado em Teoria Espectral de Grafos")
    print("=" * 70)
    print()

    # Inicializa indicador
    indicator = GlobalMacroSpectralCoherenceScanner(
        correlation_window=100,
        fiedler_threshold=0.3,
        ntl_contraction_threshold=0.1,
        star_threshold=0.4,
        min_history=5
    )

    print("Indicador inicializado!")
    print(f"  - Janela de correlação: 100")
    print(f"  - Threshold Fiedler: 0.3")
    print()

    # Gera dados multi-ativo
    returns_dict = generate_multi_asset_data(n_points=150)
    print(f"Ativos gerados: {list(returns_dict.keys())}")
    print()

    # Processa
    result = indicator.analyze(returns_dict, target_asset="EURUSD")

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Sinal: {result['signal_name']}")
    print(f"Estado Topológico: {result['topology_state']}")
    print(f"Fase de Mercado: {result['market_phase']}")
    print(f"Confiança: {result['confidence']:.2%}")
    print(f"NTL: {result['ntl']:.4f}")
    print(f"Topologia: {result['mst_topology']}")
    print(f"Fiedler (λ₂): {result['fiedler_value']:.4f}")
    print(f"Líder: {result['central_asset']}")
    print(f"Direção: {result['leader_direction']}")
    print(f"Correlação média: {result['mean_correlation']:.4f}")
    print(f"Razões: {result['reasons']}")

    return indicator, result


if __name__ == "__main__":
    indicator, result = main()
