"""
================================================================================
MULTIPLEX VISIBILITY GRAPH & KURAMOTO SYNCHRONIZATION DETECTOR (MVG-KSD)
Indicador de Forex baseado em Redes Complexas e Sincronizacao de Fase
================================================================================

Este algoritmo transforma a serie temporal do EURUSD em uma Rede Complexa
(Complex Network) e mede a Sincronizacao de Fase dos nos dessa rede para
prever rupturas de estabilidade critica (Criticality Breaking Points).

Em media volatilidade, o mercado opera no que chamamos de "Borda do Caos"
(Edge of Chaos). Este indicador detecta o exato momento em que o
sistema escorrega da borda.

A Arquitetura Matematica: De Series Temporais para Grafos
Voce nao vai processar numeros. Voce vai processar a TOPOLOGIA da
conectividade temporal.

Por que isso e "Alien Tech"?
1. Memoria Infinita: O Grafo de Visibilidade conecta o candle de agora com
   um candle de 3 dias atras se houver linha de visao. Ele captura correlacoes
   de longuissima distancia que modelos autoregressivos (ARIMA/GARCH) ignoram.
2. Deteccao de Fragilidade: Redes Complexas robustas sao "Scale-Free". Quando
   o mercado esta prestes a mudar de direcao, a rede do grafo muitas vezes
   muda de topologia (de Scale-Free para Aleatoria ou Regular) ANTES do preco
   virar. E um detector de terremotos tectonicos.
3. Filtragem Geometrica: Ruido de alta frequencia gera nos com baixo grau de
   conexao (ninguem os ve). O indicador pondera naturalmente os pivos
   estruturais e ignora o ruido, pois o ruido e "invisivel" no algoritmo NVG.

Desafio de Implementacao: O calculo de autovalores de matrizes Laplacianas
grandes (Spectral Graph Theory) e custoso. Usa scipy.sparse.linalg.

Autor: Gerado por Claude AI
Versao: 1.0.0
================================================================================
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

class MVGSignalType(Enum):
    """Tipos de sinais de trading"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    WAIT = "WAIT"  # Estado Chimera estavel


class NetworkTopology(Enum):
    """Topologia da rede"""
    SCALE_FREE = "SCALE_FREE"        # Rede robusta (hubs dominantes)
    SMALL_WORLD = "SMALL_WORLD"      # Mundo pequeno (media vol)
    RANDOM = "RANDOM"                 # Rede aleatoria (transicao)
    REGULAR = "REGULAR"               # Rede regular (rigidez)


class SyncState(Enum):
    """Estado de sincronizacao de Kuramoto"""
    INCOHERENT = "INCOHERENT"        # r ~ 0 - Ruido/Baixa vol
    CHIMERA = "CHIMERA"               # 0.3 < r < 0.7 - Media vol (alvo)
    SYNCHRONIZED = "SYNCHRONIZED"     # r ~ 1 - Tendencia forte/Crash


@dataclass
class VisibilityGraph:
    """Grafo de Visibilidade Natural"""
    adjacency_matrix: np.ndarray    # Matriz de adjacencia A
    degree_sequence: np.ndarray     # Sequencia de graus
    n_nodes: int                    # Numero de nos
    n_edges: int                    # Numero de arestas
    density: float                  # Densidade do grafo


@dataclass
class KuramotoState:
    """Estado do modelo de Kuramoto"""
    phases: np.ndarray              # theta_i - fases de cada oscilador
    order_parameter: float          # r - parametro de ordem global
    mean_phase: float               # psi - fase media
    local_order: np.ndarray         # r_i - ordem local de cada no
    sync_state: SyncState           # Estado de sincronizacao


@dataclass
class GraphSpectrum:
    """Espectro do grafo (Laplaciano)"""
    laplacian_eigenvalues: np.ndarray  # lambda_k do Laplaciano
    von_neumann_entropy: float         # S(rho) = -Tr(rho ln rho)
    algebraic_connectivity: float      # lambda_2 (segundo menor autovalor)
    spectral_gap: float                # lambda_2 - lambda_1


@dataclass
class CentralityMetrics:
    """Metricas de centralidade"""
    eigenvector_centrality: np.ndarray  # Centralidade de autovetor
    hub_indices: np.ndarray             # Indices dos hubs
    hub_concentration: float            # Concentracao em hubs
    max_centrality_node: int            # No com maior centralidade


# ==============================================================================
# ALGORITMO DE VISIBILIDADE NATURAL (NVG)
# ==============================================================================

class NaturalVisibilityGraph:
    """
    O Algoritmo de Visibilidade Natural (NVG)

    Primeiro, mapeie a serie de precos y(t) em um grafo G(V, E). Cada candle
    (ou tick) e um no (vertice). Dois nos t_a e t_b sao conectados por uma
    aresta se, e somente se, houver uma "linha de visao" direta entre eles
    no grafico, sem que nenhum candle intermediario t_c obstrua a visao.

    A condicao geometrica de convexidade e:
    y_c < y_b + (y_a - y_b) * (t_b - t_c) / (t_b - t_a)

    Nota de Engenharia: Em Python, isso e O(N^2). Para HFT, voce precisara
    implementar o algoritmo "Divide and Conquer" de Lan et al. para reduzir
    para O(N log N).
    """

    def __init__(self, use_horizontal: bool = False):
        """
        Args:
            use_horizontal: Se True, usa Horizontal Visibility Graph (mais rapido)
        """
        self.use_horizontal = use_horizontal

    def _check_visibility(self,
                         y: np.ndarray,
                         t_a: int,
                         t_b: int) -> bool:
        """
        Verifica se ha visibilidade direta entre nos t_a e t_b

        Condicao: Para todo t_c entre t_a e t_b:
        y_c < y_b + (y_a - y_b) * (t_b - t_c) / (t_b - t_a)
        """
        if abs(t_b - t_a) <= 1:
            return True

        y_a, y_b = y[t_a], y[t_b]

        # Verifica todos os pontos intermediarios
        for t_c in range(min(t_a, t_b) + 1, max(t_a, t_b)):
            y_c = y[t_c]

            # Altura da linha de visao em t_c
            visibility_line = y_b + (y_a - y_b) * (t_b - t_c) / (t_b - t_a)

            if y_c >= visibility_line:
                return False

        return True

    def _check_horizontal_visibility(self,
                                    y: np.ndarray,
                                    t_a: int,
                                    t_b: int) -> bool:
        """
        Versao simplificada: Horizontal Visibility Graph
        Dois pontos se veem se nenhum ponto intermediario e maior que ambos
        """
        if abs(t_b - t_a) <= 1:
            return True

        y_a, y_b = y[t_a], y[t_b]
        min_height = min(y_a, y_b)

        for t_c in range(min(t_a, t_b) + 1, max(t_a, t_b)):
            if y[t_c] >= min_height:
                return False

        return True

    def build_graph(self, series: np.ndarray, max_distance: int = None) -> VisibilityGraph:
        """
        Constroi o grafo de visibilidade natural

        Args:
            series: Serie temporal
            max_distance: Distancia maxima para checar visibilidade (otimizacao)
        """
        n = len(series)

        if max_distance is None:
            max_distance = n

        # Matriz de adjacencia (esparsa)
        adjacency = np.zeros((n, n), dtype=np.int8)

        # Funcao de visibilidade
        check_func = self._check_horizontal_visibility if self.use_horizontal else self._check_visibility

        # Construcao O(N^2) - pode ser otimizado
        for i in range(n):
            for j in range(i + 1, min(i + max_distance, n)):
                if check_func(series, i, j):
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1

        # Metricas
        degree_sequence = np.sum(adjacency, axis=1)
        n_edges = np.sum(adjacency) // 2
        density = 2 * n_edges / (n * (n - 1) + 1e-10)

        return VisibilityGraph(
            adjacency_matrix=adjacency,
            degree_sequence=degree_sequence,
            n_nodes=n,
            n_edges=n_edges,
            density=density
        )

    def build_graph_optimized(self, series: np.ndarray, window: int = 100) -> VisibilityGraph:
        """
        Versao otimizada com janela deslizante
        """
        return self.build_graph(series, max_distance=window)


# ==============================================================================
# MODELO DE KURAMOTO (SINCRONIZACAO DE FASE)
# ==============================================================================

class KuramotoModel:
    """
    O Parametro de Ordem de Kuramoto (r)

    Aqui entra a dinamica. Trate cada no do grafo nao como estatico, mas como
    um Oscilador de Fase Acoplado. A evolucao da fase theta_i de cada no e
    governada pela equacao de Kuramoto, ponderada pela Matriz de Adjacencia
    do grafo construido:

    dtheta_i/dt = omega_i + (K/N) * Sum A_ij * sin(theta_j - theta_i)

    Onde A_ij e 1 se os candles "se veem" e 0 se nao.

    O indicador calcula o Parametro de Ordem Global r(t):

    r(t) * e^(i*psi(t)) = (1/N) * Sum e^(i*theta_j(t))

    - r ~ 0: Incoerencia total (Ruido/Baixa Volatilidade dispersa)
    - r ~ 1: Sincronizacao total (Tendencia forte/Crash iminente)
    - Media Volatilidade (O Alvo): O sistema exibe Estados de Chimera
      (coexistencia de grupos sincronizados e desincronizados)
    """

    def __init__(self,
                 coupling_strength: float = 1.0,
                 dt: float = 0.1,
                 n_iterations: int = 100):
        """
        Args:
            coupling_strength: K - forca de acoplamento
            dt: Passo de tempo para integracao
            n_iterations: Numero de iteracoes para convergencia
        """
        self.coupling_strength = coupling_strength
        self.dt = dt
        self.n_iterations = n_iterations

    def initialize_phases(self,
                         n: int,
                         prices: np.ndarray = None) -> np.ndarray:
        """
        Inicializa fases dos osciladores

        Usa os retornos de preco para definir fases iniciais
        """
        if prices is not None and len(prices) == n:
            # Fase inicial baseada nos retornos
            returns = np.diff(np.log(prices + 1e-10))
            returns = np.append([0], returns)

            # Mapeia retornos para fases [0, 2*pi]
            phases = np.pi + np.arctan(returns * 100)  # Escala para sensibilidade
        else:
            # Fases aleatorias
            phases = np.random.uniform(0, 2 * np.pi, n)

        return phases

    def compute_natural_frequencies(self,
                                   prices: np.ndarray,
                                   base_freq: float = 1.0) -> np.ndarray:
        """
        Computa frequencias naturais omega_i de cada oscilador

        Baseado na volatilidade local
        """
        returns = np.diff(np.log(prices + 1e-10))
        returns = np.append([0], returns)

        # Frequencia proporcional a volatilidade local
        omega = base_freq * (1 + np.abs(returns) * 10)

        return omega

    def evolve(self,
              phases: np.ndarray,
              omega: np.ndarray,
              adjacency: np.ndarray) -> np.ndarray:
        """
        Evolui as fases segundo a equacao de Kuramoto

        dtheta_i/dt = omega_i + (K/N) * Sum A_ij * sin(theta_j - theta_i)
        """
        n = len(phases)
        K = self.coupling_strength

        for _ in range(self.n_iterations):
            # Calcula acoplamento
            phase_diff = phases.reshape(-1, 1) - phases.reshape(1, -1)  # theta_i - theta_j
            coupling = adjacency * np.sin(-phase_diff)  # sin(theta_j - theta_i)
            coupling_sum = np.sum(coupling, axis=1)

            # Atualiza fases
            d_theta = omega + (K / n) * coupling_sum
            phases = phases + self.dt * d_theta

            # Mantem em [0, 2*pi]
            phases = np.mod(phases, 2 * np.pi)

        return phases

    def compute_order_parameter(self, phases: np.ndarray) -> Tuple[float, float]:
        """
        Calcula o parametro de ordem global r e fase media psi

        r * e^(i*psi) = (1/N) * Sum e^(i*theta_j)
        """
        n = len(phases)

        # Soma complexa
        complex_sum = np.sum(np.exp(1j * phases)) / n

        # r = magnitude
        r = np.abs(complex_sum)

        # psi = fase
        psi = np.angle(complex_sum)

        return r, psi

    def compute_local_order(self,
                           phases: np.ndarray,
                           adjacency: np.ndarray) -> np.ndarray:
        """
        Calcula parametro de ordem local para cada no
        """
        n = len(phases)
        local_r = np.zeros(n)

        for i in range(n):
            # Vizinhos do no i
            neighbors = np.where(adjacency[i, :] > 0)[0]

            if len(neighbors) > 0:
                neighbor_phases = phases[neighbors]
                complex_sum = np.sum(np.exp(1j * neighbor_phases)) / len(neighbors)
                local_r[i] = np.abs(complex_sum)
            else:
                local_r[i] = 0.0

        return local_r

    def analyze(self,
               graph: VisibilityGraph,
               prices: np.ndarray) -> KuramotoState:
        """
        Analise completa de Kuramoto
        """
        n = graph.n_nodes

        # Inicializa
        phases = self.initialize_phases(n, prices)
        omega = self.compute_natural_frequencies(prices)

        # Evolui
        phases = self.evolve(phases, omega, graph.adjacency_matrix)

        # Parametro de ordem
        r, psi = self.compute_order_parameter(phases)

        # Ordem local
        local_r = self.compute_local_order(phases, graph.adjacency_matrix)

        # Determina estado
        if r < 0.3:
            sync_state = SyncState.INCOHERENT
        elif r > 0.7:
            sync_state = SyncState.SYNCHRONIZED
        else:
            sync_state = SyncState.CHIMERA

        return KuramotoState(
            phases=phases,
            order_parameter=r,
            mean_phase=psi,
            local_order=local_r,
            sync_state=sync_state
        )


# ==============================================================================
# ESPECTRO DO GRAFO E ENTROPIA DE VON NEUMANN
# ==============================================================================

class GraphSpectralAnalyzer:
    """
    Analise Espectral do Grafo

    Passo 2: Entropia de Von Neumann do Grafo

    Utilize o Laplaciano do Grafo L = D - A, onde D e a matriz de graus.
    Calcule os autovalores lambda_k do Laplaciano. A entropia quantica do grafo e:

    S(rho) = -Tr(rho ln rho)

    Onde rho e a matriz densidade construida a partir do espectro Laplaciano.
    """

    def __init__(self, n_eigenvalues: int = 20):
        """
        Args:
            n_eigenvalues: Numero de autovalores a calcular (para matrizes grandes)
        """
        self.n_eigenvalues = n_eigenvalues

    def compute_laplacian(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Computa o Laplaciano do grafo L = D - A

        Onde D e a matriz diagonal de graus
        """
        degrees = np.sum(adjacency, axis=1)
        D = np.diag(degrees)
        L = D - adjacency

        return L

    def compute_laplacian_eigenvalues(self,
                                     laplacian: np.ndarray,
                                     k: int = None) -> np.ndarray:
        """
        Computa autovalores do Laplaciano

        Usa scipy.sparse.linalg.eigsh para eficiencia
        """
        n = laplacian.shape[0]
        k = k or min(self.n_eigenvalues, n - 2)

        if n < 10:
            # Para grafos pequenos, usa decomposicao completa
            eigenvalues = np.linalg.eigvalsh(laplacian)
        else:
            try:
                # Usa sparse para grafos grandes
                L_sparse = csr_matrix(laplacian)
                eigenvalues, _ = eigsh(L_sparse, k=k, which='SM')
            except:
                # Fallback para decomposicao completa
                eigenvalues = np.linalg.eigvalsh(laplacian)[:k]

        # Ordena
        eigenvalues = np.sort(eigenvalues)

        return eigenvalues

    def compute_von_neumann_entropy(self, eigenvalues: np.ndarray) -> float:
        """
        Computa Entropia de Von Neumann

        S(rho) = -Tr(rho ln rho) = -Sum lambda_k ln lambda_k

        Onde lambda_k sao os autovalores normalizados do Laplaciano
        """
        # Remove autovalores zero ou negativos
        eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]

        if len(eigenvalues_pos) == 0:
            return 0.0

        # Normaliza para formar distribuicao de probabilidade
        total = np.sum(eigenvalues_pos)
        if total < 1e-10:
            return 0.0

        rho = eigenvalues_pos / total

        # Entropia de Von Neumann
        entropy = -np.sum(rho * np.log(rho + 1e-10))

        return entropy

    def analyze(self, graph: VisibilityGraph) -> GraphSpectrum:
        """
        Analise espectral completa
        """
        # Laplaciano
        L = self.compute_laplacian(graph.adjacency_matrix)

        # Autovalores
        eigenvalues = self.compute_laplacian_eigenvalues(L)

        # Entropia de Von Neumann
        entropy = self.compute_von_neumann_entropy(eigenvalues)

        # Conectividade algebrica (segundo menor autovalor)
        algebraic_connectivity = eigenvalues[1] if len(eigenvalues) > 1 else 0.0

        # Gap espectral
        spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0.0

        return GraphSpectrum(
            laplacian_eigenvalues=eigenvalues,
            von_neumann_entropy=entropy,
            algebraic_connectivity=algebraic_connectivity,
            spectral_gap=spectral_gap
        )


# ==============================================================================
# CENTRALIDADE DE AUTOVETOR
# ==============================================================================

class CentralityAnalyzer:
    """
    Passo 1: Calculo da Centralidade de Autovetor (Eigenvector Centrality)

    Calcule o autovetor principal da matriz de adjacencia do grafo atual.
    Os nos com alta centralidade sao "Hubs". Eles sao os pivos do mercado
    (suportes e resistencias reais, definidos por visibilidade, nao por
    toque de preco).
    """

    def __init__(self, hub_threshold: float = 0.8):
        """
        Args:
            hub_threshold: Percentil para considerar hub
        """
        self.hub_threshold = hub_threshold

    def compute_eigenvector_centrality(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Calcula centralidade de autovetor

        O autovetor principal da matriz de adjacencia da a centralidade
        """
        try:
            # Autovalor e autovetor principal
            eigenvalues, eigenvectors = np.linalg.eigh(adjacency)

            # Pega o autovetor do maior autovalor
            idx = np.argmax(eigenvalues)
            centrality = np.abs(eigenvectors[:, idx])

            # Normaliza
            centrality = centrality / (np.max(centrality) + 1e-10)

        except:
            # Fallback: usa grau como proxy
            centrality = np.sum(adjacency, axis=1)
            centrality = centrality / (np.max(centrality) + 1e-10)

        return centrality

    def identify_hubs(self, centrality: np.ndarray) -> np.ndarray:
        """
        Identifica nos hub (alta centralidade)
        """
        threshold = np.percentile(centrality, self.hub_threshold * 100)
        hub_indices = np.where(centrality >= threshold)[0]

        return hub_indices

    def compute_hub_concentration(self, centrality: np.ndarray) -> float:
        """
        Calcula concentracao de centralidade (Gini-like)

        Alta concentracao = poucos hubs dominam
        Baixa concentracao = centralidade distribuida
        """
        sorted_centrality = np.sort(centrality)
        n = len(centrality)

        # Indice de Gini
        cumsum = np.cumsum(sorted_centrality)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_centrality))) / (n * np.sum(sorted_centrality) + 1e-10) - (n + 1) / n

        return gini

    def analyze(self, graph: VisibilityGraph) -> CentralityMetrics:
        """
        Analise completa de centralidade
        """
        centrality = self.compute_eigenvector_centrality(graph.adjacency_matrix)
        hub_indices = self.identify_hubs(centrality)
        hub_concentration = self.compute_hub_concentration(centrality)
        max_centrality_node = np.argmax(centrality)

        return CentralityMetrics(
            eigenvector_centrality=centrality,
            hub_indices=hub_indices,
            hub_concentration=hub_concentration,
            max_centrality_node=max_centrality_node
        )


# ==============================================================================
# DETECTOR DE TOPOLOGIA
# ==============================================================================

class TopologyDetector:
    """
    Detecta a topologia da rede

    - Scale-Free: Distribuicao de graus segue lei de potencia
    - Small-World: Alto clustering, baixo path length
    - Random: Distribuicao de graus Poisson
    - Regular: Todos os nos com grau similar
    """

    def __init__(self):
        pass

    def compute_degree_distribution(self, degrees: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula distribuicao de graus"""
        unique, counts = np.unique(degrees, return_counts=True)
        prob = counts / np.sum(counts)
        return unique, prob

    def detect_scale_free(self, degrees: np.ndarray) -> float:
        """
        Detecta se e scale-free (lei de potencia)

        Retorna score [0, 1]
        """
        unique, prob = self.compute_degree_distribution(degrees)

        if len(unique) < 3:
            return 0.0

        # Log-log linear regression para detectar lei de potencia
        log_k = np.log(unique + 1)
        log_p = np.log(prob + 1e-10)

        # Fit linear
        try:
            coeffs = np.polyfit(log_k, log_p, 1)
            slope = coeffs[0]

            # Scale-free tem slope entre -2 e -3
            if -3.5 < slope < -1.5:
                return min(1.0, abs(slope + 2.5) / 1.5)
            else:
                return 0.0
        except:
            return 0.0

    def compute_clustering_coefficient(self, adjacency: np.ndarray) -> float:
        """Calcula coeficiente de clustering medio"""
        n = adjacency.shape[0]
        clustering = 0.0

        for i in range(n):
            neighbors = np.where(adjacency[i, :] > 0)[0]
            k = len(neighbors)

            if k >= 2:
                # Conta triangulos
                triangles = 0
                for j in range(len(neighbors)):
                    for l in range(j + 1, len(neighbors)):
                        if adjacency[neighbors[j], neighbors[l]] > 0:
                            triangles += 1

                clustering += 2 * triangles / (k * (k - 1))

        return clustering / n

    def detect_topology(self, graph: VisibilityGraph) -> NetworkTopology:
        """
        Detecta topologia predominante
        """
        degrees = graph.degree_sequence

        # Metricas
        scale_free_score = self.detect_scale_free(degrees)
        clustering = self.compute_clustering_coefficient(graph.adjacency_matrix)
        degree_variance = np.var(degrees) / (np.mean(degrees) + 1e-10)

        # Classificacao
        if scale_free_score > 0.5:
            return NetworkTopology.SCALE_FREE
        elif clustering > 0.3 and degree_variance > 1:
            return NetworkTopology.SMALL_WORLD
        elif degree_variance < 0.5:
            return NetworkTopology.REGULAR
        else:
            return NetworkTopology.RANDOM


# ==============================================================================
# INDICADOR MVG-KSD COMPLETO
# ==============================================================================

class MultiplexVisibilityKuramotoDetector:
    """
    Multiplex Visibility Graph & Kuramoto Synchronization Detector (MVG-KSD)

    Indicador completo que usa redes complexas e sincronizacao de fase para
    detectar rupturas de estabilidade critica.

    A Logica Operacional (O Sinal de Ruptura de Simetria):
    Em media volatilidade, o Grafo de Visibilidade mantem uma estrutura de
    "Mundo Pequeno" (Small-World Network) estavel. O indicador busca falhas
    nessa estrutura.

    O Gatilho (Sniper):
    A operacao ocorre na Transicao de Fase Topologica.

    1. Monitoramento: Em media vol, a Entropia de Von Neumann flutua dentro
       de uma banda estavel. O Parametro Kuramoto r oscila entre 0.4 e 0.6
       (Estado Chimera).

    2. SINAL DE AVISO: Ocorre uma queda subita na Entropia do Grafo (o mercado
       se torna "simples" demais) E a Centralidade se concentra em um unico no
       recente (formacao de um Hub instavel).

    3. EXECUCAO (Reversao):
       - LONG: O preco cai rapidamente, mas o Grafo de Visibilidade mostra que
         o no atual perdeu conectividade com o passado (o r de Kuramoto colapsa
         para zero localmente). Isso significa que a queda e "cega" e
         desconectada da estrutura macro. O mercado vai reverter para
         reconectar o grafo.
       - SHORT: O preco sobe, mas a visibilidade futura e bloqueada (o no se
         torna uma "folha" no grafo, sem conexoes de saida). O preco e
         topologicamente obrigado a cair para restaurar a media de
         conectividade da rede (Scale-Free property restoration).
    """

    def __init__(self,
                 # Parametros do grafo
                 visibility_window: int = 50,
                 use_horizontal_visibility: bool = True,

                 # Parametros de Kuramoto
                 kuramoto_coupling: float = 1.0,
                 kuramoto_iterations: int = 50,

                 # Parametros de deteccao
                 entropy_drop_threshold: float = 0.3,
                 hub_concentration_threshold: float = 0.7,
                 leaf_degree_threshold: int = 2,

                 # Geral
                 min_data_points: int = 100):
        """
        Inicializa o MVG-KSD
        """
        self.visibility_window = visibility_window
        self.entropy_drop_threshold = entropy_drop_threshold
        self.hub_concentration_threshold = hub_concentration_threshold
        self.leaf_degree_threshold = leaf_degree_threshold
        self.min_data_points = min_data_points

        # Componentes
        self.nvg_builder = NaturalVisibilityGraph(use_horizontal=use_horizontal_visibility)
        self.kuramoto = KuramotoModel(
            coupling_strength=kuramoto_coupling,
            n_iterations=kuramoto_iterations
        )
        self.spectral_analyzer = GraphSpectralAnalyzer()
        self.centrality_analyzer = CentralityAnalyzer()
        self.topology_detector = TopologyDetector()

        # Historico
        self.entropy_history: List[float] = []
        self.order_param_history: List[float] = []
        self.hub_concentration_history: List[float] = []

    def _detect_entropy_drop(self, current_entropy: float) -> bool:
        """Detecta queda subita na entropia"""
        if len(self.entropy_history) < 5:
            return False

        recent_mean = np.mean(self.entropy_history[-10:])
        drop = (recent_mean - current_entropy) / (recent_mean + 1e-10)

        return drop > self.entropy_drop_threshold

    def _detect_hub_instability(self,
                               centrality: CentralityMetrics,
                               n_nodes: int) -> bool:
        """Detecta formacao de hub instavel (no recente com alta centralidade)"""
        # Verifica se o no mais central e recente (ultimos 10%)
        recent_threshold = int(0.9 * n_nodes)

        if centrality.max_centrality_node >= recent_threshold:
            if centrality.hub_concentration > self.hub_concentration_threshold:
                return True

        return False

    def _check_lost_connectivity(self,
                                local_order: np.ndarray,
                                current_idx: int) -> bool:
        """Verifica se o no atual perdeu conectividade com o passado"""
        if current_idx < 5:
            return False

        # r local colapsa para zero
        return local_order[current_idx] < 0.1

    def _check_leaf_node(self,
                        degree_sequence: np.ndarray,
                        current_idx: int) -> bool:
        """Verifica se o no atual e uma folha (baixo grau)"""
        return degree_sequence[current_idx] <= self.leaf_degree_threshold

    def analyze(self,
               prices: np.ndarray,
               volumes: np.ndarray = None) -> dict:
        """
        Processa dados de mercado e retorna analise completa

        Args:
            prices: Array de precos
            volumes: Array de volumes (opcional)

        Returns:
            Dict com analise completa
        """
        n = len(prices)

        # Validacao
        if n < self.min_data_points:
            return {
                'signal': 0,
                'signal_name': 'INSUFFICIENT_DATA',
                'network_topology': 'RANDOM',
                'sync_state': 'INCOHERENT',
                'confidence': 0.0,
                'order_parameter': 0.0,
                'mean_phase': 0.0,
                'von_neumann_entropy': 0.0,
                'entropy_drop': False,
                'algebraic_connectivity': 0.0,
                'hub_concentration': 0.0,
                'hub_instability': False,
                'current_node_degree': 0,
                'is_leaf_node': False,
                'lost_connectivity': False,
                'n_nodes': n,
                'n_edges': 0,
                'graph_density': 0.0,
                'reasons': ['Dados insuficientes para analise']
            }

        # ============================================================
        # PASSO 1: CONSTRUCAO DO GRAFO DE VISIBILIDADE
        # ============================================================
        graph = self.nvg_builder.build_graph_optimized(prices, self.visibility_window)

        # ============================================================
        # PASSO 2: ANALISE DE KURAMOTO (SINCRONIZACAO)
        # ============================================================
        kuramoto_state = self.kuramoto.analyze(graph, prices)

        self.order_param_history.append(kuramoto_state.order_parameter)
        if len(self.order_param_history) > 100:
            self.order_param_history.pop(0)

        # ============================================================
        # PASSO 3: ANALISE ESPECTRAL (ENTROPIA DE VON NEUMANN)
        # ============================================================
        spectrum = self.spectral_analyzer.analyze(graph)

        self.entropy_history.append(spectrum.von_neumann_entropy)
        if len(self.entropy_history) > 100:
            self.entropy_history.pop(0)

        entropy_drop = self._detect_entropy_drop(spectrum.von_neumann_entropy)

        # ============================================================
        # PASSO 4: ANALISE DE CENTRALIDADE
        # ============================================================
        centrality = self.centrality_analyzer.analyze(graph)

        self.hub_concentration_history.append(centrality.hub_concentration)
        if len(self.hub_concentration_history) > 100:
            self.hub_concentration_history.pop(0)

        hub_instability = self._detect_hub_instability(centrality, n)

        # ============================================================
        # PASSO 5: DETECCAO DE TOPOLOGIA
        # ============================================================
        topology = self.topology_detector.detect_topology(graph)

        # ============================================================
        # PASSO 6: VERIFICACAO DO NO ATUAL
        # ============================================================
        current_idx = n - 1
        current_degree = graph.degree_sequence[current_idx]
        is_leaf = self._check_leaf_node(graph.degree_sequence, current_idx)
        lost_connectivity = self._check_lost_connectivity(
            kuramoto_state.local_order, current_idx
        )

        # ============================================================
        # PASSO 7: GERACAO DE SINAL
        # ============================================================
        # Verifica movimento de preco recente
        price_change = (prices[-1] - prices[-5]) / prices[-5] if n > 5 else 0

        signal = 0
        signal_name = 'NEUTRAL'
        confidence = 0.0
        reasons = []

        # Estado Chimera estavel - esperar
        if kuramoto_state.sync_state == SyncState.CHIMERA and not entropy_drop and not hub_instability:
            signal_name = 'WAIT'
            reasons.append(f'Estado Chimera estavel (r={kuramoto_state.order_parameter:.3f})')
            reasons.append(f'Entropia estavel (S={spectrum.von_neumann_entropy:.3f})')

        # Sincronizacao total - cuidado com crash
        elif kuramoto_state.sync_state == SyncState.SYNCHRONIZED:
            if price_change > 0.001:
                signal = -1
                signal_name = 'SHORT'
                confidence = min(1.0, kuramoto_state.order_parameter)
                reasons.append(f'Sincronizacao Total (r={kuramoto_state.order_parameter:.3f})')
                reasons.append('Preco subindo em regime critico. Crash iminente.')
            else:
                reasons.append(f'Sincronizacao alta mas sem setup claro.')

        # SINAL DE SNIPER: Entropia cai + Hub instavel
        elif entropy_drop and hub_instability:
            if price_change > 0:
                signal = -1
                signal_name = 'SHORT'
                confidence = 0.8
                reasons.append('Ruptura de Simetria: Entropia caiu + hub instavel')
                reasons.append('Preco subindo mas estrutura fragilizada')
            else:
                signal = 1
                signal_name = 'LONG'
                confidence = 0.8
                reasons.append('Ruptura de Simetria: Entropia caiu + hub instavel')
                reasons.append('Preco caindo mas estrutura vai reconectar')

        # LONG: No atual perdeu conectividade (queda cega)
        elif lost_connectivity and price_change < -0.0005:
            signal = 1
            signal_name = 'LONG'
            confidence = min(0.9, 1 - kuramoto_state.local_order[current_idx])
            reasons.append(f'Queda Cega: No desconectado (r_local={kuramoto_state.local_order[current_idx]:.3f})')
            reasons.append('Reversao para reconectar grafo')

        # SHORT: No atual e folha (sem visibilidade futura)
        elif is_leaf and price_change > 0.0005:
            signal = -1
            signal_name = 'SHORT'
            confidence = 0.7
            reasons.append(f'No Folha: Grau={current_degree}')
            reasons.append('Visibilidade bloqueada. Preco obrigado a cair.')

        # Incoerencia - ruido
        elif kuramoto_state.sync_state == SyncState.INCOHERENT:
            reasons.append(f'Incoerencia total (r={kuramoto_state.order_parameter:.3f}). Ruido dominante.')

        else:
            reasons.append(f'Sem setup claro. r={kuramoto_state.order_parameter:.3f}')

        # Ajusta confianca pela topologia
        if confidence > 0:
            if topology == NetworkTopology.SCALE_FREE:
                confidence *= 1.2  # Rede robusta, sinal mais confiavel
            elif topology == NetworkTopology.RANDOM:
                confidence *= 0.7  # Rede instavel, menos confiavel
            confidence = np.clip(confidence, 0, 1)

        return {
            'signal': signal,
            'signal_name': signal_name,
            'network_topology': topology.value,
            'sync_state': kuramoto_state.sync_state.value,
            'confidence': confidence,
            'order_parameter': kuramoto_state.order_parameter,
            'mean_phase': kuramoto_state.mean_phase,
            'von_neumann_entropy': spectrum.von_neumann_entropy,
            'entropy_drop': entropy_drop,
            'algebraic_connectivity': spectrum.algebraic_connectivity,
            'spectral_gap': spectrum.spectral_gap,
            'hub_concentration': centrality.hub_concentration,
            'hub_instability': hub_instability,
            'max_centrality_node': centrality.max_centrality_node,
            'current_node_degree': current_degree,
            'is_leaf_node': is_leaf,
            'lost_connectivity': lost_connectivity,
            'n_nodes': graph.n_nodes,
            'n_edges': graph.n_edges,
            'graph_density': graph.density,
            'reasons': reasons
        }

    def get_entropy_history(self) -> np.ndarray:
        """Retorna historico de entropia"""
        return np.array(self.entropy_history)

    def get_order_param_history(self) -> np.ndarray:
        """Retorna historico do parametro de ordem"""
        return np.array(self.order_param_history)

    def get_hub_concentration_history(self) -> np.ndarray:
        """Retorna historico de concentracao de hubs"""
        return np.array(self.hub_concentration_history)

    def reset(self):
        """Reseta o estado do indicador"""
        self.entropy_history.clear()
        self.order_param_history.clear()
        self.hub_concentration_history.clear()


# ==============================================================================
# DEMONSTRACAO
# ==============================================================================

def generate_chimera_data(n_points: int = 200, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Gera dados com estrutura de Chimera (grupos sincronizados e nao)"""
    np.random.seed(seed)

    t = np.arange(n_points)

    # Base
    base = 1.0850

    # Componente sincronizado (tendencia)
    trend = 0.0001 * np.sin(2 * np.pi * t / 50)

    # Componente caotico
    chaos = np.cumsum(np.random.randn(n_points) * 0.0001)

    # Mistura (Chimera)
    prices = base + trend + chaos

    # Volume
    volumes = 1000 + np.abs(np.diff(np.append(prices, prices[-1]))) * 100000
    volumes += np.random.randn(n_points) * 100

    return prices, volumes


def main():
    """Demonstracao do indicador MVG-KSD"""
    print("=" * 70)
    print("MULTIPLEX VISIBILITY GRAPH & KURAMOTO SYNCHRONIZATION DETECTOR")
    print("Indicador baseado em Redes Complexas")
    print("=" * 70)
    print()

    # Inicializa indicador
    indicator = MultiplexVisibilityKuramotoDetector(
        visibility_window=30,
        use_horizontal_visibility=True,
        kuramoto_coupling=1.0,
        kuramoto_iterations=30,
        entropy_drop_threshold=0.3,
        hub_concentration_threshold=0.7,
        min_data_points=100
    )

    print("Indicador inicializado!")
    print(f"  - Visibility window: 30")
    print(f"  - Kuramoto coupling: 1.0")
    print(f"  - Entropy drop threshold: 0.3")
    print()

    # Gera dados
    prices, volumes = generate_chimera_data(n_points=150)
    print(f"Dados gerados: {len(prices)} pontos")
    print()

    # Processa
    result = indicator.analyze(prices, volumes)

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Sinal: {result['signal_name']}")
    print(f"Topologia: {result['network_topology']}")
    print(f"Estado Sync: {result['sync_state']}")
    print(f"Confianca: {result['confidence']:.2%}")
    print(f"\nKuramoto:")
    print(f"  r (ordem): {result['order_parameter']:.4f}")
    print(f"  psi (fase): {result['mean_phase']:.4f}")
    print(f"\nEspectro:")
    print(f"  Entropia Von Neumann: {result['von_neumann_entropy']:.4f}")
    print(f"  Queda de entropia: {result['entropy_drop']}")
    print(f"  lambda_2 (conectividade): {result['algebraic_connectivity']:.4f}")
    print(f"\nCentralidade:")
    print(f"  Concentracao hubs: {result['hub_concentration']:.4f}")
    print(f"  Hub instavel: {result['hub_instability']}")
    print(f"\nNo atual:")
    print(f"  Grau: {result['current_node_degree']}")
    print(f"  E folha: {result['is_leaf_node']}")
    print(f"  Perdeu conectividade: {result['lost_connectivity']}")
    print(f"\nGrafo:")
    print(f"  Nos: {result['n_nodes']}")
    print(f"  Arestas: {result['n_edges']}")
    print(f"  Densidade: {result['graph_density']:.4f}")
    print(f"\nRazoes: {result['reasons']}")

    return indicator, result


if __name__ == "__main__":
    indicator, result = main()
