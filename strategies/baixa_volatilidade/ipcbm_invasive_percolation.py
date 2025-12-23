"""
================================================================================
INVASIVE PERCOLATION & CAPILLARY BREAKTHROUGH MONITOR (IP-CBM)
Indicador de Forex baseado em Fisica de Escoamento em Meios Porosos
================================================================================

Este indicador simula a injecao de um "fluido nao-molhante" (agressao de mercado)
em um meio poroso desordenado (o livro de ofertas). Ele utiliza a Fisica de
Escoamento em Meios Porosos para identificar onde e quando a "barragem" vai estourar.

A Fisica: Lei de Darcy e Invasao Capilar
Nao olharemos para medias. Olharemos para a Topologia dos Poros.

Por que usar Percolacao?
1. Geometria Realista: O mercado nao e continuo. Ele e cheio de buracos. A Percolacao
   e a unica teoria matematica feita para lidar com conectividade em meios esburacados.

2. Previsao de Caminho: Enquanto outros indicadores dizem "vai subir", o IP-CBM diz
   "vai subir por ESTE caminho de precos especificos", permitindo que voce coloque
   seus Take Profits exatamente onde a permeabilidade muda.

3. Avalanche Effect: Em baixa volatilidade, pequenas mudancas na estrutura de poros
   podem conectar dois clusters gigantes instantaneamente. O indicador ve essa
   conexao ANTES do preco passar por ela.

A Logica de Trading (Instabilidade de Saffman-Taylor)
O indicador busca um fenomeno especifico da dinamica de fluidos: o Viscous Fingering
(Dedos Viscosos). Quando um fluido menos viscoso (agressao rapida) empurra um fluido
mais viscoso (book pesado), a interface se torna instavel e forma "dedos" longos e
finos que penetram o meio rapidamente.

Autor: Claude AI
Versao: 1.0.0
================================================================================
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import warnings
import logging
import heapq

# Configuracao de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


# ==============================================================================
# CONSTANTES FISICAS
# ==============================================================================

# Dimensao fractal da percolacao 2D padrao
FRACTAL_DIMENSION_2D = 1.89

# Expoentes criticos de percolacao
PERCOLATION_EXPONENTS = {
    'nu': 4/3,      # Expoente do comprimento de correlacao
    'beta': 5/36,   # Expoente do parametro de ordem
    'gamma': 43/18, # Expoente da susceptibilidade
    'tau': 187/91,  # Expoente da distribuicao de clusters
}

# Limiar de percolacao para rede quadrada 2D
PERCOLATION_THRESHOLD_2D = 0.592746


# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

class ClusterShape(Enum):
    """Forma do cluster de invasao"""
    COMPACT = "COMPACT"           # Esferico, compacto (saturacao)
    FINGERING = "FINGERING"       # Dedos viscosos se formando
    BREAKTHROUGH = "BREAKTHROUGH" # Breakthrough iminente
    PERCOLATED = "PERCOLATED"     # Ja percolou


class FluidState(Enum):
    """Estado do fluido no sistema"""
    TRAPPED = "TRAPPED"           # Preso por capilaridade
    ADVANCING = "ADVANCING"       # Avancando lentamente
    FINGERING = "FINGERING"       # Formando dedos
    BREAKTHROUGH = "BREAKTHROUGH" # Rompimento


@dataclass
class PoreCell:
    """Uma celula do meio poroso (nivel de preco)"""
    price_level: int              # Indice do nivel de preco
    time_index: int               # Indice temporal
    volume: float                 # Volume de ordens neste nivel
    permeability: float           # k - permeabilidade local
    capillary_pressure: float     # Pc - pressao capilar de entrada
    is_invaded: bool              # True se ja foi invadido pelo fluido
    invasion_order: int           # Ordem em que foi invadido (-1 se nao)
    is_bid: bool                  # True se e nivel de bid


@dataclass
class InvasionCluster:
    """Cluster de invasao percolativa"""
    cells: Set[Tuple[int, int]]   # Celulas invadidas (price_idx, time_idx)
    perimeter: Set[Tuple[int, int]]  # Celulas na interface
    mass: int                     # Numero de celulas invadidas
    radius_of_gyration: float     # Raio de giracao
    fractal_dimension: float      # Dimensao fractal
    shape: ClusterShape           # Forma atual
    finger_length: float          # Comprimento do maior dedo
    finger_direction: str         # "UP" ou "DOWN"
    aspect_ratio: float           # Razao de aspecto (comprimento/largura)


@dataclass
class PorosityField:
    """Campo de porosidade do order book"""
    grid: np.ndarray              # Grid 2D de permeabilidades
    capillary_pressures: np.ndarray  # Grid 2D de pressoes capilares
    pore_sizes: np.ndarray        # Distribuicao de tamanhos de poros
    porosity: float               # Porosidade media phi
    permeability_mean: float      # Permeabilidade media k
    surface_tension: float        # gamma - tensao superficial (aversao ao risco)
    contact_angle: float          # theta - angulo de contato


@dataclass
class BreakthroughPath:
    """Caminho de breakthrough (percolacao)"""
    path: List[Tuple[int, int]]   # Sequencia de celulas
    total_resistance: float       # Resistencia hidraulica total
    entry_price: float            # Preco de entrada
    target_price: float           # Preco alvo (onde permeabilidade muda)
    critical_length: float        # L_crit
    is_complete: bool             # True se conecta entrada a saida


@dataclass
class SaffmanTaylorAnalysis:
    """Analise de instabilidade de Saffman-Taylor"""
    mobility_ratio: float         # M = mu_displaced / mu_displacing
    capillary_number: float       # Ca = mu*v/gamma
    finger_width: float           # Largura do dedo (fracao do canal)
    growth_rate: float            # Taxa de crescimento do dedo
    is_unstable: bool             # True se interface e instavel
    instability_wavelength: float # Comprimento de onda mais instavel


# ==============================================================================
# UNION-FIND (DISJOINT SET) PARA CLUSTERS
# ==============================================================================

class UnionFind:
    """
    Estrutura Union-Find otimizada para gerenciar clusters de percolacao

    Complexidade: O(alpha(N)) por operacao, onde alpha e a funcao inversa de Ackermann
    (praticamente constante para todos os propositos praticos)
    """

    def __init__(self, n: int):
        """Inicializa n elementos, cada um em seu proprio conjunto"""
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.n_components = n

    def find(self, x: int) -> int:
        """Encontra o representante do conjunto de x com compressao de caminho"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Une os conjuntos de x e y usando union by rank"""
        px, py = self.find(x), self.find(y)

        if px == py:
            return False

        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.parent[py] = px
        self.size[px] += self.size[py]

        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        self.n_components -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """Verifica se x e y estao no mesmo conjunto"""
        return self.find(x) == self.find(y)

    def get_size(self, x: int) -> int:
        """Retorna o tamanho do conjunto de x"""
        return self.size[self.find(x)]

    def get_components(self) -> Dict[int, List[int]]:
        """Retorna todos os componentes"""
        components = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)
        return components


# ==============================================================================
# CONSTRUTOR DO CAMPO DE POROSIDADE
# ==============================================================================

class PorosityFieldBuilder:
    """
    Passo 1: Construcao do Campo de Porosidade

    1. Mapeamento da Permeabilidade (k)
    Transforme o Depth of Market (DOM) em um reticulado (Grid) 2D/3D. Cada celula
    do grid tem uma Permeabilidade Local (k) inversamente proporcional ao volume
    de ordens naquele nivel de preco.

    - Muito volume (Wall) = Permeabilidade Baixa (Rocha impermeavel)
    - Pouco volume (Gap) = Permeabilidade Alta (Areia solta)

    2. Pressao Capilar de Entrada (Pc)
    Para o preco se mover de um nivel i para j, a "forca" de compra deve superar
    a pressao capilar da "garganta do poro" (o spread/liquidez entre os niveis).
    A Lei de Laplace define isso:

    Pc proporcional a gamma*cos(theta) / r
    """

    def __init__(self,
                 n_price_levels: int = 50,
                 n_time_steps: int = 50,
                 base_surface_tension: float = 1.0,
                 contact_angle: float = 120.0):
        """
        Args:
            n_price_levels: Numero de niveis de preco no grid
            n_time_steps: Numero de passos temporais
            base_surface_tension: gamma base (aversao ao risco)
            contact_angle: theta em graus (>90 para fluido nao-molhante)
        """
        self.n_price_levels = n_price_levels
        self.n_time_steps = n_time_steps
        self.base_surface_tension = base_surface_tension
        self.contact_angle_rad = np.radians(contact_angle)

    def build_from_ohlcv(self,
                        prices: np.ndarray,
                        volumes: np.ndarray,
                        current_price: float) -> PorosityField:
        """Constroi o campo de porosidade a partir de dados OHLCV"""
        n = len(prices)

        # Define range de precos
        price_min = np.min(prices) * 0.999
        price_max = np.max(prices) * 1.001
        price_levels = np.linspace(price_min, price_max, self.n_price_levels)

        # Inicializa grids
        permeability_grid = np.ones((self.n_price_levels, self.n_time_steps))
        capillary_grid = np.ones((self.n_price_levels, self.n_time_steps))

        # Mapeia volumes para niveis de preco
        time_step = max(1, n // self.n_time_steps)

        for t in range(self.n_time_steps):
            t_start = t * time_step
            t_end = min(t_start + time_step, n)

            if t_start >= n:
                break

            period_volumes = volumes[t_start:t_end]
            period_prices = prices[t_start:t_end]

            avg_vol = np.mean(period_volumes)

            for i, price_level in enumerate(price_levels):
                distances = np.abs(period_prices - price_level)
                weights = np.exp(-distances / (price_max - price_min) * 10)
                weighted_volume = np.sum(period_volumes * weights) / (np.sum(weights) + 1e-10)

                k = 1.0 / (1 + weighted_volume / (avg_vol + 1e-10))
                permeability_grid[i, t] = k

        # Calcula tensao superficial adaptativa
        returns = np.abs(np.diff(np.log(prices + 1e-10)))
        volatility = np.std(returns) * np.sqrt(252)
        surface_tension = self.base_surface_tension * (1 + volatility * 5)

        # Calcula pressoes capilares (Lei de Laplace)
        cos_theta = np.cos(self.contact_angle_rad)

        for i in range(self.n_price_levels):
            for t in range(self.n_time_steps):
                r = permeability_grid[i, t] + 0.01
                Pc = abs(surface_tension * cos_theta / r)
                capillary_grid[i, t] = Pc

        pore_sizes = permeability_grid.flatten()
        porosity = np.mean(permeability_grid > np.median(permeability_grid))
        permeability_mean = np.mean(permeability_grid)

        return PorosityField(
            grid=permeability_grid,
            capillary_pressures=capillary_grid,
            pore_sizes=pore_sizes,
            porosity=porosity,
            permeability_mean=permeability_mean,
            surface_tension=surface_tension,
            contact_angle=np.degrees(self.contact_angle_rad)
        )

    def identify_walls_and_gaps(self,
                               porosity_field: PorosityField) -> Tuple[np.ndarray, np.ndarray]:
        """Identifica walls (baixa permeabilidade) e gaps (alta permeabilidade)"""
        grid = porosity_field.grid
        threshold = np.median(grid)

        walls = grid < threshold * 0.5
        gaps = grid > threshold * 1.5

        return walls, gaps


# ==============================================================================
# SIMULADOR DE PERCOLACAO INVASIVA
# ==============================================================================

class InvasivePercolationSimulator:
    """
    Passo 2: Simulacao de Invasao em Tempo Real

    O Algoritmo de Invasao (Wilkinson-Willemsen)
    Diferente da difusao normal, a Percolacao Invasiva avanca APENAS pelo poro de
    menor resistencia na interface fluido-solido.
    """

    def __init__(self,
                 max_invasion_steps: int = 1000,
                 trapping_enabled: bool = True):
        self.max_invasion_steps = max_invasion_steps
        self.trapping_enabled = trapping_enabled

    def simulate_invasion(self,
                         porosity_field: PorosityField,
                         injection_point: Tuple[int, int],
                         direction: str = "UP") -> InvasionCluster:
        """
        Simula percolacao invasiva a partir de um ponto de injecao

        Algoritmo de Wilkinson-Willemsen:
        1. Identifica interface (perimeter)
        2. Encontra celula com menor Pc na interface
        3. Invade essa celula
        4. Atualiza interface
        5. Repete
        """
        n_rows, n_cols = porosity_field.grid.shape
        capillary = porosity_field.capillary_pressures

        invaded = set()
        invaded.add(injection_point)

        interface_heap = []

        for neighbor in self._get_neighbors(injection_point, n_rows, n_cols):
            if neighbor not in invaded:
                Pc = capillary[neighbor]
                heapq.heappush(interface_heap, (Pc, neighbor[0], neighbor[1]))

        in_interface = set([(h[1], h[2]) for h in interface_heap])

        invasion_order = {injection_point: 0}
        step = 1

        while interface_heap and step < self.max_invasion_steps:
            Pc_min, row, col = heapq.heappop(interface_heap)
            cell = (row, col)

            if cell in invaded:
                continue

            in_interface.discard(cell)
            invaded.add(cell)
            invasion_order[cell] = step
            step += 1

            for neighbor in self._get_neighbors(cell, n_rows, n_cols):
                if neighbor not in invaded and neighbor not in in_interface:
                    Pc = capillary[neighbor]
                    heapq.heappush(interface_heap, (Pc, neighbor[0], neighbor[1]))
                    in_interface.add(neighbor)

            if direction == "UP" and row == 0:
                break
            elif direction == "DOWN" and row == n_rows - 1:
                break

        perimeter = set()
        for cell in invaded:
            for neighbor in self._get_neighbors(cell, n_rows, n_cols):
                if neighbor not in invaded:
                    perimeter.add(neighbor)

        cluster = self._analyze_cluster(invaded, perimeter, n_rows, n_cols, direction)

        return cluster

    def _get_neighbors(self,
                      cell: Tuple[int, int],
                      n_rows: int,
                      n_cols: int) -> List[Tuple[int, int]]:
        """Retorna vizinhos validos de uma celula (4-conectividade)"""
        row, col = cell
        neighbors = []

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols:
                neighbors.append((nr, nc))

        return neighbors

    def _analyze_cluster(self,
                        invaded: Set[Tuple[int, int]],
                        perimeter: Set[Tuple[int, int]],
                        n_rows: int,
                        n_cols: int,
                        direction: str) -> InvasionCluster:
        """Analisa propriedades do cluster de invasao"""

        if not invaded:
            return InvasionCluster(
                cells=set(),
                perimeter=set(),
                mass=0,
                radius_of_gyration=0.0,
                fractal_dimension=0.0,
                shape=ClusterShape.COMPACT,
                finger_length=0.0,
                finger_direction=direction,
                aspect_ratio=1.0
            )

        cells_array = np.array(list(invaded))
        mass = len(invaded)

        center = np.mean(cells_array, axis=0)

        distances_sq = np.sum((cells_array - center)**2, axis=1)
        Rg = np.sqrt(np.mean(distances_sq)) if mass > 1 else 1.0

        if Rg > 1:
            Df = np.log(mass) / np.log(Rg)
        else:
            Df = 2.0

        min_row, max_row = np.min(cells_array[:, 0]), np.max(cells_array[:, 0])
        min_col, max_col = np.min(cells_array[:, 1]), np.max(cells_array[:, 1])

        height = max_row - min_row + 1
        width = max_col - min_col + 1

        aspect_ratio = height / (width + 1e-10)

        if direction == "UP":
            finger_length = n_rows - min_row
        else:
            finger_length = max_row

        if aspect_ratio < 1.5 and Df > 1.7:
            shape = ClusterShape.COMPACT
        elif aspect_ratio > 2.0 or Df < 1.5:
            shape = ClusterShape.FINGERING
        elif finger_length > n_rows * 0.8:
            shape = ClusterShape.BREAKTHROUGH
        else:
            shape = ClusterShape.COMPACT

        return InvasionCluster(
            cells=invaded,
            perimeter=perimeter,
            mass=mass,
            radius_of_gyration=Rg,
            fractal_dimension=Df,
            shape=shape,
            finger_length=finger_length,
            finger_direction=direction,
            aspect_ratio=aspect_ratio
        )


# ==============================================================================
# CALCULADOR DE DIMENSAO FRACTAL
# ==============================================================================

class FractalDimensionCalculator:
    """
    Passo 3: Calculo da Dimensao Fractal (Df)

    Meca a massa do cluster (M) em relacao ao seu raio de giracao (Rg):
    M proporcional a Rg^Df

    - Em percolacao normal 2D, Df aproximadamente 1.89
    - Se Df muda, a geometria do mercado mudou
    """

    def __init__(self):
        self.reference_Df = FRACTAL_DIMENSION_2D

    def calculate_box_counting_dimension(self,
                                        cells: Set[Tuple[int, int]],
                                        max_box_size: int = 32) -> float:
        """Calcula dimensao fractal usando box-counting"""
        if len(cells) < 10:
            return 2.0

        cells_array = np.array(list(cells))

        min_coords = np.min(cells_array, axis=0)
        max_coords = np.max(cells_array, axis=0)
        range_coords = max_coords - min_coords + 1

        normalized = (cells_array - min_coords) / range_coords

        box_sizes = []
        box_counts = []

        for size in [2, 4, 8, 16, 32]:
            if size > max_box_size:
                break

            epsilon = 1.0 / size
            boxes = set()

            for point in normalized:
                box_idx = tuple((point / epsilon).astype(int))
                boxes.add(box_idx)

            if len(boxes) > 0:
                box_sizes.append(epsilon)
                box_counts.append(len(boxes))

        if len(box_sizes) < 2:
            return 2.0

        log_eps = np.log(box_sizes)
        log_N = np.log(box_counts)

        slope, _ = np.polyfit(log_eps, log_N, 1)
        Df = -slope

        return np.clip(Df, 1.0, 2.0)

    def calculate_mass_radius_dimension(self,
                                       cells: Set[Tuple[int, int]]) -> float:
        """Calcula Df usando relacao massa-raio"""
        if len(cells) < 10:
            return 2.0

        cells_array = np.array(list(cells))
        center = np.mean(cells_array, axis=0)

        distances = np.sqrt(np.sum((cells_array - center)**2, axis=1))

        radii = np.linspace(1, np.max(distances), 20)
        masses = []

        for r in radii:
            mass = np.sum(distances <= r)
            if mass > 0:
                masses.append(mass)
            else:
                masses.append(1)

        valid_idx = np.array(masses) > 1
        if np.sum(valid_idx) < 2:
            return 2.0

        radii = radii[valid_idx]
        masses = np.array(masses)[valid_idx]

        log_r = np.log(radii)
        log_M = np.log(masses)

        slope, _ = np.polyfit(log_r, log_M, 1)
        Df = slope

        return np.clip(Df, 1.0, 2.0)

    def detect_geometry_change(self, Df_current: float) -> Tuple[bool, float]:
        """Detecta se a geometria do mercado mudou"""
        deviation = abs(Df_current - self.reference_Df)
        changed = deviation > 0.2

        return changed, deviation


# ==============================================================================
# ANALISADOR DE SAFFMAN-TAYLOR (VISCOUS FINGERING)
# ==============================================================================

class SaffmanTaylorAnalyzer:
    """
    A Logica de Trading (Instabilidade de Saffman-Taylor)

    O indicador busca o Viscous Fingering (Dedos Viscosos). Quando um fluido
    menos viscoso (agressao rapida) empurra um fluido mais viscoso (book pesado),
    a interface se torna instavel e forma "dedos" longos e finos.
    """

    def __init__(self,
                 critical_mobility_ratio: float = 1.0,
                 critical_capillary_number: float = 0.01,
                 finger_threshold: float = 0.3):
        self.critical_mobility_ratio = critical_mobility_ratio
        self.critical_capillary_number = critical_capillary_number
        self.finger_threshold = finger_threshold

    def analyze_stability(self,
                         cluster: InvasionCluster,
                         porosity_field: PorosityField,
                         injection_rate: float = 1.0) -> SaffmanTaylorAnalysis:
        """Analisa estabilidade da interface usando teoria de Saffman-Taylor"""

        mu_book = 1.0 / (porosity_field.permeability_mean + 0.01)
        mu_aggression = 0.1

        mobility_ratio = mu_book / mu_aggression

        velocity = injection_rate * porosity_field.permeability_mean
        gamma = porosity_field.surface_tension

        capillary_number = mu_aggression * velocity / (gamma + 0.01)

        if capillary_number < 0.01:
            finger_width = 0.5
        else:
            finger_width = 1.0 / (1 + 4 * capillary_number)

        growth_rate = cluster.aspect_ratio * mobility_ratio

        is_unstable = (mobility_ratio > self.critical_mobility_ratio and
                      capillary_number < self.critical_capillary_number)

        if velocity > 0:
            wavelength = np.sqrt(gamma / (mu_aggression * velocity + 0.01))
        else:
            wavelength = float('inf')

        return SaffmanTaylorAnalysis(
            mobility_ratio=mobility_ratio,
            capillary_number=capillary_number,
            finger_width=finger_width,
            growth_rate=growth_rate,
            is_unstable=is_unstable,
            instability_wavelength=wavelength
        )

    def detect_finger_formation(self, cluster: InvasionCluster) -> Tuple[bool, float]:
        """Detecta se um dedo se formou no cluster"""
        finger_formed = (cluster.shape == ClusterShape.FINGERING or
                        cluster.aspect_ratio > 2.0)

        finger_progress = min(1.0, cluster.finger_length / 50.0)

        return finger_formed, finger_progress

    def calculate_critical_length(self,
                                 porosity_field: PorosityField,
                                 cluster: InvasionCluster) -> float:
        """Calcula o comprimento critico L_crit"""
        n_rows = porosity_field.grid.shape[0]

        permeability_profile = np.mean(porosity_field.grid, axis=1)
        threshold = np.percentile(permeability_profile, 30)

        dense_layers = permeability_profile < threshold

        if cluster.finger_direction == "UP":
            cluster_rows = [c[0] for c in cluster.cells]
            min_row = min(cluster_rows) if cluster_rows else n_rows // 2
            L_crit = np.sum(dense_layers[:min_row])
        else:
            cluster_rows = [c[0] for c in cluster.cells]
            max_row = max(cluster_rows) if cluster_rows else n_rows // 2
            L_crit = np.sum(dense_layers[max_row:])

        return max(L_crit, 5.0)


# ==============================================================================
# ENCONTRADOR DE CAMINHO DE BREAKTHROUGH
# ==============================================================================

class BreakthroughPathFinder:
    """Encontra o caminho de menor resistencia para breakthrough"""

    def __init__(self):
        pass

    def find_path(self,
                 porosity_field: PorosityField,
                 start: Tuple[int, int],
                 direction: str = "UP") -> BreakthroughPath:
        """Encontra caminho de menor resistencia usando Dijkstra"""
        n_rows, n_cols = porosity_field.grid.shape
        capillary = porosity_field.capillary_pressures

        distances = np.full((n_rows, n_cols), np.inf)
        distances[start] = 0
        previous = {}

        pq = [(0, start[0], start[1])]
        visited = set()

        if direction == "UP":
            targets = [(0, c) for c in range(n_cols)]
        else:
            targets = [(n_rows - 1, c) for c in range(n_cols)]

        while pq:
            dist, row, col = heapq.heappop(pq)
            cell = (row, col)

            if cell in visited:
                continue
            visited.add(cell)

            if cell in targets:
                path = [cell]
                current = cell
                while current in previous:
                    current = previous[current]
                    path.append(current)
                path.reverse()

                return BreakthroughPath(
                    path=path,
                    total_resistance=dist,
                    entry_price=start[0],
                    target_price=cell[0],
                    critical_length=len(path),
                    is_complete=True
                )

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    neighbor = (nr, nc)
                    if neighbor not in visited:
                        resistance = capillary[nr, nc]
                        new_dist = dist + resistance

                        if new_dist < distances[neighbor]:
                            distances[neighbor] = new_dist
                            previous[neighbor] = cell
                            heapq.heappush(pq, (new_dist, nr, nc))

        return BreakthroughPath(
            path=[start],
            total_resistance=float('inf'),
            entry_price=start[0],
            target_price=start[0],
            critical_length=0,
            is_complete=False
        )

    def find_permeability_changes(self,
                                 porosity_field: PorosityField,
                                 path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Encontra pontos onde a permeabilidade muda significativamente"""
        if len(path) < 2:
            return []

        changes = []
        threshold = 0.3

        for i in range(1, len(path)):
            prev_perm = porosity_field.grid[path[i-1]]
            curr_perm = porosity_field.grid[path[i]]

            change = abs(curr_perm - prev_perm) / (prev_perm + 0.01)

            if change > threshold:
                changes.append(path[i])

        return changes


# ==============================================================================
# INDICADOR IP-CBM COMPLETO
# ==============================================================================

class InvasivePercolationCapillaryBreakthroughMonitor:
    """
    INVASIVE PERCOLATION & CAPILLARY BREAKTHROUGH MONITOR (IP-CBM)

    Indicador completo que usa Fisica de Escoamento em Meios Porosos para
    identificar onde e quando a "barragem" vai estourar.
    """

    def __init__(self,
                 n_price_levels: int = 50,
                 n_time_steps: int = 50,
                 base_surface_tension: float = 1.0,
                 contact_angle: float = 120.0,
                 max_invasion_steps: int = 500,
                 stop_loss_buffer: float = 0.2,
                 min_data_points: int = 50):
        """Inicializa o IP-CBM"""
        self.n_price_levels = n_price_levels
        self.n_time_steps = n_time_steps
        self.min_data_points = min_data_points
        self.stop_loss_buffer = stop_loss_buffer

        self.porosity_builder = PorosityFieldBuilder(
            n_price_levels=n_price_levels,
            n_time_steps=n_time_steps,
            base_surface_tension=base_surface_tension,
            contact_angle=contact_angle
        )

        self.invasion_simulator = InvasivePercolationSimulator(
            max_invasion_steps=max_invasion_steps
        )

        self.fractal_calculator = FractalDimensionCalculator()
        self.saffman_taylor = SaffmanTaylorAnalyzer()
        self.path_finder = BreakthroughPathFinder()

    def analyze(self,
               prices: np.ndarray,
               volumes: np.ndarray = None) -> dict:
        """Processa dados de mercado e retorna resultado da analise"""

        n = len(prices)

        if n < self.min_data_points:
            return self._create_empty_result("INSUFFICIENT_DATA")

        if volumes is None:
            volumes = np.abs(np.diff(prices))
            volumes = np.append(volumes, volumes[-1])
            volumes = volumes * 10000 + 1000

        current_price = prices[-1]
        price_min = np.min(prices)
        price_max = np.max(prices)
        price_range = price_max - price_min

        # PASSO 1: CONSTRUCAO DO CAMPO DE POROSIDADE
        porosity_field = self.porosity_builder.build_from_ohlcv(
            prices, volumes, current_price
        )

        # PASSO 2: SIMULACAO DE INVASAO (BIDIRECIONAL)
        price_levels = np.linspace(price_min * 0.999, price_max * 1.001, self.n_price_levels)
        current_price_idx = np.argmin(np.abs(price_levels - current_price))

        injection_point = (current_price_idx, self.n_time_steps // 2)

        cluster_up = self.invasion_simulator.simulate_invasion(
            porosity_field, injection_point, direction="UP"
        )

        cluster_down = self.invasion_simulator.simulate_invasion(
            porosity_field, injection_point, direction="DOWN"
        )

        # PASSO 3: ANALISE DO CLUSTER PRINCIPAL
        if cluster_up.mass > cluster_down.mass:
            main_cluster = cluster_up
            main_direction = "UP"
        else:
            main_cluster = cluster_down
            main_direction = "DOWN"

        Df_box = self.fractal_calculator.calculate_box_counting_dimension(main_cluster.cells)
        geometry_changed, deviation = self.fractal_calculator.detect_geometry_change(
            main_cluster.fractal_dimension
        )

        # PASSO 4: ANALISE DE SAFFMAN-TAYLOR
        injection_rate = np.mean(volumes[-10:]) / np.mean(volumes)

        st_analysis = self.saffman_taylor.analyze_stability(
            main_cluster, porosity_field, injection_rate
        )

        finger_formed, finger_progress = self.saffman_taylor.detect_finger_formation(main_cluster)
        L_crit = self.saffman_taylor.calculate_critical_length(porosity_field, main_cluster)

        # PASSO 5: CAMINHO DE BREAKTHROUGH
        breakthrough_path = self.path_finder.find_path(
            porosity_field, injection_point, main_direction
        )

        permeability_changes = self.path_finder.find_permeability_changes(
            porosity_field, breakthrough_path.path
        )

        # PASSO 6: DETERMINACAO DO ESTADO
        if main_cluster.finger_length >= L_crit:
            fluid_state = FluidState.BREAKTHROUGH
        elif finger_formed:
            fluid_state = FluidState.FINGERING
        elif st_analysis.is_unstable:
            fluid_state = FluidState.ADVANCING
        else:
            fluid_state = FluidState.TRAPPED

        breakthrough_imminent = main_cluster.finger_length >= L_crit * 0.8

        # PASSO 7: GERACAO DE SINAL
        signal = 0
        signal_name = "NEUTRAL"
        confidence = 0.0
        reasons = []

        # Calcula precos
        if main_direction == "UP":
            finger_base_idx = max([c[0] for c in main_cluster.cells]) if main_cluster.cells else current_price_idx
            stop_loss = price_levels[min(finger_base_idx + 2, self.n_price_levels - 1)]
        else:
            finger_base_idx = min([c[0] for c in main_cluster.cells]) if main_cluster.cells else current_price_idx
            stop_loss = price_levels[max(finger_base_idx - 2, 0)]

        if permeability_changes and len(permeability_changes) > 0:
            target_idx = permeability_changes[0][0]
            take_profit = price_levels[target_idx]
        else:
            if main_direction == "UP":
                take_profit = current_price + price_range * 0.1
            else:
                take_profit = current_price - price_range * 0.1

        # Logica de sinal
        if fluid_state == FluidState.BREAKTHROUGH:
            if main_direction == "UP":
                signal = 1
                signal_name = "LONG"
            else:
                signal = -1
                signal_name = "SHORT"

            confidence = min(0.95, 0.7 + finger_progress * 0.3)
            reasons.append(f"BREAKTHROUGH! L={main_cluster.finger_length:.1f}")
            reasons.append(f"Dir={main_direction}")

        elif fluid_state == FluidState.FINGERING and breakthrough_imminent:
            if main_direction == "UP":
                signal = 1
                signal_name = "LONG"
            else:
                signal = -1
                signal_name = "SHORT"

            confidence = min(0.85, 0.5 + finger_progress * 0.4)
            reasons.append(f"FINGERING iminente")
            reasons.append(f"L={main_cluster.finger_length:.1f}/{L_crit:.1f}")

        elif fluid_state == FluidState.FINGERING:
            signal_name = "WAIT"
            confidence = 0.5
            reasons.append(f"FINGER formado")
            reasons.append(f"Aguardando L_crit")

        elif st_analysis.is_unstable:
            signal_name = "WAIT"
            confidence = 0.4
            reasons.append("INSTABILIDADE detectada")
            reasons.append(f"M={st_analysis.mobility_ratio:.2f}")

        elif fluid_state == FluidState.TRAPPED:
            signal_name = "WAIT"
            confidence = 0.2
            reasons.append("TRAPPED por capilaridade")
            reasons.append(f"phi={porosity_field.porosity:.2f}")

        else:
            reasons.append("Cluster compacto")
            reasons.append(f"Df={main_cluster.fractal_dimension:.2f}")

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'fluid_state': fluid_state.value,
            'cluster_shape': main_cluster.shape.value,
            'porosity': porosity_field.porosity,
            'permeability_mean': porosity_field.permeability_mean,
            'surface_tension': porosity_field.surface_tension,
            'cluster_mass': main_cluster.mass,
            'radius_of_gyration': main_cluster.radius_of_gyration,
            'fractal_dimension': main_cluster.fractal_dimension,
            'fractal_dimension_box': Df_box,
            'aspect_ratio': main_cluster.aspect_ratio,
            'finger_length': main_cluster.finger_length,
            'finger_direction': main_direction,
            'critical_length': L_crit,
            'finger_progress': finger_progress,
            'finger_growth_rate': st_analysis.growth_rate,
            'mobility_ratio': st_analysis.mobility_ratio,
            'capillary_number': st_analysis.capillary_number,
            'is_unstable': st_analysis.is_unstable,
            'breakthrough_imminent': breakthrough_imminent,
            'path_length': len(breakthrough_path.path),
            'path_resistance': breakthrough_path.total_resistance,
            'geometry_changed': geometry_changed,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reasons': reasons
        }

    def _create_empty_result(self, signal_name: str) -> dict:
        """Cria resultado vazio"""
        return {
            'signal': 0,
            'signal_name': signal_name,
            'confidence': 0.0,
            'fluid_state': FluidState.TRAPPED.value,
            'cluster_shape': ClusterShape.COMPACT.value,
            'porosity': 0.0,
            'permeability_mean': 0.0,
            'surface_tension': 0.0,
            'cluster_mass': 0,
            'radius_of_gyration': 0.0,
            'fractal_dimension': 2.0,
            'fractal_dimension_box': 2.0,
            'aspect_ratio': 1.0,
            'finger_length': 0.0,
            'finger_direction': 'NEUTRAL',
            'critical_length': 0.0,
            'finger_progress': 0.0,
            'finger_growth_rate': 0.0,
            'mobility_ratio': 0.0,
            'capillary_number': 0.0,
            'is_unstable': False,
            'breakthrough_imminent': False,
            'path_length': 0,
            'path_resistance': float('inf'),
            'geometry_changed': False,
            'entry_price': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'reasons': [signal_name]
        }

    def reset(self):
        """Reseta o indicador"""
        pass


# ==============================================================================
# DEMONSTRACAO
# ==============================================================================

def generate_fingering_data(n_points: int = 100,
                           seed: int = 42,
                           with_breakthrough: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Gera dados que simulam condicoes de fingering viscoso"""
    np.random.seed(seed)

    base_price = 1.0850
    prices = [base_price]
    volumes = [1000]

    accumulation_phase = int(n_points * 0.6)

    for i in range(1, accumulation_phase):
        noise = np.random.randn() * 0.00005
        vol = 1500 + np.random.randn() * 200

        prices.append(prices[-1] + noise)
        volumes.append(vol)

    if with_breakthrough:
        finger_phase = int(n_points * 0.3)

        for i in range(finger_phase):
            trend = 0.00015
            noise = np.random.randn() * 0.00008
            vol = 2000 + i * 20 + np.random.randn() * 100

            prices.append(prices[-1] + trend + noise)
            volumes.append(vol)

        remaining = n_points - accumulation_phase - finger_phase

        for i in range(remaining):
            trend = 0.0003
            noise = np.random.randn() * 0.0001
            vol = 3000 + np.random.randn() * 500

            prices.append(prices[-1] + trend + noise)
            volumes.append(vol)
    else:
        remaining = n_points - accumulation_phase

        for i in range(remaining):
            noise = np.random.randn() * 0.00006
            vol = 1200 + np.random.randn() * 200

            prices.append(prices[-1] + noise)
            volumes.append(vol)

    return np.array(prices), np.array(volumes)


def main():
    """Demonstracao do indicador IP-CBM"""
    print("=" * 70)
    print("INVASIVE PERCOLATION & CAPILLARY BREAKTHROUGH MONITOR (IP-CBM)")
    print("Indicador baseado em Fisica de Escoamento em Meios Porosos")
    print("=" * 70)
    print()

    indicator = InvasivePercolationCapillaryBreakthroughMonitor(
        n_price_levels=50,
        n_time_steps=50,
        base_surface_tension=1.0,
        contact_angle=120.0,
        max_invasion_steps=500,
        min_data_points=50
    )

    print("Indicador inicializado!")
    print(f"  - Grid: {indicator.n_price_levels}x{indicator.n_time_steps}")
    print(f"  - Df referencia (percolacao 2D): {FRACTAL_DIMENSION_2D:.2f}")
    print()

    print("Gerando dados com condicoes de fingering e breakthrough...")
    prices, volumes = generate_fingering_data(n_points=100, seed=42, with_breakthrough=True)
    print(f"Dados gerados: {len(prices)} pontos")
    print()

    result = indicator.analyze(prices, volumes)

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Sinal: {result['signal_name']}")
    print(f"Confianca: {result['confidence']:.2%}")
    print(f"Estado: {result['fluid_state']}")
    print(f"Forma: {result['cluster_shape']}")
    print(f"Df: {result['fractal_dimension']:.3f}")
    print(f"Finger: {result['finger_length']:.1f}/{result['critical_length']:.1f}")
    print(f"Direcao: {result['finger_direction']}")
    print(f"Razoes: {result['reasons']}")

    return indicator, result


if __name__ == "__main__":
    indicator, result = main()
