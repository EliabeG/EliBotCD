"""
================================================================================
GRANULAR JAMMING & FORCE CHAIN PERCOLATOR (GJ-FCP)
Indicador de Forex baseado em Fisica da Materia Granular e Teoria da Percolacao
================================================================================

Baseado na Fisica da Materia Granular e na Teoria da Percolacao. Trataremos o
Order Book nao como numeros, mas como um empacotamento de graos de areia (ordens)
que estao sofrendo compressao.

A Fisica: Transicao de Jamming
Imagine um saco de areia a vacuo. Ele e duro como pedra (estado Jammed). Se voce
fizer um pequeno furo, ele flui como agua (estado Unjammed). Em baixa volatilidade,
o EURUSD esta no estado Jammed. As ordens de compra e venda estao tao compactadas
que o preco nao consegue se mover. Nos vamos calcular o Tensor de Tensao
(Stress Tensor) interno desse empacotamento para saber:
1. Quando ele vai quebrar? (Tempo)
2. Para que lado a "parede" vai ceder? (Direcao)

Por que isso e o "Santo Graal" da Baixa Volatilidade?
Indicadores comuns (Bollinger Squeeze) so avisam DEPOIS que a volatilidade expandiu.
O GJ-FCP mede a "fadiga do material". Ele sabe que o suporte vai quebrar porque
ele calculou que a densidade de ordens naquele ponto nao suporta a pressao do
vetor de forca acumulado, da mesma forma que um engenheiro civil sabe que uma
ponte vai cair antes dela cair.

Desafio de Codigo: Calcular Voronoi e Tensao Granular em tempo real para milhares
de ordens por segundo exige algoritmos de geometria computacional O(N log N)
extremamente otimizados. Usa scipy.spatial.Voronoi.

Autor: Gerado por Claude AI
Versao: 1.0.0
================================================================================
"""

import numpy as np
from scipy.spatial import Voronoi, Delaunay, ConvexHull
from scipy.ndimage import uniform_filter1d
from scipy.linalg import eigh
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

class JammingState(Enum):
    """Estado de jamming do sistema"""
    JAMMED = "JAMMED"              # Duro como pedra (baixa vol)
    UNJAMMING = "UNJAMMING"        # Comecando a ceder
    UNJAMMED = "UNJAMMED"          # Fluindo como agua (alta vol)
    CRITICAL = "CRITICAL"          # No ponto critico de transicao


class FailureMode(Enum):
    """Modo de falha estrutural"""
    SHEAR_FAILURE = "SHEAR_FAILURE"      # Falha por cisalhamento
    BUCKLING = "BUCKLING"                 # Flambagem das cadeias
    AVALANCHE = "AVALANCHE"               # Avalanche (SOC)
    LIQUEFACTION = "LIQUEFACTION"         # Liquefacao do book
    STABLE = "STABLE"                     # Estrutura estavel


@dataclass
class OrderParticle:
    """Uma ordem tratada como particula granular"""
    price: float                    # Posicao no eixo de preco
    time: float                     # Posicao no eixo temporal
    size: float                     # Tamanho da ordem (raio da particula)
    is_bid: bool                    # True = compra, False = venda
    aggressiveness: float           # Agressividade (0 = passiva, 1 = agressiva)


@dataclass
class VoronoiTessellation:
    """Tesselacao de Voronoi do order book"""
    cells: List[np.ndarray]         # Vertices de cada celula
    volumes: np.ndarray             # Volume de cada celula (v_f)
    neighbors: List[List[int]]      # Vizinhos de cada particula
    compactness: float              # phi = Sum V_particulas / V_total
    free_volume: np.ndarray         # Volume livre de cada celula


@dataclass
class ForceChainNetwork:
    """Rede de cadeias de forca"""
    adjacency_matrix: np.ndarray    # Matriz de adjacencia
    force_magnitudes: np.ndarray    # Magnitude das forcas
    force_directions: np.ndarray    # Direcoes das forcas
    chain_lengths: np.ndarray       # Comprimentos das cadeias
    is_anisotropic: bool            # True se anisotropico
    principal_direction: float      # Direcao principal da forca


@dataclass
class StressTensor:
    """Tensor de Tensao Granular sigma_ij"""
    tensor: np.ndarray              # Tensor 2x2
    eigenvalues: np.ndarray         # Autovalores (tensoes principais)
    eigenvectors: np.ndarray        # Autovetores (direcoes principais)
    shear_stress: float             # tau - tensao de cisalhamento
    normal_stress: float            # sigma_n - tensao normal
    deviatoric_stress: float        # Tensao desviadora
    pressure: float                 # Pressao hidrostatica


@dataclass
class MechanicalProperties:
    """Propriedades mecanicas do sistema granular"""
    shear_modulus: float            # G - Modulo de cisalhamento (rigidez)
    bulk_modulus: float             # K - Modulo volumetrico
    coordination_number: float      # Z - Numero de coordenacao medio
    coordination_critical: float    # Z_c - Numero critico (isostatico)
    friction_angle: float           # psi - Angulo de atrito interno
    cohesion: float                 # c - Coesao


@dataclass
class MohrCoulombCriterion:
    """Criterio de falha de Mohr-Coulomb"""
    shear_stress: float             # tau
    normal_stress: float            # sigma_n
    friction_angle: float           # psi
    cohesion: float                 # c
    yield_criterion: float          # tau - (sigma_n * tan(psi) + c)
    is_failing: bool                # True se tau >= sigma_n * tan(psi) + c


# ==============================================================================
# GERADOR DE PARTICULAS (SIMULACAO DE ORDER BOOK)
# ==============================================================================

class OrderBookParticleGenerator:
    """
    Gera particulas a partir de dados de preco/volume

    Na pratica real, voce usaria Level 3 Data (ordens individuais no book).
    Aqui simulamos a partir de price/volume bars.
    """

    def __init__(self,
                 price_resolution: float = 0.0001,
                 time_resolution: float = 1.0):
        self.price_resolution = price_resolution
        self.time_resolution = time_resolution

    def generate_from_ohlcv(self,
                           prices: np.ndarray,
                           volumes: np.ndarray,
                           n_particles_per_bar: int = 10) -> List[OrderParticle]:
        """
        Gera particulas a partir de dados OHLCV

        Simula ordens distribuidas em torno do preco de cada barra
        """
        particles = []
        n = len(prices)

        for i in range(n):
            price = prices[i]
            volume = volumes[i]
            time = i * self.time_resolution

            # Determina se foi mais compra ou venda
            if i > 0:
                is_bullish = prices[i] > prices[i-1]
            else:
                is_bullish = True

            # Gera particulas em torno do preco
            n_particles = max(1, int(n_particles_per_bar * volume / (np.mean(volumes) + 1e-10)))

            for j in range(n_particles):
                # Preco com pequena variacao
                particle_price = price + np.random.randn() * self.price_resolution * 5

                # Tempo com pequena variacao dentro da barra
                particle_time = time + np.random.uniform(-0.5, 0.5) * self.time_resolution

                # Tamanho proporcional ao volume
                particle_size = volume / n_particles / 1000

                # Bid ou ask
                is_bid = (np.random.random() < 0.5) if is_bullish else (np.random.random() < 0.6)

                # Agressividade
                aggressiveness = np.random.beta(2, 5)  # Maioria passiva

                particle = OrderParticle(
                    price=particle_price,
                    time=particle_time,
                    size=particle_size,
                    is_bid=is_bid,
                    aggressiveness=aggressiveness
                )
                particles.append(particle)

        return particles

    def particles_to_points(self, particles: List[OrderParticle]) -> np.ndarray:
        """Converte particulas para array de pontos 2D"""
        points = np.array([[p.price, p.time] for p in particles])
        return points


# ==============================================================================
# TESSELACAO DE VORONOI
# ==============================================================================

class VoronoiAnalyzer:
    """
    1. Tesselacao de Voronoi (Espaco de Volume Livre)

    Para cada ordem no book e cada trade realizado, mapeie-os como particulas
    num espaco 2D (Preco x Tempo). Execute uma Tesselacao de Voronoi dinamica.

    - Calcule o Volume Livre Local (v_f) de cada celula
    - Em baixa volatilidade, o volume livre tende a zero (densidade critica phi_c)
    - A variavel de estado e a Compacidade (phi): phi = Sum V_particulas / V_total
    """

    def __init__(self,
                 compactness_critical: float = 0.64):
        """
        Args:
            compactness_critical: phi_c - densidade critica de jamming (~0.64 para 2D)
        """
        self.compactness_critical = compactness_critical

    def compute_tessellation(self,
                            points: np.ndarray,
                            particle_sizes: np.ndarray) -> VoronoiTessellation:
        """
        Computa a tesselacao de Voronoi e metricas associadas
        """
        n = len(points)

        if n < 4:
            # Dados insuficientes
            return VoronoiTessellation(
                cells=[],
                volumes=np.array([]),
                neighbors=[],
                compactness=0.0,
                free_volume=np.array([])
            )

        try:
            # Adiciona pontos de borda para evitar celulas infinitas
            center = np.mean(points, axis=0)
            scale = np.std(points, axis=0) * 5

            border_points = np.array([
                center + [-scale[0], -scale[1]],
                center + [scale[0], -scale[1]],
                center + [-scale[0], scale[1]],
                center + [scale[0], scale[1]]
            ])

            all_points = np.vstack([points, border_points])

            # Tesselacao
            vor = Voronoi(all_points)

            # Calcula volumes das celulas (apenas para pontos originais)
            volumes = np.zeros(n)
            cells = []

            for i in range(n):
                region_idx = vor.point_region[i]
                region = vor.regions[region_idx]

                if -1 not in region and len(region) > 0:
                    # Celula finita
                    vertices = vor.vertices[region]
                    cells.append(vertices)

                    # Area da celula (volume em 2D)
                    try:
                        hull = ConvexHull(vertices)
                        volumes[i] = hull.volume
                    except:
                        volumes[i] = 0.01
                else:
                    cells.append(np.array([]))
                    volumes[i] = 0.01

            # Calcula vizinhos usando Delaunay
            neighbors = [[] for _ in range(n)]
            try:
                tri = Delaunay(points)
                for simplex in tri.simplices:
                    for i in range(3):
                        for j in range(i+1, 3):
                            if simplex[i] < n and simplex[j] < n:
                                neighbors[simplex[i]].append(simplex[j])
                                neighbors[simplex[j]].append(simplex[i])

                # Remove duplicatas
                neighbors = [list(set(neigh)) for neigh in neighbors]
            except:
                pass

            # Volume total
            total_volume = np.sum(volumes)

            # Volume das particulas
            particle_volumes = np.pi * particle_sizes**2  # Area em 2D
            total_particle_volume = np.sum(particle_volumes)

            # Compacidade
            compactness = total_particle_volume / (total_volume + 1e-10)

            # Volume livre de cada celula
            free_volume = volumes - particle_volumes[:n]
            free_volume = np.maximum(free_volume, 0)

            return VoronoiTessellation(
                cells=cells,
                volumes=volumes,
                neighbors=neighbors,
                compactness=compactness,
                free_volume=free_volume
            )

        except Exception as e:
            return VoronoiTessellation(
                cells=[],
                volumes=np.zeros(n),
                neighbors=[[] for _ in range(n)],
                compactness=0.0,
                free_volume=np.zeros(n)
            )


# ==============================================================================
# REDES DE CADEIAS DE FORCA
# ==============================================================================

class ForceChainAnalyzer:
    """
    2. Redes de Cadeias de Forca (Force Chains Network)

    Em materiais granulares, a forca nao se distribui igualmente. Ela se concentra
    em "raios" invisiveis chamados Cadeias de Forca. Essas cadeias seguram o preco.

    Voce deve construir um Grafo onde as arestas representam a pressao entre
    ordens de compra e venda.

    - Use a Matriz Hessiana da Energia Potencial do sistema para identificar os
      autovetores que correspondem aos "modos moles" (soft modes).
    - Se a rede de forca e isotropica (igual em todas as direcoes), o mercado
      esta morto.
    - Se a rede comeca a se alinhar (anisotropia), a ruptura ocorrera na direcao
      perpendicular a cadeia de forca principal.
    """

    def __init__(self,
                 force_threshold: float = 0.1,
                 anisotropy_threshold: float = 0.3):
        """
        Args:
            force_threshold: Limiar minimo para considerar uma forca
            anisotropy_threshold: Limiar para considerar anisotropico
        """
        self.force_threshold = force_threshold
        self.anisotropy_threshold = anisotropy_threshold

    def compute_force_network(self,
                             particles: List[OrderParticle],
                             tessellation: VoronoiTessellation) -> ForceChainNetwork:
        """
        Constroi a rede de cadeias de forca
        """
        n = len(particles)

        if n < 3 or len(tessellation.neighbors) == 0:
            return ForceChainNetwork(
                adjacency_matrix=np.zeros((1, 1)),
                force_magnitudes=np.array([0]),
                force_directions=np.array([0]),
                chain_lengths=np.array([0]),
                is_anisotropic=False,
                principal_direction=0.0
            )

        # Matriz de adjacencia e forcas
        adjacency = np.zeros((n, n))
        force_vectors = []

        for i in range(n):
            if i >= len(tessellation.neighbors):
                continue

            for j in tessellation.neighbors[i]:
                if j >= n:
                    continue

                # Calcula forca entre particulas
                p1, p2 = particles[i], particles[j]

                # Vetor distancia
                r = np.array([p2.price - p1.price, p2.time - p1.time])
                r_mag = np.linalg.norm(r) + 1e-10

                # Forca baseada em: tamanho, agressividade, e se sao opostos (bid vs ask)
                if p1.is_bid != p2.is_bid:
                    # Forcas opostas geram tensao
                    f_mag = (p1.size + p2.size) * (p1.aggressiveness + p2.aggressiveness) / r_mag
                else:
                    # Mesmo lado, menos tensao
                    f_mag = (p1.size + p2.size) * 0.5 / r_mag

                if f_mag > self.force_threshold:
                    adjacency[i, j] = f_mag
                    adjacency[j, i] = f_mag

                    # Vetor forca
                    f_vec = f_mag * r / r_mag
                    force_vectors.append(f_vec)

        # Magnitude das forcas
        force_magnitudes = np.array([np.linalg.norm(f) for f in force_vectors]) if force_vectors else np.array([0])

        # Direcoes das forcas (angulos)
        force_directions = np.array([np.arctan2(f[1], f[0]) for f in force_vectors]) if force_vectors else np.array([0])

        # Analisa anisotropia
        if len(force_vectors) > 0:
            # Tensor de direcao
            Q = np.zeros((2, 2))
            for f in force_vectors:
                f_norm = f / (np.linalg.norm(f) + 1e-10)
                Q += np.outer(f_norm, f_norm)
            Q /= len(force_vectors)

            # Autovalores
            eigenvalues, eigenvectors = np.linalg.eigh(Q)

            # Anisotropia
            anisotropy = (eigenvalues[1] - eigenvalues[0]) / (eigenvalues[1] + eigenvalues[0] + 1e-10)
            is_anisotropic = anisotropy > self.anisotropy_threshold

            # Direcao principal
            principal_direction = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
        else:
            is_anisotropic = False
            principal_direction = 0.0

        # Comprimentos das cadeias (usando DFS para encontrar cadeias conectadas)
        chain_lengths = self._find_chain_lengths(adjacency)

        return ForceChainNetwork(
            adjacency_matrix=adjacency,
            force_magnitudes=force_magnitudes,
            force_directions=force_directions,
            chain_lengths=chain_lengths,
            is_anisotropic=is_anisotropic,
            principal_direction=principal_direction
        )

    def _find_chain_lengths(self, adjacency: np.ndarray) -> np.ndarray:
        """Encontra comprimentos das cadeias de forca"""
        n = adjacency.shape[0]
        visited = np.zeros(n, dtype=bool)
        chain_lengths = []

        for start in range(n):
            if visited[start]:
                continue

            # BFS para encontrar componente conectado
            chain_length = 0
            queue = [start]

            while queue:
                node = queue.pop(0)
                if visited[node]:
                    continue
                visited[node] = True
                chain_length += 1

                for neighbor in range(n):
                    if adjacency[node, neighbor] > 0 and not visited[neighbor]:
                        queue.append(neighbor)

            if chain_length > 1:
                chain_lengths.append(chain_length)

        return np.array(chain_lengths) if chain_lengths else np.array([0])


# ==============================================================================
# TENSOR DE TENSAO GRANULAR
# ==============================================================================

class StressTensorCalculator:
    """
    3. O Tensor de Tensao Granular (sigma_ij)

    Calcule a tensao interna acumulada no sistema usando a formula de Weber
    para meios granulares:

    sigma_ij = (1/V) Sum f_i^(k) r_j^(k)

    Onde:
    - f: Vetor forca entre duas ordens (baseado na agressividade/tamanho)
    - r: Vetor distancia entre elas (spread/tempo)
    - V: Volume de controle
    """

    def __init__(self):
        pass

    def compute_stress_tensor(self,
                             particles: List[OrderParticle],
                             tessellation: VoronoiTessellation,
                             force_network: ForceChainNetwork) -> StressTensor:
        """
        Calcula o tensor de tensao usando a formula de Weber
        """
        n = len(particles)

        if n < 3:
            return StressTensor(
                tensor=np.zeros((2, 2)),
                eigenvalues=np.array([0, 0]),
                eigenvectors=np.eye(2),
                shear_stress=0.0,
                normal_stress=0.0,
                deviatoric_stress=0.0,
                pressure=0.0
            )

        # Volume de controle
        V = np.sum(tessellation.volumes) + 1e-10

        # Tensor de tensao sigma_ij = (1/V) Sum f_i^(k) r_j^(k)
        sigma = np.zeros((2, 2))

        adjacency = force_network.adjacency_matrix

        for i in range(min(n, adjacency.shape[0])):
            if i >= len(tessellation.neighbors):
                continue

            for j in tessellation.neighbors[i]:
                if j >= n or j >= adjacency.shape[0]:
                    continue

                if adjacency[i, j] > 0:
                    p1, p2 = particles[i], particles[j]

                    # Vetor distancia r
                    r = np.array([p2.price - p1.price, p2.time - p1.time])

                    # Vetor forca f (na direcao de r)
                    f_mag = adjacency[i, j]
                    r_mag = np.linalg.norm(r) + 1e-10
                    f = f_mag * r / r_mag

                    # Contribuicao para o tensor: f_i * r_j
                    sigma += np.outer(f, r)

        sigma /= V

        # Simetriza
        sigma = (sigma + sigma.T) / 2

        # Autovalores e autovetores (tensoes principais)
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)

        # Ordena (maior primeiro)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Pressao hidrostatica (media dos autovalores)
        pressure = np.mean(eigenvalues)

        # Tensao desviadora (anisotropica)
        deviatoric_stress = eigenvalues[0] - eigenvalues[1]

        # Tensao normal e de cisalhamento (para plano a 45 graus)
        sigma_n = pressure
        tau = deviatoric_stress / 2

        return StressTensor(
            tensor=sigma,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            shear_stress=tau,
            normal_stress=sigma_n,
            deviatoric_stress=deviatoric_stress,
            pressure=pressure
        )


# ==============================================================================
# PROPRIEDADES MECANICAS E CRITERIO DE MOHR-COULOMB
# ==============================================================================

class MechanicalAnalyzer:
    """
    O Algoritmo de Implementacao (The Avalanche Hunter)
    Este e um algoritmo de deteccao de Criticalidade Auto-Organizada (SOC).

    1. Monitoramento de Rigidez (G):
       Calcule o Modulo de Cisalhamento (Shear Modulus G) do mercado em tempo real.
       Em baixa vol, G e alto (mercado comporta-se como solido elastico).

    2. Calculo do Numero de Coordenacao (Z):
       Conte quantos "vizinhos" (ordens opostas) cada ordem toca.
       Existe um numero critico Z_c (ponto isostatico).
       Quando Z se aproxima de Z_c vindo de cima, o sistema perde estabilidade.

    3. Sinal de Ruptura (The Yield Point):
       Utilize o Criterio de Falha de Mohr-Coulomb adaptado para financas:
       tau >= sigma_n tan(psi) + c

       Onde:
       - tau: Tensao de cisalhamento (pressao de rompimento)
       - sigma_n: Tensao normal (volume de defesa no book)
       - psi: Angulo de atrito interno (spread/volatilidade implicita)
       - c: Coesao
    """

    def __init__(self,
                 z_critical_2d: float = 4.0,
                 friction_angle_default: float = 30.0):
        """
        Args:
            z_critical_2d: Z_c para empacotamento 2D (ponto isostatico ~ 4)
            friction_angle_default: Angulo de atrito padrao em graus
        """
        self.z_critical_2d = z_critical_2d
        self.friction_angle_default = np.radians(friction_angle_default)

    def compute_coordination_number(self, tessellation: VoronoiTessellation) -> float:
        """
        Calcula numero de coordenacao medio Z

        Z = media do numero de vizinhos por particula
        """
        if not tessellation.neighbors:
            return 0.0

        coordination_numbers = [len(n) for n in tessellation.neighbors]
        Z = np.mean(coordination_numbers) if coordination_numbers else 0.0

        return Z

    def compute_shear_modulus(self,
                             stress_tensor: StressTensor,
                             compactness: float) -> float:
        """
        Estima o modulo de cisalhamento G

        G aumenta com a compacidade e com a tensao desviadora
        """
        # Modelo simplificado: G proporcional a (phi - phi_c)^alpha perto da transicao
        phi_c = 0.64  # Densidade critica de jamming 2D

        if compactness <= phi_c:
            G = 0.0
        else:
            G = (compactness - phi_c)**1.5 * (1 + stress_tensor.pressure)

        return G

    def compute_friction_angle(self,
                              prices: np.ndarray,
                              volumes: np.ndarray) -> float:
        """
        Estima o angulo de atrito interno psi

        Baseado no spread/volatilidade implicita
        """
        # Volatilidade como proxy para atrito
        returns = np.diff(np.log(prices + 1e-10))
        volatility = np.std(returns) * np.sqrt(252)  # Anualizada

        # Volume relativo
        volume_ratio = np.std(volumes) / (np.mean(volumes) + 1e-10)

        # Angulo de atrito: maior volatilidade = maior atrito
        psi = self.friction_angle_default * (1 + volatility * 10 + volume_ratio)
        psi = np.clip(psi, 0.1, np.pi/3)  # Entre ~5 e 60 graus

        return psi

    def compute_cohesion(self, stress_tensor: StressTensor) -> float:
        """
        Estima a coesao c

        Baseado na pressao media do sistema
        """
        c = max(0.0, stress_tensor.pressure * 0.1)
        return c

    def evaluate_mohr_coulomb(self,
                             stress_tensor: StressTensor,
                             friction_angle: float,
                             cohesion: float) -> MohrCoulombCriterion:
        """
        Avalia o criterio de falha de Mohr-Coulomb

        tau >= sigma_n tan(psi) + c
        """
        tau = stress_tensor.shear_stress
        sigma_n = stress_tensor.normal_stress
        psi = friction_angle
        c = cohesion

        # Criterio de yield
        yield_strength = sigma_n * np.tan(psi) + c
        yield_criterion = tau - yield_strength

        # Esta falhando?
        is_failing = yield_criterion >= 0

        return MohrCoulombCriterion(
            shear_stress=tau,
            normal_stress=sigma_n,
            friction_angle=psi,
            cohesion=c,
            yield_criterion=yield_criterion,
            is_failing=is_failing
        )

    def compute_all_properties(self,
                              particles: List[OrderParticle],
                              tessellation: VoronoiTessellation,
                              stress_tensor: StressTensor,
                              prices: np.ndarray,
                              volumes: np.ndarray) -> MechanicalProperties:
        """
        Computa todas as propriedades mecanicas
        """
        Z = self.compute_coordination_number(tessellation)
        G = self.compute_shear_modulus(stress_tensor, tessellation.compactness)
        psi = self.compute_friction_angle(prices, volumes)
        c = self.compute_cohesion(stress_tensor)

        # Modulo volumetrico (estimativa)
        K = G * 2 * (1 + 0.3)  # Assumindo razao de Poisson ~ 0.3

        return MechanicalProperties(
            shear_modulus=G,
            bulk_modulus=K,
            coordination_number=Z,
            coordination_critical=self.z_critical_2d,
            friction_angle=psi,
            cohesion=c
        )


# ==============================================================================
# DETECTOR DE JAMMING E AVALANCHE
# ==============================================================================

class JammingTransitionDetector:
    """
    Detecta a transicao de jamming e prediz avalanches
    """

    def __init__(self,
                 g_drop_threshold: float = 0.3,
                 z_margin: float = 0.5):
        self.g_drop_threshold = g_drop_threshold
        self.z_margin = z_margin

        # Historico
        self.g_history: List[float] = []
        self.z_history: List[float] = []
        self.phi_history: List[float] = []

    def update_history(self, G: float, Z: float, phi: float):
        """Atualiza historico"""
        self.g_history.append(G)
        self.z_history.append(Z)
        self.phi_history.append(phi)

        # Mantem ultimos 50
        if len(self.g_history) > 50:
            self.g_history.pop(0)
            self.z_history.pop(0)
            self.phi_history.pop(0)

    def detect_jamming_state(self,
                            properties: MechanicalProperties,
                            tessellation: VoronoiTessellation) -> Tuple[JammingState, float]:
        """
        Detecta o estado de jamming atual

        Returns:
            Tupla (JammingState, taxa_de_queda_G)
        """
        G = properties.shear_modulus
        Z = properties.coordination_number
        Z_c = properties.coordination_critical
        phi = tessellation.compactness
        phi_c = 0.64

        self.update_history(G, Z, phi)

        # Taxa de queda de G
        if len(self.g_history) >= 5:
            g_recent = np.mean(self.g_history[-5:])
            g_older = np.mean(self.g_history[-10:-5]) if len(self.g_history) >= 10 else g_recent
            g_rate = (g_recent - g_older) / (g_older + 1e-10)
        else:
            g_rate = 0.0

        # Classificacao
        if G < 0.01:
            # Sem rigidez = liquido
            state = JammingState.UNJAMMED
        elif Z < Z_c + self.z_margin:
            # Proximo do ponto isostatico
            if g_rate < -self.g_drop_threshold:
                state = JammingState.CRITICAL
            else:
                state = JammingState.UNJAMMING
        elif phi < phi_c:
            # Abaixo da densidade critica
            state = JammingState.UNJAMMING
        else:
            # Compactado e rigido
            if g_rate < -self.g_drop_threshold:
                state = JammingState.UNJAMMING
            else:
                state = JammingState.JAMMED

        return state, g_rate

    def detect_failure_mode(self,
                           mohr_coulomb: MohrCoulombCriterion,
                           force_network: ForceChainNetwork,
                           jamming_state: JammingState,
                           g_rate: float) -> FailureMode:
        """
        Detecta o modo de falha esperado
        """
        if jamming_state == JammingState.JAMMED:
            return FailureMode.STABLE

        if mohr_coulomb.is_failing:
            return FailureMode.SHEAR_FAILURE

        if g_rate < -0.5 and force_network.is_anisotropic:
            return FailureMode.BUCKLING

        if jamming_state == JammingState.CRITICAL:
            return FailureMode.AVALANCHE

        if jamming_state == JammingState.UNJAMMED:
            return FailureMode.LIQUEFACTION

        return FailureMode.STABLE

    def reset(self):
        """Reseta historico"""
        self.g_history.clear()
        self.z_history.clear()
        self.phi_history.clear()


# ==============================================================================
# INDICADOR GJ-FCP COMPLETO
# ==============================================================================

class GranularJammingForceChainPercolator:
    """
    Granular Jamming & Force Chain Percolator (GJ-FCP)

    Indicador completo que usa fisica da materia granular para detectar
    transicoes de volatilidade em baixa vol.

    GATILHO DE OPERACAO:
    - O indicador detecta que o Modulo de Cisalhamento G comecou a cair
      exponencialmente (o "solido" esta virando "liquido").
    - As Cadeias de Forca (ForceChains) sofrem Flambagem (Buckling). Ou seja,
      a estrutura de suporte colapsa.
    - Direcao: Olhe para os autovetores do Tensor de Tensao Anisotropico
      (Deviatoric Stress). O rompimento ocorrera na direcao do autovalor
      principal positivo.
    - Acao: Entre a mercado na direcao da falha estrutural ANTES do preco se
      mover 1 pip. Voce esta comprando a liquefacao do book.
    """

    def __init__(self,
                 # Parametros de particulas
                 particles_per_bar: int = 5,

                 # Parametros de jamming
                 compactness_critical: float = 0.64,
                 z_critical: float = 4.0,

                 # Parametros de deteccao
                 g_drop_threshold: float = 0.3,
                 anisotropy_threshold: float = 0.3,

                 # Geral
                 min_data_points: int = 50):
        """
        Inicializa o GJ-FCP
        """
        self.particles_per_bar = particles_per_bar
        self.compactness_critical = compactness_critical
        self.min_data_points = min_data_points

        # Componentes
        self.particle_generator = OrderBookParticleGenerator()
        self.voronoi_analyzer = VoronoiAnalyzer(compactness_critical=compactness_critical)
        self.force_analyzer = ForceChainAnalyzer(anisotropy_threshold=anisotropy_threshold)
        self.stress_calculator = StressTensorCalculator()
        self.mechanical_analyzer = MechanicalAnalyzer(z_critical_2d=z_critical)
        self.jamming_detector = JammingTransitionDetector(g_drop_threshold=g_drop_threshold)

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
                'jamming_state': 'JAMMED',
                'failure_mode': 'STABLE',
                'confidence': 0.0,
                'shear_stress': 0.0,
                'normal_stress': 0.0,
                'deviatoric_stress': 0.0,
                'shear_modulus': 0.0,
                'shear_modulus_rate': 0.0,
                'coordination_number': 0.0,
                'coordination_critical': 4.0,
                'compactness': 0.0,
                'compactness_critical': 0.64,
                'yield_criterion': 0.0,
                'friction_angle': 0.0,
                'failure_direction': 0.0,
                'price_direction': 'NEUTRAL',
                'n_particles': 0,
                'max_chain_length': 0,
                'is_anisotropic': False,
                'reasons': ['Dados insuficientes para analise granular']
            }

        # Volumes sinteticos se nao fornecidos
        if volumes is None:
            volumes = np.abs(np.diff(prices))
            volumes = np.append(volumes, volumes[-1])
            volumes = volumes * 10000 + 1000

        # ============================================================
        # PASSO 1: GERACAO DE PARTICULAS
        # ============================================================
        particles = self.particle_generator.generate_from_ohlcv(
            prices, volumes, self.particles_per_bar
        )
        points = self.particle_generator.particles_to_points(particles)
        particle_sizes = np.array([p.size for p in particles])

        # ============================================================
        # PASSO 2: TESSELACAO DE VORONOI
        # ============================================================
        tessellation = self.voronoi_analyzer.compute_tessellation(points, particle_sizes)

        # ============================================================
        # PASSO 3: REDE DE CADEIAS DE FORCA
        # ============================================================
        force_network = self.force_analyzer.compute_force_network(particles, tessellation)

        # ============================================================
        # PASSO 4: TENSOR DE TENSAO GRANULAR
        # ============================================================
        stress_tensor = self.stress_calculator.compute_stress_tensor(
            particles, tessellation, force_network
        )

        # ============================================================
        # PASSO 5: PROPRIEDADES MECANICAS
        # ============================================================
        properties = self.mechanical_analyzer.compute_all_properties(
            particles, tessellation, stress_tensor, prices, volumes
        )

        # ============================================================
        # PASSO 6: CRITERIO DE MOHR-COULOMB
        # ============================================================
        mohr_coulomb = self.mechanical_analyzer.evaluate_mohr_coulomb(
            stress_tensor, properties.friction_angle, properties.cohesion
        )

        # ============================================================
        # PASSO 7: DETECCAO DE JAMMING
        # ============================================================
        jamming_state, g_rate = self.jamming_detector.detect_jamming_state(
            properties, tessellation
        )

        failure_mode = self.jamming_detector.detect_failure_mode(
            mohr_coulomb, force_network, jamming_state, g_rate
        )

        # ============================================================
        # PASSO 8: GERACAO DE SINAL
        # ============================================================
        signal = 0
        signal_name = 'NEUTRAL'
        confidence = 0.0
        reasons = []

        # Direcao da falha (perpendicular a cadeia de forca principal)
        failure_direction = force_network.principal_direction + np.pi/2

        # Mapeia para direcao de preco
        eigenvector_main = stress_tensor.eigenvectors[:, 0]
        price_component = eigenvector_main[0]

        if price_component > 0:
            price_direction = "UP"
        else:
            price_direction = "DOWN"

        # JAMMED - Sistema estavel, esperar
        if jamming_state == JammingState.JAMMED:
            signal_name = 'WAIT'
            reasons.append(f"JAMMED: Sistema estavel. G={properties.shear_modulus:.4f}")
            reasons.append(f"Z={properties.coordination_number:.2f}. Aguardando transicao.")

        # CRITICAL ou UNJAMMING com falha de Mohr-Coulomb
        elif mohr_coulomb.is_failing and failure_mode in [FailureMode.SHEAR_FAILURE, FailureMode.AVALANCHE]:
            if price_direction == "UP":
                signal = 1
                signal_name = 'LONG'
            else:
                signal = -1
                signal_name = 'SHORT'

            confidence = min(1.0, abs(mohr_coulomb.yield_criterion) * 10)
            reasons.append(f"Mohr-Coulomb: Criterio de falha atingido!")
            reasons.append(f"tau={mohr_coulomb.shear_stress:.4f}. Modo: {failure_mode.value}")

        # BUCKLING das cadeias de forca
        elif failure_mode == FailureMode.BUCKLING:
            if price_direction == "UP":
                signal = 1
                signal_name = 'LONG'
            else:
                signal = -1
                signal_name = 'SHORT'

            confidence = min(0.9, abs(g_rate) * 2)
            reasons.append(f"Buckling: Cadeias de forca flambando!")
            reasons.append(f"dG/dt={g_rate:.2%}. Rede anisotropica.")

        # G caindo rapidamente (transicao iminente)
        elif jamming_state == JammingState.UNJAMMING and g_rate < -0.2:
            if price_direction == "UP":
                signal = 1
                signal_name = 'LONG'
            else:
                signal = -1
                signal_name = 'SHORT'

            confidence = min(0.7, abs(g_rate))
            reasons.append(f"Unjamming: G caindo {g_rate:.2%}")
            reasons.append(f"Liquefacao iminente. Direcao: {price_direction}")

        # Z proximo do critico
        elif properties.coordination_number < properties.coordination_critical + 0.5:
            if force_network.is_anisotropic:
                if price_direction == "UP":
                    signal = 1
                    signal_name = 'LONG'
                else:
                    signal = -1
                    signal_name = 'SHORT'

                confidence = 0.6
                reasons.append(f"Z critico: Z={properties.coordination_number:.2f}")
                reasons.append(f"Proximo de Z_c={properties.coordination_critical:.2f}")
            else:
                reasons.append(f"Z proximo do critico mas rede isotropica")
                reasons.append(f"Direcao incerta. Aguardando anisotropia.")

        # UNJAMMED - Mercado ja liquido
        elif jamming_state == JammingState.UNJAMMED:
            signal_name = 'NEUTRAL'
            reasons.append("UNJAMMED: Mercado ja em estado liquido")
            reasons.append("Tendencia em andamento.")

        else:
            reasons.append(f"Sem setup claro. Estado: {jamming_state.value}")
            reasons.append(f"G={properties.shear_modulus:.4f}, Z={properties.coordination_number:.2f}")

        confidence = np.clip(confidence, 0, 1)

        # Max chain length
        max_chain = int(np.max(force_network.chain_lengths)) if len(force_network.chain_lengths) > 0 else 0

        return {
            'signal': signal,
            'signal_name': signal_name,
            'jamming_state': jamming_state.value,
            'failure_mode': failure_mode.value,
            'confidence': confidence,
            'shear_stress': stress_tensor.shear_stress,
            'normal_stress': stress_tensor.normal_stress,
            'deviatoric_stress': stress_tensor.deviatoric_stress,
            'shear_modulus': properties.shear_modulus,
            'shear_modulus_rate': g_rate,
            'coordination_number': properties.coordination_number,
            'coordination_critical': properties.coordination_critical,
            'compactness': tessellation.compactness,
            'compactness_critical': self.compactness_critical,
            'yield_criterion': mohr_coulomb.yield_criterion,
            'friction_angle': properties.friction_angle,
            'failure_direction': failure_direction,
            'price_direction': price_direction,
            'n_particles': len(particles),
            'max_chain_length': max_chain,
            'is_anisotropic': force_network.is_anisotropic,
            'reasons': reasons
        }

    def get_g_history(self) -> np.ndarray:
        """Retorna historico do modulo de cisalhamento"""
        return np.array(self.jamming_detector.g_history)

    def get_z_history(self) -> np.ndarray:
        """Retorna historico do numero de coordenacao"""
        return np.array(self.jamming_detector.z_history)

    def get_phi_history(self) -> np.ndarray:
        """Retorna historico da compacidade"""
        return np.array(self.jamming_detector.phi_history)

    def reset(self):
        """Reseta o estado do indicador"""
        self.jamming_detector.reset()


# ==============================================================================
# DEMONSTRACAO
# ==============================================================================

def generate_jamming_data(n_points: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Gera dados que simulam transicao de jamming (baixa vol -> rompimento)"""
    np.random.seed(seed)

    t = np.arange(n_points)

    # Fase 1: Baixa volatilidade (jammed) - 70% dos dados
    jammed_phase = int(n_points * 0.7)

    # Precos comprimidos
    base = 1.0850
    prices_jammed = base + np.cumsum(np.random.randn(jammed_phase) * 0.00005)

    # Fase 2: Transicao e rompimento - 30% restante
    transition_phase = n_points - jammed_phase
    prices_transition = prices_jammed[-1] + np.cumsum(np.random.randn(transition_phase) * 0.0002)

    prices = np.concatenate([prices_jammed, prices_transition])

    # Volume (alto durante compressao, pico no rompimento)
    volumes = 1000 + 500 * np.exp(-((t - jammed_phase)/10)**2)
    volumes += np.random.randn(n_points) * 100

    return prices, volumes


def main():
    """Demonstracao do indicador GJ-FCP"""
    print("=" * 70)
    print("GRANULAR JAMMING & FORCE CHAIN PERCOLATOR (GJ-FCP)")
    print("Indicador baseado em Fisica da Materia Granular")
    print("=" * 70)
    print()

    # Inicializa indicador
    indicator = GranularJammingForceChainPercolator(
        particles_per_bar=5,
        compactness_critical=0.64,
        z_critical=4.0,
        g_drop_threshold=0.3,
        min_data_points=50
    )

    print("Indicador inicializado!")
    print(f"  - Particulas por barra: 5")
    print(f"  - phi_critico: 0.64")
    print(f"  - Z_critico: 4.0")
    print()

    # Gera dados
    prices, volumes = generate_jamming_data(n_points=80, seed=42)
    print(f"Dados gerados: {len(prices)} pontos")
    print()

    # Processa
    result = indicator.analyze(prices, volumes)

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Sinal: {result['signal_name']}")
    print(f"Estado Jamming: {result['jamming_state']}")
    print(f"Modo de Falha: {result['failure_mode']}")
    print(f"Confianca: {result['confidence']:.2%}")
    print(f"\nTensao:")
    print(f"  tau (cisalhamento): {result['shear_stress']:.6f}")
    print(f"  sigma_n (normal): {result['normal_stress']:.6f}")
    print(f"  Deviatoric: {result['deviatoric_stress']:.6f}")
    print(f"\nRigidez:")
    print(f"  G (Shear Modulus): {result['shear_modulus']:.6f}")
    print(f"  dG/dt: {result['shear_modulus_rate']:.2%}")
    print(f"\nCoordenacao:")
    print(f"  Z: {result['coordination_number']:.2f}")
    print(f"  Z_c: {result['coordination_critical']:.2f}")
    print(f"\nCompacidade:")
    print(f"  phi: {result['compactness']:.4f}")
    print(f"  phi_c: {result['compactness_critical']:.4f}")
    print(f"\nMohr-Coulomb:")
    print(f"  Yield: {result['yield_criterion']:.6f}")
    print(f"  psi: {np.degrees(result['friction_angle']):.1f} graus")
    print(f"\nDirecao:")
    print(f"  Angulo falha: {np.degrees(result['failure_direction']):.1f} graus")
    print(f"  Preco: {result['price_direction']}")
    print(f"\nRazoes: {result['reasons']}")

    return indicator, result


if __name__ == "__main__":
    indicator, result = main()
