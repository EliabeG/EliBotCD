"""
================================================================================
HOLOGRAPHIC AdS/CFT BULK-BOUNDARY PROJECTOR (HBBP)
Indicador de Forex baseado em Gravidade Entrópica e Correspondência Holográfica
================================================================================

Este indicador não prevê o preço. Ele calcula a emergência do espaço-tempo do
mercado. Ele utiliza a Fórmula de Ryu-Takayanagi para calcular a Entropia de
Emaranhamento da fronteira e deduzir se um "Buraco Negro" (um atrator de preço
massivo) acabou de se formar no futuro invisível.

A Física: Gravidade Entrópica e Tensores
Baseado na hipótese de Erik Verlinde: "A gravidade não é uma força fundamental,
é uma força entrópica emergente". O preço se move não porque alguém comprou, mas
porque o sistema busca maximizar a entropia informacional.

1. O Universo na Fronteira (CFT - Conformal Field Theory)
   O Order Book (Bid/Ask) é o nosso limite 1D (ou 2D se considerarmos o tempo).
   É um sistema quântico de muitos corpos em estado crítico. Nós modelamos o
   book como uma Cadeia de Spin quântica.

2. O Universo no Bulk (AdS - Anti-de Sitter)
   O "Preço Futuro" vive em uma dimensão extra (o Bulk). A geometria desse espaço
   é hiperbólica. Para saber o que vai acontecer no Bulk (Preço), precisamos medir
   o Emaranhamento Quântico na Fronteira (Book).

3. A Fórmula de Ryu-Takayanagi (Prêmio Nobel aplicado ao lucro):
   S_A = Area(γ_A) / 4G_N

   - S_A: Entropia de Emaranhamento de uma região A do Order Book
   - γ_A: A área da superfície mínima (geodésica) que mergulha no Bulk

Tradução para Trading: A complexidade da correlação entre as ordens de compra e
venda define a "geometria" do movimento de preço. Se o emaranhamento sobe, a
geometria do espaço-tempo se curva.

Autor: Claude AI
Versão: 1.0.0
================================================================================
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Set, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import warnings
import logging
from functools import lru_cache

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


# ==============================================================================
# CONSTANTES FÍSICAS (AdS/CFT)
# ==============================================================================

# Constante gravitacional de Newton (normalizada para o mercado)
G_NEWTON = 1.0

# Raio do espaço AdS (escala do bulk)
ADS_RADIUS = 1.0

# Temperatura de Hawking normalizada
HAWKING_TEMPERATURE = 1.0

# Constante de Boltzmann (normalizada)
K_BOLTZMANN = 1.0

# Dimensão central da CFT (para cadeia crítica)
CENTRAL_CHARGE = 1.0

# Cutoff UV (escala mínima)
UV_CUTOFF = 0.01


# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

class HBBPSignalType(Enum):
    """Tipos de sinais de trading"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    WAIT = "WAIT"


class GeometryState(Enum):
    """Estado da geometria do bulk"""
    FLAT = "FLAT"                       # AdS puro, sem massa
    CURVED = "CURVED"                   # Curvatura leve
    BLACK_HOLE = "BLACK_HOLE"           # Buraco negro se formando
    SINGULARITY = "SINGULARITY"         # Singularidade (crash iminente)


class EntanglementPhase(Enum):
    """Fase do emaranhamento"""
    AREA_LAW = "AREA_LAW"               # Lei de área (baixo emaranhamento)
    VOLUME_LAW = "VOLUME_LAW"           # Lei de volume (alto emaranhamento)
    CRITICAL = "CRITICAL"               # Ponto crítico
    THERMAL = "THERMAL"                 # Estado térmico


@dataclass
class QuantumState:
    """Estado quântico do order book"""
    psi: np.ndarray                     # Vetor de estado |ψ⟩
    density_matrix: np.ndarray          # Matriz densidade ρ = |ψ⟩⟨ψ|
    n_sites: int                        # Número de sites na cadeia
    bond_dimension: int                 # Dimensão do bond (para MPS/MERA)


@dataclass
class TensorNetwork:
    """Rede de tensores MERA"""
    layers: List[np.ndarray]            # Camadas de tensores
    isometries: List[np.ndarray]        # Isometrias
    disentanglers: List[np.ndarray]     # Disentanglers
    n_layers: int                       # Número de camadas
    coarse_grained_state: np.ndarray    # Estado após renormalização


@dataclass
class EntanglementEntropy:
    """Entropia de emaranhamento"""
    S_A: float                          # Entropia de von Neumann
    S_renyi_2: float                    # Entropia de Rényi de ordem 2
    mutual_information: float           # Informação mútua I(A:B)
    entanglement_spectrum: np.ndarray   # Espectro de emaranhamento
    area: float                         # "Área" da superfície mínima (Ryu-Takayanagi)


@dataclass
class BulkGeometry:
    """Geometria do espaço bulk (AdS)"""
    metric: np.ndarray                  # Tensor métrico g_μν
    ricci_scalar: float                 # Escalar de Ricci R
    geodesic_length: float              # Comprimento da geodésica mínima
    horizon_radius: float               # Raio do horizonte (se houver buraco negro)
    curvature_tensor: np.ndarray        # Tensor de curvatura
    is_black_hole: bool                 # True se buraco negro presente


@dataclass
class EntropicForce:
    """Força entrópica emergente"""
    F_e: float                          # Magnitude da força
    direction: str                      # "UP" ou "DOWN"
    gradient_S: float                   # ∇S (gradiente da entropia)
    temperature: float                  # Temperatura efetiva T
    inertia: float                      # Inércia do mercado
    net_force: float                    # F_e - inércia


@dataclass
class HolographicProjection:
    """Projeção holográfica do bulk para a fronteira"""
    bulk_field: np.ndarray              # Campo no bulk φ(z, x)
    boundary_operator: np.ndarray       # Operador na fronteira O(x)
    correlation_function: np.ndarray    # Função de correlação ⟨O(x)O(y)⟩
    scaling_dimension: float            # Dimensão de escala Δ


@dataclass
class HBBPResult:
    """Sinal completo do HBBP"""
    signal_type: HBBPSignalType
    geometry_state: GeometryState
    entanglement_phase: EntanglementPhase
    confidence: float

    # Entropia de Emaranhamento
    S_A: float                          # Entropia de von Neumann
    S_renyi: float                      # Entropia de Rényi
    mutual_information: float           # I(A:B)
    ryu_takayanagi_area: float          # Área da superfície mínima

    # Geometria do Bulk
    ricci_scalar: float                 # Curvatura
    geodesic_length: float              # Comprimento geodésico
    horizon_radius: float               # Raio do horizonte
    is_black_hole: bool                 # Buraco negro detectado

    # Força Entrópica
    entropic_force: float               # F_e = T∇S
    force_direction: str                # Direção da força
    gradient_entropy: float             # ∇S
    temperature: float                  # T efetiva
    net_force: float                    # Força líquida

    # MERA
    n_mera_layers: int                  # Camadas de renormalização
    long_range_entanglement: float      # Emaranhamento de longo alcance

    # Trading
    entry_price: float
    stop_loss: float
    take_profit: float

    reason: str
    timestamp: str


# ==============================================================================
# TENSORIZAÇÃO DO ORDER BOOK
# ==============================================================================

class OrderBookTensorizer:
    """
    Passo 1: Tensorização do Book

    Converta o Order Book em um estado quântico |ψ⟩. Normalize os volumes em uma
    rede de tensores hierárquica (uma pirâmide de informação). A base da pirâmide
    são os ticks individuais. O topo da pirâmide é a tendência macro.

    Modelamos o book como uma Cadeia de Spin quântica onde cada "spin" representa
    a direção do fluxo de ordens em cada nível de preço.
    """

    def __init__(self,
                 n_sites: int = 64,
                 bond_dimension: int = 16):
        """
        Args:
            n_sites: Número de sites na cadeia quântica
            bond_dimension: Dimensão do bond para MPS
        """
        self.n_sites = n_sites
        self.bond_dimension = bond_dimension

    def tensorize(self,
                 prices: np.ndarray,
                 volumes: np.ndarray) -> QuantumState:
        """
        Converte dados de mercado em estado quântico

        |ψ⟩ = Σ_i c_i |σ_1, σ_2, ..., σ_n⟩

        onde σ_i ∈ {↑, ↓} representa direção do fluxo
        """
        n = len(prices)

        # Calcula "spins" a partir dos retornos
        returns = np.diff(np.log(prices + 1e-10))

        # Mapeia para spins: ↑ = +1, ↓ = -1
        spins = np.sign(returns)
        spins[spins == 0] = 1  # Neutro → ↑

        # Redimensiona para n_sites
        if len(spins) >= self.n_sites:
            # Coarse-grain
            step = len(spins) // self.n_sites
            spins_resampled = np.array([
                np.sign(np.sum(spins[i*step:(i+1)*step]))
                for i in range(self.n_sites)
            ])
            spins_resampled[spins_resampled == 0] = 1
        else:
            # Pad com zeros
            spins_resampled = np.ones(self.n_sites)
            spins_resampled[:len(spins)] = spins

        # Volume normalizado como amplitude
        if len(volumes) >= self.n_sites:
            step = len(volumes) // self.n_sites
            vol_resampled = np.array([
                np.mean(volumes[i*step:(i+1)*step])
                for i in range(self.n_sites)
            ])
        else:
            vol_resampled = np.ones(self.n_sites) * np.mean(volumes)

        # Normaliza volumes para amplitudes de probabilidade
        amplitudes = vol_resampled / (np.sum(vol_resampled) + 1e-10)
        amplitudes = np.sqrt(amplitudes)  # |c_i|² = p_i

        # Constrói vetor de estado |ψ⟩
        # Para simplificar, usamos representação de produto de estados locais
        # |ψ⟩ = |ψ_1⟩ ⊗ |ψ_2⟩ ⊗ ... ⊗ |ψ_n⟩
        # onde |ψ_i⟩ = α|↑⟩ + β|↓⟩

        psi = np.zeros((self.n_sites, 2), dtype=complex)

        for i in range(self.n_sites):
            if spins_resampled[i] > 0:
                # Mais provável ↑
                psi[i, 0] = amplitudes[i] * np.sqrt(0.8)  # |↑⟩
                psi[i, 1] = amplitudes[i] * np.sqrt(0.2)  # |↓⟩
            else:
                # Mais provável ↓
                psi[i, 0] = amplitudes[i] * np.sqrt(0.2)
                psi[i, 1] = amplitudes[i] * np.sqrt(0.8)

        # Normaliza
        norm = np.sqrt(np.sum(np.abs(psi)**2))
        psi = psi / (norm + 1e-10)

        # Matriz densidade reduzida (para cada site)
        # ρ_i = Tr_{j≠i}(|ψ⟩⟨ψ|)
        density_matrix = np.zeros((self.n_sites, 2, 2), dtype=complex)

        for i in range(self.n_sites):
            density_matrix[i] = np.outer(psi[i], np.conj(psi[i]))

        return QuantumState(
            psi=psi,
            density_matrix=density_matrix,
            n_sites=self.n_sites,
            bond_dimension=self.bond_dimension
        )


# ==============================================================================
# MERA (MULTI-SCALE ENTANGLEMENT RENORMALIZATION ANSATZ)
# ==============================================================================

class MERANetwork:
    """
    Passo 2: Renormalização em Tempo Real

    Execute o algoritmo MERA para comprimir a informação da base para o topo.
    Isso elimina as correlações de curto alcance (ruído HFT) e preserva apenas
    o emaranhamento de longo alcance (estrutura institucional).

    MERA é uma rede de tensores hierárquica que implementa renormalização
    do grupo via:
    1. Disentanglers: Removem emaranhamento de curto alcance
    2. Isometries: Fazem coarse-graining espacial
    """

    def __init__(self,
                 n_layers: int = 4,
                 bond_dim: int = 4):
        """
        Args:
            n_layers: Número de camadas de renormalização
            bond_dim: Dimensão do bond (controla precisão)
        """
        self.n_layers = n_layers
        self.bond_dim = bond_dim

    def build_network(self, quantum_state: QuantumState) -> TensorNetwork:
        """
        Constrói a rede MERA a partir do estado quântico
        """
        n_sites = quantum_state.n_sites

        layers = []
        isometries = []
        disentanglers = []

        current_state = quantum_state.psi.flatten()

        # Constrói camadas de MERA
        sites_per_layer = n_sites

        for layer in range(self.n_layers):
            if sites_per_layer < 2:
                break

            # Disentangler: Remove emaranhamento local
            # U: C² ⊗ C² → C² ⊗ C²
            disentangler = self._create_disentangler(sites_per_layer)
            disentanglers.append(disentangler)

            # Aplica disentangler (simbolicamente)
            current_state = self._apply_disentangler(current_state, disentangler)

            # Isometria: Coarse-graining 2→1
            # W: C² ⊗ C² → C^bond_dim
            isometry = self._create_isometry(sites_per_layer)
            isometries.append(isometry)

            # Aplica isometria (reduz número de sites pela metade)
            current_state = self._apply_isometry(current_state, isometry)

            # Armazena estado da camada
            layers.append(current_state.copy())

            sites_per_layer = sites_per_layer // 2

        return TensorNetwork(
            layers=layers,
            isometries=isometries,
            disentanglers=disentanglers,
            n_layers=len(layers),
            coarse_grained_state=current_state
        )

    def _create_disentangler(self, n_sites: int) -> np.ndarray:
        """
        Cria tensor disentangler

        O disentangler é uma unitária que remove correlações de curto alcance
        """
        # Disentangler simples: rotação local
        # Para uma implementação real, isso seria otimizado
        theta = np.pi / 4  # Ângulo de rotação

        # Matriz de rotação 4x4 (para 2 qubits)
        U = np.array([
            [np.cos(theta), 0, 0, -np.sin(theta)],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [np.sin(theta), 0, 0, np.cos(theta)]
        ], dtype=complex)

        return U

    def _create_isometry(self, n_sites: int) -> np.ndarray:
        """
        Cria tensor isometria

        A isometria mapeia 2 sites → 1 site (coarse-graining)
        W†W = I
        """
        # Isometria simples: projeção para estado de menor energia
        # W: C^4 → C^2
        W = np.array([
            [1, 0, 0, 0],
            [0, 1/np.sqrt(2), 1/np.sqrt(2), 0]
        ], dtype=complex) / np.sqrt(2)

        return W

    def _apply_disentangler(self, state: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Aplica disentangler ao estado"""
        # Simplificação: aplica transformação linear
        n = len(state)
        new_state = state.copy()

        # Aplica em pares de sites
        for i in range(0, n - 3, 4):
            if i + 4 <= n:
                local = state[i:i+4]
                if len(local) == 4:
                    new_state[i:i+4] = U @ local

        return new_state

    def _apply_isometry(self, state: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Aplica isometria (coarse-graining)"""
        n = len(state)

        # Reduz pela metade
        new_n = max(2, n // 2)
        new_state = np.zeros(new_n, dtype=complex)

        for i in range(new_n):
            if 2*i + 1 < n:
                # Média ponderada (simplificação)
                new_state[i] = (state[2*i] + state[2*i + 1]) / np.sqrt(2)
            elif 2*i < n:
                new_state[i] = state[2*i]

        # Normaliza
        norm = np.linalg.norm(new_state)
        if norm > 0:
            new_state = new_state / norm

        return new_state

    def compute_long_range_entanglement(self, network: TensorNetwork) -> float:
        """
        Calcula emaranhamento de longo alcance preservado pela MERA

        Após remover correlações de curto alcance, o que resta é a
        estrutura institucional de longo prazo
        """
        if len(network.layers) == 0:
            return 0.0

        # Emaranhamento no topo da pirâmide
        top_state = network.coarse_grained_state

        # Entropia do estado coarse-grained
        probs = np.abs(top_state)**2
        probs = probs[probs > 1e-10]  # Remove zeros

        if len(probs) == 0:
            return 0.0

        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Normaliza pela entropia máxima
        max_entropy = np.log2(len(top_state))

        return entropy / (max_entropy + 1e-10)


# ==============================================================================
# ENTROPIA DE EMARANHAMENTO (RYU-TAKAYANAGI)
# ==============================================================================

class EntanglementEntropyCalculator:
    """
    Fórmula de Ryu-Takayanagi:

    S_A = Area(γ_A) / 4G_N

    - S_A: Entropia de Emaranhamento de uma região A do Order Book
    - γ_A: A área da superfície mínima (geodésica) que mergulha no Bulk
    - G_N: Constante de Newton (normalizada)

    A entropia de emaranhamento na fronteira (CFT) é igual à área da
    superfície mínima no bulk (AdS) dividida por 4G_N.
    """

    def __init__(self,
                 G_N: float = G_NEWTON,
                 central_charge: float = CENTRAL_CHARGE):
        """
        Args:
            G_N: Constante gravitacional de Newton
            central_charge: Carga central da CFT
        """
        self.G_N = G_N
        self.central_charge = central_charge

    def compute_von_neumann_entropy(self,
                                   quantum_state: QuantumState,
                                   subsystem_size: int = None) -> float:
        """
        Calcula entropia de von Neumann: S = -Tr(ρ log ρ)

        Para CFT crítica 1D: S ∝ (c/3) log(L/a)
        onde c é a carga central, L é o tamanho do subsistema, a é o cutoff
        """
        if subsystem_size is None:
            subsystem_size = quantum_state.n_sites // 2

        # Matriz densidade reduzida do subsistema
        rho_A = self._compute_reduced_density_matrix(
            quantum_state, subsystem_size
        )

        # Autovalores de ρ_A
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove zeros

        if len(eigenvalues) == 0:
            return 0.0

        # S = -Σ λ_i log(λ_i)
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))

        return entropy

    def compute_renyi_entropy(self,
                             quantum_state: QuantumState,
                             order: int = 2,
                             subsystem_size: int = None) -> float:
        """
        Calcula entropia de Rényi: S_n = (1/(1-n)) log(Tr(ρ^n))

        S_2 é particularmente útil pois é mais fácil de medir experimentalmente
        """
        if subsystem_size is None:
            subsystem_size = quantum_state.n_sites // 2

        rho_A = self._compute_reduced_density_matrix(
            quantum_state, subsystem_size
        )

        # Tr(ρ^n)
        rho_n = np.linalg.matrix_power(rho_A, order)
        trace_rho_n = np.real(np.trace(rho_n))

        if trace_rho_n <= 0:
            return 0.0

        # S_n = (1/(1-n)) log(Tr(ρ^n))
        entropy = np.log(trace_rho_n + 1e-10) / (1 - order)

        return entropy

    def compute_mutual_information(self,
                                  quantum_state: QuantumState) -> float:
        """
        Calcula informação mútua: I(A:B) = S_A + S_B - S_AB

        Mede correlações totais (clássicas + quânticas) entre A e B
        """
        n = quantum_state.n_sites

        # Divide em duas metades
        S_A = self.compute_von_neumann_entropy(quantum_state, n // 2)
        S_B = self.compute_von_neumann_entropy(quantum_state, n // 2)
        S_AB = self.compute_von_neumann_entropy(quantum_state, n)

        return S_A + S_B - S_AB

    def compute_ryu_takayanagi_area(self,
                                   entropy: float) -> float:
        """
        Calcula a "área" da superfície mínima usando Ryu-Takayanagi invertida

        Area(γ_A) = 4 G_N × S_A
        """
        return 4 * self.G_N * entropy

    def _compute_reduced_density_matrix(self,
                                       quantum_state: QuantumState,
                                       subsystem_size: int) -> np.ndarray:
        """
        Calcula matriz densidade reduzida do subsistema A

        ρ_A = Tr_B(|ψ⟩⟨ψ|)
        """
        psi = quantum_state.psi
        n_sites = quantum_state.n_sites

        # Para simplificar, usamos aproximação de produto
        # ρ_A ≈ ⊗_i ρ_i para sites em A

        subsystem_size = min(subsystem_size, n_sites)

        # Constrói matriz densidade do subsistema
        # Dimensão: 2^subsystem_size (exponencial!)
        # Para evitar explosão, usamos aproximação

        if subsystem_size > 10:
            # Aproximação: média das matrizes densidade locais
            rho_A = np.zeros((2, 2), dtype=complex)
            for i in range(subsystem_size):
                rho_A += np.outer(psi[i], np.conj(psi[i]))
            rho_A /= subsystem_size
        else:
            # Produto tensorial das matrizes locais
            rho_A = np.outer(psi[0], np.conj(psi[0]))
            for i in range(1, min(subsystem_size, 4)):
                rho_i = np.outer(psi[i], np.conj(psi[i]))
                rho_A = np.kron(rho_A, rho_i)

        # Garante traço 1
        trace = np.trace(rho_A)
        if np.abs(trace) > 1e-10:
            rho_A = rho_A / trace

        return rho_A

    def compute_entanglement_spectrum(self,
                                     quantum_state: QuantumState) -> np.ndarray:
        """
        Calcula o espectro de emaranhamento

        λ_i são os autovalores de ρ_A. O espectro revela a estrutura
        do emaranhamento.
        """
        rho_A = self._compute_reduced_density_matrix(
            quantum_state, quantum_state.n_sites // 2
        )

        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Ordem decrescente

        return eigenvalues

    def analyze_entanglement(self,
                            quantum_state: QuantumState) -> EntanglementEntropy:
        """
        Análise completa do emaranhamento
        """
        S_vN = self.compute_von_neumann_entropy(quantum_state)
        S_renyi = self.compute_renyi_entropy(quantum_state, order=2)
        MI = self.compute_mutual_information(quantum_state)
        spectrum = self.compute_entanglement_spectrum(quantum_state)
        area = self.compute_ryu_takayanagi_area(S_vN)

        return EntanglementEntropy(
            S_A=S_vN,
            S_renyi_2=S_renyi,
            mutual_information=MI,
            entanglement_spectrum=spectrum,
            area=area
        )


# ==============================================================================
# GEOMETRIA DO BULK (AdS)
# ==============================================================================

class BulkGeometryAnalyzer:
    """
    Analisa a geometria do espaço bulk (Anti-de Sitter)

    A métrica AdS em coordenadas de Poincaré:
    ds² = (L²/z²)(dz² + dx²)

    onde z é a coordenada radial (direção do bulk) e x é a fronteira.

    Um buraco negro no bulk corresponde a um estado térmico na fronteira.
    """

    def __init__(self,
                 ads_radius: float = ADS_RADIUS,
                 G_N: float = G_NEWTON):
        """
        Args:
            ads_radius: Raio do espaço AdS
            G_N: Constante de Newton
        """
        self.L = ads_radius
        self.G_N = G_N

    def compute_metric(self, z: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Calcula o tensor métrico AdS

        g_μν = (L²/z²) × diag(1, 1)
        """
        n = len(z)
        metric = np.zeros((n, 2, 2))

        for i in range(n):
            factor = (self.L / (z[i] + UV_CUTOFF))**2
            metric[i] = factor * np.eye(2)

        return metric

    def compute_geodesic_length(self,
                               entanglement_entropy: EntanglementEntropy,
                               subsystem_size: float) -> float:
        """
        Calcula comprimento da geodésica mínima que conecta os pontos
        na fronteira e mergulha no bulk

        Para AdS₃ puro: Length(γ) = 2L × arcosh(L/2a)
        onde a é o tamanho do subsistema

        Pela fórmula de Ryu-Takayanagi:
        S = Length(γ) / 4G_N
        ∴ Length(γ) = 4G_N × S
        """
        # Usa a área de Ryu-Takayanagi diretamente
        geodesic_length = entanglement_entropy.area

        return geodesic_length

    def compute_ricci_scalar(self, entropy: float, subsystem_size: float) -> float:
        """
        Calcula o escalar de Ricci a partir da entropia

        Para AdS puro: R = -d(d+1)/L²

        Desvios indicam matéria/energia no bulk (buraco negro)
        """
        # AdS₃ puro: R = -6/L²
        R_ads = -6 / (self.L**2)

        # Correção devido ao emaranhamento
        # Mais emaranhamento → mais curvatura
        correction = entropy / (subsystem_size + 1)

        R = R_ads * (1 + correction)

        return R

    def detect_black_hole(self,
                         entanglement: EntanglementEntropy,
                         threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Detecta formação de buraco negro no bulk

        Um buraco negro corresponde a:
        1. Alta entropia (lei de volume em vez de lei de área)
        2. Espectro de emaranhamento térmico
        3. Horizonte de eventos

        Returns:
            (is_black_hole, horizon_radius)
        """
        # Lei de área vs lei de volume
        # Lei de área: S ∝ L^(d-1) (sistemas gapped)
        # Lei de volume: S ∝ L^d (sistemas críticos/térmicos)

        # Para detectar, olhamos a entropia normalizada
        S_normalized = entanglement.S_A

        # Se S > threshold, temos lei de volume → estado térmico → buraco negro
        # Também considera alta informação mútua como indicador
        is_black_hole = (S_normalized > threshold or
                        entanglement.mutual_information > threshold * 1.2 or
                        entanglement.area > threshold * 3)

        # Raio do horizonte (pela fórmula de Bekenstein-Hawking invertida)
        # S = A / 4G_N = π r_h² / G_N
        # ∴ r_h = sqrt(G_N × S / π)
        if is_black_hole and entanglement.S_A > 0:
            horizon_radius = np.sqrt(self.G_N * entanglement.S_A / np.pi)
        else:
            horizon_radius = 0.0

        return is_black_hole, horizon_radius

    def analyze_geometry(self,
                        entanglement: EntanglementEntropy,
                        n_sites: int) -> BulkGeometry:
        """
        Análise completa da geometria do bulk
        """
        # Coordenadas
        z = np.linspace(UV_CUTOFF, 1.0, 20)
        x = np.linspace(0, 1, n_sites)

        # Métrica
        metric = self.compute_metric(z, x[:20])

        # Geodésica
        geodesic = self.compute_geodesic_length(entanglement, n_sites)

        # Curvatura
        ricci = self.compute_ricci_scalar(entanglement.S_A, n_sites)

        # Buraco negro
        is_bh, horizon = self.detect_black_hole(entanglement)

        # Tensor de curvatura (simplificado)
        curvature = np.zeros((2, 2, 2, 2))
        curvature[0, 1, 0, 1] = ricci / 2
        curvature[1, 0, 1, 0] = ricci / 2

        return BulkGeometry(
            metric=metric,
            ricci_scalar=ricci,
            geodesic_length=geodesic,
            horizon_radius=horizon,
            curvature_tensor=curvature,
            is_black_hole=is_bh
        )


# ==============================================================================
# FORÇA ENTRÓPICA
# ==============================================================================

class EntropicForceCalculator:
    """
    Passo 3: Cálculo da Força Entrópica (F_e)

    Calcule a variação da Entropia de Emaranhamento (S) em relação à posição
    do preço (x).

    F_e = T × ∇_x S

    - Se ∇S = 0: O espaço é plano. O preço flutua sem direção (Baixa Volatilidade Absoluta).
    - Se ∇S ≠ 0: Existe uma força entrópica empurrando o preço.

    Baseado na Gravidade Entrópica de Verlinde:
    "A gravidade não é fundamental, é uma força entrópica emergente"
    """

    def __init__(self,
                 temperature: float = HAWKING_TEMPERATURE,
                 k_B: float = K_BOLTZMANN):
        """
        Args:
            temperature: Temperatura efetiva do mercado
            k_B: Constante de Boltzmann
        """
        self.T = temperature
        self.k_B = k_B

    def compute_entropy_gradient(self,
                                prices: np.ndarray,
                                entropies: np.ndarray) -> float:
        """
        Calcula ∇_x S (gradiente da entropia em relação ao preço)
        """
        if len(entropies) < 2:
            return 0.0

        # Gradiente numérico
        dS = np.diff(entropies)
        dx = np.diff(prices)

        # Evita divisão por zero
        dx[np.abs(dx) < 1e-10] = 1e-10

        gradient = np.mean(dS / dx)

        return gradient

    def compute_entropic_force(self,
                              gradient_S: float,
                              temperature: float = None) -> Tuple[float, str]:
        """
        Calcula a força entrópica: F_e = T × ∇S

        Returns:
            (magnitude, direction)
        """
        T = temperature if temperature is not None else self.T

        F_e = T * gradient_S

        if F_e > 0:
            direction = "UP"
        elif F_e < 0:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        return abs(F_e), direction

    def compute_market_inertia(self,
                              volumes: np.ndarray,
                              prices: np.ndarray) -> float:
        """
        Calcula a "inércia" do mercado

        Análogo à massa em mecânica: maior volume = maior inércia
        """
        # Inércia proporcional ao volume médio
        avg_volume = np.mean(volumes)

        # Também considera a volatilidade (mercado volátil tem menos inércia)
        returns = np.diff(np.log(prices + 1e-10))
        volatility = np.std(returns)

        inertia = avg_volume / (volatility + 1e-10)

        # Normaliza
        inertia = np.log1p(inertia)

        return inertia

    def compute_effective_temperature(self,
                                     prices: np.ndarray,
                                     volumes: np.ndarray) -> float:
        """
        Calcula a temperatura efetiva do mercado

        Análogo à temperatura de Hawking para buracos negros
        T_H = ℏc³ / (8πGMk_B)

        Para o mercado: T ∝ volatilidade
        """
        returns = np.diff(np.log(prices + 1e-10))
        volatility = np.std(returns) * np.sqrt(252)  # Anualizada

        # Temperatura proporcional à volatilidade
        T = self.T * (1 + volatility * 10)

        return T

    def analyze_force(self,
                     prices: np.ndarray,
                     volumes: np.ndarray,
                     entropy_history: List[float]) -> EntropicForce:
        """
        Análise completa da força entrópica
        """
        # Gradiente de entropia
        if len(entropy_history) >= 2:
            gradient = self.compute_entropy_gradient(
                prices[-len(entropy_history):],
                np.array(entropy_history)
            )
        else:
            gradient = 0.0

        # Temperatura efetiva
        T = self.compute_effective_temperature(prices, volumes)

        # Força entrópica
        F_magnitude, direction = self.compute_entropic_force(gradient, T)

        # Inércia
        inertia = self.compute_market_inertia(volumes, prices)

        # Força líquida
        if direction == "UP":
            net_force = F_magnitude - inertia * 0.1
        elif direction == "DOWN":
            net_force = -F_magnitude + inertia * 0.1
        else:
            net_force = 0.0

        return EntropicForce(
            F_e=F_magnitude,
            direction=direction,
            gradient_S=gradient,
            temperature=T,
            inertia=inertia,
            net_force=net_force
        )


# ==============================================================================
# DETECTOR DE SINGULARIDADE HOLOGRÁFICA
# ==============================================================================

class HolographicSingularityDetector:
    """
    A Lógica de Trading (A Singularidade Holográfica)
    O indicador busca uma Anomalia na Fronteira.

    1. O Vácuo de Baixa Volatilidade: O indicador MERA mostra que a geometria do
       Bulk é plana (AdS puro). Não há "massa" no centro. O preço é ruído térmico.

    2. A Nucleação do Buraco Negro: De repente, sem que o preço se mova, a Entropia
       de Emaranhamento no Order Book dispara em uma região específica (ex: alta
       correlação entre ordens passivas de venda e ordens agressivas de compra).
       A fórmula Ryu-Takayanagi indica que uma "Superfície Mínima" se formou no Bulk.
       Isso significa que um Atrator Gravitacional nasceu no futuro probabilístico.

    3. SINAL (O Salto Holográfico): A força entrópica F_e supera a inércia do mercado.
       - Setup: O preço está parado, mas a Entropia Holográfica sugere uma curvatura extrema.
       - Direção: O preço "cai" na direção do aumento de entropia.
       - Timing: Instantâneo. A informação na fronteira (book) e a geometria do bulk
         (preço) são duais. Assim que a entropia muda, o preço DEVE se mover para
         satisfazer a equação de Einstein emergente.
    """

    def __init__(self,
                 entropy_threshold: float = 1.5,
                 force_threshold: float = 0.5):
        """
        Args:
            entropy_threshold: Limiar para detectar nucleação de buraco negro
            force_threshold: Limiar para força entrópica significativa
        """
        self.entropy_threshold = entropy_threshold
        self.force_threshold = force_threshold
        self.entropy_history = []

    def detect_anomaly(self,
                      entanglement: EntanglementEntropy,
                      bulk_geometry: BulkGeometry,
                      entropic_force: EntropicForce) -> Tuple[bool, str, float]:
        """
        Detecta anomalia holográfica (nucleação de buraco negro)

        Returns:
            (anomaly_detected, anomaly_type, severity)
        """
        # Adiciona ao histórico
        self.entropy_history.append(entanglement.S_A)
        if len(self.entropy_history) > 100:
            self.entropy_history.pop(0)

        # Detecta spike de entropia
        entropy_spike = False
        if len(self.entropy_history) >= 3:
            recent_avg = np.mean(self.entropy_history[-3:])
            if len(self.entropy_history) > 3:
                historical_avg = np.mean(self.entropy_history[:-3])
                entropy_spike = recent_avg > historical_avg * self.entropy_threshold
            else:
                entropy_spike = recent_avg > 0.5
        else:
            entropy_spike = entanglement.S_A > self.entropy_threshold

        # Verifica buraco negro
        black_hole_formed = bulk_geometry.is_black_hole

        # Verifica força entrópica (mais sensível)
        force_significant = (abs(entropic_force.net_force) > self.force_threshold or
                           abs(entropic_force.gradient_S) > 0.001 or
                           entropic_force.F_e > self.force_threshold)

        # Alta entropia também é sinal
        high_entropy = entanglement.S_A > 1.0 or entanglement.mutual_information > 0.5

        # Classifica anomalia
        if black_hole_formed and (entropy_spike or high_entropy) and force_significant:
            return True, "BLACK_HOLE_NUCLEATION", 1.0
        elif black_hole_formed and (entropy_spike or high_entropy):
            return True, "BLACK_HOLE_NUCLEATION", 0.8
        elif (entropy_spike or high_entropy) and force_significant:
            return True, "ENTROPY_SPIKE", 0.7
        elif black_hole_formed:
            return True, "HORIZON_FORMATION", 0.5
        elif entropy_spike or high_entropy:
            return True, "ENTROPY_RISING", 0.4
        elif entanglement.S_A > 0.3:
            return False, "ENTROPY_RISING", 0.3
        else:
            return False, "VACUUM", 0.0

    def classify_geometry_state(self,
                               bulk_geometry: BulkGeometry,
                               entanglement: EntanglementEntropy) -> GeometryState:
        """
        Classifica o estado da geometria do bulk
        """
        if bulk_geometry.is_black_hole and bulk_geometry.horizon_radius > 0.5:
            return GeometryState.SINGULARITY
        elif bulk_geometry.is_black_hole:
            return GeometryState.BLACK_HOLE
        elif abs(bulk_geometry.ricci_scalar) > 10:
            return GeometryState.CURVED
        else:
            return GeometryState.FLAT

    def classify_entanglement_phase(self,
                                   entanglement: EntanglementEntropy,
                                   n_sites: int) -> EntanglementPhase:
        """
        Classifica a fase do emaranhamento

        - Lei de área: S ∝ log(L) para CFT crítica
        - Lei de volume: S ∝ L para estados térmicos
        """
        S = entanglement.S_A
        L = n_sites

        # Esperado para lei de área (CFT crítica): S ∝ log(L)
        expected_area_law = np.log(L + 1)

        # Esperado para lei de volume: S ∝ L
        expected_volume_law = L * 0.1

        # Classifica
        ratio_to_area = S / (expected_area_law + 1e-10)
        ratio_to_volume = S / (expected_volume_law + 1e-10)

        if ratio_to_volume > 0.5:
            return EntanglementPhase.VOLUME_LAW  # Térmico/Buraco negro
        elif ratio_to_area > 1.5:
            return EntanglementPhase.CRITICAL  # Ponto crítico
        elif ratio_to_area > 0.5:
            return EntanglementPhase.AREA_LAW  # Ground state
        else:
            return EntanglementPhase.THERMAL

    def reset(self):
        """Reseta o histórico"""
        self.entropy_history.clear()


# ==============================================================================
# INDICADOR HBBP COMPLETO
# ==============================================================================

class HolographicAdSCFTBulkBoundaryProjector:
    """
    HOLOGRAPHIC AdS/CFT BULK-BOUNDARY PROJECTOR (HBBP)

    Indicador completo que usa correspondência holográfica para calcular
    a emergência do espaço-tempo do mercado e detectar "buracos negros"
    (atratores de preço massivos) antes que sejam visíveis.
    """

    def __init__(self,
                 # Parâmetros quânticos
                 n_sites: int = 64,
                 bond_dimension: int = 16,

                 # Parâmetros MERA
                 n_mera_layers: int = 4,

                 # Parâmetros físicos
                 G_N: float = G_NEWTON,
                 ads_radius: float = ADS_RADIUS,
                 temperature: float = HAWKING_TEMPERATURE,

                 # Parâmetros de detecção
                 entropy_threshold: float = 1.5,
                 force_threshold: float = 0.3,

                 # Geral
                 min_data_points: int = 50):
        """
        Inicializa o HBBP
        """
        self.n_sites = n_sites
        self.min_data_points = min_data_points
        self.bond_dimension = bond_dimension
        self.n_mera_layers = n_mera_layers
        self.entropy_threshold = entropy_threshold
        self.force_threshold = force_threshold

        # Componentes
        self.tensorizer = OrderBookTensorizer(
            n_sites=n_sites,
            bond_dimension=bond_dimension
        )

        self.mera = MERANetwork(
            n_layers=n_mera_layers,
            bond_dim=bond_dimension
        )

        self.entropy_calculator = EntanglementEntropyCalculator(
            G_N=G_N,
            central_charge=CENTRAL_CHARGE
        )

        self.geometry_analyzer = BulkGeometryAnalyzer(
            ads_radius=ads_radius,
            G_N=G_N
        )

        self.force_calculator = EntropicForceCalculator(
            temperature=temperature
        )

        self.singularity_detector = HolographicSingularityDetector(
            entropy_threshold=entropy_threshold,
            force_threshold=force_threshold
        )

    def analyze(self,
               prices: np.ndarray,
               volumes: np.ndarray = None) -> dict:
        """
        Processa dados de mercado e gera resultado de análise

        Args:
            prices: Array de preços
            volumes: Array de volumes (opcional)

        Returns:
            Dict com todos os resultados da análise
        """
        from datetime import datetime

        n = len(prices)

        # Validação
        if n < self.min_data_points:
            return self._create_empty_result(
                f"INSUFFICIENT_DATA: {n} < {self.min_data_points}"
            )

        # Volumes sintéticos se não fornecidos
        if volumes is None:
            volumes = np.abs(np.diff(prices))
            volumes = np.append(volumes, volumes[-1] if len(volumes) > 0 else 1.0)
            volumes = volumes * 10000 + 1000

        current_price = prices[-1]

        # ============================================================
        # PASSO 1: TENSORIZAÇÃO DO BOOK
        # ============================================================
        quantum_state = self.tensorizer.tensorize(prices, volumes)

        # ============================================================
        # PASSO 2: RENORMALIZAÇÃO MERA
        # ============================================================
        tensor_network = self.mera.build_network(quantum_state)
        long_range_entanglement = self.mera.compute_long_range_entanglement(tensor_network)

        # ============================================================
        # PASSO 3: ENTROPIA DE EMARANHAMENTO (RYU-TAKAYANAGI)
        # ============================================================
        entanglement = self.entropy_calculator.analyze_entanglement(quantum_state)

        # ============================================================
        # PASSO 4: GEOMETRIA DO BULK (AdS)
        # ============================================================
        bulk_geometry = self.geometry_analyzer.analyze_geometry(
            entanglement, quantum_state.n_sites
        )

        # ============================================================
        # PASSO 5: FORÇA ENTRÓPICA
        # ============================================================
        entropic_force = self.force_calculator.analyze_force(
            prices, volumes,
            self.singularity_detector.entropy_history + [entanglement.S_A]
        )

        # ============================================================
        # PASSO 6: DETECÇÃO DE SINGULARIDADE
        # ============================================================
        anomaly_detected, anomaly_type, severity = self.singularity_detector.detect_anomaly(
            entanglement, bulk_geometry, entropic_force
        )

        geometry_state = self.singularity_detector.classify_geometry_state(
            bulk_geometry, entanglement
        )

        entanglement_phase = self.singularity_detector.classify_entanglement_phase(
            entanglement, quantum_state.n_sites
        )

        # ============================================================
        # PASSO 7: GERAÇÃO DE SINAL
        # ============================================================
        signal = 0
        signal_name = "NEUTRAL"
        confidence = 0.0
        reason = ""

        entry_price = current_price
        stop_loss = current_price
        take_profit = current_price

        # Lógica de sinal baseada na física holográfica

        # 1. NUCLEAÇÃO DE BURACO NEGRO - Sinal forte!
        if anomaly_type == "BLACK_HOLE_NUCLEATION":
            if entropic_force.direction == "UP":
                signal = 1
                signal_name = "LONG"
                stop_loss = current_price * 0.995
                take_profit = current_price * 1.015
            elif entropic_force.direction == "DOWN":
                signal = -1
                signal_name = "SHORT"
                stop_loss = current_price * 1.005
                take_profit = current_price * 0.985

            confidence = min(0.95, 0.6 + severity * 0.4)

            reason = (f"NUCLEACAO DE BURACO NEGRO! Entropia disparou (S={entanglement.S_A:.3f}). "
                     f"Horizonte r_h={bulk_geometry.horizon_radius:.3f}. "
                     f"Forca entropica F_e={entropic_force.F_e:.3f} empurrando {entropic_force.direction}.")

        # 2. SPIKE DE ENTROPIA - Sinal moderado
        elif anomaly_type == "ENTROPY_SPIKE":
            if entropic_force.direction == "UP":
                signal = 1
                signal_name = "LONG"
                stop_loss = current_price * 0.997
                take_profit = current_price * 1.010
            elif entropic_force.direction == "DOWN":
                signal = -1
                signal_name = "SHORT"
                stop_loss = current_price * 1.003
                take_profit = current_price * 0.990
            else:
                signal_name = "WAIT"

            confidence = min(0.80, 0.5 + severity * 0.3)

            reason = (f"SPIKE DE ENTROPIA! S subiu para {entanglement.S_A:.3f}. "
                     f"Superficie minima (Ryu-Takayanagi) = {entanglement.area:.3f}.")

        # 3. HORIZONTE SE FORMANDO
        elif anomaly_type == "HORIZON_FORMATION":
            signal_name = "WAIT"
            confidence = 0.6

            reason = (f"HORIZONTE DE EVENTOS se formando! r_h={bulk_geometry.horizon_radius:.3f}. "
                     f"Geometria: {geometry_state.value}.")

        # 4. ENTROPIA SUBINDO - Preparar entrada
        elif anomaly_type == "ENTROPY_RISING" and anomaly_detected:
            returns = np.diff(np.log(prices[-20:] + 1e-10))
            trend = np.mean(returns)

            if trend > 0:
                signal = 1
                signal_name = "LONG"
                stop_loss = current_price * 0.998
                take_profit = current_price * 1.008
            elif trend < 0:
                signal = -1
                signal_name = "SHORT"
                stop_loss = current_price * 1.002
                take_profit = current_price * 0.992
            else:
                signal_name = "WAIT"

            confidence = min(0.65, 0.3 + severity * 0.5)

            reason = (f"ENTROPIA SUBINDO com tendencia! S={entanglement.S_A:.3f}. "
                     f"Fase: {entanglement_phase.value}.")

        # 5. FORÇA ENTRÓPICA ALTA (sem anomalia clara)
        elif abs(entropic_force.F_e) > 1.0 and entropic_force.direction != "NEUTRAL":
            if entropic_force.direction == "UP":
                signal = 1
                signal_name = "LONG"
                stop_loss = current_price * 0.997
                take_profit = current_price * 1.010
            else:
                signal = -1
                signal_name = "SHORT"
                stop_loss = current_price * 1.003
                take_profit = current_price * 0.990

            confidence = min(0.70, 0.3 + abs(entropic_force.F_e) * 0.05)

            reason = (f"FORCA ENTROPICA SIGNIFICATIVA! F_e={entropic_force.F_e:.2f}, "
                     f"direcao={entropic_force.direction}.")

        # 6. ENTROPIA SUBINDO - Observar
        elif anomaly_type == "ENTROPY_RISING":
            signal_name = "WAIT"
            confidence = 0.4

            reason = (f"ENTROPIA SUBINDO. S={entanglement.S_A:.3f}. "
                     f"Fase: {entanglement_phase.value}.")

        # 7. GEOMETRIA CURVADA mas sem buraco negro
        elif geometry_state == GeometryState.CURVED:
            signal_name = "WAIT"
            confidence = 0.3

            reason = (f"GEOMETRIA CURVADA (R={bulk_geometry.ricci_scalar:.3f}). "
                     f"I(A:B)={entanglement.mutual_information:.3f}.")

        # 8. VÁCUO - Espaço plano
        elif geometry_state == GeometryState.FLAT:
            signal_name = "NEUTRAL"
            confidence = 0.1

            reason = (f"VACUO (AdS puro). Geometria plana, R={bulk_geometry.ricci_scalar:.3f}. "
                     f"Sem atrator gravitacional no bulk.")

        else:
            reason = (f"Estado indefinido. Geometria: {geometry_state.value}. "
                     f"Fase: {entanglement_phase.value}. S={entanglement.S_A:.3f}.")

        confidence = np.clip(confidence, 0, 1)

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'geometry_state': geometry_state.value,
            'entanglement_phase': entanglement_phase.value,
            'anomaly_type': anomaly_type,
            'anomaly_detected': anomaly_detected,
            'severity': severity,

            # Entropia
            'S_A': entanglement.S_A,
            'S_renyi': entanglement.S_renyi_2,
            'mutual_information': entanglement.mutual_information,
            'ryu_takayanagi_area': entanglement.area,

            # Geometria
            'ricci_scalar': bulk_geometry.ricci_scalar,
            'geodesic_length': bulk_geometry.geodesic_length,
            'horizon_radius': bulk_geometry.horizon_radius,
            'is_black_hole': bulk_geometry.is_black_hole,

            # Força entrópica
            'entropic_force': entropic_force.F_e,
            'force_direction': entropic_force.direction,
            'gradient_entropy': entropic_force.gradient_S,
            'temperature': entropic_force.temperature,
            'inertia': entropic_force.inertia,
            'net_force': entropic_force.net_force,

            # MERA
            'n_mera_layers': tensor_network.n_layers,
            'long_range_entanglement': long_range_entanglement,
            'n_sites': quantum_state.n_sites,

            # Trading
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,

            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }

    def _create_empty_result(self, reason: str) -> dict:
        """Cria resultado vazio"""
        from datetime import datetime

        return {
            'signal': 0,
            'signal_name': 'NEUTRAL',
            'confidence': 0.0,
            'geometry_state': GeometryState.FLAT.value,
            'entanglement_phase': EntanglementPhase.AREA_LAW.value,
            'anomaly_type': 'VACUUM',
            'anomaly_detected': False,
            'severity': 0.0,

            'S_A': 0.0,
            'S_renyi': 0.0,
            'mutual_information': 0.0,
            'ryu_takayanagi_area': 0.0,

            'ricci_scalar': 0.0,
            'geodesic_length': 0.0,
            'horizon_radius': 0.0,
            'is_black_hole': False,

            'entropic_force': 0.0,
            'force_direction': 'NEUTRAL',
            'gradient_entropy': 0.0,
            'temperature': 1.0,
            'inertia': 0.0,
            'net_force': 0.0,

            'n_mera_layers': 0,
            'long_range_entanglement': 0.0,
            'n_sites': 0,

            'entry_price': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,

            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }

    def reset(self):
        """Reseta o indicador"""
        self.singularity_detector.reset()


# ==============================================================================
# DEMONSTRAÇÃO
# ==============================================================================

def generate_black_hole_data(n_points: int = 100,
                            seed: int = 42,
                            with_attractor: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera dados que simulam formação de um "buraco negro" (atrator de preço)

    Args:
        n_points: Número de pontos
        seed: Semente
        with_attractor: Se True, simula formação de atrator
    """
    np.random.seed(seed)

    base_price = 1.0850
    prices = [base_price]
    volumes = [1000]

    if with_attractor:
        # Fase 1: Vácuo (baixa vol, sem direção) - 50%
        vacuum_phase = int(n_points * 0.5)

        for i in range(1, vacuum_phase):
            # Ruído térmico puro (range bound)
            noise = np.random.randn() * 0.00002
            vol = 800 + np.random.randn() * 100

            prices.append(prices[-1] + noise)
            volumes.append(max(100, vol))

        # Fase 2: Acúmulo de entropia (correlações se formando) - 30%
        # AQUI é onde o buraco negro está se formando
        accumulation_phase = int(n_points * 0.3)

        for i in range(accumulation_phase):
            # Correlações aparecem - preço ainda parado mas volume sobe
            noise = np.random.randn() * 0.00002

            # Volume aumentando exponencialmente (entropia subindo!)
            vol = 1000 * (1 + (i / accumulation_phase)**2 * 3) + np.random.randn() * 50

            # Preço quase não se move ainda
            trend = 0.000001 * i  # Tendência muito sutil

            prices.append(prices[-1] + noise + trend)
            volumes.append(max(100, vol))

        # Fase 3: Colapso gravitacional (buraco negro formado) - 20%
        collapse_phase = n_points - vacuum_phase - accumulation_phase

        for i in range(collapse_phase):
            # Movimento forte na direção do atrator
            trend = 0.0003 + 0.0001 * i  # Acelerando
            noise = np.random.randn() * 0.0001

            # Volume muito alto
            vol = 4000 + i * 100 + np.random.randn() * 200

            prices.append(prices[-1] + trend + noise)
            volumes.append(max(100, vol))
    else:
        # Apenas vácuo (ruído térmico)
        for i in range(1, n_points):
            noise = np.random.randn() * 0.00002
            vol = 800 + np.random.randn() * 100

            prices.append(prices[-1] + noise)
            volumes.append(max(100, vol))

    return np.array(prices), np.array(volumes)


def main():
    """Demonstração do indicador HBBP"""
    print("=" * 70)
    print("HOLOGRAPHIC AdS/CFT BULK-BOUNDARY PROJECTOR (HBBP)")
    print("Indicador baseado em Gravidade Entrópica e Correspondência Holográfica")
    print("=" * 70)
    print()

    # Inicializa indicador
    indicator = HolographicAdSCFTBulkBoundaryProjector(
        n_sites=64,
        bond_dimension=16,
        n_mera_layers=4,
        entropy_threshold=1.2,
        force_threshold=0.1,
        min_data_points=30
    )

    print("Indicador inicializado!")
    print(f"  - Sites na cadeia quantica: {indicator.n_sites}")
    print(f"  - Camadas MERA: 4")
    print(f"  - Formula de Ryu-Takayanagi: S_A = Area(gamma_A) / 4G_N")
    print()

    # Gera dados com atrator (buraco negro)
    print("Gerando dados com formacao de 'buraco negro' (atrator de preco)...")
    prices, volumes = generate_black_hole_data(n_points=150, seed=42, with_attractor=True)
    print(f"Dados gerados: {len(prices)} pontos")
    print(f"Range de preco: {prices.min():.5f} - {prices.max():.5f}")
    print()

    # Processa
    result = indicator.analyze(prices, volumes)

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Sinal: {result['signal_name']}")
    print(f"Confianca: {result['confidence']:.2%}")
    print()
    print("--- ENTROPIA DE EMARANHAMENTO ---")
    print(f"Entropia de von Neumann S_A: {result['S_A']:.4f}")
    print(f"Entropia de Renyi S2: {result['S_renyi']:.4f}")
    print(f"Informacao Mutua I(A:B): {result['mutual_information']:.4f}")
    print(f"Area Ryu-Takayanagi: {result['ryu_takayanagi_area']:.4f}")
    print()
    print("--- GEOMETRIA DO BULK (AdS) ---")
    print(f"Escalar de Ricci R: {result['ricci_scalar']:.4f}")
    print(f"Comprimento geodesico: {result['geodesic_length']:.4f}")
    print(f"Buraco negro: {result['is_black_hole']}")
    print(f"Raio do horizonte: {result['horizon_radius']:.4f}")
    print(f"Estado: {result['geometry_state']}")
    print()
    print("--- FORCA ENTROPICA ---")
    print(f"|F_e| = T nabla S: {result['entropic_force']:.4f}")
    print(f"Direcao: {result['force_direction']}")
    print(f"Gradiente nabla S: {result['gradient_entropy']:.6f}")
    print(f"Temperatura T: {result['temperature']:.4f}")
    print(f"Forca liquida: {result['net_force']:.4f}")
    print()
    print("--- MERA ---")
    print(f"Camadas: {result['n_mera_layers']}")
    print(f"Emaranhamento longo alcance: {result['long_range_entanglement']:.4f}")
    print(f"Fase: {result['entanglement_phase']}")
    print()
    print(f"Razao: {result['reason']}")

    return indicator, result


if __name__ == "__main__":
    indicator, result = main()
