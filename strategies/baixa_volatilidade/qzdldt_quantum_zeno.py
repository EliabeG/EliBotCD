"""
================================================================================
QUANTUM ZENO DISCORD & LINDBLAD DECOHERENCE TIMER (QZD-LDT)
Indicador de Forex baseado em Informação Quântica e Decoerência
================================================================================

Este indicador utiliza a Equação Mestra de Lindblad para medir a taxa de
interação entre o "Sistema" (Preço) e o "Ambiente" (Liquidez). Ele detecta
quando o "Olhar Zeno" dos algoritmos formadores de mercado falha, permitindo
que a função de onda do preço finalmente colapse e se mova (o rompimento).

A Física: Informação Quântica e Decoerência
Não analisaremos o preço em si, mas a Matriz Densidade (ρ) da informação do
mercado.

Por que isso é superior em Baixa Volatilidade?
1. Visão do Invisível: A análise técnica vê "Preço parado". A Mecânica Quântica
   vê "Função de Onda sob tensão máxima sendo segurada por observação constante".
2. Timing Perfeito: Rompimentos de baixa volatilidade costumam falhar (False
   Breakouts). O QZD-LDT filtra isso calculando a Fidelidade do estado. Se o
   preço se move, mas a Fidelidade do estado quântico não muda, é apenas
   flutuação quântica (ruído). Se a Fidelidade cai, houve uma transição real.
3. HFT Tracking: O modelo usa a própria atividade dos robôs de alta frequência
   (que causam o Zeno Effect) como sinal de entrada. Quando eles piscam, você atira.

Nota Técnica: Resolver a Equação Mestra de Lindblad em tempo real é computação
matricial complexa no espaço de Hilbert. Certifique-se de truncar o espaço de
estados (não use dimensão infinita, use um sistema de 2 ou 3 níveis/qubits)
para manter a latência abaixo de 5ms.

Autor: Gerado por Claude AI
Versão: 1.0.0
================================================================================
"""

import numpy as np
from scipy.linalg import expm, logm, sqrtm
from scipy.integrate import solve_ivp
from scipy.ndimage import uniform_filter1d
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import warnings
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

class SignalType(Enum):
    """Tipos de sinais de trading"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    WAIT = "WAIT"


class QuantumState(Enum):
    """Estado quântico do mercado"""
    COHERENT = "COHERENT"          # Superposição coerente (Zeno ativo)
    DECOHERING = "DECOHERING"      # Perdendo coerência
    COLLAPSED = "COLLAPSED"         # Estado colapsado (movimento iminente)
    MIXED = "MIXED"                 # Estado misto (ruído clássico)


class ZenoState(Enum):
    """Estado do Efeito Zeno"""
    ZENO_ACTIVE = "ZENO_ACTIVE"         # Preço travado por observação
    ZENO_WEAKENING = "ZENO_WEAKENING"   # Proteção enfraquecendo
    ZENO_BROKEN = "ZENO_BROKEN"         # Proteção quebrada


class BlochPosition(Enum):
    """Posição na Esfera de Bloch"""
    NORTH_POLE = "NORTH_POLE"   # |0⟩ - Dominância Ask (SHORT)
    SOUTH_POLE = "SOUTH_POLE"   # |1⟩ - Dominância Bid (LONG)
    EQUATOR = "EQUATOR"         # Superposição máxima
    MIXED_STATE = "MIXED_STATE" # Estado misto (dentro da esfera)


@dataclass
class DensityMatrix:
    """Matriz Densidade ρ do sistema"""
    rho: np.ndarray              # Matriz densidade 2x2 (ou NxN)
    purity: float                # Tr(ρ²) - pureza do estado
    von_neumann_entropy: float   # S = -Tr(ρ ln ρ)
    coherence: float             # Elementos fora da diagonal


@dataclass
class BlochVector:
    """Vetor de Bloch (representação geométrica do qubit)"""
    x: float                     # Componente x
    y: float                     # Componente y
    z: float                     # Componente z
    radius: float                # |r| - raio (1 = puro, <1 = misto)
    theta: float                 # Ângulo polar
    phi: float                   # Ângulo azimutal
    position: BlochPosition


@dataclass
class QuantumDiscord:
    """Quantum Discord - correlações quânticas não-clássicas"""
    discord: float               # Discórdia quântica
    classical_correlation: float # Correlação clássica
    mutual_information: float    # Informação mútua total


@dataclass
class ZenoMetrics:
    """Métricas do Efeito Zeno Quântico"""
    measurement_frequency: float # Frequência de "medição" (updates BBO)
    zeno_strength: float         # Força da proteção Zeno
    zeno_state: ZenoState
    time_since_last_measurement: float


@dataclass
class LindbladEvolution:
    """Evolução de Lindblad do sistema"""
    rho_evolved: np.ndarray      # ρ(t+dt)
    coherence_decay_rate: float  # Taxa de decaimento da coerência
    entropy_rate: float          # dS/dt
    fidelity: float              # F(ρ_0, ρ_t)


@dataclass
class QZDLDTSignal:
    """Sinal gerado pelo QZD-LDT"""
    signal_type: SignalType
    quantum_state: QuantumState
    zeno_state: ZenoState
    bloch_position: BlochPosition
    confidence: float

    # Matriz Densidade
    purity: float                # Tr(ρ²)
    von_neumann_entropy: float   # S = -Tr(ρ ln ρ)
    coherence: float             # |ρ_01|

    # Bloch
    bloch_z: float               # z > 0 → |0⟩ (SHORT), z < 0 → |1⟩ (LONG)
    bloch_radius: float          # Pureza geométrica

    # Zeno
    zeno_strength: float
    measurement_frequency: float

    # Lindblad
    coherence_decay_rate: float
    entropy_diverging: bool
    fidelity: float

    # Discord
    quantum_discord: float

    reason: str
    timestamp: str


# ==============================================================================
# MATRIZES DE PAULI (BASE DO QUBIT)
# ==============================================================================

# Matrizes de Pauli
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.array([[1, 0], [0, 1]], dtype=complex)

# Estados base
KET_0 = np.array([[1], [0]], dtype=complex)  # |0⟩ - Ask dominant
KET_1 = np.array([[0], [1]], dtype=complex)  # |1⟩ - Bid dominant


# ==============================================================================
# MATRIZ DENSIDADE
# ==============================================================================

class DensityMatrixCalculator:
    """
    1. A Matriz Densidade (ρ)

    Mapeie o fluxo de ordens (L2 Data) em um Qubit (Bit Quântico) na Esfera de
    Bloch.

    - Estado |0⟩: Dominância absoluta de Venda (Ask side heavy)
    - Estado |1⟩: Dominância absoluta de Compra (Bid side heavy)
    - Superposição: O mercado em equilíbrio de baixa vol é uma superposição
      coerente: |ψ⟩ = α|0⟩ + β|1⟩

    A Matriz Densidade é ρ = |ψ⟩⟨ψ|. Em um mercado misto (ruidoso), ρ descreve
    a pureza do estado.
    """

    def __init__(self, n_levels: int = 2):
        """
        Args:
            n_levels: Dimensão do espaço de Hilbert (2 para qubit)
        """
        self.n_levels = n_levels

    def construct_from_order_flow(self,
                                  bid_volume: float,
                                  ask_volume: float,
                                  bid_depth: float = 1.0,
                                  ask_depth: float = 1.0) -> DensityMatrix:
        """
        Constrói matriz densidade a partir do fluxo de ordens

        Quantum State Tomography simplificada
        """
        total = bid_volume + ask_volume + 1e-10

        # Amplitudes de probabilidade
        alpha = np.sqrt(ask_volume / total)  # |0⟩ component (ask)
        beta = np.sqrt(bid_volume / total)   # |1⟩ component (bid)

        # Fase baseada na profundidade relativa (spread/depth ratio)
        phase = np.pi * (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-10)

        # Estado puro: |ψ⟩ = α|0⟩ + βe^(iφ)|1⟩
        psi = alpha * KET_0 + beta * np.exp(1j * phase) * KET_1

        # Matriz densidade: ρ = |ψ⟩⟨ψ|
        rho = psi @ psi.conj().T

        # Adiciona ruído para simular estado misto (realidade do mercado)
        noise_level = 0.1  # 10% de ruído
        rho = (1 - noise_level) * rho + noise_level * IDENTITY / 2

        return self._analyze_density_matrix(rho)

    def construct_from_price_returns(self,
                                    returns: np.ndarray,
                                    window: int = 20) -> DensityMatrix:
        """
        Constrói matriz densidade a partir de retornos de preço

        Tomografia baseada em estatísticas dos retornos
        """
        recent = returns[-window:]

        # Estatísticas
        mean_return = np.mean(recent)
        std_return = np.std(recent) + 1e-10
        skew = np.mean(((recent - mean_return) / std_return)**3)

        # Mapeia para probabilidades do qubit
        # Retornos positivos → |1⟩ (bid), negativos → |0⟩ (ask)
        p_positive = np.mean(recent > 0)
        p_negative = 1 - p_positive

        # Amplitudes
        alpha = np.sqrt(p_negative + 1e-10)  # |0⟩
        beta = np.sqrt(p_positive + 1e-10)   # |1⟩

        # Fase baseada na assimetria (skew)
        phase = np.arctan(skew) / 2

        # Estado
        psi = alpha * KET_0 + beta * np.exp(1j * phase) * KET_1

        # Matriz densidade pura
        rho_pure = psi @ psi.conj().T

        # Mistura com estado maximamente misto baseado na volatilidade
        # Alta vol = mais misto, baixa vol = mais puro
        vol_norm = min(1.0, std_return / 0.01)  # Normaliza por vol típica
        mixing = vol_norm * 0.5  # Máximo 50% de mistura

        rho = (1 - mixing) * rho_pure + mixing * IDENTITY / 2

        return self._analyze_density_matrix(rho)

    def _analyze_density_matrix(self, rho: np.ndarray) -> DensityMatrix:
        """
        Analisa propriedades da matriz densidade
        """
        # Pureza: Tr(ρ²)
        rho2 = rho @ rho
        purity = np.real(np.trace(rho2))

        # Entropia de Von Neumann: S = -Tr(ρ ln ρ)
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove zeros
        von_neumann_entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))

        # Coerência: soma dos elementos fora da diagonal
        coherence = np.sum(np.abs(rho - np.diag(np.diag(rho))))

        return DensityMatrix(
            rho=rho,
            purity=purity,
            von_neumann_entropy=np.real(von_neumann_entropy),
            coherence=np.real(coherence)
        )


# ==============================================================================
# ESFERA DE BLOCH
# ==============================================================================

class BlochSphereAnalyzer:
    """
    Representação geométrica do qubit na Esfera de Bloch

    ρ = (1/2)(I + r·σ)

    Onde r = (x, y, z) é o vetor de Bloch e σ = (σ_x, σ_y, σ_z) são as matrizes
    de Pauli.

    - Polo Norte (z = +1): Estado |0⟩ → Ask dominant → SHORT
    - Polo Sul (z = -1): Estado |1⟩ → Bid dominant → LONG
    - Equador: Superposição máxima
    - Dentro da esfera (|r| < 1): Estado misto
    """

    def __init__(self):
        pass

    def density_to_bloch(self, rho: np.ndarray) -> BlochVector:
        """
        Converte matriz densidade 2x2 para vetor de Bloch

        Para qubit: ρ = (1/2)(I + x·σ_x + y·σ_y + z·σ_z)
        Então: x = Tr(ρ·σ_x), y = Tr(ρ·σ_y), z = Tr(ρ·σ_z)
        """
        x = np.real(np.trace(rho @ SIGMA_X))
        y = np.real(np.trace(rho @ SIGMA_Y))
        z = np.real(np.trace(rho @ SIGMA_Z))

        # Raio (pureza geométrica)
        radius = np.sqrt(x**2 + y**2 + z**2)

        # Ângulos esféricos
        theta = np.arccos(z / (radius + 1e-10))  # Ângulo polar
        phi = np.arctan2(y, x)                    # Ângulo azimutal

        # Determina posição
        if radius < 0.3:
            position = BlochPosition.MIXED_STATE
        elif z > 0.7:
            position = BlochPosition.NORTH_POLE
        elif z < -0.7:
            position = BlochPosition.SOUTH_POLE
        else:
            position = BlochPosition.EQUATOR

        return BlochVector(
            x=x, y=y, z=z,
            radius=radius,
            theta=theta,
            phi=phi,
            position=position
        )


# ==============================================================================
# QUANTUM DISCORD
# ==============================================================================

class QuantumDiscordCalculator:
    """
    2. O Efeito Zeno e a Discórdia Quântica

    O Quantum Discord mede correlações quânticas que não são emaranhamento
    (entanglement). Em baixa volatilidade, a correlação clássica (Pearson) é
    zero (o preço não vai a lugar nenhum). Mas a Discórdia Quântica está fervendo.

    - Os HFTs estão bombardeando o book com ordens de cancelamento (spoofing/pinging).
      Isso equivale a "medir" o sistema constantemente.
    - Enquanto a frequência de medição for alta, o Efeito Zeno trava o preço.
    """

    def __init__(self):
        pass

    def compute_discord(self,
                       rho: np.ndarray,
                       rho_classical: np.ndarray = None) -> QuantumDiscord:
        """
        Calcula Quantum Discord (simplificado para qubit único)

        Discord = I(A:B) - J(A:B)

        Onde I é informação mútua total e J é correlação clássica máxima
        """
        # Para um sistema de qubit único, usamos uma aproximação
        # baseada na coerência como proxy para discord

        # Informação mútua (aproximada pela entropia)
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))

        # Correlação clássica (elementos diagonais)
        diagonal = np.real(np.diag(rho))
        diagonal = diagonal[diagonal > 1e-10]
        classical_entropy = -np.sum(diagonal * np.log2(diagonal + 1e-10))

        # Discord aproximado pela diferença de entropias
        # (coerência contribui para discord)
        coherence = np.sum(np.abs(rho - np.diag(np.diag(rho))))

        discord = np.real(coherence * entropy)
        classical_correlation = classical_entropy
        mutual_information = entropy

        return QuantumDiscord(
            discord=discord,
            classical_correlation=classical_correlation,
            mutual_information=mutual_information
        )


# ==============================================================================
# EFEITO ZENO QUÂNTICO
# ==============================================================================

class ZenoEffectMonitor:
    """
    Passo 2: Monitoramento da Taxa Zeno

    Calcule a frequência de atualizações do Best Bid/Offer sem mudança de preço.
    Isso é a "frequência de medição".

    - Enquanto essa frequência for alta, o Efeito Zeno está ativo. O preço está
      "preso no olhar" do mercado.
    """

    def __init__(self,
                 zeno_threshold: float = 0.8,
                 measurement_window: int = 50):
        """
        Args:
            zeno_threshold: Limiar para considerar Zeno ativo
            measurement_window: Janela para calcular frequência
        """
        self.zeno_threshold = zeno_threshold
        self.measurement_window = measurement_window

        # Histórico
        self.measurement_times: List[float] = []
        self.price_changes: List[bool] = []
        self.zeno_history: List[float] = []

    def update(self,
              timestamp: float,
              price_changed: bool,
              bbo_updated: bool = True):
        """
        Atualiza com nova observação
        """
        if bbo_updated:
            self.measurement_times.append(timestamp)
            self.price_changes.append(price_changed)

        # Mantém janela
        if len(self.measurement_times) > self.measurement_window:
            self.measurement_times.pop(0)
            self.price_changes.pop(0)

    def compute_zeno_metrics(self) -> ZenoMetrics:
        """
        Calcula métricas do Efeito Zeno
        """
        if len(self.measurement_times) < 5:
            return ZenoMetrics(
                measurement_frequency=0.0,
                zeno_strength=0.0,
                zeno_state=ZenoState.ZENO_BROKEN,
                time_since_last_measurement=float('inf')
            )

        # Frequência de medição (updates BBO por unidade de tempo)
        time_span = self.measurement_times[-1] - self.measurement_times[0] + 1e-10
        measurement_frequency = len(self.measurement_times) / time_span

        # Força do Zeno: proporção de medições sem mudança de preço
        n_no_change = sum(1 for changed in self.price_changes if not changed)
        zeno_strength = n_no_change / len(self.price_changes)

        # Tempo desde última medição
        time_since_last = 1.0 / (measurement_frequency + 1e-10)

        # Estado
        if zeno_strength > self.zeno_threshold and measurement_frequency > 1.0:
            zeno_state = ZenoState.ZENO_ACTIVE
        elif zeno_strength > 0.5:
            zeno_state = ZenoState.ZENO_WEAKENING
        else:
            zeno_state = ZenoState.ZENO_BROKEN

        # Salva histórico
        self.zeno_history.append(zeno_strength)
        if len(self.zeno_history) > 100:
            self.zeno_history.pop(0)

        return ZenoMetrics(
            measurement_frequency=measurement_frequency,
            zeno_strength=zeno_strength,
            zeno_state=zeno_state,
            time_since_last_measurement=time_since_last
        )

    def is_zeno_breaking(self) -> bool:
        """
        Detecta se a proteção Zeno está quebrando
        """
        if len(self.zeno_history) < 10:
            return False

        recent = np.mean(self.zeno_history[-5:])
        older = np.mean(self.zeno_history[-10:-5])

        return recent < older - 0.1  # Queda de 10%

    def reset(self):
        """Reseta estado do monitor"""
        self.measurement_times.clear()
        self.price_changes.clear()
        self.zeno_history.clear()


# ==============================================================================
# EQUAÇÃO DE LINDBLAD
# ==============================================================================

class LindbladSolver:
    """
    3. A Dinâmica Dissipativa (Equação de Lindblad)

    Para prever o rompimento, precisamos saber quando esse "congelamento" vai
    derreter. Modelamos a evolução temporal de ρ considerando a dissipação
    (perda de informação para o ambiente):

    dρ/dt = -(i/ℏ)[H, ρ] + Σ γ_k (L_k ρ L_k† - (1/2){L_k† L_k, ρ})

    - H: O Hamiltoniano do sistema (a "energia" da tendência latente)
    - [H, ρ]: A evolução unitária (o que o preço quer fazer)
    - L_k: Operadores de Lindblad (Salto). Eles representam o "ruído de medição"
           dos HFTs que causa o colapso
    - γ_k: A taxa de decoerência

    Passo 3: Solução Numérica de Lindblad
    Use o solver mesolve do QuTiP para projetar a evolução de ρ para os próximos
    segundos, dado o Hamiltoniano atual (pressão de compra/venda latente).
    """

    def __init__(self,
                 hbar: float = 1.0,
                 dt: float = 0.01,
                 n_steps: int = 100):
        """
        Args:
            hbar: Constante de Planck reduzida (escala)
            dt: Passo de tempo
            n_steps: Número de passos para evolução
        """
        self.hbar = hbar
        self.dt = dt
        self.n_steps = n_steps

    def build_hamiltonian(self,
                         trend_pressure: float,
                         volatility: float) -> np.ndarray:
        """
        Constrói o Hamiltoniano do sistema

        H = ε σ_z + Ω σ_x

        - ε: Energia do gap (tendência latente)
        - Ω: Acoplamento (volatilidade)
        """
        epsilon = trend_pressure  # Pressão de tendência
        omega = volatility        # Volatilidade como acoplamento

        H = epsilon * SIGMA_Z + omega * SIGMA_X

        return H

    def build_lindblad_operators(self,
                                 decoherence_rate: float,
                                 relaxation_rate: float) -> List[Tuple[np.ndarray, float]]:
        """
        Constrói operadores de Lindblad (jump operators)

        L_1: Operador de defasagem (dephasing) - perda de coerência
        L_2: Operador de relaxação - decaimento para estado base
        """
        operators = []

        # Dephasing: L_1 = √γ_d σ_z
        if decoherence_rate > 0:
            L_dephasing = np.sqrt(decoherence_rate) * SIGMA_Z
            operators.append((L_dephasing, decoherence_rate))

        # Relaxação: L_2 = √γ_r σ_- (lowering operator)
        if relaxation_rate > 0:
            sigma_minus = (SIGMA_X - 1j * SIGMA_Y) / 2
            L_relaxation = np.sqrt(relaxation_rate) * sigma_minus
            operators.append((L_relaxation, relaxation_rate))

        return operators

    def lindblad_rhs(self,
                    t: float,
                    rho_flat: np.ndarray,
                    H: np.ndarray,
                    lindblad_ops: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        Lado direito da equação de Lindblad (para solver ODE)
        """
        n = int(np.sqrt(len(rho_flat)))
        rho = rho_flat.reshape((n, n))

        # Termo unitário: -i[H, ρ]/ℏ
        commutator = H @ rho - rho @ H
        drho = -1j * commutator / self.hbar

        # Termos dissipativos: Σ γ_k (L_k ρ L_k† - (1/2){L_k† L_k, ρ})
        for L, gamma in lindblad_ops:
            L_dag = L.conj().T
            L_dag_L = L_dag @ L

            # L ρ L†
            term1 = L @ rho @ L_dag

            # (1/2){L†L, ρ} = (1/2)(L†L ρ + ρ L†L)
            term2 = 0.5 * (L_dag_L @ rho + rho @ L_dag_L)

            drho += gamma * (term1 - term2)

        return drho.flatten()

    def evolve(self,
              rho_initial: np.ndarray,
              H: np.ndarray,
              lindblad_ops: List[Tuple[np.ndarray, float]],
              t_final: float = None) -> LindbladEvolution:
        """
        Evolui o sistema pela equação de Lindblad
        """
        if t_final is None:
            t_final = self.dt * self.n_steps

        # Estado inicial
        rho_0 = rho_initial.flatten()

        # Resolve numericamente
        t_span = (0, t_final)
        t_eval = np.linspace(0, t_final, self.n_steps)

        # Separar partes real e imaginária para solver real
        def rhs_real(t, y):
            y_complex = y[:len(y)//2] + 1j * y[len(y)//2:]
            dy_complex = self.lindblad_rhs(t, y_complex, H, lindblad_ops)
            return np.concatenate([np.real(dy_complex), np.imag(dy_complex)])

        y0 = np.concatenate([np.real(rho_0), np.imag(rho_0)])

        try:
            sol = solve_ivp(rhs_real, t_span, y0, t_eval=t_eval, method='RK45')

            # Reconstrói ρ final
            y_final = sol.y[:, -1]
            n = int(np.sqrt(len(rho_0)))
            rho_final = (y_final[:len(y_final)//2] + 1j * y_final[len(y_final)//2:]).reshape((n, n))

        except Exception as e:
            logger.warning(f"Erro no solver de Lindblad: {e}")
            rho_final = rho_initial

        # Garante traço 1 e hermitianidade
        rho_final = (rho_final + rho_final.conj().T) / 2
        rho_final = rho_final / np.trace(rho_final)

        # Métricas
        coherence_initial = np.abs(rho_initial[0, 1])
        coherence_final = np.abs(rho_final[0, 1])
        coherence_decay_rate = (coherence_initial - coherence_final) / (t_final + 1e-10)

        # Entropia
        eigenvalues_init = np.linalg.eigvalsh(rho_initial)
        eigenvalues_final = np.linalg.eigvalsh(rho_final)

        S_init = -np.sum(eigenvalues_init[eigenvalues_init > 1e-10] *
                        np.log(eigenvalues_init[eigenvalues_init > 1e-10] + 1e-10))
        S_final = -np.sum(eigenvalues_final[eigenvalues_final > 1e-10] *
                         np.log(eigenvalues_final[eigenvalues_final > 1e-10] + 1e-10))
        entropy_rate = (S_final - S_init) / (t_final + 1e-10)

        # Fidelidade: F(ρ, σ) = (Tr√(√ρ σ √ρ))²
        fidelity = self.compute_fidelity(rho_initial, rho_final)

        return LindbladEvolution(
            rho_evolved=rho_final,
            coherence_decay_rate=coherence_decay_rate,
            entropy_rate=np.real(entropy_rate),
            fidelity=fidelity
        )

    def compute_fidelity(self, rho: np.ndarray, sigma: np.ndarray) -> float:
        """
        Calcula fidelidade entre dois estados

        F(ρ, σ) = (Tr√(√ρ σ √ρ))²
        """
        try:
            sqrt_rho = sqrtm(rho)
            inner = sqrt_rho @ sigma @ sqrt_rho
            sqrt_inner = sqrtm(inner)
            fidelity = np.real(np.trace(sqrt_inner))**2
            return np.clip(fidelity, 0, 1)
        except:
            return 1.0


# ==============================================================================
# INDICADOR QZD-LDT COMPLETO
# ==============================================================================

class QuantumZenoDiscordLindbladTimer:
    """
    Quantum Zeno Discord & Lindblad Decoherence Timer (QZD-LDT)

    Indicador completo que usa informação quântica para detectar quando o
    "congelamento" do preço vai derreter.

    O GATILHO (The Decoherence Event):
    Você está procurando o momento em que a Coerência (os elementos fora da
    diagonal da matriz ρ) começa a vazar para o ambiente.

    SINAL: O indicador mostra que a "Proteção Zeno" caiu (a frequência de
    cancelamentos diminuiu ou tornou-se assíncrona) E a solução de Lindblad
    mostra uma divergência na Entropia de Von Neumann (S = -Tr(ρ ln ρ)).
    Isso significa que a "barreira quântica" que segurava o preço desapareceu.

    DIREÇÃO: Olhe para o vetor de Bloch resultante da evolução de Lindblad.
    Se ele aponta para o Polo Norte (|0⟩), é SHORT.
    Se aponta para o Polo Sul (|1⟩), é LONG.

    AÇÃO: O trade é executado no vácuo de atividade, milissegundos antes da
    explosão de volatilidade.
    """

    def __init__(self,
                 # Parâmetros de decoerência
                 decoherence_rate: float = 0.1,
                 relaxation_rate: float = 0.05,

                 # Parâmetros de Zeno
                 zeno_threshold: float = 0.7,
                 measurement_window: int = 50,

                 # Parâmetros de Lindblad
                 evolution_steps: int = 50,

                 # Geral
                 min_data_points: int = 30):
        """
        Inicializa o QZD-LDT
        """
        self.decoherence_rate = decoherence_rate
        self.relaxation_rate = relaxation_rate
        self.min_data_points = min_data_points

        # Componentes
        self.density_calculator = DensityMatrixCalculator()
        self.bloch_analyzer = BlochSphereAnalyzer()
        self.discord_calculator = QuantumDiscordCalculator()
        self.zeno_monitor = ZenoEffectMonitor(
            zeno_threshold=zeno_threshold,
            measurement_window=measurement_window
        )
        self.lindblad_solver = LindbladSolver(
            n_steps=evolution_steps
        )

        # Histórico
        self.entropy_history: List[float] = []
        self.coherence_history: List[float] = []
        self.fidelity_history: List[float] = []

    def analyze(self,
                prices: np.ndarray,
                volumes: np.ndarray = None,
                bid_volumes: np.ndarray = None,
                ask_volumes: np.ndarray = None) -> dict:
        """
        Analisa dados de mercado e retorna resultado

        Returns:
            dict com signal, confidence, e métricas quânticas
        """
        n = len(prices)

        # Validação
        if n < self.min_data_points:
            return self._empty_result("INSUFFICIENT_DATA")

        # Prepara dados
        returns = np.diff(np.log(prices + 1e-10))

        if volumes is None:
            volumes = np.abs(np.diff(prices)) * 10000 + 1000
            volumes = np.append(volumes, volumes[-1])

        if bid_volumes is None or ask_volumes is None:
            returns_extended = np.append(returns, returns[-1])
            bid_volumes = volumes * (0.5 + 0.5 * np.tanh(returns_extended * 100))
            ask_volumes = volumes - bid_volumes + 1

        # PASSO 1: TOMOGRAFIA DE ESTADO QUÂNTICO
        density = self.density_calculator.construct_from_price_returns(returns)
        density_flow = self.density_calculator.construct_from_order_flow(
            bid_volume=np.mean(bid_volumes[-10:]),
            ask_volume=np.mean(ask_volumes[-10:])
        )
        rho = 0.5 * density.rho + 0.5 * density_flow.rho
        density = self.density_calculator._analyze_density_matrix(rho)

        # PASSO 2: ANÁLISE DA ESFERA DE BLOCH
        bloch = self.bloch_analyzer.density_to_bloch(density.rho)

        # PASSO 3: QUANTUM DISCORD
        discord = self.discord_calculator.compute_discord(density.rho)

        # PASSO 4: MONITORAMENTO DO EFEITO ZENO
        for i in range(min(20, n-1)):
            price_changed = abs(prices[-i-1] - prices[-i-2]) > 1e-6
            self.zeno_monitor.update(timestamp=float(i), price_changed=price_changed)

        zeno = self.zeno_monitor.compute_zeno_metrics()
        zeno_breaking = self.zeno_monitor.is_zeno_breaking()

        # PASSO 5: EVOLUÇÃO DE LINDBLAD
        trend_pressure = np.mean(returns[-10:]) * 1000
        volatility = np.std(returns[-20:]) * 100
        H = self.lindblad_solver.build_hamiltonian(trend_pressure, volatility)
        lindblad_ops = self.lindblad_solver.build_lindblad_operators(
            decoherence_rate=self.decoherence_rate * (1 - zeno.zeno_strength),
            relaxation_rate=self.relaxation_rate
        )
        evolution = self.lindblad_solver.evolve(density.rho, H, lindblad_ops, t_final=1.0)
        bloch_evolved = self.bloch_analyzer.density_to_bloch(evolution.rho_evolved)

        # PASSO 6: DETECÇÃO DE DECOERÊNCIA
        self.entropy_history.append(density.von_neumann_entropy)
        self.coherence_history.append(density.coherence)
        self.fidelity_history.append(evolution.fidelity)

        for hist in [self.entropy_history, self.coherence_history, self.fidelity_history]:
            if len(hist) > 50:
                hist.pop(0)

        entropy_diverging = False
        if len(self.entropy_history) >= 5:
            recent_entropy = np.mean(self.entropy_history[-3:])
            older_entropy = np.mean(self.entropy_history[-6:-3])
            entropy_diverging = recent_entropy > older_entropy + 0.05

        fidelity_dropping = False
        if len(self.fidelity_history) >= 5:
            recent_fid = np.mean(self.fidelity_history[-3:])
            older_fid = np.mean(self.fidelity_history[-6:-3])
            fidelity_dropping = recent_fid < older_fid - 0.05

        # PASSO 7: CLASSIFICAÇÃO DO ESTADO QUÂNTICO
        if density.purity > 0.95:
            quantum_state = QuantumState.COHERENT
        elif density.purity < 0.6:
            quantum_state = QuantumState.MIXED
        elif entropy_diverging or fidelity_dropping:
            quantum_state = QuantumState.DECOHERING
        elif evolution.coherence_decay_rate > 0.1:
            quantum_state = QuantumState.COLLAPSED
        else:
            quantum_state = QuantumState.COHERENT

        # PASSO 8: GERAÇÃO DE SINAL
        signal = 0
        signal_name = "NEUTRAL"
        confidence = 0.0
        reasons = []

        # CONDIÇÃO 1: Estado coerente + Zeno ativo → WAIT
        if quantum_state == QuantumState.COHERENT and zeno.zeno_state == ZenoState.ZENO_ACTIVE:
            signal_name = "WAIT"
            reasons.append(f"Zeno ATIVO: Pureza={density.purity:.2f}")
            reasons.append(f"Forca Zeno={zeno.zeno_strength:.1%}")

        # CONDIÇÃO 2: Estado misto → ruído clássico
        elif quantum_state == QuantumState.MIXED:
            signal_name = "NEUTRAL"
            reasons.append(f"Estado MISTO: Pureza={density.purity:.2f}")

        # CONDIÇÃO 3: Decoerência + Zeno quebrando → SINAL!
        elif (quantum_state in [QuantumState.DECOHERING, QuantumState.COLLAPSED] and
              (zeno_breaking or zeno.zeno_state != ZenoState.ZENO_ACTIVE)):

            if bloch_evolved.z > 0.1:
                signal = -1  # SHORT
                signal_name = "SHORT"
                confidence = min(1.0, abs(bloch_evolved.z) + (1 - evolution.fidelity))
                reasons.append(f"Bloch->Norte (z={bloch_evolved.z:.2f})")
                reasons.append(f"Colapso para |0> (ASK)")
            elif bloch_evolved.z < -0.1:
                signal = 1  # LONG
                signal_name = "LONG"
                confidence = min(1.0, abs(bloch_evolved.z) + (1 - evolution.fidelity))
                reasons.append(f"Bloch->Sul (z={bloch_evolved.z:.2f})")
                reasons.append(f"Colapso para |1> (BID)")
            else:
                signal_name = "NEUTRAL"
                reasons.append(f"Direcao incerta z={bloch_evolved.z:.2f}")

        # CONDIÇÃO 4: Discord alto + Zeno enfraquecendo
        elif discord.discord > 0.5 and zeno.zeno_state == ZenoState.ZENO_WEAKENING:
            signal_name = "WAIT"
            confidence = 0.5
            reasons.append(f"Discord ALTO ({discord.discord:.2f})")
            reasons.append("Zeno enfraquecendo")

        else:
            reasons.append(f"Estado={quantum_state.value}")
            reasons.append(f"Zeno={zeno.zeno_state.value}")

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'quantum_state': quantum_state.value,
            'zeno_state': zeno.zeno_state.value,
            'bloch_position': bloch_evolved.position.value,
            'purity': density.purity,
            'von_neumann_entropy': density.von_neumann_entropy,
            'coherence': density.coherence,
            'bloch_x': bloch_evolved.x,
            'bloch_y': bloch_evolved.y,
            'bloch_z': bloch_evolved.z,
            'bloch_radius': bloch_evolved.radius,
            'zeno_strength': zeno.zeno_strength,
            'measurement_frequency': zeno.measurement_frequency,
            'zeno_breaking': zeno_breaking,
            'coherence_decay_rate': evolution.coherence_decay_rate,
            'entropy_rate': evolution.entropy_rate,
            'entropy_diverging': entropy_diverging,
            'fidelity': evolution.fidelity,
            'fidelity_dropping': fidelity_dropping,
            'quantum_discord': discord.discord,
            'classical_correlation': discord.classical_correlation,
            'reasons': reasons
        }

    def _empty_result(self, signal_name: str) -> dict:
        """Retorna resultado vazio"""
        return {
            'signal': 0,
            'signal_name': signal_name,
            'confidence': 0.0,
            'quantum_state': QuantumState.MIXED.value,
            'zeno_state': ZenoState.ZENO_BROKEN.value,
            'bloch_position': BlochPosition.MIXED_STATE.value,
            'purity': 0.0,
            'von_neumann_entropy': 0.0,
            'coherence': 0.0,
            'bloch_x': 0.0,
            'bloch_y': 0.0,
            'bloch_z': 0.0,
            'bloch_radius': 0.0,
            'zeno_strength': 0.0,
            'measurement_frequency': 0.0,
            'zeno_breaking': False,
            'coherence_decay_rate': 0.0,
            'entropy_rate': 0.0,
            'entropy_diverging': False,
            'fidelity': 1.0,
            'fidelity_dropping': False,
            'quantum_discord': 0.0,
            'classical_correlation': 0.0,
            'reasons': [signal_name]
        }

    def reset(self):
        """Reseta estado do indicador"""
        self.zeno_monitor.reset()
        self.entropy_history.clear()
        self.coherence_history.clear()
        self.fidelity_history.clear()

    def get_entropy_history(self) -> np.ndarray:
        """Retorna historico de entropia"""
        return np.array(self.entropy_history)

    def get_coherence_history(self) -> np.ndarray:
        """Retorna historico de coerencia"""
        return np.array(self.coherence_history)

    def get_fidelity_history(self) -> np.ndarray:
        """Retorna historico de fidelidade"""
        return np.array(self.fidelity_history)


# ==============================================================================
# DEMONSTRAÇÃO
# ==============================================================================

def generate_zeno_data(n_points: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Gera dados que simulam efeito Zeno (preço travado)"""
    np.random.seed(seed)

    # Preço quase estático (Zeno ativo)
    base = 1.0850

    # Ruído mínimo (alta frequência de "medição" trava o preço)
    noise = np.random.randn(n_points) * 0.00005

    prices = base + noise

    # Volume alto (muita atividade de HFT)
    volumes = 5000 + np.random.randn(n_points) * 500

    return prices, volumes


def main():
    """Demonstração do indicador QZD-LDT"""
    print("=" * 70)
    print("QUANTUM ZENO DISCORD & LINDBLAD DECOHERENCE TIMER (QZD-LDT)")
    print("Indicador baseado em Informação Quântica e Decoerência")
    print("=" * 70)
    print()

    # Inicializa indicador
    indicator = QuantumZenoDiscordLindbladTimer(
        decoherence_rate=0.1,
        relaxation_rate=0.05,
        zeno_threshold=0.7,
        measurement_window=30,
        min_data_points=30
    )

    print("Indicador inicializado!")
    print(f"  - Taxa Decoerência: 0.1")
    print(f"  - Taxa Relaxação: 0.05")
    print(f"  - Threshold Zeno: 0.7")
    print()

    # Gera dados
    prices, volumes = generate_zeno_data(n_points=60)
    print(f"Dados gerados: {len(prices)} pontos")
    print()

    # Processa
    result = indicator.analyze(prices, volumes)

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Sinal: {result['signal_name']}")
    print(f"Estado Quântico: {result['quantum_state']}")
    print(f"Estado Zeno: {result['zeno_state']}")
    print(f"Posição Bloch: {result['bloch_position']}")
    print(f"Confiança: {result['confidence']:.2%}")
    print(f"\nMatriz Densidade:")
    print(f"  Pureza: {result['purity']:.4f}")
    print(f"  Entropia VN: {result['von_neumann_entropy']:.4f}")
    print(f"  Coerência: {result['coherence']:.4f}")
    print(f"\nEsfera de Bloch:")
    print(f"  z: {result['bloch_z']:.4f}")
    print(f"  Raio: {result['bloch_radius']:.4f}")
    print(f"\nEfeito Zeno:")
    print(f"  Força: {result['zeno_strength']:.2%}")
    print(f"  Freq. Medição: {result['measurement_frequency']:.2f}")
    print(f"\nLindblad:")
    print(f"  Taxa Decay Coerência: {result['coherence_decay_rate']:.4f}")
    print(f"  Entropia Divergindo: {result['entropy_diverging']}")
    print(f"  Fidelidade: {result['fidelity']:.4f}")
    print(f"\nDiscord: {result['quantum_discord']:.4f}")
    print(f"\nRazões: {result['reasons']}")

    return indicator, result


if __name__ == "__main__":
    indicator, result = main()
