"""
================================================================================
PARISI REPLICA SYMMETRY BREAKING DETECTOR (PRSBD)
Indicador de Forex baseado na Fisica de Vidros de Spin (Nobel Giorgio Parisi)
================================================================================

Baseado no trabalho do Nobel Giorgio Parisi, este indicador detecta a quebra da
simetria de replicas. Ele preve quando o sistema vai "resolver" a frustracao e
colapsar para um novo estado de energia minima (o inicio de uma nova tendencia
violenta) ANTES que qualquer volume apareca.

A Fisica: A Quebra de Simetria de Replicas (RSB)
Nao vamos modelar o preco. Vamos modelar a Paisagem de Energia Livre (Free Energy
Landscape) do mercado.

Em baixa volatilidade, essa paisagem e rugosa, cheia de vales e picos locais
(multiplos estados de equilibrio). O preco fica preso em um desses vales locais.
A Analise Tecnica chama isso de "Consolidacao". A Fisica chama de "Estado Vitreo".

O indicador calcula o Parametro de Ordem de Edwards-Anderson (q_EA) e a funcao
de distribuicao de probabilidade da sobreposicao de replicas P(q).

Por que usar Fisica de Vidros de Spin?
1. Deteccao de Falsos Rompimentos: Se o preco rompe a caixa de consolidacao,
   mas a funcao P(q) ainda mostra estrutura ultrametrica (RSB), significa que
   o sistema ainda esta frustrado internamente. O rompimento e falso (ruido).
2. Sensibilidade Extrema: A Susceptibilidade de Spin Glass (chi_SG) e conhecida
   por detectar transicoes de fase muito antes da Susceptibilidade Ferromagnetica.
3. Independencia de Volume: Funciona em feriados ou horarios de baixa liquidez.

Autor: Gerado por Claude AI
Versao: 1.0.0
================================================================================
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import entropy
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

class PhaseState(Enum):
    """Estado de fase do sistema"""
    PARAMAGNETIC = "PARAMAGNETIC"      # Ruido puro, P(q) = delta(0)
    SPIN_GLASS = "SPIN_GLASS"          # RSB, P(q) tem suporte amplo
    FERROMAGNETIC = "FERROMAGNETIC"    # Simetria quebrada, tendencia
    CRITICAL = "CRITICAL"              # No ponto critico de transicao


class SymmetryState(Enum):
    """Estado da simetria de replicas"""
    RS = "REPLICA_SYMMETRIC"           # Todas replicas iguais
    RSB = "REPLICA_SYMMETRY_BREAKING"  # Estrutura hierarquica
    BROKEN = "SYMMETRY_BROKEN"         # Simetria totalmente quebrada


@dataclass
class SpinConfiguration:
    """Configuracao de spins de uma replica"""
    spins: np.ndarray           # S_i = +/-1
    energy: float               # H(S)
    magnetization: float        # M = (1/N) Sum S_i
    replica_id: int


@dataclass
class InteractionMatrix:
    """Matriz de interacao J_ij"""
    J: np.ndarray               # Matriz de acoplamento
    h: np.ndarray               # Campo externo
    frustration: float          # Nivel de frustracao


@dataclass
class OverlapMatrix:
    """Matriz de sobreposicao Q_ab entre replicas"""
    Q: np.ndarray               # Q_ab = (1/N) Sum S_i^a S_i^b
    q_EA: float                 # Parametro de Edwards-Anderson
    q_distribution: np.ndarray  # P(q) - distribuicao dos overlaps


@dataclass
class UltrametricStructure:
    """Estrutura ultrametrica das replicas"""
    distance_matrix: np.ndarray     # Distancia entre replicas
    is_ultrametric: bool            # True se satisfaz d(a,c) <= max(d(a,b), d(b,c))
    hierarchy_depth: int            # Profundidade da arvore
    clustering: np.ndarray          # Clusters hierarquicos


@dataclass
class SusceptibilityMetrics:
    """Metricas de susceptibilidade"""
    chi_SG: float               # Susceptibilidade de Spin Glass
    chi_FM: float               # Susceptibilidade Ferromagnetica
    divergence_ratio: float     # chi_SG / chi_FM


# ==============================================================================
# HAMILTONIANO DO VIDRO DE SPIN
# ==============================================================================

class SpinGlassHamiltonian:
    """
    1. O Hamiltoniano do Vidro de Spin (H)

    Defina o mercado como uma rede de spins S_i = +/-1 (Compra/Venda) interagindo
    atraves de uma matriz de acoplamento aleatoria J_ij.

    H = -Sum J_ij S_i S_j - Sum h_i S_i

    Onde:
    - J_ij: Correlacao cruzada entre diferentes escalas de tempo (ex: M1 vs M5 vs H1).
            Em baixa vol, J_ij e uma mistura caotica de sinais positivos e negativos
            (Frustracao).
    - h_i: Campo externo (News/Macroeconomia - que em baixa vol e desprezivel).
    """

    def __init__(self,
                 n_spins: int = 50,
                 temperature: float = 1.0,
                 external_field_strength: float = 0.0):
        """
        Args:
            n_spins: Numero de spins (ativos x timeframes)
            temperature: T - temperatura do sistema
            external_field_strength: Forca do campo externo
        """
        self.n_spins = n_spins
        self.temperature = temperature
        self.external_field_strength = external_field_strength

    def build_interaction_matrix(self,
                                returns_matrix: np.ndarray) -> InteractionMatrix:
        """
        Passo 1: Construcao da Matriz de Interacao (J_ij)

        Nao use precos. Use os retornos de diferentes ativos correlacionados
        (EURUSD, GBPUSD, USDCHF, Gold) e diferentes timeframes como os "nos" da rede.

        Args:
            returns_matrix: Matriz (n_assets x n_timepoints) de retornos
        """
        n = returns_matrix.shape[0]

        # Matriz de correlacao como J_ij
        # Adiciona ruido para simular "quenched disorder"
        correlation = np.corrcoef(returns_matrix)

        # Substitui NaN por 0
        correlation = np.nan_to_num(correlation, nan=0.0)

        # J_ij = correlacao com ruido
        noise = np.random.randn(n, n) * 0.1
        noise = (noise + noise.T) / 2  # Simetriza
        np.fill_diagonal(noise, 0)

        J = correlation + noise
        np.fill_diagonal(J, 0)

        # Campo externo (fraco em baixa vol)
        h = np.random.randn(n) * self.external_field_strength

        # Frustracao = numero de plaquetas frustradas
        # Uma plaqueta e frustrada se J_ij * J_jk * J_ki < 0
        frustration = self._compute_frustration(J)

        return InteractionMatrix(J=J, h=h, frustration=frustration)

    def _compute_frustration(self, J: np.ndarray) -> float:
        """
        Computa o nivel de frustracao do sistema

        Frustracao ocorre quando nao e possivel minimizar todas as interacoes
        """
        n = J.shape[0]
        n_frustrated = 0
        n_total = 0

        # Amostra plaquetas triangulares
        for _ in range(min(1000, n**2)):
            i, j, k = np.random.choice(n, 3, replace=False)
            product = J[i, j] * J[j, k] * J[k, i]
            if product < 0:
                n_frustrated += 1
            n_total += 1

        return n_frustrated / (n_total + 1e-10)

    def compute_energy(self,
                      spins: np.ndarray,
                      interaction: InteractionMatrix) -> float:
        """
        Calcula a energia H = -Sum J_ij S_i S_j - Sum h_i S_i
        """
        # Termo de interacao
        interaction_term = -0.5 * np.sum(interaction.J * np.outer(spins, spins))

        # Termo de campo externo
        field_term = -np.sum(interaction.h * spins)

        return interaction_term + field_term

    def compute_magnetization(self, spins: np.ndarray) -> float:
        """
        Calcula a magnetizacao M = (1/N) Sum S_i
        """
        return np.mean(spins)


# ==============================================================================
# SIMULACAO DE REPLICAS (SIMULATED ANNEALING)
# ==============================================================================

class ReplicaSimulator:
    """
    2. O Metodo das Replicas (The Replica Trick)

    Para resolver a termodinamica desse sistema desordenado, voce deve simular n
    copias identicas (replicas) do mercado rodando simultaneamente com o mesmo
    "ruido congelado" (Quenched Disorder), mas com condicoes iniciais estocasticas
    diferentes.

    Passo 2: Simulated Annealing Paralelo
    Rode 50 "Replicas" do sistema simultaneamente. Esfrie o sistema (reduza a
    temperatura algoritmica T) e veja onde cada replica "trava".
    """

    def __init__(self,
                 n_replicas: int = 50,
                 n_sweeps: int = 1000,
                 T_initial: float = 10.0,
                 T_final: float = 0.1,
                 cooling_rate: float = 0.995):
        """
        Args:
            n_replicas: Numero de replicas a simular
            n_sweeps: Numero de sweeps de Monte Carlo
            T_initial: Temperatura inicial
            T_final: Temperatura final
            cooling_rate: Taxa de resfriamento
        """
        self.n_replicas = n_replicas
        self.n_sweeps = n_sweeps
        self.T_initial = T_initial
        self.T_final = T_final
        self.cooling_rate = cooling_rate

    def initialize_replica(self,
                          n_spins: int,
                          replica_id: int) -> SpinConfiguration:
        """
        Inicializa uma replica com configuracao aleatoria de spins
        """
        # Spins aleatorios +/-1
        spins = np.random.choice([-1, 1], size=n_spins).astype(np.float64)

        return SpinConfiguration(
            spins=spins,
            energy=0.0,
            magnetization=np.mean(spins),
            replica_id=replica_id
        )

    def metropolis_sweep(self,
                        config: SpinConfiguration,
                        interaction: InteractionMatrix,
                        hamiltonian: SpinGlassHamiltonian,
                        temperature: float) -> SpinConfiguration:
        """
        Um sweep de Metropolis-Hastings
        """
        n = len(config.spins)
        spins = config.spins.copy()

        for _ in range(n):
            # Escolhe spin aleatorio
            i = np.random.randint(n)

            # Calcula mudanca de energia se flipar
            local_field = np.sum(interaction.J[i, :] * spins) + interaction.h[i]
            delta_E = 2 * spins[i] * local_field

            # Criterio de Metropolis
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / temperature):
                spins[i] *= -1

        # Atualiza energia e magnetizacao
        energy = hamiltonian.compute_energy(spins, interaction)
        magnetization = hamiltonian.compute_magnetization(spins)

        return SpinConfiguration(
            spins=spins,
            energy=energy,
            magnetization=magnetization,
            replica_id=config.replica_id
        )

    def simulated_annealing(self,
                           interaction: InteractionMatrix,
                           hamiltonian: SpinGlassHamiltonian) -> List[SpinConfiguration]:
        """
        Executa Simulated Annealing para todas as replicas
        """
        n_spins = interaction.J.shape[0]

        # Inicializa replicas
        replicas = [self.initialize_replica(n_spins, i) for i in range(self.n_replicas)]

        # Annealing
        T = self.T_initial

        for sweep in range(self.n_sweeps):
            # Atualiza cada replica
            for i in range(self.n_replicas):
                replicas[i] = self.metropolis_sweep(
                    replicas[i], interaction, hamiltonian, T
                )

            # Resfria
            T = max(self.T_final, T * self.cooling_rate)

        # Calcula energias finais
        for i in range(self.n_replicas):
            replicas[i] = SpinConfiguration(
                spins=replicas[i].spins,
                energy=hamiltonian.compute_energy(replicas[i].spins, interaction),
                magnetization=hamiltonian.compute_magnetization(replicas[i].spins),
                replica_id=i
            )

        return replicas


# ==============================================================================
# MATRIZ DE SOBREPOSICAO E PARAMETRO DE EDWARDS-ANDERSON
# ==============================================================================

class OverlapAnalyzer:
    """
    Voce deve calcular a Matriz de Sobreposicao (Overlap Matrix) Q_ab entre a
    replica alpha e a replica beta:

    Q_ab = (1/N) Sum S_i^a S_i^b

    - Se todas as replicas convergem para o mesmo estado, temos Simetria de
      Replica (RS). O mercado e simples e previsivel.

    - Se as replicas convergem para estados diferentes mas correlacionados
      hierarquicamente, temos Quebra de Simetria de Replicas (RSB). Isso e a
      assinatura matematica de que a baixa volatilidade esta escondendo uma
      complexidade explosiva.
    """

    def __init__(self, n_bins: int = 50):
        """
        Args:
            n_bins: Numero de bins para P(q)
        """
        self.n_bins = n_bins

    def compute_overlap(self,
                       spins_a: np.ndarray,
                       spins_b: np.ndarray) -> float:
        """
        Calcula o overlap entre duas configuracoes

        Q_ab = (1/N) Sum S_i^a S_i^b
        """
        N = len(spins_a)
        return np.sum(spins_a * spins_b) / N

    def compute_overlap_matrix(self,
                              replicas: List[SpinConfiguration]) -> OverlapMatrix:
        """
        Calcula a matriz de overlap completa e a distribuicao P(q)
        """
        n_replicas = len(replicas)

        # Matriz Q_ab
        Q = np.zeros((n_replicas, n_replicas))

        for alpha in range(n_replicas):
            for beta in range(n_replicas):
                Q[alpha, beta] = self.compute_overlap(
                    replicas[alpha].spins, replicas[beta].spins
                )

        # Extrai overlaps nao-diagonais
        overlaps = []
        for alpha in range(n_replicas):
            for beta in range(alpha + 1, n_replicas):
                overlaps.append(Q[alpha, beta])

        overlaps = np.array(overlaps)

        # Parametro de Edwards-Anderson: q_EA = max de P(q)
        # Aproximado pelo overlap medio dos estados de mais baixa energia
        energies = np.array([r.energy for r in replicas])
        low_energy_idx = np.argsort(energies)[:max(5, n_replicas // 10)]

        q_EA_estimates = []
        for i in low_energy_idx:
            for j in low_energy_idx:
                if i < j:
                    q_EA_estimates.append(Q[i, j])

        q_EA = np.mean(np.abs(q_EA_estimates)) if q_EA_estimates else 0.0

        # Distribuicao P(q)
        q_distribution, _ = np.histogram(overlaps, bins=self.n_bins,
                                         range=(-1, 1), density=True)

        return OverlapMatrix(
            Q=Q,
            q_EA=q_EA,
            q_distribution=q_distribution
        )


# ==============================================================================
# ESTRUTURA ULTRAMETRICA
# ==============================================================================

class UltrametricAnalyzer:
    """
    Passo 3: Topologia Ultrametrica

    Aqui esta a magica. Em um Vidro de Spin, os estados formam uma arvore
    genealogica (estrutura ultrametrica). A distancia entre dois estados depende
    do ancestral comum na arvore.

    - Calcule a distancia ultrametrica entre as solucoes das replicas.
    - Plote a funcao x(q) (a probabilidade integrada de ter uma sobreposicao q).
    """

    def __init__(self, ultrametric_tolerance: float = 0.1):
        """
        Args:
            ultrametric_tolerance: Tolerancia para verificar ultrametricidade
        """
        self.tolerance = ultrametric_tolerance

    def compute_distance_matrix(self,
                               overlap_matrix: np.ndarray) -> np.ndarray:
        """
        Converte overlap em distancia: d_ab = 1 - |Q_ab|
        """
        return 1 - np.abs(overlap_matrix)

    def check_ultrametric(self, distance_matrix: np.ndarray) -> Tuple[bool, float]:
        """
        Verifica se a matriz satisfaz a desigualdade ultrametrica:
        d(a,c) <= max(d(a,b), d(b,c))

        Em espacos ultrametricos, todos os triangulos sao isosceles com a base
        sendo o lado mais curto.
        """
        n = distance_matrix.shape[0]
        violations = 0
        total = 0

        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    d_ab = distance_matrix[a, b]
                    d_bc = distance_matrix[b, c]
                    d_ac = distance_matrix[a, c]

                    # Verifica desigualdade ultrametrica
                    max_two = max(d_ab, d_bc)
                    if d_ac > max_two + self.tolerance:
                        violations += 1

                    total += 1

        violation_rate = violations / (total + 1e-10)
        is_ultrametric = violation_rate < 0.1  # Menos de 10% de violacoes

        return is_ultrametric, 1 - violation_rate

    def compute_hierarchy(self,
                         distance_matrix: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Computa clustering hierarquico
        """
        # Converte para vetor condensado
        condensed = squareform(distance_matrix, checks=False)

        # Clustering hierarquico
        try:
            Z = linkage(condensed, method='average')

            # Profundidade da arvore
            depth = int(Z[-1, 2] * 10) if len(Z) > 0 else 0

            # Clusters
            clusters = fcluster(Z, t=0.5, criterion='distance')

            return clusters, depth
        except:
            return np.arange(distance_matrix.shape[0]), 0

    def analyze(self, overlap_matrix: OverlapMatrix) -> UltrametricStructure:
        """
        Analise completa da estrutura ultrametrica
        """
        distance = self.compute_distance_matrix(overlap_matrix.Q)
        is_ultra, ultra_score = self.check_ultrametric(distance)
        clusters, depth = self.compute_hierarchy(distance)

        return UltrametricStructure(
            distance_matrix=distance,
            is_ultrametric=is_ultra,
            hierarchy_depth=depth,
            clustering=clusters
        )


# ==============================================================================
# SUSCEPTIBILIDADE E TRANSICAO DE FASE
# ==============================================================================

class SusceptibilityAnalyzer:
    """
    3. O Gatilho (Quebra de Frustracao / Magnetizacao Espontanea)

    Quando a funcao P(q) colapsa subitamente de uma distribuicao larga para dois
    picos definidos (q_EA e -q_EA), significa que o sistema encontrou o "Caminho
    de Menor Acao" para escapar da armadilha de vidro.

    A simetria foi quebrada. O sistema deixa de ser um vidro e torna-se um
    ferromagneto (Tendencia).

    SINAL: No momento exato em que a Susceptibilidade Magnetica (chi_SG) diverge
    (tende ao infinito):

    chi_SG = (1/T) Sum [<S_i S_j>^2 - <S_i>^2<S_j>^2]

    DIRECAO: Olhe para a "Magnetizacao Remanescente" (M) da replica de menor
    energia. Se M > 0, LONG. Se M < 0, SHORT.
    """

    def __init__(self,
                 divergence_threshold: float = 5.0):
        """
        Args:
            divergence_threshold: Limiar para considerar chi divergindo
        """
        self.divergence_threshold = divergence_threshold
        self.chi_history: List[float] = []

    def compute_spin_glass_susceptibility(self,
                                         replicas: List[SpinConfiguration],
                                         temperature: float) -> float:
        """
        Calcula a susceptibilidade de Spin Glass

        chi_SG = (1/T) Sum [<S_i S_j>^2 - <S_i>^2<S_j>^2]
        """
        n_replicas = len(replicas)
        n_spins = len(replicas[0].spins)

        # Medias
        mean_Si = np.zeros(n_spins)
        mean_SiSj = np.zeros((n_spins, n_spins))

        for replica in replicas:
            mean_Si += replica.spins
            mean_SiSj += np.outer(replica.spins, replica.spins)

        mean_Si /= n_replicas
        mean_SiSj /= n_replicas

        # chi_SG
        chi_SG = 0.0
        for i in range(n_spins):
            for j in range(i + 1, n_spins):
                chi_SG += mean_SiSj[i, j]**2 - mean_Si[i]**2 * mean_Si[j]**2

        chi_SG *= 2 / (temperature + 1e-10)
        chi_SG /= (n_spins * (n_spins - 1) / 2)

        return chi_SG

    def compute_ferromagnetic_susceptibility(self,
                                            replicas: List[SpinConfiguration],
                                            temperature: float) -> float:
        """
        Calcula a susceptibilidade ferromagnetica

        chi_FM = (1/T) * [<M^2> - <M>^2]
        """
        magnetizations = np.array([r.magnetization for r in replicas])

        mean_M = np.mean(magnetizations)
        mean_M2 = np.mean(magnetizations**2)

        chi_FM = (mean_M2 - mean_M**2) / (temperature + 1e-10)

        return chi_FM

    def detect_divergence(self, chi_SG: float) -> bool:
        """
        Detecta se chi_SG esta divergindo
        """
        self.chi_history.append(chi_SG)

        if len(self.chi_history) > 20:
            self.chi_history.pop(0)

        if len(self.chi_history) < 5:
            return False

        # Verifica crescimento exponencial
        recent = np.mean(self.chi_history[-3:])
        older = np.mean(self.chi_history[-6:-3])

        return recent > older * 1.5 or chi_SG > self.divergence_threshold

    def analyze(self,
               replicas: List[SpinConfiguration],
               temperature: float) -> SusceptibilityMetrics:
        """
        Analise completa de susceptibilidade
        """
        chi_SG = self.compute_spin_glass_susceptibility(replicas, temperature)
        chi_FM = self.compute_ferromagnetic_susceptibility(replicas, temperature)

        return SusceptibilityMetrics(
            chi_SG=chi_SG,
            chi_FM=chi_FM,
            divergence_ratio=chi_SG / (chi_FM + 1e-10)
        )

    def reset(self):
        """Reseta historico"""
        self.chi_history.clear()


# ==============================================================================
# DETECTOR DE FASE
# ==============================================================================

class PhaseDetector:
    """
    A Logica de Trading (A Transicao de Fase)

    O indicador monitora a Entropia de Complexidade da funcao P(q).

    1. Fase Paramagnetica (Ruido Puro): P(q) e uma delta de Dirac centrada em zero.
       As replicas nao tem nada a ver uma com a outra.
       Diagnostico: Mercado morto. Baixa volatilidade inutil. Nao operar.

    2. Fase Vidro de Spin (Spin Glass - RSB Full): P(q) tem suporte amplo entre
       [0, q_EA]. As replicas estao presas em vales diferentes. O mercado esta
       tenso, comprimido, "frustrado".
       Diagnostico: Acumulacao Profissional (Baixa Volatilidade Ativa).

    3. O Gatilho (Quebra de Frustracao): Quando P(q) colapsa de uma distribuicao
       larga para dois picos (+/-q_EA), a simetria foi quebrada.
    """

    def __init__(self,
                 paramagnetic_threshold: float = 0.3,
                 rsb_threshold: float = 0.5):
        self.paramagnetic_threshold = paramagnetic_threshold
        self.rsb_threshold = rsb_threshold

    def compute_complexity_entropy(self, q_distribution: np.ndarray) -> float:
        """
        Calcula entropia de complexidade de P(q)
        """
        # Normaliza
        p = q_distribution / (np.sum(q_distribution) + 1e-10)
        p = p[p > 0]  # Remove zeros

        return entropy(p)

    def detect_bimodality(self, q_distribution: np.ndarray) -> bool:
        """
        Detecta se P(q) tem dois picos (simetria quebrando)
        """
        # Suaviza
        smoothed = uniform_filter1d(q_distribution, size=3)

        # Encontra picos
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                if smoothed[i] > np.max(smoothed) * 0.3:  # Pico significativo
                    peaks.append(i)

        return len(peaks) >= 2

    def detect_phase(self,
                    overlap: OverlapMatrix,
                    ultrametric: UltrametricStructure,
                    susceptibility: SusceptibilityMetrics) -> Tuple[PhaseState, SymmetryState]:
        """
        Detecta a fase e estado de simetria atual
        """
        q_EA = overlap.q_EA
        complexity = self.compute_complexity_entropy(overlap.q_distribution)
        is_bimodal = self.detect_bimodality(overlap.q_distribution)

        # Variancia do overlap
        Q_offdiag = overlap.Q[np.triu_indices_from(overlap.Q, k=1)]
        q_variance = np.var(Q_offdiag)

        # Determina fase
        if q_variance < self.paramagnetic_threshold**2:
            phase = PhaseState.PARAMAGNETIC
            symmetry = SymmetryState.RS
        elif is_bimodal:
            phase = PhaseState.FERROMAGNETIC
            symmetry = SymmetryState.BROKEN
        elif ultrametric.is_ultrametric and q_EA > self.rsb_threshold:
            phase = PhaseState.SPIN_GLASS
            symmetry = SymmetryState.RSB
        elif susceptibility.chi_SG > 2 * susceptibility.chi_FM:
            phase = PhaseState.CRITICAL
            symmetry = SymmetryState.RSB
        else:
            phase = PhaseState.PARAMAGNETIC
            symmetry = SymmetryState.RS

        return phase, symmetry


# ==============================================================================
# INDICADOR PRSBD COMPLETO
# ==============================================================================

class ParisiReplicaSymmetryBreakingDetector:
    """
    Parisi Replica Symmetry Breaking Detector (PRSBD)

    Indicador completo que usa fisica de vidros de spin para detectar
    transicoes de volatilidade antes do preco se mover.

    GATILHO:
    - chi_SG diverge (susceptibilidade de spin glass)
    - P(q) colapsa para dois picos
    - Estrutura ultrametrica se desfaz

    DIRECAO:
    - Magnetizacao M > 0 -> LONG
    - Magnetizacao M < 0 -> SHORT
    """

    def __init__(self,
                 # Parametros de replica
                 n_replicas: int = 30,
                 n_sweeps: int = 500,

                 # Parametros de temperatura
                 T_initial: float = 5.0,
                 T_final: float = 0.1,

                 # Parametros de deteccao
                 divergence_threshold: float = 3.0,

                 # Geral
                 min_data_points: int = 50):
        """
        Inicializa o PRSBD
        """
        self.n_replicas = n_replicas
        self.T_final = T_final
        self.min_data_points = min_data_points

        # Componentes
        self.hamiltonian = SpinGlassHamiltonian(temperature=T_final)
        self.replica_sim = ReplicaSimulator(
            n_replicas=n_replicas,
            n_sweeps=n_sweeps,
            T_initial=T_initial,
            T_final=T_final
        )
        self.overlap_analyzer = OverlapAnalyzer()
        self.ultrametric_analyzer = UltrametricAnalyzer()
        self.susceptibility_analyzer = SusceptibilityAnalyzer(
            divergence_threshold=divergence_threshold
        )
        self.phase_detector = PhaseDetector()

    def _prepare_returns_matrix(self,
                               prices: np.ndarray,
                               n_timeframes: int = 5) -> np.ndarray:
        """
        Prepara matriz de retornos em diferentes timeframes
        """
        returns = np.diff(np.log(prices + 1e-10))
        n = len(returns)

        # Simula diferentes timeframes agregando retornos
        returns_matrix = []

        for tf in range(1, n_timeframes + 1):
            # Agrega retornos
            aggregated = []
            for i in range(0, n - tf, tf):
                aggregated.append(np.sum(returns[i:i+tf]))

            if len(aggregated) > 10:
                returns_matrix.append(aggregated[:n // tf])

        # Preenche para ter mesmo tamanho
        min_len = min(len(r) for r in returns_matrix)
        returns_matrix = np.array([r[:min_len] for r in returns_matrix])

        return returns_matrix

    def analyze(self,
               prices: np.ndarray,
               additional_assets: List[np.ndarray] = None) -> dict:
        """
        Processa dados de mercado e retorna analise completa

        Args:
            prices: Array de precos
            additional_assets: Lista de arrays de precos de outros ativos

        Returns:
            Dict com analise completa
        """
        n = len(prices)

        # Validacao
        if n < self.min_data_points:
            return {
                'signal': 0,
                'signal_name': 'INSUFFICIENT_DATA',
                'phase_state': 'PARAMAGNETIC',
                'symmetry_state': 'REPLICA_SYMMETRIC',
                'confidence': 0.0,
                'q_EA': 0.0,
                'q_variance': 0.0,
                'chi_SG': 0.0,
                'chi_FM': 0.0,
                'chi_diverging': False,
                'magnetization': 0.0,
                'magnetization_variance': 0.0,
                'mean_energy': 0.0,
                'energy_variance': 0.0,
                'is_ultrametric': False,
                'hierarchy_depth': 0,
                'complexity_entropy': 0.0,
                'frustration': 0.0,
                'n_replicas': 0,
                'n_converged': 0,
                'n_spins': 0,
                'reasons': ['Dados insuficientes para analise de vidro de spin']
            }

        # ============================================================
        # PASSO 1: CONSTRUCAO DA MATRIZ DE INTERACAO
        # ============================================================
        # Prepara retornos em multiplos timeframes
        returns_matrix = self._prepare_returns_matrix(prices)

        # Adiciona outros ativos se disponiveis
        if additional_assets:
            for asset in additional_assets:
                asset_returns = self._prepare_returns_matrix(asset)
                min_len = min(returns_matrix.shape[1], asset_returns.shape[1])
                returns_matrix = np.vstack([
                    returns_matrix[:, :min_len],
                    asset_returns[:, :min_len]
                ])

        # Constroi matriz de interacao
        interaction = self.hamiltonian.build_interaction_matrix(returns_matrix)
        n_spins = interaction.J.shape[0]

        # ============================================================
        # PASSO 2: SIMULATED ANNEALING PARALELO
        # ============================================================
        self.hamiltonian.n_spins = n_spins
        replicas = self.replica_sim.simulated_annealing(interaction, self.hamiltonian)

        # Estatisticas das replicas
        energies = np.array([r.energy for r in replicas])
        magnetizations = np.array([r.magnetization for r in replicas])

        mean_energy = np.mean(energies)
        energy_variance = np.var(energies)
        mean_magnetization = np.mean(magnetizations)
        magnetization_variance = np.var(magnetizations)

        # ============================================================
        # PASSO 3: MATRIZ DE SOBREPOSICAO
        # ============================================================
        overlap = self.overlap_analyzer.compute_overlap_matrix(replicas)

        # Variancia do overlap
        Q_offdiag = overlap.Q[np.triu_indices_from(overlap.Q, k=1)]
        q_variance = np.var(Q_offdiag)

        # ============================================================
        # PASSO 4: ESTRUTURA ULTRAMETRICA
        # ============================================================
        ultrametric = self.ultrametric_analyzer.analyze(overlap)

        # ============================================================
        # PASSO 5: SUSCEPTIBILIDADE
        # ============================================================
        susceptibility = self.susceptibility_analyzer.analyze(replicas, self.T_final)
        chi_diverging = self.susceptibility_analyzer.detect_divergence(susceptibility.chi_SG)

        # ============================================================
        # PASSO 6: DETECCAO DE FASE
        # ============================================================
        phase_state, symmetry_state = self.phase_detector.detect_phase(
            overlap, ultrametric, susceptibility
        )

        complexity_entropy = self.phase_detector.compute_complexity_entropy(
            overlap.q_distribution
        )

        # ============================================================
        # PASSO 7: GERACAO DE SINAL
        # ============================================================
        signal = 0
        signal_name = 'NEUTRAL'
        confidence = 0.0
        reasons = []

        # Encontra replica de menor energia para direcao
        min_energy_idx = np.argmin(energies)
        best_magnetization = magnetizations[min_energy_idx]

        # PARAMAGNETICO - Mercado morto
        if phase_state == PhaseState.PARAMAGNETIC:
            signal_name = 'WAIT'
            reasons.append(f"PARAMAGNETICO: Mercado morto. q_var={q_variance:.4f}")
            reasons.append(f"Replicas nao correlacionadas. Nao operar.")

        # SPIN GLASS - Acumulacao (RSB)
        elif phase_state == PhaseState.SPIN_GLASS:
            if chi_diverging:
                # Susceptibilidade divergindo = transicao iminente
                if best_magnetization > 0.1:
                    signal = 1
                    signal_name = 'LONG'
                elif best_magnetization < -0.1:
                    signal = -1
                    signal_name = 'SHORT'

                confidence = min(1.0, susceptibility.chi_SG / 5)
                reasons.append(f"RSB -> Transicao: chi_SG={susceptibility.chi_SG:.2f} divergindo!")
                reasons.append(f"M={best_magnetization:.3f}. Simetria quebrando.")
            else:
                signal_name = 'WAIT'
                reasons.append(f"SPIN GLASS (RSB): Frustracao alta ({interaction.frustration:.1%})")
                reasons.append(f"q_EA={overlap.q_EA:.3f}. Acumulacao em andamento.")

        # FERROMAGNETICO - Simetria quebrada (SINAL!)
        elif phase_state == PhaseState.FERROMAGNETIC:
            if best_magnetization > 0.1:
                signal = 1
                signal_name = 'LONG'
            elif best_magnetization < -0.1:
                signal = -1
                signal_name = 'SHORT'

            confidence = min(1.0, abs(best_magnetization) + overlap.q_EA)
            reasons.append(f"FERROMAGNETICO: Simetria QUEBRADA!")
            reasons.append(f"P(q) bimodal. M={best_magnetization:.3f}. Tendencia iniciando.")

        # CRITICO - No ponto de transicao
        elif phase_state == PhaseState.CRITICAL:
            if chi_diverging and abs(best_magnetization) > 0.05:
                if best_magnetization > 0:
                    signal = 1
                    signal_name = 'LONG'
                else:
                    signal = -1
                    signal_name = 'SHORT'

                confidence = min(0.9, susceptibility.divergence_ratio / 5)
                reasons.append(f"CRITICO: No ponto de transicao!")
                reasons.append(f"chi_SG/chi_FM={susceptibility.divergence_ratio:.2f}. M={best_magnetization:.3f}")
            else:
                signal_name = 'WAIT'
                reasons.append(f"CRITICO: Proximo da transicao mas direcao incerta")
                reasons.append(f"M={best_magnetization:.3f}. Aguardando clareza.")

        # Conta replicas convergidas (energia similar)
        energy_threshold = mean_energy + np.sqrt(energy_variance)
        n_converged = int(np.sum(energies < energy_threshold))

        confidence = np.clip(confidence, 0, 1)

        return {
            'signal': signal,
            'signal_name': signal_name,
            'phase_state': phase_state.value,
            'symmetry_state': symmetry_state.value,
            'confidence': confidence,
            'q_EA': overlap.q_EA,
            'q_variance': q_variance,
            'chi_SG': susceptibility.chi_SG,
            'chi_FM': susceptibility.chi_FM,
            'chi_diverging': chi_diverging,
            'divergence_ratio': susceptibility.divergence_ratio,
            'magnetization': mean_magnetization,
            'best_magnetization': best_magnetization,
            'magnetization_variance': magnetization_variance,
            'mean_energy': mean_energy,
            'energy_variance': energy_variance,
            'is_ultrametric': ultrametric.is_ultrametric,
            'hierarchy_depth': ultrametric.hierarchy_depth,
            'complexity_entropy': complexity_entropy,
            'frustration': interaction.frustration,
            'n_replicas': self.n_replicas,
            'n_converged': n_converged,
            'n_spins': n_spins,
            'reasons': reasons
        }

    def get_chi_history(self) -> np.ndarray:
        """Retorna historico de susceptibilidade"""
        return np.array(self.susceptibility_analyzer.chi_history)

    def reset(self):
        """Reseta o estado do indicador"""
        self.susceptibility_analyzer.reset()


# ==============================================================================
# DEMONSTRACAO
# ==============================================================================

def generate_consolidation_data(n_points: int = 100, seed: int = 42) -> np.ndarray:
    """Gera dados de consolidacao (estado vitreo)"""
    np.random.seed(seed)

    # Preco em range apertado (frustrado)
    base = 1.0850

    # Oscilacao dentro de range
    oscillation = 0.0003 * np.sin(np.linspace(0, 10 * np.pi, n_points))

    # Ruido pequeno
    noise = np.random.randn(n_points) * 0.0001

    prices = base + oscillation + noise

    return prices


def main():
    """Demonstracao do indicador PRSBD"""
    print("=" * 70)
    print("PARISI REPLICA SYMMETRY BREAKING DETECTOR (PRSBD)")
    print("Indicador baseado em Fisica de Vidros de Spin")
    print("=" * 70)
    print()

    # Inicializa indicador (parametros reduzidos para demo)
    indicator = ParisiReplicaSymmetryBreakingDetector(
        n_replicas=20,
        n_sweeps=300,
        T_initial=3.0,
        T_final=0.1,
        divergence_threshold=2.0,
        min_data_points=50
    )

    print("Indicador inicializado!")
    print(f"  - Replicas: 20")
    print(f"  - Sweeps: 300")
    print(f"  - T_inicial: 3.0")
    print(f"  - T_final: 0.1")
    print()

    # Gera dados de consolidacao
    prices = generate_consolidation_data(n_points=80)
    print(f"Dados gerados: {len(prices)} pontos")
    print()

    # Processa
    result = indicator.analyze(prices)

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Sinal: {result['signal_name']}")
    print(f"Fase: {result['phase_state']}")
    print(f"Simetria: {result['symmetry_state']}")
    print(f"Confianca: {result['confidence']:.2%}")
    print(f"\nEdwards-Anderson:")
    print(f"  q_EA: {result['q_EA']:.4f}")
    print(f"  q_variance: {result['q_variance']:.4f}")
    print(f"\nSusceptibilidade:")
    print(f"  chi_SG: {result['chi_SG']:.4f}")
    print(f"  chi_FM: {result['chi_FM']:.4f}")
    print(f"  chi divergindo: {result['chi_diverging']}")
    print(f"\nMagnetizacao:")
    print(f"  M: {result['magnetization']:.4f}")
    print(f"  Var(M): {result['magnetization_variance']:.4f}")
    print(f"\nEnergia:")
    print(f"  E media: {result['mean_energy']:.4f}")
    print(f"  Var(E): {result['energy_variance']:.4f}")
    print(f"\nEstrutura:")
    print(f"  Ultrametrico: {result['is_ultrametric']}")
    print(f"  Entropia: {result['complexity_entropy']:.4f}")
    print(f"\nReplicas:")
    print(f"  Total: {result['n_replicas']}")
    print(f"  Convergidas: {result['n_converged']}")
    print(f"\nRazoes: {result['reasons']}")

    return indicator, result


if __name__ == "__main__":
    indicator, result = main()
