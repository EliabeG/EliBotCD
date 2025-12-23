"""
================================================================================
FEYNMAN-KAC QUANTUM PATH INTEGRAL PROPAGATOR (FK-QPIP)
Indicador de Forex baseado em Eletrodinamica Quantica (QED)
================================================================================

Este sistema nao calcula probabilidades simples. Ele calcula a Amplitude de
Probabilidade Complexa somando as contribuicoes de todas as historias possiveis
que o preco poderia ter entre o agora (t_a) e o futuro (t_b), ponderadas pela
Acao Classica do mercado.

A Fisica: Eletrodinamica Quantica (QED) aplicada ao Forex
Em vez de prever onde o preco vai, calcularemos a probabilidade de Tunelamento
Quantico atraves das barreiras de liquidez (Suporte/Resistencia).

A Lagrangiana do Mercado (L):
Na fisica, L = T - V (Energia Cinetica - Energia Potencial). No nosso modelo:
- Energia Cinetica (T): Representada pela volatilidade instantanea e momentum.
  T = (1/2)m(t)x_dot^2 onde m(t) (massa) e o volume inverso.
- Energia Potencial (V): Representada pela densidade do Order Book. Ordens
  limitadas agem como "pocos de potencial" (atraem) ou "barreiras de potencial"
  (repulsao).

A Integral de Caminho (Path Integral):
K(x_b, t_b; x_a, t_a) = Integral Dx(t) exp(i/hbar S[x(t)])

Onde S e a Acao: S[x(t)] = Integral L(x, x_dot, t)dt

Nota de Engenharia: Como estamos em financas, fazemos uma Rotacao de Wick
(t -> -i*tau) para transformar a equacao de Schrodinger em uma Equacao de Difusao
Generalizada (Heat Kernel), tornando a integral real e convergente
(Feynman-Kac Formula).

O Algoritmo: Lattice Quantum Finance
Usa Quantum Monte Carlo (QMC) sobre um reticulado de espaco-tempo (Lattice).

Stack Tecnologica:
- Matematica: Calculo Variacional, Integrais de Trajetoria, Numeros Complexos
- Computacao: Metodo de Monte Carlo via Cadeias de Markov (MCMC), Metropolis
- Hardware: Este algoritmo e massivamente paralelo. Idealmente GPU (CUDA/OpenCL)

Autor: Gerado por Claude AI
Versao: 1.0.0
================================================================================
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

class QEDSignalType(Enum):
    """Tipos de sinais de trading"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    FADE = "FADE"  # Operar contra rompimento falso


class InterferenceType(Enum):
    """Tipo de interferencia quantica"""
    CONSTRUCTIVE = "CONSTRUCTIVE"    # Caminhos convergem (Atrator)
    DESTRUCTIVE = "DESTRUCTIVE"      # Caminhos se cancelam
    MIXED = "MIXED"                  # Interferencia mista


class TunnelingState(Enum):
    """Estado de tunelamento quantico"""
    TUNNELING_LIKELY = "TUNNELING_LIKELY"      # Alta prob de atravessar
    TUNNELING_UNLIKELY = "TUNNELING_UNLIKELY"  # Baixa prob de atravessar
    NO_BARRIER = "NO_BARRIER"                  # Sem barreira detectada


@dataclass
class MarketLagrangian:
    """Lagrangiana do Mercado L = T - V"""
    kinetic_energy: np.ndarray      # T = (1/2)m(t)x_dot^2
    potential_energy: np.ndarray    # V(x) - campo de potencial
    lagrangian: np.ndarray          # L = T - V
    mass: np.ndarray                # m(t) - massa efetiva (1/volume)


@dataclass
class QuantumPath:
    """Um caminho quantico individual"""
    path: np.ndarray                # x(t) - trajetoria
    action: float                   # S - Acao do caminho
    weight: float                   # e^(-S) - Peso estatistico
    is_classical: bool              # Caminho de acao minima?


@dataclass
class TunnelingAnalysis:
    """Analise de tunelamento quantico"""
    tunneling_probability: float    # Probabilidade de atravessar
    barrier_height: float           # Altura da barreira V
    barrier_width: float            # Largura da barreira
    particle_energy: float          # Energia da "particula" (momentum)
    tunneling_state: TunnelingState


@dataclass
class WavefunctionCollapse:
    """Colapso da funcao de onda"""
    probability_density: np.ndarray # psi(x, t_future)
    price_grid: np.ndarray          # Grid de precos
    peak_prices: np.ndarray         # Picos de probabilidade (Atratores)
    peak_probabilities: np.ndarray  # Probabilidade em cada pico
    interference_type: InterferenceType


# ==============================================================================
# LAGRANGIANA DO MERCADO
# ==============================================================================

class MarketLagrangianCalculator:
    """
    A Lagrangiana do Mercado (L)

    Na fisica, L = T - V (Energia Cinetica - Energia Potencial).

    Energia Cinetica (T):
    T = (1/2)m(t)x_dot^2
    Onde m(t) (massa) e o volume inverso (quanto mais volume, mais "pesado"
    e dificil de mover e o preco).

    Energia Potencial (V):
    Representada pela densidade do Order Book. Ordens limitadas agem como
    "pocos de potencial" (atraem o preco) ou "barreiras de potencial" (repulsao).
    - Grandes concentracoes de volume = Alta barreira (V >> 0)
    - Vazios de liquidez = Vacuo (V ~ 0)
    """

    def __init__(self,
                 mass_scale: float = 1.0,
                 potential_scale: float = 1.0):
        """
        Args:
            mass_scale: Escala para a massa efetiva
            potential_scale: Escala para o potencial
        """
        self.mass_scale = mass_scale
        self.potential_scale = potential_scale

    def compute_effective_mass(self, volumes: np.ndarray) -> np.ndarray:
        """
        Computa a massa efetiva m(t) = escala / volume

        Volume alto = massa alta = preco dificil de mover
        Volume baixo = massa baixa = preco move facil
        """
        # Normaliza volumes
        volume_norm = volumes / (np.mean(volumes) + 1e-10)

        # Massa inversamente proporcional ao volume
        # (mais volume = mais "inercia" do mercado)
        mass = self.mass_scale / (volume_norm + 0.1)

        return mass

    def compute_velocity(self, prices: np.ndarray) -> np.ndarray:
        """
        Computa velocidade x_dot = dp/dt (retornos)
        """
        velocity = np.gradient(prices)
        return velocity

    def compute_kinetic_energy(self,
                              mass: np.ndarray,
                              velocity: np.ndarray) -> np.ndarray:
        """
        Energia Cinetica T = (1/2)m(t)x_dot^2
        """
        T = 0.5 * mass * velocity**2
        return T

    def compute_potential_field(self,
                               prices: np.ndarray,
                               volumes: np.ndarray,
                               n_levels: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computa o campo de potencial V(x) baseado no Volume Profile

        Areas de alto volume = barreiras de potencial
        Areas de baixo volume = vacuo (facil passagem)

        Returns:
            Tupla (price_levels, potential)
        """
        # Grid de precos
        price_min = np.min(prices)
        price_max = np.max(prices)
        price_levels = np.linspace(price_min, price_max, n_levels)

        # Acumula volume em cada nivel de preco
        volume_profile = np.zeros(n_levels)

        for p, v in zip(prices, volumes):
            # Encontra o bin mais proximo
            idx = int((p - price_min) / (price_max - price_min + 1e-10) * (n_levels - 1))
            idx = np.clip(idx, 0, n_levels - 1)
            volume_profile[idx] += v

        # Suaviza
        volume_profile = uniform_filter1d(volume_profile, size=3)

        # Potencial proporcional ao volume (barreiras)
        potential = self.potential_scale * volume_profile / (np.max(volume_profile) + 1e-10)

        return price_levels, potential

    def compute_potential_at_price(self,
                                  price: float,
                                  price_levels: np.ndarray,
                                  potential: np.ndarray) -> float:
        """
        Interpola o potencial em um preco especifico
        """
        idx = np.argmin(np.abs(price_levels - price))
        return potential[idx]

    def compute_lagrangian(self,
                          prices: np.ndarray,
                          volumes: np.ndarray) -> MarketLagrangian:
        """
        Computa a Lagrangiana completa L = T - V
        """
        # Massa efetiva
        mass = self.compute_effective_mass(volumes)

        # Velocidade
        velocity = self.compute_velocity(prices)

        # Energia cinetica
        T = self.compute_kinetic_energy(mass, velocity)

        # Campo de potencial
        price_levels, potential = self.compute_potential_field(prices, volumes)

        # V para cada ponto da serie
        V = np.array([self.compute_potential_at_price(p, price_levels, potential)
                     for p in prices])

        # Lagrangiana
        L = T - V

        return MarketLagrangian(
            kinetic_energy=T,
            potential_energy=V,
            lagrangian=L,
            mass=mass
        )


# ==============================================================================
# INTEGRAL DE CAMINHO (PATH INTEGRAL)
# ==============================================================================

class FeynmanPathIntegral:
    """
    A Integral de Caminho (Path Integral)

    A amplitude de probabilidade (Kernel) para o preco ir de x_a para x_b no
    tempo T e dada pela soma sobre TODAS as trajetorias possiveis x(t), nao
    apenas a mais provavel:

    K(x_b, t_b; x_a, t_a) = Integral Dx(t) exp(i/hbar S[x(t)])

    Onde S e a Acao:
    S[x(t)] = Integral L(x, x_dot, t)dt

    Nota de Engenharia: Fazemos uma Rotacao de Wick (t -> -i*tau) para transformar
    em Equacao de Difusao (Heat Kernel), tornando a integral real:

    K = Integral Dx(t) exp(-S[x(t)])
    """

    def __init__(self,
                 hbar: float = 1.0,
                 use_wick_rotation: bool = True):
        """
        Args:
            hbar: Constante de Planck reduzida (escala quantica)
            use_wick_rotation: Se True, usa rotacao de Wick (integral real)
        """
        self.hbar = hbar
        self.use_wick_rotation = use_wick_rotation

    def compute_action(self,
                      path: np.ndarray,
                      lagrangian_calculator: MarketLagrangianCalculator,
                      volumes: np.ndarray,
                      dt: float = 1.0) -> float:
        """
        Computa a Acao S[x(t)] = Integral L(x, x_dot, t)dt para um caminho

        Args:
            path: Trajetoria x(t)
            lagrangian_calculator: Calculador da Lagrangiana
            volumes: Volumes correspondentes
            dt: Passo de tempo
        """
        n = len(path)

        if n < 2:
            return 0.0

        # Velocidade ao longo do caminho
        velocity = np.gradient(path)

        # Massa (usa volume medio como proxy)
        avg_volume = np.mean(volumes)
        mass = lagrangian_calculator.mass_scale / (avg_volume / np.mean(volumes) + 0.1)

        # Energia cinetica
        T = 0.5 * mass * velocity**2

        # Potencial (simplificado: desvio do preco medio)
        price_mean = np.mean(path)
        V = lagrangian_calculator.potential_scale * (path - price_mean)**2

        # Lagrangiana
        L = T - V

        # Acao = integral da Lagrangiana
        action = np.sum(L) * dt

        return action

    def compute_path_weight(self, action: float) -> float:
        """
        Computa o peso estatistico de um caminho

        Com rotacao de Wick: w = exp(-S/hbar)
        Sem rotacao: w = exp(iS/hbar) (complexo)
        """
        if self.use_wick_rotation:
            # Real (Heat Kernel)
            weight = np.exp(-action / self.hbar)
        else:
            # Complexo (original)
            weight = np.exp(1j * action / self.hbar)

        return weight


# ==============================================================================
# QUANTUM MONTE CARLO (METROPOLIS-HASTINGS)
# ==============================================================================

class QuantumMonteCarlo:
    """
    O Algoritmo de Implementacao (Lattice Quantum Finance)

    Voce nao pode resolver a integral analiticamente para um potencial V(x)
    complexo como um Order Book. Voce deve simular isso usando Quantum Monte
    Carlo (QMC) sobre um reticulado de espaco-tempo (Lattice).

    Passo 1: Discretizacao do Espaco-Tempo
    Crie um grid onde o eixo X e o preco e o eixo Y e o tempo futuro
    (proximos 10-15 candles).

    Passo 2: O Campo de Potencial V(x, t)
    Mapeie o Order Book e o historico de volume perfilado (Market Profile)
    em um campo escalar.

    Passo 3: Metropolis-Hastings Path Sampling
    Gere milhares de caminhos aleatorios ("minhocas") que conectam o preco
    atual a varios pontos futuros. Para cada caminho, calcule a Acao S.
    A contribuicao estatistica do caminho e e^(-S). Caminhos com "Acao Minima"
    (que seguem a logica do mercado) contribuem muito. Caminhos malucos
    contribuem pouco.

    Passo 4: O Colapso da Funcao de Onda (Sinal)
    Ao somar todos os caminhos, voce obtem a densidade de probabilidade futura
    psi(x, t_futuro).
    """

    def __init__(self,
                 n_paths: int = 10000,
                 n_time_steps: int = 15,
                 temperature: float = 1.0,
                 step_size: float = 0.001):
        """
        Args:
            n_paths: Numero de caminhos a amostrar
            n_time_steps: Passos de tempo no futuro
            temperature: Temperatura para Metropolis
            step_size: Tamanho do passo para gerar caminhos
        """
        self.n_paths = n_paths
        self.n_time_steps = n_time_steps
        self.temperature = temperature
        self.step_size = step_size

    def generate_random_path(self,
                            start_price: float,
                            volatility: float) -> np.ndarray:
        """
        Gera um caminho aleatorio (random walk com drift)
        """
        path = np.zeros(self.n_time_steps + 1)
        path[0] = start_price

        for t in range(1, self.n_time_steps + 1):
            # Random walk com volatilidade
            dW = np.random.randn() * volatility * self.step_size
            path[t] = path[t-1] + dW

        return path

    def metropolis_step(self,
                       current_path: np.ndarray,
                       current_action: float,
                       lagrangian_calc: MarketLagrangianCalculator,
                       path_integral: FeynmanPathIntegral,
                       volumes: np.ndarray,
                       volatility: float) -> Tuple[np.ndarray, float, bool]:
        """
        Um passo do algoritmo Metropolis-Hastings

        Returns:
            Tupla (new_path, new_action, accepted)
        """
        n = len(current_path)

        # Propoe novo caminho (perturba aleatoriamente)
        proposed_path = current_path.copy()

        # Escolhe um ponto aleatorio para perturbar
        idx = np.random.randint(1, n)
        perturbation = np.random.randn() * volatility * self.step_size * 10
        proposed_path[idx] += perturbation

        # Propaga perturbacao para manter continuidade
        for i in range(idx + 1, n):
            proposed_path[i] += perturbation * (n - i) / (n - idx)

        # Calcula acao do novo caminho
        proposed_action = path_integral.compute_action(
            proposed_path, lagrangian_calc, volumes
        )

        # Criterio de Metropolis
        delta_action = proposed_action - current_action

        if delta_action < 0:
            # Aceita sempre se acao diminui
            return proposed_path, proposed_action, True
        else:
            # Aceita com probabilidade exp(-delta_S/T)
            acceptance_prob = np.exp(-delta_action / self.temperature)
            if np.random.random() < acceptance_prob:
                return proposed_path, proposed_action, True
            else:
                return current_path, current_action, False

    def sample_paths(self,
                    start_price: float,
                    volatility: float,
                    lagrangian_calc: MarketLagrangianCalculator,
                    path_integral: FeynmanPathIntegral,
                    volumes: np.ndarray,
                    n_burnin: int = 100) -> List[QuantumPath]:
        """
        Amostra caminhos via Metropolis-Hastings

        Returns:
            Lista de QuantumPath
        """
        paths = []

        # Inicializa com caminho aleatorio
        current_path = self.generate_random_path(start_price, volatility)
        current_action = path_integral.compute_action(
            current_path, lagrangian_calc, volumes
        )

        # Burn-in
        for _ in range(n_burnin):
            current_path, current_action, _ = self.metropolis_step(
                current_path, current_action, lagrangian_calc,
                path_integral, volumes, volatility
            )

        # Amostragem principal
        min_action = float('inf')

        for i in range(self.n_paths):
            # Passo Metropolis
            current_path, current_action, _ = self.metropolis_step(
                current_path, current_action, lagrangian_calc,
                path_integral, volumes, volatility
            )

            # Peso estatistico
            weight = path_integral.compute_path_weight(current_action)
            if isinstance(weight, complex):
                weight = np.abs(weight)

            # Verifica se e caminho classico (acao minima)
            is_classical = current_action < min_action
            if is_classical:
                min_action = current_action

            # Armazena
            quantum_path = QuantumPath(
                path=current_path.copy(),
                action=current_action,
                weight=weight,
                is_classical=is_classical
            )
            paths.append(quantum_path)

        return paths

    def collapse_wavefunction(self,
                             paths: List[QuantumPath],
                             n_price_bins: int = 50) -> WavefunctionCollapse:
        """
        Passo 4: O Colapso da Funcao de Onda (Sinal)

        Ao somar todos os caminhos, voce obtem a densidade de probabilidade
        futura psi(x, t_futuro).
        """
        # Extrai precos finais de todos os caminhos
        final_prices = np.array([p.path[-1] for p in paths])
        weights = np.array([p.weight for p in paths])

        # Normaliza pesos
        weights = weights / (np.sum(weights) + 1e-10)

        # Grid de precos
        price_min = np.min(final_prices)
        price_max = np.max(final_prices)
        price_grid = np.linspace(price_min, price_max, n_price_bins)

        # Histograma ponderado (densidade de probabilidade)
        probability_density = np.zeros(n_price_bins)

        for price, weight in zip(final_prices, weights):
            idx = int((price - price_min) / (price_max - price_min + 1e-10) * (n_price_bins - 1))
            idx = np.clip(idx, 0, n_price_bins - 1)
            probability_density[idx] += weight

        # Normaliza
        probability_density = probability_density / (np.sum(probability_density) + 1e-10)

        # Suaviza
        probability_density = uniform_filter1d(probability_density, size=3)

        # Encontra picos (Atratores Quanticos)
        peaks = []
        peak_probs = []

        for i in range(1, n_price_bins - 1):
            if (probability_density[i] > probability_density[i-1] and
                probability_density[i] > probability_density[i+1] and
                probability_density[i] > 0.02):  # Threshold
                peaks.append(price_grid[i])
                peak_probs.append(probability_density[i])

        peak_prices = np.array(peaks) if peaks else np.array([np.mean(price_grid)])
        peak_probabilities = np.array(peak_probs) if peak_probs else np.array([1.0])

        # Determina tipo de interferencia
        if len(peak_prices) == 1:
            interference_type = InterferenceType.CONSTRUCTIVE  # Um atrator dominante
        elif len(peak_prices) == 2:
            interference_type = InterferenceType.DESTRUCTIVE   # Dois caminhos competindo
        else:
            interference_type = InterferenceType.MIXED

        return WavefunctionCollapse(
            probability_density=probability_density,
            price_grid=price_grid,
            peak_prices=peak_prices,
            peak_probabilities=peak_probabilities,
            interference_type=interference_type
        )


# ==============================================================================
# ANALISE DE TUNELAMENTO QUANTICO
# ==============================================================================

class QuantumTunnelingAnalyzer:
    """
    A Logica de Trading (Instanton Tunneling)

    O FK-QPIP permite prever "Magica".

    1. O Efeito Tunelamento:
    A analise tecnica classica diz: "O preco nao pode passar dessa resistencia
    porque o volume e alto". O FK-QPIP calcula a probabilidade de tunelamento.
    Se a "barreira de potencial" e estreita (mesmo que alta), e o "momentum"
    da particula (preco) tem alta frequencia, a funcao de onda vaza para o
    outro lado.

    Sinal: Se psi aparecer do outro lado da resistencia antes do preco romper,
    entre LONG no rompimento antes dele acontecer. Voce esta comprando o
    tunelamento.

    2. O Vacuo Quantico (False Breakout Filter):
    Se o preco rompe um nivel, mas a Integral de Caminho mostra que a
    "densidade de probabilidade" nao fluiu junto (a Acao S para manter o preco
    la e muito alta/custosa), e um rompimento falso. O sistema forca o retorno
    ao estado de menor energia.

    Sinal: FADE (operar contra) o rompimento.
    """

    def __init__(self,
                 tunneling_threshold: float = 0.1,
                 barrier_detection_percentile: float = 0.8):
        """
        Args:
            tunneling_threshold: Probabilidade minima para considerar tunelamento
            barrier_detection_percentile: Percentil para detectar barreiras
        """
        self.tunneling_threshold = tunneling_threshold
        self.barrier_detection_percentile = barrier_detection_percentile

    def detect_barriers(self,
                       price_levels: np.ndarray,
                       potential: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Detecta barreiras de potencial (suportes/resistencias)

        Returns:
            Lista de tuplas (price, height, width)
        """
        barriers = []
        threshold = np.percentile(potential, self.barrier_detection_percentile * 100)

        in_barrier = False
        barrier_start = 0
        max_height = 0

        for i, (price, V) in enumerate(zip(price_levels, potential)):
            if V > threshold:
                if not in_barrier:
                    in_barrier = True
                    barrier_start = i
                    max_height = V
                else:
                    max_height = max(max_height, V)
            else:
                if in_barrier:
                    # Fim da barreira
                    barrier_end = i
                    barrier_price = price_levels[(barrier_start + barrier_end) // 2]
                    barrier_width = price_levels[barrier_end] - price_levels[barrier_start]
                    barriers.append((barrier_price, max_height, abs(barrier_width)))
                    in_barrier = False

        return barriers

    def compute_tunneling_probability(self,
                                     particle_energy: float,
                                     barrier_height: float,
                                     barrier_width: float,
                                     mass: float = 1.0,
                                     hbar: float = 1.0) -> float:
        """
        Calcula probabilidade de tunelamento quantico (WKB approximation)

        T ~ exp(-2 * Integral sqrt(2m(V-E)) dx / hbar)

        Para barreira retangular:
        T ~ exp(-2 * sqrt(2m(V-E)) * L / hbar)
        """
        if particle_energy >= barrier_height:
            # Energia suficiente para passar classicamente
            return 1.0

        # Coeficiente de tunelamento
        kappa = np.sqrt(2 * mass * (barrier_height - particle_energy) + 1e-10)

        # Probabilidade WKB
        exponent = -2 * kappa * barrier_width / hbar
        tunneling_prob = np.exp(exponent)

        return np.clip(tunneling_prob, 0, 1)

    def analyze_tunneling(self,
                         current_price: float,
                         price_levels: np.ndarray,
                         potential: np.ndarray,
                         kinetic_energy: float,
                         mass: float) -> TunnelingAnalysis:
        """
        Analise completa de tunelamento
        """
        # Detecta barreiras
        barriers = self.detect_barriers(price_levels, potential)

        if not barriers:
            return TunnelingAnalysis(
                tunneling_probability=1.0,
                barrier_height=0.0,
                barrier_width=0.0,
                particle_energy=kinetic_energy,
                tunneling_state=TunnelingState.NO_BARRIER
            )

        # Encontra barreira mais proxima acima do preco atual
        nearest_barrier = None
        min_distance = float('inf')

        for barrier_price, height, width in barriers:
            distance = abs(barrier_price - current_price)
            if distance < min_distance:
                min_distance = distance
                nearest_barrier = (barrier_price, height, width)

        if nearest_barrier is None:
            return TunnelingAnalysis(
                tunneling_probability=1.0,
                barrier_height=0.0,
                barrier_width=0.0,
                particle_energy=kinetic_energy,
                tunneling_state=TunnelingState.NO_BARRIER
            )

        barrier_price, barrier_height, barrier_width = nearest_barrier

        # Energia da particula
        particle_energy = kinetic_energy

        # Probabilidade de tunelamento
        tunneling_prob = self.compute_tunneling_probability(
            particle_energy, barrier_height, barrier_width, mass
        )

        # Estado
        if tunneling_prob > self.tunneling_threshold:
            tunneling_state = TunnelingState.TUNNELING_LIKELY
        else:
            tunneling_state = TunnelingState.TUNNELING_UNLIKELY

        return TunnelingAnalysis(
            tunneling_probability=tunneling_prob,
            barrier_height=barrier_height,
            barrier_width=barrier_width,
            particle_energy=particle_energy,
            tunneling_state=tunneling_state
        )


# ==============================================================================
# DETECTOR DE INTERFERENCIA
# ==============================================================================

class InterferenceDetector:
    """
    O Fenomeno de Interferencia

    Em media volatilidade, muitas vezes voce vera dois caminhos provaveis
    (ex: subir ou descer) que tem fases opostas.

    - Interferencia Destrutiva: Se os caminhos se cancelam em uma regiao, o
      preco NUNCA ira la, mesmo que a analise tecnica diga que sim.

    - Interferencia Construtiva: Se os caminhos convergem em fase num ponto
      especifico (ex: 1.08500), cria-se um pico de probabilidade. Isso e um
      Atrator Quantico.
    """

    def __init__(self, constructive_threshold: float = 0.3):
        self.constructive_threshold = constructive_threshold

    def analyze_interference(self,
                            wavefunction: WavefunctionCollapse,
                            current_price: float) -> Dict:
        """
        Analisa padroes de interferencia na funcao de onda
        """
        prob_density = wavefunction.probability_density
        price_grid = wavefunction.price_grid

        # Divide em regiao acima e abaixo do preco atual
        current_idx = np.argmin(np.abs(price_grid - current_price))

        prob_above = np.sum(prob_density[current_idx:])
        prob_below = np.sum(prob_density[:current_idx])

        # Normaliza
        total = prob_above + prob_below + 1e-10
        prob_above /= total
        prob_below /= total

        # Detecta onde ha interferencia construtiva forte
        strong_attractors = []
        for i, (price, prob) in enumerate(zip(price_grid, prob_density)):
            if prob > self.constructive_threshold:
                strong_attractors.append(price)

        return {
            'prob_up': prob_above,
            'prob_down': prob_below,
            'strong_attractors': strong_attractors,
            'expected_price': np.sum(price_grid * prob_density)
        }


# ==============================================================================
# INDICADOR FK-QPIP COMPLETO
# ==============================================================================

class FeynmanKacQuantumPathIntegralPropagator:
    """
    Feynman-Kac Quantum Path Integral Propagator (FK-QPIP)

    Indicador completo que usa Integrais de Caminho de Feynman para calcular
    a amplitude de probabilidade quantica do preco futuro.

    A Logica de Trading:

    1. O Efeito Tunelamento:
       - Se psi aparece do outro lado da resistencia antes do preco romper
       - Entre LONG no rompimento antes dele acontecer
       - Voce esta comprando o tunelamento

    2. O Vacuo Quantico (False Breakout Filter):
       - Se preco rompe mas Integral de Caminho nao acompanha
       - A Acao S para manter preco la e muito alta
       - Sinal: FADE (operar contra) o rompimento
    """

    def __init__(self,
                 # Parametros do Monte Carlo
                 n_paths: int = 5000,
                 n_time_steps: int = 15,

                 # Parametros fisicos
                 hbar: float = 0.1,
                 mass_scale: float = 1.0,
                 potential_scale: float = 1.0,

                 # Parametros de tunelamento
                 tunneling_threshold: float = 0.15,

                 # Geral
                 min_data_points: int = 100):
        """
        Inicializa o FK-QPIP
        """
        self.n_paths = n_paths
        self.n_time_steps = n_time_steps
        self.min_data_points = min_data_points

        # Componentes
        self.lagrangian_calc = MarketLagrangianCalculator(
            mass_scale=mass_scale,
            potential_scale=potential_scale
        )
        self.path_integral = FeynmanPathIntegral(
            hbar=hbar,
            use_wick_rotation=True
        )
        self.qmc = QuantumMonteCarlo(
            n_paths=n_paths,
            n_time_steps=n_time_steps
        )
        self.tunneling_analyzer = QuantumTunnelingAnalyzer(
            tunneling_threshold=tunneling_threshold
        )
        self.interference_detector = InterferenceDetector()

        # Cache
        self.last_wavefunction: Optional[WavefunctionCollapse] = None

        # Historicos
        self.prob_up_history: List[float] = []
        self.prob_down_history: List[float] = []
        self.tunneling_history: List[float] = []

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
                'interference_type': 'MIXED',
                'tunneling_state': 'NO_BARRIER',
                'confidence': 0.0,
                'quantum_attractors': [prices[-1] if n > 0 else 0.0],
                'attractor_probabilities': [1.0],
                'tunneling_probability': 0.0,
                'barrier_price': prices[-1] if n > 0 else 0.0,
                'kinetic_energy': 0.0,
                'potential_energy': 0.0,
                'total_action': 0.0,
                'prob_up': 0.5,
                'prob_down': 0.5,
                'expected_price': prices[-1] if n > 0 else 0.0,
                'n_paths_sampled': 0,
                'classical_path_price': prices[-1] if n > 0 else 0.0,
                'reasons': ['Dados insuficientes para analise QED']
            }

        # Volumes sinteticos se nao fornecidos
        if volumes is None:
            volumes = np.abs(np.diff(prices))
            volumes = np.append(volumes, volumes[-1])
            volumes = volumes * 10000 + 1000

        # ============================================================
        # PASSO 1: LAGRANGIANA DO MERCADO
        # ============================================================
        lagrangian = self.lagrangian_calc.compute_lagrangian(prices, volumes)

        current_T = lagrangian.kinetic_energy[-1]
        current_V = lagrangian.potential_energy[-1]

        # ============================================================
        # PASSO 2: CAMPO DE POTENCIAL
        # ============================================================
        price_levels, potential = self.lagrangian_calc.compute_potential_field(
            prices, volumes, n_levels=50
        )

        # ============================================================
        # PASSO 3: QUANTUM MONTE CARLO (PATH SAMPLING)
        # ============================================================
        current_price = prices[-1]
        volatility = np.std(np.diff(np.log(prices + 1e-10)))

        paths = self.qmc.sample_paths(
            start_price=current_price,
            volatility=volatility,
            lagrangian_calc=self.lagrangian_calc,
            path_integral=self.path_integral,
            volumes=volumes,
            n_burnin=50
        )

        # Calcula acao total media
        actions = [p.action for p in paths]
        total_action = np.mean(actions)

        # Encontra caminho classico
        classical_idx = np.argmin(actions)
        classical_path = paths[classical_idx]
        classical_path_price = classical_path.path[-1]

        # ============================================================
        # PASSO 4: COLAPSO DA FUNCAO DE ONDA
        # ============================================================
        wavefunction = self.qmc.collapse_wavefunction(paths)
        self.last_wavefunction = wavefunction

        # ============================================================
        # PASSO 5: ANALISE DE TUNELAMENTO
        # ============================================================
        avg_mass = np.mean(lagrangian.mass)
        tunneling = self.tunneling_analyzer.analyze_tunneling(
            current_price=current_price,
            price_levels=price_levels,
            potential=potential,
            kinetic_energy=current_T,
            mass=avg_mass
        )

        # ============================================================
        # PASSO 6: ANALISE DE INTERFERENCIA
        # ============================================================
        interference = self.interference_detector.analyze_interference(
            wavefunction, current_price
        )

        prob_up = interference['prob_up']
        prob_down = interference['prob_down']
        expected_price = interference['expected_price']

        # Atualiza historicos
        self.prob_up_history.append(prob_up)
        self.prob_down_history.append(prob_down)
        self.tunneling_history.append(tunneling.tunneling_probability)

        if len(self.prob_up_history) > 100:
            self.prob_up_history.pop(0)
            self.prob_down_history.pop(0)
            self.tunneling_history.pop(0)

        # ============================================================
        # PASSO 7: GERACAO DE SINAL
        # ============================================================
        # Verifica movimento de preco recente
        price_change = (prices[-1] - prices[-5]) / prices[-5] if n > 5 else 0

        signal = 0
        signal_name = 'NEUTRAL'
        confidence = 0.0
        reasons = []

        # SINAL 1: Tunelamento Quantico (comprar antes do rompimento)
        if tunneling.tunneling_state == TunnelingState.TUNNELING_LIKELY:
            # psi aparece do outro lado da barreira
            if prob_up > 0.6 and tunneling.tunneling_probability > 0.2:
                signal = 1
                signal_name = 'LONG'
                confidence = tunneling.tunneling_probability * prob_up
                reasons.append(f'Tunelamento: psi atravessa barreira')
                reasons.append(f'P(tunel)={tunneling.tunneling_probability:.2%}, P(up)={prob_up:.2%}')
            elif prob_down > 0.6 and tunneling.tunneling_probability > 0.2:
                signal = -1
                signal_name = 'SHORT'
                confidence = tunneling.tunneling_probability * prob_down
                reasons.append(f'Tunelamento: psi atravessa barreira para baixo')
                reasons.append(f'P(tunel)={tunneling.tunneling_probability:.2%}, P(down)={prob_down:.2%}')

        # SINAL 2: Vacuo Quantico (False Breakout)
        elif wavefunction.interference_type == InterferenceType.DESTRUCTIVE:
            # Preco se moveu mas psi nao acompanhou (interferencia destrutiva)
            if abs(price_change) > 0.001:
                if price_change > 0 and prob_up < 0.4:
                    signal = -1  # FADE = operar contra
                    signal_name = 'FADE'
                    confidence = (1 - prob_up) * 0.8
                    reasons.append(f'Vacuo Quantico: Preco subiu mas psi nao fluiu')
                    reasons.append(f'P(up)={prob_up:.2%} baixa. Rompimento falso.')
                elif price_change < 0 and prob_down < 0.4:
                    signal = 1  # FADE = operar contra
                    signal_name = 'FADE'
                    confidence = (1 - prob_down) * 0.8
                    reasons.append(f'Vacuo Quantico: Preco caiu mas psi nao fluiu')
                    reasons.append(f'P(down)={prob_down:.2%} baixa. Rompimento falso.')

        # SINAL 3: Interferencia Construtiva (Atrator Quantico forte)
        elif wavefunction.interference_type == InterferenceType.CONSTRUCTIVE:
            # Um unico atrator forte
            if len(wavefunction.peak_prices) == 1:
                attractor = wavefunction.peak_prices[0]
                attractor_prob = wavefunction.peak_probabilities[0]

                if attractor > current_price * 1.001 and attractor_prob > 0.3:
                    signal = 1
                    signal_name = 'LONG'
                    confidence = attractor_prob
                    reasons.append(f'Atrator Quantico: Interferencia construtiva em {attractor:.5f}')
                    reasons.append(f'P={attractor_prob:.2%}. Preco sera atraido.')
                elif attractor < current_price * 0.999 and attractor_prob > 0.3:
                    signal = -1
                    signal_name = 'SHORT'
                    confidence = attractor_prob
                    reasons.append(f'Atrator Quantico: Interferencia construtiva em {attractor:.5f}')
                    reasons.append(f'P={attractor_prob:.2%}. Preco sera atraido para baixo.')

        # Sem sinal claro
        if signal == 0:
            reasons.append(f'Sem setup claro. P(up)={prob_up:.2%}, P(down)={prob_down:.2%}')
            reasons.append(f'Interferencia: {wavefunction.interference_type.value}')

        # Ajusta confianca
        confidence = np.clip(confidence, 0, 1)

        # Barreira mais relevante
        barriers = self.tunneling_analyzer.detect_barriers(price_levels, potential)
        barrier_price = barriers[0][0] if barriers else current_price

        return {
            'signal': signal,
            'signal_name': signal_name,
            'interference_type': wavefunction.interference_type.value,
            'tunneling_state': tunneling.tunneling_state.value,
            'confidence': confidence,
            'quantum_attractors': wavefunction.peak_prices.tolist(),
            'attractor_probabilities': wavefunction.peak_probabilities.tolist(),
            'tunneling_probability': tunneling.tunneling_probability,
            'barrier_price': barrier_price,
            'barrier_height': tunneling.barrier_height,
            'barrier_width': tunneling.barrier_width,
            'kinetic_energy': current_T,
            'potential_energy': current_V,
            'total_action': total_action,
            'prob_up': prob_up,
            'prob_down': prob_down,
            'expected_price': expected_price,
            'n_paths_sampled': len(paths),
            'classical_path_price': classical_path_price,
            'reasons': reasons
        }

    def get_wavefunction(self) -> Optional[WavefunctionCollapse]:
        """Retorna a ultima funcao de onda calculada"""
        return self.last_wavefunction

    def get_prob_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna historico de probabilidades"""
        return np.array(self.prob_up_history), np.array(self.prob_down_history)

    def get_tunneling_history(self) -> np.ndarray:
        """Retorna historico de tunelamento"""
        return np.array(self.tunneling_history)

    def reset(self):
        """Reseta o estado do indicador"""
        self.last_wavefunction = None
        self.prob_up_history.clear()
        self.prob_down_history.clear()
        self.tunneling_history.clear()


# ==============================================================================
# DEMONSTRACAO
# ==============================================================================

def generate_barrier_data(n_points: int = 200, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Gera dados com barreira de potencial (resistencia)"""
    np.random.seed(seed)

    t = np.arange(n_points)

    # Base com tendencia
    base = 1.0850 + 0.00001 * t

    # Adiciona oscilacao perto de uma "barreira"
    oscillation = 0.0005 * np.sin(2 * np.pi * t / 30)

    # Ruido
    noise = np.random.randn(n_points) * 0.0002

    prices = base + oscillation + noise

    # Volume (alto perto da barreira)
    barrier_level = 1.0865
    volumes = 1000 + 5000 * np.exp(-((prices - barrier_level) / 0.001)**2)
    volumes += np.random.randn(n_points) * 100

    return prices, volumes


def main():
    """Demonstracao do indicador FK-QPIP"""
    print("=" * 70)
    print("FEYNMAN-KAC QUANTUM PATH INTEGRAL PROPAGATOR (FK-QPIP)")
    print("Indicador baseado em Eletrodinamica Quantica")
    print("=" * 70)
    print()

    # Inicializa indicador (parametros reduzidos para demo)
    indicator = FeynmanKacQuantumPathIntegralPropagator(
        n_paths=2000,
        n_time_steps=10,
        hbar=0.1,
        mass_scale=1.0,
        potential_scale=1.0,
        tunneling_threshold=0.15,
        min_data_points=100
    )

    print("Indicador inicializado!")
    print(f"  - Caminhos: 2000")
    print(f"  - Time steps: 10")
    print(f"  - hbar: 0.1")
    print()

    # Gera dados
    prices, volumes = generate_barrier_data(n_points=150)
    print(f"Dados gerados: {len(prices)} pontos")
    print()

    # Processa
    result = indicator.analyze(prices, volumes)

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Sinal: {result['signal_name']}")
    print(f"Interferencia: {result['interference_type']}")
    print(f"Tunelamento: {result['tunneling_state']}")
    print(f"Confianca: {result['confidence']:.2%}")
    print(f"\nAtratores Quanticos:")
    for p, prob in zip(result['quantum_attractors'], result['attractor_probabilities']):
        print(f"  {p:.5f} (P={prob:.2%})")
    print(f"\nTunelamento:")
    print(f"  Probabilidade: {result['tunneling_probability']:.2%}")
    print(f"  Barreira: {result['barrier_price']:.5f}")
    print(f"\nEnergia:")
    print(f"  T (cinetica): {result['kinetic_energy']:.6f}")
    print(f"  V (potencial): {result['potential_energy']:.6f}")
    print(f"  S (acao): {result['total_action']:.4f}")
    print(f"\nProbabilidades:")
    print(f"  P(up): {result['prob_up']:.2%}")
    print(f"  P(down): {result['prob_down']:.2%}")
    print(f"  E[preco]: {result['expected_price']:.5f}")
    print(f"\nCaminhos:")
    print(f"  Amostrados: {result['n_paths_sampled']}")
    print(f"  Preco classico: {result['classical_path_price']:.5f}")
    print(f"\nRazoes: {result['reasons']}")

    return indicator, result


if __name__ == "__main__":
    indicator, result = main()
