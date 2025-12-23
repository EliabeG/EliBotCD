"""
================================================================================
NEUROMORPHIC SPIKING SYNAPTIC PRE-POTENTIATION SCANNER (NS-PPS)
Indicador de Forex baseado em Redes Neurais de Espiking (SNN)
================================================================================

Este indicador utiliza Redes Neurais de Espiking (SNN - Spiking Neural Networks)
de terceira geração. Diferente das redes neurais de Deep Learning comuns (que
usam taxas de disparo contínuas), as SNNs operam no domínio do tempo discreto,
simulando a voltagem elétrica real de neurônios biológicos.

O objetivo é medir a Excitabilidade Cortical do mercado. Queremos saber quão
perto o "cérebro" do EURUSD está de disparar um potencial de ação coletivo.

A Lógica de Trading (Previsão de Epilepsia Financeira)
O indicador busca o estado de "Hiperexcitabilidade Silenciosa".

Por que usar Computação Neuromórfica?
1. Processamento Temporal Preciso: Redes Neurais comuns perdem a informação do
   tempo exato entre ticks. SNNs vivem no tempo. Em HFT, a diferença de 1ms muda
   a causalidade. STDP captura essa causalidade.
2. Eficiência Energética de Dados: O modelo ignora ruído (que não atinge o limiar)
   e foca apenas em acumulação de sinal real. É um filtro de ruído natural
   biologicamente plausível.
3. Antecipação de Explosão: Modelos lineares esperam o movimento começar para
   dizer "tendência". O modelo LIF mostra que a "voltagem" estava subindo horas
   antes do movimento, mesmo com o preço parado. Ele mede a PRESSÃO, não o movimento.

Desafio de Código: O ajuste dos parâmetros (τ_m, V_th) é delicado. Sugiro
implementar um Algoritmo Genético que roda em background para evoluir os
parâmetros da rede neural.

Autor: Gerado por Claude AI
Versão: 1.0.0
================================================================================
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.sparse import lil_matrix, csr_matrix
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

class CorticalState(Enum):
    """Estado cortical da rede"""
    SUBCRITICAL = "SUBCRITICAL"         # κ < 1 - sinais morrem
    CRITICAL = "CRITICAL"                # κ ≈ 1 - criticalidade
    SUPERCRITICAL = "SUPERCRITICAL"      # κ > 1 - avalanche
    PRE_ICTAL = "PRE_ICTAL"              # Pré-convulsão (estado perigoso)


class ExcitabilityState(Enum):
    """Estado de excitabilidade"""
    QUIESCENT = "QUIESCENT"              # Silêncio (baixa excitabilidade)
    BUILDING = "BUILDING"                 # Acumulando potencial
    HYPEREXCITABLE = "HYPEREXCITABLE"     # Hiperexcitável (perigo!)
    FIRING = "FIRING"                     # Disparando (explosão em andamento)


@dataclass
class LIFNeuron:
    """Neurônio Leaky Integrate-and-Fire"""
    neuron_id: int
    price_level: float              # Nível de preço que representa
    membrane_potential: float       # V(t) - Potencial de membrana
    resting_potential: float        # E_rest - Potencial de repouso
    threshold: float                # V_th - Limiar de disparo
    last_spike_time: float          # Tempo do último spike
    is_refractory: bool             # Em período refratário?


@dataclass
class SynapticConnection:
    """Conexão sináptica entre neurônios"""
    pre_neuron: int                 # Neurônio pré-sináptico
    post_neuron: int                # Neurônio pós-sináptico
    weight: float                   # Peso sináptico
    delay: float                    # Atraso sináptico


@dataclass
class SpikeEvent:
    """Evento de spike"""
    neuron_id: int
    time: float
    price_level: float


@dataclass
class STDPUpdate:
    """Atualização de plasticidade STDP"""
    pre_neuron: int
    post_neuron: int
    delta_weight: float
    timing_diff: float              # Δt = t_post - t_pre


@dataclass
class NetworkState:
    """Estado da rede neural"""
    mean_potential: float           # V̄ - Potencial médio
    potential_std: float            # Desvio padrão
    firing_rate: float              # Taxa de disparo
    avalanche_coefficient: float    # κ - coeficiente de avalanche
    synchrony_index: float          # Índice de sincronia (Kuramoto)
    cortical_state: CorticalState
    excitability_state: ExcitabilityState


@dataclass
class STDPPathway:
    """Caminho sináptico formado por STDP"""
    pathway_neurons: List[int]      # Neurônios no caminho
    total_weight: float             # Peso total do caminho
    direction: str                  # "UP" ou "DOWN"
    strength: float                 # Força relativa


# ==============================================================================
# NEURÔNIO LIF (LEAKY INTEGRATE-AND-FIRE)
# ==============================================================================

class LIFNeuronModel:
    """
    1. A Dinâmica do Neurônio Individual (V_m)

    Cada nível de preço no Order Book é um neurônio. O fluxo de ordens
    (compras/vendas) é a corrente elétrica de entrada I(t). A evolução da
    voltagem da membrana V(t) é dada pela equação diferencial estocástica:

    τ_m * dV/dt = -(V(t) - E_rest) + R_m * I(t) + σξ(t)

    - τ_m: Constante de tempo da membrana (memória do mercado)
    - E_rest: Potencial de repouso (preço justo atual)
    - I(t): Agressão do fluxo de ordens

    O Mecanismo de Disparo: Se V(t) > V_threshold (Limiar de Disparo), o neurônio
    dispara um "Spike" (sinal de trade), reseta sua voltagem e envia um sinal
    excitatório para os neurônios vizinhos.
    """

    def __init__(self,
                 tau_m: float = 20.0,           # Constante de tempo (ms)
                 R_m: float = 1.0,              # Resistência da membrana
                 E_rest: float = -70.0,         # Potencial de repouso (mV)
                 V_threshold: float = -55.0,    # Limiar de disparo (mV)
                 V_reset: float = -75.0,        # Potencial pós-spike
                 refractory_period: float = 2.0,# Período refratário (ms)
                 noise_sigma: float = 1.0):     # Ruído
        """
        Inicializa o modelo LIF
        """
        self.tau_m = tau_m
        self.R_m = R_m
        self.E_rest = E_rest
        self.V_threshold = V_threshold
        self.V_reset = V_reset
        self.refractory_period = refractory_period
        self.noise_sigma = noise_sigma

    def create_neuron(self, neuron_id: int, price_level: float) -> LIFNeuron:
        """Cria um novo neurônio"""
        return LIFNeuron(
            neuron_id=neuron_id,
            price_level=price_level,
            membrane_potential=self.E_rest + np.random.randn() * 5,
            resting_potential=self.E_rest,
            threshold=self.V_threshold,
            last_spike_time=-float('inf'),
            is_refractory=False
        )

    def update(self,
              neuron: LIFNeuron,
              I_input: float,
              dt: float,
              current_time: float) -> Tuple[LIFNeuron, bool]:
        """
        Atualiza o neurônio por um passo de tempo

        Returns:
            Tupla (neurônio atualizado, disparou spike?)
        """
        # Verifica período refratário
        if current_time - neuron.last_spike_time < self.refractory_period:
            neuron.is_refractory = True
            return neuron, False

        neuron.is_refractory = False

        # Equação LIF: τ_m * dV/dt = -(V - E_rest) + R_m * I + σξ
        noise = self.noise_sigma * np.random.randn() * np.sqrt(dt)

        dV = dt / self.tau_m * (
            -(neuron.membrane_potential - neuron.resting_potential) +
            self.R_m * I_input +
            noise
        )

        neuron.membrane_potential += dV

        # Verifica disparo
        fired = False
        if neuron.membrane_potential >= neuron.threshold:
            fired = True
            neuron.membrane_potential = self.V_reset
            neuron.last_spike_time = current_time

        return neuron, fired


# ==============================================================================
# PLASTICIDADE STDP (SPIKE-TIMING DEPENDENT PLASTICITY)
# ==============================================================================

class STDPPlasticity:
    """
    2. Plasticidade Sináptica (STDP)

    Aqui está a chave da adaptação. As conexões entre os preços não são fixas.
    Elas aprendem com a Plasticidade Dependente do Tempo de Disparo (STDP).

    - Se o neurônio A (preço 1.0500) dispara pouco antes do neurônio B (preço
      1.0501), a conexão A → B é fortalecida (LTP - Long Term Potentiation).

    - Isso cria "Estradas Neurais" invisíveis. Em baixa volatilidade, o mercado
      está pavimentando o caminho do rompimento silenciosamente através do
      fortalecimento sináptico.
    """

    def __init__(self,
                 A_plus: float = 0.01,      # Amplitude de potenciação
                 A_minus: float = 0.012,    # Amplitude de depressão
                 tau_plus: float = 20.0,    # Constante de tempo (LTP)
                 tau_minus: float = 20.0,   # Constante de tempo (LTD)
                 w_max: float = 1.0,        # Peso máximo
                 w_min: float = 0.0):       # Peso mínimo
        """
        Inicializa STDP
        """
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_max = w_max
        self.w_min = w_min

    def compute_weight_change(self,
                             delta_t: float,
                             current_weight: float) -> float:
        """
        Calcula mudança de peso baseada no timing dos spikes

        delta_t = t_post - t_pre
        - delta_t > 0: Pré antes de pós → LTP (potenciação)
        - delta_t < 0: Pós antes de pré → LTD (depressão)
        """
        if delta_t > 0:
            # LTP: A+ * exp(-Δt/τ+)
            dw = self.A_plus * np.exp(-delta_t / self.tau_plus)
        else:
            # LTD: -A- * exp(Δt/τ-)
            dw = -self.A_minus * np.exp(delta_t / self.tau_minus)

        # Aplica limites
        new_weight = np.clip(current_weight + dw, self.w_min, self.w_max)

        return new_weight - current_weight

    def update_synapse(self,
                      pre_spike_time: float,
                      post_spike_time: float,
                      current_weight: float) -> Tuple[float, STDPUpdate]:
        """
        Atualiza uma sinapse baseado nos tempos de spike
        """
        delta_t = post_spike_time - pre_spike_time
        dw = self.compute_weight_change(delta_t, current_weight)
        new_weight = current_weight + dw

        update = STDPUpdate(
            pre_neuron=-1,  # Será preenchido pelo chamador
            post_neuron=-1,
            delta_weight=dw,
            timing_diff=delta_t
        )

        return new_weight, update


# ==============================================================================
# REDE NEURAL SMALL-WORLD
# ==============================================================================

class SmallWorldNetwork:
    """
    Topologia Small-World para a rede de neurônios

    Combina alta conectividade local (como um anel) com atalhos aleatórios
    (como grafo aleatório). Isso modela como diferentes níveis de preço
    podem estar conectados através de níveis intermediários.
    """

    def __init__(self,
                 n_neurons: int = 1000,
                 k_neighbors: int = 4,
                 rewire_prob: float = 0.1):
        """
        Args:
            n_neurons: Número de neurônios
            k_neighbors: Vizinhos em cada lado
            rewire_prob: Probabilidade de reconexão
        """
        self.n_neurons = n_neurons
        self.k_neighbors = k_neighbors
        self.rewire_prob = rewire_prob

    def build_network(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constrói a rede Small-World (Watts-Strogatz)

        Returns:
            Tupla (adjacency matrix, weight matrix)
        """
        n = self.n_neurons

        # Matriz de adjacência esparsa
        adj = lil_matrix((n, n), dtype=np.float32)
        weights = lil_matrix((n, n), dtype=np.float32)

        # Conecta vizinhos em anel
        for i in range(n):
            for j in range(1, self.k_neighbors + 1):
                # Vizinho à direita
                right = (i + j) % n
                adj[i, right] = 1
                weights[i, right] = np.random.uniform(0.3, 0.7)

                # Vizinho à esquerda
                left = (i - j) % n
                adj[i, left] = 1
                weights[i, left] = np.random.uniform(0.3, 0.7)

        # Reconecta algumas arestas (Small-World)
        for i in range(n):
            for j in range(1, self.k_neighbors + 1):
                if np.random.random() < self.rewire_prob:
                    right = (i + j) % n

                    # Remove conexão antiga
                    adj[i, right] = 0
                    weights[i, right] = 0

                    # Adiciona conexão aleatória
                    new_target = np.random.randint(n)
                    while new_target == i or adj[i, new_target] > 0:
                        new_target = np.random.randint(n)

                    adj[i, new_target] = 1
                    weights[i, new_target] = np.random.uniform(0.3, 0.7)

        return csr_matrix(adj), csr_matrix(weights)


# ==============================================================================
# CODIFICADOR DE SPIKES
# ==============================================================================

class SpikeEncoder:
    """
    Passo 1: Codificação Sensorial (Spike Encoding)

    Transforme o tick data em trens de spikes.

    - Tick de alta de preço = Spike no canal excitatório
    - Tick de baixa = Spike no canal inibitório
    - Silêncio = Decaimento exponencial do potencial (Leakage)
    """

    def __init__(self,
                 base_current: float = 5.0,
                 volume_scale: float = 0.01):
        """
        Args:
            base_current: Corrente base por tick
            volume_scale: Escala do volume para corrente
        """
        self.base_current = base_current
        self.volume_scale = volume_scale

    def encode_tick(self,
                   price_change: float,
                   volume: float,
                   n_neurons: int,
                   price_levels: np.ndarray) -> np.ndarray:
        """
        Codifica um tick em correntes de entrada para os neurônios
        """
        currents = np.zeros(n_neurons)

        # Encontra neurônios próximos ao nível de preço afetado
        # (distribuição gaussiana de corrente)
        current_price = price_levels[n_neurons // 2]  # Preço central

        for i, level in enumerate(price_levels):
            distance = abs(level - current_price)

            # Corrente decai com a distância
            spatial_decay = np.exp(-distance * 100)  # Escala do preço

            # Corrente baseada na direção
            if price_change > 0:
                # Alta: excita neurônios acima
                if level >= current_price:
                    currents[i] = self.base_current * spatial_decay
                else:
                    currents[i] = -self.base_current * spatial_decay * 0.5
            elif price_change < 0:
                # Baixa: excita neurônios abaixo
                if level <= current_price:
                    currents[i] = self.base_current * spatial_decay
                else:
                    currents[i] = -self.base_current * spatial_decay * 0.5

            # Escala pelo volume
            currents[i] *= (1 + volume * self.volume_scale)

        return currents

    def encode_order_flow(self,
                         bid_volume: float,
                         ask_volume: float,
                         n_neurons: int) -> np.ndarray:
        """
        Codifica fluxo de ordens em correntes
        """
        currents = np.zeros(n_neurons)

        # Metade superior: sensível a compras
        # Metade inferior: sensível a vendas
        mid = n_neurons // 2

        bid_current = self.base_current * (bid_volume / (bid_volume + ask_volume + 1e-10))
        ask_current = self.base_current * (ask_volume / (bid_volume + ask_volume + 1e-10))

        # Distribui com decaimento
        for i in range(n_neurons):
            if i >= mid:
                # Neurônios superiores (compra/alta)
                distance = (i - mid) / (n_neurons - mid)
                currents[i] = bid_current * np.exp(-distance * 2)
            else:
                # Neurônios inferiores (venda/baixa)
                distance = (mid - i) / mid
                currents[i] = ask_current * np.exp(-distance * 2)

        return currents


# ==============================================================================
# DETECTOR DE AVALANCHE
# ==============================================================================

class AvalancheDetector:
    """
    3. O Coeficiente de Avalanche (κ)

    Baseado na neurociência de sistemas críticos. Você monitora o Branching
    Parameter (parâmetro de ramificação) da rede.

    κ = Número de neurônios ativados no passo t + 1 / Número de neurônios
        ativados no passo t

    - Em baixa volatilidade segura: κ < 1 (sinais morrem, mercado sub-crítico)
    - Em baixa volatilidade perigosa: κ → 1 (Criticalidade). O sistema torna-se
      capaz de sustentar uma atividade autopropagada infinita.
    """

    def __init__(self,
                 critical_threshold: float = 0.95,
                 supercritical_threshold: float = 1.05):
        self.critical_threshold = critical_threshold
        self.supercritical_threshold = supercritical_threshold

        # Histórico
        self.n_active_history: List[int] = []
        self.kappa_history: List[float] = []

    def compute_kappa(self, n_active_t: int, n_active_t1: int) -> float:
        """
        Calcula coeficiente de avalanche

        κ = N(t+1) / N(t)
        """
        if n_active_t == 0:
            return 0.0

        return n_active_t1 / n_active_t

    def update(self, n_active: int) -> float:
        """
        Atualiza histórico e retorna κ atual
        """
        self.n_active_history.append(n_active)

        if len(self.n_active_history) > 100:
            self.n_active_history.pop(0)

        if len(self.n_active_history) < 2:
            return 0.0

        kappa = self.compute_kappa(
            self.n_active_history[-2],
            self.n_active_history[-1]
        )

        self.kappa_history.append(kappa)
        if len(self.kappa_history) > 100:
            self.kappa_history.pop(0)

        return kappa

    def get_average_kappa(self, window: int = 10) -> float:
        """Retorna κ médio"""
        if len(self.kappa_history) < window:
            return np.mean(self.kappa_history) if self.kappa_history else 0.0
        return np.mean(self.kappa_history[-window:])

    def detect_state(self, kappa: float) -> CorticalState:
        """Detecta estado cortical baseado em κ"""
        if kappa < self.critical_threshold:
            return CorticalState.SUBCRITICAL
        elif kappa < self.supercritical_threshold:
            return CorticalState.CRITICAL
        else:
            return CorticalState.SUPERCRITICAL

    def reset(self):
        """Reseta histórico"""
        self.n_active_history.clear()
        self.kappa_history.clear()


# ==============================================================================
# DETECTOR DE SINCRONIA (KURAMOTO)
# ==============================================================================

class SynchronyDetector:
    """
    Passo 3: Detecção de Sincronia de Fase

    Meça o Índice de Sincronia de Kuramoto dos potenciais sub-limiares. Mesmo
    sem trades (sem spikes), as voltagens dos neurônios podem começar a oscilar
    em uníssono.
    """

    def __init__(self, phase_threshold: float = 0.7):
        self.phase_threshold = phase_threshold

    def compute_kuramoto_index(self, potentials: np.ndarray) -> float:
        """
        Calcula índice de sincronia de Kuramoto

        r = |1/N Σ exp(i*θ_j)|

        Onde θ_j é a fase do neurônio j (baseada no potencial)
        """
        N = len(potentials)

        if N == 0:
            return 0.0

        # Converte potenciais para fases (normaliza para [0, 2π])
        pot_min = np.min(potentials)
        pot_max = np.max(potentials)

        if pot_max - pot_min < 1e-10:
            return 1.0  # Todos iguais = sincronia perfeita

        normalized = (potentials - pot_min) / (pot_max - pot_min)
        phases = 2 * np.pi * normalized

        # Índice de Kuramoto
        complex_order = np.mean(np.exp(1j * phases))
        r = np.abs(complex_order)

        return r

    def detect_synchrony(self, potentials: np.ndarray) -> Tuple[float, bool]:
        """
        Detecta se a rede está sincronizada
        """
        r = self.compute_kuramoto_index(potentials)
        is_synchronized = r > self.phase_threshold

        return r, is_synchronized


# ==============================================================================
# ANALISADOR DE CAMINHOS STDP
# ==============================================================================

class STDPPathwayAnalyzer:
    """
    Analisa os "caminhos neurais" formados por STDP

    Detecta se o STDP formou uma "trilha forte" apontando para cima ou para baixo.
    """

    def __init__(self, pathway_threshold: float = 0.5):
        self.pathway_threshold = pathway_threshold

    def analyze_pathways(self,
                        weights: np.ndarray,
                        n_neurons: int) -> Tuple[STDPPathway, STDPPathway]:
        """
        Analisa caminhos UP e DOWN formados por STDP
        """
        mid = n_neurons // 2

        # Caminho UP: conexões de baixo para cima (i < j)
        up_weights = []
        for i in range(n_neurons - 1):
            for j in range(i + 1, min(i + 10, n_neurons)):  # Vizinhos próximos
                if weights[i, j] > self.pathway_threshold:
                    up_weights.append(weights[i, j])

        # Caminho DOWN: conexões de cima para baixo (i > j)
        down_weights = []
        for i in range(1, n_neurons):
            for j in range(max(0, i - 10), i):
                if weights[i, j] > self.pathway_threshold:
                    down_weights.append(weights[i, j])

        up_strength = np.mean(up_weights) if up_weights else 0.0
        down_strength = np.mean(down_weights) if down_weights else 0.0

        up_pathway = STDPPathway(
            pathway_neurons=[],
            total_weight=np.sum(up_weights) if up_weights else 0.0,
            direction="UP",
            strength=up_strength
        )

        down_pathway = STDPPathway(
            pathway_neurons=[],
            total_weight=np.sum(down_weights) if down_weights else 0.0,
            direction="DOWN",
            strength=down_strength
        )

        return up_pathway, down_pathway

    def compute_asymmetry(self, up_pathway: STDPPathway, down_pathway: STDPPathway) -> float:
        """
        Computa assimetria entre caminhos

        Positivo = mais força para cima
        Negativo = mais força para baixo
        """
        total = up_pathway.strength + down_pathway.strength + 1e-10
        asymmetry = (up_pathway.strength - down_pathway.strength) / total

        return asymmetry


# ==============================================================================
# SIMULADOR DA REDE NEURAL
# ==============================================================================

class SpikingNetworkSimulator:
    """
    Passo 2: Simulação da Rede Recorrente

    Execute a rede. Em baixa volatilidade, você não verá muitos spikes de saída
    (output), mas verá o Potencial de Membrana Médio (V̄) de toda a população
    subindo.
    """

    def __init__(self,
                 n_neurons: int = 500,
                 dt: float = 1.0,
                 simulation_steps: int = 100):
        """
        Args:
            n_neurons: Número de neurônios na rede
            dt: Passo de tempo (ms)
            simulation_steps: Passos de simulação
        """
        self.n_neurons = n_neurons
        self.dt = dt
        self.simulation_steps = simulation_steps

        # Componentes
        self.lif_model = LIFNeuronModel()
        self.stdp = STDPPlasticity()
        self.network_builder = SmallWorldNetwork(n_neurons=n_neurons)
        self.spike_encoder = SpikeEncoder()
        self.avalanche_detector = AvalancheDetector()
        self.synchrony_detector = SynchronyDetector()
        self.pathway_analyzer = STDPPathwayAnalyzer()

        # Inicializa rede
        self._init_network()

    def _init_network(self):
        """Inicializa a rede neural"""
        # Cria neurônios
        self.neurons: List[LIFNeuron] = []
        price_range = 0.001  # Range de preço

        for i in range(self.n_neurons):
            price_level = 1.0850 + (i - self.n_neurons // 2) * price_range / self.n_neurons
            neuron = self.lif_model.create_neuron(i, price_level)
            self.neurons.append(neuron)

        # Níveis de preço
        self.price_levels = np.array([n.price_level for n in self.neurons])

        # Constrói topologia Small-World
        self.adjacency, self.weights = self.network_builder.build_network()
        self.weights_dense = self.weights.toarray()

        # Histórico de spikes
        self.spike_history: List[SpikeEvent] = []
        self.last_spike_times = np.full(self.n_neurons, -float('inf'))

    def simulate_step(self,
                     input_currents: np.ndarray,
                     current_time: float) -> Tuple[List[SpikeEvent], int]:
        """
        Simula um passo de tempo
        """
        spikes = []
        n_fired = 0

        # Processa cada neurônio
        for i, neuron in enumerate(self.neurons):
            # Corrente de entrada (externa + sináptica)
            I_ext = input_currents[i]

            # Corrente sináptica de neurônios pré-sinápticos que dispararam
            I_syn = 0.0
            for j in range(self.n_neurons):
                if self.weights_dense[j, i] > 0:
                    # Verifica se j disparou recentemente
                    if current_time - self.last_spike_times[j] < 5.0:  # Janela de 5ms
                        I_syn += self.weights_dense[j, i] * 10  # Escala

            I_total = I_ext + I_syn

            # Atualiza neurônio
            self.neurons[i], fired = self.lif_model.update(
                neuron, I_total, self.dt, current_time
            )

            if fired:
                n_fired += 1
                self.last_spike_times[i] = current_time

                spike = SpikeEvent(
                    neuron_id=i,
                    time=current_time,
                    price_level=neuron.price_level
                )
                spikes.append(spike)
                self.spike_history.append(spike)

                # Atualiza STDP para todas as sinapses pós-sinápticas
                self._update_stdp(i, current_time)

        return spikes, n_fired

    def _update_stdp(self, post_neuron: int, post_time: float):
        """
        Atualiza pesos sinápticos via STDP após um spike
        """
        for pre_neuron in range(self.n_neurons):
            if self.weights_dense[pre_neuron, post_neuron] > 0:
                pre_time = self.last_spike_times[pre_neuron]

                if pre_time > -float('inf'):
                    new_weight, _ = self.stdp.update_synapse(
                        pre_time, post_time,
                        self.weights_dense[pre_neuron, post_neuron]
                    )
                    self.weights_dense[pre_neuron, post_neuron] = new_weight

    def get_network_state(self) -> NetworkState:
        """
        Obtém estado atual da rede
        """
        # Potenciais
        potentials = np.array([n.membrane_potential for n in self.neurons])
        mean_potential = np.mean(potentials)
        potential_std = np.std(potentials)

        # Neurônios ativos (perto do limiar)
        threshold = self.lif_model.V_threshold
        n_near_threshold = np.sum(potentials > threshold - 10)

        # Taxa de disparo
        recent_spikes = [s for s in self.spike_history[-100:]]
        firing_rate = len(recent_spikes) / (100 * self.dt / 1000)  # Hz

        # Coeficiente de avalanche
        kappa = self.avalanche_detector.get_average_kappa()

        # Sincronia
        synchrony, _ = self.synchrony_detector.detect_synchrony(potentials)

        # Estados
        cortical_state = self.avalanche_detector.detect_state(kappa)

        # Excitabilidade
        potential_ratio = (mean_potential - self.lif_model.E_rest) / \
                         (threshold - self.lif_model.E_rest)

        if potential_ratio < 0.5:
            excitability = ExcitabilityState.QUIESCENT
        elif potential_ratio < 0.8:
            excitability = ExcitabilityState.BUILDING
        elif potential_ratio < 0.95:
            excitability = ExcitabilityState.HYPEREXCITABLE
        else:
            excitability = ExcitabilityState.FIRING

        return NetworkState(
            mean_potential=mean_potential,
            potential_std=potential_std,
            firing_rate=firing_rate,
            avalanche_coefficient=kappa,
            synchrony_index=synchrony,
            cortical_state=cortical_state,
            excitability_state=excitability
        )

    def reset(self):
        """Reseta a rede"""
        self._init_network()
        self.avalanche_detector.reset()


# ==============================================================================
# INDICADOR NS-PPS COMPLETO
# ==============================================================================

class NeuromorphicSpikingPrePotentiationScanner:
    """
    Neuromorphic Spiking Synaptic Pre-Potentiation Scanner (NS-PPS)

    Indicador completo que usa redes neurais de espiking para detectar
    hiperexcitabilidade silenciosa em baixa volatilidade.

    A Lógica de Trading (Previsão de Epilepsia Financeira)
    O indicador busca o estado de "Hiperexcitabilidade Silenciosa".

    1. Monitoramento: O mercado está parado (baixa vol). O gráfico de velas é
       uma linha reta.

    2. O Gatilho Oculto:
       - A simulação mostra que o Potencial de Membrana Médio (V̄) atingiu 95%
         do limiar de disparo (V_th).
       - A plasticidade STDP formou uma "trilha forte" apontando para cima
         (sinapses fortes conectando preços atuais a preços mais altos).

    3. SINAL (O Pré-Disparo):
       - O Coeficiente de Avalanche κ cruza 1.0.
       - Ação: O mercado está em um estado "Pré-Ictal". Qualquer micro-ordem
         agora causará uma convulsão de alta volatilidade.
       - Direção: Siga a direção da assimetria das conexões sinápticas (onde
         a STDP está mais forte). Se o caminho neural para cima é mais condutivo
         ("menor resistência elétrica"), compre.
    """

    def __init__(self,
                 # Parâmetros da rede
                 n_neurons: int = 300,
                 simulation_steps: int = 50,

                 # Limiares
                 potential_threshold_ratio: float = 0.90,
                 kappa_critical: float = 0.95,

                 # Geral
                 min_data_points: int = 30):
        """
        Inicializa o NS-PPS
        """
        self.n_neurons = n_neurons
        self.simulation_steps = simulation_steps
        self.potential_threshold_ratio = potential_threshold_ratio
        self.kappa_critical = kappa_critical
        self.min_data_points = min_data_points

        # Simulador
        self.simulator = SpikingNetworkSimulator(
            n_neurons=n_neurons,
            simulation_steps=simulation_steps
        )

    def analyze(self,
               prices: np.ndarray,
               volumes: np.ndarray = None) -> dict:
        """
        Analisa dados de mercado e retorna resultado

        Returns:
            dict com signal, confidence, e métricas neurais
        """
        n = len(prices)

        # Validação
        if n < self.min_data_points:
            return self._empty_result("INSUFFICIENT_DATA")

        # Prepara dados
        if volumes is None:
            volumes = np.abs(np.diff(prices)) * 10000 + 1000
            volumes = np.append(volumes, volumes[-1])

        price_changes = np.diff(prices)
        price_changes = np.append(price_changes, 0)

        # PASSO 1: CODIFICAÇÃO SENSORIAL
        total_spikes = 0
        current_time = 0.0

        for i in range(min(n, self.simulation_steps)):
            input_currents = self.simulator.spike_encoder.encode_tick(
                price_change=price_changes[i],
                volume=volumes[i],
                n_neurons=self.n_neurons,
                price_levels=self.simulator.price_levels
            )

            spikes, n_fired = self.simulator.simulate_step(input_currents, current_time)
            total_spikes += len(spikes)
            self.simulator.avalanche_detector.update(n_fired)
            current_time += self.simulator.dt

        # PASSO 2: ESTADO DA REDE
        network_state = self.simulator.get_network_state()

        V_rest = self.simulator.lif_model.E_rest
        V_th = self.simulator.lif_model.V_threshold
        potential_ratio = (network_state.mean_potential - V_rest) / (V_th - V_rest)

        # PASSO 3: SINCRONIA DE FASE
        potentials = np.array([n.membrane_potential for n in self.simulator.neurons])
        synchrony, is_sync = self.simulator.synchrony_detector.detect_synchrony(potentials)

        # PASSO 4: ANÁLISE DE CAMINHOS STDP
        up_pathway, down_pathway = self.simulator.pathway_analyzer.analyze_pathways(
            self.simulator.weights_dense,
            self.n_neurons
        )
        asymmetry = self.simulator.pathway_analyzer.compute_asymmetry(up_pathway, down_pathway)

        # PASSO 5: CLASSIFICAÇÃO DO ESTADO
        cortical_state = network_state.cortical_state
        excitability_state = network_state.excitability_state

        is_pre_ictal = (
            potential_ratio > self.potential_threshold_ratio and
            network_state.avalanche_coefficient > self.kappa_critical
        )

        if is_pre_ictal:
            cortical_state = CorticalState.PRE_ICTAL

        # PASSO 6: GERAÇÃO DE SINAL
        signal = 0
        signal_name = "NEUTRAL"
        confidence = 0.0
        reasons = []

        n_active = np.sum(potentials > V_th - 15)

        # CONDIÇÃO 1: SUBCRÍTICO
        if cortical_state == CorticalState.SUBCRITICAL:
            signal_name = "WAIT"
            reasons.append(f"SUBCRITICO: k={network_state.avalanche_coefficient:.3f}")

        # CONDIÇÃO 2: PRÉ-ICTAL
        elif cortical_state == CorticalState.PRE_ICTAL:
            if asymmetry > 0.1:
                signal = 1  # LONG
                signal_name = "LONG"
                confidence = min(1.0, potential_ratio + abs(asymmetry))
                reasons.append(f"PRE-ICTAL: V={potential_ratio:.0%}")
                reasons.append(f"STDP UP (+{asymmetry:.2f})")
            elif asymmetry < -0.1:
                signal = -1  # SHORT
                signal_name = "SHORT"
                confidence = min(1.0, potential_ratio + abs(asymmetry))
                reasons.append(f"PRE-ICTAL: V={potential_ratio:.0%}")
                reasons.append(f"STDP DOWN ({asymmetry:.2f})")
            else:
                signal_name = "WAIT"
                reasons.append(f"PRE-ICTAL direcao incerta")

        # CONDIÇÃO 3: CRÍTICO
        elif cortical_state == CorticalState.CRITICAL:
            if excitability_state == ExcitabilityState.HYPEREXCITABLE:
                if asymmetry > 0.15:
                    signal = 1
                    signal_name = "LONG"
                    confidence = min(0.8, potential_ratio)
                    reasons.append("CRITICO hiperexcitavel")
                    reasons.append("STDP UP")
                elif asymmetry < -0.15:
                    signal = -1
                    signal_name = "SHORT"
                    confidence = min(0.8, potential_ratio)
                    reasons.append("CRITICO hiperexcitavel")
                    reasons.append("STDP DOWN")
                else:
                    signal_name = "WAIT"
                    reasons.append("CRITICO aguardando direcao")
            else:
                signal_name = "WAIT"
                reasons.append("CRITICO acumulando")

        # CONDIÇÃO 4: SUPERCRÍTICO
        elif cortical_state == CorticalState.SUPERCRITICAL:
            signal_name = "NEUTRAL"
            reasons.append("SUPERCRITICO - tarde")

        # CONDIÇÃO 5: Building
        elif excitability_state == ExcitabilityState.BUILDING:
            signal_name = "WAIT"
            reasons.append(f"BUILDING: V={potential_ratio:.0%}")

        else:
            reasons.append(f"Estado={cortical_state.value}")

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'cortical_state': cortical_state.value,
            'excitability_state': excitability_state.value,
            'mean_potential': network_state.mean_potential,
            'potential_ratio': potential_ratio,
            'avalanche_coefficient': network_state.avalanche_coefficient,
            'is_critical': network_state.avalanche_coefficient > 0.95,
            'is_pre_ictal': is_pre_ictal,
            'synchrony_index': synchrony,
            'is_synchronized': is_sync,
            'stdp_up_strength': up_pathway.strength,
            'stdp_down_strength': down_pathway.strength,
            'stdp_asymmetry': asymmetry,
            'total_spikes': total_spikes,
            'firing_rate': network_state.firing_rate,
            'n_neurons': self.n_neurons,
            'n_active': n_active,
            'reasons': reasons
        }

    def _empty_result(self, signal_name: str) -> dict:
        """Retorna resultado vazio"""
        return {
            'signal': 0,
            'signal_name': signal_name,
            'confidence': 0.0,
            'cortical_state': CorticalState.SUBCRITICAL.value,
            'excitability_state': ExcitabilityState.QUIESCENT.value,
            'mean_potential': 0.0,
            'potential_ratio': 0.0,
            'avalanche_coefficient': 0.0,
            'is_critical': False,
            'is_pre_ictal': False,
            'synchrony_index': 0.0,
            'is_synchronized': False,
            'stdp_up_strength': 0.0,
            'stdp_down_strength': 0.0,
            'stdp_asymmetry': 0.0,
            'total_spikes': 0,
            'firing_rate': 0.0,
            'n_neurons': 0,
            'n_active': 0,
            'reasons': [signal_name]
        }

    def reset(self):
        """Reseta o indicador"""
        self.simulator.reset()


# ==============================================================================
# DEMONSTRAÇÃO
# ==============================================================================

def generate_low_vol_data(n_points: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Gera dados de baixa volatilidade (acumulação silenciosa)"""
    np.random.seed(seed)

    # Preço quase estático
    base = 1.0850

    # Pequena tendência de alta escondida
    trend = np.linspace(0, 0.0002, n_points)

    # Ruído mínimo
    noise = np.random.randn(n_points) * 0.00003

    prices = base + trend + noise

    # Volume crescente (acumulação)
    volumes = 1000 + np.linspace(0, 500, n_points) + np.random.randn(n_points) * 50

    return prices, volumes


def main():
    """Demonstração do indicador NS-PPS"""
    print("=" * 70)
    print("NEUROMORPHIC SPIKING SYNAPTIC PRE-POTENTIATION SCANNER (NS-PPS)")
    print("Indicador baseado em Redes Neurais de Espiking")
    print("=" * 70)
    print()

    # Inicializa indicador
    indicator = NeuromorphicSpikingPrePotentiationScanner(
        n_neurons=200,
        simulation_steps=40,
        potential_threshold_ratio=0.85,
        kappa_critical=0.9,
        min_data_points=30
    )

    print("Indicador inicializado!")
    print(f"  - Neuronios: 200")
    print(f"  - Passos simulacao: 40")
    print(f"  - Threshold V: 85%")
    print(f"  - kappa critico: 0.9")
    print()

    # Gera dados
    prices, volumes = generate_low_vol_data(n_points=50)
    print(f"Dados gerados: {len(prices)} pontos")
    print()

    # Processa
    result = indicator.analyze(prices, volumes)

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Sinal: {result['signal_name']}")
    print(f"Estado Cortical: {result['cortical_state']}")
    print(f"Excitabilidade: {result['excitability_state']}")
    print(f"Confianca: {result['confidence']:.2%}")
    print(f"\nPotencial:")
    print(f"  V medio: {result['mean_potential']:.2f} mV")
    print(f"  V/V_th: {result['potential_ratio']:.2%}")
    print(f"\nAvalanche:")
    print(f"  kappa: {result['avalanche_coefficient']:.4f}")
    print(f"  Critico: {result['is_critical']}")
    print(f"\nSincronia (Kuramoto): {result['synchrony_index']:.4f}")
    print(f"\nSTDP:")
    print(f"  UP strength: {result['stdp_up_strength']:.4f}")
    print(f"  DOWN strength: {result['stdp_down_strength']:.4f}")
    print(f"  Assimetria: {result['stdp_asymmetry']:.4f}")
    print(f"\nSpikes:")
    print(f"  Total: {result['total_spikes']}")
    print(f"  Taxa: {result['firing_rate']:.2f} Hz")
    print(f"\nRazoes: {result['reasons']}")

    return indicator, result


if __name__ == "__main__":
    indicator, result = main()
