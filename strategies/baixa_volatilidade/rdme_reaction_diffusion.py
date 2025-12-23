"""
================================================================================
TURING-GRAY-SCOTT REACTION-DIFFUSION MORPHOGENESIS ENGINE (RD-ME)
Indicador de Forex baseado em Química de Reação-Difusão
================================================================================

Este indicador modela o Order Book como um reator químico onde ocorrem reações
autocatalíticas. Ele utiliza o Modelo de Gray-Scott para detectar a nucleação
de padrões de tendência antes que eles sejam visíveis no preço. Ele rastreia a
concentração de "Fluxo Tóxico" (Informed Trading) e prevê onde a "mancha" de
volatilidade vai aparecer.

A Química: Autocatálise e VPIN
Vamos definir as espécies químicas que reagem no seu reator (o Mercado):

1. Substrato U (Liquidez de Varejo/Noise): É o "alimento". Abundante, difunde-se
   lentamente (ordens limitadas paradas).

2. Reagente V (Fluxo Tóxico/Informed): É o "predador". Raro, difunde-se rapidamente
   (HFTs, agressão), e é autocatalítico (quando um banco agride, outros robôs
   seguem e agridem também): 2V + U → 3V

Por que usar Química de Reação-Difusão?
1. Explica o "Flash Crash": Movimentos violentos em baixa liquidez são reações
   em cadeia. O modelo uv² (termo não-linear cúbico) captura perfeitamente como
   uma pequena venda tóxica pode desencadear uma avalanche de stops e HFTs.

2. Detecção de Insider: O VPIN isolado é bom, mas o Gray-Scott mostra COMO esse
   VPIN vai interagir com a liquidez disponível. Ele prevê se a ordem tóxica vai
   morrer (ser absorvida pelo u) ou se vai explodir (gerar tendência).

3. Padrões Emergentes: Turing provou que a forma vem da química. O padrão gráfico
   (Bandeira, Triângulo) é apenas a manifestação visual dessa reação subjacente.
   Nós operamos a reação, não o desenho.

Nota de Implementação: A estabilidade numérica do solver PDE é crítica. Use o
esquema ADI (Alternating Direction Implicit) ou Crank-Nicolson para evitar que
a simulação exploda artificialmente.

Autor: Claude AI
Versão: 1.0.0
================================================================================
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import warnings
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


# ==============================================================================
# CONSTANTES DO MODELO GRAY-SCOTT
# ==============================================================================

# Parâmetros clássicos para diferentes padrões (Mapa de Pearson)
GRAY_SCOTT_PATTERNS = {
    'solitons': {'F': 0.030, 'k': 0.060},      # Manchas isoladas
    'spots': {'F': 0.035, 'k': 0.065},          # Manchas que se dividem
    'labyrinth': {'F': 0.042, 'k': 0.063},      # Padrões labirínticos
    'stripes': {'F': 0.040, 'k': 0.060},        # Listras
    'chaos': {'F': 0.026, 'k': 0.051},          # Caos
    'trivial': {'F': 0.010, 'k': 0.045},        # Homogêneo trivial
}

# Limites do diagrama de fases de Pearson
PEARSON_PHASE_DIAGRAM = {
    'turing_unstable_min_F': 0.020,
    'turing_unstable_max_F': 0.060,
    'turing_unstable_min_k': 0.045,
    'turing_unstable_max_k': 0.070,
}


# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

class RDMESignalType(Enum):
    """Tipos de sinais de trading"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    WAIT = "WAIT"


class PhaseState(Enum):
    """Estado no diagrama de fases de Pearson"""
    TRIVIAL_HOMOGENEOUS = "TRIVIAL_HOMOGENEOUS"    # Baixa vol segura
    TURING_UNSTABLE = "TURING_UNSTABLE"             # Borda da instabilidade
    SPOTS = "SPOTS"                                  # Manchas se formando
    LABYRINTH = "LABYRINTH"                         # Padrões labirínticos
    STRIPES = "STRIPES"                             # Listras (tendência)
    CHAOS = "CHAOS"                                  # Regime caótico


class PatternType(Enum):
    """Tipo de padrão de Turing detectado"""
    NONE = "NONE"                                   # Sem padrão
    NUCLEATING = "NUCLEATING"                       # Nucleando
    SPOT = "SPOT"                                   # Mancha isolada
    GROWING_SPOT = "GROWING_SPOT"                   # Mancha crescendo
    WAVE = "WAVE"                                   # Onda de choque
    SPIRAL = "SPIRAL"                               # Espiral (raro)


class ReactionType(Enum):
    """Tipo de reação química dominante"""
    ABSORPTION = "ABSORPTION"                       # V sendo absorvido por U
    AUTOCATALYSIS = "AUTOCATALYSIS"                 # V se multiplicando (2V + U → 3V)
    EQUILIBRIUM = "EQUILIBRIUM"                     # Equilíbrio químico
    EXPLOSIVE = "EXPLOSIVE"                         # Reação em cadeia


@dataclass
class ConcentrationField:
    """Campo de concentrações u (liquidez) e v (toxicidade)"""
    u: np.ndarray                                   # Concentração de substrato (liquidez)
    v: np.ndarray                                   # Concentração de reagente (toxicidade/VPIN)
    Du: float                                       # Coeficiente de difusão de u
    Dv: float                                       # Coeficiente de difusão de v
    F: float                                        # Feed rate
    k: float                                        # Kill rate
    price_levels: np.ndarray                        # Níveis de preço correspondentes
    time_steps: int                                 # Passos temporais


@dataclass
class VPINResult:
    """Resultado do cálculo de VPIN"""
    vpin: float                                     # VPIN atual (0-1)
    vpin_series: np.ndarray                         # Série temporal de VPIN
    buy_volume: float                               # Volume de compra estimado
    sell_volume: float                              # Volume de venda estimado
    toxicity_level: str                             # "LOW", "MEDIUM", "HIGH", "EXTREME"
    bucket_imbalances: np.ndarray                   # Desequilíbrios por bucket


@dataclass
class TuringPattern:
    """Padrão de Turing detectado"""
    pattern_type: PatternType
    location: int                                   # Índice no grid de preço
    amplitude: float                                # Amplitude do padrão
    wavelength: float                               # Comprimento de onda
    growth_rate: float                              # Taxa de crescimento
    direction: str                                  # "UP" ou "DOWN"
    is_critical: bool                               # True se no ponto crítico


@dataclass
class HopfBifurcation:
    """Análise de bifurcação de Hopf"""
    is_at_bifurcation: bool                         # True se no ponto de bifurcação
    eigenvalue_real: float                          # Parte real do autovalor crítico
    eigenvalue_imag: float                          # Parte imaginária
    bifurcation_parameter: float                    # Parâmetro de bifurcação (F ou k)
    stability: str                                  # "STABLE", "MARGINAL", "UNSTABLE"
    oscillation_period: float                       # Período de oscilação esperado


@dataclass
class ChemicalReactionAnalysis:
    """Análise da reação química"""
    reaction_type: ReactionType
    reaction_rate: float                            # Taxa da reação uv²
    consumption_rate: float                         # Taxa de consumo de u
    production_rate: float                          # Taxa de produção de v
    equilibrium_u: float                            # Concentração de equilíbrio de u
    equilibrium_v: float                            # Concentração de equilíbrio de v
    is_explosive: bool                              # True se reação em cadeia


@dataclass
class RDMESignal:
    """Sinal completo do RD-ME"""
    signal_type: RDMESignalType
    phase_state: PhaseState
    pattern_type: PatternType
    reaction_type: ReactionType
    confidence: float

    # Concentrações
    u_mean: float                                   # Liquidez média
    v_mean: float                                   # Toxicidade média
    u_gradient: float                               # Gradiente de liquidez
    v_gradient: float                               # Gradiente de toxicidade

    # Parâmetros Gray-Scott
    F: float                                        # Feed rate
    k: float                                        # Kill rate
    Du: float                                       # Difusão de u
    Dv: float                                       # Difusão de v

    # VPIN
    vpin: float                                     # VPIN atual
    toxicity_level: str                             # Nível de toxicidade

    # Padrão de Turing
    pattern_location: int                           # Onde o padrão está
    pattern_amplitude: float                        # Amplitude
    pattern_growth_rate: float                      # Taxa de crescimento

    # Bifurcação
    at_turing_instability: bool                     # Na borda da instabilidade
    hopf_stability: str                             # Estabilidade de Hopf

    # Reação química
    reaction_rate: float                            # Taxa uv²
    is_chain_reaction: bool                         # Reação em cadeia?

    # Trading
    target_price_idx: int                           # Índice do preço alvo
    entry_price: float
    stop_loss: float
    take_profit: float

    reason: str
    timestamp: str


# ==============================================================================
# CALCULADOR DE VPIN
# ==============================================================================

class VPINCalculator:
    """
    2. VPIN (Volume-Synchronized Probability of Informed Trading)

    Para alimentar a variável v (Toxicidade), você não pode usar volume bruto.
    Você deve implementar o algoritmo VPIN (Easley, López de Prado et al.).
    O VPIN mede o desequilíbrio de volume em "baldes de volume" (não de tempo)
    para estimar a probabilidade de que um trader com informação privilegiada
    esteja atuando.

    Alta concentração de v = Alta Toxicidade (Informed Trader presente).
    """

    def __init__(self,
                 bucket_size: float = 0.01,
                 n_buckets: int = 50,
                 volume_classification: str = "bulk"):
        """
        Args:
            bucket_size: Tamanho de cada bucket como fração do volume diário
            n_buckets: Número de buckets para calcular VPIN
            volume_classification: "bulk" (Bulk Volume Classification) ou "tick"
        """
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets
        self.volume_classification = volume_classification

    def calculate_vpin(self,
                      prices: np.ndarray,
                      volumes: np.ndarray) -> VPINResult:
        """
        Calcula VPIN usando Bulk Volume Classification

        VPIN = Σ|V_buy - V_sell| / (n × V_bucket)
        """
        n = len(prices)
        if n < 10:
            return VPINResult(
                vpin=0.0,
                vpin_series=np.array([0.0]),
                buy_volume=0.0,
                sell_volume=0.0,
                toxicity_level="LOW",
                bucket_imbalances=np.array([0.0])
            )

        # Volume total
        total_volume = np.sum(volumes)
        bucket_volume = total_volume * self.bucket_size

        if bucket_volume < 1:
            bucket_volume = np.mean(volumes) * 10

        # Classifica volume como compra ou venda usando BVC
        buy_volumes, sell_volumes = self._bulk_volume_classification(prices, volumes)

        # Agrupa em buckets
        n_actual_buckets = min(self.n_buckets, max(1, int(total_volume / bucket_volume)))

        bucket_imbalances = []
        current_buy = 0.0
        current_sell = 0.0
        current_vol = 0.0

        vpin_series = []

        for i in range(n):
            current_buy += buy_volumes[i]
            current_sell += sell_volumes[i]
            current_vol += volumes[i]

            # Bucket completo?
            if current_vol >= bucket_volume:
                imbalance = abs(current_buy - current_sell) / (current_vol + 1e-10)
                bucket_imbalances.append(imbalance)

                # VPIN parcial
                if len(bucket_imbalances) >= 5:
                    vpin_partial = np.mean(bucket_imbalances[-self.n_buckets:])
                    vpin_series.append(vpin_partial)

                current_buy = 0.0
                current_sell = 0.0
                current_vol = 0.0

        # Último bucket parcial
        if current_vol > 0:
            imbalance = abs(current_buy - current_sell) / (current_vol + 1e-10)
            bucket_imbalances.append(imbalance)

        # VPIN final
        bucket_imbalances = np.array(bucket_imbalances) if bucket_imbalances else np.array([0.0])
        vpin = np.mean(bucket_imbalances[-self.n_buckets:]) if len(bucket_imbalances) > 0 else 0.0

        # Classifica nível de toxicidade
        if vpin < 0.3:
            toxicity_level = "LOW"
        elif vpin < 0.5:
            toxicity_level = "MEDIUM"
        elif vpin < 0.7:
            toxicity_level = "HIGH"
        else:
            toxicity_level = "EXTREME"

        vpin_series = np.array(vpin_series) if vpin_series else np.array([vpin])

        return VPINResult(
            vpin=vpin,
            vpin_series=vpin_series,
            buy_volume=np.sum(buy_volumes),
            sell_volume=np.sum(sell_volumes),
            toxicity_level=toxicity_level,
            bucket_imbalances=bucket_imbalances
        )

    def _bulk_volume_classification(self,
                                   prices: np.ndarray,
                                   volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bulk Volume Classification (BVC)

        Classifica volume como compra ou venda baseado na posição do preço
        dentro da barra (aproximação quando tick data não está disponível)
        """
        from scipy.stats import norm

        n = len(prices)
        buy_volumes = np.zeros(n)
        sell_volumes = np.zeros(n)

        for i in range(1, n):
            # Retorno normalizado
            ret = (prices[i] - prices[i-1]) / (prices[i-1] + 1e-10)

            # Estimativa da volatilidade local
            if i >= 10:
                local_vol = np.std(np.diff(np.log(prices[max(0, i-10):i] + 1e-10)))
            else:
                local_vol = 0.001

            # Z-score do retorno
            z = ret / (local_vol + 1e-10)

            # Probabilidade de compra usando CDF normal
            prob_buy = norm.cdf(z)

            buy_volumes[i] = volumes[i] * prob_buy
            sell_volumes[i] = volumes[i] * (1 - prob_buy)

        return buy_volumes, sell_volumes


# ==============================================================================
# CONSTRUTOR DO CAMPO DE CONCENTRAÇÃO
# ==============================================================================

class ConcentrationFieldBuilder:
    """
    Passo 1: Mapeamento da Concentração Inicial

    Crie um grid 1D (Preço) ou 2D (Preço x Tempo). Preencha o grid com a
    densidade de ordens limitadas (u) e injete o VPIN calculado (v) como
    sementes de perturbação.
    """

    def __init__(self,
                 n_price_levels: int = 100,
                 Du: float = 0.16,
                 Dv: float = 0.08,
                 default_F: float = 0.035,
                 default_k: float = 0.065):
        """
        Args:
            n_price_levels: Número de níveis de preço no grid
            Du: Coeficiente de difusão do substrato (liquidez)
            Dv: Coeficiente de difusão do reagente (toxicidade)
            default_F: Feed rate padrão
            default_k: Kill rate padrão

        Nota: Para instabilidade de Turing, é necessário que Dv << Du
        (a toxicidade se concentra localmente enquanto a liquidez é global)
        """
        self.n_price_levels = n_price_levels
        self.Du = Du
        self.Dv = Dv
        self.default_F = default_F
        self.default_k = default_k

    def build_from_market_data(self,
                              prices: np.ndarray,
                              volumes: np.ndarray,
                              vpin_result: VPINResult) -> ConcentrationField:
        """
        Constrói o campo de concentração a partir de dados de mercado
        """
        n = len(prices)

        # Define níveis de preço
        price_min = np.min(prices) * 0.995
        price_max = np.max(prices) * 1.005
        price_levels = np.linspace(price_min, price_max, self.n_price_levels)

        # Inicializa campos com equilíbrio Gray-Scott padrão
        # u = 1, v = 0 é o equilíbrio trivial
        # Mas queremos começar perto do equilíbrio não-trivial
        u = np.ones(self.n_price_levels) * 0.5  # Liquidez parcial
        v = np.ones(self.n_price_levels) * 0.25  # Toxicidade base

        # Mapeia volumes para níveis de preço (modula u)
        volume_profile = np.zeros(self.n_price_levels)
        for i in range(n):
            idx = np.argmin(np.abs(price_levels - prices[i]))
            volume_profile[idx] += volumes[i]

        # Normaliza e aplica ao u (mais volume = mais liquidez disponível)
        if np.max(volume_profile) > 0:
            volume_profile = volume_profile / np.max(volume_profile)
            u = 0.3 + 0.7 * volume_profile  # u entre 0.3 e 1.0

        # Injeta VPIN como sementes de v (toxicidade)
        current_price_idx = np.argmin(np.abs(price_levels - prices[-1]))

        # Cria perturbação localizada baseada no VPIN
        vpin = vpin_result.vpin

        # Semente de toxicidade centrada no preço atual
        sigma = self.n_price_levels // 8
        for i in range(self.n_price_levels):
            dist = abs(i - current_price_idx)
            # Perturbação na direção do flow
            if vpin_result.buy_volume > vpin_result.sell_volume:
                # Toxicidade acima do preço (buy pressure)
                if i > current_price_idx:
                    v[i] += vpin * 0.3 * np.exp(-dist**2 / (2 * sigma**2))
            else:
                # Toxicidade abaixo do preço (sell pressure)
                if i < current_price_idx:
                    v[i] += vpin * 0.3 * np.exp(-dist**2 / (2 * sigma**2))

        # Adiciona ruído para quebrar simetria (importante para padrões de Turing!)
        v += np.random.uniform(0, 0.05, self.n_price_levels)

        # Garante valores no range apropriado
        u = np.clip(u, 0.1, 1.0)
        v = np.clip(v, 0.0, 0.5)

        # Calcula parâmetros F e k baseados nas condições de mercado
        F, k = self._estimate_parameters(prices, volumes, vpin_result)

        return ConcentrationField(
            u=u,
            v=v,
            Du=self.Du,
            Dv=self.Dv,
            F=F,
            k=k,
            price_levels=price_levels,
            time_steps=0
        )

    def _estimate_parameters(self,
                            prices: np.ndarray,
                            volumes: np.ndarray,
                            vpin_result: VPINResult) -> Tuple[float, float]:
        """
        Estima F (Feed Rate) e k (Kill Rate) das condições de mercado

        F: Taxa de chegada de novas ordens limitadas
        k: Taxa de execução/consumo de ordens
        """
        # F proporcional à taxa de chegada de volume
        volume_rate = np.mean(np.diff(np.cumsum(volumes)))
        F = self.default_F * (1 + volume_rate / (np.mean(volumes) + 1e-10) * 0.3)

        # k proporcional à taxa de execução (proxy: volatilidade)
        returns = np.diff(np.log(prices + 1e-10))
        volatility = np.std(returns) * np.sqrt(252)
        k = self.default_k * (1 + volatility * 0.5)

        # Ajusta para condições de VPIN - mantém na região de Turing
        if vpin_result.toxicity_level == "EXTREME":
            # Move para região de spots/stripes
            F = max(0.035, min(0.045, F))
            k = max(0.058, min(0.068, k * 0.9))
        elif vpin_result.toxicity_level == "HIGH":
            F = max(0.032, min(0.042, F))
            k = max(0.060, min(0.068, k * 0.95))
        elif vpin_result.toxicity_level == "MEDIUM":
            F = max(0.030, min(0.040, F))
            k = max(0.062, min(0.070, k))

        # Garante que estamos na região de instabilidade quando há toxicidade
        F = np.clip(F, 0.025, 0.055)
        k = np.clip(k, 0.050, 0.072)

        return F, k


# ==============================================================================
# SOLVER NUMÉRICO (GRAY-SCOTT)
# ==============================================================================

class GrayScottSolver:
    """
    Passo 2: Solver Numérico (Método de Diferenças Finitas)

    Resolva as equações de Gray-Scott iterativamente. Utilize um operador
    Laplaciano discreto para o termo de difusão ∇².

    Equações:
    ∂u/∂t = Du∇²u - uv² + F(1-u)
    ∂v/∂t = Dv∇²v + uv² - (F+k)v

    Nota de Implementação: A estabilidade numérica do solver PDE é crítica.
    Use o esquema ADI (Alternating Direction Implicit) ou Crank-Nicolson para
    evitar que a simulação exploda artificialmente.
    """

    def __init__(self,
                 dx: float = 1.0,
                 dt: float = 0.5,  # Passo temporal menor para estabilidade
                 method: str = "crank_nicolson"):
        """
        Args:
            dx: Espaçamento espacial
            dt: Passo temporal
            method: "euler", "crank_nicolson", ou "adi"
        """
        self.dx = dx
        self.dt = dt
        self.method = method

    def laplacian_1d(self, field: np.ndarray) -> np.ndarray:
        """
        Operador Laplaciano discreto 1D

        ∇²f = (f[i+1] - 2f[i] + f[i-1]) / dx²

        Com condições de contorno periódicas
        """
        n = len(field)
        lap = np.zeros(n)

        for i in range(n):
            ip1 = (i + 1) % n  # Índice i+1 com wrapping
            im1 = (i - 1) % n  # Índice i-1 com wrapping

            lap[i] = (field[ip1] - 2 * field[i] + field[im1]) / (self.dx ** 2)

        return lap

    def reaction_term_u(self, u: np.ndarray, v: np.ndarray, F: float) -> np.ndarray:
        """
        Termo de reação para u: -uv² + F(1-u)

        - uv²: Consumo de u pela reação autocatalítica
        - F(1-u): Alimentação de u (feed)
        """
        return -u * v**2 + F * (1 - u)

    def reaction_term_v(self, u: np.ndarray, v: np.ndarray, F: float, k: float) -> np.ndarray:
        """
        Termo de reação para v: uv² - (F+k)v

        - uv²: Produção autocatalítica de v (2V + U → 3V)
        - (F+k)v: Remoção de v (kill + wash-out)
        """
        return u * v**2 - (F + k) * v

    def step_euler(self,
                  u: np.ndarray,
                  v: np.ndarray,
                  Du: float,
                  Dv: float,
                  F: float,
                  k: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Passo temporal usando Euler explícito

        Simples mas pode ser instável para dt grande
        """
        # Difusão
        lap_u = self.laplacian_1d(u)
        lap_v = self.laplacian_1d(v)

        # Reação
        react_u = self.reaction_term_u(u, v, F)
        react_v = self.reaction_term_v(u, v, F, k)

        # Atualização
        u_new = u + self.dt * (Du * lap_u + react_u)
        v_new = v + self.dt * (Dv * lap_v + react_v)

        # Garante valores positivos
        u_new = np.clip(u_new, 0, 1)
        v_new = np.clip(v_new, 0, 1)

        return u_new, v_new

    def step_crank_nicolson(self,
                           u: np.ndarray,
                           v: np.ndarray,
                           Du: float,
                           Dv: float,
                           F: float,
                           k: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Passo temporal usando Crank-Nicolson

        Implícito, mais estável que Euler
        """
        n = len(u)

        # Lado direito (explícito)
        lap_u = self.laplacian_1d(u)
        lap_v = self.laplacian_1d(v)

        react_u = self.reaction_term_u(u, v, F)
        react_v = self.reaction_term_v(u, v, F, k)

        # Semi-implícito: difusão meio implícita, reação explícita
        rhs_u = u + self.dt * (0.5 * Du * lap_u + react_u)
        rhs_v = v + self.dt * (0.5 * Dv * lap_v + react_v)

        # Iteração de ponto fixo para parte implícita
        u_new = rhs_u.copy()
        v_new = rhs_v.copy()

        for _ in range(5):  # Mais iterações para convergência
            lap_u_new = self.laplacian_1d(u_new)
            lap_v_new = self.laplacian_1d(v_new)

            u_new = rhs_u + 0.5 * self.dt * Du * lap_u_new
            v_new = rhs_v + 0.5 * self.dt * Dv * lap_v_new

            # Clipa durante iteração para estabilidade
            u_new = np.clip(u_new, 0, 1)
            v_new = np.clip(v_new, 0, 1)

        return u_new, v_new

    def solve(self,
             concentration: ConcentrationField,
             n_steps: int = 1000) -> Tuple[ConcentrationField, List[np.ndarray], List[np.ndarray]]:
        """
        Resolve as equações de Gray-Scott por n_steps passos

        Returns:
            (campo final, histórico de u, histórico de v)
        """
        u = concentration.u.copy()
        v = concentration.v.copy()
        Du = concentration.Du
        Dv = concentration.Dv
        F = concentration.F
        k = concentration.k

        u_history = [u.copy()]
        v_history = [v.copy()]

        # Seleciona método
        if self.method == "crank_nicolson":
            step_func = self.step_crank_nicolson
        else:
            step_func = self.step_euler

        for step in range(n_steps):
            u, v = step_func(u, v, Du, Dv, F, k)

            # Armazena histórico a cada 50 passos
            if step % 50 == 0:
                u_history.append(u.copy())
                v_history.append(v.copy())

        # Campo final
        final_field = ConcentrationField(
            u=u,
            v=v,
            Du=Du,
            Dv=Dv,
            F=F,
            k=k,
            price_levels=concentration.price_levels,
            time_steps=n_steps
        )

        return final_field, u_history, v_history


# ==============================================================================
# DIAGRAMA DE FASES DE PEARSON
# ==============================================================================

class PearsonPhaseDiagram:
    """
    Passo 3: Diagrama de Fases (O Mapa de Pearson)

    Monitore os parâmetros F e k do mercado atual e plote-os no Diagrama de
    Fases de Pearson.

    - O mercado pode estar na fase de "Solitons", "Labirinto", "Manchas" ou "Caos"
    - Em baixa volatilidade segura, o sistema está na região "Homogênea Trivial"
    """

    def __init__(self):
        # Regiões do diagrama de fases
        self.regions = GRAY_SCOTT_PATTERNS
        self.boundaries = PEARSON_PHASE_DIAGRAM

    def classify_phase(self, F: float, k: float) -> PhaseState:
        """
        Classifica o estado atual no diagrama de fases
        """
        # Região trivial homogênea (baixa vol segura)
        if F < 0.02 or k < 0.04:
            return PhaseState.TRIVIAL_HOMOGENEOUS

        # Região caótica
        if F < 0.03 and k > 0.045 and k < 0.055:
            return PhaseState.CHAOS

        # Região de spots
        if 0.03 <= F <= 0.04 and 0.06 <= k <= 0.07:
            return PhaseState.SPOTS

        # Região de labirinto
        if 0.04 <= F <= 0.05 and 0.06 <= k <= 0.065:
            return PhaseState.LABYRINTH

        # Região de stripes
        if 0.035 <= F <= 0.045 and 0.055 <= k <= 0.065:
            return PhaseState.STRIPES

        # Borda da instabilidade de Turing
        if (self.boundaries['turing_unstable_min_F'] <= F <= self.boundaries['turing_unstable_max_F'] and
            self.boundaries['turing_unstable_min_k'] <= k <= self.boundaries['turing_unstable_max_k']):
            return PhaseState.TURING_UNSTABLE

        return PhaseState.TRIVIAL_HOMOGENEOUS

    def distance_to_instability(self, F: float, k: float) -> float:
        """
        Calcula distância até a borda da instabilidade de Turing
        """
        # Centro aproximado da região de instabilidade
        F_center = 0.04
        k_center = 0.06

        # Distância normalizada
        dF = (F - F_center) / 0.02
        dk = (k - k_center) / 0.01

        distance = np.sqrt(dF**2 + dk**2)

        return distance

    def get_expected_pattern(self, F: float, k: float) -> str:
        """
        Retorna o padrão esperado para os parâmetros dados
        """
        phase = self.classify_phase(F, k)

        pattern_map = {
            PhaseState.TRIVIAL_HOMOGENEOUS: "Sem padrão (estável)",
            PhaseState.TURING_UNSTABLE: "Padrões de Turing emergindo",
            PhaseState.SPOTS: "Manchas (spots) se dividindo",
            PhaseState.LABYRINTH: "Padrões labirínticos",
            PhaseState.STRIPES: "Listras (tendência direcional)",
            PhaseState.CHAOS: "Regime caótico",
        }

        return pattern_map.get(phase, "Indefinido")


# ==============================================================================
# DETECTOR DE PADRÕES DE TURING
# ==============================================================================

class TuringPatternDetector:
    """
    A Lógica de Trading (Bifurcação de Hopf)
    O indicador busca a Quebra de Simetria Espontânea.

    1. O Estado Crítico: O mercado está parado, mas o indicador mostra que os
       parâmetros F (Feed) e k (Kill) moveram o sistema para a Borda da
       Instabilidade de Turing. Isso significa que o estado homogêneo (preço
       parado) tornou-se matematicamente instável. Qualquer flutuação microscópica
       vai crescer exponencialmente.

    2. O Gatilho (Spot Formation): A simulação mostra a formação súbita de um
       "Padrão de Turing" (uma concentração alta de v) em um nível de preço
       específico P_target. A reação autocatalítica (uv²) começou. O fluxo tóxico
       está consumindo a liquidez de varejo em uma reação em cadeia.

    3. SINAL:
       - Direção: Onde a concentração de v está crescendo? (Acima do preço =
         Buy Side Toxicity; Abaixo = Sell Side Toxicity)
       - Ação: O preço vai ser "sugado" para onde a reação química está mais forte.
         Entre a favor da Toxicidade. Você está surfando a onda de choque química.
    """

    def __init__(self,
                 pattern_threshold: float = 0.1,
                 growth_threshold: float = 0.05):
        """
        Args:
            pattern_threshold: Limiar para detectar padrão
            growth_threshold: Limiar para crescimento significativo
        """
        self.pattern_threshold = pattern_threshold
        self.growth_threshold = growth_threshold

    def detect_patterns(self,
                       v_current: np.ndarray,
                       v_previous: np.ndarray,
                       price_levels: np.ndarray,
                       current_price: float) -> List[TuringPattern]:
        """
        Detecta padrões de Turing no campo de concentração
        """
        n = len(v_current)
        patterns = []

        # Encontra índice do preço atual
        current_idx = np.argmin(np.abs(price_levels - current_price))

        # Detecta picos de concentração
        mean_v = np.mean(v_current)
        std_v = np.std(v_current)

        threshold = mean_v + 2 * std_v

        for i in range(1, n - 1):
            # É um pico local?
            if v_current[i] > v_current[i-1] and v_current[i] > v_current[i+1]:
                if v_current[i] > threshold:
                    # Calcula amplitude
                    amplitude = v_current[i] - mean_v

                    # Calcula taxa de crescimento
                    if len(v_previous) > 0:
                        growth_rate = (v_current[i] - v_previous[i]) / (v_previous[i] + 1e-10)
                    else:
                        growth_rate = 0.0

                    # Direção relativa ao preço atual
                    if i < current_idx:
                        direction = "DOWN"
                    elif i > current_idx:
                        direction = "UP"
                    else:
                        direction = "AT_PRICE"

                    # Estima comprimento de onda
                    wavelength = self._estimate_wavelength(v_current, i)

                    # Determina tipo de padrão
                    if amplitude < self.pattern_threshold:
                        pattern_type = PatternType.NUCLEATING
                    elif growth_rate > self.growth_threshold:
                        pattern_type = PatternType.GROWING_SPOT
                    elif wavelength < n / 5:
                        pattern_type = PatternType.WAVE
                    else:
                        pattern_type = PatternType.SPOT

                    # É crítico? (Alta amplitude + alto crescimento)
                    is_critical = amplitude > 0.2 and growth_rate > 0.1

                    pattern = TuringPattern(
                        pattern_type=pattern_type,
                        location=i,
                        amplitude=amplitude,
                        wavelength=wavelength,
                        growth_rate=growth_rate,
                        direction=direction,
                        is_critical=is_critical
                    )
                    patterns.append(pattern)

        # Ordena por amplitude
        patterns.sort(key=lambda p: p.amplitude, reverse=True)

        return patterns

    def _estimate_wavelength(self, v: np.ndarray, peak_idx: int) -> float:
        """
        Estima comprimento de onda do padrão
        """
        n = len(v)

        # Encontra próximo pico
        for i in range(peak_idx + 2, n - 1):
            if v[i] > v[i-1] and v[i] > v[i+1]:
                return float(i - peak_idx)

        # Encontra pico anterior
        for i in range(peak_idx - 2, 0, -1):
            if v[i] > v[i-1] and v[i] > v[i+1]:
                return float(peak_idx - i)

        return float(n)  # Sem periodicidade clara

    def analyze_hopf_bifurcation(self,
                                concentration: ConcentrationField) -> HopfBifurcation:
        """
        Analisa estabilidade usando teoria de bifurcação de Hopf

        No ponto de bifurcação, autovalores do sistema linearizado cruzam
        o eixo imaginário
        """
        F = concentration.F
        k = concentration.k
        Du = concentration.Du
        Dv = concentration.Dv

        # Estado de equilíbrio homogêneo
        # u* = 1, v* = 0 é sempre solução
        # Mas também existe u* = (F + k)² / F, v* = F / (F + k)² quando u*v* > 0

        # Para o segundo equilíbrio (não-trivial):
        if F > 0 and k > 0:
            u_eq = (F + k)**2 / F
            v_eq = F / (F + k)**2

            # Jacobiano linearizado
            # J = [[-F - v²,  -2uv], [v², 2uv - F - k]]

            J11 = -F - v_eq**2
            J12 = -2 * u_eq * v_eq
            J21 = v_eq**2
            J22 = 2 * u_eq * v_eq - F - k

            # Autovalores (simplificado para 2x2)
            trace = J11 + J22
            det = J11 * J22 - J12 * J21

            discriminant = trace**2 - 4 * det

            if discriminant >= 0:
                lambda1 = (trace + np.sqrt(discriminant)) / 2
                lambda2 = (trace - np.sqrt(discriminant)) / 2
                eigenvalue_real = max(lambda1, lambda2)
                eigenvalue_imag = 0.0
            else:
                eigenvalue_real = trace / 2
                eigenvalue_imag = np.sqrt(-discriminant) / 2

            # Período de oscilação
            if eigenvalue_imag > 0:
                oscillation_period = 2 * np.pi / eigenvalue_imag
            else:
                oscillation_period = float('inf')

            # Estabilidade
            if eigenvalue_real < -0.01:
                stability = "STABLE"
            elif eigenvalue_real > 0.01:
                stability = "UNSTABLE"
            else:
                stability = "MARGINAL"

            # No ponto de bifurcação?
            is_at_bifurcation = abs(eigenvalue_real) < 0.02 and eigenvalue_imag > 0

        else:
            eigenvalue_real = -1.0
            eigenvalue_imag = 0.0
            stability = "STABLE"
            is_at_bifurcation = False
            oscillation_period = float('inf')

        return HopfBifurcation(
            is_at_bifurcation=is_at_bifurcation,
            eigenvalue_real=eigenvalue_real,
            eigenvalue_imag=eigenvalue_imag,
            bifurcation_parameter=F,
            stability=stability,
            oscillation_period=oscillation_period
        )


# ==============================================================================
# ANALISADOR DE REAÇÃO QUÍMICA
# ==============================================================================

class ChemicalReactionAnalyzer:
    """
    Analisa a dinâmica da reação química no sistema
    """

    def __init__(self):
        pass

    def analyze_reaction(self,
                        concentration: ConcentrationField) -> ChemicalReactionAnalysis:
        """
        Analisa o tipo e taxa da reação química
        """
        u = concentration.u
        v = concentration.v
        F = concentration.F
        k = concentration.k

        # Taxa da reação autocatalítica: uv²
        reaction_rate = np.mean(u * v**2)

        # Taxa de consumo de u: uv² (mesmo termo)
        consumption_rate = reaction_rate

        # Taxa de produção de v: uv² - (F+k)v
        production_rate = np.mean(u * v**2 - (F + k) * v)

        # Equilíbrios
        if F > 0:
            equilibrium_u = (F + k)**2 / F
            equilibrium_v = F / (F + k)**2
        else:
            equilibrium_u = 1.0
            equilibrium_v = 0.0

        # Classifica tipo de reação
        if production_rate > 0.1:
            reaction_type = ReactionType.EXPLOSIVE
            is_explosive = True
        elif production_rate > 0.01:
            reaction_type = ReactionType.AUTOCATALYSIS
            is_explosive = False
        elif production_rate < -0.01:
            reaction_type = ReactionType.ABSORPTION
            is_explosive = False
        else:
            reaction_type = ReactionType.EQUILIBRIUM
            is_explosive = False

        return ChemicalReactionAnalysis(
            reaction_type=reaction_type,
            reaction_rate=reaction_rate,
            consumption_rate=consumption_rate,
            production_rate=production_rate,
            equilibrium_u=equilibrium_u,
            equilibrium_v=equilibrium_v,
            is_explosive=is_explosive
        )


# ==============================================================================
# INDICADOR RD-ME COMPLETO
# ==============================================================================

class TuringGrayScottReactionDiffusionMorphogenesisEngine:
    """
    TURING-GRAY-SCOTT REACTION-DIFFUSION MORPHOGENESIS ENGINE (RD-ME)

    Indicador completo que usa Química de Reação-Difusão para detectar
    nucleação de padrões de tendência antes que eles sejam visíveis no preço.
    """

    def __init__(self,
                 # Parâmetros do grid
                 n_price_levels: int = 100,

                 # Parâmetros de difusão
                 Du: float = 0.16,
                 Dv: float = 0.08,

                 # Parâmetros Gray-Scott padrão
                 default_F: float = 0.035,
                 default_k: float = 0.065,

                 # Parâmetros do solver
                 solver_steps: int = 500,
                 solver_method: str = "crank_nicolson",

                 # Parâmetros de detecção
                 pattern_threshold: float = 0.1,

                 # VPIN
                 vpin_buckets: int = 50,

                 # Geral
                 min_data_points: int = 50):
        """
        Inicializa o RD-ME
        """
        self.n_price_levels = n_price_levels
        self.min_data_points = min_data_points

        # Componentes
        self.vpin_calculator = VPINCalculator(n_buckets=vpin_buckets)

        self.concentration_builder = ConcentrationFieldBuilder(
            n_price_levels=n_price_levels,
            Du=Du,
            Dv=Dv,
            default_F=default_F,
            default_k=default_k
        )

        self.solver = GrayScottSolver(
            dx=1.0,
            dt=1.0,
            method=solver_method
        )
        self.solver_steps = solver_steps

        self.phase_diagram = PearsonPhaseDiagram()

        self.pattern_detector = TuringPatternDetector(
            pattern_threshold=pattern_threshold
        )

        self.reaction_analyzer = ChemicalReactionAnalyzer()

    def analyze(self,
               prices: np.ndarray,
               volumes: np.ndarray = None) -> dict:
        """
        Processa dados de mercado e gera resultado de análise

        Returns:
            Dict com todos os resultados da análise
        """
        from datetime import datetime

        n = len(prices)

        # Validação
        if n < self.min_data_points:
            return self._create_empty_result("INSUFFICIENT_DATA")

        # Volumes sintéticos se não fornecidos
        if volumes is None:
            volumes = np.abs(np.diff(prices))
            volumes = np.append(volumes, volumes[-1])
            volumes = volumes * 10000 + 1000

        current_price = prices[-1]

        # PASSO 1: CÁLCULO DO VPIN
        vpin_result = self.vpin_calculator.calculate_vpin(prices, volumes)

        # PASSO 2: CONSTRUÇÃO DO CAMPO DE CONCENTRAÇÃO
        concentration = self.concentration_builder.build_from_market_data(
            prices, volumes, vpin_result
        )

        # PASSO 3: DIAGRAMA DE FASES DE PEARSON
        phase_state = self.phase_diagram.classify_phase(concentration.F, concentration.k)
        distance_to_instability = self.phase_diagram.distance_to_instability(
            concentration.F, concentration.k
        )

        # PASSO 4: SIMULAÇÃO GRAY-SCOTT
        v_initial = concentration.v.copy()

        final_concentration, u_history, v_history = self.solver.solve(
            concentration, n_steps=self.solver_steps
        )

        # PASSO 5: DETECÇÃO DE PADRÕES DE TURING
        patterns = self.pattern_detector.detect_patterns(
            final_concentration.v,
            v_initial,
            final_concentration.price_levels,
            current_price
        )

        main_pattern = patterns[0] if patterns else None

        # PASSO 6: ANÁLISE DE BIFURCAÇÃO DE HOPF
        hopf = self.pattern_detector.analyze_hopf_bifurcation(final_concentration)

        # PASSO 7: ANÁLISE DA REAÇÃO QUÍMICA
        reaction = self.reaction_analyzer.analyze_reaction(final_concentration)

        # PASSO 8: GERAÇÃO DE SINAL
        signal = 0
        signal_name = "NEUTRAL"
        confidence = 0.0
        reasons = []

        # Preços
        price_levels = final_concentration.price_levels
        current_price_idx = np.argmin(np.abs(price_levels - current_price))

        entry_price = current_price
        stop_loss = current_price
        take_profit = current_price
        target_price_idx = current_price_idx

        # Gradientes
        u_gradient = np.gradient(final_concentration.u)[current_price_idx]
        v_gradient = np.gradient(final_concentration.v)[current_price_idx]

        # Lógica de sinal

        # 1. PADRÃO CRÍTICO DETECTADO
        if main_pattern and main_pattern.is_critical:
            if main_pattern.direction == "UP":
                signal = 1
                signal_name = "LONG"
                target_price_idx = main_pattern.location
                take_profit = price_levels[target_price_idx]
                stop_loss = current_price - (take_profit - current_price) * 0.5
            elif main_pattern.direction == "DOWN":
                signal = -1
                signal_name = "SHORT"
                target_price_idx = main_pattern.location
                take_profit = price_levels[target_price_idx]
                stop_loss = current_price + (current_price - take_profit) * 0.5
            else:
                signal_name = "WAIT"

            confidence = min(0.95, 0.6 + main_pattern.amplitude + main_pattern.growth_rate)
            reasons.append(f"PADRÃO DE TURING CRÍTICO em nível {target_price_idx}")
            reasons.append(f"Amplitude={main_pattern.amplitude:.3f}")

        # 2. REAÇÃO EXPLOSIVA
        elif reaction.is_explosive:
            if v_gradient > 0:
                signal = 1
                signal_name = "LONG"
                stop_loss = current_price * 0.998
                take_profit = current_price * 1.005
            elif v_gradient < 0:
                signal = -1
                signal_name = "SHORT"
                stop_loss = current_price * 1.002
                take_profit = current_price * 0.995
            else:
                signal_name = "WAIT"

            confidence = min(0.85, 0.5 + abs(reaction.production_rate) * 10)
            reasons.append("REAÇÃO EM CADEIA detectada")
            reasons.append(f"Taxa produção={reaction.production_rate:.4f}")

        # 3. FASE DE TURING ATIVA + VPIN alto
        elif phase_state in [PhaseState.LABYRINTH, PhaseState.SPOTS, PhaseState.STRIPES] and vpin_result.vpin > 0.5:
            if vpin_result.buy_volume > vpin_result.sell_volume * 1.2:
                signal = 1
                signal_name = "LONG"
                stop_loss = current_price * 0.997
                take_profit = current_price * 1.008
            elif vpin_result.sell_volume > vpin_result.buy_volume * 1.2:
                signal = -1
                signal_name = "SHORT"
                stop_loss = current_price * 1.003
                take_profit = current_price * 0.992
            else:
                signal_name = "WAIT"

            confidence = min(0.80, 0.4 + vpin_result.vpin * 0.5)
            reasons.append(f"FASE {phase_state.value} + ALTA TOXICIDADE")
            reasons.append(f"VPIN={vpin_result.vpin:.3f}")

        # 4. NA BORDA DA INSTABILIDADE DE TURING
        elif phase_state == PhaseState.TURING_UNSTABLE or hopf.is_at_bifurcation:
            signal_name = "WAIT"
            confidence = 0.6
            reasons.append("NA BORDA DA INSTABILIDADE")
            reasons.append(f"Aguardando nucleação")

        # 5. PADRÃO EM CRESCIMENTO
        elif main_pattern and main_pattern.pattern_type == PatternType.GROWING_SPOT:
            if main_pattern.direction == "UP":
                signal = 1
                signal_name = "LONG"
            elif main_pattern.direction == "DOWN":
                signal = -1
                signal_name = "SHORT"
            else:
                signal_name = "WAIT"

            target_price_idx = main_pattern.location
            take_profit = price_levels[target_price_idx]
            stop_loss = current_price - np.sign(take_profit - current_price) * abs(take_profit - current_price) * 0.3

            confidence = min(0.75, 0.4 + main_pattern.growth_rate * 5)
            reasons.append(f"SPOT EM CRESCIMENTO em nível {main_pattern.location}")
            reasons.append(f"Crescimento={main_pattern.growth_rate:.2%}")

        # 6. TOXICIDADE ALTA mas sem padrão
        elif vpin_result.toxicity_level in ["HIGH", "EXTREME"]:
            signal_name = "WAIT"
            confidence = 0.4
            reasons.append(f"TOXICIDADE {vpin_result.toxicity_level}")
            reasons.append("Aguardando padrão")

        # 7. REGIME CAÓTICO
        elif phase_state == PhaseState.CHAOS:
            signal_name = "NEUTRAL"
            confidence = 0.2
            reasons.append("REGIME CAÓTICO")
            reasons.append("Evitar trades")

        # 8. HOMOGÊNEO TRIVIAL
        else:
            signal_name = "NEUTRAL"
            confidence = 0.1
            reasons.append("ESTADO HOMOGÊNEO TRIVIAL")
            reasons.append("Mercado estável")

        confidence = np.clip(confidence, 0, 1)

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'phase_state': phase_state.value,
            'pattern_type': main_pattern.pattern_type.value if main_pattern else PatternType.NONE.value,
            'reaction_type': reaction.reaction_type.value,
            'u_mean': float(np.mean(final_concentration.u)),
            'v_mean': float(np.mean(final_concentration.v)),
            'u_gradient': float(u_gradient),
            'v_gradient': float(v_gradient),
            'F': concentration.F,
            'k': concentration.k,
            'Du': concentration.Du,
            'Dv': concentration.Dv,
            'vpin': vpin_result.vpin,
            'toxicity_level': vpin_result.toxicity_level,
            'pattern_location': main_pattern.location if main_pattern else -1,
            'pattern_amplitude': main_pattern.amplitude if main_pattern else 0.0,
            'pattern_growth_rate': main_pattern.growth_rate if main_pattern else 0.0,
            'pattern_direction': main_pattern.direction if main_pattern else "NONE",
            'pattern_is_critical': main_pattern.is_critical if main_pattern else False,
            'at_turing_instability': bool(phase_state == PhaseState.TURING_UNSTABLE),
            'hopf_stability': hopf.stability,
            'hopf_eigenvalue_real': hopf.eigenvalue_real,
            'hopf_eigenvalue_imag': hopf.eigenvalue_imag,
            'is_at_bifurcation': hopf.is_at_bifurcation,
            'reaction_rate': reaction.reaction_rate,
            'production_rate': reaction.production_rate,
            'is_chain_reaction': reaction.is_explosive,
            'distance_to_instability': distance_to_instability,
            'target_price_idx': target_price_idx,
            'entry_price': entry_price,
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
            'phase_state': PhaseState.TRIVIAL_HOMOGENEOUS.value,
            'pattern_type': PatternType.NONE.value,
            'reaction_type': ReactionType.EQUILIBRIUM.value,
            'u_mean': 0.0,
            'v_mean': 0.0,
            'u_gradient': 0.0,
            'v_gradient': 0.0,
            'F': 0.035,
            'k': 0.065,
            'Du': 0.16,
            'Dv': 0.08,
            'vpin': 0.0,
            'toxicity_level': "LOW",
            'pattern_location': -1,
            'pattern_amplitude': 0.0,
            'pattern_growth_rate': 0.0,
            'pattern_direction': "NONE",
            'pattern_is_critical': False,
            'at_turing_instability': False,
            'hopf_stability': "STABLE",
            'hopf_eigenvalue_real': 0.0,
            'hopf_eigenvalue_imag': 0.0,
            'is_at_bifurcation': False,
            'reaction_rate': 0.0,
            'production_rate': 0.0,
            'is_chain_reaction': False,
            'distance_to_instability': 1.0,
            'target_price_idx': 0,
            'entry_price': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'reasons': [signal_name]
        }

    def reset(self):
        """Reseta o indicador"""
        pass


# ==============================================================================
# DEMONSTRAÇÃO
# ==============================================================================

def generate_toxic_flow_data(n_points: int = 100,
                            seed: int = 42,
                            with_informed_trader: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera dados que simulam fluxo tóxico (informed trading)
    """
    np.random.seed(seed)

    base_price = 1.0850
    prices = [base_price]
    volumes = [1000]

    # Fase 1: Mercado normal
    normal_phase = int(n_points * 0.5)

    for i in range(1, normal_phase):
        noise = np.random.randn() * 0.00005
        vol = 1000 + np.random.randn() * 200

        prices.append(prices[-1] + noise)
        volumes.append(max(100, vol))

    if with_informed_trader:
        # Fase 2: Informed trader entra
        accumulation_phase = int(n_points * 0.3)

        for i in range(accumulation_phase):
            noise = np.random.randn() * 0.00003
            trend = 0.000002
            vol = 1500 + i * 10 + np.random.randn() * 100

            prices.append(prices[-1] + noise + trend)
            volumes.append(max(100, vol))

        # Fase 3: Reação em cadeia
        remaining = n_points - normal_phase - accumulation_phase

        for i in range(remaining):
            trend = 0.0003 * (1 + i * 0.1)
            noise = np.random.randn() * 0.0001
            vol = 3000 + i * 100 + np.random.randn() * 200

            prices.append(prices[-1] + trend + noise)
            volumes.append(max(100, vol))
    else:
        remaining = n_points - normal_phase

        for i in range(remaining):
            noise = np.random.randn() * 0.00005
            vol = 1000 + np.random.randn() * 200

            prices.append(prices[-1] + noise)
            volumes.append(max(100, vol))

    return np.array(prices), np.array(volumes)


def main():
    """Demonstração do indicador RD-ME"""
    print("=" * 70)
    print("TURING-GRAY-SCOTT REACTION-DIFFUSION MORPHOGENESIS ENGINE (RD-ME)")
    print("Indicador baseado em Química de Reação-Difusão")
    print("=" * 70)
    print()

    indicator = TuringGrayScottReactionDiffusionMorphogenesisEngine(
        n_price_levels=100,
        Du=0.2,
        Dv=0.1,
        default_F=0.037,
        default_k=0.06,
        solver_steps=1000,
        solver_method="crank_nicolson",
        min_data_points=50
    )

    print("Indicador inicializado!")
    print(f"  - Grid: {indicator.n_price_levels} níveis de preço")
    print(f"  - Du/Dv = 2.0 (requisito para Turing: Du > Dv)")
    print(f"  - Solver: Crank-Nicolson ({indicator.solver_steps} passos)")
    print()

    print("Gerando dados com fluxo tóxico...")
    prices, volumes = generate_toxic_flow_data(n_points=100, seed=42, with_informed_trader=True)
    print(f"Dados gerados: {len(prices)} pontos")
    print()

    result = indicator.analyze(prices, volumes)

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Sinal: {result['signal_name']}")
    print(f"Confiança: {result['confidence']:.2%}")
    print(f"Fase: {result['phase_state']}")
    print(f"Padrão: {result['pattern_type']}")
    print(f"Reação: {result['reaction_type']}")
    print(f"VPIN: {result['vpin']:.4f}")
    print(f"Toxicidade: {result['toxicity_level']}")
    print(f"Razões: {result['reasons']}")

    return indicator, result


if __name__ == "__main__":
    indicator, result = main()
