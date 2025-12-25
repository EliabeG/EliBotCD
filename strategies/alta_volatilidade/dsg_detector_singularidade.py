"""
Detector de Singularidade Gravitacional (DSG)
==============================================
Nível de Complexidade: Astrofísica Computacional / Relatividade Numérica.

Premissa Teórica: O mercado é uma variedade pseudo-Riemanniana 4D. As coordenadas são
(t, P, V_bid, V_ask). Grandes ordens institucionais não "empurram" o preço; elas criam uma
curvatura no espaço-tempo financeiro. O preço segue a Geodésica (o caminho de menor
resistência). Alta volatilidade é o equivalente a cair em um buraco negro: a curvatura se torna
infinita e as regras normais cessam.

Dependências Críticas: jax ou tensorflow (para operações tensoriais aceleradas e diferenciação
automática), numpy (uso extensivo de einsum), scipy.spatial

VERSÃO V3.4 - CORREÇÕES DA QUARTA AUDITORIA (25/12/2025)
===========================================================
Correções aplicadas (V2.0):
1. Substituído gaussian_filter1d (não-causal) por EMA causal
2. Direção da geodésica calculada apenas com barras FECHADAS
3. Adicionado suporte para modo "online" sem look-ahead

Correções aplicadas (V3.0 - Auditoria):
4. NÃO calcular último ponto (n-1) no método analyze() - barra atual
5. Volumes determinísticos SEM look-ahead (backward diff)
6. Histórico de Ricci e distância SEM contaminação da barra atual
7. Centro de massa EXCLUI barra atual
8. Direção geodésica usa _coords_history[:-2] para excluir completamente barra atual
9. ricci_collapsing e crossing_horizon usam histórico ANTES da barra atual

Correções aplicadas (V3.1 - Auditoria Completa):
10. VALIDAÇÃO DE INPUTS: Verificação de NaN, Inf, negativos, ordem temporal
11. VOLUMES CENTRALIZADOS: Usa config/volume_generator.py para consistência
12. SUBSAMPLING ADAPTATIVO: step dinâmico baseado em volatilidade
13. THREAD-SAFETY: Lock para proteger estado compartilhado
14. LOOK-AHEAD RESIDUAL: Vetor tangente retorna [1,0,0,0] quando histórico vazio

Correções aplicadas (V3.2 - Segunda Auditoria 24/12/2025):
15. STEP FUNCTION: Usa NaN propagado ao invés de 0.0 para índices sem dados
16. CENTRO DE MASSA: Retorna NaN quando histórico vazio (não usa preço atual)
17. GEODESIC_DIRECTION: Calculado com histórico ANTES de adicionar coords
18. THREAD-SAFETY COMPLETO: Lock em analyze_point() para modificar históricos
19. RICCI_COLLAPSING: Acesso ao histórico dentro do lock
20. SIGNAL_NAME: Unificado para 'NEUTRO' (era inconsistente 'HOLD')
21. MIN_HISTORY: Exige mínimo de pontos antes de gerar sinais

Correções aplicadas (V3.3 - Terceira Auditoria 25/12/2025):
22. EH_DISTANCE: Usa preço da última barra FECHADA, não preço atual (look-ahead fix)
23. _GENERATE_SIGNAL: Verifica history_length mínimo antes de gerar sinais
24. EMA_CAUSAL: Tratamento robusto de NaN na entrada (encontra primeiro válido)

Correções aplicadas (V3.4 - Quarta Auditoria 25/12/2025):
25. RICCI_THRESHOLD: Valor padrão corrigido para escala real (-50500 ao invés de -0.5)
26. RICCI_CONDITION: Usa percentil dinâmico do histórico para filtrar sinais
27. DIRECTION_FALLBACK: Removido fallback que gerava sinais com histórico insuficiente
"""

import numpy as np
# REMOVIDO: from scipy.ndimage import gaussian_filter1d (era não-causal, substituído por EMA)
from scipy.spatial.distance import cdist
import warnings
import threading
from typing import Tuple, Optional
warnings.filterwarnings('ignore')

# CORREÇÃO V3.1: Importar gerador de volumes centralizado
try:
    from config.volume_generator import generate_synthetic_volumes, VOLUME_MULTIPLIER, VOLUME_BASE
    VOLUME_GENERATOR_AVAILABLE = True
except ImportError:
    # Fallback se módulo não disponível (para testes isolados)
    VOLUME_GENERATOR_AVAILABLE = False
    VOLUME_MULTIPLIER = 10000.0
    VOLUME_BASE = 50.0

# Tentar importar JAX para compilação JIT
JAX_AVAILABLE = False
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, jacfwd, jacrev
    JAX_AVAILABLE = True
except ImportError:
    pass

# Fallback para numpy puro (mais lento mas funcional)
if not JAX_AVAILABLE:
    jnp = np
    def jit(func):
        return func


class TensorMetricoFinanceiro:
    """
    Módulo 1: O Tensor Métrico Financeiro (g_μν)

    Primeiro, defina a geometria do espaço. Não use distância Euclidiana.

    Construa o Tensor Métrico g_μν (4x4) dinamicamente.
    A métrica deve acoplar o tempo (dt) com a liquidez (dV).

    ds² = -c(V)²dt² + dP² + γ(dV²_bid + dV²_ask)

    Onde c(V) é a "velocidade da luz" local do mercado (velocidade máxima de
    preenchimento de ordens baseada na latência do servidor).
    """

    def __init__(self, c_base: float = 1.0, gamma: float = 0.1, eps: float = 1e-10):
        """
        Parâmetros:
        -----------
        c_base : float
            Velocidade base da luz financeira (normalizada)
        gamma : float
            Fator de acoplamento para volume bid/ask
        eps : float
            Epsilon para estabilidade numérica
        """
        self.c_base = c_base
        self.gamma = gamma
        self.eps = eps
        self.dim = 4  # Dimensão do espaço-tempo (t, P, V_bid, V_ask)

    def calculate_speed_of_light(self, V_bid: float, V_ask: float) -> float:
        """
        c(V) - Velocidade da luz local do mercado

        Depende da liquidez disponível. Maior liquidez = maior velocidade de propagação
        """
        total_volume = V_bid + V_ask + self.eps
        # c(V) aumenta com a liquidez
        c_local = self.c_base * np.sqrt(1 + np.log1p(total_volume))
        return c_local

    def construct_metric_tensor(self, t: float, P: float,
                                 V_bid: float, V_ask: float) -> np.ndarray:
        """
        Constrói o tensor métrico g_μν no ponto (t, P, V_bid, V_ask)

        Métrica pseudo-Riemanniana (assinatura -,+,+,+):

        ds² = -c(V)²dt² + dP² + γ(dV²_bid + dV²_ask)

        g_μν = diag(-c²(V), 1, γ, γ)

        Com termos de acoplamento não-diagonais para capturar interações
        """
        c = self.calculate_speed_of_light(V_bid, V_ask)

        # Tensor métrico 4x4
        g = np.zeros((4, 4))

        # Componentes diagonais (assinatura de Lorentz)
        g[0, 0] = -c**2  # g_tt (temporal, negativo para métrica pseudo-Riemanniana)
        g[1, 1] = 1.0    # g_PP (preço)
        g[2, 2] = self.gamma  # g_VbidVbid
        g[3, 3] = self.gamma  # g_VaskVask

        # Termos de acoplamento não-diagonais (espaço-tempo financeiro curvo)
        # Acoplamento tempo-volume (ordens grandes distorcem o tempo local)
        imbalance = (V_bid - V_ask) / (V_bid + V_ask + self.eps)
        total_vol = V_bid + V_ask + self.eps

        g[0, 1] = g[1, 0] = 0.1 * imbalance  # Acoplamento t-P
        g[0, 2] = g[2, 0] = 0.01 * V_bid / total_vol
        g[0, 3] = g[3, 0] = 0.01 * V_ask / total_vol

        # Acoplamento preço-volume
        g[1, 2] = g[2, 1] = 0.05 * imbalance
        g[1, 3] = g[3, 1] = -0.05 * imbalance

        # Acoplamento bid-ask
        g[2, 3] = g[3, 2] = 0.02 * np.abs(imbalance)

        return g

    def inverse_metric(self, g: np.ndarray) -> np.ndarray:
        """
        Calcula o tensor métrico inverso g^μν
        """
        try:
            g_inv = np.linalg.inv(g)
        except np.linalg.LinAlgError:
            # Regularização se singular
            g_reg = g + self.eps * np.eye(4)
            g_inv = np.linalg.inv(g_reg)
        return g_inv


class SimbolosChristoffel:
    """
    Módulo 2: Símbolos de Christoffel (Γ^λ_μν)

    Para saber como o mercado "gira" e "acelera", calcule a conexão de Levi-Civita.

    Utilize diferenciação automática (jax.grad) para calcular as derivadas parciais
    do tensor métrico em relação às coordenadas.

    Fórmula (implementar via np.einsum para não explodir a CPU):

    Γ^λ_μν = (1/2) g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
    """

    def __init__(self, metric_calculator: TensorMetricoFinanceiro,
                 delta: float = 1e-5):
        """
        Parâmetros:
        -----------
        metric_calculator : TensorMetricoFinanceiro
            Calculador do tensor métrico
        delta : float
            Passo para diferenciação numérica
        """
        self.metric_calc = metric_calculator
        self.delta = delta
        self.dim = 4

    def _metric_at_point(self, coords: np.ndarray) -> np.ndarray:
        """Wrapper para calcular métrica em coordenadas"""
        return self.metric_calc.construct_metric_tensor(
            coords[0], coords[1], coords[2], coords[3]
        )

    def compute_metric_derivatives(self, coords: np.ndarray) -> np.ndarray:
        """
        Calcula ∂_σ g_μν para todas as coordenadas σ

        Retorna tensor de forma (4, 4, 4) onde:
        dg[σ, μ, ν] = ∂g_μν/∂x^σ
        """
        dg = np.zeros((4, 4, 4))

        for sigma in range(4):
            # Diferença central para derivada
            coords_plus = coords.copy()
            coords_minus = coords.copy()

            coords_plus[sigma] += self.delta
            coords_minus[sigma] -= self.delta

            g_plus = self._metric_at_point(coords_plus)
            g_minus = self._metric_at_point(coords_minus)

            dg[sigma] = (g_plus - g_minus) / (2 * self.delta)

        return dg

    def compute_christoffel_symbols(self, coords: np.ndarray) -> np.ndarray:
        """
        Calcula os símbolos de Christoffel Γ^λ_μν

        Γ^λ_μν = (1/2) g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)

        Implementado com np.einsum para eficiência
        """
        # Métrica e inversa
        g = self._metric_at_point(coords)
        g_inv = self.metric_calc.inverse_metric(g)

        # Derivadas da métrica
        dg = self.compute_metric_derivatives(coords)

        # Calcular Γ^λ_μν usando einsum
        # Termo 1: ∂_μ g_νσ -> dg[μ, ν, σ]
        # Termo 2: ∂_ν g_μσ -> dg[ν, μ, σ]
        # Termo 3: ∂_σ g_μν -> dg[σ, μ, ν]

        # Primeiro, construir (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
        # Shape: (4, 4, 4) indexado por (μ, ν, σ)
        bracket = np.zeros((4, 4, 4))

        for mu in range(4):
            for nu in range(4):
                for sigma in range(4):
                    bracket[mu, nu, sigma] = (
                        dg[mu, nu, sigma] +  # ∂_μ g_νσ
                        dg[nu, mu, sigma] -  # ∂_ν g_μσ
                        dg[sigma, mu, nu]    # ∂_σ g_μν
                    )

        # Γ^λ_μν = (1/2) g^λσ * bracket[μ, ν, σ]
        # Usar einsum: 'ls,mns->lmn'
        christoffel = 0.5 * np.einsum('ls,mns->lmn', g_inv, bracket)

        return christoffel


class TensorCurvaturaRicci:
    """
    Módulo 3: O Tensor de Curvatura de Ricci (R_μν) e Escalar de Ricci (R)

    Aqui está a detecção da volatilidade real.

    O Escalar de Ricci (R) mede o volume do espaço-tempo financeiro:
    - R = 0: Espaço plano. Mercado eficiente/calmo.
    - R > 0: Espaço esférico. Convergência de ordens (Consolidação).
    - R << 0 (Muito Negativo): Espaço hiperbólico. Divergência explosiva. ALTA VOLATILIDADE.

    Calcule R contraindo o Tensor de Riemann.
    """

    def __init__(self, christoffel_calc: SimbolosChristoffel, delta: float = 1e-5):
        """
        Parâmetros:
        -----------
        christoffel_calc : SimbolosChristoffel
            Calculador dos símbolos de Christoffel
        delta : float
            Passo para diferenciação numérica
        """
        self.christoffel_calc = christoffel_calc
        self.delta = delta
        self.dim = 4

    def compute_christoffel_derivatives(self, coords: np.ndarray) -> np.ndarray:
        """
        Calcula ∂_ρ Γ^λ_μν

        Retorna tensor de forma (4, 4, 4, 4) onde:
        dGamma[ρ, λ, μ, ν] = ∂Γ^λ_μν/∂x^ρ
        """
        dGamma = np.zeros((4, 4, 4, 4))

        for rho in range(4):
            coords_plus = coords.copy()
            coords_minus = coords.copy()

            coords_plus[rho] += self.delta
            coords_minus[rho] -= self.delta

            Gamma_plus = self.christoffel_calc.compute_christoffel_symbols(coords_plus)
            Gamma_minus = self.christoffel_calc.compute_christoffel_symbols(coords_minus)

            dGamma[rho] = (Gamma_plus - Gamma_minus) / (2 * self.delta)

        return dGamma

    def compute_riemann_tensor(self, coords: np.ndarray) -> np.ndarray:
        """
        Calcula o Tensor de Riemann R^ρ_σμν

        R^ρ_σμν = ∂_μ Γ^ρ_νσ - ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ - Γ^ρ_νλ Γ^λ_μσ

        O Desafio de Programação (Cálculo Tensorial):
        O pesadelo aqui é a notação de índices. Se você errar um índice no np.einsum
        (ex: ik, kj -> ij), você inverte a causalidade do universo e seu indicador
        vai prever o passado.
        """
        Gamma = self.christoffel_calc.compute_christoffel_symbols(coords)
        dGamma = self.compute_christoffel_derivatives(coords)

        # Tensor de Riemann R^ρ_σμν (4^4 = 256 componentes)
        Riemann = np.zeros((4, 4, 4, 4))

        for rho in range(4):
            for sigma in range(4):
                for mu in range(4):
                    for nu in range(4):
                        # Termo 1: ∂_μ Γ^ρ_νσ
                        term1 = dGamma[mu, rho, nu, sigma]

                        # Termo 2: -∂_ν Γ^ρ_μσ
                        term2 = -dGamma[nu, rho, mu, sigma]

                        # Termo 3: Γ^ρ_μλ Γ^λ_νσ (soma em λ)
                        term3 = 0.0
                        for lam in range(4):
                            term3 += Gamma[rho, mu, lam] * Gamma[lam, nu, sigma]

                        # Termo 4: -Γ^ρ_νλ Γ^λ_μσ (soma em λ)
                        term4 = 0.0
                        for lam in range(4):
                            term4 -= Gamma[rho, nu, lam] * Gamma[lam, mu, sigma]

                        Riemann[rho, sigma, mu, nu] = term1 + term2 + term3 + term4

        return Riemann

    def compute_ricci_tensor(self, coords: np.ndarray) -> np.ndarray:
        """
        Calcula o Tensor de Ricci R_μν por contração do Tensor de Riemann

        R_μν = R^ρ_μρν (contração no primeiro e terceiro índice)
        """
        Riemann = self.compute_riemann_tensor(coords)

        # Contração: R_μν = R^ρ_μρν
        # Usar einsum: 'rmrn->mn'
        Ricci = np.einsum('rmrn->mn', Riemann)

        return Ricci

    def compute_ricci_scalar(self, coords: np.ndarray) -> float:
        """
        Calcula o Escalar de Ricci R

        R = g^μν R_μν (contração do tensor de Ricci com a métrica inversa)

        Este é o indicador principal de curvatura:
        - R = 0: Espaço plano (mercado calmo)
        - R > 0: Espaço esférico (consolidação)
        - R << 0: Espaço hiperbólico (ALTA VOLATILIDADE)
        """
        g = self.christoffel_calc.metric_calc.construct_metric_tensor(
            coords[0], coords[1], coords[2], coords[3]
        )
        g_inv = self.christoffel_calc.metric_calc.inverse_metric(g)

        Ricci = self.compute_ricci_tensor(coords)

        # R = g^μν R_μν
        # Usar einsum: 'mn,mn->'
        R = np.einsum('mn,mn->', g_inv, Ricci)

        return R


class DesvioGeodesico:
    """
    Módulo 4: A Equação do Desvio da Geodésica (Forças de Maré)

    Não queremos apenas saber se há gravidade, mas se ela vai "rasgar" o preço.

    Modele a separação entre duas ordens próximas (ξ^μ).

    Aceleração da separação:
    D²ξ^μ/dτ² = -R^μ_νρσ T^ν ξ^ρ T^σ

    (Onde T é o vetor tangente de velocidade do preço).

    Sinal: Se a "Força de Maré" (Tidal Force) for alta, significa que o spread está sendo
    rasgado por fluxo institucional agressivo em direções opostas (batalha de liquidez)
    logo antes de um rompimento.
    """

    def __init__(self, ricci_calc: TensorCurvaturaRicci):
        """
        Parâmetros:
        -----------
        ricci_calc : TensorCurvaturaRicci
            Calculador do tensor de Ricci/Riemann
        """
        self.ricci_calc = ricci_calc
        self.dim = 4

    def compute_tangent_vector(self, coords_history: np.ndarray) -> np.ndarray:
        """
        Calcula o vetor tangente T^μ = dx^μ/dτ (velocidade no espaço-tempo)

        CORREÇÃO V3.1: Quando histórico insuficiente, retorna vetor temporal puro
        sem usar coordenadas atuais (evita look-ahead bias).
        """
        # CORREÇÃO V3.1: Verificar se coords_history é válido
        if coords_history is None or len(coords_history) == 0:
            # Sem histórico: retornar vetor temporal puro (sem look-ahead)
            return np.array([1.0, 0.0, 0.0, 0.0])

        if len(coords_history) < 2:
            # Apenas 1 ponto: retornar vetor temporal puro (sem look-ahead)
            # ANTES: usava coords atual como fallback (look-ahead!)
            return np.array([1.0, 0.0, 0.0, 0.0])

        # Derivada numérica das coordenadas (usa apenas histórico)
        T = coords_history[-1] - coords_history[-2]

        # Normalizar
        norm = np.sqrt(np.abs(np.sum(T**2)) + 1e-10)
        T = T / norm

        return T

    def compute_separation_vector(self, bid_volume: float, ask_volume: float) -> np.ndarray:
        """
        Vetor de separação ξ^μ entre ordens bid e ask

        Modela a "distância" entre as duas ordens no espaço-tempo
        """
        xi = np.zeros(4)

        # Separação temporal (zero para ordens simultâneas)
        xi[0] = 0.0

        # Separação de preço (spread implícito)
        spread_proxy = np.abs(bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)
        xi[1] = spread_proxy

        # Separação em volume
        xi[2] = bid_volume / (bid_volume + ask_volume + 1e-10)
        xi[3] = ask_volume / (bid_volume + ask_volume + 1e-10)

        return xi

    def compute_tidal_force(self, coords: np.ndarray,
                            T: np.ndarray,
                            xi: np.ndarray) -> np.ndarray:
        """
        Calcula a Força de Maré (aceleração do desvio geodésico)

        D²ξ^μ/dτ² = -R^μ_νρσ T^ν ξ^ρ T^σ

        Retorna o vetor de aceleração da separação
        """
        Riemann = self.ricci_calc.compute_riemann_tensor(coords)

        # Calcular a contração: -R^μ_νρσ T^ν ξ^ρ T^σ
        # Usar einsum: 'mnrs,n,r,s->m'
        tidal_acceleration = -np.einsum('mnrs,n,r,s->m', Riemann, T, xi, T)

        return tidal_acceleration

    def compute_tidal_force_magnitude(self, coords: np.ndarray,
                                       T: np.ndarray,
                                       xi: np.ndarray) -> float:
        """
        Calcula a magnitude da força de maré

        |F_tidal| = √(Σ (D²ξ^μ/dτ²)²)
        """
        tidal_acc = self.compute_tidal_force(coords, T, xi)

        # Magnitude (norma euclidiana como aproximação)
        magnitude = np.sqrt(np.sum(tidal_acc**2))

        return magnitude


class HorizonteEventos:
    """
    Módulo 5: O Horizonte de Eventos (Gatilho de Entrada)

    Calcule o Raio de Schwarzschild Financeiro (r_s) baseado no volume acumulado
    (massa) nos últimos minutos.

    Condição de Disparo:
    1. O Escalar de Ricci (R) cai verticalmente (Colapso geométrico).
    2. A distância do preço atual para o centro de massa do volume (r) se aproxima de r_s.
    3. Estamos cruzando o "Horizonte de Eventos". A partir daqui, a inércia é tão grande
       que o preço não pode voltar.
    4. Direção: Seguir o sinal da componente temporal da Geodésica (d²P/dτ²).
    """

    def __init__(self, gravitational_constant: float = 1.0):
        """
        Parâmetros:
        -----------
        gravitational_constant : float
            Constante gravitacional financeira (escala)
        """
        self.G = gravitational_constant
        self.c = 1.0  # Velocidade da luz (normalizada)

    def compute_schwarzschild_radius(self, accumulated_volume: float) -> float:
        """
        Calcula o Raio de Schwarzschild Financeiro

        r_s = 2GM/c²

        Onde M é a "massa" (volume acumulado)
        """
        M = accumulated_volume
        r_s = 2 * self.G * M / (self.c**2)

        # Normalizar para escala de preço
        r_s = r_s * 0.0001  # Escalar para pips/pontos

        return r_s

    def compute_volume_center_of_mass(self, prices: np.ndarray,
                                       volumes: np.ndarray) -> float:
        """
        Calcula o centro de massa do volume (VWAP local)
        """
        if len(prices) == 0 or len(volumes) == 0:
            return prices[-1] if len(prices) > 0 else 0.0

        total_volume = np.sum(volumes) + 1e-10
        center_of_mass = np.sum(prices * volumes) / total_volume

        return center_of_mass

    def compute_event_horizon_distance(self, current_price: float,
                                        center_of_mass: float,
                                        schwarzschild_radius: float) -> float:
        """
        Calcula a distância ao horizonte de eventos

        Distância = |preço_atual - centro_massa| - r_s

        Se distância <= 0, estamos dentro do horizonte de eventos
        """
        r = np.abs(current_price - center_of_mass)
        distance = r - schwarzschild_radius

        return distance

    def is_crossing_event_horizon(self, distance: float,
                                   distance_history: list = None) -> bool:
        """
        Detecta se estamos cruzando o horizonte de eventos

        Condição: distância estava positiva e agora está se aproximando de zero/negativo
        """
        if distance_history is None or len(distance_history) < 3:
            return distance <= 0

        # Verificar tendência de aproximação
        recent = distance_history[-5:]
        if len(recent) >= 3:
            trend = np.polyfit(range(len(recent)), recent, 1)[0]
            approaching = trend < 0
        else:
            approaching = False

        return distance <= 0 or (distance < 0.001 and approaching)


class DetectorSingularidadeGravitacional:
    """
    Implementação completa do Detector de Singularidade Gravitacional (DSG)

    Módulos:
    1. O Tensor Métrico Financeiro (g_μν)
    2. Símbolos de Christoffel (Γ^λ_μν)
    3. O Tensor de Curvatura de Ricci (R_μν) e Escalar de Ricci (R)
    4. A Equação do Desvio da Geodésica (Forças de Maré)
    5. O Horizonte de Eventos (Gatilho de Entrada)
    6. Output e Visualização
    """

    def __init__(self,
                 c_base: float = 1.0,
                 gamma: float = 0.1,
                 gravitational_constant: float = 1.0,
                 ricci_collapse_threshold: float = -50500.0,  # CORREÇÃO V3.4: Escala real (era -0.5)
                 tidal_force_threshold: float = 0.1,
                 event_horizon_threshold: float = 0.001,
                 lookback_window: int = 50):
        """
        Inicialização do Detector de Singularidade Gravitacional

        Parâmetros:
        -----------
        c_base : float
            Velocidade base da luz financeira

        gamma : float
            Fator de acoplamento volume bid/ask

        gravitational_constant : float
            Constante gravitacional financeira

        ricci_collapse_threshold : float
            Limiar para colapso do escalar de Ricci (R << 0)
            CORREÇÃO V3.4: Valores reais de Ricci estão na faixa -51000 a -49500
            O threshold padrão -50500 está no meio desta faixa
            Valores mais negativos que o threshold indicam colapso

        tidal_force_threshold : float
            Limiar para força de maré alta

        event_horizon_threshold : float
            Limiar de distância ao horizonte de eventos

        lookback_window : int
            Janela de lookback para cálculos
        """
        self.c_base = c_base
        self.gamma = gamma
        self.G = gravitational_constant
        self.ricci_collapse_threshold = ricci_collapse_threshold
        self.tidal_force_threshold = tidal_force_threshold
        self.event_horizon_threshold = event_horizon_threshold
        self.lookback_window = lookback_window

        # Inicializar módulos
        self.metric_calc = TensorMetricoFinanceiro(c_base, gamma)
        self.christoffel_calc = SimbolosChristoffel(self.metric_calc)
        self.ricci_calc = TensorCurvaturaRicci(self.christoffel_calc)
        self.geodesic_calc = DesvioGeodesico(self.ricci_calc)
        self.horizon_calc = HorizonteEventos(gravitational_constant)

        # Cache e histórico
        self._ricci_history = []
        self._distance_history = []
        self._coords_history = []

        # CORREÇÃO V3.1: Lock para thread-safety
        self._lock = threading.Lock()

        # CORREÇÃO V3.1: Mínimo de preços para análise
        self.min_prices = 10

        # CORREÇÃO V3.2: Mínimo de histórico para gerar sinais confiáveis
        # Precisa de pelo menos N pontos no _coords_history para:
        # - Calcular geodesic_direction (4 pontos)
        # - Calcular ricci_collapsing (5 pontos)
        # - Ter vetor tangente válido (2 pontos)
        self.min_history_for_signal = 6

    def _validate_inputs(self, prices: np.ndarray,
                         bid_volumes: np.ndarray = None,
                         ask_volumes: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        CORREÇÃO V3.1: Valida e sanitiza inputs antes do processamento.

        Verifica:
        - Array de preços não vazio
        - Sem NaN ou Inf
        - Sem valores negativos (preços devem ser positivos)
        - Mínimo de dados para análise

        Args:
            prices: Array de preços de fechamento
            bid_volumes: Array de volumes de bid (opcional)
            ask_volumes: Array de volumes de ask (opcional)

        Returns:
            Tuple (prices, bid_volumes, ask_volumes) sanitizados

        Raises:
            ValueError: Se dados inválidos não puderem ser corrigidos
        """
        # Verificar array de preços
        if prices is None or len(prices) == 0:
            raise ValueError("Array de preços vazio ou None")

        # Converter para numpy se necessário
        prices = np.asarray(prices, dtype=np.float64)

        # Verificar NaN e Inf em preços
        nan_mask = np.isnan(prices)
        inf_mask = np.isinf(prices)

        if np.any(nan_mask) or np.any(inf_mask):
            bad_count = nan_mask.sum() + inf_mask.sum()

            # Se primeiro preço é inválido, não podemos corrigir
            if nan_mask[0] or inf_mask[0]:
                raise ValueError(f"Primeiro preço é NaN ou Inf, impossível corrigir")

            # Se menos de 10% dos dados são ruins, forward-fill
            if bad_count < len(prices) * 0.1:
                bad_mask = nan_mask | inf_mask
                for i in range(1, len(prices)):
                    if bad_mask[i]:
                        prices[i] = prices[i-1]
            else:
                raise ValueError(f"Dados contêm {bad_count} valores NaN/Inf ({bad_count/len(prices)*100:.1f}%)")

        # Verificar preços negativos
        if np.any(prices <= 0):
            neg_count = np.sum(prices <= 0)
            raise ValueError(f"Preços contêm {neg_count} valores negativos ou zero")

        # Verificar mínimo de dados
        if len(prices) < self.min_prices:
            raise ValueError(f"Mínimo de {self.min_prices} preços necessário, recebido {len(prices)}")

        # Validar volumes se fornecidos
        if bid_volumes is not None:
            bid_volumes = np.asarray(bid_volumes, dtype=np.float64)
            if len(bid_volumes) != len(prices):
                raise ValueError(f"Tamanho de bid_volumes ({len(bid_volumes)}) diferente de prices ({len(prices)})")
            # Substituir NaN/Inf/negativos por valor base
            bad_mask = np.isnan(bid_volumes) | np.isinf(bid_volumes) | (bid_volumes < 0)
            if np.any(bad_mask):
                bid_volumes[bad_mask] = VOLUME_BASE

        if ask_volumes is not None:
            ask_volumes = np.asarray(ask_volumes, dtype=np.float64)
            if len(ask_volumes) != len(prices):
                raise ValueError(f"Tamanho de ask_volumes ({len(ask_volumes)}) diferente de prices ({len(prices)})")
            # Substituir NaN/Inf/negativos por valor base
            bad_mask = np.isnan(ask_volumes) | np.isinf(ask_volumes) | (ask_volumes < 0)
            if np.any(bad_mask):
                ask_volumes[bad_mask] = VOLUME_BASE

        return prices, bid_volumes, ask_volumes

    def _calculate_adaptive_step(self, prices: np.ndarray, last_closed_idx: int) -> int:
        """
        CORREÇÃO V3.1: Calcula step adaptativo baseado em volatilidade.

        Em períodos de alta volatilidade, usa step menor para não perder eventos.
        Em períodos calmos, usa step maior para performance.

        Args:
            prices: Array de preços
            last_closed_idx: Índice da última barra fechada

        Returns:
            Step adaptativo (mínimo 1, máximo baseado em volatilidade)
        """
        # Calcular volatilidade recente (últimas 20 barras ou disponíveis)
        window = min(20, last_closed_idx)
        if window < 2:
            return 1

        recent_prices = prices[last_closed_idx - window:last_closed_idx + 1]
        returns = np.diff(recent_prices) / recent_prices[:-1]

        # Volatilidade como desvio padrão dos retornos
        volatility = np.std(returns)

        # Definir step baseado em volatilidade
        # Alta vol (>0.5%) -> step=1 (calcular todos os pontos)
        # Média vol (0.1-0.5%) -> step=2-5
        # Baixa vol (<0.1%) -> step até max_step
        max_step = max(1, last_closed_idx // 50)  # Mínimo 50 pontos

        if volatility > 0.005:  # >0.5%
            step = 1
        elif volatility > 0.001:  # 0.1-0.5%
            step = max(1, min(3, max_step))
        else:  # <0.1%
            step = max(1, min(max_step, last_closed_idx // 100))

        return step

    def _prepare_coordinates(self, t: int, price: float,
                              bid_vol: float, ask_vol: float) -> np.ndarray:
        """
        Prepara coordenadas do espaço-tempo financeiro
        """
        # Normalizar coordenadas
        t_norm = t / 1000.0  # Normalizar tempo
        p_norm = price  # Preço já em escala adequada
        vb_norm = np.log1p(bid_vol)  # Log-transformar volume
        va_norm = np.log1p(ask_vol)

        return np.array([t_norm, p_norm, vb_norm, va_norm])

    def analyze_point(self, t: int, price: float,
                      bid_vol: float, ask_vol: float,
                      _use_lock: bool = False) -> dict:
        """
        Analisa um único ponto no espaço-tempo financeiro

        CORREÇÃO V3.2: Parâmetro _use_lock para thread-safety quando chamado
        fora do contexto do lock em analyze(). Por padrão False para manter
        compatibilidade com chamadas dentro do 'with self._lock:' em analyze().
        """
        coords = self._prepare_coordinates(t, price, bid_vol, ask_vol)

        # 1. Tensor Métrico
        g = self.metric_calc.construct_metric_tensor(
            coords[0], coords[1], coords[2], coords[3]
        )

        # 2. Escalar de Ricci
        R = self.ricci_calc.compute_ricci_scalar(coords)

        # CORREÇÃO V3.0: Calcular vetor tangente e centro de massa ANTES de adicionar
        # a barra atual ao histórico, para evitar look-ahead bias

        # CORREÇÃO V3.2: Operações que leem/escrevem histórico precisam de lock
        # Quando chamado de analyze() já estamos dentro do lock, então _use_lock=False
        if _use_lock:
            self._lock.acquire()

        try:
            # 3. Vetor tangente (usa histórico SEM a barra atual)
            # CORREÇÃO V3.1: NÃO usar coords como fallback (isso é look-ahead!)
            # Se histórico vazio, compute_tangent_vector retorna vetor temporal puro
            T = self.geodesic_calc.compute_tangent_vector(
                np.array(self._coords_history) if len(self._coords_history) > 0 else None
            )

            # 4. Vetor de separação e força de maré
            xi = self.geodesic_calc.compute_separation_vector(bid_vol, ask_vol)
            tidal_magnitude = self.geodesic_calc.compute_tidal_force_magnitude(coords, T, xi)

            # 5. Horizonte de eventos
            accumulated_vol = bid_vol + ask_vol
            r_s = self.horizon_calc.compute_schwarzschild_radius(accumulated_vol)

            # CORREÇÃO V3.2: Centro de massa calculado APENAS com barras ANTERIORES
            # Quando histórico vazio, NÃO usar preço atual (isso é look-ahead!)
            # Usar NaN para indicar "sem dados suficientes"
            if len(self._coords_history) > 1:
                # Usar apenas histórico existente (sem a barra atual)
                prices_hist = [c[1] for c in self._coords_history[-20:]]
                vols_hist = [np.exp(c[2]) + np.exp(c[3]) for c in self._coords_history[-20:]]
                com = self.horizon_calc.compute_volume_center_of_mass(
                    np.array(prices_hist), np.array(vols_hist)
                )
            elif len(self._coords_history) == 1:
                # Apenas uma barra anterior, usar seu preço
                com = self._coords_history[0][1]
            else:
                # CORREÇÃO V3.2: Sem histórico, usar NaN (não preço atual!)
                # Comentário anterior dizia "barra anterior" mas usava preço atual
                com = np.nan

            # Calcular distância ao horizonte
            # CORREÇÃO V3.3: Usar preço da última barra FECHADA, não preço atual!
            # O 'price' parâmetro é da barra atual (look-ahead!)
            # Devemos usar o preço da última barra já no histórico
            if len(self._coords_history) > 0:
                last_closed_price = self._coords_history[-1][1]
                eh_distance = self.horizon_calc.compute_event_horizon_distance(
                    last_closed_price, com, r_s
                )
            else:
                # Sem histórico, sem distância calculável
                eh_distance = np.nan

            # AGORA adicionar a barra atual ao histórico (DEPOIS dos cálculos)
            self._coords_history.append(coords)
            if len(self._coords_history) > self.lookback_window:
                self._coords_history = self._coords_history[-self.lookback_window:]

            # Atualizar histórico de Ricci e distância
            self._ricci_history.append(R)
            self._distance_history.append(eh_distance)

            if len(self._ricci_history) > self.lookback_window:
                self._ricci_history = self._ricci_history[-self.lookback_window:]
                self._distance_history = self._distance_history[-self.lookback_window:]

        finally:
            if _use_lock:
                self._lock.release()

        return {
            'ricci_scalar': R,
            'tidal_force': tidal_magnitude,
            'event_horizon_distance': eh_distance,
            'schwarzschild_radius': r_s,
            'center_of_mass': com,
            'tangent_vector': T,
            'metric_tensor': g,
            'coordinates': coords
        }

    def analyze(self, prices: np.ndarray,
                bid_volumes: np.ndarray = None,
                ask_volumes: np.ndarray = None) -> dict:
        """
        Execução completa do Detector de Singularidade Gravitacional

        Output: [Ricci_Scalar, Tidal_Force_Magnitude, Event_Horizon_Distance]

        CORREÇÃO V3.1: Agora com validação de inputs, volumes centralizados,
        subsampling adaptativo e thread-safety.
        """
        # CORREÇÃO V3.1: Validação de inputs
        try:
            prices, bid_volumes, ask_volumes = self._validate_inputs(
                prices, bid_volumes, ask_volumes
            )
        except ValueError as e:
            # Se validação falhar, retornar resultado vazio com erro
            result = self._empty_result()
            result['error'] = str(e)
            return result

        n = len(prices)

        if n < self.min_prices:
            return self._empty_result()

        # CORREÇÃO V3.1: Gerar volumes usando função CENTRALIZADA
        # Isso garante consistência entre indicador, estratégia, backtest e otimizador
        if bid_volumes is None or ask_volumes is None:
            if VOLUME_GENERATOR_AVAILABLE:
                bid_volumes, ask_volumes = generate_synthetic_volumes(prices)
            else:
                # Fallback se módulo não disponível
                bid_volumes = np.zeros(n)
                ask_volumes = np.zeros(n)
                bid_volumes[0] = VOLUME_BASE
                ask_volumes[0] = VOLUME_BASE
                bid_volumes[1] = VOLUME_BASE
                ask_volumes[1] = VOLUME_BASE
                for i in range(2, n):
                    change = np.abs(prices[i-1] - prices[i-2])
                    bid_volumes[i] = change * VOLUME_MULTIPLIER + VOLUME_BASE
                    ask_volumes[i] = change * VOLUME_MULTIPLIER + VOLUME_BASE

        # CORREÇÃO V3.1: Thread-safety - adquirir lock antes de modificar estado
        with self._lock:
            # Resetar histórico
            self._ricci_history = []
            self._distance_history = []
            self._coords_history = []

        # Arrays para resultados
        ricci_series = np.zeros(n)
        tidal_series = np.zeros(n)
        distance_series = np.zeros(n)

        # CORREÇÃO V3.0 CRÍTICA: Subsampling SEM calcular a barra atual (n-1)
        # =================================================================
        # PROBLEMA V2.0: O código forçava o cálculo do último ponto (n-1),
        # que é a barra ATUAL ainda não fechada. Isso é look-ahead bias porque:
        # 1. O escalar de Ricci final usa o close da barra atual
        # 2. A força de maré final usa o close da barra atual
        # 3. A distância ao horizonte usa o close da barra atual
        # Em tempo real, você NÃO tem acesso ao close até a barra fechar!
        #
        # SOLUÇÃO V3.0: Calcular apenas até n-2 (última barra FECHADA)
        # O sinal é baseado apenas em barras completamente fechadas.
        # =================================================================

        # n-1 é a barra ATUAL (não fechada), n-2 é a última FECHADA
        last_closed_idx = n - 2
        if last_closed_idx < self.min_prices:
            return self._empty_result()

        # CORREÇÃO V3.1: Subsampling ADAPTATIVO baseado em volatilidade
        # Em períodos de alta volatilidade, calcula mais pontos para não perder eventos
        step = self._calculate_adaptive_step(prices, last_closed_idx)

        # Armazenar pontos calculados (apenas barras FECHADAS)
        calculated_indices = []
        calculated_ricci = []
        calculated_tidal = []
        calculated_distance = []

        # CORREÇÃO V3.1: Thread-safety - adquirir lock durante cálculos que modificam estado
        with self._lock:
            for i in range(0, last_closed_idx + 1, step):  # Até last_closed_idx inclusive
                result = self.analyze_point(
                    i, prices[i], bid_volumes[i], ask_volumes[i]
                )

                calculated_indices.append(i)
                calculated_ricci.append(result['ricci_scalar'])
                calculated_tidal.append(result['tidal_force'])
                calculated_distance.append(result['event_horizon_distance'])

            # Garantir que o último ponto FECHADO (n-2) foi calculado
            if calculated_indices[-1] != last_closed_idx:
                result = self.analyze_point(
                    last_closed_idx, prices[last_closed_idx],
                    bid_volumes[last_closed_idx], ask_volumes[last_closed_idx]
                )
                calculated_indices.append(last_closed_idx)
                calculated_ricci.append(result['ricci_scalar'])
                calculated_tidal.append(result['tidal_force'])
                calculated_distance.append(result['event_horizon_distance'])

        # CORREÇÃO: Step Function CAUSAL (Zero-Order Hold)
        # Usa apenas o ÚLTIMO valor calculado até cada índice - SEM look-ahead
        ricci_series = self._apply_step_function_causal(n, calculated_indices, calculated_ricci)
        tidal_series = self._apply_step_function_causal(n, calculated_indices, calculated_tidal)
        distance_series = self._apply_step_function_causal(n, calculated_indices, calculated_distance)

        # CORREÇÃO #1: Suavizar séries com EMA CAUSAL (não gaussian_filter1d que é não-causal)
        # gaussian_filter1d usa convolução simétrica que olha para o futuro!
        # EMA é 100% causal: só usa dados passados
        ricci_series = self._apply_ema_causal(ricci_series, alpha=0.3)
        tidal_series = self._apply_ema_causal(tidal_series, alpha=0.3)

        # CORREÇÃO V3.0: Valores da última barra FECHADA (não a atual)
        # CORREÇÃO V3.2: Tratar NaN - se o valor é NaN, usar 0.0 como fallback
        current_ricci = ricci_series[last_closed_idx] if last_closed_idx < len(ricci_series) else ricci_series[-1]
        current_tidal = tidal_series[last_closed_idx] if last_closed_idx < len(tidal_series) else tidal_series[-1]
        current_distance = distance_series[last_closed_idx] if last_closed_idx < len(distance_series) else distance_series[-1]

        # CORREÇÃO V3.2: Se valores são NaN (insuficiente histórico), usar 0.0
        if np.isnan(current_ricci):
            current_ricci = 0.0
        if np.isnan(current_tidal):
            current_tidal = 0.0
        if np.isnan(current_distance):
            current_distance = 0.0

        # CORREÇÃO V3.2: Acesso aos históricos DENTRO do lock para thread-safety
        with self._lock:
            # CORREÇÃO V3.0: Detectar colapso de Ricci SEM contaminação da barra atual
            # O _ricci_history agora contém apenas valores de barras FECHADAS (até n-2)
            # CORREÇÃO V3.2: Exigir mínimo de histórico para sinais confiáveis
            if len(self._ricci_history) >= self.min_history_for_signal:
                # Usar os últimos 5 valores do histórico (todos de barras fechadas)
                # Filtrar NaN antes de calcular diff
                valid_ricci = [r for r in self._ricci_history[-5:] if not np.isnan(r)]
                if len(valid_ricci) >= 2:
                    ricci_change = np.diff(valid_ricci)
                    ricci_collapsing = np.mean(ricci_change) < -0.1
                else:
                    ricci_collapsing = False
            else:
                ricci_collapsing = False

            # CORREÇÃO V3.0: Detectar cruzamento do horizonte SEM contaminação
            # O _distance_history agora contém apenas valores de barras FECHADAS
            # Filtrar NaN do histórico
            valid_distance_history = [d for d in self._distance_history if not np.isnan(d)]
            crossing_horizon = self.horizon_calc.is_crossing_event_horizon(
                current_distance, valid_distance_history
            )

            # CORREÇÃO V3.4: Direção geodésica usando APENAS barras COMPLETAMENTE FECHADAS
            # =============================================================================
            # O _coords_history contém coords de barras FECHADAS
            # Exigir mínimo de histórico para calcular direção confiável
            # CORREÇÃO V3.4: REMOVIDO fallback que gerava sinais com histórico insuficiente
            # Agora só gera direção quando há histórico mínimo (min_history_for_signal)
            # =============================================================================
            if len(self._coords_history) >= self.min_history_for_signal:
                # Usar os últimos 4 pontos (todos já fechados)
                prices_past = [c[1] for c in self._coords_history[-4:]]
                geodesic_direction = int(np.sign(prices_past[-1] - prices_past[0]))
            else:
                # CORREÇÃO V3.4: Histórico insuficiente, direção = 0 (neutro)
                # Removido fallback com 2+ pontos que gerava sinais não confiáveis
                geodesic_direction = 0

        # Gerar sinal
        # CORREÇÃO V3.3: Passar tamanho do histórico para validação
        # CORREÇÃO V3.4: Passar histórico de Ricci para percentil dinâmico
        signal_result = self._generate_signal(
            current_ricci, current_tidal, current_distance,
            ricci_collapsing, crossing_horizon, geodesic_direction,
            history_length=len(self._coords_history),
            ricci_history=self._ricci_history.copy()  # Cópia para evitar modificação
        )

        # Output principal
        output_vector = [current_ricci, current_tidal, current_distance]

        return {
            # Vetor de saída principal
            'output_vector': output_vector,
            'Ricci_Scalar': output_vector[0],
            'Tidal_Force_Magnitude': output_vector[1],
            'Event_Horizon_Distance': output_vector[2],

            # Séries temporais
            'ricci_series': ricci_series,
            'tidal_series': tidal_series,
            'distance_series': distance_series,

            # Diagnósticos
            'ricci_collapsing': ricci_collapsing,
            'crossing_horizon': crossing_horizon,
            'geodesic_direction': geodesic_direction,

            # Sinal
            'signal': signal_result['signal'],
            'signal_name': signal_result['signal_name'],
            'confidence': signal_result['confidence'],
            'reasons': signal_result['reasons'],

            # Classificação de curvatura
            'curvature_class': self._classify_curvature(current_ricci),

            # Metadados
            'n_observations': n,
            'last_closed_idx': last_closed_idx,  # NOVO V3.0: índice da última barra fechada
            'current_price': prices[last_closed_idx],  # CORREÇÃO V3.0: preço da última FECHADA
            'jax_available': JAX_AVAILABLE
        }

    def _classify_curvature(self, ricci: float) -> dict:
        """
        Classifica a curvatura do espaço-tempo financeiro
        """
        if np.abs(ricci) < 0.01:
            return {
                'class': 'PLANO',
                'description': 'Espaço plano - Mercado eficiente/calmo',
                'volatility': 'BAIXA'
            }
        elif ricci > 0:
            return {
                'class': 'ESFERICO',
                'description': 'Espaço esférico - Convergência de ordens (Consolidação)',
                'volatility': 'MODERADA'
            }
        elif ricci > self.ricci_collapse_threshold:
            return {
                'class': 'HIPERBOLICO_LEVE',
                'description': 'Espaço hiperbólico leve - Divergência moderada',
                'volatility': 'MODERADA-ALTA'
            }
        else:
            return {
                'class': 'HIPERBOLICO_EXTREMO',
                'description': 'Espaço hiperbólico extremo - Divergência explosiva',
                'volatility': 'EXTREMA'
            }

    def _generate_signal(self, ricci: float, tidal: float, distance: float,
                         ricci_collapsing: bool, crossing_horizon: bool,
                         geodesic_direction: float,
                         history_length: int = 0,
                         ricci_history: list = None) -> dict:
        """
        Gera sinal de trading baseado na análise gravitacional

        Condição de Disparo:
        1. O Escalar de Ricci (R) cai verticalmente (Colapso geométrico).
        2. A distância do preço atual para o centro de massa se aproxima de r_s.
        3. Estamos cruzando o "Horizonte de Eventos".
        4. Direção: Seguir o sinal da componente temporal da Geodésica (d²P/dτ²).

        CORREÇÃO V3.3: Parâmetro history_length adicionado para validar se há
        histórico suficiente antes de gerar sinais. Sinais gerados com histórico
        insuficiente não são confiáveis.

        CORREÇÃO V3.4: Parâmetro ricci_history adicionado para usar percentil
        dinâmico ao invés de threshold fixo. Isso garante que o filtro de Ricci
        seja efetivo na prática.
        """
        # CORREÇÃO V3.3: Verificar histórico mínimo ANTES de gerar sinal
        # Sinais com histórico insuficiente não são confiáveis
        if history_length < self.min_history_for_signal:
            return {
                'signal': 0,
                'signal_name': 'NEUTRO',
                'confidence': 0.0,
                'reasons': [f'Histórico insuficiente ({history_length} < {self.min_history_for_signal})'],
                'conditions_met': 0
            }

        signal = 0
        signal_name = "NEUTRO"
        confidence = 0.0
        reasons = []

        conditions_met = 0

        # CORREÇÃO V3.4: Condição 1 reformulada - Ricci em colapso
        # Usar percentil dinâmico se houver histórico suficiente
        # Isso garante que o filtro seja efetivo independente da escala
        ricci_is_collapsing = ricci_collapsing  # Flag de mudança rápida
        ricci_below_threshold = False

        if ricci_history is not None and len(ricci_history) >= 10:
            # Usar P25 como threshold dinâmico - Ricci deve estar no quartil inferior
            valid_ricci = [r for r in ricci_history if not np.isnan(r)]
            if len(valid_ricci) >= 10:
                dynamic_threshold = np.percentile(valid_ricci, 25)
                ricci_below_threshold = ricci < dynamic_threshold
        else:
            # Fallback para threshold fixo se não há histórico suficiente
            ricci_below_threshold = ricci < self.ricci_collapse_threshold

        if ricci_is_collapsing or ricci_below_threshold:
            conditions_met += 1
            reasons.append("Colapso do Escalar de Ricci (curvatura negativa)")

        # Condição 2: Alta força de maré
        if tidal > self.tidal_force_threshold:
            conditions_met += 1
            reasons.append("Força de maré elevada (spread sendo rasgado)")

        # Condição 3: Cruzando horizonte de eventos
        if crossing_horizon:
            conditions_met += 1
            reasons.append("Cruzando Horizonte de Eventos (ponto de não-retorno)")

        # Determinar sinal se condições suficientes
        if conditions_met >= 2:
            if geodesic_direction > 0:
                signal = 1
                signal_name = "LONG (Singularidade Gravitacional)"
                reasons.append("Geodésica aponta para cima")
            elif geodesic_direction < 0:
                signal = -1
                signal_name = "SHORT (Singularidade Gravitacional)"
                reasons.append("Geodésica aponta para baixo")
            else:
                signal = 0
                signal_name = "ALERTA (Singularidade detectada)"
                reasons.append("Direção da geodésica indeterminada")

            confidence = conditions_met / 3

        elif conditions_met == 1:
            signal = 0
            signal_name = "ALERTA (Pré-singularidade)"
            confidence = 0.3

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'reasons': reasons,
            'conditions_met': conditions_met
        }

    def _apply_step_function_causal(self, n: int, indices: list, values: list) -> np.ndarray:
        """
        CORREÇÃO CRÍTICA: Step Function (Zero-Order Hold) - 100% CAUSAL

        Substitui np.interp() que causa LOOK-AHEAD BIAS.

        A step function usa apenas o ÚLTIMO valor calculado até cada índice,
        nunca olhando para valores futuros.

        CORREÇÃO V3.2: Índices antes do primeiro cálculo mantêm NaN
        O valor 0.0 NÃO é neutro para o Escalar de Ricci (que pode ser
        positivo, negativo ou zero significativamente).

        Exemplo:
        - indices = [5, 15, 25]
        - values = [1.0, 2.0, 3.0]
        - Para índices 0-4: NaN (sem dados ainda)
        - Para índices 5-14: usa valor 1.0
        - Para índices 15-24: usa valor 2.0
        - Para índices 25+: usa valor 3.0

        Args:
            n: Tamanho total do array de saída
            indices: Lista de índices onde valores foram calculados
            values: Lista de valores calculados

        Returns:
            Array de tamanho n com step function aplicada (causal)
        """
        # CORREÇÃO V3.2: Inicializar com NaN ao invés de zeros
        # NaN indica explicitamente "sem dados" e não será confundido
        # com um valor calculado
        result = np.full(n, np.nan)

        if not indices or not values:
            return result

        # Converter para arrays numpy para eficiência
        indices_arr = np.array(indices)
        values_arr = np.array(values)

        for i in range(n):
            # Encontrar o índice do último valor calculado que é <= i
            # searchsorted retorna onde i seria inserido para manter ordem
            # Com 'right', retorna posição após elementos iguais
            pos = np.searchsorted(indices_arr, i, side='right')

            if pos == 0:
                # CORREÇÃO V3.2: Nenhum valor calculado ANTES deste índice
                # Manter NaN - NÃO substituir por 0.0!
                # O código que usa esta série deve verificar NaN
                pass  # result[i] já é NaN
            else:
                # Usar o último valor calculado (pos-1)
                result[i] = values_arr[pos - 1]

        # CORREÇÃO V3.2: NÃO fazer forward-fill com 0.0
        # NaN antes do primeiro cálculo é o comportamento correto
        # O chamador deve tratar NaN apropriadamente

        return result

    def _apply_ema_causal(self, series: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        CORREÇÃO #1: Aplica EMA (Exponential Moving Average) CAUSAL

        Substitui gaussian_filter1d que é NÃO-CAUSAL (olha para o futuro).
        EMA só usa dados PASSADOS: EMA[t] = alpha * X[t] + (1-alpha) * EMA[t-1]

        CORREÇÃO V3.3: Tratamento robusto de NaN na entrada
        - Se valor atual é NaN, mantém EMA anterior
        - Se EMA anterior é NaN, usa valor atual
        - Se ambos NaN, propaga NaN

        Args:
            series: Série temporal a suavizar
            alpha: Fator de suavização (0-1). Maior = menos suavização.

        Returns:
            Série suavizada de forma causal
        """
        if len(series) == 0:
            return series

        result = np.zeros_like(series, dtype=np.float64)

        # CORREÇÃO V3.3: Encontrar primeiro valor não-NaN para inicializar
        first_valid_idx = 0
        for i in range(len(series)):
            if not np.isnan(series[i]):
                first_valid_idx = i
                break
        else:
            # Todos NaN - retornar série de NaN
            return np.full_like(series, np.nan, dtype=np.float64)

        # Preencher com NaN até primeiro valor válido
        result[:first_valid_idx] = np.nan
        result[first_valid_idx] = series[first_valid_idx]

        # CORREÇÃO V3.3: EMA com tratamento de NaN
        for i in range(first_valid_idx + 1, len(series)):
            if np.isnan(series[i]):
                # Se valor atual é NaN, manter EMA anterior
                result[i] = result[i-1]
            elif np.isnan(result[i-1]):
                # Se EMA anterior é NaN mas valor atual não, usar valor atual
                result[i] = series[i]
            else:
                # Caso normal: calcular EMA
                result[i] = alpha * series[i] + (1 - alpha) * result[i-1]

        return result

    def _empty_result(self) -> dict:
        """
        Retorna resultado vazio quando não há dados suficientes

        CORREÇÃO V3.2: Unificado signal_name para 'NEUTRO' (era 'HOLD')
        para consistência com _generate_signal()
        """
        return {
            'output_vector': [0.0, 0.0, 0.0],
            'Ricci_Scalar': 0.0,
            'Tidal_Force_Magnitude': 0.0,
            'Event_Horizon_Distance': 0.0,
            'ricci_series': np.array([]),
            'tidal_series': np.array([]),
            'distance_series': np.array([]),
            'ricci_collapsing': False,
            'crossing_horizon': False,
            'geodesic_direction': 0,
            'signal': 0,
            'signal_name': 'NEUTRO',  # CORREÇÃO V3.2: Era 'HOLD', agora consistente
            'confidence': 0.0,
            'reasons': ['Dados insuficientes'],
            'curvature_class': {'class': 'UNKNOWN', 'description': 'Dados insuficientes', 'volatility': 'N/A'},
            'n_observations': 0,
            'last_closed_idx': 0,  # CORREÇÃO V3.2: Adicionado campo faltante
            'current_price': 0.0,
            'jax_available': JAX_AVAILABLE
        }

    def get_signal(self, prices: np.ndarray,
                   bid_volumes: np.ndarray = None,
                   ask_volumes: np.ndarray = None) -> int:
        """
        Retorna sinal simplificado:
        1 = LONG
        0 = NEUTRO
        -1 = SHORT
        """
        result = self.analyze(prices, bid_volumes, ask_volumes)
        return result['signal']


# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# =============================================================================

def plot_dsg_analysis(prices: np.ndarray,
                      bid_volumes: np.ndarray = None,
                      ask_volumes: np.ndarray = None,
                      save_path: str = None):
    """
    Visualização 3D: Plotar a "Superfície de Curvatura" (Embedding Diagram).
    Um funil de gravidade se formando no gráfico. O preço é uma bola orbitando o funil.
    Se cair, é trade.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
    except ImportError:
        print("matplotlib não disponível para visualização")
        return None

    # Criar detector e analisar
    dsg = DetectorSingularidadeGravitacional(
        ricci_collapse_threshold=-0.5,
        tidal_force_threshold=0.1
    )
    result = dsg.analyze(prices, bid_volumes, ask_volumes)

    # Criar figura
    fig = plt.figure(figsize=(18, 14))

    # Layout
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1.5])

    time = np.arange(len(prices))

    # =========================================================================
    # Plot 1: Superfície de Curvatura 3D (Embedding Diagram)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    # Criar superfície tipo funil baseada na curvatura
    n_grid = 50
    theta = np.linspace(0, 2 * np.pi, n_grid)
    r = np.linspace(0.1, 2, n_grid)
    THETA, R = np.meshgrid(theta, r)

    # Curvatura determina a profundidade do funil
    ricci_current = result['Ricci_Scalar']
    depth_factor = np.abs(ricci_current) * 10 if ricci_current < 0 else 0

    # Superfície do funil
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    Z = -depth_factor * np.log(R + 0.1)  # Funil logarítmico

    # Plotar superfície
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')

    # Plotar "partícula" (preço atual)
    price_normalized = (prices[-1] - np.mean(prices)) / (np.std(prices) + 1e-10)
    ball_r = 1.0 + 0.3 * price_normalized
    ball_theta = (time[-1] % 100) / 100 * 2 * np.pi
    ball_x = ball_r * np.cos(ball_theta)
    ball_y = ball_r * np.sin(ball_theta)
    ball_z = -depth_factor * np.log(ball_r + 0.1) + 0.5

    ax1.scatter([ball_x], [ball_y], [ball_z], c='red', s=200, marker='o',
               label='Preço Atual')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Curvatura')
    ax1.set_title('Superfície de Curvatura (Embedding Diagram)', fontsize=12)

    # Info
    curvature = result['curvature_class']
    info = (
        f"R = {ricci_current:.4f}\n"
        f"Classe: {curvature['class']}\n"
        f"Vol: {curvature['volatility']}"
    )
    ax1.text2D(0.02, 0.95, info, transform=ax1.transAxes, fontsize=9,
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # =========================================================================
    # Plot 2: Preço com Horizonte de Eventos
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.plot(time, prices, 'b-', linewidth=1.5, label='Preço', zorder=3)

    # Marcar horizonte de eventos
    distance_series = result['distance_series']
    if len(distance_series) > 0:
        horizon_mask = distance_series <= 0

        if np.any(horizon_mask):
            ax2.fill_between(time, prices.min(), prices.max(),
                            where=horizon_mask, alpha=0.3, color='black',
                            label='Dentro do Horizonte')

    # Sinal
    signal = result['signal']
    if signal == 1:
        ax2.scatter([time[-1]], [prices[-1]], c='green', s=300, marker='^',
                   zorder=5, label='LONG')
    elif signal == -1:
        ax2.scatter([time[-1]], [prices[-1]], c='red', s=300, marker='v',
                   zorder=5, label='SHORT')

    ax2.set_xlabel('Tempo')
    ax2.set_ylabel('Preço')
    ax2.set_title(f'Preço e Horizonte de Eventos | Sinal: {result["signal_name"]}', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 3: Escalar de Ricci (R)
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    ricci_series = result['ricci_series']

    if len(ricci_series) > 0:
        ax3.plot(time, ricci_series, 'purple', linewidth=1.5, label='Escalar de Ricci R')
        ax3.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='R=0 (Plano)')
        ax3.axhline(y=dsg.ricci_collapse_threshold, color='red', linestyle='--',
                   alpha=0.7, label=f'Threshold ({dsg.ricci_collapse_threshold})')

        # Colorir regiões
        ax3.fill_between(time, ricci_series, 0, where=ricci_series > 0,
                        alpha=0.3, color='blue', label='Esférico (Consolidação)')
        ax3.fill_between(time, ricci_series, 0, where=ricci_series < 0,
                        alpha=0.3, color='red', label='Hiperbólico (Volatilidade)')

    ax3.set_xlabel('Tempo')
    ax3.set_ylabel('Escalar de Ricci R')
    ax3.set_title('Curvatura do Espaço-Tempo Financeiro', fontsize=12)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 4: Força de Maré (Tidal Force)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    tidal_series = result['tidal_series']

    if len(tidal_series) > 0:
        ax4.fill_between(time, 0, tidal_series, alpha=0.5, color='orange')
        ax4.plot(time, tidal_series, 'orange', linewidth=1.5, label='Força de Maré')
        ax4.axhline(y=dsg.tidal_force_threshold, color='red', linestyle='--',
                   label=f'Threshold ({dsg.tidal_force_threshold})')

        # Marcar regiões de alta força de maré
        high_tidal = tidal_series > dsg.tidal_force_threshold
        if np.any(high_tidal):
            ax4.fill_between(time, 0, tidal_series.max(), where=high_tidal,
                            alpha=0.2, color='red', label='Spread sendo rasgado')

    ax4.set_xlabel('Tempo')
    ax4.set_ylabel('|Força de Maré|')
    ax4.set_title('Desvio Geodésico (Forças de Maré)', fontsize=12)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 5: Distância ao Horizonte de Eventos
    # =========================================================================
    ax5 = fig.add_subplot(gs[2, 0])

    if len(distance_series) > 0:
        ax5.plot(time, distance_series, 'black', linewidth=1.5,
                label='Distância ao Horizonte')
        ax5.axhline(y=0, color='red', linestyle='-', linewidth=2,
                   label='Horizonte de Eventos')
        ax5.fill_between(time, distance_series, 0, where=distance_series < 0,
                        alpha=0.5, color='black', label='Dentro do Horizonte')

        # Marcar ponto atual
        ax5.scatter([time[-1]], [distance_series[-1]], c='red', s=100, zorder=5)

    ax5.set_xlabel('Tempo')
    ax5.set_ylabel('Distância (r - r_s)')
    ax5.set_title('Aproximação do Horizonte de Eventos', fontsize=12)
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 6: Tensor Métrico (Heatmap)
    # =========================================================================
    ax6 = fig.add_subplot(gs[2, 1])

    # Obter tensor métrico do último ponto
    coords = dsg._coords_history[-1] if dsg._coords_history else np.zeros(4)
    g = dsg.metric_calc.construct_metric_tensor(
        coords[0], coords[1], coords[2], coords[3]
    )

    im = ax6.imshow(g, cmap='RdBu_r', aspect='equal')
    plt.colorbar(im, ax=ax6, label='g_μν')

    # Labels
    labels = ['t', 'P', 'V_bid', 'V_ask']
    ax6.set_xticks(range(4))
    ax6.set_yticks(range(4))
    ax6.set_xticklabels(labels)
    ax6.set_yticklabels(labels)

    # Adicionar valores
    for i in range(4):
        for j in range(4):
            ax6.text(j, i, f'{g[i,j]:.3f}', ha='center', va='center',
                    fontsize=8, color='white' if np.abs(g[i,j]) > 0.5 else 'black')

    ax6.set_title('Tensor Métrico Financeiro g_μν', fontsize=12)

    # =========================================================================
    # Resumo
    # =========================================================================
    output = result['output_vector']
    reasons = '; '.join(result['reasons'][:2]) if result['reasons'] else 'Nenhuma'

    summary = (
        f"DSG Output: [R={output[0]:.4f}, F_tidal={output[1]:.4f}, d_EH={output[2]:.4f}] | "
        f"Curvatura: {result['curvature_class']['class']} | "
        f"Sinal: {result['signal_name']} | "
        f"Condições: {result['reasons'][0] if result['reasons'] else 'Nenhuma'}"
    )

    fig.text(0.5, 0.01, summary, fontsize=10, ha='center',
            bbox=dict(boxstyle='round',
                     facecolor='purple' if result['signal'] != 0 else 'lightblue',
                     alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DETECTOR DE SINGULARIDADE GRAVITACIONAL (DSG)")
    print("Astrofísica Computacional / Relatividade Numérica aplicada a Finanças")
    print("=" * 80)
    print(f"\nJAX disponível: {JAX_AVAILABLE}")

    # Gerar dados simulados com evento de singularidade
    np.random.seed(42)
    n_points = 300

    # Fase 1: Mercado calmo (espaço plano)
    calm = 1.1000 + 0.0002 * np.cumsum(np.random.randn(100))

    # Fase 2: Consolidação (espaço esférico)
    consolidation = calm[-1] + 0.0001 * np.cumsum(np.random.randn(80))

    # Fase 3: Singularidade gravitacional (espaço hiperbólico)
    # Movimento explosivo após "cruzar o horizonte de eventos"
    singularity = consolidation[-1] + np.linspace(0, 0.02, 50) + 0.001 * np.cumsum(np.random.randn(50))

    # Fase 4: Pós-singularidade
    post = singularity[-1] + 0.0003 * np.cumsum(np.random.randn(70))

    prices = np.concatenate([calm, consolidation, singularity, post])

    # Volumes simulados
    bid_volumes = np.abs(np.diff(prices, prepend=prices[0])) * 50000 + np.random.rand(len(prices)) * 1000
    ask_volumes = np.abs(np.diff(prices, prepend=prices[0])) * 50000 + np.random.rand(len(prices)) * 1000

    # Aumentar volume na singularidade
    singularity_start = len(calm) + len(consolidation)
    bid_volumes[singularity_start:singularity_start+50] *= 3
    ask_volumes[singularity_start:singularity_start+50] *= 3

    print(f"\nDados simulados: {len(prices)} pontos")
    print(f"Preço inicial: {prices[0]:.5f}")
    print(f"Preço final: {prices[-1]:.5f}")
    print(f"Evento de singularidade: ponto {singularity_start}")

    # Criar detector
    dsg = DetectorSingularidadeGravitacional(
        ricci_collapse_threshold=-0.3,
        tidal_force_threshold=0.05,
        lookback_window=30
    )

    # Executar análise
    print("\n" + "-" * 40)
    print("Executando análise DSG...")
    print("-" * 40)

    result = dsg.analyze(prices, bid_volumes, ask_volumes)

    # Mostrar resultados
    print("\n📊 VETOR DE SAÍDA:")
    print(f"   [Ricci_Scalar, Tidal_Force_Magnitude, Event_Horizon_Distance]")
    output = result['output_vector']
    print(f"   [{output[0]:.4f}, {output[1]:.4f}, {output[2]:.4f}]")

    print("\n🌌 CURVATURA DO ESPAÇO-TEMPO:")
    curvature = result['curvature_class']
    print(f"   Classe: {curvature['class']}")
    print(f"   Descrição: {curvature['description']}")
    print(f"   Volatilidade: {curvature['volatility']}")

    print("\n🌊 FORÇA DE MARÉ:")
    print(f"   Magnitude: {result['Tidal_Force_Magnitude']:.6f}")
    print(f"   Threshold: {dsg.tidal_force_threshold}")
    print(f"   Alta: {'SIM' if result['Tidal_Force_Magnitude'] > dsg.tidal_force_threshold else 'NAO'}")

    print("\n⚫ HORIZONTE DE EVENTOS:")
    print(f"   Distância: {result['Event_Horizon_Distance']:.6f}")
    print(f"   Cruzando: {'SIM' if result['crossing_horizon'] else 'NAO'}")
    print(f"   Ricci colapsando: {'SIM' if result['ricci_collapsing'] else 'NAO'}")

    print("\n🎯 SINAL:")
    print(f"   Sinal: {result['signal_name']}")
    print(f"   Confiança: {result['confidence']:.2%}")
    print(f"   Direção Geodésica: {result['geodesic_direction']}")
    print(f"   Razões: {', '.join(result['reasons']) if result['reasons'] else 'Nenhuma'}")

    print("\n" + "=" * 80)
    if result['signal'] == 1:
        print("SINGULARIDADE GRAVITACIONAL - LONG!")
        print("   O preço cruzou o horizonte de eventos. Movimento explosivo para cima.")
    elif result['signal'] == -1:
        print("SINGULARIDADE GRAVITACIONAL - SHORT!")
        print("   O preço cruzou o horizonte de eventos. Movimento explosivo para baixo.")
    elif 'ALERTA' in result['signal_name']:
        print("PRE-SINGULARIDADE DETECTADA!")
        print("   Curvatura do espaço-tempo se intensificando. Monitorar.")
    else:
        print("ESPACO-TEMPO ESTAVEL")
        print(f"   Curvatura: {curvature['class']}")
    print("=" * 80)

    # Gerar visualização
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print("\nGerando visualização...")
        fig = plot_dsg_analysis(prices, bid_volumes, ask_volumes,
                                save_path='/tmp/dsg_analysis.png')
        print("Visualização salva como '/tmp/dsg_analysis.png'")
        plt.close()
    except Exception as e:
        print(f"\nNão foi possível gerar visualização: {e}")

    print("\n✅ Teste do DSG concluído com sucesso!")
