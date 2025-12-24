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

VERSÃO CORRIGIDA - SEM LOOK-AHEAD BIAS
======================================
Correções aplicadas:
1. Substituído gaussian_filter1d (não-causal) por EMA causal
2. Direção da geodésica calculada apenas com barras FECHADAS
3. Adicionado suporte para modo "online" sem look-ahead
"""

import numpy as np
# REMOVIDO: from scipy.ndimage import gaussian_filter1d (era não-causal, substituído por EMA)
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

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
        """
        if len(coords_history) < 2:
            return np.array([1.0, 0.0, 0.0, 0.0])  # Apenas passagem do tempo

        # Derivada numérica das coordenadas
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
                 ricci_collapse_threshold: float = -0.5,
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
                      bid_vol: float, ask_vol: float) -> dict:
        """
        Analisa um único ponto no espaço-tempo financeiro
        """
        coords = self._prepare_coordinates(t, price, bid_vol, ask_vol)

        # 1. Tensor Métrico
        g = self.metric_calc.construct_metric_tensor(
            coords[0], coords[1], coords[2], coords[3]
        )

        # 2. Escalar de Ricci
        R = self.ricci_calc.compute_ricci_scalar(coords)

        # 3. Vetor tangente (se temos histórico)
        self._coords_history.append(coords)
        if len(self._coords_history) > self.lookback_window:
            self._coords_history = self._coords_history[-self.lookback_window:]

        T = self.geodesic_calc.compute_tangent_vector(
            np.array(self._coords_history)
        )

        # 4. Vetor de separação e força de maré
        xi = self.geodesic_calc.compute_separation_vector(bid_vol, ask_vol)
        tidal_magnitude = self.geodesic_calc.compute_tidal_force_magnitude(coords, T, xi)

        # 5. Horizonte de eventos
        accumulated_vol = bid_vol + ask_vol
        r_s = self.horizon_calc.compute_schwarzschild_radius(accumulated_vol)

        # Centro de massa simplificado (usando preço atual e histórico)
        if len(self._coords_history) > 1:
            prices_hist = [c[1] for c in self._coords_history[-20:]]
            vols_hist = [np.exp(c[2]) + np.exp(c[3]) for c in self._coords_history[-20:]]
            com = self.horizon_calc.compute_volume_center_of_mass(
                np.array(prices_hist), np.array(vols_hist)
            )
        else:
            com = price

        eh_distance = self.horizon_calc.compute_event_horizon_distance(price, com, r_s)

        # Atualizar histórico
        self._ricci_history.append(R)
        self._distance_history.append(eh_distance)

        if len(self._ricci_history) > self.lookback_window:
            self._ricci_history = self._ricci_history[-self.lookback_window:]
            self._distance_history = self._distance_history[-self.lookback_window:]

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
        """
        n = len(prices)

        if n < 10:
            return self._empty_result()

        # CORREÇÃO C1: Gerar volumes DETERMINÍSTICOS se não fornecidos
        # ANTES: Usava np.random.rand() que tornava backtests não-reproduzíveis
        # DEPOIS: Volumes baseados apenas na variação de preço (determinístico)
        if bid_volumes is None:
            price_changes = np.abs(np.diff(prices, prepend=prices[0]))
            bid_volumes = price_changes * 1000 + 50  # Valor base fixo, sem random
        if ask_volumes is None:
            price_changes = np.abs(np.diff(prices, prepend=prices[0]))
            ask_volumes = price_changes * 1000 + 50  # Valor base fixo, sem random

        # Resetar histórico
        self._ricci_history = []
        self._distance_history = []
        self._coords_history = []

        # Arrays para resultados
        ricci_series = np.zeros(n)
        tidal_series = np.zeros(n)
        distance_series = np.zeros(n)

        # CORREÇÃO CRÍTICA: Subsampling com STEP FUNCTION (Zero-Order Hold)
        # =================================================================
        # PROBLEMA ANTERIOR: np.interp() causa LOOK-AHEAD BIAS porque a
        # interpolação linear usa pontos FUTUROS para calcular valores intermediários.
        # Exemplo: Para obter valor no índice 5, interp usa índices 0 e 10,
        # mas o índice 10 ainda não aconteceu no momento 5!
        #
        # SOLUÇÃO: Step Function (Zero-Order Hold) - 100% CAUSAL
        # Cada barra usa o valor do ÚLTIMO ponto calculado, sem olhar para frente.
        # =================================================================
        step = max(1, n // 100)  # Máximo 100 pontos para cálculo completo

        # Armazenar pontos calculados
        calculated_indices = []
        calculated_ricci = []
        calculated_tidal = []
        calculated_distance = []

        for i in range(0, n, step):
            result = self.analyze_point(
                i, prices[i], bid_volumes[i], ask_volumes[i]
            )

            calculated_indices.append(i)
            calculated_ricci.append(result['ricci_scalar'])
            calculated_tidal.append(result['tidal_force'])
            calculated_distance.append(result['event_horizon_distance'])

        # Adicionar último ponto se não foi calculado
        if calculated_indices[-1] != n - 1:
            result = self.analyze_point(
                n - 1, prices[n - 1], bid_volumes[n - 1], ask_volumes[n - 1]
            )
            calculated_indices.append(n - 1)
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

        # Valores atuais
        current_ricci = ricci_series[-1]
        current_tidal = tidal_series[-1]
        current_distance = distance_series[-1]

        # Detectar colapso de Ricci
        if len(self._ricci_history) > 5:
            ricci_change = np.diff(self._ricci_history[-5:])
            ricci_collapsing = np.mean(ricci_change) < -0.1
        else:
            ricci_collapsing = False

        # Detectar cruzamento do horizonte
        crossing_horizon = self.horizon_calc.is_crossing_event_horizon(
            current_distance, self._distance_history
        )

        # CORREÇÃO #2: Determinar direção da geodésica usando apenas barras FECHADAS
        # ANTES (ERRADO): Usava self._coords_history[-3:] que inclui a barra atual
        # DEPOIS (CORRETO): Usar [-4:-1] para excluir a barra atual (ainda não fechou)
        #
        # No momento da decisão:
        # - _coords_history[-1] = barra atual (close ainda pode mudar em tempo real)
        # - _coords_history[-2] = última barra fechada
        # - _coords_history[-4] = 3 barras atrás (fechada)
        if len(self._coords_history) >= 4:
            # Usar apenas barras COMPLETAMENTE FECHADAS
            prices_past = [c[1] for c in self._coords_history[-4:-1]]  # Exclui barra atual
            geodesic_direction = np.sign(prices_past[-1] - prices_past[0])
        elif len(self._coords_history) >= 2:
            # Fallback com menos dados (ainda exclui barra atual)
            prices_past = [c[1] for c in self._coords_history[:-1]]
            if len(prices_past) >= 2:
                geodesic_direction = np.sign(prices_past[-1] - prices_past[0])
            else:
                geodesic_direction = 0
        else:
            geodesic_direction = 0

        # Gerar sinal
        signal_result = self._generate_signal(
            current_ricci, current_tidal, current_distance,
            ricci_collapsing, crossing_horizon, geodesic_direction
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
            'current_price': prices[-1],
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
                         geodesic_direction: float) -> dict:
        """
        Gera sinal de trading baseado na análise gravitacional

        Condição de Disparo:
        1. O Escalar de Ricci (R) cai verticalmente (Colapso geométrico).
        2. A distância do preço atual para o centro de massa se aproxima de r_s.
        3. Estamos cruzando o "Horizonte de Eventos".
        4. Direção: Seguir o sinal da componente temporal da Geodésica (d²P/dτ²).
        """
        signal = 0
        signal_name = "NEUTRO"
        confidence = 0.0
        reasons = []

        conditions_met = 0

        # Condição 1: Ricci colapsando
        if ricci_collapsing or ricci < self.ricci_collapse_threshold:
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

        Exemplo:
        - indices = [0, 10, 20]
        - values = [1.0, 2.0, 3.0]
        - Para índice 5: usa valor 1.0 (último calculado até 5)
        - Para índice 15: usa valor 2.0 (último calculado até 15)

        Args:
            n: Tamanho total do array de saída
            indices: Lista de índices onde valores foram calculados
            values: Lista de valores calculados

        Returns:
            Array de tamanho n com step function aplicada (causal)
        """
        result = np.zeros(n)

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
                # CORREÇÃO CRÍTICA: Nenhum valor calculado ANTES deste índice
                # Usar np.nan para indicar "sem dados ainda" - 100% causal
                # ANTES (ERRADO): result[i] = values_arr[0] ← Isso é look-ahead!
                # O values_arr[0] corresponde ao índice indices_arr[0], que pode ser > i
                result[i] = np.nan
            else:
                # Usar o último valor calculado (pos-1)
                result[i] = values_arr[pos - 1]

        # Substituir NaN pelo primeiro valor válido (forward-fill do primeiro calculado)
        # Isso é causal porque só preenchemos APÓS o primeiro cálculo real
        if np.isnan(result[0]) and len(values_arr) > 0:
            first_valid_idx = indices_arr[0]
            # Preencher do início até o primeiro cálculo com NaN ou valor neutro
            result[:first_valid_idx] = 0.0  # Valor neutro antes do primeiro cálculo

        return result

    def _apply_ema_causal(self, series: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        CORREÇÃO #1: Aplica EMA (Exponential Moving Average) CAUSAL

        Substitui gaussian_filter1d que é NÃO-CAUSAL (olha para o futuro).
        EMA só usa dados PASSADOS: EMA[t] = alpha * X[t] + (1-alpha) * EMA[t-1]

        Args:
            series: Série temporal a suavizar
            alpha: Fator de suavização (0-1). Maior = menos suavização.

        Returns:
            Série suavizada de forma causal
        """
        if len(series) == 0:
            return series

        result = np.zeros_like(series)
        result[0] = series[0]

        for i in range(1, len(series)):
            result[i] = alpha * series[i] + (1 - alpha) * result[i-1]

        return result

    def _empty_result(self) -> dict:
        """Retorna resultado vazio quando não há dados suficientes"""
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
            'signal_name': 'HOLD',
            'confidence': 0.0,
            'reasons': ['Dados insuficientes'],
            'curvature_class': {'class': 'UNKNOWN', 'description': 'Dados insuficientes', 'volatility': 'N/A'},
            'n_observations': 0,
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
