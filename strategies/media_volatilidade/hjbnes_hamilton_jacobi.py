"""
================================================================================
HAMILTON-JACOBI-BELLMAN NASH EQUILIBRIUM SOLVER (HJB-NES)
Indicador de Forex baseado em Mean Field Games (MFG)
================================================================================

Este indicador nao preve o preco. Ele calcula, a cada tick, qual e o
Preco de Equilibrio de Nash do sistema. Em media volatilidade, o mercado
SEMPRE converge para o equilibrio de Nash. Se o preco real desviar disso,
e uma anomalia matematica que SERA corrigida.

O Conceito: O Problema do Agente Representativo
Em vez de modelar cada trader individualmente (impossivel), a MFG aproxima o
mercado por um continuum de agentes. Cada agente tenta maximizar seu lucro
(funcao utilidade) sabendo que todos os outros estao tentando fazer o mesmo.

O sistema e governado por um par de Equacoes Diferenciais Parciais (PDEs)
acopladas que funcionam em direcoes opostas no tempo:

1. Equacao de Hamilton-Jacobi-Bellman (HJB) - Backward in Time
   Descreve a estrategia otima do "agente medio". Olha do futuro para o
   presente para determinar a melhor acao (comprar/vender) agora.

2. Equacao de Fokker-Planck-Kolmogorov (FPK) - Forward in Time
   Descreve como a massa de traders (distribuicao m) evolui no tempo,
   dada a estrategia otima calculada na HJB.

Por que usar MFG (Mean Field Games)?
1. Antecipacao de Crowd Behavior: A maioria dos indicadores segue o preco.
   A MFG modela o MOTIVO pelo qual o preco se move (a interacao dos agentes).
2. Resiliencia em Media Volatilidade: Em regimes extremos (panico), as equacoes
   quebram porque os agentes agem irracionalmente. Em media volatilidade, os
   agentes institucionais (robos de bancos) operam exatamente segundo essas
   equacoes de otimizacao de custo.
3. Arbitragem Estrutural: Voce esta arbitrando a diferenca entre o estado atual
   do mercado e o estado de equilibrio matematico. E a forma mais pura de Alpha.

Nota de Programacao: Este calculo e extremamente pesado O(N^3) ou mais.
Usa numpy vetorizado para performance.

Autor: Gerado por Claude AI
Versao: 1.0.0
================================================================================
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

class MFGSignalType(Enum):
    """Tipos de sinais de trading"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    NO_CONVERGENCE = "NO_CONVERGENCE"  # Solver nao convergiu


class MarketRegime(Enum):
    """Regime do mercado"""
    EQUILIBRIUM = "EQUILIBRIUM"        # No equilibrio de Nash
    OVERVALUED = "OVERVALUED"          # Acima do equilibrio
    UNDERVALUED = "UNDERVALUED"        # Abaixo do equilibrio
    UNSTABLE = "UNSTABLE"              # Regime instavel


@dataclass
class HJBSolution:
    """Solucao da equacao Hamilton-Jacobi-Bellman"""
    value_function: np.ndarray      # u(x, t) - funcao de valor
    optimal_control: np.ndarray     # alpha*(x, t) - controle otimo
    x_grid: np.ndarray              # Grid espacial
    t_grid: np.ndarray              # Grid temporal


@dataclass
class FPKSolution:
    """Solucao da equacao Fokker-Planck-Kolmogorov"""
    density: np.ndarray             # m(x, t) - distribuicao de traders
    drift_field: np.ndarray         # Campo de drift resultante
    x_grid: np.ndarray
    t_grid: np.ndarray


@dataclass
class NashEquilibrium:
    """Equilibrio de Nash do sistema MFG"""
    equilibrium_price: float        # Preco de equilibrio
    equilibrium_drift: float        # Drift de equilibrio (v*)
    value_function: np.ndarray
    density: np.ndarray
    converged: bool
    iterations: int
    error: float


@dataclass
class MeanFieldDiscrepancy:
    """Discrepancia entre preco real e equilibrio de Nash"""
    real_drift: float               # v_real - velocidade atual
    nash_drift: float               # v* - drift de equilibrio
    discrepancy: float              # v_real - v*
    is_irrational: bool             # Movimento irracional detectado
    correction_direction: str       # "UP" ou "DOWN"


# ==============================================================================
# HAMILTONIANO DO MERCADO
# ==============================================================================

class MarketHamiltonian:
    """
    Definicao do Hamiltoniano (H)

    Modela o custo de transacao e a aversao ao risco. Em media volatilidade,
    o Hamiltoniano e quadratico (custo de impacto linear):

    H(p, q) = -1/(2*gamma) * q^2

    Onde gamma e a aversao ao risco do mercado (calibrada pela volatilidade
    historica recente).
    """

    def __init__(self, risk_aversion: float = 1.0):
        """
        Args:
            risk_aversion: gamma - parametro de aversao ao risco
        """
        self.risk_aversion = risk_aversion

    def H(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Hamiltoniano H(p, q) = -1/(2*gamma) * q^2

        Args:
            p: Posicao (preco)
            q: Momento (gradiente do valor)
        """
        return -1.0 / (2 * self.risk_aversion) * q**2

    def dH_dq(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Derivada do Hamiltoniano em relacao a q

        dH/dq = -q/gamma
        """
        return -q / self.risk_aversion

    def optimal_control(self, grad_u: np.ndarray) -> np.ndarray:
        """
        Controle otimo derivado do Hamiltoniano

        alpha* = -dH/dq = q/gamma = nabla_u/gamma
        """
        return grad_u / self.risk_aversion

    def calibrate_risk_aversion(self, returns: np.ndarray, window: int = 50) -> float:
        """
        Calibra gamma pela volatilidade historica recente

        gamma = 1 / (sigma^2 * fator_escala)
        """
        if len(returns) < window:
            return self.risk_aversion

        recent_vol = np.std(returns[-window:])

        # gamma inversamente proporcional a volatilidade
        # Alta vol = baixa aversao (agentes arriscam mais)
        # Baixa vol = alta aversao (agentes mais conservadores)
        gamma = 1.0 / (recent_vol**2 + 1e-10)
        gamma = np.clip(gamma, 0.1, 100.0)  # Limita para estabilidade

        return gamma


# ==============================================================================
# SOLVER HJB (BACKWARD IN TIME)
# ==============================================================================

class HJBSolver:
    """
    Solver para a Equacao de Hamilton-Jacobi-Bellman

    d_t u(x,t) + (sigma^2/2)*Delta_u(x,t) + H(x, nabla_u(x,t)) = F(x, m(x,t))

    Esta equacao descreve a estrategia otima do "agente medio". Ela olha do
    futuro para o presente para determinar a melhor acao (comprar/vender)
    agora para maximizar o valor futuro.

    - u(x, t): A funcao de valor (lucro esperado)
    - H: O Hamiltoniano (a dinamica do mercado)
    - m(x, t): A distribuicao da populacao de traders (densidade de ordens)
    - F: Custo de interacao (running cost)
    """

    def __init__(self,
                 n_space: int = 100,
                 n_time: int = 50,
                 x_min: float = -1.0,
                 x_max: float = 1.0,
                 T: float = 1.0):
        """
        Args:
            n_space: Pontos no grid espacial
            n_time: Pontos no grid temporal
            x_min, x_max: Limites do dominio espacial
            T: Horizonte de tempo
        """
        self.n_space = n_space
        self.n_time = n_time
        self.x_min = x_min
        self.x_max = x_max
        self.T = T

        # Grids
        self.dx = (x_max - x_min) / (n_space - 1)
        self.dt = T / (n_time - 1)
        self.x = np.linspace(x_min, x_max, n_space)
        self.t = np.linspace(0, T, n_time)

    def _compute_gradient(self, u: np.ndarray) -> np.ndarray:
        """Computa gradiente nabla_u via diferencas centrais"""
        grad = np.zeros_like(u)
        grad[1:-1] = (u[2:] - u[:-2]) / (2 * self.dx)
        grad[0] = (u[1] - u[0]) / self.dx
        grad[-1] = (u[-1] - u[-2]) / self.dx
        return grad

    def _compute_laplacian(self, u: np.ndarray) -> np.ndarray:
        """Computa Laplaciano Delta_u via diferencas finitas"""
        lap = np.zeros_like(u)
        lap[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / self.dx**2
        # Condicoes de contorno de Neumann (derivada zero)
        lap[0] = lap[1]
        lap[-1] = lap[-2]
        return lap

    def _running_cost(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Custo de running F(x, m)

        Modela o custo de estar em uma posicao x dado a distribuicao m
        Penaliza posicoes onde ha muita concentracao de traders
        """
        # F(x, m) = x^2 + log(m + epsilon) - interacao com a massa
        return x**2 + np.log(m + 1e-6)

    def _terminal_cost(self, x: np.ndarray) -> np.ndarray:
        """
        Custo terminal g(x)

        Condicao final em t = T
        """
        # g(x) = x^2 (penaliza desvio de zero)
        return x**2

    def solve(self,
             m: np.ndarray,
             sigma: float,
             hamiltonian: MarketHamiltonian) -> HJBSolution:
        """
        Resolve a HJB backward in time usando diferencas finitas implicitas

        Args:
            m: Distribuicao de traders m(x) no tempo atual
            sigma: Volatilidade
            hamiltonian: Objeto Hamiltoniano

        Returns:
            HJBSolution com funcao de valor e controle otimo
        """
        n_x = self.n_space
        n_t = self.n_time

        # Inicializa funcao de valor
        u = np.zeros((n_t, n_x))

        # Condicao terminal: u(x, T) = g(x)
        u[-1, :] = self._terminal_cost(self.x)

        # Interpola m para o grid se necessario
        if len(m) != n_x:
            interp = interp1d(np.linspace(0, 1, len(m)), m,
                            kind='linear', fill_value='extrapolate')
            m_grid = interp(np.linspace(0, 1, n_x))
        else:
            m_grid = m

        # Resolve backward in time
        # du/dt + (sigma^2/2)*d^2u/dx^2 + H(x, du/dx) = F(x, m)

        sigma2_half = 0.5 * sigma**2

        for n in range(n_t - 2, -1, -1):
            u_current = u[n + 1, :].copy()

            # Gradiente e Laplaciano
            grad_u = self._compute_gradient(u_current)
            lap_u = self._compute_laplacian(u_current)

            # Hamiltoniano
            H_val = hamiltonian.H(self.x, grad_u)

            # Running cost
            F_val = self._running_cost(self.x, m_grid)

            # Esquema implicito simplificado (Euler backward)
            # u^n = u^{n+1} - dt * (sigma^2/2 Delta_u + H - F)
            u[n, :] = u_current - self.dt * (sigma2_half * lap_u + H_val - F_val)

            # Condicoes de contorno (Dirichlet)
            u[n, 0] = self._terminal_cost(np.array([self.x_min]))[0]
            u[n, -1] = self._terminal_cost(np.array([self.x_max]))[0]

        # Controle otimo
        grad_u_final = self._compute_gradient(u[0, :])
        optimal_control = hamiltonian.optimal_control(grad_u_final)

        return HJBSolution(
            value_function=u,
            optimal_control=optimal_control,
            x_grid=self.x,
            t_grid=self.t
        )


# ==============================================================================
# SOLVER FPK (FORWARD IN TIME)
# ==============================================================================

class FPKSolver:
    """
    Solver para a Equacao de Fokker-Planck-Kolmogorov

    d_t m(x,t) - (sigma^2/2)*Delta_m(x,t) + div(m(x,t) * nabla_p H) = 0

    Esta equacao descreve como a massa de traders (distribuicao m) evolui
    no tempo, dada a estrategia otima calculada na HJB.

    - m(x, t): A distribuicao da populacao de traders (densidade de ordens)
    """

    def __init__(self,
                 n_space: int = 100,
                 n_time: int = 50,
                 x_min: float = -1.0,
                 x_max: float = 1.0,
                 T: float = 1.0):
        """
        Args:
            n_space: Pontos no grid espacial
            n_time: Pontos no grid temporal
            x_min, x_max: Limites do dominio espacial
            T: Horizonte de tempo
        """
        self.n_space = n_space
        self.n_time = n_time
        self.x_min = x_min
        self.x_max = x_max
        self.T = T

        # Grids
        self.dx = (x_max - x_min) / (n_space - 1)
        self.dt = T / (n_time - 1)
        self.x = np.linspace(x_min, x_max, n_space)
        self.t = np.linspace(0, T, n_time)

    def _compute_divergence(self, flux: np.ndarray) -> np.ndarray:
        """Computa divergencia div(flux)"""
        div = np.zeros_like(flux)
        div[1:-1] = (flux[2:] - flux[:-2]) / (2 * self.dx)
        div[0] = (flux[1] - flux[0]) / self.dx
        div[-1] = (flux[-1] - flux[-2]) / self.dx
        return div

    def _compute_laplacian(self, m: np.ndarray) -> np.ndarray:
        """Computa Laplaciano Delta_m"""
        lap = np.zeros_like(m)
        lap[1:-1] = (m[2:] - 2*m[1:-1] + m[:-2]) / self.dx**2
        lap[0] = lap[1]
        lap[-1] = lap[-2]
        return lap

    def solve(self,
             m0: np.ndarray,
             optimal_control: np.ndarray,
             sigma: float,
             hamiltonian: MarketHamiltonian) -> FPKSolution:
        """
        Resolve a FPK forward in time

        Args:
            m0: Distribuicao inicial m(x, 0)
            optimal_control: alpha*(x) do solver HJB
            sigma: Volatilidade
            hamiltonian: Objeto Hamiltoniano

        Returns:
            FPKSolution com densidade e campo de drift
        """
        n_x = self.n_space
        n_t = self.n_time

        # Inicializa densidade
        m = np.zeros((n_t, n_x))

        # Condicao inicial
        if len(m0) != n_x:
            interp = interp1d(np.linspace(0, 1, len(m0)), m0,
                            kind='linear', fill_value='extrapolate')
            m[0, :] = interp(np.linspace(0, 1, n_x))
        else:
            m[0, :] = m0

        # Normaliza
        m[0, :] = m[0, :] / (np.sum(m[0, :]) * self.dx + 1e-10)

        # Interpola controle otimo
        if len(optimal_control) != n_x:
            interp = interp1d(np.linspace(0, 1, len(optimal_control)), optimal_control,
                            kind='linear', fill_value='extrapolate')
            alpha = interp(np.linspace(0, 1, n_x))
        else:
            alpha = optimal_control

        sigma2_half = 0.5 * sigma**2

        # Resolve forward in time
        # dm/dt = (sigma^2/2)*Delta_m - div(m * v)
        # onde v = -dH/dq = alpha (velocidade induzida pelo controle)

        for n in range(n_t - 1):
            m_current = m[n, :].copy()

            # Laplaciano (difusao)
            lap_m = self._compute_laplacian(m_current)

            # Fluxo advectivo: m * alpha
            flux = m_current * alpha
            div_flux = self._compute_divergence(flux)

            # Euler forward
            # dm/dt = sigma^2/2 * Delta_m - div(m * alpha)
            m[n + 1, :] = m_current + self.dt * (sigma2_half * lap_m - div_flux)

            # Garante positividade e normalizacao
            m[n + 1, :] = np.maximum(m[n + 1, :], 1e-10)
            m[n + 1, :] = m[n + 1, :] / (np.sum(m[n + 1, :]) * self.dx + 1e-10)

        # Campo de drift resultante
        drift_field = alpha  # O drift e dado pelo controle otimo

        return FPKSolution(
            density=m,
            drift_field=drift_field,
            x_grid=self.x,
            t_grid=self.t
        )


# ==============================================================================
# SOLVER ACOPLADO HJB-FPK (ITERACAO DE PONTO FIXO)
# ==============================================================================

class MFGSolver:
    """
    Solver do Sistema Acoplado (The Master Equation)

    O desafio de engenharia: A HJB depende da distribuicao m (que vem da FPK),
    e a FPK depende da estrategia nabla_u (que vem da HJB). Voce precisa resolver
    este sistema de Forward-Backward Stochastic Differential Equations (FBSDEs)
    iterativamente ate a convergencia.

    Ciclo de Iteracao do Solver:
    1. Chute inicial para a trajetoria do preco
    2. Resolva a HJB de T (futuro) ate t (agora) para achar a estrategia otima
    3. Com essa estrategia, resolva a FPK de t ate T para ver onde a massa vai
    4. Compare a distribuicao resultante com a inicial
    5. Repita ate o erro epsilon < 10^-6
    """

    def __init__(self,
                 n_space: int = 100,
                 n_time: int = 50,
                 max_iterations: int = 100,
                 tolerance: float = 1e-6,
                 relaxation: float = 0.3):
        """
        Args:
            n_space: Pontos no grid espacial
            n_time: Pontos no grid temporal
            max_iterations: Maximo de iteracoes
            tolerance: Tolerancia para convergencia
            relaxation: Fator de relaxacao (0 < lambda < 1) para estabilidade
        """
        self.n_space = n_space
        self.n_time = n_time
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.relaxation = relaxation

        # Solvers individuais
        self.hjb_solver = HJBSolver(n_space, n_time)
        self.fpk_solver = FPKSolver(n_space, n_time)

    def _compute_error(self, m_old: np.ndarray, m_new: np.ndarray) -> float:
        """Computa erro L^2 entre distribuicoes"""
        return np.sqrt(np.mean((m_old - m_new)**2))

    def solve(self,
             m0: np.ndarray,
             sigma: float,
             hamiltonian: MarketHamiltonian) -> NashEquilibrium:
        """
        Resolve o sistema MFG acoplado via iteracao de ponto fixo

        Args:
            m0: Distribuicao inicial de traders
            sigma: Volatilidade do mercado
            hamiltonian: Hamiltoniano do mercado

        Returns:
            NashEquilibrium com a solucao convergida
        """
        # Inicializa com a distribuicao dada
        m_current = m0.copy()
        if len(m_current) != self.n_space:
            x_orig = np.linspace(0, 1, len(m_current))
            x_new = np.linspace(0, 1, self.n_space)
            interp = interp1d(x_orig, m_current, kind='linear', fill_value='extrapolate')
            m_current = interp(x_new)

        # Normaliza
        m_current = m_current / (np.sum(m_current) + 1e-10)

        converged = False
        iteration = 0
        error = float('inf')

        hjb_solution = None
        fpk_solution = None

        for iteration in range(self.max_iterations):
            # Passo 1: Resolve HJB backward com m atual
            hjb_solution = self.hjb_solver.solve(m_current, sigma, hamiltonian)

            # Passo 2: Resolve FPK forward com alpha* da HJB
            fpk_solution = self.fpk_solver.solve(
                m_current, hjb_solution.optimal_control, sigma, hamiltonian
            )

            # Passo 3: Extrai nova distribuicao em t=T
            m_new = fpk_solution.density[-1, :]

            # Passo 4: Computa erro
            error = self._compute_error(m_current, m_new)

            # Passo 5: Atualiza com relaxacao
            m_current = self.relaxation * m_new + (1 - self.relaxation) * m_current
            m_current = np.maximum(m_current, 1e-10)
            m_current = m_current / (np.sum(m_current) + 1e-10)

            # Verifica convergencia
            if error < self.tolerance:
                converged = True
                break

        # Calcula preco e drift de equilibrio
        x_grid = self.hjb_solver.x

        # Preco de equilibrio = media ponderada pela densidade
        equilibrium_price = np.sum(x_grid * m_current) / (np.sum(m_current) + 1e-10)

        # Drift de equilibrio = controle otimo medio
        equilibrium_drift = np.mean(hjb_solution.optimal_control)

        return NashEquilibrium(
            equilibrium_price=equilibrium_price,
            equilibrium_drift=equilibrium_drift,
            value_function=hjb_solution.value_function if hjb_solution else np.array([]),
            density=m_current,
            converged=converged,
            iterations=iteration + 1,
            error=error
        )


# ==============================================================================
# ESTIMADOR DE DISTRIBUICAO INICIAL
# ==============================================================================

class InitialDistributionEstimator:
    """
    Entrada de Dados (O Campo Medio Inicial)

    Utilize o Order Book (Profundidade de Mercado) para estimar a distribuicao
    inicial m(x, t0). Onde estao as ordens limitadas? Essa e a densidade da
    populacao de agentes.
    """

    def __init__(self, n_points: int = 100):
        self.n_points = n_points

    def estimate_from_prices(self,
                            prices: np.ndarray,
                            volumes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estima distribuicao inicial a partir de precos e volumes

        Como nao temos acesso real ao Order Book L2, aproximamos pela
        distribuicao historica de precos ponderada por volume.
        """
        # Normaliza precos para [-1, 1]
        price_min = np.min(prices)
        price_max = np.max(prices)
        price_range = price_max - price_min + 1e-10

        prices_norm = 2 * (prices - price_min) / price_range - 1

        # Se temos volume, usa como peso
        if volumes is not None:
            weights = volumes / (np.sum(volumes) + 1e-10)
        else:
            weights = np.ones(len(prices)) / len(prices)

        # Estima PDF via histograma ponderado
        x_grid = np.linspace(-1, 1, self.n_points)
        m0 = np.zeros(self.n_points)

        for p, w in zip(prices_norm, weights):
            # Encontra bin mais proximo
            idx = int((p + 1) / 2 * (self.n_points - 1))
            idx = np.clip(idx, 0, self.n_points - 1)
            m0[idx] += w

        # Suaviza
        m0 = uniform_filter1d(m0, size=5)

        # Normaliza para ser distribuicao de probabilidade
        m0 = m0 / (np.sum(m0) + 1e-10)

        return m0

    def estimate_from_volume_profile(self,
                                     prices: np.ndarray,
                                     volumes: np.ndarray,
                                     window: int = 50) -> np.ndarray:
        """
        Estima distribuicao a partir do Volume Profile recente
        """
        if len(prices) < window:
            return self.estimate_from_prices(prices, volumes)

        recent_prices = prices[-window:]
        recent_volumes = volumes[-window:]

        return self.estimate_from_prices(recent_prices, recent_volumes)


# ==============================================================================
# CALCULADOR DE DISCREPANCIA DE CAMPO MEDIO
# ==============================================================================

class MeanFieldDiscrepancyCalculator:
    """
    Calcula a discrepancia entre o preco real e o equilibrio de Nash

    Gatilho de Entrada:
    - Discrepancia de Campo Medio: Se o preco real esta subindo (v_real > 0),
      mas o Fluxo Otimo de Nash aponta para baixo (v* < 0), significa que o
      movimento atual e IRRACIONAL e insustentavel (compradores estao pagando
      acima do valor de utilidade marginal).
    - Acao: Entre contra o movimento irracional (Reversao a Media). O preco e
      matematicamente obrigado a convergir para a solucao da HJB.
    """

    def __init__(self, discrepancy_threshold: float = 0.1):
        self.discrepancy_threshold = discrepancy_threshold

    def compute_real_drift(self, prices: np.ndarray, window: int = 10) -> float:
        """
        Calcula o drift real (velocidade atual do preco)
        """
        if len(prices) < window:
            return 0.0

        # Drift = tendencia recente
        recent = prices[-window:]
        drift = (recent[-1] - recent[0]) / (window * (recent[0] + 1e-10))

        return drift

    def compute_discrepancy(self,
                           real_drift: float,
                           nash_drift: float,
                           current_price: float,
                           nash_price: float) -> MeanFieldDiscrepancy:
        """
        Computa a discrepancia entre estado real e equilibrio de Nash
        """
        # Discrepancia de drift
        drift_discrepancy = real_drift - nash_drift

        # Movimento irracional: direcoes opostas
        is_irrational = (real_drift * nash_drift < 0 and
                        abs(drift_discrepancy) > self.discrepancy_threshold)

        # Direcao de correcao (para onde o preco "deveria" ir)
        if nash_drift > 0:
            correction_direction = "UP"
        elif nash_drift < 0:
            correction_direction = "DOWN"
        else:
            correction_direction = "NEUTRAL"

        return MeanFieldDiscrepancy(
            real_drift=real_drift,
            nash_drift=nash_drift,
            discrepancy=drift_discrepancy,
            is_irrational=is_irrational,
            correction_direction=correction_direction
        )


# ==============================================================================
# INDICADOR HJB-NES COMPLETO
# ==============================================================================

class HJBNashEquilibriumSolver:
    """
    Hamilton-Jacobi-Bellman Nash Equilibrium Solver (HJB-NES)

    Indicador completo que calcula o Preco de Equilibrio de Nash do mercado
    usando Mean Field Games.

    O Output (O Sinal de Trading):
    Ao convergir, voce tera o campo vetorial de Fluxo Otimo de Nash.

    - Calculo do "Drift de Equilibrio": O solver lhe dira qual e a "Deriva" (v*)
      que o mercado deveria ter para que todos os agentes estivessem satisfeitos.
    - Calculo do "Drift Real": Meca a velocidade atual do preco.
    - Entre contra movimentos irracionais: O preco e matematicamente obrigado a
      convergir para a solucao da HJB.
    """

    def __init__(self,
                 # Parametros do solver
                 n_space: int = 50,
                 n_time: int = 25,
                 max_iterations: int = 50,
                 tolerance: float = 1e-4,

                 # Parametros do modelo
                 base_risk_aversion: float = 1.0,

                 # Parametros de discrepancia
                 discrepancy_threshold: float = 0.05,

                 # Geral
                 min_data_points: int = 100,
                 volatility_window: int = 50):
        """
        Inicializa o HJB-NES
        """
        self.n_space = n_space
        self.n_time = n_time
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.base_risk_aversion = base_risk_aversion
        self.discrepancy_threshold = discrepancy_threshold
        self.min_data_points = min_data_points
        self.volatility_window = volatility_window

        # Componentes
        self.mfg_solver = MFGSolver(
            n_space=n_space,
            n_time=n_time,
            max_iterations=max_iterations,
            tolerance=tolerance
        )
        self.distribution_estimator = InitialDistributionEstimator(n_points=n_space)
        self.discrepancy_calc = MeanFieldDiscrepancyCalculator(
            discrepancy_threshold=discrepancy_threshold
        )
        self.hamiltonian = MarketHamiltonian(risk_aversion=base_risk_aversion)

        # Cache
        self.last_equilibrium: Optional[NashEquilibrium] = None

        # Historicos
        self.equilibrium_history: List[float] = []
        self.drift_history: List[float] = []

    def _compute_returns(self, prices: np.ndarray) -> np.ndarray:
        """Calcula retornos logaritmicos"""
        return np.diff(np.log(prices + 1e-10))

    def _compute_volatility(self, prices: np.ndarray) -> float:
        """Calcula volatilidade"""
        returns = self._compute_returns(prices)
        if len(returns) < self.volatility_window:
            return np.std(returns) + 1e-10
        return np.std(returns[-self.volatility_window:]) + 1e-10

    def _normalize_price(self, price: float, prices: np.ndarray) -> float:
        """Normaliza preco para [-1, 1]"""
        price_min = np.min(prices)
        price_max = np.max(prices)
        price_range = price_max - price_min + 1e-10
        return 2 * (price - price_min) / price_range - 1

    def _denormalize_price(self, norm_price: float, prices: np.ndarray) -> float:
        """Desnormaliza preco de [-1, 1]"""
        price_min = np.min(prices)
        price_max = np.max(prices)
        price_range = price_max - price_min + 1e-10
        return (norm_price + 1) / 2 * price_range + price_min

    def _determine_market_regime(self,
                                price_discrepancy: float,
                                converged: bool) -> MarketRegime:
        """Determina o regime do mercado"""
        if not converged:
            return MarketRegime.UNSTABLE

        if abs(price_discrepancy) < 0.05:
            return MarketRegime.EQUILIBRIUM
        elif price_discrepancy > 0.05:
            return MarketRegime.OVERVALUED
        else:
            return MarketRegime.UNDERVALUED

    def analyze(self,
                prices: np.ndarray,
                volumes: Optional[np.ndarray] = None) -> dict:
        """
        Processa dados de mercado e retorna analise completa

        Args:
            prices: Array de precos
            volumes: Array de volumes (opcional)

        Returns:
            Dict com analise completa do equilibrio de Nash
        """
        n = len(prices)

        # Validacao
        if n < self.min_data_points:
            return {
                'signal': 0,
                'signal_name': 'INSUFFICIENT_DATA',
                'confidence': 0.0,
                'nash_price': prices[-1] if n > 0 else 0.0,
                'nash_drift': 0.0,
                'current_price': prices[-1] if n > 0 else 0.0,
                'current_drift': 0.0,
                'price_discrepancy': 0.0,
                'drift_discrepancy': 0.0,
                'market_regime': 'UNSTABLE',
                'solver_converged': False,
                'solver_iterations': 0,
                'solver_error': float('inf'),
                'risk_aversion': self.base_risk_aversion,
                'volatility': 0.0,
                'reasons': ['Dados insuficientes para analise MFG']
            }

        # Gera volumes sinteticos se nao fornecidos
        if volumes is None:
            volumes = np.ones(n)

        # ============================================================
        # PASSO 1: CALIBRACAO DE PARAMETROS
        # ============================================================
        returns = self._compute_returns(prices)
        sigma = self._compute_volatility(prices)

        # Calibra aversao ao risco
        gamma = self.hamiltonian.calibrate_risk_aversion(returns, self.volatility_window)
        self.hamiltonian.risk_aversion = gamma

        # ============================================================
        # PASSO 2: ESTIMACAO DA DISTRIBUICAO INICIAL
        # ============================================================
        m0 = self.distribution_estimator.estimate_from_volume_profile(
            prices, volumes, window=min(50, n // 2)
        )

        # ============================================================
        # PASSO 3: RESOLVER SISTEMA MFG ACOPLADO
        # ============================================================
        equilibrium = self.mfg_solver.solve(m0, sigma, self.hamiltonian)
        self.last_equilibrium = equilibrium

        # ============================================================
        # PASSO 4: CALCULAR DISCREPANCIA
        # ============================================================
        # Drift real
        real_drift = self.discrepancy_calc.compute_real_drift(prices)

        # Preco atual normalizado
        current_price_norm = self._normalize_price(prices[-1], prices)

        # Preco de Nash desnormalizado
        nash_price = self._denormalize_price(equilibrium.equilibrium_price, prices)

        # Discrepancia
        discrepancy = self.discrepancy_calc.compute_discrepancy(
            real_drift=real_drift,
            nash_drift=equilibrium.equilibrium_drift,
            current_price=current_price_norm,
            nash_price=equilibrium.equilibrium_price
        )

        price_discrepancy = current_price_norm - equilibrium.equilibrium_price

        # Atualiza historicos
        self.equilibrium_history.append(nash_price)
        self.drift_history.append(equilibrium.equilibrium_drift)
        if len(self.equilibrium_history) > 100:
            self.equilibrium_history.pop(0)
            self.drift_history.pop(0)

        # ============================================================
        # PASSO 5: GERACAO DE SINAL
        # ============================================================
        market_regime = self._determine_market_regime(
            price_discrepancy, equilibrium.converged
        )

        signal = 0
        signal_name = 'NEUTRAL'
        confidence = 0.0
        reasons = []

        # Solver nao convergiu - nao opera
        if not equilibrium.converged:
            signal_name = 'HIBERNATE'
            reasons.append(f'Solver nao convergiu (iter={equilibrium.iterations}, err={equilibrium.error:.2e})')
            reasons.append('Mercado possivelmente em regime extremo')

        # Movimento irracional detectado - entra contra
        elif discrepancy.is_irrational:

            if discrepancy.correction_direction == "UP":
                # Preco deveria subir (Nash drift positivo), mas esta caindo
                signal = 1
                signal_name = 'LONG'
                confidence = min(1.0, abs(discrepancy.discrepancy) * 5)
                reasons.append('Arbitragem Estrutural: Preco contra Nash')
                reasons.append(f'v_real={real_drift:.4f}, v*={equilibrium.equilibrium_drift:.4f}')
                reasons.append('Convergencia para cima esperada')

            elif discrepancy.correction_direction == "DOWN":
                # Preco deveria cair (Nash drift negativo), mas esta subindo
                signal = -1
                signal_name = 'SHORT'
                confidence = min(1.0, abs(discrepancy.discrepancy) * 5)
                reasons.append('Arbitragem Estrutural: Preco contra Nash')
                reasons.append(f'v_real={real_drift:.4f}, v*={equilibrium.equilibrium_drift:.4f}')
                reasons.append('Convergencia para baixo esperada')

            else:
                reasons.append('Direcao de correcao neutra')

        # Mercado desequilibrado (preco longe do Nash)
        elif market_regime == MarketRegime.OVERVALUED:
            signal = -1
            signal_name = 'SHORT'
            confidence = min(0.7, abs(price_discrepancy) * 2)
            reasons.append('Overvalued: Preco acima do equilibrio de Nash')
            reasons.append(f'Discrepancia={price_discrepancy:.4f}')

        elif market_regime == MarketRegime.UNDERVALUED:
            signal = 1
            signal_name = 'LONG'
            confidence = min(0.7, abs(price_discrepancy) * 2)
            reasons.append('Undervalued: Preco abaixo do equilibrio de Nash')
            reasons.append(f'Discrepancia={price_discrepancy:.4f}')

        # No equilibrio
        elif market_regime == MarketRegime.EQUILIBRIUM:
            reasons.append('Equilibrio: Preco no ponto de Nash')
            reasons.append(f'Discrepancia={price_discrepancy:.4f}')

        else:
            reasons.append(f'Regime {market_regime.value}. Aguardando estabilizacao.')

        # Ajusta confianca pelo erro do solver
        if confidence > 0 and equilibrium.converged:
            solver_confidence = 1.0 / (1.0 + equilibrium.error * 1000)
            confidence *= solver_confidence
            confidence = np.clip(confidence, 0, 1)

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'nash_price': nash_price,
            'nash_drift': equilibrium.equilibrium_drift,
            'current_price': prices[-1],
            'current_drift': real_drift,
            'price_discrepancy': price_discrepancy,
            'drift_discrepancy': discrepancy.discrepancy,
            'market_regime': market_regime.value,
            'is_irrational': discrepancy.is_irrational,
            'correction_direction': discrepancy.correction_direction,
            'solver_converged': equilibrium.converged,
            'solver_iterations': equilibrium.iterations,
            'solver_error': equilibrium.error,
            'risk_aversion': gamma,
            'volatility': sigma,
            'reasons': reasons
        }

    def get_equilibrium(self) -> Optional[NashEquilibrium]:
        """Retorna o ultimo equilibrio calculado"""
        return self.last_equilibrium

    def get_equilibrium_history(self) -> np.ndarray:
        """Retorna historico de precos de equilibrio"""
        return np.array(self.equilibrium_history)

    def get_drift_history(self) -> np.ndarray:
        """Retorna historico de drifts de equilibrio"""
        return np.array(self.drift_history)

    def reset(self):
        """Reseta o estado do indicador"""
        self.last_equilibrium = None
        self.equilibrium_history.clear()
        self.drift_history.clear()


# ==============================================================================
# DEMONSTRACAO
# ==============================================================================

def generate_mean_reverting_data(n_points: int = 500, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Gera dados com mean-reversion (onde MFG funciona bem)"""
    np.random.seed(seed)

    # Processo Ornstein-Uhlenbeck
    mu = 1.0850
    theta = 0.15  # Velocidade de reversao
    sigma = 0.0003

    prices = np.zeros(n_points)
    prices[0] = mu

    for i in range(1, n_points):
        dW = np.random.randn()
        prices[i] = prices[i-1] + theta * (mu - prices[i-1]) + sigma * dW

    # Volume correlacionado com volatilidade
    returns = np.diff(prices)
    volumes = 1000 + 500 * np.abs(np.append([0], returns)) * 10000
    volumes += np.random.randn(n_points) * 100
    volumes = np.maximum(volumes, 100)

    return prices, volumes


def main():
    """Demonstracao do indicador HJB-NES"""
    print("=" * 70)
    print("HAMILTON-JACOBI-BELLMAN NASH EQUILIBRIUM SOLVER (HJB-NES)")
    print("Indicador baseado em Mean Field Games")
    print("=" * 70)
    print()

    # Inicializa indicador
    indicator = HJBNashEquilibriumSolver(
        n_space=50,
        n_time=25,
        max_iterations=50,
        tolerance=1e-4,
        base_risk_aversion=1.0,
        discrepancy_threshold=0.05,
        min_data_points=100
    )

    print("Indicador inicializado!")
    print(f"  - Grid espacial: 50 pontos")
    print(f"  - Grid temporal: 25 pontos")
    print(f"  - Max iteracoes: 50")
    print(f"  - Tolerancia: 1e-4")
    print()

    # Gera dados
    prices, volumes = generate_mean_reverting_data(n_points=200)
    print(f"Dados gerados: {len(prices)} pontos")
    print()

    # Processa
    result = indicator.analyze(prices, volumes)

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Sinal: {result['signal_name']}")
    print(f"Regime: {result['market_regime']}")
    print(f"Confianca: {result['confidence']:.2%}")
    print(f"\nEquilibrio de Nash:")
    print(f"  Preco Nash: {result['nash_price']:.5f}")
    print(f"  Drift Nash (v*): {result['nash_drift']:.6f}")
    print(f"  Preco atual: {result['current_price']:.5f}")
    print(f"  Drift real (v): {result['current_drift']:.6f}")
    print(f"\nDiscrepancias:")
    print(f"  Preco: {result['price_discrepancy']:.6f}")
    print(f"  Drift: {result['drift_discrepancy']:.6f}")
    print(f"\nSolver:")
    print(f"  Convergiu: {result['solver_converged']}")
    print(f"  Iteracoes: {result['solver_iterations']}")
    print(f"  Erro: {result['solver_error']:.2e}")
    print(f"\nParametros:")
    print(f"  gamma (aversao): {result['risk_aversion']:.4f}")
    print(f"  sigma (vol): {result['volatility']:.6f}")
    print(f"\nRazoes: {result['reasons']}")

    return indicator, result


if __name__ == "__main__":
    indicator, result = main()
