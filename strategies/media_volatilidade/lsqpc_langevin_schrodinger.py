"""
================================================================================
LANGEVIN-SCHRÖDINGER QUANTUM PROBABILITY CLOUD (LSQPC)
Indicador de Forex para EURUSD baseado em Física Estatística Avançada
================================================================================

Este modelo NÃO tenta adivinhar onde o preço vai. Ele calcula a Densidade de
Probabilidade (PDF) da função de onda do preço e opera o colapso dessa função.

Arquitetura de 2 Camadas (filtro de regime é externo):
1. Motor Dinâmico (Cérebro) - Equação de Langevin Generalizada
2. Sinal Quântico (Gatilho) - Equação de Fokker-Planck

Nota: O filtro de regime (volatilidade) é gerenciado externamente pelo sistema.
================================================================================
"""

import numpy as np
from numpy.fft import fft, ifft
from scipy import integrate
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import curve_fit
from scipy.stats import levy_stable
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.interpolate import interp1d
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
from enum import Enum

warnings.filterwarnings('ignore')


# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

class SignalType(Enum):
    """Tipos de sinais de trading"""
    LONG = "LONG"
    SHORT = "SHORT"
    NO_SIGNAL = "NO_SIGNAL"


@dataclass
class LangevinSimulation:
    """Resultado da simulação de Langevin"""
    trajectories: np.ndarray       # Trajetórias simuladas
    time_grid: np.ndarray          # Grid temporal
    mean_trajectory: np.ndarray    # Média das trajetórias
    std_trajectory: np.ndarray     # Desvio padrão das trajetórias
    lambda_param: float            # Parâmetro de reversão
    sigma_L: float                 # Volatilidade de Lévy


@dataclass
class FokkerPlanckSolution:
    """Solução da equação de Fokker-Planck"""
    probability_density: np.ndarray  # P(x, t)
    price_grid: np.ndarray           # Grid de preços
    time_grid: np.ndarray            # Grid temporal
    effective_potential: np.ndarray  # V(x) = -ln(P_steady)
    upper_95_isoline: np.ndarray     # Isolinha 95% superior
    lower_95_isoline: np.ndarray     # Isolinha 95% inferior
    current_probability: float       # Probabilidade atual


@dataclass
class QuantumAnalysisResult:
    """Resultado completo da análise quântica"""
    signal: int                      # 1=LONG, -1=SHORT, 0=NO_SIGNAL
    signal_name: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    probability_breach: float
    lambda_param: float              # Parâmetro de reversão à média
    sigma_levy: float                # Volatilidade de Lévy
    mean_trajectory: np.ndarray      # Trajetória média prevista
    upper_bound: float               # Limite superior 95%
    lower_bound: float               # Limite inferior 95%
    effective_potential: np.ndarray  # Potencial efetivo V(x)
    reasons: List[str]


# ==============================================================================
# CAMADA 1: MOTOR DINÂMICO (CÉREBRO)
# Equação de Langevin Generalizada
# ==============================================================================

class MemoryKernel:
    """
    Kernel de Memória K(t-τ) para a Equação de Langevin

    Usa decaimento de lei de potência (Power-law decay) para capturar
    a persistência de longo prazo típica do EURUSD.
    """

    def __init__(self,
                 alpha: float = 0.5,  # Expoente do power-law
                 tau_0: float = 1.0,  # Escala de tempo característica
                 cutoff: float = 100.0):  # Tempo de corte
        """
        Args:
            alpha: Expoente do decaimento (0 < α < 1 para memória longa)
            tau_0: Escala de tempo
            cutoff: Tempo máximo de memória
        """
        self.alpha = alpha
        self.tau_0 = tau_0
        self.cutoff = cutoff

    def __call__(self, t: float) -> float:
        """
        Avalia o kernel em t

        K(t) = (t/τ₀)^(-α) * exp(-t/cutoff)
        """
        if t <= 0:
            return 1.0
        return (t / self.tau_0) ** (-self.alpha) * np.exp(-t / self.cutoff)

    def evaluate_array(self, t_array: np.ndarray) -> np.ndarray:
        """Avalia kernel em array de tempos"""
        result = np.ones_like(t_array, dtype=float)
        positive = t_array > 0
        result[positive] = ((t_array[positive] / self.tau_0) ** (-self.alpha) *
                           np.exp(-t_array[positive] / self.cutoff))
        return result


class LevyNoise:
    """
    Ruído Fracionário de Lévy

    Modela as caudas gordas ("fat tails") típicas de mercados financeiros.
    """

    def __init__(self,
                 alpha: float = 1.7,  # Índice de estabilidade (1 < α ≤ 2)
                 beta: float = 0.0,   # Parâmetro de assimetria
                 scale: float = 1.0):
        """
        Args:
            alpha: Índice de estabilidade (α=2 é Gaussiano)
            beta: Assimetria (-1 a 1)
            scale: Escala
        """
        self.alpha = alpha
        self.beta = beta
        self.scale = scale

    def generate(self, size: int, dt: float = 1.0) -> np.ndarray:
        """Gera amostras de ruído de Lévy"""
        try:
            samples = levy_stable.rvs(
                alpha=self.alpha,
                beta=self.beta,
                scale=self.scale * (dt ** (1/self.alpha)),
                size=size
            )
        except:
            # Fallback para distribuição normal com caudas pesadas
            samples = np.random.standard_t(df=4, size=size) * self.scale * np.sqrt(dt)
        return samples


class GeneralizedLangevinEquation:
    """
    Equação de Langevin Generalizada

    dp/dt = -λ ∫₋∞ᵗ K(t-τ)v(τ)dτ + σ_L ξ(t)

    Onde:
    - p: Log-preço
    - λ: Coeficiente de reversão (damping)
    - K(t-τ): Kernel de memória
    - v(τ): "Velocidade" do preço
    - σ_L: Volatilidade de Lévy
    - ξ(t): Ruído de Lévy
    """

    def __init__(self,
                 lambda_param: float = 0.1,
                 sigma_L: float = 0.01,
                 memory_alpha: float = 0.5,
                 levy_alpha: float = 1.7,
                 dt: float = 1/1440,  # 1 minuto em dias
                 n_trajectories: int = 5000,
                 forecast_horizon: int = 15):  # 15 minutos
        """
        Args:
            lambda_param: Coeficiente de reversão
            sigma_L: Volatilidade de Lévy
            memory_alpha: Expoente do kernel de memória
            levy_alpha: Índice de estabilidade de Lévy
            dt: Passo temporal
            n_trajectories: Número de trajetórias Monte Carlo
            forecast_horizon: Horizonte de previsão em minutos
        """
        self.lambda_param = lambda_param
        self.sigma_L = sigma_L
        self.dt = dt
        self.n_trajectories = n_trajectories
        self.forecast_horizon = forecast_horizon

        self.kernel = MemoryKernel(alpha=memory_alpha)
        self.noise = LevyNoise(alpha=levy_alpha)

    def calibrate(self, returns: np.ndarray) -> Tuple[float, float]:
        """
        Calibra parâmetros λ e σ_L instantaneamente baseado nos dados

        Returns:
            Tupla (lambda_calibrado, sigma_L_calibrado)
        """
        n = len(returns)
        mean = np.mean(returns)
        var = np.var(returns)

        if var == 0:
            var = 1e-10

        # Autocorrelação lag-1
        if n > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0

        # λ estimado (velocidade de reversão)
        if autocorr > 0 and autocorr < 1:
            self.lambda_param = -np.log(autocorr) / self.dt
        else:
            self.lambda_param = 1.0  # Default

        # σ_L estimado via variância
        self.sigma_L = np.sqrt(var / self.dt)

        # Limita valores extremos
        self.lambda_param = np.clip(self.lambda_param, 0.01, 10.0)
        self.sigma_L = np.clip(self.sigma_L, 1e-6, 0.1)

        return self.lambda_param, self.sigma_L

    def _integrate_memory(self,
                         velocity_history: np.ndarray,
                         time_history: np.ndarray,
                         current_time: float) -> float:
        """
        Calcula a integral de memória ∫K(t-τ)v(τ)dτ
        """
        if len(velocity_history) == 0:
            return 0.0

        # Diferenças de tempo
        tau_diffs = current_time - time_history

        # Kernel avaliado
        kernel_values = self.kernel.evaluate_array(tau_diffs)

        # Integral via soma de Riemann
        dt_array = np.diff(time_history, prepend=time_history[0] - self.dt)
        integral = np.sum(kernel_values * velocity_history * np.abs(dt_array))

        return integral

    def simulate(self,
                initial_price: float,
                returns_history: np.ndarray) -> LangevinSimulation:
        """
        Simula trajetórias via Monte Carlo

        Args:
            initial_price: Preço inicial (log-preço)
            returns_history: Histórico de retornos para memória

        Returns:
            LangevinSimulation com todas as trajetórias
        """
        # Calibra parâmetros
        self.calibrate(returns_history)

        # Grid temporal
        n_steps = self.forecast_horizon
        time_grid = np.arange(n_steps + 1) * self.dt

        # Prepara histórico de velocidade
        memory_len = min(50, len(returns_history))
        velocity_history = returns_history[-memory_len:] / self.dt
        time_history = np.arange(-len(velocity_history), 0) * self.dt

        # Array para trajetórias
        trajectories = np.zeros((self.n_trajectories, n_steps + 1))
        trajectories[:, 0] = initial_price

        # Gera todo o ruído de uma vez
        noise = self.noise.generate(
            size=(self.n_trajectories, n_steps),
            dt=self.dt
        )

        # Simula cada trajetória (Euler-Maruyama)
        for traj in range(self.n_trajectories):
            p = initial_price
            v = returns_history[-1] / self.dt if len(returns_history) > 0 else 0

            vel_hist = velocity_history.copy()
            t_hist = time_history.copy()

            for t_idx in range(n_steps):
                current_time = time_grid[t_idx]

                # Integral de memória simplificada
                memory_integral = self._integrate_memory(vel_hist, t_hist, current_time)

                # Equação de Langevin discretizada
                dv = -self.lambda_param * (v + 0.1 * memory_integral) * self.dt
                dv += self.sigma_L * noise[traj, t_idx]

                # Atualiza velocidade e preço
                v = np.clip(v + dv, -0.01, 0.01)  # Limita velocidade
                p += v * self.dt

                trajectories[traj, t_idx + 1] = p

                # Atualiza histórico
                vel_hist = np.append(vel_hist[1:], v)
                t_hist = t_hist + self.dt

        # Estatísticas das trajetórias
        mean_trajectory = np.mean(trajectories, axis=0)
        std_trajectory = np.std(trajectories, axis=0)

        return LangevinSimulation(
            trajectories=trajectories,
            time_grid=time_grid,
            mean_trajectory=mean_trajectory,
            std_trajectory=std_trajectory,
            lambda_param=self.lambda_param,
            sigma_L=self.sigma_L
        )


# ==============================================================================
# CAMADA 2: SINAL QUÂNTICO (GATILHO)
# Equação de Fokker-Planck
# ==============================================================================

class FokkerPlanckSolver:
    """
    Solver para a Equação de Fokker-Planck

    ∂P/∂t = -∂/∂x[D⁽¹⁾(x,t)P] + ∂²/∂x²[D⁽²⁾(x,t)P]

    Onde:
    - P(x,t): Densidade de probabilidade
    - D⁽¹⁾: Coeficiente de deriva (drift)
    - D⁽²⁾: Coeficiente de difusão
    """

    def __init__(self,
                 n_grid: int = 200,
                 n_time: int = 100,
                 price_range_std: float = 4.0):
        """
        Args:
            n_grid: Pontos no grid espacial
            n_time: Pontos no grid temporal
            price_range_std: Range em desvios padrão
        """
        self.n_grid = n_grid
        self.n_time = n_time
        self.price_range_std = price_range_std

    def _drift_coefficient(self, x: np.ndarray, lambda_param: float,
                          mean_price: float) -> np.ndarray:
        """
        Coeficiente de deriva D⁽¹⁾(x)

        Modelo de reversão à média: D⁽¹⁾ = -λ(x - μ)
        """
        return -lambda_param * (x - mean_price)

    def _diffusion_coefficient(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """
        Coeficiente de difusão D⁽²⁾(x)

        Difusão constante: D⁽²⁾ = σ²/2
        """
        return np.ones_like(x) * (sigma ** 2) / 2

    def solve(self,
             langevin_result: LangevinSimulation,
             current_price: float) -> FokkerPlanckSolution:
        """
        Resolve a equação de Fokker-Planck numericamente

        Args:
            langevin_result: Resultado da simulação de Langevin
            current_price: Preço atual

        Returns:
            FokkerPlanckSolution
        """
        # Extrai parâmetros
        lambda_param = langevin_result.lambda_param
        sigma = langevin_result.sigma_L
        trajectories = langevin_result.trajectories

        # Estatísticas
        all_prices = trajectories.flatten()
        mean_price = np.mean(all_prices)
        std_price = np.std(all_prices)

        if std_price < 1e-10:
            std_price = 0.001

        # Grid espacial
        x_min = mean_price - self.price_range_std * std_price
        x_max = mean_price + self.price_range_std * std_price
        x_grid = np.linspace(x_min, x_max, self.n_grid)
        dx = x_grid[1] - x_grid[0]

        # Grid temporal
        t_max = langevin_result.time_grid[-1]
        t_grid = np.linspace(0, t_max, self.n_time)
        dt = t_grid[1] - t_grid[0] if len(t_grid) > 1 else 0.001

        # Condição inicial: gaussiana centrada no preço atual
        P = np.exp(-((x_grid - current_price) ** 2) / (2 * (std_price/10) ** 2))
        P = P / (np.sum(P) * dx + 1e-10)

        # Coeficientes
        D1 = self._drift_coefficient(x_grid, lambda_param, mean_price)
        D2 = self._diffusion_coefficient(x_grid, sigma)

        # Matriz para evolução
        P_history = np.zeros((self.n_time, self.n_grid))
        P_history[0, :] = P

        # Resolve via diferenças finitas
        for t_idx in range(1, self.n_time):
            P_new = P.copy()

            for i in range(1, self.n_grid - 1):
                # Termo de deriva
                drift = -D1[i] * (P[i+1] - P[i-1]) / (2 * dx)
                drift -= P[i] * (D1[min(i+1, self.n_grid-1)] - D1[max(i-1, 0)]) / (2 * dx)

                # Termo de difusão
                diffusion = D2[i] * (P[i+1] - 2*P[i] + P[i-1]) / (dx**2)

                P_new[i] = P[i] + dt * (drift + diffusion)

            # Condições de contorno
            P_new[0] = 0
            P_new[-1] = 0

            # Positividade e normalização
            P_new = np.maximum(P_new, 0)
            if np.sum(P_new) > 0:
                P_new = P_new / (np.sum(P_new) * dx)

            P = P_new
            P_history[t_idx, :] = P

        # Distribuição estacionária
        P_steady = P_history[-1, :]

        # Potencial efetivo
        P_steady_safe = np.maximum(P_steady, 1e-20)
        effective_potential = -np.log(P_steady_safe)

        # Calcula quantis
        P_final = P_history[-1, :]
        cdf = np.cumsum(P_final) * dx
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]

        try:
            lower_idx = np.searchsorted(cdf, 0.025)
            upper_idx = np.searchsorted(cdf, 0.975)

            lower_95 = x_grid[max(0, lower_idx)]
            upper_95 = x_grid[min(len(x_grid)-1, upper_idx)]
        except:
            lower_95 = x_grid[0]
            upper_95 = x_grid[-1]

        # Probabilidade atual
        current_idx = np.searchsorted(x_grid, current_price)
        if current_idx < len(cdf):
            current_prob = cdf[current_idx]
        else:
            current_prob = 1.0

        # Isolinhas
        upper_isoline = np.ones(self.n_time) * upper_95
        lower_isoline = np.ones(self.n_time) * lower_95

        return FokkerPlanckSolution(
            probability_density=P_history,
            price_grid=x_grid,
            time_grid=t_grid,
            effective_potential=effective_potential,
            upper_95_isoline=upper_isoline,
            lower_95_isoline=lower_isoline,
            current_probability=current_prob
        )


class QuantumSignalGenerator:
    """
    Gerador de Sinais Quânticos

    Analisa a solução de Fokker-Planck para gerar sinais de trading
    baseados em "tunelamento quântico" (probabilidade de romper barreiras).
    """

    def __init__(self,
                 probability_threshold: float = 0.05,
                 confidence_multiplier: float = 1.5):
        """
        Args:
            probability_threshold: Limiar de probabilidade (P < 5%)
            confidence_multiplier: Multiplicador para stop/take profit
        """
        self.probability_threshold = probability_threshold
        self.confidence_multiplier = confidence_multiplier

    def generate_signal(self,
                       fokker_planck: FokkerPlanckSolution,
                       langevin: LangevinSimulation,
                       current_price: float) -> QuantumAnalysisResult:
        """
        Gera sinal de trading baseado na análise quântica
        """
        # Isolinhas 95%
        upper_95 = fokker_planck.upper_95_isoline[0]
        lower_95 = fokker_planck.lower_95_isoline[0]

        # Range de preço
        price_range = upper_95 - lower_95
        if price_range == 0:
            price_range = 0.001

        # Probabilidade
        prob = fokker_planck.current_probability

        # Determina sinal
        signal = 0
        signal_name = "NO_SIGNAL"
        confidence = 0.0
        stop_loss = current_price
        take_profit = current_price
        reasons = []

        # SHORT: Preço na isolinha superior
        if current_price >= upper_95 or prob >= (1 - self.probability_threshold):
            signal = -1
            signal_name = "SHORT"
            confidence = min(1.0, (prob - 0.95) / 0.05 + 0.5)
            confidence = max(0.5, confidence)

            stop_loss = current_price + price_range * 0.2
            take_profit = lower_95 + price_range * 0.3

            reasons.append("price_at_upper_95")
            reasons.append(f"prob={prob:.3f}")

        # LONG: Preço na isolinha inferior
        elif current_price <= lower_95 or prob <= self.probability_threshold:
            signal = 1
            signal_name = "LONG"
            confidence = min(1.0, (0.05 - prob) / 0.05 + 0.5)
            confidence = max(0.5, confidence)

            stop_loss = current_price - price_range * 0.2
            take_profit = upper_95 - price_range * 0.3

            reasons.append("price_at_lower_95")
            reasons.append(f"prob={prob:.3f}")

        return QuantumAnalysisResult(
            signal=signal,
            signal_name=signal_name,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            probability_breach=prob,
            lambda_param=langevin.lambda_param,
            sigma_levy=langevin.sigma_L,
            mean_trajectory=langevin.mean_trajectory,
            upper_bound=upper_95,
            lower_bound=lower_95,
            effective_potential=fokker_planck.effective_potential,
            reasons=reasons
        )


# ==============================================================================
# SISTEMA COMPLETO: LANGEVIN-SCHRÖDINGER QUANTUM PROBABILITY CLOUD
# ==============================================================================

class LangevinSchrodingerQuantumIndicator:
    """
    Sistema Completo de Trading baseado em Física Estatística

    Integra as duas camadas:
    1. Motor Dinâmico (Langevin Generalizada)
    2. Sinal Quântico (Fokker-Planck)

    Nota: O filtro de regime é gerenciado externamente.
    """

    def __init__(self,
                 # Parâmetros do Motor Dinâmico
                 n_trajectories: int = 5000,
                 forecast_horizon: int = 15,  # minutos
                 memory_alpha: float = 0.5,
                 levy_alpha: float = 1.7,

                 # Parâmetros do Sinal Quântico
                 probability_threshold: float = 0.05,

                 # Parâmetros gerais
                 min_data_points: int = 100):
        """
        Inicializa o sistema completo
        """
        self.n_trajectories = n_trajectories
        self.forecast_horizon = forecast_horizon
        self.min_data_points = min_data_points

        # Inicializa componentes
        self.langevin_engine = GeneralizedLangevinEquation(
            n_trajectories=n_trajectories,
            forecast_horizon=forecast_horizon,
            memory_alpha=memory_alpha,
            levy_alpha=levy_alpha
        )
        self.fokker_planck_solver = FokkerPlanckSolver()
        self.signal_generator = QuantumSignalGenerator(
            probability_threshold=probability_threshold
        )

        # Estado interno
        self.last_langevin: Optional[LangevinSimulation] = None
        self.last_fokker_planck: Optional[FokkerPlanckSolution] = None

    def analyze(self, prices: np.ndarray) -> dict:
        """
        Analisa série de preços e gera resultado completo

        Args:
            prices: Array de preços

        Returns:
            Dict com resultado da análise
        """
        n = len(prices)

        if n < self.min_data_points:
            return {
                'signal': 0,
                'signal_name': 'INSUFFICIENT_DATA',
                'confidence': 0.0,
                'entry_price': prices[-1] if n > 0 else 0,
                'stop_loss': 0,
                'take_profit': 0,
                'probability_breach': 0.5,
                'lambda_param': 0,
                'sigma_levy': 0,
                'upper_bound': 0,
                'lower_bound': 0,
                'reasons': ['insufficient_data'],
                'current_price': prices[-1] if n > 0 else 0
            }

        # Log-retornos
        log_prices = np.log(prices)
        returns = np.diff(log_prices)
        current_log_price = log_prices[-1]
        current_price = prices[-1]

        # ============================================================
        # CAMADA 1: MOTOR DINÂMICO (LANGEVIN)
        # ============================================================
        langevin_result = self.langevin_engine.simulate(
            initial_price=current_log_price,
            returns_history=returns
        )
        self.last_langevin = langevin_result

        # ============================================================
        # CAMADA 2: SINAL QUÂNTICO (FOKKER-PLANCK)
        # ============================================================
        fokker_planck_result = self.fokker_planck_solver.solve(
            langevin_result=langevin_result,
            current_price=current_log_price
        )
        self.last_fokker_planck = fokker_planck_result

        # Gera sinal
        quantum_result = self.signal_generator.generate_signal(
            fokker_planck=fokker_planck_result,
            langevin=langevin_result,
            current_price=current_log_price
        )

        # Converte preços de volta de log
        if quantum_result.signal != 0:
            log_entry = np.log(current_price)
            log_sl = np.log(current_price) + (quantum_result.stop_loss - current_log_price)
            log_tp = np.log(current_price) + (quantum_result.take_profit - current_log_price)
            stop_loss = np.exp(log_sl)
            take_profit = np.exp(log_tp)
        else:
            stop_loss = current_price
            take_profit = current_price

        return {
            'signal': quantum_result.signal,
            'signal_name': quantum_result.signal_name,
            'confidence': quantum_result.confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'probability_breach': quantum_result.probability_breach,
            'lambda_param': quantum_result.lambda_param,
            'sigma_levy': quantum_result.sigma_levy,
            'mean_trajectory': quantum_result.mean_trajectory,
            'upper_bound': np.exp(quantum_result.upper_bound),
            'lower_bound': np.exp(quantum_result.lower_bound),
            'effective_potential': quantum_result.effective_potential,
            'reasons': quantum_result.reasons,
            'current_price': current_price,
            'n_trajectories': self.n_trajectories,
            'forecast_horizon': self.forecast_horizon
        }

    def get_probability_heatmap(self) -> Optional[np.ndarray]:
        """Retorna o mapa de calor de probabilidade"""
        if self.last_fokker_planck is not None:
            return self.last_fokker_planck.probability_density
        return None

    def get_effective_potential(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retorna o potencial efetivo V(x)"""
        if self.last_fokker_planck is not None:
            return (self.last_fokker_planck.price_grid,
                   self.last_fokker_planck.effective_potential)
        return None

    def get_trajectories(self) -> Optional[np.ndarray]:
        """Retorna as trajetórias simuladas"""
        if self.last_langevin is not None:
            return self.last_langevin.trajectories
        return None


# ==============================================================================
# EXEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LANGEVIN-SCHRÖDINGER QUANTUM PROBABILITY CLOUD (LSQPC)")
    print("Indicador de Física Estatística para Média Volatilidade")
    print("=" * 70)

    np.random.seed(42)

    # Gera dados sintéticos
    initial_price = 1.0850
    n_points = 500

    prices = [initial_price]
    for _ in range(n_points - 1):
        current = prices[-1]
        mean_rev = 0.1 * (initial_price - current)
        noise = np.random.standard_t(df=4) * 0.0002
        new_price = current + mean_rev + noise
        prices.append(max(new_price, 1.0))

    prices = np.array(prices)

    print(f"\nDados: {len(prices)} pontos")
    print(f"Preço: {prices[0]:.5f} -> {prices[-1]:.5f}")

    # Cria indicador
    indicator = LangevinSchrodingerQuantumIndicator(
        n_trajectories=5000,
        forecast_horizon=15,
        memory_alpha=0.5,
        levy_alpha=1.7,
        probability_threshold=0.05,
        min_data_points=100
    )

    print("\nExecutando análise quântica...")

    # Analisa
    result = indicator.analyze(prices)

    print("\n" + "-" * 40)
    print("RESULTADO:")
    print(f"  Sinal: {result['signal_name']}")
    print(f"  Confiança: {result['confidence']:.0%}")

    print("\nPARÂMETROS CALIBRADOS:")
    print(f"  λ (reversão): {result['lambda_param']:.4f}")
    print(f"  σ_L (Lévy): {result['sigma_levy']:.6f}")

    print("\nLIMITES 95%:")
    print(f"  Superior: {result['upper_bound']:.5f}")
    print(f"  Inferior: {result['lower_bound']:.5f}")
    print(f"  Preço atual: {result['current_price']:.5f}")

    print("\nPROBABILIDADE:")
    print(f"  P(breach): {result['probability_breach']:.4f}")

    if result['signal'] != 0:
        print("\nNÍVEIS DE TRADE:")
        print(f"  Entry: {result['entry_price']:.5f}")
        print(f"  Stop Loss: {result['stop_loss']:.5f}")
        print(f"  Take Profit: {result['take_profit']:.5f}")

    print("\n" + "=" * 70)
    print("Teste concluído!")
    print("=" * 70)
