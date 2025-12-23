"""
================================================================================
FISHER-SHANNON INFORMATION GRAVITY ENGINE (FSIGE)
Indicador de Forex baseado em Geometria da Informacao
================================================================================

Este indicador baseia-se na premissa da Termodinamica Estatistica: um sistema
macroscopico (o mercado) evolui para estados que maximizam a entropia dada uma
restricao de energia. Aqui, utilizaremos a Geometria da Informacao (uma fusao
de Estatistica e Geometria Diferencial) para prever o fluxo de preco.

O Conceito: O Mercado como uma Variedade Estatistica
Nao olhe para o grafico de precos (x, y). Imagine que a cada instante t, a
distribuicao de probabilidade dos retornos do EURUSD, p(x|theta), e um "ponto" em
uma variedade (manifold) estatistica abstrata.

As coordenadas desse espaco nao sao precos, mas os parametros estatisticos
theta = (mu, sigma, kappa, ...) - media, desvio padrao, curtose, etc.

Por que usar Geometria da Informacao?
1. Independencia de Escala: A metrica de Fisher e invariante a mudancas de
   parametrizacao. A "distancia informacional" e absoluta.
2. Deteccao de Manipulacao: Movimentos manipulados (Stop Hunts) geralmente
   reduzem a entropia local (sao artificiais). Este indicador grita quando o
   preco vai para onde "nao deveria" estatisticamente.
3. Predictive Power: A Forca Entropica Causal preve o movimento antes da
   liquidez aparecer. O mercado se move para onde ele tem mais "opcoes" futuras.
================================================================================
"""

import numpy as np
from scipy.stats import gaussian_kde, entropy as scipy_entropy
from scipy.stats import moment, skew, kurtosis
from scipy.ndimage import uniform_filter1d
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

class SignalType(Enum):
    """Tipos de sinais de trading"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    HIBERNATE = "HIBERNATE"  # Sistema nao-adiabatico


class ThermodynamicState(Enum):
    """Estado termodinamico do mercado"""
    ADIABATIC = "ADIABATIC"        # Temperatura estavel (media vol)
    HEATING = "HEATING"             # Temperatura subindo (vol aumentando)
    COOLING = "COOLING"             # Temperatura caindo (vol diminuindo)
    PHASE_TRANSITION = "PHASE_TRANSITION"  # Mudanca de regime


@dataclass
class FisherMetric:
    """Tensor Metrico de Informacao de Fisher"""
    g: np.ndarray               # g_ij - tensor metrico
    g_inv: np.ndarray           # g^ij - inversa (contravariante)
    determinant: float          # det(g)
    eigenvalues: np.ndarray     # Autovalores (curvatura principal)
    curvature_scalar: float     # Curvatura escalar do espaco


@dataclass
class EntropicForce:
    """Forca Entropica Causal F_e = T*grad(S)"""
    force_vector: np.ndarray    # Vetor de forca no espaco de parametros
    magnitude: float            # |F_e|
    direction: np.ndarray       # Direcao unitaria
    temperature: float          # T - "temperatura" do mercado
    entropy_gradient: np.ndarray # grad(S)


@dataclass
class InformationFlowState:
    """Estado do fluxo de informacao"""
    theta: np.ndarray           # Parametros atuais theta = (mu, sigma, skew, kurt)
    d_theta_dt: np.ndarray      # Velocidade no espaco de parametros
    free_energy: float          # Phi - Potencial de energia livre
    entropy: float              # S - Entropia de Shannon


@dataclass
class FSIGESignal:
    """Sinal gerado pelo FSIGE"""
    signal_type: SignalType
    thermodynamic_state: ThermodynamicState
    confidence: float

    # Metricas de Fisher
    fisher_curvature: float
    information_resistance: float

    # Forca Entropica
    entropic_force_magnitude: float
    entropic_force_direction: str  # "BULLISH" ou "BEARISH"

    # Termodinamica
    temperature: float
    entropy: float
    free_energy: float

    # Tensao
    thermodynamic_tension: float   # Preco vs Gradiente Entropico
    tension_derivative: float      # Derivada da tensao (trigger)

    # Deteccao
    manipulation_score: float      # Score de manipulacao detectada

    reason: str
    timestamp: str


# ==============================================================================
# ESTIMACAO DE DENSIDADE KERNEL (KDE) DINAMICA
# ==============================================================================

class DynamicKDE:
    """
    Estimacao de Densidade Kernel (KDE) Dinamica

    A cada tick, usa uma janela deslizante para estimar a PDF (Probability
    Density Function) nao-parametrica dos retornos. Nao assume que e Gaussiana.
    Usa KDE com largura de banda adaptativa.
    """

    def __init__(self,
                 window_size: int = 100,
                 bandwidth_method: str = 'silverman'):
        """
        Args:
            window_size: Tamanho da janela para KDE
            bandwidth_method: Metodo para largura de banda ('scott' ou 'silverman')
        """
        self.window_size = window_size
        self.bandwidth_method = bandwidth_method

    def compute_returns(self, prices: np.ndarray) -> np.ndarray:
        """Calcula retornos logaritmicos"""
        returns = np.diff(np.log(prices + 1e-10))
        return returns

    def estimate_pdf(self,
                    returns: np.ndarray,
                    n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estima a PDF dos retornos via KDE

        Returns:
            Tupla (x_grid, pdf_values)
        """
        if len(returns) < 10:
            x = np.linspace(-0.01, 0.01, n_points)
            pdf = np.exp(-x**2 / 0.0001) / np.sqrt(np.pi * 0.0001)
            return x, pdf / (np.sum(pdf) * (x[1] - x[0]))

        try:
            kde = gaussian_kde(returns, bw_method=self.bandwidth_method)

            # Grid para avaliacao
            x_min = np.min(returns) - 3 * np.std(returns)
            x_max = np.max(returns) + 3 * np.std(returns)
            x = np.linspace(x_min, x_max, n_points)

            pdf = kde(x)

            # Normaliza
            pdf = pdf / (np.sum(pdf) * (x[1] - x[0]) + 1e-10)

        except Exception as e:
            x = np.linspace(-0.01, 0.01, n_points)
            std = np.std(returns) + 1e-10
            pdf = np.exp(-x**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))

        return x, pdf

    def compute_entropy(self, pdf: np.ndarray, dx: float) -> float:
        """
        Calcula Entropia de Shannon: H = -integral p(x) log p(x) dx
        """
        # Evita log(0)
        pdf_safe = np.clip(pdf, 1e-10, None)

        # Entropia diferencial
        entropy = -np.sum(pdf * np.log(pdf_safe)) * dx

        return entropy

    def compute_statistical_moments(self, returns: np.ndarray) -> np.ndarray:
        """
        Computa os momentos estatisticos theta = (mu, sigma, skewness, kurtosis)
        """
        if len(returns) < 4:
            return np.array([0.0, 0.001, 0.0, 3.0])

        mu = np.mean(returns)
        sigma = np.std(returns) + 1e-10
        skewness = skew(returns)
        kurt = kurtosis(returns, fisher=False)  # Kurtosis de Pearson

        # Limita valores extremos
        skewness = np.clip(skewness, -10, 10)
        kurt = np.clip(kurt, 1, 100)

        return np.array([mu, sigma, skewness, kurt])


# ==============================================================================
# TENSOR METRICO DE INFORMACAO DE FISHER
# ==============================================================================

class FisherInformationMetric:
    """
    Tensor Metrico de Informacao de Fisher (g_ij)

    Para medir a "distancia" entre o estado atual do mercado e o estado futuro,
    a distancia Euclidiana e inutil. Precisamos da Metrica de Fisher. Ela mede
    o quao distinguiveis sao duas distribuicoes de probabilidade.

    g_ij(theta) = E[(d/d_theta_i log p(x|theta))(d/d_theta_j log p(x|theta))]

    Este tensor define a curvatura do "espaco de informacao".

    Interpretacao: Onde a curvatura de Fisher e alta, a "resistencia informacional"
    a mudanca de preco e alta (o mercado "sabe" o preco justo). Onde e baixa,
    o preco desliza livremente.
    """

    def __init__(self,
                 n_parameters: int = 4,
                 regularization: float = 1e-6):
        """
        Args:
            n_parameters: Numero de parametros (4: mu, sigma, skew, kurt)
            regularization: Regularizacao de Tikhonov para inversao
        """
        self.n_parameters = n_parameters
        self.regularization = regularization

    def _numerical_derivative(self,
                             kde_func,
                             returns: np.ndarray,
                             x: np.ndarray,
                             param_idx: int,
                             epsilon: float = 1e-6) -> np.ndarray:
        """
        Calcula derivada numerica d(log p) / d(theta_i)
        """
        # Perturba os retornos simulando mudanca no parametro
        returns_plus = returns.copy()
        returns_minus = returns.copy()

        if param_idx == 0:  # media
            returns_plus = returns + epsilon
            returns_minus = returns - epsilon
        elif param_idx == 1:  # variancia
            returns_plus = returns * (1 + epsilon)
            returns_minus = returns * (1 - epsilon)
        elif param_idx == 2:  # skewness
            returns_plus = returns + epsilon * returns**2 * np.sign(returns)
            returns_minus = returns - epsilon * returns**2 * np.sign(returns)
        elif param_idx == 3:  # kurtosis
            returns_plus = returns + epsilon * returns**3
            returns_minus = returns - epsilon * returns**3

        try:
            kde_plus = gaussian_kde(returns_plus, bw_method='silverman')
            kde_minus = gaussian_kde(returns_minus, bw_method='silverman')

            pdf_plus = kde_plus(x) + 1e-10
            pdf_minus = kde_minus(x) + 1e-10

            # d(log p) / d(theta) ~= (log p+ - log p-) / (2*epsilon)
            d_log_p = (np.log(pdf_plus) - np.log(pdf_minus)) / (2 * epsilon)

        except:
            d_log_p = np.zeros_like(x)

        return d_log_p

    def compute_fisher_metric(self,
                             returns: np.ndarray,
                             x: np.ndarray,
                             pdf: np.ndarray) -> FisherMetric:
        """
        Calcula o Tensor de Fisher g_ij

        g_ij = E[(d log p / d theta_i)(d log p / d theta_j)]
             = integral (d log p / d theta_i)(d log p / d theta_j) p(x) dx
        """
        n = self.n_parameters
        g = np.zeros((n, n))

        dx = x[1] - x[0] if len(x) > 1 else 0.01

        # Calcula derivadas para cada parametro
        derivatives = []
        for i in range(n):
            d_log_p = self._numerical_derivative(gaussian_kde, returns, x, i)
            derivatives.append(d_log_p)

        # Monta tensor g_ij
        for i in range(n):
            for j in range(n):
                # g_ij = integral (d log p / d theta_i)(d log p / d theta_j) p(x) dx
                integrand = derivatives[i] * derivatives[j] * pdf
                g[i, j] = np.sum(integrand) * dx

        # Simetriza
        g = (g + g.T) / 2

        # Regularizacao de Tikhonov para estabilidade
        g = g + self.regularization * np.eye(n)

        # Garante positividade
        eigenvalues = np.linalg.eigvalsh(g)
        if np.min(eigenvalues) < self.regularization:
            g = g + (self.regularization - np.min(eigenvalues) + 0.01) * np.eye(n)
            eigenvalues = np.linalg.eigvalsh(g)

        # Calcula inversa com Tikhonov
        try:
            g_inv = np.linalg.inv(g)
        except:
            g_inv = np.linalg.pinv(g)

        # Determinante
        det = np.linalg.det(g)

        # Curvatura escalar (aproximacao via traco da Hessiana)
        curvature = np.trace(g) / n

        return FisherMetric(
            g=g,
            g_inv=g_inv,
            determinant=det,
            eigenvalues=eigenvalues,
            curvature_scalar=curvature
        )

    def compute_information_resistance(self, fisher_metric: FisherMetric) -> float:
        """
        Calcula a "resistencia informacional" a mudanca de preco

        Alta curvatura = mercado "sabe" o preco justo
        Baixa curvatura = preco desliza livremente
        """
        # Resistencia proporcional ao traco da metrica
        resistance = np.trace(fisher_metric.g)

        return resistance


# ==============================================================================
# FORCA ENTROPICA CAUSAL
# ==============================================================================

class CausalEntropicForce:
    """
    A Forca Entropica Causal (F_e)

    Teoria proposta por Wissner-Gross (Harvard/MIT). A inteligencia (ou o mercado)
    tenta maximizar a liberdade futura de acao. A forca que move o preco nao e
    compra/venda, e uma forca emergente estatistica:

    F_e = T*grad(S)

    Onde:
    - T e a "temperatura" do mercado (volatilidade latente)
    - grad(S) e o gradiente da Entropia de Shannon projetada no horizonte de tempo tau
      (Causal Path Entropy)

    O Calculo: Voce deve simular milhares de micro-caminhos futuros possiveis.
    A direcao "real" do preco sera aquela que maximiza a diversidade de caminhos
    futuros acessiveis (maxima entropia causal).
    """

    def __init__(self,
                 n_simulations: int = 1000,
                 horizon: int = 10,
                 n_directions: int = 8):
        """
        Args:
            n_simulations: Numero de simulacoes Monte Carlo por direcao
            horizon: Horizonte de tempo tau (em barras)
            n_directions: Numero de direcoes a explorar
        """
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.n_directions = n_directions

    def compute_temperature(self, returns: np.ndarray, window: int = 20) -> float:
        """
        Calcula a "temperatura" T do mercado (volatilidade latente)
        """
        if len(returns) < window:
            return np.std(returns) + 1e-10

        # Volatilidade realizada (temperatura)
        recent_returns = returns[-window:]
        temperature = np.std(recent_returns) + 1e-10

        return temperature

    def simulate_future_paths(self,
                             current_price: float,
                             returns: np.ndarray,
                             direction: float = 0.0) -> np.ndarray:
        """
        Simula caminhos futuros com vies direcional

        Args:
            current_price: Preco atual
            returns: Historico de retornos
            direction: Vies direcional (-1 a 1)

        Returns:
            Array de precos finais simulados
        """
        if len(returns) < 10:
            returns = np.random.randn(100) * 0.001

        mu = np.mean(returns) + direction * np.std(returns) * 0.5
        sigma = np.std(returns) + 1e-10

        # Simula caminhos via Monte Carlo
        final_prices = np.zeros(self.n_simulations)

        for sim in range(self.n_simulations):
            price = current_price
            for t in range(self.horizon):
                ret = np.random.normal(mu, sigma)
                price = price * np.exp(ret)
            final_prices[sim] = price

        return final_prices

    def compute_path_entropy(self, final_prices: np.ndarray) -> float:
        """
        Calcula a entropia dos caminhos futuros

        Mais diversidade de precos finais = maior entropia = mais "opcoes"
        """
        if len(final_prices) < 10:
            return 0.0

        # Usa KDE para estimar PDF dos precos finais
        try:
            kde = gaussian_kde(final_prices, bw_method='silverman')
            x = np.linspace(np.min(final_prices), np.max(final_prices), 100)
            pdf = kde(x)
            pdf = pdf / (np.sum(pdf) + 1e-10)

            # Entropia
            entropy = -np.sum(pdf * np.log(pdf + 1e-10))
        except:
            entropy = np.log(np.std(final_prices) + 1e-10)

        return entropy

    def compute_entropy_gradient(self,
                                current_price: float,
                                returns: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calcula o gradiente da entropia grad(S)

        Perturba o sistema em direcoes diferentes e mede a variacao da
        Entropia de Shannon. O vetor que aponta para o maior ganho de
        entropia e a Direcao da Forca Entropica.
        """
        directions = np.linspace(-1, 1, self.n_directions)
        entropies = []

        for direction in directions:
            final_prices = self.simulate_future_paths(
                current_price, returns, direction
            )
            entropy = self.compute_path_entropy(final_prices)
            entropies.append(entropy)

        entropies = np.array(entropies)

        # Encontra direcao de maxima entropia
        max_idx = np.argmax(entropies)

        # Gradiente: diferenca entre entropia maxima e atual
        current_entropy = entropies[len(directions) // 2]  # Direcao neutra
        gradient_magnitude = entropies[max_idx] - current_entropy
        gradient_direction = directions[max_idx]

        # Vetor gradiente (simplificado para 1D no espaco de preco)
        # Em teoria completa seria no espaco de parametros theta
        gradient = np.array([gradient_direction, gradient_magnitude, 0, 0])

        return gradient, entropies[max_idx]

    def compute_entropic_force(self,
                              current_price: float,
                              returns: np.ndarray,
                              theta: np.ndarray) -> EntropicForce:
        """
        Calcula a Forca Entropica Causal F_e = T*grad(S)
        """
        # Temperatura
        temperature = self.compute_temperature(returns)

        # Gradiente de entropia
        gradient, max_entropy = self.compute_entropy_gradient(current_price, returns)

        # Forca entropica
        force_vector = temperature * gradient
        magnitude = np.linalg.norm(force_vector)

        if magnitude > 0:
            direction = force_vector / magnitude
        else:
            direction = np.zeros_like(force_vector)

        return EntropicForce(
            force_vector=force_vector,
            magnitude=magnitude,
            direction=direction,
            temperature=temperature,
            entropy_gradient=gradient
        )


# ==============================================================================
# EQUACAO DE DIFUSAO DE INFORMACAO
# ==============================================================================

class InformationDiffusion:
    """
    A Equacao de Difusao de Informacao

    Combine os dois conceitos. O fluxo de preco segue o Fluxo Gradiente
    (Gradient Flow) na variedade de Fisher:

    d(theta)/dt = -g^ij * d(Phi)/d(theta_j)

    Onde Phi e o Potencial de Energia Livre (Free Energy Potential) do sistema.
    """

    def __init__(self):
        pass

    def compute_free_energy(self,
                           entropy: float,
                           temperature: float,
                           internal_energy: float = 0.0) -> float:
        """
        Calcula o Potencial de Energia Livre

        Phi = U - TS (termodinamica)

        Para o mercado, U e aproximado pela "energia" do movimento de preco
        """
        # Energia livre de Helmholtz
        free_energy = internal_energy - temperature * entropy

        return free_energy

    def compute_gradient_flow(self,
                             theta: np.ndarray,
                             fisher_metric: FisherMetric,
                             free_energy_gradient: np.ndarray) -> np.ndarray:
        """
        Calcula o fluxo gradiente d(theta)/dt = -g^ij * d(Phi)/d(theta_j)
        """
        g_inv = fisher_metric.g_inv

        # d(theta)/dt = -g^ij * d(Phi)/d(theta_j)
        d_theta_dt = -np.dot(g_inv, free_energy_gradient)

        return d_theta_dt

    def compute_information_state(self,
                                 theta: np.ndarray,
                                 fisher_metric: FisherMetric,
                                 entropy: float,
                                 temperature: float,
                                 entropic_force: EntropicForce) -> InformationFlowState:
        """
        Computa o estado completo do fluxo de informacao
        """
        # Energia interna (aproximada pelo momento de preco)
        internal_energy = 0.5 * np.sum(theta**2)

        # Energia livre
        free_energy = self.compute_free_energy(entropy, temperature, internal_energy)

        # Gradiente da energia livre (aproximado pela forca entropica negativa)
        free_energy_gradient = -entropic_force.force_vector

        # Fluxo gradiente
        d_theta_dt = self.compute_gradient_flow(theta, fisher_metric, free_energy_gradient)

        return InformationFlowState(
            theta=theta,
            d_theta_dt=d_theta_dt,
            free_energy=free_energy,
            entropy=entropy
        )


# ==============================================================================
# DETECTOR DE MANIPULACAO
# ==============================================================================

class ManipulationDetector:
    """
    Detecta movimentos manipulados (Stop Hunts)

    Movimentos manipulados geralmente reduzem a entropia local (sao artificiais).
    Este indicador "grita" quando o preco vai para onde "nao deveria"
    estatisticamente.
    """

    def __init__(self, window: int = 20, threshold: float = 2.0):
        self.window = window
        self.threshold = threshold

    def compute_manipulation_score(self,
                                   returns: np.ndarray,
                                   entropy_history: List[float]) -> float:
        """
        Calcula score de manipulacao

        Score alto = movimento provavelmente artificial
        """
        if len(entropy_history) < 5:
            return 0.0

        # Verifica queda subita de entropia (manipulacao)
        recent_entropy = np.array(entropy_history[-5:])
        entropy_drop = recent_entropy[0] - recent_entropy[-1]

        # Verifica movimento extremo de preco
        if len(returns) >= self.window:
            recent_returns = returns[-self.window:]
            z_score = np.abs(recent_returns[-1]) / (np.std(recent_returns) + 1e-10)
        else:
            z_score = 0.0

        # Score: combinacao de queda de entropia e movimento extremo
        manipulation_score = max(0, entropy_drop) * z_score

        return manipulation_score


# ==============================================================================
# INDICADOR FSIGE COMPLETO
# ==============================================================================

class FisherShannonInformationGravityEngine:
    """
    Fisher-Shannon Information Gravity Engine (FSIGE)

    Indicador completo que usa Geometria da Informacao e Termodinamica
    Estatistica para prever movimentos de preco.

    Sinal de Trading (O Colapso da Incerteza):
    Em media volatilidade, o mercado oscila entre estados de alta entropia
    (incerteza maxima) e baixa entropia (informacao resolvida).

    SINAL DE SNIPER:
    - Quando o preco se move CONTRA o Gradiente Entropico (o preco esta sendo
      forcado artificialmente para uma zona de baixa probabilidade/baixa entropia
      por um player grande), gera-se uma "tensao termodinamica".
    - A Geodesica de Fisher atua como um elastico.
    - Entrada: No momento em que a derivada da Forca Entropica inverte (o mercado
      desiste de lutar contra a entropia), o preco ira COLAPSAR violentamente na
      direcao do Gradiente de Entropia Maxima para restaurar o equilibrio.

    Basicamente, voce aposta a favor da 2a Lei da Termodinamica. E a aposta mais
    segura do universo.
    """

    def __init__(self,
                 # Parametros de KDE
                 kde_window: int = 100,

                 # Parametros de Fisher
                 n_parameters: int = 4,
                 regularization: float = 1e-4,

                 # Parametros de Forca Entropica
                 n_simulations: int = 500,
                 entropy_horizon: int = 10,

                 # Parametros de regime
                 temperature_stable_threshold: float = 0.3,

                 # Parametros de tensao
                 tension_threshold: float = 0.5,

                 # Geral
                 min_data_points: int = 100):
        """
        Inicializa o FSIGE
        """
        self.kde_window = kde_window
        self.n_parameters = n_parameters
        self.temperature_stable_threshold = temperature_stable_threshold
        self.tension_threshold = tension_threshold
        self.min_data_points = min_data_points

        # Componentes
        self.kde = DynamicKDE(window_size=kde_window)
        self.fisher = FisherInformationMetric(
            n_parameters=n_parameters,
            regularization=regularization
        )
        self.entropic_force = CausalEntropicForce(
            n_simulations=n_simulations,
            horizon=entropy_horizon
        )
        self.diffusion = InformationDiffusion()
        self.manipulation_detector = ManipulationDetector()

        # Historico
        self.entropy_history: List[float] = []
        self.temperature_history: List[float] = []
        self.tension_history: List[float] = []
        self.force_direction_history: List[float] = []

    def _compute_thermodynamic_tension(self,
                                       price_direction: float,
                                       entropic_direction: float) -> float:
        """
        Calcula a "tensao termodinamica" entre preco e entropia

        Tensao alta = preco movendo contra o gradiente entropico
        """
        # Tensao = quanto o preco esta "lutando" contra a entropia
        tension = -price_direction * entropic_direction

        return tension

    def _determine_thermodynamic_state(self) -> ThermodynamicState:
        """
        Determina o estado termodinamico atual
        """
        if len(self.temperature_history) < 5:
            return ThermodynamicState.PHASE_TRANSITION

        recent_temp = np.array(self.temperature_history[-5:])
        temp_change = (recent_temp[-1] - recent_temp[0]) / (recent_temp[0] + 1e-10)

        if abs(temp_change) < self.temperature_stable_threshold:
            return ThermodynamicState.ADIABATIC
        elif temp_change > self.temperature_stable_threshold:
            return ThermodynamicState.HEATING
        elif temp_change < -self.temperature_stable_threshold:
            return ThermodynamicState.COOLING
        else:
            return ThermodynamicState.PHASE_TRANSITION

    def _detect_tension_reversal(self) -> Tuple[bool, float]:
        """
        Detecta reversao da tensao termodinamica (trigger de entrada)
        """
        if len(self.tension_history) < 3:
            return False, 0.0

        recent_tension = np.array(self.tension_history[-5:])

        # Derivada da tensao
        tension_derivative = np.gradient(recent_tension)[-1]

        # Reversao: tensao alta seguida de queda
        if (recent_tension[-2] > self.tension_threshold and
            tension_derivative < -0.1):
            return True, tension_derivative

        return False, tension_derivative

    def analyze(self, prices: np.ndarray) -> dict:
        """
        Analisa dados de preco e retorna resultado completo

        Args:
            prices: Array de precos

        Returns:
            Dict com analise completa
        """
        from datetime import datetime

        n = len(prices)

        # Validacao
        if n < self.min_data_points:
            return {
                'signal': 0,
                'signal_name': 'INSUFFICIENT_DATA',
                'thermodynamic_state': 'PHASE_TRANSITION',
                'confidence': 0.0,
                'fisher_curvature': 0.0,
                'information_resistance': 0.0,
                'entropic_force_magnitude': 0.0,
                'entropic_force_direction': 'NEUTRAL',
                'temperature': 0.0,
                'entropy': 0.0,
                'free_energy': 0.0,
                'thermodynamic_tension': 0.0,
                'tension_derivative': 0.0,
                'manipulation_score': 0.0,
                'reasons': ['dados_insuficientes'],
                'current_price': prices[-1]
            }

        # PASSO 1: KDE E MOMENTOS ESTATISTICOS
        returns = self.kde.compute_returns(prices)
        x, pdf = self.kde.estimate_pdf(returns[-self.kde_window:])
        dx = x[1] - x[0] if len(x) > 1 else 0.01

        theta = self.kde.compute_statistical_moments(returns[-self.kde_window:])
        entropy = self.kde.compute_entropy(pdf, dx)

        self.entropy_history.append(entropy)
        if len(self.entropy_history) > 50:
            self.entropy_history.pop(0)

        # PASSO 2: TENSOR DE FISHER
        fisher_metric = self.fisher.compute_fisher_metric(
            returns[-self.kde_window:], x, pdf
        )
        information_resistance = self.fisher.compute_information_resistance(fisher_metric)

        # PASSO 3: FORCA ENTROPICA CAUSAL
        entropic = self.entropic_force.compute_entropic_force(
            prices[-1], returns, theta
        )

        self.temperature_history.append(entropic.temperature)
        if len(self.temperature_history) > 50:
            self.temperature_history.pop(0)

        # Direcao entropica
        entropic_direction = entropic.direction[0] if len(entropic.direction) > 0 else 0.0
        self.force_direction_history.append(entropic_direction)
        if len(self.force_direction_history) > 50:
            self.force_direction_history.pop(0)

        # PASSO 4: ESTADO TERMODINAMICO E TENSAO
        thermo_state = self._determine_thermodynamic_state()

        # Direcao do preco recente
        if len(returns) >= 5:
            price_direction = np.sign(np.mean(returns[-5:]))
        else:
            price_direction = 0.0

        # Tensao termodinamica
        tension = self._compute_thermodynamic_tension(price_direction, entropic_direction)
        self.tension_history.append(tension)
        if len(self.tension_history) > 50:
            self.tension_history.pop(0)

        # PASSO 5: DETECCAO DE MANIPULACAO
        manipulation_score = self.manipulation_detector.compute_manipulation_score(
            returns, self.entropy_history
        )

        # PASSO 6: GERACAO DE SINAL
        tension_reversal, tension_derivative = self._detect_tension_reversal()

        # Energia livre
        free_energy = self.diffusion.compute_free_energy(
            entropy, entropic.temperature
        )

        # Logica de sinal
        signal = 0
        signal_name = "NEUTRAL"
        confidence = 0.0
        reasons = []

        entropic_dir_str = "BULLISH" if entropic_direction > 0.1 else (
            "BEARISH" if entropic_direction < -0.1 else "NEUTRAL"
        )

        # Sistema nao-adiabatico - hiberna
        if thermo_state in [ThermodynamicState.HEATING, ThermodynamicState.PHASE_TRANSITION]:
            signal_name = "HIBERNATE"
            reasons.append(f"sistema_{thermo_state.value}")
            reasons.append(f"T={entropic.temperature:.6f}")

        # Sistema adiabatico - procura sinais
        elif thermo_state == ThermodynamicState.ADIABATIC:

            # SINAL DE SNIPER: Reversao de tensao
            if tension_reversal and abs(tension) > self.tension_threshold:

                # Direcao do colapso = direcao do gradiente entropico
                if entropic_direction > 0.1:
                    signal = 1
                    signal_name = "LONG"
                    confidence = min(1.0, abs(tension) * entropic.magnitude)
                    reasons.append("colapso_entropico_bullish")
                    reasons.append(f"tensao={tension:.3f}")
                    reasons.append("2a_lei_termodinamica")

                elif entropic_direction < -0.1:
                    signal = -1
                    signal_name = "SHORT"
                    confidence = min(1.0, abs(tension) * entropic.magnitude)
                    reasons.append("colapso_entropico_bearish")
                    reasons.append(f"tensao={tension:.3f}")
                    reasons.append("2a_lei_termodinamica")

                else:
                    reasons.append("tensao_alta_direcao_neutra")

            # Alta tensao mas sem reversao ainda
            elif abs(tension) > self.tension_threshold:
                reasons.append(f"tensao_alta_{tension:.3f}")
                reasons.append("aguardando_reversao")

            # Manipulacao detectada
            elif manipulation_score > 1.5:
                if entropic_direction > 0.1:
                    signal = 1
                    signal_name = "LONG"
                    confidence = min(0.7, manipulation_score * 0.3)
                    reasons.append("anti_manipulacao_bullish")
                    reasons.append(f"score={manipulation_score:.2f}")
                elif entropic_direction < -0.1:
                    signal = -1
                    signal_name = "SHORT"
                    confidence = min(0.7, manipulation_score * 0.3)
                    reasons.append("anti_manipulacao_bearish")
                    reasons.append(f"score={manipulation_score:.2f}")
                else:
                    reasons.append("manipulacao_direcao_indefinida")

            else:
                reasons.append("sistema_adiabatico")
                reasons.append("sem_trigger")

        else:
            reasons.append(f"estado_{thermo_state.value}")
            reasons.append("aguardando_estabilizacao")

        # Ajusta confianca pela resistencia informacional
        if confidence > 0:
            resistance_factor = 1.0 / (1.0 + information_resistance * 0.1)
            confidence *= resistance_factor
            confidence = np.clip(confidence, 0, 1)

        return {
            'signal': signal,
            'signal_name': signal_name,
            'thermodynamic_state': thermo_state.value,
            'confidence': confidence,
            'fisher_curvature': fisher_metric.curvature_scalar,
            'information_resistance': information_resistance,
            'entropic_force_magnitude': entropic.magnitude,
            'entropic_force_direction': entropic_dir_str,
            'temperature': entropic.temperature,
            'entropy': entropy,
            'free_energy': free_energy,
            'thermodynamic_tension': tension,
            'tension_derivative': tension_derivative,
            'manipulation_score': manipulation_score,
            'theta': theta.tolist(),
            'reasons': reasons,
            'current_price': prices[-1]
        }

    def get_entropy_history(self) -> np.ndarray:
        """Retorna historico de entropia"""
        return np.array(self.entropy_history)

    def get_temperature_history(self) -> np.ndarray:
        """Retorna historico de temperatura"""
        return np.array(self.temperature_history)

    def get_tension_history(self) -> np.ndarray:
        """Retorna historico de tensao"""
        return np.array(self.tension_history)

    def reset(self):
        """Reseta o estado do indicador"""
        self.entropy_history.clear()
        self.temperature_history.clear()
        self.tension_history.clear()
        self.force_direction_history.clear()


# ==============================================================================
# EXEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FISHER-SHANNON INFORMATION GRAVITY ENGINE (FSIGE)")
    print("Indicador baseado em Geometria da Informacao")
    print("=" * 70)

    np.random.seed(42)

    # Gera dados mean-reverting
    n_points = 200
    mu = 1.0850
    theta_ou = 0.1
    sigma = 0.0003

    prices = np.zeros(n_points)
    prices[0] = mu

    for i in range(1, n_points):
        dW = np.random.randn()
        prices[i] = prices[i-1] + theta_ou * (mu - prices[i-1]) + sigma * dW

    print(f"\nDados: {len(prices)} pontos")
    print(f"Preco: {prices[0]:.5f} -> {prices[-1]:.5f}")

    # Cria indicador
    indicator = FisherShannonInformationGravityEngine(
        kde_window=80,
        n_simulations=300,
        entropy_horizon=10,
        temperature_stable_threshold=0.3,
        tension_threshold=0.3,
        min_data_points=80
    )

    print("\nAnalisando Geometria da Informacao...")

    result = indicator.analyze(prices)

    print("\n" + "-" * 40)
    print("RESULTADO:")
    print(f"  Sinal: {result['signal_name']}")
    print(f"  Estado: {result['thermodynamic_state']}")
    print(f"  Confianca: {result['confidence']:.0%}")

    print("\nFISHER:")
    print(f"  Curvatura: {result['fisher_curvature']:.4f}")
    print(f"  Resistencia: {result['information_resistance']:.4f}")

    print("\nENTROPIA:")
    print(f"  Forca: {result['entropic_force_magnitude']:.6f}")
    print(f"  Direcao: {result['entropic_force_direction']}")

    print("\nTERMODINAMICA:")
    print(f"  Temperatura: {result['temperature']:.6f}")
    print(f"  Entropia: {result['entropy']:.4f}")
    print(f"  Energia Livre: {result['free_energy']:.4f}")

    print("\nTENSAO:")
    print(f"  Tensao: {result['thermodynamic_tension']:.4f}")
    print(f"  Derivada: {result['tension_derivative']:.4f}")

    print(f"\nManipulacao: {result['manipulation_score']:.4f}")

    print("\n" + "=" * 70)
    print("Teste concluido!")
    print("=" * 70)
