"""
================================================================================
KORTEWEG-DE VRIES SOLITON HUNTER (KdV-SH)
Indicador de Forex baseado em Equacoes Diferenciais Parciais
================================================================================

Este indicador resolve Equacoes Diferenciais Parciais (PDEs) para detectar
Ondas Solitarias (Solitons). Um soliton e uma onda que mantem sua forma
enquanto viaja a velocidade constante.

No mercado, isso e o movimento perfeito de swing: limpo, forte e previsivel,
que ocorre APENAS quando a dispersao e a nao-linearidade do mercado se cancelam
perfeitamente (o equilibrio da media volatilidade).

A Fisica do Problema:
Voce vai simular o "livro de ofertas" (Order Book) como um canal de agua raso
onde o preco flutua.

Por que usar Dinamica de Fluidos (KdV)?
1. Fisica Real: O mercado e um fluxo de ordens. Trata-lo como fluido e mais
   realista do que trata-lo como estatistica.
2. Preditividade Deterministica: Diferente de modelos probabilisticos (GARCH),
   a equacao KdV e deterministica. Se um Soliton existe, ele VAI se comportar
   de maneira X a menos que uma forca externa (News) destrua o sistema.
3. Separacao Sinal/Ruido: O metodo Inverse Scattering Transform separa
   matematicamente a parte "Soliton" (Sinal Estrutural) da parte "Radiacao"
   (Ruido Oscilatorio). Voce opera o sinal puro.

Arquitetura:
1. Modelo Hidrodinamico (Navier-Stokes Simplificado)
2. Equacao Mestra: Korteweg-de Vries (KdV)
3. Inverse Scattering Transform (IST)
4. Deteccao de Solitons via autovalores de Schrodinger
5. Simulacao Preditiva Split-Step Fourier
================================================================================
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.fft import fft, ifft, fftfreq
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

class SignalType(Enum):
    """Tipos de sinais de trading"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    HIBERNATE = "HIBERNATE"  # Fora do regime KdV


class SolitonType(Enum):
    """Tipos de solitons detectados"""
    BUYING_SOLITON = "BUYING_SOLITON"    # Onda de compra
    SELLING_SOLITON = "SELLING_SOLITON"  # Onda de venda
    NO_SOLITON = "NO_SOLITON"            # Apenas radiacao dispersiva


@dataclass
class Soliton:
    """Representa um soliton detectado"""
    eigenvalue: float           # lambda_n - autovalor
    amplitude: float            # Amplitude do soliton
    velocity: float             # Velocidade de propagacao
    position: float             # Posicao atual
    soliton_type: SolitonType   # Tipo (compra/venda)
    width: float                # Largura do soliton

    @property
    def energy(self) -> float:
        """Energia do soliton (proporcional a amplitude^2)"""
        return self.amplitude ** 2


@dataclass
class ScatteringData:
    """Dados do Inverse Scattering Transform"""
    eigenvalues: np.ndarray           # Autovalores discretos (solitons)
    eigenfunctions: np.ndarray        # Autofuncoes correspondentes
    reflection_coefficient: np.ndarray # Coeficiente de reflexao (radiacao)
    n_solitons: int                   # Numero de solitons detectados


@dataclass
class KdVSolution:
    """Solucao da equacao KdV"""
    phi: np.ndarray              # Campo phi(x, t)
    x_grid: np.ndarray           # Grid espacial
    t_current: float             # Tempo atual
    t_predicted: float           # Tempo de previsao
    phi_predicted: np.ndarray    # Campo previsto phi(x, t+delta)


@dataclass
class UrsellNumber:
    """Numero de Ursell para regime check"""
    value: float
    amplitude: float
    wavelength: float
    depth: float
    is_kdv_regime: bool

    @property
    def regime_description(self) -> str:
        if self.value < 0.5:
            return "LINEAR (Ur << 1)"
        elif self.value > 2.0:
            return "TURBULENTO (Ur >> 1)"
        else:
            return "KdV (Ur ~= 1)"


@dataclass
class KdVSignal:
    """Sinal gerado pelo KdV-SH"""
    signal_type: SignalType
    confidence: float
    ursell_number: float
    n_solitons: int
    solitons: List[Soliton]
    dominant_soliton: Optional[Soliton]
    collision_predicted: bool
    collision_time: float          # Tempo ate colisao (em barras)
    collision_type: str            # "SUPPORT" ou "RESISTANCE"
    predicted_reversal_price: float
    reason: str
    timestamp: str


# ==============================================================================
# MODELO HIDRODINAMICO
# ==============================================================================

class HydrodynamicModel:
    """
    Modelo Hidrodinamico (Navier-Stokes Simplificado)

    Primeiro, defina o campo de velocidade do mercado u(x,t) e a altura
    da "superficie" do preco eta(x,t). A viscosidade nu do fluido e dada
    pela liquidez do Order Book (L2 Data).

    - Alta liquidez = Alta viscosidade (o preco se move devagar, como mel)
    - Baixa liquidez = Baixa viscosidade (o preco se move rapido, como agua)
    """

    def __init__(self,
                 base_viscosity: float = 0.01,
                 depth: float = 1.0):
        """
        Args:
            base_viscosity: Viscosidade base do fluido
            depth: Profundidade do canal (normalizada)
        """
        self.base_viscosity = base_viscosity
        self.depth = depth

    def compute_velocity_field(self,
                               prices: np.ndarray,
                               volumes: np.ndarray) -> np.ndarray:
        """
        Computa o campo de velocidade u(x,t) do mercado

        u = dp/dt ponderado pelo volume
        """
        # Velocidade = variacao de preco
        velocity = np.gradient(prices)

        # Pondera pelo volume (normalizado)
        volume_norm = volumes / (np.mean(volumes) + 1e-10)

        # Velocidade efetiva
        u = velocity * np.sqrt(volume_norm)

        return u

    def compute_surface_height(self, prices: np.ndarray) -> np.ndarray:
        """
        Computa a altura da superficie eta(x,t)

        eta = preco normalizado relativo a media
        """
        mean_price = np.mean(prices)
        std_price = np.std(prices) + 1e-10

        # Altura normalizada
        eta = (prices - mean_price) / std_price

        return eta

    def compute_viscosity(self, volumes: np.ndarray) -> np.ndarray:
        """
        Computa viscosidade variavel baseada na liquidez

        nu = nu_base * (V / V_mean)
        """
        volume_norm = volumes / (np.mean(volumes) + 1e-10)

        # Alta liquidez = alta viscosidade
        viscosity = self.base_viscosity * volume_norm

        return viscosity

    def compute_topography(self,
                          prices: np.ndarray,
                          volumes: np.ndarray) -> np.ndarray:
        """
        Computa a topografia de fundo B(x) baseada no Volume Profile

        O "fluido" do preco corre sobre essa topografia.
        Regioes de alto volume = "montanhas" (resistencia)
        Regioes de baixo volume = "vales" (suporte)
        """
        # Suaviza o volume para criar perfil
        window = min(20, len(volumes) // 5)
        volume_profile = uniform_filter1d(volumes, size=window)

        # Normaliza
        volume_profile = volume_profile / (np.max(volume_profile) + 1e-10)

        # Topografia: alto volume = barreira
        topography = volume_profile

        return topography


# ==============================================================================
# EQUACAO KORTEWEG-DE VRIES
# ==============================================================================

class KdVEquation:
    """
    A Equacao Mestra: Korteweg-de Vries (KdV)

    A equacao KdV modela a evolucao de ondas em meios dispersivos fracos.
    E aqui que a magica acontece. Voce deve resolver numericamente para
    o campo de preco phi(x,t):

    dphi/dt + d^3phi/dx^3 + 6*phi*(dphi/dx) = 0

    Onde:
    - Termo Dispersivo (d^3phi/dx^3): Representa a microestrutura do mercado
      (HFTs e arbitragem) que tenta ESPALHAR o preco.
    - Termo Nao-Linear (6*phi*dphi/dx): Representa a pressao de momentum dos
      grandes players que tenta CONCENTRAR o preco (formar tendencia).

    O Milagre da Media Volatilidade: Quando esses dois termos se equilibram
    perfeitamente, nasce um Soliton. Uma onda de reversao que nao se dissipa.
    """

    def __init__(self,
                 n_points: int = 256,
                 x_length: float = 100.0,
                 nonlinear_coeff: float = 6.0):
        """
        Args:
            n_points: Numero de pontos no grid espacial (potencia de 2 para FFT)
            x_length: Comprimento do dominio espacial
            nonlinear_coeff: Coeficiente do termo nao-linear (padrao 6 para KdV)
        """
        self.n_points = n_points
        self.x_length = x_length
        self.nonlinear_coeff = nonlinear_coeff

        # Grid espacial
        self.dx = x_length / n_points
        self.x = np.linspace(0, x_length, n_points, endpoint=False)

        # Numeros de onda para FFT
        self.k = fftfreq(n_points, d=self.dx) * 2 * np.pi

    def initial_condition_from_prices(self, prices: np.ndarray) -> np.ndarray:
        """
        Converte serie de precos em condicao inicial phi(x, 0)
        """
        # Normaliza precos
        prices_norm = (prices - np.mean(prices)) / (np.std(prices) + 1e-10)

        # Interpola para o grid
        x_original = np.linspace(0, self.x_length, len(prices))
        interpolator = interp1d(x_original, prices_norm, kind='cubic',
                               fill_value='extrapolate')

        phi_0 = interpolator(self.x)

        return phi_0

    def solve_split_step_fourier(self,
                                 phi_0: np.ndarray,
                                 dt: float = 0.001,
                                 n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resolve a equacao KdV usando o metodo Split-Step Fourier

        O metodo divide a evolucao em:
        1. Passo linear (dispersao) no espaco de Fourier
        2. Passo nao-linear no espaco real

        Args:
            phi_0: Condicao inicial
            dt: Passo de tempo
            n_steps: Numero de passos

        Returns:
            Tupla (phi_final, phi_history)
        """
        phi = phi_0.copy().astype(complex)
        phi_history = [phi.real.copy()]

        # Operador linear (dispersao): exp(-i k^3 dt)
        linear_operator = np.exp(-1j * self.k**3 * dt)

        for step in range(n_steps):
            # Passo 1: Meio passo nao-linear
            phi = phi * np.exp(-1j * self.nonlinear_coeff * np.abs(phi)**2 * dt / 2)

            # Passo 2: Passo linear completo (no espaco de Fourier)
            phi_hat = fft(phi)
            phi_hat = phi_hat * linear_operator
            phi = ifft(phi_hat)

            # Passo 3: Meio passo nao-linear
            phi = phi * np.exp(-1j * self.nonlinear_coeff * np.abs(phi)**2 * dt / 2)

            # Metodo alternativo mais estavel para KdV real
            # Usa a forma: dphi/dt = -d^3phi/dx^3 - 6*phi*dphi/dx
            phi_real = phi.real

            # Termo dispersivo via FFT
            phi_hat = fft(phi_real)
            d3phi_dx3 = ifft(-(1j * self.k)**3 * phi_hat).real

            # Termo nao-linear
            dphi_dx = ifft(1j * self.k * phi_hat).real
            nonlinear = self.nonlinear_coeff * phi_real * dphi_dx

            # Evolui
            phi = phi_real - dt * (d3phi_dx3 + nonlinear)
            phi = phi.astype(complex)

            if step % 10 == 0:
                phi_history.append(phi.real.copy())

        return phi.real, np.array(phi_history)

    def solve_pseudospectral(self,
                            phi_0: np.ndarray,
                            dt: float = 0.0001,
                            n_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resolve a equacao KdV usando metodo pseudo-espectral

        Mais estavel para a equacao KdV padrao.
        """
        phi = phi_0.copy()
        phi_history = [phi.copy()]

        # Integrador RK4 no espaco de Fourier
        def kdv_rhs(phi):
            """Lado direito da equacao KdV"""
            phi_hat = fft(phi)

            # d^3phi/dx^3
            d3phi = ifft(-(1j * self.k)**3 * phi_hat).real

            # dphi/dx
            dphi = ifft(1j * self.k * phi_hat).real

            # -d^3phi/dx^3 - 6*phi*dphi/dx
            return -d3phi - self.nonlinear_coeff * phi * dphi

        for step in range(n_steps):
            # Runge-Kutta 4
            k1 = dt * kdv_rhs(phi)
            k2 = dt * kdv_rhs(phi + 0.5 * k1)
            k3 = dt * kdv_rhs(phi + 0.5 * k2)
            k4 = dt * kdv_rhs(phi + k3)

            phi = phi + (k1 + 2*k2 + 2*k3 + k4) / 6

            # Filtro de estabilizacao (remove altas frequencias)
            phi_hat = fft(phi)
            # Filtro suave
            filter_mask = np.exp(-(self.k / (0.5 * np.max(np.abs(self.k))))**8)
            phi = ifft(phi_hat * filter_mask).real

            if step % 100 == 0:
                phi_history.append(phi.copy())

        return phi, np.array(phi_history)


# ==============================================================================
# INVERSE SCATTERING TRANSFORM (IST)
# ==============================================================================

class InverseScatteringTransform:
    """
    A Transformada Inversa de Espalhamento (Inverse Scattering Transform - IST)

    Esta e a parte pesada. O metodo IST e usado para resolver a equacao KdV
    nao-linear.

    Passo 1: Considere o perfil de preco atual como um "potencial" na equacao
    de Schrodinger (sim, a fisica quantica retorna aqui como ferramenta
    matematica auxiliar):

    -d^2psi/dx^2 + V(x)*psi = lambda*psi

    Onde V(x) e o negativo do movimento de preco recente.

    Passo 2: Calcule os autovalores discretos (lambda_n) desse operador de Schrodinger.

    O SINAL (Eigenvalue Discrete Spectrum):
    - Se nao houver autovalores discretos (lambda), o mercado e apenas radiacao
      dispersiva (ruido).
    - Cada autovalor discreto (lambda_n) corresponde a UM SOLITON escondido nos dados.
    - A magnitude de lambda_n determina a amplitude e a velocidade do Soliton.
    """

    def __init__(self,
                 n_points: int = 256,
                 eigenvalue_threshold: float = 0.01):
        """
        Args:
            n_points: Pontos no grid
            eigenvalue_threshold: Threshold para considerar autovalor como soliton
        """
        self.n_points = n_points
        self.eigenvalue_threshold = eigenvalue_threshold

    def construct_schrodinger_potential(self,
                                        prices: np.ndarray) -> np.ndarray:
        """
        Constroi o potencial V(x) para a equacao de Schrodinger

        V(x) = -phi(x) onde phi e o perfil de preco normalizado

        Isso transforma o problema de encontrar solitons em encontrar
        estados ligados de uma particula quantica.
        """
        # Normaliza
        phi = (prices - np.mean(prices)) / (np.std(prices) + 1e-10)

        # Interpola para grid uniforme
        x_orig = np.linspace(0, 1, len(prices))
        x_new = np.linspace(0, 1, self.n_points)

        interpolator = interp1d(x_orig, phi, kind='cubic', fill_value='extrapolate')
        phi_interp = interpolator(x_new)

        # Potencial: V(x) = -6 * phi(x) para KdV padrao
        # O fator 6 vem da equacao KdV
        V = -6 * phi_interp

        return V

    def solve_schrodinger_eigenvalues(self,
                                      V: np.ndarray,
                                      dx: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resolve o problema de autovalores de Schrodinger

        -d^2psi/dx^2 + V(x)*psi = lambda*psi

        Usando discretizacao de diferencas finitas, isso se torna um problema
        de autovalores tridiagonal.

        Usa scipy.linalg.eigh_tridiagonal para eficiencia.
        """
        n = len(V)

        # Discretizacao: -psi''(x) ~= -(psi_{i+1} - 2*psi_i + psi_{i-1}) / dx^2
        # Isso da uma matriz tridiagonal:
        # d (diagonal): 2/dx^2 + V_i
        # e (off-diagonal): -1/dx^2

        # Diagonal principal
        d = 2.0 / dx**2 + V

        # Off-diagonal (simetrica)
        e = -np.ones(n - 1) / dx**2

        try:
            # Resolve problema de autovalores tridiagonal
            eigenvalues, eigenvectors = eigh_tridiagonal(d, e)

            # Ordena por autovalor (menores primeiro - estados ligados)
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        except Exception as ex:
            eigenvalues = np.array([])
            eigenvectors = np.array([])

        return eigenvalues, eigenvectors

    def detect_solitons(self,
                       eigenvalues: np.ndarray,
                       eigenvectors: np.ndarray,
                       V: np.ndarray) -> List[Soliton]:
        """
        Detecta solitons a partir dos autovalores discretos

        Autovalores negativos (estados ligados) correspondem a solitons.
        lambda_n < 0 -> Soliton com:
          - Amplitude proporcional a |lambda_n|
          - Velocidade proporcional a 4*|lambda_n|
        """
        solitons = []

        # Filtra autovalores negativos (estados ligados = solitons)
        bound_states = eigenvalues < -self.eigenvalue_threshold

        for i, (eigenval, is_bound) in enumerate(zip(eigenvalues, bound_states)):
            if is_bound and i < len(eigenvectors.T):
                eigenvec = eigenvectors[:, i]

                # Amplitude do soliton: A = 2|lambda|
                amplitude = 2 * np.abs(eigenval)

                # Velocidade: v = 4|lambda| (para KdV)
                velocity = 4 * np.abs(eigenval)

                # Posicao: centro da autofuncao
                position = np.sum(np.arange(len(eigenvec)) * eigenvec**2) / (np.sum(eigenvec**2) + 1e-10)
                position = position / len(eigenvec)  # Normaliza para [0, 1]

                # Largura: sigma do pico
                width = 1.0 / np.sqrt(np.abs(eigenval) + 0.1)

                # Tipo: baseado na forma da autofuncao e posicao no potencial
                # Se o soliton esta em regiao de potencial negativo (preco subindo) = compra
                # Se esta em regiao de potencial positivo (preco caindo) = venda
                pos_idx = int(position * len(V))
                pos_idx = min(pos_idx, len(V) - 1)

                if V[pos_idx] < 0:
                    soliton_type = SolitonType.BUYING_SOLITON
                else:
                    soliton_type = SolitonType.SELLING_SOLITON

                soliton = Soliton(
                    eigenvalue=eigenval,
                    amplitude=amplitude,
                    velocity=velocity,
                    position=position,
                    soliton_type=soliton_type,
                    width=width
                )

                solitons.append(soliton)

        # Ordena por amplitude (mais forte primeiro)
        solitons.sort(key=lambda s: s.amplitude, reverse=True)

        return solitons

    def compute_scattering_data(self, prices: np.ndarray) -> ScatteringData:
        """
        Computa dados completos do Inverse Scattering Transform
        """
        # Constroi potencial
        V = self.construct_schrodinger_potential(prices)

        # Resolve Schrodinger
        eigenvalues, eigenvectors = self.solve_schrodinger_eigenvalues(V)

        # Detecta solitons
        solitons = self.detect_solitons(eigenvalues, eigenvectors, V)

        # Coeficiente de reflexao (aproximacao simplificada)
        # Representa a parte de "radiacao" que nao e soliton
        reflection = np.abs(fft(V))**2
        reflection = reflection / (np.max(reflection) + 1e-10)

        return ScatteringData(
            eigenvalues=eigenvalues,
            eigenfunctions=eigenvectors,
            reflection_coefficient=reflection,
            n_solitons=len(solitons)
        )


# ==============================================================================
# NUMERO DE URSELL - REGIME CHECK
# ==============================================================================

class UrsellCalculator:
    """
    Calcula o Numero de Ursell para verificar regime KdV

    Ur = (Amplitude * Comprimento^2) / Profundidade^3

    - Se Ur ~= 1, estamos no regime KdV (Media Volatilidade)
    - Se Ur << 1 (Linear) ou Ur >> 1 (Turbulento), o indicador hiberna
    """

    def __init__(self,
                 ur_min: float = 0.5,
                 ur_max: float = 2.0):
        """
        Args:
            ur_min: Minimo para regime KdV
            ur_max: Maximo para regime KdV
        """
        self.ur_min = ur_min
        self.ur_max = ur_max

    def calculate(self,
                 prices: np.ndarray,
                 volumes: np.ndarray) -> UrsellNumber:
        """
        Calcula o Numero de Ursell
        """
        # Amplitude: amplitude das oscilacoes de preco
        amplitude = np.std(prices)

        # Comprimento de onda: periodo dominante via FFT
        prices_detrend = prices - np.mean(prices)
        fft_result = np.abs(fft(prices_detrend))
        freqs = fftfreq(len(prices))

        # Encontra frequencia dominante (ignora DC)
        fft_result[0] = 0
        dominant_idx = np.argmax(fft_result[:len(fft_result)//2])

        if freqs[dominant_idx] != 0:
            wavelength = 1.0 / np.abs(freqs[dominant_idx])
        else:
            wavelength = len(prices)

        # Profundidade: relacionada a liquidez media
        depth = np.mean(volumes) / (np.max(volumes) + 1e-10)
        depth = max(depth, 0.1)  # Evita divisao por zero

        # Normaliza para escala adequada
        amplitude_norm = amplitude / (np.mean(prices) + 1e-10)
        wavelength_norm = wavelength / len(prices)

        # Numero de Ursell
        ur = (amplitude_norm * wavelength_norm**2) / (depth**3 + 1e-10)

        # Escala para range util
        ur = ur * 1000  # Ajuste empirico

        # Verifica regime
        is_kdv = self.ur_min <= ur <= self.ur_max

        return UrsellNumber(
            value=ur,
            amplitude=amplitude,
            wavelength=wavelength,
            depth=depth,
            is_kdv_regime=is_kdv
        )


# ==============================================================================
# DETECTOR DE COLISOES
# ==============================================================================

class CollisionDetector:
    """
    Detecta colisoes iminentes entre solitons e barreiras de liquidez

    A solucao da KdV permite prever interacoes:
    - COMPRA: Soliton de Venda prestes a colidir com barreira de liquidez
      (topografia do fundo) e refletir. A matematica garante a conservacao
      de energia na reflexao elastica. Voce compra ANTES da reflexao visual
      no grafico.
    - VENDA: Soliton de Compra colide com uma densidade de liquidez (wall)
      e reverte.
    """

    def __init__(self,
                 collision_horizon: int = 10,  # Barras a frente
                 proximity_threshold: float = 0.1):
        """
        Args:
            collision_horizon: Horizonte de previsao em barras
            proximity_threshold: Distancia para considerar colisao iminente
        """
        self.collision_horizon = collision_horizon
        self.proximity_threshold = proximity_threshold

    def find_liquidity_barriers(self,
                                volumes: np.ndarray,
                                prices: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Encontra barreiras de liquidez (suportes e resistencias)

        Returns:
            Tupla (supports, resistances)
        """
        n = len(volumes)
        window = max(5, n // 20)

        # Suaviza volume
        volume_smooth = uniform_filter1d(volumes, size=window)

        # Encontra picos de volume (barreiras)
        supports = []
        resistances = []

        for i in range(window, n - window):
            # Pico local de volume
            if (volume_smooth[i] > volume_smooth[i-1] and
                volume_smooth[i] > volume_smooth[i+1] and
                volume_smooth[i] > np.mean(volume_smooth) * 1.2):

                price_at_peak = prices[i]
                current_price = prices[-1]

                if price_at_peak < current_price:
                    supports.append(price_at_peak)
                else:
                    resistances.append(price_at_peak)

        return supports, resistances

    def predict_collision(self,
                         soliton: Soliton,
                         supports: List[float],
                         resistances: List[float],
                         current_price: float,
                         price_range: float) -> Tuple[bool, float, str]:
        """
        Preve se um soliton vai colidir com uma barreira

        Returns:
            Tupla (collision_predicted, time_to_collision, barrier_type)
        """
        # Posicao do soliton em preco
        soliton_price = current_price + (soliton.position - 0.5) * price_range

        # Velocidade efetiva (direcao baseada no tipo)
        if soliton.soliton_type == SolitonType.SELLING_SOLITON:
            # Soliton de venda move para baixo
            velocity = -soliton.velocity
            barriers = supports
            barrier_type = "SUPPORT"
        else:
            # Soliton de compra move para cima
            velocity = soliton.velocity
            barriers = resistances
            barrier_type = "RESISTANCE"

        if not barriers:
            return False, np.inf, ""

        # Encontra barreira mais proxima na direcao do movimento
        min_time = np.inf
        collision = False

        for barrier in barriers:
            distance = barrier - soliton_price

            # Verifica se esta na direcao correta
            if velocity != 0 and np.sign(distance) == np.sign(velocity):
                time_to_barrier = abs(distance / (velocity * price_range / self.collision_horizon + 1e-10))

                if time_to_barrier < min_time and time_to_barrier < self.collision_horizon:
                    min_time = time_to_barrier
                    collision = True

        return collision, min_time, barrier_type


# ==============================================================================
# INDICADOR KdV-SH COMPLETO
# ==============================================================================

class KdVSolitonHunter:
    """
    Korteweg-de Vries Soliton Hunter (KdV-SH)

    Indicador completo que detecta solitons no mercado e preve reversoes
    baseadas na dinamica de fluidos.

    Logica de Trading (Sniper de Solitons):
    O indicador gera um sinal baseado na Colisao de Solitons.

    1. Regime Check: Verifica Numero de Ursell
    2. Deteccao: Encontra solitons via IST
    3. Previsao: Simula evolucao e detecta colisoes
    4. Sinal: Gera entrada baseada na reflexao prevista
    """

    def __init__(self,
                 # Parametros do regime
                 ur_min: float = 0.3,
                 ur_max: float = 3.0,

                 # Parametros do IST
                 n_points: int = 256,
                 eigenvalue_threshold: float = 0.001,

                 # Parametros de colisao
                 collision_horizon: int = 10,

                 # Parametros gerais
                 min_data_points: int = 100):
        """
        Inicializa o KdV Soliton Hunter
        """
        self.ur_min = ur_min
        self.ur_max = ur_max
        self.n_points = n_points
        self.eigenvalue_threshold = eigenvalue_threshold
        self.collision_horizon = collision_horizon
        self.min_data_points = min_data_points

        # Componentes
        self.hydro = HydrodynamicModel()
        self.kdv = KdVEquation(n_points=n_points)
        self.ist = InverseScatteringTransform(
            n_points=n_points,
            eigenvalue_threshold=eigenvalue_threshold
        )
        self.ursell_calc = UrsellCalculator(ur_min=ur_min, ur_max=ur_max)
        self.collision_detector = CollisionDetector(collision_horizon=collision_horizon)

        # Cache
        self.last_solitons: List[Soliton] = []
        self.last_scattering: Optional[ScatteringData] = None
        self.last_ursell: Optional[UrsellNumber] = None

    def analyze(self,
               prices: np.ndarray,
               volumes: np.ndarray = None) -> dict:
        """
        Analisa dados de mercado e retorna resultado completo

        Args:
            prices: Array de precos
            volumes: Array de volumes (opcional)

        Returns:
            Dict com analise completa
        """
        from datetime import datetime

        n = len(prices)

        # Gera volumes sinteticos se nao fornecidos
        if volumes is None:
            volumes = np.abs(np.diff(prices, prepend=prices[0])) * 50000 + \
                      np.random.rand(n) * 1000 + 500

        # Validacao
        if n < self.min_data_points:
            return {
                'signal': 0,
                'signal_name': 'INSUFFICIENT_DATA',
                'confidence': 0.0,
                'ursell_number': 0.0,
                'n_solitons': 0,
                'solitons': [],
                'dominant_soliton': None,
                'collision_predicted': False,
                'collision_time': np.inf,
                'collision_type': '',
                'predicted_reversal_price': prices[-1],
                'reasons': ['dados_insuficientes'],
                'current_price': prices[-1]
            }

        # PASSO 1: REGIME CHECK (Numero de Ursell)
        ursell = self.ursell_calc.calculate(prices, volumes)
        self.last_ursell = ursell

        if not ursell.is_kdv_regime:
            return {
                'signal': 0,
                'signal_name': 'HIBERNATE',
                'confidence': 0.0,
                'ursell_number': ursell.value,
                'n_solitons': 0,
                'solitons': [],
                'dominant_soliton': None,
                'collision_predicted': False,
                'collision_time': np.inf,
                'collision_type': '',
                'predicted_reversal_price': prices[-1],
                'reasons': [f'regime_{ursell.regime_description}', f'Ur={ursell.value:.3f}'],
                'current_price': prices[-1]
            }

        # PASSO 2: INVERSE SCATTERING TRANSFORM
        scattering_data = self.ist.compute_scattering_data(prices)
        self.last_scattering = scattering_data

        # Detecta solitons
        V = self.ist.construct_schrodinger_potential(prices)
        eigenvalues, eigenvectors = self.ist.solve_schrodinger_eigenvalues(V)
        solitons = self.ist.detect_solitons(eigenvalues, eigenvectors, V)
        self.last_solitons = solitons

        if len(solitons) == 0:
            return {
                'signal': 0,
                'signal_name': 'NEUTRAL',
                'confidence': 0.0,
                'ursell_number': ursell.value,
                'n_solitons': 0,
                'solitons': [],
                'dominant_soliton': None,
                'collision_predicted': False,
                'collision_time': np.inf,
                'collision_type': '',
                'predicted_reversal_price': prices[-1],
                'reasons': ['apenas_radiacao_dispersiva', 'sem_solitons'],
                'current_price': prices[-1]
            }

        # Soliton dominante
        dominant = solitons[0]

        # PASSO 3: DETECCAO DE COLISAO
        supports, resistances = self.collision_detector.find_liquidity_barriers(
            volumes, prices
        )

        # Preve colisao
        price_range = np.max(prices) - np.min(prices)
        collision, time_to_collision, barrier_type = self.collision_detector.predict_collision(
            dominant, supports, resistances, prices[-1], price_range
        )

        # PASSO 4: GERACAO DE SINAL
        signal = 0
        signal_name = "NEUTRAL"
        confidence = 0.0
        reasons = []
        predicted_price = prices[-1]

        if collision:
            # COMPRA: Soliton de venda vai colidir com suporte e refletir
            if (dominant.soliton_type == SolitonType.SELLING_SOLITON and
                barrier_type == "SUPPORT"):

                signal = 1
                signal_name = "LONG"
                confidence = min(1.0, dominant.amplitude * 0.5)
                predicted_price = min(supports) if supports else prices[-1]
                reasons.append("reflexao_soliton_venda")
                reasons.append(f"colisao_suporte_em_{time_to_collision:.1f}_barras")
                reasons.append(f"amplitude={dominant.amplitude:.3f}")

            # VENDA: Soliton de compra vai colidir com resistencia e refletir
            elif (dominant.soliton_type == SolitonType.BUYING_SOLITON and
                  barrier_type == "RESISTANCE"):

                signal = -1
                signal_name = "SHORT"
                confidence = min(1.0, dominant.amplitude * 0.5)
                predicted_price = max(resistances) if resistances else prices[-1]
                reasons.append("reflexao_soliton_compra")
                reasons.append(f"colisao_resistencia_em_{time_to_collision:.1f}_barras")
                reasons.append(f"amplitude={dominant.amplitude:.3f}")

            else:
                reasons.append("colisao_sem_setup_claro")
        else:
            reasons.append(f"soliton_{dominant.soliton_type.value}")
            reasons.append("sem_colisao_iminente")

        # Ajusta confianca pelo numero de Ursell
        ur_confidence = 1.0 - abs(ursell.value - 1.0) / max(ursell.value, 1.0)
        confidence *= ur_confidence
        confidence = np.clip(confidence, 0, 1)

        # Serializa solitons para output
        soliton_dicts = []
        for s in solitons[:5]:  # Limita a 5 solitons
            soliton_dicts.append({
                'eigenvalue': s.eigenvalue,
                'amplitude': s.amplitude,
                'velocity': s.velocity,
                'position': s.position,
                'type': s.soliton_type.value,
                'width': s.width,
                'energy': s.energy
            })

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'ursell_number': ursell.value,
            'ursell_regime': ursell.regime_description,
            'n_solitons': len(solitons),
            'solitons': soliton_dicts,
            'dominant_soliton': soliton_dicts[0] if soliton_dicts else None,
            'collision_predicted': collision,
            'collision_time': time_to_collision,
            'collision_type': barrier_type,
            'n_supports': len(supports),
            'n_resistances': len(resistances),
            'predicted_reversal_price': predicted_price,
            'reasons': reasons,
            'current_price': prices[-1]
        }

    def get_solitons(self) -> List[Soliton]:
        """Retorna os ultimos solitons detectados"""
        return self.last_solitons

    def get_scattering_data(self) -> Optional[ScatteringData]:
        """Retorna os ultimos dados de scattering"""
        return self.last_scattering

    def get_ursell_number(self) -> Optional[UrsellNumber]:
        """Retorna o ultimo numero de Ursell"""
        return self.last_ursell

    def simulate_evolution(self,
                          prices: np.ndarray,
                          n_steps: int = 100) -> KdVSolution:
        """
        Simula a evolucao temporal da equacao KdV

        Uma vez identificados os autovalores (os "DNAs" das ondas presentes),
        voce evolui a equacao KdV no tempo t -> t + delta. Como os solitons sao
        estaveis, a solucao matematica lhe dira EXATAMENTE onde a onda estara
        daqui a 5 ou 10 minutos com precisao assustadora, ignorando o ruido
        de superficie.
        """
        # Condicao inicial
        phi_0 = self.kdv.initial_condition_from_prices(prices)

        # Resolve KdV
        phi_final, phi_history = self.kdv.solve_pseudospectral(
            phi_0, dt=0.0001, n_steps=n_steps
        )

        return KdVSolution(
            phi=phi_0,
            x_grid=self.kdv.x,
            t_current=0.0,
            t_predicted=n_steps * 0.0001,
            phi_predicted=phi_final
        )

    def reset(self):
        """Reseta o estado do indicador"""
        self.last_solitons = []
        self.last_scattering = None
        self.last_ursell = None


# ==============================================================================
# EXEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("KORTEWEG-DE VRIES SOLITON HUNTER (KdV-SH)")
    print("Indicador baseado em Dinamica de Fluidos")
    print("=" * 70)

    np.random.seed(42)

    # Gera dados com soliton
    n_points = 200
    t = np.arange(n_points)

    # Tendencia base
    trend = 1.0850 + 0.00002 * t

    # Soliton analitico: phi(x,t) = A * sech^2(sqrt(A/12) * (x - ct))
    A = 0.005  # Amplitude
    c = 0.1    # Velocidade
    x0 = n_points / 2  # Posicao inicial

    soliton = A / np.cosh(np.sqrt(A / 12) * (t - x0 - c * t / 10))**2

    # Ruido baixo
    noise = np.random.randn(n_points) * 0.0002

    prices = trend + soliton + noise

    # Volume correlacionado
    base_volume = 1000
    volumes = base_volume + 500 * np.abs(np.gradient(prices)) * 10000
    volumes += np.random.randn(n_points) * 100
    volumes = np.maximum(volumes, 100)

    print(f"\nDados: {len(prices)} pontos")
    print(f"Preco: {prices[0]:.5f} -> {prices[-1]:.5f}")

    # Cria indicador
    indicator = KdVSolitonHunter(
        ur_min=0.1,
        ur_max=5.0,
        n_points=256,
        eigenvalue_threshold=0.001,
        collision_horizon=10,
        min_data_points=100
    )

    print("\nAnalisando dinamica de fluidos do mercado...")

    result = indicator.analyze(prices, volumes)

    print("\n" + "-" * 40)
    print("RESULTADO:")
    print(f"  Sinal: {result['signal_name']}")
    print(f"  Confianca: {result['confidence']:.0%}")
    print(f"  Numero de Ursell: {result['ursell_number']:.4f}")
    print(f"  Solitons detectados: {result['n_solitons']}")

    if result['dominant_soliton']:
        s = result['dominant_soliton']
        print(f"\n  Soliton Dominante:")
        print(f"    Tipo: {s['type']}")
        print(f"    Amplitude: {s['amplitude']:.4f}")
        print(f"    Velocidade: {s['velocity']:.4f}")

    print(f"\n  Colisao prevista: {result['collision_predicted']}")
    if result['collision_predicted']:
        print(f"    Tempo: {result['collision_time']:.1f} barras")
        print(f"    Tipo: {result['collision_type']}")

    print("\n" + "=" * 70)
    print("Teste concluido!")
    print("=" * 70)
