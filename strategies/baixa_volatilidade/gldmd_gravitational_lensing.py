"""
================================================================================
GRAVITATIONAL LENSING & DARK MATTER DETECTOR (GL-DMD)
Indicador de Forex baseado em Relatividade Geral e Distorcao de Cisalhamento
================================================================================

Este indicador resolve o Problema Inverso da Gravitacao. Ele monitora as
micro-trajetorias dos ticks (o "fundo de estrelas") e utiliza tecnicas de
Lente Gravitacional Fraca (Weak Lensing) para reconstruir o mapa de densidade
da "Materia Escura" (liquidez oculta) que esta distorcendo o mercado.

A Fisica: Relatividade Geral e Distorcao de Cisalhamento
Nao estamos analisando tendencia. Estamos analisando a curvatura causada por
massa estatica.

Por que usar Astrofisica?
1. Detecta o Invisivel: A maioria dos indicadores usa dados passados (Preco/Volume
   realizados). Lente Gravitacional detecta a ESTRUTURA que vai causar o movimento
   futuro. Ela ve a causa, nao o efeito.
2. Imune a Spoofing: HFTs podem colocar e tirar ordens falsas (spoofing) em
   milissegundos para enganar indicadores de book. Mas para gerar distorcao de
   cisalhamento estatistico (gamma), a ordem precisa interagir com o fluxo real por
   um tempo. Spoofing nao gera gravidade suficiente para curvar a estatistica.
3. Precisao de Sniper: O algoritmo Kaiser-Squires diz exatamente ONDE esta a massa
   (Preco) e QUAO densa ela e (Tamanho do Lote), sem nunca ter visto a ordem na tela.

Nota de Engenharia: Isso requer processamento de imagem em tensores 2D. Use
bibliotecas de astronomia como lenstronomy ou galpy adaptadas para dados 1D de
series temporais convertidos em espaco de fase 2D. A complexidade e O(N log N)
devido a FFT.

Autor: Gerado por Claude AI
Versao: 1.0.0
================================================================================
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import warnings
import logging

# Configuracao de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

class LensType(Enum):
    """Tipo de lente gravitacional detectada"""
    NO_LENS = "NO_LENS"                 # Sem massa significativa
    ATTRACTIVE = "ATTRACTIVE"            # Massa atratora (Buy Wall)
    REPULSIVE = "REPULSIVE"              # Massa repulsora (Sell Wall)
    EINSTEIN_RING = "EINSTEIN_RING"      # Anel de Einstein (alinhamento perfeito)


class MassType(Enum):
    """Tipo de massa detectada"""
    VISIBLE = "VISIBLE"                  # Ordens visiveis no book
    DARK = "DARK"                        # Liquidez oculta (Iceberg)
    MIXED = "MIXED"                      # Combinacao


@dataclass
class GravitationalPotential:
    """Potencial gravitacional Phi do mercado"""
    phi: np.ndarray              # Campo de potencial 2D
    phi_gradient: np.ndarray     # Gradiente do potencial
    mass_center: Tuple[float, float]  # Centro de massa
    total_mass: float            # Massa total detectada


@dataclass
class DeflectionField:
    """Campo de deflexao alpha"""
    alpha_x: np.ndarray          # Componente x da deflexao
    alpha_y: np.ndarray          # Componente y da deflexao
    alpha_magnitude: np.ndarray  # Magnitude |alpha|
    max_deflection: float        # Deflexao maxima


@dataclass
class ShearField:
    """Campo de cisalhamento gamma (Cosmic Shear)"""
    gamma1: np.ndarray           # gamma1 - componente real
    gamma2: np.ndarray           # gamma2 - componente imaginaria
    gamma_complex: np.ndarray    # gamma = gamma1 + i*gamma2
    gamma_magnitude: np.ndarray  # |gamma|
    shear_angle: np.ndarray      # Angulo do cisalhamento


@dataclass
class ConvergenceMap:
    """Mapa de convergencia kappa (densidade de massa)"""
    kappa: np.ndarray            # kappa - densidade superficial de massa
    kappa_smoothed: np.ndarray   # kappa suavizado
    peaks: List[Tuple[int, int, float]]  # Picos de massa (x, y, valor)
    dark_matter_fraction: float  # Fracao de materia escura


@dataclass
class DarkMatterDetection:
    """Deteccao de materia escura"""
    position: Tuple[float, float]    # Posicao (preco, tempo)
    mass: float                       # Massa estimada
    distance_from_price: float        # Distancia do preco atual
    is_attractive: bool               # True = Buy, False = Sell
    confidence: float


# ==============================================================================
# POTENCIAL GRAVITACIONAL
# ==============================================================================

class GravitationalPotentialCalculator:
    """
    1. O Potencial Gravitacional do Mercado (Phi)

    Assuma que o espaco de preco-tempo e plano (Random Walk Gaussiano) na
    ausencia de grandes players. Quando uma ordem massiva M e colocada em um
    preco P_dark, ela cria um potencial:

    Phi(x) = -G integral rho(x') / |x - x'| d2x'

    Onde rho(x') e a densidade de ordens (visiveis + ocultas).
    """

    def __init__(self,
                 G: float = 1.0,
                 softening: float = 0.1):
        self.G = G
        self.softening = softening

    def compute_potential(self,
                         density_field: np.ndarray) -> GravitationalPotential:
        """
        Calcula o potencial gravitacional a partir do campo de densidade
        Usa a equacao de Poisson: nabla^2 Phi = 4*pi*G*rho
        Resolvida via FFT
        """
        ny, nx = density_field.shape

        kx = fftfreq(nx) * 2 * np.pi
        ky = fftfreq(ny) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2

        K2[0, 0] = 1.0

        rho_k = fft2(density_field)

        phi_k = -4 * np.pi * self.G * rho_k / (K2 + self.softening**2)
        phi_k[0, 0] = 0

        phi = np.real(ifft2(phi_k))

        phi_gradient = np.gradient(phi)

        total_mass = np.sum(density_field)
        if total_mass > 0:
            y_coords, x_coords = np.mgrid[0:ny, 0:nx]
            cx = np.sum(x_coords * density_field) / total_mass
            cy = np.sum(y_coords * density_field) / total_mass
        else:
            cx, cy = nx / 2, ny / 2

        return GravitationalPotential(
            phi=phi,
            phi_gradient=phi_gradient,
            mass_center=(cx, cy),
            total_mass=total_mass
        )


# ==============================================================================
# ANGULO DE DEFLEXAO
# ==============================================================================

class DeflectionCalculator:
    """
    2. O Angulo de Deflexao (alpha)

    Um "raio de luz" (uma sequencia de micro-trades) que passa perto dessa
    massa sofre um desvio:

    alpha(theta) = nabla_theta Phi(theta)
    """

    def __init__(self):
        pass

    def compute_deflection(self,
                          phi: np.ndarray) -> DeflectionField:
        """Calcula o campo de deflexao alpha = nabla Phi"""
        grad_y, grad_x = np.gradient(phi)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return DeflectionField(
            alpha_x=grad_x,
            alpha_y=grad_y,
            alpha_magnitude=magnitude,
            max_deflection=np.max(magnitude)
        )


# ==============================================================================
# CISALHAMENTO COSMICO (COSMIC SHEAR)
# ==============================================================================

class CosmicShearCalculator:
    """
    3. Cisalhamento Cosmico (Cosmic Shear - gamma)

    Esta e a variavel chave. A gravidade nao apenas desloca a imagem, ela a
    distorce (estica). O Tensor de Cisalhamento Complexo descreve como uma nuvem
    circular de pontos de dados (trades aleatorios) se torna eliptica na presenca
    de massa.

    gamma = gamma1 + i*gamma2

    - Se detectarmos que a nuvem de ruido do mercado esta ficando eliptica e
      alinhada tangencialmente a um ponto de preco vazio, ACHAMOS A MATERIA ESCURA.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def compute_shear_from_jacobian(self,
                                   deflection: DeflectionField) -> ShearField:
        """
        Calcula cisalhamento a partir da matriz Jacobiana da deflexao

        A = | 1 - kappa - gamma1    -gamma2      |
            |   -gamma2      1 - kappa + gamma1  |
        """
        dalpha_x_dx = np.gradient(deflection.alpha_x, axis=1)
        dalpha_y_dy = np.gradient(deflection.alpha_y, axis=0)
        dalpha_x_dy = np.gradient(deflection.alpha_x, axis=0)

        gamma1 = 0.5 * (dalpha_x_dx - dalpha_y_dy)
        gamma2 = dalpha_x_dy

        gamma_complex = gamma1 + 1j * gamma2
        gamma_magnitude = np.abs(gamma_complex)

        shear_angle = 0.5 * np.arctan2(gamma2, gamma1)

        return ShearField(
            gamma1=gamma1,
            gamma2=gamma2,
            gamma_complex=gamma_complex,
            gamma_magnitude=gamma_magnitude,
            shear_angle=shear_angle
        )

    def measure_ellipticity(self,
                           points: np.ndarray,
                           window_center: Tuple[int, int]) -> Tuple[float, float]:
        """Mede a elipticidade local de uma nuvem de pontos"""
        if len(points) < 5:
            return 0.0, 0.0

        cx = np.mean(points[:, 0])
        cy = np.mean(points[:, 1])

        Qxx = np.mean((points[:, 0] - cx)**2)
        Qyy = np.mean((points[:, 1] - cy)**2)
        Qxy = np.mean((points[:, 0] - cx) * (points[:, 1] - cy))

        denom = Qxx + Qyy + 1e-10
        e1 = (Qxx - Qyy) / denom
        e2 = 2 * Qxy / denom

        return e1, e2


# ==============================================================================
# RECONSTRUCAO DE MASSA (KAISER-SQUIRES)
# ==============================================================================

class KaiserSquiresReconstructor:
    """
    O Algoritmo de Implementacao (The Kaiser-Squires Reconstruction)

    A densidade de massa kappa pode ser recuperada do campo de cisalhamento gamma
    no espaco de Fourier:

    kappa_tilde(k) = [(k_x^2 - k_y^2) + 2i*k_x*k_y] / (k_x^2 + k_y^2) * gamma_tilde(k)

    Ao fazer a Transformada Inversa de Fourier, voce obtem um Mapa de Calor 2D
    da Liquidez Oculta.
    """

    def __init__(self, smoothing_scale: float = 2.0):
        self.smoothing_scale = smoothing_scale

    def reconstruct_convergence(self,
                               shear: ShearField) -> ConvergenceMap:
        """
        Reconstroi o mapa de convergencia kappa a partir do cisalhamento gamma
        usando o algoritmo Kaiser-Squires
        """
        ny, nx = shear.gamma_complex.shape

        kx = fftfreq(nx) * 2 * np.pi
        ky = fftfreq(ny) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2

        K2[K2 == 0] = 1e-10

        D_real = (KX**2 - KY**2) / K2
        D_imag = 2 * KX * KY / K2
        D = D_real + 1j * D_imag

        gamma_k = fft2(shear.gamma_complex)

        kappa_k = np.conj(D) * gamma_k
        kappa_k[0, 0] = 0

        kappa = np.real(ifft2(kappa_k))

        kappa_smoothed = gaussian_filter(kappa, sigma=self.smoothing_scale)

        peaks = self._find_peaks(kappa_smoothed)

        dark_fraction = self._estimate_dark_fraction(kappa_smoothed)

        return ConvergenceMap(
            kappa=kappa,
            kappa_smoothed=kappa_smoothed,
            peaks=peaks,
            dark_matter_fraction=dark_fraction
        )

    def _find_peaks(self,
                   kappa: np.ndarray,
                   threshold_sigma: float = 2.0) -> List[Tuple[int, int, float]]:
        """Encontra picos no mapa de convergencia"""
        peaks = []

        mean_kappa = np.mean(kappa)
        std_kappa = np.std(kappa)
        threshold = mean_kappa + threshold_sigma * std_kappa

        ny, nx = kappa.shape

        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                val = kappa[y, x]
                if val > threshold:
                    neighbors = kappa[y-1:y+2, x-1:x+2]
                    if val == np.max(neighbors):
                        peaks.append((x, y, val))

        peaks.sort(key=lambda p: -p[2])

        return peaks[:10]

    def _estimate_dark_fraction(self, kappa: np.ndarray) -> float:
        """Estima a fracao de massa que e escura"""
        total_mass = np.sum(np.abs(kappa))

        if total_mass == 0:
            return 0.0

        median_kappa = np.median(np.abs(kappa))
        dark_mass = np.sum(np.abs(kappa[np.abs(kappa) > 2 * median_kappa]))

        return dark_mass / total_mass


# ==============================================================================
# DETECTOR DE MATERIA ESCURA
# ==============================================================================

class DarkMatterDetector:
    """
    A Logica de Trading (O Efeito Estilingue)

    O indicador plota no grafico manchas de calor onde NAO EXISTE ordem no book,
    mas existe gravidade.
    """

    def __init__(self,
                 detection_threshold: float = 2.0,
                 min_mass: float = 0.1):
        self.detection_threshold = detection_threshold
        self.min_mass = min_mass

    def detect_dark_matter(self,
                          convergence: ConvergenceMap,
                          visible_density: np.ndarray,
                          price_levels: np.ndarray) -> List[DarkMatterDetection]:
        """
        Detecta materia escura (liquidez oculta)

        Compara o mapa de convergencia (massa gravitacional) com a densidade
        visivel (ordens no book). A diferenca e materia escura.
        """
        detections = []

        kappa = convergence.kappa_smoothed
        ny, nx = kappa.shape

        if visible_density.shape != kappa.shape:
            visible_scaled = np.zeros_like(kappa)
        else:
            visible_scaled = visible_density / (np.max(visible_density) + 1e-10)
            visible_scaled *= np.max(np.abs(kappa))

        dark_matter = kappa - visible_scaled

        mean_dm = np.mean(dark_matter)
        std_dm = np.std(dark_matter)
        threshold_pos = mean_dm + self.detection_threshold * std_dm

        for peak in convergence.peaks:
            x, y, val = peak

            if val > threshold_pos:
                if dark_matter[y, x] > threshold_pos * 0.5:
                    if len(price_levels) > 0:
                        price_idx = int(x * len(price_levels) / nx)
                        price_idx = np.clip(price_idx, 0, len(price_levels) - 1)
                        price = price_levels[price_idx]
                    else:
                        price = x

                    detection = DarkMatterDetection(
                        position=(float(x), float(y)),
                        mass=float(val),
                        distance_from_price=float(x - nx / 2),
                        is_attractive=val > 0,
                        confidence=min(1.0, abs(val) / (threshold_pos + 1e-10))
                    )
                    detections.append(detection)

        return detections


# ==============================================================================
# DETECTOR DE ANEL DE EINSTEIN
# ==============================================================================

class EinsteinRingDetector:
    """
    3. SINAL (O Anel de Einstein)

    Quando a distorcao de cisalhamento atinge o formato de um anel perfeito ao
    redor de um ponto de preco, a "lente" esta perfeitamente alinhada.

    Acao: Coloque sua ordem limite exatamente no centro de massa da Materia
    Escura calculada. Voce estara front-running a baleia invisivel com precisao
    matematica.
    """

    def __init__(self,
                 ring_threshold: float = 0.8,
                 min_radius: int = 3,
                 max_radius: int = 20):
        self.ring_threshold = ring_threshold
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect_einstein_ring(self,
                            shear: ShearField,
                            convergence: ConvergenceMap) -> Tuple[bool, float, Tuple[int, int]]:
        """Detecta anel de Einstein"""
        if len(convergence.peaks) == 0:
            return False, 0.0, (0, 0)

        peak_x, peak_y, peak_val = convergence.peaks[0]

        ny, nx = shear.gamma_magnitude.shape

        best_ring = False
        best_radius = 0
        best_score = 0.0

        for radius in range(self.min_radius, min(self.max_radius, min(nx, ny) // 4)):
            tangential_score = self._compute_tangential_score(
                shear, peak_x, peak_y, radius
            )

            if tangential_score > best_score:
                best_score = tangential_score
                best_radius = radius

                if tangential_score > self.ring_threshold:
                    best_ring = True

        return best_ring, float(best_radius), (peak_x, peak_y)

    def _compute_tangential_score(self,
                                  shear: ShearField,
                                  cx: int, cy: int,
                                  radius: int) -> float:
        """Computa score de alinhamento tangencial do cisalhamento"""
        ny, nx = shear.shear_angle.shape

        n_samples = min(36, 2 * np.pi * radius)
        angles = np.linspace(0, 2 * np.pi, int(n_samples), endpoint=False)

        tangential_alignments = []

        for theta in angles:
            x = int(cx + radius * np.cos(theta))
            y = int(cy + radius * np.sin(theta))

            if 0 <= x < nx and 0 <= y < ny:
                expected_angle = theta + np.pi / 2
                measured_angle = shear.shear_angle[y, x]
                diff = 2 * (measured_angle - expected_angle)
                alignment = np.cos(diff)

                tangential_alignments.append(alignment)

        if len(tangential_alignments) == 0:
            return 0.0

        return np.mean(tangential_alignments)


# ==============================================================================
# CONVERSOR DE DADOS PARA ESPACO DE FASE
# ==============================================================================

class PhaseSpaceConverter:
    """
    Passo 1: Definicao do Campo de Fundo

    Colete os ultimos 1000 micro-ticks. Calcule a elipticidade intrinseca do
    ruido de mercado (baseline). Em um mercado sem ordens grandes, essa
    distribuicao deve ser isotropica (redonda).

    Converte series temporais 1D em espaco de fase 2D para analise de lensing.
    """

    def __init__(self,
                 grid_size: int = 64,
                 embedding_delay: int = 1):
        self.grid_size = grid_size
        self.embedding_delay = embedding_delay

    def create_phase_space(self,
                          prices: np.ndarray,
                          volumes: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria espaco de fase 2D a partir de serie temporal

        Eixo X: Preco (ou derivada do preco)
        Eixo Y: Tempo (ou segunda derivada)
        """
        n = len(prices)

        returns = np.diff(prices)
        acceleration = np.diff(returns)

        delay = self.embedding_delay
        m = len(acceleration) - delay

        if m < 10:
            density = np.zeros((self.grid_size, self.grid_size))
            return density, np.array([])

        x = acceleration[:-delay]
        y = acceleration[delay:]

        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10) * (self.grid_size - 1)
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-10) * (self.grid_size - 1)

        density = np.zeros((self.grid_size, self.grid_size))

        for i in range(len(x_norm)):
            xi = int(x_norm[i])
            yi = int(y_norm[i])
            xi = np.clip(xi, 0, self.grid_size - 1)
            yi = np.clip(yi, 0, self.grid_size - 1)

            if volumes is not None and i < len(volumes):
                density[yi, xi] += volumes[i]
            else:
                density[yi, xi] += 1

        price_levels = np.linspace(np.min(prices), np.max(prices), self.grid_size)

        return density, price_levels

    def measure_baseline_ellipticity(self,
                                     density: np.ndarray) -> Tuple[float, float]:
        """Mede elipticidade intrinseca do campo de fundo"""
        ny, nx = density.shape

        total = np.sum(density)
        if total == 0:
            return 0.0, 0.0

        y_coords, x_coords = np.mgrid[0:ny, 0:nx]
        cx = np.sum(x_coords * density) / total
        cy = np.sum(y_coords * density) / total

        Qxx = np.sum((x_coords - cx)**2 * density) / total
        Qyy = np.sum((y_coords - cy)**2 * density) / total
        Qxy = np.sum((x_coords - cx) * (y_coords - cy) * density) / total

        denom = Qxx + Qyy + 1e-10
        e1 = (Qxx - Qyy) / denom
        e2 = 2 * Qxy / denom

        return e1, e2


# ==============================================================================
# INDICADOR GL-DMD COMPLETO
# ==============================================================================

class GravitationalLensingDarkMatterDetector:
    """
    Gravitational Lensing & Dark Matter Detector (GL-DMD)

    Indicador completo que usa tecnicas de weak lensing para detectar liquidez
    oculta no mercado.

    A Logica de Trading (O Efeito Estilingue):

    1. Deteccao de Anomalia: Preco lateralizado + Alta convergencia kappa em regiao
       vazia do book = Materia Escura (Iceberg order)

    2. Evento de Horizonte: Preco "cai" em direcao a massa
       - Cenario A (Absorcao): Massa atrativa -> Rebote apos tocar
       - Cenario B (Repulsao): Massa repulsiva -> Curva antes de tocar

    3. Anel de Einstein: Cisalhamento forma anel perfeito -> Trade no centro de
       massa com precisao matematica
    """

    def __init__(self,
                 grid_size: int = 64,
                 G: float = 1.0,
                 smoothing_scale: float = 2.0,
                 detection_threshold: float = 2.0,
                 ring_threshold: float = 0.7,
                 min_data_points: int = 50):
        """Inicializa o GL-DMD"""
        self.grid_size = grid_size
        self.min_data_points = min_data_points

        self.phase_converter = PhaseSpaceConverter(grid_size=grid_size)
        self.potential_calculator = GravitationalPotentialCalculator(G=G)
        self.deflection_calculator = DeflectionCalculator()
        self.shear_calculator = CosmicShearCalculator()
        self.kaiser_squires = KaiserSquiresReconstructor(smoothing_scale=smoothing_scale)
        self.dark_detector = DarkMatterDetector(detection_threshold=detection_threshold)
        self.ring_detector = EinsteinRingDetector(ring_threshold=ring_threshold)

    def analyze(self,
               prices: np.ndarray,
               volumes: np.ndarray = None) -> dict:
        """Processa dados de mercado e retorna resultado da analise"""

        n = len(prices)

        if n < self.min_data_points:
            return self._create_empty_result("INSUFFICIENT_DATA")

        if volumes is None:
            volumes = np.abs(np.diff(prices)) * 10000 + 1000
            volumes = np.append(volumes, volumes[-1])

        # PASSO 1: CONVERSAO PARA ESPACO DE FASE
        density, price_levels = self.phase_converter.create_phase_space(prices, volumes)

        e1_baseline, e2_baseline = self.phase_converter.measure_baseline_ellipticity(density)
        baseline_ellipticity = np.sqrt(e1_baseline**2 + e2_baseline**2)

        # PASSO 2: POTENCIAL GRAVITACIONAL
        potential = self.potential_calculator.compute_potential(density)

        # PASSO 3: CAMPO DE DEFLEXAO
        deflection = self.deflection_calculator.compute_deflection(potential.phi)

        # PASSO 4: CISALHAMENTO COSMICO
        shear = self.shear_calculator.compute_shear_from_jacobian(deflection)

        mean_shear = np.mean(shear.gamma_magnitude)
        max_shear = np.max(shear.gamma_magnitude)

        mean_angle = np.mean(shear.shear_angle[shear.gamma_magnitude > mean_shear])

        # PASSO 5: INVERSAO KAISER-SQUIRES
        convergence = self.kaiser_squires.reconstruct_convergence(shear)

        max_kappa = np.max(convergence.kappa_smoothed)
        n_peaks = len(convergence.peaks)

        if convergence.peaks:
            peak_x, peak_y, peak_val = convergence.peaks[0]
            if len(price_levels) > 0:
                price_idx = int(peak_x * len(price_levels) / self.grid_size)
                price_idx = np.clip(price_idx, 0, len(price_levels) - 1)
                convergence_price = price_levels[price_idx]
            else:
                convergence_price = peak_x
        else:
            convergence_price = prices[-1]

        # PASSO 6: DETECCAO DE MATERIA ESCURA
        dark_detections = self.dark_detector.detect_dark_matter(
            convergence, density, price_levels
        )

        n_dark = len(dark_detections)

        dark_mass = 0.0
        dark_position = 0.0

        if dark_detections:
            largest = max(dark_detections, key=lambda d: d.mass)
            dark_mass = largest.mass
            dark_position = largest.position[0]

        # PASSO 7: DETECCAO DE ANEL DE EINSTEIN
        is_ring, einstein_radius, ring_center = self.ring_detector.detect_einstein_ring(
            shear, convergence
        )

        # PASSO 8: CLASSIFICACAO E GERACAO DE SINAL
        signal = 0
        signal_name = "NEUTRAL"
        confidence = 0.0
        reasons = []

        current_price = prices[-1]

        lens_type = LensType.NO_LENS
        mass_type = MassType.VISIBLE

        if max_kappa > 0.1:
            if is_ring:
                lens_type = LensType.EINSTEIN_RING
            elif convergence.peaks:
                peak_x, peak_y, peak_val = convergence.peaks[0]
                if peak_val > 0:
                    lens_type = LensType.ATTRACTIVE
                else:
                    lens_type = LensType.REPULSIVE

        if convergence.dark_matter_fraction > 0.5:
            mass_type = MassType.DARK
        elif convergence.dark_matter_fraction > 0.2:
            mass_type = MassType.MIXED

        mass_direction = "ABOVE" if convergence_price > current_price else "BELOW"

        # CONDICAO 1: ANEL DE EINSTEIN
        if lens_type == LensType.EINSTEIN_RING:
            if mass_direction == "BELOW":
                signal = 1
                signal_name = "LONG"
            else:
                signal = -1
                signal_name = "SHORT"

            confidence = 0.9
            reasons.append(f"ANEL DE EINSTEIN R={einstein_radius:.1f}")
            reasons.append(f"Massa {mass_direction}")

        # CONDICAO 2: LENTE ATRATIVA
        elif lens_type == LensType.ATTRACTIVE and mass_type in [MassType.DARK, MassType.MIXED]:
            if mass_direction == "BELOW":
                signal = 1
                signal_name = "LONG"
                confidence = min(0.8, max_kappa * 2 + convergence.dark_matter_fraction)
                reasons.append("LENTE ATRATIVA (Buy Wall)")
                reasons.append(f"DM={convergence.dark_matter_fraction:.0%}")
            else:
                signal = 1
                signal_name = "LONG"
                confidence = min(0.6, max_kappa * 2)
                reasons.append("LENTE ATRATIVA ACIMA")
                reasons.append(f"kappa={max_kappa:.3f}")

        # CONDICAO 3: LENTE REPULSIVA
        elif lens_type == LensType.REPULSIVE and mass_type in [MassType.DARK, MassType.MIXED]:
            if mass_direction == "ABOVE":
                signal = -1
                signal_name = "SHORT"
                confidence = min(0.8, abs(max_kappa) * 2 + convergence.dark_matter_fraction)
                reasons.append("LENTE REPULSIVA (Sell Wall)")
                reasons.append(f"DM={convergence.dark_matter_fraction:.0%}")
            else:
                signal = -1
                signal_name = "SHORT"
                confidence = min(0.6, abs(max_kappa) * 2)
                reasons.append("LENTE REPULSIVA ABAIXO")
                reasons.append(f"kappa={max_kappa:.3f}")

        # CONDICAO 4: Cisalhamento alto
        elif mean_shear > 0.05 and lens_type == LensType.NO_LENS:
            signal_name = "WAIT"
            confidence = 0.4
            reasons.append(f"Shear alto gamma={mean_shear:.3f}")
            reasons.append("Aguardando clareza")

        # CONDICAO 5: Sem lente
        else:
            reasons.append("Sem lente significativa")
            reasons.append(f"kappa={max_kappa:.3f}")

        confidence = np.clip(confidence, 0, 1)

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'lens_type': lens_type.value,
            'mass_type': mass_type.value,
            'total_mass': potential.total_mass,
            'mass_center_x': potential.mass_center[0],
            'mass_center_y': potential.mass_center[1],
            'mass_center_price': convergence_price,
            'mean_shear': mean_shear,
            'max_shear': max_shear,
            'shear_direction': float(np.degrees(mean_angle)),
            'max_convergence': max_kappa,
            'convergence_position': convergence_price,
            'n_peaks': n_peaks,
            'dark_matter_fraction': convergence.dark_matter_fraction,
            'dark_matter_mass': dark_mass,
            'dark_matter_position': dark_position,
            'n_dark_detections': n_dark,
            'einstein_radius': einstein_radius,
            'is_ring_detected': is_ring,
            'baseline_ellipticity': baseline_ellipticity,
            'max_deflection': deflection.max_deflection,
            'reasons': reasons
        }

    def _create_empty_result(self, signal_name: str) -> dict:
        """Cria resultado vazio"""
        return {
            'signal': 0,
            'signal_name': signal_name,
            'confidence': 0.0,
            'lens_type': LensType.NO_LENS.value,
            'mass_type': MassType.VISIBLE.value,
            'total_mass': 0.0,
            'mass_center_x': 0.0,
            'mass_center_y': 0.0,
            'mass_center_price': 0.0,
            'mean_shear': 0.0,
            'max_shear': 0.0,
            'shear_direction': 0.0,
            'max_convergence': 0.0,
            'convergence_position': 0.0,
            'n_peaks': 0,
            'dark_matter_fraction': 0.0,
            'dark_matter_mass': 0.0,
            'dark_matter_position': 0.0,
            'n_dark_detections': 0,
            'einstein_radius': 0.0,
            'is_ring_detected': False,
            'baseline_ellipticity': 0.0,
            'max_deflection': 0.0,
            'reasons': [signal_name]
        }

    def reset(self):
        """Reseta o indicador"""
        pass


# ==============================================================================
# DEMONSTRACAO
# ==============================================================================

def generate_iceberg_data(n_points: int = 200, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Gera dados com ordem iceberg oculta (materia escura)"""
    np.random.seed(seed)

    base = 1.0850

    returns = np.random.randn(n_points) * 0.0001

    iceberg_level = base - 0.0005

    prices = [base]
    for i in range(1, n_points):
        distance = prices[-1] - iceberg_level
        gravity = -0.00001 / (abs(distance) + 0.0001)

        new_price = prices[-1] + returns[i] + gravity
        prices.append(new_price)

    prices = np.array(prices)

    volumes = 1000 + np.random.randn(n_points) * 100

    return prices, volumes


def main():
    """Demonstracao do indicador GL-DMD"""
    print("=" * 70)
    print("GRAVITATIONAL LENSING & DARK MATTER DETECTOR (GL-DMD)")
    print("Indicador baseado em Relatividade Geral e Weak Lensing")
    print("=" * 70)
    print()

    indicator = GravitationalLensingDarkMatterDetector(
        grid_size=64,
        G=1.0,
        smoothing_scale=2.0,
        detection_threshold=2.0,
        ring_threshold=0.7,
        min_data_points=50
    )

    print("Indicador inicializado!")
    print(f"  - Grid: 64x64")
    print(f"  - G: 1.0")
    print(f"  - Smoothing: 2.0")
    print()

    prices, volumes = generate_iceberg_data(n_points=100)
    print(f"Dados gerados: {len(prices)} pontos")
    print()

    result = indicator.analyze(prices, volumes)

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Sinal: {result['signal_name']}")
    print(f"Confianca: {result['confidence']:.2%}")
    print(f"Lente: {result['lens_type']}")
    print(f"Massa: {result['mass_type']}")
    print(f"kappa max: {result['max_convergence']:.4f}")
    print(f"Shear medio: {result['mean_shear']:.4f}")
    print(f"DM fraction: {result['dark_matter_fraction']:.2%}")
    print(f"Anel: {result['is_ring_detected']}")
    print(f"Razoes: {result['reasons']}")

    return indicator, result


if __name__ == "__main__":
    indicator, result = main()
