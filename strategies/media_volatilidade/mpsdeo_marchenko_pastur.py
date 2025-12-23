"""
================================================================================
MARCHENKO-PASTUR SPECTRAL DE-NOISER & EIGEN-ENTROPIC OSCILLATOR (MP-SDEO)
Indicador de Forex baseado em Random Matrix Theory (RMT)
================================================================================

Este sistema utiliza a universalidade espectral de matrizes de Wishart para
CIRURGICAMENTE LOBOTOMIZAR o ruido do mercado. Ele nao suaviza o preco; ele
decompoe matematicamente a estrutura de correlacao interna da serie temporal
e REMOVE os componentes que obedecem a distribuicao de ruido aleatorio,
reconstruindo um "Preco Fantasma" purificado que contem apenas a informacao
estrutural deterministica.

A Base Matematica: RMT e a Lei de Marchenko-Pastur
Voce nao aplicara isso a uma carteira de ativos, mas sim a Auto-Correlacao
da Serie Temporal do EURUSD (Time-Lagged Correlation Matrix).

Por que RMT e superior?
1. Matematica "Hardcore": Nao ha parametros subjetivos como "periodo 14".
   Os limites lambda_min/max sao constantes universais derivadas da dimensionalidade
   da matriz. Ou e ruido, ou nao e.
2. Visao de Raio-X: Enquanto o RSI grita "sobrecomprado" porque o preco subiu,
   o RMT pode dizer "Ignore, esse subida e 99% composta por autovalores dentro
   da zona de Marchenko-Pastur, logo, e ruido sem sustentacao institucional".
3. Sinergia com Media Vol: Em media vol, o sinal (lambda_dominante) raramente e
   forte o suficiente para ser visto a olho nu, mas ele viola o limite de
   Marchenko-Pastur matematicamente. O indicador captura tendencias invisiveis.

Implementacao: Usa numpy.linalg.eigh para matrizes hermitianas/simetricas,
que e mais rapido e estavel que eig.

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

class RMTSignalType(Enum):
    """Tipos de sinais de trading"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    WAIT = "WAIT"  # Mercado delocalizado


class EigenRegime(Enum):
    """Regime baseado na localizacao de autovetor"""
    LOCALIZED = "LOCALIZED"          # Tendencia coerente (Alto IPR)
    DELOCALIZED = "DELOCALIZED"      # Mercado confuso (Baixo IPR)
    TRANSITIONAL = "TRANSITIONAL"    # Transicao entre regimes


@dataclass
class MarchenkoPasturBounds:
    """Limites teoricos de Marchenko-Pastur"""
    lambda_min: float       # lambda_min - limite inferior do ruido
    lambda_max: float       # lambda_max - limite superior do ruido
    Q: float               # Q = T/N - razao de aspecto
    sigma_sq: float        # sigma^2 - variancia estimada


@dataclass
class SpectralDecomposition:
    """Decomposicao espectral da matriz de correlacao"""
    eigenvalues: np.ndarray         # Todos os autovalores
    eigenvectors: np.ndarray        # Todos os autovetores
    signal_eigenvalues: np.ndarray  # lambda > lambda_max (SINAL)
    noise_eigenvalues: np.ndarray   # lambda in [lambda_min, lambda_max] (RUIDO)
    signal_indices: np.ndarray      # Indices dos sinais
    noise_indices: np.ndarray       # Indices do ruido
    n_signal: int                   # Numero de componentes de sinal
    n_noise: int                    # Numero de componentes de ruido


@dataclass
class CleanedSignal:
    """Sinal limpo apos filtragem espectral"""
    price_clean: np.ndarray         # Preco "Fantasma" purificado
    price_raw: np.ndarray           # Preco original
    correlation_clean: np.ndarray   # Matriz de correlacao limpa
    signal_strength: float          # Forca do sinal (soma lambda_signal)
    noise_level: float              # Nivel de ruido (soma lambda_noise)
    snr: float                      # Signal-to-Noise Ratio


@dataclass
class EigenEntropy:
    """Entropia espectral e metricas de localizacao"""
    entropy: float                  # S = -Sum rho(lambda) ln rho(lambda)
    ipr: float                      # Inverse Participation Ratio
    localization: float             # Grau de localizacao [0, 1]
    dominant_eigenvalue: float      # Maior autovalor
    dominant_eigenvector: np.ndarray  # Autovetor dominante


# ==============================================================================
# CONSTRUCAO DA MATRIZ DE TRAJETORIA (WISHART)
# ==============================================================================

class TrajectoryMatrixBuilder:
    """
    Construcao da Matriz de Trajetoria (Matriz de Wishart)

    Crie uma matriz M de dimensoes N x T (onde N e o numero de atrasos/lags
    e T e a janela de tempo). Calcule a Matriz de Correlacao Empirica C:

    C = (1/T) * M * M^T
    """

    def __init__(self, n_lags: int = 50, window: int = 200):
        """
        Args:
            n_lags: N - numero de lags (linhas da matriz M)
            window: T - janela de tempo (colunas da matriz M)
        """
        self.n_lags = n_lags
        self.window = window

    def build_trajectory_matrix(self, prices: np.ndarray) -> np.ndarray:
        """
        Constroi a matriz de trajetoria M (N x T)

        Cada linha e a serie de precos deslocada por um lag diferente
        """
        # Calcula retornos
        returns = np.diff(np.log(prices + 1e-10))

        # Normaliza
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)

        # Dimensoes efetivas
        T = min(self.window, len(returns) - self.n_lags)
        N = self.n_lags

        if T < N:
            T = len(returns) - N

        # Constroi matriz de trajetoria
        M = np.zeros((N, T))

        for i in range(N):
            M[i, :] = returns[i:i + T]

        return M

    def compute_correlation_matrix(self, M: np.ndarray) -> np.ndarray:
        """
        Calcula a Matriz de Correlacao Empirica C = (1/T) * M * M^T
        """
        N, T = M.shape
        C = (1.0 / T) * np.dot(M, M.T)

        return C

    def get_aspect_ratio(self, M: np.ndarray) -> float:
        """
        Calcula Q = T/N (razao de aspecto da matriz)
        """
        N, T = M.shape
        return T / N


# ==============================================================================
# LEI DE MARCHENKO-PASTUR
# ==============================================================================

class MarchenkoPasturDistribution:
    """
    O Limite Universal do Ruido (Marchenko-Pastur)

    Em sistemas complexos, autovalores provenientes de ruido aleatorio puro
    seguem estritamente a distribuicao de probabilidade de Marchenko-Pastur:

    P(lambda) = (Q / 2*pi*sigma^2) * sqrt((lambda_max - lambda)(lambda - lambda_min)) / lambda

    Os limites teoricos do ruido sao:
    lambda_min,max = sigma^2 * (1 +- sqrt(1/Q))^2

    Onde Q = T/N.

    A Logica de Engenharia:
    - Qualquer autovalor lambda que caia dentro do intervalo [lambda_min, lambda_max] e
      LIXO MATEMATICO (ruido).
    - Qualquer autovalor lambda > lambda_max e SINAL DE MERCADO genuino (informacao
      estrutural).
    """

    def __init__(self):
        pass

    def compute_bounds(self,
                      Q: float,
                      sigma_sq: float = 1.0) -> MarchenkoPasturBounds:
        """
        Calcula os limites de Marchenko-Pastur

        lambda_min,max = sigma^2 * (1 +- sqrt(1/Q))^2

        Args:
            Q: Razao de aspecto T/N
            sigma_sq: Variancia estimada (geralmente 1 para dados normalizados)
        """
        sqrt_inv_Q = np.sqrt(1.0 / Q)

        lambda_min = sigma_sq * (1 - sqrt_inv_Q)**2
        lambda_max = sigma_sq * (1 + sqrt_inv_Q)**2

        return MarchenkoPasturBounds(
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            Q=Q,
            sigma_sq=sigma_sq
        )

    def pdf(self,
           lambdas: np.ndarray,
           bounds: MarchenkoPasturBounds) -> np.ndarray:
        """
        Calcula a PDF de Marchenko-Pastur

        P(lambda) = (Q / 2*pi*sigma^2) * sqrt((lambda_max - lambda)(lambda - lambda_min)) / lambda
        """
        Q = bounds.Q
        sigma_sq = bounds.sigma_sq
        lambda_min = bounds.lambda_min
        lambda_max = bounds.lambda_max

        pdf = np.zeros_like(lambdas)

        # Apenas dentro do suporte [lambda_min, lambda_max]
        mask = (lambdas >= lambda_min) & (lambdas <= lambda_max)

        if np.any(mask):
            lam = lambdas[mask]
            numerator = np.sqrt((lambda_max - lam) * (lam - lambda_min))
            denominator = 2 * np.pi * sigma_sq * lam
            pdf[mask] = Q * numerator / (denominator + 1e-10)

        return pdf

    def classify_eigenvalues(self,
                            eigenvalues: np.ndarray,
                            bounds: MarchenkoPasturBounds) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Classifica autovalores em SINAL e RUIDO

        Returns:
            Tupla (signal_eigenvalues, noise_eigenvalues, signal_indices, noise_indices)
        """
        # Sinal: lambda > lambda_max
        signal_mask = eigenvalues > bounds.lambda_max

        # Ruido: lambda in [lambda_min, lambda_max]
        noise_mask = (eigenvalues >= bounds.lambda_min) & (eigenvalues <= bounds.lambda_max)

        signal_eigenvalues = eigenvalues[signal_mask]
        noise_eigenvalues = eigenvalues[noise_mask]
        signal_indices = np.where(signal_mask)[0]
        noise_indices = np.where(noise_mask)[0]

        return signal_eigenvalues, noise_eigenvalues, signal_indices, noise_indices


# ==============================================================================
# FILTRAGEM ESPECTRAL (SPECTRAL FILTERING)
# ==============================================================================

class SpectralFilter:
    """
    O Algoritmo de Implementacao (Spectral Filtering)

    Voce deve codificar um processo de Diagonalizacao de Matriz em tempo real.

    1. Decomposicao Espectral: Calcule todos os autovalores lambda_i e autovetores
       v_i da matriz C: Cv_i = lambda_i v_i

    2. Limpeza Espectral (Clipping): Identifique os autovalores que estao
       dentro do "Mar de Marchenko-Pastur" (ruido). Substitua todos esses
       autovalores pela media deles (para preservar o traco da matriz, mas
       matando a variancia do ruido). Mantenha intactos apenas os autovalores
       "gigantes" (lambda >> lambda_max).

    3. Reconstrucao do Sinal Limpo: Reverta a operacao. Reconstrua a matriz
       de correlacao limpa C_clean e, a partir dela, extraia a serie temporal
       filtrada do preco.

       C_clean = Sum lambda_i v_i v_i^T (signal) + Sum <lambda_noise> v_j v_j^T (noise)
    """

    def __init__(self):
        self.mp_distribution = MarchenkoPasturDistribution()

    def decompose(self, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decomposicao espectral: Cv_i = lambda_i v_i

        Usa numpy.linalg.eigh para matrizes simetricas (mais rapido e estavel)
        """
        # eigh retorna autovalores ordenados em ordem crescente
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        # Ordena em ordem decrescente (maior autovalor primeiro)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors

    def spectral_clipping(self,
                         eigenvalues: np.ndarray,
                         eigenvectors: np.ndarray,
                         bounds: MarchenkoPasturBounds) -> Tuple[np.ndarray, SpectralDecomposition]:
        """
        Limpeza Espectral (Clipping)

        Substitui autovalores de ruido pela media (preserva traco, mata variancia)
        """
        # Classifica autovalores
        signal_eig, noise_eig, signal_idx, noise_idx = self.mp_distribution.classify_eigenvalues(
            eigenvalues, bounds
        )

        # Copia autovalores
        eigenvalues_clean = eigenvalues.copy()

        # Substitui ruido pela media
        if len(noise_eig) > 0:
            noise_mean = np.mean(noise_eig)
            eigenvalues_clean[noise_idx] = noise_mean

        # Autovalores abaixo de lambda_min tambem sao ruido
        below_min = eigenvalues < bounds.lambda_min
        if np.any(below_min):
            if len(noise_eig) > 0:
                eigenvalues_clean[below_min] = noise_mean
            else:
                eigenvalues_clean[below_min] = bounds.lambda_min

        decomposition = SpectralDecomposition(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            signal_eigenvalues=signal_eig,
            noise_eigenvalues=noise_eig,
            signal_indices=signal_idx,
            noise_indices=noise_idx,
            n_signal=len(signal_eig),
            n_noise=len(noise_eig)
        )

        return eigenvalues_clean, decomposition

    def reconstruct_correlation_matrix(self,
                                       eigenvalues_clean: np.ndarray,
                                       eigenvectors: np.ndarray) -> np.ndarray:
        """
        Reconstroi matriz de correlacao limpa

        C_clean = Sum lambda_i v_i v_i^T
        """
        N = len(eigenvalues_clean)
        C_clean = np.zeros((N, N))

        for i in range(N):
            v = eigenvectors[:, i].reshape(-1, 1)
            C_clean += eigenvalues_clean[i] * np.dot(v, v.T)

        return C_clean

    def extract_clean_signal(self,
                            M: np.ndarray,
                            C_clean: np.ndarray) -> np.ndarray:
        """
        Extrai serie temporal filtrada a partir da matriz limpa

        Usa a primeira componente principal como o "Preco Fantasma"
        """
        # Decomposicao da matriz limpa
        eigenvalues, eigenvectors = np.linalg.eigh(C_clean)

        # Ordena (maior primeiro)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Projeta os dados originais na direcao principal
        v1 = eigenvectors[:, 0]

        # Sinal limpo = projecao de M no autovetor dominante
        clean_signal = np.dot(v1, M)

        return clean_signal

    def filter_prices(self,
                     prices: np.ndarray,
                     n_lags: int = 50,
                     window: int = 200) -> CleanedSignal:
        """
        Pipeline completo de filtragem espectral
        """
        # Constroi matriz de trajetoria
        builder = TrajectoryMatrixBuilder(n_lags=n_lags, window=window)
        M = builder.build_trajectory_matrix(prices)

        # Matriz de correlacao
        C = builder.compute_correlation_matrix(M)
        Q = builder.get_aspect_ratio(M)

        # Limites de Marchenko-Pastur
        bounds = self.mp_distribution.compute_bounds(Q)

        # Decomposicao espectral
        eigenvalues, eigenvectors = self.decompose(C)

        # Limpeza espectral
        eigenvalues_clean, decomposition = self.spectral_clipping(
            eigenvalues, eigenvectors, bounds
        )

        # Reconstroi matriz limpa
        C_clean = self.reconstruct_correlation_matrix(eigenvalues_clean, eigenvectors)

        # Extrai sinal limpo
        clean_signal = self.extract_clean_signal(M, C_clean)

        # Converte sinal limpo de volta para "precos"
        # Normaliza para escala do preco original
        price_mean = np.mean(prices[-len(clean_signal):])
        price_std = np.std(prices[-len(clean_signal):])

        clean_signal_norm = (clean_signal - np.mean(clean_signal)) / (np.std(clean_signal) + 1e-10)
        price_clean = price_mean + clean_signal_norm * price_std * 0.1  # Escala reduzida

        # Metricas
        signal_strength = np.sum(decomposition.signal_eigenvalues) if len(decomposition.signal_eigenvalues) > 0 else 0
        noise_level = np.sum(decomposition.noise_eigenvalues) if len(decomposition.noise_eigenvalues) > 0 else 1e-10
        snr = signal_strength / (noise_level + 1e-10)

        return CleanedSignal(
            price_clean=price_clean,
            price_raw=prices[-len(price_clean):],
            correlation_clean=C_clean,
            signal_strength=signal_strength,
            noise_level=noise_level,
            snr=snr
        )


# ==============================================================================
# ENTROPIA DE AUTOVETOR E IPR (LOCALIZACAO DE ANDERSON)
# ==============================================================================

class EigenEntropyCalculator:
    """
    O Gatilho Operacional: Entropia de Autovetor e IPR

    Apenas ter o preco limpo nao basta. Precisamos de um trigger de reversao
    para media volatilidade. Usaremos o conceito de Localizacao de Anderson
    (fisica de estado solido).

    Calcularemos a Razao de Participacao Inversa (Inverse Participation Ratio - IPR)
    para o autovetor dominante v_1 (que representa a tendencia principal):

    IPR(v_1) = Sum (v_{1,k})^4

    E a Eigen-Entropy (Entropia Espectral):

    S = -Sum rho(lambda) ln rho(lambda)

    A Logica de Trading:
    1. Regime de Delocalizacao (Alta Entropia / Baixo IPR): O autovetor esta
       "espalhado" por todos os componentes. O mercado esta confuso, sem
       direcao clara ou em movimento browniano. Acao: Esperar.

    2. Regime de Localizacao (Baixa Entropia / Alto IPR): O autovetor sofre
       um "colapso" e se localiza em poucos componentes temporais. Isso ocorre
       quando uma estrutura de tendencia coerente emerge do caos. E o
       equivalente matematico a um feixe de laser se formando.
    """

    def __init__(self,
                 ipr_threshold: float = 0.1,
                 entropy_threshold: float = 2.0):
        """
        Args:
            ipr_threshold: Threshold para considerar IPR alto (localizado)
            entropy_threshold: Threshold para considerar entropia baixa
        """
        self.ipr_threshold = ipr_threshold
        self.entropy_threshold = entropy_threshold

    def compute_ipr(self, eigenvector: np.ndarray) -> float:
        """
        Calcula IPR (Inverse Participation Ratio)

        IPR(v) = Sum v_k^4

        IPR alto = autovetor localizado (tendencia clara)
        IPR baixo = autovetor delocalizado (ruido)
        """
        # Normaliza autovetor
        v_norm = eigenvector / (np.linalg.norm(eigenvector) + 1e-10)

        # IPR = soma das quartas potencias
        ipr = np.sum(v_norm**4)

        return ipr

    def compute_eigen_entropy(self, eigenvalues: np.ndarray) -> float:
        """
        Calcula Entropia Espectral

        S = -Sum rho(lambda) ln rho(lambda)

        onde rho(lambda) = lambda / Sum_lambda (distribuicao normalizada de autovalores)
        """
        # Apenas autovalores positivos
        eigenvalues_pos = np.maximum(eigenvalues, 1e-10)

        # Normaliza para distribuicao de probabilidade
        rho = eigenvalues_pos / (np.sum(eigenvalues_pos) + 1e-10)

        # Entropia de Shannon
        entropy = -np.sum(rho * np.log(rho + 1e-10))

        return entropy

    def compute_localization(self, ipr: float, n_components: int) -> float:
        """
        Calcula grau de localizacao [0, 1]

        0 = completamente delocalizado (IPR = 1/N)
        1 = completamente localizado (IPR = 1)
        """
        # IPR minimo teorico (delocalizado uniforme)
        ipr_min = 1.0 / n_components

        # Normaliza
        localization = (ipr - ipr_min) / (1.0 - ipr_min + 1e-10)
        localization = np.clip(localization, 0, 1)

        return localization

    def analyze(self,
               eigenvalues: np.ndarray,
               eigenvectors: np.ndarray) -> EigenEntropy:
        """
        Analise completa de entropia e localizacao
        """
        # Autovetor dominante (maior autovalor)
        dominant_eigenvector = eigenvectors[:, 0]
        dominant_eigenvalue = eigenvalues[0]

        # IPR
        ipr = self.compute_ipr(dominant_eigenvector)

        # Entropia
        entropy = self.compute_eigen_entropy(eigenvalues)

        # Localizacao
        localization = self.compute_localization(ipr, len(eigenvalues))

        return EigenEntropy(
            entropy=entropy,
            ipr=ipr,
            localization=localization,
            dominant_eigenvalue=dominant_eigenvalue,
            dominant_eigenvector=dominant_eigenvector
        )

    def detect_ipr_spike(self,
                        ipr_history: List[float],
                        current_ipr: float,
                        spike_factor: float = 1.5) -> bool:
        """
        Detecta spike no IPR (subita localizacao)
        """
        if len(ipr_history) < 5:
            return False

        recent_mean = np.mean(ipr_history[-10:])
        recent_std = np.std(ipr_history[-10:]) + 1e-10

        # Spike = IPR significativamente acima da media recente
        z_score = (current_ipr - recent_mean) / recent_std

        return z_score > spike_factor or current_ipr > self.ipr_threshold

    def determine_regime(self, eigen_entropy: EigenEntropy) -> EigenRegime:
        """
        Determina o regime baseado em entropia e IPR
        """
        if eigen_entropy.ipr > self.ipr_threshold and eigen_entropy.entropy < self.entropy_threshold:
            return EigenRegime.LOCALIZED
        elif eigen_entropy.ipr < self.ipr_threshold * 0.5 and eigen_entropy.entropy > self.entropy_threshold * 1.5:
            return EigenRegime.DELOCALIZED
        else:
            return EigenRegime.TRANSITIONAL


# ==============================================================================
# INDICADOR MP-SDEO COMPLETO
# ==============================================================================

class MarchenkoPasturSpectralDeNoiser:
    """
    Marchenko-Pastur Spectral De-Noiser & Eigen-Entropic Oscillator (MP-SDEO)

    Indicador completo que usa Random Matrix Theory para cirurgicamente
    remover ruido e identificar tendencias estruturais genuinas.

    SINAL DE SNIPER:
    - Setup: Estamos em Media Volatilidade. O preco P_raw desvia do P_clean.
    - Disparo: O IPR dispara (spike), indicando uma localizacao subita de
      energia informacional (um consenso institucional oculto se formou).

    Se P_raw > P_clean e IPR Spikes -> SHORT
    (O ruido puxou o preco para cima, mas a estrutura espectral do mercado
    esta embaixo.)

    Se P_raw < P_clean e IPR Spikes -> LONG
    """

    def __init__(self,
                 # Parametros da matriz
                 n_lags: int = 50,
                 window: int = 200,

                 # Parametros de entropia/IPR
                 ipr_threshold: float = 0.08,
                 entropy_threshold: float = 3.0,
                 ipr_spike_factor: float = 1.5,

                 # Parametros de desvio
                 deviation_threshold: float = 0.0005,

                 # Geral
                 min_data_points: int = 250):
        """
        Inicializa o MP-SDEO
        """
        self.n_lags = n_lags
        self.window = window
        self.ipr_threshold = ipr_threshold
        self.entropy_threshold = entropy_threshold
        self.ipr_spike_factor = ipr_spike_factor
        self.deviation_threshold = deviation_threshold
        self.min_data_points = min_data_points

        # Componentes
        self.trajectory_builder = TrajectoryMatrixBuilder(n_lags=n_lags, window=window)
        self.mp_distribution = MarchenkoPasturDistribution()
        self.spectral_filter = SpectralFilter()
        self.entropy_calculator = EigenEntropyCalculator(
            ipr_threshold=ipr_threshold,
            entropy_threshold=entropy_threshold
        )

        # Historico
        self.ipr_history: List[float] = []
        self.entropy_history: List[float] = []
        self.price_clean_history: List[float] = []

    def analyze(self, prices: np.ndarray) -> dict:
        """
        Processa dados de preco e retorna analise completa

        Args:
            prices: Array de precos

        Returns:
            Dict com analise completa
        """
        n = len(prices)

        # Validacao
        if n < self.min_data_points:
            return {
                'signal': 0,
                'signal_name': 'INSUFFICIENT_DATA',
                'eigen_regime': 'DELOCALIZED',
                'confidence': 0.0,
                'price_raw': prices[-1] if n > 0 else 0.0,
                'price_clean': prices[-1] if n > 0 else 0.0,
                'price_deviation': 0.0,
                'lambda_max': 0.0,
                'lambda_min': 0.0,
                'n_signal_components': 0,
                'n_noise_components': 0,
                'signal_strength': 0.0,
                'noise_level': 0.0,
                'snr': 0.0,
                'eigen_entropy': 0.0,
                'ipr': 0.0,
                'ipr_spike': False,
                'localization': 0.0,
                'reasons': ['Dados insuficientes para analise RMT']
            }

        # ============================================================
        # PASSO 1: CONSTRUCAO DA MATRIZ DE TRAJETORIA
        # ============================================================
        M = self.trajectory_builder.build_trajectory_matrix(prices)
        N, T = M.shape
        Q = T / N

        # ============================================================
        # PASSO 2: MATRIZ DE CORRELACAO E LIMITES MP
        # ============================================================
        C = self.trajectory_builder.compute_correlation_matrix(M)
        bounds = self.mp_distribution.compute_bounds(Q)

        # ============================================================
        # PASSO 3: DECOMPOSICAO ESPECTRAL
        # ============================================================
        eigenvalues, eigenvectors = self.spectral_filter.decompose(C)

        # Classifica autovalores
        signal_eig, noise_eig, signal_idx, noise_idx = self.mp_distribution.classify_eigenvalues(
            eigenvalues, bounds
        )

        n_signal = len(signal_eig)
        n_noise = len(noise_eig)

        # ============================================================
        # PASSO 4: LIMPEZA ESPECTRAL E RECONSTRUCAO
        # ============================================================
        eigenvalues_clean, decomposition = self.spectral_filter.spectral_clipping(
            eigenvalues, eigenvectors, bounds
        )

        C_clean = self.spectral_filter.reconstruct_correlation_matrix(
            eigenvalues_clean, eigenvectors
        )

        clean_signal = self.spectral_filter.extract_clean_signal(M, C_clean)

        # Converte para escala de preco
        price_mean = np.mean(prices[-len(clean_signal):])
        price_std = np.std(prices[-len(clean_signal):])

        clean_norm = (clean_signal - np.mean(clean_signal)) / (np.std(clean_signal) + 1e-10)
        price_clean = price_mean + clean_norm * price_std * 0.1

        # Pega ultimo valor limpo
        price_clean_current = price_clean[-1] if len(price_clean) > 0 else prices[-1]

        self.price_clean_history.append(price_clean_current)
        if len(self.price_clean_history) > 100:
            self.price_clean_history.pop(0)

        # Metricas
        signal_strength = np.sum(signal_eig) if n_signal > 0 else 0
        noise_level = np.sum(noise_eig) if n_noise > 0 else 1e-10
        snr = signal_strength / (noise_level + 1e-10)

        # ============================================================
        # PASSO 5: ENTROPIA E IPR
        # ============================================================
        eigen_entropy = self.entropy_calculator.analyze(eigenvalues, eigenvectors)

        self.ipr_history.append(eigen_entropy.ipr)
        self.entropy_history.append(eigen_entropy.entropy)

        if len(self.ipr_history) > 100:
            self.ipr_history.pop(0)
            self.entropy_history.pop(0)

        # Detecta spike de IPR
        ipr_spike = self.entropy_calculator.detect_ipr_spike(
            self.ipr_history[:-1] if len(self.ipr_history) > 1 else [],
            eigen_entropy.ipr,
            self.ipr_spike_factor
        )

        # Regime
        eigen_regime = self.entropy_calculator.determine_regime(eigen_entropy)

        # ============================================================
        # PASSO 6: GERACAO DE SINAL
        # ============================================================
        # Desvio entre preco raw e clean
        price_raw = prices[-1]
        price_deviation = price_raw - price_clean_current
        deviation_normalized = price_deviation / (price_std + 1e-10)

        signal = 0
        signal_name = 'NEUTRAL'
        confidence = 0.0
        reasons = []

        # Regime delocalizado - esperar
        if eigen_regime == EigenRegime.DELOCALIZED:
            signal_name = 'WAIT'
            reasons.append(f'DELOCALIZADO: Alta entropia ({eigen_entropy.entropy:.2f}), baixo IPR ({eigen_entropy.ipr:.4f})')
            reasons.append('Mercado sem direcao. Esperar.')

        # Regime localizado ou transicional com spike
        elif eigen_regime in [EigenRegime.LOCALIZED, EigenRegime.TRANSITIONAL]:

            # SINAL DE SNIPER: Desvio + IPR Spike
            if ipr_spike and abs(deviation_normalized) > 0.5:

                if price_raw > price_clean_current:
                    # P_raw > P_clean + IPR Spike -> SHORT
                    signal = -1
                    signal_name = 'SHORT'
                    confidence = min(1.0, abs(deviation_normalized) * eigen_entropy.localization * snr)
                    reasons.append(f'Spectral Sniper: P_raw > P_clean por {deviation_normalized:.2f} sigma')
                    reasons.append(f'IPR spike detectado ({eigen_entropy.ipr:.4f})')
                    reasons.append('Ruido puxou preco para CIMA, estrutura espectral esta EMBAIXO')

                else:
                    # P_raw < P_clean + IPR Spike -> LONG
                    signal = 1
                    signal_name = 'LONG'
                    confidence = min(1.0, abs(deviation_normalized) * eigen_entropy.localization * snr)
                    reasons.append(f'Spectral Sniper: P_raw < P_clean por {abs(deviation_normalized):.2f} sigma')
                    reasons.append(f'IPR spike detectado ({eigen_entropy.ipr:.4f})')
                    reasons.append('Ruido puxou preco para BAIXO, estrutura espectral esta ACIMA')

            # Desvio significativo sem spike
            elif abs(deviation_normalized) > 1.0:
                if price_raw > price_clean_current:
                    signal = -1
                    signal_name = 'SHORT'
                    confidence = min(0.6, abs(deviation_normalized) * 0.3)
                    reasons.append(f'Desvio: P_raw > P_clean por {deviation_normalized:.2f} sigma')
                    reasons.append('Aguardando confirmacao de IPR spike')
                else:
                    signal = 1
                    signal_name = 'LONG'
                    confidence = min(0.6, abs(deviation_normalized) * 0.3)
                    reasons.append(f'Desvio: P_raw < P_clean por {abs(deviation_normalized):.2f} sigma')
                    reasons.append('Aguardando confirmacao de IPR spike')

            # Localizado mas sem desvio significativo
            else:
                reasons.append(f'LOCALIZADO mas desvio pequeno ({deviation_normalized:.2f} sigma)')
                reasons.append('Tendencia estrutural presente, aguardando entrada')

        # Ajusta confianca pelo numero de componentes de sinal
        if confidence > 0:
            if n_signal == 0:
                confidence *= 0.3  # Sem sinal genuino, reduz confianca
            else:
                confidence *= min(1.0, n_signal * 0.5)
            confidence = np.clip(confidence, 0, 1)

        return {
            'signal': signal,
            'signal_name': signal_name,
            'eigen_regime': eigen_regime.value,
            'confidence': confidence,
            'price_raw': price_raw,
            'price_clean': price_clean_current,
            'price_deviation': price_deviation,
            'deviation_normalized': deviation_normalized,
            'lambda_max': bounds.lambda_max,
            'lambda_min': bounds.lambda_min,
            'n_signal_components': n_signal,
            'n_noise_components': n_noise,
            'signal_strength': signal_strength,
            'noise_level': noise_level,
            'snr': snr,
            'eigen_entropy': eigen_entropy.entropy,
            'ipr': eigen_entropy.ipr,
            'ipr_spike': ipr_spike,
            'localization': eigen_entropy.localization,
            'dominant_eigenvalue': eigen_entropy.dominant_eigenvalue,
            'reasons': reasons
        }

    def get_ipr_history(self) -> np.ndarray:
        """Retorna historico de IPR"""
        return np.array(self.ipr_history)

    def get_entropy_history(self) -> np.ndarray:
        """Retorna historico de entropia"""
        return np.array(self.entropy_history)

    def get_clean_price_history(self) -> np.ndarray:
        """Retorna historico de preco limpo"""
        return np.array(self.price_clean_history)

    def reset(self):
        """Reseta o estado do indicador"""
        self.ipr_history.clear()
        self.entropy_history.clear()
        self.price_clean_history.clear()


# ==============================================================================
# DEMONSTRACAO
# ==============================================================================

def generate_noisy_trend_data(n_points: int = 500, seed: int = 42) -> np.ndarray:
    """Gera dados com tendencia + ruido (para testar RMT)"""
    np.random.seed(seed)

    t = np.arange(n_points)

    # Tendencia estrutural (o "sinal")
    trend = 1.0850 + 0.00005 * t + 0.001 * np.sin(2 * np.pi * t / 100)

    # Ruido (o que sera removido pelo RMT)
    noise = np.random.randn(n_points) * 0.0003

    prices = trend + noise

    return prices


def main():
    """Demonstracao do indicador MP-SDEO"""
    print("=" * 70)
    print("MARCHENKO-PASTUR SPECTRAL DE-NOISER & EIGEN-ENTROPIC OSCILLATOR")
    print("Indicador baseado em Random Matrix Theory (RMT)")
    print("=" * 70)
    print()

    # Inicializa indicador
    indicator = MarchenkoPasturSpectralDeNoiser(
        n_lags=30,
        window=100,
        ipr_threshold=0.08,
        entropy_threshold=3.0,
        deviation_threshold=0.0005,
        min_data_points=150
    )

    print("Indicador inicializado!")
    print(f"  - N (lags): 30")
    print(f"  - T (window): 100")
    print(f"  - IPR threshold: 0.08")
    print(f"  - Entropy threshold: 3.0")
    print()

    # Gera dados
    prices = generate_noisy_trend_data(n_points=300)
    print(f"Dados gerados: {len(prices)} pontos")
    print()

    # Processa
    result = indicator.analyze(prices)

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Sinal: {result['signal_name']}")
    print(f"Regime: {result['eigen_regime']}")
    print(f"Confianca: {result['confidence']:.2%}")
    print(f"\nPrecos:")
    print(f"  P_raw: {result['price_raw']:.5f}")
    print(f"  P_clean: {result['price_clean']:.5f}")
    print(f"  Desvio: {result['price_deviation']:.6f}")
    print(f"\nMarchenko-Pastur:")
    print(f"  lambda_min: {result['lambda_min']:.6f}")
    print(f"  lambda_max: {result['lambda_max']:.6f}")
    print(f"  Componentes SINAL: {result['n_signal_components']}")
    print(f"  Componentes RUIDO: {result['n_noise_components']}")
    print(f"  SNR: {result['snr']:.4f}")
    print(f"\nEntropia/IPR:")
    print(f"  Entropia: {result['eigen_entropy']:.4f}")
    print(f"  IPR: {result['ipr']:.6f}")
    print(f"  IPR Spike: {result['ipr_spike']}")
    print(f"\nRazoes: {result['reasons']}")

    return indicator, result


if __name__ == "__main__":
    indicator, result = main()
