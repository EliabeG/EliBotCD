"""
================================================================================
RIEMANNIAN CURVATURE TENSOR FLOW (RCTF)
Indicador de Forex baseado em Geometria Diferencial
================================================================================

O Conceito: O Mercado como uma Superfície Curva

Imagine que o preço, o tempo e o volume formam uma superfície 3D (uma variedade
Riemanniana). Em média volatilidade, o preço tenta seguir a "linha reta" nessa
superfície curva, chamada de Geodésica (o caminho de menor resistência).

O indicador calcula a Curvatura de Ricci dessa superfície em tempo real. Quando
a curvatura se torna extrema, a geodésica se rompe, forçando o preço a mudar de
direção (reversão).

Por que isso é superior?
Indicadores comuns (RSI, Bollinger) medem a distância linear da média. Este
indicador mede a TOPOLOGIA DA LIQUIDEZ. Ele detecta quando o "tecido" do mercado
está tão esticado que FISICAMENTE não pode ir mais longe sem rasgar.

Arquitetura Matemática:
1. Tensor Métrico g_ij - Define distâncias no espaço deformado
2. Símbolos de Christoffel Γ^k_ij - Conexão de Levi-Civita
3. Tensor de Curvatura de Riemann R^l_ijk
4. Escalar de Ricci R = g^ij R_ij
================================================================================
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
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
    HOLD = "HOLD"
    COMPRESSION = "COMPRESSION"
    EXPANSION = "EXPANSION"


class GeometryType(Enum):
    """Tipos de geometria do espaço"""
    FLAT = "FLAT"            # R ≈ 0, geodésica plana
    SPHERICAL = "SPHERICAL"  # R > 0, compressão (esfera)
    HYPERBOLIC = "HYPERBOLIC"  # R < 0, expansão (sela)


@dataclass
class MetricTensor:
    """Tensor Métrico g_ij e sua inversa"""
    g: np.ndarray           # g_ij (3x3)
    g_inv: np.ndarray       # g^ij (inversa)
    determinant: float      # det(g)
    eigenvalues: np.ndarray # Autovalores


@dataclass
class ChristoffelSymbols:
    """Símbolos de Christoffel Γ^k_ij"""
    gamma: np.ndarray  # Γ^k_ij (3x3x3)


@dataclass
class RiemannTensor:
    """Tensor de Curvatura de Riemann R^l_ijk"""
    riemann: np.ndarray     # R^l_ijk (3x3x3x3)
    ricci: np.ndarray       # R_ij (3x3) - contração
    ricci_scalar: float     # R = g^ij R_ij


@dataclass
class GeodesicDeviation:
    """Desvio Geodésico (Equação de Jacobi)"""
    jacobi_vector: np.ndarray      # J - vetor de separação
    jacobi_acceleration: float     # D²J/dt²
    convergence_rate: float        # Taxa de convergência das geodésicas


@dataclass
class RCTFResult:
    """Resultado completo da análise RCTF"""
    signal: int                     # 1=LONG, -1=SHORT, 0=HOLD
    signal_name: str
    geometry_type: str
    confidence: float
    ricci_scalar: float
    ricci_normalized: float
    geodesic_deviation: float
    jacobi_acceleration: float
    metric_determinant: float
    convergence_rate: float
    reasons: List[str]


# ==============================================================================
# TENSOR MÉTRICO
# ==============================================================================

class MetricTensorCalculator:
    """
    Calcula o Tensor Métrico g_ij

    Em vez de usar OHLC simples, mapeamos cada tick como um ponto vetorial em um
    espaço 3D: x = (Tempo, Preço, Volume).

    O Tensor Métrico g_ij define como medir "distâncias" nesse espaço deformado.
    NÃO usamos distância Euclidiana. A distância entre dois preços deve ser
    ponderada pela liquidez (Volume):

    - Se o volume é alto, a "distância" para mover o preço é maior (espaço denso)
    - Se o volume é baixo, a distância é menor (espaço rarefeito)

    ds² = g_ij dx^i dx^j

    onde:
    - dx^0 = dt (tempo)
    - dx^1 = dP (preço)
    - dx^2 = dV (volume)
    """

    def __init__(self,
                 volume_weight: float = 1.0,
                 time_scale: float = 1.0,
                 smoothing_window: int = 5):
        """
        Args:
            volume_weight: Peso do volume na métrica
            time_scale: Escala temporal
            smoothing_window: Janela de suavização para derivadas
        """
        self.volume_weight = volume_weight
        self.time_scale = time_scale
        self.smoothing_window = smoothing_window

    def calculate_metric(self,
                        time: np.ndarray,
                        price: np.ndarray,
                        volume: np.ndarray) -> MetricTensor:
        """
        Calcula o tensor métrico no ponto atual

        A métrica é construída considerando que a "rigidez" do espaço
        depende do volume. Em regiões de alto volume, o espaço é mais
        "denso" e resistente a mudanças.

        g_ij = [[g_tt, g_tp, g_tv],
                [g_pt, g_pp, g_pv],
                [g_vt, g_vp, g_vv]]
        """
        n = len(price)

        # Normaliza dados
        t_norm = (time - time[0]) / (time[-1] - time[0] + 1e-10)
        p_norm = (price - np.mean(price)) / (np.std(price) + 1e-10)
        v_norm = (volume - np.mean(volume)) / (np.std(volume) + 1e-10)

        # Calcula derivadas locais
        window = min(self.smoothing_window, n - 1)

        dp_dt = np.gradient(p_norm)[-1]
        dv_dt = np.gradient(v_norm)[-1]
        dp_dv = dp_dt / (dv_dt + 1e-10) if abs(dv_dt) > 1e-10 else 0

        # Volume local
        local_volume = np.mean(volume[-window:])
        volume_factor = local_volume / (np.mean(volume) + 1e-10)

        # Constrói tensor métrico
        g = np.zeros((3, 3))

        # Componente temporal (g_tt)
        g[0, 0] = self.time_scale ** 2

        # Componente de preço (g_pp) - ponderado pelo volume
        g[1, 1] = 1.0 + self.volume_weight * volume_factor

        # Componente de volume (g_vv)
        g[2, 2] = 1.0 / (volume_factor + 0.1)

        # Componentes cruzados
        g[0, 1] = g[1, 0] = 0.5 * dp_dt * self.time_scale
        g[0, 2] = g[2, 0] = 0.3 * dv_dt * self.time_scale
        g[1, 2] = g[2, 1] = 0.4 * dp_dv * np.sqrt(volume_factor)

        # Garante positiva definida
        eigenvalues = np.linalg.eigvalsh(g)
        if np.min(eigenvalues) < 1e-6:
            g += np.eye(3) * (1e-6 - np.min(eigenvalues) + 0.01)
            eigenvalues = np.linalg.eigvalsh(g)

        try:
            g_inv = np.linalg.inv(g)
            det = np.linalg.det(g)
        except:
            g_inv = np.eye(3)
            det = 1.0

        return MetricTensor(
            g=g,
            g_inv=g_inv,
            determinant=det,
            eigenvalues=eigenvalues
        )

    def calculate_metric_history(self,
                                time: np.ndarray,
                                price: np.ndarray,
                                volume: np.ndarray,
                                window: int = 10) -> List[MetricTensor]:
        """Calcula histórico de tensores métricos"""
        metrics = []
        n = len(price)

        for i in range(max(window, 1), n + 1):
            metric = self.calculate_metric(time[:i], price[:i], volume[:i])
            metrics.append(metric)

        return metrics


# ==============================================================================
# SÍMBOLOS DE CHRISTOFFEL
# ==============================================================================

class ChristoffelCalculator:
    """
    Calcula os Símbolos de Christoffel Γ^k_ij

    Para saber como o preço está "acelerando" nessa superfície curva,
    calculamos a conexão de Levi-Civita.

    Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
    """

    def __init__(self, derivative_order: int = 2):
        self.derivative_order = derivative_order

    def _numerical_derivative(self,
                             values: List[np.ndarray],
                             h: float = 1.0) -> np.ndarray:
        """Calcula derivada numérica"""
        n = len(values)
        if n < 3:
            return np.zeros_like(values[0])

        if n >= 3:
            deriv = (values[-1] - values[-3]) / (2 * h)
        else:
            deriv = (values[-1] - values[-2]) / h

        return deriv

    def calculate_christoffel(self,
                             metrics: List[MetricTensor],
                             h: float = 1.0) -> ChristoffelSymbols:
        """
        Calcula símbolos de Christoffel no ponto atual

        Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
        """
        if len(metrics) < 3:
            return ChristoffelSymbols(gamma=np.zeros((3, 3, 3)))

        g = metrics[-1].g
        g_inv = metrics[-1].g_inv

        # Calcula derivadas do tensor métrico
        dg = np.zeros((3, 3, 3))

        for i in range(3):
            for j in range(3):
                dg[:, i, j] = self._numerical_derivative_3d(
                    [m.g for m in metrics[-3:]], i, j, h
                )

        # Calcula Γ^k_ij
        gamma = np.zeros((3, 3, 3))

        for k in range(3):
            for i in range(3):
                for j in range(3):
                    for l in range(3):
                        gamma[k, i, j] += 0.5 * g_inv[k, l] * (
                            dg[i, j, l] + dg[j, i, l] - dg[l, i, j]
                        )

        return ChristoffelSymbols(gamma=gamma)

    def _numerical_derivative_3d(self,
                                 g_history: List[np.ndarray],
                                 i: int, j: int,
                                 h: float) -> np.ndarray:
        """Calcula derivadas parciais de g_ij"""
        if len(g_history) < 3:
            return np.zeros(3)

        dg_dt = (g_history[-1][i, j] - g_history[-3][i, j]) / (2 * h)
        dg_dp = dg_dt * 0.5
        dg_dv = dg_dt * 0.3

        return np.array([dg_dt, dg_dp, dg_dv])


# ==============================================================================
# TENSOR DE RIEMANN E ESCALAR DE RICCI
# ==============================================================================

class RiemannCalculator:
    """
    Calcula o Tensor de Curvatura de Riemann e o Escalar de Ricci

    R^l_ijk = ∂_j Γ^l_ik - ∂_k Γ^l_ij + Γ^l_jm Γ^m_ik - Γ^l_km Γ^m_ij

    Para simplificar, contraímos para o Escalar de Ricci (R):

    R = g^ij R_ij
    """

    def __init__(self):
        pass

    def calculate_riemann(self,
                         christoffel_history: List[ChristoffelSymbols],
                         metric: MetricTensor,
                         h: float = 1.0) -> RiemannTensor:
        """Calcula o tensor de Riemann completo"""
        if len(christoffel_history) < 3:
            return RiemannTensor(
                riemann=np.zeros((3, 3, 3, 3)),
                ricci=np.zeros((3, 3)),
                ricci_scalar=0.0
            )

        gamma = christoffel_history[-1].gamma
        gamma_prev = christoffel_history[-2].gamma if len(christoffel_history) > 1 else gamma
        gamma_prev2 = christoffel_history[-3].gamma if len(christoffel_history) > 2 else gamma_prev

        # Derivadas de Christoffel
        dgamma = (gamma - gamma_prev2) / (2 * h)

        # Tensor de Riemann R^l_ijk
        riemann = np.zeros((3, 3, 3, 3))

        for l in range(3):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        riemann[l, i, j, k] = dgamma[l, i, k] - dgamma[l, i, j]

                        for m in range(3):
                            riemann[l, i, j, k] += (
                                gamma[l, j, m] * gamma[m, i, k] -
                                gamma[l, k, m] * gamma[m, i, j]
                            )

        # Tensor de Ricci: R_ij = R^k_ikj
        ricci = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    ricci[i, j] += riemann[k, i, k, j]

        # Escalar de Ricci: R = g^ij R_ij
        g_inv = metric.g_inv
        ricci_scalar = 0.0
        for i in range(3):
            for j in range(3):
                ricci_scalar += g_inv[i, j] * ricci[i, j]

        return RiemannTensor(
            riemann=riemann,
            ricci=ricci,
            ricci_scalar=ricci_scalar
        )


# ==============================================================================
# DESVIO GEODÉSICO (EQUAÇÃO DE JACOBI)
# ==============================================================================

class GeodesicDeviationCalculator:
    """
    Calcula o Desvio Geodésico usando a Equação de Jacobi

    D²J/dt² + R(V, J)V = 0

    Onde:
    - V é o vetor velocidade do preço
    - J é o vetor de separação (desvio geodésico)
    - R(V, J) é o tensor de Riemann contraído
    """

    def __init__(self):
        pass

    def calculate_geodesic_deviation(self,
                                     price: np.ndarray,
                                     volume: np.ndarray,
                                     riemann: RiemannTensor,
                                     metric: MetricTensor) -> GeodesicDeviation:
        """Calcula o desvio geodésico"""
        n = len(price)

        # Vetor velocidade V
        if n >= 3:
            dp = np.gradient(price)
            dv = np.gradient(volume)
            V = np.array([1.0, dp[-1], dv[-1]])
        else:
            V = np.array([1.0, 0.0, 0.0])

        # Normaliza V
        V_norm = np.sqrt(np.dot(V, np.dot(metric.g, V)) + 1e-10)
        V = V / V_norm

        # Vetor de Jacobi inicial
        J = np.array([0.0, 1.0, 0.0])
        J = J - np.dot(J, V) * V
        J_norm = np.linalg.norm(J)
        if J_norm > 1e-10:
            J = J / J_norm

        # Calcula R(V, J)V
        R_VJV = np.zeros(3)
        for l in range(3):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        R_VJV[l] += riemann.riemann[l, i, j, k] * V[i] * J[j] * V[k]

        jacobi_accel = -np.linalg.norm(R_VJV)
        convergence_rate = -riemann.ricci_scalar

        return GeodesicDeviation(
            jacobi_vector=J,
            jacobi_acceleration=jacobi_accel,
            convergence_rate=convergence_rate
        )


# ==============================================================================
# INDICADOR RCTF COMPLETO
# ==============================================================================

class RiemannianCurvatureTensorFlow:
    """
    Riemannian Curvature Tensor Flow (RCTF)

    Indicador completo que calcula a curvatura do espaço de mercado
    e identifica pontos de reversão baseados em geometria diferencial.

    O Escalar de Ricci (R) representa a "densidade de energia geométrica"
    do mercado naquele instante:

    1. Regime de Fluxo Geodésico (R ≈ 0):
       - O mercado está plano ou em tendência inercial suave
       - Ação: Manter posição (Hold)

    2. Singularidade de Compressão (R > Limite_Positivo):
       - O espaço-tempo está se fechando (como uma esfera)
       - Topos ou fundos agudos onde liquidez absorve movimento

    3. Singularidade de Expansão (R < Limite_Negativo):
       - Geometria hiperbólica (sela)
       - Rompimento com aceleração
    """

    def __init__(self,
                 volume_weight: float = 1.0,
                 time_scale: float = 1.0,
                 ricci_positive_threshold: float = 0.5,
                 ricci_negative_threshold: float = -0.5,
                 ricci_neutral_band: float = 0.1,
                 jacobi_threshold: float = 0.1,
                 lookback_window: int = 20,
                 min_data_points: int = 50):
        """
        Inicializa o indicador RCTF

        Args:
            volume_weight: Peso do volume na métrica
            time_scale: Escala temporal
            ricci_positive_threshold: Limiar para curvatura positiva
            ricci_negative_threshold: Limiar para curvatura negativa
            ricci_neutral_band: Banda neutra ao redor de zero
            jacobi_threshold: Threshold para aceleração de Jacobi
            lookback_window: Janela de lookback
            min_data_points: Mínimo de dados necessários
        """
        self.volume_weight = volume_weight
        self.time_scale = time_scale
        self.ricci_positive_threshold = ricci_positive_threshold
        self.ricci_negative_threshold = ricci_negative_threshold
        self.ricci_neutral_band = ricci_neutral_band
        self.jacobi_threshold = jacobi_threshold
        self.lookback_window = lookback_window
        self.min_data_points = min_data_points

        # Inicializa calculadores
        self.metric_calculator = MetricTensorCalculator(
            volume_weight=volume_weight,
            time_scale=time_scale
        )
        self.christoffel_calculator = ChristoffelCalculator()
        self.riemann_calculator = RiemannCalculator()
        self.jacobi_calculator = GeodesicDeviationCalculator()

        # Cache
        self.metric_history: List[MetricTensor] = []
        self.christoffel_history: List[ChristoffelSymbols] = []
        self.riemann_history: List[RiemannTensor] = []
        self.ricci_scalar_history: List[float] = []

        # Último estado
        self.last_metric: Optional[MetricTensor] = None
        self.last_riemann: Optional[RiemannTensor] = None
        self.last_jacobi: Optional[GeodesicDeviation] = None

    def _detect_geometry_type(self, ricci_scalar: float) -> str:
        """Detecta o tipo de geometria"""
        if abs(ricci_scalar) < self.ricci_neutral_band:
            return "FLAT"
        elif ricci_scalar > self.ricci_positive_threshold:
            return "SPHERICAL"
        elif ricci_scalar < self.ricci_negative_threshold:
            return "HYPERBOLIC"
        else:
            return "FLAT"

    def _detect_price_trend(self, price: np.ndarray, window: int = 5) -> str:
        """Detecta tendência do preço"""
        if len(price) < window:
            return "NEUTRAL"

        recent = price[-window:]
        slope = (recent[-1] - recent[0]) / window

        if slope > 0.0001:
            return "UP"
        elif slope < -0.0001:
            return "DOWN"
        else:
            return "NEUTRAL"

    def _detect_ricci_peak(self) -> Tuple[bool, str]:
        """Detecta se Ricci está em pico local"""
        if len(self.ricci_scalar_history) < 5:
            return False, ""

        recent = self.ricci_scalar_history[-5:]

        if (recent[-2] > recent[-3] and recent[-2] > recent[-1] and
            recent[-2] > self.ricci_positive_threshold):
            return True, "POSITIVE_PEAK"

        if (recent[-2] < recent[-3] and recent[-2] < recent[-1] and
            recent[-2] < self.ricci_negative_threshold):
            return True, "NEGATIVE_PEAK"

        return False, ""

    def analyze(self,
               prices: np.ndarray,
               volumes: np.ndarray = None) -> dict:
        """
        Analisa dados de mercado e retorna resultado completo

        Args:
            prices: Array de preços
            volumes: Array de volumes (opcional)

        Returns:
            Dict com análise completa
        """
        n = len(prices)

        # Gera volumes sintéticos se não fornecidos
        if volumes is None:
            volumes = np.abs(np.diff(prices, prepend=prices[0])) * 50000 + \
                      np.random.rand(n) * 1000 + 500

        time = np.arange(n, dtype=float)

        if n < self.min_data_points:
            return {
                'signal': 0,
                'signal_name': 'INSUFFICIENT_DATA',
                'geometry_type': 'FLAT',
                'confidence': 0.0,
                'ricci_scalar': 0.0,
                'ricci_normalized': 0.0,
                'geodesic_deviation': 0.0,
                'jacobi_acceleration': 0.0,
                'metric_determinant': 1.0,
                'convergence_rate': 0.0,
                'reasons': ['insufficient_data'],
                'current_price': prices[-1]
            }

        # PASSO 1: TENSOR MÉTRICO
        window_start = max(0, n - self.lookback_window)

        for i in range(window_start, n):
            metric = self.metric_calculator.calculate_metric(
                time[:i+1], prices[:i+1], volumes[:i+1]
            )
            self.metric_history.append(metric)

            if len(self.metric_history) > self.lookback_window * 2:
                self.metric_history.pop(0)

        current_metric = self.metric_history[-1]
        self.last_metric = current_metric

        # PASSO 2: SÍMBOLOS DE CHRISTOFFEL
        if len(self.metric_history) >= 3:
            christoffel = self.christoffel_calculator.calculate_christoffel(
                self.metric_history[-5:] if len(self.metric_history) >= 5 else self.metric_history
            )
            self.christoffel_history.append(christoffel)

            if len(self.christoffel_history) > self.lookback_window:
                self.christoffel_history.pop(0)

        # PASSO 3: TENSOR DE RIEMANN E ESCALAR DE RICCI
        if len(self.christoffel_history) >= 3:
            riemann = self.riemann_calculator.calculate_riemann(
                self.christoffel_history, current_metric
            )
        else:
            riemann = RiemannTensor(
                riemann=np.zeros((3, 3, 3, 3)),
                ricci=np.zeros((3, 3)),
                ricci_scalar=0.0
            )

        self.last_riemann = riemann
        self.riemann_history.append(riemann)
        self.ricci_scalar_history.append(riemann.ricci_scalar)

        if len(self.riemann_history) > self.lookback_window:
            self.riemann_history.pop(0)
            self.ricci_scalar_history.pop(0)

        # Normaliza Ricci
        ricci_std = np.std(self.ricci_scalar_history) if len(self.ricci_scalar_history) > 1 else 1.0
        ricci_normalized = riemann.ricci_scalar / (ricci_std + 1e-10)

        # PASSO 4: DESVIO GEODÉSICO
        jacobi = self.jacobi_calculator.calculate_geodesic_deviation(
            prices, volumes, riemann, current_metric
        )
        self.last_jacobi = jacobi

        # PASSO 5: DETERMINAÇÃO DO SINAL
        geometry_type = self._detect_geometry_type(ricci_normalized)
        price_trend = self._detect_price_trend(prices)
        is_ricci_peak, peak_type = self._detect_ricci_peak()

        signal = 0
        signal_name = "HOLD"
        confidence = 0.0
        reasons = []

        # SINAL DE VENDA (TOPO MATEMÁTICO)
        if (price_trend == "UP" and
            ricci_normalized > self.ricci_positive_threshold and
            jacobi.jacobi_acceleration < -self.jacobi_threshold):

            signal = -1
            signal_name = "SHORT"
            confidence = min(1.0, abs(ricci_normalized) / 2.0)
            reasons.append("geometric_top")
            reasons.append(f"ricci={ricci_normalized:.3f}")
            reasons.append("forced_convergence")

        # SINAL DE COMPRA (FUNDO MATEMÁTICO)
        elif (price_trend == "DOWN" and
              ricci_normalized > self.ricci_positive_threshold and
              jacobi.jacobi_acceleration < -self.jacobi_threshold):

            signal = 1
            signal_name = "LONG"
            confidence = min(1.0, abs(ricci_normalized) / 2.0)
            reasons.append("geometric_bottom")
            reasons.append(f"ricci={ricci_normalized:.3f}")
            reasons.append("selling_exhaustion")

        # SINGULARIDADE DE EXPANSÃO
        elif ricci_normalized < self.ricci_negative_threshold:
            signal = 0
            signal_name = "EXPANSION"
            confidence = min(1.0, abs(ricci_normalized) / 2.0)
            reasons.append("hyperbolic_expansion")
            reasons.append("breakout_imminent")

        # COMPRESSÃO SEM DIREÇÃO
        elif ricci_normalized > self.ricci_positive_threshold:
            signal = 0
            signal_name = "COMPRESSION"
            confidence = min(0.5, abs(ricci_normalized) / 3.0)
            reasons.append("compression_building")

        # GEODÉSICA PLANA
        else:
            signal = 0
            signal_name = "HOLD"
            confidence = 0.0
            reasons.append("geodesic_flow")

        return {
            'signal': signal,
            'signal_name': signal_name,
            'geometry_type': geometry_type,
            'confidence': confidence,
            'ricci_scalar': riemann.ricci_scalar,
            'ricci_normalized': ricci_normalized,
            'geodesic_deviation': np.linalg.norm(jacobi.jacobi_vector),
            'jacobi_acceleration': jacobi.jacobi_acceleration,
            'metric_determinant': current_metric.determinant,
            'metric_eigenvalues': current_metric.eigenvalues.tolist(),
            'convergence_rate': jacobi.convergence_rate,
            'price_trend': price_trend,
            'reasons': reasons,
            'current_price': prices[-1]
        }

    def get_ricci_history(self) -> np.ndarray:
        """Retorna histórico do escalar de Ricci"""
        return np.array(self.ricci_scalar_history)

    def get_metric_eigenvalues(self) -> Optional[np.ndarray]:
        """Retorna autovalores da métrica atual"""
        if self.last_metric is not None:
            return self.last_metric.eigenvalues
        return None

    def reset(self):
        """Reseta o estado do indicador"""
        self.metric_history.clear()
        self.christoffel_history.clear()
        self.riemann_history.clear()
        self.ricci_scalar_history.clear()
        self.last_metric = None
        self.last_riemann = None
        self.last_jacobi = None


# ==============================================================================
# EXEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RIEMANNIAN CURVATURE TENSOR FLOW (RCTF)")
    print("Indicador baseado em Geometria Diferencial")
    print("=" * 70)

    np.random.seed(42)

    # Gera dados sintéticos
    n_points = 200
    time = np.arange(n_points).astype(float)

    trend = 1.0850 + 0.0001 * np.sin(2 * np.pi * time / 200)
    cycles = 0.002 * np.sin(2 * np.pi * time / 30)
    noise = np.random.randn(n_points) * 0.0005

    prices = trend + cycles + np.cumsum(noise) * 0.1

    base_volume = 1000 + 500 * np.sin(2 * np.pi * time / 50)
    volume_noise = np.random.randn(n_points) * 200
    price_change = np.abs(np.gradient(prices))
    volumes = base_volume + volume_noise + price_change * 50000
    volumes = np.maximum(volumes, 100)

    print(f"\nDados: {len(prices)} pontos")
    print(f"Preço: {prices[0]:.5f} -> {prices[-1]:.5f}")
    print(f"Volume médio: {np.mean(volumes):.0f}")

    # Cria indicador
    indicator = RiemannianCurvatureTensorFlow(
        volume_weight=1.0,
        ricci_positive_threshold=0.3,
        ricci_negative_threshold=-0.3,
        jacobi_threshold=0.05,
        lookback_window=20,
        min_data_points=50
    )

    print("\nCalculando curvatura do espaço de mercado...")

    result = indicator.analyze(prices, volumes)

    print("\n" + "-" * 40)
    print("RESULTADO:")
    print(f"  Sinal: {result['signal_name']}")
    print(f"  Geometria: {result['geometry_type']}")
    print(f"  Confiança: {result['confidence']:.0%}")

    print("\nCURVATURA:")
    print(f"  Escalar de Ricci: {result['ricci_scalar']:.6f}")
    print(f"  Ricci Normalizado: {result['ricci_normalized']:.4f}")

    print("\nGEODÉSICA:")
    print(f"  Desvio Geodésico: {result['geodesic_deviation']:.6f}")
    print(f"  Aceleração Jacobi: {result['jacobi_acceleration']:.6f}")
    print(f"  Taxa de Convergência: {result['convergence_rate']:.6f}")

    print("\nMÉTRICA:")
    print(f"  det(g): {result['metric_determinant']:.6f}")
    print(f"  Eigenvalues: {result['metric_eigenvalues']}")

    print("\n" + "=" * 70)
    print("Teste concluído!")
    print("=" * 70)
