"""
================================================================================
HILBERT-HUANG PHASE-LOCK OSCILLATOR (H2-PLO)
Indicador de Forex baseado em Decomposição Espectral Adaptativa
================================================================================

Conceito: Decomposição Espectral Adaptativa
Em vez de perguntar "o preço subiu nos últimos 14 períodos?", nós decompomos o
preço em suas frequências intrínsecas fundamentais e medimos a Sincronização de
Fase Instantânea.

Este indicador NÃO usa "períodos". Ele se adapta à frequência natural do mercado
naquele exato momento.

Arquitetura:
1. CEEMDAN - Decomposição em Modos Intrínsecos (IMFs)
2. Transformada de Hilbert - Fase e Frequência Instantâneas
3. Phase Locking Value (PLV) - Sincronização entre ciclos
4. Gatilho Sniper - Detecção de pontos de inflexão sem lag
================================================================================
"""

import numpy as np
from scipy.signal import hilbert
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


@dataclass
class IMFAnalysis:
    """Resultado da análise de um IMF"""
    imf_index: int
    imf_data: np.ndarray
    instantaneous_phase: np.ndarray
    instantaneous_frequency: np.ndarray
    amplitude_envelope: np.ndarray
    mean_frequency: float
    energy: float


@dataclass
class PhaseAnalysis:
    """Análise de fase entre IMFs"""
    phase_difference: np.ndarray
    plv: float  # Phase Locking Value
    coherence: float
    dominant_phase_relation: float


@dataclass
class H2PLOResult:
    """Resultado completo da análise H2-PLO"""
    signal: int                      # 1=LONG, -1=SHORT, 0=NEUTRAL
    signal_name: str
    confidence: float
    cycle_position: float            # Posição no ciclo (0 a 2π)
    frequency_acceleration: float    # dω/dt
    phase_fast: float
    phase_slow: float
    plv: float                       # Phase Locking Value
    amplitude: float
    n_imfs: int
    imf_energies: List[float]
    reasons: List[str]


# ==============================================================================
# IMPLEMENTAÇÃO DO CEEMDAN
# ==============================================================================

class CEEMDAN:
    """
    Complete Ensemble Empirical Mode Decomposition with Adaptive Noise

    Não usamos EMD simples devido ao problema de mistura de modos.
    O CEEMDAN injeta ruído branco gaussiano no preço, decompõe os modos,
    calcula a média e repete isso centenas de vezes para extrair as
    Funções de Modo Intrínseco (IMFs) puras.

    O preço X(t) é decomposto em n modos (IMFs) e um resíduo r(t):
    X(t) = Σ c_i(t) + r(t)

    Onde cada c_i(t) representa uma oscilação do mercado (da alta
    frequência/ruído até a baixa frequência/tendência macro), extraída
    sem funções de base pré-definidas (diferente de Wavelets ou Fourier).
    """

    def __init__(self,
                 trials: int = 100,
                 epsilon: float = 0.005,
                 max_imfs: int = 10,
                 max_siftings: int = 100,
                 sifting_threshold: float = 0.05):
        """
        Args:
            trials: Número de injeções de ruído (quanto maior, mais preciso)
            epsilon: Amplitude do ruído adicionado (fração do desvio padrão)
            max_imfs: Número máximo de IMFs a extrair
            max_siftings: Máximo de iterações de sifting por IMF
            sifting_threshold: Critério de parada para sifting
        """
        self.trials = trials
        self.epsilon = epsilon
        self.max_imfs = max_imfs
        self.max_siftings = max_siftings
        self.sifting_threshold = sifting_threshold

    def _find_extrema(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encontra índices de máximos e mínimos locais"""
        n = len(signal)

        # Derivada do sinal
        diff = np.diff(signal)

        # Encontra onde a derivada muda de sinal
        max_indices = []
        min_indices = []

        for i in range(1, len(diff)):
            if diff[i-1] > 0 and diff[i] <= 0:
                max_indices.append(i)
            elif diff[i-1] < 0 and diff[i] >= 0:
                min_indices.append(i)

        return np.array(max_indices), np.array(min_indices)

    def _interpolate_envelope(self,
                              indices: np.ndarray,
                              values: np.ndarray,
                              n: int) -> np.ndarray:
        """Interpola envelope usando spline cúbico"""
        if len(indices) < 4:
            if len(indices) < 2:
                return np.full(n, np.mean(values) if len(values) > 0 else 0)
            return np.interp(np.arange(n), indices, values)

        from scipy.interpolate import CubicSpline

        # Adiciona pontos de borda
        if indices[0] > 0:
            indices = np.concatenate([[0], indices])
            values = np.concatenate([[values[0]], values])
        if indices[-1] < n - 1:
            indices = np.concatenate([indices, [n-1]])
            values = np.concatenate([values, [values[-1]]])

        try:
            cs = CubicSpline(indices, values, bc_type='natural')
            return cs(np.arange(n))
        except:
            return np.interp(np.arange(n), indices, values)

    def _sift(self, signal: np.ndarray) -> np.ndarray:
        """
        Processo de sifting para extrair um IMF

        1. Encontra máximos e mínimos locais
        2. Interpola envelopes superior e inferior
        3. Calcula média dos envelopes
        4. Subtrai média do sinal
        5. Repete até convergência
        """
        h = signal.copy()
        n = len(h)

        for _ in range(self.max_siftings):
            max_idx, min_idx = self._find_extrema(h)

            if len(max_idx) < 3 or len(min_idx) < 3:
                break

            # Cria envelopes
            upper_env = self._interpolate_envelope(max_idx, h[max_idx], n)
            lower_env = self._interpolate_envelope(min_idx, h[min_idx], n)

            # Média dos envelopes
            mean_env = (upper_env + lower_env) / 2

            # Subtrai média
            h_new = h - mean_env

            # Critério de parada
            if np.std(h) > 0:
                sd = np.sum((h_new - h)**2) / np.sum(h**2 + 1e-10)
                if sd < self.sifting_threshold:
                    h = h_new
                    break

            h = h_new

        return h

    def _is_imf(self, signal: np.ndarray) -> bool:
        """Verifica se o sinal satisfaz as condições de IMF"""
        max_idx, min_idx = self._find_extrema(signal)
        n_extrema = len(max_idx) + len(min_idx)
        n_zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)

        return abs(n_extrema - n_zero_crossings) <= 1

    def _emd(self, signal: np.ndarray) -> List[np.ndarray]:
        """Empirical Mode Decomposition básico"""
        imfs = []
        residue = signal.copy()

        for _ in range(self.max_imfs):
            imf = self._sift(residue)

            if np.std(imf) < 1e-10:
                break

            imfs.append(imf)
            residue = residue - imf

            max_idx, min_idx = self._find_extrema(residue)
            if len(max_idx) < 2 or len(min_idx) < 2:
                break

        if np.std(residue) > 1e-10:
            imfs.append(residue)

        return imfs

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Executa decomposição CEEMDAN

        Args:
            signal: Série temporal a decompor

        Returns:
            Array 2D com IMFs (cada linha é um IMF)
        """
        n = len(signal)
        std_signal = np.std(signal)

        if std_signal == 0:
            return np.array([signal])

        all_imfs = []
        max_n_imfs = 0

        for trial in range(self.trials):
            noise = np.random.randn(n) * self.epsilon * std_signal
            noisy_signal = signal + noise

            imfs = self._emd(noisy_signal)
            all_imfs.append(imfs)
            max_n_imfs = max(max_n_imfs, len(imfs))

        # Calcula média dos IMFs
        final_imfs = []

        for i in range(max_n_imfs):
            imf_sum = np.zeros(n)
            count = 0

            for trial_imfs in all_imfs:
                if i < len(trial_imfs):
                    imf_sum += trial_imfs[i]
                    count += 1

            if count > 0:
                final_imfs.append(imf_sum / count)

        return np.array(final_imfs)

    def decompose(self, signal: np.ndarray) -> np.ndarray:
        """Alias para __call__"""
        return self.__call__(signal)


# ==============================================================================
# ANÁLISE DE HILBERT
# ==============================================================================

class HilbertAnalyzer:
    """
    Análise via Transformada de Hilbert

    Para cada IMF c_i(t) relevante, aplicamos a Transformada de Hilbert H[c_i(t)]
    para obter o sinal analítico Z(t):

    Z(t) = c_i(t) + j*H[c_i(t)] = a(t)*e^(jθ(t))

    Onde:
    - a(t): Amplitude Instantânea - A "energia" do movimento atual
    - θ(t): Fase Instantânea - A posição cíclica exata do mercado
    - ω(t): Frequência Instantânea - A derivada da fase (ω = dθ/dt)
    """

    def __init__(self, sampling_rate: float = 1.0):
        """
        Args:
            sampling_rate: Taxa de amostragem
        """
        self.sampling_rate = sampling_rate

    def analyze_imf(self, imf: np.ndarray, imf_index: int = 0) -> IMFAnalysis:
        """
        Analisa um IMF via Transformada de Hilbert

        Args:
            imf: Função de Modo Intrínseco
            imf_index: Índice do IMF

        Returns:
            IMFAnalysis com todos os parâmetros instantâneos
        """
        # Sinal analítico via Transformada de Hilbert
        analytic_signal = hilbert(imf)

        # Amplitude Instantânea (envelope)
        amplitude_envelope = np.abs(analytic_signal)

        # Fase Instantânea (unwrapped para continuidade)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

        # Frequência Instantânea
        instantaneous_freq = np.zeros_like(instantaneous_phase)
        instantaneous_freq[1:] = (np.diff(instantaneous_phase) / (2.0 * np.pi)) * self.sampling_rate
        instantaneous_freq[0] = instantaneous_freq[1]

        # Estatísticas
        mean_frequency = np.mean(np.abs(instantaneous_freq))
        energy = np.sum(imf ** 2)

        return IMFAnalysis(
            imf_index=imf_index,
            imf_data=imf,
            instantaneous_phase=instantaneous_phase,
            instantaneous_frequency=instantaneous_freq,
            amplitude_envelope=amplitude_envelope,
            mean_frequency=mean_frequency,
            energy=energy
        )

    def calculate_phase_locking_value(self,
                                       phase1: np.ndarray,
                                       phase2: np.ndarray,
                                       n: int = 1,
                                       m: int = 1) -> PhaseAnalysis:
        """
        Calcula o Phase Locking Value (PLV) entre dois IMFs

        Em média volatilidade, os ciclos de curto e médio prazo tendem a se
        alinhar nos pontos de virada. O PLV mede esta sincronização.

        Δθ_{n,m}(t) = |n * θ_2(t) - m * θ_3(t)|

        PLV = |<e^(j*Δθ)>| onde <> é a média temporal

        PLV ≈ 1: Ciclos fortemente sincronizados
        PLV ≈ 0: Ciclos não relacionados

        Args:
            phase1: Fase do primeiro IMF
            phase2: Fase do segundo IMF
            n, m: Razão de frequência para sincronização n:m

        Returns:
            PhaseAnalysis com PLV e métricas relacionadas
        """
        phase_diff = n * phase1 - m * phase2

        plv = np.abs(np.mean(np.exp(1j * phase_diff)))

        coherence = plv

        dominant_phase = np.mean(np.mod(phase_diff, 2 * np.pi))

        return PhaseAnalysis(
            phase_difference=phase_diff,
            plv=plv,
            coherence=coherence,
            dominant_phase_relation=dominant_phase
        )


# ==============================================================================
# INDICADOR H2-PLO COMPLETO
# ==============================================================================

class HilbertHuangPhaseLockOscillator:
    """
    Hilbert-Huang Phase-Lock Oscillator (H2-PLO)

    Indicador que identifica pontos de inflexão cíclica instantânea sem lag
    em regimes de média volatilidade.

    Funcionamento:
    1. Decompõe preço via CEEMDAN em IMFs
    2. Seleciona IMFs de ciclo rápido (1) e lento (2)
    3. Calcula fase e frequência instantâneas via Hilbert
    4. Detecta sincronização de fase (PLV)
    5. Gatilho quando:
       - Frequência está em ponto estacionário (dω/dt ≈ 0)
       - Fase em extremo cíclico (π/2 ou 3π/2)
    """

    def __init__(self,
                 ceemdan_trials: int = 50,
                 ceemdan_epsilon: float = 0.005,
                 fast_imf_index: int = 1,
                 slow_imf_index: int = 2,
                 phase_peak_range: Tuple[float, float] = (1.4, 1.7),
                 phase_trough_range: Tuple[float, float] = (4.5, 4.9),
                 freq_threshold: float = 0.001,
                 plv_threshold: float = 0.3,
                 min_data_points: int = 100):
        """
        Args:
            ceemdan_trials: Número de trials do CEEMDAN
            ceemdan_epsilon: Amplitude do ruído CEEMDAN
            fast_imf_index: Índice do IMF rápido (ciclo curto)
            slow_imf_index: Índice do IMF lento (ciclo médio)
            phase_peak_range: Range de fase para detectar topo (radianos)
            phase_trough_range: Range de fase para detectar fundo (radianos)
            freq_threshold: Threshold para frequência estacionária
            plv_threshold: Threshold mínimo de PLV para sinal válido
            min_data_points: Mínimo de dados necessários
        """
        self.ceemdan = CEEMDAN(trials=ceemdan_trials, epsilon=ceemdan_epsilon)
        self.hilbert = HilbertAnalyzer()

        self.fast_imf_index = fast_imf_index
        self.slow_imf_index = slow_imf_index
        self.phase_peak_range = phase_peak_range
        self.phase_trough_range = phase_trough_range
        self.freq_threshold = freq_threshold
        self.plv_threshold = plv_threshold
        self.min_data_points = min_data_points

        # Cache
        self.last_imfs: Optional[np.ndarray] = None
        self.last_imf_fast_analysis: Optional[IMFAnalysis] = None
        self.last_imf_slow_analysis: Optional[IMFAnalysis] = None
        self.last_phase_analysis: Optional[PhaseAnalysis] = None

    def decompose_signal(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompõe o preço em Modos Intrínsecos (IMFs)

        Em média volatilidade:
        - IMF 0 e 1: Ruído de microestrutura (spread/HFT)
        - IMF 2 e 3: Ciclo de trade (Swing/DayTrade)
        - IMF > 4: Tendência macro

        Returns:
            Tupla (imf_fast, imf_slow)
        """
        imfs = self.ceemdan(prices)
        self.last_imfs = imfs

        n_imfs = len(imfs)

        fast_idx = min(self.fast_imf_index, n_imfs - 1)
        slow_idx = min(self.slow_imf_index, n_imfs - 1)

        if fast_idx >= slow_idx:
            slow_idx = min(fast_idx + 1, n_imfs - 1)

        imf_fast = imfs[fast_idx]
        imf_slow = imfs[slow_idx]

        return imf_fast, imf_slow

    def calculate_instantaneous_parameters(self,
                                           imf: np.ndarray,
                                           imf_index: int = 0) -> IMFAnalysis:
        """
        Aplica Transformada de Hilbert para extrair parâmetros instantâneos
        """
        return self.hilbert.analyze_imf(imf, imf_index)

    def get_sniper_trigger(self,
                           phase: float,
                           freq_acceleration: float) -> Tuple[bool, bool, str]:
        """
        Gatilho de Entrada (Sniper)

        Quando a derivada da Frequência Instantânea (dω/dt) atinge zero
        (ponto estacionário) E a Fase Instantânea atinge π/2 ou -π/2
        (máxima excursão cíclica), temos uma reversão matemática.

        Args:
            phase: Fase atual normalizada (0 a 2π)
            freq_acceleration: Aceleração da frequência (dω/dt)

        Returns:
            Tupla (is_peak, is_trough, description)
        """
        is_peak = False
        is_trough = False
        description = ""

        cycle_position = np.mod(phase, 2 * np.pi)

        # TOPO CÍCLICO: fase próxima de π/2 (1.57 rad)
        if (self.phase_peak_range[0] < cycle_position < self.phase_peak_range[1]):
            if freq_acceleration <= 0:
                is_peak = True
                description = f"Cycle Peak: phase={cycle_position:.3f}, freq_accel={freq_acceleration:.6f}"

        # FUNDO CÍCLICO: fase próxima de 3π/2 (4.71 rad)
        if (self.phase_trough_range[0] < cycle_position < self.phase_trough_range[1]):
            if freq_acceleration >= 0:
                is_trough = True
                description = f"Cycle Trough: phase={cycle_position:.3f}, freq_accel={freq_acceleration:.6f}"

        return is_peak, is_trough, description

    def analyze(self, prices: np.ndarray) -> dict:
        """
        Analisa série de preços e retorna resultado completo

        Args:
            prices: Array numpy de preços

        Returns:
            Dict com resultado da análise
        """
        n = len(prices)

        if n < self.min_data_points:
            return {
                'signal': 0,
                'signal_name': 'INSUFFICIENT_DATA',
                'confidence': 0.0,
                'cycle_position': 0.0,
                'frequency_acceleration': 0.0,
                'phase_fast': 0.0,
                'phase_slow': 0.0,
                'plv': 0.0,
                'amplitude': 0.0,
                'n_imfs': 0,
                'imf_energies': [],
                'reasons': ['insufficient_data'],
                'current_price': prices[-1] if n > 0 else 0
            }

        # PASSO 1: Decomposição CEEMDAN
        imf_fast, imf_slow = self.decompose_signal(prices)

        # PASSO 2: Análise de Hilbert
        imf_fast_analysis = self.calculate_instantaneous_parameters(
            imf_fast, self.fast_imf_index
        )
        imf_slow_analysis = self.calculate_instantaneous_parameters(
            imf_slow, self.slow_imf_index
        )

        self.last_imf_fast_analysis = imf_fast_analysis
        self.last_imf_slow_analysis = imf_slow_analysis

        # PASSO 3: Phase Locking Value
        phase_analysis = self.hilbert.calculate_phase_locking_value(
            imf_fast_analysis.instantaneous_phase,
            imf_slow_analysis.instantaneous_phase
        )
        self.last_phase_analysis = phase_analysis

        # PASSO 4: Extração de parâmetros atuais
        curr_phase = imf_fast_analysis.instantaneous_phase[-1]

        freq = imf_fast_analysis.instantaneous_frequency
        if len(freq) >= 2:
            freq_acceleration = freq[-1] - freq[-2]
        else:
            freq_acceleration = 0.0

        cycle_position = np.mod(curr_phase, 2 * np.pi)
        curr_amplitude = imf_fast_analysis.amplitude_envelope[-1]

        # PASSO 5: Geração de Sinal
        is_peak, is_trough, trigger_desc = self.get_sniper_trigger(
            cycle_position, freq_acceleration
        )

        signal = 0
        signal_name = "NEUTRAL"
        confidence = 0.0
        reasons = []

        # SINAL DE VENDA (TOPO CÍCLICO)
        if is_peak:
            signal = -1
            signal_name = "SHORT"
            phase_conf = 1.0 - abs(cycle_position - np.pi/2) / (np.pi/2)
            plv_conf = phase_analysis.plv
            confidence = (phase_conf * 0.6 + plv_conf * 0.4)
            confidence = np.clip(confidence, 0, 1)
            reasons.append("cycle_peak")
            reasons.append(f"phase={cycle_position:.2f}")

        # SINAL DE COMPRA (FUNDO CÍCLICO)
        elif is_trough:
            signal = 1
            signal_name = "LONG"
            phase_conf = 1.0 - abs(cycle_position - 3*np.pi/2) / (np.pi/2)
            plv_conf = phase_analysis.plv
            confidence = (phase_conf * 0.6 + plv_conf * 0.4)
            confidence = np.clip(confidence, 0, 1)
            reasons.append("cycle_trough")
            reasons.append(f"phase={cycle_position:.2f}")

        # Ajusta confiança baseado no PLV
        if phase_analysis.plv < self.plv_threshold:
            confidence *= 0.5
            reasons.append("low_plv")

        # Calcula energias dos IMFs
        imf_energies = [np.sum(imf**2) for imf in self.last_imfs] if self.last_imfs is not None else []

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'cycle_position': cycle_position,
            'cycle_position_degrees': np.degrees(cycle_position),
            'frequency_acceleration': freq_acceleration,
            'phase_fast': curr_phase,
            'phase_slow': imf_slow_analysis.instantaneous_phase[-1],
            'plv': phase_analysis.plv,
            'amplitude': curr_amplitude,
            'n_imfs': len(self.last_imfs) if self.last_imfs is not None else 0,
            'imf_energies': imf_energies,
            'mean_frequency_fast': imf_fast_analysis.mean_frequency,
            'mean_frequency_slow': imf_slow_analysis.mean_frequency,
            'reasons': reasons,
            'current_price': prices[-1],
            'trigger_description': trigger_desc if is_peak or is_trough else ''
        }

    def get_imfs(self) -> Optional[np.ndarray]:
        """Retorna os últimos IMFs calculados"""
        return self.last_imfs

    def get_phase_analysis(self) -> Optional[PhaseAnalysis]:
        """Retorna a última análise de fase"""
        return self.last_phase_analysis

    def get_hilbert_spectrum(self) -> Optional[Dict]:
        """
        Retorna dados para plotar o espectro de Hilbert-Huang
        """
        if self.last_imfs is None:
            return None

        spectrum_data = []

        for i, imf in enumerate(self.last_imfs):
            analysis = self.calculate_instantaneous_parameters(imf, i)
            spectrum_data.append({
                'imf_index': i,
                'time': np.arange(len(imf)),
                'frequency': analysis.instantaneous_frequency,
                'amplitude': analysis.amplitude_envelope,
                'energy': analysis.energy
            })

        return spectrum_data


# ==============================================================================
# EXEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HILBERT-HUANG PHASE-LOCK OSCILLATOR (H2-PLO)")
    print("Demonstração do Indicador")
    print("=" * 70)

    np.random.seed(42)

    # Gera dados sintéticos com comportamento cíclico
    n_points = 300
    t = np.arange(n_points)

    # Tendência lenta
    trend = 1.0850 + 0.0001 * t + 0.001 * np.sin(2 * np.pi * t / 500)

    # Ciclo de trade (período ~50)
    trade_cycle = 0.002 * np.sin(2 * np.pi * t / 50 + np.random.rand() * 2 * np.pi)

    # Ciclo rápido (período ~10)
    fast_cycle = 0.0008 * np.sin(2 * np.pi * t / 10 + np.random.rand() * 2 * np.pi)

    # Ruído
    noise = np.random.randn(n_points) * 0.0003

    prices = trend + trade_cycle + fast_cycle + noise

    print(f"\nDados: {len(prices)} pontos")
    print(f"Preço: {prices[0]:.5f} -> {prices[-1]:.5f}")

    # Cria indicador
    indicator = HilbertHuangPhaseLockOscillator(
        ceemdan_trials=30,  # Reduzido para demo
        fast_imf_index=1,
        slow_imf_index=2,
        plv_threshold=0.2,
        min_data_points=100
    )

    print("\nExecutando análise H2-PLO...")

    # Analisa
    result = indicator.analyze(prices)

    print("\n" + "-" * 40)
    print("RESULTADO:")
    print(f"  Sinal: {result['signal_name']}")
    print(f"  Confiança: {result['confidence']:.0%}")

    print("\nPARÂMETROS INSTANTÂNEOS:")
    print(f"  Posição no Ciclo: {result['cycle_position']:.4f} rad ({result['cycle_position_degrees']:.1f}°)")
    print(f"  Aceleração Freq: {result['frequency_acceleration']:.6f}")
    print(f"  Fase Rápida: {result['phase_fast']:.4f} rad")
    print(f"  Fase Lenta: {result['phase_slow']:.4f} rad")

    print("\nSINCRONIZAÇÃO:")
    print(f"  PLV: {result['plv']:.4f}")
    print(f"  Amplitude: {result['amplitude']:.6f}")

    print("\nIMFs:")
    print(f"  Número de IMFs: {result['n_imfs']}")
    for i, e in enumerate(result['imf_energies'][:5]):
        print(f"    IMF {i}: energia = {e:.6f}")

    if result['trigger_description']:
        print(f"\nGatilho: {result['trigger_description']}")

    print("\n" + "=" * 70)
    print("Teste concluído!")
    print("=" * 70)
