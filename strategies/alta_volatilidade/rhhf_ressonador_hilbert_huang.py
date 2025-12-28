"""
Ressonador Hilbert-Huang Fractal (RHHF)
=======================================
Nível de Complexidade: Engenharia Aeroespacial / Processamento de Sinal Não-Linear Adaptativo.

Premissa Teórica: Não existem "tendências" e "ruídos" fixos. O preço é uma superposição de
Funções de Modo Intrínseco (IMFs). Para operar alta volatilidade com precisão cirúrgica,
precisamos decompor o preço nessas funções e aplicar a Transformada de Hilbert para
obter a frequência instantânea. O sinal de entrada não é o preço, mas a singularidade de
frequência (Chirp Signal) que precede um colapso ou explosão, similar ao sinal gravitacional
de dois buracos negros colidindo.

Dependências Críticas: EMD-signal (ou implementação própria de Ensemble EMD),
scipy.signal (Hilbert), numpy
"""

import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings('ignore')

# Tentar importar EMD-signal
try:
    from PyEMD import EEMD, EMD
    PYEMD_AVAILABLE = True
except ImportError:
    PYEMD_AVAILABLE = False


class RessonadorHilbertHuangFractal:
    """
    Implementação completa do Ressonador Hilbert-Huang Fractal (RHHF)

    Módulos:
    1. A Decomposição: Ensemble Empirical Mode Decomposition (EEMD)
    2. A Transformada de Hilbert e o Espectro de Energia
    3. O Detector de "Chirp" (A Física da Volatilidade)
    4. Análise Fractal da Frequência (Meta-Hurst)
    5. Output e Execução
    """

    def __init__(self,
                 n_ensembles: int = 100,
                 noise_amplitude: float = 0.2,
                 n_imfs: int = None,
                 mirror_extension: int = 50,
                 use_predictive_extension: bool = True,
                 ar_order: int = 20,
                 chirp_threshold: float = 0.0,
                 fractal_threshold: float = 1.35,
                 energy_percentile: float = 70,
                 smoothing_sigma: float = 3):
        """
        Inicialização do Ressonador Hilbert-Huang Fractal

        Parâmetros:
        -----------
        n_ensembles : int
            Número de repetições para EEMD (default: 100)

        noise_amplitude : float
            Amplitude do ruído branco adicionado (como fração do std)

        n_imfs : int ou None
            Número máximo de IMFs a extrair (None = automático)

        mirror_extension : int
            Número de pontos para extensão de espelho preditiva (default: 50)

        use_predictive_extension : bool
            Usar extensão preditiva para mitigar efeito de borda

        ar_order : int
            Ordem do modelo AR para extensão preditiva

        chirp_threshold : float
            Limiar para detecção de chirp (dω/dt > threshold)

        fractal_threshold : float
            Limiar de dimensão fractal para gatilho (< 1.2)

        energy_percentile : float
            Percentil de energia para detecção de concentração

        smoothing_sigma : float
            Sigma para suavização gaussiana
        """
        self.n_ensembles = n_ensembles
        self.noise_amplitude = noise_amplitude
        self.n_imfs = n_imfs
        self.mirror_extension = mirror_extension
        self.use_predictive_extension = use_predictive_extension
        self.ar_order = ar_order
        self.chirp_threshold = chirp_threshold
        self.fractal_threshold = fractal_threshold
        self.energy_percentile = energy_percentile
        self.smoothing_sigma = smoothing_sigma

        # Epsilon para estabilidade numérica
        self.eps = 1e-10

        # Cache
        self._cache = {}

    # =========================================================================
    # EXTENSÃO DE ESPELHO PREDITIVA (Predictive Mirror Extension)
    # =========================================================================

    def _fit_ar_model(self, signal: np.ndarray, order: int) -> np.ndarray:
        """
        Ajusta um modelo autoregressivo (AR) simples.

        x[t] = Σ φ[i] * x[t-i] + ε[t]

        Usa equações de Yule-Walker para estimar coeficientes.
        """
        n = len(signal)
        if n <= order:
            order = n - 1

        # Autocorrelação
        r = np.correlate(signal, signal, mode='full')
        r = r[n-1:]  # Apenas lag >= 0
        r = r[:order+1]

        # Matriz de Toeplitz para Yule-Walker
        R = np.zeros((order, order))
        for i in range(order):
            for j in range(order):
                R[i, j] = r[abs(i-j)]

        # Vetor r
        r_vec = r[1:order+1]

        # Resolver sistema linear: R * φ = r
        try:
            phi = np.linalg.solve(R, r_vec)
        except np.linalg.LinAlgError:
            phi = np.linalg.lstsq(R, r_vec, rcond=None)[0]

        return phi

    def _predict_ar(self, signal: np.ndarray, phi: np.ndarray, n_ahead: int) -> np.ndarray:
        """
        Prediz n_ahead pontos usando modelo AR.
        """
        order = len(phi)
        predictions = np.zeros(n_ahead)

        # Buffer com últimos 'order' pontos
        buffer = signal[-order:].copy()

        for i in range(n_ahead):
            # Predição: x[t] = Σ φ[i] * x[t-i]
            pred = np.sum(phi * buffer[::-1])
            predictions[i] = pred

            # Atualizar buffer
            buffer = np.roll(buffer, -1)
            buffer[-1] = pred

        return predictions

    def apply_predictive_mirror_extension(self, signal: np.ndarray) -> tuple:
        """
        O Desafio de Implementação para o Programador: O Efeito de Borda (End Effect).
        A Transformada de Hilbert e a EMD distorcem terrivelmente os dados no final da série
        (exatamente o candle atual, onde precisamos tomar a decisão).

        A Exigência: Você deve implementar um algoritmo de Extensão de Espelho Preditiva
        (Predictive Mirror Extension): Use um modelo autoregressivo simples (AR) para
        "alucinar" 50 candles no futuro, rodar a EEMD + Hilbert nesses dados estendidos,
        e depois cortar a ponta. Isso trará a distorção para o futuro "imaginário" e
        deixará o sinal do candle atual limpo (Spectral Leakage Mitigation).
        """
        n = len(signal)
        n_ext = self.mirror_extension

        # 1. Ajustar modelo AR
        phi = self._fit_ar_model(signal, self.ar_order)

        # 2. Predizer extensão futura
        future_ext = self._predict_ar(signal, phi, n_ext)

        # 3. Criar extensão espelhada no início (para simetria)
        # Espelhar os primeiros n_ext pontos
        past_ext = 2 * signal[0] - signal[1:n_ext+1][::-1]

        # 4. Concatenar: [passado_espelhado | sinal | futuro_predito]
        signal_extended = np.concatenate([past_ext, signal, future_ext])

        # Índices para recortar depois
        start_idx = n_ext
        end_idx = n_ext + n

        return signal_extended, start_idx, end_idx

    # =========================================================================
    # MÓDULO 1: Ensemble Empirical Mode Decomposition (EEMD)
    # =========================================================================

    def _sift(self, signal: np.ndarray, max_iterations: int = 1000,
              tol: float = 0.05) -> np.ndarray:
        """
        Processo de sifting para extrair uma IMF.
        """
        h = signal.copy()

        for _ in range(max_iterations):
            # Encontrar extremos locais
            maxima_idx = self._find_extrema(h, 'max')
            minima_idx = self._find_extrema(h, 'min')

            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break

            # Interpolar envelopes
            t = np.arange(len(h))

            try:
                # Envelope superior (máximos)
                cs_max = CubicSpline(maxima_idx, h[maxima_idx], bc_type='natural')
                upper = cs_max(t)

                # Envelope inferior (mínimos)
                cs_min = CubicSpline(minima_idx, h[minima_idx], bc_type='natural')
                lower = cs_min(t)
            except:
                break

            # Média dos envelopes
            mean_env = (upper + lower) / 2

            # Subtrair média
            h_new = h - mean_env

            # Verificar convergência
            sd = np.sum((h - h_new)**2) / (np.sum(h**2) + self.eps)
            h = h_new

            if sd < tol:
                break

        return h

    def _find_extrema(self, signal: np.ndarray, extrema_type: str) -> np.ndarray:
        """
        Encontra índices de extremos locais.
        """
        n = len(signal)
        extrema = []

        for i in range(1, n - 1):
            if extrema_type == 'max':
                if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                    extrema.append(i)
            else:  # min
                if signal[i] < signal[i-1] and signal[i] < signal[i+1]:
                    extrema.append(i)

        # Adicionar pontos de borda para interpolação
        if len(extrema) > 0:
            if extrema[0] != 0:
                extrema = [0] + extrema
            if extrema[-1] != n - 1:
                extrema = extrema + [n - 1]
        else:
            extrema = [0, n - 1]

        return np.array(extrema)

    def _emd_decompose(self, signal: np.ndarray, n_imfs: int = None) -> list:
        """
        Decomposição EMD básica.
        """
        if n_imfs is None:
            n_imfs = int(np.log2(len(signal))) + 1

        imfs = []
        residue = signal.copy()

        for _ in range(n_imfs):
            imf = self._sift(residue)

            # Verificar se é uma IMF válida
            if np.std(imf) < self.eps:
                break

            imfs.append(imf)
            residue = residue - imf

            # Verificar se residue é monotônico
            extrema_max = self._find_extrema(residue, 'max')
            extrema_min = self._find_extrema(residue, 'min')

            if len(extrema_max) <= 3 or len(extrema_min) <= 3:
                break

        # Adicionar resíduo
        imfs.append(residue)

        return imfs

    def eemd_decompose(self, signal: np.ndarray) -> dict:
        """
        O algoritmo EMD padrão sofre de "mistura de modos". Você deve implementar o EEMD
        (Ensemble EMD).

        Processo: Adicione ruído branco ao sinal de preço original, decomponha em IMFs,
        repita 100 vezes e tire a média.

        O Que Obter: Você terá um conjunto de IMFs (c1, c2, ..., cn) e um resíduo r.
        - c1: Ruído de alta frequência (HFT).
        - c2 - c4: Ciclos de mercado operáveis (Volatilidade Institucional).
        - cn: Tendência macroeconômica.

        Filtragem: Descarte c1 e o resíduo. Focaremos na soma reconstruída de c2 + c3.
        """
        n = len(signal)
        std_signal = np.std(signal)

        # Usar PyEMD se disponível
        if PYEMD_AVAILABLE:
            eemd = EEMD(trials=self.n_ensembles, noise_width=self.noise_amplitude)
            imfs = eemd.eemd(signal)
            imfs = [imfs[i] for i in range(imfs.shape[0])]
        else:
            # Implementação própria do EEMD
            all_imfs = []

            for i in range(self.n_ensembles):
                # Adicionar ruído branco
                noise = np.random.randn(n) * self.noise_amplitude * std_signal
                signal_noisy = signal + noise

                # Decompor
                imfs = self._emd_decompose(signal_noisy, self.n_imfs)
                all_imfs.append(imfs)

            # Calcular média das IMFs
            # Primeiro, encontrar número máximo de IMFs
            max_n_imfs = max(len(imfs) for imfs in all_imfs)

            # Média por IMF
            imfs = []
            for j in range(max_n_imfs):
                imf_ensemble = []
                for trial_imfs in all_imfs:
                    if j < len(trial_imfs):
                        imf_ensemble.append(trial_imfs[j])

                if imf_ensemble:
                    # Alinhar tamanhos (padding se necessário)
                    max_len = max(len(imf) for imf in imf_ensemble)
                    aligned = []
                    for imf in imf_ensemble:
                        if len(imf) < max_len:
                            imf = np.pad(imf, (0, max_len - len(imf)), mode='edge')
                        aligned.append(imf[:n])  # Garantir tamanho original

                    imfs.append(np.mean(aligned, axis=0))

        # Separar em categorias
        n_imfs = len(imfs)

        # c1: Ruído de alta frequência (primeira IMF)
        c1_hft = imfs[0] if n_imfs > 0 else np.zeros(n)

        # c2-c4: Ciclos operáveis (IMFs 2, 3, 4)
        c2 = imfs[1] if n_imfs > 1 else np.zeros(n)
        c3 = imfs[2] if n_imfs > 2 else np.zeros(n)
        c4 = imfs[3] if n_imfs > 3 else np.zeros(n)

        # cn: Tendência (última IMF ou resíduo)
        cn_trend = imfs[-1] if n_imfs > 0 else np.zeros(n)

        # Soma reconstruída de c2 + c3 (foco principal)
        reconstructed_operable = c2 + c3

        # Nuvem reconstruída (c2 + c3 + cn para contexto)
        cloud = c2 + c3 + cn_trend

        return {
            'imfs': imfs,
            'n_imfs': n_imfs,
            'c1_hft': c1_hft,
            'c2': c2,
            'c3': c3,
            'c4': c4,
            'cn_trend': cn_trend,
            'reconstructed_operable': reconstructed_operable,
            'cloud': cloud,
            'original': signal
        }

    # =========================================================================
    # MÓDULO 2: A Transformada de Hilbert e o Espectro de Energia
    # =========================================================================

    def apply_hilbert_transform(self, imf: np.ndarray) -> dict:
        """
        Para cada IMF selecionada, aplique a Transformada de Hilbert:

        y(t) = (1/π) P ∫ c(τ)/(t-τ) dτ

        (Onde P é o valor principal de Cauchy).

        - Calcule o sinal analítico z(t) = c(t) + iy(t).
        - Extraia a Amplitude Instantânea A(t) e a Frequência Instantânea ω(t) = dθ/dt.
        """
        # Transformada de Hilbert
        analytic_signal = hilbert(imf)

        # Amplitude Instantânea: A(t) = |z(t)|
        amplitude = np.abs(analytic_signal)

        # Fase Instantânea: θ(t) = arg(z(t))
        phase = np.unwrap(np.angle(analytic_signal))

        # Frequência Instantânea: ω(t) = dθ/dt
        # Usar diferença central para melhor precisão
        frequency = np.gradient(phase)

        # Garantir frequência positiva (física)
        frequency = np.abs(frequency)

        # Suavizar para reduzir ruído
        frequency_smooth = gaussian_filter1d(frequency, sigma=self.smoothing_sigma)
        amplitude_smooth = gaussian_filter1d(amplitude, sigma=self.smoothing_sigma)

        return {
            'analytic_signal': analytic_signal,
            'amplitude': amplitude,
            'amplitude_smooth': amplitude_smooth,
            'phase': phase,
            'frequency': frequency,
            'frequency_smooth': frequency_smooth
        }

    def calculate_hilbert_energy(self, amplitude: np.ndarray) -> np.ndarray:
        """
        Cálculo da Energia de Hilbert (H_E): H_E(t) = A(t)²
        """
        return amplitude ** 2

    def calculate_hilbert_spectrum(self, imfs: list) -> dict:
        """
        Calcula o espectro de Hilbert completo para todas as IMFs.

        Retorna matriz Tempo x Frequência x Energia para visualização 3D.
        """
        n_imfs = len(imfs)
        n_time = len(imfs[0]) if n_imfs > 0 else 0

        # Arrays para espectro
        frequencies = []
        amplitudes = []
        energies = []

        for imf in imfs:
            ht = self.apply_hilbert_transform(imf)
            frequencies.append(ht['frequency_smooth'])
            amplitudes.append(ht['amplitude_smooth'])
            energies.append(self.calculate_hilbert_energy(ht['amplitude_smooth']))

        return {
            'frequencies': np.array(frequencies),
            'amplitudes': np.array(amplitudes),
            'energies': np.array(energies),
            'time': np.arange(n_time),
            'n_imfs': n_imfs
        }

    # =========================================================================
    # MÓDULO 3: O Detector de "Chirp" (A Física da Volatilidade)
    # =========================================================================

    def detect_chirp(self, frequency: np.ndarray, amplitude: np.ndarray) -> dict:
        """
        Em sistemas físicos prestes a romper (como uma represa rachando ou um mercado
        crashando), a frequência instantânea exibe um comportamento específico chamado "Chirp"
        (aumento exponencial de frequência e amplitude).

        Detecção de Fase: Monitore a derivada da frequência instantânea dω/dt.

        O Sinal de Perigo:
        - Se dω/dt > 0 (Frequência subindo) E A(t) está crescendo exponencialmente nos
          modos c2 ou c3.
        - Isso indica que a volatilidade não é apenas "ruído", é uma ressonância construtiva.
          O mercado está entrando em auto-excitação.
        """
        n = len(frequency)

        # Derivada da frequência: dω/dt
        freq_derivative = np.gradient(frequency)

        # Derivada da amplitude: dA/dt
        amp_derivative = np.gradient(amplitude)

        # Suavizar derivadas
        freq_derivative_smooth = gaussian_filter1d(freq_derivative, sigma=self.smoothing_sigma)
        amp_derivative_smooth = gaussian_filter1d(amp_derivative, sigma=self.smoothing_sigma)

        # Detectar chirp: dω/dt > 0 E dA/dt > 0
        chirp_mask = (freq_derivative_smooth > self.chirp_threshold) & (amp_derivative_smooth > 0)

        # Força do chirp (produto das derivadas quando positivas)
        chirp_strength = np.where(chirp_mask,
                                   freq_derivative_smooth * amp_derivative_smooth,
                                   0)

        # Chirp atual (últimos pontos)
        lookback = 10
        recent_chirp = chirp_mask[-lookback:] if n >= lookback else chirp_mask
        chirp_detected = np.sum(recent_chirp) >= lookback // 2  # Maioria dos pontos

        # Direção do chirp
        freq_trend = freq_derivative_smooth[-lookback:].mean() if n >= lookback else 0
        amp_trend = amp_derivative_smooth[-lookback:].mean() if n >= lookback else 0

        if freq_trend > 0 and amp_trend > 0:
            chirp_direction = "UP"  # Ressonância crescente
        elif freq_trend > 0 and amp_trend < 0:
            chirp_direction = "DOWN"  # Frequência subindo mas amplitude caindo (dispersão)
        else:
            chirp_direction = "NEUTRAL"

        return {
            'freq_derivative': freq_derivative,
            'freq_derivative_smooth': freq_derivative_smooth,
            'amp_derivative': amp_derivative,
            'amp_derivative_smooth': amp_derivative_smooth,
            'chirp_mask': chirp_mask,
            'chirp_strength': chirp_strength,
            'chirp_detected': chirp_detected,
            'chirp_direction': chirp_direction,
            'freq_trend': freq_trend,
            'amp_trend': amp_trend
        }

    # =========================================================================
    # MÓDULO 4: Análise Fractal da Frequência (Meta-Hurst)
    # =========================================================================

    def calculate_hurst_exponent(self, series: np.ndarray, min_window: int = 10) -> float:
        """
        Calcula o Expoente de Hurst usando R/S Analysis (Rescaled Range).

        Este método usa os RETORNOS (diferenças) da série para calcular R/S,
        que é o método correto para séries temporais financeiras.

        Relação com Dimensão Fractal: D = 2 - H
        - H ≈ 0.5: Passeio aleatório (D ≈ 1.5)
        - H > 0.5: Série persistente/trending (D < 1.5)
        - H < 0.5: Série anti-persistente/mean-reverting (D > 1.5)

        Retorna o Expoente de Hurst (0 a 1).
        """
        n = len(series)
        if n < min_window * 2:
            return 0.5  # Default para dados insuficientes

        # Calcular retornos (diferenças) da série
        returns = np.diff(series)
        if len(returns) < min_window:
            return 0.5

        n_returns = len(returns)

        # Gerar tamanhos de janela
        max_window = n_returns // 2
        window_sizes = []
        w = min_window
        while w <= max_window:
            window_sizes.append(w)
            w = int(w * 1.4)  # Crescimento moderado

        if len(window_sizes) < 3:
            return 0.5

        rs_values = []
        valid_sizes = []

        for window_size in window_sizes:
            n_windows = n_returns // window_size
            if n_windows < 2:
                continue

            rs_list = []

            for i in range(n_windows):
                start = i * window_size
                end = start + window_size
                window = returns[start:end]

                if len(window) < 3:
                    continue

                # Média e desvio padrão da janela de retornos
                mean = np.mean(window)
                std = np.std(window, ddof=1)  # Sample std

                if std < self.eps:
                    continue

                # Série de desvios cumulativos
                cumsum = np.cumsum(window - mean)

                # Range (max - min dos desvios cumulativos)
                R = np.max(cumsum) - np.min(cumsum)

                # R/S (Rescaled Range)
                rs = R / std
                rs_list.append(rs)

            if rs_list:
                rs_values.append(np.mean(rs_list))
                valid_sizes.append(window_size)

        if len(valid_sizes) < 3:
            return 0.5

        # Regressão log-log: log(R/S) vs log(n)
        log_n = np.log(np.array(valid_sizes))
        log_rs = np.log(np.array(rs_values))

        try:
            coeffs = np.polyfit(log_n, log_rs, 1)
            hurst = coeffs[0]  # Slope = Hurst exponent
        except:
            hurst = 0.5

        # Limitar a valores válidos [0.01, 0.99]
        hurst = np.clip(hurst, 0.01, 0.99)

        return hurst

    def calculate_box_counting_dimension(self, curve: np.ndarray,
                                          n_scales: int = 20) -> float:
        """
        Calcula a Dimensão Fractal usando Expoente de Hurst (R/S Analysis).

        Esta implementação usa D = 2 - H onde H é o Expoente de Hurst,
        que é mais robusto para séries temporais financeiras do que
        o box counting tradicional.

        Lógica:
        - Dimensão ≈ 1.5: Passeio aleatório de frequência. (Ignorar).
        - Dimensão → 1.0: A frequência está se tornando linear/determinística.

        Gatilho: Quando detectamos um "Chirp" e a Dimensão Fractal da frequência cai
        abaixo de 1.2, significa que a instabilidade é direcionada.
        """
        n = len(curve)
        if n < 16:
            return 1.5  # Default para dados insuficientes

        # Verificar se a curva é constante
        curve_range = np.max(curve) - np.min(curve)
        if curve_range < self.eps:
            return 1.0  # Curva constante = linha

        # Calcular Hurst exponent
        hurst = self.calculate_hurst_exponent(curve)

        # Converter para dimensão fractal: D = 2 - H
        dimension = 2.0 - hurst

        # Limitar a valores razoáveis [1, 2]
        dimension = np.clip(dimension, 1.0, 2.0)

        return dimension

    def analyze_frequency_fractal(self, frequency: np.ndarray) -> dict:
        """
        Análise fractal completa da frequência instantânea.
        """
        # Dimensão fractal
        fractal_dim = self.calculate_box_counting_dimension(frequency)

        # Classificação
        if fractal_dim >= 1.4:
            fractal_class = "ALEATORIO"
            is_deterministic = False
        elif fractal_dim <= 1.1:
            fractal_class = "LINEAR"
            is_deterministic = True
        else:
            fractal_class = "TRANSICAO"
            is_deterministic = fractal_dim < self.fractal_threshold

        # Gatilho: dimensão < 1.2
        fractal_trigger = fractal_dim < self.fractal_threshold

        return {
            'fractal_dimension': fractal_dim,
            'fractal_class': fractal_class,
            'is_deterministic': is_deterministic,
            'fractal_trigger': fractal_trigger,
            'threshold': self.fractal_threshold
        }

    # =========================================================================
    # MÓDULO 5: Output e Execução
    # =========================================================================

    def detect_energy_concentration(self, energies: np.ndarray, imf_index: int) -> dict:
        """
        Detecta concentração de energia em uma IMF específica.
        """
        if energies.shape[0] <= imf_index:
            return {'concentrated': False, 'ratio': 0.0}

        # Energia total
        total_energy = np.sum(energies)

        # Energia na IMF específica
        imf_energy = np.sum(energies[imf_index])

        # Razão
        energy_ratio = imf_energy / (total_energy + self.eps)

        # Percentil da energia recente
        recent_energy = energies[imf_index][-20:] if energies.shape[1] >= 20 else energies[imf_index]
        energy_percentile_value = np.percentile(recent_energy, self.energy_percentile)

        # Concentração detectada se energia acima do percentil
        concentrated = energy_ratio > 0.2  # Mais de 20% da energia total

        return {
            'concentrated': concentrated,
            'ratio': energy_ratio,
            'imf_energy': imf_energy,
            'total_energy': total_energy,
            'percentile_value': energy_percentile_value
        }

    def generate_signal(self, prices: np.ndarray, eemd_result: dict,
                        chirp_result: dict, fractal_result: dict,
                        hilbert_results: dict) -> dict:
        """
        Sinal de Compra:
        1. EEMD detecta energia concentrada na IMF 2.
        2. Frequência Instantânea ω(t) em ascensão (Chirp positivo).
        3. Preço acima da nuvem reconstruída (Superposição de cn).

        Sinal de Venda:
        1. EEMD detecta energia na IMF 2.
        2. Frequência Instantânea em ascensão (a volatilidade acelera para baixo também).
        3. Preço abaixo da nuvem reconstruída.
        """
        current_price = prices[-1]
        cloud = eemd_result['cloud']
        current_cloud = cloud[-1]

        # Verificar concentração de energia em IMF2
        c2_hilbert = hilbert_results.get('c2', {})
        c2_energy = self.calculate_hilbert_energy(c2_hilbert.get('amplitude_smooth', np.array([0])))

        c3_hilbert = hilbert_results.get('c3', {})
        c3_energy = self.calculate_hilbert_energy(c3_hilbert.get('amplitude_smooth', np.array([0])))

        # Energia concentrada em c2 ou c3?
        c2_energy_mean = np.mean(c2_energy[-20:]) if len(c2_energy) >= 20 else np.mean(c2_energy)
        c3_energy_mean = np.mean(c3_energy[-20:]) if len(c3_energy) >= 20 else np.mean(c3_energy)
        total_energy_mean = c2_energy_mean + c3_energy_mean + self.eps

        energy_in_c2 = c2_energy_mean / total_energy_mean > 0.3
        energy_in_c3 = c3_energy_mean / total_energy_mean > 0.3
        energy_concentrated = energy_in_c2 or energy_in_c3

        # Chirp positivo?
        chirp_positive = chirp_result['chirp_detected'] and chirp_result['freq_trend'] > 0

        # Dimensão fractal baixa (determinística)?
        fractal_low = fractal_result['fractal_trigger']

        # Usar momentum do preco REAL (ultimas 5 barras vs anteriores 5)
        if len(prices) >= 10:
            recent_price = np.mean(prices[-5:])
            older_price = np.mean(prices[-10:-5])
            momentum = (recent_price - older_price) / older_price if older_price > 0 else 0
        else:
            momentum = 0

        # Preco em alta = momentum positivo, em baixa = momentum negativo
        price_above_cloud = momentum > 0.0003  # 0.03% threshold
        price_below_cloud = momentum < -0.0003

        # Determinar sinal
        signal = 0
        signal_name = "NEUTRO"
        confidence = 0.0
        reasons = []

        # Verificar condições
        conditions_met = 0

        if energy_concentrated:
            conditions_met += 1
            reasons.append("Energia concentrada em IMF operável")

        if chirp_positive:
            conditions_met += 1
            reasons.append("Chirp positivo detectado")

        if fractal_low:
            conditions_met += 1
            reasons.append("Frequência determinística (D < 1.2)")

        # Sinal de Compra
        if conditions_met >= 2 and price_above_cloud:
            signal = 1
            signal_name = "COMPRA"
            confidence = conditions_met / 3
            reasons.append("Preço acima da nuvem")

        # Sinal de Venda
        elif conditions_met >= 2 and price_below_cloud:
            signal = -1
            signal_name = "VENDA"
            confidence = conditions_met / 3
            reasons.append("Preço abaixo da nuvem")

        # Alerta (pré-sinal)
        elif conditions_met >= 2:
            signal = 0
            signal_name = "ALERTA"
            confidence = conditions_met / 3
            reasons.append("Aguardando confirmação de preço")

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'reasons': reasons,
            'conditions_met': conditions_met,
            'energy_concentrated': energy_concentrated,
            'energy_in_c2': energy_in_c2,
            'energy_in_c3': energy_in_c3,
            'chirp_positive': chirp_positive,
            'fractal_trigger': fractal_low,
            'price_above_cloud': price_above_cloud,
            'price_below_cloud': price_below_cloud,
            'current_price': current_price,
            'current_cloud': current_cloud
        }

    def analyze(self, prices: np.ndarray) -> dict:
        """
        Execução completa do Ressonador Hilbert-Huang Fractal.
        """
        prices = np.array(prices, dtype=float)
        n = len(prices)

        if n < 100:
            raise ValueError("Dados insuficientes. Necessário mínimo de 100 pontos.")

        # 1. Aplicar extensão preditiva para mitigar efeito de borda
        if self.use_predictive_extension:
            prices_extended, start_idx, end_idx = self.apply_predictive_mirror_extension(prices)
        else:
            prices_extended = prices
            start_idx = 0
            end_idx = n

        # 2. EEMD Decomposition
        eemd_result_ext = self.eemd_decompose(prices_extended)

        # 3. Recortar para remover extensão (eliminar efeito de borda)
        eemd_result = {
            'imfs': [imf[start_idx:end_idx] for imf in eemd_result_ext['imfs']],
            'n_imfs': eemd_result_ext['n_imfs'],
            'c1_hft': eemd_result_ext['c1_hft'][start_idx:end_idx],
            'c2': eemd_result_ext['c2'][start_idx:end_idx],
            'c3': eemd_result_ext['c3'][start_idx:end_idx],
            'c4': eemd_result_ext['c4'][start_idx:end_idx],
            'cn_trend': eemd_result_ext['cn_trend'][start_idx:end_idx],
            'reconstructed_operable': eemd_result_ext['reconstructed_operable'][start_idx:end_idx],
            'cloud': eemd_result_ext['cloud'][start_idx:end_idx],
            'original': prices
        }

        # 4. Transformada de Hilbert para IMFs relevantes
        hilbert_results = {}

        # c2 (IMF principal para operação)
        hilbert_results['c2'] = self.apply_hilbert_transform(eemd_result['c2'])

        # c3
        hilbert_results['c3'] = self.apply_hilbert_transform(eemd_result['c3'])

        # Soma c2 + c3
        hilbert_results['operable'] = self.apply_hilbert_transform(eemd_result['reconstructed_operable'])

        # 5. Espectro de Hilbert completo
        spectrum = self.calculate_hilbert_spectrum(eemd_result['imfs'])

        # 6. Detecção de Chirp (usar IMF operável)
        chirp_result = self.detect_chirp(
            hilbert_results['operable']['frequency_smooth'],
            hilbert_results['operable']['amplitude_smooth']
        )

        # 7. Análise Fractal da Frequência
        fractal_result = self.analyze_frequency_fractal(
            hilbert_results['operable']['frequency_smooth']
        )

        # 8. Gerar Sinal
        signal_result = self.generate_signal(
            prices, eemd_result, chirp_result, fractal_result, hilbert_results
        )

        return {
            # Sinal principal
            'signal': signal_result['signal'],
            'signal_name': signal_result['signal_name'],
            'confidence': signal_result['confidence'],
            'reasons': signal_result['reasons'],

            # EEMD
            'eemd': eemd_result,
            'n_imfs': eemd_result['n_imfs'],

            # Hilbert
            'hilbert': hilbert_results,
            'spectrum': spectrum,

            # Chirp
            'chirp': chirp_result,
            'chirp_detected': chirp_result['chirp_detected'],
            'chirp_direction': chirp_result['chirp_direction'],

            # Fractal
            'fractal': fractal_result,
            'fractal_dimension': fractal_result['fractal_dimension'],
            'fractal_trigger': fractal_result['fractal_trigger'],

            # Detalhes do sinal
            'signal_details': signal_result,

            # Metadados
            'n_observations': n,
            'current_price': prices[-1],
            'cloud_value': eemd_result['cloud'][-1]
        }

    def get_signal(self, prices: np.ndarray) -> int:
        """
        Retorna sinal simplificado:
        1 = COMPRA
        0 = NEUTRO
        -1 = VENDA
        """
        result = self.analyze(prices)
        return result['signal']


# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# =============================================================================

def plot_rhhf_analysis(prices: np.ndarray, save_path: str = None):
    """
    Visualização:
    Plotar o Espectro de Hilbert 3D: Tempo (X) vs Frequência (Y) vs Energia (Cor/Z).
    A alta volatilidade aparecerá como "línguas de fogo" (amarelo/vermelho) surgindo nas
    frequências médias.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import LogNorm

    # Criar indicador e analisar
    rhhf = RessonadorHilbertHuangFractal(n_ensembles=50)
    result = rhhf.analyze(prices)

    # Criar figura
    fig = plt.figure(figsize=(18, 14))

    # Layout
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1.5, 1.5])

    # =========================================================================
    # Plot 1: Preço com Nuvem Reconstruída e IMFs
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :2])

    time = np.arange(len(prices))
    eemd = result['eemd']

    # Preço original
    ax1.plot(time, prices, 'b-', linewidth=1.5, label='Preço', zorder=3)

    # Nuvem reconstruída (c2 + c3 + cn)
    cloud = eemd['cloud']
    cloud_upper = cloud + np.std(eemd['reconstructed_operable'])
    cloud_lower = cloud - np.std(eemd['reconstructed_operable'])

    ax1.fill_between(time, cloud_lower, cloud_upper, alpha=0.3, color='purple',
                    label='Nuvem (c2+c3+cn)', zorder=1)
    ax1.plot(time, cloud, 'purple', linewidth=1, linestyle='--', alpha=0.7, zorder=2)

    # Sinal
    signal = result['signal_details']
    if signal['signal'] == 1:
        ax1.scatter([time[-1]], [prices[-1]], c='green', s=300, marker='^',
                   zorder=5, label='COMPRA')
    elif signal['signal'] == -1:
        ax1.scatter([time[-1]], [prices[-1]], c='red', s=300, marker='v',
                   zorder=5, label='VENDA')

    # Info
    info = (
        f"Sinal: {result['signal_name']}\n"
        f"Chirp: {'SIM' if result['chirp_detected'] else 'NAO'}\n"
        f"D_fractal: {result['fractal_dimension']:.3f}\n"
        f"IMFs: {result['n_imfs']}"
    )
    ax1.text(0.02, 0.95, info, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_title('Ressonador Hilbert-Huang Fractal (RHHF)', fontsize=14)
    ax1.set_ylabel('Preço')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 2: IMFs Decompostas
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 2])

    imfs = eemd['imfs']
    n_show = min(5, len(imfs))

    for i in range(n_show):
        offset = i * np.std(imfs[i]) * 3
        ax2.plot(time, imfs[i] + offset, linewidth=0.8,
                label=f'IMF {i+1}' if i < 4 else f'Trend')

    ax2.set_title('Decomposição EEMD', fontsize=12)
    ax2.set_ylabel('IMFs (offset)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 3: Espectro de Hilbert (Heatmap)
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    spectrum = result['spectrum']
    energies = spectrum['energies']

    # Criar heatmap
    if energies.shape[0] > 0 and energies.shape[1] > 0:
        im = ax3.imshow(energies, aspect='auto', cmap='hot',
                       extent=[0, energies.shape[1], energies.shape[0], 0],
                       norm=LogNorm(vmin=energies.max()*0.001 + 1e-10,
                                   vmax=energies.max() + 1e-10))
        plt.colorbar(im, ax=ax3, label='Energia H_E(t)')

    ax3.set_xlabel('Tempo')
    ax3.set_ylabel('IMF (Frequência)')
    ax3.set_title('Espectro de Hilbert (Energia)', fontsize=12)

    # =========================================================================
    # Plot 4: Frequência Instantânea e Chirp
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    hilbert_op = result['hilbert']['operable']
    freq = hilbert_op['frequency_smooth']
    chirp = result['chirp']

    ax4.plot(time, freq, 'blue', linewidth=1, label='ω(t) Frequência')

    # Derivada da frequência
    ax4_twin = ax4.twinx()
    ax4_twin.plot(time, chirp['freq_derivative_smooth'], 'red', linewidth=1,
                  alpha=0.7, label='dω/dt (Chirp)')
    ax4_twin.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Marcar regiões de chirp
    chirp_mask = chirp['chirp_mask']
    for i in range(len(time)):
        if chirp_mask[i]:
            ax4.axvspan(time[i]-0.5, time[i]+0.5, alpha=0.1, color='orange')

    ax4.set_xlabel('Tempo')
    ax4.set_ylabel('Frequência ω(t)', color='blue')
    ax4_twin.set_ylabel('dω/dt', color='red')
    ax4.set_title(f'Detector de Chirp | Direção: {chirp["chirp_direction"]}', fontsize=12)
    ax4.legend(loc='upper left', fontsize=8)
    ax4_twin.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 5: Amplitude e Energia
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 2])

    amplitude = hilbert_op['amplitude_smooth']
    energy = rhhf.calculate_hilbert_energy(amplitude)

    ax5.fill_between(time, 0, energy, alpha=0.5, color='orange', label='Energia H_E(t)')
    ax5.plot(time, amplitude, 'purple', linewidth=1.5, label='Amplitude A(t)')

    ax5.set_xlabel('Tempo')
    ax5.set_ylabel('Amplitude / Energia')
    ax5.set_title('Energia de Hilbert', fontsize=12)
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 6: Análise Fractal da Frequência
    # =========================================================================
    ax6 = fig.add_subplot(gs[2, 0])

    fractal = result['fractal']

    # Plotar frequência normalizada
    freq_norm = (freq - freq.min()) / (freq.max() - freq.min() + 1e-10)
    ax6.plot(time, freq_norm, 'blue', linewidth=1)

    # Indicar dimensão fractal
    ax6.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    dim_text = (
        f"Dimensão Fractal: {fractal['fractal_dimension']:.3f}\n"
        f"Classe: {fractal['fractal_class']}\n"
        f"Threshold: {fractal['threshold']}\n"
        f"Determinística: {'SIM' if fractal['is_deterministic'] else 'NAO'}"
    )
    ax6.text(0.02, 0.95, dim_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='green' if fractal['fractal_trigger'] else 'lightgray', alpha=0.5))

    ax6.set_xlabel('Tempo')
    ax6.set_ylabel('Frequência Normalizada')
    ax6.set_title('Geometria Fractal da Frequência', fontsize=12)
    ax6.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 7: Espectro 3D de Hilbert
    # =========================================================================
    ax7 = fig.add_subplot(gs[2, 1:], projection='3d')

    # Preparar dados para superfície 3D
    n_time_show = min(200, energies.shape[1])
    time_3d = np.arange(n_time_show)
    imf_3d = np.arange(energies.shape[0])
    T, I = np.meshgrid(time_3d, imf_3d)

    E = energies[:, -n_time_show:] if energies.shape[1] >= n_time_show else energies

    # Plotar superfície
    surf = ax7.plot_surface(T, I, E, cmap='hot', alpha=0.8)

    ax7.set_xlabel('Tempo')
    ax7.set_ylabel('IMF')
    ax7.set_zlabel('Energia')
    ax7.set_title('Espectro de Hilbert 3D', fontsize=12)

    # =========================================================================
    # Resumo
    # =========================================================================
    reasons_str = ', '.join(result['reasons'][:3]) if result['reasons'] else 'Nenhum'
    summary = (
        f"RHHF | Sinal: {result['signal_name']} (Conf: {result['confidence']:.0%}) | "
        f"Chirp: {result['chirp_direction']} | "
        f"D_fractal: {result['fractal_dimension']:.3f} | "
        f"Razões: {reasons_str}"
    )

    fig.text(0.5, 0.01, summary, fontsize=11, ha='center',
            bbox=dict(boxstyle='round',
                     facecolor='green' if result['signal'] == 1 else
                              'red' if result['signal'] == -1 else 'lightblue',
                     alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")

    return fig


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RESSONADOR HILBERT-HUANG FRACTAL (RHHF)")
    print("Engenharia Aeroespacial / Processamento de Sinal Não-Linear Adaptativo")
    print("=" * 80)
    print(f"\nPyEMD disponível: {PYEMD_AVAILABLE}")

    # Gerar dados simulados com características de chirp
    np.random.seed(42)
    n_points = 500
    t = np.linspace(0, 10, n_points)

    # Componentes do sinal
    trend = 1.1000 + 0.02 * t
    cycle1 = 0.005 * np.sin(2 * np.pi * 0.5 * t)
    cycle2 = 0.003 * np.sin(2 * np.pi * 1.5 * t)
    noise = 0.001 * np.random.randn(n_points)

    # CHIRP (aumento exponencial de frequência no final)
    chirp_start = 400
    chirp_freq = np.zeros(n_points)
    chirp_amp = np.zeros(n_points)
    for i in range(chirp_start, n_points):
        progress = (i - chirp_start) / (n_points - chirp_start)
        chirp_freq[i] = 2 * np.pi * (3 + 5 * progress**2)
        chirp_amp[i] = 0.005 * np.exp(2 * progress)

    chirp_signal = chirp_amp * np.sin(np.cumsum(chirp_freq) * 0.02)
    prices = trend + cycle1 + cycle2 + noise + chirp_signal

    print(f"\nDados simulados: {n_points} pontos")
    print(f"Preço inicial: {prices[0]:.5f}")
    print(f"Preço final: {prices[-1]:.5f}")

    # Criar indicador
    rhhf = RessonadorHilbertHuangFractal(
        n_ensembles=50,
        noise_amplitude=0.2,
        mirror_extension=50,
        use_predictive_extension=True,
        chirp_threshold=0.0,
        fractal_threshold=1.2
    )

    # Executar análise
    print("\n" + "-" * 40)
    print("Executando análise RHHF...")
    print("-" * 40)

    result = rhhf.analyze(prices)

    # Mostrar resultados
    print("\nRESULTADO PRINCIPAL:")
    print(f"   Sinal: {result['signal_name']}")
    print(f"   Confiança: {result['confidence']:.2%}")
    print(f"   Razões: {', '.join(result['reasons']) if result['reasons'] else 'Nenhuma'}")

    print("\nDECOMPOSIÇÃO EEMD:")
    print(f"   Número de IMFs: {result['n_imfs']}")
    print(f"   Energia em c2: {'SIM' if result['signal_details']['energy_in_c2'] else 'NAO'}")
    print(f"   Energia em c3: {'SIM' if result['signal_details']['energy_in_c3'] else 'NAO'}")

    print("\nDETECÇÃO DE CHIRP:")
    chirp = result['chirp']
    print(f"   Chirp Detectado: {'SIM' if result['chirp_detected'] else 'NAO'}")
    print(f"   Direção: {result['chirp_direction']}")

    print("\nANÁLISE FRACTAL:")
    fractal = result['fractal']
    print(f"   Dimensão Fractal: {result['fractal_dimension']:.4f}")
    print(f"   Classe: {fractal['fractal_class']}")
    print(f"   Gatilho (D < 1.2): {'SIM' if result['fractal_trigger'] else 'NAO'}")

    print("\nPOSIÇÃO vs NUVEM:")
    signal_det = result['signal_details']
    print(f"   Preço Atual: {signal_det['current_price']:.5f}")
    print(f"   Valor da Nuvem: {signal_det['current_cloud']:.5f}")
    print(f"   Acima da Nuvem: {'SIM' if signal_det['price_above_cloud'] else 'NAO'}")

    print("\n" + "=" * 80)
    if result['signal'] == 1:
        print("SINAL DE COMPRA!")
    elif result['signal'] == -1:
        print("SINAL DE VENDA!")
    elif result['signal_name'] == "ALERTA":
        print("ALERTA! Pré-condições atendidas.")
    else:
        print("SEM SINAL - Mercado em modo normal.")
    print("=" * 80)

    # Gerar visualização
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print("\nGerando visualização...")
        fig = plot_rhhf_analysis(prices, save_path='rhhf_analysis.png')
        print("Visualização salva como 'rhhf_analysis.png'")
        plt.close()
    except Exception as e:
        print(f"\nNão foi possível gerar visualização: {e}")
        import traceback
        traceback.print_exc()
