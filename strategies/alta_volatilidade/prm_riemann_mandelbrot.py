"""
Protocolo Riemann-Mandelbrot (PRM)
==================================
Nível de Complexidade: Ph.D. / Institutional Quant

Premissa Teórica: O preço do EURUSD não é uma linha 2D, é uma projeção de uma variedade
(manifold) multidimensional. A alta volatilidade real ocorre quando há uma Transição de Fase
na microestrutura do mercado (mudança de estado líquido para gasoso). O indicador deve
detectar a coerência dessa transição.

Dependências Críticas: PyWavelets, hmmlearn, nolds, scipy.optimize, numpy
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Dependências Críticas
try:
    import pywt
    from hmmlearn.hmm import GaussianHMM
    import nolds
except ImportError as e:
    print(f"Instale as dependências: pip install PyWavelets hmmlearn nolds")
    raise e


class ProtocoloRiemannMandelbrot:
    """
    Implementação completa do Protocolo Riemann-Mandelbrot (PRM)

    Módulos:
    1. Detecção de Regime: Hidden Markov Models (HMM) Gaussianos
    2. Processamento de Sinal: Transformada Wavelet Contínua (CWT)
    3. Coração Matemático: Expoente de Lyapunov Máximo (λ_max)
    4. Gatilho de Disparo: Curvatura Tensorial (Geometria Diferencial)
    """

    def __init__(self,
                 n_states: int = 3,
                 hmm_threshold: float = 0.85,
                 lyapunov_threshold_k: float = 0.5,
                 wavelet_mother: str = 'cmor1.5-1.0',
                 garch_omega: float = 0.00001,
                 garch_alpha: float = 0.1,
                 garch_beta: float = 0.85,
                 curvature_threshold: float = 0.1,
                 lookback_window: int = 100):
        """
        Inicialização do Protocolo Riemann-Mandelbrot

        Parâmetros:
        -----------
        n_states : int
            Número de estados latentes do HMM (default: 3)
            - Estado 0: Movimento Browniano (Ruído/Consolidação)
            - Estado 1: Alta Volatilidade Direcional (Tendência/Fluxo Institucional)
            - Estado 2: Choque de Volatilidade (Flash crashes/News spikes)

        hmm_threshold : float
            Limiar de Probabilidade Posterior para ativação (default: 0.85)

        lyapunov_threshold_k : float
            Limiar empírico K para Caos Determinístico (default: 0.5)

        wavelet_mother : str
            Wavelet-mãe para CWT (default: 'cmor1.5-1.0' - Morlet Complexa)

        garch_omega, garch_alpha, garch_beta : float
            Parâmetros do modelo GARCH(1,1) para estimação de volatilidade

        curvature_threshold : float
            Limite crítico para Aceleração da Curvatura (Δκ)

        lookback_window : int
            Janela de lookback para cálculos deslizantes
        """
        self.n_states = n_states
        self.hmm_threshold = hmm_threshold
        self.lyapunov_threshold_k = lyapunov_threshold_k
        self.wavelet_mother = wavelet_mother
        self.garch_omega = garch_omega
        self.garch_alpha = garch_alpha
        self.garch_beta = garch_beta
        self.curvature_threshold = curvature_threshold
        self.lookback_window = lookback_window

        # Modelo HMM
        self.hmm_model = None
        self.is_fitted = False

        # Cache de resultados
        self._cache = {}

    # =========================================================================
    # MÓDULO 1: Detecção de Regime - Hidden Markov Models (HMM) Gaussianos
    # =========================================================================

    def _calculate_log_returns(self, prices: np.ndarray) -> np.ndarray:
        """Calcula retornos logarítmicos"""
        return np.diff(np.log(prices))

    def _estimate_garch_volatility(self, returns: np.ndarray) -> np.ndarray:
        """
        Estima volatilidade usando GARCH(1,1)

        σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
        """
        n = len(returns)
        variance = np.zeros(n)
        variance[0] = np.var(returns)

        omega = self.garch_omega
        alpha = self.garch_alpha
        beta = self.garch_beta

        for t in range(1, n):
            variance[t] = omega + alpha * (returns[t-1] ** 2) + beta * variance[t-1]

        return np.sqrt(variance)

    def _prepare_hmm_features(self, prices: np.ndarray, volume: np.ndarray = None) -> np.ndarray:
        """
        Prepara features para o HMM

        Input do HMM: Retornos logarítmicos, Volatilidade GARCH(1,1) estimada e Tick Volume
        """
        returns = self._calculate_log_returns(prices)
        volatility = self._estimate_garch_volatility(returns)

        # Se volume não for fornecido, usar proxy baseado em volatilidade
        if volume is None:
            volume = np.abs(returns) * 1000  # Proxy simples
        else:
            volume = volume[1:]  # Alinhar com retornos

        # Normalizar features
        features = np.column_stack([
            (returns - np.mean(returns)) / (np.std(returns) + 1e-10),
            (volatility - np.mean(volatility)) / (np.std(volatility) + 1e-10),
            (volume - np.mean(volume)) / (np.std(volume) + 1e-10)
        ])

        return features

    def fit_hmm(self, prices: np.ndarray, volume: np.ndarray = None, n_iter: int = 100):
        """
        Treina o modelo HMM Gaussiano em tempo real com 3 estados latentes

        Estados:
        - Estado 0: Movimento Browniano (Ruído/Consolidação)
        - Estado 1: Alta Volatilidade Direcional (Tendência/Fluxo Institucional)
        - Estado 2: Choque de Volatilidade (Flash crashes/News spikes)
        """
        features = self._prepare_hmm_features(prices, volume)

        self.hmm_model = GaussianHMM(
            n_components=self.n_states,
            covariance_type='full',
            n_iter=n_iter,
            random_state=42,
            verbose=False
        )

        self.hmm_model.fit(features)
        self.is_fitted = True

        return self

    def get_hmm_probabilities(self, prices: np.ndarray, volume: np.ndarray = None) -> dict:
        """
        Obtém probabilidades posteriores do HMM

        Gatilho: O algoritmo só "acorda" quando a Probabilidade Posterior
        do Estado 1 ou 2 for > 0.85
        """
        if not self.is_fitted:
            self.fit_hmm(prices, volume)

        features = self._prepare_hmm_features(prices, volume)

        # Probabilidades posteriores
        posterior_probs = self.hmm_model.predict_proba(features)

        # Estado mais provável
        states = self.hmm_model.predict(features)

        # Probabilidade do estado atual (último ponto)
        current_state = states[-1]
        current_prob = posterior_probs[-1, current_state]

        # Verificar se Estado 1 ou 2 tem probabilidade > threshold
        prob_state_1 = posterior_probs[-1, 1] if self.n_states > 1 else 0
        prob_state_2 = posterior_probs[-1, 2] if self.n_states > 2 else 0

        hmm_activated = (prob_state_1 > self.hmm_threshold) or (prob_state_2 > self.hmm_threshold)
        high_volatility_state = current_state in [1, 2] and hmm_activated

        return {
            'posterior_probs': posterior_probs,
            'states': states,
            'current_state': current_state,
            'current_prob': current_prob,
            'prob_state_0': posterior_probs[-1, 0],
            'prob_state_1': prob_state_1,
            'prob_state_2': prob_state_2,
            'hmm_activated': hmm_activated,
            'high_volatility_state': high_volatility_state,
            'Prob_HMM': max(prob_state_1, prob_state_2)
        }

    # =========================================================================
    # MÓDULO 2: Processamento de Sinal - Transformada Wavelet Contínua (CWT)
    # =========================================================================

    def apply_cwt(self, prices: np.ndarray, scales: np.ndarray = None) -> dict:
        """
        Aplica a Transformada Wavelet Contínua (CWT) usando wavelet-mãe Morlet Complexa

        A Análise de Fourier perde a informação do tempo. Para "snipar" a entrada,
        precisamos de resolução tempo-frequência.

        Objetivo: Isolar coeficientes de alta energia em escalas específicas que
        correspondem aos ciclos de liquidez dos bancos centrais (intraday).
        """
        if scales is None:
            # Escalas que capturam ciclos de diferentes frequências
            # Baixas = alta frequência (HFT), Altas = baixa frequência (macro)
            scales = np.arange(1, min(128, len(prices) // 4))

        # Aplicar CWT com Morlet Complexa
        coefficients, frequencies = pywt.cwt(prices, scales, self.wavelet_mother)

        # Calcular potência (magnitude ao quadrado)
        power = np.abs(coefficients) ** 2

        return {
            'coefficients': coefficients,
            'frequencies': frequencies,
            'scales': scales,
            'power': power
        }

    def filter_cwt_reconstruct(self, prices: np.ndarray,
                                low_scale_cutoff: int = 5,
                                high_scale_cutoff: int = 64) -> np.ndarray:
        """
        Filtro: Reconstruir o sinal (iCWT) descartando os coeficientes de alta
        frequência (ruído de HFT) e de frequência ultra-baixa (tendência macro),
        focando apenas na "Banda de Volatilidade Operável".

        Parâmetros:
        -----------
        low_scale_cutoff : int
            Escala mínima (descartar ruído HFT)
        high_scale_cutoff : int
            Escala máxima (descartar tendência macro)
        """
        scales = np.arange(1, min(128, len(prices) // 4))
        cwt_result = self.apply_cwt(prices, scales)
        coefficients = cwt_result['coefficients']

        # Filtrar: manter apenas escalas na "Banda de Volatilidade Operável"
        filtered_coeffs = coefficients.copy()

        # Zerar coeficientes fora da banda
        for i, scale in enumerate(scales):
            if scale < low_scale_cutoff or scale > high_scale_cutoff:
                filtered_coeffs[i, :] = 0

        # Reconstrução aproximada (soma ponderada dos coeficientes filtrados)
        # Usando método de reconstrução por soma
        reconstructed = np.real(np.sum(filtered_coeffs, axis=0))

        # Normalizar para mesma escala do preço original
        reconstructed = reconstructed - np.mean(reconstructed)
        reconstructed = reconstructed / (np.std(reconstructed) + 1e-10)
        reconstructed = reconstructed * np.std(prices) + np.mean(prices)

        return reconstructed

    def get_wavelet_power_spectrogram(self, prices: np.ndarray) -> dict:
        """
        Visualização: Plotar o Espectrograma de Potência da Wavelet em 3D (Heatmap)
        abaixo do gráfico de preço para confirmar visualmente a densidade de energia.
        """
        cwt_result = self.apply_cwt(prices)

        return {
            'power': cwt_result['power'],
            'scales': cwt_result['scales'],
            'frequencies': cwt_result['frequencies'],
            'time': np.arange(len(prices))
        }

    # =========================================================================
    # MÓDULO 3: Coração Matemático - Expoente de Lyapunov Máximo (λ_max)
    # =========================================================================

    def calculate_lyapunov_exponent(self, series: np.ndarray,
                                     emb_dim: int = 10,
                                     matrix_dim: int = 4) -> float:
        """
        Calcular o λ_max sobre a série temporal reconstruída pela Wavelet
        em uma janela deslizante.

        Equação de referência para a divergência de trajetórias:
        d(t) = C * e^(λt)

        Precisamos saber se a volatilidade é determinística (operável)
        ou estocástica pura (aleatória).
        """
        try:
            # Usar biblioteca nolds para cálculo robusto do expoente de Lyapunov
            lyap = nolds.lyap_r(series, emb_dim=emb_dim, lag=1, min_tsep=None)
            return lyap
        except Exception:
            # Fallback: método de Rosenstein simplificado
            return self._lyapunov_rosenstein(series, emb_dim)

    def _lyapunov_rosenstein(self, series: np.ndarray, emb_dim: int = 10) -> float:
        """
        Implementação do algoritmo de Rosenstein para cálculo do expoente de Lyapunov
        """
        n = len(series)
        if n < emb_dim * 2:
            return 0.0

        # Criar matriz de embedding (reconstrução do espaço de fase)
        m = n - emb_dim + 1
        embedded = np.zeros((m, emb_dim))
        for i in range(m):
            embedded[i] = series[i:i + emb_dim]

        # Encontrar vizinhos mais próximos (excluindo temporalmente próximos)
        min_tsep = emb_dim
        divergence = []

        for i in range(m - min_tsep):
            # Distâncias para todos os outros pontos
            distances = np.linalg.norm(embedded - embedded[i], axis=1)
            distances[max(0, i - min_tsep):min(m, i + min_tsep)] = np.inf

            # Vizinho mais próximo
            j = np.argmin(distances)
            if distances[j] < np.inf:
                # Rastrear divergência
                max_k = min(m - i - 1, m - j - 1, 20)
                for k in range(1, max_k):
                    d = np.linalg.norm(embedded[i + k] - embedded[j + k])
                    if d > 0:
                        divergence.append((k, np.log(d)))

        if len(divergence) < 10:
            return 0.0

        # Regressão linear para estimar λ
        divergence = np.array(divergence)
        k_vals = divergence[:, 0]
        log_d = divergence[:, 1]

        # Fit linear
        slope, _ = np.polyfit(k_vals, log_d, 1)

        return slope

    def get_lyapunov_analysis(self, prices: np.ndarray) -> dict:
        """
        Lógica de Ativação do Lyapunov:

        - Se λ_max < 0: O sistema é estável/atrator fixo (Consolidação). IGNORAR.
        - Se λ_max >> 0: O sistema é puramente caótico imprevisível. IGNORAR.
        - Se 0 < λ_max < K (onde K é um limiar empírico ajustado, ex: 0.5):
          O sistema está em Caos Determinístico. Existe uma "ordem oculta" na volatilidade.
          ESTE É O PONTO DE ENTRADA.
        """
        # Usar série filtrada pela wavelet
        filtered_series = self.filter_cwt_reconstruct(prices)

        # Calcular λ_max em janela deslizante
        window = min(self.lookback_window, len(filtered_series) - 20)
        if window < 30:
            window = len(filtered_series) - 1

        recent_series = filtered_series[-window:]
        lyapunov = self.calculate_lyapunov_exponent(recent_series)

        # Classificação
        K = self.lyapunov_threshold_k

        if lyapunov < 0:
            classification = "ESTAVEL_CONSOLIDACAO"
            action = "IGNORAR"
            is_entry_point = False
        elif lyapunov > K:
            classification = "CAOTICO_PURO"
            action = "IGNORAR"
            is_entry_point = False
        else:  # 0 < lyapunov < K
            classification = "CAOS_DETERMINISTICO"
            action = "PONTO_DE_ENTRADA"
            is_entry_point = True

        return {
            'lyapunov_max': lyapunov,
            'threshold_K': K,
            'classification': classification,
            'action': action,
            'is_entry_point': is_entry_point,
            'Lyapunov_Score': lyapunov
        }

    # =========================================================================
    # MÓDULO 4: Gatilho de Disparo - Curvatura Tensorial (Geometria Diferencial)
    # =========================================================================

    def calculate_curvature(self, prices: np.ndarray) -> np.ndarray:
        """
        Trataremos o preço e o tempo como uma curva no espaço vetorial.

        Calcula a Curvatura (κ) local:
        κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)

        (Onde x é tempo e y é preço)

        Passos:
        1. Calcule o vetor tangente unitário (T) e o vetor normal unitário (N)
           da série de preços suavizada pela Wavelet.
        2. Calcule a Curvatura (κ) local.
        """
        # Usar série suavizada pela wavelet
        filtered_prices = self.filter_cwt_reconstruct(prices)

        n = len(filtered_prices)
        t = np.arange(n, dtype=float)  # x = tempo
        y = filtered_prices  # y = preço

        # Derivadas numéricas (diferenças finitas centradas)
        # x' = dx/dt = 1 (tempo é uniforme)
        x_prime = np.ones(n)

        # y' = dy/dt
        y_prime = np.gradient(y)

        # x'' = d²x/dt² = 0
        x_double_prime = np.zeros(n)

        # y'' = d²y/dt²
        y_double_prime = np.gradient(y_prime)

        # Curvatura: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(x_prime * y_double_prime - y_prime * x_double_prime)
        denominator = (x_prime**2 + y_prime**2)**(3/2)

        # Evitar divisão por zero
        curvature = np.where(denominator > 1e-10, numerator / denominator, 0)

        return curvature

    def calculate_tangent_normal_vectors(self, prices: np.ndarray) -> dict:
        """
        Calcule o vetor tangente unitário (T) e o vetor normal unitário (N)
        """
        filtered_prices = self.filter_cwt_reconstruct(prices)

        n = len(filtered_prices)

        # Derivadas
        dy = np.gradient(filtered_prices)
        dx = np.ones(n)  # dt = 1

        # Vetor tangente
        tangent = np.column_stack([dx, dy])
        tangent_magnitude = np.linalg.norm(tangent, axis=1, keepdims=True)
        T = tangent / (tangent_magnitude + 1e-10)  # Unitário

        # Vetor normal (perpendicular ao tangente, rotacionado 90°)
        N = np.column_stack([-T[:, 1], T[:, 0]])

        return {
            'tangent_vector': T,
            'normal_vector': N,
            'tangent_magnitude': tangent_magnitude.flatten()
        }

    def get_curvature_signal(self, prices: np.ndarray) -> dict:
        """
        Sinal de Disparo baseado na Curvatura:

        1. HMM diz que estamos em Estado de Alta Volatilidade.
        2. Lyapunov indica Caos Determinístico (tendência sustentável).
        3. A Aceleração da Curvatura (Δκ) excede um limite crítico (indica que a
           "força G" do mercado mudou de direção drasticamente, sinalizando
           entrada de grandes players).
        """
        curvature = self.calculate_curvature(prices)

        # Aceleração da Curvatura (Δκ)
        curvature_acceleration = np.gradient(curvature)

        # Verificar se excede limite crítico
        current_acceleration = curvature_acceleration[-1]
        exceeds_threshold = np.abs(current_acceleration) > self.curvature_threshold

        # Média móvel da curvatura para suavização
        window = min(10, len(curvature) // 4)
        if window < 2:
            window = 2
        curvature_ma = np.convolve(curvature, np.ones(window)/window, mode='valid')

        return {
            'curvature': curvature,
            'curvature_acceleration': curvature_acceleration,
            'current_curvature': curvature[-1],
            'current_acceleration': current_acceleration,
            'exceeds_threshold': exceeds_threshold,
            'threshold': self.curvature_threshold,
            'curvature_ma': curvature_ma,
            'Curvature_Signal': 1 if exceeds_threshold else 0
        }

    # =========================================================================
    # MÓDULO 5: Output e Execução
    # =========================================================================

    def analyze(self, prices: np.ndarray, volume: np.ndarray = None) -> dict:
        """
        Execução completa do Protocolo Riemann-Mandelbrot

        Retorno: O script deve retornar um vetor [Prob_HMM, Lyapunov_Score, Curvature_Signal]

        Condição Booleana Final: True apenas se todos os subsistemas validarem
        a hipótese de "Singularidade de Preço".
        """
        prices = np.array(prices, dtype=float)

        if len(prices) < 50:
            raise ValueError("Dados insuficientes. Necessário mínimo de 50 pontos de preço.")

        # 1. Análise HMM
        hmm_result = self.get_hmm_probabilities(prices, volume)

        # 2. Análise Wavelet (já utilizada internamente)
        wavelet_spectrogram = self.get_wavelet_power_spectrogram(prices)

        # 3. Análise Lyapunov
        lyapunov_result = self.get_lyapunov_analysis(prices)

        # 4. Análise de Curvatura
        curvature_result = self.get_curvature_signal(prices)

        # Vetor de saída: [Prob_HMM, Lyapunov_Score, Curvature_Signal]
        output_vector = [
            hmm_result['Prob_HMM'],
            lyapunov_result['Lyapunov_Score'],
            curvature_result['Curvature_Signal']
        ]

        # Condição Booleana Final: True apenas se TODOS os subsistemas validarem
        # a hipótese de "Singularidade de Preço"
        singularity_detected = (
            hmm_result['high_volatility_state'] and      # HMM em estado de alta volatilidade
            lyapunov_result['is_entry_point'] and        # Lyapunov indica caos determinístico
            curvature_result['exceeds_threshold']         # Curvatura excede limite crítico
        )

        return {
            # Vetor de saída principal
            'output_vector': output_vector,
            'Prob_HMM': output_vector[0],
            'Lyapunov_Score': output_vector[1],
            'Curvature_Signal': output_vector[2],

            # Condição final
            'singularity_detected': singularity_detected,
            'price_singularity': singularity_detected,

            # Detalhes de cada módulo
            'hmm_analysis': hmm_result,
            'lyapunov_analysis': lyapunov_result,
            'curvature_analysis': curvature_result,
            'wavelet_spectrogram': wavelet_spectrogram,

            # Metadados
            'n_observations': len(prices),
            'current_price': prices[-1]
        }

    def get_signal(self, prices: np.ndarray, volume: np.ndarray = None) -> int:
        """
        Retorna sinal simplificado:
        1 = Singularidade detectada (entrada potencial)
        0 = Sem sinal
        """
        result = self.analyze(prices, volume)
        return 1 if result['singularity_detected'] else 0


# =============================================================================
# FUNÇÕES AUXILIARES PARA VISUALIZAÇÃO
# =============================================================================

def plot_prm_analysis(prices: np.ndarray, volume: np.ndarray = None,
                       save_path: str = None):
    """
    Visualização: Plotar o Espectrograma de Potência da Wavelet em 3D (Heatmap)
    abaixo do gráfico de preço para confirmar visualmente a densidade de energia.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # Criar instância e analisar
    prm = ProtocoloRiemannMandelbrot()
    result = prm.analyze(prices, volume)

    # Criar figura com subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 12),
                             gridspec_kw={'height_ratios': [2, 2, 1, 1]})

    time = np.arange(len(prices))

    # 1. Gráfico de Preço com Estados do HMM
    ax1 = axes[0]
    states = result['hmm_analysis']['states']
    colors = ['green', 'orange', 'red']
    state_names = ['Consolidação', 'Alta Vol. Direcional', 'Choque de Vol.']

    # Alinhar arrays (states tem len(prices)-1 elementos)
    prices_aligned = prices[1:]
    time_aligned = time[1:]

    # Plotar preço
    ax1.plot(time_aligned, prices_aligned, 'b-', linewidth=1, label='Preço')

    # Colorir fundo por estado
    for state in range(3):
        mask = states == state
        ax1.fill_between(time_aligned, prices_aligned.min(), prices_aligned.max(),
                        where=mask, alpha=0.2, color=colors[state],
                        label=f'Estado {state}: {state_names[state]}')

    ax1.set_title('Protocolo Riemann-Mandelbrot (PRM) - Análise Completa', fontsize=14)
    ax1.set_ylabel('Preço')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Adicionar indicador de singularidade
    if result['singularity_detected']:
        ax1.axvline(x=time_aligned[-1], color='purple', linestyle='--', linewidth=2,
                   label='SINGULARIDADE DETECTADA')
        ax1.text(time_aligned[-1], prices_aligned.max(), 'SINGULARIDADE', fontsize=10,
                color='purple', ha='right', fontweight='bold')

    # 2. Espectrograma de Potência da Wavelet (Heatmap)
    ax2 = axes[1]
    wavelet_data = result['wavelet_spectrogram']
    power = wavelet_data['power']
    scales = wavelet_data['scales']

    # Plotar heatmap
    im = ax2.imshow(power, aspect='auto', cmap='hot',
                    extent=[0, len(prices), scales[-1], scales[0]],
                    norm=LogNorm())
    ax2.set_ylabel('Escala (Frequência Inversa)')
    ax2.set_title('Espectrograma de Potência da Wavelet (CWT)', fontsize=12)
    plt.colorbar(im, ax=ax2, label='Potência')

    # 3. Expoente de Lyapunov e Curvatura
    ax3 = axes[2]
    curvature = result['curvature_analysis']['curvature']
    curvature_time = np.arange(len(curvature))
    ax3.plot(curvature_time, curvature, 'g-', linewidth=1, label='Curvatura')
    ax3.axhline(y=prm.curvature_threshold, color='r', linestyle='--',
               label=f'Threshold = {prm.curvature_threshold}')
    ax3.axhline(y=-prm.curvature_threshold, color='r', linestyle='--')

    # Adicionar λ_max como texto
    lyap = result['lyapunov_analysis']['lyapunov_max']
    lyap_class = result['lyapunov_analysis']['classification']
    ax3.text(0.02, 0.95, f'λ_max = {lyap:.4f}\n{lyap_class}',
            transform=ax3.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax3.set_ylabel('Curvatura')
    ax3.set_title('Curvatura Tensorial e Expoente de Lyapunov', fontsize=12)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Probabilidades do HMM
    ax4 = axes[3]
    posterior = result['hmm_analysis']['posterior_probs']
    ax4.stackplot(time_aligned, posterior.T, labels=[f'P(Estado {i})' for i in range(3)],
                  colors=colors, alpha=0.7)
    ax4.axhline(y=prm.hmm_threshold, color='black', linestyle='--',
               label=f'Threshold = {prm.hmm_threshold}')
    ax4.set_ylabel('Probabilidade')
    ax4.set_xlabel('Tempo')
    ax4.set_title('Probabilidades Posteriores do HMM', fontsize=12)
    ax4.legend(loc='upper left', fontsize=8)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    # Vetor de saída
    output = result['output_vector']
    fig.text(0.02, 0.02,
             f"Output: [Prob_HMM={output[0]:.4f}, Lyapunov_Score={output[1]:.4f}, "
             f"Curvature_Signal={output[2]}] | "
             f"Singularidade: {'SIM' if result['singularity_detected'] else 'NAO'}",
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")

    return fig


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PROTOCOLO RIEMANN-MANDELBROT (PRM)")
    print("Indicador de Detecção de Singularidade de Preço")
    print("=" * 80)

    # Gerar dados simulados (em produção, usar dados reais do EURUSD)
    np.random.seed(42)
    n_points = 500

    # Simular preços com diferentes regimes
    t = np.linspace(0, 10, n_points)

    # Regime 1: Consolidação (0-150)
    prices_1 = 1.1000 + 0.001 * np.cumsum(np.random.randn(150))

    # Regime 2: Tendência com alta volatilidade (150-350)
    trend = np.linspace(0, 0.02, 200)
    prices_2 = prices_1[-1] + trend + 0.003 * np.cumsum(np.random.randn(200))

    # Regime 3: Choque de volatilidade (350-500)
    shock = np.zeros(150)
    shock[50:70] = -0.005  # Flash crash
    prices_3 = prices_2[-1] + shock + 0.002 * np.cumsum(np.random.randn(150))

    # Concatenar
    prices = np.concatenate([prices_1, prices_2, prices_3])

    # Volume simulado
    volume = np.abs(np.diff(prices)) * 100000 + np.random.rand(n_points - 1) * 10000
    volume = np.concatenate([[volume[0]], volume])

    print(f"\nDados simulados: {len(prices)} pontos de preço")
    print(f"Preço inicial: {prices[0]:.5f}")
    print(f"Preço final: {prices[-1]:.5f}")

    # Criar instância do PRM
    prm = ProtocoloRiemannMandelbrot(
        n_states=3,
        hmm_threshold=0.85,
        lyapunov_threshold_k=0.5,
        curvature_threshold=0.1
    )

    # Executar análise completa
    print("\n" + "-" * 40)
    print("Executando análise PRM...")
    print("-" * 40)

    result = prm.analyze(prices, volume)

    # Mostrar resultados
    print("\nVETOR DE SAIDA:")
    print(f"   [Prob_HMM, Lyapunov_Score, Curvature_Signal]")
    print(f"   [{result['Prob_HMM']:.4f}, {result['Lyapunov_Score']:.4f}, {result['Curvature_Signal']}]")

    print("\nANALISE DO HMM:")
    hmm = result['hmm_analysis']
    print(f"   Estado Atual: {hmm['current_state']}")
    print(f"   P(Estado 0 - Consolidação): {hmm['prob_state_0']:.4f}")
    print(f"   P(Estado 1 - Alta Vol. Direcional): {hmm['prob_state_1']:.4f}")
    print(f"   P(Estado 2 - Choque de Vol.): {hmm['prob_state_2']:.4f}")
    print(f"   HMM Ativado (P > 0.85): {'SIM' if hmm['hmm_activated'] else 'NAO'}")

    print("\nANALISE DE LYAPUNOV:")
    lyap = result['lyapunov_analysis']
    print(f"   lambda_max: {lyap['lyapunov_max']:.6f}")
    print(f"   Threshold K: {lyap['threshold_K']}")
    print(f"   Classificação: {lyap['classification']}")
    print(f"   Ação: {lyap['action']}")

    print("\nANALISE DE CURVATURA:")
    curv = result['curvature_analysis']
    print(f"   Curvatura Atual: {curv['current_curvature']:.6f}")
    print(f"   Aceleração Atual: {curv['current_acceleration']:.6f}")
    print(f"   Threshold: {curv['threshold']}")
    print(f"   Excede Limite: {'SIM' if curv['exceeds_threshold'] else 'NAO'}")

    print("\n" + "=" * 80)
    if result['singularity_detected']:
        print("SINGULARIDADE DE PRECO DETECTADA!")
        print("   Todos os subsistemas validaram a hipótese.")
        print("   Considere entrada no mercado.")
    else:
        print("Singularidade NAO detectada.")
        print("   Aguardar alinhamento de todos os subsistemas.")
    print("=" * 80)
