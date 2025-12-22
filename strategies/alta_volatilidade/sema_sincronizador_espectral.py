"""
Sincronizador Espectral (SEMA)
==============================
Nível de Complexidade: Gestão de Risco Sistêmico / Engenharia de Redes Complexas.

Premissa Teórica: O mercado é um grafo onde os nós são ativos (EURUSD, DXY, Gold, US10Y,
SPX500) e as arestas são forças de correlação. A maior parte da correlação é ruído (explicada
pela distribuição de Marchenko-Pastur). O sinal real ("O Modo de Mercado") reside nos
autovalores (eigenvalues) que escapam dessa distribuição de ruído. Alta volatilidade no
EURUSD é precedida por uma mudança na Conectividade Algébrica global.

Dependências Críticas: numpy.linalg (lapack), scipy.sparse, networkx, pandas
"""

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Para grafos
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("AVISO: networkx não disponível. Algumas funcionalidades serão limitadas.")


class SincronizadorEspectral:
    """
    Implementação completa do Sincronizador Espectral (SEMA)

    Módulos:
    1. A Infraestrutura de Dados (Input Multivariado)
    2. Matriz de Correlação e Filtragem de Ruído (RMT)
    3. Teoria Espectral de Grafos (O Cálculo Pesado)
    4. Entropia de Von Neumann (A Termodinâmica do Grafo)
    5. O Algoritmo de Decisão (O Sniper Espectral)
    6. Output e Visualização
    """

    def __init__(self,
                 correlation_window: int = 120,
                 fiedler_percentile_threshold: float = 90,
                 entropy_drop_threshold: float = 0.3,
                 entropy_critical_threshold: float = 0.5,
                 use_lanczos: bool = True,
                 perturbation_update: bool = True,
                 n_top_eigenvalues: int = 5,
                 asset_names: list = None):
        """
        Inicialização do Sincronizador Espectral

        Parâmetros:
        -----------
        correlation_window : int
            Janela móvel para cálculo da matriz de correlação (default: 120 períodos)

        fiedler_percentile_threshold : float
            Percentil histórico do λ₂ para pré-requisito (default: 90)

        entropy_drop_threshold : float
            Limiar de queda súbita na entropia para gatilho (default: 0.3)

        entropy_critical_threshold : float
            Limiar crítico absoluto de entropia (default: 0.5)

        use_lanczos : bool
            Usar algoritmo de Lanczos para eigenvalues (mais eficiente)

        perturbation_update : bool
            Usar métodos perturbativos para atualização incremental

        n_top_eigenvalues : int
            Número de autovalores a calcular (para Lanczos)

        asset_names : list
            Nomes dos ativos (default: ['EURUSD', 'DXY', 'XAUUSD', 'US10Y', 'SPX500'])
        """
        self.correlation_window = correlation_window
        self.fiedler_percentile_threshold = fiedler_percentile_threshold
        self.entropy_drop_threshold = entropy_drop_threshold
        self.entropy_critical_threshold = entropy_critical_threshold
        self.use_lanczos = use_lanczos
        self.perturbation_update = perturbation_update
        self.n_top_eigenvalues = n_top_eigenvalues

        # Nomes dos ativos
        if asset_names is None:
            self.asset_names = ['EURUSD', 'DXY', 'XAUUSD', 'US10Y', 'SPX500']
        else:
            self.asset_names = asset_names

        self.n_assets = len(self.asset_names)

        # Cache para otimização
        self._cache = {
            'last_correlation': None,
            'last_eigenvalues': None,
            'last_eigenvectors': None,
            'fiedler_history': [],
            'entropy_history': []
        }

        # Epsilon para estabilidade numérica
        self.eps = 1e-10

    # =========================================================================
    # MÓDULO 1: A Infraestrutura de Dados (Input Multivariado)
    # =========================================================================

    def _calculate_log_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Este indicador não funciona só com o OHLC do EURUSD. O script deve ingerir,
        em tempo real, os dados normalizados (retornos logarítmicos) de uma cesta
        de ativos correlatos.

        Ativos:
        - EURUSD (O Alvo)
        - DXY (Índice Dólar - O Driver)
        - XAUUSD (Ouro - Hedge)
        - US10Y (Yields de 10 anos - Risco Livre)
        - SPX500 (Risco de Mercado)
        """
        return np.diff(np.log(prices), axis=0)

    def prepare_multivariate_data(self, data_dict: dict) -> pd.DataFrame:
        """
        Prepara dados multivariados a partir de dicionário de preços.

        Parâmetros:
        -----------
        data_dict : dict
            Dicionário com arrays de preços para cada ativo.
            Ex: {'EURUSD': [...], 'DXY': [...], 'XAUUSD': [...], ...}

        Retorno:
        --------
        DataFrame com retornos logarítmicos alinhados
        """
        # Criar DataFrame com preços
        prices_df = pd.DataFrame(data_dict)

        # Calcular retornos logarítmicos
        returns_df = np.log(prices_df).diff().dropna()

        return returns_df

    # =========================================================================
    # MÓDULO 2: Matriz de Correlação e Filtragem de Ruído (RMT)
    # =========================================================================

    def calculate_correlation_matrix(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculamos a Matriz de Correlação de Pearson (C) numa janela móvel (ex: 120 períodos).
        Mas C é suja.
        """
        n_samples, n_assets = returns.shape

        # Correlação de Pearson
        corr_matrix = np.corrcoef(returns.T)

        # Garantir que é simétrica e bem formada
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1.0)

        # Tratar NaN/Inf
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

        return corr_matrix

    def marchenko_pastur_bounds(self, n_samples: int, n_assets: int,
                                 sigma: float = 1.0) -> tuple:
        """
        Limpeza via RMT: Compare os autovalores de C com a distribuição teórica de
        Marchenko-Pastur:

        p(λ) = (1 / 2πσ²) * √((λ₊ - λ)(λ - λ₋)) / λ

        Os limites são:
        λ₊ = σ²(1 + √(n_assets/n_samples))²
        λ₋ = σ²(1 - √(n_assets/n_samples))²
        """
        q = n_assets / n_samples

        # Limites de Marchenko-Pastur
        lambda_plus = sigma**2 * (1 + np.sqrt(q))**2
        lambda_minus = sigma**2 * (1 - np.sqrt(q))**2

        # Garantir λ₋ >= 0
        lambda_minus = max(0, lambda_minus)

        return lambda_minus, lambda_plus

    def marchenko_pastur_pdf(self, lambdas: np.ndarray, n_samples: int,
                              n_assets: int, sigma: float = 1.0) -> np.ndarray:
        """
        Calcula a PDF de Marchenko-Pastur para comparação.

        p(λ) = (1 / 2πσ²) * √((λ₊ - λ)(λ - λ₋)) / λ
        """
        lambda_minus, lambda_plus = self.marchenko_pastur_bounds(n_samples, n_assets, sigma)

        pdf = np.zeros_like(lambdas)

        # Apenas para λ dentro dos bounds
        mask = (lambdas >= lambda_minus) & (lambdas <= lambda_plus)

        if np.any(mask):
            q = n_assets / n_samples
            numerator = np.sqrt((lambda_plus - lambdas[mask]) * (lambdas[mask] - lambda_minus))
            denominator = 2 * np.pi * sigma**2 * lambdas[mask]
            pdf[mask] = numerator / (denominator + self.eps)

        return pdf

    def clean_correlation_matrix_rmt(self, corr_matrix: np.ndarray,
                                      n_samples: int) -> tuple:
        """
        Ação: Zere todos os autovalores dentro dos limites [λ₋, λ₊] (ruído) e reconstrua
        a "Matriz de Correlação Limpa" (C_clean). Isso elimina a "espuma" do mercado e
        deixa apenas as conexões estruturais reais.
        """
        n_assets = corr_matrix.shape[0]

        # Calcular autovalores e autovetores
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

        # Ordenar em ordem decrescente
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Bounds de Marchenko-Pastur
        lambda_minus, lambda_plus = self.marchenko_pastur_bounds(n_samples, n_assets)

        # Identificar autovalores de ruído vs sinal
        noise_mask = (eigenvalues >= lambda_minus) & (eigenvalues <= lambda_plus)
        signal_mask = ~noise_mask

        # Autovalores limpos: zerar os que são ruído (ou substituir pela média)
        eigenvalues_clean = eigenvalues.copy()

        # Substituir ruído pelo valor médio esperado (preserva traço)
        if np.any(noise_mask):
            mean_noise = np.mean(eigenvalues[noise_mask])
            eigenvalues_clean[noise_mask] = mean_noise

        # Reconstruir matriz limpa: C_clean = V @ Λ_clean @ V^T
        corr_clean = eigenvectors @ np.diag(eigenvalues_clean) @ eigenvectors.T

        # Garantir correlações válidas [-1, 1]
        corr_clean = np.clip(corr_clean, -1, 1)
        np.fill_diagonal(corr_clean, 1.0)

        return {
            'correlation_clean': corr_clean,
            'correlation_raw': corr_matrix,
            'eigenvalues_raw': eigenvalues,
            'eigenvalues_clean': eigenvalues_clean,
            'eigenvectors': eigenvectors,
            'lambda_minus': lambda_minus,
            'lambda_plus': lambda_plus,
            'noise_mask': noise_mask,
            'signal_mask': signal_mask,
            'n_signal_components': np.sum(signal_mask),
            'n_noise_components': np.sum(noise_mask)
        }

    # =========================================================================
    # MÓDULO 3: Teoria Espectral de Grafos (O Cálculo Pesado)
    # =========================================================================

    def correlation_to_adjacency(self, corr_clean: np.ndarray,
                                  threshold: float = 0.0) -> np.ndarray:
        """
        Transforme C_clean em uma Matriz de Adjacência.
        Usamos correlações absolutas como pesos das arestas.
        """
        # Usar valor absoluto da correlação como peso
        adjacency = np.abs(corr_clean.copy())

        # Zerar diagonal (sem self-loops)
        np.fill_diagonal(adjacency, 0)

        # Opcional: aplicar threshold
        if threshold > 0:
            adjacency[adjacency < threshold] = 0

        return adjacency

    def calculate_laplacian_matrix(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Calcule a Matriz Laplaciana (L):

        L = D - A

        (Onde D é a matriz de graus e A é a adjacência baseada na correlação limpa).
        """
        # Matriz de graus (diagonal com soma das linhas)
        degrees = np.sum(adjacency, axis=1)
        D = np.diag(degrees)

        # Laplaciana
        L = D - adjacency

        return L

    def calculate_normalized_laplacian(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Laplaciana normalizada para melhor estabilidade numérica.
        L_norm = I - D^(-1/2) @ A @ D^(-1/2)
        """
        degrees = np.sum(adjacency, axis=1)

        # Evitar divisão por zero
        degrees = np.maximum(degrees, self.eps)

        # D^(-1/2)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))

        # Laplaciana normalizada
        n = adjacency.shape[0]
        L_norm = np.eye(n) - D_inv_sqrt @ adjacency @ D_inv_sqrt

        return L_norm

    def calculate_fiedler_value(self, laplacian: np.ndarray,
                                 use_lanczos: bool = None) -> dict:
        """
        O Sinal: Calcule os autovalores de L. Focaremos no Autovalor de Fiedler (λ₂).

        - O λ₂ mede a "Conectividade Algébrica" do mercado.
        - λ₂ ≈ 0: O mercado está fragmentado. Movimentos no EURUSD são idiossincráticos e fracos.
        - λ₂ sobe rapidamente: O mercado está se "apertando". Todos os ativos começam a se
          mover em uníssono. Isso precede um evento de liquidez massiva.

        Dica do Engenheiro: Use o algoritmo de Lanczos para encontrar apenas os top-k
        autovalores (já que só nos importamos com λ₂ e λ_max).
        """
        if use_lanczos is None:
            use_lanczos = self.use_lanczos

        n = laplacian.shape[0]

        try:
            if use_lanczos and n > 5:
                # Algoritmo de Lanczos para os menores autovalores
                # k deve ser menor que n-1
                k = min(self.n_top_eigenvalues, n - 2)

                # Usar sparse eigensolver para menores autovalores
                # which='SM' = Smallest Magnitude
                eigenvalues, eigenvectors = eigsh(laplacian.astype(float),
                                                   k=k, which='SM',
                                                   maxiter=1000)

                # Ordenar
                idx = np.argsort(eigenvalues)
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
            else:
                # Decomposição completa
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
                idx = np.argsort(eigenvalues)
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]

        except (ArpackNoConvergence, np.linalg.LinAlgError):
            # Fallback para decomposição completa
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        # λ₁ deve ser ≈ 0 (conectividade do grafo)
        # λ₂ é o Fiedler value (conectividade algébrica)
        lambda_1 = eigenvalues[0] if len(eigenvalues) > 0 else 0
        lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0  # Fiedler
        lambda_max = eigenvalues[-1] if len(eigenvalues) > 0 else 0

        # Autovetor de Fiedler
        fiedler_vector = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else np.zeros(n)

        # Autovetor principal (para λ_max)
        principal_vector = eigenvectors[:, -1] if eigenvectors.shape[1] > 0 else np.zeros(n)

        return {
            'fiedler_value': lambda_2,
            'lambda_1': lambda_1,
            'lambda_max': lambda_max,
            'eigenvalues': eigenvalues,
            'fiedler_vector': fiedler_vector,
            'principal_vector': principal_vector,
            'eigenvectors': eigenvectors
        }

    def calculate_eigenvector_centrality(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Calcular o Autovetor Principal (Eigenvector Centrality) associado ao maior
        autovalor (λ_max).

        Este vetor indica a "importância" de cada ativo na rede.
        """
        # Autovalores e autovetores da matriz de adjacência
        eigenvalues, eigenvectors = np.linalg.eigh(adjacency)

        # Autovetor associado ao maior autovalor
        max_idx = np.argmax(eigenvalues)
        centrality = np.abs(eigenvectors[:, max_idx])

        # Normalizar
        centrality = centrality / (np.sum(centrality) + self.eps)

        return centrality

    # =========================================================================
    # MÓDULO 4: Entropia de Von Neumann (A Termodinâmica do Grafo)
    # =========================================================================

    def calculate_von_neumann_entropy(self, eigenvalues: np.ndarray) -> float:
        """
        Para confirmar que a volatilidade é explorável e não apenas pânico,
        calcularemos a entropia quântica da matriz de densidade (ρ):

        S(ρ) = -tr(ρ ln ρ) = -Σ λᵢ ln λᵢ

        (Onde os λᵢ são os autovalores normalizados da matriz de covariância).

        Lógica de Gatilho:
        - Alta Entropia = Mercado diversificado (Normal).
        - Queda Súbita na Entropia: Colapso de diversificação. Todos os players estão
          correndo para o mesmo lado (ex: Dólar forte). Isso gera uma Singularidade de
          Volatilidade.
        """
        # Normalizar autovalores para formar distribuição de probabilidade
        # (matriz de densidade tem traço = 1)
        eigenvalues_pos = np.maximum(eigenvalues, self.eps)
        eigenvalues_norm = eigenvalues_pos / (np.sum(eigenvalues_pos) + self.eps)

        # Entropia de Von Neumann: S = -Σ λᵢ ln(λᵢ)
        entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + self.eps))

        # Normalizar pela entropia máxima (log(n))
        max_entropy = np.log(len(eigenvalues))
        entropy_normalized = entropy / (max_entropy + self.eps)

        return entropy_normalized

    def detect_entropy_collapse(self, entropy_history: list,
                                 current_entropy: float) -> dict:
        """
        Detecta queda súbita na entropia (colapso de diversificação).
        """
        if len(entropy_history) < 10:
            return {
                'collapse_detected': False,
                'entropy_change': 0.0,
                'entropy_zscore': 0.0
            }

        recent_entropy = np.array(entropy_history[-20:])
        mean_entropy = np.mean(recent_entropy)
        std_entropy = np.std(recent_entropy) + self.eps

        # Mudança relativa
        entropy_change = (current_entropy - mean_entropy) / mean_entropy

        # Z-score
        entropy_zscore = (current_entropy - mean_entropy) / std_entropy

        # Detectar colapso
        collapse_detected = (
            entropy_change < -self.entropy_drop_threshold or
            current_entropy < self.entropy_critical_threshold
        )

        return {
            'collapse_detected': collapse_detected,
            'entropy_change': entropy_change,
            'entropy_zscore': entropy_zscore,
            'mean_entropy': mean_entropy,
            'std_entropy': std_entropy
        }

    # =========================================================================
    # MÓDULO 5: O Algoritmo de Decisão (O Sniper Espectral)
    # =========================================================================

    def calculate_fiedler_percentile(self, fiedler_history: list,
                                      current_fiedler: float) -> float:
        """
        Calcula o percentil do Fiedler value atual em relação ao histórico.
        """
        if len(fiedler_history) < 10:
            return 50.0

        history = np.array(fiedler_history)
        percentile = (np.sum(history < current_fiedler) / len(history)) * 100

        return percentile

    def determine_direction(self, principal_vector: np.ndarray,
                            returns: np.ndarray) -> dict:
        """
        Direção:
        - Calcular o Autovetor Principal (Eigenvector Centrality) associado ao maior
          autovalor (λ_max).
        - Se o componente do EURUSD no autovetor principal tiver o mesmo sinal que o
          componente do DXY -> Anomalia (Venda Falsa/Correção).
        - Se tiverem sinais opostos (correlação negativa padrão amplificada) -> Tendência Real.
        - Executar na direção do momentum imediato do EURUSD assim que o colapso de
          entropia for detectado.
        """
        # Índices dos ativos (assumindo ordem padrão)
        try:
            eurusd_idx = self.asset_names.index('EURUSD')
            dxy_idx = self.asset_names.index('DXY')
        except ValueError:
            eurusd_idx = 0
            dxy_idx = 1

        # Componentes do autovetor principal
        eurusd_component = principal_vector[eurusd_idx]
        dxy_component = principal_vector[dxy_idx]

        # Verificar sinais
        same_sign = np.sign(eurusd_component) == np.sign(dxy_component)

        if same_sign:
            pattern = "ANOMALIA"
            pattern_description = "Correlação anômala EURUSD-DXY (mesmo sinal)"
            is_real_trend = False
        else:
            pattern = "TENDENCIA_REAL"
            pattern_description = "Correlação negativa amplificada (sinais opostos)"
            is_real_trend = True

        # Momentum do EURUSD
        if returns.shape[0] > 0 and returns.shape[1] > eurusd_idx:
            eurusd_momentum = np.mean(returns[-5:, eurusd_idx]) if returns.shape[0] >= 5 else returns[-1, eurusd_idx]
        else:
            eurusd_momentum = 0.0

        # Direção do trade
        if eurusd_momentum > 0:
            direction = 1  # LONG
            direction_name = "LONG"
        elif eurusd_momentum < 0:
            direction = -1  # SHORT
            direction_name = "SHORT"
        else:
            direction = 0
            direction_name = "NEUTRO"

        return {
            'eurusd_component': eurusd_component,
            'dxy_component': dxy_component,
            'same_sign': same_sign,
            'pattern': pattern,
            'pattern_description': pattern_description,
            'is_real_trend': is_real_trend,
            'eurusd_momentum': eurusd_momentum,
            'direction': direction,
            'direction_name': direction_name,
            'principal_vector': principal_vector
        }

    def spectral_sniper_decision(self, fiedler_value: float,
                                  fiedler_percentile: float,
                                  entropy_result: dict,
                                  direction_result: dict) -> dict:
        """
        O indicador cruza a informação da rede com a direção do fluxo local.

        Pré-requisito: λ₂ (Fiedler) deve estar acima do percentil 90 histórico
        (O mercado está hiper-conectado).

        Gatilho: A Entropia de Von Neumann cruza abaixo de um limiar crítico
        (Colapso de complexidade).
        """
        # Pré-requisito: Fiedler acima do percentil 90
        fiedler_condition = fiedler_percentile >= self.fiedler_percentile_threshold

        # Gatilho: Colapso de entropia
        entropy_collapse = entropy_result['collapse_detected']

        # Sinal global de sincronização
        global_sync = fiedler_condition and entropy_collapse

        # Trade signal
        if global_sync:
            if direction_result['is_real_trend']:
                signal = direction_result['direction']
                signal_name = f"{direction_result['direction_name']} (Tendência Real)"
                confidence = 0.9
            else:
                # Anomalia - operar com cautela ou contrário
                signal = 0
                signal_name = "AGUARDAR (Anomalia detectada)"
                confidence = 0.3
        else:
            signal = 0
            signal_name = "SEM SINAL"
            confidence = 0.0

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'global_sync': global_sync,
            'fiedler_condition': fiedler_condition,
            'entropy_collapse': entropy_collapse,
            'fiedler_value': fiedler_value,
            'fiedler_percentile': fiedler_percentile,
            'pattern': direction_result['pattern'],
            'is_real_trend': direction_result['is_real_trend']
        }

    # =========================================================================
    # MÓDULO 6: Output e Visualização
    # =========================================================================

    def analyze(self, data: dict or pd.DataFrame) -> dict:
        """
        Execução completa do Sincronizador Espectral.

        Parâmetros:
        -----------
        data : dict ou DataFrame
            Dados de preços/retornos para múltiplos ativos.
            Se dict: {'EURUSD': [...], 'DXY': [...], ...}
            Se DataFrame: colunas são ativos, linhas são tempo
        """
        # Preparar dados
        if isinstance(data, dict):
            returns_df = self.prepare_multivariate_data(data)
        elif isinstance(data, pd.DataFrame):
            # Assumir que já são retornos ou converter
            if data.shape[0] > 1:
                # Verificar se parece preço (valores > 0.1 típicos) ou retornos
                if data.iloc[0, 0] > 0.1:
                    returns_df = np.log(data).diff().dropna()
                else:
                    returns_df = data
            else:
                returns_df = data
        else:
            raise ValueError("data deve ser dict ou DataFrame")

        returns = returns_df.values
        n_samples, n_assets = returns.shape

        if n_samples < self.correlation_window:
            raise ValueError(f"Dados insuficientes. Necessário mínimo de {self.correlation_window} amostras.")

        # Usar janela móvel
        returns_window = returns[-self.correlation_window:]

        # 1. Calcular Matriz de Correlação
        corr_matrix = self.calculate_correlation_matrix(returns_window)

        # 2. Limpar com RMT
        rmt_result = self.clean_correlation_matrix_rmt(corr_matrix, self.correlation_window)

        # 3. Converter para grafo
        adjacency = self.correlation_to_adjacency(rmt_result['correlation_clean'])

        # 4. Calcular Laplaciana
        laplacian = self.calculate_laplacian_matrix(adjacency)

        # 5. Calcular Fiedler value e autovalores
        spectral_result = self.calculate_fiedler_value(laplacian)

        # 6. Eigenvector Centrality
        centrality = self.calculate_eigenvector_centrality(adjacency)

        # 7. Entropia de Von Neumann
        # Usar autovalores da matriz de correlação limpa
        entropy = self.calculate_von_neumann_entropy(rmt_result['eigenvalues_clean'])

        # Atualizar histórico
        self._cache['fiedler_history'].append(spectral_result['fiedler_value'])
        self._cache['entropy_history'].append(entropy)

        # Limitar tamanho do histórico
        max_history = 500
        if len(self._cache['fiedler_history']) > max_history:
            self._cache['fiedler_history'] = self._cache['fiedler_history'][-max_history:]
            self._cache['entropy_history'] = self._cache['entropy_history'][-max_history:]

        # 8. Detectar colapso de entropia
        entropy_result = self.detect_entropy_collapse(
            self._cache['entropy_history'][:-1], entropy
        )

        # 9. Percentil do Fiedler
        fiedler_percentile = self.calculate_fiedler_percentile(
            self._cache['fiedler_history'][:-1], spectral_result['fiedler_value']
        )

        # 10. Determinar direção
        direction_result = self.determine_direction(
            spectral_result['principal_vector'], returns
        )

        # 11. Decisão final
        decision = self.spectral_sniper_decision(
            spectral_result['fiedler_value'],
            fiedler_percentile,
            entropy_result,
            direction_result
        )

        return {
            # Decisão principal
            'signal': decision['signal'],
            'signal_name': decision['signal_name'],
            'confidence': decision['confidence'],
            'global_sync': decision['global_sync'],

            # Fiedler
            'fiedler_value': spectral_result['fiedler_value'],
            'fiedler_percentile': fiedler_percentile,
            'fiedler_condition': decision['fiedler_condition'],

            # Entropia
            'entropy': entropy,
            'entropy_collapse': entropy_result['collapse_detected'],
            'entropy_change': entropy_result['entropy_change'],

            # Padrão
            'pattern': direction_result['pattern'],
            'is_real_trend': direction_result['is_real_trend'],
            'direction': direction_result['direction_name'],

            # Detalhes espectrais
            'eigenvalues_raw': rmt_result['eigenvalues_raw'],
            'eigenvalues_clean': rmt_result['eigenvalues_clean'],
            'lambda_plus': rmt_result['lambda_plus'],
            'lambda_minus': rmt_result['lambda_minus'],
            'n_signal_components': rmt_result['n_signal_components'],

            # Centrality
            'eigenvector_centrality': dict(zip(self.asset_names, centrality)),

            # Matrizes
            'correlation_raw': corr_matrix,
            'correlation_clean': rmt_result['correlation_clean'],
            'adjacency': adjacency,
            'laplacian': laplacian,

            # Histórico
            'fiedler_history': self._cache['fiedler_history'].copy(),
            'entropy_history': self._cache['entropy_history'].copy(),

            # Metadados
            'n_samples': n_samples,
            'n_assets': n_assets,
            'asset_names': self.asset_names
        }

    def get_signal(self, data: dict or pd.DataFrame) -> int:
        """
        Retorna sinal simplificado:
        1 = LONG
        0 = NEUTRO
        -1 = SHORT
        """
        result = self.analyze(data)
        return result['signal']


# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# =============================================================================

def plot_sema_analysis(data: dict or pd.DataFrame, save_path: str = None):
    """
    Output e Visualização:

    - Plot 1: O "Espectro de Autovalores" em tempo real (Heatmap das eigen-frequencies).
    - Plot 2: Linha do Valor de Fiedler (λ₂). Quando cruzar o limiar vermelho, sinal de alerta.
    - Ação no Gráfico: Pintar o fundo do gráfico de preço de Roxo (cor espectral) quando o
      sistema detectar "Sincronização Global". É o momento de alavancagem máxima.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    # Criar indicador e analisar
    sema = SincronizadorEspectral()
    result = sema.analyze(data)

    # Preparar dados de preço (EURUSD)
    if isinstance(data, dict):
        eurusd_prices = np.array(data.get('EURUSD', data[list(data.keys())[0]]))
    else:
        eurusd_prices = data.iloc[:, 0].values

    # Criar figura
    fig = plt.figure(figsize=(16, 14))

    # Layout: 2x2 grid + 1 row
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1.5])

    # =========================================================================
    # Plot 1: Preço EURUSD com indicação de Sincronização Global
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :])

    time = np.arange(len(eurusd_prices))
    ax1.plot(time, eurusd_prices, 'b-', linewidth=1.5, label='EURUSD')

    # Pintar fundo roxo se sincronização global detectada
    if result['global_sync']:
        ax1.axvspan(time[-50], time[-1], alpha=0.3, color='purple',
                   label='Sincronização Global')
        ax1.scatter([time[-1]], [eurusd_prices[-1]], c='purple', s=300,
                   marker='*', zorder=5, label='SYNC DETECTADA')

    # Sinal
    signal_color = {'LONG': 'green', 'SHORT': 'red', 'NEUTRO': 'gray'}
    direction = result['direction']

    if result['signal'] != 0:
        marker = '^' if result['signal'] == 1 else 'v'
        ax1.scatter([time[-1]], [eurusd_prices[-1]],
                   c=signal_color.get(direction, 'gray'),
                   s=200, marker=marker, zorder=6,
                   label=f'Sinal: {result["signal_name"]}')

    # Info box
    info_text = (
        f"Fiedler (λ₂): {result['fiedler_value']:.4f}\n"
        f"Percentil: {result['fiedler_percentile']:.1f}%\n"
        f"Entropia: {result['entropy']:.4f}\n"
        f"Padrão: {result['pattern']}"
    )
    ax1.text(0.02, 0.95, info_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_title('Sincronizador Espectral (SEMA) - Análise de Rede de Mercado', fontsize=14)
    ax1.set_ylabel('Preço EURUSD')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 2: Espectro de Autovalores (Heatmap)
    # =========================================================================
    ax2 = fig.add_subplot(gs[1, 0])

    eigenvalues_raw = result['eigenvalues_raw']
    eigenvalues_clean = result['eigenvalues_clean']

    # Criar heatmap de autovalores
    eigen_matrix = np.vstack([eigenvalues_raw, eigenvalues_clean])

    im = ax2.imshow(eigen_matrix, aspect='auto', cmap='hot',
                    extent=[0, len(eigenvalues_raw), 0, 2])

    # Marcar bounds de Marchenko-Pastur
    lambda_minus = result['lambda_minus']
    lambda_plus = result['lambda_plus']

    ax2.set_yticks([0.5, 1.5])
    ax2.set_yticklabels(['Limpos', 'Raw'])
    ax2.set_xlabel('Índice do Autovalor')
    ax2.set_title('Espectro de Autovalores (RMT Filtered)', fontsize=12)
    plt.colorbar(im, ax=ax2, label='Valor')

    # Adicionar texto com bounds
    ax2.text(0.98, 0.95, f'λ₋={lambda_minus:.3f}\nλ₊={lambda_plus:.3f}',
            transform=ax2.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # =========================================================================
    # Plot 3: Matriz de Correlação Limpa (Heatmap)
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 1])

    corr_clean = result['correlation_clean']

    im3 = ax3.imshow(corr_clean, cmap='RdBu_r', vmin=-1, vmax=1)

    # Labels
    ax3.set_xticks(range(len(result['asset_names'])))
    ax3.set_yticks(range(len(result['asset_names'])))
    ax3.set_xticklabels(result['asset_names'], rotation=45, ha='right')
    ax3.set_yticklabels(result['asset_names'])
    ax3.set_title('Matriz de Correlação Limpa (C_clean)', fontsize=12)
    plt.colorbar(im3, ax=ax3, label='Correlação')

    # Adicionar valores
    for i in range(len(result['asset_names'])):
        for j in range(len(result['asset_names'])):
            ax3.text(j, i, f'{corr_clean[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if abs(corr_clean[i, j]) > 0.5 else 'black')

    # =========================================================================
    # Plot 4: Fiedler Value (λ₂) - Oscilador
    # =========================================================================
    ax4 = fig.add_subplot(gs[2, 0])

    fiedler_history = result['fiedler_history']
    time_fiedler = np.arange(len(fiedler_history))

    ax4.plot(time_fiedler, fiedler_history, 'purple', linewidth=1.5, label='λ₂ (Fiedler)')

    # Threshold (percentil 90 histórico)
    if len(fiedler_history) > 10:
        threshold_90 = np.percentile(fiedler_history, 90)
        ax4.axhline(y=threshold_90, color='red', linestyle='--',
                   label=f'Threshold P90: {threshold_90:.4f}')

    # Marcar valor atual
    ax4.scatter([time_fiedler[-1]], [fiedler_history[-1]], c='purple', s=100, zorder=5)

    # Colorir fundo quando acima do threshold
    if result['fiedler_condition']:
        ax4.axvspan(time_fiedler[-1] - 5, time_fiedler[-1], alpha=0.3, color='red',
                   label='Hiper-conectado!')

    ax4.set_ylabel('Fiedler Value (λ₂)')
    ax4.set_xlabel('Tempo')
    ax4.set_title('Conectividade Algébrica do Mercado', fontsize=12)
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 5: Entropia de Von Neumann
    # =========================================================================
    ax5 = fig.add_subplot(gs[2, 1])

    entropy_history = result['entropy_history']
    time_entropy = np.arange(len(entropy_history))

    ax5.plot(time_entropy, entropy_history, 'green', linewidth=1.5,
            label='Entropia de Von Neumann')

    # Threshold crítico
    ax5.axhline(y=sema.entropy_critical_threshold, color='red', linestyle='--',
               label=f'Threshold Crítico: {sema.entropy_critical_threshold}')

    # Marcar valor atual
    ax5.scatter([time_entropy[-1]], [entropy_history[-1]], c='green', s=100, zorder=5)

    # Colorir fundo quando colapso detectado
    if result['entropy_collapse']:
        ax5.axvspan(time_entropy[-1] - 5, time_entropy[-1], alpha=0.3, color='orange',
                   label='Colapso de Entropia!')

    ax5.set_ylabel('Entropia Normalizada')
    ax5.set_xlabel('Tempo')
    ax5.set_title('Entropia de Von Neumann (Diversificação)', fontsize=12)
    ax5.legend(loc='upper left', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Interpretação
    interpretation = "Alta Entropia = Mercado Diversificado" if result['entropy'] > 0.7 else \
                    "ALERTA: Baixa Entropia = Colapso de Diversificação" if result['entropy_collapse'] else \
                    "Entropia Moderada"
    ax5.text(0.98, 0.05, interpretation, transform=ax5.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # =========================================================================
    # Resumo Final
    # =========================================================================
    summary = (
        f"SEMA | Fiedler: {result['fiedler_value']:.4f} (P{result['fiedler_percentile']:.0f}) | "
        f"Entropia: {result['entropy']:.4f} | "
        f"Sync Global: {'SIM' if result['global_sync'] else 'NAO'} | "
        f"Padrão: {result['pattern']} | "
        f"Sinal: {result['signal_name']}"
    )

    fig.text(0.5, 0.01, summary, fontsize=11, ha='center',
            bbox=dict(boxstyle='round',
                     facecolor='purple' if result['global_sync'] else 'lightblue',
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
    print("SINCRONIZADOR ESPECTRAL (SEMA)")
    print("Gestão de Risco Sistêmico / Engenharia de Redes Complexas")
    print("=" * 80)

    # Gerar dados simulados para múltiplos ativos
    np.random.seed(42)
    n_points = 500

    # Simular correlações realistas entre ativos
    # EURUSD e DXY são tipicamente negativamente correlacionados
    # XAUUSD (ouro) tem correlação inversa com dólar
    # US10Y e SPX500 têm suas próprias dinâmicas

    # Fator comum (mercado)
    market_factor = np.cumsum(np.random.randn(n_points) * 0.001)

    # Fator de dólar
    dollar_factor = np.cumsum(np.random.randn(n_points) * 0.0008)

    # Fator de risco
    risk_factor = np.cumsum(np.random.randn(n_points) * 0.0005)

    # Gerar preços
    eurusd = 1.1000 + market_factor - 0.5 * dollar_factor + np.cumsum(np.random.randn(n_points) * 0.0003)
    dxy = 100 + 0.8 * dollar_factor + np.cumsum(np.random.randn(n_points) * 0.05)
    xauusd = 1900 - 0.3 * dollar_factor + 0.2 * risk_factor + np.cumsum(np.random.randn(n_points) * 2)
    us10y = 4.0 + 0.1 * market_factor + np.cumsum(np.random.randn(n_points) * 0.01)
    spx500 = 4500 + 100 * market_factor + 50 * risk_factor + np.cumsum(np.random.randn(n_points) * 5)

    # Criar evento de sincronização (últimos 50 pontos)
    sync_start = n_points - 50
    sync_factor = np.cumsum(np.random.randn(50) * 0.002)
    eurusd[sync_start:] += -sync_factor * 0.5
    dxy[sync_start:] += sync_factor * 50
    xauusd[sync_start:] += -sync_factor * 100
    us10y[sync_start:] += sync_factor * 0.5
    spx500[sync_start:] += -sync_factor * 200

    # Dicionário de dados
    data = {
        'EURUSD': eurusd,
        'DXY': dxy,
        'XAUUSD': xauusd,
        'US10Y': us10y,
        'SPX500': spx500
    }

    print(f"\nDados simulados: {n_points} pontos")
    print(f"Ativos: {list(data.keys())}")
    print(f"\nPreços finais:")
    for asset, prices in data.items():
        print(f"  {asset}: {prices[-1]:.4f}")

    # Criar indicador
    sema = SincronizadorEspectral(
        correlation_window=120,
        fiedler_percentile_threshold=90,
        entropy_drop_threshold=0.3,
        entropy_critical_threshold=0.5,
        use_lanczos=True
    )

    # Executar análise múltiplas vezes para construir histórico
    print("\n" + "-" * 40)
    print("Construindo histórico de análise...")
    print("-" * 40)

    for i in range(50, n_points, 10):
        subset_data = {k: v[:i] for k, v in data.items()}
        try:
            _ = sema.analyze(subset_data)
        except:
            pass

    # Análise final
    print("\n" + "-" * 40)
    print("Executando análise SEMA final...")
    print("-" * 40)

    result = sema.analyze(data)

    # Mostrar resultados
    print("\nRESULTADO PRINCIPAL:")
    print(f"   Sinal: {result['signal_name']}")
    print(f"   Confiança: {result['confidence']:.2%}")
    print(f"   Sincronização Global: {'SIM' if result['global_sync'] else 'NAO'}")

    print("\nCONECTIVIDADE (Fiedler λ₂):")
    print(f"   Fiedler Value: {result['fiedler_value']:.6f}")
    print(f"   Percentil Histórico: {result['fiedler_percentile']:.1f}%")
    print(f"   Condição (P90): {'SIM Acima' if result['fiedler_condition'] else 'NAO Abaixo'}")

    print("\nENTROPIA DE VON NEUMANN:")
    print(f"   Entropia: {result['entropy']:.4f}")
    print(f"   Colapso Detectado: {'SIM' if result['entropy_collapse'] else 'NAO'}")
    print(f"   Mudança: {result['entropy_change']:.2%}")

    print("\nPADRÃO DETECTADO:")
    print(f"   Tipo: {result['pattern']}")
    print(f"   Tendência Real: {'SIM' if result['is_real_trend'] else 'NAO (Anomalia)'}")
    print(f"   Direção: {result['direction']}")

    print("\nEIGENVECTOR CENTRALITY:")
    for asset, centrality in result['eigenvector_centrality'].items():
        print(f"   {asset}: {centrality:.4f}")

    print("\nFILTRAGEM RMT:")
    print(f"   λ₋ (Marchenko-Pastur): {result['lambda_minus']:.4f}")
    print(f"   λ₊ (Marchenko-Pastur): {result['lambda_plus']:.4f}")
    print(f"   Componentes de Sinal: {result['n_signal_components']}")

    print("\n" + "=" * 80)
    if result['global_sync']:
        print("SINCRONIZAÇÃO GLOBAL DETECTADA!")
        print("   Todos os ativos estão se movendo em uníssono.")
        print("   Momento de alavancagem máxima.")
        if result['is_real_trend']:
            print(f"   Direção recomendada: {result['direction']}")
        else:
            print("   ATENÇÃO: Padrão anômalo - proceder com cautela.")
    else:
        print("Mercado fragmentado - Sem sincronização global.")
        if result['fiedler_condition']:
            print("   Mercado hiper-conectado, aguardando colapso de entropia.")
        else:
            print("   Conectividade ainda baixa.")
    print("=" * 80)

    # Gerar visualização
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print("\nGerando visualização...")
        fig = plot_sema_analysis(data, save_path='sema_analysis.png')
        print("Visualização salva como 'sema_analysis.png'")
        plt.close()
    except Exception as e:
        print(f"\nNão foi possível gerar visualização: {e}")
        import traceback
        traceback.print_exc()
