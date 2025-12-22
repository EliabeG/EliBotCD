"""
Detector de Tunelamento Topológico (DTT)
=========================================
Nível de Complexidade: Experimental / Deep Quant

Premissa Teórica: Mercados laterais são topologicamente "contráteis" (sem características
interessantes, Betti numbers = 0). Mercados de alta volatilidade criam estruturas geométricas
complexas (loops e voids). O indicador busca o momento exato em que a topologia do
mercado muda de um conjunto desconexo para uma estrutura com "ciclos persistentes" (Betti-1),
indicando uma armadilha de liquidez pronta para romper.

Dependências Críticas: gudhi ou ripser, scikit-learn, numpy, scipy.stats
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from scipy.linalg import eigh_tridiagonal
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Dependências para TDA (Topological Data Analysis)
try:
    import gudhi
    TDA_BACKEND = 'gudhi'
except ImportError:
    try:
        import ripser
        TDA_BACKEND = 'ripser'
    except ImportError:
        print("AVISO: Instale gudhi ou ripser para Homologia Persistente")
        print("pip install gudhi  # Recomendado (C++ backend)")
        print("ou: pip install ripser")
        TDA_BACKEND = None


class DetectorTunelamentoTopologico:
    """
    Implementação completa do Detector de Tunelamento Topológico (DTT)

    Módulos:
    1. Pré-processamento: Embedding de Takens (Reconstrução do Espaço de Fase)
    2. O Motor: Homologia Persistente (Vietoris-Rips Filtration)
    3. O Potencial Quântico (Equação de Schrödinger Financeira)
    4. O Sintetizador de Decisão (Gatilho Lógico)
    5. Output e Visualização
    """

    def __init__(self,
                 embedding_dim: int = None,  # None = calcular automaticamente
                 time_delay: int = None,     # None = calcular automaticamente
                 max_embedding_dim: int = 7,
                 min_embedding_dim: int = 5,
                 max_points: int = 200,      # Limitar para performance
                 use_dimensionality_reduction: bool = True,
                 reduction_method: str = 'pca',  # 'pca' ou 'tsne'
                 reduction_components: int = 3,
                 persistence_entropy_threshold: float = 0.5,
                 tunneling_probability_threshold: float = 0.15,
                 hbar: float = 1.0,          # Constante de Planck reduzida (ajustável)
                 particle_mass: float = 1.0,  # Massa da partícula (ajustável)
                 n_eigenstates: int = 10,    # Número de autoestados a calcular
                 kde_bandwidth: str = 'scott'):
        """
        Inicialização do Detector de Tunelamento Topológico

        Parâmetros:
        -----------
        embedding_dim : int ou None
            Dimensão do embedding de Takens. Se None, calcula via False Nearest Neighbors.

        time_delay : int ou None
            Delay temporal para embedding. Se None, calcula via Average Mutual Information.

        max_points : int
            Número máximo de pontos para análise topológica (default: 200).
            Recomendação técnica: limitar a 100-200 candles.

        use_dimensionality_reduction : bool
            Se True, aplica redução de dimensionalidade antes da topologia.

        reduction_method : str
            'pca' ou 'tsne' para redução de dimensionalidade.

        persistence_entropy_threshold : float
            Limiar mínimo de entropia de persistência para sinal válido.

        tunneling_probability_threshold : float
            Limiar de probabilidade de tunelamento para disparo do sinal.

        hbar, particle_mass : float
            Parâmetros da equação de Schrödinger (ajustáveis para calibração).
        """
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.max_embedding_dim = max_embedding_dim
        self.min_embedding_dim = min_embedding_dim
        self.max_points = max_points
        self.use_dimensionality_reduction = use_dimensionality_reduction
        self.reduction_method = reduction_method
        self.reduction_components = reduction_components
        self.persistence_entropy_threshold = persistence_entropy_threshold
        self.tunneling_probability_threshold = tunneling_probability_threshold
        self.hbar = hbar
        self.particle_mass = particle_mass
        self.n_eigenstates = n_eigenstates
        self.kde_bandwidth = kde_bandwidth

        # Cache de resultados
        self._cache = {}

    # =========================================================================
    # MÓDULO 1: Pré-processamento - Embedding de Takens
    # =========================================================================

    def _calculate_log_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Não use o preço bruto (Close). O preço é apenas uma sombra de um sistema dinâmico.
        Usar retornos logarítmicos.
        """
        return np.diff(np.log(prices))

    def _calculate_ami(self, series: np.ndarray, max_lag: int = 50) -> int:
        """
        Time Delay (τ): Calcular usando o primeiro mínimo da Informação Mútua Média (AMI).

        A AMI mede a dependência não-linear entre a série e sua versão defasada.
        """
        n = len(series)
        if max_lag > n // 4:
            max_lag = n // 4

        ami_values = []

        for lag in range(1, max_lag + 1):
            # Discretizar as séries para cálculo de MI
            x = series[:-lag]
            y = series[lag:]

            # Número de bins (regra de Sturges)
            n_bins = int(1 + 3.322 * np.log10(len(x)))
            n_bins = max(n_bins, 10)

            # Histogramas marginais e conjuntos
            hist_x, _ = np.histogram(x, bins=n_bins, density=True)
            hist_y, _ = np.histogram(y, bins=n_bins, density=True)
            hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins, density=True)

            # Evitar log(0)
            hist_x = hist_x + 1e-10
            hist_y = hist_y + 1e-10
            hist_xy = hist_xy + 1e-10

            # Normalizar
            hist_x = hist_x / hist_x.sum()
            hist_y = hist_y / hist_y.sum()
            hist_xy = hist_xy / hist_xy.sum()

            # Entropias
            h_x = -np.sum(hist_x * np.log(hist_x))
            h_y = -np.sum(hist_y * np.log(hist_y))
            h_xy = -np.sum(hist_xy * np.log(hist_xy))

            # Informação Mútua
            mi = h_x + h_y - h_xy
            ami_values.append(mi)

        ami_values = np.array(ami_values)

        # Encontrar primeiro mínimo local
        for i in range(1, len(ami_values) - 1):
            if ami_values[i] < ami_values[i-1] and ami_values[i] < ami_values[i+1]:
                return i + 1  # lag começa em 1

        # Se não encontrar mínimo, usar o ponto de inflexão ou default
        return max(1, np.argmin(ami_values) + 1)

    def _calculate_fnn(self, series: np.ndarray, max_dim: int = 10,
                       delay: int = 1, rtol: float = 15.0) -> int:
        """
        Dimensão de Embedding (m): Calcular dinamicamente usando o método de
        False Nearest Neighbors (geralmente entre 5 e 7 para FX).
        """
        n = len(series)

        fnn_ratios = []

        for dim in range(1, max_dim + 1):
            # Verificar se temos pontos suficientes
            n_points = n - (dim + 1) * delay
            if n_points < 10:
                break

            # Criar embeddings para dim e dim+1
            embedded_d = self._create_embedding(series, dim, delay)
            embedded_d1 = self._create_embedding(series, dim + 1, delay)

            # Alinhar tamanhos
            min_len = min(len(embedded_d), len(embedded_d1))
            embedded_d = embedded_d[:min_len]
            embedded_d1 = embedded_d1[:min_len]

            if len(embedded_d) < 2:
                break

            # Encontrar vizinhos mais próximos em dim
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
            nbrs.fit(embedded_d)
            distances, indices = nbrs.kneighbors(embedded_d)

            # Contar falsos vizinhos
            n_false = 0
            n_total = 0

            for i in range(len(embedded_d)):
                j = indices[i, 1]  # Vizinho mais próximo (não ele mesmo)

                if j >= len(embedded_d1):
                    continue

                d_d = distances[i, 1]  # Distância em dim
                if d_d < 1e-10:
                    continue

                # Distância em dim+1
                d_d1 = np.linalg.norm(embedded_d1[i] - embedded_d1[j])

                # Verificar se é falso vizinho
                if (d_d1 / d_d) > rtol:
                    n_false += 1
                n_total += 1

            fnn_ratio = n_false / max(n_total, 1)
            fnn_ratios.append(fnn_ratio)

            # Se FNN < 1%, dimensão suficiente
            if fnn_ratio < 0.01:
                return max(dim, self.min_embedding_dim)

        # Se não convergir, usar dimensão com menor FNN
        if fnn_ratios:
            optimal_dim = np.argmin(fnn_ratios) + 1
            return max(optimal_dim, self.min_embedding_dim)

        return self.min_embedding_dim

    def _create_embedding(self, series: np.ndarray, dim: int, delay: int) -> np.ndarray:
        """
        Ação: Criar um "Time Delay Embedding" da série de retornos logarítmicos.

        Resultado: Uma nuvem de pontos X em R^m.
        """
        n = len(series)
        n_points = n - (dim - 1) * delay

        if n_points <= 0:
            raise ValueError(f"Série muito curta para embedding: n={n}, dim={dim}, delay={delay}")

        embedded = np.zeros((n_points, dim))

        for i in range(dim):
            start = i * delay
            end = start + n_points
            embedded[:, i] = series[start:end]

        return embedded

    def get_takens_embedding(self, prices: np.ndarray) -> dict:
        """
        Executa o Embedding de Takens completo com cálculo automático de parâmetros.
        """
        # Calcular retornos logarítmicos
        returns = self._calculate_log_returns(prices)

        # Calcular Time Delay se não especificado
        if self.time_delay is None:
            tau = self._calculate_ami(returns)
        else:
            tau = self.time_delay

        # Calcular Dimensão de Embedding se não especificada
        if self.embedding_dim is None:
            m = self._calculate_fnn(returns, max_dim=self.max_embedding_dim, delay=tau)
            m = min(max(m, self.min_embedding_dim), self.max_embedding_dim)
        else:
            m = self.embedding_dim

        # Criar embedding
        point_cloud = self._create_embedding(returns, m, tau)

        # Limitar número de pontos (recomendação técnica: 100-200 candles)
        if len(point_cloud) > self.max_points:
            point_cloud = point_cloud[-self.max_points:]

        # Aplicar redução de dimensionalidade se configurado
        point_cloud_reduced = None
        if self.use_dimensionality_reduction and m > self.reduction_components:
            if self.reduction_method == 'pca':
                reducer = PCA(n_components=self.reduction_components)
                point_cloud_reduced = reducer.fit_transform(point_cloud)
            elif self.reduction_method == 'tsne':
                # t-SNE para preservar estrutura local
                perplexity = min(30, len(point_cloud) // 4)
                reducer = TSNE(n_components=min(3, self.reduction_components),
                              perplexity=max(5, perplexity),
                              random_state=42)
                point_cloud_reduced = reducer.fit_transform(point_cloud)

        return {
            'point_cloud': point_cloud,
            'point_cloud_reduced': point_cloud_reduced if point_cloud_reduced is not None else point_cloud,
            'embedding_dim': m,
            'time_delay': tau,
            'n_points': len(point_cloud),
            'returns': returns
        }

    # =========================================================================
    # MÓDULO 2: O Motor - Homologia Persistente (Vietoris-Rips Filtration)
    # =========================================================================

    def _compute_persistence_gudhi(self, point_cloud: np.ndarray, max_dim: int = 1) -> dict:
        """
        Cálculo: Gerar o Persistence Diagram usando a filtração de Vietoris-Rips.
        Usando biblioteca GUDHI (C++ backend, mais eficiente).
        """
        # Criar complexo de Rips
        rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=np.inf)

        # Criar simplex tree com dimensão limitada
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim + 1)

        # Calcular persistência
        simplex_tree.compute_persistence()

        # Extrair pares de persistência por dimensão
        persistence_pairs = {}
        for dim in range(max_dim + 1):
            pairs = simplex_tree.persistence_intervals_in_dimension(dim)
            # Filtrar infinitos
            finite_pairs = pairs[np.isfinite(pairs[:, 1])] if len(pairs) > 0 else np.array([])
            persistence_pairs[dim] = finite_pairs

        return {
            'persistence_pairs': persistence_pairs,
            'simplex_tree': simplex_tree
        }

    def _compute_persistence_ripser(self, point_cloud: np.ndarray, max_dim: int = 1) -> dict:
        """
        Alternativa usando ripser.
        """
        result = ripser.ripser(point_cloud, maxdim=max_dim)

        persistence_pairs = {}
        for dim in range(max_dim + 1):
            pairs = result['dgms'][dim]
            # Filtrar infinitos
            finite_pairs = pairs[np.isfinite(pairs[:, 1])] if len(pairs) > 0 else np.array([])
            persistence_pairs[dim] = finite_pairs

        return {
            'persistence_pairs': persistence_pairs,
            'diagrams': result['dgms']
        }

    def _compute_persistence_fallback(self, point_cloud: np.ndarray, max_dim: int = 1) -> dict:
        """
        Fallback simplificado se gudhi/ripser não estiverem disponíveis.
        Usa apenas análise de distâncias para estimar estrutura.
        """
        from scipy.spatial.distance import pdist, squareform

        # Matriz de distâncias
        distances = squareform(pdist(point_cloud))

        # Simular persistência baseada em componentes conexas
        n = len(point_cloud)

        # Estimar "ciclos" baseado em distribuição de distâncias
        dist_flat = distances[np.triu_indices(n, k=1)]

        # Criar pares sintéticos baseados em percentis
        percentiles = np.percentile(dist_flat, [10, 25, 50, 75, 90])

        # Pares de dimensão 0 (componentes conexas)
        pairs_0 = np.array([[0, percentiles[i]] for i in range(len(percentiles))])

        # Pares de dimensão 1 (ciclos) - estimativa baseada em variância local
        pairs_1 = []
        n_samples = min(50, n // 2)
        for i in range(n_samples):
            idx = np.random.randint(0, n)
            local_dist = np.sort(distances[idx])[:10]
            if len(local_dist) > 2:
                birth = local_dist[1]
                death = local_dist[-1]
                if death > birth:
                    pairs_1.append([birth, death])

        pairs_1 = np.array(pairs_1) if pairs_1 else np.array([]).reshape(0, 2)

        return {
            'persistence_pairs': {0: pairs_0, 1: pairs_1},
            'estimated': True
        }

    def compute_persistent_homology(self, point_cloud: np.ndarray) -> dict:
        """
        Aqui está o gargalo computacional. Você deve construir complexos simpliciais
        sobre a nuvem de pontos para extrair a forma dos dados.

        Foco: Analisar os Números de Betti (B1). B1 representa buracos/ciclos 1-dimensionais.
        """
        # Escolher backend
        if TDA_BACKEND == 'gudhi':
            result = self._compute_persistence_gudhi(point_cloud, max_dim=1)
        elif TDA_BACKEND == 'ripser':
            result = self._compute_persistence_ripser(point_cloud, max_dim=1)
        else:
            result = self._compute_persistence_fallback(point_cloud, max_dim=1)

        persistence_pairs = result['persistence_pairs']

        # Calcular Números de Betti médios
        betti_0 = len(persistence_pairs.get(0, []))  # Componentes conexas
        betti_1 = len(persistence_pairs.get(1, []))  # Ciclos 1D (loops)

        # Calcular lifetimes (persistências)
        lifetimes_1 = []
        if len(persistence_pairs.get(1, [])) > 0:
            pairs_1 = persistence_pairs[1]
            lifetimes_1 = pairs_1[:, 1] - pairs_1[:, 0]

        return {
            'persistence_pairs': persistence_pairs,
            'betti_0': betti_0,
            'betti_1': betti_1,
            'lifetimes_dim1': np.array(lifetimes_1),
            'raw_result': result
        }

    def calculate_persistence_entropy(self, persistence_result: dict) -> dict:
        """
        Filtro: Calcular a "Entropia da Persistência" da dimensão 1 (H_pers).
        Se H_pers for baixo, ignore (mercado trivial).

        Lógica de Alta Volatilidade:
        - Em mercado calmo (ruído), os ciclos nascem e morrem muito rápido (baixa persistência).
        - Em preparação para um movimento explosivo, a estrutura de correlação entre os lags
          cria "loops" duradouros no espaço de fase.
        """
        lifetimes = persistence_result['lifetimes_dim1']

        if len(lifetimes) == 0:
            return {
                'persistence_entropy': 0.0,
                'mean_lifetime': 0.0,
                'max_lifetime': 0.0,
                'n_significant_cycles': 0,
                'is_trivial_market': True
            }

        # Normalizar lifetimes para distribuição de probabilidade
        total_lifetime = np.sum(lifetimes)
        if total_lifetime < 1e-10:
            return {
                'persistence_entropy': 0.0,
                'mean_lifetime': 0.0,
                'max_lifetime': 0.0,
                'n_significant_cycles': 0,
                'is_trivial_market': True
            }

        probs = lifetimes / total_lifetime

        # Entropia de Shannon normalizada
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(lifetimes)) if len(lifetimes) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Estatísticas dos ciclos
        mean_lifetime = np.mean(lifetimes)
        max_lifetime = np.max(lifetimes)

        # Ciclos significativos (lifetime > média)
        n_significant = np.sum(lifetimes > mean_lifetime)

        # Mercado trivial se entropia muito baixa
        is_trivial = normalized_entropy < self.persistence_entropy_threshold

        return {
            'persistence_entropy': normalized_entropy,
            'raw_entropy': entropy,
            'mean_lifetime': mean_lifetime,
            'max_lifetime': max_lifetime,
            'n_significant_cycles': n_significant,
            'n_total_cycles': len(lifetimes),
            'is_trivial_market': is_trivial
        }

    # =========================================================================
    # MÓDULO 3: O Potencial Quântico (Equação de Schrödinger Financeira)
    # =========================================================================

    def estimate_potential(self, prices: np.ndarray, n_points: int = 100) -> dict:
        """
        Estime V(x) assumindo que é o inverso da densidade de probabilidade histórica recente
        (Kernel Density Estimation - KDE). Onde há muito volume (consolidação),
        o potencial é baixo (fundo do poço).

        Trate o preço x como uma partícula em um poço de potencial V(x).
        """
        # Usar preços recentes
        recent_prices = prices[-min(len(prices), 500):]

        # Criar grid de preços
        price_min = np.min(recent_prices) * 0.995
        price_max = np.max(recent_prices) * 1.005
        x_grid = np.linspace(price_min, price_max, n_points)

        # Estimar densidade via KDE
        kde = stats.gaussian_kde(recent_prices, bw_method=self.kde_bandwidth)
        density = kde(x_grid)

        # Potencial = inverso da densidade (normalizado)
        # Onde densidade é alta (consolidação), potencial é baixo
        density_normalized = density / (np.max(density) + 1e-10)
        potential = 1.0 / (density_normalized + 0.01)  # Evitar divisão por zero

        # Normalizar potencial
        potential = potential - np.min(potential)
        potential = potential / (np.max(potential) + 1e-10)

        # Identificar "poços" de potencial (mínimos locais)
        wells, _ = find_peaks(-potential, distance=5)

        # Identificar "barreiras" de potencial (máximos locais)
        barriers, _ = find_peaks(potential, distance=5)

        return {
            'x_grid': x_grid,
            'potential': potential,
            'density': density,
            'wells': wells,
            'barriers': barriers,
            'current_price': prices[-1],
            'dx': x_grid[1] - x_grid[0]
        }

    def solve_schrodinger(self, potential_result: dict) -> dict:
        """
        Equação: Resolva a Equação de Schrödinger independente do tempo para
        encontrar a função de onda psi(x):

        -hbar^2/(2m) * d^2psi/dx^2 + V(x)psi = E*psi

        Usamos método de diferenças finitas para discretizar.
        """
        V = potential_result['potential']
        x = potential_result['x_grid']
        dx = potential_result['dx']
        n = len(x)

        # Constantes
        hbar = self.hbar
        m = self.particle_mass

        # Coeficiente cinético
        kinetic_coeff = hbar**2 / (2 * m * dx**2)

        # Construir Hamiltoniano tridiagonal
        # H = -kinetic_coeff * (psi_{i+1} - 2psi_i + psi_{i-1}) + V_i * psi_i

        # Diagonal principal
        diagonal = 2 * kinetic_coeff + V

        # Diagonais secundárias
        off_diagonal = -kinetic_coeff * np.ones(n - 1)

        # Resolver autovalores/autovetores (estados estacionários)
        try:
            # Usar solver tridiagonal eficiente
            n_states = min(self.n_eigenstates, n - 2)
            eigenvalues, eigenvectors = eigh_tridiagonal(diagonal, off_diagonal)

            # Selecionar primeiros n_states
            eigenvalues = eigenvalues[:n_states]
            eigenvectors = eigenvectors[:, :n_states]

        except Exception as e:
            # Fallback: usar autovalores aproximados
            eigenvalues = np.sort(V)[:self.n_eigenstates]
            eigenvectors = np.eye(n)[:, :self.n_eigenstates]

        # Normalizar funções de onda
        for i in range(eigenvectors.shape[1]):
            norm = np.sqrt(np.sum(eigenvectors[:, i]**2) * dx)
            eigenvectors[:, i] /= (norm + 1e-10)

        return {
            'eigenvalues': eigenvalues,  # Energias E_n
            'eigenvectors': eigenvectors,  # Funções de onda psi_n(x)
            'ground_state_energy': eigenvalues[0] if len(eigenvalues) > 0 else 0,
            'x_grid': x,
            'potential': V
        }

    def calculate_tunneling_probability(self, schrodinger_result: dict,
                                        potential_result: dict) -> dict:
        """
        O Sinal de Tunelamento:

        - Normalmente, o preço fica preso no poço V(x).
        - O indicador dispara quando a probabilidade da partícula estar fora do poço
          (|psi(x_out)|^2) excede um limiar crítico, significando que o preço tem energia
          suficiente para "atravessar a parede" de liquidez (Breakout via Tunelamento Quântico).
        """
        x = schrodinger_result['x_grid']
        V = schrodinger_result['potential']
        psi = schrodinger_result['eigenvectors']
        energies = schrodinger_result['eigenvalues']

        current_price = potential_result['current_price']
        dx = potential_result['dx']
        barriers = potential_result['barriers']
        wells = potential_result['wells']

        # Encontrar índice do preço atual
        current_idx = np.argmin(np.abs(x - current_price))

        # Usar estado fundamental ou superposição dos primeiros estados
        # Estado efetivo = superposição ponderada por Boltzmann
        kT = 0.1  # "Temperatura" efetiva do mercado

        # Tratar energias para evitar overflow
        energies_shifted = energies - np.min(energies)  # Shift para evitar exp grande
        weights = np.exp(-energies_shifted / kT)
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=0.0)
        weight_sum = np.sum(weights)
        if weight_sum > 1e-10:
            weights /= weight_sum
        else:
            weights = np.ones(len(energies)) / len(energies)

        # Função de onda efetiva
        psi_effective = np.zeros(len(x))
        for i in range(len(weights)):
            psi_effective += weights[i] * psi[:, i]**2

        # Normalizar e tratar NaN/Inf
        psi_effective = np.nan_to_num(psi_effective, nan=0.0, posinf=0.0, neginf=0.0)
        psi_sum = np.sum(psi_effective) * dx
        if psi_sum > 1e-10:
            psi_effective /= psi_sum
        else:
            # Fallback: distribuição uniforme
            psi_effective = np.ones(len(x)) / (len(x) * dx)

        # Identificar região "dentro do poço" vs "fora do poço"
        if len(barriers) >= 2:
            # Região entre barreiras é o "poço"
            left_barrier = x[barriers[0]] if len(barriers) > 0 else x[0]
            right_barrier = x[barriers[-1]] if len(barriers) > 1 else x[-1]
        else:
            # Usar desvio padrão como proxy
            std = np.std(x)
            mean = np.mean(x)
            left_barrier = mean - std
            right_barrier = mean + std

        # Probabilidade fora do poço (tunelamento)
        inside_mask = (x >= left_barrier) & (x <= right_barrier)
        outside_mask = ~inside_mask

        prob_inside = np.nansum(psi_effective[inside_mask]) * dx
        prob_outside = np.nansum(psi_effective[outside_mask]) * dx

        # Garantir valores válidos
        if np.isnan(prob_inside) or np.isinf(prob_inside):
            prob_inside = 0.5
        if np.isnan(prob_outside) or np.isinf(prob_outside):
            prob_outside = 0.5

        # Probabilidade no preço atual
        prob_current = psi_effective[current_idx] * dx
        if np.isnan(prob_current):
            prob_current = 0.0

        # Verificar se preço está fora do poço
        price_outside_well = current_price < left_barrier or current_price > right_barrier

        # Energia do estado atual vs barreira
        barrier_height = np.max(V[barriers]) if len(barriers) > 0 else np.max(V)
        current_energy = V[current_idx]

        # Energia suficiente para tunelamento?
        energy_ratio = current_energy / (barrier_height + 1e-10)

        # Sinal de tunelamento
        tunneling_signal = prob_outside > self.tunneling_probability_threshold

        # Direção provável (gradiente do momento)
        # Fluxo de probabilidade J = (hbar/m) * Im(psi* grad psi)
        psi_sqrt = np.sqrt(psi_effective + 1e-10)
        psi_gradient = np.gradient(psi_sqrt)
        momentum_direction = np.sign(psi_gradient[current_idx])

        # Tratar NaN
        if np.isnan(momentum_direction):
            momentum_direction = 0

        return {
            'tunneling_probability': prob_outside,
            'prob_inside_well': prob_inside,
            'prob_outside_well': prob_outside,
            'prob_at_current_price': prob_current,
            'price_outside_well': price_outside_well,
            'barrier_height': barrier_height,
            'current_energy': current_energy,
            'energy_ratio': energy_ratio,
            'tunneling_signal': tunneling_signal,
            'momentum_direction': momentum_direction,  # +1 = up, -1 = down
            'left_barrier': left_barrier,
            'right_barrier': right_barrier,
            'psi_effective': psi_effective
        }

    # =========================================================================
    # MÓDULO 4: O Sintetizador de Decisão (Gatilho Lógico)
    # =========================================================================

    def synthesize_decision(self, topology_result: dict,
                           entropy_result: dict,
                           tunneling_result: dict) -> dict:
        """
        Combinando Topologia e Quântica:

        1. Verificação de Regime (Topologia): A Entropia de Persistência (B1) está subindo?
           (O mercado está se organizando geometricamente, saindo do caos aleatório).

        2. Verificação de Energia (Quântica): O nível de energia do autoestado atual (E_n)
           é superior à barreira de potencial local?

        3. Direção: Dada pelo gradiente do momento da função de onda (fluxo de probabilidade).
        """
        # 1. Verificação de Regime (Topologia)
        persistence_entropy = entropy_result['persistence_entropy']
        is_organizing = not entropy_result['is_trivial_market']
        n_significant_cycles = entropy_result['n_significant_cycles']

        # Mercado com estrutura topológica interessante
        topology_valid = is_organizing and n_significant_cycles > 0

        # 2. Verificação de Energia (Quântica)
        tunneling_prob = tunneling_result['tunneling_probability']
        energy_ratio = tunneling_result['energy_ratio']
        tunneling_signal = tunneling_result['tunneling_signal']

        # Energia suficiente para atravessar barreira
        energy_valid = tunneling_signal or energy_ratio > 0.5

        # 3. Direção
        direction = tunneling_result['momentum_direction']
        direction_str = 'LONG' if direction > 0 else 'SHORT' if direction < 0 else 'NEUTRAL'

        # Decisão final: TRADE ON apenas se ambas condições forem satisfeitas
        # Muitas linhas longas = Alta complexidade/Volatilidade estruturada = TRADE ON
        # Apenas linhas curtas = Ruído branco = SYSTEM OFF

        trade_on = topology_valid and energy_valid

        # Força do sinal (0 a 1)
        signal_strength = (
            0.4 * min(persistence_entropy, 1.0) +
            0.4 * min(tunneling_prob * 2 if not np.isnan(tunneling_prob) else 0, 1.0) +
            0.2 * min(n_significant_cycles / 5, 1.0)
        )

        # Garantir valor válido
        if np.isnan(signal_strength):
            signal_strength = 0.0

        return {
            # Verificações individuais
            'topology_valid': topology_valid,
            'energy_valid': energy_valid,

            # Métricas
            'persistence_entropy': persistence_entropy,
            'tunneling_probability': tunneling_prob,
            'energy_ratio': energy_ratio,
            'n_significant_cycles': n_significant_cycles,

            # Decisão
            'trade_on': trade_on,
            'direction': direction_str,
            'direction_numeric': direction,
            'signal_strength': signal_strength,

            # Status do sistema
            'system_status': 'TRADE ON' if trade_on else 'SYSTEM OFF'
        }

    # =========================================================================
    # MÓDULO 5: Output e Visualização
    # =========================================================================

    def analyze(self, prices: np.ndarray) -> dict:
        """
        Execução completa do Detector de Tunelamento Topológico.

        Retorna análise completa com todos os subsistemas.
        """
        prices = np.array(prices, dtype=float)

        if len(prices) < 100:
            raise ValueError("Dados insuficientes. Necessário mínimo de 100 pontos de preço.")

        # 1. Embedding de Takens
        embedding_result = self.get_takens_embedding(prices)

        # 2. Homologia Persistente
        point_cloud = embedding_result['point_cloud_reduced']
        topology_result = self.compute_persistent_homology(point_cloud)

        # 3. Entropia de Persistência
        entropy_result = self.calculate_persistence_entropy(topology_result)

        # 4. Potencial Quântico
        potential_result = self.estimate_potential(prices)

        # 5. Equação de Schrödinger
        schrodinger_result = self.solve_schrodinger(potential_result)

        # 6. Probabilidade de Tunelamento
        tunneling_result = self.calculate_tunneling_probability(
            schrodinger_result, potential_result
        )

        # 7. Sintetizar Decisão
        decision = self.synthesize_decision(
            topology_result, entropy_result, tunneling_result
        )

        return {
            # Decisão principal
            'trade_on': decision['trade_on'],
            'system_status': decision['system_status'],
            'direction': decision['direction'],
            'signal_strength': decision['signal_strength'],

            # Resultados detalhados
            'embedding': embedding_result,
            'topology': topology_result,
            'entropy': entropy_result,
            'potential': potential_result,
            'schrodinger': schrodinger_result,
            'tunneling': tunneling_result,
            'decision': decision,

            # Metadados
            'n_observations': len(prices),
            'current_price': prices[-1],
            'tda_backend': TDA_BACKEND
        }

    def get_signal(self, prices: np.ndarray) -> int:
        """
        Retorna sinal simplificado:
        1 = TRADE ON (tunneling detectado)
        0 = SYSTEM OFF
        """
        result = self.analyze(prices)
        return 1 if result['trade_on'] else 0

    def get_barcode_data(self, topology_result: dict) -> dict:
        """
        Subjanela: Plotar o "Barcode de Persistência" (linhas horizontais que
        representam a vida útil dos ciclos topológicos).

        - Muitas linhas longas = Alta complexidade/Volatilidade estruturada = TRADE ON
        - Apenas linhas curtas = Ruído branco = SYSTEM OFF
        """
        persistence_pairs = topology_result['persistence_pairs']

        barcode_data = {}
        for dim, pairs in persistence_pairs.items():
            if len(pairs) > 0:
                births = pairs[:, 0]
                deaths = pairs[:, 1]
                lifetimes = deaths - births

                # Ordenar por lifetime
                sort_idx = np.argsort(lifetimes)[::-1]

                barcode_data[dim] = {
                    'births': births[sort_idx],
                    'deaths': deaths[sort_idx],
                    'lifetimes': lifetimes[sort_idx]
                }
            else:
                barcode_data[dim] = {
                    'births': np.array([]),
                    'deaths': np.array([]),
                    'lifetimes': np.array([])
                }

        return barcode_data


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DETECTOR DE TUNELAMENTO TOPOLOGICO (DTT)")
    print("Experimental / Deep Quant")
    print("=" * 80)
    print(f"\nBackend TDA: {TDA_BACKEND or 'fallback (instale gudhi ou ripser)'}")

    # Gerar dados simulados
    np.random.seed(42)
    n_points = 500

    # Simular preços com diferentes regimes topológicos
    # Regime 1: Mercado lateral (topologicamente trivial)
    consolidation = 1.1000 + 0.0005 * np.cumsum(np.random.randn(150))

    # Regime 2: Formação de estrutura (ciclos persistentes)
    t = np.linspace(0, 4*np.pi, 200)
    structured = consolidation[-1] + 0.005 * np.sin(t) + 0.001 * np.cumsum(np.random.randn(200))

    # Regime 3: Breakout (tunelamento)
    breakout = structured[-1] + np.linspace(0, 0.015, 150) + 0.0008 * np.cumsum(np.random.randn(150))

    prices = np.concatenate([consolidation, structured, breakout])

    print(f"\nDados simulados: {len(prices)} pontos")
    print(f"Preco inicial: {prices[0]:.5f}")
    print(f"Preco final: {prices[-1]:.5f}")

    # Criar detector
    dtt = DetectorTunelamentoTopologico(
        max_points=200,
        use_dimensionality_reduction=True,
        reduction_method='pca',
        persistence_entropy_threshold=0.3,
        tunneling_probability_threshold=0.15
    )

    # Executar análise
    print("\n" + "-" * 40)
    print("Executando analise DTT...")
    print("-" * 40)

    result = dtt.analyze(prices)

    # Mostrar resultados
    print("\nRESULTADO PRINCIPAL:")
    print(f"   Status: {result['system_status']}")
    print(f"   Direcao: {result['direction']}")
    print(f"   Forca do Sinal: {result['signal_strength']:.4f}")

    print("\nEMBEDDING DE TAKENS:")
    emb = result['embedding']
    print(f"   Dimensao (m): {emb['embedding_dim']}")
    print(f"   Time Delay (tau): {emb['time_delay']}")
    print(f"   Pontos na nuvem: {emb['n_points']}")

    print("\nHOMOLOGIA PERSISTENTE:")
    topo = result['topology']
    ent = result['entropy']
    print(f"   Betti_0 (componentes): {topo['betti_0']}")
    print(f"   Betti_1 (ciclos): {topo['betti_1']}")
    print(f"   Entropia de Persistencia: {ent['persistence_entropy']:.4f}")
    print(f"   Ciclos Significativos: {ent['n_significant_cycles']}")
    print(f"   Mercado Trivial: {'Sim' if ent['is_trivial_market'] else 'Nao'}")

    print("\nPOTENCIAL QUANTICO:")
    tunn = result['tunneling']
    print(f"   P(Tunelamento): {tunn['tunneling_probability']:.4f}")
    print(f"   P(Dentro do Poco): {tunn['prob_inside_well']:.4f}")
    print(f"   P(Fora do Poco): {tunn['prob_outside_well']:.4f}")
    print(f"   Razao de Energia: {tunn['energy_ratio']:.4f}")
    print(f"   Direcao do Momento: {'+1 (UP)' if tunn['momentum_direction'] > 0 else '-1 (DOWN)'}")

    print("\nDECISAO FINAL:")
    dec = result['decision']
    print(f"   Topologia Valida: {'SIM' if dec['topology_valid'] else 'NAO'}")
    print(f"   Energia Valida: {'SIM' if dec['energy_valid'] else 'NAO'}")

    print("\n" + "=" * 80)
    if result['trade_on']:
        print(f"TRADE ON - TUNNELLING EVENT DETECTADO!")
        print(f"   Direcao recomendada: {result['direction']}")
        print(f"   O mercado esta atravessando a barreira de liquidez.")
    else:
        print("SYSTEM OFF - Aguardando estrutura topologica.")
        print("   Mercado em ruido branco ou sem energia suficiente.")
    print("=" * 80)
