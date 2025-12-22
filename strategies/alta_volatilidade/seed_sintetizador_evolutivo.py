"""
Sintetizador Evolutivo de Estruturas Dissipativas (SEED)
=========================================================
Nível de Complexidade: Termodinâmica Estatística / Biologia Evolutiva.

Premissa Teórica:
1. Termodinâmica: O mercado é um sistema aberto longe do equilíbrio. Segundo Prigogine,
   nesses sistemas, flutuações gigantescas não geram caos, mas Estruturas Dissipativas
   (auto-organização, como um tornado ou uma tendência de alta).

2. Evolução: Modelamos Bulls e Bears não como forças estáticas, mas como populações em
   uma Dinâmica de Replicador. A "aptidão" (fitness) de cada espécie depende da
   eficiência com que ela dissipa a entropia do mercado.

Dependências Críticas: scipy.integrate (ODEINT), scipy.spatial (Voronoi), numpy.

O Desafio Final de Programação (Simulação Dinâmica):
Você precisará resolver o sistema de equações diferenciais da Dinâmica de Replicador
(odeint) a cada tick ou a cada fechamento de vela, realimentando o resultado anterior
como condição inicial. Isso cria um sistema com memória evolutiva.
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.spatial import Voronoi, ConvexHull
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class EntropyProductionRate:
    """
    Módulo 1: Taxa de Produção de Entropia (σ)

    Não olhe para o preço. Olhe para a dissipação de energia.

    - Defina o Fluxo Termodinâmico (J): O volume de ordens executadas (agressão).
    - Defina a Força Termodinâmica (X): O gradiente de volatilidade local (∇σ_vol).

    Cálculo da Produção de Entropia Local:
        σ(t) = J(t) · X(t)

    Lógica:
    - Perto do equilíbrio (consolidação), σ é mínimo (Teorema da Produção de Entropia Mínima).
    - Em alta volatilidade, σ explode. Se σ ultrapassa um limiar crítico, o sistema sai do
      regime linear e cria uma Estrutura Dissipativa (Tendência Forte).
    """

    def __init__(self, critical_threshold: float = 2.0,
                 smoothing_window: int = 5):
        """
        Parâmetros:
        -----------
        critical_threshold : float
            Limiar crítico para σ (regime não-linear)
        smoothing_window : int
            Janela de suavização para cálculos
        """
        self.critical_threshold = critical_threshold
        self.smoothing_window = smoothing_window
        self.eps = 1e-10

        # Histórico
        self.sigma_history = []

    def compute_thermodynamic_flux(self, volumes: np.ndarray,
                                    prices: np.ndarray) -> np.ndarray:
        """
        Calcula o Fluxo Termodinâmico J(t).

        J = Volume de ordens executadas × direção (agressão)

        Representa a "corrente" de entropia fluindo pelo sistema.
        """
        if len(volumes) < 2:
            return np.zeros(len(prices))

        # Direção do movimento (sinal)
        price_direction = np.sign(np.diff(prices, prepend=prices[0]))

        # Fluxo = Volume × Direção
        J = volumes * price_direction

        # Normalizar pelo volume médio
        J_normalized = J / (np.mean(np.abs(volumes)) + self.eps)

        return J_normalized

    def compute_thermodynamic_force(self, prices: np.ndarray,
                                     window: int = 10) -> np.ndarray:
        """
        Calcula a Força Termodinâmica X(t).

        X = Gradiente de volatilidade local (∇σ_vol)

        Representa o "potencial" que impulsiona o fluxo de entropia.
        """
        n = len(prices)

        if n < window:
            window = max(2, n // 2)

        # Calcular volatilidade local (rolling std)
        volatility = np.zeros(n)

        for i in range(n):
            start = max(0, i - window + 1)
            window_prices = prices[start:i+1]
            if len(window_prices) > 1:
                returns = np.diff(window_prices) / (window_prices[:-1] + self.eps)
                volatility[i] = np.std(returns)

        # Gradiente da volatilidade
        X = np.gradient(volatility)

        return X

    def compute_entropy_production(self, prices: np.ndarray,
                                    volumes: np.ndarray = None) -> dict:
        """
        Calcula a Taxa de Produção de Entropia Local:
            σ(t) = J(t) · X(t)
        """
        n = len(prices)

        # Gerar volumes sintéticos se não fornecidos
        if volumes is None:
            volumes = np.abs(np.diff(prices, prepend=prices[0])) * 10000 + \
                      np.random.rand(n) * 1000

        # Calcular fluxo e força
        J = self.compute_thermodynamic_flux(volumes, prices)
        X = self.compute_thermodynamic_force(prices)

        # Produção de entropia: σ = J · X
        sigma = J * X

        # Suavizar
        sigma_smooth = gaussian_filter1d(sigma, sigma=self.smoothing_window)

        # Normalizar para escala interpretável
        sigma_normalized = sigma_smooth / (np.std(sigma_smooth) + self.eps)

        # Atualizar histórico
        self.sigma_history.extend(sigma_normalized.tolist())
        if len(self.sigma_history) > 1000:
            self.sigma_history = self.sigma_history[-1000:]

        # Detectar regime
        current_sigma = sigma_normalized[-1] if len(sigma_normalized) > 0 else 0
        is_nonlinear = np.abs(current_sigma) > self.critical_threshold

        # Detectar formação de estrutura dissipativa
        if len(sigma_normalized) > 10:
            sigma_trend = np.polyfit(range(10), sigma_normalized[-10:], 1)[0]
            structure_forming = sigma_trend > 0.1 and is_nonlinear
        else:
            sigma_trend = 0
            structure_forming = False

        return {
            'sigma': sigma_normalized,
            'sigma_raw': sigma,
            'J': J,
            'X': X,
            'current_sigma': current_sigma,
            'is_nonlinear_regime': is_nonlinear,
            'structure_forming': structure_forming,
            'sigma_trend': sigma_trend,
            'threshold': self.critical_threshold
        }


class ReplicatorDynamics:
    """
    Módulo 2: Dinâmica de Replicador (Evolutionary Game Theory)

    Simule a batalha entre três espécies de agentes no tempo t:
    - x₁: Bulls (Compradores de Tendência)
    - x₂: Bears (Vendedores de Tendência)
    - x₃: Mean Reverters (Apostam no retorno à média/Ruído)

    A evolução das populações segue a equação diferencial:
        ẋᵢ = xᵢ(fᵢ(x) - f̄(x))

    Onde fᵢ é o Fitness da espécie i (lucratividade instantânea baseada no movimento recente).
    f̄ é o fitness médio do ecossistema.
    Restrição: Σxᵢ = 1 (Simplex).

    O Desafio: Resolver o sistema de EDOs (odeint) a cada tick, realimentando o resultado
    anterior como condição inicial. Sistema com memória evolutiva.
    """

    def __init__(self,
                 fitness_window: int = 20,
                 integration_steps: int = 100,
                 mutation_rate: float = 0.01):
        """
        Parâmetros:
        -----------
        fitness_window : int
            Janela para cálculo de fitness
        integration_steps : int
            Passos de integração por período
        mutation_rate : float
            Taxa de mutação (estabiliza dinâmica)
        """
        self.fitness_window = fitness_window
        self.integration_steps = integration_steps
        self.mutation_rate = mutation_rate
        self.eps = 1e-10

        # Estado atual das populações [x1, x2, x3]
        # Iniciar em equilíbrio simétrico
        self.populations = np.array([1/3, 1/3, 1/3])

        # Histórico evolutivo
        self.population_history = []
        self.fitness_history = []

    def compute_fitness(self, prices: np.ndarray,
                        window: int = None) -> np.ndarray:
        """
        Calcula o fitness de cada espécie baseado no movimento recente.

        - Bulls (x₁): Lucram quando preço sobe
        - Bears (x₂): Lucram quando preço cai
        - Mean Reverters (x₃): Lucram em reversões

        Retorna: [f₁, f₂, f₃]
        """
        if window is None:
            window = self.fitness_window

        if len(prices) < window:
            window = len(prices)

        recent_prices = prices[-window:]

        if len(recent_prices) < 2:
            return np.array([0.0, 0.0, 0.0])

        # Retornos
        returns = np.diff(recent_prices) / (recent_prices[:-1] + self.eps)

        # Fitness dos Bulls: soma dos retornos positivos
        f1 = np.sum(np.maximum(returns, 0))

        # Fitness dos Bears: soma dos retornos negativos (invertido)
        f2 = np.sum(np.maximum(-returns, 0))

        # Fitness dos Mean Reverters: lucram com reversões
        # Detectar reversões (mudança de sinal nos retornos)
        sign_changes = np.sum(np.abs(np.diff(np.sign(returns)))) / 2
        mean_reversion_profit = sign_changes * np.std(returns)
        f3 = mean_reversion_profit

        # Normalizar
        total = np.abs(f1) + np.abs(f2) + np.abs(f3) + self.eps
        fitness = np.array([f1, f2, f3]) / total

        return fitness

    def replicator_equations(self, x: np.ndarray, t: float,
                              fitness: np.ndarray) -> np.ndarray:
        """
        Sistema de equações diferenciais da Dinâmica de Replicador.

        ẋᵢ = xᵢ(fᵢ(x) - f̄(x))

        Com termo de mutação para estabilidade.
        """
        # Garantir que estamos no simplex
        x = np.maximum(x, self.eps)
        x = x / np.sum(x)

        # Fitness médio do ecossistema
        f_bar = np.sum(x * fitness)

        # Dinâmica de replicador
        dx = x * (fitness - f_bar)

        # Adicionar mutação (evita extinção completa)
        mutation = self.mutation_rate * (1/3 - x)
        dx += mutation

        return dx

    def evolve_populations(self, prices: np.ndarray,
                           dt: float = 1.0) -> dict:
        """
        Evolui as populações usando ODEINT.

        Realimenta o resultado anterior como condição inicial,
        criando um sistema com memória evolutiva.
        """
        # Calcular fitness atual
        fitness = self.compute_fitness(prices)

        # Integrar equações de replicador
        t_span = np.linspace(0, dt, self.integration_steps)

        try:
            # Usar odeint para resolver EDOs
            solution = odeint(
                self.replicator_equations,
                self.populations,
                t_span,
                args=(fitness,)
            )

            # Pegar estado final
            new_populations = solution[-1]
        except:
            # Fallback: Euler simples
            dx = self.replicator_equations(self.populations, 0, fitness)
            new_populations = self.populations + dx * dt

        # Garantir que estamos no simplex
        new_populations = np.maximum(new_populations, self.eps)
        new_populations = new_populations / np.sum(new_populations)

        # Atualizar estado
        self.populations = new_populations

        # Salvar histórico
        self.population_history.append(new_populations.copy())
        self.fitness_history.append(fitness.copy())

        if len(self.population_history) > 1000:
            self.population_history = self.population_history[-1000:]
            self.fitness_history = self.fitness_history[-1000:]

        # Calcular derivadas (velocidade de mudança)
        if len(self.population_history) > 1:
            prev = self.population_history[-2]
            dx = new_populations - prev
        else:
            dx = np.zeros(3)

        return {
            'populations': new_populations,
            'x1_bulls': new_populations[0],
            'x2_bears': new_populations[1],
            'x3_mean_reverters': new_populations[2],
            'fitness': fitness,
            'f1_bulls': fitness[0],
            'f2_bears': fitness[1],
            'f3_mean_reverters': fitness[2],
            'dx': dx,
            'dx1': dx[0],
            'dx2': dx[1],
            'dx3': dx[2],
            'dominant_species': ['Bulls', 'Bears', 'Mean Reverters'][np.argmax(new_populations)]
        }


class HakenSlavingPrinciple:
    """
    Módulo 3: O Princípio de Escravização de Haken (Synergetics)

    Em transições de fase, as variáveis rápidas (ruído/Mean Reverters) são
    "escravizadas" pelas variáveis lentas (Tendência/Estrutura Dissipativa).

    Cálculo: Monitore a amplitude da população x₃ (Mean Reverters).

    O Sinal: Quando a população x₃ colapsa para perto de 0 e uma das outras (x₁ ou x₂)
    domina o Simplex, as "modas de ordem" escravizaram o sistema. A volatilidade deixou de
    ser ruído e virou sinal.
    """

    def __init__(self,
                 slaving_threshold: float = 0.15,
                 dominance_threshold: float = 0.5):
        """
        Parâmetros:
        -----------
        slaving_threshold : float
            Limiar abaixo do qual x₃ é considerado "escravizado"
        dominance_threshold : float
            Limiar acima do qual uma espécie é considerada dominante
        """
        self.slaving_threshold = slaving_threshold
        self.dominance_threshold = dominance_threshold
        self.eps = 1e-10

    def check_slaving(self, populations: np.ndarray,
                      population_history: List[np.ndarray] = None) -> dict:
        """
        Verifica se o Princípio de Escravização está ativo.

        Condições:
        1. x₃ (Mean Reverters) colapsou para próximo de 0
        2. Uma das espécies de tendência (x₁ ou x₂) domina
        """
        x1, x2, x3 = populations

        # Verificar se Mean Reverters colapsaram
        mean_reverters_collapsed = x3 < self.slaving_threshold

        # Verificar dominância
        bulls_dominant = x1 > self.dominance_threshold
        bears_dominant = x2 > self.dominance_threshold

        # Princípio de escravização ativo
        slaving_active = mean_reverters_collapsed and (bulls_dominant or bears_dominant)

        # Determinar modo de ordem dominante
        if slaving_active:
            if bulls_dominant:
                order_mode = "BULLS"
                direction = 1
            else:
                order_mode = "BEARS"
                direction = -1
        else:
            order_mode = "NONE"
            direction = 0

        # Calcular velocidade de colapso de x₃
        collapse_rate = 0.0
        if population_history is not None and len(population_history) > 5:
            x3_recent = [p[2] for p in population_history[-5:]]
            collapse_rate = x3_recent[0] - x3_recent[-1]  # Positivo = colapsando

        # Força do sinal (quão forte é a escravização)
        signal_strength = 0.0
        if slaving_active:
            # Quanto menor x₃ e maior o dominante, mais forte
            dominance = max(x1, x2)
            signal_strength = (1 - x3) * dominance

        return {
            'slaving_active': slaving_active,
            'mean_reverters_collapsed': mean_reverters_collapsed,
            'order_mode': order_mode,
            'direction': direction,
            'x3_amplitude': x3,
            'collapse_rate': collapse_rate,
            'signal_strength': signal_strength,
            'bulls_dominant': bulls_dominant,
            'bears_dominant': bears_dominant,
            'noise_became_signal': slaving_active
        }


class VoronoiPhaseSpace:
    """
    Módulo 4: Diagrama de Voronoi do Espaço de Fase

    Para visualizar a estabilidade, mapeie o estado (x₁, x₂, x₃) em um triângulo
    ternário e calcule as células de Voronoi baseadas em atratores históricos.

    Se o ponto do estado atual cruzar a fronteira de Voronoi para a região de
    "Dominância Bull" com alta Produção de Entropia (σ), é o Ponto de Ignição.
    """

    def __init__(self):
        """
        Inicializa o espaço de fase com atratores conhecidos.
        """
        self.eps = 1e-10

        # Atratores históricos no simplex
        # Formato: (x1, x2, x3) - soma = 1
        self.attractors = {
            'bull_dominance': np.array([0.7, 0.2, 0.1]),
            'bear_dominance': np.array([0.2, 0.7, 0.1]),
            'equilibrium': np.array([0.33, 0.33, 0.34]),
            'high_noise': np.array([0.25, 0.25, 0.5]),
            'bull_bear_battle': np.array([0.45, 0.45, 0.1])
        }

        # Histórico de estados
        self.state_history = []

    def simplex_to_cartesian(self, x: np.ndarray) -> np.ndarray:
        """
        Converte coordenadas do simplex (x1, x2, x3) para coordenadas
        cartesianas no triângulo equilátero.

        Vértices do triângulo:
        - Bulls (x1=1): (0, 0)
        - Bears (x2=1): (1, 0)
        - Mean Reverters (x3=1): (0.5, √3/2)
        """
        x1, x2, x3 = x

        # Coordenadas cartesianas
        cart_x = x2 + 0.5 * x3
        cart_y = (np.sqrt(3) / 2) * x3

        return np.array([cart_x, cart_y])

    def cartesian_to_simplex(self, cart: np.ndarray) -> np.ndarray:
        """
        Converte coordenadas cartesianas de volta para o simplex.
        """
        cart_x, cart_y = cart

        x3 = cart_y / (np.sqrt(3) / 2)
        x2 = cart_x - 0.5 * x3
        x1 = 1 - x2 - x3

        return np.array([x1, x2, x3])

    def identify_region(self, state: np.ndarray) -> dict:
        """
        Identifica a região do espaço de fase onde o estado se encontra.
        """
        # Calcular distância para cada atrator
        distances = {}
        for name, attractor in self.attractors.items():
            dist = np.linalg.norm(state - attractor)
            distances[name] = dist

        # Região mais próxima
        closest_region = min(distances, key=distances.get)

        return {
            'region': closest_region,
            'distances': distances,
            'distance_to_closest': distances[closest_region]
        }

    def check_boundary_crossing(self, current_state: np.ndarray,
                                 previous_state: np.ndarray = None) -> dict:
        """
        Verifica se houve cruzamento de fronteira entre regiões.

        Se o ponto cruzar para "bull_dominance" com alta σ → Ponto de Ignição.
        """
        current_region = self.identify_region(current_state)

        if previous_state is None and len(self.state_history) > 0:
            previous_state = self.state_history[-1]

        boundary_crossed = False
        from_region = None
        to_region = current_region['region']

        if previous_state is not None:
            prev_region = self.identify_region(previous_state)
            from_region = prev_region['region']
            boundary_crossed = from_region != to_region

        # Salvar estado
        self.state_history.append(current_state.copy())
        if len(self.state_history) > 500:
            self.state_history = self.state_history[-500:]

        # Calcular trajetória (últimos N pontos em cartesianas)
        trajectory = []
        for state in self.state_history[-50:]:
            cart = self.simplex_to_cartesian(state)
            trajectory.append(cart)

        return {
            'current_region': to_region,
            'previous_region': from_region,
            'boundary_crossed': boundary_crossed,
            'is_ignition_point': boundary_crossed and to_region == 'bull_dominance',
            'trajectory': np.array(trajectory) if trajectory else None,
            'distances': current_region['distances']
        }

    def compute_voronoi_cells(self) -> dict:
        """
        Calcula as células de Voronoi para os atratores no espaço cartesiano.
        """
        # Converter atratores para cartesianas
        points = []
        names = []

        for name, attractor in self.attractors.items():
            cart = self.simplex_to_cartesian(attractor)
            points.append(cart)
            names.append(name)

        points = np.array(points)

        # Calcular Voronoi
        try:
            vor = Voronoi(points)
            return {
                'voronoi': vor,
                'points': points,
                'names': names,
                'vertices': vor.vertices,
                'regions': vor.regions,
                'ridge_vertices': vor.ridge_vertices
            }
        except:
            return {
                'voronoi': None,
                'points': points,
                'names': names
            }


class SintetizadorEvolutivoEstruturasDissipativas:
    """
    Implementação completa do Sintetizador Evolutivo de Estruturas Dissipativas (SEED)

    Módulos:
    1. Taxa de Produção de Entropia (σ)
    2. Dinâmica de Replicador (Evolutionary Game Theory)
    3. O Princípio de Escravização de Haken (Synergetics)
    4. Diagrama de Voronoi do Espaço de Fase
    5. Output e Execução

    Sinal de Compra:
    1. Produção de Entropia σ > Limiar Crítico (Regime Não-Linear).
    2. População de Bulls x₁ está crescendo exponencialmente (ẋ₁ >> 0).
    3. População de Mean Reverters x₃ colapsou (Princípio de Escravização ativo).
    """

    def __init__(self,
                 critical_sigma: float = 1.5,
                 slaving_threshold: float = 0.15,
                 dominance_threshold: float = 0.5,
                 fitness_window: int = 20,
                 dx_threshold: float = 0.02):
        """
        Inicialização do SEED.

        Parâmetros:
        -----------
        critical_sigma : float
            Limiar crítico para σ (regime não-linear)
        slaving_threshold : float
            Limiar para colapso de x₃
        dominance_threshold : float
            Limiar de dominância de espécie
        fitness_window : int
            Janela para cálculo de fitness
        dx_threshold : float
            Limiar para crescimento exponencial
        """
        self.critical_sigma = critical_sigma
        self.dx_threshold = dx_threshold

        # Módulos
        self.entropy = EntropyProductionRate(critical_sigma)
        self.replicator = ReplicatorDynamics(fitness_window)
        self.haken = HakenSlavingPrinciple(slaving_threshold, dominance_threshold)
        self.voronoi = VoronoiPhaseSpace()

        # Histórico
        self._sigma_history = []
        self._signal_history = []

    def analyze(self, prices: np.ndarray,
                volumes: np.ndarray = None) -> dict:
        """
        Execução completa do SEED.

        Resolve o sistema de EDOs da Dinâmica de Replicador a cada chamada,
        realimentando o resultado anterior. Sistema com memória evolutiva.
        """
        n = len(prices)

        if n < 10:
            raise ValueError("Dados insuficientes. Necessário mínimo de 10 pontos.")

        # =====================================================================
        # 1. Calcular Taxa de Produção de Entropia
        # =====================================================================
        entropy_result = self.entropy.compute_entropy_production(prices, volumes)

        sigma = entropy_result['current_sigma']
        is_nonlinear = entropy_result['is_nonlinear_regime']
        structure_forming = entropy_result['structure_forming']

        self._sigma_history.append(sigma)
        if len(self._sigma_history) > 500:
            self._sigma_history = self._sigma_history[-500:]

        # =====================================================================
        # 2. Evoluir Populações (Dinâmica de Replicador)
        # =====================================================================
        pop_result = self.replicator.evolve_populations(prices)

        populations = pop_result['populations']
        dx = pop_result['dx']
        x1, x2, x3 = populations

        # Verificar crescimento exponencial
        bulls_growing = dx[0] > self.dx_threshold
        bears_growing = dx[1] > self.dx_threshold

        # =====================================================================
        # 3. Verificar Princípio de Escravização
        # =====================================================================
        slaving_result = self.haken.check_slaving(
            populations,
            self.replicator.population_history
        )

        slaving_active = slaving_result['slaving_active']
        order_mode = slaving_result['order_mode']

        # =====================================================================
        # 4. Análise do Espaço de Fase
        # =====================================================================
        voronoi_result = self.voronoi.check_boundary_crossing(populations)

        current_region = voronoi_result['current_region']
        boundary_crossed = voronoi_result['boundary_crossed']
        is_ignition = voronoi_result['is_ignition_point'] and is_nonlinear

        # =====================================================================
        # 5. Gerar Sinal de Trading
        # =====================================================================
        signal_result = self._generate_signal(
            sigma, is_nonlinear, structure_forming,
            x1, x2, x3, dx,
            bulls_growing, bears_growing,
            slaving_active, order_mode,
            current_region, is_ignition
        )

        return {
            # Sinal principal
            'signal': signal_result['signal'],
            'signal_name': signal_result['signal_name'],
            'confidence': signal_result['confidence'],
            'reasons': signal_result['reasons'],

            # Entropia
            'sigma': sigma,
            'sigma_series': entropy_result['sigma'],
            'is_nonlinear_regime': is_nonlinear,
            'structure_forming': structure_forming,
            'J': entropy_result['J'],
            'X': entropy_result['X'],

            # Populações
            'populations': populations,
            'x1_bulls': x1,
            'x2_bears': x2,
            'x3_mean_reverters': x3,
            'dx': dx,
            'dx1_bulls': dx[0],
            'dx2_bears': dx[1],
            'dx3_mean_reverters': dx[2],
            'fitness': pop_result['fitness'],
            'dominant_species': pop_result['dominant_species'],
            'bulls_growing_exp': bulls_growing,
            'bears_growing_exp': bears_growing,

            # Escravização
            'slaving_active': slaving_active,
            'order_mode': order_mode,
            'noise_became_signal': slaving_result['noise_became_signal'],
            'signal_strength': slaving_result['signal_strength'],

            # Espaço de fase
            'phase_region': current_region,
            'boundary_crossed': boundary_crossed,
            'is_ignition_point': is_ignition,
            'trajectory': voronoi_result['trajectory'],

            # Histórico
            'population_history': self.replicator.population_history,
            'sigma_history': self._sigma_history,

            # Metadados
            'n_observations': n,
            'current_price': prices[-1]
        }

    def _generate_signal(self, sigma: float, is_nonlinear: bool,
                         structure_forming: bool,
                         x1: float, x2: float, x3: float,
                         dx: np.ndarray,
                         bulls_growing: bool, bears_growing: bool,
                         slaving_active: bool, order_mode: str,
                         current_region: str, is_ignition: bool) -> dict:
        """
        Gera sinal de trading baseado nos critérios evolutivos.

        Sinal de Compra:
        1. Produção de Entropia σ > Limiar Crítico (Regime Não-Linear).
        2. População de Bulls x₁ está crescendo exponencialmente (ẋ₁ >> 0).
        3. População de Mean Reverters x₃ colapsou (Princípio de Escravização ativo).
        """
        signal = 0
        signal_name = "NEUTRO"
        confidence = 0.0
        reasons = []

        # Condições de COMPRA
        buy_conditions = {
            'sigma_nonlinear': is_nonlinear and sigma > 0,
            'bulls_growing_exp': bulls_growing,
            'slaving_bulls': slaving_active and order_mode == "BULLS",
            'structure_forming': structure_forming,
            'ignition_point': is_ignition,
            'bull_region': current_region == 'bull_dominance'
        }

        # Condições de VENDA
        sell_conditions = {
            'sigma_nonlinear_neg': is_nonlinear and sigma < 0,
            'bears_growing_exp': bears_growing,
            'slaving_bears': slaving_active and order_mode == "BEARS",
            'bear_region': current_region == 'bear_dominance'
        }

        # Contar condições
        buy_count = sum(buy_conditions.values())
        sell_count = sum(sell_conditions.values())

        # Gerar sinal de COMPRA
        if buy_count >= 3:
            signal = 1
            signal_name = "COMPRA (Estrutura Dissipativa Bull)"
            confidence = min(buy_count / 5, 1.0)
            reasons = [k for k, v in buy_conditions.items() if v]

        # Gerar sinal de VENDA
        elif sell_count >= 3:
            signal = -1
            signal_name = "VENDA (Estrutura Dissipativa Bear)"
            confidence = min(sell_count / 4, 1.0)
            reasons = [k for k, v in sell_conditions.items() if v]

        # Ponto de ignição (sinal forte)
        elif is_ignition:
            if x1 > x2:
                signal = 1
                signal_name = "IGNIÇÃO BULL!"
                confidence = 0.9
                reasons = ['ignition_point', 'bulls_dominant']
            else:
                signal = -1
                signal_name = "IGNIÇÃO BEAR!"
                confidence = 0.9
                reasons = ['ignition_point', 'bears_dominant']

        # Alerta
        elif buy_count >= 2 or sell_count >= 2:
            signal = 0
            signal_name = "ALERTA (Estrutura em Formação)"
            confidence = 0.4
            reasons = ['pre_structure']

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'reasons': reasons,
            'buy_conditions': buy_conditions,
            'sell_conditions': sell_conditions,
            'buy_count': buy_count,
            'sell_count': sell_count
        }

    def get_signal(self, prices: np.ndarray,
                   volumes: np.ndarray = None) -> int:
        """Retorna sinal simplificado."""
        result = self.analyze(prices, volumes)
        return result['signal']

    def reset_memory(self):
        """Reseta a memória evolutiva do sistema."""
        self.replicator.populations = np.array([1/3, 1/3, 1/3])
        self.replicator.population_history = []
        self.replicator.fitness_history = []
        self.entropy.sigma_history = []
        self.voronoi.state_history = []
        self._sigma_history = []


def plot_seed_analysis(prices: np.ndarray,
                       volumes: np.ndarray = None,
                       save_path: str = None):
    """
    Visualização do SEED.

    Plot 1: O "Simplex Evolutivo" (Triângulo Equilátero). Um ponto se move dentro dele.
            As pontas são Bull, Bear, Ruído. A trilha do ponto mostra a história evolutiva.

    Plot 2: Série temporal da Produção de Entropia (σ). Picos indicam formação de estrutura.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import matplotlib.tri as mtri

    print("Iniciando análise SEED...")
    seed = SintetizadorEvolutivoEstruturasDissipativas(
        critical_sigma=1.5,
        slaving_threshold=0.15
    )

    # Analisar progressivamente para ter histórico
    window = 30
    for i in range(window, len(prices), 10):
        _ = seed.analyze(prices[:i], volumes[:i] if volumes is not None else None)

    # Análise final
    result = seed.analyze(prices, volumes)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])

    # =========================================================================
    # Plot 1: Simplex Evolutivo (Triângulo Ternário)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    # Desenhar triângulo equilátero
    # Vértices: Bulls (0,0), Bears (1,0), Mean Reverters (0.5, √3/2)
    triangle = np.array([
        [0, 0],           # Bulls
        [1, 0],           # Bears
        [0.5, np.sqrt(3)/2]  # Mean Reverters
    ])

    # Desenhar triângulo
    tri = Polygon(triangle, fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(tri)

    # Colorir regiões
    # Região Bull (canto esquerdo)
    bull_region = Polygon([[0, 0], [0.35, 0], [0.25, 0.35]],
                          alpha=0.2, color='green', label='Bull Zone')
    ax1.add_patch(bull_region)

    # Região Bear (canto direito)
    bear_region = Polygon([[1, 0], [0.65, 0], [0.75, 0.35]],
                          alpha=0.2, color='red', label='Bear Zone')
    ax1.add_patch(bear_region)

    # Região Ruído (topo)
    noise_region = Polygon([[0.5, np.sqrt(3)/2], [0.3, 0.5], [0.7, 0.5]],
                           alpha=0.2, color='gray', label='Noise Zone')
    ax1.add_patch(noise_region)

    # Plotar trajetória evolutiva
    trajectory = result.get('trajectory')
    if trajectory is not None and len(trajectory) > 1:
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-',
                alpha=0.5, linewidth=1, label='Trajetória')

        # Ponto atual
        ax1.scatter([trajectory[-1, 0]], [trajectory[-1, 1]],
                   c='blue', s=200, marker='o', zorder=5, label='Estado Atual')

        # Ponto inicial
        ax1.scatter([trajectory[0, 0]], [trajectory[0, 1]],
                   c='lightblue', s=100, marker='s', zorder=4, label='Início')

    # Labels dos vértices
    ax1.text(-0.05, -0.05, 'BULLS\n(x₁)', fontsize=11, ha='center', weight='bold', color='green')
    ax1.text(1.05, -0.05, 'BEARS\n(x₂)', fontsize=11, ha='center', weight='bold', color='red')
    ax1.text(0.5, np.sqrt(3)/2 + 0.05, 'NOISE\n(x₃)', fontsize=11, ha='center', weight='bold', color='gray')

    # Info
    x1, x2, x3 = result['populations']
    info = (f"x₁={x1:.2f} | x₂={x2:.2f} | x₃={x3:.2f}\n"
            f"Dominante: {result['dominant_species']}\n"
            f"Escravização: {'✓' if result['slaving_active'] else '✗'}")
    ax1.text(0.5, 0.3, info, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.set_xlim(-0.15, 1.15)
    ax1.set_ylim(-0.15, 1.0)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Simplex Evolutivo - Dinâmica de Replicador', fontsize=12)
    ax1.legend(loc='upper right', fontsize=8)

    # =========================================================================
    # Plot 2: Produção de Entropia σ(t)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    sigma_series = result['sigma_series']
    time = np.arange(len(sigma_series))

    ax2.plot(time, sigma_series, 'purple', linewidth=1.5, label='σ(t)')
    ax2.fill_between(time, 0, sigma_series, where=sigma_series > 0,
                    alpha=0.3, color='green', label='σ > 0 (Bull)')
    ax2.fill_between(time, 0, sigma_series, where=sigma_series < 0,
                    alpha=0.3, color='red', label='σ < 0 (Bear)')

    # Limiar crítico
    ax2.axhline(y=seed.critical_sigma, color='red', linestyle='--',
               label=f'Limiar crítico ({seed.critical_sigma})')
    ax2.axhline(y=-seed.critical_sigma, color='red', linestyle='--')

    # Marcar regime não-linear
    if result['is_nonlinear_regime']:
        ax2.axvspan(len(sigma_series)-20, len(sigma_series),
                   alpha=0.2, color='yellow', label='Regime Não-Linear')

    # Marcar formação de estrutura
    if result['structure_forming']:
        ax2.text(0.5, 0.95, '🌀 ESTRUTURA DISSIPATIVA FORMANDO',
                transform=ax2.transAxes, ha='center', fontsize=11,
                color='purple', weight='bold')

    ax2.set_xlabel('Tempo')
    ax2.set_ylabel('σ(t) = J(t) · X(t)')
    ax2.set_title('Taxa de Produção de Entropia', fontsize=12)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 3: Preço + Sinal
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    price_time = np.arange(len(prices))
    ax3.plot(price_time, prices, 'b-', linewidth=1.5, label='Preço')

    # Colorir fundo baseado na região
    region = result['phase_region']
    if region == 'bull_dominance':
        ax3.axvspan(0, len(prices), alpha=0.1, color='green')
    elif region == 'bear_dominance':
        ax3.axvspan(0, len(prices), alpha=0.1, color='red')

    # Sinal
    if result['signal'] == 1:
        ax3.scatter([price_time[-1]], [prices[-1]], c='green', s=300,
                   marker='^', zorder=5, label='COMPRA')
    elif result['signal'] == -1:
        ax3.scatter([price_time[-1]], [prices[-1]], c='red', s=300,
                   marker='v', zorder=5, label='VENDA')

    ax3.set_xlabel('Tempo')
    ax3.set_ylabel('Preço')
    ax3.set_title(f'Preço | Região: {region} | {result["signal_name"]}', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 4: Evolução das Populações
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    pop_history = np.array(result['population_history']) if result['population_history'] else None

    if pop_history is not None and len(pop_history) > 1:
        pop_time = np.arange(len(pop_history))
        ax4.plot(pop_time, pop_history[:, 0], 'g-', linewidth=2, label='x₁ Bulls')
        ax4.plot(pop_time, pop_history[:, 1], 'r-', linewidth=2, label='x₂ Bears')
        ax4.plot(pop_time, pop_history[:, 2], 'gray', linewidth=2, label='x₃ Noise')

        # Marcar limiar de escravização
        ax4.axhline(y=seed.haken.slaving_threshold, color='orange',
                   linestyle='--', label=f'Limiar x₃ ({seed.haken.slaving_threshold})')

    ax4.set_xlabel('Iteração')
    ax4.set_ylabel('População')
    ax4.set_title('Evolução das Populações (Memória Evolutiva)', fontsize=12)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)

    # =========================================================================
    # Resumo
    # =========================================================================
    summary = (
        f"SEED | σ={result['sigma']:.3f} | "
        f"x₁={x1:.2f} x₂={x2:.2f} x₃={x3:.2f} | "
        f"Escravização: {'ATIVO' if result['slaving_active'] else 'não'} | "
        f"Sinal: {result['signal_name']}"
    )

    color = 'green' if result['signal'] == 1 else 'red' if result['signal'] == -1 else 'lightblue'
    fig.text(0.5, 0.01, summary, fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo: {save_path}")

    return fig


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SINTETIZADOR EVOLUTIVO DE ESTRUTURAS DISSIPATIVAS (SEED)")
    print("Termodinâmica Estatística / Biologia Evolutiva")
    print("=" * 70)

    np.random.seed(42)

    # Simular dados com diferentes regimes
    # Fase 1: Equilíbrio (baixa entropia)
    equilibrium = 1.1 + 0.0001 * np.cumsum(np.random.randn(80))

    # Fase 2: Formação de estrutura dissipativa (alta entropia, tendência)
    # Movimento organizado (Prigogine)
    structure = equilibrium[-1] + 0.0004 * np.cumsum(np.ones(70) + 0.2 * np.random.randn(70))

    # Fase 3: Colapso da estrutura
    collapse = structure[-1] + 0.0002 * np.cumsum(np.random.randn(50))

    prices = np.concatenate([equilibrium, structure, collapse])

    # Volumes simulados
    volumes = np.abs(np.diff(prices, prepend=prices[0])) * 50000 + np.random.rand(len(prices)) * 5000
    # Aumentar volume na formação de estrutura
    volumes[80:150] *= 2

    print(f"\nDados: {len(prices)} pontos")
    print(f"Preço: {prices[0]:.5f} → {prices[-1]:.5f}")

    # Criar sintetizador
    seed = SintetizadorEvolutivoEstruturasDissipativas(
        critical_sigma=1.5,
        slaving_threshold=0.15,
        fitness_window=20
    )

    print("\n" + "-" * 40)
    print("Executando análise SEED com memória evolutiva...")
    print("-" * 40)

    # Simular evolução temporal (memória)
    print("\n  Simulando evolução temporal...")
    for i in range(30, len(prices), 20):
        _ = seed.analyze(prices[:i], volumes[:i])

    # Análise final
    result = seed.analyze(prices, volumes)

    # Mostrar resultados
    print("\n RESULTADO:")
    print(f"   Sinal: {result['signal_name']}")
    print(f"   Confiança: {result['confidence']:.0%}")

    print("\n TERMODINÂMICA:")
    print(f"   σ (Produção de Entropia): {result['sigma']:.4f}")
    print(f"   Regime Não-Linear: {'SIM' if result['is_nonlinear_regime'] else 'NÃO'}")
    print(f"   Estrutura Formando: {'SIM' if result['structure_forming'] else 'NÃO'}")

    print("\n DINÂMICA DE REPLICADOR:")
    print(f"   x₁ (Bulls): {result['x1_bulls']:.3f} (ẋ₁={result['dx1_bulls']:+.4f})")
    print(f"   x₂ (Bears): {result['x2_bears']:.3f} (ẋ₂={result['dx2_bears']:+.4f})")
    print(f"   x₃ (Noise): {result['x3_mean_reverters']:.3f} (ẋ₃={result['dx3_mean_reverters']:+.4f})")
    print(f"   Espécie Dominante: {result['dominant_species']}")

    print("\n PRINCÍPIO DE ESCRAVIZAÇÃO (HAKEN):")
    print(f"   Escravização Ativa: {'SIM' if result['slaving_active'] else 'NÃO'}")
    print(f"   Modo de Ordem: {result['order_mode']}")
    print(f"   Ruído -> Sinal: {'SIM' if result['noise_became_signal'] else 'NÃO'}")

    print("\n ESPAÇO DE FASE:")
    print(f"   Região Atual: {result['phase_region']}")
    print(f"   Fronteira Cruzada: {'SIM' if result['boundary_crossed'] else 'NÃO'}")
    print(f"   Ponto de Ignição: {'SIM!' if result['is_ignition_point'] else 'NÃO'}")

    print("\n" + "=" * 70)
    if result['signal'] == 1:
        print("COMPRA - ESTRUTURA DISSIPATIVA BULL!")
        print("   σ > limiar crítico")
        print("   Bulls crescendo exponencialmente")
        print("   Mean Reverters escravizados")
    elif result['signal'] == -1:
        print("VENDA - ESTRUTURA DISSIPATIVA BEAR!")
        print("   Sistema organizando-se para baixo")
    elif result['structure_forming']:
        print("ESTRUTURA DISSIPATIVA EM FORMAÇÃO!")
        print("   Auto-organização em progresso")
    else:
        print("NEUTRO - Equilíbrio termodinâmico")
        print(f"   σ = {result['sigma']:.3f}")
    print("=" * 70)
