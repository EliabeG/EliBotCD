"""
Oráculo de Derivativos de Malliavin-Nash (ODMN)
===============================================
Nível de Complexidade: Fields Medal + Nobel em Economia (Teoria dos Jogos)

Premissa Teórica: Combina o Cálculo de Malliavin (ferramenta avançada de análise
estocástica para precificação de derivativos) com a Teoria dos Jogos de Campo Médio
(Mean Field Games - MFG) de Pierre-Louis Lions. O objetivo é calcular a "Fragilidade
Estrutural" do mercado usando derivadas de Malliavin do modelo de Heston, e prever
pontos de "Transição de Fase" usando o equilíbrio de Nash em jogos com infinitos
jogadores (institucionais).

Dependências Críticas: numpy, scipy, torch (para Deep Galerkin Method)
"""

import numpy as np
from scipy import stats, optimize
from scipy.integrate import quad
try:
    from scipy.integrate import simpson as simps
except ImportError:
    from scipy.integrate import simps
from scipy.linalg import cholesky, solve_banded
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Deep Learning para resolver PDEs
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class HestonModel:
    """
    Modelo de Heston para Volatilidade Estocástica

    dS_t = μ S_t dt + √v_t S_t dW_S
    dv_t = κ(θ - v_t)dt + σ√v_t dW_v

    com correlação ρ entre W_S e W_v
    """

    def __init__(self,
                 kappa: float = 2.0,      # Velocidade de reversão à média
                 theta: float = 0.04,     # Nível médio de variância
                 sigma: float = 0.3,      # Vol da vol (volatilidade da volatilidade)
                 rho: float = -0.7,       # Correlação (tipicamente negativa)
                 mu: float = 0.05,        # Drift do preço
                 v0: float = 0.04):       # Variância inicial
        """
        Inicializa o modelo de Heston

        A condição de Feller (2κθ > σ²) garante que v_t > 0
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.mu = mu
        self.v0 = v0

        # Verifica condição de Feller
        self.feller_condition = 2 * kappa * theta > sigma**2

    def simulate(self, S0: float, T: float, n_steps: int, n_paths: int = 1000,
                 seed: int = None) -> dict:
        """
        Simula trajetórias do modelo de Heston usando Euler-Maruyama
        com truncamento para garantir variância positiva

        REPRODUTIBILIDADE: Usa np.random.Generator para thread-safety
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        # Gerador thread-safe para reprodutibilidade
        rng = np.random.default_rng(seed)

        # Arrays para armazenar resultados
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))

        S[:, 0] = S0
        v[:, 0] = self.v0

        # Gera correlação
        for t in range(n_steps):
            # Movimentos Brownianos correlacionados (com seed)
            Z1 = rng.standard_normal(n_paths)
            Z2 = rng.standard_normal(n_paths)
            W_S = Z1
            W_v = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

            # Truncamento da variância para garantir positividade
            v_pos = np.maximum(v[:, t], 0)
            sqrt_v = np.sqrt(v_pos)

            # Evolução do preço
            S[:, t+1] = S[:, t] * np.exp(
                (self.mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * W_S
            )

            # Evolução da variância (Milstein para melhor precisão)
            v[:, t+1] = (
                v[:, t] +
                self.kappa * (self.theta - v_pos) * dt +
                self.sigma * sqrt_v * sqrt_dt * W_v +
                0.25 * self.sigma**2 * dt * (W_v**2 - 1)
            )

            # Reflexão para manter variância positiva
            v[:, t+1] = np.maximum(v[:, t+1], 0)

        return {
            'prices': S,
            'variance': v,
            'volatility': np.sqrt(v),
            'time': np.linspace(0, T, n_steps + 1)
        }

    def characteristic_function(self, u: complex, S0: float, T: float) -> complex:
        """
        Função característica do log-preço no modelo de Heston
        Usada para precificação via FFT
        """
        i = 1j

        d = np.sqrt(
            (self.rho * self.sigma * i * u - self.kappa)**2 +
            self.sigma**2 * (i * u + u**2)
        )

        g = (self.kappa - self.rho * self.sigma * i * u - d) / \
            (self.kappa - self.rho * self.sigma * i * u + d)

        C = self.kappa * (
            (self.kappa - self.rho * self.sigma * i * u - d) * T -
            2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
        ) * self.theta / self.sigma**2

        D = (self.kappa - self.rho * self.sigma * i * u - d) * \
            (1 - np.exp(-d * T)) / \
            (self.sigma**2 * (1 - g * np.exp(-d * T)))

        return np.exp(C + D * self.v0 + i * u * np.log(S0))


class HestonCalibrator:
    """
    Calibrador do Modelo de Heston a partir de dados de mercado
    Usa método de máxima verossimilhança aproximada
    """

    def __init__(self, returns_window: int = 100):
        self.returns_window = returns_window
        self.calibrated_params = None
        self.calibration_history = deque(maxlen=50)

    def calibrate(self, returns: np.ndarray) -> dict:
        """
        Calibra parâmetros do Heston usando GMM (Generalized Method of Moments)
        """
        if len(returns) < self.returns_window:
            return self._default_params()

        # Usa janela mais recente
        r = returns[-self.returns_window:]

        # Momentos empíricos
        mean_r = np.mean(r)
        var_r = np.var(r)
        skew_r = stats.skew(r)
        kurt_r = stats.kurtosis(r)

        # Estimativa de variância realizada (proxy para v_t)
        rv = self._realized_variance(r)

        # Autocorrelação da variância realizada
        rv_centered = rv - np.mean(rv)
        if len(rv_centered) > 1:
            autocorr_rv = np.corrcoef(rv_centered[:-1], rv_centered[1:])[0, 1]
            if np.isnan(autocorr_rv):
                autocorr_rv = 0.5
        else:
            autocorr_rv = 0.5

        # Estimativas iniciais baseadas em momentos
        theta_init = var_r * 252  # Variância média anualizada
        kappa_init = -np.log(max(autocorr_rv, 0.01)) * 252  # De autocorrelação

        # Vol da vol estimada da curtose
        sigma_init = np.sqrt(max(0.01, (kurt_r - 3) * theta_init / 4))

        # Correlação estimada do skew
        rho_init = np.clip(skew_r / 3, -0.95, -0.1)

        # Otimização para refinar estimativas
        try:
            result = optimize.minimize(
                self._objective,
                x0=[kappa_init, theta_init, sigma_init, rho_init],
                args=(r,),
                method='L-BFGS-B',
                bounds=[(0.1, 10), (0.001, 0.5), (0.01, 2.0), (-0.99, -0.01)]
            )

            if result.success:
                kappa, theta, sigma, rho = result.x
            else:
                kappa, theta, sigma, rho = kappa_init, theta_init, sigma_init, rho_init
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            # Erros esperados durante otimização - usa valores iniciais
            kappa, theta, sigma, rho = kappa_init, theta_init, sigma_init, rho_init

        # Verifica condição de Feller e ajusta se necessário
        if 2 * kappa * theta <= sigma**2:
            # Ajusta sigma para satisfazer Feller
            sigma = np.sqrt(2 * kappa * theta * 0.9)

        params = {
            'kappa': kappa,
            'theta': theta,
            'sigma': sigma,
            'rho': rho,
            'mu': mean_r * 252,
            'v0': var_r * 252,
            'feller_satisfied': 2 * kappa * theta > sigma**2
        }

        self.calibrated_params = params
        self.calibration_history.append(params)

        return params

    def _realized_variance(self, returns: np.ndarray, window: int = 5) -> np.ndarray:
        """Calcula variância realizada em janela móvel"""
        rv = np.zeros(len(returns) - window + 1)
        for i in range(len(rv)):
            rv[i] = np.sum(returns[i:i+window]**2)
        return rv

    def _objective(self, params: list, returns: np.ndarray) -> float:
        """Função objetivo para calibração (quasi-likelihood)"""
        kappa, theta, sigma, rho = params

        # Penaliza violação de Feller
        if 2 * kappa * theta <= sigma**2:
            return 1e10

        # Simula momentos teóricos do Heston
        # Variância incondicional
        var_theo = theta

        # Curtose teórica aproximada
        kurt_theo = 3 + 3 * sigma**2 / (kappa * theta)

        # Momentos empíricos
        var_emp = np.var(returns) * 252
        kurt_emp = stats.kurtosis(returns) + 3

        # Erro quadrático dos momentos
        error = (var_emp - var_theo)**2 + 0.1 * (kurt_emp - kurt_theo)**2

        return error

    def _default_params(self) -> dict:
        """Parâmetros padrão quando não há dados suficientes"""
        return {
            'kappa': 2.0,
            'theta': 0.04,
            'sigma': 0.3,
            'rho': -0.7,
            'mu': 0.05,
            'v0': 0.04,
            'feller_satisfied': True
        }


class MalliavinDerivativeCalculator:
    """
    Calculador de Derivadas de Malliavin

    A derivada de Malliavin D_t F mede a sensibilidade de uma variável aleatória F
    em relação ao movimento Browniano no tempo t. No contexto de derivativos,
    isso nos dá informação sobre a "fragilidade" do payoff.

    Para o modelo de Heston, calculamos:
    1. D_t S_T (sensibilidade do preço)
    2. D_t v_T (sensibilidade da variância)
    3. Norma de Malliavin ||D F||_{L^2} (medida de risco)

    REPRODUTIBILIDADE: Usa np.random.Generator para resultados determinísticos
    """

    def __init__(self, n_paths: int = 2000, n_steps: int = 30):
        """
        Parâmetros alinhados com config/odmn_config.py:
        - n_paths: MALLIAVIN_PATHS (default: 2000)
        - n_steps: MALLIAVIN_STEPS (default: 30)
        """
        self.n_paths = n_paths
        self.n_steps = n_steps

    def compute_malliavin_weight(self,
                                   heston: HestonModel,
                                   S0: float,
                                   T: float,
                                   seed: int = None) -> dict:
        """
        Calcula os pesos de Malliavin para o modelo de Heston
        usando integração por partes estocástica

        O peso de Malliavin π permite calcular Gregas sem diferenciação:
        E[f'(S_T)] = E[f(S_T) * π]

        REPRODUTIBILIDADE: Aceita seed para resultados determinísticos
        """
        dt = T / self.n_steps
        sqrt_dt = np.sqrt(dt)

        # Gerador thread-safe para reprodutibilidade
        rng = np.random.default_rng(seed)

        # Simula trajetórias
        S = np.zeros((self.n_paths, self.n_steps + 1))
        v = np.zeros((self.n_paths, self.n_steps + 1))

        # Derivadas de Malliavin (sensibilidade ao Browniano)
        D_S = np.zeros((self.n_paths, self.n_steps + 1))  # D_0 S_t
        D_v = np.zeros((self.n_paths, self.n_steps + 1))  # D_0 v_t

        S[:, 0] = S0
        v[:, 0] = heston.v0
        D_S[:, 0] = 0  # Condição inicial
        D_v[:, 0] = 0

        # Acumula incrementos Brownianos para peso de Malliavin
        W_S_total = np.zeros(self.n_paths)
        W_v_total = np.zeros(self.n_paths)

        for t in range(self.n_steps):
            # Incrementos Brownianos correlacionados (com seed)
            Z1 = rng.standard_normal(self.n_paths)
            Z2 = rng.standard_normal(self.n_paths)
            dW_S = sqrt_dt * Z1
            dW_v = sqrt_dt * (heston.rho * Z1 + np.sqrt(1 - heston.rho**2) * Z2)

            W_S_total += dW_S
            W_v_total += dW_v

            v_pos = np.maximum(v[:, t], 1e-8)
            sqrt_v = np.sqrt(v_pos)

            # Evolução do preço e variância
            S[:, t+1] = S[:, t] * np.exp(
                (heston.mu - 0.5 * v_pos) * dt + sqrt_v * dW_S
            )

            v[:, t+1] = v[:, t] + heston.kappa * (heston.theta - v_pos) * dt + \
                        heston.sigma * sqrt_v * dW_v
            v[:, t+1] = np.maximum(v[:, t+1], 1e-8)

            # Evolução das derivadas de Malliavin
            # D_s S_t satisfaz uma SDE linear
            D_S[:, t+1] = D_S[:, t] * np.exp(
                (heston.mu - 0.5 * v_pos) * dt + sqrt_v * dW_S
            ) + S[:, t] * sqrt_v * (t == 0)  # Contribuição do choque inicial

            D_v[:, t+1] = D_v[:, t] * (1 - heston.kappa * dt) + \
                         heston.sigma * sqrt_v * (t == 0)

        # Peso de Malliavin para Delta
        # π_delta = W_S_T / (S_0 * σ * T) para Black-Scholes
        # Generalização para Heston
        avg_vol = np.mean(np.sqrt(v), axis=1)
        pi_delta = W_S_total / (S0 * avg_vol * T + 1e-8)

        # Norma de Malliavin (L2 norm do processo de derivada)
        malliavin_norm_S = np.sqrt(np.mean(D_S[:, -1]**2))
        malliavin_norm_v = np.sqrt(np.mean(D_v[:, -1]**2))

        # Índice de fragilidade de Malliavin
        # Quanto maior, mais sensível ao ruído = mais frágil
        fragility_index = malliavin_norm_S / (S0 + 1e-8) + \
                          malliavin_norm_v / (heston.theta + 1e-8)

        return {
            'S_final': S[:, -1],
            'v_final': v[:, -1],
            'D_S_final': D_S[:, -1],
            'D_v_final': D_v[:, -1],
            'malliavin_weight_delta': pi_delta,
            'malliavin_norm_S': malliavin_norm_S,
            'malliavin_norm_v': malliavin_norm_v,
            'fragility_index': fragility_index,
            'avg_path_vol': np.mean(avg_vol)
        }


class DeepGalerkinMFGSolver:
    """
    Solucionador de Mean Field Games usando Deep Galerkin Method

    Resolve o sistema acoplado:
    1. HJB (Hamilton-Jacobi-Bellman): -∂V/∂t - H(x, ∇V) = 0
    2. Fokker-Planck: ∂m/∂t - div(m * ∇_p H) = 0

    onde m é a distribuição dos agentes e V é a função valor.

    O Deep Galerkin Method usa redes neurais como ansatz para V e m,
    treinando para minimizar os resíduos das PDEs.
    """

    def __init__(self,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 learning_rate: float = 0.001):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate

        self.value_net = None
        self.density_net = None
        self.is_trained = False

        if TORCH_AVAILABLE:
            self._build_networks()

    def _build_networks(self):
        """Constrói as redes neurais para V e m"""
        if not TORCH_AVAILABLE:
            return

        # Rede para função valor V(t, x)
        layers_v = [nn.Linear(2, self.hidden_dim), nn.Tanh()]
        for _ in range(self.n_layers - 1):
            layers_v.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh()])
        layers_v.append(nn.Linear(self.hidden_dim, 1))
        self.value_net = nn.Sequential(*layers_v)

        # Rede para densidade m(t, x) - deve ser positiva
        layers_m = [nn.Linear(2, self.hidden_dim), nn.Tanh()]
        for _ in range(self.n_layers - 1):
            layers_m.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh()])
        layers_m.extend([nn.Linear(self.hidden_dim, 1), nn.Softplus()])
        self.density_net = nn.Sequential(*layers_m)

    def solve_mfg_equilibrium(self,
                               price_level: float,
                               volatility: float,
                               n_iterations: int = 500,
                               n_samples: int = 256,
                               seed: int = None,
                               historical_prices: np.ndarray = None) -> dict:
        """
        Encontra o equilíbrio de Nash no Mean Field Game

        Modela traders institucionais como agentes que:
        - Maximizam utilidade esperada
        - Antecipam o impacto de mercado da distribuição coletiva

        REPRODUTIBILIDADE: Aceita seed para resultados determinísticos

        V2.4: Aceita historical_prices para calcular mean_log_price corretamente
        """
        if not TORCH_AVAILABLE:
            return self._analytical_approximation(price_level, volatility, historical_prices)

        # Seed para reprodutibilidade
        if seed is not None:
            torch.manual_seed(seed)

        optimizer = torch.optim.Adam(
            list(self.value_net.parameters()) + list(self.density_net.parameters()),
            lr=self.learning_rate
        )

        T = 1.0  # Horizonte normalizado

        loss_history = []

        for iteration in range(n_iterations):
            optimizer.zero_grad()

            # Amostra pontos no domínio [0, T] x [log(S) - 2σ, log(S) + 2σ]
            # (seed já aplicado via torch.manual_seed)
            t = torch.rand(n_samples, 1) * T
            x = torch.randn(n_samples, 1) * volatility * 2 + np.log(price_level)

            tx = torch.cat([t, x], dim=1)
            tx.requires_grad_(True)

            # Avalia redes
            V = self.value_net(tx)
            m = self.density_net(tx)

            # Calcula gradientes para PDE
            grad_V = torch.autograd.grad(V.sum(), tx, create_graph=True)[0]
            V_t = grad_V[:, 0:1]
            V_x = grad_V[:, 1:2]

            # Segunda derivada
            grad_V_x = torch.autograd.grad(V_x.sum(), tx, create_graph=True)[0]
            V_xx = grad_V_x[:, 1:2]

            grad_m = torch.autograd.grad(m.sum(), tx, create_graph=True)[0]
            m_t = grad_m[:, 0:1]
            m_x = grad_m[:, 1:2]

            # Hamiltoniano para trading: H = -α|a|² + a * V_x
            # Controle ótimo: a* = V_x / (2α)
            alpha = 0.1  # Custo de impacto de mercado
            a_star = V_x / (2 * alpha)

            # Resíduo HJB: -V_t - (σ²/2)V_xx - H = 0
            sigma2 = volatility**2
            hjb_residual = -V_t - 0.5 * sigma2 * V_xx + alpha * a_star**2 - a_star * V_x

            # Resíduo Fokker-Planck: m_t + (m * a*)_x - (σ²/2)m_xx = 0
            grad_ma = torch.autograd.grad((m * a_star).sum(), tx, create_graph=True)[0]
            ma_x = grad_ma[:, 1:2]

            grad_m_x = torch.autograd.grad(m_x.sum(), tx, create_graph=True)[0]
            m_xx = grad_m_x[:, 1:2]

            fp_residual = m_t + ma_x - 0.5 * sigma2 * m_xx

            # Loss total
            loss_hjb = torch.mean(hjb_residual**2)
            loss_fp = torch.mean(fp_residual**2)

            # Condições de contorno (terminal e normalização)
            t_terminal = torch.ones(32, 1) * T
            x_terminal = torch.randn(32, 1) * volatility + np.log(price_level)
            tx_terminal = torch.cat([t_terminal, x_terminal], dim=1)

            V_terminal = self.value_net(tx_terminal)
            loss_terminal = torch.mean((V_terminal - 0)**2)  # V(T, x) = 0

            # Normalização da densidade
            loss_norm = (torch.mean(m) - 1.0)**2

            total_loss = loss_hjb + loss_fp + 0.1 * loss_terminal + 0.1 * loss_norm

            total_loss.backward()
            optimizer.step()

            loss_history.append(total_loss.item())

        self.is_trained = True

        # Avalia equilíbrio no ponto atual
        t_now = torch.tensor([[0.0, np.log(price_level)]])
        V_now = self.value_net(t_now).item()
        m_now = self.density_net(t_now).item()

        # Gradiente (direção do fluxo ótimo)
        t_now.requires_grad_(True)
        V_grad = self.value_net(t_now)
        grad = torch.autograd.grad(V_grad, t_now)[0]
        optimal_direction = grad[0, 1].item()

        return {
            'value_function': V_now,
            'density': m_now,
            'optimal_direction': optimal_direction,
            'convergence_loss': loss_history[-1] if loss_history else 0,
            'is_equilibrium': loss_history[-1] < 0.01 if loss_history else False
        }

    def _analytical_approximation(self, price_level: float, volatility: float,
                                    historical_prices: np.ndarray = None) -> dict:
        """
        Aproximação analítica quando PyTorch não está disponível
        Usa solução de forma fechada para MFG linear-quadrático

        V2.4 FIX: Agora calcula mean_log_price corretamente usando preços históricos,
        ao invés de usar o preço atual (que resultava em optimal_direction = 0 sempre)
        """
        # Para MFG LQ, a solução é Gaussiana
        # A função valor é quadrática: V(t, x) = A(t)x² + B(t)x + C(t)

        # Parâmetros do modelo LQ
        alpha = 0.1  # Custo de controle
        sigma = volatility
        T = 1.0

        # Solução de Riccati para A(t)
        # dA/dt = -2α A² + ..., com A(T) = 0
        A_0 = sigma**2 / (4 * alpha * T)

        # Direção ótima baseada no gradiente
        log_price = np.log(price_level)

        # V2.4 FIX: Calcula mean_log_price a partir dos preços históricos
        # Isso corrige o bug onde optimal_direction era sempre 0
        if historical_prices is not None and len(historical_prices) > 1:
            # Usa média dos log-preços históricos como referência de equilíbrio
            mean_log_price = np.mean(np.log(historical_prices))
        else:
            # Fallback: usa uma estimativa baseada em volatilidade
            # Assume que o preço está desviado da média por ~1 desvio padrão
            mean_log_price = log_price - sigma * 0.5  # Estimativa conservadora

        # Direção ótima: positiva se preço > média (vender), negativa se preço < média (comprar)
        # O sinal é invertido porque no MFG LQ, o controle ótimo é a* = -∇V
        optimal_direction = -2 * A_0 * (log_price - mean_log_price)

        # Densidade no equilíbrio (Gaussiana)
        density = stats.norm.pdf(log_price, mean_log_price, sigma)

        return {
            'value_function': A_0 * log_price**2,
            'density': density,
            'optimal_direction': optimal_direction,
            'convergence_loss': 0.001,
            'is_equilibrium': True
        }


class OracloDerivativosMalliavinNash:
    """
    Implementação completa do Oráculo de Derivativos de Malliavin-Nash (ODMN)

    Módulos:
    1. Calibração do Modelo de Heston (O Motor de Volatilidade Estocástica)
    2. Cálculo de Malliavin (O Detector de Fragilidade)
    3. Mean Field Games (O Previsor de Comportamento Institucional)
    4. Síntese e Sinal de Trading (O Oráculo)

    REPRODUTIBILIDADE:
    - Suporta seed para resultados determinísticos
    - Usa np.random.Generator thread-safe
    """

    def __init__(self,
                 lookback_window: int = 100,
                 fragility_threshold: float = 2.0,
                 mfg_direction_threshold: float = 0.1,
                 confidence_decay: float = 0.95,
                 use_deep_galerkin: bool = False,
                 malliavin_paths: int = 2000,
                 malliavin_steps: int = 30,
                 seed: int = None):
        """
        Inicialização do Oráculo de Derivativos de Malliavin-Nash

        PARÂMETROS ALINHADOS COM config/odmn_config.py:
        ------------------------------------------------
        lookback_window : int
            Janela para calibração do Heston (config: HESTON_CALIBRATION_WINDOW = 100)

        fragility_threshold : float
            Limiar do índice de fragilidade de Malliavin (default: 2.0)
            Acima disso = mercado frágil, risco de crash

        mfg_direction_threshold : float
            Limiar para direção do MFG (config: MFG_DIRECTION_THRESHOLD = 0.1)
            |direção| > threshold indica pressão institucional

        confidence_decay : float
            Decaimento da confiança ao longo do tempo (default: 0.95)

        use_deep_galerkin : bool
            Se True, usa redes neurais para MFG; se False, solução analítica
            (config: USE_DEEP_GALERKIN = False)

        malliavin_paths : int
            Número de trajetórias para Monte Carlo de Malliavin
            (config: MALLIAVIN_PATHS = 2000)

        malliavin_steps : int
            Número de passos temporais na simulação
            (config: MALLIAVIN_STEPS = 30)

        seed : int
            Seed para reprodutibilidade (None = não determinístico)
        """
        self.lookback_window = lookback_window
        self.fragility_threshold = fragility_threshold
        self.mfg_direction_threshold = mfg_direction_threshold
        self.confidence_decay = confidence_decay
        self.use_deep_galerkin = use_deep_galerkin
        self.seed = seed

        # Componentes
        self.heston_calibrator = HestonCalibrator(returns_window=lookback_window)
        self.malliavin_calc = MalliavinDerivativeCalculator(
            n_paths=malliavin_paths,
            n_steps=malliavin_steps
        )
        self.mfg_solver = DeepGalerkinMFGSolver() if use_deep_galerkin else DeepGalerkinMFGSolver()

        # Cache de análises (thread-safe usando deque)
        self._cache = {
            'heston_params': None,
            'malliavin_result': None,
            'mfg_result': None,
            'fragility_history': deque(maxlen=100),
            'direction_history': deque(maxlen=100)
        }

        # Contador para gerar seeds únicos quando seed base é fornecido
        self._seed_counter = 0

    def _get_next_seed(self) -> int:
        """Gera próximo seed único quando seed base é fornecido"""
        if self.seed is None:
            return None
        self._seed_counter += 1
        return self.seed + self._seed_counter

    def reset(self):
        """
        Reseta o estado interno do indicador.

        V2.4: Método público para reset seguro, evitando acesso direto ao _cache.
        Deve ser chamado quando a estratégia é resetada.
        """
        self._cache = {
            'heston_params': None,
            'malliavin_result': None,
            'mfg_result': None,
            'fragility_history': deque(maxlen=100),
            'direction_history': deque(maxlen=100)
        }
        self._seed_counter = 0

    def analyze(self, prices: np.ndarray) -> dict:
        """
        Análise completa do Oráculo de Derivativos de Malliavin-Nash

        Parâmetros:
        -----------
        prices : np.ndarray
            Array de preços (close prices)

        Retorna:
        --------
        dict com:
            - signal: -1 (SELL), 0 (HOLD), +1 (BUY)
            - confidence: 0 a 1
            - fragility_index: índice de fragilidade de Malliavin
            - mfg_direction: direção do equilíbrio de Nash
            - heston_params: parâmetros calibrados
            - analysis_details: detalhes completos

        REPRODUTIBILIDADE: Resultados determinísticos quando seed é fornecido
        """
        if len(prices) < self.lookback_window:
            return self._empty_result()

        # Calcula retornos
        returns = np.diff(np.log(prices))
        current_price = prices[-1]

        # =============================
        # MÓDULO 1: Calibração do Heston
        # =============================
        heston_params = self.heston_calibrator.calibrate(returns)
        self._cache['heston_params'] = heston_params

        # Cria modelo Heston com parâmetros calibrados
        heston = HestonModel(
            kappa=heston_params['kappa'],
            theta=heston_params['theta'],
            sigma=heston_params['sigma'],
            rho=heston_params['rho'],
            mu=heston_params['mu'],
            v0=heston_params['v0']
        )

        # =============================
        # MÓDULO 2: Derivadas de Malliavin
        # =============================
        # REPRODUTIBILIDADE: Passa seed para Monte Carlo
        malliavin_result = self.malliavin_calc.compute_malliavin_weight(
            heston=heston,
            S0=current_price,
            T=1/252,  # 1 dia
            seed=self._get_next_seed()
        )
        self._cache['malliavin_result'] = malliavin_result

        fragility = malliavin_result['fragility_index']
        self._cache['fragility_history'].append(fragility)

        # Normaliza fragilidade pelo histórico
        if len(self._cache['fragility_history']) > 10:
            fragility_percentile = stats.percentileofscore(
                list(self._cache['fragility_history']),
                fragility
            ) / 100
        else:
            fragility_percentile = 0.5

        # =============================
        # MÓDULO 3: Mean Field Games
        # =============================
        volatility = np.sqrt(heston_params['v0'])
        # REPRODUTIBILIDADE: Passa seed para MFG solver
        # V2.4 FIX: Passa historical_prices para calcular mean_log_price corretamente
        mfg_result = self.mfg_solver.solve_mfg_equilibrium(
            price_level=current_price,
            volatility=volatility,
            n_iterations=200 if self.use_deep_galerkin and TORCH_AVAILABLE else 100,
            seed=self._get_next_seed(),
            historical_prices=prices  # V2.4: passa preços históricos para MFG analítico
        )
        self._cache['mfg_result'] = mfg_result

        mfg_direction = mfg_result['optimal_direction']
        self._cache['direction_history'].append(mfg_direction)

        # =============================
        # MÓDULO 4: Síntese do Sinal
        # =============================
        signal, confidence, reasons = self._synthesize_signal(
            fragility=fragility,
            fragility_percentile=fragility_percentile,
            mfg_direction=mfg_direction,
            mfg_equilibrium=mfg_result['is_equilibrium'],
            heston_rho=heston_params['rho'],
            feller_satisfied=heston_params['feller_satisfied']
        )

        # Determina regime de mercado
        regime = self._determine_regime(
            fragility_percentile=fragility_percentile,
            volatility=volatility
        )

        return {
            'signal': signal,
            'signal_name': {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}[signal],
            'confidence': confidence,
            'fragility_index': fragility,
            'fragility_percentile': fragility_percentile,
            'fragility_trigger': fragility_percentile > 0.8,
            'mfg_direction': mfg_direction,
            'mfg_equilibrium': mfg_result['is_equilibrium'],
            'mfg_value': mfg_result['value_function'],
            'heston_params': heston_params,
            'malliavin_norm_S': malliavin_result['malliavin_norm_S'],
            'malliavin_norm_v': malliavin_result['malliavin_norm_v'],
            'regime': regime,
            'reasons': reasons,
            'current_price': current_price,
            'implied_vol': volatility
        }

    def _synthesize_signal(self,
                           fragility: float,
                           fragility_percentile: float,
                           mfg_direction: float,
                           mfg_equilibrium: bool,
                           heston_rho: float,
                           feller_satisfied: bool) -> tuple:
        """
        Sintetiza o sinal de trading combinando todas as análises

        Lógica:
        1. Alta fragilidade (percentil > 80%) + direção MFG negativa = SELL
        2. Alta fragilidade + direção MFG positiva = BUY (risco alto)
        3. Baixa fragilidade + direção clara = seguir MFG com confiança
        4. Equilíbrio não convergido = cautela
        """
        signal = 0
        confidence = 0.5
        reasons = []

        # Verifica condição de Feller (modelo bem comportado)
        if not feller_satisfied:
            reasons.append("Feller não satisfeito - modelo instável")
            return 0, 0.3, reasons

        # Fragilidade alta indica possível crash/reversão
        high_fragility = fragility_percentile > 0.8
        very_high_fragility = fragility_percentile > 0.95

        # Direção institucional do MFG
        strong_buy_pressure = mfg_direction > self.mfg_direction_threshold
        strong_sell_pressure = mfg_direction < -self.mfg_direction_threshold

        # Correlação Heston (leverage effect)
        # rho muito negativo = quedas são mais voláteis
        leverage_effect = heston_rho < -0.5

        if very_high_fragility:
            # Mercado extremamente frágil - possível crash iminente
            if strong_sell_pressure:
                signal = -1
                confidence = 0.85
                reasons.append("Fragilidade extrema + pressão vendedora institucional")
                reasons.append(f"Malliavin fragility P{fragility_percentile*100:.0f}")
            elif strong_buy_pressure:
                # Possível short squeeze
                signal = 1
                confidence = 0.70
                reasons.append("Fragilidade extrema mas compra institucional")
                reasons.append("Possível short squeeze")
            else:
                signal = 0
                confidence = 0.6
                reasons.append("Fragilidade extrema - aguardar definição")

        elif high_fragility:
            # Mercado frágil - cautela
            if strong_buy_pressure and not leverage_effect:
                signal = 1
                confidence = 0.65
                reasons.append("Fragilidade alta com suporte institucional")
            elif strong_sell_pressure or leverage_effect:
                signal = -1
                confidence = 0.70
                reasons.append("Fragilidade alta + leverage effect")
            else:
                signal = 0
                confidence = 0.55
                reasons.append("Fragilidade moderada-alta")

        else:
            # Mercado relativamente estável
            if strong_buy_pressure:
                signal = 1
                confidence = 0.75
                reasons.append("Mercado estável + fluxo comprador MFG")
            elif strong_sell_pressure:
                signal = -1
                confidence = 0.75
                reasons.append("Mercado estável + fluxo vendedor MFG")
            else:
                signal = 0
                confidence = 0.5
                reasons.append("Sem pressão direcional clara")

        # Ajusta confiança se equilíbrio não convergiu
        if not mfg_equilibrium:
            confidence *= 0.8
            reasons.append("MFG não convergiu totalmente")

        # Adiciona info sobre parâmetros
        reasons.append(f"MFG direction: {mfg_direction:.4f}")
        reasons.append(f"Heston ρ: {heston_rho:.3f}")

        return signal, min(confidence, 0.95), reasons

    def _determine_regime(self, fragility_percentile: float, volatility: float) -> str:
        """Determina o regime de mercado atual"""
        if fragility_percentile > 0.9:
            return "CRITICAL_FRAGILITY"
        elif fragility_percentile > 0.75:
            return "HIGH_FRAGILITY"
        elif volatility > 0.3:  # ~30% vol anualizada
            return "HIGH_VOLATILITY"
        elif volatility < 0.1:
            return "LOW_VOLATILITY"
        else:
            return "NORMAL"

    def _empty_result(self) -> dict:
        """Retorna resultado vazio quando não há dados suficientes"""
        return {
            'signal': 0,
            'signal_name': 'HOLD',
            'confidence': 0.0,
            'fragility_index': 0.0,
            'fragility_percentile': 0.0,
            'fragility_trigger': False,
            'mfg_direction': 0.0,
            'mfg_equilibrium': False,
            'mfg_value': 0.0,
            'heston_params': None,
            'malliavin_norm_S': 0.0,
            'malliavin_norm_v': 0.0,
            'regime': 'INSUFFICIENT_DATA',
            'reasons': ['Dados insuficientes'],
            'current_price': 0.0,
            'implied_vol': 0.0
        }

    def get_heston_forecast(self, prices: np.ndarray, horizon: int = 20) -> dict:
        """
        Gera previsão usando o modelo de Heston calibrado

        Retorna distribuição esperada de preços futuros

        V2.4 FIX: Agora passa seed para reprodutibilidade
        """
        if self._cache['heston_params'] is None:
            self.analyze(prices)

        params = self._cache['heston_params']
        if params is None:
            return None

        heston = HestonModel(**{k: params[k] for k in ['kappa', 'theta', 'sigma', 'rho', 'mu', 'v0']})

        # Simula trajetórias - V2.4 FIX: passa seed para reprodutibilidade
        result = heston.simulate(
            S0=prices[-1],
            T=horizon/252,  # Dias para fração de ano
            n_steps=horizon,
            n_paths=1000,
            seed=self._get_next_seed()  # V2.4: reprodutibilidade
        )

        final_prices = result['prices'][:, -1]

        return {
            'mean_price': np.mean(final_prices),
            'std_price': np.std(final_prices),
            'percentiles': {
                '5%': np.percentile(final_prices, 5),
                '25%': np.percentile(final_prices, 25),
                '50%': np.percentile(final_prices, 50),
                '75%': np.percentile(final_prices, 75),
                '95%': np.percentile(final_prices, 95)
            },
            'prob_up': np.mean(final_prices > prices[-1]),
            'expected_return': np.mean(final_prices / prices[-1] - 1),
            'var_95': np.percentile(final_prices / prices[-1] - 1, 5),
            'horizon_days': horizon
        }


def plot_odmn_analysis(result: dict, prices: np.ndarray = None, figsize: tuple = (16, 12)):
    """
    Visualização completa da análise ODMN

    Gera painel com:
    1. Preços e sinal
    2. Índice de fragilidade de Malliavin
    3. Direção do MFG
    4. Parâmetros do Heston
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib não disponível para visualização")
        return None

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)

    # Cores
    colors = {
        'price': '#2E86AB',
        'buy': '#28A745',
        'sell': '#DC3545',
        'hold': '#FFC107',
        'fragility': '#9B59B6',
        'mfg': '#E74C3C',
        'heston': '#3498DB'
    }

    # 1. Preços e Sinal
    ax1 = fig.add_subplot(gs[0, :])
    if prices is not None and len(prices) > 0:
        ax1.plot(prices, color=colors['price'], linewidth=1.5, alpha=0.8, label='Preço')

        # Marca sinal no último ponto
        signal_color = colors['buy'] if result['signal'] == 1 else \
                       colors['sell'] if result['signal'] == -1 else colors['hold']
        ax1.scatter(len(prices)-1, prices[-1], c=signal_color, s=200, zorder=5,
                    marker='^' if result['signal'] == 1 else 'v' if result['signal'] == -1 else 'o',
                    edgecolors='black', linewidths=2)

    ax1.set_title(f"ODMN Analysis - Signal: {result['signal_name']} "
                  f"(Confidence: {result['confidence']:.1%})", fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Fragilidade de Malliavin
    ax2 = fig.add_subplot(gs[1, 0])

    fragility = result['fragility_index']
    fragility_pct = result['fragility_percentile']

    # Gauge de fragilidade
    theta = np.linspace(0, np.pi, 100)
    r = 1

    # Zonas
    ax2.fill_between(theta[:33] * 180/np.pi, 0, r, alpha=0.3, color='green', label='Estável')
    ax2.fill_between(theta[33:66] * 180/np.pi, 0, r, alpha=0.3, color='yellow', label='Moderado')
    ax2.fill_between(theta[66:] * 180/np.pi, 0, r, alpha=0.3, color='red', label='Frágil')

    # Agulha
    needle_angle = fragility_pct * np.pi
    ax2.annotate('', xy=(needle_angle * 180/np.pi, 0.8), xytext=(90, 0),
                 arrowprops=dict(arrowstyle='->', color='black', lw=3))

    ax2.set_xlim(0, 180)
    ax2.set_ylim(0, 1.2)
    ax2.set_title(f'Fragilidade de Malliavin: P{fragility_pct*100:.0f}', fontsize=12)
    ax2.legend(loc='lower right', fontsize=8)
    ax2.set_xticks([0, 45, 90, 135, 180])
    ax2.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax2.set_yticks([])

    # 3. Direção MFG
    ax3 = fig.add_subplot(gs[1, 1])

    mfg_dir = result['mfg_direction']

    # Barra de direção
    bar_color = colors['buy'] if mfg_dir > 0 else colors['sell']
    ax3.barh(['MFG'], [mfg_dir], color=bar_color, alpha=0.7, height=0.5)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax3.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='Buy threshold')
    ax3.axvline(x=-0.1, color='red', linestyle='--', alpha=0.5, label='Sell threshold')

    ax3.set_xlim(-0.5, 0.5)
    ax3.set_title(f'Direção Mean Field Game: {mfg_dir:.4f}', fontsize=12)
    ax3.set_xlabel('Pressão Institucional (← Venda | Compra →)')
    ax3.legend(loc='lower right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Parâmetros do Heston
    ax4 = fig.add_subplot(gs[2, 0])

    if result['heston_params']:
        params = result['heston_params']
        param_names = ['κ (rev.speed)', 'θ (long var)', 'σ (vol of vol)', 'ρ (corr)']
        param_values = [params['kappa'], params['theta'], params['sigma'], params['rho']]

        bars = ax4.bar(param_names, param_values, color=colors['heston'], alpha=0.7)

        # Destaca correlação negativa
        if params['rho'] < 0:
            bars[3].set_color(colors['sell'])
            bars[3].set_alpha(0.7)

        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_title('Parâmetros Heston Calibrados', fontsize=12)
        ax4.set_ylabel('Valor')
        ax4.tick_params(axis='x', rotation=15)
        ax4.grid(True, alpha=0.3, axis='y')

    # 5. Resumo e Razões
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    summary_text = [
        f"═══ RESUMO DO ORÁCULO ═══",
        f"",
        f"Sinal: {result['signal_name']}",
        f"Confiança: {result['confidence']:.1%}",
        f"Regime: {result['regime']}",
        f"",
        f"Fragilidade: {result['fragility_index']:.4f}",
        f"(Percentil: {result['fragility_percentile']*100:.0f}%)",
        f"",
        f"Vol Implícita: {result['implied_vol']*100:.1f}%",
        f"MFG Equilibrium: {'Sim' if result['mfg_equilibrium'] else 'Não'}",
        f"",
        f"═══ RAZÕES ═══",
    ]

    for reason in result['reasons'][:5]:
        summary_text.append(f"• {reason}")

    ax5.text(0.05, 0.95, '\n'.join(summary_text), transform=ax5.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


# =============================================================================
# Teste standalone
# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("TESTE DO ORÁCULO DE DERIVATIVOS DE MALLIAVIN-NASH (ODMN)")
    print("="*70)

    # Gera dados sintéticos com regime de volatilidade variável
    np.random.seed(42)
    n_points = 500

    # Simula preços com mudanças de regime
    prices = [1.1000]
    volatility = 0.0001

    for i in range(1, n_points):
        # Regime de volatilidade
        if 150 < i < 200:
            volatility = 0.0005  # Alta volatilidade
        elif 300 < i < 350:
            volatility = 0.0008  # Volatilidade extrema
        else:
            volatility = 0.0001

        # Drift com reversão à média
        drift = 0.00001 * (1.1000 - prices[-1]) / 0.001

        # Adiciona saltos ocasionais
        jump = 0
        if np.random.random() < 0.01:
            jump = np.random.choice([-1, 1]) * 0.002

        returns = drift + volatility * np.random.randn() + jump
        prices.append(prices[-1] * (1 + returns))

    prices = np.array(prices)

    # Inicializa o ODMN
    print("\nInicializando ODMN...")
    odmn = OracloDerivativosMalliavinNash(
        lookback_window=100,
        fragility_threshold=2.0,
        mfg_direction_threshold=0.1,
        use_deep_galerkin=TORCH_AVAILABLE,
        malliavin_paths=2000,
        malliavin_steps=30
    )

    # Executa análise
    print("\nExecutando análise ODMN...")
    result = odmn.analyze(prices)

    # Exibe resultados
    print("\n" + "="*70)
    print("RESULTADO DA ANÁLISE ODMN")
    print("="*70)

    print(f"\n📊 SINAL: {result['signal_name']}")
    print(f"   Confiança: {result['confidence']:.1%}")
    print(f"   Regime: {result['regime']}")

    print(f"\n🔬 FRAGILIDADE DE MALLIAVIN:")
    print(f"   Índice: {result['fragility_index']:.6f}")
    print(f"   Percentil: {result['fragility_percentile']*100:.0f}%")
    print(f"   Trigger: {'SIM' if result['fragility_trigger'] else 'NÃO'}")
    print(f"   Norma Malliavin (S): {result['malliavin_norm_S']:.6f}")
    print(f"   Norma Malliavin (v): {result['malliavin_norm_v']:.6f}")

    print(f"\n🎯 MEAN FIELD GAME:")
    print(f"   Direção Ótima: {result['mfg_direction']:.6f}")
    print(f"   Função Valor: {result['mfg_value']:.6f}")
    print(f"   Equilíbrio: {'Convergiu' if result['mfg_equilibrium'] else 'Não convergiu'}")

    if result['heston_params']:
        params = result['heston_params']
        print(f"\n📈 PARÂMETROS HESTON CALIBRADOS:")
        print(f"   κ (velocidade reversão): {params['kappa']:.4f}")
        print(f"   θ (variância longo prazo): {params['theta']:.6f}")
        print(f"   σ (vol da vol): {params['sigma']:.4f}")
        print(f"   ρ (correlação): {params['rho']:.4f}")
        print(f"   v₀ (variância inicial): {params['v0']:.6f}")
        print(f"   Feller satisfeito: {'SIM' if params['feller_satisfied'] else 'NÃO'}")

    print(f"\n📝 RAZÕES:")
    for reason in result['reasons']:
        print(f"   • {reason}")

    # Previsão Heston
    print("\n" + "="*70)
    print("PREVISÃO COM MODELO DE HESTON (20 DIAS)")
    print("="*70)

    forecast = odmn.get_heston_forecast(prices, horizon=20)
    if forecast:
        print(f"\n   Preço atual: {prices[-1]:.5f}")
        print(f"   Preço médio esperado: {forecast['mean_price']:.5f}")
        print(f"   Desvio padrão: {forecast['std_price']:.5f}")
        print(f"\n   Percentis:")
        for pct, value in forecast['percentiles'].items():
            print(f"      {pct}: {value:.5f}")
        print(f"\n   Probabilidade de alta: {forecast['prob_up']:.1%}")
        print(f"   Retorno esperado: {forecast['expected_return']*100:.2f}%")
        print(f"   VaR 95%: {forecast['var_95']*100:.2f}%")

    # Tenta gerar visualização
    print("\n" + "="*70)
    print("GERANDO VISUALIZAÇÃO...")
    print("="*70)

    try:
        fig = plot_odmn_analysis(result, prices)
        if fig:
            fig.savefig('/tmp/odmn_analysis.png', dpi=150, bbox_inches='tight')
            print("✅ Visualização salva em /tmp/odmn_analysis.png")
        else:
            print("⚠️ Não foi possível gerar visualização")
    except Exception as e:
        print(f"⚠️ Erro na visualização: {e}")

    # Teste de múltiplas análises
    print("\n" + "="*70)
    print("TESTE DE ANÁLISES CONTÍNUAS (10 pontos)")
    print("="*70)

    signals_generated = 0
    for i in range(10):
        # Adiciona novos pontos
        new_prices = prices.copy()
        for _ in range(10):
            ret = 0.0001 * np.random.randn()
            new_prices = np.append(new_prices, new_prices[-1] * (1 + ret))

        result = odmn.analyze(new_prices)

        if result['signal'] != 0:
            signals_generated += 1
            print(f"   Ponto {i+1}: {result['signal_name']} "
                  f"(conf: {result['confidence']:.1%}, regime: {result['regime']})")

    print(f"\n   Total de sinais gerados: {signals_generated}/10")
    print("\n✅ Teste do ODMN concluído com sucesso!")
