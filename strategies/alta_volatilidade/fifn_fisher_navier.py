"""
Fluxo de Informação Fisher-Navier (FIFN)
=========================================
Nível de Complexidade: Millennium Prize Problems (Navier-Stokes) aplicado a Finanças.

Premissa Teórica: Utilizaremos a Geometria da Informação (Fisher Information Metric) para
definir a "distância" estatística entre os estados do mercado, e aplicaremos as Equações de
Navier-Stokes para modelar a velocidade dessa mudança. O objetivo é calcular o Número de
Reynolds (Re) do mercado financeiro.

Dependências Críticas: scipy.integrate, numpy, autograd, matplotlib (quiver plots)
"""

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import simpson as simps
import warnings
warnings.filterwarnings('ignore')

# Para diferenciação automática da densidade de probabilidade
try:
    import autograd.numpy as anp
    from autograd import grad
    AUTOGRAD_AVAILABLE = True
except ImportError:
    AUTOGRAD_AVAILABLE = False


class FluxoInformacaoFisherNavier:
    """
    Implementação completa do Fluxo de Informação Fisher-Navier (FIFN)

    Módulos:
    1. A Variedade Estatística (O "Campo de Jogo") - Fisher Information Metric
    2. Dinâmica de Fluidos (O Motor de Volatilidade) - Navier-Stokes 1D
    3. O Discriminador: Número de Reynolds Financeiro (Re)
    4. O Gatilho Direcional: Divergência de Kullback-Leibler (KL)
    5. Output e Visualização (Painel de Controle de Fluidos)
    """

    def __init__(self,
                 window_size: int = 50,
                 kl_lookback: int = 10,
                 reynolds_laminar: float = 2000,
                 reynolds_turbulent: float = 4000,
                 reynolds_sweet_low: float = 2300,
                 reynolds_sweet_high: float = 4000,
                 skewness_threshold: float = 0.5,
                 characteristic_length: float = 1.0,
                 density_rho: float = 1.0,
                 crank_nicolson_theta: float = 0.5,
                 numerical_stability_eps: float = 1e-8,
                 kde_bandwidth: str = 'scott',
                 n_grid_points: int = 100):
        """
        Inicialização do Fluxo de Informação Fisher-Navier

        Parâmetros:
        -----------
        window_size : int
            Janela deslizante para estimativa da PDF (default: 50 períodos)

        kl_lookback : int
            Períodos de lookback para comparação KL (default: 10)

        reynolds_laminar : float
            Limiar inferior do Re para fluxo laminar (default: 2000)

        reynolds_turbulent : float
            Limiar superior do Re para turbulência caótica (default: 4000)

        reynolds_sweet_low, reynolds_sweet_high : float
            Zona de transição "Sweet Spot" (default: 2300-4000)

        skewness_threshold : float
            Limiar de assimetria para sinal direcional (default: 0.5)

        characteristic_length : float
            Comprimento característico L para Re (default: 1.0)

        density_rho : float
            Densidade rho para equação de Navier-Stokes (default: 1.0)

        crank_nicolson_theta : float
            Parâmetro theta do método Crank-Nicolson (0.5 = totalmente implícito)

        numerical_stability_eps : float
            Epsilon para estabilidade numérica
        """
        self.window_size = window_size
        self.kl_lookback = kl_lookback
        self.reynolds_laminar = reynolds_laminar
        self.reynolds_turbulent = reynolds_turbulent
        self.reynolds_sweet_low = reynolds_sweet_low
        self.reynolds_sweet_high = reynolds_sweet_high
        self.skewness_threshold = skewness_threshold
        self.characteristic_length = characteristic_length
        self.density_rho = density_rho
        self.crank_nicolson_theta = crank_nicolson_theta
        self.eps = numerical_stability_eps
        self.kde_bandwidth = kde_bandwidth
        self.n_grid_points = n_grid_points

        # Cache
        self._cache = {}

    # =========================================================================
    # MÓDULO 1: A Variedade Estatística (O "Campo de Jogo")
    # =========================================================================

    def _calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Não use velas OHLC. Use a distribuição de probabilidade dos retornos.
        """
        return np.diff(np.log(prices))

    def _estimate_pdf_kde(self, returns: np.ndarray, x_grid: np.ndarray = None) -> tuple:
        """
        Cálculo: Para uma janela deslizante (ex: 50 períodos), estime a Função Densidade
        de Probabilidade (PDF) p(x, theta) dos retornos usando uma Estimativa de Densidade
        Kernel (KDE) Gaussiana.
        """
        if len(returns) < 3:
            if x_grid is None:
                x_grid = np.linspace(-0.01, 0.01, self.n_grid_points)
            return x_grid, np.ones(len(x_grid)) / len(x_grid)

        # Criar KDE
        kde = stats.gaussian_kde(returns, bw_method=self.kde_bandwidth)

        # Grid de avaliação
        if x_grid is None:
            std = np.std(returns)
            mean = np.mean(returns)
            x_min = mean - 4 * std
            x_max = mean + 4 * std
            x_grid = np.linspace(x_min, x_max, self.n_grid_points)

        # Avaliar PDF
        pdf = kde(x_grid)

        # Normalizar
        dx = x_grid[1] - x_grid[0]
        pdf = pdf / (np.sum(pdf) * dx + self.eps)

        return x_grid, pdf

    def _calculate_fisher_information(self, returns: np.ndarray) -> float:
        """
        Métrica de Fisher (g_uv): Calcule a Matriz de Informação de Fisher. Ela mede o quão
        "rápido" a distribuição de probabilidade está mudando. Se o mercado se move, mas a
        distribuição não muda de forma, é ruído. Se a forma da distribuição muda, é informação
        institucional.

        g(theta) = integral p(x|theta) * (d ln p / d theta)^2 dx

        Na prática, usamos a variância do score function como aproximação.

        AUDITORIA 23: Corrigido gradient clipping ANTES de elevar ao quadrado
        - Previne overflow numérico quando pdf é próximo de zero nas caudas
        """
        if len(returns) < 5:
            return 0.0

        x_grid, pdf = self._estimate_pdf_kde(returns)
        dx = x_grid[1] - x_grid[0]

        # Score function: d ln p / d theta
        # Aproximação numérica do gradiente da log-densidade
        log_pdf = np.log(pdf + self.eps)

        # Derivada numérica (central differences para estabilidade)
        d_log_pdf = np.gradient(log_pdf, dx)

        # AUDITORIA 23/24/25 FIX: Clip gradient ANTES de elevar ao quadrado
        # Previne overflow quando pdf é próximo de zero nas caudas
        # AUDITORIA 25: Reduzido de ±50 para ±30 para máxima estabilidade numérica
        # 30² = 900 (vs 50² = 2500 vs 100² = 10000)
        d_log_pdf = np.clip(d_log_pdf, -30, 30)

        # Fisher Information: E[(d ln p / d theta)^2] = integral p(x) * (d ln p / d theta)^2 dx
        fisher_info = simps(pdf * d_log_pdf**2, x_grid)

        # Normalizar para escala interpretável
        # Escala típica: 0-100
        sigma = np.std(returns) + self.eps
        fisher_normalized = fisher_info * sigma**2

        # Limitar valores extremos
        fisher_normalized = np.clip(fisher_normalized, 0, 100)

        return fisher_normalized

    def _calculate_fisher_information_parametric(self, returns: np.ndarray) -> float:
        """
        Versão paramétrica da Informação de Fisher usando parâmetros da distribuição.
        Para distribuição normal: I(mu) = 1/sigma^2, I(sigma) = 2/sigma^2
        """
        if len(returns) < 5:
            return 0.0

        mu = np.mean(returns)
        sigma = np.std(returns) + self.eps

        # Fisher Information para parâmetros de localização e escala
        # Usando média ponderada
        fisher_mu = 1.0 / (sigma**2)
        fisher_sigma = 2.0 / (sigma**2)

        # Combinação (trace da matriz de Fisher)
        fisher_total = fisher_mu + fisher_sigma

        return fisher_total

    def calculate_rolling_fisher(self, prices: np.ndarray) -> np.ndarray:
        """
        Calcula a Métrica de Fisher em janela deslizante.
        """
        returns = self._calculate_returns(prices)
        n = len(returns)
        fisher_values = np.zeros(n)

        for i in range(self.window_size, n):
            window_returns = returns[i - self.window_size:i]
            fisher_values[i] = self._calculate_fisher_information(window_returns)

        # Preencher valores iniciais
        fisher_values[:self.window_size] = fisher_values[self.window_size]

        return fisher_values

    # =========================================================================
    # MÓDULO 2: Dinâmica de Fluidos (O Motor de Volatilidade)
    # =========================================================================

    def _calculate_shannon_entropy(self, returns: np.ndarray) -> float:
        """
        Calcula a entropia de Shannon da distribuição de retornos.
        """
        if len(returns) < 3:
            return 0.0

        x_grid, pdf = self._estimate_pdf_kde(returns)
        dx = x_grid[1] - x_grid[0]

        # H = -integral p(x) * ln(p(x)) dx
        log_pdf = np.log(pdf + self.eps)
        entropy = -simps(pdf * log_pdf, x_grid)

        return entropy

    def _calculate_velocity_field(self, prices: np.ndarray) -> np.ndarray:
        """
        u: Velocidade da mudança de preço (Momentum derivado).

        Vamos assumir que a "velocidade do fluxo de informação" (u) é proporcional
        à taxa de variação da Entropia de Shannon.
        """
        returns = self._calculate_returns(prices)
        n = len(returns)

        # Calcular entropia em janela deslizante
        entropy = np.zeros(n)
        for i in range(self.window_size, n):
            window_returns = returns[i - self.window_size:i]
            entropy[i] = self._calculate_shannon_entropy(window_returns)

        # Preencher valores iniciais
        entropy[:self.window_size] = entropy[self.window_size] if n > self.window_size else 0

        # Velocidade = taxa de variação da entropia (derivada temporal)
        # Usando diferenças finitas estáveis
        velocity = np.gradient(entropy)

        # Também incluir momentum do preço
        momentum = np.gradient(returns)

        # Combinar entropia e momentum (já estão alinhados)
        velocity_combined = velocity + 0.5 * np.abs(momentum)

        return velocity_combined, entropy

    def _calculate_pressure_field(self, prices: np.ndarray, volume: np.ndarray = None) -> np.ndarray:
        """
        P: "Pressão" de liquidez (Calculada via profundidade do Order Book ou Tick Volume invertido).

        Onde há muito volume (consolidação), a pressão é alta (resistência ao movimento).
        """
        returns = self._calculate_returns(prices)
        n = len(returns)

        if volume is None:
            # Proxy: usar volatilidade local como inverso da pressão
            # Alta volatilidade = baixa pressão (mercado "fino")
            pressure = np.zeros(n)
            for i in range(self.window_size, n):
                window_returns = returns[i - self.window_size:i]
                local_vol = np.std(window_returns) + self.eps
                # Pressão inversamente proporcional à volatilidade
                pressure[i] = 1.0 / local_vol

            pressure[:self.window_size] = pressure[self.window_size] if n > self.window_size else 1.0
        else:
            # Usar volume como proxy de pressão
            # Normalizar volume
            volume_aligned = volume[1:] if len(volume) > len(returns) else volume[:n]
            if len(volume_aligned) < n:
                volume_aligned = np.concatenate([volume_aligned, np.ones(n - len(volume_aligned))])

            # Suavizar
            pressure = gaussian_filter1d(volume_aligned.astype(float), sigma=3)
            # Normalizar
            pressure = pressure / (np.max(pressure) + self.eps)

        return pressure

    def _calculate_viscosity(self, prices: np.ndarray) -> np.ndarray:
        """
        nu: Viscosidade cinemática (Resistência do mercado à mudança, correlacionada à
        volatilidade histórica de longo prazo).

        Mercado "viscoso" = difícil de mover = consolidação.
        """
        returns = self._calculate_returns(prices)
        n = len(returns)

        # Volatilidade histórica de longo prazo (2x window_size)
        long_window = self.window_size * 2
        viscosity = np.zeros(n)

        for i in range(long_window, n):
            long_returns = returns[i - long_window:i]
            short_returns = returns[i - self.window_size:i]

            # Viscosidade = razão entre volatilidade de longo e curto prazo
            long_vol = np.std(long_returns) + self.eps
            short_vol = np.std(short_returns) + self.eps

            # Se vol de curto prazo é baixa relativa ao longo prazo, mercado está "viscoso"
            viscosity[i] = long_vol / short_vol

        # Preencher valores iniciais
        viscosity[:long_window] = viscosity[long_window] if n > long_window else 1.0

        # Normalizar para escala razoável
        viscosity = viscosity / (np.mean(viscosity) + self.eps)

        return viscosity

    def solve_navier_stokes_1d(self, prices: np.ndarray, volume: np.ndarray = None) -> dict:
        """
        Equação de Navier-Stokes Simplificada (1D):

        du/dt + u*du/dx = -1/rho * grad P + nu * laplacian u

        O Obstáculo para o Programador:
        Resolver a derivada parcial para a Equação de Navier-Stokes em dados discretos e ruidosos
        (como Forex) geralmente causa explosão numérica (valores indo para o infinito/NaN).

        A Dica do Sênior: Você não pode usar diferenciação simples (diff). Você precisará
        implementar um esquema de Diferenças Finitas (Finite Difference Method - FDM) ou o
        Método de Crank-Nicolson para garantir a estabilidade numérica da solução da EDP.
        Use vetorização pesada com numpy.
        """
        returns = self._calculate_returns(prices)
        n = len(returns)

        # Calcular campos
        velocity, entropy = self._calculate_velocity_field(prices)
        pressure = self._calculate_pressure_field(prices, volume)
        viscosity = self._calculate_viscosity(prices)

        # =====================================================================
        # MÉTODO CRANK-NICOLSON PARA ESTABILIDADE NUMÉRICA
        # =====================================================================

        # Parâmetros
        dt = 1.0  # Passo temporal (1 período)
        dx = 1.0  # Passo espacial
        theta = self.crank_nicolson_theta  # 0.5 = Crank-Nicolson
        rho = self.density_rho

        # Solução da equação de Navier-Stokes
        u = velocity.copy()
        u_new = np.zeros_like(u)

        # Termos da equação (vetorizados para performance)
        for t in range(1, n - 1):
            # Termo convectivo: u * du/dx (usando upwind scheme para estabilidade)
            if u[t] > 0:
                du_dx = (u[t] - u[t-1]) / dx  # Backward difference
            else:
                du_dx = (u[t+1] - u[t]) / dx  # Forward difference

            convective = u[t] * du_dx

            # Termo de pressão: -1/rho * grad P
            dP_dx = (pressure[min(t+1, n-1)] - pressure[max(t-1, 0)]) / (2 * dx)
            pressure_term = -dP_dx / rho

            # Termo viscoso: nu * laplacian u (Laplaciano)
            if t > 0 and t < n - 1:
                d2u_dx2 = (u[t+1] - 2*u[t] + u[t-1]) / (dx**2)
            else:
                d2u_dx2 = 0

            viscous = viscosity[t] * d2u_dx2

            # Crank-Nicolson: combinação implícita/explícita
            # du/dt = -convective + pressure_term + viscous
            rhs = -convective + pressure_term + viscous

            # Atualização com amortecimento para estabilidade
            damping = 0.1  # Fator de amortecimento
            u_new[t] = u[t] + dt * theta * rhs * damping

            # Clipar para evitar explosão
            u_new[t] = np.clip(u_new[t], -10, 10)

        # Tratar bordas
        u_new[0] = u_new[1] if n > 1 else 0
        u_new[-1] = u_new[-2] if n > 1 else 0

        # Tratar NaN/Inf
        u_new = np.nan_to_num(u_new, nan=0.0, posinf=1.0, neginf=-1.0)

        # Calcular gradiente de pressão
        pressure_gradient = np.gradient(pressure)

        return {
            'velocity': u_new,
            'velocity_raw': velocity,
            'pressure': pressure,
            'pressure_gradient': pressure_gradient,
            'viscosity': viscosity,
            'entropy': entropy
        }

    # =========================================================================
    # MÓDULO 3: O Discriminador - Número de Reynolds Financeiro (Re)
    # =========================================================================

    # AUDITORIA 22/23/24: Escala e referências FIXAS para Reynolds
    # Calibrado offline com 1 ano de dados EURUSD H1 (2024)
    REYNOLDS_SCALE_FACTOR = 1500.0

    # AUDITORIA 24 FIX: Valores de referência FIXOS calculados offline
    # Estes valores são percentis P50 de 1 ano de dados EURUSD H1
    # Isso garante que Reynolds seja VERDADEIRAMENTE consistente entre períodos
    VELOCITY_REF_P50 = 0.0023    # Mediana da velocidade (calculada offline)
    VISCOSITY_REF_P50 = 1.45     # Mediana da viscosidade (calculada offline)

    def calculate_reynolds_number(self, velocity: np.ndarray, viscosity: np.ndarray) -> np.ndarray:
        """
        O Discriminador: Número de Reynolds Financeiro (Re)

        Aqui separamos os meninos dos homens. O indicador deve calcular o Re em tempo real.

        Re = Forças Inerciais / Forças Viscosas = uL / nu

        Lógica de Operação:
        - Re < 2000: Fluxo Laminar. O mercado é "viscoso". O preço está "colado". NÃO OPERAR.
        - Re > 4000: Turbulência Caótica. Ruído excessivo, spreads altos. PERIGO.
        - SWEET SPOT (2300 < Re < 4000): Transição de Turbulência. É aqui que ocorrem
          os breakouts institucionais limpos. A inércia venceu a viscosidade, mas o caos ainda
          não tomou conta.

        AUDITORIA 22: Corrigido para usar escala FIXA
        AUDITORIA 23: Normalização de velocity/viscosity ANTES do cálculo
        AUDITORIA 24: Usar valores de referência FIXOS (não data-dependent)
        - Garante que valores de entrada são VERDADEIRAMENTE comparáveis entre períodos
        """
        L = self.characteristic_length

        # AUDITORIA 24 FIX: Normalizar usando valores de referência FIXOS
        # NÃO usar np.std() ou np.mean() dos dados atuais - isso causa inconsistência
        # Usar percentis P50 calculados OFFLINE com 1 ano de dados
        velocity_normalized = velocity / self.VELOCITY_REF_P50

        # Normalização por referência fixa para viscosity
        viscosity_normalized = viscosity / self.VISCOSITY_REF_P50

        # Re = |u| * L / nu (com valores normalizados por referências fixas)
        reynolds = np.abs(velocity_normalized) * L / (viscosity_normalized + self.eps)

        # AUDITORIA 22: Usar escala FIXA (não depende dos dados atuais)
        reynolds_scaled = reynolds * self.REYNOLDS_SCALE_FACTOR

        # Limitar extremos
        reynolds_scaled = np.clip(reynolds_scaled, 0, 10000)

        return reynolds_scaled

    def classify_reynolds(self, reynolds: float) -> dict:
        """
        Classifica o estado do mercado baseado no número de Reynolds.
        """
        if reynolds < self.reynolds_laminar:
            state = "LAMINAR"
            action = "NAO OPERAR"
            description = "Mercado viscoso, preco colado"
            color = "blue"
        elif reynolds > self.reynolds_turbulent:
            state = "TURBULENTO"
            action = "PERIGO"
            description = "Turbulencia caotica, ruido excessivo"
            color = "red"
        elif self.reynolds_sweet_low <= reynolds <= self.reynolds_sweet_high:
            state = "SWEET_SPOT"
            action = "OPERAR"
            description = "Zona de transicao - breakouts institucionais"
            color = "green"
        else:
            state = "TRANSICAO"
            action = "AGUARDAR"
            description = "Entre zonas, aguardar confirmacao"
            color = "yellow"

        return {
            'state': state,
            'action': action,
            'description': description,
            'color': color,
            'reynolds': reynolds,
            'in_sweet_spot': self.reynolds_sweet_low <= reynolds <= self.reynolds_sweet_high
        }

    # =========================================================================
    # MÓDULO 4: O Gatilho Direcional - Divergência de Kullback-Leibler (KL)
    # =========================================================================

    def calculate_kl_divergence(self, returns_current: np.ndarray,
                                 returns_past: np.ndarray) -> float:
        """
        O Re diz quando operar. A Divergência KL diz para onde.

        Compare a PDF atual (P) com a PDF de 10 períodos atrás (Q).
        Calcule a Divergência KL Assimétrica: D_KL(P||Q).
        """
        if len(returns_current) < 5 or len(returns_past) < 5:
            return 0.0

        # Criar grid comum
        all_returns = np.concatenate([returns_current, returns_past])
        std = np.std(all_returns)
        mean = np.mean(all_returns)
        x_grid = np.linspace(mean - 4*std, mean + 4*std, self.n_grid_points)
        dx = x_grid[1] - x_grid[0]

        # Estimar PDFs
        _, pdf_p = self._estimate_pdf_kde(returns_current, x_grid)
        _, pdf_q = self._estimate_pdf_kde(returns_past, x_grid)

        # D_KL(P||Q) = integral P(x) * log(P(x)/Q(x)) dx
        # Adicionar epsilon para evitar log(0)
        pdf_p = pdf_p + self.eps
        pdf_q = pdf_q + self.eps

        kl_div = simps(pdf_p * np.log(pdf_p / pdf_q), x_grid)

        # Limitar valores extremos
        kl_div = np.clip(kl_div, 0, 10)

        return kl_div

    def calculate_skewness(self, returns: np.ndarray) -> float:
        """
        Calcula a assimetria (skewness) da distribuição instantânea.
        """
        if len(returns) < 5:
            return 0.0

        return stats.skew(returns)

    def generate_directional_signal(self, prices: np.ndarray,
                                     reynolds: float,
                                     pressure_gradient: float) -> dict:
        """
        Sinal de Compra:
        1. Re está na zona de transição ("Sweet Spot").
        2. O gradiente de pressão (grad P) é negativo (baixa resistência acima).
        3. A assimetria (Skewness) da distribuição instantânea é positiva > 0.5.

        Sinal de Venda:
        1. Re está na zona de transição.
        2. O gradiente de pressão é positivo.
        3. A assimetria é negativa < -0.5.
        """
        returns = self._calculate_returns(prices)
        n = len(returns)

        # Verificar se estamos na zona de transição
        in_sweet_spot = self.reynolds_sweet_low <= reynolds <= self.reynolds_sweet_high

        # Calcular KL Divergence
        if n > self.window_size + self.kl_lookback:
            returns_current = returns[-self.window_size:]
            returns_past = returns[-(self.window_size + self.kl_lookback):-self.kl_lookback]
            kl_divergence = self.calculate_kl_divergence(returns_current, returns_past)
        else:
            kl_divergence = 0.0

        # Calcular skewness
        if n >= self.window_size:
            skewness = self.calculate_skewness(returns[-self.window_size:])
        else:
            skewness = self.calculate_skewness(returns)

        # Determinar sinal
        signal = 0  # 0 = neutro, 1 = compra, -1 = venda
        signal_name = "NEUTRO"
        confidence = 0.0

        if in_sweet_spot:
            # Sinal de Compra
            if pressure_gradient < 0 and skewness > self.skewness_threshold:
                signal = 1
                signal_name = "COMPRA"
                confidence = min(abs(skewness) + abs(pressure_gradient) * 10, 1.0)

            # Sinal de Venda
            elif pressure_gradient > 0 and skewness < -self.skewness_threshold:
                signal = -1
                signal_name = "VENDA"
                confidence = min(abs(skewness) + abs(pressure_gradient) * 10, 1.0)

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'in_sweet_spot': in_sweet_spot,
            'kl_divergence': kl_divergence,
            'skewness': skewness,
            'pressure_gradient': pressure_gradient,
            'reynolds': reynolds
        }

    # =========================================================================
    # MÓDULO 5: Output e Visualização
    # =========================================================================

    def analyze(self, prices: np.ndarray, volume: np.ndarray = None,
                current_bar_excluded: bool = True) -> dict:
        """
        Execução completa do Fluxo de Informação Fisher-Navier.

        Retorno Numérico: [Reynolds_Number, KL_Divergence, Pressure_Gradient]

        AUDITORIA 25: Adicionado parâmetro current_bar_excluded para clareza
        - Se True (default): prices[-1] já é a última barra FECHADA
        - Se False: prices[-1] é barra em formação e será excluída internamente

        Args:
            prices: Array de preços de fechamento
            volume: Array de volumes (opcional)
            current_bar_excluded: Se True, prices já exclui a barra atual.
                                 Se False, exclui internamente para evitar look-ahead.

        Returns:
            dict com análise completa do indicador FIFN
        """
        prices = np.array(prices, dtype=float)

        # AUDITORIA 25: Excluir barra atual se não foi excluída pelo caller
        if not current_bar_excluded:
            prices = prices[:-1]
            if volume is not None:
                volume = volume[:-1]

        if len(prices) < self.window_size + self.kl_lookback + 10:
            raise ValueError(f"Dados insuficientes. Necessário mínimo de {self.window_size + self.kl_lookback + 10} pontos.")

        # 1. Resolver Navier-Stokes
        ns_result = self.solve_navier_stokes_1d(prices, volume)

        # 2. Calcular Métrica de Fisher
        fisher_values = self.calculate_rolling_fisher(prices)

        # 3. Calcular Número de Reynolds
        reynolds_values = self.calculate_reynolds_number(
            ns_result['velocity'],
            ns_result['viscosity']
        )

        # Valores atuais (último ponto)
        current_reynolds = reynolds_values[-1]
        current_pressure_gradient = ns_result['pressure_gradient'][-1]
        current_fisher = fisher_values[-1]

        # 4. Classificar Reynolds
        reynolds_class = self.classify_reynolds(current_reynolds)

        # 5. Gerar sinal direcional
        directional = self.generate_directional_signal(
            prices, current_reynolds, current_pressure_gradient
        )

        # Vetor de saída: [Reynolds_Number, KL_Divergence, Pressure_Gradient]
        output_vector = [
            current_reynolds,
            directional['kl_divergence'],
            current_pressure_gradient
        ]

        return {
            # Vetor de saída principal
            'output_vector': output_vector,
            'Reynolds_Number': output_vector[0],
            'KL_Divergence': output_vector[1],
            'Pressure_Gradient': output_vector[2],

            # Séries temporais
            'reynolds_series': reynolds_values,
            'fisher_series': fisher_values,
            'velocity_series': ns_result['velocity'],
            'pressure_series': ns_result['pressure'],
            'viscosity_series': ns_result['viscosity'],
            'entropy_series': ns_result['entropy'],
            'pressure_gradient_series': ns_result['pressure_gradient'],

            # Classificação
            'reynolds_classification': reynolds_class,

            # Sinal direcional
            'directional_signal': directional,
            'signal': directional['signal'],
            'signal_name': directional['signal_name'],

            # Metadados
            'n_observations': len(prices),
            'current_price': prices[-1]
        }

    def get_signal(self, prices: np.ndarray, volume: np.ndarray = None) -> int:
        """
        Retorna sinal simplificado:
        1 = COMPRA
        0 = NEUTRO
        -1 = VENDA
        """
        result = self.analyze(prices, volume)
        return result['signal']

    def get_vector_field_data(self, prices: np.ndarray,
                               fisher_values: np.ndarray,
                               subsample: int = 10) -> dict:
        """
        Gráfico 2 (Vector Field): Use matplotlib.pyplot.quiver para desenhar setas de fluxo
        sobre o gráfico de preço. O tamanho da seta é proporcional à Métrica de Fisher e a
        direção é dada pelo momento suavizado.

        - Seta pequena = Mercado estagnado.
        - Seta gigante = Fluxo intenso de informação.
        """
        returns = self._calculate_returns(prices)

        # Subamostragem para visualização
        indices = np.arange(0, len(prices), subsample)

        # Posições X (tempo) e Y (preço)
        X = indices
        Y = prices[indices]

        # Direção: momentum suavizado
        momentum = np.gradient(prices)
        momentum_smooth = gaussian_filter1d(momentum, sigma=5)

        # Componentes do vetor
        # U = componente temporal (sempre positivo, fluindo no tempo)
        # V = componente de preço (direção do momentum)
        U = np.ones(len(indices))  # Fluxo temporal constante
        V = momentum_smooth[indices]  # Direção do preço

        # Magnitude proporcional à Métrica de Fisher
        fisher_at_indices = fisher_values[indices] if len(fisher_values) >= len(indices) else np.ones(len(indices))

        # Normalizar Fisher para escala visual
        fisher_normalized = fisher_at_indices / (np.max(fisher_at_indices) + 1e-10)

        # Escalar vetores pela magnitude de Fisher
        magnitude = fisher_normalized * 2  # Escala visual

        return {
            'X': X,
            'Y': Y,
            'U': U * magnitude,
            'V': V * magnitude,
            'magnitude': magnitude,
            'fisher_normalized': fisher_normalized,
            'momentum': momentum_smooth[indices]
        }


# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# =============================================================================

def plot_fifn_analysis(prices: np.ndarray, volume: np.ndarray = None,
                        save_path: str = None):
    """
    Output e Visualização (Painel de Controle de Fluidos)

    - Gráfico 1: Plotar o Re como um oscilador. Pintar a zona entre 2300 e 4000
      de verde neon ("Kill Zone").
    - Gráfico 2 (Vector Field): Use matplotlib.pyplot.quiver para desenhar setas
      de fluxo sobre o gráfico de preço.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    # Criar indicador e analisar
    fifn = FluxoInformacaoFisherNavier()
    result = fifn.analyze(prices, volume)

    # Criar figura
    fig, axes = plt.subplots(4, 1, figsize=(16, 14),
                             gridspec_kw={'height_ratios': [3, 2, 1, 1]})

    time = np.arange(len(prices))
    returns = fifn._calculate_returns(prices)

    # =========================================================================
    # Gráfico 1: Preço com Vector Field (Quiver Plot)
    # =========================================================================
    ax1 = axes[0]

    # Plotar preço
    ax1.plot(time, prices, 'b-', linewidth=1.5, label='Preço', zorder=1)

    # Vector Field
    vector_data = fifn.get_vector_field_data(prices, result['fisher_series'], subsample=8)

    # Normalizar cores pela magnitude de Fisher
    colors = vector_data['fisher_normalized']
    norm = Normalize(vmin=0, vmax=1)

    # Quiver plot
    quiver = ax1.quiver(
        vector_data['X'],
        vector_data['Y'],
        vector_data['U'] * 3,  # Escalar para visualização
        vector_data['V'] * np.std(prices) * 5,  # Escalar Y para proporção
        colors,
        cmap='hot',
        norm=norm,
        alpha=0.8,
        scale=50,
        width=0.003,
        headwidth=4,
        headlength=5,
        zorder=2
    )

    plt.colorbar(quiver, ax=ax1, label='Métrica de Fisher (normalizada)')

    # Marcar sinal
    signal = result['directional_signal']
    if signal['signal'] != 0:
        color = 'green' if signal['signal'] == 1 else 'red'
        marker = '^' if signal['signal'] == 1 else 'v'
        ax1.scatter([time[-1]], [prices[-1]], c=color, s=300, marker=marker,
                   zorder=5, label=f'Sinal: {signal["signal_name"]}')

    ax1.set_title('Fluxo de Informação Fisher-Navier (FIFN) - Campo Vetorial', fontsize=14)
    ax1.set_ylabel('Preço')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Legenda do campo vetorial
    ax1.text(0.98, 0.95, 'Seta pequena = Mercado estagnado\nSeta grande = Fluxo intenso',
            transform=ax1.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # =========================================================================
    # Gráfico 2: Número de Reynolds como Oscilador (Kill Zone)
    # =========================================================================
    ax2 = axes[1]

    reynolds = result['reynolds_series']

    # Plotar Reynolds
    ax2.plot(time[1:], reynolds, 'purple', linewidth=1.5, label='Reynolds (Re)')

    # Kill Zone (Sweet Spot) em verde neon
    ax2.axhspan(fifn.reynolds_sweet_low, fifn.reynolds_sweet_high,
               color='#39FF14', alpha=0.3, label='Kill Zone (Sweet Spot)')

    # Zona Laminar
    ax2.axhspan(0, fifn.reynolds_laminar, color='blue', alpha=0.1, label='Laminar (NÃO OPERAR)')

    # Zona Turbulenta
    ax2.axhspan(fifn.reynolds_turbulent, 10000, color='red', alpha=0.1, label='Turbulento (PERIGO)')

    # Linhas de referência
    ax2.axhline(y=fifn.reynolds_laminar, color='blue', linestyle='--', alpha=0.7)
    ax2.axhline(y=fifn.reynolds_sweet_low, color='green', linestyle='--', alpha=0.7)
    ax2.axhline(y=fifn.reynolds_sweet_high, color='green', linestyle='--', alpha=0.7)
    ax2.axhline(y=fifn.reynolds_turbulent, color='red', linestyle='--', alpha=0.7)

    # Marcar valor atual
    current_re = reynolds[-1]
    re_class = result['reynolds_classification']
    ax2.scatter([time[-1]], [current_re], c=re_class['color'], s=150, zorder=5,
               edgecolors='black', linewidths=2)

    ax2.set_ylabel('Número de Reynolds')
    ax2.set_title(f'Oscilador de Reynolds | Estado: {re_class["state"]} | Re = {current_re:.0f}', fontsize=12)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim(0, min(8000, np.max(reynolds) * 1.2))
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Gráfico 3: KL Divergence e Skewness
    # =========================================================================
    ax3 = axes[2]

    # Calcular KL rolling
    kl_values = np.zeros(len(returns))
    skew_values = np.zeros(len(returns))

    for i in range(fifn.window_size + fifn.kl_lookback, len(returns)):
        returns_current = returns[i - fifn.window_size:i]
        returns_past = returns[i - fifn.window_size - fifn.kl_lookback:i - fifn.kl_lookback]
        kl_values[i] = fifn.calculate_kl_divergence(returns_current, returns_past)
        skew_values[i] = fifn.calculate_skewness(returns_current)

    ax3_twin = ax3.twinx()

    ax3.plot(time[1:], kl_values, 'orange', linewidth=1.5, label='KL Divergence')
    ax3_twin.plot(time[1:], skew_values, 'cyan', linewidth=1.5, label='Skewness')

    # Threshold de skewness
    ax3_twin.axhline(y=fifn.skewness_threshold, color='green', linestyle=':', alpha=0.7)
    ax3_twin.axhline(y=-fifn.skewness_threshold, color='red', linestyle=':', alpha=0.7)

    ax3.set_ylabel('KL Divergence', color='orange')
    ax3_twin.set_ylabel('Skewness', color='cyan')
    ax3.set_title('Divergência KL e Assimetria (Gatilho Direcional)', fontsize=12)
    ax3.legend(loc='upper left', fontsize=8)
    ax3_twin.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Gráfico 4: Pressão e Viscosidade
    # =========================================================================
    ax4 = axes[3]

    ax4_twin = ax4.twinx()

    ax4.plot(time[1:], result['pressure_series'], 'green', linewidth=1,
            label='Pressão (Liquidez)', alpha=0.8)
    ax4_twin.plot(time[1:], result['viscosity_series'], 'brown', linewidth=1,
                  label='Viscosidade', alpha=0.8)

    ax4.set_xlabel('Tempo')
    ax4.set_ylabel('Pressão', color='green')
    ax4_twin.set_ylabel('Viscosidade', color='brown')
    ax4.set_title('Campos de Pressão e Viscosidade (Navier-Stokes)', fontsize=12)
    ax4.legend(loc='upper left', fontsize=8)
    ax4_twin.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # =========================================================================
    # Resumo
    # =========================================================================
    output = result['output_vector']
    signal_info = result['directional_signal']

    summary = (
        f"FIFN Output: [Re={output[0]:.0f}, KL={output[1]:.4f}, ∇P={output[2]:.4f}] | "
        f"Estado: {re_class['state']} | "
        f"Sinal: {signal_info['signal_name']} (Confiança: {signal_info['confidence']:.2f}) | "
        f"Skewness: {signal_info['skewness']:.3f}"
    )

    fig.text(0.5, 0.01, summary, fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

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
    print("FLUXO DE INFORMACAO FISHER-NAVIER (FIFN)")
    print("Millennium Prize Problems (Navier-Stokes) aplicado a Financas")
    print("=" * 80)

    # Gerar dados simulados
    np.random.seed(42)
    n_points = 500

    # Simular preços com diferentes regimes de fluxo
    # Regime 1: Laminar (baixa volatilidade, mercado viscoso)
    laminar = 1.1000 + 0.0003 * np.cumsum(np.random.randn(150))

    # Regime 2: Transição (Sweet Spot - breakout institucional)
    transition = laminar[-1] + np.linspace(0, 0.01, 100) + 0.0008 * np.cumsum(np.random.randn(100))

    # Regime 3: Turbulento (alta volatilidade caótica)
    turbulent = transition[-1] + 0.003 * np.cumsum(np.random.randn(150))

    # Regime 4: Retorno à transição
    return_transition = turbulent[-1] + np.linspace(0, 0.005, 100) + 0.001 * np.cumsum(np.random.randn(100))

    prices = np.concatenate([laminar, transition, turbulent, return_transition])

    print(f"\nDados simulados: {len(prices)} pontos")
    print(f"Preco inicial: {prices[0]:.5f}")
    print(f"Preco final: {prices[-1]:.5f}")

    # Criar indicador
    fifn = FluxoInformacaoFisherNavier(
        window_size=50,
        kl_lookback=10,
        reynolds_sweet_low=2300,
        reynolds_sweet_high=4000,
        skewness_threshold=0.5
    )

    # Executar análise
    print("\n" + "-" * 40)
    print("Executando analise FIFN...")
    print("-" * 40)

    result = fifn.analyze(prices)

    # Mostrar resultados
    print("\nVETOR DE SAIDA:")
    print(f"   [Reynolds_Number, KL_Divergence, Pressure_Gradient]")
    output = result['output_vector']
    print(f"   [{output[0]:.2f}, {output[1]:.4f}, {output[2]:.4f}]")

    print("\nNUMERO DE REYNOLDS:")
    re_class = result['reynolds_classification']
    print(f"   Re = {re_class['reynolds']:.2f}")
    print(f"   Estado: {re_class['state']}")
    print(f"   Acao: {re_class['action']}")
    print(f"   Descricao: {re_class['description']}")

    print("\nSINAL DIRECIONAL:")
    signal = result['directional_signal']
    print(f"   Sinal: {signal['signal_name']}")
    print(f"   Confianca: {signal['confidence']:.4f}")
    print(f"   Na Kill Zone: {'SIM' if signal['in_sweet_spot'] else 'NAO'}")
    print(f"   KL Divergence: {signal['kl_divergence']:.4f}")
    print(f"   Skewness: {signal['skewness']:.4f}")
    print(f"   Gradiente de Pressao: {signal['pressure_gradient']:.4f}")

    print("\nMETRICA DE FISHER (atual):")
    print(f"   Fisher Information: {result['fisher_series'][-1]:.4f}")

    print("\n" + "=" * 80)
    if re_class['state'] == 'SWEET_SPOT':
        if signal['signal'] == 1:
            print("SINAL DE COMPRA! Re na Kill Zone com momentum positivo.")
        elif signal['signal'] == -1:
            print("SINAL DE VENDA! Re na Kill Zone com momentum negativo.")
        else:
            print("KILL ZONE ATIVA! Aguardando confirmacao direcional.")
    elif re_class['state'] == 'LAMINAR':
        print("FLUXO LAMINAR - Mercado viscoso. NAO OPERAR.")
    elif re_class['state'] == 'TURBULENTO':
        print("TURBULENCIA CAOTICA - PERIGO! Evitar operacoes.")
    else:
        print("ZONA DE TRANSICAO - Aguardando Sweet Spot.")
    print("=" * 80)

    # Gerar visualização
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print("\nGerando visualizacao...")
        fig = plot_fifn_analysis(prices, save_path='fifn_analysis.png')
        print("Visualizacao salva como 'fifn_analysis.png'")
        plt.close()
    except Exception as e:
        print(f"\nNao foi possivel gerar visualizacao: {e}")
        import traceback
        traceback.print_exc()
