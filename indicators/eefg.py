"""
EEFG-v1: Estimador de Entropia Fractal GARCH
=============================================
Este indicador mede a QUALIDADE da volatilidade, distinguindo entre
volatilidade direcional (tendência forte) e ruído caótico (mercado lateral perigoso).

Componentes:
1. Volatilidade de Parkinson (usa High/Low)
2. GARCH(1,1) Recursivo
3. Expoente de Hurst (persistência/antipersistência)
4. Entropia de Shannon (incerteza da informação)
5. Fórmula de Fusão: V_final = σ_GARCH * e^(1-H) * (1 + S_norm)

Output: Z-score do V_final em janela móvel de 100 períodos

Classificação:
- Z-score < -1.0: Baixa Volatilidade (Zona de Acumulação)
- -1.0 <= Z-score <= 1.5: Média Volatilidade (Operável)
- Z-score > 1.5: Alta Volatilidade (Crítico/Exaustão)
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from arch import arch_model
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class EEFGIndicator:
    """
    Estimador de Entropia Fractal GARCH (EEFG-v1)
    """

    def __init__(
        self,
        parkinson_window: int = 10,
        hurst_window: int = 100,
        entropy_window: int = 20,
        zscore_window: int = 100,
        garch_window: int = 252
    ):
        """
        Parâmetros:
        -----------
        parkinson_window : int
            Janela para cálculo da Volatilidade de Parkinson (default: 10)
        hurst_window : int
            Janela para cálculo do Expoente de Hurst (default: 100)
        entropy_window : int
            Janela para cálculo da Entropia de Shannon (default: 20)
        zscore_window : int
            Janela para normalização Z-score do V_final (default: 100)
        garch_window : int
            Janela para ajuste do modelo GARCH (default: 252)
        """
        self.parkinson_window = parkinson_window
        self.hurst_window = hurst_window
        self.entropy_window = entropy_window
        self.zscore_window = zscore_window
        self.garch_window = garch_window

    def calculate_log_returns(self, close: np.ndarray) -> np.ndarray:
        """
        Calcula log-returns: ln(C_t / C_{t-1})
        """
        returns = np.log(close[1:] / close[:-1])
        return np.concatenate([[np.nan], returns])

    def calculate_parkinson_volatility(
        self,
        high: np.ndarray,
        low: np.ndarray,
        window: Optional[int] = None
    ) -> np.ndarray:
        """
        Volatilidade de Parkinson normalizada.

        Fórmula: σ_Parkinson = sqrt( (1 / (4n * ln(2))) * Σ(ln(H_i/L_i))^2 )

        Captura a expansão real do spread intradiário usando High e Low.
        """
        if window is None:
            window = self.parkinson_window

        n = len(high)
        parkinson_vol = np.full(n, np.nan)

        # Calcula ln(H/L)^2 para cada candle
        log_hl_squared = np.log(high / low) ** 2

        for i in range(window - 1, n):
            sum_log_hl_sq = np.sum(log_hl_squared[i - window + 1:i + 1])
            parkinson_vol[i] = np.sqrt(sum_log_hl_sq / (4 * window * np.log(2)))

        return parkinson_vol

    def fit_garch_rolling(
        self,
        returns: np.ndarray,
        window: Optional[int] = None
    ) -> np.ndarray:
        """
        Ajusta GARCH(1,1) em janelas deslizantes e prevê a variância do próximo passo.

        Modelo: σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}

        Onde:
        - ω: intercepto (viés de longo prazo)
        - α: coeficiente de reação a choques recentes ("susto" do mercado)
        - β: persistência da volatilidade antiga ("memória" do mercado)
        """
        if window is None:
            window = self.garch_window

        n = len(returns)
        garch_vol = np.full(n, np.nan)

        # Remove NaN inicial dos retornos
        valid_start = np.where(~np.isnan(returns))[0]
        if len(valid_start) == 0:
            return garch_vol

        start_idx = valid_start[0]

        for i in range(start_idx + window, n):
            try:
                # Janela de dados para ajuste
                window_returns = returns[i - window:i]

                # Remove NaNs
                window_returns = window_returns[~np.isnan(window_returns)]

                if len(window_returns) < 50:  # Mínimo de dados
                    continue

                # Escala para percentual (arch_model espera retornos em %)
                scaled_returns = window_returns * 100

                # Ajusta GARCH(1,1)
                model = arch_model(
                    scaled_returns,
                    vol='Garch',
                    p=1,
                    q=1,
                    mean='Zero',
                    rescale=False
                )
                result = model.fit(disp='off', show_warning=False)

                # Previsão da variância para o próximo período
                forecast = result.forecast(horizon=1)
                variance_forecast = forecast.variance.values[-1, 0]

                # Converte de volta para decimal (de % para decimal)
                garch_vol[i] = np.sqrt(variance_forecast) / 100

            except Exception:
                # Se falhar, usa volatilidade realizada simples
                window_returns = returns[i - window:i]
                window_returns = window_returns[~np.isnan(window_returns)]
                if len(window_returns) > 0:
                    garch_vol[i] = np.std(window_returns)

        return garch_vol

    def calculate_hurst_exponent(
        self,
        series: np.ndarray,
        window: Optional[int] = None
    ) -> np.ndarray:
        """
        Calcula o Expoente de Hurst usando análise R/S (Rescaled Range).

        Fórmula: (R/S)_n = (Max(Z_1...Z_n) - Min(Z_1...Z_n)) / sqrt(Var(X))

        Interpretação:
        - H > 0.5: Série persistente (tendência fractal)
        - H < 0.5: Série antipersistente (reversão à média/choque)
        - H = 0.5: Passeio aleatório puro
        """
        if window is None:
            window = self.hurst_window

        n = len(series)
        hurst = np.full(n, np.nan)

        for i in range(window - 1, n):
            try:
                window_data = series[i - window + 1:i + 1]

                # Remove NaNs
                window_data = window_data[~np.isnan(window_data)]

                if len(window_data) < 20:
                    continue

                # Calcula Hurst via R/S Analysis
                h = self._rs_analysis(window_data)
                hurst[i] = h

            except Exception:
                continue

        return hurst

    def _rs_analysis(self, series: np.ndarray) -> float:
        """
        Análise R/S para calcular o expoente de Hurst.
        """
        n = len(series)

        if n < 20:
            return 0.5

        # Divide em subseries de diferentes tamanhos
        max_k = int(np.log2(n))
        rs_values = []
        ns_values = []

        for k in range(2, max_k + 1):
            subset_size = 2 ** k
            if subset_size > n:
                break

            num_subsets = n // subset_size
            rs_list = []

            for j in range(num_subsets):
                subset = series[j * subset_size:(j + 1) * subset_size]

                # Média da subserie
                mean_subset = np.mean(subset)

                # Desvios acumulados da média
                cumdev = np.cumsum(subset - mean_subset)

                # Range
                R = np.max(cumdev) - np.min(cumdev)

                # Desvio padrão
                S = np.std(subset, ddof=1)

                if S > 0:
                    rs_list.append(R / S)

            if len(rs_list) > 0:
                rs_values.append(np.mean(rs_list))
                ns_values.append(subset_size)

        if len(rs_values) < 2:
            return 0.5

        # Regressão log-log para encontrar H
        log_n = np.log(ns_values)
        log_rs = np.log(rs_values)

        # Regressão linear: log(R/S) = H * log(n) + c
        coeffs = np.polyfit(log_n, log_rs, 1)
        hurst = coeffs[0]

        # Limita entre 0 e 1
        hurst = np.clip(hurst, 0.01, 0.99)

        return hurst

    def calculate_shannon_entropy(
        self,
        returns: np.ndarray,
        window: Optional[int] = None,
        bins: int = 10
    ) -> np.ndarray:
        """
        Calcula a Entropia de Shannon da distribuição dos retornos recentes.

        Mede a incerteza da informação - quanto maior, mais imprevisível.
        """
        if window is None:
            window = self.entropy_window

        n = len(returns)
        shannon = np.full(n, np.nan)

        for i in range(window - 1, n):
            window_returns = returns[i - window + 1:i + 1]

            # Remove NaNs
            window_returns = window_returns[~np.isnan(window_returns)]

            if len(window_returns) < 5:
                continue

            try:
                # Cria histograma para estimar distribuição
                hist, _ = np.histogram(window_returns, bins=bins, density=True)

                # Normaliza para somar 1 (probabilidades)
                hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist

                # Remove zeros para evitar log(0)
                hist = hist[hist > 0]

                # Entropia de Shannon
                shannon[i] = entropy(hist, base=2)

            except Exception:
                continue

        return shannon

    def normalize_entropy(self, shannon: np.ndarray) -> np.ndarray:
        """
        Normaliza a entropia de Shannon para o intervalo [0, 1].
        """
        # Normalização min-max em janela rolling
        s_norm = np.full(len(shannon), np.nan)

        for i in range(self.zscore_window - 1, len(shannon)):
            window_data = shannon[i - self.zscore_window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]

            if len(window_data) < 10:
                continue

            min_val = np.min(window_data)
            max_val = np.max(window_data)

            if max_val > min_val:
                s_norm[i] = (shannon[i] - min_val) / (max_val - min_val)
            else:
                s_norm[i] = 0.5

        return s_norm

    def calculate_v_final(
        self,
        garch_vol: np.ndarray,
        hurst: np.ndarray,
        s_norm: np.ndarray
    ) -> np.ndarray:
        """
        Calcula o indicador final de volatilidade.

        Fórmula: V_final = σ_GARCH * e^(1-H) * (1 + S_norm)

        Onde:
        - e^(1-H) penaliza volatilidade em tendência suave (H → 1)
                  e amplifica quando mercado está caótico (H → 0)
        - (1 + S_norm) aumenta com a incerteza
        """
        # Componente de ajuste pelo Hurst
        hurst_factor = np.exp(1 - hurst)

        # Componente de ajuste pela entropia
        entropy_factor = 1 + s_norm

        # Fórmula de fusão
        v_final = garch_vol * hurst_factor * entropy_factor

        return v_final

    def calculate_zscore(
        self,
        v_final: np.ndarray,
        window: Optional[int] = None
    ) -> np.ndarray:
        """
        Calcula o Z-score do V_final em janela móvel.

        Z = (V_final - μ) / σ
        """
        if window is None:
            window = self.zscore_window

        n = len(v_final)
        zscore = np.full(n, np.nan)

        for i in range(window - 1, n):
            window_data = v_final[i - window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]

            if len(window_data) < 10:
                continue

            mean = np.mean(window_data)
            std = np.std(window_data, ddof=1)

            if std > 0:
                zscore[i] = (v_final[i] - mean) / std
            else:
                zscore[i] = 0

        return zscore

    def classify_volatility(self, zscore: float) -> str:
        """
        Classifica o regime de volatilidade baseado no Z-score.

        Retorna:
        - "BAIXA" (Z < -1.0): Zona de Acumulação
        - "MEDIA" (-1.0 <= Z <= 1.5): Operável
        - "ALTA" (Z > 1.5): Crítico/Exaustão
        """
        if np.isnan(zscore):
            return "INDEFINIDO"
        elif zscore < -1.0:
            return "BAIXA"
        elif zscore > 1.5:
            return "ALTA"
        else:
            return "MEDIA"

    def calculate(
        self,
        open_prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Calcula o indicador EEFG completo.

        Parâmetros:
        -----------
        open_prices : np.ndarray
            Preços de abertura
        high : np.ndarray
            Preços máximos
        low : np.ndarray
            Preços mínimos
        close : np.ndarray
            Preços de fechamento
        verbose : bool
            Se True, imprime progresso

        Retorna:
        --------
        pd.DataFrame com colunas:
            - log_returns: Log-retornos
            - parkinson_vol: Volatilidade de Parkinson
            - garch_vol: Volatilidade GARCH(1,1)
            - hurst: Expoente de Hurst
            - shannon: Entropia de Shannon
            - s_norm: Entropia normalizada
            - v_final: Indicador de volatilidade final
            - zscore: Z-score do V_final
            - classification: Classificação do regime
        """
        n = len(close)

        if verbose:
            print("Calculando EEFG-v1...")
            print(f"  [1/6] Log-Returns...")
        log_returns = self.calculate_log_returns(close)

        if verbose:
            print(f"  [2/6] Volatilidade de Parkinson (janela={self.parkinson_window})...")
        parkinson_vol = self.calculate_parkinson_volatility(high, low)

        if verbose:
            print(f"  [3/6] GARCH(1,1) Rolling (janela={self.garch_window})...")
        garch_vol = self.fit_garch_rolling(log_returns)

        if verbose:
            print(f"  [4/6] Expoente de Hurst (janela={self.hurst_window})...")
        hurst = self.calculate_hurst_exponent(log_returns)

        if verbose:
            print(f"  [5/6] Entropia de Shannon (janela={self.entropy_window})...")
        shannon = self.calculate_shannon_entropy(log_returns)
        s_norm = self.normalize_entropy(shannon)

        if verbose:
            print(f"  [6/6] Fusão e Z-Score (janela={self.zscore_window})...")
        v_final = self.calculate_v_final(garch_vol, hurst, s_norm)
        zscore = self.calculate_zscore(v_final)

        # Classificação
        classification = [self.classify_volatility(z) for z in zscore]

        # Monta DataFrame
        result = pd.DataFrame({
            'log_returns': log_returns,
            'parkinson_vol': parkinson_vol,
            'garch_vol': garch_vol,
            'hurst': hurst,
            'shannon': shannon,
            's_norm': s_norm,
            'v_final': v_final,
            'zscore': zscore,
            'classification': classification
        })

        if verbose:
            print("Cálculo concluído!")

        return result


def run_eefg(
    df: pd.DataFrame,
    open_col: str = 'Open',
    high_col: str = 'High',
    low_col: str = 'Low',
    close_col: str = 'Close',
    **kwargs
) -> pd.DataFrame:
    """
    Função wrapper para calcular o EEFG a partir de um DataFrame.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com dados OHLC
    open_col, high_col, low_col, close_col : str
        Nomes das colunas
    **kwargs : dict
        Parâmetros adicionais para EEFGIndicator

    Retorna:
    --------
    pd.DataFrame com dados originais + colunas do indicador
    """
    indicator = EEFGIndicator(**kwargs)

    result = indicator.calculate(
        open_prices=df[open_col].values,
        high=df[high_col].values,
        low=df[low_col].values,
        close=df[close_col].values
    )

    # Combina com dados originais
    for col in result.columns:
        df[f'eefg_{col}'] = result[col].values

    return df
