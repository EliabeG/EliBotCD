#!/usr/bin/env python3
"""
Otimização do Indicador de Volatilidade
Usa dados históricos para calibrar os thresholds de classificação
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import entropy, percentileofscore
import warnings
warnings.filterwarnings('ignore')

# Tenta importar yfinance para dados históricos
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("yfinance não instalado. Instalando...")
    import subprocess
    subprocess.run(["pip", "install", "yfinance", "-q"])
    import yfinance as yf
    HAS_YFINANCE = True


class VolatilityOptimizer:
    """Otimiza thresholds do indicador de volatilidade"""

    def __init__(self):
        self.data = None
        self.volatility_stats = {}
        self.optimal_thresholds = {}

    def load_data(self, symbol: str = "EURUSD=X", period: str = "1y", interval: str = "1h"):
        """
        Carrega dados históricos do Yahoo Finance

        Args:
            symbol: Par de moedas (EURUSD=X para Yahoo)
            period: Período (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)
            interval: Intervalo (1m, 5m, 15m, 30m, 1h, 1d)
        """
        print(f"Carregando dados históricos de {symbol}...")
        print(f"  Período: {period}, Intervalo: {interval}")

        try:
            self.data = yf.download(symbol, period=period, interval=interval, progress=False)

            if len(self.data) == 0:
                print("  Sem dados para EURUSD=X, tentando EUR=X...")
                self.data = yf.download("EUR=X", period=period, interval=interval, progress=False)

            if len(self.data) == 0:
                raise Exception("Não foi possível baixar dados")

            print(f"  Dados carregados: {len(self.data)} candles")
            print(f"  De {self.data.index[0]} até {self.data.index[-1]}")

            return True

        except Exception as e:
            print(f"  Erro ao carregar dados: {e}")
            return False

    def calculate_parkinson_volatility(self, high: np.ndarray, low: np.ndarray, window: int = 20) -> np.ndarray:
        """Calcula volatilidade de Parkinson em janela móvel"""
        n = len(high)
        vol = np.full(n, np.nan)

        for i in range(window - 1, n):
            h = high[i - window + 1:i + 1]
            l = low[i - window + 1:i + 1]

            # Evita divisão por zero
            ratio = np.clip(h / l, 1.0001, None)
            log_hl_sq = np.log(ratio) ** 2
            vol[i] = np.sqrt(np.sum(log_hl_sq) / (4 * window * np.log(2)))

        return vol

    def calculate_realized_volatility(self, close: np.ndarray, window: int = 20) -> np.ndarray:
        """Calcula volatilidade realizada (desvio padrão dos retornos)"""
        returns = np.diff(np.log(close))
        returns = np.concatenate([[np.nan], returns])

        n = len(returns)
        vol = np.full(n, np.nan)

        for i in range(window, n):
            vol[i] = np.std(returns[i - window + 1:i + 1])

        return vol

    def calculate_future_move(self, close: np.ndarray, forward_periods: int = 5) -> np.ndarray:
        """
        Calcula o movimento futuro do preço (APENAS PARA ANÁLISE EXPLORATÓRIA).

        ⚠️ AVISO CRÍTICO - LOOK-AHEAD BIAS:
        Esta função usa dados FUTUROS e NÃO pode ser usada para:
        - Validação de estratégias
        - Otimização de parâmetros para trading real
        - Qualquer decisão de trading

        Uso permitido APENAS para:
        - Análise exploratória offline
        - Estudos acadêmicos
        - Entender a distribuição histórica de volatilidade

        Args:
            close: Array de preços de fechamento
            forward_periods: Número de períodos à frente para calcular

        Returns:
            Array com o range em pips nos próximos N períodos (DADOS FUTUROS!)
        """
        n = len(close)
        future_move = np.full(n, np.nan)

        # ⚠️ LOOK-AHEAD: Este loop usa close[i:i + forward_periods + 1] que são preços FUTUROS!
        for i in range(n - forward_periods):
            future_prices = close[i:i + forward_periods + 1]
            move = (np.max(future_prices) - np.min(future_prices)) * 10000  # em pips
            future_move[i] = move

        return future_move

    def calculate_historical_volatility(self, close: np.ndarray, lookback: int = 20) -> np.ndarray:
        """
        Calcula a volatilidade histórica SEM look-ahead bias.

        Esta é a alternativa SEGURA para uso em trading real.
        Usa apenas dados passados disponíveis até cada momento.

        Args:
            close: Array de preços de fechamento
            lookback: Número de períodos passados para calcular

        Returns:
            Array com a volatilidade histórica em pips
        """
        n = len(close)
        hist_vol = np.full(n, np.nan)

        for i in range(lookback, n):
            past_prices = close[i - lookback:i + 1]  # Usa apenas dados PASSADOS
            vol = (np.max(past_prices) - np.min(past_prices)) * 10000  # em pips
            hist_vol[i] = vol

        return hist_vol

    def analyze_volatility_distribution(self):
        """Analisa a distribuição da volatilidade nos dados históricos"""
        if self.data is None:
            print("Dados não carregados!")
            return

        print("\n" + "=" * 60)
        print("ANÁLISE DA DISTRIBUIÇÃO DE VOLATILIDADE")
        print("=" * 60)

        # Extrai OHLC (flatten para garantir 1D)
        high = self.data['High'].values.flatten()
        low = self.data['Low'].values.flatten()
        close = self.data['Close'].values.flatten()

        # Calcula volatilidades
        parkinson_vol = self.calculate_parkinson_volatility(high, low, window=20)
        realized_vol = self.calculate_realized_volatility(close, window=20)
        future_move = self.calculate_future_move(close, forward_periods=5)

        # Converte para pips
        parkinson_pips = parkinson_vol * 10000
        realized_pips = realized_vol * 10000

        # Remove NaN
        valid_idx = ~np.isnan(parkinson_pips) & ~np.isnan(future_move)
        parkinson_valid = parkinson_pips[valid_idx]
        future_valid = future_move[valid_idx]

        # Estatísticas
        print(f"\nVolatilidade Parkinson (pips):")
        print(f"  Mínimo:     {np.min(parkinson_valid):.4f}")
        print(f"  Máximo:     {np.max(parkinson_valid):.4f}")
        print(f"  Média:      {np.mean(parkinson_valid):.4f}")
        print(f"  Mediana:    {np.median(parkinson_valid):.4f}")
        print(f"  Std:        {np.std(parkinson_valid):.4f}")

        # Percentis
        percentiles = [10, 25, 50, 75, 90, 95]
        print(f"\nPercentis da Volatilidade:")
        for p in percentiles:
            val = np.percentile(parkinson_valid, p)
            print(f"  P{p}: {val:.4f} pips")

        self.volatility_stats = {
            'min': np.min(parkinson_valid),
            'max': np.max(parkinson_valid),
            'mean': np.mean(parkinson_valid),
            'median': np.median(parkinson_valid),
            'std': np.std(parkinson_valid),
            'percentiles': {p: np.percentile(parkinson_valid, p) for p in percentiles}
        }

        # Correlação com movimento futuro
        correlation = np.corrcoef(parkinson_valid, future_valid)[0, 1]
        print(f"\nCorrelação Volatilidade vs Movimento Futuro: {correlation:.4f}")

        return parkinson_valid, future_valid

    def optimize_thresholds(self):
        """
        Otimiza os thresholds de classificação baseado nos dados históricos

        Estratégia:
        - BAIXA: Percentil 0-30 (volatilidade abaixo do normal)
        - MÉDIA: Percentil 30-80 (volatilidade normal/operável)
        - ALTA: Percentil 80-100 (volatilidade elevada)
        """
        if self.data is None:
            print("Dados não carregados!")
            return

        print("\n" + "=" * 60)
        print("OTIMIZAÇÃO DOS THRESHOLDS")
        print("=" * 60)

        parkinson_valid, future_valid = self.analyze_volatility_distribution()

        # Define thresholds baseado em percentis
        # BAIXA: abaixo do percentil 30
        # MÉDIA: entre percentil 30 e 80
        # ALTA: acima do percentil 80

        low_threshold = np.percentile(parkinson_valid, 30)
        high_threshold = np.percentile(parkinson_valid, 80)

        print(f"\nThresholds Otimizados (baseado em percentis):")
        print(f"  BAIXA: volatilidade < {low_threshold:.4f} pips (P30)")
        print(f"  MÉDIA: {low_threshold:.4f} <= vol < {high_threshold:.4f} pips (P30-P80)")
        print(f"  ALTA:  volatilidade >= {high_threshold:.4f} pips (P80)")

        self.optimal_thresholds = {
            'low_threshold': low_threshold,
            'high_threshold': high_threshold
        }

        # Valida os thresholds
        self._validate_thresholds(parkinson_valid, future_valid)

        return self.optimal_thresholds

    def _validate_thresholds(self, volatility: np.ndarray, future_move: np.ndarray):
        """Valida se os thresholds classificam corretamente"""

        low_th = self.optimal_thresholds['low_threshold']
        high_th = self.optimal_thresholds['high_threshold']

        # Classifica cada ponto
        classifications = []
        for v in volatility:
            if v < low_th:
                classifications.append('BAIXA')
            elif v >= high_th:
                classifications.append('ALTA')
            else:
                classifications.append('MEDIA')

        classifications = np.array(classifications)

        # Calcula movimento médio por classificação
        print(f"\n" + "-" * 60)
        print("VALIDAÇÃO DOS THRESHOLDS")
        print("-" * 60)

        for cls in ['BAIXA', 'MEDIA', 'ALTA']:
            mask = classifications == cls
            count = np.sum(mask)
            pct = 100 * count / len(classifications)

            if count > 0:
                avg_move = np.mean(future_move[mask])
                std_move = np.std(future_move[mask])
                avg_vol = np.mean(volatility[mask])

                print(f"\n{cls}:")
                print(f"  Ocorrências: {count} ({pct:.1f}%)")
                print(f"  Vol média: {avg_vol:.4f} pips")
                print(f"  Movimento futuro médio: {avg_move:.2f} pips")
                print(f"  Movimento futuro std: {std_move:.2f} pips")

        # Verifica se a classificação faz sentido
        print(f"\n" + "-" * 60)
        print("VERIFICAÇÃO:")

        baixa_move = np.mean(future_move[classifications == 'BAIXA'])
        media_move = np.mean(future_move[classifications == 'MEDIA'])
        alta_move = np.mean(future_move[classifications == 'ALTA'])

        if baixa_move < media_move < alta_move:
            print("✓ Thresholds válidos: BAIXA < MÉDIA < ALTA em movimento futuro")
        else:
            print("⚠ Aviso: Ordem dos movimentos não é monotônica")
            print(f"  BAIXA: {baixa_move:.2f} pips")
            print(f"  MÉDIA: {media_move:.2f} pips")
            print(f"  ALTA: {alta_move:.2f} pips")

    def generate_optimized_code(self):
        """Gera código Python com os thresholds otimizados"""

        if not self.optimal_thresholds:
            print("Execute optimize_thresholds() primeiro!")
            return

        low_th = self.optimal_thresholds['low_threshold']
        high_th = self.optimal_thresholds['high_threshold']

        print(f"\n" + "=" * 60)
        print("CÓDIGO OTIMIZADO PARA realtime_volatility.py")
        print("=" * 60)

        code = f'''
    def _classify(self) -> str:
        """Classifica volatilidade (OTIMIZADO)"""
        vol = self._cache.get('parkinson_vol')
        ent = self._cache.get('entropy')

        if vol is None:
            return "INDEFINIDO"

        vol_pips = vol * 10000

        # Thresholds otimizados com dados históricos
        LOW_THRESHOLD = {low_th:.4f}   # Percentil 30
        HIGH_THRESHOLD = {high_th:.4f}  # Percentil 80

        if vol_pips < LOW_THRESHOLD:
            classification = "BAIXA"
        elif vol_pips >= HIGH_THRESHOLD:
            classification = "ALTA"
        else:
            classification = "MEDIA"

        # Ajuste fino pela entropia
        if ent is not None:
            # Alta entropia (incerteza) pode indicar transição
            if ent > 0.85 and classification == "MEDIA":
                # Verificar se está perto do threshold alto
                if vol_pips > (LOW_THRESHOLD + HIGH_THRESHOLD) / 2:
                    classification = "ALTA"
            elif ent < 0.25 and classification == "MEDIA":
                # Baixa entropia com vol média-baixa
                if vol_pips < (LOW_THRESHOLD + HIGH_THRESHOLD) / 2:
                    classification = "BAIXA"

        return classification
'''
        print(code)

        return {
            'low_threshold': low_th,
            'high_threshold': high_th
        }


def main():
    print("=" * 60)
    print("  OTIMIZAÇÃO DO INDICADOR DE VOLATILIDADE")
    print("=" * 60)

    optimizer = VolatilityOptimizer()

    # Carrega dados de diferentes timeframes para análise robusta
    print("\n[1/3] Carregando dados históricos...")

    # Tenta carregar dados de 1 hora por 1 ano
    if not optimizer.load_data(symbol="EURUSD=X", period="1y", interval="1h"):
        # Fallback para dados diários
        if not optimizer.load_data(symbol="EURUSD=X", period="2y", interval="1d"):
            print("Não foi possível carregar dados históricos")
            return

    # Otimiza thresholds
    print("\n[2/3] Otimizando thresholds...")
    thresholds = optimizer.optimize_thresholds()

    # Gera código
    print("\n[3/3] Gerando código otimizado...")
    result = optimizer.generate_optimized_code()

    print("\n" + "=" * 60)
    print("RESUMO DA OTIMIZAÇÃO")
    print("=" * 60)
    print(f"\nThresholds otimizados para EURUSD:")
    print(f"  BAIXA:  volatilidade < {result['low_threshold']:.4f} pips")
    print(f"  MÉDIA:  {result['low_threshold']:.4f} <= vol < {result['high_threshold']:.4f} pips")
    print(f"  ALTA:   volatilidade >= {result['high_threshold']:.4f} pips")

    return result


if __name__ == "__main__":
    result = main()
