"""
Indicador de Volatilidade em Tempo Real
Carrega dados históricos e atualiza instantaneamente com cada tick
"""
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Callable
from collections import deque
from scipy.stats import entropy


class RealtimeVolatility:
    """
    Indicador de volatilidade instantâneo.
    - Carrega dados históricos no início
    - Atualiza em tempo real com cada tick
    """

    def __init__(
        self,
        candle_seconds: int = 5,
        max_candles: int = 500,
        parkinson_window: int = 20,
        hurst_window: int = 50,
        entropy_window: int = 20
    ):
        self.candle_seconds = candle_seconds
        self.max_candles = max_candles
        self.parkinson_window = parkinson_window
        self.hurst_window = hurst_window
        self.entropy_window = entropy_window

        # Buffer de candles (deque para eficiência)
        self.candles: deque = deque(maxlen=max_candles)

        # Candle atual sendo formado
        self.current_candle: Optional[dict] = None
        self.current_candle_start: Optional[datetime] = None

        # Cache de cálculos (evita recalcular tudo)
        self._cache = {
            'parkinson_vol': None,
            'hurst': None,
            'entropy': None,
            'classification': 'INDEFINIDO',
            'last_update': None
        }

        # Callbacks
        self.on_update: Optional[Callable] = None

    def load_historical(self, df: pd.DataFrame):
        """
        Carrega dados históricos (OHLC) no buffer.
        DataFrame deve ter colunas: timestamp, open, high, low, close
        """
        self.candles.clear()

        for _, row in df.iterrows():
            candle = {
                'timestamp': row['timestamp'] if isinstance(row['timestamp'], datetime) else pd.to_datetime(row['timestamp']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'tick_count': row.get('volume', 1)
            }
            self.candles.append(candle)

        # Calcula indicadores iniciais
        self._update_indicators()

        print(f"Histórico carregado: {len(self.candles)} candles")

    def add_tick(self, price: float, timestamp: datetime):
        """
        Adiciona um tick e atualiza indicadores instantaneamente.
        """
        # Determina o início do candle atual
        candle_start = timestamp.replace(
            second=(timestamp.second // self.candle_seconds) * self.candle_seconds,
            microsecond=0
        )

        # Novo candle?
        if self.current_candle_start != candle_start:
            # Salva candle anterior se existir
            if self.current_candle is not None:
                self.candles.append(self.current_candle)

            # Inicia novo candle
            self.current_candle_start = candle_start
            self.current_candle = {
                'timestamp': candle_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'tick_count': 1
            }

            # Recalcula indicadores (novo candle = recálculo)
            self._update_indicators()
        else:
            # Atualiza candle atual
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price
            self.current_candle['tick_count'] += 1

            # Atualização rápida (só o preço mudou no candle atual)
            self._quick_update(price)

        # Notifica callback se existir
        if self.on_update:
            self.on_update(self.get_state())

    def _get_all_candles(self) -> list:
        """Retorna todos os candles incluindo o atual"""
        all_candles = list(self.candles)
        if self.current_candle:
            all_candles.append(self.current_candle)
        return all_candles

    def _update_indicators(self):
        """Recalcula todos os indicadores"""
        candles = self._get_all_candles()

        if len(candles) < 3:
            return

        # Extrai arrays
        high = np.array([c['high'] for c in candles])
        low = np.array([c['low'] for c in candles])
        close = np.array([c['close'] for c in candles])

        # Parkinson Volatility
        self._cache['parkinson_vol'] = self._calc_parkinson(high, low)

        # Hurst Exponent
        self._cache['hurst'] = self._calc_hurst(close)

        # Shannon Entropy
        returns = np.diff(np.log(close))
        self._cache['entropy'] = self._calc_entropy(returns)

        # Classificação
        self._cache['classification'] = self._classify()
        self._cache['last_update'] = datetime.now(timezone.utc)

    def _quick_update(self, price: float):
        """Atualização rápida quando só o preço atual muda"""
        # Por enquanto, faz recálculo completo
        # Pode ser otimizado para atualização incremental
        self._update_indicators()

    def _calc_parkinson(self, high: np.ndarray, low: np.ndarray) -> float:
        """Volatilidade de Parkinson"""
        window = min(self.parkinson_window, len(high))
        h = high[-window:]
        l = low[-window:]

        # Evita divisão por zero apenas quando low é zero
        l_safe = np.where(l == 0, 1e-10, l)
        ratio = h / l_safe

        # Clip mínimo muito pequeno para evitar log(1) = 0, mas não inflar volatilidade
        ratio = np.maximum(ratio, 1.0000001)

        log_hl_sq = np.log(ratio) ** 2
        return np.sqrt(np.sum(log_hl_sq) / (4 * len(h) * np.log(2)))

    def _calc_hurst(self, prices: np.ndarray) -> float:
        """Hurst simplificado via autocorrelação"""
        window = min(self.hurst_window, len(prices))
        p = prices[-window:]

        if len(p) < 10:
            return 0.5

        returns = np.diff(np.log(p))
        if len(returns) < 5:
            return 0.5

        mean_ret = np.mean(returns)
        var_ret = np.var(returns)

        if var_ret == 0:
            return 0.5

        autocorr = np.sum((returns[:-1] - mean_ret) * (returns[1:] - mean_ret)) / (len(returns) - 1) / var_ret
        hurst = 0.5 + autocorr * 0.3

        return np.clip(hurst, 0.1, 0.9)

    def _calc_entropy(self, returns: np.ndarray) -> float:
        """Entropia de Shannon normalizada"""
        window = min(self.entropy_window, len(returns))
        r = returns[-window:]

        if len(r) < 5:
            return 0.5

        bins = 10
        hist, _ = np.histogram(r, bins=bins, density=True)
        hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        hist = hist[hist > 0]

        if len(hist) == 0:
            return 0.5

        max_entropy = np.log2(bins)
        return entropy(hist, base=2) / max_entropy

    def _classify(self) -> str:
        """
        Classifica volatilidade (OTIMIZADO com dados históricos EURUSD)

        Thresholds calibrados com 1 ano de dados históricos:
        - Baseado em percentis P30 e P80 da volatilidade de Parkinson
        - Ajustados para candles de 5 segundos
        - Validados com correlação positiva ao movimento futuro
        """
        vol = self._cache.get('parkinson_vol')
        ent = self._cache.get('entropy')

        if vol is None:
            return "INDEFINIDO"

        vol_pips = vol * 10000

        # Thresholds otimizados (calibrados com dados históricos)
        # Originais para 1h: LOW=6.38, HIGH=10.33 pips
        # Ajustados para 5s: scale = sqrt(5/3600) = 0.0373
        LOW_THRESHOLD = 0.238   # Percentil 30 (escalado)
        HIGH_THRESHOLD = 0.385  # Percentil 80 (escalado)

        if vol_pips < LOW_THRESHOLD:
            classification = "BAIXA"
        elif vol_pips >= HIGH_THRESHOLD:
            classification = "ALTA"
        else:
            classification = "MEDIA"

        # Ajuste fino pela entropia (incerteza do mercado)
        if ent is not None:
            mid_threshold = (LOW_THRESHOLD + HIGH_THRESHOLD) / 2

            # Alta entropia + vol média-alta → pode ser ALTA
            if ent > 0.85 and classification == "MEDIA":
                if vol_pips > mid_threshold:
                    classification = "ALTA"

            # Baixa entropia + vol média-baixa → pode ser BAIXA
            elif ent < 0.25 and classification == "MEDIA":
                if vol_pips < mid_threshold:
                    classification = "BAIXA"

        return classification

    def get_state(self) -> dict:
        """Retorna estado atual do indicador"""
        candles = self._get_all_candles()

        vol = self._cache.get('parkinson_vol')
        vol_pips = vol * 10000 if vol is not None else 0

        return {
            'volatility': vol_pips,
            'hurst': self._cache.get('hurst') or 0.5,
            'entropy': self._cache.get('entropy') or 0.5,
            'classification': self._cache.get('classification', 'INDEFINIDO'),
            'candles_count': len(candles),
            'last_price': candles[-1]['close'] if candles else 0,
            'last_update': self._cache.get('last_update')
        }

    def get_classification(self) -> str:
        """Retorna classificação atual"""
        return self._cache.get('classification', 'INDEFINIDO')

    def get_volatility_pips(self) -> float:
        """Retorna volatilidade em pips"""
        vol = self._cache.get('parkinson_vol', 0)
        return vol * 10000
