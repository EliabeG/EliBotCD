#!/usr/bin/env python3
"""
================================================================================
OTIMIZACAO DO HMM PARA M5
================================================================================

Testa diferentes abordagens para melhorar o Win Rate:
1. Retornos padronizados (z-score)
2. Retornos absolutos
3. Features multiplas (retorno + volatilidade)
4. 2 estados em vez de 3
5. Volatilidade realizada direta (sem HMM)
6. ATR como feature
================================================================================
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from enum import Enum
import numpy as np

from api.fxopen_historical_ws import Bar, get_historical_data_with_spread_sync

try:
    from hmmlearn import hmm
except ImportError:
    print("ERRO: hmmlearn nao instalado")
    sys.exit(1)

# ===============================================================================
# PARAMETROS FIXOS
# ===============================================================================
STOP_LOSS_PIPS = 10.0
TAKE_PROFIT_PIPS = 20.0
MIN_PRICES_WARMUP = 6624
SIGNAL_COOLDOWN = 10
TREND_LOOKBACK = 5
# ===============================================================================


class PositionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    type: PositionType
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float


@dataclass
class BacktestResult:
    name: str
    total_trades: int
    win_rate: float
    pnl_pips: float
    profit_factor: float
    max_drawdown: float


# ===============================================================================
# ABORDAGEM 1: HMM com Retornos Padronizados
# ===============================================================================
class HMM_Standardized:
    """HMM treinado com retornos padronizados (z-score)"""

    def __init__(self, n_states=3, window=200):
        self.n_states = n_states
        self.window = window
        self.hmm_model = None
        self.mean = 0
        self.std = 1
        self.returns_buffer = deque(maxlen=window)

    def train(self, prices: np.ndarray):
        returns = np.diff(np.log(prices))
        self.mean = np.mean(returns)
        self.std = np.std(returns)

        # Padronizar
        standardized = (returns - self.mean) / self.std
        rets = standardized.reshape(-1, 1)

        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.hmm_model.fit(rets)

        # Inicializar buffer
        for r in standardized[-self.window:]:
            self.returns_buffer.append(r)

    def add_price(self, price: float, prev_price: float):
        if prev_price > 0:
            ret = np.log(price / prev_price)
            standardized = (ret - self.mean) / self.std
            self.returns_buffer.append(standardized)

    def get_signal_strength(self) -> float:
        """Retorna forca do sinal (0-1)"""
        if len(self.returns_buffer) < 10:
            return 0.0

        try:
            rets = np.array(list(self.returns_buffer)).reshape(-1, 1)
            probs = self.hmm_model.predict_proba(rets)
            current_probs = probs[-1]

            # Estado de alta volatilidade (maior covariancia)
            # Com padronizacao, estados ficam mais bem definidos
            high_vol_prob = max(current_probs[1:]) if len(current_probs) > 1 else 0
            return high_vol_prob
        except:
            return 0.0


# ===============================================================================
# ABORDAGEM 2: HMM com Retornos Absolutos
# ===============================================================================
class HMM_AbsoluteReturns:
    """HMM treinado com retornos absolutos (captura volatilidade)"""

    def __init__(self, n_states=3, window=200):
        self.n_states = n_states
        self.window = window
        self.hmm_model = None
        self.returns_buffer = deque(maxlen=window)

    def train(self, prices: np.ndarray):
        returns = np.abs(np.diff(np.log(prices)))
        rets = returns.reshape(-1, 1)

        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.hmm_model.fit(rets)

        for r in returns[-self.window:]:
            self.returns_buffer.append(r)

    def add_price(self, price: float, prev_price: float):
        if prev_price > 0:
            ret = abs(np.log(price / prev_price))
            self.returns_buffer.append(ret)

    def get_signal_strength(self) -> float:
        if len(self.returns_buffer) < 10:
            return 0.0

        try:
            rets = np.array(list(self.returns_buffer)).reshape(-1, 1)
            probs = self.hmm_model.predict_proba(rets)
            current_probs = probs[-1]

            # Identificar estado de alta volatilidade (maior media)
            means = [self.hmm_model.means_[i][0] for i in range(self.n_states)]
            high_vol_state = np.argmax(means)

            return current_probs[high_vol_state]
        except:
            return 0.0


# ===============================================================================
# ABORDAGEM 3: HMM com Features Multiplas
# ===============================================================================
class HMM_MultiFeature:
    """HMM com retorno + volatilidade realizada"""

    def __init__(self, n_states=3, window=200, vol_window=20):
        self.n_states = n_states
        self.window = window
        self.vol_window = vol_window
        self.hmm_model = None
        self.features_buffer = deque(maxlen=window)
        self.returns_for_vol = deque(maxlen=vol_window)

    def train(self, prices: np.ndarray):
        returns = np.diff(np.log(prices))

        # Calcular features: [retorno, volatilidade]
        features = []
        vol_buffer = deque(maxlen=self.vol_window)

        for i, ret in enumerate(returns):
            vol_buffer.append(ret)
            if len(vol_buffer) >= self.vol_window:
                vol = np.std(list(vol_buffer))
                features.append([ret, vol])

        features = np.array(features)

        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.hmm_model.fit(features)

        for f in features[-self.window:]:
            self.features_buffer.append(f)

        for r in returns[-self.vol_window:]:
            self.returns_for_vol.append(r)

    def add_price(self, price: float, prev_price: float):
        if prev_price > 0:
            ret = np.log(price / prev_price)
            self.returns_for_vol.append(ret)

            if len(self.returns_for_vol) >= self.vol_window:
                vol = np.std(list(self.returns_for_vol))
                self.features_buffer.append([ret, vol])

    def get_signal_strength(self) -> float:
        if len(self.features_buffer) < 10:
            return 0.0

        try:
            features = np.array(list(self.features_buffer))
            probs = self.hmm_model.predict_proba(features)
            current_probs = probs[-1]

            # Estado de alta volatilidade
            means = [self.hmm_model.means_[i][1] for i in range(self.n_states)]  # Coluna de vol
            high_vol_state = np.argmax(means)

            return current_probs[high_vol_state]
        except:
            return 0.0


# ===============================================================================
# ABORDAGEM 4: HMM com 2 Estados
# ===============================================================================
class HMM_TwoStates:
    """HMM com apenas 2 estados (baixa/alta volatilidade)"""

    def __init__(self, window=200):
        self.window = window
        self.hmm_model = None
        self.returns_buffer = deque(maxlen=window)

    def train(self, prices: np.ndarray):
        returns = np.abs(np.diff(np.log(prices)))
        rets = returns.reshape(-1, 1)

        self.hmm_model = hmm.GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.hmm_model.fit(rets)

        for r in returns[-self.window:]:
            self.returns_buffer.append(r)

    def add_price(self, price: float, prev_price: float):
        if prev_price > 0:
            ret = abs(np.log(price / prev_price))
            self.returns_buffer.append(ret)

    def get_signal_strength(self) -> float:
        if len(self.returns_buffer) < 10:
            return 0.0

        try:
            rets = np.array(list(self.returns_buffer)).reshape(-1, 1)
            probs = self.hmm_model.predict_proba(rets)
            current_probs = probs[-1]

            # Estado de alta volatilidade
            means = [self.hmm_model.means_[i][0] for i in range(2)]
            high_vol_state = np.argmax(means)

            return current_probs[high_vol_state]
        except:
            return 0.0


# ===============================================================================
# ABORDAGEM 5: Volatilidade Realizada (sem HMM)
# ===============================================================================
class RealizedVolatility:
    """Usa volatilidade realizada diretamente"""

    def __init__(self, window=50):
        self.window = window
        self.returns_buffer = deque(maxlen=window)
        self.vol_percentile_75 = 0
        self.vol_percentile_90 = 0

    def train(self, prices: np.ndarray):
        returns = np.diff(np.log(prices))

        # Calcular volatilidades historicas
        vols = []
        for i in range(self.window, len(returns)):
            vol = np.std(returns[i-self.window:i])
            vols.append(vol)

        self.vol_percentile_75 = np.percentile(vols, 75)
        self.vol_percentile_90 = np.percentile(vols, 90)

        for r in returns[-self.window:]:
            self.returns_buffer.append(r)

    def add_price(self, price: float, prev_price: float):
        if prev_price > 0:
            ret = np.log(price / prev_price)
            self.returns_buffer.append(ret)

    def get_signal_strength(self) -> float:
        if len(self.returns_buffer) < self.window:
            return 0.0

        vol = np.std(list(self.returns_buffer))

        # Normalizar: 0 se abaixo de P75, 1 se acima de P90
        if vol <= self.vol_percentile_75:
            return 0.0
        elif vol >= self.vol_percentile_90:
            return 1.0
        else:
            return (vol - self.vol_percentile_75) / (self.vol_percentile_90 - self.vol_percentile_75)


# ===============================================================================
# ABORDAGEM 6: ATR Normalizado
# ===============================================================================
class ATR_Indicator:
    """Usa ATR normalizado como indicador de volatilidade"""

    def __init__(self, window=14):
        self.window = window
        self.atr_buffer = deque(maxlen=window)
        self.atr_percentile_75 = 0
        self.atr_percentile_90 = 0

    def train(self, bars: List[Bar]):
        # Calcular TR para cada barra
        trs = []
        for i in range(1, len(bars)):
            high = bars[i].high
            low = bars[i].low
            prev_close = bars[i-1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            trs.append(tr)

        # Calcular ATRs historicos
        atrs = []
        for i in range(self.window, len(trs)):
            atr = np.mean(trs[i-self.window:i])
            atrs.append(atr)

        self.atr_percentile_75 = np.percentile(atrs, 75)
        self.atr_percentile_90 = np.percentile(atrs, 90)

        for tr in trs[-self.window:]:
            self.atr_buffer.append(tr)

    def add_bar(self, bar: Bar, prev_close: float):
        tr = max(
            bar.high - bar.low,
            abs(bar.high - prev_close),
            abs(bar.low - prev_close)
        )
        self.atr_buffer.append(tr)

    def get_signal_strength(self) -> float:
        if len(self.atr_buffer) < self.window:
            return 0.0

        atr = np.mean(list(self.atr_buffer))

        if atr <= self.atr_percentile_75:
            return 0.0
        elif atr >= self.atr_percentile_90:
            return 1.0
        else:
            return (atr - self.atr_percentile_75) / (self.atr_percentile_90 - self.atr_percentile_75)


# ===============================================================================
# FUNCAO DE BACKTEST GENERICA
# ===============================================================================
def run_backtest(
    bars: List[Bar],
    warmup_bars: int,
    indicator,
    indicator_type: str,
    threshold: float,
    direction: str = 'contra'
) -> BacktestResult:
    """Backtest generico para qualquer indicador"""

    pip = 0.0001
    total_bars = len(bars)

    trades_pnl = []
    position: Optional[Position] = None
    pending_signal = None
    cooldown_counter = 0

    # Buffer de closes para direcao
    closes = deque(maxlen=TREND_LOOKBACK + 10)
    for bar in bars[:warmup_bars]:
        closes.append(bar.close)

    prev_price = bars[warmup_bars - 1].close

    for i in range(warmup_bars, total_bars):
        bar = bars[i]
        closes.append(bar.close)

        # Atualizar indicador
        if indicator_type == 'atr':
            indicator.add_bar(bar, prev_price)
        else:
            indicator.add_price(bar.close, prev_price)

        prev_price = bar.close

        # 1. Executar sinal pendente
        if pending_signal and position is None:
            spread = bar.spread_pips * pip if bar.has_spread_data else 0.2 * pip

            if pending_signal == 'BUY':
                entry_price = bar.ask_open if bar.ask_open else bar.open + spread/2
                pos_type = PositionType.LONG
                sl = entry_price - (STOP_LOSS_PIPS * pip)
                tp = entry_price + (TAKE_PROFIT_PIPS * pip)
            else:
                entry_price = bar.bid_open if bar.bid_open else bar.open - spread/2
                pos_type = PositionType.SHORT
                sl = entry_price + (STOP_LOSS_PIPS * pip)
                tp = entry_price - (TAKE_PROFIT_PIPS * pip)

            position = Position(pos_type, entry_price, bar.timestamp, sl, tp)
            pending_signal = None

        # 2. Verificar stop/take
        if position:
            exit_price = None

            bid_low = bar.bid_low if bar.bid_low else bar.low
            ask_high = bar.ask_high if bar.ask_high else bar.high

            if position.type == PositionType.LONG:
                if bid_low <= position.stop_loss:
                    exit_price = position.stop_loss
                elif ask_high >= position.take_profit:
                    exit_price = position.take_profit
            else:
                if ask_high >= position.stop_loss:
                    exit_price = position.stop_loss
                elif bid_low <= position.take_profit:
                    exit_price = position.take_profit

            if exit_price:
                if position.type == PositionType.LONG:
                    pnl = (exit_price - position.entry_price) / pip
                else:
                    pnl = (position.entry_price - exit_price) / pip

                trades_pnl.append(pnl)
                position = None

        # 3. Gerar sinal
        if position is None and pending_signal is None:
            if cooldown_counter > 0:
                cooldown_counter -= 1
            else:
                signal_strength = indicator.get_signal_strength()

                if signal_strength >= threshold:
                    # Calcular direcao
                    if len(closes) >= TREND_LOOKBACK + 2:
                        closes_list = list(closes)
                        recent = closes_list[-2]
                        past = closes_list[-(TREND_LOOKBACK + 2)]
                        trend = 1 if recent > past else -1

                        if direction == 'contra':
                            pending_signal = 'SELL' if trend == 1 else 'BUY'
                        else:
                            pending_signal = 'BUY' if trend == 1 else 'SELL'

                        cooldown_counter = SIGNAL_COOLDOWN

    # Calcular metricas
    if not trades_pnl:
        return BacktestResult("", 0, 0.0, 0.0, 0.0, 0.0)

    pnls = np.array(trades_pnl)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    total_trades = len(pnls)
    win_rate = len(wins) / total_trades * 100
    pnl_total = float(np.sum(pnls))
    profit_factor = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else 0

    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max(peak - equity))

    return BacktestResult("", total_trades, win_rate, pnl_total, profit_factor, max_dd)


def main():
    print("=" * 70)
    print("  OTIMIZACAO DO HMM PARA M5")
    print("=" * 70)

    breakeven = STOP_LOSS_PIPS / (STOP_LOSS_PIPS + TAKE_PROFIT_PIPS) * 100
    print(f"\n  Breakeven WR: {breakeven:.1f}%")
    print(f"  SL: {STOP_LOSS_PIPS} pips, TP: {TAKE_PROFIT_PIPS} pips")

    # Baixar dados
    end_time = datetime.now(timezone.utc)
    start_time = datetime(2025, 7, 1, tzinfo=timezone.utc)

    print(f"\n  Periodo: {start_time.date()} a {end_time.date()}")
    print("\n  Baixando dados...")

    bars = get_historical_data_with_spread_sync(
        symbol="EURUSD",
        periodicity="M5",
        start_time=start_time,
        end_time=end_time
    )

    if not bars or len(bars) < MIN_PRICES_WARMUP + 100:
        print("  ERRO: Dados insuficientes")
        return

    print(f"  Barras: {len(bars)}")

    # Preparar dados de warmup
    warmup_prices = np.array([bar.close for bar in bars[:MIN_PRICES_WARMUP]])
    warmup_bars_list = bars[:MIN_PRICES_WARMUP]

    # Thresholds para testar
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    results = []

    # =========================================================================
    # TESTE 1: HMM Padronizado
    # =========================================================================
    print("\n" + "=" * 70)
    print("  TESTE 1: HMM com Retornos Padronizados")
    print("=" * 70)

    for threshold in thresholds:
        indicator = HMM_Standardized(n_states=3, window=200)
        indicator.train(warmup_prices)

        result = run_backtest(bars, MIN_PRICES_WARMUP, indicator, 'hmm', threshold, 'contra')
        result.name = f"HMM_Std_t{threshold}"

        if result.total_trades >= 50:
            results.append(result)
            print(f"    Threshold {threshold}: {result.total_trades} trades, "
                  f"WR={result.win_rate:.1f}%, PnL={result.pnl_pips:+.0f}")

    # =========================================================================
    # TESTE 2: HMM com Retornos Absolutos
    # =========================================================================
    print("\n" + "=" * 70)
    print("  TESTE 2: HMM com Retornos Absolutos")
    print("=" * 70)

    for threshold in thresholds:
        indicator = HMM_AbsoluteReturns(n_states=3, window=200)
        indicator.train(warmup_prices)

        result = run_backtest(bars, MIN_PRICES_WARMUP, indicator, 'hmm', threshold, 'contra')
        result.name = f"HMM_Abs_t{threshold}"

        if result.total_trades >= 50:
            results.append(result)
            print(f"    Threshold {threshold}: {result.total_trades} trades, "
                  f"WR={result.win_rate:.1f}%, PnL={result.pnl_pips:+.0f}")

    # =========================================================================
    # TESTE 3: HMM Multi-Feature
    # =========================================================================
    print("\n" + "=" * 70)
    print("  TESTE 3: HMM Multi-Feature (retorno + volatilidade)")
    print("=" * 70)

    for threshold in thresholds:
        indicator = HMM_MultiFeature(n_states=3, window=200, vol_window=20)
        indicator.train(warmup_prices)

        result = run_backtest(bars, MIN_PRICES_WARMUP, indicator, 'hmm', threshold, 'contra')
        result.name = f"HMM_Multi_t{threshold}"

        if result.total_trades >= 50:
            results.append(result)
            print(f"    Threshold {threshold}: {result.total_trades} trades, "
                  f"WR={result.win_rate:.1f}%, PnL={result.pnl_pips:+.0f}")

    # =========================================================================
    # TESTE 4: HMM com 2 Estados
    # =========================================================================
    print("\n" + "=" * 70)
    print("  TESTE 4: HMM com 2 Estados")
    print("=" * 70)

    for threshold in thresholds:
        indicator = HMM_TwoStates(window=200)
        indicator.train(warmup_prices)

        result = run_backtest(bars, MIN_PRICES_WARMUP, indicator, 'hmm', threshold, 'contra')
        result.name = f"HMM_2St_t{threshold}"

        if result.total_trades >= 50:
            results.append(result)
            print(f"    Threshold {threshold}: {result.total_trades} trades, "
                  f"WR={result.win_rate:.1f}%, PnL={result.pnl_pips:+.0f}")

    # =========================================================================
    # TESTE 5: Volatilidade Realizada (sem HMM)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  TESTE 5: Volatilidade Realizada (sem HMM)")
    print("=" * 70)

    for threshold in thresholds:
        indicator = RealizedVolatility(window=50)
        indicator.train(warmup_prices)

        result = run_backtest(bars, MIN_PRICES_WARMUP, indicator, 'vol', threshold, 'contra')
        result.name = f"RealVol_t{threshold}"

        if result.total_trades >= 50:
            results.append(result)
            print(f"    Threshold {threshold}: {result.total_trades} trades, "
                  f"WR={result.win_rate:.1f}%, PnL={result.pnl_pips:+.0f}")

    # =========================================================================
    # TESTE 6: ATR
    # =========================================================================
    print("\n" + "=" * 70)
    print("  TESTE 6: ATR Normalizado")
    print("=" * 70)

    for threshold in thresholds:
        indicator = ATR_Indicator(window=14)
        indicator.train(warmup_bars_list)

        result = run_backtest(bars, MIN_PRICES_WARMUP, indicator, 'atr', threshold, 'contra')
        result.name = f"ATR_t{threshold}"

        if result.total_trades >= 50:
            results.append(result)
            print(f"    Threshold {threshold}: {result.total_trades} trades, "
                  f"WR={result.win_rate:.1f}%, PnL={result.pnl_pips:+.0f}")

    # =========================================================================
    # RESULTADOS FINAIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("  RANKING - ORDENADO POR WIN RATE")
    print("=" * 70)

    results.sort(key=lambda x: x.win_rate, reverse=True)

    print(f"\n  {'#':<3} {'Nome':<20} {'Trades':<8} {'WR%':<8} {'PnL':<10} {'PF':<6} {'DD':<8}")
    print("  " + "-" * 70)

    for i, r in enumerate(results[:20]):
        print(f"  {i+1:<3} {r.name:<20} {r.total_trades:<8} {r.win_rate:<8.1f} "
              f"{r.pnl_pips:+<10.0f} {r.profit_factor:<6.2f} {r.max_drawdown:<8.0f}")

    # Melhor por WR
    if results:
        best_wr = max(results, key=lambda x: x.win_rate)
        print(f"\n  MELHOR WIN RATE:")
        print(f"    {best_wr.name}: WR={best_wr.win_rate:.1f}%, PnL={best_wr.pnl_pips:+.0f}, PF={best_wr.profit_factor:.2f}")

        best_pnl = max(results, key=lambda x: x.pnl_pips)
        print(f"\n  MELHOR PNL:")
        print(f"    {best_pnl.name}: WR={best_pnl.win_rate:.1f}%, PnL={best_pnl.pnl_pips:+.0f}, PF={best_pnl.profit_factor:.2f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
