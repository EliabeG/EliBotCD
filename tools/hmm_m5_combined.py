#!/usr/bin/env python3
"""
================================================================================
OTIMIZACAO COMBINADA PARA MAXIMIZAR WIN RATE
================================================================================

Combina os melhores indicadores encontrados:
1. Volatilidade Realizada (melhor WR)
2. ATR (segundo melhor)
3. Combinacao de ambos
4. Adiciona filtros extras (Lyapunov, momentum)
================================================================================
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from typing import List, Optional
from dataclasses import dataclass
from collections import deque
from enum import Enum
import numpy as np
import itertools

from api.fxopen_historical_ws import Bar, get_historical_data_with_spread_sync

# ===============================================================================
# PARAMETROS FIXOS
# ===============================================================================
STOP_LOSS_PIPS = 10.0
TAKE_PROFIT_PIPS = 20.0
MIN_PRICES_WARMUP = 6624
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
    params: dict
    total_trades: int
    win_rate: float
    pnl_pips: float
    profit_factor: float
    max_drawdown: float
    avg_trade: float


class CombinedIndicator:
    """Indicador combinado com multiplos filtros"""

    def __init__(self, vol_window=50, atr_window=14, lyap_window=50, momentum_window=10):
        self.vol_window = vol_window
        self.atr_window = atr_window
        self.lyap_window = lyap_window
        self.momentum_window = momentum_window

        self.returns_buffer = deque(maxlen=max(vol_window, lyap_window))
        self.tr_buffer = deque(maxlen=atr_window)
        self.prices_buffer = deque(maxlen=momentum_window + 1)

        # Percentis calibrados no warmup
        self.vol_p75 = 0
        self.vol_p90 = 0
        self.atr_p75 = 0
        self.atr_p90 = 0
        self.lyap_p75 = 0
        self.lyap_p90 = 0

    def train(self, bars: List[Bar]):
        """Calibra os percentis no warmup"""
        prices = [bar.close for bar in bars]
        returns = np.diff(np.log(prices))

        # Volatilidades
        vols = []
        for i in range(self.vol_window, len(returns)):
            vol = np.std(returns[i-self.vol_window:i])
            vols.append(vol)

        self.vol_p75 = np.percentile(vols, 75)
        self.vol_p90 = np.percentile(vols, 90)

        # ATRs
        trs = []
        for i in range(1, len(bars)):
            tr = max(
                bars[i].high - bars[i].low,
                abs(bars[i].high - bars[i-1].close),
                abs(bars[i].low - bars[i-1].close)
            )
            trs.append(tr)

        atrs = []
        for i in range(self.atr_window, len(trs)):
            atr = np.mean(trs[i-self.atr_window:i])
            atrs.append(atr)

        self.atr_p75 = np.percentile(atrs, 75)
        self.atr_p90 = np.percentile(atrs, 90)

        # Lyapunovs
        lyaps = []
        for i in range(self.lyap_window, len(returns)):
            window_returns = returns[i-self.lyap_window:i]
            abs_returns = np.abs(window_returns)
            abs_returns = abs_returns[abs_returns > 1e-10]
            if len(abs_returns) > 0:
                lyap = np.mean(np.log(abs_returns)) + 10
                lyaps.append(max(0, lyap))

        self.lyap_p75 = np.percentile(lyaps, 75)
        self.lyap_p90 = np.percentile(lyaps, 90)

        # Inicializar buffers
        for r in returns[-self.vol_window:]:
            self.returns_buffer.append(r)

        for i in range(len(bars)-self.atr_window, len(bars)):
            if i > 0:
                tr = max(
                    bars[i].high - bars[i].low,
                    abs(bars[i].high - bars[i-1].close),
                    abs(bars[i].low - bars[i-1].close)
                )
                self.tr_buffer.append(tr)

        for bar in bars[-self.momentum_window-1:]:
            self.prices_buffer.append(bar.close)

    def add_bar(self, bar: Bar, prev_close: float):
        # Retorno
        if prev_close > 0:
            ret = np.log(bar.close / prev_close)
            self.returns_buffer.append(ret)

        # TR
        tr = max(
            bar.high - bar.low,
            abs(bar.high - prev_close),
            abs(bar.low - prev_close)
        )
        self.tr_buffer.append(tr)

        # Preco
        self.prices_buffer.append(bar.close)

    def get_volatility_score(self) -> float:
        """Score de volatilidade normalizado (0-1)"""
        if len(self.returns_buffer) < self.vol_window:
            return 0.0

        vol = np.std(list(self.returns_buffer)[-self.vol_window:])

        if vol <= self.vol_p75:
            return 0.0
        elif vol >= self.vol_p90:
            return 1.0
        else:
            return (vol - self.vol_p75) / (self.vol_p90 - self.vol_p75)

    def get_atr_score(self) -> float:
        """Score de ATR normalizado (0-1)"""
        if len(self.tr_buffer) < self.atr_window:
            return 0.0

        atr = np.mean(list(self.tr_buffer))

        if atr <= self.atr_p75:
            return 0.0
        elif atr >= self.atr_p90:
            return 1.0
        else:
            return (atr - self.atr_p75) / (self.atr_p90 - self.atr_p75)

    def get_lyapunov_score(self) -> float:
        """Score de Lyapunov normalizado (0-1)"""
        if len(self.returns_buffer) < self.lyap_window:
            return 0.0

        window_returns = list(self.returns_buffer)[-self.lyap_window:]
        abs_returns = np.abs(window_returns)
        abs_returns = abs_returns[abs_returns > 1e-10]

        if len(abs_returns) == 0:
            return 0.0

        lyap = np.mean(np.log(abs_returns)) + 10
        lyap = max(0, lyap)

        if lyap <= self.lyap_p75:
            return 0.0
        elif lyap >= self.lyap_p90:
            return 1.0
        else:
            return (lyap - self.lyap_p75) / (self.lyap_p90 - self.lyap_p75)

    def get_momentum(self) -> float:
        """Retorna momentum (-1 a 1)"""
        if len(self.prices_buffer) < self.momentum_window + 1:
            return 0.0

        prices = list(self.prices_buffer)
        return (prices[-1] - prices[0]) / prices[0]

    def get_trend_direction(self, lookback: int = 5) -> int:
        """Retorna direcao da tendencia"""
        if len(self.prices_buffer) < lookback + 1:
            return 0

        prices = list(self.prices_buffer)
        if prices[-1] > prices[-lookback-1]:
            return 1
        else:
            return -1


def run_backtest(
    bars: List[Bar],
    warmup_bars: int,
    vol_threshold: float,
    atr_threshold: float,
    lyap_threshold: float,
    use_vol: bool,
    use_atr: bool,
    use_lyap: bool,
    combine_mode: str,  # 'and', 'or', 'avg'
    signal_cooldown: int,
    trend_lookback: int,
    direction: str
) -> BacktestResult:
    """Backtest com indicador combinado"""

    pip = 0.0001
    total_bars = len(bars)

    indicator = CombinedIndicator(vol_window=50, atr_window=14, lyap_window=50)
    indicator.train(bars[:warmup_bars])

    trades_pnl = []
    position: Optional[Position] = None
    pending_signal = None
    cooldown_counter = 0

    prev_close = bars[warmup_bars - 1].close

    for i in range(warmup_bars, total_bars):
        bar = bars[i]
        indicator.add_bar(bar, prev_close)
        prev_close = bar.close

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
                # Calcular scores
                scores = []
                if use_vol:
                    scores.append(('vol', indicator.get_volatility_score(), vol_threshold))
                if use_atr:
                    scores.append(('atr', indicator.get_atr_score(), atr_threshold))
                if use_lyap:
                    scores.append(('lyap', indicator.get_lyapunov_score(), lyap_threshold))

                # Combinar scores
                signal_valid = False
                if scores:
                    if combine_mode == 'and':
                        signal_valid = all(score >= thresh for _, score, thresh in scores)
                    elif combine_mode == 'or':
                        signal_valid = any(score >= thresh for _, score, thresh in scores)
                    elif combine_mode == 'avg':
                        avg_score = np.mean([score for _, score, _ in scores])
                        avg_thresh = np.mean([thresh for _, _, thresh in scores])
                        signal_valid = avg_score >= avg_thresh

                if signal_valid:
                    trend = indicator.get_trend_direction(trend_lookback)
                    if trend != 0:
                        if direction == 'contra':
                            pending_signal = 'SELL' if trend == 1 else 'BUY'
                        else:
                            pending_signal = 'BUY' if trend == 1 else 'SELL'
                        cooldown_counter = signal_cooldown

    # Calcular metricas
    if not trades_pnl:
        return BacktestResult("", {}, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

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
    avg_trade = float(np.mean(pnls))

    return BacktestResult("", {}, total_trades, win_rate, pnl_total, profit_factor, max_dd, avg_trade)


def main():
    print("=" * 70)
    print("  OTIMIZACAO COMBINADA PARA MAXIMIZAR WIN RATE")
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

    # Parametros para otimizar
    vol_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    atr_thresholds = [0.3, 0.4, 0.5, 0.6]
    lyap_thresholds = [0.3, 0.4, 0.5, 0.6]
    cooldowns = [5, 10, 15, 20]
    lookbacks = [3, 5, 10]
    directions = ['contra']  # Contra-tendencia foi melhor

    # Combinacoes a testar
    combinations = [
        # (use_vol, use_atr, use_lyap, combine_mode, name)
        (True, False, False, 'and', 'Vol_Only'),
        (False, True, False, 'and', 'ATR_Only'),
        (False, False, True, 'and', 'Lyap_Only'),
        (True, True, False, 'and', 'Vol+ATR_AND'),
        (True, False, True, 'and', 'Vol+Lyap_AND'),
        (True, True, False, 'or', 'Vol+ATR_OR'),
        (True, True, True, 'and', 'Vol+ATR+Lyap_AND'),
        (True, True, True, 'avg', 'Vol+ATR+Lyap_AVG'),
    ]

    print("\n" + "=" * 70)
    print("  EXECUTANDO OTIMIZACAO")
    print("=" * 70)

    all_results = []
    total_combos = len(combinations) * len(vol_thresholds) * len(cooldowns) * len(lookbacks)
    tested = 0

    for use_vol, use_atr, use_lyap, combine_mode, combo_name in combinations:
        for vol_t in vol_thresholds:
            for cool in cooldowns:
                for look in lookbacks:
                    result = run_backtest(
                        bars, MIN_PRICES_WARMUP,
                        vol_threshold=vol_t,
                        atr_threshold=vol_t,  # Mesmo threshold para simplificar
                        lyap_threshold=vol_t,
                        use_vol=use_vol,
                        use_atr=use_atr,
                        use_lyap=use_lyap,
                        combine_mode=combine_mode,
                        signal_cooldown=cool,
                        trend_lookback=look,
                        direction='contra'
                    )

                    if result.total_trades >= 30:  # Minimo de trades
                        result.name = f"{combo_name}_t{vol_t}_c{cool}_l{look}"
                        result.params = {
                            'combo': combo_name,
                            'threshold': vol_t,
                            'cooldown': cool,
                            'lookback': look
                        }
                        all_results.append(result)

                    tested += 1
                    if tested % 100 == 0:
                        print(f"    Testado: {tested}/{total_combos}")

    print(f"\n  Total de combinacoes validas: {len(all_results)}")

    # Ordenar por Win Rate
    all_results.sort(key=lambda x: x.win_rate, reverse=True)

    # Resultados
    print("\n" + "=" * 70)
    print("  TOP 20 - ORDENADO POR WIN RATE")
    print("=" * 70)

    print(f"\n  {'#':<3} {'Nome':<30} {'Trades':<7} {'WR%':<7} {'PnL':<8} {'PF':<6} {'AvgT':<7}")
    print("  " + "-" * 75)

    for i, r in enumerate(all_results[:20]):
        print(f"  {i+1:<3} {r.name:<30} {r.total_trades:<7} {r.win_rate:<7.1f} "
              f"{r.pnl_pips:+<8.0f} {r.profit_factor:<6.2f} {r.avg_trade:<+7.1f}")

    # Analise por tipo de combinacao
    print("\n" + "=" * 70)
    print("  MELHOR POR TIPO DE COMBINACAO")
    print("=" * 70)

    combo_types = set([r.params['combo'] for r in all_results])
    for combo in sorted(combo_types):
        combo_results = [r for r in all_results if r.params['combo'] == combo]
        if combo_results:
            best = max(combo_results, key=lambda x: x.win_rate)
            print(f"\n  {combo}:")
            print(f"    Melhor: {best.name}")
            print(f"    WR={best.win_rate:.1f}%, PnL={best.pnl_pips:+.0f}, Trades={best.total_trades}")

    # Melhor resultado geral
    if all_results:
        best = all_results[0]
        print("\n" + "=" * 70)
        print("  MELHOR RESULTADO GERAL")
        print("=" * 70)

        print(f"\n  Configuracao: {best.name}")
        print(f"  Parametros: {best.params}")
        print(f"  Trades: {best.total_trades}")
        print(f"  Win Rate: {best.win_rate:.1f}%")
        print(f"  PnL: {best.pnl_pips:+.0f} pips")
        print(f"  Profit Factor: {best.profit_factor:.2f}")
        print(f"  Avg Trade: {best.avg_trade:+.1f} pips")
        print(f"  Max Drawdown: {best.max_drawdown:.0f} pips")

    # WR acima de 40%
    wr_40_plus = [r for r in all_results if r.win_rate >= 40]
    if wr_40_plus:
        print("\n" + "=" * 70)
        print(f"  COMBINACOES COM WR >= 40% ({len(wr_40_plus)} encontradas)")
        print("=" * 70)

        for r in wr_40_plus[:10]:
            print(f"    {r.name}: WR={r.win_rate:.1f}%, PnL={r.pnl_pips:+.0f}, Trades={r.total_trades}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
