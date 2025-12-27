#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR PRM - 3000 COMBINACOES
================================================================================

Parametros fixos:
- hmm_threshold = 0.4
- SL = 10 pips, TP = 20 pips

Parametros otimizados:
- hmm_states_allowed: 7 combinacoes
- lyapunov_threshold: 12 valores
- signal_cooldown: 8 valores
- direction: 2 (tendencia/contra)
- trend_lookback: 3 valores

Total: ~3000 combinacoes
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque
import numpy as np
import itertools
import random

from api.fxopen_historical_ws import Bar, get_historical_data_with_spread_sync

try:
    from hmmlearn import hmm
except ImportError:
    print("ERRO: hmmlearn nao instalado")
    sys.exit(1)

# ===============================================================================
# PARAMETROS FIXOS
# ===============================================================================
HMM_THRESHOLD = 0.4
STOP_LOSS_PIPS = 10.0
TAKE_PROFIT_PIPS = 20.0
MIN_PRICES_WARMUP = 6624
TARGET_COMBINATIONS = 3000
# ===============================================================================

# ===============================================================================
# PARAMETROS A OTIMIZAR
# ===============================================================================
STATES_OPTIONS = [
    [0],
    [1],
    [2],
    [0, 1],
    [0, 2],
    [1, 2],
    [0, 1, 2]
]

LYAPUNOV_OPTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
COOLDOWN_OPTIONS = [3, 5, 10, 15, 20, 30, 50, 100]
DIRECTION_OPTIONS = ['trend', 'contra']
LOOKBACK_OPTIONS = [5, 10, 20]
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
class Trade:
    type: PositionType
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    pnl_pips: float
    exit_reason: str


@dataclass
class OptimizationResult:
    states_allowed: List[int]
    lyapunov_threshold: float
    signal_cooldown: int
    direction: str
    trend_lookback: int
    total_trades: int
    win_rate: float
    pnl_pips: float
    profit_factor: float
    max_drawdown: float


class IncrementalPRMIndicator:
    """PRM com calculo incremental"""

    def __init__(self, n_states=3, hmm_window=200, lyap_window=50):
        self.n_states = n_states
        self.hmm_window = hmm_window
        self.lyap_window = lyap_window
        self.hmm_model = None
        self.is_trained = False
        self.returns_buffer = deque(maxlen=hmm_window)
        self.prices_buffer = deque(maxlen=lyap_window + 10)
        self.last_state = 0
        self.last_hmm_prob = 0.0
        self.last_lyapunov = 0.0

    def initialize_buffers(self, prices: np.ndarray):
        returns = np.diff(np.log(prices))
        self.returns_buffer.clear()
        for r in returns[-self.hmm_window:]:
            self.returns_buffer.append(r)
        self.prices_buffer.clear()
        for p in prices[-(self.lyap_window + 10):]:
            self.prices_buffer.append(p)

    def train_hmm(self, prices: np.ndarray):
        if len(prices) < self.hmm_window:
            return False
        returns = np.diff(np.log(prices)).reshape(-1, 1)
        try:
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            self.hmm_model.fit(returns)
            self.is_trained = True
            self.initialize_buffers(prices)
            return True
        except:
            return False

    def add_price(self, price: float):
        if len(self.prices_buffer) > 0:
            last_price = self.prices_buffer[-1]
            if last_price > 0 and price > 0:
                new_return = np.log(price / last_price)
                self.returns_buffer.append(new_return)
        self.prices_buffer.append(price)

    def compute_state_and_prob(self) -> Tuple[int, float]:
        if not self.is_trained or len(self.returns_buffer) < 10:
            return 0, 0.0
        try:
            returns = np.array(list(self.returns_buffer)).reshape(-1, 1)
            probs = self.hmm_model.predict_proba(returns)
            current_probs = probs[-1]
            state = np.argmax(current_probs)
            hmm_prob = max(
                current_probs[1] if len(current_probs) > 1 else 0,
                current_probs[2] if len(current_probs) > 2 else 0
            )
            self.last_state = state
            self.last_hmm_prob = hmm_prob
            return state, hmm_prob
        except:
            return self.last_state, self.last_hmm_prob

    def compute_lyapunov(self) -> float:
        if len(self.prices_buffer) < self.lyap_window + 1:
            return 0.0
        try:
            prices = list(self.prices_buffer)[-self.lyap_window - 1:]
            returns = np.diff(np.log(prices))
            abs_returns = np.abs(returns)
            abs_returns = abs_returns[abs_returns > 1e-10]
            if len(abs_returns) == 0:
                return 0.0
            lyapunov = np.mean(np.log(abs_returns)) + 10
            lyapunov = max(0, lyapunov)
            self.last_lyapunov = lyapunov
            return lyapunov
        except:
            return self.last_lyapunov

    def analyze(self) -> dict:
        state, hmm_prob = self.compute_state_and_prob()
        lyapunov = self.compute_lyapunov()
        return {
            'state': state,
            'hmm_prob': hmm_prob,
            'lyapunov': lyapunov,
            'valid': self.is_trained and len(self.returns_buffer) >= 10
        }


def calculate_direction(closes: deque, lookback: int) -> int:
    min_bars = lookback + 2
    if len(closes) < min_bars:
        return 0
    closes_list = list(closes)
    recent = closes_list[-2]
    past = closes_list[-(lookback + 2)]
    trend = recent - past
    return 1 if trend > 0 else -1


def run_backtest(
    bars: List[Bar],
    warmup_bars: int,
    prm: IncrementalPRMIndicator,
    closes_buffer_init: deque,
    states_allowed: List[int],
    lyapunov_threshold: float,
    signal_cooldown: int,
    direction: str,
    trend_lookback: int
) -> OptimizationResult:
    """Backtest rapido com parametros especificos"""

    trades = []
    position: Optional[Position] = None
    pip = 0.0001
    cooldown_counter = 0

    # Resetar buffers do PRM
    prm.returns_buffer = deque(prm.returns_buffer, maxlen=prm.hmm_window)
    prm.prices_buffer = deque(prm.prices_buffer, maxlen=prm.lyap_window + 10)

    closes_buffer = deque(closes_buffer_init, maxlen=500)

    pending_signal = None
    total_bars = len(bars)

    for i in range(warmup_bars, total_bars):
        bar = bars[i]

        # Adicionar preco
        prm.add_price(bar.close)
        closes_buffer.append(bar.close)

        # 1. Executar sinal pendente
        if pending_signal and position is None:
            spread = bar.spread_pips * pip if bar.has_spread_data else 0.2 * pip

            if pending_signal == 'BUY':
                entry_price = (bar.ask_open if bar.ask_open else bar.open + spread/2)
                pos_type = PositionType.LONG
                sl = entry_price - (STOP_LOSS_PIPS * pip)
                tp = entry_price + (TAKE_PROFIT_PIPS * pip)
            else:
                entry_price = (bar.bid_open if bar.bid_open else bar.open - spread/2)
                pos_type = PositionType.SHORT
                sl = entry_price + (STOP_LOSS_PIPS * pip)
                tp = entry_price - (TAKE_PROFIT_PIPS * pip)

            position = Position(pos_type, entry_price, bar.timestamp, sl, tp)
            pending_signal = None

        # 2. Verificar stop/take
        if position:
            exit_price = None
            exit_reason = None

            bid_low = bar.bid_low if bar.bid_low else bar.low
            ask_high = bar.ask_high if bar.ask_high else bar.high

            if position.type == PositionType.LONG:
                if bid_low <= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = "stop_loss"
                elif ask_high >= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "take_profit"
            else:
                if ask_high >= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = "stop_loss"
                elif bid_low <= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "take_profit"

            if exit_price:
                if position.type == PositionType.LONG:
                    pnl_pips = (exit_price - position.entry_price) / pip
                else:
                    pnl_pips = (position.entry_price - exit_price) / pip

                trades.append(Trade(
                    position.type, position.entry_price, position.entry_time,
                    exit_price, bar.timestamp, pnl_pips, exit_reason
                ))
                position = None

        # 3. Gerar sinal
        if position is None and pending_signal is None:
            if cooldown_counter > 0:
                cooldown_counter -= 1
            else:
                result = prm.analyze()

                if result['valid']:
                    # Verificar estado
                    if result['state'] in states_allowed:
                        # Verificar thresholds
                        if (result['hmm_prob'] >= HMM_THRESHOLD and
                            result['lyapunov'] >= lyapunov_threshold):

                            dir_val = calculate_direction(closes_buffer, trend_lookback)
                            if dir_val != 0:
                                if direction == 'trend':
                                    pending_signal = 'BUY' if dir_val == 1 else 'SELL'
                                else:  # contra
                                    pending_signal = 'SELL' if dir_val == 1 else 'BUY'
                                cooldown_counter = signal_cooldown

    # Calcular metricas
    if not trades:
        return OptimizationResult(
            states_allowed, lyapunov_threshold, signal_cooldown,
            direction, trend_lookback, 0, 0.0, 0.0, 0.0, 0.0
        )

    pnls = [t.pnl_pips for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_trades = len(trades)
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    pnl_total = sum(pnls)
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0

    # Drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    max_dd = np.max(peak - equity) if len(equity) > 0 else 0

    return OptimizationResult(
        states_allowed, lyapunov_threshold, signal_cooldown,
        direction, trend_lookback, total_trades, win_rate,
        pnl_total, profit_factor, max_dd
    )


def main():
    print("=" * 70)
    print("  OTIMIZADOR PRM - 3000 COMBINACOES")
    print("=" * 70)

    breakeven = STOP_LOSS_PIPS / (STOP_LOSS_PIPS + TAKE_PROFIT_PIPS) * 100

    print(f"\n  PARAMETROS FIXOS:")
    print(f"    hmm_threshold: {HMM_THRESHOLD}")
    print(f"    stop_loss_pips: {STOP_LOSS_PIPS}")
    print(f"    take_profit_pips: {TAKE_PROFIT_PIPS}")
    print(f"    warmup_bars: {MIN_PRICES_WARMUP}")
    print(f"    BREAKEVEN WR: {breakeven:.1f}%")

    # Gerar todas as combinacoes
    all_combinations = list(itertools.product(
        STATES_OPTIONS,
        LYAPUNOV_OPTIONS,
        COOLDOWN_OPTIONS,
        DIRECTION_OPTIONS,
        LOOKBACK_OPTIONS
    ))

    total_possible = len(all_combinations)
    print(f"\n  Combinacoes possiveis: {total_possible}")

    # Amostrar 3000 combinacoes
    if total_possible > TARGET_COMBINATIONS:
        random.seed(42)
        combinations = random.sample(all_combinations, TARGET_COMBINATIONS)
    else:
        combinations = all_combinations

    print(f"  Combinacoes a testar: {len(combinations)}")

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

    # Treinar HMM uma vez
    print("\n  Treinando HMM...")
    warmup_prices = np.array([bar.close for bar in bars[:MIN_PRICES_WARMUP]])

    prm_template = IncrementalPRMIndicator(n_states=3, hmm_window=200, lyap_window=50)
    if not prm_template.train_hmm(warmup_prices):
        print("  ERRO: Falha ao treinar HMM")
        return

    # Preparar closes buffer inicial
    closes_buffer_init = deque(maxlen=500)
    for bar in bars[:MIN_PRICES_WARMUP]:
        closes_buffer_init.append(bar.close)

    # Otimizacao
    print("\n" + "=" * 70)
    print("  EXECUTANDO OTIMIZACAO")
    print("=" * 70)

    results = []
    start_time_opt = datetime.now()

    # Top 20 resultados
    top_results = []

    for idx, combo in enumerate(combinations):
        states, lyap, cooldown, direction, lookback = combo

        # Criar copia do PRM para este teste
        prm = IncrementalPRMIndicator(n_states=3, hmm_window=200, lyap_window=50)
        prm.hmm_model = prm_template.hmm_model
        prm.is_trained = True
        prm.initialize_buffers(warmup_prices)

        # Rodar backtest
        result = run_backtest(
            bars, MIN_PRICES_WARMUP, prm, closes_buffer_init,
            states, lyap, cooldown, direction, lookback
        )

        results.append(result)

        # Atualizar top 20
        if result.total_trades >= 50:  # Minimo de trades
            top_results.append(result)
            top_results.sort(key=lambda x: x.pnl_pips, reverse=True)
            top_results = top_results[:20]

        # Progresso
        if (idx + 1) % 100 == 0:
            elapsed = (datetime.now() - start_time_opt).total_seconds()
            speed = (idx + 1) / elapsed
            eta = (len(combinations) - idx - 1) / speed

            best_pnl = top_results[0].pnl_pips if top_results else 0

            print(f"    [{idx+1}/{len(combinations)}] "
                  f"{speed:.0f} comb/s | "
                  f"ETA: {eta:.0f}s | "
                  f"Melhor PnL: {best_pnl:+.0f} pips")

    elapsed_total = (datetime.now() - start_time_opt).total_seconds()
    print(f"\n  Otimizacao concluida em {elapsed_total:.1f}s")

    # Resultados
    print("\n" + "=" * 70)
    print("  TOP 20 MELHORES COMBINACOES")
    print("=" * 70)

    print(f"\n  {'#':<3} {'States':<12} {'Lyap':<6} {'Cool':<5} {'Dir':<7} "
          f"{'Look':<5} {'Trades':<7} {'WR%':<6} {'PnL':<10} {'PF':<5}")
    print("  " + "-" * 80)

    for i, r in enumerate(top_results):
        states_str = str(r.states_allowed)
        print(f"  {i+1:<3} {states_str:<12} {r.lyapunov_threshold:<6.1f} "
              f"{r.signal_cooldown:<5} {r.direction:<7} {r.trend_lookback:<5} "
              f"{r.total_trades:<7} {r.win_rate:<6.1f} {r.pnl_pips:+<10.0f} "
              f"{r.profit_factor:<5.2f}")

    # Estatisticas gerais
    print("\n" + "=" * 70)
    print("  ESTATISTICAS GERAIS")
    print("=" * 70)

    profitable = [r for r in results if r.pnl_pips > 0 and r.total_trades >= 50]
    losing = [r for r in results if r.pnl_pips <= 0 and r.total_trades >= 50]
    no_trades = [r for r in results if r.total_trades < 50]

    print(f"\n  Combinacoes lucrativas: {len(profitable)}")
    print(f"  Combinacoes perdedoras: {len(losing)}")
    print(f"  Combinacoes sem trades suficientes: {len(no_trades)}")

    if profitable:
        best = max(profitable, key=lambda x: x.pnl_pips)
        print(f"\n  MELHOR COMBINACAO:")
        print(f"    states_allowed: {best.states_allowed}")
        print(f"    lyapunov_threshold: {best.lyapunov_threshold}")
        print(f"    signal_cooldown: {best.signal_cooldown}")
        print(f"    direction: {best.direction}")
        print(f"    trend_lookback: {best.trend_lookback}")
        print(f"    total_trades: {best.total_trades}")
        print(f"    win_rate: {best.win_rate:.1f}%")
        print(f"    pnl_pips: {best.pnl_pips:+.0f}")
        print(f"    profit_factor: {best.profit_factor:.2f}")
        print(f"    max_drawdown: {best.max_drawdown:.0f} pips")

    # Analise por estado
    print("\n" + "=" * 70)
    print("  ANALISE POR ESTADO")
    print("=" * 70)

    for states in STATES_OPTIONS:
        state_results = [r for r in results if r.states_allowed == states and r.total_trades >= 50]
        if state_results:
            avg_pnl = np.mean([r.pnl_pips for r in state_results])
            avg_wr = np.mean([r.win_rate for r in state_results])
            best_pnl = max(r.pnl_pips for r in state_results)
            print(f"  {str(states):<12}: {len(state_results)} combos, "
                  f"Avg PnL={avg_pnl:+.0f}, Avg WR={avg_wr:.1f}%, Best={best_pnl:+.0f}")
        else:
            print(f"  {str(states):<12}: Sem trades suficientes")

    # Analise por direcao
    print("\n" + "=" * 70)
    print("  ANALISE POR DIRECAO")
    print("=" * 70)

    for direction in DIRECTION_OPTIONS:
        dir_results = [r for r in results if r.direction == direction and r.total_trades >= 50]
        if dir_results:
            avg_pnl = np.mean([r.pnl_pips for r in dir_results])
            avg_wr = np.mean([r.win_rate for r in dir_results])
            profitable_count = len([r for r in dir_results if r.pnl_pips > 0])
            print(f"  {direction:<7}: {len(dir_results)} combos, "
                  f"Avg PnL={avg_pnl:+.0f}, Avg WR={avg_wr:.1f}%, "
                  f"Lucrativos={profitable_count}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
