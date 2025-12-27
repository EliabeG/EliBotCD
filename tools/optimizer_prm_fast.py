#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR PRM ULTRA-RAPIDO - 3000 COMBINACOES
================================================================================

Otimizacao: Pre-calcula TODOS os valores do indicador uma vez,
depois apenas aplica diferentes filtros.

Isso reduz o tempo de horas para segundos.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from typing import List, Tuple
from dataclasses import dataclass
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


def precompute_indicators(bars: List[Bar], warmup_bars: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Pre-calcula TODOS os valores do indicador para todas as barras.
    Retorna arrays de: states, hmm_probs, lyapunovs, e directions por lookback
    """
    print("  Pre-calculando indicadores...")

    total_bars = len(bars)
    trading_bars = total_bars - warmup_bars

    # Arrays para armazenar resultados
    states = np.zeros(trading_bars, dtype=np.int32)
    hmm_probs = np.zeros(trading_bars)
    lyapunovs = np.zeros(trading_bars)

    # Directions para diferentes lookbacks
    directions = {}
    for lookback in LOOKBACK_OPTIONS:
        directions[lookback] = np.zeros(trading_bars, dtype=np.int32)

    # Treinar HMM
    prices = np.array([bar.close for bar in bars])
    warmup_prices = prices[:warmup_bars]

    returns = np.diff(np.log(warmup_prices)).reshape(-1, 1)

    hmm_model = hmm.GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    hmm_model.fit(returns)

    # Buffer de retornos
    hmm_window = 200
    lyap_window = 50
    returns_buffer = deque(maxlen=hmm_window)

    # Inicializar com warmup
    warmup_returns = np.diff(np.log(warmup_prices))
    for r in warmup_returns[-hmm_window:]:
        returns_buffer.append(r)

    # Closes para directions
    closes = [bar.close for bar in bars]

    # Calcular para cada barra
    for i in range(warmup_bars, total_bars):
        idx = i - warmup_bars

        # Adicionar novo retorno
        if i > 0:
            new_return = np.log(prices[i] / prices[i-1])
            returns_buffer.append(new_return)

        # HMM state e prob
        if len(returns_buffer) >= 10:
            try:
                rets = np.array(list(returns_buffer)).reshape(-1, 1)
                probs = hmm_model.predict_proba(rets)
                current_probs = probs[-1]
                states[idx] = np.argmax(current_probs)
                hmm_probs[idx] = max(
                    current_probs[1] if len(current_probs) > 1 else 0,
                    current_probs[2] if len(current_probs) > 2 else 0
                )
            except:
                pass

        # Lyapunov
        if i >= lyap_window:
            window_prices = prices[i-lyap_window:i+1]
            window_returns = np.diff(np.log(window_prices))
            abs_returns = np.abs(window_returns)
            abs_returns = abs_returns[abs_returns > 1e-10]
            if len(abs_returns) > 0:
                lyapunovs[idx] = max(0, np.mean(np.log(abs_returns)) + 10)

        # Directions para cada lookback
        for lookback in LOOKBACK_OPTIONS:
            if i >= lookback + 2:
                recent = closes[i-1]  # -2 do original, mas i ja eh o indice atual
                past = closes[i - lookback - 1]
                trend = recent - past
                directions[lookback][idx] = 1 if trend > 0 else -1

        # Progresso
        if (idx + 1) % 5000 == 0:
            print(f"    Processado: {idx+1}/{trading_bars}")

    print(f"  Pre-calculo concluido!")

    return states, hmm_probs, lyapunovs, directions


def run_fast_backtest(
    bars: List[Bar],
    warmup_bars: int,
    states: np.ndarray,
    hmm_probs: np.ndarray,
    lyapunovs: np.ndarray,
    directions: dict,
    states_allowed: List[int],
    lyapunov_threshold: float,
    signal_cooldown: int,
    direction_mode: str,
    trend_lookback: int
) -> OptimizationResult:
    """Backtest ultra-rapido usando valores pre-calculados"""

    pip = 0.0001
    total_bars = len(bars)
    trading_bars = total_bars - warmup_bars

    dir_array = directions[trend_lookback]

    # Listas para trades
    trade_pnls = []

    # Estado
    in_position = False
    position_type = 0  # 1=LONG, -1=SHORT
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    cooldown_counter = 0
    pending_signal = 0  # 1=BUY, -1=SELL, 0=none

    for idx in range(trading_bars):
        bar_idx = idx + warmup_bars
        bar = bars[bar_idx]

        # 1. Executar sinal pendente
        if pending_signal != 0 and not in_position:
            spread = bar.spread_pips * pip if bar.has_spread_data else 0.2 * pip

            if pending_signal == 1:  # BUY
                entry_price = bar.ask_open if bar.ask_open else bar.open + spread/2
                position_type = 1
                stop_loss = entry_price - (STOP_LOSS_PIPS * pip)
                take_profit = entry_price + (TAKE_PROFIT_PIPS * pip)
            else:  # SELL
                entry_price = bar.bid_open if bar.bid_open else bar.open - spread/2
                position_type = -1
                stop_loss = entry_price + (STOP_LOSS_PIPS * pip)
                take_profit = entry_price - (TAKE_PROFIT_PIPS * pip)

            in_position = True
            pending_signal = 0

        # 2. Verificar stop/take
        if in_position:
            exit_price = None

            bid_low = bar.bid_low if bar.bid_low else bar.low
            ask_high = bar.ask_high if bar.ask_high else bar.high

            if position_type == 1:  # LONG
                if bid_low <= stop_loss:
                    exit_price = stop_loss
                elif ask_high >= take_profit:
                    exit_price = take_profit
            else:  # SHORT
                if ask_high >= stop_loss:
                    exit_price = stop_loss
                elif bid_low <= take_profit:
                    exit_price = take_profit

            if exit_price is not None:
                if position_type == 1:
                    pnl_pips = (exit_price - entry_price) / pip
                else:
                    pnl_pips = (entry_price - exit_price) / pip

                trade_pnls.append(pnl_pips)
                in_position = False

        # 3. Gerar sinal
        if not in_position and pending_signal == 0:
            if cooldown_counter > 0:
                cooldown_counter -= 1
            else:
                # Verificar filtros usando valores pre-calculados
                state = states[idx]
                hmm_prob = hmm_probs[idx]
                lyap = lyapunovs[idx]
                dir_val = dir_array[idx]

                if state in states_allowed:
                    if hmm_prob >= HMM_THRESHOLD and lyap >= lyapunov_threshold:
                        if dir_val != 0:
                            if direction_mode == 'trend':
                                pending_signal = dir_val
                            else:  # contra
                                pending_signal = -dir_val
                            cooldown_counter = signal_cooldown

    # Calcular metricas
    if not trade_pnls:
        return OptimizationResult(
            states_allowed, lyapunov_threshold, signal_cooldown,
            direction_mode, trend_lookback, 0, 0.0, 0.0, 0.0, 0.0
        )

    pnls = np.array(trade_pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    total_trades = len(pnls)
    win_rate = len(wins) / total_trades * 100
    pnl_total = float(np.sum(pnls))
    profit_factor = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else 0

    # Drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max(peak - equity))

    return OptimizationResult(
        states_allowed, lyapunov_threshold, signal_cooldown,
        direction_mode, trend_lookback, total_trades, win_rate,
        pnl_total, profit_factor, max_dd
    )


def main():
    print("=" * 70)
    print("  OTIMIZADOR PRM ULTRA-RAPIDO - 3000 COMBINACOES")
    print("=" * 70)

    breakeven = STOP_LOSS_PIPS / (STOP_LOSS_PIPS + TAKE_PROFIT_PIPS) * 100

    print(f"\n  PARAMETROS FIXOS:")
    print(f"    hmm_threshold: {HMM_THRESHOLD}")
    print(f"    stop_loss_pips: {STOP_LOSS_PIPS}")
    print(f"    take_profit_pips: {TAKE_PROFIT_PIPS}")
    print(f"    warmup_bars: {MIN_PRICES_WARMUP}")
    print(f"    BREAKEVEN WR: {breakeven:.1f}%")

    # Gerar combinacoes
    all_combinations = list(itertools.product(
        STATES_OPTIONS,
        LYAPUNOV_OPTIONS,
        COOLDOWN_OPTIONS,
        DIRECTION_OPTIONS,
        LOOKBACK_OPTIONS
    ))

    total_possible = len(all_combinations)
    print(f"\n  Combinacoes possiveis: {total_possible}")

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

    # Pre-calcular indicadores (UMA VEZ)
    print("\n" + "=" * 70)
    print("  PRE-CALCULANDO INDICADORES")
    print("=" * 70)

    states, hmm_probs, lyapunovs, directions = precompute_indicators(bars, MIN_PRICES_WARMUP)

    # Estatisticas dos indicadores
    print(f"\n  Estatisticas dos indicadores:")
    print(f"    HMM States: 0={np.sum(states==0)}, 1={np.sum(states==1)}, 2={np.sum(states==2)}")
    print(f"    HMM Prob: min={np.min(hmm_probs):.3f}, max={np.max(hmm_probs):.3f}, mean={np.mean(hmm_probs):.3f}")
    print(f"    Lyapunov: min={np.min(lyapunovs):.3f}, max={np.max(lyapunovs):.3f}, mean={np.mean(lyapunovs):.3f}")

    # Otimizacao
    print("\n" + "=" * 70)
    print("  EXECUTANDO OTIMIZACAO (Ultra-rapido)")
    print("=" * 70)

    results = []
    start_time_opt = datetime.now()

    for idx, combo in enumerate(combinations):
        states_allowed, lyap, cooldown, direction, lookback = combo

        result = run_fast_backtest(
            bars, MIN_PRICES_WARMUP,
            states, hmm_probs, lyapunovs, directions,
            list(states_allowed), lyap, cooldown, direction, lookback
        )

        results.append(result)

        # Progresso
        if (idx + 1) % 500 == 0:
            elapsed = (datetime.now() - start_time_opt).total_seconds()
            speed = (idx + 1) / elapsed
            eta = (len(combinations) - idx - 1) / speed if speed > 0 else 0
            print(f"    [{idx+1}/{len(combinations)}] {speed:.0f} comb/s | ETA: {eta:.0f}s")

    elapsed_total = (datetime.now() - start_time_opt).total_seconds()
    print(f"\n  Otimizacao concluida em {elapsed_total:.1f}s ({len(combinations)/elapsed_total:.0f} comb/s)")

    # Filtrar resultados com trades suficientes
    valid_results = [r for r in results if r.total_trades >= 50]

    # Ordenar por PnL
    valid_results.sort(key=lambda x: x.pnl_pips, reverse=True)

    # Resultados
    print("\n" + "=" * 70)
    print("  TOP 20 MELHORES COMBINACOES")
    print("=" * 70)

    print(f"\n  {'#':<3} {'States':<12} {'Lyap':<6} {'Cool':<5} {'Dir':<7} "
          f"{'Look':<5} {'Trades':<7} {'WR%':<6} {'PnL':<10} {'PF':<5}")
    print("  " + "-" * 80)

    for i, r in enumerate(valid_results[:20]):
        states_str = str(r.states_allowed)
        print(f"  {i+1:<3} {states_str:<12} {r.lyapunov_threshold:<6.1f} "
              f"{r.signal_cooldown:<5} {r.direction:<7} {r.trend_lookback:<5} "
              f"{r.total_trades:<7} {r.win_rate:<6.1f} {r.pnl_pips:+<10.0f} "
              f"{r.profit_factor:<5.2f}")

    # Estatisticas gerais
    print("\n" + "=" * 70)
    print("  ESTATISTICAS GERAIS")
    print("=" * 70)

    profitable = [r for r in valid_results if r.pnl_pips > 0]
    losing = [r for r in valid_results if r.pnl_pips <= 0]

    print(f"\n  Combinacoes com 50+ trades: {len(valid_results)}")
    print(f"  Combinacoes lucrativas: {len(profitable)} ({len(profitable)/len(valid_results)*100:.1f}%)")
    print(f"  Combinacoes perdedoras: {len(losing)} ({len(losing)/len(valid_results)*100:.1f}%)")

    if profitable:
        best = profitable[0]
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

    for states_opt in STATES_OPTIONS:
        state_results = [r for r in valid_results if r.states_allowed == list(states_opt)]
        if state_results:
            avg_pnl = np.mean([r.pnl_pips for r in state_results])
            avg_wr = np.mean([r.win_rate for r in state_results])
            best_pnl = max(r.pnl_pips for r in state_results)
            profitable_count = len([r for r in state_results if r.pnl_pips > 0])
            print(f"  {str(list(states_opt)):<12}: {len(state_results):>3} combos, "
                  f"Avg PnL={avg_pnl:+6.0f}, Avg WR={avg_wr:5.1f}%, "
                  f"Best={best_pnl:+6.0f}, Lucrativos={profitable_count}")

    # Analise por direcao
    print("\n" + "=" * 70)
    print("  ANALISE POR DIRECAO")
    print("=" * 70)

    for direction in DIRECTION_OPTIONS:
        dir_results = [r for r in valid_results if r.direction == direction]
        if dir_results:
            avg_pnl = np.mean([r.pnl_pips for r in dir_results])
            avg_wr = np.mean([r.win_rate for r in dir_results])
            profitable_count = len([r for r in dir_results if r.pnl_pips > 0])
            print(f"  {direction:<7}: {len(dir_results):>4} combos, "
                  f"Avg PnL={avg_pnl:+6.0f}, Avg WR={avg_wr:5.1f}%, "
                  f"Lucrativos={profitable_count}")

    # Top 5 para cada direcao
    print("\n" + "=" * 70)
    print("  TOP 5 POR DIRECAO")
    print("=" * 70)

    for direction in DIRECTION_OPTIONS:
        dir_results = [r for r in valid_results if r.direction == direction]
        print(f"\n  {direction.upper()}:")
        for i, r in enumerate(dir_results[:5]):
            print(f"    {i+1}. States={r.states_allowed}, Lyap={r.lyapunov_threshold:.1f}, "
                  f"Cool={r.signal_cooldown}, Look={r.trend_lookback} -> "
                  f"{r.total_trades} trades, WR={r.win_rate:.1f}%, PnL={r.pnl_pips:+.0f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
