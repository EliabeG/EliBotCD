#!/usr/bin/env python3
"""
================================================================================
BACKTEST PRM COM PARAMETROS VIAVEIS
================================================================================

Baseado no diagnostico:
- Prob_HMM max = 0.4451 (threshold 0.9 impossivel!)
- 100% estado 0 (estados 1,2 nunca ocorrem)

Solucao: Remover filtro de estado e usar threshold viavel.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from api.fxopen_historical_ws import Bar, get_historical_data_with_spread_sync

try:
    from hmmlearn import hmm
except ImportError:
    print("ERRO: hmmlearn nao instalado")
    sys.exit(1)

# Custos
SPREAD_PIPS = 0.2
SLIPPAGE_PIPS = 0.0

# ===============================================================================
# PARAMETROS AJUSTADOS (baseado no diagnostico)
# ===============================================================================
# Original: hmm_threshold = 0.9 (IMPOSSIVEL - max eh 0.44!)
# Ajustado: usar threshold viavel
HMM_THRESHOLD = 0.44  # Proximo do P90 (10% das barras passam)
LYAPUNOV_THRESHOLD = 0.9  # Proximo do P75 (25% das barras passam)
STOP_LOSS_PIPS = 10.0
TAKE_PROFIT_PIPS = 10.0
SIGNAL_COOLDOWN = 10
TREND_LOOKBACK = 10
MIN_PRICES_WARMUP = 6624
# Estado removido - sempre 0
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


class ViablePRMIndicator:
    """PRM com parametros viaveis"""

    def __init__(self, n_states=3, training_window=200):
        self.n_states = n_states
        self.training_window = training_window
        self.hmm_model = None
        self.is_trained = False

    def train_hmm(self, prices: np.ndarray):
        """Treina HMM"""
        if len(prices) < self.training_window:
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
            return True
        except:
            return False

    def compute_lyapunov(self, prices: np.ndarray, window: int = 50) -> float:
        """Lyapunov simplificado"""
        if len(prices) < window + 1:
            return 0.0

        returns = np.diff(np.log(prices[-window-1:]))
        abs_returns = np.abs(returns)
        abs_returns = abs_returns[abs_returns > 1e-10]

        if len(abs_returns) == 0:
            return 0.0

        lyapunov = np.mean(np.log(abs_returns)) + 10
        return max(0, lyapunov)

    def analyze(self, prices: np.ndarray) -> dict:
        """Analise sem filtro de estado"""
        result = {'hmm_prob': 0.0, 'lyapunov': 0.0, 'valid': False}

        if not self.is_trained or len(prices) < 10:
            return result

        try:
            returns = np.diff(np.log(prices[-self.training_window:])).reshape(-1, 1)
            probs = self.hmm_model.predict_proba(returns)
            current_probs = probs[-1]

            # Prob max de estados 1 e 2
            hmm_prob = max(current_probs[1] if len(current_probs) > 1 else 0,
                          current_probs[2] if len(current_probs) > 2 else 0)

            lyapunov = self.compute_lyapunov(prices)

            result = {
                'hmm_prob': hmm_prob,
                'lyapunov': lyapunov,
                'valid': True
            }
        except:
            pass

        return result


def calculate_direction(closes: List[float], lookback: int) -> int:
    """Calcula direcao"""
    min_bars = lookback + 2
    if len(closes) < min_bars:
        return 0

    recent = closes[-2]
    past = closes[-(lookback + 2)]
    trend = recent - past

    return 1 if trend > 0 else -1


def run_backtest(bars: List[Bar], warmup_bars: int) -> List[Trade]:
    """Backtest com parametros viaveis"""
    trades = []
    position: Optional[Position] = None
    pip = 0.0001
    signal_cooldown = 0

    prm = ViablePRMIndicator(n_states=3, training_window=200)

    total_bars = len(bars)
    trading_start = warmup_bars

    print(f"\n  Total de barras: {total_bars}")
    print(f"  Barras de warmup: {warmup_bars}")
    print(f"  Barras de trading: {total_bars - warmup_bars}")

    # Warmup
    print(f"\n  [WARMUP] Treinando HMM...")
    warmup_prices = np.array([bar.close for bar in bars[:warmup_bars]])
    warmup_closes = [bar.close for bar in bars[:warmup_bars]]

    if not prm.train_hmm(warmup_prices):
        print("  ERRO: Falha ao treinar HMM")
        return []
    print(f"  [WARMUP] HMM treinado!")

    prices_buffer = list(warmup_prices)
    closes_buffer = list(warmup_closes)

    # Trading
    print(f"\n  [TRADING] Iniciando...")
    pending_signal = None
    signals_generated = 0

    for i in range(warmup_bars, total_bars):
        bar = bars[i]
        prices_buffer.append(bar.close)
        closes_buffer.append(bar.close)

        # Executar sinal pendente
        if pending_signal and position is None:
            if bar.has_spread_data:
                spread = bar.spread_pips * pip
            else:
                spread = SPREAD_PIPS * pip

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

        # Verificar stop/take
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

        # Gerar sinal
        if position is None and pending_signal is None:
            if signal_cooldown > 0:
                signal_cooldown -= 1
            else:
                prices_array = np.array(prices_buffer[-500:])
                result = prm.analyze(prices_array)

                if result['valid']:
                    # SEM filtro de estado (sempre 0)
                    if (result['hmm_prob'] >= HMM_THRESHOLD and
                        result['lyapunov'] >= LYAPUNOV_THRESHOLD):

                        direction = calculate_direction(closes_buffer, TREND_LOOKBACK)
                        if direction != 0:
                            pending_signal = 'BUY' if direction == 1 else 'SELL'
                            signal_cooldown = SIGNAL_COOLDOWN
                            signals_generated += 1

        # Progresso
        trading_bar = i - warmup_bars + 1
        total_trading = total_bars - warmup_bars
        if trading_bar % 1000 == 0:
            print(f"    Trading: {trading_bar}/{total_trading}, {len(trades)} trades, {signals_generated} sinais")

    # Fechar posicao
    if position:
        last_bar = bars[-1]
        if position.type == PositionType.LONG:
            exit_price = last_bar.bid_close if last_bar.bid_close else last_bar.close
            pnl_pips = (exit_price - position.entry_price) / pip
        else:
            exit_price = last_bar.ask_close if last_bar.ask_close else last_bar.close
            pnl_pips = (position.entry_price - exit_price) / pip

        trades.append(Trade(
            position.type, position.entry_price, position.entry_time,
            exit_price, last_bar.timestamp, pnl_pips, "end_of_data"
        ))

    print(f"\n  Sinais gerados: {signals_generated}")

    return trades


def main():
    print("=" * 70)
    print("  BACKTEST PRM - PARAMETROS VIAVEIS")
    print("=" * 70)

    print(f"\n  NOTA: hmm_threshold=0.9 eh impossivel (max=0.44)")
    print(f"  Usando parametros ajustados baseados no diagnostico.")

    print(f"\n  PARAMETROS:")
    print(f"    hmm_threshold: {HMM_THRESHOLD} (ajustado de 0.9)")
    print(f"    lyapunov_threshold: {LYAPUNOV_THRESHOLD}")
    print(f"    stop_loss_pips: {STOP_LOSS_PIPS}")
    print(f"    take_profit_pips: {TAKE_PROFIT_PIPS}")
    print(f"    signal_cooldown: {SIGNAL_COOLDOWN}")
    print(f"    trend_lookback: {TREND_LOOKBACK}")
    print(f"    warmup_bars: {MIN_PRICES_WARMUP}")

    # Baixar dados
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=90)

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

    # Backtest
    print("\n" + "=" * 70)
    print("  EXECUTANDO BACKTEST")
    print("=" * 70)

    trades = run_backtest(bars, MIN_PRICES_WARMUP)

    # Resultados
    print("\n" + "=" * 70)
    print("  RESULTADOS")
    print("=" * 70)

    if not trades:
        print("\n  NENHUM TRADE!")
        return

    pnls = [t.pnl_pips for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)

    win_rate = len(wins) / len(trades)
    breakeven = STOP_LOSS_PIPS / (TAKE_PROFIT_PIPS + STOP_LOSS_PIPS)
    profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')

    print(f"\n  TRADES:")
    print(f"    Total: {len(trades)}")
    print(f"    Vencedores: {len(wins)}")
    print(f"    Perdedores: {len(losses)}")
    print(f"    Win Rate: {win_rate:.1%}")
    print(f"    Breakeven: {breakeven:.1%}")

    print(f"\n  PERFORMANCE:")
    print(f"    PnL Total: ${total_pnl * 0.10:.2f} ({total_pnl:.1f} pips)")
    print(f"    Profit Factor: {profit_factor:.2f}")
    print(f"    Avg Trade: {np.mean(pnls):.1f} pips")
    print(f"    Max Win: {max(pnls):.1f} pips")
    print(f"    Max Loss: {min(pnls):.1f} pips")

    # Drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    max_dd = np.max(peak - equity)

    print(f"\n  RISCO:")
    print(f"    Max Drawdown: {max_dd:.1f} pips")

    stops = len([t for t in trades if t.exit_reason == "stop_loss"])
    takes = len([t for t in trades if t.exit_reason == "take_profit"])
    print(f"\n  SAIDAS:")
    print(f"    Stop Loss: {stops} ({stops/len(trades)*100:.0f}%)")
    print(f"    Take Profit: {takes} ({takes/len(trades)*100:.0f}%)")

    print(f"\n  ULTIMOS 10 TRADES:")
    for t in trades[-10:]:
        sign = "+" if t.pnl_pips > 0 else ""
        print(f"    {t.entry_time.strftime('%Y-%m-%d %H:%M')} | "
              f"{t.type.name:5} | {sign}{t.pnl_pips:.1f} pips | {t.exit_reason}")

    print("\n" + "=" * 70)
    print("  CONCLUSAO")
    print("=" * 70)

    if total_pnl > 0:
        print(f"\n  ESTRATEGIA LUCRATIVA!")
    else:
        print(f"\n  ESTRATEGIA NAO LUCRATIVA")
        if win_rate < breakeven:
            print(f"    Win Rate ({win_rate:.1%}) < Breakeven ({breakeven:.1%})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
