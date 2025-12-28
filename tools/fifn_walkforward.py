#!/usr/bin/env python3
"""
FIFN Walk-Forward Validation
===============================================================================
Divide os dados em 3 periodos e valida consistencia do FIFN com filtros.
"""

import sys
import os
import urllib.request
import ssl
import json
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.alta_volatilidade.fifn_fisher_navier import FluxoInformacaoFisherNavier

# =============================================================================
# PARAMETROS
# =============================================================================

SL_PIPS = 22.5
TP_PIPS = 27.9
COOLDOWN = 3
SPREAD_PIPS = 1.2
SLIPPAGE_PIPS = 0.5
PIP_VALUE = 0.0001

REYNOLDS_LOW = 2521.0
REYNOLDS_HIGH = 3786.0
SKEWNESS_THRESHOLD = 0.3091

API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"

@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

def download_bars(period: str, count: int) -> List[Bar]:
    bars = []
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    current_ts = int(time.time() * 1000)
    total = 0

    print(f"  Baixando {count} barras {period}...")
    while total < count:
        remaining = min(1000, count - total)
        url = f"{API_BASE}/{SYMBOL}/{period}/bars/ask?timestamp={current_ts}&count=-{remaining}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30, context=ctx) as response:
                data = json.loads(response.read().decode())
                batch = data.get("Bars", [])
                if not batch:
                    break
                for b in batch:
                    ts = datetime.fromtimestamp(b["Timestamp"] / 1000, tz=timezone.utc)
                    bars.append(Bar(ts, b["Open"], b["High"], b["Low"], b["Close"], b.get("Volume", 0)))
                total += len(batch)
                oldest = min(batch, key=lambda x: x["Timestamp"])
                current_ts = oldest["Timestamp"] - 1
                if total % 2000 == 0:
                    print(f"    {total} barras...")
                time.sleep(0.1)
        except Exception as e:
            print(f"  Erro: {e}")
            break

    bars.sort(key=lambda x: x.timestamp)
    return bars

def run_fold(bars: List[Bar], start_pct: float, end_pct: float) -> dict:
    """Executa um fold do walk-forward."""
    n = len(bars)
    start_idx = int(n * start_pct / 100)
    end_idx = int(n * end_pct / 100)

    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])

    # FIFN
    fifn = FluxoInformacaoFisherNavier(
        window_size=50,
        kl_lookback=10,
        reynolds_sweet_low=REYNOLDS_LOW,
        reynolds_sweet_high=REYNOLDS_HIGH,
        skewness_threshold=SKEWNESS_THRESHOLD
    )

    try:
        result = fifn.analyze(closes[:end_idx])
        reynolds_series = result['reynolds_series']
        pressure_gradient = result['pressure_gradient_series']

        from scipy import stats
        returns = np.diff(np.log(closes[:end_idx]))
        skewness_arr = np.zeros(len(returns))
        for i in range(fifn.window_size, len(returns)):
            skewness_arr[i] = stats.skew(returns[i - fifn.window_size:i])

    except Exception as e:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0}

    # EMA e RSI
    ema_12 = np.zeros(end_idx)
    ema_26 = np.zeros(end_idx)
    rsi_14 = np.full(end_idx, 50.0)

    alpha_12 = 2.0 / 13
    alpha_26 = 2.0 / 27
    ema_12[12] = np.mean(closes[:12])
    ema_26[26] = np.mean(closes[:26])
    for i in range(13, end_idx):
        ema_12[i] = alpha_12 * closes[i-1] + (1 - alpha_12) * ema_12[i-1]
    for i in range(27, end_idx):
        ema_26[i] = alpha_26 * closes[i-1] + (1 - alpha_26) * ema_26[i-1]

    deltas = np.diff(closes[:end_idx])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    for i in range(15, end_idx):
        avg_gain = np.mean(gains[max(0, i-14):i])
        avg_loss = np.mean(losses[max(0, i-14):i])
        if avg_loss == 0:
            rsi_14[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_14[i] = 100.0 - (100.0 / (1.0 + rs))

    # Gerar sinais
    signals = []
    for i in range(max(100, start_idx), end_idx - 1):
        re = reynolds_series[i] if i < len(reynolds_series) else 0
        skew = skewness_arr[i] if i < len(skewness_arr) else 0
        pg = pressure_gradient[i] if i < len(pressure_gradient) else 0

        in_sweet_spot = REYNOLDS_LOW <= re <= REYNOLDS_HIGH
        if not in_sweet_spot:
            continue

        trend_up = ema_12[i] > ema_26[i]
        trend_down = ema_12[i] < ema_26[i]
        rsi = rsi_14[i]
        hour = bars[i].timestamp.hour
        session_ok = 8 <= hour <= 20

        if pg < 0 and skew > SKEWNESS_THRESHOLD:
            if session_ok and (trend_up or abs(ema_12[i] - ema_26[i]) < 0.001) and 35 <= rsi <= 75:
                signals.append((i, 'LONG'))

        elif pg > 0 and skew < -SKEWNESS_THRESHOLD:
            if session_ok and (trend_down or abs(ema_12[i] - ema_26[i]) < 0.001) and 25 <= rsi <= 65:
                signals.append((i, 'SHORT'))

    # Simular trades
    trades = []
    last_trade_idx = -999

    for idx, direction in signals:
        if idx - last_trade_idx < COOLDOWN:
            continue
        if idx + 1 >= n:
            continue

        entry = bars[idx + 1].open
        sl = entry - SL_PIPS * PIP_VALUE if direction == 'LONG' else entry + SL_PIPS * PIP_VALUE
        tp = entry + TP_PIPS * PIP_VALUE if direction == 'LONG' else entry - TP_PIPS * PIP_VALUE

        for j in range(idx + 2, min(idx + 200, n)):
            bar = bars[j]
            if direction == 'LONG':
                if bar.low <= sl:
                    trades.append(-SL_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS)
                    break
                if bar.high >= tp:
                    trades.append(TP_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS)
                    break
            else:
                if bar.high >= sl:
                    trades.append(-SL_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS)
                    break
                if bar.low <= tp:
                    trades.append(TP_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS)
                    break
        last_trade_idx = idx

    if not trades:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0}

    wins = len([t for t in trades if t > 0])
    total = len(trades)
    wr = wins / total * 100
    be = SL_PIPS / (SL_PIPS + TP_PIPS) * 100
    edge = wr - be
    pnl = sum(trades)
    gp = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))
    pf = gp / gl if gl > 0 else 0

    return {'total': total, 'wins': wins, 'wr': wr, 'be': be, 'edge': edge, 'pnl': pnl, 'pf': pf}

def main():
    print("=" * 70)
    print("  FIFN WALK-FORWARD VALIDATION")
    print("=" * 70)

    bars = download_bars("H1", 5000)
    if len(bars) < 1000:
        print(f"  ERRO: Apenas {len(bars)} barras")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"\n  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()}")
    print(f"  Duracao: {days} dias (~{days/30:.1f} meses)")
    print(f"  Barras: {len(bars)}")

    # Walk-Forward: 3 folds
    print("\n" + "=" * 70)
    print("  WALK-FORWARD (3 FOLDS)")
    print("=" * 70)

    folds = []
    for i in range(3):
        start = i * 33
        end = (i + 1) * 33 if i < 2 else 100
        print(f"\n  Fold {i+1}/3: {start}%-{end}%...")
        result = run_fold(bars, start, end)
        folds.append(result)

        if result['total'] > 0:
            status = "[OK]" if result['edge'] > 0 else "[FAIL]"
            print(f"    Trades: {result['total']} | WR: {result['wr']:.1f}% | Edge: {result['edge']:+.1f}% | PF: {result['pf']:.2f} {status}")
        else:
            print(f"    Nenhum trade")

    # Agregado
    total_trades = sum(f['total'] for f in folds)
    total_wins = sum(f.get('wins', 0) for f in folds)
    total_pnl = sum(f['pnl'] for f in folds)

    print("\n" + "-" * 70)
    if total_trades > 0:
        agg_wr = total_wins / total_trades * 100
        agg_be = SL_PIPS / (SL_PIPS + TP_PIPS) * 100
        agg_edge = agg_wr - agg_be
        print(f"  AGREGADO: {total_trades} trades | WR: {agg_wr:.1f}% | Edge: {agg_edge:+.1f}% | PnL: {total_pnl:+.1f}")

    # Veredicto
    print("\n" + "=" * 70)
    print("  VEREDICTO")
    print("=" * 70)

    positive_folds = len([f for f in folds if f['edge'] > 0])
    criteria = [
        ("Edge agregado > 0", agg_edge > 0 if total_trades > 0 else False),
        (">=50% folds positivos", positive_folds >= 2),
        ("Total trades >= 10", total_trades >= 10),
    ]

    passed = 0
    for name, ok in criteria:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")
        if ok:
            passed += 1

    print(f"\n  Resultado: {passed}/{len(criteria)} criterios")

    if passed >= 2:
        print("\n  *** APROVADO PARA PAPER TRADING ***")
    else:
        print("\n  *** REQUER MAIS AJUSTES ***")

    print("=" * 70)

if __name__ == "__main__":
    main()
