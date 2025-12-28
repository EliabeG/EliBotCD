#!/usr/bin/env python3
"""
FIFN Multi-Timeframe Test
===============================================================================
Testa o FIFN em diferentes timeframes: M5, M15, M30, H1, H4
"""

import sys
import os
import urllib.request
import ssl
import json
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.alta_volatilidade.fifn_fisher_navier import FluxoInformacaoFisherNavier

# =============================================================================
# PARAMETROS BASE
# =============================================================================

REYNOLDS_LOW = 2521.0
REYNOLDS_HIGH = 3786.0
SKEWNESS_THRESHOLD = 0.3091
COOLDOWN_BARS = 3
SPREAD_PIPS = 1.2
SLIPPAGE_PIPS = 0.5
PIP_VALUE = 0.0001

API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"

# SL/TP por timeframe (proporcional à volatilidade)
TIMEFRAME_PARAMS = {
    "M5":  {"sl": 8.0,  "tp": 16.0, "bars": 5000, "window": 30},
    "M15": {"sl": 12.0, "tp": 24.0, "bars": 4000, "window": 35},
    "M30": {"sl": 15.0, "tp": 30.0, "bars": 3500, "window": 40},
    "H1":  {"sl": 22.5, "tp": 27.9, "bars": 3000, "window": 50},
    "H4":  {"sl": 40.0, "tp": 80.0, "bars": 2000, "window": 50},
}

@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

def download_bars(period: str, count: int) -> List[Bar]:
    """Download barras da API."""
    bars = []
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    current_ts = int(time.time() * 1000)
    total = 0

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
                time.sleep(0.1)
        except Exception as e:
            print(f"      Erro: {e}")
            break

    bars.sort(key=lambda x: x.timestamp)
    return bars

def test_timeframe(tf: str, params: Dict) -> Dict:
    """Testa FIFN em um timeframe específico."""

    sl_pips = params["sl"]
    tp_pips = params["tp"]
    bar_count = params["bars"]
    window_size = params["window"]

    print(f"\n    Baixando {bar_count} barras...")
    bars = download_bars(tf, bar_count)

    if len(bars) < 500:
        return {"error": f"Apenas {len(bars)} barras"}

    days = (bars[-1].timestamp - bars[0].timestamp).days
    n = len(bars)

    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])

    # FIFN
    print(f"    Calculando FIFN...")
    fifn = FluxoInformacaoFisherNavier(
        window_size=window_size,
        kl_lookback=10,
        reynolds_sweet_low=REYNOLDS_LOW,
        reynolds_sweet_high=REYNOLDS_HIGH,
        skewness_threshold=SKEWNESS_THRESHOLD
    )

    try:
        result = fifn.analyze(closes)
        reynolds_series = result['reynolds_series']
        pressure_gradient = result['pressure_gradient_series']

        from scipy import stats
        returns = np.diff(np.log(closes))
        skewness_arr = np.zeros(len(returns))
        for i in range(window_size, len(returns)):
            skewness_arr[i] = stats.skew(returns[i - window_size:i])

    except Exception as e:
        return {"error": str(e)}

    # Filtros técnicos
    print(f"    Calculando filtros...")
    ema_12 = np.zeros(n)
    ema_26 = np.zeros(n)
    rsi_14 = np.full(n, 50.0)

    alpha_12 = 2.0 / 13
    alpha_26 = 2.0 / 27
    if n > 12:
        ema_12[12] = np.mean(closes[:12])
    if n > 26:
        ema_26[26] = np.mean(closes[:26])
    for i in range(13, n):
        ema_12[i] = alpha_12 * closes[i-1] + (1 - alpha_12) * ema_12[i-1]
    for i in range(27, n):
        ema_26[i] = alpha_26 * closes[i-1] + (1 - alpha_26) * ema_26[i-1]

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    for i in range(15, n):
        avg_gain = np.mean(gains[max(0, i-14):i])
        avg_loss = np.mean(losses[max(0, i-14):i])
        if avg_loss == 0:
            rsi_14[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_14[i] = 100.0 - (100.0 / (1.0 + rs))

    # Gerar sinais
    print(f"    Gerando sinais...")
    signals_raw = []
    signals_filtered = []

    for i in range(100, n - 1):
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

        # Sinais FIFN puro
        if pg < 0 and skew > SKEWNESS_THRESHOLD:
            signals_raw.append((i, 'LONG'))
            if session_ok and (trend_up or abs(ema_12[i] - ema_26[i]) < 0.001) and 35 <= rsi <= 75:
                signals_filtered.append((i, 'LONG'))

        elif pg > 0 and skew < -SKEWNESS_THRESHOLD:
            signals_raw.append((i, 'SHORT'))
            if session_ok and (trend_down or abs(ema_12[i] - ema_26[i]) < 0.001) and 25 <= rsi <= 65:
                signals_filtered.append((i, 'SHORT'))

    # Simular trades
    def simulate(signal_list, sl, tp):
        trades = []
        last_idx = -999

        for idx, direction in signal_list:
            if idx - last_idx < COOLDOWN_BARS:
                continue
            if idx + 1 >= n:
                continue

            entry = bars[idx + 1].open
            sl_price = entry - sl * PIP_VALUE if direction == 'LONG' else entry + sl * PIP_VALUE
            tp_price = entry + tp * PIP_VALUE if direction == 'LONG' else entry - tp * PIP_VALUE

            for j in range(idx + 2, min(idx + 200, n)):
                bar = bars[j]
                if direction == 'LONG':
                    if bar.low <= sl_price:
                        trades.append(-sl - SPREAD_PIPS - SLIPPAGE_PIPS)
                        break
                    if bar.high >= tp_price:
                        trades.append(tp - SPREAD_PIPS - SLIPPAGE_PIPS)
                        break
                else:
                    if bar.high >= sl_price:
                        trades.append(-sl - SPREAD_PIPS - SLIPPAGE_PIPS)
                        break
                    if bar.low <= tp_price:
                        trades.append(tp - SPREAD_PIPS - SLIPPAGE_PIPS)
                        break
            last_idx = idx

        return trades

    print(f"    Simulando trades...")
    trades_raw = simulate(signals_raw, sl_pips, tp_pips)
    trades_filtered = simulate(signals_filtered, sl_pips, tp_pips)

    # Calcular métricas
    def calc_metrics(trades, sl, tp):
        if not trades:
            return {"total": 0, "wr": 0, "be": 0, "edge": 0, "pnl": 0, "pf": 0}

        wins = len([t for t in trades if t > 0])
        total = len(trades)
        wr = wins / total * 100
        be = sl / (sl + tp) * 100
        edge = wr - be
        pnl = sum(trades)
        gp = sum(t for t in trades if t > 0)
        gl = abs(sum(t for t in trades if t < 0))
        pf = gp / gl if gl > 0 else 0

        return {"total": total, "wins": wins, "wr": wr, "be": be, "edge": edge, "pnl": pnl, "pf": pf}

    return {
        "days": days,
        "bars": n,
        "sl": sl_pips,
        "tp": tp_pips,
        "signals_raw": len(signals_raw),
        "signals_filtered": len(signals_filtered),
        "raw": calc_metrics(trades_raw, sl_pips, tp_pips),
        "filtered": calc_metrics(trades_filtered, sl_pips, tp_pips)
    }

def main():
    print("=" * 80)
    print("  FIFN MULTI-TIMEFRAME TEST")
    print("=" * 80)

    results = {}
    timeframes = ["M5", "M15", "M30", "H1", "H4"]

    for tf in timeframes:
        print(f"\n{'='*80}")
        print(f"  TIMEFRAME: {tf}")
        print(f"{'='*80}")

        params = TIMEFRAME_PARAMS[tf]
        print(f"    SL: {params['sl']} pips | TP: {params['tp']} pips | Window: {params['window']}")

        result = test_timeframe(tf, params)
        results[tf] = result

        if "error" in result:
            print(f"    ERRO: {result['error']}")
            continue

        print(f"\n    Periodo: {result['days']} dias | Barras: {result['bars']}")

        print(f"\n    SEM FILTROS:")
        raw = result['raw']
        if raw['total'] > 0:
            print(f"      Trades: {raw['total']} | WR: {raw['wr']:.1f}% (BE: {raw['be']:.1f}%)")
            print(f"      Edge: {raw['edge']:+.1f}% | PnL: {raw['pnl']:+.1f} pips | PF: {raw['pf']:.2f}")
        else:
            print(f"      Nenhum trade")

        print(f"\n    COM FILTROS:")
        filt = result['filtered']
        if filt['total'] > 0:
            print(f"      Trades: {filt['total']} | WR: {filt['wr']:.1f}% (BE: {filt['be']:.1f}%)")
            print(f"      Edge: {filt['edge']:+.1f}% | PnL: {filt['pnl']:+.1f} pips | PF: {filt['pf']:.2f}")
        else:
            print(f"      Nenhum trade")

    # Resumo comparativo
    print("\n" + "=" * 80)
    print("  RESUMO COMPARATIVO")
    print("=" * 80)

    print("\n  Timeframe | Dias | Trades | WR%   | Edge%  | PnL     | PF   | Status")
    print("  " + "-" * 75)

    best_tf = None
    best_edge = -999

    for tf in timeframes:
        r = results[tf]
        if "error" in r:
            print(f"  {tf:9s} | ERRO")
            continue

        filt = r['filtered']
        if filt['total'] > 0:
            status = "OK" if filt['edge'] > 0 and filt['pf'] > 1.0 else "WARN" if filt['edge'] > 0 else "FAIL"
            print(f"  {tf:9s} | {r['days']:4d} | {filt['total']:6d} | {filt['wr']:5.1f} | {filt['edge']:+5.1f}% | {filt['pnl']:+7.1f} | {filt['pf']:4.2f} | [{status}]")

            if filt['edge'] > best_edge and filt['total'] >= 5:
                best_edge = filt['edge']
                best_tf = tf
        else:
            print(f"  {tf:9s} | {r['days']:4d} |      0 |   -   |    -   |     -   |   -  | [FAIL]")

    print("\n" + "=" * 80)
    print("  CONCLUSAO")
    print("=" * 80)

    if best_tf:
        print(f"\n  Melhor timeframe: {best_tf}")
        print(f"  Edge: {best_edge:+.1f}%")
        r = results[best_tf]
        print(f"  SL/TP: {r['sl']}/{r['tp']} pips")

        if best_edge > 5:
            print("\n  *** RECOMENDADO PARA PAPER TRADING ***")
        elif best_edge > 0:
            print("\n  *** APROVADO COM RESSALVAS ***")
        else:
            print("\n  *** NAO RECOMENDADO ***")
    else:
        print("\n  Nenhum timeframe com resultados validos")

    print("=" * 80)

if __name__ == "__main__":
    main()
