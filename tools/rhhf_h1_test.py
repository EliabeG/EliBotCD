#!/usr/bin/env python3
"""RHHF H1 Test - Testar em timeframe menor"""

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
from strategies.alta_volatilidade.rhhf_ressonador_hilbert_huang import RessonadorHilbertHuangFractal

API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"
SPREAD_PIPS = 1.2
SLIPPAGE_PIPS = 0.5
PIP_VALUE = 0.0001

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
    while total < count:
        remaining = min(1000, count - total)
        url = f"{API_BASE}/{SYMBOL}/{period}/bars/ask?timestamp={current_ts}&count=-{remaining}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30, context=ctx) as response:
                data = json.loads(response.read().decode())
                batch = data.get("Bars", [])
                if not batch: break
                for b in batch:
                    ts = datetime.fromtimestamp(b["Timestamp"] / 1000, tz=timezone.utc)
                    bars.append(Bar(ts, b["Open"], b["High"], b["Low"], b["Close"], b.get("Volume", 0)))
                total += len(batch)
                oldest = min(batch, key=lambda x: x["Timestamp"])
                current_ts = oldest["Timestamp"] - 1
                time.sleep(0.1)
        except: break
    bars.sort(key=lambda x: x.timestamp)
    return bars

def run_backtest(bars, sl, tp, cooldown, n_ens=15, frac_th=1.2, start_pct=0, end_pct=100):
    n = len(bars)
    start_idx = int(n * start_pct / 100)
    end_idx = int(n * end_pct / 100)
    if end_idx - start_idx < 150:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'max_dd': 0, 'wins': 0, 'be': sl/(sl+tp)*100}

    closes = np.array([b.close for b in bars[start_idx:end_idx]])
    rhhf = RessonadorHilbertHuangFractal(n_ensembles=n_ens, noise_amplitude=0.2, mirror_extension=30,
                                          use_predictive_extension=True, fractal_threshold=frac_th)

    signals = []
    for i in range(120, len(closes) - 1, 8):
        try:
            result = rhhf.analyze(closes[:i])
            if result['signal_details']['conditions_met'] >= 2 and result['signal'] == 1:
                signals.append((start_idx + i, 'LONG', bars[start_idx + i].timestamp))
        except: continue

    trades = []
    last_idx = -999
    for idx, direction, _ in signals:
        if idx - last_idx < cooldown or idx + 1 >= n: continue
        entry = bars[idx + 1].open
        sl_price = entry - sl * PIP_VALUE
        tp_price = entry + tp * PIP_VALUE
        for j in range(idx + 2, min(idx + 60, n)):
            bar = bars[j]
            if bar.low <= sl_price:
                trades.append(-sl - SPREAD_PIPS - SLIPPAGE_PIPS)
                break
            if bar.high >= tp_price:
                trades.append(tp - SPREAD_PIPS - SLIPPAGE_PIPS)
                break
        last_idx = idx

    if not trades:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'max_dd': 0, 'wins': 0, 'be': sl/(sl+tp)*100}

    wins = len([t for t in trades if t > 0])
    total = len(trades)
    wr = wins / total * 100
    be = sl / (sl + tp) * 100
    pnl = sum(trades)
    gp = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))
    cumsum = np.cumsum(trades)
    max_dd = np.max(np.maximum.accumulate(cumsum) - cumsum)

    return {'total': total, 'wins': wins, 'wr': wr, 'be': be, 'edge': wr - be,
            'pnl': pnl, 'pf': gp/gl if gl > 0 else 0, 'max_dd': max_dd}

def main():
    print("=" * 70)
    print("  RHHF H1 TEST")
    print("=" * 70)

    print("\n  Baixando H1...", end=" ", flush=True)
    bars = download_bars("H1", 4000)
    print(f"{len(bars)} barras")

    if len(bars) < 500:
        print("  Dados insuficientes!")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()} ({days} dias)")

    sl, tp, cooldown = 25, 50, 4

    print("\n  Testando configurações...")
    configs = [
        {'ens': 15, 'frac': 1.1},
        {'ens': 15, 'frac': 1.2},
        {'ens': 20, 'frac': 1.2},
    ]

    best = None
    best_score = -999

    for c in configs:
        print(f"    ens={c['ens']}, frac={c['frac']}...", end=" ", flush=True)
        r = run_backtest(bars, sl, tp, cooldown, n_ens=c['ens'], frac_th=c['frac'])
        print(f"{r['total']} trades, edge={r['edge']:+.1f}%")
        if r['total'] >= 10:
            score = r['edge'] * np.sqrt(r['total'])
            if r['pf'] < 1.0: score -= 15
            if score > best_score:
                best_score = score
                best = r
                best['config'] = c

    if not best:
        print("  Nenhuma config viável!")
        return

    print(f"\n  Melhor: ens={best['config']['ens']}, frac={best['config']['frac']}")
    print(f"  Resultado: {best['total']} trades | WR={best['wr']:.1f}% | Edge={best['edge']:+.1f}% | PF={best['pf']:.2f}")

    # Walk-forward
    print("\n  Walk-forward...")
    folds = []
    for i in range(4):
        f = run_backtest(bars, sl, tp, cooldown, n_ens=best['config']['ens'],
                         frac_th=best['config']['frac'], start_pct=i*25, end_pct=(i+1)*25)
        f['period'] = f"Q{i+1}"
        folds.append(f)

    pos = sum(1 for f in folds if f['edge'] > 0 and f['total'] > 0)
    tot_t = sum(f['total'] for f in folds)
    tot_w = sum(f['wins'] for f in folds)
    tot_pnl = sum(f['pnl'] for f in folds)

    for f in folds:
        if f['total'] > 0:
            s = "[OK]" if f['edge'] > 0 else "[FAIL]"
            print(f"    {f['period']}: {f['total']} trades, edge={f['edge']:+.1f}% {s}")

    print(f"\n  Folds positivos: {pos}/4")

    if tot_t > 0:
        agg_wr = tot_w / tot_t * 100
        agg_edge = agg_wr - sl/(sl+tp)*100
        gp = sum(f['pnl'] for f in folds if f['pnl'] > 0)
        gl = abs(sum(f['pnl'] for f in folds if f['pnl'] < 0))
        agg_pf = gp/gl if gl > 0 else 0
        print(f"  Agregado: {tot_t} trades, edge={agg_edge:+.1f}%, PF={agg_pf:.2f}")
    else:
        agg_edge = agg_pf = 0

    # Veredicto
    criteria = [agg_edge > 0, agg_pf > 1.0, pos >= 2, best['max_dd'] < 200, tot_t >= 15]
    passed = sum(criteria)
    print(f"\n  Resultado: {passed}/5 criterios")

    if passed >= 4:
        print("  *** APROVADO ***")
        verdict = "APROVADO"
    elif passed >= 3:
        print("  *** APROVADO COM RESSALVAS ***")
        verdict = "APROVADO_COM_RESSALVAS"
    else:
        print("  *** NAO RECOMENDADO ***")
        verdict = "REPROVADO"

    # Salvar
    config = {
        "strategy": "RHHF-H1",
        "symbol": SYMBOL,
        "periodicity": "H1",
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "mode": "LONG_ONLY",
        "parameters": {
            "n_ensembles": best['config']['ens'],
            "fractal_threshold": best['config']['frac'],
            "stop_loss_pips": sl,
            "take_profit_pips": tp,
            "cooldown_bars": cooldown
        },
        "performance": {
            "trades": tot_t, "win_rate": agg_wr if tot_t > 0 else 0,
            "edge": agg_edge if tot_t > 0 else 0, "profit_factor": agg_pf if tot_t > 0 else 0,
            "total_pnl_pips": tot_pnl, "max_drawdown_pips": best['max_dd']
        },
        "walk_forward": {"folds_positive": pos, "total_folds": 4},
        "criteria_passed": f"{passed}/5",
        "verdict": verdict
    }

    path = "/home/azureuser/EliBotCD/configs/rhhf_h1_optimized.json"
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config: {path}")


if __name__ == "__main__":
    main()
