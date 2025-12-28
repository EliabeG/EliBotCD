#!/usr/bin/env python3
"""
PHM Ultra Fast Test - Versão ultra-otimizada
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

from strategies.alta_volatilidade.phm_projetor_holografico import ProjetorHolograficoMaldacena

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
                if not batch:
                    break
                for b in batch:
                    ts = datetime.fromtimestamp(b["Timestamp"] / 1000, tz=timezone.utc)
                    bars.append(Bar(ts, b["Open"], b["High"], b["Low"], b["Close"], b.get("Volume", 0)))
                total += len(batch)
                oldest = min(batch, key=lambda x: x["Timestamp"])
                current_ts = oldest["Timestamp"] - 1
                time.sleep(0.1)
        except:
            break
    bars.sort(key=lambda x: x.timestamp)
    return bars


def run_phm_backtest(bars: List[Bar], sl_pips: float, tp_pips: float, cooldown: int,
                     mag_th: float = 0.03, long_only: bool = True,
                     start_pct: float = 0, end_pct: float = 100) -> Dict:
    n = len(bars)
    start_idx = int(n * start_pct / 100)
    end_idx = int(n * end_pct / 100)

    if end_idx - start_idx < 100:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'max_dd': 0,
                'wins': 0, 'be': sl_pips/(sl_pips+tp_pips)*100}

    closes = np.array([b.close for b in bars[start_idx:end_idx]])

    # PHM ultra-light
    phm = ProjetorHolograficoMaldacena(window_size=48, bond_dim=3, n_layers=2)

    signals = []
    min_bars = 60

    # Iteração a cada 8 barras para máxima eficiência
    for i in range(min_bars, len(closes) - 1, 8):
        try:
            result = phm.analyze(closes[:i])

            mag = result['magnetization']
            horizon = result['horizon_forming']
            spike = result['spike_magnitude']

            if not (horizon or spike >= 0.8):
                continue

            if mag > mag_th:
                signals.append((start_idx + i, 'LONG', bars[start_idx + i].timestamp))
            elif mag < -mag_th and not long_only:
                signals.append((start_idx + i, 'SHORT', bars[start_idx + i].timestamp))

        except:
            continue

    trades = []
    last_trade_idx = -999

    for idx, direction, _ in signals:
        if idx - last_trade_idx < cooldown:
            continue
        if idx + 1 >= n:
            continue

        entry = bars[idx + 1].open
        sl = entry - sl_pips * PIP_VALUE if direction == 'LONG' else entry + sl_pips * PIP_VALUE
        tp = entry + tp_pips * PIP_VALUE if direction == 'LONG' else entry - tp_pips * PIP_VALUE

        for j in range(idx + 2, min(idx + 60, n)):
            bar = bars[j]
            if direction == 'LONG':
                if bar.low <= sl:
                    trades.append(-sl_pips - SPREAD_PIPS - SLIPPAGE_PIPS)
                    break
                if bar.high >= tp:
                    trades.append(tp_pips - SPREAD_PIPS - SLIPPAGE_PIPS)
                    break
            else:
                if bar.high >= sl:
                    trades.append(-sl_pips - SPREAD_PIPS - SLIPPAGE_PIPS)
                    break
                if bar.low <= tp:
                    trades.append(tp_pips - SPREAD_PIPS - SLIPPAGE_PIPS)
                    break
        last_trade_idx = idx

    if not trades:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'max_dd': 0,
                'wins': 0, 'be': sl_pips/(sl_pips+tp_pips)*100}

    wins = len([t for t in trades if t > 0])
    total = len(trades)
    wr = wins / total * 100
    be = sl_pips / (sl_pips + tp_pips) * 100
    edge = wr - be
    pnl = sum(trades)
    gp = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))
    pf = gp / gl if gl > 0 else 0

    cumsum = np.cumsum(trades)
    running_max = np.maximum.accumulate(cumsum)
    max_dd = np.max(running_max - cumsum)

    return {'total': total, 'wins': wins, 'wr': wr, 'be': be, 'edge': edge,
            'pnl': pnl, 'pf': pf, 'max_dd': max_dd}


def main():
    print("=" * 70)
    print("  PHM ULTRA FAST OPTIMIZATION")
    print("=" * 70)

    print("\n  Baixando dados H4 (menor overhead)...")
    bars = download_bars("H4", 2500)

    if len(bars) < 500:
        print("  Dados insuficientes!")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()} ({days} dias)")
    print(f"  Barras: {len(bars)}")

    sl, tp, cooldown = 40, 80, 3

    # Testar diferentes thresholds de magnetização
    print("\n  Testando configurações...")
    print(f"\n  {'Mag':>5} | {'Trades':>6} | {'WR%':>6} | {'Edge%':>7} | {'PnL':>8} | {'PF':>5} | Status")
    print("  " + "-" * 55)

    best = None
    best_score = -999

    for mag_th in [0.02, 0.025, 0.03, 0.035, 0.04, 0.05]:
        result = run_phm_backtest(bars, sl, tp, cooldown, mag_th=mag_th, long_only=True)

        if result['total'] >= 5:
            status = "[OK]" if result['edge'] > 0 and result['pf'] > 1.0 else "[FAIL]"
            print(f"  {mag_th:>5} | {result['total']:>6} | {result['wr']:>5.1f}% | "
                  f"{result['edge']:>+6.1f}% | {result['pnl']:>+7.1f} | {result['pf']:>5.2f} | {status}")

            score = result['edge'] * np.sqrt(result['total'])
            if result['pf'] < 1.0:
                score -= 15
            if score > best_score:
                best_score = score
                best = result
                best['mag_th'] = mag_th
        else:
            print(f"  {mag_th:>5} | {result['total']:>6} | {'N/A':>6} | {'N/A':>7} | {'N/A':>8} | {'N/A':>5} | [SKIP]")

    if not best or best['total'] < 5:
        print("\n  Nenhuma config viável!")
        return

    print(f"\n  Melhor: Mag={best['mag_th']}")

    # Walk-forward
    print("\n" + "=" * 70)
    print("  WALK-FORWARD")
    print("=" * 70)

    folds = []
    for i in range(4):
        fold = run_phm_backtest(bars, sl, tp, cooldown, mag_th=best['mag_th'],
                                long_only=True, start_pct=i*25, end_pct=(i+1)*25)
        fold['period'] = f"Q{i+1}"
        folds.append(fold)

    print(f"\n  {'Fold':<6} | {'Trades':>6} | {'WR%':>6} | {'Edge%':>7} | {'PnL':>8} | Status")
    print("  " + "-" * 50)

    pos_folds = 0
    total_t = total_w = total_pnl = 0

    for f in folds:
        if f['total'] > 0:
            s = "[OK]" if f['edge'] > 0 else "[FAIL]"
            if f['edge'] > 0: pos_folds += 1
            total_t += f['total']
            total_w += f['wins']
            total_pnl += f['pnl']
            print(f"  {f['period']:<6} | {f['total']:>6} | {f['wr']:>5.1f}% | {f['edge']:>+6.1f}% | {f['pnl']:>+7.1f} | {s}")
        else:
            print(f"  {f['period']:<6} | {'N/A':>6} | {'N/A':>6} | {'N/A':>7} | {'N/A':>8} | [SKIP]")

    print("  " + "-" * 50)

    if total_t > 0:
        agg_wr = total_w / total_t * 100
        agg_be = sl / (sl + tp) * 100
        agg_edge = agg_wr - agg_be
        gp = sum(f['pnl'] for f in folds if f['pnl'] > 0)
        gl = abs(sum(f['pnl'] for f in folds if f['pnl'] < 0))
        agg_pf = gp / gl if gl > 0 else 0
        print(f"  {'AGG':<6} | {total_t:>6} | {agg_wr:>5.1f}% | {agg_edge:>+6.1f}% | {total_pnl:>+7.1f} |")
    else:
        agg_wr = agg_edge = agg_pf = 0

    print(f"\n  Folds positivos: {pos_folds}/4")

    # Veredicto
    print("\n" + "=" * 70)
    print("  VEREDICTO PHM")
    print("=" * 70)

    criteria = [
        ("Edge agregado > 0%", agg_edge > 0),
        ("Profit Factor > 1.0", agg_pf > 1.0),
        (">=2 folds positivos", pos_folds >= 2),
        ("Max DD < 300 pips", best['max_dd'] < 300),
        ("Trades >= 10", total_t >= 10),
    ]

    passed = sum(1 for _, ok in criteria if ok)
    for name, ok in criteria:
        print(f"  {'[PASS]' if ok else '[FAIL]'} {name}")

    print(f"\n  Resultado: {passed}/5")

    if passed >= 4:
        verdict = "APROVADO"
        print("\n  *** APROVADO PARA PAPER TRADING ***")
    elif passed >= 3:
        verdict = "APROVADO_COM_RESSALVAS"
        print("\n  *** APROVADO COM RESSALVAS ***")
    else:
        verdict = "REPROVADO"
        print("\n  *** NAO RECOMENDADO ***")

    # Salvar
    config = {
        "strategy": "PHM-UltraFast-H4",
        "symbol": SYMBOL,
        "periodicity": "H4",
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "mode": "LONG_ONLY",
        "parameters": {
            "window_size": 48,
            "bond_dim": 3,
            "n_layers": 2,
            "magnetization_threshold": best['mag_th'],
            "stop_loss_pips": sl,
            "take_profit_pips": tp,
            "cooldown_bars": cooldown
        },
        "performance": {
            "trades": total_t,
            "win_rate": agg_wr,
            "edge": agg_edge,
            "profit_factor": agg_pf,
            "total_pnl_pips": total_pnl,
            "max_drawdown_pips": best['max_dd']
        },
        "walk_forward": {"folds_positive": pos_folds, "total_folds": 4},
        "criteria_passed": f"{passed}/5",
        "verdict": verdict
    }

    path = "/home/azureuser/EliBotCD/configs/phm_h4_optimized.json"
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
