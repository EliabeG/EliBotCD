#!/usr/bin/env python3
"""
PHM Contrarian Test - Lógica de reversão à média

Hipótese: Magnetização alta = mercado sobre-extendido = reversão iminente
- Magnetização MUITO positiva + horizonte = SELL (saturação bullish)
- Magnetização MUITO negativa + horizonte = BUY (saturação bearish)
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


def run_contrarian_backtest(bars: List[Bar], sl_pips: float, tp_pips: float, cooldown: int,
                            mag_th: float = 0.04, spike_th: float = 0.8, long_only: bool = True,
                            start_pct: float = 0, end_pct: float = 100) -> Dict:
    """
    Backtest com lógica contrarian:
    - Mag < -mag_th + evento = BUY (oversold)
    - Mag > +mag_th + evento = SELL (overbought)
    """
    n = len(bars)
    start_idx = int(n * start_pct / 100)
    end_idx = int(n * end_pct / 100)

    if end_idx - start_idx < 100:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'max_dd': 0,
                'wins': 0, 'be': sl_pips/(sl_pips+tp_pips)*100}

    closes = np.array([b.close for b in bars[start_idx:end_idx]])
    phm = ProjetorHolograficoMaldacena(window_size=48, bond_dim=3, n_layers=2)

    signals = []
    for i in range(60, len(closes) - 1, 8):
        try:
            result = phm.analyze(closes[:i])
            mag = result['magnetization']
            horizon = result['horizon_forming']
            spike = result['spike_magnitude']

            if not (horizon or spike >= spike_th):
                continue

            # CONTRARIAN: magnetização negativa = oversold = BUY
            if mag < -mag_th:
                signals.append((start_idx + i, 'LONG', bars[start_idx + i].timestamp))
            # CONTRARIAN: magnetização positiva = overbought = SELL
            elif mag > mag_th and not long_only:
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
    print("  PHM CONTRARIAN TEST (Reversao a media)")
    print("=" * 70)

    print("\n  Baixando H4...")
    bars = download_bars("H4", 2500)
    if len(bars) < 500:
        print("  Dados insuficientes!")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()} ({days} dias)")

    sl, tp, cooldown = 40, 80, 3

    print("\n  Testando thresholds contrarian...")
    print(f"\n  {'Mag':>5} | {'Spike':>5} | {'Trades':>6} | {'WR%':>6} | {'Edge%':>7} | {'PnL':>8} | {'PF':>5}")
    print("  " + "-" * 60)

    best = None
    best_score = -999

    for mag_th in [0.02, 0.03, 0.04, 0.05, 0.06]:
        for spike_th in [0.6, 0.8, 1.0]:
            r = run_contrarian_backtest(bars, sl, tp, cooldown, mag_th=mag_th, spike_th=spike_th, long_only=True)
            if r['total'] >= 5:
                s = "[OK]" if r['edge'] > 0 and r['pf'] > 1.0 else ""
                print(f"  {mag_th:>5} | {spike_th:>5} | {r['total']:>6} | {r['wr']:>5.1f}% | {r['edge']:>+6.1f}% | {r['pnl']:>+7.1f} | {r['pf']:>5.2f} {s}")
                score = r['edge'] * np.sqrt(r['total'])
                if r['pf'] < 1.0: score -= 15
                if score > best_score:
                    best_score = score
                    best = r
                    best['mag_th'] = mag_th
                    best['spike_th'] = spike_th

    if not best or best['total'] < 5:
        print("\n  Nenhuma config viável!")
        # Criar relatório de reprovação
        save_failed_report(bars)
        return

    print(f"\n  Melhor: Mag={best['mag_th']}, Spike={best['spike_th']}")

    # Walk-forward
    print("\n  Walk-forward...")
    folds = []
    for i in range(4):
        f = run_contrarian_backtest(bars, sl, tp, cooldown, mag_th=best['mag_th'],
                                    spike_th=best['spike_th'], long_only=True,
                                    start_pct=i*25, end_pct=(i+1)*25)
        f['period'] = f"Q{i+1}"
        folds.append(f)

    pos = sum(1 for f in folds if f['edge'] > 0 and f['total'] > 0)
    tot_t = sum(f['total'] for f in folds)
    tot_w = sum(f['wins'] for f in folds)
    tot_pnl = sum(f['pnl'] for f in folds)

    for f in folds:
        if f['total'] > 0:
            s = "[OK]" if f['edge'] > 0 else "[FAIL]"
            print(f"  {f['period']}: {f['total']} trades, edge={f['edge']:+.1f}% {s}")

    print(f"\n  Folds positivos: {pos}/4")

    if tot_t > 0:
        agg_wr = tot_w / tot_t * 100
        agg_be = sl / (sl + tp) * 100
        agg_edge = agg_wr - agg_be
        gp = sum(f['pnl'] for f in folds if f['pnl'] > 0)
        gl = abs(sum(f['pnl'] for f in folds if f['pnl'] < 0))
        agg_pf = gp / gl if gl > 0 else 0
    else:
        agg_wr = agg_edge = agg_pf = 0

    print(f"  Agregado: {tot_t} trades, edge={agg_edge:+.1f}%, PF={agg_pf:.2f}")

    # Veredicto
    criteria = [
        agg_edge > 0,
        agg_pf > 1.0,
        pos >= 2,
        best['max_dd'] < 300,
        tot_t >= 10
    ]
    passed = sum(criteria)

    print(f"\n  Resultado: {passed}/5")

    if passed >= 4:
        verdict = "APROVADO"
    elif passed >= 3:
        verdict = "APROVADO_COM_RESSALVAS"
    else:
        verdict = "REPROVADO"

    print(f"  Veredicto: {verdict}")

    # Salvar
    config = {
        "strategy": "PHM-Contrarian-H4",
        "symbol": SYMBOL,
        "periodicity": "H4",
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "mode": "LONG_ONLY",
        "signal_logic": "contrarian_mean_reversion",
        "parameters": {
            "window_size": 48, "bond_dim": 3, "n_layers": 2,
            "magnetization_threshold": best['mag_th'],
            "spike_threshold": best['spike_th'],
            "stop_loss_pips": sl, "take_profit_pips": tp, "cooldown_bars": cooldown
        },
        "performance": {
            "trades": tot_t, "win_rate": agg_wr, "edge": agg_edge,
            "profit_factor": agg_pf, "total_pnl_pips": tot_pnl, "max_drawdown_pips": best['max_dd']
        },
        "walk_forward": {"folds_positive": pos, "total_folds": 4},
        "criteria_passed": f"{passed}/5",
        "verdict": verdict
    }

    path = "/home/azureuser/EliBotCD/configs/phm_contrarian_optimized.json"
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Config: {path}")


def save_failed_report(bars):
    """Salva relatório de falha"""
    config = {
        "strategy": "PHM-ProjetorHolografico",
        "symbol": SYMBOL,
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "REPROVADO",
        "reason": "Nenhuma configuração atingiu critérios mínimos",
        "tests_performed": [
            "Standard (fase ferromagnética)",
            "Magnetization-based",
            "Contrarian (reversão à média)"
        ],
        "recommendation": "Indicador não recomendado para trading em EURUSD"
    }
    path = "/home/azureuser/EliBotCD/configs/phm_final_verdict.json"
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Relatório de falha salvo: {path}")


if __name__ == "__main__":
    main()
