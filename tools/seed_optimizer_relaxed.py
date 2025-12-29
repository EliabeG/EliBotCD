#!/usr/bin/env python3
"""
SEED Relaxed Optimizer
======================
Versao com condicoes mais relaxadas para encontrar configuracao viavel
"""

import sys
import os
import urllib.request
import ssl
import json
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"
SPREAD_PIPS = 1.5
SLIPPAGE_PIPS = 0.8
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
                time.sleep(0.03)
        except:
            break
    bars.sort(key=lambda x: x.timestamp)
    return bars

def compute_signals_relaxed(bars: List[Bar], critical_sigma: float, slaving_threshold: float,
                            dominance_threshold: float, fitness_window: int) -> List[Tuple[int, int]]:
    """Computa sinais SEED com condicoes relaxadas"""
    from strategies.alta_volatilidade.seed_sintetizador_evolutivo import SintetizadorEvolutivoEstruturasDissipativas

    closes = np.array([b.close for b in bars])
    volumes = np.array([b.volume if b.volume > 0 else abs(closes[i] - closes[max(0,i-1)]) * 50000
                       for i, b in enumerate(bars)])

    seed = SintetizadorEvolutivoEstruturasDissipativas(
        critical_sigma=critical_sigma,
        slaving_threshold=slaving_threshold,
        dominance_threshold=dominance_threshold,
        fitness_window=fitness_window,
        dx_threshold=0.01  # Mais baixo
    )

    signals = []
    for i in range(50, len(closes) - 1, 4):
        try:
            result = seed.analyze(closes[:i], volumes[:i])

            # Sinal RELAXADO: basta sigma alto OU populacao dominante
            x1 = result['x1_bulls']
            x2 = result['x2_bears']
            sigma = result['sigma']

            sig = 0

            # Condicao 1: Sigma muito alto (extremo)
            if abs(sigma) > critical_sigma:
                if sigma > 0:
                    sig = 1  # Bull
                else:
                    sig = -1  # Bear

            # Condicao 2: Bulls muito dominante
            elif x1 > dominance_threshold and x1 > x2 * 1.5:
                sig = 1

            # Condicao 3: Bears muito dominante
            elif x2 > dominance_threshold and x2 > x1 * 1.5:
                sig = -1

            # Condicao 4: Sinal original do SEED
            elif result['signal'] != 0:
                sig = result['signal']

            if sig != 0:
                signals.append((i, sig))
        except:
            continue

    return signals

def simulate_trades(bars: List[Bar], signals: List[Tuple[int, int]],
                   sl: float, tp: float, cooldown: int,
                   start_idx: int = 0, end_idx: int = None,
                   direction: int = 1) -> Dict:
    """Simula trades"""
    if end_idx is None:
        end_idx = len(bars)

    n = len(bars)
    trades = []
    last_idx = -999

    for idx, sig in signals:
        if idx < start_idx or idx >= end_idx:
            continue
        if sig != direction:
            continue
        if idx - last_idx < cooldown:
            continue
        if idx + 1 >= n:
            continue

        entry = bars[idx + 1].open

        if direction == 1:
            sl_price = entry - sl * PIP_VALUE
            tp_price = entry + tp * PIP_VALUE
        else:
            sl_price = entry + sl * PIP_VALUE
            tp_price = entry - tp * PIP_VALUE

        for j in range(idx + 2, min(idx + 100, n)):
            bar = bars[j]
            if direction == 1:
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

    if not trades:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0}

    wins = len([t for t in trades if t > 0])
    total = len(trades)
    wr = wins / total * 100
    be = sl / (sl + tp) * 100
    pnl = sum(trades)
    gp = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))

    return {
        'total': total,
        'wins': wins,
        'wr': wr,
        'edge': wr - be,
        'pnl': pnl,
        'pf': gp / gl if gl > 0 else 0,
        'trades': trades
    }

def walk_forward(bars: List[Bar], signals: List[Tuple[int, int]],
                 sl: float, tp: float, cooldown: int, n_folds: int = 4) -> Dict:
    """Walk-forward simples"""
    n = len(bars)
    fold_size = n // n_folds
    results = []
    all_trades = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size
        r = simulate_trades(bars, signals, sl, tp, cooldown, start, end)
        if r['total'] >= 5:
            results.append(r)
            all_trades.extend(r['trades'])

    if not results:
        return {'valid': False}

    folds_pos = sum(1 for r in results if r['edge'] > 0)
    wins = len([t for t in all_trades if t > 0])
    total = len(all_trades)

    return {
        'valid': True,
        'folds_pos': folds_pos,
        'folds_total': len(results),
        'total': total,
        'wr': wins/total*100 if total > 0 else 0,
        'edge': (wins/total*100 - sl/(sl+tp)*100) if total > 0 else 0,
        'pnl': sum(all_trades)
    }

def main():
    print("=" * 80)
    print("  SEED RELAXED OPTIMIZER")
    print("=" * 80)

    start_time = time.time()

    print("\n[1/4] Baixando dados H1...", end=" ", flush=True)
    bars = download_bars("H1", 4000)
    print(f"{len(bars)} barras")

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()} ({days} dias)")

    # Grid relaxado
    print("\n[2/4] Grid de parametros relaxado...")

    critical_sigma_list = [0.5, 0.8, 1.0, 1.2, 1.5]  # Mais baixo
    slaving_threshold_list = [0.20, 0.25, 0.30]  # Mais alto (mais facil ativar)
    dominance_threshold_list = [0.40, 0.45, 0.50]  # Mais baixo
    fitness_window_list = [15, 20]

    sl_list = [30, 35, 40]
    tp_list = [45, 50, 60]
    cooldown_list = [6, 8]

    indicator_combos = list(itertools.product(
        critical_sigma_list,
        slaving_threshold_list,
        dominance_threshold_list,
        fitness_window_list
    ))

    trade_combos = list(itertools.product(sl_list, tp_list, cooldown_list))

    print(f"  Configs indicador: {len(indicator_combos)}")
    print(f"  Configs trade: {len(trade_combos)}")
    print(f"  Total: {len(indicator_combos) * len(trade_combos)}")

    # Pre-computar sinais
    print("\n[3/4] Computando sinais relaxados...")

    signals_cache = {}
    for i, (crit_sig, slav_th, dom_th, fit_win) in enumerate(indicator_combos):
        key = (crit_sig, slav_th, dom_th, fit_win)
        print(f"  {i+1}/{len(indicator_combos)} - sigma={crit_sig}, slav={slav_th}, dom={dom_th}...", end=" ", flush=True)

        try:
            t0 = time.time()
            signals_cache[key] = compute_signals_relaxed(bars, crit_sig, slav_th, dom_th, fit_win)
            elapsed = time.time() - t0
            print(f"{len(signals_cache[key])} sinais ({elapsed:.1f}s)")
        except Exception as e:
            print(f"Erro: {e}")
            signals_cache[key] = []

    # Testar combinacoes
    print("\n[4/4] Testando combinacoes...")

    results = []
    tested = 0

    for key, signals in signals_cache.items():
        if len(signals) < 20:
            continue

        crit_sig, slav_th, dom_th, fit_win = key

        for sl, tp, cooldown in trade_combos:
            r = simulate_trades(bars, signals, sl, tp, cooldown, direction=1)

            if r['total'] >= 30 and r['edge'] > 0:
                wf = walk_forward(bars, signals, sl, tp, cooldown)

                if wf.get('valid') and wf.get('folds_pos', 0) >= 2:
                    results.append({
                        'crit_sig': crit_sig,
                        'slav_th': slav_th,
                        'dom_th': dom_th,
                        'fit_win': fit_win,
                        'sl': sl,
                        'tp': tp,
                        'cooldown': cooldown,
                        'trades': r['total'],
                        'wr': r['wr'],
                        'edge': r['edge'],
                        'pf': r['pf'],
                        'wf_folds': wf['folds_pos'],
                        'wf_total': wf['folds_total'],
                        'wf_edge': wf['edge']
                    })

            tested += 1

    elapsed = time.time() - start_time
    print(f"\n  Testadas: {tested}, Validas: {len(results)} em {elapsed:.1f}s")

    if not results:
        print("\n  NENHUMA configuracao encontrada!")
        print("  O indicador SEED nao tem edge no periodo testado.")

        # Mostrar diagnostico
        print("\n  Diagnostico:")
        for key, signals in signals_cache.items():
            if len(signals) > 0:
                crit_sig, slav_th, dom_th, fit_win = key
                r = simulate_trades(bars, signals, 35, 50, 6, direction=1)
                print(f"    sigma={crit_sig}, slav={slav_th}: {len(signals)} sinais, {r['total']} trades, edge={r['edge']:.1f}%")

        return

    # Ordenar
    results.sort(key=lambda x: x['edge'], reverse=True)

    print("\n" + "=" * 80)
    print("  TOP 10 CONFIGURACOES")
    print("=" * 80)
    print(f"\n  {'#':>2} | {'Sig':>3} | {'Slv':>4} | {'Dom':>4} | {'SL':>3} | {'TP':>3} | {'CD':>2} | {'Trd':>4} | {'WR':>5} | {'Edge':>5} | {'PF':>5} | {'WF':>4}")
    print("-" * 85)

    for i, r in enumerate(results[:10]):
        wf_str = f"{r['wf_folds']}/{r['wf_total']}"
        print(f"  {i+1:2} | {r['crit_sig']:.1f} | {r['slav_th']:.2f} | {r['dom_th']:.2f} | {r['sl']:3.0f} | {r['tp']:3.0f} | {r['cooldown']:2} | {r['trades']:4} | {r['wr']:5.1f} | {r['edge']:+5.1f} | {r['pf']:5.2f} | {wf_str:>4}")

    # Salvar melhor
    best = results[0]

    config = {
        "strategy": "SEED-RELAXED",
        "symbol": SYMBOL,
        "periodicity": "H1",
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "mode": "LONG_ONLY",
        "note": "Configuracao com condicoes relaxadas - REQUER VALIDACAO ADICIONAL",
        "parameters": {
            "critical_sigma": float(best['crit_sig']),
            "slaving_threshold": float(best['slav_th']),
            "dominance_threshold": float(best['dom_th']),
            "fitness_window": int(best['fit_win']),
            "stop_loss_pips": int(best['sl']),
            "take_profit_pips": int(best['tp']),
            "cooldown_bars": int(best['cooldown'])
        },
        "performance": {
            "trades": int(best['trades']),
            "win_rate": float(best['wr']),
            "edge": float(best['edge']),
            "profit_factor": float(best['pf']),
            "walk_forward_folds": best['wf_folds'],
            "walk_forward_total": best['wf_total']
        }
    }

    path = "/home/azureuser/EliBotCD/configs/seed_relaxed.json"
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config salva: {path}")

    print(f"\n  Tempo total: {elapsed/60:.1f} minutos")

if __name__ == "__main__":
    main()
