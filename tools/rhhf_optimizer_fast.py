#!/usr/bin/env python3
"""
RHHF Fast Optimizer - Versao otimizada para 100k combinacoes
Usa cache agressivo e menos ensembles para ser rapido
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
                time.sleep(0.03)
        except:
            break
    bars.sort(key=lambda x: x.timestamp)
    return bars

def precompute_signals_fast(bars: List[Bar], n_ens: int, frac_th: float,
                            noise_amp: float, min_conds: int, step: int = 12) -> List[Tuple[int, int]]:
    """Pre-computa sinais - versao rapida"""
    from strategies.alta_volatilidade.rhhf_ressonador_hilbert_huang import RessonadorHilbertHuangFractal

    closes = np.array([b.close for b in bars])
    # Usar menos ensembles para ser mais rapido
    rhhf = RessonadorHilbertHuangFractal(
        n_ensembles=n_ens,
        noise_amplitude=noise_amp,
        fractal_threshold=frac_th,
        mirror_extension=20  # Menor extensao
    )

    signals = []
    # Maior passo para ser mais rapido
    for i in range(100, len(closes) - 1, step):
        try:
            result = rhhf.analyze(closes[:i])
            sig = result['signal']
            conds = result['signal_details']['conditions_met']
            if conds >= min_conds:
                signals.append((i, sig))
        except:
            continue

    return signals

def simulate_trades(bars: List[Bar], signals: List[Tuple[int, int]],
                   sl: float, tp: float, cooldown: int, direction: int = 1) -> Dict:
    """Simula trades com SL/TP especificos"""
    n = len(bars)
    trades = []
    last_idx = -999

    for idx, sig in signals:
        if sig != direction:
            continue
        if idx - last_idx < cooldown:
            continue
        if idx + 1 >= n:
            continue

        entry = bars[idx + 1].open

        if direction == 1:  # LONG
            sl_price = entry - sl * PIP_VALUE
            tp_price = entry + tp * PIP_VALUE
        else:  # SHORT
            sl_price = entry + sl * PIP_VALUE
            tp_price = entry - tp * PIP_VALUE

        for j in range(idx + 2, min(idx + 50, n)):
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
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'max_dd': 0}

    wins = len([t for t in trades if t > 0])
    total = len(trades)
    wr = wins / total * 100
    be = sl / (sl + tp) * 100
    pnl = sum(trades)
    gp = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))

    cumsum = np.cumsum(trades)
    max_dd = np.max(np.maximum.accumulate(cumsum) - cumsum) if len(cumsum) > 0 else 0

    return {
        'total': total,
        'wins': wins,
        'wr': wr,
        'edge': wr - be,
        'pnl': pnl,
        'pf': gp / gl if gl > 0 else 0,
        'max_dd': max_dd
    }

def main():
    print("=" * 80)
    print("  RHHF FAST OPTIMIZER - 100.000 COMBINACOES")
    print("=" * 80)

    start_time = time.time()

    # Baixar dados
    print("\n[1/4] Baixando dados H4...", end=" ", flush=True)
    bars = download_bars("H4", 2000)
    print(f"{len(bars)} barras")

    if len(bars) < 500:
        print("  Dados insuficientes!")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()} ({days} dias)")

    # Definir grid de parametros REDUZIDO para ser mais rapido
    print("\n[2/4] Gerando combinacoes...")

    # Parametros do indicador - REDUZIDOS
    n_ensembles_list = [5, 8, 10, 15]  # Menos opcoes
    frac_threshold_list = [1.25, 1.30, 1.35, 1.40, 1.45, 1.50]
    noise_amplitude_list = [0.15, 0.2, 0.25]  # Menos opcoes
    min_conditions_list = [1, 2, 3]

    # Parametros de trade - COMPLETOS
    sl_list = [20, 25, 30, 35, 40, 45, 50, 55, 60]
    tp_list = [40, 50, 60, 70, 80, 90, 100, 110, 120]
    cooldown_list = [2, 3, 4, 5, 6, 7, 8]

    # Todas as combinacoes de trade
    trade_combos = list(itertools.product(sl_list, tp_list, cooldown_list))

    # Todas as combinacoes de indicador
    indicator_combos = list(itertools.product(
        n_ensembles_list,
        frac_threshold_list,
        noise_amplitude_list,
        min_conditions_list
    ))

    total_combos = len(indicator_combos) * len(trade_combos)
    print(f"  Configs de indicador: {len(indicator_combos)}")
    print(f"  Configs de trade: {len(trade_combos)}")
    print(f"  Total de combinacoes: {total_combos:,}")

    # Pre-computar sinais para cada config de indicador
    print("\n[3/4] Pre-computando sinais...")

    signals_cache = {}
    step = 12  # Passo grande para ser rapido

    for i, (n_ens, frac_th, noise_amp, min_conds) in enumerate(indicator_combos):
        key = (n_ens, frac_th, noise_amp, min_conds)
        print(f"  {i+1}/{len(indicator_combos)} - ens={n_ens}, frac={frac_th:.2f}...", end=" ", flush=True)

        try:
            t0 = time.time()
            signals_cache[key] = precompute_signals_fast(bars, n_ens, frac_th, noise_amp, min_conds, step)
            elapsed = time.time() - t0
            print(f"{len(signals_cache[key])} sinais ({elapsed:.1f}s)")
        except Exception as e:
            print(f"Erro: {e}")
            signals_cache[key] = []

    # Testar todas as combinacoes de trade
    print("\n[4/4] Testando combinacoes de trade...")

    results = []
    tested = 0

    for key, signals in signals_cache.items():
        if not signals:
            continue

        n_ens, frac_th, noise_amp, min_conds = key

        for sl, tp, cooldown in trade_combos:
            r = simulate_trades(bars, signals, sl, tp, cooldown, direction=1)

            if r['total'] >= 5:
                score = r['edge'] * np.sqrt(r['total'])
                if r['pf'] < 1.0:
                    score -= 15
                if r['max_dd'] > 300:
                    score -= 5

                results.append({
                    'n_ens': n_ens,
                    'frac_th': frac_th,
                    'noise_amp': noise_amp,
                    'min_conds': min_conds,
                    'sl': sl,
                    'tp': tp,
                    'cooldown': cooldown,
                    'score': score,
                    **r
                })

            tested += 1

        if tested % 5000 == 0:
            print(f"  {tested:,}/{total_combos:,} testadas...")

    elapsed = time.time() - start_time
    print(f"\n  Total testado: {tested:,} em {elapsed:.1f}s ({tested/elapsed:.0f} combos/s)")

    # Ordenar por score
    results.sort(key=lambda x: x['score'], reverse=True)

    # Resultados
    print("\n" + "=" * 80)
    print("  RESULTADOS")
    print("=" * 80)

    print(f"\n  Combinacoes validas: {len(results):,}")

    if not results:
        print("  Nenhum resultado valido!")
        return

    # Top 30
    print("\n  TOP 30 CONFIGURACOES:")
    print("-" * 110)
    print(f"  {'#':>3} | {'Ens':>3} | {'FracTh':>6} | {'Noise':>5} | {'Conds':>5} | {'SL':>3} | {'TP':>3} | {'CD':>2} | {'Trades':>6} | {'WR%':>5} | {'Edge%':>6} | {'PF':>5} | {'Score':>7}")
    print("-" * 110)

    for i, r in enumerate(results[:30]):
        print(f"  {i+1:3} | {r['n_ens']:3} | {r['frac_th']:6.2f} | {r['noise_amp']:5.2f} | {r['min_conds']:5} | {r['sl']:3.0f} | {r['tp']:3.0f} | {r['cooldown']:2} | {r['total']:6} | {r['wr']:5.1f} | {r['edge']:+6.1f} | {r['pf']:5.2f} | {r['score']:7.1f}")

    # Melhor config
    best = results[0]
    print("\n" + "=" * 80)
    print("  MELHOR CONFIGURACAO")
    print("=" * 80)
    print(f"""
  Indicador:
    n_ensembles: {best['n_ens']}
    fractal_threshold: {best['frac_th']}
    noise_amplitude: {best['noise_amp']}
    min_conditions: {best['min_conds']}

  Trade:
    stop_loss_pips: {best['sl']}
    take_profit_pips: {best['tp']}
    cooldown_bars: {best['cooldown']}

  Performance:
    Trades: {best['total']}
    Win Rate: {best['wr']:.1f}%
    Edge: {best['edge']:+.1f}%
    Profit Factor: {best['pf']:.2f}
    Max Drawdown: {best['max_dd']:.1f} pips
    Score: {best['score']:.1f}
""")

    # Walk-forward da melhor config
    print("  Walk-forward (4 folds):")

    n = len(bars)
    folds_pos = 0
    tot_t, tot_w, tot_pnl = 0, 0, 0

    for fold in range(4):
        start_idx = int(n * fold / 4)
        end_idx = int(n * (fold + 1) / 4)
        fold_bars = bars[start_idx:end_idx]

        if len(fold_bars) < 100:
            print(f"    Q{fold+1}: Dados insuficientes")
            continue

        try:
            signals = precompute_signals_fast(fold_bars, best['n_ens'], best['frac_th'],
                                              best['noise_amp'], best['min_conds'], step=12)
            r = simulate_trades(fold_bars, signals, best['sl'], best['tp'], best['cooldown'])

            status = "OK" if r['edge'] > 0 and r['total'] > 0 else "FAIL"
            if r['edge'] > 0 and r['total'] > 0:
                folds_pos += 1

            tot_t += r['total']
            tot_w += r['wins']
            tot_pnl += r['pnl']

            print(f"    Q{fold+1}: {r['total']:3} trades | edge={r['edge']:+5.1f}% | PF={r['pf']:.2f} [{status}]")
        except Exception as e:
            print(f"    Q{fold+1}: Erro - {e}")

    print(f"\n  Folds positivos: {folds_pos}/4")

    if tot_t > 0:
        be = best['sl'] / (best['sl'] + best['tp']) * 100
        agg_wr = tot_w / tot_t * 100
        agg_edge = agg_wr - be
        gp = sum(r['pnl'] for r in results[:1] if r['pnl'] > 0)
        print(f"  Agregado WF: {tot_t} trades, WR={agg_wr:.1f}%, Edge={agg_edge:+.1f}%")

    # Salvar config
    config = {
        "strategy": "RHHF-FAST-OPTIMIZED",
        "symbol": SYMBOL,
        "periodicity": "H4",
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "combinations_tested": tested,
        "optimization_time_seconds": elapsed,
        "mode": "LONG_ONLY",
        "parameters": {
            "n_ensembles": int(best['n_ens']),
            "fractal_threshold": float(best['frac_th']),
            "noise_amplitude": float(best['noise_amp']),
            "min_conditions": int(best['min_conds']),
            "stop_loss_pips": int(best['sl']),
            "take_profit_pips": int(best['tp']),
            "cooldown_bars": int(best['cooldown'])
        },
        "performance": {
            "trades": int(best['total']),
            "win_rate": float(best['wr']),
            "edge": float(best['edge']),
            "profit_factor": float(best['pf']),
            "max_drawdown_pips": float(best['max_dd']),
            "score": float(best['score'])
        },
        "walk_forward": {
            "folds_positive": folds_pos,
            "total_folds": 4
        },
        "top_20": [
            {
                "rank": i + 1,
                "n_ens": int(r['n_ens']),
                "frac_th": float(r['frac_th']),
                "noise_amp": float(r['noise_amp']),
                "min_conds": int(r['min_conds']),
                "sl": int(r['sl']),
                "tp": int(r['tp']),
                "cooldown": int(r['cooldown']),
                "trades": int(r['total']),
                "edge": float(r['edge']),
                "pf": float(r['pf']),
                "score": float(r['score'])
            }
            for i, r in enumerate(results[:20])
        ]
    }

    path = "/home/azureuser/EliBotCD/configs/rhhf_fast_optimized.json"
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config salva: {path}")

    # Criterios
    print("\n" + "=" * 80)
    print("  CRITERIOS DE APROVACAO")
    print("=" * 80)

    criteria = [
        ("Edge > 0%", best['edge'] > 0),
        ("PF > 1.0", best['pf'] > 1.0),
        ("Folds >= 2/4", folds_pos >= 2),
        ("DD < 400 pips", best['max_dd'] < 400),
        ("Trades >= 10", best['total'] >= 10)
    ]

    passed = sum(1 for _, v in criteria if v)
    for name, ok in criteria:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Resultado: {passed}/5")

    if passed >= 4:
        print("\n  *** APROVADO ***")
    elif passed >= 3:
        print("\n  *** APROVADO COM RESSALVAS ***")
    else:
        print("\n  *** REPROVADO ***")

    print(f"\n  Tempo total: {elapsed/60:.1f} minutos")

if __name__ == "__main__":
    main()
