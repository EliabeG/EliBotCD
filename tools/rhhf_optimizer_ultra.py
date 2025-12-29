#!/usr/bin/env python3
"""
RHHF Ultra Optimizer - 100.000 combinacoes
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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
                time.sleep(0.05)
        except:
            break
    bars.sort(key=lambda x: x.timestamp)
    return bars

# Pré-calcular sinais uma vez e reutilizar para diferentes SL/TP
def precompute_signals(bars: List[Bar], n_ens: int, frac_th: float,
                       noise_amp: float, min_conds: int, step: int = 8) -> List[Tuple[int, int]]:
    """Pre-computa sinais para uma config de indicador"""
    from strategies.alta_volatilidade.rhhf_ressonador_hilbert_huang import RessonadorHilbertHuangFractal

    closes = np.array([b.close for b in bars])
    rhhf = RessonadorHilbertHuangFractal(
        n_ensembles=n_ens,
        noise_amplitude=noise_amp,
        fractal_threshold=frac_th,
        mirror_extension=30
    )

    signals = []
    for i in range(120, len(closes) - 1, step):
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
    """Simula trades com SL/TP específicos"""
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

        for j in range(idx + 2, min(idx + 60, n)):
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

def evaluate_config(args):
    """Avalia uma configuração completa"""
    bars, n_ens, frac_th, noise_amp, min_conds, sl, tp, cooldown, step = args

    try:
        signals = precompute_signals(bars, n_ens, frac_th, noise_amp, min_conds, step)
        result = simulate_trades(bars, signals, sl, tp, cooldown, direction=1)

        # Score composto
        if result['total'] < 5:
            score = -1000
        else:
            score = result['edge'] * np.sqrt(result['total'])
            if result['pf'] < 1.0:
                score -= 20
            if result['max_dd'] > 300:
                score -= 10

        return {
            'n_ens': n_ens,
            'frac_th': frac_th,
            'noise_amp': noise_amp,
            'min_conds': min_conds,
            'sl': sl,
            'tp': tp,
            'cooldown': cooldown,
            'score': score,
            **result
        }
    except Exception as e:
        return {'score': -9999, 'error': str(e)}

def main():
    print("=" * 80)
    print("  RHHF ULTRA OPTIMIZER - 100.000 COMBINACOES")
    print("=" * 80)

    # Baixar dados
    print("\n[1/4] Baixando dados H4...", end=" ", flush=True)
    bars = download_bars("H4", 2500)
    print(f"{len(bars)} barras")

    if len(bars) < 500:
        print("  Dados insuficientes!")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()} ({days} dias)")

    # Definir grid de parametros
    print("\n[2/4] Gerando combinacoes...")

    # Parametros do indicador
    n_ensembles_list = [5, 8, 10, 12, 15]
    frac_threshold_list = [1.25, 1.30, 1.35, 1.40, 1.45, 1.50]
    noise_amplitude_list = [0.1, 0.15, 0.2, 0.25, 0.3]
    min_conditions_list = [1, 2, 3]

    # Parametros de trade
    sl_list = [20, 25, 30, 35, 40, 45, 50, 60]
    tp_list = [40, 50, 60, 70, 80, 90, 100, 120]
    cooldown_list = [2, 3, 4, 5, 6]

    # Gerar todas as combinacoes
    all_combos = list(itertools.product(
        n_ensembles_list,
        frac_threshold_list,
        noise_amplitude_list,
        min_conditions_list,
        sl_list,
        tp_list,
        cooldown_list
    ))

    total_combos = len(all_combos)
    print(f"  Total de combinacoes: {total_combos:,}")

    # Limitar a 100.000 se necessario (amostragem aleatoria)
    if total_combos > 100000:
        np.random.seed(42)
        indices = np.random.choice(total_combos, 100000, replace=False)
        all_combos = [all_combos[i] for i in indices]
        print(f"  Amostrado para: 100,000 combinacoes")

    # Estrategia: pre-computar sinais para cada config de indicador
    # e depois testar rapidamente diferentes SL/TP/cooldown
    print("\n[3/4] Otimizando (estrategia de cache)...")

    # Agrupar por config de indicador
    indicator_configs = set()
    for combo in all_combos:
        n_ens, frac_th, noise_amp, min_conds, _, _, _ = combo
        indicator_configs.add((n_ens, frac_th, noise_amp, min_conds))

    print(f"  Configs de indicador unicas: {len(indicator_configs)}")

    # Pre-computar sinais para cada config de indicador
    print("  Pre-computando sinais...", end=" ", flush=True)
    signals_cache = {}
    step = 8  # Passo de amostragem

    for i, (n_ens, frac_th, noise_amp, min_conds) in enumerate(indicator_configs):
        key = (n_ens, frac_th, noise_amp, min_conds)
        try:
            signals_cache[key] = precompute_signals(bars, n_ens, frac_th, noise_amp, min_conds, step)
        except:
            signals_cache[key] = []

        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(indicator_configs)}", end=" ", flush=True)

    print("OK")

    # Testar todas as combinacoes de SL/TP/cooldown
    print("  Testando combinacoes de trade...", end=" ", flush=True)

    results = []
    tested = 0

    for combo in all_combos:
        n_ens, frac_th, noise_amp, min_conds, sl, tp, cooldown = combo
        key = (n_ens, frac_th, noise_amp, min_conds)

        signals = signals_cache.get(key, [])
        if not signals:
            continue

        r = simulate_trades(bars, signals, sl, tp, cooldown, direction=1)

        if r['total'] >= 5:
            score = r['edge'] * np.sqrt(r['total'])
            if r['pf'] < 1.0:
                score -= 20
            if r['max_dd'] > 300:
                score -= 10

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
        if tested % 10000 == 0:
            print(f"{tested:,}", end=" ", flush=True)

    print(f"OK ({tested:,} testadas)")

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

    # Top 20
    print("\n  TOP 20 CONFIGURACOES:")
    print("-" * 100)
    print(f"  {'#':>3} | {'Ens':>3} | {'FracTh':>6} | {'Noise':>5} | {'Conds':>5} | {'SL':>3} | {'TP':>3} | {'CD':>2} | {'Trades':>6} | {'WR%':>5} | {'Edge%':>6} | {'PF':>5} | {'Score':>7}")
    print("-" * 100)

    for i, r in enumerate(results[:20]):
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

    key = (best['n_ens'], best['frac_th'], best['noise_amp'], best['min_conds'])
    n = len(bars)
    folds_pos = 0

    for fold in range(4):
        start_idx = int(n * fold / 4)
        end_idx = int(n * (fold + 1) / 4)
        fold_bars = bars[start_idx:end_idx]

        if len(fold_bars) < 150:
            print(f"    Q{fold+1}: Dados insuficientes")
            continue

        try:
            signals = precompute_signals(fold_bars, best['n_ens'], best['frac_th'],
                                        best['noise_amp'], best['min_conds'], step=8)
            r = simulate_trades(fold_bars, signals, best['sl'], best['tp'], best['cooldown'])

            status = "OK" if r['edge'] > 0 and r['total'] > 0 else "FAIL"
            if r['edge'] > 0 and r['total'] > 0:
                folds_pos += 1

            print(f"    Q{fold+1}: {r['total']:3} trades | edge={r['edge']:+5.1f}% | PF={r['pf']:.2f} [{status}]")
        except Exception as e:
            print(f"    Q{fold+1}: Erro - {e}")

    print(f"\n  Folds positivos: {folds_pos}/4")

    # Salvar config
    config = {
        "strategy": "RHHF-ULTRA-OPTIMIZED",
        "symbol": SYMBOL,
        "periodicity": "H4",
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "combinations_tested": tested,
        "mode": "LONG_ONLY",
        "parameters": {
            "n_ensembles": best['n_ens'],
            "fractal_threshold": best['frac_th'],
            "noise_amplitude": best['noise_amp'],
            "min_conditions": best['min_conds'],
            "stop_loss_pips": best['sl'],
            "take_profit_pips": best['tp'],
            "cooldown_bars": best['cooldown']
        },
        "performance": {
            "trades": best['total'],
            "win_rate": best['wr'],
            "edge": best['edge'],
            "profit_factor": best['pf'],
            "max_drawdown_pips": best['max_dd'],
            "score": best['score']
        },
        "walk_forward": {
            "folds_positive": folds_pos,
            "total_folds": 4
        },
        "top_10": [
            {
                "rank": i + 1,
                "n_ens": r['n_ens'],
                "frac_th": r['frac_th'],
                "noise_amp": r['noise_amp'],
                "min_conds": r['min_conds'],
                "sl": r['sl'],
                "tp": r['tp'],
                "cooldown": r['cooldown'],
                "trades": r['total'],
                "edge": r['edge'],
                "pf": r['pf'],
                "score": r['score']
            }
            for i, r in enumerate(results[:10])
        ]
    }

    path = "/home/azureuser/EliBotCD/configs/rhhf_ultra_optimized.json"
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
        print("  *** APROVADO ***")
    elif passed >= 3:
        print("  *** APROVADO COM RESSALVAS ***")
    else:
        print("  *** REPROVADO ***")

if __name__ == "__main__":
    main()
