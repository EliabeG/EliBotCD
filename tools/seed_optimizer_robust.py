#!/usr/bin/env python3
"""
SEED Robust Optimizer - Anti-Overfitting
==========================================
Sintetizador Evolutivo de Estruturas Dissipativas

Tecnicas aplicadas:
1. Minimo 50 trades para validar
2. Walk-forward anchored (5 folds)
3. Monte Carlo shuffle para validar robustez
4. Penalizacao por complexidade
5. Out-of-sample final (20% dados nunca vistos)
6. Custos conservadores (spread + slippage)
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

# Anti-overfitting settings
MIN_TRADES = 50
MIN_TRADES_PER_FOLD = 10
OOS_RATIO = 0.20
MONTE_CARLO_RUNS = 100

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

def compute_signals(bars: List[Bar], critical_sigma: float, slaving_threshold: float,
                   dominance_threshold: float, fitness_window: int,
                   dx_threshold: float, min_conditions: int) -> List[Tuple[int, int]]:
    """Computa sinais SEED para toda a serie"""
    from strategies.alta_volatilidade.seed_sintetizador_evolutivo import SintetizadorEvolutivoEstruturasDissipativas

    closes = np.array([b.close for b in bars])
    volumes = np.array([b.volume if b.volume > 0 else abs(closes[i] - closes[max(0,i-1)]) * 50000
                       for i, b in enumerate(bars)])

    seed = SintetizadorEvolutivoEstruturasDissipativas(
        critical_sigma=critical_sigma,
        slaving_threshold=slaving_threshold,
        dominance_threshold=dominance_threshold,
        fitness_window=fitness_window,
        dx_threshold=dx_threshold
    )

    signals = []
    # Passo de 4 barras para H1 (mais granular)
    for i in range(50, len(closes) - 1, 4):
        try:
            result = seed.analyze(closes[:i], volumes[:i])
            sig = result['signal']

            # Contar condicoes atendidas
            conds = 0
            if result['is_nonlinear_regime']:
                conds += 1
            if result['slaving_active']:
                conds += 1
            if result['structure_forming']:
                conds += 1
            if result.get('bulls_growing_exp') or result.get('bears_growing_exp'):
                conds += 1
            if result.get('is_ignition_point'):
                conds += 1

            if conds >= min_conditions and sig != 0:
                signals.append((i, sig))
        except:
            continue

    return signals

def simulate_trades(bars: List[Bar], signals: List[Tuple[int, int]],
                   sl: float, tp: float, cooldown: int,
                   start_idx: int = 0, end_idx: int = None,
                   direction: int = 1) -> Dict:
    """Simula trades em uma janela especifica"""
    if end_idx is None:
        end_idx = len(bars)

    n = len(bars)
    trades = []
    trade_details = []
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

        if direction == 1:  # LONG
            sl_price = entry - sl * PIP_VALUE
            tp_price = entry + tp * PIP_VALUE
        else:  # SHORT
            sl_price = entry + sl * PIP_VALUE
            tp_price = entry - tp * PIP_VALUE

        result = None
        exit_idx = None

        for j in range(idx + 2, min(idx + 100, n)):
            bar = bars[j]
            if direction == 1:
                if bar.low <= sl_price:
                    pnl = -sl - SPREAD_PIPS - SLIPPAGE_PIPS
                    result = 'loss'
                    exit_idx = j
                    break
                if bar.high >= tp_price:
                    pnl = tp - SPREAD_PIPS - SLIPPAGE_PIPS
                    result = 'win'
                    exit_idx = j
                    break
            else:
                if bar.high >= sl_price:
                    pnl = -sl - SPREAD_PIPS - SLIPPAGE_PIPS
                    result = 'loss'
                    exit_idx = j
                    break
                if bar.low <= tp_price:
                    pnl = tp - SPREAD_PIPS - SLIPPAGE_PIPS
                    result = 'win'
                    exit_idx = j
                    break

        if result:
            trades.append(pnl)
            trade_details.append({
                'entry_idx': idx,
                'exit_idx': exit_idx,
                'pnl': pnl,
                'result': result
            })

        last_idx = idx

    if not trades:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'trades': [], 'details': []}

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
        'trades': trades,
        'details': trade_details
    }

def anchored_walk_forward(bars: List[Bar], signals: List[Tuple[int, int]],
                          sl: float, tp: float, cooldown: int,
                          direction: int = 1, n_folds: int = 5) -> Dict:
    """Anchored Walk-Forward validation"""
    n = len(bars)
    fold_size = n // (n_folds + 1)

    results = []
    all_oos_trades = []

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = test_start + fold_size

        if test_end > n:
            break

        r = simulate_trades(bars, signals, sl, tp, cooldown, test_start, test_end, direction)

        if r['total'] >= MIN_TRADES_PER_FOLD:
            results.append({
                'fold': fold + 1,
                'trades': r['total'],
                'edge': r['edge'],
                'pf': r['pf'],
                'pnl': r['pnl']
            })
            all_oos_trades.extend(r['trades'])

    if not results:
        return {'valid': False, 'folds_positive': 0, 'total_folds': n_folds}

    folds_positive = sum(1 for r in results if r['edge'] > 0)

    if all_oos_trades:
        wins = len([t for t in all_oos_trades if t > 0])
        total = len(all_oos_trades)
        wr = wins / total * 100
        be = sl / (sl + tp) * 100
        gp = sum(t for t in all_oos_trades if t > 0)
        gl = abs(sum(t for t in all_oos_trades if t < 0))

        return {
            'valid': True,
            'folds_positive': folds_positive,
            'total_folds': len(results),
            'oos_trades': total,
            'oos_wr': wr,
            'oos_edge': wr - be,
            'oos_pf': gp / gl if gl > 0 else 0,
            'oos_pnl': sum(all_oos_trades),
            'details': results
        }

    return {'valid': False, 'folds_positive': 0, 'total_folds': n_folds}

def monte_carlo_validation(trades: List[float], n_runs: int = 100) -> Dict:
    """Monte Carlo shuffle validation"""
    if len(trades) < 10:
        return {'valid': False, 'stability': 0}

    drawdowns = []
    for _ in range(n_runs):
        shuffled = trades.copy()
        np.random.shuffle(shuffled)
        cumsum = np.cumsum(shuffled)
        dd = np.max(np.maximum.accumulate(cumsum) - cumsum)
        drawdowns.append(dd)

    dd_std = np.std(drawdowns)
    dd_mean = np.mean(drawdowns)
    stability = 1.0 - (dd_std / (dd_mean + 0.01))

    return {
        'valid': True,
        'stability': max(0, stability),
        'dd_mean': dd_mean,
        'dd_std': dd_std,
        'dd_95th': np.percentile(drawdowns, 95)
    }

def calculate_complexity_penalty(critical_sigma: float, slaving_threshold: float,
                                 dominance_threshold: float, min_conditions: int) -> float:
    """Penaliza configuracoes mais complexas"""
    penalty = 0

    # critical_sigma: valores extremos penalizados
    if critical_sigma < 1.0 or critical_sigma > 2.5:
        penalty += 2

    # slaving_threshold: muito baixo ou muito alto
    if slaving_threshold < 0.10 or slaving_threshold > 0.25:
        penalty += 2

    # dominance_threshold: muito baixo
    if dominance_threshold < 0.4:
        penalty += 1

    # min_conditions: muito baixo e agressivo
    if min_conditions < 2:
        penalty += 3

    return penalty

def robust_score(result: Dict, wf: Dict, mc: Dict, complexity: float) -> float:
    """Score robusto anti-overfitting"""
    if not wf.get('valid') or wf.get('oos_trades', 0) < MIN_TRADES:
        return -1000

    oos_edge = wf.get('oos_edge', 0)
    oos_pf = wf.get('oos_pf', 0)
    pf_bonus = min(5, (oos_pf - 1) * 3) if oos_pf > 1 else (oos_pf - 1) * 10
    folds_ratio = wf.get('folds_positive', 0) / max(1, wf.get('total_folds', 1))
    fold_bonus = folds_ratio * 10
    trade_bonus = min(10, wf.get('oos_trades', 0) / 10)
    stability_bonus = mc.get('stability', 0) * 5
    complexity_penalty = complexity * 2

    score = (
        oos_edge * 2 +
        pf_bonus +
        fold_bonus +
        trade_bonus +
        stability_bonus -
        complexity_penalty
    )

    return score

def main():
    print("=" * 80)
    print("  SEED ROBUST OPTIMIZER - ANTI-OVERFITTING")
    print("  Sintetizador Evolutivo de Estruturas Dissipativas")
    print("=" * 80)
    print(f"""
  Tecnicas Anti-Overfitting:
  - Minimo {MIN_TRADES} trades para validar
  - Anchored Walk-Forward (5 folds)
  - Monte Carlo Validation ({MONTE_CARLO_RUNS} shuffles)
  - Penalizacao por complexidade
  - Out-of-sample final ({int(OOS_RATIO*100)}%)
  - Custos conservadores: {SPREAD_PIPS + SLIPPAGE_PIPS} pips
""")

    start_time = time.time()

    # Baixar dados H1 (mais dados que H4)
    print("[1/5] Baixando dados H1...", end=" ", flush=True)
    all_bars = download_bars("H1", 4000)
    print(f"{len(all_bars)} barras")

    if len(all_bars) < 1500:
        print("  Dados insuficientes!")
        return

    # Separar dados: 80% otimizacao, 20% final OOS
    oos_split = int(len(all_bars) * (1 - OOS_RATIO))
    opt_bars = all_bars[:oos_split]
    final_oos_bars = all_bars[oos_split:]

    days = (all_bars[-1].timestamp - all_bars[0].timestamp).days
    print(f"  Periodo total: {all_bars[0].timestamp.date()} a {all_bars[-1].timestamp.date()} ({days} dias)")
    print(f"  Otimizacao: {len(opt_bars)} barras | Final OOS: {len(final_oos_bars)} barras")

    # Grid de parametros
    print("\n[2/5] Definindo grid de parametros...")

    # Parametros do indicador SEED
    critical_sigma_list = [1.2, 1.5, 1.8, 2.0]
    slaving_threshold_list = [0.12, 0.15, 0.18, 0.20]
    dominance_threshold_list = [0.45, 0.50, 0.55]
    fitness_window_list = [15, 20, 25]
    dx_threshold_list = [0.02]  # Fixo
    min_conditions_list = [2, 3]

    # Parametros de trade
    sl_list = [30, 35, 40, 45]
    tp_list = [45, 50, 60, 70]
    cooldown_list = [6, 8, 10]

    indicator_combos = list(itertools.product(
        critical_sigma_list,
        slaving_threshold_list,
        dominance_threshold_list,
        fitness_window_list,
        dx_threshold_list,
        min_conditions_list
    ))

    trade_combos = list(itertools.product(sl_list, tp_list, cooldown_list))

    total_combos = len(indicator_combos) * len(trade_combos)
    print(f"  Configs de indicador: {len(indicator_combos)}")
    print(f"  Configs de trade: {len(trade_combos)}")
    print(f"  Total: {total_combos} combinacoes")

    # Pre-computar sinais
    print("\n[3/5] Pre-computando sinais...")

    signals_cache = {}

    for i, (crit_sig, slav_th, dom_th, fit_win, dx_th, min_cond) in enumerate(indicator_combos):
        key = (crit_sig, slav_th, dom_th, fit_win, dx_th, min_cond)
        print(f"  {i+1}/{len(indicator_combos)} - sigma={crit_sig}, slav={slav_th}, dom={dom_th}, conds={min_cond}...", end=" ", flush=True)

        try:
            t0 = time.time()
            signals_cache[key] = compute_signals(opt_bars, crit_sig, slav_th, dom_th, fit_win, dx_th, min_cond)
            elapsed = time.time() - t0
            print(f"{len(signals_cache[key])} sinais ({elapsed:.1f}s)")
        except Exception as e:
            print(f"Erro: {e}")
            signals_cache[key] = []

    # Testar combinacoes
    print("\n[4/5] Validacao robusta (Walk-Forward + Monte Carlo)...")

    results = []
    tested = 0

    for key, signals in signals_cache.items():
        if len(signals) < 20:
            continue

        crit_sig, slav_th, dom_th, fit_win, dx_th, min_cond = key
        complexity = calculate_complexity_penalty(crit_sig, slav_th, dom_th, min_cond)

        for sl, tp, cooldown in trade_combos:
            # Simular in-sample (LONG only)
            r = simulate_trades(opt_bars, signals, sl, tp, cooldown, direction=1)

            if r['total'] < MIN_TRADES:
                tested += 1
                continue

            # Walk-forward
            wf = anchored_walk_forward(opt_bars, signals, sl, tp, cooldown, direction=1, n_folds=5)

            if not wf.get('valid') or wf.get('oos_trades', 0) < MIN_TRADES:
                tested += 1
                continue

            # Monte Carlo
            mc = monte_carlo_validation(r['trades'], MONTE_CARLO_RUNS)

            # Score robusto
            score = robust_score(r, wf, mc, complexity)

            if score > 0:
                results.append({
                    'crit_sig': crit_sig,
                    'slav_th': slav_th,
                    'dom_th': dom_th,
                    'fit_win': fit_win,
                    'dx_th': dx_th,
                    'min_cond': min_cond,
                    'sl': sl,
                    'tp': tp,
                    'cooldown': cooldown,
                    'is_trades': r['total'],
                    'is_wr': r['wr'],
                    'is_edge': r['edge'],
                    'is_pf': r['pf'],
                    'oos_trades': wf['oos_trades'],
                    'oos_wr': wf['oos_wr'],
                    'oos_edge': wf['oos_edge'],
                    'oos_pf': wf['oos_pf'],
                    'folds_pos': wf['folds_positive'],
                    'folds_total': wf['total_folds'],
                    'mc_stability': mc.get('stability', 0),
                    'complexity': complexity,
                    'score': score
                })

            tested += 1

            if tested % 50 == 0:
                print(f"  {tested}/{total_combos} testadas, {len(results)} validas...")

    elapsed = time.time() - start_time
    print(f"\n  Total: {tested} testadas, {len(results)} validas em {elapsed:.1f}s")

    if not results:
        print("\n  NENHUMA CONFIGURACAO PASSOU NOS CRITERIOS!")
        print("  O indicador SEED pode nao ter edge real no timeframe H1.")

        # Tentar com criterios mais relaxados
        print("\n  Tentando com criterios relaxados (30 trades minimo)...")
        MIN_TRADES_RELAXED = 30

        for key, signals in signals_cache.items():
            if len(signals) < 15:
                continue
            crit_sig, slav_th, dom_th, fit_win, dx_th, min_cond = key

            for sl, tp, cooldown in trade_combos:
                r = simulate_trades(opt_bars, signals, sl, tp, cooldown, direction=1)
                if r['total'] >= MIN_TRADES_RELAXED and r['edge'] > 0 and r['pf'] > 1.0:
                    print(f"    Encontrado: sigma={crit_sig}, {r['total']} trades, edge={r['edge']:.1f}%, PF={r['pf']:.2f}")

        return

    # Ordenar por score
    results.sort(key=lambda x: x['score'], reverse=True)

    # Mostrar top 15
    print("\n" + "=" * 80)
    print("  TOP 15 CONFIGURACOES ROBUSTAS")
    print("=" * 80)
    print(f"\n  {'#':>2} | {'Sig':>3} | {'Slv':>4} | {'Dom':>4} | {'Cnd':>3} | {'SL':>3} | {'TP':>3} | {'CD':>2} | {'IS_T':>4} | {'IS_E':>5} | {'OOS_T':>5} | {'OOS_E':>5} | {'OOS_PF':>5} | {'Folds':>5} | {'Score':>6}")
    print("-" * 110)

    for i, r in enumerate(results[:15]):
        folds_str = f"{r['folds_pos']}/{r['folds_total']}"
        print(f"  {i+1:2} | {r['crit_sig']:.1f} | {r['slav_th']:.2f} | {r['dom_th']:.2f} | {r['min_cond']:3} | {r['sl']:3.0f} | {r['tp']:3.0f} | {r['cooldown']:2} | {r['is_trades']:4} | {r['is_edge']:+5.1f} | {r['oos_trades']:5} | {r['oos_edge']:+5.1f} | {r['oos_pf']:5.2f} | {folds_str:>5} | {r['score']:6.1f}")

    # Melhor config
    best = results[0]

    # Validacao final OOS
    print("\n" + "=" * 80)
    print("  [5/5] VALIDACAO FINAL OUT-OF-SAMPLE (dados nunca vistos)")
    print("=" * 80)

    # Recomputar sinais para dados completos
    final_signals = compute_signals(all_bars, best['crit_sig'], best['slav_th'],
                                    best['dom_th'], best['fit_win'], best['dx_th'], best['min_cond'])

    # Testar apenas no periodo final OOS
    final_r = simulate_trades(all_bars, final_signals, best['sl'], best['tp'],
                              best['cooldown'], oos_split, len(all_bars), direction=1)

    print(f"""
  Periodo Final OOS: {final_oos_bars[0].timestamp.date()} a {final_oos_bars[-1].timestamp.date()}

  Resultados:
    Trades: {final_r['total']}
    Win Rate: {final_r['wr']:.1f}%
    Edge: {final_r['edge']:+.1f}%
    Profit Factor: {final_r['pf']:.2f}
    PnL: {final_r['pnl']:.1f} pips
""")

    final_passed = (
        final_r['total'] >= 5 and
        final_r['edge'] > 0 and
        final_r['pf'] > 1.0
    )

    if final_passed:
        print("  *** VALIDACAO FINAL: APROVADO ***")
    else:
        print("  *** VALIDACAO FINAL: REPROVADO ***")

    # Resumo
    print("\n" + "=" * 80)
    print("  CONFIGURACAO ROBUSTA FINAL")
    print("=" * 80)
    print(f"""
  Indicador SEED:
    critical_sigma: {best['crit_sig']}
    slaving_threshold: {best['slav_th']}
    dominance_threshold: {best['dom_th']}
    fitness_window: {best['fit_win']}
    dx_threshold: {best['dx_th']}
    min_conditions: {best['min_cond']}

  Trade:
    stop_loss_pips: {best['sl']}
    take_profit_pips: {best['tp']}
    cooldown_bars: {best['cooldown']}

  Performance In-Sample:
    Trades: {best['is_trades']}
    Edge: {best['is_edge']:+.1f}%
    PF: {best['is_pf']:.2f}

  Performance Out-of-Sample (Walk-Forward):
    Trades: {best['oos_trades']}
    Edge: {best['oos_edge']:+.1f}%
    PF: {best['oos_pf']:.2f}
    Folds Positivos: {best['folds_pos']}/{best['folds_total']}

  Performance Final OOS (nunca visto):
    Trades: {final_r['total']}
    Edge: {final_r['edge']:+.1f}%
    PF: {final_r['pf']:.2f}

  Metricas Anti-Overfitting:
    Monte Carlo Stability: {best['mc_stability']:.2f}
    Complexity Penalty: {best['complexity']}
    Robust Score: {best['score']:.1f}
""")

    # Expectativa realista
    expected_edge = (
        best['is_edge'] * 0.2 +
        best['oos_edge'] * 0.5 +
        final_r['edge'] * 0.3
    )

    print(f"  EXPECTATIVA REALISTA DE EDGE: {expected_edge:+.1f}%")

    # Salvar config
    config = {
        "strategy": "SEED-ROBUST-ANTI-OVERFITTING",
        "symbol": SYMBOL,
        "periodicity": "H1",
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "mode": "LONG_ONLY",
        "anti_overfitting": {
            "min_trades": MIN_TRADES,
            "walk_forward_folds": 5,
            "monte_carlo_runs": MONTE_CARLO_RUNS,
            "oos_ratio": OOS_RATIO,
            "spread_pips": SPREAD_PIPS,
            "slippage_pips": SLIPPAGE_PIPS
        },
        "parameters": {
            "critical_sigma": float(best['crit_sig']),
            "slaving_threshold": float(best['slav_th']),
            "dominance_threshold": float(best['dom_th']),
            "fitness_window": int(best['fit_win']),
            "dx_threshold": float(best['dx_th']),
            "min_conditions": int(best['min_cond']),
            "stop_loss_pips": int(best['sl']),
            "take_profit_pips": int(best['tp']),
            "cooldown_bars": int(best['cooldown'])
        },
        "performance": {
            "in_sample": {
                "trades": int(best['is_trades']),
                "edge": float(best['is_edge']),
                "profit_factor": float(best['is_pf'])
            },
            "out_of_sample_wf": {
                "trades": int(best['oos_trades']),
                "edge": float(best['oos_edge']),
                "profit_factor": float(best['oos_pf']),
                "folds_positive": best['folds_pos'],
                "folds_total": best['folds_total']
            },
            "final_oos": {
                "trades": int(final_r['total']),
                "edge": float(final_r['edge']),
                "profit_factor": float(final_r['pf'])
            },
            "expected_edge": float(expected_edge),
            "monte_carlo_stability": float(best['mc_stability']),
            "robust_score": float(best['score'])
        },
        "validation": {
            "final_oos_passed": final_passed,
            "overfitting_risk": "LOW" if final_passed and expected_edge > 3 else "MODERATE"
        }
    }

    path = "/home/azureuser/EliBotCD/configs/seed_robust_optimized.json"
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config salva: {path}")

    # Criterios finais
    print("\n" + "=" * 80)
    print("  CRITERIOS ANTI-OVERFITTING")
    print("=" * 80)

    criteria = [
        ("OOS Edge > 0%", best['oos_edge'] > 0),
        ("OOS PF > 1.0", best['oos_pf'] > 1.0),
        ("OOS Trades >= 50", best['oos_trades'] >= 50),
        ("Folds >= 3/5", best['folds_pos'] >= 3),
        ("Final OOS Edge > 0%", final_r['edge'] > 0),
        ("Final OOS Trades >= 5", final_r['total'] >= 5),
        ("MC Stability > 0.5", best['mc_stability'] > 0.5),
        ("Expected Edge > 3%", expected_edge > 3)
    ]

    passed = sum(1 for _, v in criteria if v)
    for name, ok in criteria:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Resultado: {passed}/{len(criteria)}")

    if passed >= 7:
        print("\n  *** APROVADO - BAIXO RISCO DE OVERFITTING ***")
    elif passed >= 5:
        print("\n  *** APROVADO COM RESSALVAS - RISCO MODERADO ***")
    else:
        print("\n  *** REPROVADO - ALTO RISCO DE OVERFITTING ***")

    print(f"\n  Tempo total: {(time.time() - start_time)/60:.1f} minutos")

if __name__ == "__main__":
    main()
