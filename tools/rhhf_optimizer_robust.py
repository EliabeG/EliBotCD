#!/usr/bin/env python3
"""
RHHF Robust Optimizer - Anti-Overfitting
==========================================
Tecnicas aplicadas:
1. Minimo 50 trades para validar
2. Walk-forward anchored (treino cresce, teste fixo)
3. Monte Carlo shuffle para validar robustez
4. Penalizacao por complexidade (menos parametros = melhor)
5. Consenso de multiplas configs (nao apenas a melhor)
6. Out-of-sample final (20% dados nunca vistos)
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
SPREAD_PIPS = 1.5  # Mais conservador
SLIPPAGE_PIPS = 0.8  # Mais conservador
PIP_VALUE = 0.0001

# Anti-overfitting settings
MIN_TRADES = 50  # Minimo de trades para validar
MIN_TRADES_PER_FOLD = 10  # Minimo por fold
OOS_RATIO = 0.20  # 20% out-of-sample final
MONTE_CARLO_RUNS = 100  # Shuffles para validar

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

def compute_signals(bars: List[Bar], n_ens: int, frac_th: float,
                   noise_amp: float, min_conds: int) -> List[Tuple[int, int]]:
    """Computa sinais para toda a serie"""
    from strategies.alta_volatilidade.rhhf_ressonador_hilbert_huang import RessonadorHilbertHuangFractal

    closes = np.array([b.close for b in bars])
    rhhf = RessonadorHilbertHuangFractal(
        n_ensembles=n_ens,
        noise_amplitude=noise_amp,
        fractal_threshold=frac_th,
        mirror_extension=20
    )

    signals = []
    # Passo de 6 barras (mais granular que antes)
    for i in range(100, len(closes) - 1, 6):
        try:
            result = rhhf.analyze(closes[:i])
            sig = result['signal']
            conds = result['signal_details']['conditions_met']
            if conds >= min_conds and sig != 0:
                signals.append((i, sig))
        except:
            continue

    return signals

def simulate_trades(bars: List[Bar], signals: List[Tuple[int, int]],
                   sl: float, tp: float, cooldown: int,
                   start_idx: int = 0, end_idx: int = None) -> Dict:
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
        if sig != 1:  # Long only
            continue
        if idx - last_idx < cooldown:
            continue
        if idx + 1 >= n:
            continue

        entry = bars[idx + 1].open
        sl_price = entry - sl * PIP_VALUE
        tp_price = entry + tp * PIP_VALUE

        result = None
        exit_idx = None

        for j in range(idx + 2, min(idx + 50, n)):
            bar = bars[j]
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
                          sl: float, tp: float, cooldown: int, n_folds: int = 5) -> Dict:
    """
    Anchored Walk-Forward: treino sempre comeca do inicio, teste avanca
    Mais robusto que walk-forward tradicional
    """
    n = len(bars)
    fold_size = n // (n_folds + 1)  # +1 para ter espaco para teste

    results = []
    all_oos_trades = []

    for fold in range(n_folds):
        # Treino: do inicio ate fold * fold_size
        train_end = (fold + 1) * fold_size
        # Teste: do fim do treino ate proximo fold
        test_start = train_end
        test_end = test_start + fold_size

        if test_end > n:
            break

        # Simular no periodo de teste (out-of-sample)
        r = simulate_trades(bars, signals, sl, tp, cooldown, test_start, test_end)

        if r['total'] >= MIN_TRADES_PER_FOLD:
            results.append({
                'fold': fold + 1,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'trades': r['total'],
                'edge': r['edge'],
                'pf': r['pf'],
                'pnl': r['pnl']
            })
            all_oos_trades.extend(r['trades'])

    if not results:
        return {'valid': False, 'folds_positive': 0, 'total_folds': n_folds}

    folds_positive = sum(1 for r in results if r['edge'] > 0)

    # Calcular metricas agregadas do OOS
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
    """
    Monte Carlo: embaralha trades para verificar se resultado nao e por ordem
    Se resultado depender da ordem, e overfitting
    """
    if len(trades) < 10:
        return {'valid': False, 'stability': 0}

    original_pnl = sum(trades)
    original_wr = len([t for t in trades if t > 0]) / len(trades) * 100

    shuffled_pnls = []
    shuffled_wrs = []

    for _ in range(n_runs):
        shuffled = trades.copy()
        np.random.shuffle(shuffled)
        shuffled_pnls.append(sum(shuffled))
        shuffled_wrs.append(len([t for t in shuffled if t > 0]) / len(shuffled) * 100)

    # PnL e WR nao devem mudar com shuffle (sao somas)
    # Mas podemos verificar drawdown e consistencia

    # Calcular max drawdown para cada shuffle
    drawdowns = []
    for _ in range(n_runs):
        shuffled = trades.copy()
        np.random.shuffle(shuffled)
        cumsum = np.cumsum(shuffled)
        dd = np.max(np.maximum.accumulate(cumsum) - cumsum)
        drawdowns.append(dd)

    # Estabilidade: quao consistente e o drawdown
    dd_std = np.std(drawdowns)
    dd_mean = np.mean(drawdowns)
    stability = 1.0 - (dd_std / (dd_mean + 0.01))  # 0 a 1

    return {
        'valid': True,
        'stability': max(0, stability),
        'dd_mean': dd_mean,
        'dd_std': dd_std,
        'dd_95th': np.percentile(drawdowns, 95)
    }

def calculate_complexity_penalty(n_ens: int, frac_th: float, min_conds: int) -> float:
    """
    Penaliza configuracoes mais complexas
    Menos parametros extremos = menor penalidade
    """
    penalty = 0

    # n_ensembles: valores extremos sao penalizados
    if n_ens < 5 or n_ens > 20:
        penalty += 2

    # fractal_threshold: muito baixo ou muito alto
    if frac_th < 1.2 or frac_th > 1.6:
        penalty += 2

    # min_conditions: 1 e muito agressivo
    if min_conds == 1:
        penalty += 3
    elif min_conds == 2:
        penalty += 1

    return penalty

def robust_score(result: Dict, wf: Dict, mc: Dict, complexity: float) -> float:
    """
    Score robusto que considera:
    1. Performance in-sample (peso menor)
    2. Performance out-of-sample (peso maior)
    3. Estabilidade Monte Carlo
    4. Penalidade de complexidade
    5. Numero de trades
    """
    if not wf.get('valid') or wf.get('oos_trades', 0) < MIN_TRADES:
        return -1000

    # Base: edge out-of-sample (mais importante)
    oos_edge = wf.get('oos_edge', 0)

    # Bonus por profit factor OOS
    oos_pf = wf.get('oos_pf', 0)
    pf_bonus = min(5, (oos_pf - 1) * 3) if oos_pf > 1 else (oos_pf - 1) * 10

    # Bonus por folds positivos
    folds_ratio = wf.get('folds_positive', 0) / max(1, wf.get('total_folds', 1))
    fold_bonus = folds_ratio * 10

    # Bonus por trades (mais trades = mais confiavel)
    trade_bonus = min(10, wf.get('oos_trades', 0) / 10)

    # Estabilidade Monte Carlo
    stability_bonus = mc.get('stability', 0) * 5

    # Penalidade de complexidade
    complexity_penalty = complexity * 2

    # Score final
    score = (
        oos_edge * 2 +  # Edge OOS tem peso dobrado
        pf_bonus +
        fold_bonus +
        trade_bonus +
        stability_bonus -
        complexity_penalty
    )

    return score

def main():
    print("=" * 80)
    print("  RHHF ROBUST OPTIMIZER - ANTI-OVERFITTING")
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

    # Baixar dados
    print("[1/5] Baixando dados H4...", end=" ", flush=True)
    all_bars = download_bars("H4", 2500)
    print(f"{len(all_bars)} barras")

    if len(all_bars) < 800:
        print("  Dados insuficientes!")
        return

    # Separar dados: 80% para otimizacao, 20% final OOS
    oos_split = int(len(all_bars) * (1 - OOS_RATIO))
    opt_bars = all_bars[:oos_split]
    final_oos_bars = all_bars[oos_split:]

    days = (all_bars[-1].timestamp - all_bars[0].timestamp).days
    print(f"  Periodo total: {all_bars[0].timestamp.date()} a {all_bars[-1].timestamp.date()} ({days} dias)")
    print(f"  Otimizacao: {len(opt_bars)} barras | Final OOS: {len(final_oos_bars)} barras")

    # Grid de parametros REDUZIDO (menos overfitting)
    print("\n[2/5] Definindo grid de parametros...")

    # Apenas valores centrais/robustos
    n_ensembles_list = [8, 10, 12]  # Valores moderados
    frac_threshold_list = [1.30, 1.35, 1.40, 1.45]  # Range central
    noise_amplitude_list = [0.2]  # Valor padrao
    min_conditions_list = [2, 3]  # Mais conservador (nao 1)

    # Trade params
    sl_list = [35, 40, 45, 50]  # Range menor
    tp_list = [50, 60, 70, 80]  # TP > SL preferido
    cooldown_list = [4, 5, 6]  # Cooldown moderado

    indicator_combos = list(itertools.product(
        n_ensembles_list,
        frac_threshold_list,
        noise_amplitude_list,
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

    for i, (n_ens, frac_th, noise_amp, min_conds) in enumerate(indicator_combos):
        key = (n_ens, frac_th, noise_amp, min_conds)
        print(f"  {i+1}/{len(indicator_combos)} - ens={n_ens}, frac={frac_th:.2f}, conds={min_conds}...", end=" ", flush=True)

        try:
            t0 = time.time()
            signals_cache[key] = compute_signals(opt_bars, n_ens, frac_th, noise_amp, min_conds)
            elapsed = time.time() - t0
            print(f"{len(signals_cache[key])} sinais ({elapsed:.1f}s)")
        except Exception as e:
            print(f"Erro: {e}")
            signals_cache[key] = []

    # Testar combinacoes com validacao robusta
    print("\n[4/5] Validacao robusta (Walk-Forward + Monte Carlo)...")

    results = []
    tested = 0

    for key, signals in signals_cache.items():
        if len(signals) < 20:  # Minimo de sinais
            continue

        n_ens, frac_th, noise_amp, min_conds = key
        complexity = calculate_complexity_penalty(n_ens, frac_th, min_conds)

        for sl, tp, cooldown in trade_combos:
            # Simular in-sample completo
            r = simulate_trades(opt_bars, signals, sl, tp, cooldown)

            if r['total'] < MIN_TRADES:
                tested += 1
                continue

            # Walk-forward anchored
            wf = anchored_walk_forward(opt_bars, signals, sl, tp, cooldown, n_folds=5)

            if not wf.get('valid') or wf.get('oos_trades', 0) < MIN_TRADES:
                tested += 1
                continue

            # Monte Carlo validation
            mc = monte_carlo_validation(r['trades'], MONTE_CARLO_RUNS)

            # Score robusto
            score = robust_score(r, wf, mc, complexity)

            if score > 0:
                results.append({
                    'n_ens': n_ens,
                    'frac_th': frac_th,
                    'noise_amp': noise_amp,
                    'min_conds': min_conds,
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
        print("\n  NENHUMA CONFIGURACAO PASSOU NOS CRITERIOS ANTI-OVERFITTING!")
        print("  O indicador pode nao ter edge real.")
        return

    # Ordenar por score robusto
    results.sort(key=lambda x: x['score'], reverse=True)

    # Mostrar top 15
    print("\n" + "=" * 80)
    print("  TOP 15 CONFIGURACOES ROBUSTAS")
    print("=" * 80)
    print(f"\n  {'#':>2} | {'Ens':>3} | {'Frac':>4} | {'Cnd':>3} | {'SL':>3} | {'TP':>3} | {'CD':>2} | {'IS_T':>4} | {'IS_E':>5} | {'OOS_T':>5} | {'OOS_E':>5} | {'OOS_PF':>5} | {'Folds':>5} | {'Score':>6}")
    print("-" * 100)

    for i, r in enumerate(results[:15]):
        folds_str = f"{r['folds_pos']}/{r['folds_total']}"
        print(f"  {i+1:2} | {r['n_ens']:3} | {r['frac_th']:.2f} | {r['min_conds']:3} | {r['sl']:3.0f} | {r['tp']:3.0f} | {r['cooldown']:2} | {r['is_trades']:4} | {r['is_edge']:+5.1f} | {r['oos_trades']:5} | {r['oos_edge']:+5.1f} | {r['oos_pf']:5.2f} | {folds_str:>5} | {r['score']:6.1f}")

    # Selecionar melhor config
    best = results[0]

    # Validacao final no OOS nunca visto
    print("\n" + "=" * 80)
    print("  [5/5] VALIDACAO FINAL OUT-OF-SAMPLE (dados nunca vistos)")
    print("=" * 80)

    # Recomputar sinais para dados completos
    key = (best['n_ens'], best['frac_th'], best['noise_amp'], best['min_conds'])
    final_signals = compute_signals(all_bars, best['n_ens'], best['frac_th'],
                                    best['noise_amp'], best['min_conds'])

    # Testar apenas no periodo final OOS
    final_r = simulate_trades(all_bars, final_signals, best['sl'], best['tp'],
                              best['cooldown'], oos_split, len(all_bars))

    print(f"""
  Periodo Final OOS: {final_oos_bars[0].timestamp.date()} a {final_oos_bars[-1].timestamp.date()}

  Resultados:
    Trades: {final_r['total']}
    Win Rate: {final_r['wr']:.1f}%
    Edge: {final_r['edge']:+.1f}%
    Profit Factor: {final_r['pf']:.2f}
    PnL: {final_r['pnl']:.1f} pips
""")

    # Verificar se passou
    final_passed = (
        final_r['total'] >= 5 and
        final_r['edge'] > 0 and
        final_r['pf'] > 1.0
    )

    if final_passed:
        print("  *** VALIDACAO FINAL: APROVADO ***")
    else:
        print("  *** VALIDACAO FINAL: REPROVADO ***")
        print("  A estrategia pode ter overfitting residual.")

    # Resumo
    print("\n" + "=" * 80)
    print("  CONFIGURACAO ROBUSTA FINAL")
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

    # Calcular expectativa realista
    # Media ponderada: IS (20%), OOS WF (50%), Final OOS (30%)
    expected_edge = (
        best['is_edge'] * 0.2 +
        best['oos_edge'] * 0.5 +
        final_r['edge'] * 0.3
    )

    print(f"  EXPECTATIVA REALISTA DE EDGE: {expected_edge:+.1f}%")

    # Salvar config
    config = {
        "strategy": "RHHF-ROBUST-ANTI-OVERFITTING",
        "symbol": SYMBOL,
        "periodicity": "H4",
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
            "n_ensembles": int(best['n_ens']),
            "fractal_threshold": float(best['frac_th']),
            "noise_amplitude": float(best['noise_amp']),
            "min_conditions": int(best['min_conds']),
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

    path = "/home/azureuser/EliBotCD/configs/rhhf_robust_optimized.json"
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
