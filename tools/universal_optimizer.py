#!/usr/bin/env python3
"""
Universal Indicator Optimizer
=============================
Otimizador robusto para todos os indicadores de alta volatilidade
"""

import sys
import os
import urllib.request
import ssl
import json
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable
import time
import itertools
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"
SPREAD_PIPS = 1.5
SLIPPAGE_PIPS = 0.8
PIP_VALUE = 0.0001

# Criterios de qualidade
MIN_TRADES = 25
MIN_EDGE = 3
MIN_PF = 1.1
MIN_WF_FOLDS = 2

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
                time.sleep(0.02)
        except:
            break
    bars.sort(key=lambda x: x.timestamp)
    return bars

def simulate(bars: List[Bar], signals: List[Tuple[int, int]],
            sl: float, tp: float, cooldown: int,
            start_idx: int = 0, end_idx: int = None,
            direction: int = 1) -> Dict:
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
        return {'total': 0, 'wr': 0, 'edge': 0, 'pf': 0, 'pnl': 0}

    wins = len([t for t in trades if t > 0])
    total = len(trades)
    wr = wins / total * 100
    be = sl / (sl + tp) * 100
    gp = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))

    return {
        'total': total,
        'wins': wins,
        'wr': wr,
        'edge': wr - be,
        'pnl': sum(trades),
        'pf': gp / gl if gl > 0 else 0,
        'trades_list': trades
    }

def walk_forward(bars: List[Bar], signals: List[Tuple[int, int]],
                sl: float, tp: float, cooldown: int, direction: int = 1, n_folds: int = 4) -> Dict:
    n = len(bars)
    fold_size = n // n_folds
    results = []
    all_trades = []

    for f in range(n_folds):
        start = f * fold_size
        end = (f + 1) * fold_size
        fold_signals = [(i, s) for i, s in signals if start <= i < end]
        r = simulate(bars, fold_signals, sl, tp, cooldown, direction=direction)
        if r['total'] >= 3:
            results.append(r)
            all_trades.extend(r.get('trades_list', []))

    if len(results) < 2 or len(all_trades) < 10:
        return {'valid': False}

    folds_pos = sum(1 for r in results if r['edge'] > 0)
    wins = len([t for t in all_trades if t > 0])
    total = len(all_trades)
    gp = sum(t for t in all_trades if t > 0)
    gl = abs(sum(t for t in all_trades if t < 0))

    return {
        'valid': True,
        'folds_pos': folds_pos,
        'folds_total': len(results),
        'total': total,
        'wr': wins/total*100,
        'edge': wins/total*100 - sl/(sl+tp)*100,
        'pf': gp / gl if gl > 0 else 0
    }


# ============================================================================
# INDICATOR SIGNAL GENERATORS
# ============================================================================

def compute_dsg_signals(bars: List[Bar], params: dict) -> List[Tuple[int, int]]:
    """DSG - Detector de Singularidade Gravitacional"""
    from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional

    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    volumes = np.array([b.volume if b.volume > 0 else 50000 for b in bars])

    dsg = DetectorSingularidadeGravitacional(
        mass_window=params.get('mass_window', 20),
        curvature_threshold=params.get('curvature_threshold', 0.01),
        geodesic_deviation_threshold=params.get('geodesic_deviation_threshold', 0.005)
    )

    signals = []
    step = params.get('step', 8)

    for i in range(100, len(closes) - 1, step):
        try:
            result = dsg.analyze(closes[:i], highs[:i], lows[:i], volumes[:i])
            if result['signal'] != 0:
                signals.append((i, result['signal']))
        except:
            continue

    return signals


def compute_dtt_signals(bars: List[Bar], params: dict) -> List[Tuple[int, int]]:
    """DTT - Tunelamento Topológico"""
    from strategies.alta_volatilidade.dtt_tunelamento_topologico import DetectorTunelamentoTopologico

    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    volumes = np.array([b.volume if b.volume > 0 else 50000 for b in bars])

    dtt = DetectorTunelamentoTopologico(
        barrier_window=params.get('barrier_window', 50),
        tunneling_threshold=params.get('tunneling_threshold', 0.3),
        persistence_threshold=params.get('persistence_threshold', 0.1)
    )

    signals = []
    step = params.get('step', 8)

    for i in range(100, len(closes) - 1, step):
        try:
            result = dtt.analyze(closes[:i], highs[:i], lows[:i], volumes[:i])
            if result['signal'] != 0:
                signals.append((i, result['signal']))
        except:
            continue

    return signals


def compute_fifn_signals(bars: List[Bar], params: dict) -> List[Tuple[int, int]]:
    """FIFN - Fisher Information Field Navigator"""
    from strategies.alta_volatilidade.fifn_fisher_navier import FisherInformationFieldNavigator

    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    volumes = np.array([b.volume if b.volume > 0 else 50000 for b in bars])

    fifn = FisherInformationFieldNavigator(
        window_size=params.get('window_size', 30),
        fisher_threshold=params.get('fisher_threshold', 0.5),
        flow_threshold=params.get('flow_threshold', 0.01)
    )

    signals = []
    step = params.get('step', 8)

    for i in range(100, len(closes) - 1, step):
        try:
            result = fifn.analyze(closes[:i], highs[:i], lows[:i], volumes[:i])
            if result['signal'] != 0:
                signals.append((i, result['signal']))
        except:
            continue

    return signals


def compute_odmn_signals(bars: List[Bar], params: dict) -> List[Tuple[int, int]]:
    """ODMN - Malliavin-Nash Optimizer"""
    from strategies.alta_volatilidade.odmn_malliavin_nash import OtimizadorDinamicoMalliavinNash

    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    volumes = np.array([b.volume if b.volume > 0 else 50000 for b in bars])

    odmn = OtimizadorDinamicoMalliavinNash(
        malliavin_window=params.get('malliavin_window', 30),
        nash_iterations=params.get('nash_iterations', 50),
        equilibrium_threshold=params.get('equilibrium_threshold', 0.1)
    )

    signals = []
    step = params.get('step', 8)

    for i in range(100, len(closes) - 1, step):
        try:
            result = odmn.analyze(closes[:i], highs[:i], lows[:i], volumes[:i])
            if result['signal'] != 0:
                signals.append((i, result['signal']))
        except:
            continue

    return signals


def compute_phm_signals(bars: List[Bar], params: dict) -> List[Tuple[int, int]]:
    """PHM - Projetor Holográfico de Mercado"""
    from strategies.alta_volatilidade.phm_projetor_holografico import ProjetorHolograficoMercado

    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    volumes = np.array([b.volume if b.volume > 0 else 50000 for b in bars])

    phm = ProjetorHolograficoMercado(
        embedding_dim=params.get('embedding_dim', 10),
        projection_threshold=params.get('projection_threshold', 0.5),
        entropy_threshold=params.get('entropy_threshold', 0.3)
    )

    signals = []
    step = params.get('step', 8)

    for i in range(100, len(closes) - 1, step):
        try:
            result = phm.analyze(closes[:i], highs[:i], lows[:i], volumes[:i])
            if result['signal'] != 0:
                signals.append((i, result['signal']))
        except:
            continue

    return signals


def compute_prm_signals(bars: List[Bar], params: dict) -> List[Tuple[int, int]]:
    """PRM - Riemann-Mandelbrot"""
    from strategies.alta_volatilidade.prm_riemann_mandelbrot import ProjetorRiemannMandelbrot

    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    volumes = np.array([b.volume if b.volume > 0 else 50000 for b in bars])

    prm = ProjetorRiemannMandelbrot(
        fractal_dimension_threshold=params.get('fractal_dimension_threshold', 1.5),
        curvature_threshold=params.get('curvature_threshold', 0.01),
        window_size=params.get('window_size', 50)
    )

    signals = []
    step = params.get('step', 8)

    for i in range(100, len(closes) - 1, step):
        try:
            result = prm.analyze(closes[:i], highs[:i], lows[:i], volumes[:i])
            if result['signal'] != 0:
                signals.append((i, result['signal']))
        except:
            continue

    return signals


def compute_stgk_signals(bars: List[Bar], params: dict) -> List[Tuple[int, int]]:
    """STGK - Sintetizador Topos Grothendieck"""
    from strategies.alta_volatilidade.stgk_sintetizador_topos import SintetizadorToposGrothendieck

    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    volumes = np.array([b.volume if b.volume > 0 else 50000 for b in bars])

    stgk = SintetizadorToposGrothendieck(
        sheaf_window=params.get('sheaf_window', 30),
        cohomology_threshold=params.get('cohomology_threshold', 0.5),
        topos_dimension=params.get('topos_dimension', 5)
    )

    signals = []
    step = params.get('step', 8)

    for i in range(100, len(closes) - 1, step):
        try:
            result = stgk.analyze(closes[:i], highs[:i], lows[:i], volumes[:i])
            if result['signal'] != 0:
                signals.append((i, result['signal']))
        except:
            continue

    return signals


# ============================================================================
# INDICATOR CONFIGURATIONS
# ============================================================================

INDICATORS = {
    'DSG': {
        'name': 'Detector de Singularidade Gravitacional',
        'compute_fn': compute_dsg_signals,
        'param_grid': {
            'mass_window': [15, 20, 25, 30],
            'curvature_threshold': [0.005, 0.01, 0.015],
            'geodesic_deviation_threshold': [0.003, 0.005, 0.008],
            'step': [6, 8]
        }
    },
    'DTT': {
        'name': 'Tunelamento Topológico',
        'compute_fn': compute_dtt_signals,
        'param_grid': {
            'barrier_window': [40, 50, 60],
            'tunneling_threshold': [0.2, 0.3, 0.4],
            'persistence_threshold': [0.08, 0.1, 0.12],
            'step': [6, 8]
        }
    },
    'FIFN': {
        'name': 'Fisher Information Field Navigator',
        'compute_fn': compute_fifn_signals,
        'param_grid': {
            'window_size': [25, 30, 35, 40],
            'fisher_threshold': [0.4, 0.5, 0.6],
            'flow_threshold': [0.008, 0.01, 0.012],
            'step': [6, 8]
        }
    },
    'ODMN': {
        'name': 'Malliavin-Nash Optimizer',
        'compute_fn': compute_odmn_signals,
        'param_grid': {
            'malliavin_window': [25, 30, 35],
            'nash_iterations': [40, 50, 60],
            'equilibrium_threshold': [0.08, 0.1, 0.12],
            'step': [6, 8]
        }
    },
    'PHM': {
        'name': 'Projetor Holográfico de Mercado',
        'compute_fn': compute_phm_signals,
        'param_grid': {
            'embedding_dim': [8, 10, 12],
            'projection_threshold': [0.4, 0.5, 0.6],
            'entropy_threshold': [0.25, 0.3, 0.35],
            'step': [6, 8]
        }
    },
    'PRM': {
        'name': 'Riemann-Mandelbrot',
        'compute_fn': compute_prm_signals,
        'param_grid': {
            'fractal_dimension_threshold': [1.4, 1.5, 1.6],
            'curvature_threshold': [0.008, 0.01, 0.012],
            'window_size': [40, 50, 60],
            'step': [6, 8]
        }
    },
    'STGK': {
        'name': 'Sintetizador Topos Grothendieck',
        'compute_fn': compute_stgk_signals,
        'param_grid': {
            'sheaf_window': [25, 30, 35],
            'cohomology_threshold': [0.4, 0.5, 0.6],
            'topos_dimension': [4, 5, 6],
            'step': [6, 8]
        }
    }
}

# Trade parameters grid
TRADE_GRID = {
    'sl': [25, 30, 35],
    'tp': [50, 55, 60, 70],
    'cooldown': [5, 6, 8]
}


def optimize_indicator(indicator_key: str, bars: List[Bar], verbose: bool = True) -> Dict:
    """Otimiza um indicador específico"""

    if indicator_key not in INDICATORS:
        print(f"Indicador {indicator_key} não encontrado!")
        return None

    config = INDICATORS[indicator_key]
    compute_fn = config['compute_fn']
    param_grid = config['param_grid']

    if verbose:
        print(f"\n{'='*70}")
        print(f"  {indicator_key} - {config['name']}")
        print(f"{'='*70}")

    # Generate parameter combinations
    param_keys = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_keys]
    param_combos = list(itertools.product(*param_values))

    trade_keys = list(TRADE_GRID.keys())
    trade_values = [TRADE_GRID[k] for k in trade_keys]
    trade_combos = list(itertools.product(*trade_values))

    if verbose:
        print(f"  Configs indicador: {len(param_combos)}")
        print(f"  Configs trade: {len(trade_combos)}")
        print(f"  Total: {len(param_combos) * len(trade_combos)}")

    # Compute signals for each parameter set
    if verbose:
        print(f"\n  Computando sinais...")

    signals_cache = {}
    for i, combo in enumerate(param_combos):
        params = dict(zip(param_keys, combo))

        if verbose and i % 10 == 0:
            print(f"    {i+1}/{len(param_combos)}...", end=" ", flush=True)

        try:
            t0 = time.time()
            signals = compute_fn(bars, params)
            elapsed = time.time() - t0

            if verbose and i % 10 == 0:
                print(f"{len(signals)} sinais ({elapsed:.1f}s)")

            signals_cache[combo] = signals
        except Exception as e:
            if verbose and i % 10 == 0:
                print(f"Erro: {e}")
            signals_cache[combo] = []

    # Test all combinations
    if verbose:
        print(f"\n  Testando combinacoes...")

    results = []

    for param_combo, signals in signals_cache.items():
        if len(signals) < 10:
            continue

        params = dict(zip(param_keys, param_combo))

        for trade_combo in trade_combos:
            sl, tp, cooldown = trade_combo

            # Test LONG
            r = simulate(bars, signals, sl, tp, cooldown, direction=1)

            if r['total'] < MIN_TRADES:
                continue
            if r['edge'] < MIN_EDGE:
                continue
            if r['pf'] < MIN_PF:
                continue

            wf = walk_forward(bars, signals, sl, tp, cooldown, direction=1)

            if not wf.get('valid') or wf.get('folds_pos', 0) < MIN_WF_FOLDS:
                continue

            score = (
                r['pf'] * 10 +
                r['edge'] * 0.5 +
                np.sqrt(r['total']) * 1.5 +
                wf.get('folds_pos', 0) * 4
            )

            results.append({
                **params,
                'sl': sl,
                'tp': tp,
                'cooldown': cooldown,
                'trades': r['total'],
                'wr': r['wr'],
                'edge': r['edge'],
                'pf': r['pf'],
                'pnl': r['pnl'],
                'wf_folds': wf.get('folds_pos', 0),
                'wf_total': wf.get('folds_total', 0),
                'wf_edge': wf.get('edge', 0),
                'wf_pf': wf.get('pf', 0),
                'score': score
            })

    if verbose:
        print(f"\n  Configuracoes validas: {len(results)}")

    if not results:
        if verbose:
            print(f"\n  NENHUMA configuracao valida para {indicator_key}")

            # Diagnostic
            print("\n  Diagnostico:")
            for combo, signals in signals_cache.items():
                if len(signals) >= 10:
                    params = dict(zip(param_keys, combo))
                    r = simulate(bars, signals, 30, 55, 6, direction=1)
                    if r['total'] >= 10:
                        print(f"    {params}: {r['total']} trades, WR={r['wr']:.1f}%, Edge={r['edge']:+.1f}%, PF={r['pf']:.2f}")

        return None

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    if verbose:
        print(f"\n  TOP 5 CONFIGURACOES:")
        print(f"  {'#':>2} | {'Trd':>4} | {'WR%':>5} | {'Edge':>5} | {'PF':>5} | {'WF':>4}")
        print("-" * 50)
        for i, r in enumerate(results[:5]):
            wf_str = f"{r['wf_folds']}/{r['wf_total']}"
            print(f"  {i+1:2} | {r['trades']:4} | {r['wr']:5.1f} | {r['edge']:+5.1f} | {r['pf']:5.2f} | {wf_str:>4}")

    best = results[0]

    # Build config
    indicator_params = {k: best[k] for k in param_keys if k != 'step'}

    config_out = {
        "strategy": f"{indicator_key}-OPTIMIZED",
        "symbol": SYMBOL,
        "periodicity": "H1",
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "mode": "LONG_ONLY",
        "parameters": {
            **indicator_params,
            "stop_loss_pips": int(best['sl']),
            "take_profit_pips": int(best['tp']),
            "cooldown_bars": int(best['cooldown'])
        },
        "performance": {
            "trades": int(best['trades']),
            "win_rate": float(best['wr']),
            "edge": float(best['edge']),
            "profit_factor": float(best['pf']),
            "pnl_pips": float(best['pnl'])
        },
        "walk_forward": {
            "folds_positive": best['wf_folds'],
            "folds_total": best['wf_total'],
            "edge": float(best['wf_edge']),
            "profit_factor": float(best['wf_pf'])
        }
    }

    # Save config
    path = f"/home/azureuser/EliBotCD/configs/{indicator_key.lower()}_robust_final.json"
    with open(path, 'w') as f:
        json.dump(config_out, f, indent=2)

    if verbose:
        print(f"\n  MELHOR {indicator_key}:")
        print(f"    Trades: {best['trades']}")
        print(f"    Win Rate: {best['wr']:.1f}%")
        print(f"    Edge: {best['edge']:+.1f}%")
        print(f"    PF: {best['pf']:.2f}")
        print(f"    Walk-Forward: {best['wf_folds']}/{best['wf_total']}")
        print(f"\n  Salvo: {path}")

    return config_out


def main():
    parser = argparse.ArgumentParser(description='Universal Indicator Optimizer')
    parser.add_argument('--indicator', '-i', type=str, default='ALL',
                       help='Indicator to optimize (DSG, DTT, FIFN, ODMN, PHM, PRM, STGK, or ALL)')
    parser.add_argument('--bars', '-b', type=int, default=4000,
                       help='Number of bars to download')
    args = parser.parse_args()

    print("=" * 70)
    print("  UNIVERSAL INDICATOR OPTIMIZER")
    print("=" * 70)

    start_time = time.time()

    print(f"\n[1] Baixando dados H1...", end=" ", flush=True)
    bars = download_bars("H1", args.bars)
    print(f"{len(bars)} barras")

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()} ({days} dias)")

    # Determine which indicators to optimize
    if args.indicator.upper() == 'ALL':
        indicators_to_optimize = list(INDICATORS.keys())
    else:
        indicators_to_optimize = [args.indicator.upper()]

    print(f"\n[2] Otimizando indicadores: {', '.join(indicators_to_optimize)}")

    results_summary = {}

    for indicator in indicators_to_optimize:
        try:
            result = optimize_indicator(indicator, bars, verbose=True)
            if result:
                results_summary[indicator] = result
        except Exception as e:
            print(f"\n  ERRO otimizando {indicator}: {e}")

    elapsed = time.time() - start_time

    # Final summary
    print("\n" + "=" * 70)
    print("  RESUMO FINAL")
    print("=" * 70)

    if results_summary:
        print(f"\n  {'Indicador':<10} | {'Trades':>6} | {'WR%':>6} | {'Edge':>6} | {'PF':>6} | {'WF':>5}")
        print("-" * 60)

        for ind, cfg in results_summary.items():
            perf = cfg['performance']
            wf = cfg['walk_forward']
            wf_str = f"{wf['folds_positive']}/{wf['folds_total']}"
            print(f"  {ind:<10} | {perf['trades']:6} | {perf['win_rate']:6.1f} | {perf['edge']:+6.1f} | {perf['profit_factor']:6.2f} | {wf_str:>5}")
    else:
        print("\n  Nenhum indicador otimizado com sucesso.")

    print(f"\n  Tempo total: {elapsed/60:.1f} minutos")


if __name__ == "__main__":
    main()
