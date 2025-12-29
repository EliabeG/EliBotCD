#!/usr/bin/env python3
"""
Quick Indicator Optimizer
=========================
Otimizador ultra-rápido com grid mínimo para todos os indicadores
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
import argparse

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
                time.sleep(0.02)
        except:
            break
    bars.sort(key=lambda x: x.timestamp)
    return bars

def simulate(bars: List[Bar], signals: List[Tuple[int, int]],
            sl: float, tp: float, cooldown: int, direction: int = 1) -> Dict:
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
        'wr': wr,
        'edge': wr - be,
        'pnl': sum(trades),
        'pf': gp / gl if gl > 0 else 0,
        'trades': trades
    }

def walk_forward(bars: List[Bar], signals: List[Tuple[int, int]],
                sl: float, tp: float, cooldown: int, n_folds: int = 4) -> Dict:
    n = len(bars)
    fold_size = n // n_folds
    results = []
    all_trades = []

    for f in range(n_folds):
        start = f * fold_size
        end = (f + 1) * fold_size
        fold_signals = [(i, s) for i, s in signals if start <= i < end]
        r = simulate(bars, fold_signals, sl, tp, cooldown)
        if r['total'] >= 2:
            results.append(r)
            all_trades.extend(r.get('trades', []))

    if len(results) < 2 or len(all_trades) < 5:
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


def optimize_dsg(bars: List[Bar]) -> Dict:
    """Otimiza DSG com grid mínimo"""
    print("\n  [DSG] Detector Singularidade Gravitacional")
    from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional

    closes = np.array([b.close for b in bars])

    dsg = DetectorSingularidadeGravitacional()

    print("    Computando sinais...", end=" ", flush=True)
    t0 = time.time()
    signals = []
    for i in range(100, len(closes) - 1, 15):
        try:
            result = dsg.analyze(closes[:i])
            if result.get('signal', 0) != 0:
                signals.append((i, result['signal']))
        except:
            continue
    print(f"{len(signals)} sinais ({time.time()-t0:.1f}s)")

    return test_trade_params(bars, signals, "DSG")


def optimize_dtt(bars: List[Bar]) -> Dict:
    """Otimiza DTT com grid mínimo"""
    print("\n  [DTT] Tunelamento Topológico")
    from strategies.alta_volatilidade.dtt_tunelamento_topologico import DetectorTunelamentoTopologico

    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])

    dtt = DetectorTunelamentoTopologico()

    print("    Computando sinais...", end=" ", flush=True)
    t0 = time.time()
    signals = []
    for i in range(100, len(closes) - 1, 15):
        try:
            result = dtt.analyze(closes[:i], highs[:i], lows[:i])
            if result.get('signal', 0) != 0:
                signals.append((i, result['signal']))
        except:
            continue
    print(f"{len(signals)} sinais ({time.time()-t0:.1f}s)")

    return test_trade_params(bars, signals, "DTT")


def optimize_fifn(bars: List[Bar]) -> Dict:
    """Otimiza FIFN com grid mínimo"""
    print("\n  [FIFN] Fluxo Fisher-Navier")
    from strategies.alta_volatilidade.fifn_fisher_navier import FluxoInformacaoFisherNavier

    closes = np.array([b.close for b in bars])

    fifn = FluxoInformacaoFisherNavier()

    print("    Computando sinais...", end=" ", flush=True)
    t0 = time.time()
    signals = []
    for i in range(100, len(closes) - 1, 15):
        try:
            result = fifn.analyze(closes[:i])
            if result.get('signal', 0) != 0:
                signals.append((i, result['signal']))
        except:
            continue
    print(f"{len(signals)} sinais ({time.time()-t0:.1f}s)")

    return test_trade_params(bars, signals, "FIFN")


def optimize_odmn(bars: List[Bar]) -> Dict:
    """Otimiza ODMN com grid mínimo"""
    print("\n  [ODMN] Oráculo Malliavin-Nash")
    from strategies.alta_volatilidade.odmn_malliavin_nash import OracloDerivativosMalliavinNash

    closes = np.array([b.close for b in bars])

    odmn = OracloDerivativosMalliavinNash()

    print("    Computando sinais...", end=" ", flush=True)
    t0 = time.time()
    signals = []
    for i in range(120, len(closes) - 1, 15):
        try:
            result = odmn.analyze(closes[:i])
            if result.get('signal', 0) != 0:
                signals.append((i, result['signal']))
        except:
            continue
    print(f"{len(signals)} sinais ({time.time()-t0:.1f}s)")

    return test_trade_params(bars, signals, "ODMN")


def optimize_phm(bars: List[Bar]) -> Dict:
    """Otimiza PHM com grid mínimo"""
    print("\n  [PHM] Projetor Holográfico")
    from strategies.alta_volatilidade.phm_projetor_holografico import ProjetorHolograficoMaldacena

    closes = np.array([b.close for b in bars])

    phm = ProjetorHolograficoMaldacena()

    print("    Computando sinais...", end=" ", flush=True)
    t0 = time.time()
    signals = []
    for i in range(150, len(closes) - 1, 15):
        try:
            result = phm.analyze(closes[:i])
            if result.get('signal', 0) != 0:
                signals.append((i, result['signal']))
        except:
            continue
    print(f"{len(signals)} sinais ({time.time()-t0:.1f}s)")

    return test_trade_params(bars, signals, "PHM")


def optimize_prm(bars: List[Bar]) -> Dict:
    """Otimiza PRM com grid mínimo"""
    print("\n  [PRM] Protocolo Riemann-Mandelbrot")
    from strategies.alta_volatilidade.prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot

    closes = np.array([b.close for b in bars])
    volumes = np.array([b.volume if b.volume > 0 else 50000 for b in bars])

    prm = ProtocoloRiemannMandelbrot()

    print("    Computando sinais...", end=" ", flush=True)
    t0 = time.time()
    signals = []
    for i in range(150, len(closes) - 1, 15):
        try:
            result = prm.analyze(closes[:i], volumes[:i])
            if result.get('signal', 0) != 0:
                signals.append((i, result['signal']))
        except:
            continue
    print(f"{len(signals)} sinais ({time.time()-t0:.1f}s)")

    return test_trade_params(bars, signals, "PRM")


def optimize_stgk(bars: List[Bar]) -> Dict:
    """Otimiza STGK com grid mínimo"""
    print("\n  [STGK] Sintetizador Topos Grothendieck")
    from strategies.alta_volatilidade.stgk_sintetizador_topos import SintetizadorToposGrothendieckKolmogorov

    closes = np.array([b.close for b in bars])

    stgk = SintetizadorToposGrothendieckKolmogorov()

    print("    Computando sinais...", end=" ", flush=True)
    t0 = time.time()
    signals = []
    for i in range(120, len(closes) - 1, 15):
        try:
            result = stgk.analyze(closes[:i])
            if result.get('signal', 0) != 0:
                signals.append((i, result['signal']))
        except:
            continue
    print(f"{len(signals)} sinais ({time.time()-t0:.1f}s)")

    return test_trade_params(bars, signals, "STGK")


def test_trade_params(bars: List[Bar], signals: List[Tuple[int, int]], name: str) -> Dict:
    """Testa parâmetros de trade com grid mínimo"""

    if len(signals) < 5:
        print(f"    Poucos sinais ({len(signals)}). Pulando.")
        return None

    trade_configs = [
        (25, 50, 5), (25, 55, 6), (30, 55, 5), (30, 60, 6),
        (35, 60, 5), (35, 65, 6), (30, 70, 5), (35, 70, 6)
    ]

    results = []

    for sl, tp, cd in trade_configs:
        r = simulate(bars, signals, sl, tp, cd, direction=1)

        if r['total'] < 10:
            continue
        if r['edge'] < 0:
            continue
        if r['pf'] < 1.0:
            continue

        wf = walk_forward(bars, signals, sl, tp, cd)

        if wf.get('valid') and wf.get('folds_pos', 0) >= 2:
            results.append({
                'sl': sl,
                'tp': tp,
                'cd': cd,
                'trades': r['total'],
                'wr': r['wr'],
                'edge': r['edge'],
                'pf': r['pf'],
                'pnl': r['pnl'],
                'wf_folds': wf['folds_pos'],
                'wf_total': wf['folds_total'],
                'wf_edge': wf['edge']
            })

    if not results:
        print(f"    Nenhuma config válida.")
        # Show diagnostic
        r = simulate(bars, signals, 30, 55, 5, direction=1)
        print(f"    Diag: {r['total']} trades, WR={r['wr']:.1f}%, Edge={r['edge']:+.1f}%, PF={r['pf']:.2f}")
        return None

    # Sort by score
    results.sort(key=lambda x: x['pf'] * x['edge'] * np.sqrt(x['trades']), reverse=True)
    best = results[0]

    print(f"    MELHOR: {best['trades']} trades | WR={best['wr']:.1f}% | Edge={best['edge']:+.1f}% | PF={best['pf']:.2f} | WF={best['wf_folds']}/{best['wf_total']}")

    config = {
        "strategy": f"{name}-QUICK",
        "symbol": SYMBOL,
        "periodicity": "H1",
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "mode": "LONG_ONLY",
        "parameters": {
            "stop_loss_pips": int(best['sl']),
            "take_profit_pips": int(best['tp']),
            "cooldown_bars": int(best['cd'])
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
            "edge": float(best['wf_edge'])
        }
    }

    path = f"/home/azureuser/EliBotCD/configs/{name.lower()}_quick.json"
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"    Salvo: {path}")

    return config


def main():
    print("=" * 70)
    print("  QUICK INDICATOR OPTIMIZER")
    print("=" * 70)

    start_time = time.time()

    print("\n[1] Baixando dados H1...", end=" ", flush=True)
    bars = download_bars("H1", 2500)
    print(f"{len(bars)} barras")

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()} ({days} dias)")

    print("\n[2] Otimizando indicadores...")

    results = {}

    indicators = [
        ('DSG', optimize_dsg),
        ('DTT', optimize_dtt),
        ('FIFN', optimize_fifn),
        ('ODMN', optimize_odmn),
        ('PHM', optimize_phm),
        ('PRM', optimize_prm),
        ('STGK', optimize_stgk),
    ]

    for name, func in indicators:
        try:
            result = func(bars)
            if result:
                results[name] = result
        except Exception as e:
            print(f"\n  [{name}] ERRO: {e}")

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("  RESUMO FINAL")
    print("=" * 70)

    if results:
        print(f"\n  {'Indicador':<10} | {'Trades':>6} | {'WR%':>6} | {'Edge':>6} | {'PF':>6} | {'WF':>5}")
        print("-" * 60)

        for ind, cfg in results.items():
            perf = cfg['performance']
            wf = cfg['walk_forward']
            wf_str = f"{wf['folds_positive']}/{wf['folds_total']}"
            print(f"  {ind:<10} | {perf['trades']:6} | {perf['win_rate']:6.1f} | {perf['edge']:+6.1f} | {perf['profit_factor']:6.2f} | {wf_str:>5}")

        print(f"\n  Indicadores otimizados: {len(results)}/{len(indicators)}")
    else:
        print("\n  Nenhum indicador otimizado com sucesso.")

    print(f"\n  Tempo total: {elapsed/60:.1f} minutos")


if __name__ == "__main__":
    main()
