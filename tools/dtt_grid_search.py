#!/usr/bin/env python3
"""
DTT Grid Search - Busca de Melhores Parametros
Testa varias combinacoes para encontrar edge positivo
"""

import sys
import os
import urllib.request
import ssl
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import List, Tuple
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.alta_volatilidade.dtt_tunelamento_topologico import DetectorTunelamentoTopologico

# =============================================================================
# CONFIGURACAO
# =============================================================================

API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"

SPREAD_PIPS = 1.5
SLIPPAGE_PIPS = 0.8
TRADING_START_HOUR = 7
TRADING_END_HOUR = 20
COOLDOWN = 15
MIN_WARMUP = 150
PIP_VALUE = 0.0001

@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

# =============================================================================
# DOWNLOAD
# =============================================================================

def download_real_bars(period: str, count: int) -> List[Bar]:
    bars = []
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    current_ts = int(time.time() * 1000)
    total_downloaded = 0

    print(f"    Baixando {count} barras {period}...")

    while total_downloaded < count:
        remaining = min(1000, count - total_downloaded)
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
                    bar = Bar(ts, float(b["Open"]), float(b["High"]),
                              float(b["Low"]), float(b["Close"]), float(b.get("Volume", 0)))
                    bars.append(bar)

                total_downloaded += len(batch)
                oldest = min(batch, key=lambda x: x["Timestamp"])
                current_ts = oldest["Timestamp"] - 1
                print(f"      +{len(batch)} (total: {total_downloaded})")
                time.sleep(0.15)

        except Exception as e:
            print(f"    Erro: {e}")
            break

    bars.sort(key=lambda x: x.timestamp)
    return bars

# =============================================================================
# PRE-CALCULAR SINAIS DTT
# =============================================================================

def precompute_signals(bars: List[Bar]) -> List[dict]:
    """Pre-calcula todos os sinais DTT para acelerar grid search."""

    dtt = DetectorTunelamentoTopologico(
        max_points=150,
        use_dimensionality_reduction=True,
        reduction_method='pca',
        persistence_entropy_threshold=0.1,
        tunneling_probability_threshold=0.05,
        auto_calibrate_quantum=True
    )

    signals = []
    prices_buf = deque(maxlen=500)

    print(f"\n  Pre-calculando sinais DTT ({len(bars)} barras)...")

    for i, bar in enumerate(bars):
        prices_buf.append(bar.close)

        if len(prices_buf) < MIN_WARMUP:
            continue

        if i >= len(bars) - 1:
            continue

        hour = bar.timestamp.hour
        if hour < TRADING_START_HOUR or hour >= TRADING_END_HOUR:
            continue

        try:
            result = dtt.analyze(np.array(prices_buf))

            if result is None or 'error' in result:
                continue

            entropy_info = result.get('entropy', {})
            tunneling_info = result.get('tunneling', {})

            signals.append({
                'bar_idx': i,
                'timestamp': bar.timestamp,
                'persistence_entropy': entropy_info.get('persistence_entropy', 0),
                'tunneling_probability': tunneling_info.get('tunneling_probability', 0),
                'signal_strength': result.get('signal_strength', 0),
                'trade_on': result.get('trade_on', False),
                'direction': result.get('direction', ''),
                'entry_price': bars[i + 1].open
            })

        except Exception:
            continue

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(bars)} barras...")

    print(f"  {len(signals)} sinais pre-calculados")
    return signals

# =============================================================================
# BACKTEST RAPIDO
# =============================================================================

def fast_backtest(signals: List[dict], bars: List[Bar],
                  entropy_thresh: float, tunneling_thresh: float,
                  strength_thresh: float, sl_pips: float, tp_pips: float) -> dict:
    """Backtest rapido usando sinais pre-calculados."""

    trades = []
    last_signal = -999

    for s in signals:
        # Cooldown
        if s['bar_idx'] - last_signal < COOLDOWN:
            continue

        # Verificar thresholds
        if s['persistence_entropy'] < entropy_thresh:
            continue
        if s['tunneling_probability'] < tunneling_thresh:
            continue
        if s['signal_strength'] < strength_thresh:
            continue
        if not s['trade_on']:
            continue

        direction = s['direction']
        if direction == 'LONG':
            signal = 1
        elif direction == 'SHORT':
            signal = -1
        else:
            continue

        # Simular trade
        entry = s['entry_price']

        if signal == 1:
            entry += SLIPPAGE_PIPS * PIP_VALUE
            sl = entry - sl_pips * PIP_VALUE
            tp = entry + tp_pips * PIP_VALUE
        else:
            entry -= SLIPPAGE_PIPS * PIP_VALUE
            sl = entry + sl_pips * PIP_VALUE
            tp = entry - tp_pips * PIP_VALUE

        # Verificar resultado
        bar_idx = s['bar_idx']
        for j in range(bar_idx + 2, min(bar_idx + 200, len(bars))):
            bar = bars[j]

            if signal == 1:
                if bar.low <= sl:
                    trades.append(('LOSS', -sl_pips - SPREAD_PIPS))
                    break
                if bar.high >= tp:
                    trades.append(('WIN', tp_pips - SPREAD_PIPS))
                    break
            else:
                if bar.high >= sl:
                    trades.append(('LOSS', -sl_pips - SPREAD_PIPS))
                    break
                if bar.low <= tp:
                    trades.append(('WIN', tp_pips - SPREAD_PIPS))
                    break

        last_signal = bar_idx

    # Estatisticas
    if not trades:
        return {'total_trades': 0, 'edge': -100, 'pf': 0}

    wins = len([t for t in trades if t[0] == 'WIN'])
    total = len(trades)
    win_rate = wins / total * 100

    breakeven = sl_pips / (sl_pips + tp_pips) * 100
    edge = win_rate - breakeven

    total_pnl = sum(t[1] for t in trades)

    gross_profit = sum(t[1] for t in trades if t[1] > 0)
    gross_loss = abs(sum(t[1] for t in trades if t[1] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    return {
        'total_trades': total,
        'wins': wins,
        'losses': total - wins,
        'win_rate': win_rate,
        'breakeven': breakeven,
        'edge': edge,
        'total_pnl': total_pnl,
        'pf': pf,
        'entropy_thresh': entropy_thresh,
        'tunneling_thresh': tunneling_thresh,
        'strength_thresh': strength_thresh,
        'sl_pips': sl_pips,
        'tp_pips': tp_pips
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("  DTT GRID SEARCH - BUSCA DE MELHORES PARAMETROS")
    print("=" * 80)

    # Baixar dados
    bars = download_real_bars("H1", 2500)

    if len(bars) < MIN_WARMUP + 100:
        print(f"  ERRO: Apenas {len(bars)} barras!")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"\n  Periodo: {bars[0].timestamp.strftime('%Y-%m-%d')} a {bars[-1].timestamp.strftime('%Y-%m-%d')}")
    print(f"  Dias: {days} | Barras: {len(bars)}")

    # Pre-calcular sinais
    signals = precompute_signals(bars)

    if len(signals) < 100:
        print(f"  ERRO: Apenas {len(signals)} sinais!")
        return

    # Grid Search
    print(f"\n{'='*80}")
    print("  GRID SEARCH")
    print(f"{'='*80}")

    entropy_values = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    tunneling_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    strength_values = [0.2, 0.25, 0.3, 0.35]
    sl_values = [20, 25, 30, 35]
    tp_values = [40, 50, 60, 70, 80]

    total_combos = len(entropy_values) * len(tunneling_values) * len(strength_values) * len(sl_values) * len(tp_values)
    print(f"  Total de combinacoes: {total_combos}")

    results = []
    tested = 0

    for entropy in entropy_values:
        for tunneling in tunneling_values:
            for strength in strength_values:
                for sl in sl_values:
                    for tp in tp_values:
                        result = fast_backtest(signals, bars, entropy, tunneling, strength, sl, tp)
                        tested += 1

                        # Guardar se tiver trades suficientes
                        if result['total_trades'] >= 20:
                            results.append(result)

                        if tested % 1000 == 0:
                            print(f"    {tested}/{total_combos} testadas, {len(results)} validas")

    print(f"\n  Combinacoes validas (>=20 trades): {len(results)}")

    # Filtrar e ordenar
    profitable = [r for r in results if r['edge'] > 0 and r['pf'] > 1.1]

    if profitable:
        print(f"\n  COMBINACOES LUCRATIVAS (edge > 0, PF > 1.1): {len(profitable)}")

        # Top 10 por edge
        profitable.sort(key=lambda x: x['edge'], reverse=True)

        print(f"\n  TOP 10 POR EDGE:")
        print(f"  {'-'*100}")
        print(f"  {'#':<3} {'Entropy':<8} {'Tunnel':<8} {'Strength':<9} {'SL/TP':<10} {'Trades':<8} {'Win%':<7} {'Edge':<8} {'PF':<6} {'PnL'}")
        print(f"  {'-'*100}")

        for i, r in enumerate(profitable[:10], 1):
            sl_tp = f"{r['sl_pips']:.0f}/{r['tp_pips']:.0f}"
            print(f"  {i:<3} {r['entropy_thresh']:<8.2f} {r['tunneling_thresh']:<8.2f} {r['strength_thresh']:<9.2f} "
                  f"{sl_tp:<10} {r['total_trades']:<8} {r['win_rate']:.1f}%{'':<2} {r['edge']:+.1f}%{'':<3} "
                  f"{r['pf']:.2f}{'':<2} {r['total_pnl']:+.0f}")

        # Melhor
        best = profitable[0]
        print(f"\n{'='*80}")
        print(f"  MELHOR CONFIGURACAO")
        print(f"{'='*80}")
        print(f"  persistence_entropy_threshold: {best['entropy_thresh']}")
        print(f"  tunneling_probability_threshold: {best['tunneling_thresh']}")
        print(f"  min_signal_strength: {best['strength_thresh']}")
        print(f"  stop_loss_pips: {best['sl_pips']}")
        print(f"  take_profit_pips: {best['tp_pips']}")
        print(f"\n  PERFORMANCE:")
        print(f"    Trades: {best['total_trades']}")
        print(f"    Win Rate: {best['win_rate']:.1f}%")
        print(f"    Breakeven: {best['breakeven']:.1f}%")
        print(f"    Edge: {best['edge']:+.1f}%")
        print(f"    PnL: {best['total_pnl']:+.0f} pips")
        print(f"    Profit Factor: {best['pf']:.2f}")

    else:
        print(f"\n  NENHUMA COMBINACAO LUCRATIVA ENCONTRADA!")

        # Mostrar melhores mesmo assim
        if results:
            results.sort(key=lambda x: x['edge'], reverse=True)
            print(f"\n  Melhores resultados (mesmo com edge negativo):")
            for i, r in enumerate(results[:5], 1):
                sl_tp = f"{r['sl_pips']:.0f}/{r['tp_pips']:.0f}"
                print(f"  {i}. Edge={r['edge']:+.1f}% PF={r['pf']:.2f} Trades={r['total_trades']} "
                      f"Entropy={r['entropy_thresh']} Tunnel={r['tunneling_thresh']}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
