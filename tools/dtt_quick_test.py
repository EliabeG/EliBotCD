#!/usr/bin/env python3
"""
DTT Quick Test - Testa em H1 e H4 com parametros reotimizados
"""

import sys
import os
import urllib.request
import ssl
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import List
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.alta_volatilidade.dtt_tunelamento_topologico import DetectorTunelamentoTopologico

# =============================================================================
# PARAMETROS REOTIMIZADOS (Grid Search 28/12/2025)
# =============================================================================

PERSISTENCE_ENTROPY_THRESHOLD = 0.5
TUNNELING_PROBABILITY_THRESHOLD = 0.35
MIN_SIGNAL_STRENGTH = 0.2

# Custos
SPREAD_PIPS = 1.5
SLIPPAGE_PIPS = 0.8

# Trading
TRADING_START_HOUR = 7
TRADING_END_HOUR = 20
COOLDOWN = 15
MIN_WARMUP = 150
PIP_VALUE = 0.0001

# API
API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"

@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class TimeframeConfig:
    name: str
    period: str
    sl_pips: float
    tp_pips: float
    bars_to_download: int
    cooldown: int

TIMEFRAME_CONFIGS = {
    'H1': TimeframeConfig('H1', 'H1', 20.0, 40.0, 2500, 15),
    'H4': TimeframeConfig('H4', 'H4', 35.0, 70.0, 1000, 8),
}

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
# BACKTEST
# =============================================================================

def run_backtest(bars: List[Bar], config: TimeframeConfig) -> dict:
    dtt = DetectorTunelamentoTopologico(
        max_points=150,
        use_dimensionality_reduction=True,
        reduction_method='pca',
        persistence_entropy_threshold=0.1,
        tunneling_probability_threshold=0.05,
        auto_calibrate_quantum=True
    )

    trades = []
    prices = [b.close for b in bars]
    last_signal = -999
    total_bars = len(bars)
    progress_step = max(1, total_bars // 10)

    print(f"    Processando {total_bars} barras...")

    for i in range(MIN_WARMUP, len(bars) - 1):
        if i % progress_step == 0:
            pct = i * 100 // total_bars
            print(f"      {pct}%... (trades: {len(trades)})")

        if i - last_signal < config.cooldown:
            continue

        hour = bars[i].timestamp.hour
        if hour < TRADING_START_HOUR or hour >= TRADING_END_HOUR:
            continue

        try:
            prices_arr = np.array(prices[:i+1])
            result = dtt.analyze(prices_arr)

            if result is None or 'error' in result:
                continue

            entropy_info = result.get('entropy', {})
            tunneling_info = result.get('tunneling', {})

            persistence_entropy = entropy_info.get('persistence_entropy', 0)
            tunneling_prob = tunneling_info.get('tunneling_probability', 0)
            signal_strength = result.get('signal_strength', 0)

            if persistence_entropy < PERSISTENCE_ENTROPY_THRESHOLD:
                continue
            if tunneling_prob < TUNNELING_PROBABILITY_THRESHOLD:
                continue
            if signal_strength < MIN_SIGNAL_STRENGTH:
                continue

            if not result.get('trade_on', False):
                continue

            direction = result.get('direction', '')
            if direction == 'LONG':
                signal = 1
            elif direction == 'SHORT':
                signal = -1
            else:
                continue

        except Exception:
            continue

        # Simular trade
        entry = bars[i + 1].open

        if signal == 1:
            entry += SLIPPAGE_PIPS * PIP_VALUE
            sl = entry - config.sl_pips * PIP_VALUE
            tp = entry + config.tp_pips * PIP_VALUE
        else:
            entry -= SLIPPAGE_PIPS * PIP_VALUE
            sl = entry + config.sl_pips * PIP_VALUE
            tp = entry - config.tp_pips * PIP_VALUE

        for j in range(i + 2, min(i + 200, len(bars))):
            bar = bars[j]

            if signal == 1:
                if bar.low <= sl:
                    trades.append(('LOSS', -config.sl_pips - SPREAD_PIPS))
                    break
                if bar.high >= tp:
                    trades.append(('WIN', config.tp_pips - SPREAD_PIPS))
                    break
            else:
                if bar.high >= sl:
                    trades.append(('LOSS', -config.sl_pips - SPREAD_PIPS))
                    break
                if bar.low <= tp:
                    trades.append(('WIN', config.tp_pips - SPREAD_PIPS))
                    break

        last_signal = i

    print(f"      100% (trades: {len(trades)})")

    if not trades:
        return {
            'timeframe': config.name,
            'total_trades': 0,
            'win_rate': 0,
            'breakeven': config.sl_pips / (config.sl_pips + config.tp_pips) * 100,
            'edge': 0,
            'total_pnl': 0,
            'profit_factor': 0,
            'trades_per_month': 0,
            'sl_pips': config.sl_pips,
            'tp_pips': config.tp_pips
        }

    wins = len([t for t in trades if t[0] == 'WIN'])
    total = len(trades)
    win_rate = wins / total * 100

    breakeven = config.sl_pips / (config.sl_pips + config.tp_pips) * 100
    edge = win_rate - breakeven

    total_pnl = sum(t[1] for t in trades)

    gross_profit = sum(t[1] for t in trades if t[1] > 0)
    gross_loss = abs(sum(t[1] for t in trades if t[1] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    if len(bars) > 1:
        days = (bars[-1].timestamp - bars[0].timestamp).days
        months = days / 30.0 if days > 0 else 1
        trades_per_month = total / months
    else:
        trades_per_month = 0

    return {
        'timeframe': config.name,
        'total_trades': total,
        'wins': wins,
        'losses': total - wins,
        'win_rate': win_rate,
        'breakeven': breakeven,
        'edge': edge,
        'total_pnl': total_pnl,
        'profit_factor': pf,
        'trades_per_month': trades_per_month,
        'sl_pips': config.sl_pips,
        'tp_pips': config.tp_pips
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("  DTT QUICK TEST - H1 E H4 COM PARAMETROS REOTIMIZADOS")
    print("=" * 80)
    print()
    print("  PARAMETROS REOTIMIZADOS (Grid Search 28/12/2025):")
    print(f"    persistence_entropy >= {PERSISTENCE_ENTROPY_THRESHOLD}")
    print(f"    tunneling_probability >= {TUNNELING_PROBABILITY_THRESHOLD}")
    print(f"    signal_strength >= {MIN_SIGNAL_STRENGTH}")
    print()

    results = []

    for tf_name in ['H1', 'H4']:
        config = TIMEFRAME_CONFIGS[tf_name]

        print(f"\n{'='*75}")
        print(f"  TESTANDO {tf_name}")
        print(f"{'='*75}")
        print(f"  SL: {config.sl_pips} pips | TP: {config.tp_pips} pips | Cooldown: {config.cooldown}")
        print(f"  Breakeven: {config.sl_pips / (config.sl_pips + config.tp_pips) * 100:.1f}%")
        print()

        bars = download_real_bars(config.period, config.bars_to_download)

        if len(bars) < MIN_WARMUP + 100:
            print(f"  ERRO: Apenas {len(bars)} barras!")
            continue

        days = (bars[-1].timestamp - bars[0].timestamp).days
        print(f"\n    Periodo: {bars[0].timestamp.strftime('%Y-%m-%d')} a {bars[-1].timestamp.strftime('%Y-%m-%d')}")
        print(f"    Dias: {days} | Barras: {len(bars)}")
        print()

        result = run_backtest(bars, config)
        results.append(result)

        print(f"\n  RESULTADO {tf_name}:")
        print(f"  {'-'*55}")
        print(f"  Trades: {result['total_trades']} ({result['trades_per_month']:.1f}/mes)")
        if result['total_trades'] > 0:
            print(f"  Wins: {result['wins']} | Losses: {result['losses']}")
            print(f"  Win Rate: {result['win_rate']:.1f}%")
            print(f"  Breakeven: {result['breakeven']:.1f}%")
            print(f"  Edge: {result['edge']:+.1f}%")
            print(f"  PnL Total: {result['total_pnl']:+.0f} pips")
            print(f"  Profit Factor: {result['profit_factor']:.2f}")

            status = "[APROVADO]" if result['edge'] > 0 else "[REPROVADO]"
            print(f"\n  {status}")
        else:
            print("  NENHUM TRADE GERADO!")

    # Tabela final
    print("\n")
    print("=" * 90)
    print("  TABELA COMPARATIVA - DTT PARAMETROS REOTIMIZADOS")
    print("=" * 90)
    print()
    print(f"{'TF':<5} {'SL/TP':<10} {'Trades':<8} {'T/Mes':<7} {'Win%':<7} {'BE%':<7} {'Edge':<8} {'PnL':<9} {'PF':<6} {'Status'}")
    print("-" * 90)

    for r in results:
        status = "APROVADO" if r['edge'] > 0 else "REPROVADO"
        sl_tp = f"{r['sl_pips']:.0f}/{r['tp_pips']:.0f}"
        print(f"{r['timeframe']:<5} {sl_tp:<10} {r['total_trades']:<8} {r['trades_per_month']:.1f}{'':<3} {r['win_rate']:.1f}%{'':<2} {r['breakeven']:.1f}%{'':<2} {r['edge']:+.1f}%{'':<3} {r['total_pnl']:+.0f}{'':<4} {r['profit_factor']:.2f}{'':<2} {status}")

    # Conclusao
    print("\n" + "=" * 90)
    print("  CONCLUSAO")
    print("=" * 90)

    approved = [r for r in results if r['edge'] > 0]

    if approved:
        best = max(approved, key=lambda x: x['edge'])
        print(f"\n  MELHOR TIMEFRAME: {best['timeframe']}")
        print(f"    - Edge: {best['edge']:+.1f}%")
        print(f"    - Trades/mes: {best['trades_per_month']:.1f}")
        print(f"    - Profit Factor: {best['profit_factor']:.2f}")
        print(f"    - SL/TP: {best['sl_pips']:.0f}/{best['tp_pips']:.0f} pips")
    else:
        print("\n  NENHUM TIMEFRAME COM EDGE POSITIVO!")
        if results:
            best = max(results, key=lambda x: x['edge'])
            print(f"  Melhor: {best['timeframe']} com edge {best['edge']:+.1f}%")

    print("\n" + "=" * 90)

if __name__ == "__main__":
    main()
