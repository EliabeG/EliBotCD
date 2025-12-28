#!/usr/bin/env python3
"""
================================================================================
DTT VALIDATION FAST - Versao Otimizada
================================================================================

Validacao em H1 (mais rapida) e depois verificacao em M5.
Usa parametros do DTT M5 aprovado mas testa em H1 para validacao mais rapida.

================================================================================
"""

import sys
import os
import urllib.request
import ssl
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import List, Dict
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.alta_volatilidade.dtt_tunelamento_topologico import DetectorTunelamentoTopologico

# =============================================================================
# PARAMETROS
# =============================================================================

PERSISTENCE_ENTROPY_THRESHOLD = 0.5
TUNNELING_PROBABILITY_THRESHOLD = 0.15
MIN_SIGNAL_STRENGTH = 0.35

# Para H1: SL/TP proporcionalmente maiores
SL_PIPS = 20.0   # 2.5x do M5
TP_PIPS = 50.0   # 2.5x do M5

LONG_ONLY_MODE = True
COOLDOWN = 2  # 2 horas entre trades
MIN_WARMUP = 100
SPREAD_PIPS = 1.2
SLIPPAGE_PIPS = 0.5
PIP_VALUE = 0.0001

API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"

# Datas de noticias 2025
NEWS_DATES = {
    "2025-01-03", "2025-01-29", "2025-02-07", "2025-03-07", "2025-03-19",
    "2025-04-04", "2025-05-02", "2025-05-07", "2025-06-06", "2025-06-18",
    "2025-07-03", "2025-07-30", "2025-08-01", "2025-09-05", "2025-09-17",
    "2025-10-03", "2025-11-05", "2025-11-07", "2025-12-05", "2025-12-17"
}

@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class Trade:
    entry_time: datetime
    direction: str
    pnl_pips: float
    result: str
    is_news_day: bool = False

def download_bars(period: str, count: int) -> List[Bar]:
    """Download barras da API."""
    bars = []
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    current_ts = int(time.time() * 1000)
    total = 0

    print(f"    Baixando {count} barras {period}...")

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

                if total % 2000 == 0:
                    print(f"      {total} barras...")
                time.sleep(0.15)
        except Exception as e:
            print(f"    Erro: {e}")
            break

    bars.sort(key=lambda x: x.timestamp)
    return bars

def run_backtest(bars: List[Bar], start_pct: float = 0, end_pct: float = 100) -> Dict:
    """Executa backtest em um periodo."""

    n = len(bars)
    start_idx = int(n * start_pct / 100)
    end_idx = int(n * end_pct / 100)

    dtt = DetectorTunelamentoTopologico(
        max_points=80,  # Menor para velocidade
        use_dimensionality_reduction=True,
        persistence_entropy_threshold=PERSISTENCE_ENTROPY_THRESHOLD,
        tunneling_probability_threshold=TUNNELING_PROBABILITY_THRESHOLD,
        auto_calibrate_quantum=True
    )

    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])

    trades = []
    last_signal = -999
    equity = 0
    peak = 0
    max_dd = 0
    losing_streak = 0
    max_losing_streak = 0

    for i in range(max(start_idx, MIN_WARMUP), min(end_idx, n - 1)):
        if i - last_signal < COOLDOWN:
            continue

        hour = bars[i].timestamp.hour
        date_str = bars[i].timestamp.strftime("%Y-%m-%d")
        is_news = date_str in NEWS_DATES

        try:
            result = dtt.analyze(
                prices=closes[:i+1],
                highs=highs[:i+1],
                lows=lows[:i+1],
                current_hour=hour,
                use_filters=True
            )

            if not result or not result.get('trade_on', False):
                continue

            if result.get('entropy', {}).get('persistence_entropy', 0) < PERSISTENCE_ENTROPY_THRESHOLD:
                continue
            if result.get('signal_strength', 0) < MIN_SIGNAL_STRENGTH:
                continue

            direction = result.get('direction', '')
            if direction == 'LONG':
                signal = 1
            elif direction == 'SHORT' and not LONG_ONLY_MODE:
                signal = -1
            else:
                continue

            filters = result.get('filters', {})
            if filters and (not filters.get('filters_ok', True) or filters.get('total_score', 0) < 0.5):
                continue

        except:
            continue

        # Simular trade
        entry = bars[i + 1].open
        if signal == 1:
            sl = entry - SL_PIPS * PIP_VALUE
            tp = entry + TP_PIPS * PIP_VALUE
        else:
            sl = entry + SL_PIPS * PIP_VALUE
            tp = entry - TP_PIPS * PIP_VALUE

        trade_result = None
        for j in range(i + 2, min(i + 100, n)):
            bar = bars[j]
            if signal == 1:
                if bar.low <= sl:
                    trade_result = ('LOSS', -SL_PIPS - SPREAD_PIPS)
                    break
                if bar.high >= tp:
                    trade_result = ('WIN', TP_PIPS - SPREAD_PIPS)
                    break
            else:
                if bar.high >= sl:
                    trade_result = ('LOSS', -SL_PIPS - SPREAD_PIPS)
                    break
                if bar.low <= tp:
                    trade_result = ('WIN', TP_PIPS - SPREAD_PIPS)
                    break

        if trade_result:
            trades.append(Trade(
                bars[i].timestamp,
                'LONG' if signal == 1 else 'SHORT',
                trade_result[1],
                trade_result[0],
                is_news
            ))

            equity += trade_result[1]
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

            if trade_result[0] == 'LOSS':
                losing_streak += 1
                max_losing_streak = max(max_losing_streak, losing_streak)
            else:
                losing_streak = 0

        last_signal = i

    # Calcular metricas
    if not trades:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'dd': 0, 'streak': 0}

    wins = len([t for t in trades if t.result == 'WIN'])
    total = len(trades)
    wr = wins / total * 100
    be = SL_PIPS / (SL_PIPS + TP_PIPS) * 100
    edge = wr - be
    pnl = sum(t.pnl_pips for t in trades)

    gp = sum(t.pnl_pips for t in trades if t.pnl_pips > 0)
    gl = abs(sum(t.pnl_pips for t in trades if t.pnl_pips < 0))
    pf = gp / gl if gl > 0 else 0

    # News analysis
    news_trades = [t for t in trades if t.is_news_day]
    news_pnl = sum(t.pnl_pips for t in news_trades)

    # Monthly
    monthly = {}
    for t in trades:
        m = t.entry_time.strftime("%Y-%m")
        if m not in monthly:
            monthly[m] = 0
        monthly[m] += t.pnl_pips

    profitable_months = len([v for v in monthly.values() if v > 0])

    return {
        'total': total,
        'wins': wins,
        'wr': wr,
        'be': be,
        'edge': edge,
        'pnl': pnl,
        'pf': pf,
        'dd': max_dd,
        'streak': max_losing_streak,
        'news_trades': len(news_trades),
        'news_pnl': news_pnl,
        'months': len(monthly),
        'profitable_months': profitable_months,
        'trades': trades
    }

def main():
    print("=" * 70)
    print("  DTT VALIDATION - H1 (12 meses)")
    print("=" * 70)

    # H1: 24 barras/dia * 260 dias = ~6240 barras para 12 meses
    bars = download_bars("H1", 8000)  # ~13 meses

    if len(bars) < 1000:
        print(f"  ERRO: Apenas {len(bars)} barras")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"\n  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()}")
    print(f"  Duracao: {days} dias ({days/30:.1f} meses)")
    print(f"  Barras: {len(bars)}")

    # Walk-Forward: 4 folds (25% cada)
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (4 FOLDS)")
    print("=" * 70)

    folds = []
    for i in range(4):
        start = i * 25
        end = (i + 1) * 25
        print(f"\n  Fold {i+1}/4: {start}%-{end}%...")
        result = run_backtest(bars, start, end)
        folds.append(result)

        if result['total'] > 0:
            status = "[OK]" if result['edge'] > 0 else "[FAIL]"
            print(f"    Trades: {result['total']} | WR: {result['wr']:.1f}% | Edge: {result['edge']:+.1f}% | PnL: {result['pnl']:+.1f} {status}")
        else:
            print(f"    Nenhum trade")

    # Agregado
    total_trades = sum(f['total'] for f in folds)
    total_wins = sum(f['wins'] for f in folds)
    total_pnl = sum(f['pnl'] for f in folds)
    all_trades = []
    for f in folds:
        all_trades.extend(f.get('trades', []))

    print("\n" + "-" * 70)
    if total_trades > 0:
        agg_wr = total_wins / total_trades * 100
        agg_be = SL_PIPS / (SL_PIPS + TP_PIPS) * 100
        agg_edge = agg_wr - agg_be
        print(f"  AGREGADO: {total_trades} trades | WR: {agg_wr:.1f}% | Edge: {agg_edge:+.1f}% | PnL: {total_pnl:+.1f}")

    # Full backtest para metricas completas
    print("\n" + "=" * 70)
    print("  BACKTEST COMPLETO")
    print("=" * 70)

    full = run_backtest(bars, 0, 100)

    print(f"\n  METRICAS PRINCIPAIS:")
    print(f"  {'-' * 50}")
    print(f"  Trades: {full['total']}")
    print(f"  Win Rate: {full['wr']:.1f}% (BE: {full['be']:.1f}%)")
    print(f"  Edge: {full['edge']:+.1f}%")
    print(f"  PnL Total: {full['pnl']:+.1f} pips")
    print(f"  Profit Factor: {full['pf']:.2f}")

    print(f"\n  DRAWDOWN:")
    print(f"  {'-' * 50}")
    print(f"  Max Drawdown: {full['dd']:.1f} pips")
    print(f"  Longest Losing Streak: {full['streak']} trades")

    print(f"\n  STRESS TEST (Noticias):")
    print(f"  {'-' * 50}")
    print(f"  Trades em dias NFP/FOMC: {full['news_trades']}")
    print(f"  PnL em dias de noticias: {full['news_pnl']:+.1f} pips")

    print(f"\n  ANALISE MENSAL:")
    print(f"  {'-' * 50}")
    print(f"  Meses analisados: {full['months']}")
    print(f"  Meses lucrativos: {full['profitable_months']} ({full['profitable_months']/max(1,full['months'])*100:.0f}%)")

    # Veredicto
    print("\n" + "=" * 70)
    print("  VEREDICTO")
    print("=" * 70)

    criteria = [
        ("Edge WF > 0", agg_edge > 0 if total_trades > 0 else False),
        ("PF > 1.0", full['pf'] > 1.0),
        ("DD < 100 pips", full['dd'] < 100),
        (">=50% Folds OK", len([f for f in folds if f['edge'] > 0]) >= 2),
        (">=50% Meses OK", full['profitable_months'] >= full['months'] * 0.5),
        ("News PnL >= -30", full['news_pnl'] >= -30),
    ]

    passed = 0
    for name, ok in criteria:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")
        if ok:
            passed += 1

    print(f"\n  Resultado: {passed}/{len(criteria)} criterios")

    if passed >= 5:
        print("\n  *** APROVADO PARA PAPER TRADING ***")
    elif passed >= 4:
        print("\n  *** APROVADO COM RESSALVAS ***")
    else:
        print("\n  *** REPROVADO ***")

    print("=" * 70)

if __name__ == "__main__":
    main()
