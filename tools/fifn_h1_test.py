#!/usr/bin/env python3
"""
FIFN H1 - 1 Year Comprehensive Test (LONG-ONLY)
===============================================================================
Teste do timeframe H1 para comparacao com M30.
"""

import sys
import os
import urllib.request
import ssl
import json
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.alta_volatilidade.fifn_fisher_navier import FluxoInformacaoFisherNavier

# =============================================================================
# PARAMETROS H1
# =============================================================================

SL_PIPS = 20.0
TP_PIPS = 40.0
WINDOW_SIZE = 40
COOLDOWN = 2  # 2 horas entre trades
SPREAD_PIPS = 1.2
SLIPPAGE_PIPS = 0.5
PIP_VALUE = 0.0001
LONG_ONLY = True

REYNOLDS_LOW = 2521.0
REYNOLDS_HIGH = 3786.0
SKEWNESS_THRESHOLD = 0.3091

API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"

BARS_TARGET = 9000  # ~1 year of H1 data

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

    print(f"  Baixando {count} barras {period}...")
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
                    print(f"    {total} barras...")
                time.sleep(0.1)
        except Exception as e:
            print(f"  Erro: {e}")
            break

    bars.sort(key=lambda x: x.timestamp)
    return bars

def run_backtest(bars: List[Bar], use_filters: bool = True) -> dict:
    """Executa backtest completo."""
    n = len(bars)
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])

    # FIFN
    fifn = FluxoInformacaoFisherNavier(
        window_size=WINDOW_SIZE,
        kl_lookback=10,
        reynolds_sweet_low=REYNOLDS_LOW,
        reynolds_sweet_high=REYNOLDS_HIGH,
        skewness_threshold=SKEWNESS_THRESHOLD
    )

    try:
        result = fifn.analyze(closes)
        reynolds_series = result['reynolds_series']
        pressure_gradient = result['pressure_gradient_series']

        from scipy import stats
        returns = np.diff(np.log(closes))
        skewness_arr = np.zeros(len(returns))
        for i in range(WINDOW_SIZE, len(returns)):
            skewness_arr[i] = stats.skew(returns[i - WINDOW_SIZE:i])

    except Exception as e:
        return {'total': 0, 'error': str(e)}

    # EMA e RSI
    ema_12 = np.zeros(n)
    ema_26 = np.zeros(n)
    rsi_14 = np.full(n, 50.0)

    alpha_12 = 2.0 / 13
    alpha_26 = 2.0 / 27
    if n > 12:
        ema_12[12] = np.mean(closes[:12])
    if n > 26:
        ema_26[26] = np.mean(closes[:26])
    for i in range(13, n):
        ema_12[i] = alpha_12 * closes[i-1] + (1 - alpha_12) * ema_12[i-1]
    for i in range(27, n):
        ema_26[i] = alpha_26 * closes[i-1] + (1 - alpha_26) * ema_26[i-1]

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    for i in range(15, n):
        avg_gain = np.mean(gains[max(0, i-14):i])
        avg_loss = np.mean(losses[max(0, i-14):i])
        if avg_loss == 0:
            rsi_14[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_14[i] = 100.0 - (100.0 / (1.0 + rs))

    # Gerar sinais
    signals = []
    for i in range(100, n - 1):
        re = reynolds_series[i] if i < len(reynolds_series) else 0
        skew = skewness_arr[i] if i < len(skewness_arr) else 0
        pg = pressure_gradient[i] if i < len(pressure_gradient) else 0

        in_sweet_spot = REYNOLDS_LOW <= re <= REYNOLDS_HIGH
        if not in_sweet_spot:
            continue

        if use_filters:
            trend_up = ema_12[i] > ema_26[i]
            trend_down = ema_12[i] < ema_26[i]
            rsi = rsi_14[i]
            hour = bars[i].timestamp.hour
            session_ok = 8 <= hour <= 20

            if pg < 0 and skew > SKEWNESS_THRESHOLD:
                if session_ok and (trend_up or abs(ema_12[i] - ema_26[i]) < 0.001) and 35 <= rsi <= 75:
                    signals.append((i, 'LONG'))
            elif pg > 0 and skew < -SKEWNESS_THRESHOLD:
                if not LONG_ONLY:
                    if session_ok and (trend_down or abs(ema_12[i] - ema_26[i]) < 0.001) and 25 <= rsi <= 65:
                        signals.append((i, 'SHORT'))
        else:
            if pg < 0 and skew > SKEWNESS_THRESHOLD:
                signals.append((i, 'LONG'))
            elif pg > 0 and skew < -SKEWNESS_THRESHOLD:
                if not LONG_ONLY:
                    signals.append((i, 'SHORT'))

    # Simular trades
    trades = []
    trade_details = []
    last_trade_idx = -999

    for idx, direction in signals:
        if idx - last_trade_idx < COOLDOWN:
            continue
        if idx + 1 >= n:
            continue

        entry = bars[idx + 1].open
        sl = entry - SL_PIPS * PIP_VALUE if direction == 'LONG' else entry + SL_PIPS * PIP_VALUE
        tp = entry + TP_PIPS * PIP_VALUE if direction == 'LONG' else entry - TP_PIPS * PIP_VALUE

        for j in range(idx + 2, min(idx + 100, n)):
            bar = bars[j]
            if direction == 'LONG':
                if bar.low <= sl:
                    pnl = -SL_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS
                    trades.append(pnl)
                    trade_details.append({'date': bars[idx].timestamp, 'dir': direction, 'pnl': pnl})
                    break
                if bar.high >= tp:
                    pnl = TP_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS
                    trades.append(pnl)
                    trade_details.append({'date': bars[idx].timestamp, 'dir': direction, 'pnl': pnl})
                    break
            else:
                if bar.high >= sl:
                    pnl = -SL_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS
                    trades.append(pnl)
                    trade_details.append({'date': bars[idx].timestamp, 'dir': direction, 'pnl': pnl})
                    break
                if bar.low <= tp:
                    pnl = TP_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS
                    trades.append(pnl)
                    trade_details.append({'date': bars[idx].timestamp, 'dir': direction, 'pnl': pnl})
                    break
        last_trade_idx = idx

    if not trades:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'details': []}

    wins = len([t for t in trades if t > 0])
    total = len(trades)
    wr = wins / total * 100
    be = SL_PIPS / (SL_PIPS + TP_PIPS) * 100
    edge = wr - be
    pnl = sum(trades)
    gp = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))
    pf = gp / gl if gl > 0 else 0

    # Max Drawdown
    cumsum = np.cumsum(trades)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

    return {
        'total': total,
        'wins': wins,
        'wr': wr,
        'be': be,
        'edge': edge,
        'pnl': pnl,
        'pf': pf,
        'max_dd': max_dd,
        'details': trade_details
    }

def run_fold(bars: List[Bar], start_pct: float, end_pct: float) -> dict:
    """Executa um fold do walk-forward."""
    n = len(bars)
    start_idx = int(n * start_pct / 100)
    end_idx = int(n * end_pct / 100)

    fold_bars = bars[start_idx:end_idx]
    if len(fold_bars) < 200:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0}

    return run_backtest(fold_bars, use_filters=True)

def main():
    print("=" * 80)
    print("  FIFN H1 - 1 YEAR COMPREHENSIVE TEST (LONG-ONLY)")
    print("=" * 80)

    bars = download_bars("H1", BARS_TARGET)
    if len(bars) < 1000:
        print(f"  ERRO: Apenas {len(bars)} barras")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"\n  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()}")
    print(f"  Duracao: {days} dias (~{days/30:.1f} meses)")
    print(f"  Barras: {len(bars)}")

    # Backtest completo
    print("\n" + "=" * 80)
    print("  BACKTEST COMPLETO")
    print("=" * 80)

    print("\n  Sem filtros...")
    result_no_filter = run_backtest(bars, use_filters=False)

    print("  Com filtros...")
    result_filter = run_backtest(bars, use_filters=True)

    print("\n  COMPARACAO:")
    print("  " + "-" * 70)
    print(f"  {'Metrica':<20} | {'Sem Filtros':>15} | {'Com Filtros':>15}")
    print("  " + "-" * 70)
    print(f"  {'Trades':<20} | {result_no_filter['total']:>15} | {result_filter['total']:>15}")
    print(f"  {'Win Rate':<20} | {result_no_filter['wr']:>14.1f}% | {result_filter['wr']:>14.1f}%")
    print(f"  {'Breakeven':<20} | {result_no_filter['be']:>14.1f}% | {result_filter['be']:>14.1f}%")
    print(f"  {'Edge':<20} | {result_no_filter['edge']:>+14.1f}% | {result_filter['edge']:>+14.1f}%")
    print(f"  {'PnL (pips)':<20} | {result_no_filter['pnl']:>15.1f} | {result_filter['pnl']:>15.1f}")
    print(f"  {'Profit Factor':<20} | {result_no_filter['pf']:>15.2f} | {result_filter['pf']:>15.2f}")
    print(f"  {'Max Drawdown':<20} | {result_no_filter.get('max_dd', 0):>14.1f} | {result_filter.get('max_dd', 0):>14.1f}")

    # Walk-Forward
    print("\n" + "=" * 80)
    print("  WALK-FORWARD VALIDATION (8 FOLDS)")
    print("=" * 80)

    folds = []
    for i in range(8):
        start = i * 12.5
        end = (i + 1) * 12.5
        print(f"\n  Fold {i+1}/8: {start:.1f}%-{end:.1f}%...")
        result = run_fold(bars, start, end)
        folds.append(result)

        if result['total'] > 0:
            status = "[OK]" if result['edge'] > 0 else "[FAIL]"
            print(f"    Trades: {result['total']:>3} | WR: {result['wr']:>5.1f}% | Edge: {result['edge']:>+6.1f}% | PnL: {result['pnl']:>+8.1f} | PF: {result['pf']:.2f} {status}")
        else:
            print(f"    Nenhum trade")

    # Agregado
    total_trades = sum(f['total'] for f in folds)
    total_wins = sum(f.get('wins', 0) for f in folds)
    total_pnl = sum(f['pnl'] for f in folds)

    print("\n" + "-" * 70)
    if total_trades > 0:
        agg_wr = total_wins / total_trades * 100
        agg_be = SL_PIPS / (SL_PIPS + TP_PIPS) * 100
        agg_edge = agg_wr - agg_be
        gp = sum(f['pnl'] for f in folds if f['pnl'] > 0)
        gl = abs(sum(f['pnl'] for f in folds if f['pnl'] < 0))
        agg_pf = gp / gl if gl > 0 else 0
        print(f"  AGREGADO: {total_trades} trades | WR: {agg_wr:.1f}% | Edge: {agg_edge:+.1f}% | PnL: {total_pnl:+.1f} | PF: {agg_pf:.2f}")
        positive_folds = len([f for f in folds if f['edge'] > 0])
        print(f"  Folds positivos: {positive_folds}/8")

    # Analise Mensal
    print("\n" + "=" * 80)
    print("  ANALISE MENSAL")
    print("=" * 80)

    details = result_filter.get('details', [])
    monthly_pnl = {}
    for t in details:
        month_key = t['date'].strftime('%Y-%m')
        monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + t['pnl']

    print(f"\n  {'Mes':<10} | {'PnL (pips)':>12} | {'Status':<8}")
    print("  " + "-" * 40)

    profitable_months = 0
    for month in sorted(monthly_pnl.keys()):
        pnl = monthly_pnl[month]
        status = "[OK]" if pnl > 0 else "[LOSS]"
        if pnl > 0:
            profitable_months += 1
        print(f"  {month:<10} | {pnl:>+12.1f} | {status:<8}")

    total_months = len(monthly_pnl)
    print("  " + "-" * 40)
    print(f"  Meses lucrativos: {profitable_months}/{total_months}")
    if total_months > 0:
        print(f"  Percentual: {profitable_months/total_months*100:.0f}%")

    # Veredicto
    print("\n" + "=" * 80)
    print("  VEREDICTO FINAL")
    print("=" * 80)

    positive_folds = len([f for f in folds if f['edge'] > 0])
    agg_edge = (total_wins / total_trades * 100 - SL_PIPS / (SL_PIPS + TP_PIPS) * 100) if total_trades > 0 else 0
    gp = sum(f['pnl'] for f in folds if f['pnl'] > 0)
    gl = abs(sum(f['pnl'] for f in folds if f['pnl'] < 0))
    agg_pf = gp / gl if gl > 0 else 0

    criteria = [
        ("Edge agregado > 0%", agg_edge > 0),
        ("Profit Factor > 1.0", agg_pf > 1.0),
        (">=5 folds positivos (de 8)", positive_folds >= 5),
        (">=50% meses lucrativos", profitable_months >= total_months * 0.5 if total_months > 0 else False),
        ("Max Drawdown < 100 pips", result_filter.get('max_dd', 999) < 100),
        ("Total trades >= 30", total_trades >= 30),
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
        print("\n  *** REQUER MAIS AJUSTES ***")

    print("=" * 80)

if __name__ == "__main__":
    main()
