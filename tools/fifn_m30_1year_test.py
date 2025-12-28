#!/usr/bin/env python3
"""
FIFN M30 - 1 Year Comprehensive Test
===============================================================================
Teste completo do FIFN M30 com 1 ano de dados historicos.
Inclui walk-forward validation com 4 folds trimestrais.
"""

import sys
import os
import urllib.request
import ssl
import json
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.alta_volatilidade.fifn_fisher_navier import FluxoInformacaoFisherNavier

# =============================================================================
# PARAMETROS M30 OTIMIZADOS
# =============================================================================

SL_PIPS = 15.0
TP_PIPS = 30.0
WINDOW_SIZE = 40
COOLDOWN = 3
SPREAD_PIPS = 1.2
SLIPPAGE_PIPS = 0.5
PIP_VALUE = 0.0001

REYNOLDS_LOW = 2521.0
REYNOLDS_HIGH = 3786.0
SKEWNESS_THRESHOLD = 0.3091

# Modo de operacao
LONG_ONLY = True  # Testar apenas LONG

API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"

# M30: 48 barras/dia * 260 dias * 2 anos = ~25000 barras para 2 anos
BARS_2_YEARS = 26000

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
    entry_price: float
    exit_price: float
    pnl_pips: float
    result: str

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

def run_backtest(bars: List[Bar], start_pct: float = 0, end_pct: float = 100,
                 use_filters: bool = True) -> Dict:
    """Executa backtest em um periodo especifico."""

    n = len(bars)
    start_idx = int(n * start_pct / 100)
    end_idx = int(n * end_pct / 100)

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
        result = fifn.analyze(closes[:end_idx])
        reynolds_series = result['reynolds_series']
        pressure_gradient = result['pressure_gradient_series']

        from scipy import stats
        returns = np.diff(np.log(closes[:end_idx]))
        skewness_arr = np.zeros(len(returns))
        for i in range(WINDOW_SIZE, len(returns)):
            skewness_arr[i] = stats.skew(returns[i - WINDOW_SIZE:i])

    except Exception as e:
        return {'total': 0, 'error': str(e)}

    # Filtros tecnicos
    ema_12 = np.zeros(end_idx)
    ema_26 = np.zeros(end_idx)
    rsi_14 = np.full(end_idx, 50.0)

    alpha_12 = 2.0 / 13
    alpha_26 = 2.0 / 27
    if end_idx > 12:
        ema_12[12] = np.mean(closes[:12])
    if end_idx > 26:
        ema_26[26] = np.mean(closes[:26])
    for i in range(13, end_idx):
        ema_12[i] = alpha_12 * closes[i-1] + (1 - alpha_12) * ema_12[i-1]
    for i in range(27, end_idx):
        ema_26[i] = alpha_26 * closes[i-1] + (1 - alpha_26) * ema_26[i-1]

    deltas = np.diff(closes[:end_idx])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    for i in range(15, end_idx):
        avg_gain = np.mean(gains[max(0, i-14):i])
        avg_loss = np.mean(losses[max(0, i-14):i])
        if avg_loss == 0:
            rsi_14[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_14[i] = 100.0 - (100.0 / (1.0 + rs))

    # Gerar sinais
    signals = []
    for i in range(max(100, start_idx), end_idx - 1):
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
                    signals.append((i, 'LONG', bars[i].timestamp))
            elif pg > 0 and skew < -SKEWNESS_THRESHOLD and not LONG_ONLY:
                if session_ok and (trend_down or abs(ema_12[i] - ema_26[i]) < 0.001) and 25 <= rsi <= 65:
                    signals.append((i, 'SHORT', bars[i].timestamp))
        else:
            if pg < 0 and skew > SKEWNESS_THRESHOLD:
                signals.append((i, 'LONG', bars[i].timestamp))
            elif pg > 0 and skew < -SKEWNESS_THRESHOLD and not LONG_ONLY:
                signals.append((i, 'SHORT', bars[i].timestamp))

    # Simular trades
    trades = []
    last_trade_idx = -999
    equity = 0
    peak = 0
    max_dd = 0

    for idx, direction, timestamp in signals:
        if idx - last_trade_idx < COOLDOWN:
            continue
        if idx + 1 >= n:
            continue

        entry = bars[idx + 1].open
        sl = entry - SL_PIPS * PIP_VALUE if direction == 'LONG' else entry + SL_PIPS * PIP_VALUE
        tp = entry + TP_PIPS * PIP_VALUE if direction == 'LONG' else entry - TP_PIPS * PIP_VALUE

        trade_result = None
        exit_price = 0
        for j in range(idx + 2, min(idx + 200, n)):
            bar = bars[j]
            if direction == 'LONG':
                if bar.low <= sl:
                    trade_result = ('LOSS', -SL_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS)
                    exit_price = sl
                    break
                if bar.high >= tp:
                    trade_result = ('WIN', TP_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS)
                    exit_price = tp
                    break
            else:
                if bar.high >= sl:
                    trade_result = ('LOSS', -SL_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS)
                    exit_price = sl
                    break
                if bar.low <= tp:
                    trade_result = ('WIN', TP_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS)
                    exit_price = tp
                    break

        if trade_result:
            trades.append(Trade(timestamp, direction, entry, exit_price,
                               trade_result[1], trade_result[0]))
            equity += trade_result[1]
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        last_trade_idx = idx

    # Calcular metricas
    if not trades:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'dd': 0}

    wins = len([t for t in trades if t.result == 'WIN'])
    total = len(trades)
    wr = wins / total * 100
    be = SL_PIPS / (SL_PIPS + TP_PIPS) * 100
    edge = wr - be
    pnl = sum(t.pnl_pips for t in trades)
    gp = sum(t.pnl_pips for t in trades if t.pnl_pips > 0)
    gl = abs(sum(t.pnl_pips for t in trades if t.pnl_pips < 0))
    pf = gp / gl if gl > 0 else 0

    # Analise mensal
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
        'trades': trades,
        'monthly': monthly,
        'profitable_months': profitable_months,
        'total_months': len(monthly)
    }

def main():
    print("=" * 80)
    print("  FIFN M30 - 2 YEAR COMPREHENSIVE TEST")
    print("=" * 80)

    # Baixar 2 anos de dados M30
    bars = download_bars("M30", BARS_2_YEARS)

    if len(bars) < 5000:
        print(f"  ERRO: Apenas {len(bars)} barras")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"\n  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()}")
    print(f"  Duracao: {days} dias (~{days/30:.1f} meses)")
    print(f"  Barras: {len(bars)}")

    # ==========================================================================
    # TESTE COMPLETO (1 ANO)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  BACKTEST COMPLETO (1 ANO)")
    print("=" * 80)

    print("\n  Sem filtros...")
    result_raw = run_backtest(bars, 0, 100, use_filters=False)

    print("  Com filtros...")
    result_filtered = run_backtest(bars, 0, 100, use_filters=True)

    print("\n  COMPARACAO:")
    print("  " + "-" * 70)
    print(f"  {'Metrica':<20} | {'Sem Filtros':>15} | {'Com Filtros':>15}")
    print("  " + "-" * 70)

    if result_raw['total'] > 0:
        print(f"  {'Trades':<20} | {result_raw['total']:>15} | {result_filtered['total']:>15}")
        print(f"  {'Win Rate':<20} | {result_raw['wr']:>14.1f}% | {result_filtered['wr']:>14.1f}%")
        print(f"  {'Breakeven':<20} | {result_raw['be']:>14.1f}% | {result_filtered['be']:>14.1f}%")
        print(f"  {'Edge':<20} | {result_raw['edge']:>+14.1f}% | {result_filtered['edge']:>+14.1f}%")
        print(f"  {'PnL (pips)':<20} | {result_raw['pnl']:>+15.1f} | {result_filtered['pnl']:>+15.1f}")
        print(f"  {'Profit Factor':<20} | {result_raw['pf']:>15.2f} | {result_filtered['pf']:>15.2f}")
        print(f"  {'Max Drawdown':<20} | {result_raw['dd']:>14.1f} | {result_filtered['dd']:>14.1f}")

    # ==========================================================================
    # WALK-FORWARD VALIDATION (8 FOLDS TRIMESTRAIS - 2 ANOS)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  WALK-FORWARD VALIDATION (8 FOLDS TRIMESTRAIS - 2 ANOS)")
    print("=" * 80)

    folds = []
    for i in range(8):
        start = i * 12.5
        end = (i + 1) * 12.5
        print(f"\n  Fold {i+1}/8 (Q{(i%4)+1}/Y{(i//4)+1}): {start:.1f}%-{end:.1f}%...")
        result = run_backtest(bars, start, end, use_filters=True)
        folds.append(result)

        if result['total'] > 0:
            status = "[OK]" if result['edge'] > 0 else "[FAIL]"
            print(f"    Trades: {result['total']:3d} | WR: {result['wr']:5.1f}% | Edge: {result['edge']:+6.1f}% | PnL: {result['pnl']:+7.1f} | PF: {result['pf']:.2f} {status}")
        else:
            print(f"    Nenhum trade")

    # Agregado
    total_trades = sum(f['total'] for f in folds)
    total_wins = sum(f.get('wins', 0) for f in folds)
    total_pnl = sum(f['pnl'] for f in folds)
    positive_folds = len([f for f in folds if f['edge'] > 0])

    print("\n" + "-" * 70)
    if total_trades > 0:
        agg_wr = total_wins / total_trades * 100
        agg_be = SL_PIPS / (SL_PIPS + TP_PIPS) * 100
        agg_edge = agg_wr - agg_be
        gp = sum(f['pnl'] for f in folds if f['pnl'] > 0)
        gl = abs(sum(f['pnl'] for f in folds if f['pnl'] < 0))
        agg_pf = gp / gl if gl > 0 else 0
        print(f"  AGREGADO: {total_trades} trades | WR: {agg_wr:.1f}% | Edge: {agg_edge:+.1f}% | PnL: {total_pnl:+.1f} | PF: {agg_pf:.2f}")
        print(f"  Folds positivos: {positive_folds}/4")

    # ==========================================================================
    # ANALISE MENSAL
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  ANALISE MENSAL")
    print("=" * 80)

    if result_filtered['total'] > 0 and result_filtered.get('monthly'):
        monthly = result_filtered['monthly']
        print(f"\n  {'Mes':<10} | {'PnL (pips)':>12} | {'Status':<8}")
        print("  " + "-" * 40)

        for month in sorted(monthly.keys()):
            pnl = monthly[month]
            status = "OK" if pnl > 0 else "LOSS"
            print(f"  {month:<10} | {pnl:>+12.1f} | [{status}]")

        print("  " + "-" * 40)
        print(f"  Meses lucrativos: {result_filtered['profitable_months']}/{result_filtered['total_months']}")
        pct_profitable = result_filtered['profitable_months'] / result_filtered['total_months'] * 100 if result_filtered['total_months'] > 0 else 0
        print(f"  Percentual: {pct_profitable:.0f}%")

    # ==========================================================================
    # VEREDICTO FINAL
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  VEREDICTO FINAL")
    print("=" * 80)

    if result_filtered['total'] > 0:
        criteria = [
            ("Edge agregado > 0%", agg_edge > 0 if total_trades > 0 else False),
            ("Profit Factor > 1.0", result_filtered['pf'] > 1.0),
            (">=5 folds positivos (de 8)", positive_folds >= 5),
            (">=50% meses lucrativos", pct_profitable >= 50),
            ("Max Drawdown < 150 pips", result_filtered['dd'] < 150),
            ("Total trades >= 50", result_filtered['total'] >= 50),
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
