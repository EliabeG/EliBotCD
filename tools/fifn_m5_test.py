#!/usr/bin/env python3
"""
FIFN M5 Test - Teste do Fluxo de Informacao Fisher-Navier com filtros tecnicos
===============================================================================
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
# PARAMETROS
# =============================================================================

# Parametros do config otimizado fifn-fishernavier_robust.json
SL_PIPS = 22.5
TP_PIPS = 27.9
LONG_ONLY_MODE = False  # Testar ambas direcoes
COOLDOWN = 2  # Barras entre sinais (2 horas)
MIN_WARMUP = 100
SPREAD_PIPS = 1.2
SLIPPAGE_PIPS = 0.5
PIP_VALUE = 0.0001

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
class Trade:
    entry_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    pnl_pips: float
    result: str

def download_bars(period: str, count: int) -> List[Bar]:
    """Download barras da API."""
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

def run_backtest(bars: List[Bar], use_filters: bool = True, verbose: bool = False) -> dict:
    """Executa backtest do FIFN."""

    # Criar indicador FIFN com parametros otimizados (fifn-fishernavier_robust.json)
    fifn = FluxoInformacaoFisherNavier(
        window_size=30,  # Reduzido de 50 para velocidade
        kl_lookback=8,   # Reduzido de 10
        reynolds_sweet_low=2521.0,   # Do config otimizado
        reynolds_sweet_high=3786.0,  # Do config otimizado
        skewness_threshold=0.3091    # Do config otimizado
    )

    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    n = len(bars)

    trades = []
    last_signal = -999
    equity = 0
    peak = 0
    max_dd = 0

    print(f"  Testando {n} barras com filtros={'ON' if use_filters else 'OFF'}...")

    signals_count = 0
    sweet_spot_count = 0

    for i in range(MIN_WARMUP, n - 1, 36):  # Step 36 para velocidade (FIFN muito caro)
        if i - last_signal < COOLDOWN:
            continue

        hour = bars[i].timestamp.hour

        try:
            # FIFN precisa de mais dados
            if i < fifn.window_size + fifn.kl_lookback + 20:
                continue

            result = fifn.analyze(
                prices=closes[:i+1],
                highs=highs[:i+1] if use_filters else None,
                lows=lows[:i+1] if use_filters else None,
                current_hour=hour,
                use_filters=use_filters
            )

            # Debug: contar sinais e sweet spots
            in_sweet_spot = result.get('reynolds_classification', {}).get('in_sweet_spot', False)
            fifn_sig = result.get('fifn_signal', 0)
            trade_on = result.get('trade_on', False)

            if in_sweet_spot:
                sweet_spot_count += 1
            if fifn_sig != 0:
                signals_count += 1
                # Debug quando ha sinal
                direction = result.get('direction', '')
                if verbose:
                    print(f"    Sinal em i={i}: fifn={fifn_sig}, direction={direction}, sweet_spot={in_sweet_spot}, trade_on={trade_on}")

            if not trade_on:
                continue

            direction = result.get('direction', '')
            # FIFN retorna 'COMPRA'/'VENDA' ou 'LONG'/'SHORT'
            if direction in ('LONG', 'COMPRA'):
                signal = 1
            elif direction in ('SHORT', 'VENDA') and not LONG_ONLY_MODE:
                signal = -1
            else:
                if direction in ('LONG', 'COMPRA'):
                    signal = 1
                else:
                    continue

            # Verificar filtros se ativados
            if use_filters:
                filters = result.get('filters', {})
                if filters and filters.get('total_score', 0) < 0.5:
                    continue

        except Exception as e:
            if verbose:
                print(f"    Erro em i={i}: {e}")
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
        for j in range(i + 2, min(i + 200, n)):
            bar = bars[j]
            if signal == 1:
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
            trades.append(Trade(
                bars[i].timestamp,
                'LONG' if signal == 1 else 'SHORT',
                entry,
                exit_price,
                trade_result[1],
                trade_result[0]
            ))

            equity += trade_result[1]
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

            if verbose:
                print(f"    Trade {len(trades)}: {trade_result[0]} {trade_result[1]:+.1f} pips @ {bars[i].timestamp}")

        last_signal = i

    # Debug output
    print(f"    Sweet spot count: {sweet_spot_count}")
    print(f"    FIFN signals count: {signals_count}")
    print(f"    Trades executados: {len(trades)}")

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

    return {
        'total': total,
        'wins': wins,
        'wr': wr,
        'be': be,
        'edge': edge,
        'pnl': pnl,
        'pf': pf,
        'dd': max_dd,
        'trades': trades
    }

def main():
    print("=" * 70)
    print("  FIFN TEST - Fluxo de Informacao Fisher-Navier")
    print("=" * 70)

    # Usar H1 para validacao - 4000 barras (~5 meses)
    bars = download_bars("H1", 4000)

    if len(bars) < 1000:
        print(f"  ERRO: Apenas {len(bars)} barras")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"\n  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()}")
    print(f"  Duracao: {days} dias")
    print(f"  Barras: {len(bars)}")

    # Teste sem filtros
    print("\n" + "-" * 70)
    print("  TESTE SEM FILTROS (FIFN puro):")
    print("-" * 70)

    result_no_filter = run_backtest(bars, use_filters=False, verbose=False)

    if result_no_filter['total'] > 0:
        print(f"  Trades: {result_no_filter['total']}")
        print(f"  Win Rate: {result_no_filter['wr']:.1f}% (BE: {result_no_filter['be']:.1f}%)")
        print(f"  Edge: {result_no_filter['edge']:+.1f}%")
        print(f"  PnL: {result_no_filter['pnl']:+.1f} pips")
        print(f"  Profit Factor: {result_no_filter['pf']:.2f}")
        print(f"  Max DD: {result_no_filter['dd']:.1f} pips")
    else:
        print("  Nenhum trade")

    # Teste com filtros
    print("\n" + "-" * 70)
    print("  TESTE COM FILTROS (FIFN + ATR/EMA/RSI/Sessao):")
    print("-" * 70)

    result_filter = run_backtest(bars, use_filters=True, verbose=False)

    if result_filter['total'] > 0:
        print(f"  Trades: {result_filter['total']}")
        print(f"  Win Rate: {result_filter['wr']:.1f}% (BE: {result_filter['be']:.1f}%)")
        print(f"  Edge: {result_filter['edge']:+.1f}%")
        print(f"  PnL: {result_filter['pnl']:+.1f} pips")
        print(f"  Profit Factor: {result_filter['pf']:.2f}")
        print(f"  Max DD: {result_filter['dd']:.1f} pips")

        # Mostrar trades
        if result_filter['trades']:
            print("\n  Ultimos 5 trades:")
            for t in result_filter['trades'][-5:]:
                print(f"    {t.entry_time.strftime('%Y-%m-%d %H:%M')} | {t.direction:5s} | {t.result:4s} | {t.pnl_pips:+.1f} pips")
    else:
        print("  Nenhum trade")

    # Veredicto
    print("\n" + "=" * 70)
    print("  VEREDICTO:")
    print("=" * 70)

    if result_filter['total'] > 0 and result_filter['edge'] > 0:
        print(f"  [OK] Edge positivo: {result_filter['edge']:+.1f}%")
    else:
        print(f"  [FAIL] Edge: {result_filter['edge'] if result_filter['total'] > 0 else 0:.1f}%")

    if result_filter['pf'] > 1.0:
        print(f"  [OK] Profit Factor > 1: {result_filter['pf']:.2f}")
    else:
        print(f"  [FAIL] Profit Factor: {result_filter['pf']:.2f}")

    if result_filter['total'] >= 5:
        print(f"  [OK] Sample size adequado: {result_filter['total']} trades")
    else:
        print(f"  [WARN] Sample size pequeno: {result_filter['total']} trades")

    print("=" * 70)

if __name__ == "__main__":
    main()
