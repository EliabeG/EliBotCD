#!/usr/bin/env python3
"""
================================================================================
DTT OUT-OF-SAMPLE VALIDATION (OOS)
================================================================================

Valida DTT com separacao temporal ESTRITA:
- Treino: Ago-Out 2025 (3 meses)
- Teste: Nov-Dez 2025 (2 meses) - DADOS NAO VISTOS na otimizacao

Este e o teste mais importante para detectar overfitting.

================================================================================
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
# PARAMETROS OTIMIZADOS (do Grid Search em Ago-Dez)
# =============================================================================

PERSISTENCE_ENTROPY_THRESHOLD = 0.5
TUNNELING_PROBABILITY_THRESHOLD = 0.35
MIN_SIGNAL_STRENGTH = 0.2
SL_PIPS = 20.0
TP_PIPS = 40.0

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

def run_backtest(bars: List[Bar], description: str) -> dict:
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

    print(f"\n    Processando {len(bars)} barras [{description}]...")

    for i in range(MIN_WARMUP, len(bars) - 1):
        if i - last_signal < COOLDOWN:
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
            sl = entry - SL_PIPS * PIP_VALUE
            tp = entry + TP_PIPS * PIP_VALUE
        else:
            entry -= SLIPPAGE_PIPS * PIP_VALUE
            sl = entry + SL_PIPS * PIP_VALUE
            tp = entry - TP_PIPS * PIP_VALUE

        for j in range(i + 2, min(i + 200, len(bars))):
            bar = bars[j]

            if signal == 1:
                if bar.low <= sl:
                    trades.append(('LOSS', -SL_PIPS - SPREAD_PIPS, bars[i].timestamp))
                    break
                if bar.high >= tp:
                    trades.append(('WIN', TP_PIPS - SPREAD_PIPS, bars[i].timestamp))
                    break
            else:
                if bar.high >= sl:
                    trades.append(('LOSS', -SL_PIPS - SPREAD_PIPS, bars[i].timestamp))
                    break
                if bar.low <= tp:
                    trades.append(('WIN', TP_PIPS - SPREAD_PIPS, bars[i].timestamp))
                    break

        last_signal = i

    if not trades:
        return {
            'description': description,
            'total_trades': 0,
            'win_rate': 0,
            'breakeven': SL_PIPS / (SL_PIPS + TP_PIPS) * 100,
            'edge': 0,
            'total_pnl': 0,
            'profit_factor': 0
        }

    wins = len([t for t in trades if t[0] == 'WIN'])
    total = len(trades)
    win_rate = wins / total * 100

    breakeven = SL_PIPS / (SL_PIPS + TP_PIPS) * 100
    edge = win_rate - breakeven

    total_pnl = sum(t[1] for t in trades)

    gross_profit = sum(t[1] for t in trades if t[1] > 0)
    gross_loss = abs(sum(t[1] for t in trades if t[1] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    return {
        'description': description,
        'total_trades': total,
        'wins': wins,
        'losses': total - wins,
        'win_rate': win_rate,
        'breakeven': breakeven,
        'edge': edge,
        'total_pnl': total_pnl,
        'profit_factor': pf,
        'trades': trades
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("  DTT OUT-OF-SAMPLE VALIDATION")
    print("=" * 80)
    print()
    print("  OBJETIVO: Validar se os parametros otimizados funcionam em dados NAO VISTOS")
    print()
    print("  PARAMETROS (otimizados em Ago-Dez 2025):")
    print(f"    persistence_entropy >= {PERSISTENCE_ENTROPY_THRESHOLD}")
    print(f"    tunneling_probability >= {TUNNELING_PROBABILITY_THRESHOLD}")
    print(f"    signal_strength >= {MIN_SIGNAL_STRENGTH}")
    print(f"    SL/TP: {SL_PIPS}/{TP_PIPS} pips")
    print()

    # Baixar dados
    bars = download_real_bars("H1", 2500)

    if len(bars) < 500:
        print("  ERRO: Dados insuficientes!")
        return

    # Separar em periodos
    # Encontrar data de corte (1 Nov 2025)
    cutoff_date = datetime(2025, 11, 1, tzinfo=timezone.utc)

    train_bars = [b for b in bars if b.timestamp < cutoff_date]
    test_bars = [b for b in bars if b.timestamp >= cutoff_date]

    print(f"\n  SEPARACAO TEMPORAL:")
    print(f"    TREINO: {len(train_bars)} barras")
    if train_bars:
        print(f"            {train_bars[0].timestamp.strftime('%Y-%m-%d')} a {train_bars[-1].timestamp.strftime('%Y-%m-%d')}")
    print(f"    TESTE:  {len(test_bars)} barras")
    if test_bars:
        print(f"            {test_bars[0].timestamp.strftime('%Y-%m-%d')} a {test_bars[-1].timestamp.strftime('%Y-%m-%d')}")
    print()

    results = []

    # Testar em TREINO (In-Sample)
    if len(train_bars) > MIN_WARMUP + 100:
        result_train = run_backtest(train_bars, "TREINO (In-Sample)")
        results.append(result_train)

        print(f"\n  RESULTADO TREINO (In-Sample):")
        print(f"  {'-'*55}")
        print(f"  Trades: {result_train['total_trades']}")
        if result_train['total_trades'] > 0:
            print(f"  Win Rate: {result_train['win_rate']:.1f}%")
            print(f"  Breakeven: {result_train['breakeven']:.1f}%")
            print(f"  Edge: {result_train['edge']:+.1f}%")
            print(f"  PnL: {result_train['total_pnl']:+.0f} pips")
            print(f"  PF: {result_train['profit_factor']:.2f}")

    # Testar em TESTE (Out-of-Sample)
    if len(test_bars) > MIN_WARMUP + 100:
        result_test = run_backtest(test_bars, "TESTE (Out-of-Sample)")
        results.append(result_test)

        print(f"\n  RESULTADO TESTE (Out-of-Sample):")
        print(f"  {'-'*55}")
        print(f"  Trades: {result_test['total_trades']}")
        if result_test['total_trades'] > 0:
            print(f"  Win Rate: {result_test['win_rate']:.1f}%")
            print(f"  Breakeven: {result_test['breakeven']:.1f}%")
            print(f"  Edge: {result_test['edge']:+.1f}%")
            print(f"  PnL: {result_test['total_pnl']:+.0f} pips")
            print(f"  PF: {result_test['profit_factor']:.2f}")
    else:
        print("\n  AVISO: Dados de teste insuficientes!")

    # Comparacao
    print(f"\n{'='*80}")
    print("  COMPARACAO IN-SAMPLE vs OUT-OF-SAMPLE")
    print(f"{'='*80}")
    print()
    print(f"{'Periodo':<25} {'Trades':<8} {'Win%':<8} {'Edge':<10} {'PnL':<10} {'PF':<8} {'Status'}")
    print("-" * 80)

    for r in results:
        status = "APROVADO" if r['edge'] > 0 else "REPROVADO"
        print(f"{r['description']:<25} {r['total_trades']:<8} {r['win_rate']:.1f}%{'':<3} {r['edge']:+.1f}%{'':<5} {r['total_pnl']:+.0f}{'':<5} {r['profit_factor']:.2f}{'':<4} {status}")

    # Analise de degradacao
    print(f"\n{'='*80}")
    print("  ANALISE DE DEGRADACAO")
    print(f"{'='*80}")

    if len(results) >= 2:
        train_r = results[0]
        test_r = results[1]

        if train_r['total_trades'] > 0 and test_r['total_trades'] > 0:
            edge_degradation = train_r['edge'] - test_r['edge']
            wr_degradation = train_r['win_rate'] - test_r['win_rate']
            pf_ratio = test_r['profit_factor'] / train_r['profit_factor'] if train_r['profit_factor'] > 0 else 0

            print()
            print(f"  Degradacao de Edge: {edge_degradation:+.1f}%")
            print(f"  Degradacao de Win Rate: {wr_degradation:+.1f}%")
            print(f"  Ratio PF (OOS/IS): {pf_ratio:.2f}")

            # Criterios de validacao
            print(f"\n  CRITERIOS DE VALIDACAO:")
            print(f"  {'-'*55}")

            edge_ok = test_r['edge'] > 0
            degradation_ok = edge_degradation < 10  # Menos de 10% de degradacao
            pf_ok = test_r['profit_factor'] > 1.0

            print(f"  [{'OK' if edge_ok else 'FALHA'}] Edge OOS > 0: {test_r['edge']:+.1f}%")
            print(f"  [{'OK' if degradation_ok else 'FALHA'}] Degradacao < 10%: {edge_degradation:+.1f}%")
            print(f"  [{'OK' if pf_ok else 'FALHA'}] PF OOS > 1.0: {test_r['profit_factor']:.2f}")

            if edge_ok and degradation_ok and pf_ok:
                print(f"\n  VEREDICTO: ESTRATEGIA VALIDADA")
                print(f"  Os parametros funcionam em dados nao vistos.")
            elif edge_ok:
                print(f"\n  VEREDICTO: ESTRATEGIA PARCIALMENTE VALIDADA")
                print(f"  Edge positivo em OOS, mas com degradacao significativa.")
            else:
                print(f"\n  VEREDICTO: ESTRATEGIA REPROVADA")
                print(f"  Os parametros nao funcionam em dados nao vistos (overfitting).")
        else:
            print("\n  Trades insuficientes para analise de degradacao.")
    else:
        print("\n  Dados insuficientes para analise de degradacao.")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
