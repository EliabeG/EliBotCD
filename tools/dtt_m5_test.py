#!/usr/bin/env python3
"""
================================================================================
DTT M5 TEST - Teste com Filtros Tecnicos Adicionais
================================================================================

Testa DTT V3.3 em M5 com:
- Filtros tecnicos: ATR, EMA, RSI, Sessao
- Dados OHLC reais da API FXOpen
- Parametros conservadores (sem overfitting)

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
from typing import List, Tuple
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.alta_volatilidade.dtt_tunelamento_topologico import DetectorTunelamentoTopologico

# =============================================================================
# PARAMETROS CONSERVADORES PARA M5 (sem otimizacao - genericos)
# =============================================================================

# Thresholds DTT - valores conservadores padrao
PERSISTENCE_ENTROPY_THRESHOLD = 0.5
TUNNELING_PROBABILITY_THRESHOLD = 0.15
MIN_SIGNAL_STRENGTH = 0.35  # Ajustado para M5

# SL/TP para M5 - ratio 1:2.5 com SL menor
SL_PIPS = 8.0    # Stop loss mais apertado
TP_PIPS = 20.0   # Take profit mantido (breakeven = 28.6%)

# Modo LONG-only (SHORT tem performance ruim)
LONG_ONLY_MODE = True

# Custos realistas
SPREAD_PIPS = 1.2
SLIPPAGE_PIPS = 0.5

# Trading
COOLDOWN = 6  # 30 minutos entre trades (6 barras M5)
MIN_WARMUP = 120  # Minimo de barras para analise (10 horas)
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
# DOWNLOAD DADOS OHLC
# =============================================================================

def download_ohlc_bars(period: str, count: int) -> List[Bar]:
    """Download barras OHLC da API FXOpen."""
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
                    bar = Bar(
                        timestamp=ts,
                        open=float(b["Open"]),
                        high=float(b["High"]),
                        low=float(b["Low"]),
                        close=float(b["Close"]),
                        volume=float(b.get("Volume", 0))
                    )
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
# BACKTEST COM FILTROS
# =============================================================================

def run_backtest(bars: List[Bar]) -> dict:
    """Executa backtest com DTT V3.3 e filtros tecnicos."""

    # Criar detector (max_points=100 para M5 - mais rapido)
    dtt = DetectorTunelamentoTopologico(
        max_points=100,
        use_dimensionality_reduction=True,
        reduction_method='pca',
        persistence_entropy_threshold=PERSISTENCE_ENTROPY_THRESHOLD,
        tunneling_probability_threshold=TUNNELING_PROBABILITY_THRESHOLD,
        auto_calibrate_quantum=True
    )

    # Arrays OHLC
    opens = np.array([b.open for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])

    trades = []
    last_signal = -999
    total_bars = len(bars)
    progress_step = max(1, total_bars // 10)

    print(f"\n    Processando {total_bars} barras M5...")
    print(f"    Parametros: Entropy>={PERSISTENCE_ENTROPY_THRESHOLD}, Strength>={MIN_SIGNAL_STRENGTH}")
    print(f"    SL/TP: {SL_PIPS}/{TP_PIPS} pips")

    signals_generated = 0
    signals_filtered = 0

    for i in range(MIN_WARMUP, len(bars) - 1):
        if i % progress_step == 0:
            pct = i * 100 // total_bars
            print(f"      {pct}%... (trades: {len(trades)}, sinais: {signals_generated})")

        # Cooldown
        if i - last_signal < COOLDOWN:
            continue

        hour = bars[i].timestamp.hour

        try:
            # Analise DTT com filtros
            result = dtt.analyze(
                prices=closes[:i+1],
                highs=highs[:i+1],
                lows=lows[:i+1],
                current_hour=hour,
                use_filters=True
            )

            if result is None or 'error' in result:
                continue

            # Verificar thresholds
            entropy_info = result.get('entropy', {})
            persistence_entropy = entropy_info.get('persistence_entropy', 0)
            signal_strength = result.get('signal_strength', 0)

            if persistence_entropy < PERSISTENCE_ENTROPY_THRESHOLD:
                continue
            if signal_strength < MIN_SIGNAL_STRENGTH:
                continue

            # Verificar trade_on (ja inclui filtros)
            if not result.get('trade_on', False):
                continue

            signals_generated += 1

            # Verificar direcao
            direction = result.get('direction', '')
            if direction == 'LONG':
                signal = 1
            elif direction == 'SHORT':
                if LONG_ONLY_MODE:
                    signals_filtered += 1
                    continue  # Ignorar SHORT em modo LONG-only
                signal = -1
            else:
                signals_filtered += 1
                continue

            # Verificar filtros extras
            filters = result.get('filters', {})
            if filters:
                # Verificar se filtros estao ok
                if not filters.get('filters_ok', True):
                    signals_filtered += 1
                    continue

                # Score minimo dos filtros
                total_score = filters.get('total_score', 0)
                if total_score < 0.5:
                    signals_filtered += 1
                    continue

        except Exception as e:
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

        trade_result = None
        for j in range(i + 2, min(i + 100, len(bars))):  # Max 100 barras = ~8 horas
            bar = bars[j]

            if signal == 1:
                if bar.low <= sl:
                    trade_result = ('LOSS', -SL_PIPS - SPREAD_PIPS, bars[i].timestamp, 'LONG')
                    break
                if bar.high >= tp:
                    trade_result = ('WIN', TP_PIPS - SPREAD_PIPS, bars[i].timestamp, 'LONG')
                    break
            else:
                if bar.high >= sl:
                    trade_result = ('LOSS', -SL_PIPS - SPREAD_PIPS, bars[i].timestamp, 'SHORT')
                    break
                if bar.low <= tp:
                    trade_result = ('WIN', TP_PIPS - SPREAD_PIPS, bars[i].timestamp, 'SHORT')
                    break

        if trade_result:
            trades.append(trade_result)

        last_signal = i

    print(f"      100% (trades: {len(trades)}, sinais: {signals_generated})")
    print(f"    Sinais filtrados: {signals_filtered}")

    # Calcular metricas
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'breakeven': SL_PIPS / (SL_PIPS + TP_PIPS) * 100,
            'edge': 0,
            'total_pnl': 0,
            'profit_factor': 0,
            'signals_generated': signals_generated,
            'signals_filtered': signals_filtered
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

    # Contar por direcao
    long_trades = [t for t in trades if t[3] == 'LONG']
    short_trades = [t for t in trades if t[3] == 'SHORT']
    long_wins = len([t for t in long_trades if t[0] == 'WIN'])
    short_wins = len([t for t in short_trades if t[0] == 'WIN'])

    # Trades por mes
    if len(bars) > 1:
        days = (bars[-1].timestamp - bars[0].timestamp).days
        months = days / 30.0 if days > 0 else 1
        trades_per_month = total / months
    else:
        trades_per_month = 0

    return {
        'total_trades': total,
        'wins': wins,
        'losses': total - wins,
        'win_rate': win_rate,
        'breakeven': breakeven,
        'edge': edge,
        'total_pnl': total_pnl,
        'profit_factor': pf,
        'trades_per_month': trades_per_month,
        'long_trades': len(long_trades),
        'long_wins': long_wins,
        'short_trades': len(short_trades),
        'short_wins': short_wins,
        'signals_generated': signals_generated,
        'signals_filtered': signals_filtered,
        'trades': trades
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("  DTT M5 TEST - COM FILTROS TECNICOS (V3.3)")
    print("=" * 80)
    print()
    print("  FILTROS APLICADOS:")
    print("    - ATR: Volatilidade dentro do range ideal (2-20 pips)")
    print("    - EMA 12/26: Tendencia definida")
    print("    - RSI 14: Momentum confirmado")
    print("    - Sessao: London/NY (07:00-21:00 UTC)")
    print()
    print("  PARAMETROS CONSERVADORES (sem overfitting):")
    print(f"    persistence_entropy >= {PERSISTENCE_ENTROPY_THRESHOLD}")
    print(f"    signal_strength >= {MIN_SIGNAL_STRENGTH}")
    print(f"    SL/TP: {SL_PIPS}/{TP_PIPS} pips (ratio: 1:{TP_PIPS/SL_PIPS:.1f})")
    print(f"    Breakeven: {SL_PIPS / (SL_PIPS + TP_PIPS) * 100:.1f}%")
    print(f"    Modo: {'LONG-ONLY' if LONG_ONLY_MODE else 'BIDIRECIONAL'}")
    print()

    # Baixar dados M5
    bars = download_ohlc_bars("M5", 4000)  # ~14 dias de dados

    if len(bars) < MIN_WARMUP + 100:
        print(f"  ERRO: Apenas {len(bars)} barras!")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"\n  DADOS:")
    print(f"    Periodo: {bars[0].timestamp.strftime('%Y-%m-%d')} a {bars[-1].timestamp.strftime('%Y-%m-%d')}")
    print(f"    Dias: {days} | Barras: {len(bars)}")

    # Executar backtest
    result = run_backtest(bars)

    # Mostrar resultados
    print(f"\n{'='*80}")
    print("  RESULTADO DTT M5 COM FILTROS")
    print(f"{'='*80}")
    print()
    print(f"  METRICAS PRINCIPAIS:")
    print(f"  {'-'*55}")
    print(f"  Trades: {result['total_trades']} ({result.get('trades_per_month', 0):.1f}/mes)")

    if result['total_trades'] > 0:
        print(f"  Wins: {result['wins']} | Losses: {result['losses']}")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Breakeven: {result['breakeven']:.1f}%")
        print(f"  Edge: {result['edge']:+.1f}%")
        print(f"  PnL Total: {result['total_pnl']:+.1f} pips")
        print(f"  Profit Factor: {result['profit_factor']:.2f}")

        print(f"\n  POR DIRECAO:")
        print(f"  {'-'*55}")
        if result['long_trades'] > 0:
            long_wr = result['long_wins'] / result['long_trades'] * 100
            print(f"  LONG:  {result['long_trades']} trades, {result['long_wins']} wins ({long_wr:.1f}%)")
        if result['short_trades'] > 0:
            short_wr = result['short_wins'] / result['short_trades'] * 100
            print(f"  SHORT: {result['short_trades']} trades, {result['short_wins']} wins ({short_wr:.1f}%)")

        print(f"\n  FILTRAGEM:")
        print(f"  {'-'*55}")
        print(f"  Sinais DTT gerados: {result['signals_generated']}")
        print(f"  Sinais filtrados: {result['signals_filtered']}")
        if result['signals_generated'] > 0:
            filter_rate = result['signals_filtered'] / result['signals_generated'] * 100
            print(f"  Taxa de filtragem: {filter_rate:.1f}%")

        # Veredicto
        print(f"\n{'='*80}")
        if result['edge'] > 5 and result['profit_factor'] > 1.2:
            print("  [APROVADO] Estrategia com edge positivo!")
        elif result['edge'] > 0:
            print("  [MARGINAL] Edge positivo mas baixo - precisa mais testes")
        else:
            print("  [REPROVADO] Edge negativo")
        print(f"{'='*80}")

    else:
        print("  NENHUM TRADE GERADO!")
        print(f"  Sinais DTT: {result['signals_generated']}")
        print(f"  Filtrados: {result['signals_filtered']}")

    print()

if __name__ == "__main__":
    main()
