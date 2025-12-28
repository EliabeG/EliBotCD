#!/usr/bin/env python3
"""
================================================================================
DSG MULTI-TIMEFRAME TEST - DADOS REAIS (CORRIGIDO)
================================================================================

Testa a estrategia DSG em M5, M15, M30, H1 e H4 com dados REAIS.
Usa parametros CORRETOS da estrategia de producao.

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

from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional

# =============================================================================
# CONFIGURACAO - PARAMETROS DE PRODUCAO CORRETOS
# =============================================================================

@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread_pips: float = 0.2

@dataclass
class TimeframeConfig:
    name: str
    period: str
    sl_pips: float
    tp_pips: float
    bars_to_download: int
    cooldown: int  # Barras entre sinais (ajustado por TF)

# PARAMETROS DE PRODUCAO
RICCI_THRESHOLD = -50500.0
TIDAL_THRESHOLD = 0.01
LOOKBACK_WINDOW = 30
MIN_WARMUP_BARS = 200  # IMPORTANTE: Era 200, nao 40!
SLIPPAGE_PIPS = 0.3
MAX_SPREAD_PIPS = 2.0
TRADING_START_HOUR = 7
TRADING_END_HOUR = 20
PIP_VALUE = 0.0001

# Configuracoes por timeframe (SL/TP ajustados proporcionalmente)
TIMEFRAME_CONFIGS = {
    'M5': TimeframeConfig(
        name='M5', period='M5',
        sl_pips=10.0, tp_pips=20.0,  # 1:2 ratio
        bars_to_download=8000,  # ~28 dias
        cooldown=30  # 30 barras = 2.5 horas em M5
    ),
    'M15': TimeframeConfig(
        name='M15', period='M15',
        sl_pips=15.0, tp_pips=30.0,
        bars_to_download=4000,  # ~42 dias
        cooldown=20  # 20 barras = 5 horas em M15
    ),
    'M30': TimeframeConfig(
        name='M30', period='M30',
        sl_pips=20.0, tp_pips=40.0,
        bars_to_download=3000,  # ~63 dias
        cooldown=15  # 15 barras = 7.5 horas em M30
    ),
    'H1': TimeframeConfig(
        name='H1', period='H1',
        sl_pips=30.0, tp_pips=60.0,  # Original
        bars_to_download=2000,  # ~83 dias
        cooldown=30  # Original: 30 barras
    ),
    'H4': TimeframeConfig(
        name='H4', period='H4',
        sl_pips=50.0, tp_pips=100.0,
        bars_to_download=1000,  # ~167 dias
        cooldown=8  # 8 barras = 32 horas
    )
}

# API Configuration
API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"

# =============================================================================
# DOWNLOAD DE DADOS REAIS
# =============================================================================

def download_real_bars(period: str, count: int) -> List[Bar]:
    """Baixa barras REAIS da API FXOpen Live."""
    bars = []

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    current_ts = int(time.time() * 1000)
    total_downloaded = 0
    batch_size = 1000

    print(f"    Baixando {count} barras {period}...")

    while total_downloaded < count:
        remaining = min(batch_size, count - total_downloaded)
        url = f"{API_BASE}/{SYMBOL}/{period}/bars/ask?timestamp={current_ts}&count=-{remaining}"

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30, context=ctx) as response:
                data = json.loads(response.read().decode())
                batch_bars = data.get("Bars", [])

                if not batch_bars:
                    break

                for bar_data in batch_bars:
                    ts = datetime.fromtimestamp(bar_data["Timestamp"] / 1000, tz=timezone.utc)
                    bar = Bar(
                        timestamp=ts,
                        open=float(bar_data["Open"]),
                        high=float(bar_data["High"]),
                        low=float(bar_data["Low"]),
                        close=float(bar_data["Close"]),
                        volume=float(bar_data.get("Volume", 0)),
                        spread_pips=0.2
                    )
                    bars.append(bar)

                total_downloaded += len(batch_bars)

                oldest = min(batch_bars, key=lambda x: x["Timestamp"])
                current_ts = oldest["Timestamp"] - 1

                print(f"      +{len(batch_bars)} barras (total: {total_downloaded})")
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
    """Run backtest com parametros de producao."""

    dsg = DetectorSingularidadeGravitacional(
        ricci_collapse_threshold=RICCI_THRESHOLD,
        tidal_force_threshold=TIDAL_THRESHOLD,
        lookback_window=LOOKBACK_WINDOW
    )

    trades = []
    prices = [b.close for b in bars]

    last_signal = -999

    total_bars = len(bars)
    progress_step = max(1, total_bars // 5)

    print(f"    Processando {total_bars} barras (warmup={MIN_WARMUP_BARS}, cooldown={config.cooldown})...")

    for i in range(MIN_WARMUP_BARS, len(bars) - 1):
        if i % progress_step == 0:
            pct = i * 100 // total_bars
            print(f"      {pct}%...")

        # Cooldown
        if i - last_signal < config.cooldown:
            continue

        # Trading hours
        hour = bars[i].timestamp.hour
        if hour < TRADING_START_HOUR or hour >= TRADING_END_HOUR:
            continue

        # Analisar
        result = dsg.analyze(np.array(prices[:i+1]))
        signal = result.get('signal', 0)

        if signal == 0:
            continue

        # Simular trade
        entry = bars[i + 1].open
        spread = bars[i].spread_pips

        # Verificar spread maximo
        if spread > MAX_SPREAD_PIPS:
            continue

        if signal == 1:
            entry += SLIPPAGE_PIPS * PIP_VALUE
            sl = entry - config.sl_pips * PIP_VALUE
            tp = entry + config.tp_pips * PIP_VALUE
        else:
            entry -= SLIPPAGE_PIPS * PIP_VALUE
            sl = entry + config.sl_pips * PIP_VALUE
            tp = entry - config.tp_pips * PIP_VALUE

        # Verificar resultado
        for j in range(i + 2, min(i + 200, len(bars))):
            bar = bars[j]

            if signal == 1:
                if bar.low <= sl:
                    pnl = -config.sl_pips - spread
                    trades.append(('LOSS', pnl, bars[i].timestamp))
                    break
                if bar.high >= tp:
                    pnl = config.tp_pips - spread
                    trades.append(('WIN', pnl, bars[i].timestamp))
                    break
            else:
                if bar.high >= sl:
                    pnl = -config.sl_pips - spread
                    trades.append(('LOSS', pnl, bars[i].timestamp))
                    break
                if bar.low <= tp:
                    pnl = config.tp_pips - spread
                    trades.append(('WIN', pnl, bars[i].timestamp))
                    break

        last_signal = i

    print(f"      100%")

    # Estatisticas
    if not trades:
        return {
            'timeframe': config.name,
            'total_trades': 0,
            'win_rate': 0,
            'breakeven': config.sl_pips / (config.sl_pips + config.tp_pips) * 100,
            'edge': 0,
            'total_pnl': 0,
            'profit_factor': 0,
            'trades_per_month': 0
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

    # Trades por mes
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
    print("  DSG MULTI-TIMEFRAME TEST - DADOS REAIS (PARAMETROS DE PRODUCAO)")
    print("=" * 80)
    print(f"  Ricci Threshold: {RICCI_THRESHOLD}")
    print(f"  Tidal Threshold: {TIDAL_THRESHOLD}")
    print(f"  Warmup: {MIN_WARMUP_BARS} barras")
    print(f"  Slippage: {SLIPPAGE_PIPS} pips")
    print(f"  Trading Hours: {TRADING_START_HOUR}:00-{TRADING_END_HOUR}:00 UTC")
    print()

    results = []

    for tf_name in ['M5', 'M15', 'M30', 'H1', 'H4']:
        config = TIMEFRAME_CONFIGS[tf_name]

        print(f"\n{'='*75}")
        print(f"  TESTANDO {tf_name}")
        print(f"{'='*75}")
        print(f"  SL: {config.sl_pips} pips | TP: {config.tp_pips} pips | Cooldown: {config.cooldown}")
        print(f"  Breakeven: {config.sl_pips / (config.sl_pips + config.tp_pips) * 100:.1f}%")
        print()

        # Download
        bars = download_real_bars(config.period, config.bars_to_download)

        if len(bars) < MIN_WARMUP_BARS + 100:
            print(f"  ERRO: Apenas {len(bars)} barras!")
            continue

        print(f"\n    Periodo: {bars[0].timestamp.strftime('%Y-%m-%d')} a {bars[-1].timestamp.strftime('%Y-%m-%d')}")
        days = (bars[-1].timestamp - bars[0].timestamp).days
        print(f"    Dias: {days} | Barras: {len(bars)}")
        print()

        # Backtest
        result = run_backtest(bars, config)
        results.append(result)

        # Resultado
        print(f"\n  RESULTADO {tf_name}:")
        print(f"  {'-'*55}")
        print(f"  Trades: {result['total_trades']} ({result['trades_per_month']:.1f}/mes)")
        print(f"  Wins: {result['wins']} | Losses: {result['losses']}")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Breakeven: {result['breakeven']:.1f}%")
        print(f"  Edge: {result['edge']:+.1f}%")
        print(f"  PnL Total: {result['total_pnl']:+.0f} pips")
        print(f"  Profit Factor: {result['profit_factor']:.2f}")

        status = "[APROVADO]" if result['edge'] > 0 else "[REPROVADO]"
        print(f"\n  {status}")

    # Tabela final
    print("\n")
    print("=" * 90)
    print("  TABELA COMPARATIVA - DADOS REAIS (PARAMETROS DE PRODUCAO)")
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

        if len(approved) > 1:
            print(f"\n  TIMEFRAMES APROVADOS: {', '.join([r['timeframe'] for r in approved])}")
    else:
        print("\n  NENHUM TIMEFRAME COM EDGE POSITIVO!")
        # Mostrar o melhor mesmo que negativo
        if results:
            best = max(results, key=lambda x: x['edge'])
            print(f"\n  Melhor resultado (ainda negativo): {best['timeframe']} com edge {best['edge']:+.1f}%")

    print("\n" + "=" * 90)

if __name__ == "__main__":
    main()
