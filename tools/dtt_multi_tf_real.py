#!/usr/bin/env python3
"""
================================================================================
DTT MULTI-TIMEFRAME TEST - DADOS REAIS
================================================================================

Testa a estrategia DTT (Detector de Tunelamento Topologico) em M5, M15, M30, H1 e H4
com dados REAIS da FXOpen.

Usa parametros OTIMIZADOS do config dtt-tunelamentotopologico_robust.json

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
# PARAMETROS OTIMIZADOS DO CONFIG
# =============================================================================

# PARAMETROS REOTIMIZADOS (Grid Search 28/12/2025)
PERSISTENCE_ENTROPY_THRESHOLD = 0.5
TUNNELING_PROBABILITY_THRESHOLD = 0.35
MIN_SIGNAL_STRENGTH = 0.2

# Risk Management reotimizado
SL_PIPS_H1 = 20.0
TP_PIPS_H1 = 40.0

# Custos
SLIPPAGE_PIPS = 0.8
SPREAD_PIPS = 1.5

# Trading
COOLDOWN = 30
MIN_WARMUP = 200
TRADING_START_HOUR = 7
TRADING_END_HOUR = 20
PIP_VALUE = 0.0001

# API
API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"

# =============================================================================
# CONFIGURACAO POR TIMEFRAME
# =============================================================================

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

# SL/TP proporcionais ao timeframe (baseado no H1 20/40)
TIMEFRAME_CONFIGS = {
    'M5': TimeframeConfig(
        name='M5', period='M5',
        sl_pips=8.0, tp_pips=16.0,  # Proporcional
        bars_to_download=5000,
        cooldown=20
    ),
    'M15': TimeframeConfig(
        name='M15', period='M15',
        sl_pips=12.0, tp_pips=24.0,
        bars_to_download=3000,
        cooldown=15
    ),
    'M30': TimeframeConfig(
        name='M30', period='M30',
        sl_pips=15.0, tp_pips=30.0,
        bars_to_download=2000,
        cooldown=12
    ),
    'H1': TimeframeConfig(
        name='H1', period='H1',
        sl_pips=SL_PIPS_H1, tp_pips=TP_PIPS_H1,  # Otimizado via grid search
        bars_to_download=2500,
        cooldown=COOLDOWN
    ),
    'H4': TimeframeConfig(
        name='H4', period='H4',
        sl_pips=35.0, tp_pips=70.0,
        bars_to_download=1000,
        cooldown=8
    )
}

# =============================================================================
# DOWNLOAD DE DADOS
# =============================================================================

def download_real_bars(period: str, count: int) -> List[Bar]:
    """Baixa barras REAIS da API FXOpen Live."""
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
# BACKTEST
# =============================================================================

def run_backtest(bars: List[Bar], config: TimeframeConfig) -> dict:
    """Run backtest com parametros otimizados."""

    # Criar indicador DTT com parametros otimizados
    dtt = DetectorTunelamentoTopologico(
        max_points=150,
        use_dimensionality_reduction=True,
        reduction_method='pca',
        persistence_entropy_threshold=0.1,  # Baixo para calcular tudo
        tunneling_probability_threshold=0.05,
        auto_calibrate_quantum=True
    )

    trades = []
    prices = [b.close for b in bars]

    last_signal = -999
    total_bars = len(bars)
    progress_step = max(1, total_bars // 5)

    print(f"    Processando {total_bars} barras (warmup={MIN_WARMUP}, cooldown={config.cooldown})...")

    for i in range(MIN_WARMUP, len(bars) - 1):
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

        # Analisar com DTT
        try:
            prices_arr = np.array(prices[:i+1])
            result = dtt.analyze(prices_arr)

            # Verificar thresholds otimizados
            if result is None or 'error' in result:
                continue

            # DTT retorna entropy e tunneling em subdicts
            entropy_info = result.get('entropy', {})
            tunneling_info = result.get('tunneling', {})

            persistence_entropy = entropy_info.get('persistence_entropy', 0)
            tunneling_prob = tunneling_info.get('tunneling_probability', 0)
            signal_strength = result.get('signal_strength', 0)

            # Filtrar por thresholds otimizados
            if persistence_entropy < PERSISTENCE_ENTROPY_THRESHOLD:
                continue
            if tunneling_prob < TUNNELING_PROBABILITY_THRESHOLD:
                continue
            if signal_strength < MIN_SIGNAL_STRENGTH:
                continue

            # DTT usa trade_on e direction
            if not result.get('trade_on', False):
                continue

            direction = result.get('direction', '')
            if direction == 'LONG':
                signal = 1
            elif direction == 'SHORT':
                signal = -1
            else:
                continue

        except Exception as e:
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

        # Verificar resultado
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
    print("  DTT MULTI-TIMEFRAME TEST - DADOS REAIS")
    print("=" * 80)
    print()
    print("  PARAMETROS OTIMIZADOS (dtt-tunelamentotopologico_robust.json):")
    print(f"    persistence_entropy_threshold: {PERSISTENCE_ENTROPY_THRESHOLD}")
    print(f"    tunneling_probability_threshold: {TUNNELING_PROBABILITY_THRESHOLD}")
    print(f"    min_signal_strength: {MIN_SIGNAL_STRENGTH}")
    print(f"    SL/TP H1: {SL_PIPS_H1}/{TP_PIPS_H1} pips")
    print(f"    Slippage: {SLIPPAGE_PIPS} pips | Spread: {SPREAD_PIPS} pips")
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

        if len(bars) < MIN_WARMUP + 100:
            print(f"  ERRO: Apenas {len(bars)} barras!")
            continue

        days = (bars[-1].timestamp - bars[0].timestamp).days
        print(f"\n    Periodo: {bars[0].timestamp.strftime('%Y-%m-%d')} a {bars[-1].timestamp.strftime('%Y-%m-%d')}")
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
    print("  TABELA COMPARATIVA - DTT DADOS REAIS")
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
        if results:
            best = max(results, key=lambda x: x['edge'])
            print(f"\n  Melhor resultado: {best['timeframe']} com edge {best['edge']:+.1f}%")

    # Comparar com config
    print("\n" + "=" * 90)
    print("  COMPARACAO COM CONFIG OTIMIZADO (H1)")
    print("=" * 90)
    h1_result = next((r for r in results if r['timeframe'] == 'H1'), None)
    if h1_result:
        print()
        print(f"  {'Metrica':<20} {'Config Test':<15} {'Meu Teste':<15}")
        print(f"  {'-'*50}")
        print(f"  {'Win Rate':<20} {'42.4%':<15} {h1_result['win_rate']:.1f}%")
        print(f"  {'Profit Factor':<20} {'1.66':<15} {h1_result['profit_factor']:.2f}")
        print(f"  {'Edge':<20} {'+12.8%':<15} {h1_result['edge']:+.1f}%")

    print("\n" + "=" * 90)

if __name__ == "__main__":
    main()
