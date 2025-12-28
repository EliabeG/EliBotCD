#!/usr/bin/env python3
"""
DTT Test com Parametros Otimizados do Config
Usa os thresholds corretos do dtt-tunelamentotopologico_robust.json
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

PERSISTENCE_ENTROPY_THRESHOLD = 0.9071
TUNNELING_PROBABILITY_THRESHOLD = 0.5
MIN_SIGNAL_STRENGTH = 0.3111
SL_PIPS = 27.5
TP_PIPS = 65.5

# Custos
SPREAD_PIPS = 1.5
SLIPPAGE_PIPS = 0.8

# Trading
TRADING_START_HOUR = 7
TRADING_END_HOUR = 20
COOLDOWN = 30
MIN_WARMUP = 150
PIP_VALUE = 0.0001

# API
API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

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

def run_backtest(bars: List[Bar]) -> dict:
    """Run backtest com parametros otimizados."""

    # Criar DTT
    dtt = DetectorTunelamentoTopologico(
        max_points=150,
        use_dimensionality_reduction=True,
        reduction_method='pca',
        persistence_entropy_threshold=0.1,  # Threshold baixo para gerar analise
        tunneling_probability_threshold=0.05,
        auto_calibrate_quantum=True
    )

    trades = []
    prices = [b.close for b in bars]

    last_signal = -999
    total_bars = len(bars)
    progress_step = max(1, total_bars // 10)

    signals_checked = 0
    above_entropy = 0
    above_tunneling = 0
    above_strength = 0

    print(f"\n  Thresholds Otimizados:")
    print(f"    persistence_entropy >= {PERSISTENCE_ENTROPY_THRESHOLD}")
    print(f"    tunneling_probability >= {TUNNELING_PROBABILITY_THRESHOLD}")
    print(f"    signal_strength >= {MIN_SIGNAL_STRENGTH}")
    print(f"\n  Processando {total_bars} barras...")

    for i in range(MIN_WARMUP, len(bars) - 1):
        if i % progress_step == 0:
            pct = i * 100 // total_bars
            print(f"    {pct}%... (trades: {len(trades)})")

        # Cooldown
        if i - last_signal < COOLDOWN:
            continue

        # Trading hours
        hour = bars[i].timestamp.hour
        if hour < TRADING_START_HOUR or hour >= TRADING_END_HOUR:
            continue

        try:
            prices_arr = np.array(prices[:i+1])
            result = dtt.analyze(prices_arr)

            if result is None or 'error' in result:
                continue

            signals_checked += 1

            # Extrair metricas
            entropy_info = result.get('entropy', {})
            tunneling_info = result.get('tunneling', {})

            persistence_entropy = entropy_info.get('persistence_entropy', 0)
            tunneling_prob = tunneling_info.get('tunneling_probability', 0)
            signal_strength = result.get('signal_strength', 0)

            # Contar quantos passam cada threshold
            if persistence_entropy >= PERSISTENCE_ENTROPY_THRESHOLD:
                above_entropy += 1
            if tunneling_prob >= TUNNELING_PROBABILITY_THRESHOLD:
                above_tunneling += 1
            if signal_strength >= MIN_SIGNAL_STRENGTH:
                above_strength += 1

            # Verificar thresholds otimizados
            if persistence_entropy < PERSISTENCE_ENTROPY_THRESHOLD:
                continue
            if tunneling_prob < TUNNELING_PROBABILITY_THRESHOLD:
                continue
            if signal_strength < MIN_SIGNAL_STRENGTH:
                continue

            # trade_on e direction
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
            sl = entry - SL_PIPS * PIP_VALUE
            tp = entry + TP_PIPS * PIP_VALUE
        else:
            entry -= SLIPPAGE_PIPS * PIP_VALUE
            sl = entry + SL_PIPS * PIP_VALUE
            tp = entry - TP_PIPS * PIP_VALUE

        # Verificar resultado
        for j in range(i + 2, min(i + 200, len(bars))):
            bar = bars[j]

            if signal == 1:
                if bar.low <= sl:
                    trades.append(('LOSS', -SL_PIPS - SPREAD_PIPS))
                    break
                if bar.high >= tp:
                    trades.append(('WIN', TP_PIPS - SPREAD_PIPS))
                    break
            else:
                if bar.high >= sl:
                    trades.append(('LOSS', -SL_PIPS - SPREAD_PIPS))
                    break
                if bar.low <= tp:
                    trades.append(('WIN', TP_PIPS - SPREAD_PIPS))
                    break

        last_signal = i

    print(f"    100% (trades: {len(trades)})")

    # Estatisticas de threshold
    print(f"\n  ANALISE DE THRESHOLDS:")
    print(f"    Sinais analisados: {signals_checked}")
    print(f"    Acima entropy >= {PERSISTENCE_ENTROPY_THRESHOLD}: {above_entropy} ({100*above_entropy/max(1,signals_checked):.1f}%)")
    print(f"    Acima tunneling >= {TUNNELING_PROBABILITY_THRESHOLD}: {above_tunneling} ({100*above_tunneling/max(1,signals_checked):.1f}%)")
    print(f"    Acima strength >= {MIN_SIGNAL_STRENGTH}: {above_strength} ({100*above_strength/max(1,signals_checked):.1f}%)")

    # Estatisticas
    if not trades:
        return {
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
        'total_trades': total,
        'wins': wins,
        'losses': total - wins,
        'win_rate': win_rate,
        'breakeven': breakeven,
        'edge': edge,
        'total_pnl': total_pnl,
        'profit_factor': pf
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("  DTT TEST COM PARAMETROS OTIMIZADOS")
    print("=" * 80)
    print()
    print("  CONFIG: dtt-tunelamentotopologico_robust.json")
    print(f"    persistence_entropy_threshold: {PERSISTENCE_ENTROPY_THRESHOLD}")
    print(f"    tunneling_probability_threshold: {TUNNELING_PROBABILITY_THRESHOLD}")
    print(f"    min_signal_strength: {MIN_SIGNAL_STRENGTH}")
    print(f"    SL/TP: {SL_PIPS}/{TP_PIPS} pips")
    print(f"    Breakeven: {SL_PIPS / (SL_PIPS + TP_PIPS) * 100:.1f}%")
    print()

    # Baixar dados H1
    print("  BAIXANDO DADOS H1...")
    bars = download_real_bars("H1", 2500)

    if len(bars) < MIN_WARMUP + 100:
        print(f"  ERRO: Apenas {len(bars)} barras!")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"\n  Periodo: {bars[0].timestamp.strftime('%Y-%m-%d')} a {bars[-1].timestamp.strftime('%Y-%m-%d')}")
    print(f"  Dias: {days} | Barras: {len(bars)}")

    # Backtest
    result = run_backtest(bars)

    # Resultado
    print(f"\n{'='*80}")
    print(f"  RESULTADO")
    print(f"{'='*80}")
    print(f"  Trades: {result['total_trades']}")
    if result['total_trades'] > 0:
        print(f"  Wins: {result['wins']} | Losses: {result['losses']}")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Breakeven: {result['breakeven']:.1f}%")
        print(f"  Edge: {result['edge']:+.1f}%")
        print(f"  PnL Total: {result['total_pnl']:+.0f} pips")
        print(f"  Profit Factor: {result['profit_factor']:.2f}")

        if result['edge'] > 0:
            print(f"\n  [APROVADO] Edge positivo!")
        else:
            print(f"\n  [REPROVADO] Edge negativo!")
    else:
        print("  NENHUM TRADE GERADO!")
        print("  Thresholds podem estar muito restritivos ou dados insuficientes")

    # Comparar com config
    print(f"\n{'='*80}")
    print(f"  COMPARACAO COM CONFIG OTIMIZADO")
    print(f"{'='*80}")
    print(f"  {'Metrica':<20} {'Config Test':<15} {'Meu Teste':<15}")
    print(f"  {'-'*50}")
    print(f"  {'Trades':<20} {'59':<15} {result['total_trades']}")
    print(f"  {'Win Rate':<20} {'42.4%':<15} {result['win_rate']:.1f}%")
    print(f"  {'Profit Factor':<20} {'1.66':<15} {result['profit_factor']:.2f}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
