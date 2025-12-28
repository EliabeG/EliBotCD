#!/usr/bin/env python3
"""
================================================================================
DSG MULTI-TIMEFRAME TEST (VERSAO RAPIDA)
================================================================================

Testa a estrategia DSG em M5, M15, M30 e H4.
Versao otimizada para execucao rapida.

================================================================================
"""

import sys
import os
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional

# =============================================================================
# CONFIGURACAO
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
    lookback_window: int
    n_bars: int
    volatility_factor: float

# Configuracoes reduzidas para teste rapido
TIMEFRAME_CONFIGS = {
    'M5': TimeframeConfig(
        name='M5', period='M5', sl_pips=10.0, tp_pips=20.0,
        lookback_window=30, n_bars=2000, volatility_factor=0.00015
    ),
    'M15': TimeframeConfig(
        name='M15', period='M15', sl_pips=15.0, tp_pips=30.0,
        lookback_window=30, n_bars=1500, volatility_factor=0.00025
    ),
    'M30': TimeframeConfig(
        name='M30', period='M30', sl_pips=20.0, tp_pips=40.0,
        lookback_window=30, n_bars=1000, volatility_factor=0.00035
    ),
    'H4': TimeframeConfig(
        name='H4', period='H4', sl_pips=40.0, tp_pips=80.0,
        lookback_window=30, n_bars=500, volatility_factor=0.0012
    )
}

SLIPPAGE_PIPS = 0.3
PIP_VALUE = 0.0001
TRADING_START_HOUR = 7
TRADING_END_HOUR = 20
RICCI_THRESHOLD = -50500.0
TIDAL_THRESHOLD = 0.01

# =============================================================================
# GERACAO DE DADOS
# =============================================================================

def generate_bars(config: TimeframeConfig, seed: int = 42) -> List[Bar]:
    """Gera barras sinteticas rapidamente."""
    np.random.seed(seed)

    bars = []
    price = 1.0850

    intervals = {'M5': 5, 'M15': 15, 'M30': 30, 'H4': 240}
    interval_min = intervals.get(config.period, 60)

    start = datetime.now(timezone.utc) - timedelta(minutes=config.n_bars * interval_min)

    for i in range(config.n_bars):
        ts = start + timedelta(minutes=i * interval_min)
        hour = ts.hour

        # Volatilidade por sessao
        vol_mult = 1.2 if 7 <= hour <= 16 else 0.8
        vol = config.volatility_factor * vol_mult

        ret = np.random.normal(0, vol)
        new_price = price * np.exp(ret)

        bar = Bar(
            timestamp=ts,
            open=price,
            high=max(price, new_price) * (1 + abs(ret) * 0.3),
            low=min(price, new_price) * (1 - abs(ret) * 0.3),
            close=new_price,
            volume=1000,
            spread_pips=0.2
        )
        bars.append(bar)
        price = new_price

    return bars

# =============================================================================
# BACKTEST
# =============================================================================

def run_backtest(bars: List[Bar], config: TimeframeConfig) -> dict:
    """Run backtest otimizado."""

    dsg = DetectorSingularidadeGravitacional(
        ricci_collapse_threshold=RICCI_THRESHOLD,
        tidal_force_threshold=TIDAL_THRESHOLD,
        lookback_window=config.lookback_window
    )

    trades = []
    prices = [b.close for b in bars]

    min_bars = config.lookback_window + 10
    last_signal = -999
    cooldown = 10

    # Processar em blocos para velocidade
    for i in range(min_bars, len(bars) - 1):
        if i - last_signal < cooldown:
            continue

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

        if signal == 1:
            entry += SLIPPAGE_PIPS * PIP_VALUE
            sl = entry - config.sl_pips * PIP_VALUE
            tp = entry + config.tp_pips * PIP_VALUE
        else:
            entry -= SLIPPAGE_PIPS * PIP_VALUE
            sl = entry + config.sl_pips * PIP_VALUE
            tp = entry - config.tp_pips * PIP_VALUE

        # Verificar resultado
        for j in range(i + 2, min(i + 100, len(bars))):
            bar = bars[j]

            if signal == 1:
                if bar.low <= sl:
                    pnl = -config.sl_pips - spread
                    trades.append(('LOSS', pnl))
                    break
                if bar.high >= tp:
                    pnl = config.tp_pips - spread
                    trades.append(('WIN', pnl))
                    break
            else:
                if bar.high >= sl:
                    pnl = -config.sl_pips - spread
                    trades.append(('LOSS', pnl))
                    break
                if bar.low <= tp:
                    pnl = config.tp_pips - spread
                    trades.append(('WIN', pnl))
                    break

        last_signal = i

    # Calcular estatisticas
    if not trades:
        return {
            'timeframe': config.name,
            'total_trades': 0,
            'win_rate': 0,
            'breakeven': config.sl_pips / (config.sl_pips + config.tp_pips) * 100,
            'edge': 0,
            'total_pnl': 0,
            'profit_factor': 0
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

    return {
        'timeframe': config.name,
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
    print("  DSG MULTI-TIMEFRAME TEST")
    print("  (Dados sinteticos - API FXOpen indisponivel)")
    print("=" * 80)

    results = []

    for tf_name in ['M5', 'M15', 'M30', 'H4']:
        config = TIMEFRAME_CONFIGS[tf_name]

        print(f"\n{'='*60}")
        print(f"  TESTANDO {tf_name}")
        print(f"{'='*60}")
        print(f"  SL: {config.sl_pips} pips | TP: {config.tp_pips} pips")
        print(f"  Breakeven: {config.sl_pips / (config.sl_pips + config.tp_pips) * 100:.1f}%")

        print(f"  Gerando {config.n_bars} barras...")
        bars = generate_bars(config, seed=42)

        print(f"  Executando backtest...")
        result = run_backtest(bars, config)
        results.append(result)

        print(f"\n  RESULTADO:")
        print(f"  Trades: {result['total_trades']}")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Breakeven: {result['breakeven']:.1f}%")
        print(f"  Edge: {result['edge']:+.1f}%")
        print(f"  PnL: {result['total_pnl']:+.0f} pips")
        print(f"  PF: {result['profit_factor']:.2f}")

        status = "[APROVADO]" if result['edge'] > 0 else "[REPROVADO]"
        print(f"\n  {status}")

    # Tabela final
    print("\n")
    print("=" * 80)
    print("  TABELA COMPARATIVA")
    print("=" * 80)
    print()
    print(f"{'TF':<6} {'Trades':<8} {'Win%':<8} {'BE%':<8} {'Edge':<8} {'PnL':<10} {'PF':<6} {'Status'}")
    print("-" * 80)

    for r in results:
        status = "APROVADO" if r['edge'] > 0 else "REPROVADO"
        print(f"{r['timeframe']:<6} {r['total_trades']:<8} {r['win_rate']:.1f}%{'':<3} {r['breakeven']:.1f}%{'':<3} {r['edge']:+.1f}%{'':<3} {r['total_pnl']:+.0f}{'':<5} {r['profit_factor']:.2f}{'':<2} {status}")

    # Conclusao
    print("\n" + "=" * 80)
    print("  CONCLUSAO")
    print("=" * 80)

    approved = [r for r in results if r['edge'] > 0]

    if approved:
        best = max(approved, key=lambda x: x['edge'])
        print(f"\n  MELHOR TIMEFRAME: {best['timeframe']}")
        print(f"    Edge: {best['edge']:+.1f}%")
        print(f"    PF: {best['profit_factor']:.2f}")
    else:
        print("\n  NENHUM TIMEFRAME COM EDGE POSITIVO")

    print("\n  NOTA: Resultados preliminares com dados sinteticos.")
    print("=" * 80)

if __name__ == "__main__":
    main()
