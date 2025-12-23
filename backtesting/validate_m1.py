#!/usr/bin/env python3
"""
================================================================================
VALIDACAO PRM em M1 - Dados REAIS FXOpen
================================================================================
"""

import sys
import os
import urllib.request
import ssl
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import deque
from dataclasses import dataclass
from typing import List
import time as time_module
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.alta_volatilidade.prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot


@dataclass
class Bar:
    """Representa uma barra/candle OHLCV"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


# Configuracao Melhor Score
CONFIG_BEST_SCORE = {
    "name": "MELHOR SCORE (M1)",
    "min_prices": 30,
    "stop_loss_pips": 46.9,
    "take_profit_pips": 117.2,
    "hmm_threshold": 0.68,
    "lyapunov_threshold": 0.1249,
    "hmm_states_allowed": [1, 2]
}


def download_fxopen_data(symbol: str, periodicity: str,
                         start_date: datetime, end_date: datetime) -> List[Bar]:
    """Baixa dados historicos REAIS da API REST publica da FXOpen"""
    print(f"\n  Baixando dados REAIS da FXOpen API...")
    print(f"  Simbolo: {symbol}")
    print(f"  Periodicidade: {periodicity}")
    print(f"  Periodo: {start_date.date()} a {end_date.date()}")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    all_bars = []
    current_ts = int(end_date.timestamp() * 1000)
    start_ts = int(start_date.timestamp() * 1000)
    batch = 0
    max_retries = 3

    while current_ts > start_ts:
        batch += 1
        url = (f"https://marginalttdemowebapi.fxopen.net/api/v2/public/"
               f"quotehistory/{symbol}/{periodicity}/bars/bid"
               f"?timestamp={current_ts}&count=-1000")

        for retry in range(max_retries):
            try:
                with urllib.request.urlopen(url, context=ctx, timeout=30) as response:
                    data = json.loads(response.read().decode())

                if "Bars" not in data or not data["Bars"]:
                    print(f"    Batch {batch}: Sem mais dados")
                    return _finalize_bars(all_bars)

                for bar_data in data["Bars"]:
                    if start_ts <= bar_data["Timestamp"] <= int(end_date.timestamp() * 1000):
                        ts = datetime.fromtimestamp(bar_data["Timestamp"] / 1000, tz=timezone.utc)
                        all_bars.append(Bar(
                            timestamp=ts,
                            open=float(bar_data["Open"]),
                            high=float(bar_data["High"]),
                            low=float(bar_data["Low"]),
                            close=float(bar_data["Close"]),
                            volume=float(bar_data.get("Volume", 0))
                        ))

                oldest = min(data["Bars"], key=lambda x: x["Timestamp"])
                new_ts = oldest["Timestamp"] - 1

                if new_ts >= current_ts:
                    return _finalize_bars(all_bars)
                current_ts = new_ts

                print(f"    Batch {batch}: {len(all_bars)} barras...")
                time_module.sleep(0.5)  # Rate limiting
                break

            except Exception as e:
                if retry < max_retries - 1:
                    print(f"    Retry {retry+1}/{max_retries} - {e}")
                    time_module.sleep(2)
                else:
                    print(f"  Erro no batch {batch}: {e}")
                    return _finalize_bars(all_bars)

    return _finalize_bars(all_bars)


def _finalize_bars(all_bars: List[Bar]) -> List[Bar]:
    """Remove duplicatas e ordena"""
    seen = set()
    unique_bars = []
    for bar in all_bars:
        ts_key = int(bar.timestamp.timestamp())
        if ts_key not in seen:
            seen.add(ts_key)
            unique_bars.append(bar)

    unique_bars.sort(key=lambda b: b.timestamp)
    print(f"  Total de barras: {len(unique_bars)}")
    if unique_bars:
        print(f"  Periodo real: {unique_bars[0].timestamp} a {unique_bars[-1].timestamp}")
    return unique_bars


def run_backtest(bars: List[Bar], config: dict) -> dict:
    """Executa backtest"""
    print(f"\n  Testando: {config['name']}")
    print(f"  Barras disponiveis: {len(bars)}")

    pip = 0.0001
    spread = 1.0

    prm = ProtocoloRiemannMandelbrot(
        n_states=3,
        hmm_threshold=0.1,
        lyapunov_threshold_k=0.001,
        curvature_threshold=0.0001,
        lookback_window=100
    )

    prices_buf = deque(maxlen=500)
    volumes_buf = deque(maxlen=500)

    # Pre-calcula valores PRM
    prm_data = []
    print(f"  Calculando PRM (pode demorar)...")

    for i, bar in enumerate(bars):
        prices_buf.append(bar.close)
        volumes_buf.append(bar.volume if bar.volume > 0 else 1.0)

        if len(prices_buf) < config['min_prices']:
            continue

        try:
            result = prm.analyze(np.array(prices_buf), np.array(volumes_buf))
            prm_data.append({
                'idx': i,
                'timestamp': bar.timestamp,
                'price': bar.close,
                'high': bar.high,
                'low': bar.low,
                'hmm_prob': result['Prob_HMM'],
                'lyapunov': result['Lyapunov_Score'],
                'hmm_state': result['hmm_analysis']['current_state']
            })
        except:
            continue

        if (i + 1) % 5000 == 0:
            print(f"    Processado: {i+1}/{len(bars)} barras...")

    print(f"  Pontos PRM calculados: {len(prm_data)}")

    # Encontra sinais
    signals = []
    for d in prm_data:
        if (d['hmm_prob'] >= config['hmm_threshold'] and
            d['lyapunov'] >= config['lyapunov_threshold'] and
            d['hmm_state'] in config['hmm_states_allowed']):

            if d['idx'] >= 10:
                trend = d['price'] - bars[d['idx'] - 10].close
                direction = 1 if trend > 0 else -1
            else:
                direction = 1
            signals.append({
                'idx': d['idx'],
                'price': d['price'],
                'direction': direction,
                'timestamp': d['timestamp']
            })

    print(f"  Sinais gerados: {len(signals)}")

    if len(signals) < 1:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0,
                'win_rate': 0, 'pf': 0, 'max_dd': 0, 'signals': 0}

    # Executa trades
    pnls = []
    trade_details = []
    sl = config['stop_loss_pips']
    tp = config['take_profit_pips']

    for sig in signals:
        bar_idx = sig['idx']
        entry = sig['price']
        direction = sig['direction']

        sl_price = entry - direction * sl * pip
        tp_price = entry + direction * tp * pip

        pnl = 0
        exit_reason = "timeout"

        for j in range(bar_idx + 1, min(bar_idx + 500, len(bars))):
            b = bars[j]
            if direction == 1:
                if b.low <= sl_price:
                    pnl = -sl - spread
                    exit_reason = "stop_loss"
                    break
                if b.high >= tp_price:
                    pnl = tp - spread
                    exit_reason = "take_profit"
                    break
            else:
                if b.high >= sl_price:
                    pnl = -sl - spread
                    exit_reason = "stop_loss"
                    break
                if b.low <= tp_price:
                    pnl = tp - spread
                    exit_reason = "take_profit"
                    break

        if pnl == 0:
            exit_idx = min(bar_idx + 100, len(bars) - 1)
            exit_price = bars[exit_idx].close
            pnl = direction * (exit_price - entry) / pip - spread

        pnls.append(pnl)
        trade_details.append({
            'timestamp': sig['timestamp'],
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'pnl_pips': round(pnl, 1),
            'result': 'WIN' if pnl > 0 else 'LOSS',
            'reason': exit_reason
        })

    # Calcula metricas
    wins = sum(1 for p in pnls if p > 0)
    losses = len(pnls) - wins
    total = sum(pnls)
    wr = wins / len(pnls) if pnls else 0
    gp = sum(p for p in pnls if p > 0) or 0.001
    gl = abs(sum(p for p in pnls if p <= 0)) or 0.001
    pf = gp / gl

    eq = np.cumsum([0] + pnls)
    peak = np.maximum.accumulate(eq + 10000)
    dd = np.max((peak - (eq + 10000)) / peak) if len(peak) > 0 else 0

    return {
        'trades': len(pnls),
        'wins': wins,
        'losses': losses,
        'pnl': round(total, 1),
        'win_rate': round(wr, 4),
        'pf': round(pf, 2),
        'max_dd': round(dd * 100, 2),
        'signals': len(signals),
        'trade_details': trade_details
    }


def main():
    print("=" * 70)
    print("  VALIDACAO PRM em M1 - Dados REAIS FXOpen")
    print("=" * 70)

    # Periodo: ultimos 7 dias (M1 gera muitos dados)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)

    print(f"\n  Periodo: {start.date()} a {end.date()} (7 dias)")

    bars = download_fxopen_data("EURUSD", "M1", start, end)

    if not bars:
        print("  ERRO: Nao foi possivel carregar dados!")
        return

    result = run_backtest(bars, CONFIG_BEST_SCORE)

    print(f"\n{'='*70}")
    print(f"  RESULTADO M1: Config Melhor Score")
    print(f"{'='*70}")
    print(f"  Trades: {result['trades']}")
    print(f"  Wins: {result['wins']} | Losses: {result['losses']}")
    print(f"  Win Rate: {result['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {result['pf']:.2f}")
    print(f"  PnL Total: {result['pnl']:.1f} pips")
    print(f"  Max Drawdown: {result['max_dd']:.1f}%")

    if result.get('trade_details'):
        print(f"\n  Ultimos 10 trades:")
        for t in result['trade_details'][-10:]:
            ts_str = t['timestamp'].strftime('%Y-%m-%d %H:%M')
            print(f"    {t['direction']:5s} | {ts_str} | "
                  f"PnL: {t['pnl_pips']:+7.1f} pips | {t['result']:4s} | {t['reason']}")

    # Comparacao com H1
    print(f"\n{'='*70}")
    print(f"  COMPARACAO M1 vs H1")
    print(f"{'='*70}")
    print(f"  {'Metrica':<20} {'H1':>15} {'M1':>15}")
    print(f"  {'-'*50}")
    print(f"  {'Win Rate %':<20} {'50.0':>15} {result['win_rate']*100:>15.1f}")
    print(f"  {'Profit Factor':<20} {'2.43':>15} {result['pf']:>15.2f}")
    print(f"  {'Trades':<20} {'6':>15} {result['trades']:>15}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
