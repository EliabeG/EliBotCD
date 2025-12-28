#!/usr/bin/env python3
"""
FIFN Fast Test - Versao otimizada que pre-calcula indicadores
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
# PARAMETROS (do config otimizado)
# =============================================================================

SL_PIPS = 22.5
TP_PIPS = 27.9
COOLDOWN = 3  # Barras entre trades
SPREAD_PIPS = 1.2
SLIPPAGE_PIPS = 0.5
PIP_VALUE = 0.0001

# Reynolds thresholds otimizados
REYNOLDS_LOW = 2521.0
REYNOLDS_HIGH = 3786.0
SKEWNESS_THRESHOLD = 0.3091

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

def main():
    print("=" * 70)
    print("  FIFN FAST TEST - Pre-calculo de Indicadores")
    print("=" * 70)

    # Baixar dados H1
    bars = download_bars("H1", 3000)

    if len(bars) < 500:
        print(f"  ERRO: Apenas {len(bars)} barras")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"\n  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()}")
    print(f"  Duracao: {days} dias (~{days/30:.1f} meses)")
    print(f"  Barras: {len(bars)}")

    # Arrays
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    n = len(bars)

    # Criar FIFN
    print("\n  Calculando FIFN (pode demorar)...")
    fifn = FluxoInformacaoFisherNavier(
        window_size=50,
        kl_lookback=10,
        reynolds_sweet_low=REYNOLDS_LOW,
        reynolds_sweet_high=REYNOLDS_HIGH,
        skewness_threshold=SKEWNESS_THRESHOLD
    )

    # Calcular FIFN uma unica vez para todo o dataset
    try:
        result = fifn.analyze(closes)
        reynolds_series = result['reynolds_series']
        skewness_arr = np.zeros(n - 1)

        # Calcular skewness em janela deslizante
        returns = np.diff(np.log(closes))
        from scipy import stats
        for i in range(fifn.window_size, len(returns)):
            skewness_arr[i] = stats.skew(returns[i - fifn.window_size:i])

        print(f"  Reynolds calculado: min={reynolds_series.min():.0f}, max={reynolds_series.max():.0f}")
        print(f"  Skewness calculado: min={skewness_arr.min():.3f}, max={skewness_arr.max():.3f}")

    except Exception as e:
        print(f"  Erro ao calcular FIFN: {e}")
        return

    # Pre-calcular EMA 12/26 e RSI para filtros
    print("\n  Calculando filtros tecnicos...")
    ema_12 = np.zeros(n)
    ema_26 = np.zeros(n)
    rsi_14 = np.full(n, 50.0)

    # EMA
    alpha_12 = 2.0 / 13
    alpha_26 = 2.0 / 27
    ema_12[12] = np.mean(closes[:12])
    ema_26[26] = np.mean(closes[:26])
    for i in range(13, n):
        ema_12[i] = alpha_12 * closes[i-1] + (1 - alpha_12) * ema_12[i-1]
    for i in range(27, n):
        ema_26[i] = alpha_26 * closes[i-1] + (1 - alpha_26) * ema_26[i-1]

    # RSI
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    for i in range(15, n):
        avg_gain = np.mean(gains[i-14:i])
        avg_loss = np.mean(losses[i-14:i])
        if avg_loss == 0:
            rsi_14[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_14[i] = 100.0 - (100.0 / (1.0 + rs))

    # Gerar sinais
    print("\n  Gerando sinais com filtros...")
    signals = []
    signals_filtered = []
    pressure_gradient = result['pressure_gradient_series']

    for i in range(100, n - 1):
        re = reynolds_series[i]
        skew = skewness_arr[i] if i < len(skewness_arr) else 0
        pg = pressure_gradient[i] if i < len(pressure_gradient) else 0

        # Sweet spot
        in_sweet_spot = REYNOLDS_LOW <= re <= REYNOLDS_HIGH

        if not in_sweet_spot:
            continue

        # Filtros tecnicos
        trend_up = ema_12[i] > ema_26[i]
        trend_down = ema_12[i] < ema_26[i]
        rsi = rsi_14[i]

        # Sessao (London/NY overlap 12-16 UTC)
        hour = bars[i].timestamp.hour
        session_ok = 8 <= hour <= 20

        # Sinal de compra: gradiente de pressao negativo + skewness positivo
        if pg < 0 and skew > SKEWNESS_THRESHOLD:
            signals.append((i, 'LONG', bars[i].timestamp))
            # Filtro: tendencia de alta ou neutra + RSI nao sobrecomprado
            if session_ok and (trend_up or abs(ema_12[i] - ema_26[i]) < 0.001) and 35 <= rsi <= 75:
                signals_filtered.append((i, 'LONG', bars[i].timestamp))

        # Sinal de venda: gradiente de pressao positivo + skewness negativo
        elif pg > 0 and skew < -SKEWNESS_THRESHOLD:
            signals.append((i, 'SHORT', bars[i].timestamp))
            # Filtro: tendencia de baixa ou neutra + RSI nao sobrevendido
            if session_ok and (trend_down or abs(ema_12[i] - ema_26[i]) < 0.001) and 25 <= rsi <= 65:
                signals_filtered.append((i, 'SHORT', bars[i].timestamp))

    print(f"  Sinais gerados (sem filtro): {len(signals)}")
    print(f"  Sinais gerados (com filtro): {len(signals_filtered)}")

    if not signals:
        print("\n  NENHUM SINAL GERADO")
        print("  Possíveis causas:")
        print(f"    - Reynolds fora do sweet spot ({REYNOLDS_LOW}-{REYNOLDS_HIGH})")
        print(f"    - Skewness abaixo do threshold ({SKEWNESS_THRESHOLD})")
        return

    # Funcao para simular trades
    def simulate_trades(signal_list, label):
        trades = []
        last_trade_idx = -999

        for idx, direction, timestamp in signal_list:
            if idx - last_trade_idx < COOLDOWN:
                continue

            entry = bars[idx + 1].open
            sl = entry - SL_PIPS * PIP_VALUE if direction == 'LONG' else entry + SL_PIPS * PIP_VALUE
            tp = entry + TP_PIPS * PIP_VALUE if direction == 'LONG' else entry - TP_PIPS * PIP_VALUE

            trade_result = None
            for j in range(idx + 2, min(idx + 200, n)):
                bar = bars[j]
                if direction == 'LONG':
                    if bar.low <= sl:
                        trade_result = ('LOSS', -SL_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS)
                        break
                    if bar.high >= tp:
                        trade_result = ('WIN', TP_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS)
                        break
                else:
                    if bar.high >= sl:
                        trade_result = ('LOSS', -SL_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS)
                        break
                    if bar.low <= tp:
                        trade_result = ('WIN', TP_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS)
                        break

            if trade_result:
                trades.append({
                    'time': timestamp,
                    'direction': direction,
                    'result': trade_result[0],
                    'pnl': trade_result[1]
                })
                last_trade_idx = idx

        return trades

    # Simular ambos
    print("\n" + "-" * 70)
    print("  TESTE SEM FILTROS:")
    print("-" * 70)
    trades_no_filter = simulate_trades(signals, "sem filtro")

    if trades_no_filter:
        total = len(trades_no_filter)
        wins = len([t for t in trades_no_filter if t['result'] == 'WIN'])
        wr = wins / total * 100
        be = SL_PIPS / (SL_PIPS + TP_PIPS) * 100
        edge = wr - be
        pnl = sum(t['pnl'] for t in trades_no_filter)
        gp = sum(t['pnl'] for t in trades_no_filter if t['pnl'] > 0)
        gl = abs(sum(t['pnl'] for t in trades_no_filter if t['pnl'] < 0))
        pf = gp / gl if gl > 0 else 0

        print(f"  Trades: {total}")
        print(f"  Win Rate: {wr:.1f}% (BE: {be:.1f}%)")
        print(f"  Edge: {edge:+.1f}%")
        print(f"  PnL: {pnl:+.1f} pips")
        print(f"  Profit Factor: {pf:.2f}")
    else:
        print("  Nenhum trade")

    print("\n" + "-" * 70)
    print("  TESTE COM FILTROS (EMA/RSI/Sessao):")
    print("-" * 70)
    trades_filtered = simulate_trades(signals_filtered, "com filtro")

    if trades_filtered:
        total = len(trades_filtered)
        wins = len([t for t in trades_filtered if t['result'] == 'WIN'])
        wr = wins / total * 100
        be = SL_PIPS / (SL_PIPS + TP_PIPS) * 100
        edge = wr - be
        pnl = sum(t['pnl'] for t in trades_filtered)
        gp = sum(t['pnl'] for t in trades_filtered if t['pnl'] > 0)
        gl = abs(sum(t['pnl'] for t in trades_filtered if t['pnl'] < 0))
        pf = gp / gl if gl > 0 else 0

        print(f"  Trades: {total}")
        print(f"  Win Rate: {wr:.1f}% (BE: {be:.1f}%)")
        print(f"  Edge: {edge:+.1f}%")
        print(f"  PnL: {pnl:+.1f} pips")
        print(f"  Profit Factor: {pf:.2f}")

        print("\n  Ultimos 5 trades:")
        for t in trades_filtered[-5:]:
            print(f"    {t['time'].strftime('%Y-%m-%d %H:%M')} | {t['direction']:5s} | {t['result']:4s} | {t['pnl']:+.1f} pips")
    else:
        print("  Nenhum trade")

    # Veredicto
    print("\n" + "=" * 70)
    print("  VEREDICTO (com filtros):")
    print("=" * 70)

    if trades_filtered:
        total = len(trades_filtered)
        wins = len([t for t in trades_filtered if t['result'] == 'WIN'])
        wr = wins / total * 100
        be = SL_PIPS / (SL_PIPS + TP_PIPS) * 100
        edge = wr - be
        pnl = sum(t['pnl'] for t in trades_filtered)
        gp = sum(t['pnl'] for t in trades_filtered if t['pnl'] > 0)
        gl = abs(sum(t['pnl'] for t in trades_filtered if t['pnl'] < 0))
        pf = gp / gl if gl > 0 else 0

        if edge > 0:
            print(f"  [OK] Edge positivo: {edge:+.1f}%")
        else:
            print(f"  [FAIL] Edge negativo: {edge:+.1f}%")

        if pf > 1.0:
            print(f"  [OK] Profit Factor > 1: {pf:.2f}")
        else:
            print(f"  [FAIL] Profit Factor < 1: {pf:.2f}")

        if total >= 10:
            print(f"  [OK] Sample size adequado: {total} trades")
        else:
            print(f"  [WARN] Sample size pequeno: {total} trades")
    else:
        print("  [FAIL] Nenhum trade com filtros")

    print("=" * 70)

if __name__ == "__main__":
    main()
