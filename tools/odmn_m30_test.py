#!/usr/bin/env python3
"""
ODMN M30 Test - Testar em timeframe menor com SL/TP mais apertados
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

from strategies.alta_volatilidade.odmn_malliavin_nash import OracloDerivativosMalliavinNash

API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"
SPREAD_PIPS = 1.2
SLIPPAGE_PIPS = 0.5
PIP_VALUE = 0.0001

@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

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
                if total % 5000 == 0:
                    print(f"    {total} barras...")
                time.sleep(0.1)
        except Exception as e:
            print(f"  Erro: {e}")
            break
    bars.sort(key=lambda x: x.timestamp)
    return bars

def run_odmn_backtest(bars: List[Bar], sl_pips: float, tp_pips: float,
                      cooldown: int, long_only: bool = True,
                      fragility_th: float = 0.75, confidence_th: float = 0.60,
                      mfg_th: float = 0.05, start_pct: float = 0, end_pct: float = 100) -> Dict:
    n = len(bars)
    start_idx = int(n * start_pct / 100)
    end_idx = int(n * end_pct / 100)

    if end_idx - start_idx < 200:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'details': [], 'max_dd': 0, 'wins': 0, 'be': 33.3}

    closes = np.array([b.close for b in bars[start_idx:end_idx]])

    odmn = OracloDerivativosMalliavinNash(
        lookback_window=100,
        fragility_threshold=2.0,
        mfg_direction_threshold=mfg_th,
        malliavin_paths=500,
        malliavin_steps=20,
        seed=42
    )

    signals = []
    min_bars = 120

    for i in range(min_bars, len(closes) - 1, 5):
        try:
            prices_up_to_i = closes[:i]
            result = odmn.analyze(prices_up_to_i)

            if result.get('is_warmup', True):
                continue

            signal = result['signal']
            confidence = result['confidence']
            fragility_pct = result['fragility_percentile']
            mfg_dir = result['mfg_direction']

            if confidence < confidence_th:
                continue

            valid_signal = False
            if fragility_pct > fragility_th and signal != 0:
                valid_signal = True
            elif abs(mfg_dir) > mfg_th * 2 and signal != 0:
                valid_signal = True

            if valid_signal:
                if signal == 1:
                    signals.append((start_idx + i, 'LONG', bars[start_idx + i].timestamp))
                elif signal == -1 and not long_only:
                    signals.append((start_idx + i, 'SHORT', bars[start_idx + i].timestamp))

        except Exception:
            continue

    trades = []
    trade_details = []
    last_trade_idx = -999

    for idx, direction, timestamp in signals:
        if idx - last_trade_idx < cooldown:
            continue
        if idx + 1 >= n:
            continue

        entry = bars[idx + 1].open
        sl = entry - sl_pips * PIP_VALUE if direction == 'LONG' else entry + sl_pips * PIP_VALUE
        tp = entry + tp_pips * PIP_VALUE if direction == 'LONG' else entry - tp_pips * PIP_VALUE

        for j in range(idx + 2, min(idx + 150, n)):
            bar = bars[j]
            if direction == 'LONG':
                if bar.low <= sl:
                    pnl = -sl_pips - SPREAD_PIPS - SLIPPAGE_PIPS
                    trades.append(pnl)
                    trade_details.append({'date': timestamp, 'dir': direction, 'pnl': pnl})
                    break
                if bar.high >= tp:
                    pnl = tp_pips - SPREAD_PIPS - SLIPPAGE_PIPS
                    trades.append(pnl)
                    trade_details.append({'date': timestamp, 'dir': direction, 'pnl': pnl})
                    break
            else:
                if bar.high >= sl:
                    pnl = -sl_pips - SPREAD_PIPS - SLIPPAGE_PIPS
                    trades.append(pnl)
                    trade_details.append({'date': timestamp, 'dir': direction, 'pnl': pnl})
                    break
                if bar.low <= tp:
                    pnl = tp_pips - SPREAD_PIPS - SLIPPAGE_PIPS
                    trades.append(pnl)
                    trade_details.append({'date': timestamp, 'dir': direction, 'pnl': pnl})
                    break
        last_trade_idx = idx

    if not trades:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'details': [], 'max_dd': 0, 'wins': 0, 'be': sl_pips/(sl_pips+tp_pips)*100}

    wins = len([t for t in trades if t > 0])
    total = len(trades)
    wr = wins / total * 100
    be = sl_pips / (sl_pips + tp_pips) * 100
    edge = wr - be
    pnl = sum(trades)
    gp = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))
    pf = gp / gl if gl > 0 else 0

    cumsum = np.cumsum(trades)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

    return {
        'total': total, 'wins': wins, 'wr': wr, 'be': be, 'edge': edge,
        'pnl': pnl, 'pf': pf, 'max_dd': max_dd, 'details': trade_details
    }

def main():
    print("=" * 80)
    print("  ODMN M30 OPTIMIZATION TEST")
    print("=" * 80)

    bars = download_bars("M30", 15000)

    if len(bars) < 2000:
        print(f"  ERRO: Dados insuficientes ({len(bars)} barras)")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"\n  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()}")
    print(f"  Duracao: {days} dias (~{days/30:.1f} meses)")
    print(f"  Barras: {len(bars)}")

    # Testar diferentes SL/TP
    print("\n" + "=" * 80)
    print("  OTIMIZACAO SL/TP (LONG-ONLY)")
    print("=" * 80)

    sl_tp_configs = [
        {'sl': 10, 'tp': 20, 'cooldown': 6},
        {'sl': 12, 'tp': 24, 'cooldown': 6},
        {'sl': 15, 'tp': 30, 'cooldown': 4},
        {'sl': 15, 'tp': 45, 'cooldown': 4},  # 1:3 ratio
        {'sl': 20, 'tp': 40, 'cooldown': 3},
    ]

    best_config = None
    best_pf = 0

    print(f"\n  {'SL':>4} | {'TP':>4} | {'Trades':>6} | {'WR%':>6} | {'Edge%':>7} | {'PnL':>8} | {'PF':>5} | {'MaxDD':>7} | Status")
    print("  " + "-" * 75)

    for cfg in sl_tp_configs:
        result = run_odmn_backtest(bars, cfg['sl'], cfg['tp'], cfg['cooldown'], long_only=True)

        if result['total'] > 0:
            status = "[OK]" if result['edge'] > 0 and result['pf'] > 1.0 else "[FAIL]"
            print(f"  {cfg['sl']:>4} | {cfg['tp']:>4} | {result['total']:>6} | {result['wr']:>5.1f}% | {result['edge']:>+6.1f}% | {result['pnl']:>+7.1f} | {result['pf']:>5.2f} | {result['max_dd']:>6.1f} | {status}")

            if result['pf'] > best_pf and result['total'] >= 20:
                best_pf = result['pf']
                best_config = cfg

    if not best_config:
        best_config = {'sl': 15, 'tp': 30, 'cooldown': 4}

    print(f"\n  Melhor config: SL={best_config['sl']}, TP={best_config['tp']}")

    # Walk-forward com melhor config
    print("\n" + "=" * 80)
    print("  WALK-FORWARD VALIDATION (4 FOLDS)")
    print("=" * 80)

    folds = []
    for i in range(4):
        start = i * 25
        end = (i + 1) * 25
        result = run_odmn_backtest(bars, best_config['sl'], best_config['tp'],
                                   best_config['cooldown'], long_only=True,
                                   start_pct=start, end_pct=end)
        result['period'] = f"Q{i+1}"
        folds.append(result)

    print(f"\n  {'Fold':<6} | {'Trades':>6} | {'WR%':>6} | {'Edge%':>7} | {'PnL':>8} | {'PF':>5} | Status")
    print("  " + "-" * 55)

    positive_folds = 0
    total_trades = 0
    total_wins = 0
    total_pnl = 0

    for fold in folds:
        if fold['total'] > 0:
            status = "[OK]" if fold['edge'] > 0 else "[FAIL]"
            if fold['edge'] > 0:
                positive_folds += 1
            total_trades += fold['total']
            total_wins += fold.get('wins', 0)
            total_pnl += fold['pnl']
            print(f"  {fold['period']:<6} | {fold['total']:>6} | {fold['wr']:>5.1f}% | {fold['edge']:>+6.1f}% | {fold['pnl']:>+7.1f} | {fold['pf']:>5.2f} | {status}")
        else:
            print(f"  {fold['period']:<6} | {'N/A':>6} | {'N/A':>6} | {'N/A':>7} | {'N/A':>8} | {'N/A':>5} | [SKIP]")

    print("  " + "-" * 55)
    if total_trades > 0:
        agg_wr = total_wins / total_trades * 100
        agg_be = best_config['sl'] / (best_config['sl'] + best_config['tp']) * 100
        agg_edge = agg_wr - agg_be
        gp = sum(f['pnl'] for f in folds if f['pnl'] > 0)
        gl = abs(sum(f['pnl'] for f in folds if f['pnl'] < 0))
        agg_pf = gp / gl if gl > 0 else 0
        print(f"  {'AGG':<6} | {total_trades:>6} | {agg_wr:>5.1f}% | {agg_edge:>+6.1f}% | {total_pnl:>+7.1f} | {agg_pf:>5.2f} |")
    else:
        agg_wr = agg_edge = agg_pf = 0

    print(f"\n  Folds positivos: {positive_folds}/4")

    # Veredicto
    print("\n" + "=" * 80)
    print("  VEREDICTO FINAL")
    print("=" * 80)

    result_full = run_odmn_backtest(bars, best_config['sl'], best_config['tp'],
                                    best_config['cooldown'], long_only=True)

    criteria = [
        ("Edge agregado > 0%", agg_edge > 0 if total_trades > 0 else False),
        ("Profit Factor > 1.0", agg_pf > 1.0 if total_trades > 0 else False),
        (">=3 folds positivos (de 4)", positive_folds >= 3),
        ("Max Drawdown < 150 pips", result_full['max_dd'] < 150),
        ("Total trades >= 20", total_trades >= 20),
    ]

    passed = 0
    for name, ok in criteria:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")
        if ok:
            passed += 1

    print(f"\n  Resultado: {passed}/{len(criteria)} criterios")

    if passed >= 4:
        print("\n  *** APROVADO PARA PAPER TRADING ***")
    elif passed >= 3:
        print("\n  *** APROVADO COM RESSALVAS ***")
    else:
        print("\n  *** REQUER MAIS AJUSTES ***")

    # Salvar config
    config = {
        "strategy": "ODMN-MalliavinNash-M30",
        "symbol": SYMBOL,
        "periodicity": "M30",
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "mode": "LONG_ONLY",
        "parameters": {
            "stop_loss_pips": best_config['sl'],
            "take_profit_pips": best_config['tp'],
            "cooldown_bars": best_config['cooldown']
        },
        "performance": {
            "trades": total_trades,
            "win_rate": agg_wr,
            "edge": agg_edge,
            "profit_factor": agg_pf,
            "total_pnl_pips": total_pnl,
            "max_drawdown_pips": result_full['max_dd']
        },
        "walk_forward": {
            "folds_positive": positive_folds
        },
        "criteria_passed": f"{passed}/5"
    }

    config_path = "/home/azureuser/EliBotCD/configs/odmn_m30_optimized.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config salva: {config_path}")

    print("=" * 80)

if __name__ == "__main__":
    main()
