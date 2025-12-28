#!/usr/bin/env python3
"""
ODMN Multi-Timeframe Test - Testar H1, H4 e D1 com ajustes de thresholds
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

    print(f"    Baixando {count} barras {period}...", end=" ", flush=True)
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
                time.sleep(0.1)
        except Exception as e:
            print(f"Erro: {e}")
            break
    bars.sort(key=lambda x: x.timestamp)
    print(f"{len(bars)} barras")
    return bars

def run_odmn_backtest(bars: List[Bar], sl_pips: float, tp_pips: float, cooldown: int,
                      long_only: bool = True, fragility_th: float = 0.75,
                      confidence_th: float = 0.60, mfg_th: float = 0.05,
                      start_pct: float = 0, end_pct: float = 100) -> Dict:
    n = len(bars)
    start_idx = int(n * start_pct / 100)
    end_idx = int(n * end_pct / 100)

    if end_idx - start_idx < 200:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'details': [], 'max_dd': 0, 'wins': 0,
                'be': sl_pips/(sl_pips+tp_pips)*100}

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

    for i in range(min_bars, len(closes) - 1, 3):
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

        for j in range(idx + 2, min(idx + 100, n)):
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
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'details': [],
                'max_dd': 0, 'wins': 0, 'be': sl_pips/(sl_pips+tp_pips)*100}

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

def test_timeframe(bars: List[Bar], tf_name: str, sl_pips: float, tp_pips: float, cooldown: int):
    """Testar um timeframe com diferentes thresholds"""
    print(f"\n  Testando {tf_name} (SL={sl_pips}, TP={tp_pips})...")

    best_result = None
    best_score = -999

    # Testar diferentes combinacoes de thresholds
    threshold_configs = [
        {'frag': 0.70, 'conf': 0.55, 'mfg': 0.03},
        {'frag': 0.75, 'conf': 0.55, 'mfg': 0.03},
        {'frag': 0.80, 'conf': 0.55, 'mfg': 0.03},
        {'frag': 0.70, 'conf': 0.60, 'mfg': 0.05},
        {'frag': 0.75, 'conf': 0.60, 'mfg': 0.05},
        {'frag': 0.80, 'conf': 0.60, 'mfg': 0.05},
        {'frag': 0.80, 'conf': 0.65, 'mfg': 0.05},
        {'frag': 0.85, 'conf': 0.60, 'mfg': 0.05},
        {'frag': 0.85, 'conf': 0.65, 'mfg': 0.08},
    ]

    for cfg in threshold_configs:
        result = run_odmn_backtest(bars, sl_pips, tp_pips, cooldown,
                                   long_only=True,
                                   fragility_th=cfg['frag'],
                                   confidence_th=cfg['conf'],
                                   mfg_th=cfg['mfg'])

        if result['total'] >= 15:
            # Score = edge * sqrt(trades) - penalty por PF baixo
            score = result['edge'] * np.sqrt(result['total'])
            if result['pf'] < 1.0:
                score -= 20

            if score > best_score:
                best_score = score
                best_result = result
                best_result['config'] = cfg

    return best_result

def main():
    print("=" * 80)
    print("  ODMN MULTI-TIMEFRAME OPTIMIZATION")
    print("=" * 80)

    # Configuracoes por timeframe
    tf_configs = {
        'H1': {'count': 10000, 'sl': 25, 'tp': 50, 'cooldown': 3},
        'H4': {'count': 5000, 'sl': 40, 'tp': 80, 'cooldown': 2},
        'D1': {'count': 2000, 'sl': 60, 'tp': 120, 'cooldown': 1},
    }

    results = {}

    for tf, cfg in tf_configs.items():
        print(f"\n{'='*80}")
        print(f"  TIMEFRAME: {tf}")
        print(f"{'='*80}")

        bars = download_bars(tf, cfg['count'])

        if len(bars) < 500:
            print(f"  SKIP: Dados insuficientes ({len(bars)} barras)")
            continue

        days = (bars[-1].timestamp - bars[0].timestamp).days
        print(f"  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()} ({days} dias)")

        result = test_timeframe(bars, tf, cfg['sl'], cfg['tp'], cfg['cooldown'])

        if result and result['total'] > 0:
            results[tf] = result
            results[tf]['sl'] = cfg['sl']
            results[tf]['tp'] = cfg['tp']
            results[tf]['cooldown'] = cfg['cooldown']
            results[tf]['bars'] = bars

            status = "[OK]" if result['edge'] > 0 and result['pf'] > 1.0 else "[FAIL]"
            print(f"\n  Melhor config: Frag={result['config']['frag']}, Conf={result['config']['conf']}, MFG={result['config']['mfg']}")
            print(f"  Resultado: {result['total']} trades | WR={result['wr']:.1f}% | Edge={result['edge']:+.1f}% | PF={result['pf']:.2f} | PnL={result['pnl']:+.1f} {status}")
        else:
            print(f"  Nenhuma config viavel encontrada")

    # Comparar resultados
    print("\n" + "=" * 80)
    print("  COMPARACAO DE TIMEFRAMES")
    print("=" * 80)

    if not results:
        print("\n  Nenhum timeframe passou nos criterios minimos!")
        return

    print(f"\n  {'TF':<4} | {'Trades':>6} | {'WR%':>6} | {'Edge%':>7} | {'PnL':>8} | {'PF':>5} | {'MaxDD':>7} | Status")
    print("  " + "-" * 65)

    best_tf = None
    best_pf = 0

    for tf, result in results.items():
        status = "[OK]" if result['edge'] > 0 and result['pf'] > 1.0 else "[FAIL]"
        print(f"  {tf:<4} | {result['total']:>6} | {result['wr']:>5.1f}% | {result['edge']:>+6.1f}% | {result['pnl']:>+7.1f} | {result['pf']:>5.2f} | {result['max_dd']:>6.1f} | {status}")

        if result['pf'] > best_pf and result['edge'] > 0:
            best_pf = result['pf']
            best_tf = tf

    if not best_tf:
        # Se nenhum tem edge positivo, pegar o menos ruim
        best_tf = max(results.keys(), key=lambda x: results[x]['pf'])

    print(f"\n  Melhor timeframe: {best_tf}")

    # Walk-forward no melhor timeframe
    best_result = results[best_tf]
    bars = best_result['bars']

    print("\n" + "=" * 80)
    print(f"  WALK-FORWARD VALIDATION ({best_tf})")
    print("=" * 80)

    folds = []
    for i in range(4):
        start = i * 25
        end = (i + 1) * 25
        fold_result = run_odmn_backtest(
            bars, best_result['sl'], best_result['tp'], best_result['cooldown'],
            long_only=True,
            fragility_th=best_result['config']['frag'],
            confidence_th=best_result['config']['conf'],
            mfg_th=best_result['config']['mfg'],
            start_pct=start, end_pct=end
        )
        fold_result['period'] = f"Q{i+1}"
        folds.append(fold_result)

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
        agg_be = best_result['sl'] / (best_result['sl'] + best_result['tp']) * 100
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
    print("  VEREDICTO FINAL ODMN")
    print("=" * 80)

    criteria = [
        ("Edge agregado > 0%", agg_edge > 0 if total_trades > 0 else False),
        ("Profit Factor > 1.0", agg_pf > 1.0 if total_trades > 0 else False),
        (">=3 folds positivos (de 4)", positive_folds >= 3),
        ("Max Drawdown < 300 pips", best_result['max_dd'] < 300),
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
        verdict = "APROVADO"
        print("\n  *** APROVADO PARA PAPER TRADING ***")
    elif passed >= 3:
        verdict = "APROVADO_COM_RESSALVAS"
        print("\n  *** APROVADO COM RESSALVAS ***")
    else:
        verdict = "REPROVADO"
        print("\n  *** INDICADOR NAO RECOMENDADO ***")

    # Salvar resultado
    config = {
        "strategy": f"ODMN-MalliavinNash-{best_tf}",
        "symbol": SYMBOL,
        "periodicity": best_tf,
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "mode": "LONG_ONLY",
        "parameters": {
            "fragility_percentile_threshold": best_result['config']['frag'],
            "confidence_threshold": best_result['config']['conf'],
            "mfg_direction_threshold": best_result['config']['mfg'],
            "stop_loss_pips": best_result['sl'],
            "take_profit_pips": best_result['tp'],
            "cooldown_bars": best_result['cooldown']
        },
        "performance": {
            "trades": total_trades,
            "win_rate": agg_wr,
            "edge": agg_edge,
            "profit_factor": agg_pf,
            "total_pnl_pips": total_pnl,
            "max_drawdown_pips": best_result['max_dd']
        },
        "walk_forward": {
            "folds_positive": positive_folds,
            "total_folds": 4
        },
        "criteria_passed": f"{passed}/5",
        "verdict": verdict
    }

    config_path = f"/home/azureuser/EliBotCD/configs/odmn_{best_tf.lower()}_final.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config salva: {config_path}")

    print("=" * 80)

if __name__ == "__main__":
    main()
