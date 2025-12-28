#!/usr/bin/env python3
"""
ODMN - Comprehensive Optimization Test
===============================================================================
Teste completo do ODMN (Oraculo de Derivativos de Malliavin-Nash) similar ao FIFN.
Inclui multi-timeframe, LONG-only vs BOTH, e walk-forward validation.
"""

import sys
import os
import urllib.request
import ssl
import json
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.alta_volatilidade.odmn_malliavin_nash import OracloDerivativosMalliavinNash

# =============================================================================
# CONFIGURACAO
# =============================================================================

API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"
SPREAD_PIPS = 1.2
SLIPPAGE_PIPS = 0.5
PIP_VALUE = 0.0001

# Parametros por timeframe
TIMEFRAME_PARAMS = {
    'M5': {'sl': 10, 'tp': 20, 'cooldown': 12, 'bars': 50000},
    'M15': {'sl': 12, 'tp': 24, 'cooldown': 8, 'bars': 30000},
    'M30': {'sl': 15, 'tp': 30, 'cooldown': 6, 'bars': 15000},
    'H1': {'sl': 25, 'tp': 50, 'cooldown': 3, 'bars': 9000},
    'H4': {'sl': 40, 'tp': 80, 'cooldown': 2, 'bars': 3000},
}

# Thresholds ODMN
FRAGILITY_THRESHOLD = 0.80  # Percentil de fragilidade para sinal
CONFIDENCE_THRESHOLD = 0.60  # Confianca minima
MFG_DIRECTION_THRESHOLD = 0.05  # Direcao MFG minima

@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

def download_bars(period: str, count: int) -> List[Bar]:
    """Baixa barras historicas da API."""
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
                      cooldown: int, long_only: bool = False,
                      start_pct: float = 0, end_pct: float = 100) -> Dict:
    """Executa backtest do ODMN."""
    n = len(bars)
    start_idx = int(n * start_pct / 100)
    end_idx = int(n * end_pct / 100)

    if end_idx - start_idx < 200:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0}

    closes = np.array([b.close for b in bars[start_idx:end_idx]])

    # Inicializa ODMN
    odmn = OracloDerivativosMalliavinNash(
        lookback_window=100,
        fragility_threshold=2.0,
        mfg_direction_threshold=MFG_DIRECTION_THRESHOLD,
        malliavin_paths=500,  # Reduzido para velocidade
        malliavin_steps=20,
        seed=42
    )

    # Gerar sinais
    signals = []
    min_bars = 150  # Minimo de barras para ODMN

    for i in range(min_bars, len(closes) - 1, 5):  # Step de 5 para velocidade
        try:
            prices_up_to_i = closes[:i]
            result = odmn.analyze(prices_up_to_i)

            if result.get('is_warmup', True):
                continue

            signal = result['signal']
            confidence = result['confidence']
            fragility_pct = result['fragility_percentile']
            mfg_dir = result['mfg_direction']

            # Filtros
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            if fragility_pct < FRAGILITY_THRESHOLD and abs(mfg_dir) < MFG_DIRECTION_THRESHOLD:
                continue

            if signal == 1:  # BUY
                signals.append((start_idx + i, 'LONG', bars[start_idx + i].timestamp))
            elif signal == -1 and not long_only:  # SELL
                signals.append((start_idx + i, 'SHORT', bars[start_idx + i].timestamp))

        except Exception as e:
            continue

    # Simular trades
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

        for j in range(idx + 2, min(idx + 200, n)):
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
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'details': []}

    wins = len([t for t in trades if t > 0])
    total = len(trades)
    wr = wins / total * 100
    be = sl_pips / (sl_pips + tp_pips) * 100
    edge = wr - be
    pnl = sum(trades)
    gp = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))
    pf = gp / gl if gl > 0 else 0

    # Max Drawdown
    cumsum = np.cumsum(trades)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

    return {
        'total': total,
        'wins': wins,
        'wr': wr,
        'be': be,
        'edge': edge,
        'pnl': pnl,
        'pf': pf,
        'max_dd': max_dd,
        'details': trade_details
    }

def test_timeframe(tf: str, bars: List[Bar]) -> Dict:
    """Testa um timeframe especifico."""
    params = TIMEFRAME_PARAMS[tf]

    print(f"\n  Testando {tf}...")
    print(f"    SL: {params['sl']} | TP: {params['tp']} | Cooldown: {params['cooldown']}")

    # BOTH directions
    result_both = run_odmn_backtest(
        bars, params['sl'], params['tp'], params['cooldown'],
        long_only=False
    )

    # LONG only
    result_long = run_odmn_backtest(
        bars, params['sl'], params['tp'], params['cooldown'],
        long_only=True
    )

    return {
        'timeframe': tf,
        'both': result_both,
        'long_only': result_long,
        'params': params
    }

def run_walk_forward(bars: List[Bar], params: Dict, long_only: bool, n_folds: int = 4) -> List[Dict]:
    """Executa walk-forward validation."""
    folds = []
    fold_size = 100.0 / n_folds

    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size

        result = run_odmn_backtest(
            bars, params['sl'], params['tp'], params['cooldown'],
            long_only=long_only, start_pct=start, end_pct=end
        )
        result['fold'] = i + 1
        result['period'] = f"{start:.0f}%-{end:.0f}%"
        folds.append(result)

    return folds

def main():
    print("=" * 80)
    print("  ODMN - COMPREHENSIVE OPTIMIZATION TEST")
    print("=" * 80)

    # ==========================================================================
    # FASE 1: MULTI-TIMEFRAME TEST
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  FASE 1: MULTI-TIMEFRAME ANALYSIS")
    print("=" * 80)

    results = {}
    best_tf = None
    best_edge = -999

    for tf in ['H1', 'M30', 'H4']:  # Testar principais timeframes
        params = TIMEFRAME_PARAMS[tf]
        bars = download_bars(tf, min(params['bars'], 5000))  # Limitado para velocidade

        if len(bars) < 500:
            print(f"  {tf}: Dados insuficientes ({len(bars)} barras)")
            continue

        days = (bars[-1].timestamp - bars[0].timestamp).days
        print(f"\n  {tf}: {len(bars)} barras (~{days} dias)")

        result = test_timeframe(tf, bars)
        results[tf] = result

        # Melhor baseado em edge LONG-only
        if result['long_only']['total'] > 0:
            edge = result['long_only']['edge']
            if edge > best_edge:
                best_edge = edge
                best_tf = tf

    # Mostrar comparacao
    print("\n" + "=" * 80)
    print("  COMPARACAO MULTI-TIMEFRAME")
    print("=" * 80)

    print(f"\n  {'TF':<6} | {'Mode':<10} | {'Trades':>6} | {'WR%':>6} | {'Edge%':>7} | {'PnL':>8} | {'PF':>5} | Status")
    print("  " + "-" * 75)

    for tf, result in results.items():
        for mode in ['both', 'long_only']:
            r = result[mode]
            if r['total'] > 0:
                status = "[OK]" if r['edge'] > 0 and r['pf'] > 1.0 else "[FAIL]"
                mode_name = "BOTH" if mode == 'both' else "LONG"
                print(f"  {tf:<6} | {mode_name:<10} | {r['total']:>6} | {r['wr']:>5.1f}% | {r['edge']:>+6.1f}% | {r['pnl']:>+7.1f} | {r['pf']:>5.2f} | {status}")

    if best_tf:
        print(f"\n  Melhor timeframe: {best_tf} (Edge: {best_edge:+.1f}%)")
    else:
        print("\n  Nenhum timeframe passou nos criterios")
        return

    # ==========================================================================
    # FASE 2: TESTE DETALHADO NO MELHOR TIMEFRAME
    # ==========================================================================
    print("\n" + "=" * 80)
    print(f"  FASE 2: TESTE DETALHADO - {best_tf}")
    print("=" * 80)

    params = TIMEFRAME_PARAMS[best_tf]
    bars = download_bars(best_tf, params['bars'])

    if len(bars) < 1000:
        print(f"  ERRO: Dados insuficientes ({len(bars)} barras)")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"\n  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()}")
    print(f"  Duracao: {days} dias (~{days/30:.1f} meses)")
    print(f"  Barras: {len(bars)}")

    # ==========================================================================
    # FASE 3: LONG-ONLY vs BOTH
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  FASE 3: LONG-ONLY vs BOTH DIRECTIONS")
    print("=" * 80)

    result_both = run_odmn_backtest(bars, params['sl'], params['tp'], params['cooldown'], long_only=False)
    result_long = run_odmn_backtest(bars, params['sl'], params['tp'], params['cooldown'], long_only=True)

    print(f"\n  {'Metrica':<20} | {'BOTH':>15} | {'LONG-only':>15}")
    print("  " + "-" * 55)
    print(f"  {'Trades':<20} | {result_both['total']:>15} | {result_long['total']:>15}")
    print(f"  {'Win Rate':<20} | {result_both['wr']:>14.1f}% | {result_long['wr']:>14.1f}%")
    print(f"  {'Edge':<20} | {result_both['edge']:>+14.1f}% | {result_long['edge']:>+14.1f}%")
    print(f"  {'PnL (pips)':<20} | {result_both['pnl']:>+15.1f} | {result_long['pnl']:>+15.1f}")
    print(f"  {'Profit Factor':<20} | {result_both['pf']:>15.2f} | {result_long['pf']:>15.2f}")
    print(f"  {'Max Drawdown':<20} | {result_both['max_dd']:>14.1f} | {result_long['max_dd']:>14.1f}")

    # Determinar melhor modo
    best_mode = 'long_only' if result_long['edge'] > result_both['edge'] else 'both'
    best_result = result_long if best_mode == 'long_only' else result_both

    print(f"\n  Melhor modo: {'LONG-only' if best_mode == 'long_only' else 'BOTH'}")

    # ==========================================================================
    # FASE 4: WALK-FORWARD VALIDATION
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  FASE 4: WALK-FORWARD VALIDATION (4 FOLDS)")
    print("=" * 80)

    folds = run_walk_forward(bars, params, long_only=(best_mode == 'long_only'), n_folds=4)

    print(f"\n  {'Fold':<10} | {'Trades':>6} | {'WR%':>6} | {'Edge%':>7} | {'PnL':>8} | {'PF':>5} | Status")
    print("  " + "-" * 60)

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
            print(f"  {fold['period']:<10} | {fold['total']:>6} | {fold['wr']:>5.1f}% | {fold['edge']:>+6.1f}% | {fold['pnl']:>+7.1f} | {fold['pf']:>5.2f} | {status}")
        else:
            print(f"  {fold['period']:<10} | {'N/A':>6} | {'N/A':>6} | {'N/A':>7} | {'N/A':>8} | {'N/A':>5} | [SKIP]")

    # Agregado
    print("  " + "-" * 60)
    if total_trades > 0:
        agg_wr = total_wins / total_trades * 100
        agg_be = params['sl'] / (params['sl'] + params['tp']) * 100
        agg_edge = agg_wr - agg_be
        gp = sum(f['pnl'] for f in folds if f['pnl'] > 0)
        gl = abs(sum(f['pnl'] for f in folds if f['pnl'] < 0))
        agg_pf = gp / gl if gl > 0 else 0
        print(f"  {'AGREGADO':<10} | {total_trades:>6} | {agg_wr:>5.1f}% | {agg_edge:>+6.1f}% | {total_pnl:>+7.1f} | {agg_pf:>5.2f} |")

    print(f"\n  Folds positivos: {positive_folds}/4 ({positive_folds/4*100:.0f}%)")

    # ==========================================================================
    # FASE 5: ANALISE MENSAL
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  FASE 5: ANALISE MENSAL")
    print("=" * 80)

    details = best_result.get('details', [])
    monthly_pnl = {}
    for t in details:
        month_key = t['date'].strftime('%Y-%m')
        monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + t['pnl']

    print(f"\n  {'Mes':<10} | {'PnL (pips)':>12} | Status")
    print("  " + "-" * 35)

    profitable_months = 0
    for month in sorted(monthly_pnl.keys()):
        pnl = monthly_pnl[month]
        status = "[OK]" if pnl > 0 else "[LOSS]"
        if pnl > 0:
            profitable_months += 1
        print(f"  {month:<10} | {pnl:>+12.1f} | {status}")

    total_months = len(monthly_pnl)
    print("  " + "-" * 35)
    if total_months > 0:
        print(f"  Meses lucrativos: {profitable_months}/{total_months} ({profitable_months/total_months*100:.0f}%)")

    # ==========================================================================
    # FASE 6: VEREDICTO FINAL
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  VEREDICTO FINAL")
    print("=" * 80)

    criteria = [
        ("Edge agregado > 0%", agg_edge > 0 if total_trades > 0 else False),
        ("Profit Factor > 1.0", agg_pf > 1.0 if total_trades > 0 else False),
        (">=3 folds positivos (de 4)", positive_folds >= 3),
        (">=50% meses lucrativos", profitable_months >= total_months * 0.5 if total_months > 0 else False),
        ("Max Drawdown < 150 pips", best_result['max_dd'] < 150),
        ("Total trades >= 20", total_trades >= 20),
    ]

    passed = 0
    for name, ok in criteria:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")
        if ok:
            passed += 1

    print(f"\n  Resultado: {passed}/{len(criteria)} criterios")

    if passed >= 5:
        print("\n  *** APROVADO PARA PAPER TRADING ***")
    elif passed >= 4:
        print("\n  *** APROVADO COM RESSALVAS ***")
    else:
        print("\n  *** REQUER MAIS AJUSTES ***")

    # ==========================================================================
    # SALVAR CONFIGURACAO OTIMIZADA
    # ==========================================================================
    config = {
        "strategy": f"ODMN-MalliavinNash-{best_tf}",
        "symbol": SYMBOL,
        "periodicity": best_tf,
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "mode": "LONG_ONLY" if best_mode == 'long_only' else "BOTH",
        "parameters": {
            "fragility_percentile_threshold": FRAGILITY_THRESHOLD,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "mfg_direction_threshold": MFG_DIRECTION_THRESHOLD,
            "stop_loss_pips": params['sl'],
            "take_profit_pips": params['tp'],
            "cooldown_bars": params['cooldown']
        },
        "performance": {
            "trades": total_trades,
            "win_rate": agg_wr if total_trades > 0 else 0,
            "edge": agg_edge if total_trades > 0 else 0,
            "profit_factor": agg_pf if total_trades > 0 else 0,
            "total_pnl_pips": total_pnl,
            "max_drawdown_pips": best_result['max_dd']
        },
        "walk_forward": {
            "folds_tested": 4,
            "folds_positive": positive_folds,
            "pass_rate": f"{positive_folds}/4"
        },
        "monthly": {
            "profitable_months": profitable_months,
            "total_months": total_months,
            "pass_rate": f"{profitable_months}/{total_months}" if total_months > 0 else "N/A"
        },
        "criteria_passed": f"{passed}/6",
        "status": "APPROVED" if passed >= 5 else "APPROVED_WITH_RESERVATIONS" if passed >= 4 else "NEEDS_ADJUSTMENT"
    }

    config_path = f"/home/azureuser/EliBotCD/configs/odmn_{best_tf.lower()}_optimized.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Configuracao salva em: {config_path}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
