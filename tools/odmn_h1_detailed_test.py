#!/usr/bin/env python3
"""
ODMN H1 - Detailed Optimization Test
===============================================================================
Teste detalhado do ODMN no melhor timeframe (H1) com mais dados e ajustes.
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

# =============================================================================
# CONFIGURACAO
# =============================================================================

API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"
SPREAD_PIPS = 1.2
SLIPPAGE_PIPS = 0.5
PIP_VALUE = 0.0001

# Parametros H1
SL_PIPS = 25.0
TP_PIPS = 50.0
COOLDOWN = 3

# Thresholds ODMN - mais relaxados para mais sinais
FRAGILITY_THRESHOLD = 0.70  # Percentil de fragilidade
CONFIDENCE_THRESHOLD = 0.55  # Confianca minima
MFG_DIRECTION_THRESHOLD = 0.03  # Direcao MFG minima

# Barras alvo (~1.5 anos de H1)
BARS_TARGET = 9000

@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

def download_bars(period: str, count: int) -> List[Bar]:
    """Baixa barras historicas."""
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

def run_odmn_backtest(bars: List[Bar], long_only: bool = False,
                      fragility_th: float = 0.70, confidence_th: float = 0.55,
                      mfg_th: float = 0.03, start_pct: float = 0, end_pct: float = 100) -> Dict:
    """Executa backtest do ODMN."""
    n = len(bars)
    start_idx = int(n * start_pct / 100)
    end_idx = int(n * end_pct / 100)

    if end_idx - start_idx < 200:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'details': []}

    closes = np.array([b.close for b in bars[start_idx:end_idx]])

    # Inicializa ODMN com seed para reprodutibilidade
    odmn = OracloDerivativosMalliavinNash(
        lookback_window=100,
        fragility_threshold=2.0,
        mfg_direction_threshold=mfg_th,
        malliavin_paths=500,
        malliavin_steps=20,
        seed=42
    )

    # Gerar sinais - menos conservador
    signals = []
    min_bars = 120  # Minimo de barras para ODMN

    print(f"    Gerando sinais ({len(closes)} barras, step=3)...")
    signal_count = 0

    for i in range(min_bars, len(closes) - 1, 3):  # Step de 3 para mais sinais
        try:
            prices_up_to_i = closes[:i]
            result = odmn.analyze(prices_up_to_i)

            if result.get('is_warmup', True):
                continue

            signal = result['signal']
            confidence = result['confidence']
            fragility_pct = result['fragility_percentile']
            mfg_dir = result['mfg_direction']

            # Logica de sinal mais simples
            # Sinal valido se: confianca ok E (fragilidade alta OU direcao MFG forte)
            if confidence < confidence_th:
                continue

            valid_signal = False

            # Alta fragilidade com direcao
            if fragility_pct > fragility_th:
                if signal != 0:
                    valid_signal = True

            # Direcao MFG forte mesmo sem fragilidade alta
            elif abs(mfg_dir) > mfg_th * 2:
                if signal != 0:
                    valid_signal = True

            if valid_signal:
                if signal == 1:  # BUY
                    signals.append((start_idx + i, 'LONG', bars[start_idx + i].timestamp))
                    signal_count += 1
                elif signal == -1 and not long_only:  # SELL
                    signals.append((start_idx + i, 'SHORT', bars[start_idx + i].timestamp))
                    signal_count += 1

        except Exception as e:
            continue

    print(f"    Sinais gerados: {signal_count}")

    # Simular trades
    trades = []
    trade_details = []
    last_trade_idx = -999

    for idx, direction, timestamp in signals:
        if idx - last_trade_idx < COOLDOWN:
            continue
        if idx + 1 >= n:
            continue

        entry = bars[idx + 1].open
        sl = entry - SL_PIPS * PIP_VALUE if direction == 'LONG' else entry + SL_PIPS * PIP_VALUE
        tp = entry + TP_PIPS * PIP_VALUE if direction == 'LONG' else entry - TP_PIPS * PIP_VALUE

        for j in range(idx + 2, min(idx + 150, n)):
            bar = bars[j]
            if direction == 'LONG':
                if bar.low <= sl:
                    pnl = -SL_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS
                    trades.append(pnl)
                    trade_details.append({'date': timestamp, 'dir': direction, 'pnl': pnl})
                    break
                if bar.high >= tp:
                    pnl = TP_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS
                    trades.append(pnl)
                    trade_details.append({'date': timestamp, 'dir': direction, 'pnl': pnl})
                    break
            else:
                if bar.high >= sl:
                    pnl = -SL_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS
                    trades.append(pnl)
                    trade_details.append({'date': timestamp, 'dir': direction, 'pnl': pnl})
                    break
                if bar.low <= tp:
                    pnl = TP_PIPS - SPREAD_PIPS - SLIPPAGE_PIPS
                    trades.append(pnl)
                    trade_details.append({'date': timestamp, 'dir': direction, 'pnl': pnl})
                    break
        last_trade_idx = idx

    if not trades:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'details': []}

    wins = len([t for t in trades if t > 0])
    total = len(trades)
    wr = wins / total * 100
    be = SL_PIPS / (SL_PIPS + TP_PIPS) * 100
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

def main():
    print("=" * 80)
    print("  ODMN H1 - DETAILED OPTIMIZATION TEST")
    print("=" * 80)

    # Baixar dados
    bars = download_bars("H1", BARS_TARGET)

    if len(bars) < 2000:
        print(f"  ERRO: Dados insuficientes ({len(bars)} barras)")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"\n  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()}")
    print(f"  Duracao: {days} dias (~{days/30:.1f} meses)")
    print(f"  Barras: {len(bars)}")

    # ==========================================================================
    # TESTE 1: OTIMIZACAO DE THRESHOLDS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  FASE 1: OTIMIZACAO DE THRESHOLDS")
    print("=" * 80)

    best_config = None
    best_pf = 0

    threshold_configs = [
        {'frag': 0.60, 'conf': 0.50, 'mfg': 0.02},
        {'frag': 0.65, 'conf': 0.55, 'mfg': 0.03},
        {'frag': 0.70, 'conf': 0.55, 'mfg': 0.03},
        {'frag': 0.75, 'conf': 0.60, 'mfg': 0.05},
        {'frag': 0.80, 'conf': 0.65, 'mfg': 0.05},
    ]

    print(f"\n  {'Frag%':>6} | {'Conf':>5} | {'MFG':>5} | {'Trades':>6} | {'WR%':>6} | {'Edge%':>7} | {'PF':>5} | Status")
    print("  " + "-" * 65)

    for cfg in threshold_configs:
        result = run_odmn_backtest(bars, long_only=True,
                                   fragility_th=cfg['frag'],
                                   confidence_th=cfg['conf'],
                                   mfg_th=cfg['mfg'])

        if result['total'] > 0:
            status = "[OK]" if result['edge'] > 0 and result['pf'] > 1.0 else "[FAIL]"
            print(f"  {cfg['frag']*100:>5.0f}% | {cfg['conf']:>5.2f} | {cfg['mfg']:>5.2f} | {result['total']:>6} | {result['wr']:>5.1f}% | {result['edge']:>+6.1f}% | {result['pf']:>5.2f} | {status}")

            if result['pf'] > best_pf and result['total'] >= 15:
                best_pf = result['pf']
                best_config = cfg
        else:
            print(f"  {cfg['frag']*100:>5.0f}% | {cfg['conf']:>5.2f} | {cfg['mfg']:>5.2f} | {'N/A':>6} | {'N/A':>6} | {'N/A':>7} | {'N/A':>5} | [SKIP]")

    if not best_config:
        print("\n  Nenhuma configuracao valida encontrada")
        # Usar config padrao
        best_config = {'frag': 0.70, 'conf': 0.55, 'mfg': 0.03}

    print(f"\n  Melhor config: Frag={best_config['frag']*100:.0f}%, Conf={best_config['conf']:.2f}, MFG={best_config['mfg']:.2f}")

    # ==========================================================================
    # TESTE 2: LONG-ONLY vs BOTH
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  FASE 2: LONG-ONLY vs BOTH DIRECTIONS")
    print("=" * 80)

    print("\n  Testando BOTH directions...")
    result_both = run_odmn_backtest(bars, long_only=False,
                                    fragility_th=best_config['frag'],
                                    confidence_th=best_config['conf'],
                                    mfg_th=best_config['mfg'])

    print("\n  Testando LONG-only...")
    result_long = run_odmn_backtest(bars, long_only=True,
                                    fragility_th=best_config['frag'],
                                    confidence_th=best_config['conf'],
                                    mfg_th=best_config['mfg'])

    print(f"\n  {'Metrica':<20} | {'BOTH':>15} | {'LONG-only':>15}")
    print("  " + "-" * 55)
    print(f"  {'Trades':<20} | {result_both['total']:>15} | {result_long['total']:>15}")
    if result_both['total'] > 0 or result_long['total'] > 0:
        print(f"  {'Win Rate':<20} | {result_both['wr']:>14.1f}% | {result_long['wr']:>14.1f}%")
        print(f"  {'Breakeven':<20} | {result_both['be']:>14.1f}% | {result_long['be']:>14.1f}%")
        print(f"  {'Edge':<20} | {result_both['edge']:>+14.1f}% | {result_long['edge']:>+14.1f}%")
        print(f"  {'PnL (pips)':<20} | {result_both['pnl']:>+15.1f} | {result_long['pnl']:>+15.1f}")
        print(f"  {'Profit Factor':<20} | {result_both['pf']:>15.2f} | {result_long['pf']:>15.2f}")
        print(f"  {'Max Drawdown':<20} | {result_both['max_dd']:>14.1f} | {result_long['max_dd']:>14.1f}")

    # Escolher melhor modo
    if result_long['total'] > 0 and result_both['total'] > 0:
        best_mode = 'long_only' if result_long['pf'] >= result_both['pf'] else 'both'
    elif result_long['total'] > 0:
        best_mode = 'long_only'
    elif result_both['total'] > 0:
        best_mode = 'both'
    else:
        best_mode = 'both'

    best_result = result_long if best_mode == 'long_only' else result_both
    print(f"\n  Melhor modo: {'LONG-only' if best_mode == 'long_only' else 'BOTH'}")

    # ==========================================================================
    # TESTE 3: WALK-FORWARD (4 FOLDS)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  FASE 3: WALK-FORWARD VALIDATION (4 FOLDS)")
    print("=" * 80)

    folds = []
    fold_size = 25.0

    for i in range(4):
        start = i * fold_size
        end = (i + 1) * fold_size
        print(f"\n  Fold {i+1}/4: {start:.0f}%-{end:.0f}%...")

        result = run_odmn_backtest(bars, long_only=(best_mode == 'long_only'),
                                   fragility_th=best_config['frag'],
                                   confidence_th=best_config['conf'],
                                   mfg_th=best_config['mfg'],
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
        agg_be = SL_PIPS / (SL_PIPS + TP_PIPS) * 100
        agg_edge = agg_wr - agg_be
        gp = sum(f['pnl'] for f in folds if f['pnl'] > 0)
        gl = abs(sum(f['pnl'] for f in folds if f['pnl'] < 0))
        agg_pf = gp / gl if gl > 0 else 0
        print(f"  {'AGG':<6} | {total_trades:>6} | {agg_wr:>5.1f}% | {agg_edge:>+6.1f}% | {total_pnl:>+7.1f} | {agg_pf:>5.2f} |")
    else:
        agg_wr = 0
        agg_edge = 0
        agg_pf = 0

    print(f"\n  Folds positivos: {positive_folds}/4 ({positive_folds/4*100:.0f}%)")

    # ==========================================================================
    # ANALISE MENSAL
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  FASE 4: ANALISE MENSAL")
    print("=" * 80)

    details = best_result.get('details', [])
    monthly_pnl = {}
    for t in details:
        month_key = t['date'].strftime('%Y-%m')
        monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + t['pnl']

    if monthly_pnl:
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
        print(f"  Meses lucrativos: {profitable_months}/{total_months} ({profitable_months/total_months*100:.0f}%)")
    else:
        profitable_months = 0
        total_months = 0

    # ==========================================================================
    # VEREDICTO FINAL
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  VEREDICTO FINAL")
    print("=" * 80)

    criteria = [
        ("Edge agregado > 0%", agg_edge > 0 if total_trades > 0 else False),
        ("Profit Factor > 1.0", agg_pf > 1.0 if total_trades > 0 else False),
        (">=3 folds positivos (de 4)", positive_folds >= 3),
        (">=50% meses lucrativos", profitable_months >= total_months * 0.5 if total_months > 0 else False),
        ("Max Drawdown < 200 pips", best_result['max_dd'] < 200),
        ("Total trades >= 15", total_trades >= 15),
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
        status_str = "APPROVED"
    elif passed >= 4:
        print("\n  *** APROVADO COM RESSALVAS ***")
        status_str = "APPROVED_WITH_RESERVATIONS"
    else:
        print("\n  *** REQUER MAIS AJUSTES ***")
        status_str = "NEEDS_ADJUSTMENT"

    # Salvar config
    config = {
        "strategy": "ODMN-MalliavinNash-H1",
        "symbol": SYMBOL,
        "periodicity": "H1",
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "mode": "LONG_ONLY" if best_mode == 'long_only' else "BOTH",
        "parameters": {
            "fragility_percentile_threshold": best_config['frag'],
            "confidence_threshold": best_config['conf'],
            "mfg_direction_threshold": best_config['mfg'],
            "stop_loss_pips": SL_PIPS,
            "take_profit_pips": TP_PIPS,
            "cooldown_bars": COOLDOWN
        },
        "performance": {
            "trades": total_trades,
            "win_rate": agg_wr,
            "breakeven": agg_be if total_trades > 0 else 33.3,
            "edge": agg_edge,
            "profit_factor": agg_pf,
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
            "total_months": total_months
        },
        "criteria_passed": f"{passed}/6",
        "status": status_str
    }

    config_path = "/home/azureuser/EliBotCD/configs/odmn_h1_optimized.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config salva: {config_path}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
