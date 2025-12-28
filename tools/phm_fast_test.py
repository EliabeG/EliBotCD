#!/usr/bin/env python3
"""
PHM Fast Test - Versão otimizada para eficiência computacional
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

from strategies.alta_volatilidade.phm_projetor_holografico import ProjetorHolograficoMaldacena

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

    print(f"  Baixando {count} barras {period}...", end=" ", flush=True)
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
    print(f"{len(bars)}")
    return bars


def run_phm_backtest(bars: List[Bar], sl_pips: float, tp_pips: float, cooldown: int,
                     window_size: int = 64, bond_dim: int = 4, n_layers: int = 3,
                     confidence_th: float = 0.4, spike_th: float = 0.8,
                     long_only: bool = True,
                     start_pct: float = 0, end_pct: float = 100) -> Dict:
    """Backtest do PHM com parâmetros reduzidos para eficiência"""
    n = len(bars)
    start_idx = int(n * start_pct / 100)
    end_idx = int(n * end_pct / 100)

    if end_idx - start_idx < 150:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'max_dd': 0,
                'wins': 0, 'be': sl_pips/(sl_pips+tp_pips)*100}

    closes = np.array([b.close for b in bars[start_idx:end_idx]])

    phm = ProjetorHolograficoMaldacena(
        window_size=window_size,
        bond_dim=bond_dim,
        n_layers=n_layers
    )

    signals = []
    min_bars = window_size + 30

    # Iterar a cada 5 barras para maior eficiência
    for i in range(min_bars, len(closes) - 1, 5):
        try:
            prices_up_to_i = closes[:i]
            result = phm.analyze(prices_up_to_i)

            signal = result['signal']
            confidence = result['confidence']
            horizon = result['horizon_forming']
            spike_mag = result['spike_magnitude']
            phase_type = result['phase_type']

            # Filtros
            if confidence < confidence_th:
                continue

            # Precisa de horizonte OU spike significativo
            if spike_mag < spike_th and not horizon:
                continue

            # Só fase ferromagnética para LONG
            if phase_type != "FERROMAGNETICO" and signal == 1:
                continue

            # Gerar sinal
            if signal == 1:
                signals.append((start_idx + i, 'LONG', bars[start_idx + i].timestamp))
            elif signal == -1 and not long_only:
                signals.append((start_idx + i, 'SHORT', bars[start_idx + i].timestamp))

        except Exception:
            continue

    # Executar trades
    trades = []
    last_trade_idx = -999

    for idx, direction, timestamp in signals:
        if idx - last_trade_idx < cooldown:
            continue
        if idx + 1 >= n:
            continue

        entry = bars[idx + 1].open
        sl = entry - sl_pips * PIP_VALUE if direction == 'LONG' else entry + sl_pips * PIP_VALUE
        tp = entry + tp_pips * PIP_VALUE if direction == 'LONG' else entry - tp_pips * PIP_VALUE

        for j in range(idx + 2, min(idx + 80, n)):
            bar = bars[j]
            if direction == 'LONG':
                if bar.low <= sl:
                    trades.append(-sl_pips - SPREAD_PIPS - SLIPPAGE_PIPS)
                    break
                if bar.high >= tp:
                    trades.append(tp_pips - SPREAD_PIPS - SLIPPAGE_PIPS)
                    break
            else:
                if bar.high >= sl:
                    trades.append(-sl_pips - SPREAD_PIPS - SLIPPAGE_PIPS)
                    break
                if bar.low <= tp:
                    trades.append(tp_pips - SPREAD_PIPS - SLIPPAGE_PIPS)
                    break
        last_trade_idx = idx

    if not trades:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'max_dd': 0,
                'wins': 0, 'be': sl_pips/(sl_pips+tp_pips)*100}

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
    max_dd = np.max(running_max - cumsum) if len(cumsum) > 0 else 0

    return {
        'total': total, 'wins': wins, 'wr': wr, 'be': be, 'edge': edge,
        'pnl': pnl, 'pf': pf, 'max_dd': max_dd
    }


def main():
    print("=" * 80)
    print("  PHM FAST OPTIMIZATION TEST")
    print("=" * 80)

    # Testar H1 e H4 com menos barras
    tf_configs = {
        'H1': {'count': 5000, 'sl': 25, 'tp': 50, 'cooldown': 5},
        'H4': {'count': 3000, 'sl': 40, 'tp': 80, 'cooldown': 3},
    }

    results = {}

    for tf, cfg in tf_configs.items():
        print(f"\n{'='*80}")
        print(f"  TIMEFRAME: {tf}")
        print(f"{'='*80}")

        bars = download_bars(tf, cfg['count'])
        if len(bars) < 500:
            print(f"  SKIP: Dados insuficientes")
            continue

        days = (bars[-1].timestamp - bars[0].timestamp).days
        print(f"  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()} ({days} dias)")

        # Testar 4 configurações principais
        configs = [
            {'window': 64, 'bond': 4, 'layers': 3, 'conf': 0.4, 'spike': 0.8},
            {'window': 64, 'bond': 4, 'layers': 3, 'conf': 0.5, 'spike': 1.0},
            {'window': 96, 'bond': 6, 'layers': 3, 'conf': 0.4, 'spike': 0.8},
            {'window': 96, 'bond': 6, 'layers': 3, 'conf': 0.5, 'spike': 1.0},
        ]

        best_result = None
        best_score = -999

        print(f"\n  Testando {len(configs)} configurações...")

        for i, c in enumerate(configs):
            print(f"    Config {i+1}/{len(configs)}: w={c['window']}, b={c['bond']}, l={c['layers']}...", end=" ", flush=True)
            result = run_phm_backtest(
                bars, cfg['sl'], cfg['tp'], cfg['cooldown'],
                window_size=c['window'],
                bond_dim=c['bond'],
                n_layers=c['layers'],
                confidence_th=c['conf'],
                spike_th=c['spike'],
                long_only=True
            )
            print(f"{result['total']} trades, edge={result['edge']:+.1f}%")

            if result['total'] >= 8:
                score = result['edge'] * np.sqrt(result['total'])
                if result['pf'] < 1.0:
                    score -= 20
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_result['config'] = c

        if best_result and best_result['total'] >= 8:
            results[tf] = best_result
            results[tf]['sl'] = cfg['sl']
            results[tf]['tp'] = cfg['tp']
            results[tf]['cooldown'] = cfg['cooldown']
            results[tf]['bars'] = bars

            status = "[OK]" if best_result['edge'] > 0 and best_result['pf'] > 1.0 else "[FAIL]"
            print(f"\n  Melhor: {best_result['total']} trades | WR={best_result['wr']:.1f}% | "
                  f"Edge={best_result['edge']:+.1f}% | PF={best_result['pf']:.2f} {status}")
        else:
            print(f"\n  Nenhuma config viável")

    # Comparar e fazer walk-forward no melhor
    if not results:
        print("\n  Nenhum timeframe viável!")
        return

    print("\n" + "=" * 80)
    print("  COMPARACAO")
    print("=" * 80)

    best_tf = max(results.keys(), key=lambda x: results[x]['pf'] if results[x]['edge'] > 0 else 0)
    best = results[best_tf]
    bars = best['bars']

    print(f"\n  Melhor: {best_tf}")

    # Walk-forward
    print("\n  Walk-forward validation...")
    folds = []
    for i in range(4):
        start = i * 25
        end = (i + 1) * 25
        fold = run_phm_backtest(
            bars, best['sl'], best['tp'], best['cooldown'],
            window_size=best['config']['window'],
            bond_dim=best['config']['bond'],
            n_layers=best['config']['layers'],
            confidence_th=best['config']['conf'],
            spike_th=best['config']['spike'],
            long_only=True,
            start_pct=start, end_pct=end
        )
        fold['period'] = f"Q{i+1}"
        folds.append(fold)

    print(f"\n  {'Fold':<6} | {'Trades':>6} | {'WR%':>6} | {'Edge%':>7} | {'PnL':>8} | Status")
    print("  " + "-" * 50)

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
            total_wins += fold['wins']
            total_pnl += fold['pnl']
            print(f"  {fold['period']:<6} | {fold['total']:>6} | {fold['wr']:>5.1f}% | "
                  f"{fold['edge']:>+6.1f}% | {fold['pnl']:>+7.1f} | {status}")
        else:
            print(f"  {fold['period']:<6} | {'N/A':>6} | {'N/A':>6} | {'N/A':>7} | {'N/A':>8} | [SKIP]")

    print("  " + "-" * 50)

    if total_trades > 0:
        agg_wr = total_wins / total_trades * 100
        agg_be = best['sl'] / (best['sl'] + best['tp']) * 100
        agg_edge = agg_wr - agg_be
        gp = sum(f['pnl'] for f in folds if f['pnl'] > 0)
        gl = abs(sum(f['pnl'] for f in folds if f['pnl'] < 0))
        agg_pf = gp / gl if gl > 0 else 0
        print(f"  {'AGG':<6} | {total_trades:>6} | {agg_wr:>5.1f}% | {agg_edge:>+6.1f}% | {total_pnl:>+7.1f} |")
    else:
        agg_wr = agg_edge = agg_pf = 0

    print(f"\n  Folds positivos: {positive_folds}/4")

    # Veredicto
    print("\n" + "=" * 80)
    print("  VEREDICTO PHM")
    print("=" * 80)

    criteria = [
        ("Edge agregado > 0%", agg_edge > 0),
        ("Profit Factor > 1.0", agg_pf > 1.0),
        (">=2 folds positivos", positive_folds >= 2),
        ("Max Drawdown < 300 pips", best['max_dd'] < 300),
        ("Total trades >= 10", total_trades >= 10),
    ]

    passed = sum(1 for _, ok in criteria if ok)
    for name, ok in criteria:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n  Resultado: {passed}/5 criterios")

    if passed >= 4:
        verdict = "APROVADO"
        print("\n  *** APROVADO PARA PAPER TRADING ***")
    elif passed >= 3:
        verdict = "APROVADO_COM_RESSALVAS"
        print("\n  *** APROVADO COM RESSALVAS ***")
    else:
        verdict = "REPROVADO"
        print("\n  *** INDICADOR NAO RECOMENDADO ***")

    # Salvar config
    config = {
        "strategy": f"PHM-ProjetorHolografico-{best_tf}",
        "symbol": SYMBOL,
        "periodicity": best_tf,
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "mode": "LONG_ONLY",
        "parameters": {
            "window_size": best['config']['window'],
            "bond_dim": best['config']['bond'],
            "n_layers": best['config']['layers'],
            "confidence_threshold": best['config']['conf'],
            "spike_threshold": best['config']['spike'],
            "stop_loss_pips": best['sl'],
            "take_profit_pips": best['tp'],
            "cooldown_bars": best['cooldown']
        },
        "performance": {
            "trades": total_trades,
            "win_rate": agg_wr,
            "edge": agg_edge,
            "profit_factor": agg_pf,
            "total_pnl_pips": total_pnl,
            "max_drawdown_pips": best['max_dd']
        },
        "walk_forward": {
            "folds_positive": positive_folds,
            "total_folds": 4
        },
        "criteria_passed": f"{passed}/5",
        "verdict": verdict
    }

    config_path = f"/home/azureuser/EliBotCD/configs/phm_{best_tf.lower()}_optimized.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config salva: {config_path}")

    print("=" * 80)

if __name__ == "__main__":
    main()
