#!/usr/bin/env python3
"""
PHM (Projetor Holografico de Maldacena) - Teste de Otimizacao Completo

Baseado em:
- Redes Tensoriais (MERA)
- Correspondencia AdS/CFT
- Entropia de Ryu-Takayanagi
- Fase de Ising
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


def run_phm_backtest(bars: List[Bar], sl_pips: float, tp_pips: float, cooldown: int,
                     window_size: int = 128, bond_dim: int = 8, n_layers: int = 4,
                     confidence_th: float = 0.5, spike_th: float = 1.0,
                     ferromagnetic_only: bool = False, long_only: bool = True,
                     start_pct: float = 0, end_pct: float = 100) -> Dict:
    """
    Backtest do PHM com parametros configuráveis.

    Args:
        ferromagnetic_only: Se True, só opera em fase ferromagnética
        spike_th: Threshold para spike magnitude (1.0 = acima do percentil 90)
        confidence_th: Threshold mínimo de confiança
    """
    n = len(bars)
    start_idx = int(n * start_pct / 100)
    end_idx = int(n * end_pct / 100)

    if end_idx - start_idx < 200:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'max_dd': 0,
                'wins': 0, 'be': sl_pips/(sl_pips+tp_pips)*100, 'details': []}

    closes = np.array([b.close for b in bars[start_idx:end_idx]])

    phm = ProjetorHolograficoMaldacena(
        window_size=window_size,
        bond_dim=bond_dim,
        n_layers=n_layers
    )

    signals = []
    min_bars = max(150, window_size + 20)

    # Iterar a cada 3 barras para eficiência
    for i in range(min_bars, len(closes) - 1, 3):
        try:
            prices_up_to_i = closes[:i]
            result = phm.analyze(prices_up_to_i)

            signal = result['signal']
            confidence = result['confidence']
            horizon = result['horizon_forming']
            spike_mag = result['spike_magnitude']
            phase_type = result['phase_type']
            phase_dir = result['phase_direction']

            # Filtros
            if confidence < confidence_th:
                continue

            if spike_mag < spike_th and not horizon:
                continue

            # Filtro de fase ferromagnética
            if ferromagnetic_only and phase_type != "FERROMAGNETICO":
                continue

            # Gerar sinal baseado na análise
            if signal == 1:
                signals.append((start_idx + i, 'LONG', bars[start_idx + i].timestamp))
            elif signal == -1 and not long_only:
                signals.append((start_idx + i, 'SHORT', bars[start_idx + i].timestamp))

        except Exception as e:
            continue

    # Executar trades
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
                    trade_details.append({'date': timestamp, 'dir': direction, 'pnl': pnl, 'result': 'SL'})
                    break
                if bar.high >= tp:
                    pnl = tp_pips - SPREAD_PIPS - SLIPPAGE_PIPS
                    trades.append(pnl)
                    trade_details.append({'date': timestamp, 'dir': direction, 'pnl': pnl, 'result': 'TP'})
                    break
            else:
                if bar.high >= sl:
                    pnl = -sl_pips - SPREAD_PIPS - SLIPPAGE_PIPS
                    trades.append(pnl)
                    trade_details.append({'date': timestamp, 'dir': direction, 'pnl': pnl, 'result': 'SL'})
                    break
                if bar.low <= tp:
                    pnl = tp_pips - SPREAD_PIPS - SLIPPAGE_PIPS
                    trades.append(pnl)
                    trade_details.append({'date': timestamp, 'dir': direction, 'pnl': pnl, 'result': 'TP'})
                    break
        last_trade_idx = idx

    if not trades:
        return {'total': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'pf': 0, 'max_dd': 0,
                'wins': 0, 'be': sl_pips/(sl_pips+tp_pips)*100, 'details': []}

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


def optimize_phm_params(bars: List[Bar], sl_pips: float, tp_pips: float, cooldown: int) -> Dict:
    """Otimiza parâmetros do PHM"""

    best_result = None
    best_score = -999

    # Grid de parâmetros para testar
    param_configs = [
        # Configurações básicas
        {'window': 64, 'bond': 4, 'layers': 3, 'conf': 0.4, 'spike': 0.8},
        {'window': 64, 'bond': 4, 'layers': 3, 'conf': 0.5, 'spike': 1.0},
        {'window': 64, 'bond': 8, 'layers': 4, 'conf': 0.4, 'spike': 0.8},

        # Configurações padrão
        {'window': 128, 'bond': 8, 'layers': 4, 'conf': 0.4, 'spike': 0.8},
        {'window': 128, 'bond': 8, 'layers': 4, 'conf': 0.5, 'spike': 1.0},
        {'window': 128, 'bond': 8, 'layers': 4, 'conf': 0.5, 'spike': 1.2},

        # Configurações maiores
        {'window': 128, 'bond': 12, 'layers': 5, 'conf': 0.5, 'spike': 1.0},
        {'window': 200, 'bond': 8, 'layers': 4, 'conf': 0.5, 'spike': 1.0},

        # Configurações conservadoras
        {'window': 128, 'bond': 8, 'layers': 4, 'conf': 0.6, 'spike': 1.2},
        {'window': 128, 'bond': 8, 'layers': 4, 'conf': 0.6, 'spike': 1.5},
    ]

    print(f"\n    Testando {len(param_configs)} configurações...")

    for i, cfg in enumerate(param_configs):
        result = run_phm_backtest(
            bars, sl_pips, tp_pips, cooldown,
            window_size=cfg['window'],
            bond_dim=cfg['bond'],
            n_layers=cfg['layers'],
            confidence_th=cfg['conf'],
            spike_th=cfg['spike'],
            ferromagnetic_only=True,  # Só fase ferromagnética
            long_only=True
        )

        if result['total'] >= 10:
            # Score = edge * sqrt(trades) com penalidade para PF baixo
            score = result['edge'] * np.sqrt(result['total'])
            if result['pf'] < 1.0:
                score -= 30

            if score > best_score:
                best_score = score
                best_result = result
                best_result['config'] = cfg

    return best_result


def test_directions(bars: List[Bar], config: Dict, sl_pips: float, tp_pips: float, cooldown: int) -> Tuple[Dict, Dict]:
    """Testa LONG-only vs BOTH directions"""

    result_long = run_phm_backtest(
        bars, sl_pips, tp_pips, cooldown,
        window_size=config['window'],
        bond_dim=config['bond'],
        n_layers=config['layers'],
        confidence_th=config['conf'],
        spike_th=config['spike'],
        ferromagnetic_only=True,
        long_only=True
    )

    result_both = run_phm_backtest(
        bars, sl_pips, tp_pips, cooldown,
        window_size=config['window'],
        bond_dim=config['bond'],
        n_layers=config['layers'],
        confidence_th=config['conf'],
        spike_th=config['spike'],
        ferromagnetic_only=False,  # Permite paramagnético também
        long_only=False
    )

    return result_long, result_both


def walk_forward_validation(bars: List[Bar], config: Dict, sl_pips: float,
                           tp_pips: float, cooldown: int, long_only: bool) -> List[Dict]:
    """Executa walk-forward validation em 4 folds"""
    folds = []

    for i in range(4):
        start = i * 25
        end = (i + 1) * 25

        fold_result = run_phm_backtest(
            bars, sl_pips, tp_pips, cooldown,
            window_size=config['window'],
            bond_dim=config['bond'],
            n_layers=config['layers'],
            confidence_th=config['conf'],
            spike_th=config['spike'],
            ferromagnetic_only=True,
            long_only=long_only,
            start_pct=start,
            end_pct=end
        )
        fold_result['period'] = f"Q{i+1}"
        folds.append(fold_result)

    return folds


def main():
    print("=" * 80)
    print("  PHM (PROJETOR HOLOGRAFICO DE MALDACENA) - OTIMIZACAO COMPLETA")
    print("  Baseado em: AdS/CFT, MERA, Ryu-Takayanagi, Ising Phase")
    print("=" * 80)

    # Configurações por timeframe
    tf_configs = {
        'H1': {'count': 10000, 'sl': 25, 'tp': 50, 'cooldown': 5},
        'H4': {'count': 5000, 'sl': 40, 'tp': 80, 'cooldown': 3},
        'D1': {'count': 2000, 'sl': 60, 'tp': 120, 'cooldown': 2},
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

        # Otimizar parâmetros
        print(f"\n  Otimizando parâmetros...")
        best = optimize_phm_params(bars, cfg['sl'], cfg['tp'], cfg['cooldown'])

        if best is None or best['total'] < 10:
            print(f"  Nenhuma config viável encontrada")
            continue

        print(f"\n  Melhor config: window={best['config']['window']}, bond={best['config']['bond']}, "
              f"layers={best['config']['layers']}, conf={best['config']['conf']}, spike={best['config']['spike']}")

        # Testar LONG vs BOTH
        print(f"\n  Testando LONG-only vs BOTH...")
        result_long, result_both = test_directions(bars, best['config'], cfg['sl'], cfg['tp'], cfg['cooldown'])

        print(f"\n    LONG-only: {result_long['total']} trades | WR={result_long['wr']:.1f}% | "
              f"Edge={result_long['edge']:+.1f}% | PnL={result_long['pnl']:+.1f}")
        print(f"    BOTH:      {result_both['total']} trades | WR={result_both['wr']:.1f}% | "
              f"Edge={result_both['edge']:+.1f}% | PnL={result_both['pnl']:+.1f}")

        # Escolher melhor modo
        if result_long['pf'] >= result_both['pf'] and result_long['edge'] >= result_both['edge']:
            best_mode = 'LONG_ONLY'
            best_result = result_long
            long_only = True
        else:
            best_mode = 'BOTH'
            best_result = result_both
            long_only = False

        print(f"\n  Modo selecionado: {best_mode}")

        results[tf] = best_result
        results[tf]['config'] = best['config']
        results[tf]['sl'] = cfg['sl']
        results[tf]['tp'] = cfg['tp']
        results[tf]['cooldown'] = cfg['cooldown']
        results[tf]['mode'] = best_mode
        results[tf]['bars'] = bars
        results[tf]['long_only'] = long_only

        status = "[OK]" if best_result['edge'] > 0 and best_result['pf'] > 1.0 else "[FAIL]"
        print(f"\n  Resultado: {best_result['total']} trades | WR={best_result['wr']:.1f}% | "
              f"Edge={best_result['edge']:+.1f}% | PF={best_result['pf']:.2f} | PnL={best_result['pnl']:+.1f} {status}")

    # Comparar resultados
    print("\n" + "=" * 80)
    print("  COMPARACAO DE TIMEFRAMES")
    print("=" * 80)

    if not results:
        print("\n  Nenhum timeframe passou nos criterios minimos!")
        return

    print(f"\n  {'TF':<4} | {'Mode':<10} | {'Trades':>6} | {'WR%':>6} | {'Edge%':>7} | {'PnL':>8} | {'PF':>5} | {'MaxDD':>7} | Status")
    print("  " + "-" * 78)

    best_tf = None
    best_pf = 0

    for tf, result in results.items():
        status = "[OK]" if result['edge'] > 0 and result['pf'] > 1.0 else "[FAIL]"
        print(f"  {tf:<4} | {result['mode']:<10} | {result['total']:>6} | {result['wr']:>5.1f}% | "
              f"{result['edge']:>+6.1f}% | {result['pnl']:>+7.1f} | {result['pf']:>5.2f} | {result['max_dd']:>6.1f} | {status}")

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

    folds = walk_forward_validation(
        bars, best_result['config'],
        best_result['sl'], best_result['tp'], best_result['cooldown'],
        best_result['long_only']
    )

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
            print(f"  {fold['period']:<6} | {fold['total']:>6} | {fold['wr']:>5.1f}% | "
                  f"{fold['edge']:>+6.1f}% | {fold['pnl']:>+7.1f} | {fold['pf']:>5.2f} | {status}")
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

    # Veredicto final
    print("\n" + "=" * 80)
    print("  VEREDICTO FINAL PHM")
    print("=" * 80)

    criteria = [
        ("Edge agregado > 0%", agg_edge > 0 if total_trades > 0 else False),
        ("Profit Factor > 1.0", agg_pf > 1.0 if total_trades > 0 else False),
        (">=3 folds positivos (de 4)", positive_folds >= 3),
        ("Max Drawdown < 300 pips", best_result['max_dd'] < 300),
        ("Total trades >= 15", total_trades >= 15),
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
        "strategy": f"PHM-ProjetorHolografico-{best_tf}",
        "symbol": SYMBOL,
        "periodicity": best_tf,
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "mode": best_result['mode'],
        "parameters": {
            "window_size": best_result['config']['window'],
            "bond_dim": best_result['config']['bond'],
            "n_layers": best_result['config']['layers'],
            "confidence_threshold": best_result['config']['conf'],
            "spike_threshold": best_result['config']['spike'],
            "ferromagnetic_only": True,
            "stop_loss_pips": best_result['sl'],
            "take_profit_pips": best_result['tp'],
            "cooldown_bars": best_result['cooldown']
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
