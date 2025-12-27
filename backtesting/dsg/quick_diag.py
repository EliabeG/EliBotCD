#!/usr/bin/env python3
"""
Diagnóstico rápido: Mostra métricas de algumas configurações SEM filtros
"""
import sys
import os
import random
import numpy as np
from datetime import datetime, timezone
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
from api.fxopen_historical_ws import download_historical_data
from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional
from backtesting.common.robust_optimizer import RobustBacktester
from config.execution_costs import SPREAD_PIPS, get_pip_value
from config.optimizer_filters import (
    MIN_TRADES_TRAIN, MIN_TRADES_TEST,
    MIN_WIN_RATE, MAX_WIN_RATE,
    MIN_PROFIT_FACTOR, MAX_PROFIT_FACTOR,
    MAX_DRAWDOWN,
)


async def main():
    print("=" * 70)
    print("  DIAGNÓSTICO RÁPIDO DSG")
    print("=" * 70)

    # Baixar dados
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)
    split_date = datetime(2025, 1, 1, tzinfo=timezone.utc)

    bars = await download_historical_data(
        symbol="EURUSD",
        periodicity="H1",
        start_time=start_date,
        end_time=end_date
    )
    print(f"Barras: {len(bars)}")

    # Split
    split_idx = 0
    for i, bar in enumerate(bars):
        if bar.timestamp >= split_date:
            split_idx = i
            break

    train_bars = bars[:split_idx]
    test_bars = bars[split_idx:]
    print(f"Train: {len(train_bars)}, Test: {len(test_bars)}")

    # Pre-calcular sinais DSG (igual ao optimizer.py)
    print("\nPré-calculando sinais DSG...")
    dsg = DetectorSingularidadeGravitacional(
        ricci_collapse_threshold=-0.5,
        tidal_force_threshold=0.1,
        lookback_window=30
    )

    prices_buf = deque(maxlen=100)
    signals = []
    min_prices = 50

    for i, bar in enumerate(bars):
        prices_buf.append(bar.close)

        if len(prices_buf) < min_prices:
            continue
        if i >= len(bars) - 1:
            continue

        try:
            prices_arr = np.array(prices_buf)
            result = dsg.analyze(prices_arr)  # Sem volumes (gera internamente)

            if 'error' in result and result['error']:
                continue

            next_bar = bars[i + 1]
            signals.append({
                'bar_idx': i,
                'next_bar_idx': i + 1,
                'entry_price': next_bar.open,
                'high': next_bar.high,
                'low': next_bar.low,
                'ricci_scalar': result['Ricci_Scalar'],
                'tidal_force': result['Tidal_Force_Magnitude'],
                'geodesic_direction': result['geodesic_direction'],
                'ricci_collapsing': result['ricci_collapsing'],
                'crossing_horizon': result['crossing_horizon'],
            })
        except:
            continue

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(bars)} barras...")

    train_signals = [s for s in signals if s['bar_idx'] < split_idx]
    test_signals = [s for s in signals if s['bar_idx'] >= split_idx]

    print(f"\nSinais: train={len(train_signals)}, test={len(test_signals)}")

    if not train_signals:
        print("ERRO: Sem sinais!")
        return

    # Estatísticas
    ricci_vals = [s['ricci_scalar'] for s in signals]
    tidal_vals = [s['tidal_force'] for s in signals]
    print(f"\nDistribuição:")
    print(f"  Ricci: min={min(ricci_vals):.2f}, max={max(ricci_vals):.2f}")
    print(f"  Tidal: min={min(tidal_vals):.6f}, max={max(tidal_vals):.6f}")

    # Testar configurações
    print("\n" + "=" * 70)
    print("  TESTANDO CONFIGURAÇÕES (sem filtros)")
    print("=" * 70)

    backtester = RobustBacktester(pip=get_pip_value("EURUSD"), spread=SPREAD_PIPS)

    # Testar várias configurações
    configs = [
        {'ricci': -49000, 'tidal': 0.001, 'sl': 30, 'tp': 50},
        {'ricci': -48000, 'tidal': 0.0015, 'sl': 25, 'tp': 45},
        {'ricci': -47000, 'tidal': 0.002, 'sl': 20, 'tp': 35},
        {'ricci': -46000, 'tidal': 0.003, 'sl': 30, 'tp': 60},
        {'ricci': -50000, 'tidal': 0.0005, 'sl': 25, 'tp': 40},
    ]

    for cfg in configs:
        print(f"\n--- ricci={cfg['ricci']}, tidal={cfg['tidal']}, sl={cfg['sl']}, tp={cfg['tp']} ---")

        def get_entries(sigs, bar_off):
            entries = []
            for s in sigs:
                ricci_cond = s['ricci_scalar'] < cfg['ricci'] or s['ricci_collapsing']
                tidal_cond = s['tidal_force'] > cfg['tidal']
                cross_cond = s['crossing_horizon']
                conds = sum([ricci_cond, tidal_cond, cross_cond])
                if conds >= 2 and s['geodesic_direction'] != 0:
                    exec_idx = s['next_bar_idx'] - bar_off
                    entries.append((exec_idx, s['entry_price'], s['geodesic_direction']))
            return entries

        # Train
        train_entries = get_entries(train_signals, 0)
        if len(train_entries) < 3:
            print(f"  Train: {len(train_entries)} entradas (muito poucas)")
            continue

        train_pnls = []
        last_exit = -1
        for ex_idx, ep, d in train_entries:
            if ex_idx < 0 or ex_idx >= len(train_bars) - 1 or ex_idx <= last_exit:
                continue
            trade = backtester.execute_trade(
                bars=train_bars, entry_idx=ex_idx, entry_price=ep,
                direction=d, sl_pips=cfg['sl'], tp_pips=cfg['tp'], max_bars=200
            )
            train_pnls.append(trade.pnl_pips)
            last_exit = trade.exit_idx

        if not train_pnls:
            print("  Train: 0 trades executados")
            continue

        train_res = backtester.calculate_backtest_result(train_pnls)
        print(f"  TRAIN: {train_res.trades} trades, WR={train_res.win_rate:.1%}, PF={train_res.profit_factor:.2f}, DD={train_res.max_drawdown:.1%}")

        # Verificar filtros train
        train_fails = []
        if train_res.trades < MIN_TRADES_TRAIN:
            train_fails.append(f"trades<{MIN_TRADES_TRAIN}")
        if train_res.win_rate < MIN_WIN_RATE:
            train_fails.append(f"WR<{MIN_WIN_RATE:.0%}")
        if train_res.win_rate > MAX_WIN_RATE:
            train_fails.append(f"WR>{MAX_WIN_RATE:.0%}")
        if train_res.profit_factor < MIN_PROFIT_FACTOR:
            train_fails.append(f"PF<{MIN_PROFIT_FACTOR}")
        if train_res.profit_factor > MAX_PROFIT_FACTOR:
            train_fails.append(f"PF>{MAX_PROFIT_FACTOR}")
        if train_res.max_drawdown > MAX_DRAWDOWN:
            train_fails.append(f"DD>{MAX_DRAWDOWN:.0%}")

        if train_fails:
            print(f"  -> Train falhou: {', '.join(train_fails)}")

        # Test
        test_entries = get_entries(test_signals, split_idx)
        if len(test_entries) < 3:
            print(f"  Test: {len(test_entries)} entradas (muito poucas)")
            continue

        test_pnls = []
        last_exit = -1
        for ex_idx, ep, d in test_entries:
            if ex_idx < 0 or ex_idx >= len(test_bars) - 1 or ex_idx <= last_exit:
                continue
            trade = backtester.execute_trade(
                bars=test_bars, entry_idx=ex_idx, entry_price=ep,
                direction=d, sl_pips=cfg['sl'], tp_pips=cfg['tp'], max_bars=200
            )
            test_pnls.append(trade.pnl_pips)
            last_exit = trade.exit_idx

        if not test_pnls:
            print("  Test: 0 trades executados")
            continue

        test_res = backtester.calculate_backtest_result(test_pnls)
        print(f"  TEST:  {test_res.trades} trades, WR={test_res.win_rate:.1%}, PF={test_res.profit_factor:.2f}, DD={test_res.max_drawdown:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
