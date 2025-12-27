#!/usr/bin/env python3
"""
Diagnóstico: Por que nenhuma configuração passa nos filtros?
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
    MIN_WIN_RATE_TEST, MAX_WIN_RATE_TEST,
    MIN_PROFIT_FACTOR_TEST, MAX_PROFIT_FACTOR_TEST,
    MAX_DRAWDOWN_TEST,
    MIN_PF_RATIO, MIN_WR_RATIO,
)
from config.volume_generator import generate_synthetic_volumes, generate_single_volume


async def main():
    print("=" * 70)
    print("  DIAGNÓSTICO DSG")
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

    # Pre-calcular sinais DSG
    print("\nPré-calculando sinais DSG...")
    dsg = DetectorSingularidadeGravitacional(
        ricci_collapse_threshold=-0.5,
        tidal_force_threshold=0.1,
        lookback_window=30
    )

    prices_buf = deque(maxlen=100)
    signals = []
    min_prices = 50
    valid_results = 0
    empty_results = 0
    error_results = 0
    first_result = None

    for i, bar in enumerate(bars):
        if i % 500 == 0:
            print(f"  Processando barra {i}/{len(bars)}...")
        prices_buf.append(bar.close)

        if len(prices_buf) >= min_prices and i < len(bars) - 1:
            prices = list(prices_buf)
            prices_arr = np.array(prices)
            volumes, _ = generate_synthetic_volumes(prices_arr)
            result = dsg.analyze(prices, volumes)

            # Verificar se resultado é válido
            if not result:
                empty_results += 1
                continue
            if isinstance(result, dict) and result.get('error'):
                error_results += 1
                if error_results <= 3:
                    print(f"  ERRO na barra {i}: {result.get('error')}")
                continue

            valid_results += 1

            # Debug: mostrar primeiro resultado
            if first_result is None:
                first_result = result
                print(f"\n  Primeiro resultado DSG (barra {i}):")
                print(f"    Ricci_Scalar: {result.get('Ricci_Scalar')}")
                print(f"    Tidal_Force: {result.get('Tidal_Force_Magnitude')}")
                print(f"    geodesic_direction: {result.get('geodesic_direction')}")
                print(f"    signal: {result.get('signal')}")
                print(f"    ricci_collapsing: {result.get('ricci_collapsing')}")
                print(f"    crossing_horizon: {result.get('crossing_horizon')}")

            # Extrair campos do resultado
            ricci = result.get('Ricci_Scalar', 0)
            tidal = result.get('Tidal_Force_Magnitude', 0)
            eh_dist = result.get('Event_Horizon_Distance', 0)
            signal = result.get('signal', 0)
            geodesic_dir = result.get('geodesic_direction', 0)
            ricci_col = result.get('ricci_collapsing', False)
            cross_hor = result.get('crossing_horizon', False)

            signals.append({
                'bar_idx': i,
                'next_bar_idx': i + 1,
                'entry_price': bars[i + 1].open,
                'high': bars[i + 1].high,
                'low': bars[i + 1].low,
                'ricci_scalar': ricci,
                'tidal_force': tidal,
                'event_horizon_distance': eh_dist,
                'signal': signal,
                'geodesic_direction': geodesic_dir,
                'ricci_collapsing': ricci_col,
                'crossing_horizon': cross_hor
            })

    print(f"\n  Resultados: valid={valid_results}, empty={empty_results}, error={error_results}")

    train_signals = [s for s in signals if s['bar_idx'] < split_idx]
    test_signals = [s for s in signals if s['bar_idx'] >= split_idx]

    print(f"Sinais train: {len(train_signals)}, test: {len(test_signals)}")

    # Se não há sinais, abortar
    if len(signals) == 0:
        print("\nERRO: Nenhum sinal DSG foi gerado!")
        print("O indicador DSG não está produzindo resultados válidos.")
        return

    # Estatísticas dos sinais
    ricci_vals = [s['ricci_scalar'] for s in signals if s['ricci_scalar'] != 0]
    tidal_vals = [s['tidal_force'] for s in signals if s['tidal_force'] != 0]

    print(f"\nDistribuição Ricci: min={min(ricci_vals):.2f}, max={max(ricci_vals):.2f}, mean={np.mean(ricci_vals):.2f}")
    print(f"Distribuição Tidal: min={min(tidal_vals):.6f}, max={max(tidal_vals):.6f}, mean={np.mean(tidal_vals):.6f}")

    # Testar algumas configurações e mostrar por que falham
    print("\n" + "=" * 70)
    print("  TESTANDO CONFIGURAÇÕES")
    print("=" * 70)

    backtester = RobustBacktester(pip=get_pip_value("EURUSD"), spread=SPREAD_PIPS)

    # Parâmetros a testar
    test_configs = [
        {'ricci': -49000, 'tidal': 0.001, 'sl': 30, 'tp': 50},
        {'ricci': -48000, 'tidal': 0.002, 'sl': 25, 'tp': 40},
        {'ricci': -47000, 'tidal': 0.005, 'sl': 20, 'tp': 35},
        {'ricci': -50000, 'tidal': 0.001, 'sl': 30, 'tp': 45},
        {'ricci': -46000, 'tidal': 0.01, 'sl': 25, 'tp': 50},
    ]

    for cfg in test_configs:
        print(f"\n--- Config: ricci={cfg['ricci']}, tidal={cfg['tidal']}, sl={cfg['sl']}, tp={cfg['tp']} ---")

        # Gerar entradas para treino
        def generate_entries(sigs, bar_offset):
            entries = []
            for s in sigs:
                ricci_collapse = s['ricci_scalar'] < cfg['ricci'] or s.get('ricci_collapsing', False)
                high_tidal = s['tidal_force'] > cfg['tidal']
                crossing = s.get('crossing_horizon', False)

                conditions = sum([ricci_collapse, high_tidal, crossing])

                if conditions >= 2 and s.get('geodesic_direction', 0) != 0:
                    exec_idx = s['next_bar_idx'] - bar_offset
                    entries.append((exec_idx, s['entry_price'], s['geodesic_direction']))
            return entries

        train_entries = generate_entries(train_signals, 0)
        print(f"  Entradas TRAIN: {len(train_entries)}")

        if len(train_entries) < 3:
            print("  -> FALHA: Menos de 3 entradas")
            continue

        # Executar trades
        train_pnls = []
        last_exit_idx = -1
        for exec_idx, entry_price, direction in train_entries:
            if exec_idx < 0 or exec_idx >= len(train_bars) - 1:
                continue
            if exec_idx <= last_exit_idx:
                continue

            trade = backtester.execute_trade(
                bars=train_bars,
                entry_idx=exec_idx,
                entry_price=entry_price,
                direction=direction,
                sl_pips=cfg['sl'],
                tp_pips=cfg['tp'],
                max_bars=200
            )
            train_pnls.append(trade.pnl_pips)
            last_exit_idx = trade.exit_idx

        if not train_pnls:
            print("  -> FALHA: Nenhum trade executado")
            continue

        train_result = backtester.calculate_backtest_result(train_pnls)

        print(f"  TREINO: {train_result.trades} trades, WR={train_result.win_rate:.1%}, PF={train_result.profit_factor:.2f}, DD={train_result.max_drawdown:.1%}")

        # Verificar cada filtro
        failures = []
        if train_result.trades < MIN_TRADES_TRAIN:
            failures.append(f"trades={train_result.trades} < {MIN_TRADES_TRAIN}")
        if train_result.win_rate < MIN_WIN_RATE:
            failures.append(f"win_rate={train_result.win_rate:.1%} < {MIN_WIN_RATE:.1%}")
        if train_result.win_rate > MAX_WIN_RATE:
            failures.append(f"win_rate={train_result.win_rate:.1%} > {MAX_WIN_RATE:.1%}")
        if train_result.profit_factor < MIN_PROFIT_FACTOR:
            failures.append(f"pf={train_result.profit_factor:.2f} < {MIN_PROFIT_FACTOR:.2f}")
        if train_result.profit_factor > MAX_PROFIT_FACTOR:
            failures.append(f"pf={train_result.profit_factor:.2f} > {MAX_PROFIT_FACTOR:.2f}")
        if train_result.max_drawdown > MAX_DRAWDOWN:
            failures.append(f"dd={train_result.max_drawdown:.1%} > {MAX_DRAWDOWN:.1%}")

        if failures:
            print(f"  -> FALHA TREINO: {', '.join(failures)}")
            continue

        print("  TREINO PASSOU!")

        # Teste
        test_entries = generate_entries(test_signals, split_idx)
        print(f"  Entradas TESTE: {len(test_entries)}")

        if len(test_entries) < 3:
            print("  -> FALHA: Menos de 3 entradas no teste")
            continue

        test_pnls = []
        last_exit_idx = -1
        for exec_idx, entry_price, direction in test_entries:
            if exec_idx < 0 or exec_idx >= len(test_bars) - 1:
                continue
            if exec_idx <= last_exit_idx:
                continue

            trade = backtester.execute_trade(
                bars=test_bars,
                entry_idx=exec_idx,
                entry_price=entry_price,
                direction=direction,
                sl_pips=cfg['sl'],
                tp_pips=cfg['tp'],
                max_bars=200
            )
            test_pnls.append(trade.pnl_pips)
            last_exit_idx = trade.exit_idx

        if not test_pnls:
            print("  -> FALHA: Nenhum trade executado no teste")
            continue

        test_result = backtester.calculate_backtest_result(test_pnls)

        print(f"  TESTE: {test_result.trades} trades, WR={test_result.win_rate:.1%}, PF={test_result.profit_factor:.2f}, DD={test_result.max_drawdown:.1%}")

        # Verificar filtros do teste
        failures = []
        if test_result.trades < MIN_TRADES_TEST:
            failures.append(f"trades={test_result.trades} < {MIN_TRADES_TEST}")
        if test_result.win_rate < MIN_WIN_RATE_TEST:
            failures.append(f"win_rate={test_result.win_rate:.1%} < {MIN_WIN_RATE_TEST:.1%}")
        if test_result.win_rate > MAX_WIN_RATE_TEST:
            failures.append(f"win_rate={test_result.win_rate:.1%} > {MAX_WIN_RATE_TEST:.1%}")
        if test_result.profit_factor < MIN_PROFIT_FACTOR_TEST:
            failures.append(f"pf={test_result.profit_factor:.2f} < {MIN_PROFIT_FACTOR_TEST:.2f}")
        if test_result.profit_factor > MAX_PROFIT_FACTOR_TEST:
            failures.append(f"pf={test_result.profit_factor:.2f} > {MAX_PROFIT_FACTOR_TEST:.2f}")
        if test_result.max_drawdown > MAX_DRAWDOWN_TEST:
            failures.append(f"dd={test_result.max_drawdown:.1%} > {MAX_DRAWDOWN_TEST:.1%}")

        if failures:
            print(f"  -> FALHA TESTE: {', '.join(failures)}")
            continue

        print("  TESTE PASSOU!")

        # Verificar robustez
        pf_ratio = test_result.profit_factor / train_result.profit_factor if train_result.profit_factor > 0 else 0
        wr_ratio = test_result.win_rate / train_result.win_rate if train_result.win_rate > 0 else 0

        print(f"  Robustez: PF_ratio={pf_ratio:.2f}, WR_ratio={wr_ratio:.2f}")

        failures = []
        if pf_ratio < MIN_PF_RATIO:
            failures.append(f"pf_ratio={pf_ratio:.2f} < {MIN_PF_RATIO:.2f}")
        if wr_ratio < MIN_WR_RATIO:
            failures.append(f"wr_ratio={wr_ratio:.2f} < {MIN_WR_RATIO:.2f}")
        if test_result.profit_factor < 1.0:
            failures.append(f"pf_test={test_result.profit_factor:.2f} < 1.0")

        if failures:
            print(f"  -> FALHA ROBUSTEZ: {', '.join(failures)}")
        else:
            print("  >>> CONFIGURAÇÃO ROBUSTA ENCONTRADA! <<<")


if __name__ == "__main__":
    asyncio.run(main())
