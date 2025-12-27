#!/usr/bin/env python3
"""
================================================================================
TESTE DO BOT - VALIDACAO DE SINAIS
================================================================================

Este script:
1. Carrega dados historicos reais
2. Testa cada estrategia individualmente
3. Mostra quais sinais seriam gerados
4. Valida que o bot esta funcionando corretamente

Uso:
    python test_bot_signals.py
"""

import sys
import asyncio
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from api.fxopen_historical_ws import download_historical_data, Bar
from strategies.strategy_factory import create_all_strategies
from strategies.base import SignalType


async def test_signals():
    print("=" * 70)
    print("  TESTE DO BOT - VALIDACAO DE SINAIS")
    print("  Carregando dados historicos reais e testando estrategias")
    print("=" * 70)

    # 1. Carregar dados historicos
    print("\n[1/3] Carregando dados historicos reais...")
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=7)  # Ultimos 7 dias

    bars = await download_historical_data(
        symbol='EURUSD',
        periodicity='H1',
        start_time=start_date,
        end_time=end_date,
        price_type='bid'
    )

    if not bars:
        print("ERRO: Nao foi possivel carregar dados!")
        return

    print(f"  Barras carregadas: {len(bars)}")
    print(f"  Periodo: {bars[0].timestamp} a {bars[-1].timestamp}")

    # 2. Criar todas as estrategias
    print("\n[2/3] Inicializando estrategias...")
    strategies = create_all_strategies()

    total_strategies = sum(len(strats) for strats in strategies.values())
    print(f"  Total de estrategias: {total_strategies}")

    # 3. Testar cada estrategia
    print("\n[3/3] Testando sinais de cada estrategia...")
    print("-" * 70)

    results = defaultdict(lambda: {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'errors': 0})

    for level, strats in strategies.items():
        print(f"\n{level} VOLATILIDADE ({len(strats)} estrategias):")
        print("-" * 50)

        for strategy in strats:
            buy_count = 0
            sell_count = 0
            hold_count = 0
            error_count = 0

            # Feed das ultimas 200 barras
            test_bars = bars[-200:] if len(bars) >= 200 else bars

            for bar in test_bars:
                try:
                    signal = strategy.analyze(
                        price=bar.close,
                        timestamp=bar.timestamp,
                        volatility=0.5,  # Volatilidade media
                        hurst=0.5,
                        entropy=0.5,
                        volume=bar.volume if hasattr(bar, 'volume') else None
                    )

                    if signal:
                        if signal.type == SignalType.BUY:
                            buy_count += 1
                        elif signal.type == SignalType.SELL:
                            sell_count += 1
                        else:
                            hold_count += 1
                    else:
                        hold_count += 1

                except Exception as e:
                    error_count += 1

            total = buy_count + sell_count + hold_count

            # Status
            if error_count > 0:
                status = f"[ERRO: {error_count}]"
            elif buy_count == 0 and sell_count == 0:
                status = "[SEM SINAIS]"
            else:
                status = "[OK]"

            print(f"  {strategy.name:35} BUY:{buy_count:3} SELL:{sell_count:3} HOLD:{hold_count:3} {status}")

            results[strategy.name] = {
                'BUY': buy_count,
                'SELL': sell_count,
                'HOLD': hold_count,
                'errors': error_count,
                'level': level
            }

    # 4. Resumo
    print("\n" + "=" * 70)
    print("  RESUMO")
    print("=" * 70)

    # Estrategias funcionando
    working = [name for name, data in results.items()
               if data['BUY'] > 0 or data['SELL'] > 0]
    not_working = [name for name, data in results.items()
                   if data['BUY'] == 0 and data['SELL'] == 0 and data['errors'] == 0]
    with_errors = [name for name, data in results.items()
                   if data['errors'] > 0]

    print(f"\n  Estrategias gerando sinais: {len(working)}/{len(results)}")
    print(f"  Estrategias sem sinais: {len(not_working)}")
    print(f"  Estrategias com erros: {len(with_errors)}")

    if working:
        print("\n  ESTRATEGIAS FUNCIONANDO:")
        for name in working:
            data = results[name]
            print(f"    - {name}: {data['BUY']} BUY, {data['SELL']} SELL")

    if with_errors:
        print("\n  ESTRATEGIAS COM ERROS:")
        for name in with_errors:
            print(f"    - {name}: {results[name]['errors']} erros")

    print("\n" + "=" * 70)
    print("  TESTE CONCLUIDO")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_signals())
