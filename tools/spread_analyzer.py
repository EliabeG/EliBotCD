#!/usr/bin/env python3
"""
================================================================================
ANALISADOR DE SPREAD EM TEMPO REAL
Baixa dados históricos de BID e ASK e calcula o spread real
================================================================================
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
import numpy as np
from api.fxopen_historical_ws import FXOpenHistoricalClient


async def analyze_spread(symbol: str = "EURUSD",
                         periodicity: str = "M1",
                         hours: int = 24):
    """
    Analisa o spread real baixando dados de BID e ASK

    Args:
        symbol: Par de moedas
        periodicity: Periodicidade (M1, M5, H1, etc)
        hours: Horas de histórico para analisar
    """
    print("=" * 70)
    print("  ANALISADOR DE SPREAD EM TEMPO REAL")
    print("=" * 70)
    print(f"  Simbolo: {symbol}")
    print(f"  Periodicidade: {periodicity}")
    print(f"  Periodo: ultimas {hours} horas")
    print("=" * 70)

    client = FXOpenHistoricalClient()

    try:
        connected = await client.connect()
        if not connected:
            print("ERRO: Falha na conexao")
            return

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)

        end_ts = int(end_time.timestamp() * 1000)

        # Baixa dados BID
        print("\n  Baixando dados BID...")
        bid_bars = await client.get_bars(
            symbol=symbol,
            periodicity=periodicity,
            timestamp=end_ts,
            count=-500,
            price_type="bid"
        )
        print(f"    Barras BID: {len(bid_bars)}")

        # Baixa dados ASK
        print("  Baixando dados ASK...")
        ask_bars = await client.get_bars(
            symbol=symbol,
            periodicity=periodicity,
            timestamp=end_ts,
            count=-500,
            price_type="ask"
        )
        print(f"    Barras ASK: {len(ask_bars)}")

        if not bid_bars or not ask_bars:
            print("\n  ERRO: Nao foi possivel baixar dados")
            return

        # Criar dicionário de barras por timestamp
        bid_dict = {int(b.timestamp.timestamp()): b for b in bid_bars}
        ask_dict = {int(a.timestamp.timestamp()): a for a in ask_bars}

        # Calcular spread para cada timestamp comum
        spreads = []
        spread_data = []

        common_timestamps = set(bid_dict.keys()) & set(ask_dict.keys())

        for ts in sorted(common_timestamps):
            bid = bid_dict[ts]
            ask = ask_dict[ts]

            # Spread no close
            spread_close = (ask.close - bid.close) * 10000  # Em pips
            spreads.append(spread_close)

            spread_data.append({
                'timestamp': bid.timestamp,
                'bid_close': bid.close,
                'ask_close': ask.close,
                'spread_pips': spread_close
            })

        if not spreads:
            print("\n  ERRO: Nenhum dado comum encontrado")
            return

        spreads = np.array(spreads)

        # Estatísticas
        print("\n" + "=" * 70)
        print("  ESTATISTICAS DO SPREAD")
        print("=" * 70)
        print(f"\n  Barras analisadas: {len(spreads)}")
        print(f"\n  SPREAD (em pips):")
        print(f"    Minimo:    {spreads.min():.2f} pips")
        print(f"    Maximo:    {spreads.max():.2f} pips")
        print(f"    Media:     {spreads.mean():.2f} pips")
        print(f"    Mediana:   {np.median(spreads):.2f} pips")
        print(f"    Std Dev:   {spreads.std():.2f} pips")

        # Percentis
        print(f"\n  PERCENTIS:")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"    P{p:02d}:       {np.percentile(spreads, p):.2f} pips")

        # Contagem de spreads zero ou muito baixos
        zero_spreads = np.sum(spreads <= 0.1)
        low_spreads = np.sum(spreads <= 0.3)
        print(f"\n  DISTRIBUICAO:")
        print(f"    Spread <= 0.1 pips: {zero_spreads} ({zero_spreads/len(spreads)*100:.1f}%)")
        print(f"    Spread <= 0.3 pips: {low_spreads} ({low_spreads/len(spreads)*100:.1f}%)")

        # Últimos 10 spreads
        print(f"\n  ULTIMOS 10 SPREADS:")
        print(f"  {'Timestamp':<20} {'BID':<12} {'ASK':<12} {'Spread':<10}")
        print("  " + "-" * 54)
        for d in spread_data[-10:]:
            print(f"  {d['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{d['bid_close']:.5f}    {d['ask_close']:.5f}    "
                  f"{d['spread_pips']:.2f} pips")

        # Recomendação
        print("\n" + "=" * 70)
        print("  RECOMENDACAO PARA BACKTEST")
        print("=" * 70)
        recommended_spread = np.percentile(spreads, 75)
        print(f"\n  Spread recomendado (P75): {recommended_spread:.2f} pips")
        print(f"  Spread medio:             {spreads.mean():.2f} pips")
        print(f"\n  Para backtest conservador, use: {recommended_spread:.1f} pips")
        print(f"  Para backtest realista, use:    {spreads.mean():.1f} pips")

        return {
            'min': spreads.min(),
            'max': spreads.max(),
            'mean': spreads.mean(),
            'median': np.median(spreads),
            'std': spreads.std(),
            'p75': np.percentile(spreads, 75),
            'data': spread_data
        }

    finally:
        await client.disconnect()


def main():
    """Funcao principal"""
    import argparse

    parser = argparse.ArgumentParser(description="Analisador de Spread")
    parser.add_argument("--symbol", default="EURUSD", help="Par de moedas")
    parser.add_argument("--periodicity", default="M1", help="Periodicidade")
    parser.add_argument("--hours", type=int, default=24, help="Horas de historico")

    args = parser.parse_args()

    asyncio.run(analyze_spread(
        symbol=args.symbol,
        periodicity=args.periodicity,
        hours=args.hours
    ))


if __name__ == "__main__":
    main()
