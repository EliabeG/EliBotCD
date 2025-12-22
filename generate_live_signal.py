#!/usr/bin/env python3
"""
Script para gerar sinais ao vivo usando a API REST
"""
import asyncio
import sys
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiohttp
import numpy as np
from config.settings import CONFIG
from strategies.mean_reversion.bollinger_fade import BollingerFadeStrategy
from strategies.mean_reversion.zscore_vwap import ZScoreVWAPStrategy
from strategies.optimized.optimized_rsi import OptimizedRSIStrategy
from strategies.base_strategy import Signal

async def fetch_quote(session: aiohttp.ClientSession, symbol: str = "EURUSD") -> Optional[Dict]:
    """Busca cotação atual via REST API"""
    url = f"https://marginalttdemowebapi.fxopen.net/api/v2/public/ticker/{symbol}"
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data and len(data) > 0:
                    return data[0]
    except Exception as e:
        print(f"Erro ao buscar cotação: {e}")
    return None

async def fetch_candles(session: aiohttp.ClientSession, symbol: str = "EURUSD",
                        periodicity: str = "M1", count: int = 200) -> Optional[list]:
    """Busca candles históricos via REST API"""
    url = f"https://marginalttdemowebapi.fxopen.net/api/v2/public/quotehistory/{symbol}/{periodicity}/bars/ask"
    params = {"count": count}
    try:
        async with session.get(url, params=params, timeout=15) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("Bars", [])
    except Exception as e:
        print(f"Erro ao buscar candles: {e}")
    return None

def candles_to_ticks(candles: List[Dict]) -> List[Dict]:
    """Converte candles em formato de ticks para as estratégias"""
    ticks = []
    for c in candles:
        close = c.get("Close", 0)
        ticks.append({
            "bid": close - 0.00005,
            "ask": close + 0.00005,
            "mid": close,
            "high": c.get("High", close),
            "low": c.get("Low", close),
            "volume": c.get("Volume", 1000),
            "timestamp": c.get("Timestamp", 0)
        })
    return ticks

def build_market_context(quote: Dict, candles: list) -> Dict[str, Any]:
    """Constrói contexto de mercado para as estratégias"""
    bid = quote.get("BestBid", 0)
    ask = quote.get("BestAsk", 0)
    mid = (bid + ask) / 2
    spread = ask - bid

    # Criar lista de ticks a partir dos candles
    recent_ticks = candles_to_ticks(candles)

    return {
        "symbol": "EURUSD",
        "timestamp": datetime.now(timezone.utc),
        "tick": {"symbol": "EURUSD", "bid": bid, "ask": ask, "mid": mid},
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": spread,
        "volume": quote.get("DailyTradedTotalVolume", 0),
        "recent_ticks": recent_ticks,
        "dom": {"bids": [], "asks": []},
        "regime": "range"
    }

async def analyze_rsi_strategy(session: aiohttp.ClientSession, candles: list, quote: Dict):
    """Analisa RSI e mostra diagnóstico completo"""
    import talib

    print("\n" + "=" * 60)
    print("ANÁLISE RSI DETALHADA")
    print("=" * 60)

    # Extrair preços
    closes = np.array([c.get("Close", 0) for c in candles], dtype=float)
    highs = np.array([c.get("High", 0) for c in candles], dtype=float)
    lows = np.array([c.get("Low", 0) for c in candles], dtype=float)

    # Calcular RSI
    rsi = talib.RSI(closes, timeperiod=14)
    current_rsi = rsi[-1]

    # Calcular ATR
    atr = talib.ATR(highs, lows, closes, timeperiod=14)
    current_atr = atr[-1]
    atr_pips = current_atr / 0.0001

    # Momentum
    momentum_pips = abs(closes[-1] - closes[-4]) / 0.0001

    print(f"\nPreço Atual: {quote['BestBid']:.5f}")
    print(f"RSI(14): {current_rsi:.2f}")
    print(f"ATR(14): {atr_pips:.1f} pips")
    print(f"Momentum(3): {momentum_pips:.1f} pips")

    # Condições de sinal
    print("\nCondições para COMPRA (RSI Oversold):")
    print(f"  RSI < 30? {current_rsi:.2f} < 30 = {'SIM ✓' if current_rsi < 30 else 'NÃO ✗'}")
    print(f"  ATR > 5 pips? {atr_pips:.1f} > 5 = {'SIM ✓' if atr_pips > 5 else 'NÃO ✗'}")
    print(f"  ATR < 50 pips? {atr_pips:.1f} < 50 = {'SIM ✓' if atr_pips < 50 else 'NÃO ✗'}")
    print(f"  Momentum > 2 pips? {momentum_pips:.1f} > 2 = {'SIM ✓' if momentum_pips > 2 else 'NÃO ✗'}")

    print("\nCondições para VENDA (RSI Overbought):")
    print(f"  RSI > 70? {current_rsi:.2f} > 70 = {'SIM ✓' if current_rsi > 70 else 'NÃO ✗'}")

    # Últimos RSIs
    print(f"\nÚltimos 5 RSIs: {', '.join([f'{r:.1f}' for r in rsi[-5:]])}")

    return current_rsi, atr_pips

async def analyze_bollinger_strategy(candles: list, quote: Dict):
    """Analisa Bollinger Bands"""
    import talib

    print("\n" + "=" * 60)
    print("ANÁLISE BOLLINGER BANDS")
    print("=" * 60)

    closes = np.array([c.get("Close", 0) for c in candles], dtype=float)

    # Calcular Bollinger Bands
    upper, middle, lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)

    current_price = quote['BestBid']
    current_upper = upper[-1]
    current_lower = lower[-1]
    current_middle = middle[-1]

    # Posição do preço nas bandas (percentil)
    band_width = current_upper - current_lower
    price_position = (current_price - current_lower) / band_width * 100 if band_width > 0 else 50

    print(f"\nPreço Atual: {current_price:.5f}")
    print(f"Banda Superior: {current_upper:.5f}")
    print(f"Banda Média: {current_middle:.5f}")
    print(f"Banda Inferior: {current_lower:.5f}")
    print(f"Posição do preço: {price_position:.1f}%")

    print("\nCondições para COMPRA (Fade na banda inferior):")
    print(f"  Preço < Banda Inferior? {current_price:.5f} < {current_lower:.5f} = {'SIM ✓' if current_price < current_lower else 'NÃO ✗'}")

    print("\nCondições para VENDA (Fade na banda superior):")
    print(f"  Preço > Banda Superior? {current_price:.5f} > {current_upper:.5f} = {'SIM ✓' if current_price > current_upper else 'NÃO ✗'}")

    return price_position

async def generate_signals():
    """Gera sinais usando as estratégias ativas"""
    print("=" * 60)
    print(f"GERADOR DE SINAIS AO VIVO")
    print(f"Data/Hora: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        # Buscar dados
        print("\n[1] Buscando cotação atual...")
        quote = await fetch_quote(session)
        if not quote:
            print("ERRO: Não foi possível obter cotação")
            return

        print(f"    EURUSD: Bid={quote['BestBid']:.5f} Ask={quote['BestAsk']:.5f}")
        spread_pips = (quote['BestAsk'] - quote['BestBid']) / 0.0001
        print(f"    Spread: {spread_pips:.1f} pips")

        print("\n[2] Buscando candles M1...")
        candles = await fetch_candles(session, count=200)
        if not candles:
            print("ERRO: Não foi possível obter candles")
            return
        print(f"    {len(candles)} candles obtidos")

        # Analisar RSI
        rsi, atr_pips = await analyze_rsi_strategy(session, candles, quote)

        # Analisar Bollinger
        bb_position = await analyze_bollinger_strategy(candles, quote)

        # Construir contexto
        market_context = build_market_context(quote, candles)

        # Inicializar e testar estratégias
        print("\n" + "=" * 60)
        print("EXECUÇÃO DAS ESTRATÉGIAS")
        print("=" * 60)

        strategies = [
            ("OptimizedRSI", OptimizedRSIStrategy()),
            ("BollingerFade", BollingerFadeStrategy()),
            ("ZScoreVWAP", ZScoreVWAPStrategy()),
        ]

        signals_found = []
        for name, strategy in strategies:
            try:
                print(f"\n>>> Executando {name}...")

                # Calcular indicadores
                await strategy.calculate_indicators(market_context)

                # Mostrar indicadores
                indicators = getattr(strategy, 'current_indicators', {})
                if indicators:
                    print(f"    Indicadores calculados: {len(indicators)} valores")
                    for key, value in list(indicators.items())[:5]:
                        if isinstance(value, float):
                            print(f"      {key}: {value:.5f}")
                        elif isinstance(value, bool):
                            print(f"      {key}: {'Sim' if value else 'Não'}")

                # Gerar sinal
                signal = await strategy.generate_signal(market_context)

                if signal and signal.is_valid():
                    signals_found.append((name, signal))
                    print(f"\n    *** SINAL GERADO ***")
                    print(f"    Direção: {signal.side.upper()}")
                    print(f"    Entrada: {signal.entry_price or 'MARKET'}")
                    if signal.stop_loss:
                        print(f"    Stop Loss: {signal.stop_loss:.5f}")
                    if signal.take_profit:
                        print(f"    Take Profit: {signal.take_profit:.5f}")
                    if hasattr(signal, 'confidence'):
                        print(f"    Confiança: {signal.confidence:.1%}")
                else:
                    print(f"    Resultado: Sem sinal")

            except Exception as e:
                import traceback
                print(f"    ERRO: {e}")
                traceback.print_exc()

        # Resumo final
        print("\n" + "=" * 60)
        print("RESUMO")
        print("=" * 60)

        if signals_found:
            print(f"\n*** {len(signals_found)} SINAL(IS) GERADO(S) ***\n")
            for name, signal in signals_found:
                print(f"  {name}: {signal.side.upper()} EURUSD")
                if signal.stop_loss and signal.take_profit:
                    sl_pips = abs(signal.entry_price - signal.stop_loss) / 0.0001 if signal.entry_price else 0
                    tp_pips = abs(signal.take_profit - signal.entry_price) / 0.0001 if signal.entry_price else 0
                    print(f"    SL: {sl_pips:.0f} pips | TP: {tp_pips:.0f} pips")
        else:
            print("\n*** NENHUM SINAL NO MOMENTO ***")
            print("\nMotivos possíveis:")
            if rsi > 30 and rsi < 70:
                print(f"  - RSI em zona neutra ({rsi:.1f})")
            if atr_pips < 5:
                print(f"  - Volatilidade muito baixa ({atr_pips:.1f} pips)")
            if bb_position > 10 and bb_position < 90:
                print(f"  - Preço dentro das Bandas de Bollinger ({bb_position:.0f}%)")

        print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(generate_signals())
