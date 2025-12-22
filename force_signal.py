#!/usr/bin/env python3
"""
Força geração de sinal de teste baseado em análise técnica simplificada
"""
import asyncio
import aiohttp
from datetime import datetime, timezone

async def get_market_data():
    """Busca dados de mercado"""
    url = "https://marginalttdemowebapi.fxopen.net/api/v2/public/ticker/EURUSD"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data[0] if data else None
    return None

def analyze_and_generate_signal(quote: dict) -> dict:
    """
    Analisa mercado e gera sinal baseado em:
    - Posição do preço em relação ao range do dia
    - Spread
    - Momentum baseado em last trade prices
    """
    bid = quote.get("BestBid", 0)
    ask = quote.get("BestAsk", 0)
    mid = (bid + ask) / 2
    spread_pips = (ask - bid) / 0.0001

    # Preços do dia
    daily_buy = quote.get("DailyBestBuyPrice", mid)
    daily_sell = quote.get("DailyBestSellPrice", mid)
    daily_range = abs(daily_sell - daily_buy) / 0.0001  # em pips

    # Últimas transações
    last_buy = quote.get("LastBuyPrice", mid)
    last_sell = quote.get("LastSellPrice", mid)

    # Calcular posição do preço no range
    if daily_sell != daily_buy:
        position_pct = (mid - daily_buy) / (daily_sell - daily_buy) * 100
    else:
        position_pct = 50

    # Momentum (diferença entre últimas compras e vendas)
    momentum = (last_sell - last_buy) / 0.0001  # positivo = bullish

    print("=" * 70)
    print("ANÁLISE DE MERCADO - EURUSD")
    print(f"Horário: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 70)
    print(f"\nCotação Atual:")
    print(f"  Bid: {bid:.5f}")
    print(f"  Ask: {ask:.5f}")
    print(f"  Mid: {mid:.5f}")
    print(f"  Spread: {spread_pips:.1f} pips")

    print(f"\nRange do Dia:")
    print(f"  Melhor compra: {daily_buy:.5f}")
    print(f"  Melhor venda: {daily_sell:.5f}")
    print(f"  Range: {daily_range:.1f} pips")
    print(f"  Posição atual: {position_pct:.1f}%")

    print(f"\nÚltimas Transações:")
    print(f"  Última compra: {last_buy:.5f}")
    print(f"  Última venda: {last_sell:.5f}")
    print(f"  Momentum: {momentum:.1f} pips")

    # Gerar sinal
    signal = {
        "type": None,
        "entry": None,
        "stop_loss": None,
        "take_profit": None,
        "confidence": 0,
        "reason": ""
    }

    # Estratégia de Mean Reversion: comprar perto do fundo, vender perto do topo
    if position_pct < 30:
        # Perto do fundo do dia - sinal de COMPRA
        signal["type"] = "BUY"
        signal["entry"] = ask
        signal["stop_loss"] = round(ask - 0.0025, 5)  # 25 pips
        signal["take_profit"] = round(ask + 0.0025, 5)  # 25 pips
        signal["confidence"] = 0.75
        signal["reason"] = f"Preço perto do fundo diário ({position_pct:.1f}%), expectativa de reversão"

    elif position_pct > 70:
        # Perto do topo do dia - sinal de VENDA
        signal["type"] = "SELL"
        signal["entry"] = bid
        signal["stop_loss"] = round(bid + 0.0025, 5)
        signal["take_profit"] = round(bid - 0.0025, 5)
        signal["confidence"] = 0.75
        signal["reason"] = f"Preço perto do topo diário ({position_pct:.1f}%), expectativa de reversão"

    elif momentum > 3:
        # Momentum bullish forte
        signal["type"] = "BUY"
        signal["entry"] = ask
        signal["stop_loss"] = round(ask - 0.0020, 5)  # 20 pips
        signal["take_profit"] = round(ask + 0.0030, 5)  # 30 pips
        signal["confidence"] = 0.65
        signal["reason"] = f"Momentum bullish forte ({momentum:.1f} pips)"

    elif momentum < -3:
        # Momentum bearish forte
        signal["type"] = "SELL"
        signal["entry"] = bid
        signal["stop_loss"] = round(bid + 0.0020, 5)
        signal["take_profit"] = round(bid - 0.0030, 5)
        signal["confidence"] = 0.65
        signal["reason"] = f"Momentum bearish forte ({momentum:.1f} pips)"

    else:
        # Condição neutra - gerar sinal baseado em tendência
        if mid > (daily_buy + daily_sell) / 2:
            signal["type"] = "BUY"
            signal["entry"] = ask
            signal["stop_loss"] = round(ask - 0.0015, 5)  # 15 pips
            signal["take_profit"] = round(ask + 0.0020, 5)  # 20 pips
            signal["confidence"] = 0.55
            signal["reason"] = f"Tendência de alta intraday (acima da média: {(daily_buy + daily_sell) / 2:.5f})"
        else:
            signal["type"] = "SELL"
            signal["entry"] = bid
            signal["stop_loss"] = round(bid + 0.0015, 5)
            signal["take_profit"] = round(bid - 0.0020, 5)
            signal["confidence"] = 0.55
            signal["reason"] = f"Tendência de baixa intraday (abaixo da média: {(daily_buy + daily_sell) / 2:.5f})"

    return signal

async def main():
    """Gera e exibe sinal"""
    quote = await get_market_data()
    if not quote:
        print("ERRO: Não foi possível obter dados de mercado")
        return

    signal = analyze_and_generate_signal(quote)

    print("\n" + "=" * 70)
    print("*** SINAL GERADO ***")
    print("=" * 70)
    print(f"\n  Tipo: {signal['type']}")
    print(f"  Par: EURUSD")
    print(f"  Entrada: {signal['entry']:.5f}")
    print(f"  Stop Loss: {signal['stop_loss']:.5f}")
    print(f"  Take Profit: {signal['take_profit']:.5f}")

    sl_pips = abs(signal['entry'] - signal['stop_loss']) / 0.0001
    tp_pips = abs(signal['take_profit'] - signal['entry']) / 0.0001
    rr = tp_pips / sl_pips if sl_pips > 0 else 0

    print(f"\n  SL: {sl_pips:.0f} pips")
    print(f"  TP: {tp_pips:.0f} pips")
    print(f"  Risco/Retorno: 1:{rr:.1f}")
    print(f"  Confiança: {signal['confidence']:.0%}")
    print(f"\n  Razão: {signal['reason']}")

    print("\n" + "=" * 70)
    print("EXECUÇÃO:")
    print(f"  Para executar: {signal['type']} 0.01 lotes EURUSD @ {signal['entry']:.5f}")
    print(f"  SL: {signal['stop_loss']:.5f} | TP: {signal['take_profit']:.5f}")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
