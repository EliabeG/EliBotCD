#!/usr/bin/env python3
"""
Gerador rápido de sinais baseado em cotação em tempo real
"""
import asyncio
import aiohttp
from datetime import datetime, timezone
from typing import Dict, List
import json

# Configurações de trading
CONFIG = {
    "symbol": "EURUSD",
    "pip_size": 0.0001,
    "stop_loss_pips": 25,
    "take_profit_pips": 25,
    "lot_size": 0.01,
}

class QuickSignalGenerator:
    def __init__(self):
        self.price_history: List[float] = []
        self.max_history = 50

    async def fetch_quote(self, session: aiohttp.ClientSession) -> Dict:
        """Busca cotação atual"""
        url = "https://marginalttdemowebapi.fxopen.net/api/v2/public/ticker/EURUSD"
        async with session.get(url, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data[0] if data else {}
        return {}

    def add_price(self, price: float):
        """Adiciona preço ao histórico"""
        self.price_history.append(price)
        if len(self.price_history) > self.max_history:
            self.price_history = self.price_history[-self.max_history:]

    def calculate_rsi(self, period: int = 14) -> float:
        """Calcula RSI simples"""
        if len(self.price_history) < period + 1:
            return 50.0

        prices = self.price_history[-(period + 1):]
        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains) / period if gains else 0.0001
        avg_loss = sum(losses) / period if losses else 0.0001

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_sma(self, period: int = 20) -> float:
        """Calcula SMA"""
        if len(self.price_history) < period:
            return self.price_history[-1] if self.price_history else 0
        return sum(self.price_history[-period:]) / period

    def calculate_volatility(self, period: int = 10) -> float:
        """Calcula volatilidade em pips"""
        if len(self.price_history) < period:
            return 0
        prices = self.price_history[-period:]
        high = max(prices)
        low = min(prices)
        return (high - low) / CONFIG["pip_size"]

    def generate_signal(self, quote: Dict) -> Dict:
        """Gera sinal baseado nas condições atuais"""
        bid = quote.get("BestBid", 0)
        ask = quote.get("BestAsk", 0)
        mid = (bid + ask) / 2
        spread_pips = (ask - bid) / CONFIG["pip_size"]

        # Adicionar preço ao histórico
        self.add_price(mid)

        # Calcular indicadores
        rsi = self.calculate_rsi()
        sma = self.calculate_sma()
        volatility = self.calculate_volatility()

        # Análise de tendência
        price_vs_sma = "above" if mid > sma else "below"
        sma_diff_pips = abs(mid - sma) / CONFIG["pip_size"]

        signal = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": {
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_pips": round(spread_pips, 1)
            },
            "indicators": {
                "rsi": round(rsi, 2),
                "sma_20": round(sma, 5),
                "volatility_pips": round(volatility, 1),
                "price_vs_sma": price_vs_sma,
                "sma_diff_pips": round(sma_diff_pips, 1)
            },
            "signal": None,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "reason": None
        }

        # Condições de COMPRA
        if rsi < 30 and volatility > 2:
            signal["signal"] = "BUY"
            signal["entry"] = ask
            signal["stop_loss"] = round(ask - (CONFIG["stop_loss_pips"] * CONFIG["pip_size"]), 5)
            signal["take_profit"] = round(ask + (CONFIG["take_profit_pips"] * CONFIG["pip_size"]), 5)
            signal["reason"] = f"RSI oversold ({rsi:.1f}), volatilidade {volatility:.1f} pips"

        # Condições de VENDA
        elif rsi > 70 and volatility > 2:
            signal["signal"] = "SELL"
            signal["entry"] = bid
            signal["stop_loss"] = round(bid + (CONFIG["stop_loss_pips"] * CONFIG["pip_size"]), 5)
            signal["take_profit"] = round(bid - (CONFIG["take_profit_pips"] * CONFIG["pip_size"]), 5)
            signal["reason"] = f"RSI overbought ({rsi:.1f}), volatilidade {volatility:.1f} pips"

        # Sem sinal
        else:
            signal["reason"] = f"RSI neutro ({rsi:.1f}), aguardando condições extremas"

        return signal

async def monitor_signals(duration_seconds: int = 60):
    """Monitora sinais por um período"""
    generator = QuickSignalGenerator()

    print("=" * 70)
    print(f"MONITOR DE SINAIS AO VIVO - EURUSD")
    print(f"Iniciado: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Duração: {duration_seconds} segundos")
    print("=" * 70)

    signals_generated = []
    start_time = asyncio.get_event_loop().time()

    async with aiohttp.ClientSession() as session:
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= duration_seconds:
                break

            try:
                quote = await generator.fetch_quote(session)
                if not quote:
                    await asyncio.sleep(1)
                    continue

                signal = generator.generate_signal(quote)

                # Mostrar status
                remaining = int(duration_seconds - elapsed)
                print(f"\r[{remaining:3d}s] EURUSD: {signal['price']['bid']:.5f} | "
                      f"RSI: {signal['indicators']['rsi']:5.1f} | "
                      f"Vol: {signal['indicators']['volatility_pips']:4.1f}p | ", end="")

                if signal["signal"]:
                    print(f"\n{'='*70}")
                    print(f"*** SINAL: {signal['signal']} EURUSD ***")
                    print(f"    Entrada: {signal['entry']:.5f}")
                    print(f"    Stop Loss: {signal['stop_loss']:.5f}")
                    print(f"    Take Profit: {signal['take_profit']:.5f}")
                    print(f"    Razão: {signal['reason']}")
                    print(f"{'='*70}")
                    signals_generated.append(signal)
                else:
                    print(f"Aguardando...", end="")

                await asyncio.sleep(1)

            except Exception as e:
                print(f"\nErro: {e}")
                await asyncio.sleep(2)

    # Resumo final
    print(f"\n\n{'='*70}")
    print("RESUMO DA SESSÃO")
    print(f"{'='*70}")
    print(f"Duração: {duration_seconds} segundos")
    print(f"Ticks analisados: {len(generator.price_history)}")
    print(f"Sinais gerados: {len(signals_generated)}")

    if signals_generated:
        print(f"\nSinais:")
        for s in signals_generated:
            print(f"  - {s['timestamp']}: {s['signal']} @ {s['entry']}")
    else:
        print(f"\nNenhum sinal gerado durante o período.")
        print(f"RSI final: {generator.calculate_rsi():.1f}")
        print(f"Para gerar sinal BUY: RSI < 30")
        print(f"Para gerar sinal SELL: RSI > 70")

    print(f"{'='*70}")

if __name__ == "__main__":
    import sys
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    asyncio.run(monitor_signals(duration))
