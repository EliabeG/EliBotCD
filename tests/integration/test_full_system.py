"""
================================================================================
TESTE INTEGRADO DO SISTEMA COMPLETO
EliBotCD - Trading Bot com Estrategias baseadas em Volatilidade
================================================================================

Este teste verifica o fluxo completo:
1. Captura de ticks (simulado ou API real)
2. Classificacao de volatilidade
3. Roteamento para estrategias apropriadas
4. Geracao de sinais
5. Agregacao e decisao

Uso:
    python -m tests.integration.test_full_system [--real-api] [--duration SECONDS]
"""

import sys
import os
import asyncio
import numpy as np
from datetime import datetime, timezone
from typing import List, Optional
import argparse

# Adiciona o diretorio raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.orchestrator import StrategyOrchestrator, VolatilityLevel
from strategies.base import Signal, SignalType


# ==============================================================================
# ESTRATEGIAS DE TESTE (Simplificadas)
# ==============================================================================

class SimpleTestStrategy:
    """Estrategia simples para teste"""

    def __init__(self, name: str, bias: str = "neutral", threshold: float = 0.3):
        self.name = name
        self.bias = bias  # "bullish", "bearish", "neutral"
        self.threshold = threshold
        self.is_active = True
        self.last_signal = None
        self.price_history = []
        self.signal_count = 0

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """Analisa e gera sinal"""
        self.price_history.append(price)

        if len(self.price_history) < 10:
            return None

        # Calcula momentum simples
        recent = self.price_history[-5:]
        older = self.price_history[-10:-5]
        momentum = (sum(recent) / len(recent)) - (sum(older) / len(older))

        volatility = indicators.get('volatility', 0)
        entropy = indicators.get('entropy', 0.5)

        # Gera sinal baseado no momentum e bias
        signal_type = None
        confidence = 0.0

        if self.bias == "bullish":
            if momentum > 0.00001:
                signal_type = SignalType.BUY
                confidence = min(0.9, 0.4 + abs(momentum) * 10000)
        elif self.bias == "bearish":
            if momentum < -0.00001:
                signal_type = SignalType.SELL
                confidence = min(0.9, 0.4 + abs(momentum) * 10000)
        else:  # neutral
            if momentum > 0.00002:
                signal_type = SignalType.BUY
                confidence = min(0.8, 0.3 + abs(momentum) * 10000)
            elif momentum < -0.00002:
                signal_type = SignalType.SELL
                confidence = min(0.8, 0.3 + abs(momentum) * 10000)

        if signal_type and confidence >= self.threshold:
            pip_value = 0.0001
            if signal_type == SignalType.BUY:
                stop_loss = price - 15 * pip_value
                take_profit = price + 30 * pip_value
            else:
                stop_loss = price + 15 * pip_value
                take_profit = price - 30 * pip_value

            self.signal_count += 1
            signal = Signal(
                type=signal_type,
                price=price,
                timestamp=timestamp,
                strategy_name=self.name,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"Momentum: {momentum*10000:.2f} pips"
            )
            self.last_signal = signal
            return signal

        return None

    def reset(self):
        self.price_history.clear()
        self.signal_count = 0
        self.last_signal = None


# ==============================================================================
# GERADOR DE DADOS SIMULADOS
# ==============================================================================

def generate_market_data(n_ticks: int = 1000,
                        base_price: float = 1.0850,
                        seed: int = 42,
                        regime: str = "mixed") -> List[tuple]:
    """
    Gera dados de mercado simulados

    Args:
        n_ticks: Numero de ticks
        base_price: Preco base
        seed: Semente random
        regime: "low_vol", "high_vol", "trending", "mixed"

    Returns:
        Lista de (price, timestamp, volume)
    """
    np.random.seed(seed)
    data = []
    price = base_price
    timestamp = datetime.now(timezone.utc)

    for i in range(n_ticks):
        # Ajusta volatilidade baseada no regime
        if regime == "low_vol":
            vol = 0.00002
        elif regime == "high_vol":
            vol = 0.0002
        elif regime == "trending":
            vol = 0.00005
            trend = 0.00001 if i < n_ticks // 2 else -0.00001
            price += trend
        else:  # mixed
            # Alterna regimes
            phase = (i // 200) % 3
            if phase == 0:
                vol = 0.00002  # Baixa vol
            elif phase == 1:
                vol = 0.00005  # Media vol
            else:
                vol = 0.0002   # Alta vol

        # Movimento de preco
        change = np.random.randn() * vol
        price += change
        price = max(0.5, min(2.0, price))  # Limites

        # Volume aleatorio
        volume = 1000 + np.random.exponential(500)

        # Timestamp
        from datetime import timedelta
        timestamp = timestamp + timedelta(milliseconds=100)

        data.append((price, timestamp, volume))

    return data


# ==============================================================================
# TESTE COM DADOS SIMULADOS
# ==============================================================================

def test_with_simulated_data(n_ticks: int = 500, regime: str = "mixed"):
    """Testa o sistema com dados simulados"""
    print("=" * 70)
    print("TESTE INTEGRADO - DADOS SIMULADOS")
    print("=" * 70)

    # Cria orquestrador
    orch = StrategyOrchestrator(
        symbol="EURUSD",
        min_confidence=0.3,
        signal_cooldown_ticks=20
    )

    # Cria estrategias de teste
    # Alta volatilidade - estrategias mais agressivas
    alta_strategies = [
        SimpleTestStrategy("ALTA-Momentum", bias="neutral", threshold=0.3),
        SimpleTestStrategy("ALTA-Breakout", bias="bullish", threshold=0.4),
    ]

    # Media volatilidade - estrategias balanceadas
    media_strategies = [
        SimpleTestStrategy("MEDIA-Swing", bias="neutral", threshold=0.35),
        SimpleTestStrategy("MEDIA-Trend", bias="neutral", threshold=0.4),
    ]

    # Baixa volatilidade - estrategias de reversao
    baixa_strategies = [
        SimpleTestStrategy("BAIXA-MeanRev", bias="neutral", threshold=0.3),
        SimpleTestStrategy("BAIXA-Range", bias="neutral", threshold=0.35),
    ]

    # Registra estrategias
    for s in alta_strategies:
        orch.register_strategy(s, "ALTA")
    for s in media_strategies:
        orch.register_strategy(s, "MEDIA")
    for s in baixa_strategies:
        orch.register_strategy(s, "BAIXA")

    print(f"\nEstrategias registradas:")
    for level, names in orch.get_registered_strategies().items():
        print(f"  {level}: {names}")

    # Gera dados
    print(f"\nGerando {n_ticks} ticks ({regime})...")
    data = generate_market_data(n_ticks=n_ticks, regime=regime)

    # Processa ticks
    print("\nProcessando ticks...")
    print("-" * 70)

    signals_received = []
    level_counts = {"ALTA": 0, "MEDIA": 0, "BAIXA": 0, "INDEFINIDO": 0}

    for i, (price, timestamp, volume) in enumerate(data):
        signal = orch.process_tick(price, timestamp, volume)

        state = orch.get_state()
        level_counts[state.volatility_level] = level_counts.get(state.volatility_level, 0) + 1

        if signal:
            signals_received.append(signal)
            direction = "COMPRA" if signal.type == SignalType.BUY else "VENDA"
            print(f"[SINAL] Tick {i:4d} | {direction:6s} @ {signal.price:.5f} | "
                  f"Conf: {signal.confidence:.0%} | {signal.strategy_name}")

        if i % 100 == 0 and i > 0:
            print(f"[INFO] Tick {i:4d} | Vol: {state.current_volatility:.3f} pips | "
                  f"Level: {state.volatility_level:10s} | "
                  f"Sinais: {len(signals_received)}")

    # Resultados
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    stats = orch.get_stats()
    print(f"\nTicks processados: {stats['ticks_processed']}")
    print(f"Sinais gerados: {stats['signals_generated']}")

    print(f"\nDistribuicao de volatilidade:")
    for level, count in level_counts.items():
        pct = count / n_ticks * 100
        print(f"  {level:10s}: {count:4d} ({pct:.1f}%)")

    print(f"\nSinais por nivel:")
    for level, count in stats['signals_by_level'].items():
        print(f"  {level:10s}: {count}")

    print(f"\nSinais por tipo:")
    for stype, count in stats['signals_by_type'].items():
        print(f"  {stype:10s}: {count}")

    if signals_received:
        buy_signals = [s for s in signals_received if s.type == SignalType.BUY]
        sell_signals = [s for s in signals_received if s.type == SignalType.SELL]
        avg_conf = sum(s.confidence for s in signals_received) / len(signals_received)

        print(f"\nResumo de sinais:")
        print(f"  Total: {len(signals_received)}")
        print(f"  Compras: {len(buy_signals)}")
        print(f"  Vendas: {len(sell_signals)}")
        print(f"  Confianca media: {avg_conf:.1%}")

    return orch, signals_received


# ==============================================================================
# TESTE COM API REAL
# ==============================================================================

async def test_with_real_api(duration_seconds: int = 60):
    """Testa o sistema com API real do FXOpen"""
    print("=" * 70)
    print("TESTE INTEGRADO - API REAL (FXOpen)")
    print("=" * 70)

    try:
        from api.fxopen_client import FXOpenClient, Tick
        from config import settings
    except ImportError as e:
        print(f"Erro ao importar modulos: {e}")
        print("Execute do diretorio raiz: python -m tests.integration.test_full_system --real-api")
        return None, []

    # Cria orquestrador
    orch = StrategyOrchestrator(
        symbol=settings.SYMBOL,
        min_confidence=0.3,
        signal_cooldown_ticks=50
    )

    # Cria estrategias de teste
    alta_strategies = [
        SimpleTestStrategy("ALTA-Momentum", bias="neutral", threshold=0.35),
    ]
    media_strategies = [
        SimpleTestStrategy("MEDIA-Swing", bias="neutral", threshold=0.35),
    ]
    baixa_strategies = [
        SimpleTestStrategy("BAIXA-MeanRev", bias="neutral", threshold=0.3),
    ]

    for s in alta_strategies:
        orch.register_strategy(s, "ALTA")
    for s in media_strategies:
        orch.register_strategy(s, "MEDIA")
    for s in baixa_strategies:
        orch.register_strategy(s, "BAIXA")

    print(f"\nConectando a {settings.WS_FEED_URL}...")
    print(f"Simbolo: {settings.SYMBOL}")
    print(f"Duracao: {duration_seconds} segundos")

    # Cliente FXOpen
    client = FXOpenClient()
    signals_received = []

    async def on_tick(tick: Tick):
        """Callback para cada tick"""
        signal = orch.process_tick(tick.mid, tick.timestamp, tick.bid_volume + tick.ask_volume)

        state = orch.get_state()

        # Exibe status periodicamente
        if state.tick_count % 10 == 0:
            print(f"[TICK] {tick.symbol} | {tick.mid:.5f} | Vol: {state.current_volatility:.3f} | "
                  f"{state.volatility_level}")

        if signal:
            signals_received.append(signal)
            direction = "COMPRA" if signal.type == SignalType.BUY else "VENDA"
            print(f"\n{'='*50}")
            print(f"[SINAL] {direction} @ {signal.price:.5f}")
            print(f"  Confianca: {signal.confidence:.1%}")
            print(f"  Stop: {signal.stop_loss:.5f}")
            print(f"  Take: {signal.take_profit:.5f}")
            print(f"  Estrategia: {signal.strategy_name}")
            print(f"  Razao: {signal.reason}")
            print(f"{'='*50}\n")

    client.on_tick = on_tick

    try:
        if await client.connect():
            await client.subscribe(settings.SYMBOL)

            print(f"\nRecebendo ticks por {duration_seconds} segundos...")
            print("-" * 70)

            await asyncio.sleep(duration_seconds)

            print("\n" + "=" * 70)
            print("RESULTADOS")
            print("=" * 70)

            stats = orch.get_stats()
            print(f"\nTicks processados: {stats['ticks_processed']}")
            print(f"Sinais gerados: {stats['signals_generated']}")

            state = orch.get_state()
            print(f"\nEstado final:")
            print(f"  Preco: {state.current_price:.5f}")
            print(f"  Volatilidade: {state.current_volatility:.3f} pips")
            print(f"  Nivel: {state.volatility_level}")
            print(f"  Hurst: {state.hurst:.3f}")
            print(f"  Entropy: {state.entropy:.3f}")

        else:
            print("Falha ao conectar")

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario")
    finally:
        await client.disconnect()

    return orch, signals_received


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Teste integrado do EliBotCD")
    parser.add_argument("--real-api", action="store_true", help="Usar API real do FXOpen")
    parser.add_argument("--duration", type=int, default=60, help="Duracao do teste em segundos (para API real)")
    parser.add_argument("--ticks", type=int, default=500, help="Numero de ticks (para simulacao)")
    parser.add_argument("--regime", type=str, default="mixed",
                       choices=["low_vol", "high_vol", "trending", "mixed"],
                       help="Regime de mercado para simulacao")

    args = parser.parse_args()

    if args.real_api:
        asyncio.run(test_with_real_api(args.duration))
    else:
        test_with_simulated_data(args.ticks, args.regime)


if __name__ == "__main__":
    main()
