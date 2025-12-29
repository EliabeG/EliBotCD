"""
================================================================================
STRATEGY ORCHESTRATOR
Orquestrador de Estrategias baseado em Volatilidade
================================================================================

Este modulo gerencia:
1. Recepcao de ticks da API
2. Classificacao de volatilidade em tempo real
3. Roteamento para estrategias apropriadas
4. Agregacao de sinais
5. Decisao de execucao
"""

import numpy as np
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Callable
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .base import BaseStrategy, Signal, SignalType
from indicators.realtime_volatility import RealtimeVolatility


class VolatilityLevel(Enum):
    """Nivel de volatilidade"""
    ALTA = "ALTA"
    MEDIA = "MEDIA"
    BAIXA = "BAIXA"
    INDEFINIDO = "INDEFINIDO"


@dataclass
class StrategyResult:
    """Resultado de uma estrategia"""
    strategy_name: str
    signal: Optional[Signal]
    confidence: float
    volatility_level: str
    processing_time_ms: float


@dataclass
class OrchestratorState:
    """Estado do orquestrador"""
    current_price: float = 0.0
    current_volatility: float = 0.0
    volatility_level: str = "INDEFINIDO"
    hurst: float = 0.5
    entropy: float = 0.5
    active_strategies: int = 0
    signals_generated: int = 0
    last_signal: Optional[Signal] = None
    tick_count: int = 0
    last_update: Optional[datetime] = None


class StrategyOrchestrator:
    """
    Orquestrador de Estrategias

    Fluxo:
    1. Recebe tick da API
    2. Atualiza indicador de volatilidade
    3. Classifica volatilidade (ALTA, MEDIA, BAIXA)
    4. Seleciona estrategias apropriadas
    5. Executa analise em cada estrategia
    6. Agrega sinais
    7. Decide execucao
    """

    def __init__(self,
                 symbol: str = "EURUSD",
                 min_confidence: float = 0.5,
                 signal_cooldown_ticks: int = 100,
                 max_signals_history: int = 1000):
        """
        Inicializa o orquestrador

        Args:
            symbol: Simbolo do par
            min_confidence: Confianca minima para sinal
            signal_cooldown_ticks: Ticks de cooldown entre sinais
            max_signals_history: Maximo de sinais no historico
        """
        self.symbol = symbol
        self.min_confidence = min_confidence
        self.signal_cooldown_ticks = signal_cooldown_ticks

        # Indicador de volatilidade
        self.volatility = RealtimeVolatility(
            candle_seconds=5,
            max_candles=500,
            parkinson_window=20,
            hurst_window=50,
            entropy_window=20
        )

        # Estrategias por nivel de volatilidade
        self.strategies: Dict[str, List[BaseStrategy]] = {
            'ALTA': [],
            'MEDIA': [],
            'BAIXA': []
        }

        # Estado
        self.state = OrchestratorState()
        self.signals_history: deque = deque(maxlen=max_signals_history)
        self.ticks_since_last_signal = 0

        # Callbacks
        self.on_signal: Optional[Callable[[Signal], None]] = None
        self.on_state_update: Optional[Callable[[OrchestratorState], None]] = None

        # Estatisticas
        self.stats = {
            'ticks_processed': 0,
            'signals_generated': 0,
            'signals_by_level': {'ALTA': 0, 'MEDIA': 0, 'BAIXA': 0},
            'signals_by_type': {'BUY': 0, 'SELL': 0}
        }

    def register_strategy(self, strategy: BaseStrategy, volatility_level: str):
        """
        Registra uma estrategia para um nivel de volatilidade

        Args:
            strategy: Estrategia a registrar
            volatility_level: 'ALTA', 'MEDIA' ou 'BAIXA'
        """
        level = volatility_level.upper()
        if level not in self.strategies:
            print(f"Nivel de volatilidade invalido: {level}")
            return

        self.strategies[level].append(strategy)
        print(f"Estrategia '{strategy.name}' registrada para volatilidade {level}")

    def register_strategies_batch(self, strategies_config: Dict[str, List[BaseStrategy]]):
        """
        Registra multiplas estrategias de uma vez

        Args:
            strategies_config: Dict com nivel -> lista de estrategias
        """
        for level, strategies in strategies_config.items():
            for strategy in strategies:
                self.register_strategy(strategy, level)

    def process_tick(self, price: float, timestamp: datetime, volume: float = None) -> Optional[Signal]:
        """
        Processa um tick e retorna sinal se houver

        Args:
            price: Preco do tick
            timestamp: Timestamp do tick
            volume: Volume (opcional)

        Returns:
            Signal se houver, None caso contrario
        """
        import time
        start_time = time.time()

        # Atualiza volatilidade
        self.volatility.add_tick(price, timestamp)
        vol_state = self.volatility.get_state()

        # Atualiza estado
        self.state.current_price = price
        self.state.current_volatility = vol_state['volatility']
        self.state.volatility_level = vol_state['classification']
        self.state.hurst = vol_state['hurst']
        self.state.entropy = vol_state['entropy']
        self.state.tick_count += 1
        self.state.last_update = timestamp

        self.stats['ticks_processed'] += 1
        self.ticks_since_last_signal += 1

        # Verifica cooldown
        if self.ticks_since_last_signal < self.signal_cooldown_ticks:
            if self.on_state_update:
                self.on_state_update(self.state)
            return None

        # Seleciona estrategias para o nivel atual
        level = vol_state['classification']
        if level not in self.strategies or level == 'INDEFINIDO':
            if self.on_state_update:
                self.on_state_update(self.state)
            return None

        active_strategies = self.strategies[level]
        self.state.active_strategies = len(active_strategies)

        if not active_strategies:
            if self.on_state_update:
                self.on_state_update(self.state)
            return None

        # Executa cada estrategia
        signals = []
        for strategy in active_strategies:
            try:
                signal = strategy.analyze(
                    price=price,
                    timestamp=timestamp,
                    volatility=vol_state['volatility'],
                    hurst=vol_state['hurst'],
                    entropy=vol_state['entropy'],
                    volume=volume
                )

                if signal and signal.confidence >= self.min_confidence:
                    signals.append(signal)

            except Exception as e:
                print(f"Erro na estrategia {strategy.name}: {e}")

        # Agrega sinais
        final_signal = self._aggregate_signals(signals, level)

        if final_signal:
            self.signals_history.append(final_signal)
            self.state.signals_generated += 1
            self.state.last_signal = final_signal
            self.ticks_since_last_signal = 0

            self.stats['signals_generated'] += 1
            self.stats['signals_by_level'][level] += 1
            if final_signal.type == SignalType.BUY:
                self.stats['signals_by_type']['BUY'] += 1
            elif final_signal.type == SignalType.SELL:
                self.stats['signals_by_type']['SELL'] += 1

            if self.on_signal:
                self.on_signal(final_signal)

        if self.on_state_update:
            self.on_state_update(self.state)

        return final_signal

    def _aggregate_signals(self, signals: List[Signal], volatility_level: str) -> Optional[Signal]:
        """
        Agrega multiplos sinais em um sinal final

        Estrategia de agregacao:
        - Voto majoritario por tipo
        - Confianca media ponderada
        - Stop/Take mais conservador
        """
        if not signals:
            return None

        # Conta votos
        buy_signals = [s for s in signals if s.type == SignalType.BUY]
        sell_signals = [s for s in signals if s.type == SignalType.SELL]

        # Voto majoritario
        if len(buy_signals) > len(sell_signals):
            winning_signals = buy_signals
            signal_type = SignalType.BUY
        elif len(sell_signals) > len(buy_signals):
            winning_signals = sell_signals
            signal_type = SignalType.SELL
        else:
            # Empate - usa confianca
            buy_conf = sum(s.confidence for s in buy_signals) / len(buy_signals) if buy_signals else 0
            sell_conf = sum(s.confidence for s in sell_signals) / len(sell_signals) if sell_signals else 0

            if buy_conf > sell_conf:
                winning_signals = buy_signals
                signal_type = SignalType.BUY
            elif sell_conf > buy_conf:
                winning_signals = sell_signals
                signal_type = SignalType.SELL
            else:
                return None  # Sem decisao

        if not winning_signals:
            return None

        # Calcula confianca agregada (media ponderada)
        total_conf = sum(s.confidence for s in winning_signals)
        avg_confidence = total_conf / len(winning_signals)

        # Pega preco medio
        avg_price = sum(s.price for s in winning_signals) / len(winning_signals)

        # Stop/Take mais conservador
        if signal_type == SignalType.BUY:
            stop_losses = [s.stop_loss for s in winning_signals if s.stop_loss]
            take_profits = [s.take_profit for s in winning_signals if s.take_profit]
            stop_loss = max(stop_losses) if stop_losses else None  # Mais proximo
            take_profit = min(take_profits) if take_profits else None  # Mais proximo
        else:
            stop_losses = [s.stop_loss for s in winning_signals if s.stop_loss]
            take_profits = [s.take_profit for s in winning_signals if s.take_profit]
            stop_loss = min(stop_losses) if stop_losses else None  # Mais proximo
            take_profit = max(take_profits) if take_profits else None  # Mais proximo

        # Preserva stop_loss_pips e take_profit_pips dos sinais originais
        sl_pips_list = [getattr(s, 'stop_loss_pips', None) for s in winning_signals if getattr(s, 'stop_loss_pips', None)]
        tp_pips_list = [getattr(s, 'take_profit_pips', None) for s in winning_signals if getattr(s, 'take_profit_pips', None)]
        stop_loss_pips = max(sl_pips_list) if sl_pips_list else None
        take_profit_pips = min(tp_pips_list) if tp_pips_list else None

        # Gera razao
        strategy_names = [s.strategy_name for s in winning_signals]
        reason = f"Agregado de {len(winning_signals)} estrategias ({volatility_level}): {', '.join(strategy_names)}"

        return Signal(
            type=signal_type,
            price=avg_price,
            timestamp=winning_signals[0].timestamp,
            strategy_name=f"Orchestrator-{volatility_level}",
            confidence=avg_confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
            stop_loss_pips=stop_loss_pips,
            take_profit_pips=take_profit_pips
        )

    def get_state(self) -> OrchestratorState:
        """Retorna estado atual"""
        return self.state

    def get_stats(self) -> dict:
        """Retorna estatisticas"""
        return self.stats.copy()

    def get_registered_strategies(self) -> Dict[str, List[str]]:
        """Retorna estrategias registradas por nivel"""
        return {
            level: [s.name for s in strategies]
            for level, strategies in self.strategies.items()
        }

    def reset(self):
        """Reseta o orquestrador"""
        self.state = OrchestratorState()
        self.signals_history.clear()
        self.ticks_since_last_signal = 0

        for strategies in self.strategies.values():
            for strategy in strategies:
                strategy.reset()

        self.stats = {
            'ticks_processed': 0,
            'signals_generated': 0,
            'signals_by_level': {'ALTA': 0, 'MEDIA': 0, 'BAIXA': 0},
            'signals_by_type': {'BUY': 0, 'SELL': 0}
        }


# ==============================================================================
# TESTE SIMPLES
# ==============================================================================

def test_orchestrator():
    """Testa o orquestrador com dados simulados"""
    print("=" * 70)
    print("TESTE DO STRATEGY ORCHESTRATOR")
    print("=" * 70)

    # Cria orquestrador
    orch = StrategyOrchestrator(
        symbol="EURUSD",
        min_confidence=0.3,
        signal_cooldown_ticks=10
    )

    print(f"\nOrquestrador criado para {orch.symbol}")
    print(f"Confianca minima: {orch.min_confidence}")
    print(f"Cooldown: {orch.signal_cooldown_ticks} ticks")

    # Registra estrategias (placeholder)
    print("\n" + "-" * 70)
    print("Estrategias disponiveis (nao registradas ainda):")
    print(f"  ALTA: {len(orch.strategies['ALTA'])} estrategias")
    print(f"  MEDIA: {len(orch.strategies['MEDIA'])} estrategias")
    print(f"  BAIXA: {len(orch.strategies['BAIXA'])} estrategias")

    # Simula ticks
    print("\n" + "-" * 70)
    print("Simulando 100 ticks...")

    base_price = 1.0850
    np.random.seed(42)

    for i in range(100):
        price = base_price + np.random.randn() * 0.0005
        timestamp = datetime.now(timezone.utc)

        signal = orch.process_tick(price, timestamp)

        if i % 20 == 0:
            state = orch.get_state()
            print(f"Tick {i:3d}: Vol={state.current_volatility:.3f} pips | "
                  f"Level={state.volatility_level:10s} | "
                  f"Price={price:.5f}")

    # Resumo
    print("\n" + "-" * 70)
    print("RESUMO:")
    stats = orch.get_stats()
    print(f"Ticks processados: {stats['ticks_processed']}")
    print(f"Sinais gerados: {stats['signals_generated']}")
    print(f"Estado final: {orch.get_state().volatility_level}")

    return orch


if __name__ == "__main__":
    test_orchestrator()
