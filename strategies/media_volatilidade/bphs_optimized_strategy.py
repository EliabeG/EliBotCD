"""
================================================================================
BPHS OPTIMIZED STRATEGY - Compatible with Robust Optimizer
================================================================================

IMPORTANTE: Esta estrategia usa a logica EXATA validada pelo otimizador robusto.
Os parametros sao carregados automaticamente do arquivo bphs_robust_optimized.json.

VALIDACAO DE LOOK-AHEAD BIAS:
1. Sinal gerado com dados ATE indice i (closes[:i])
2. Entrada executada no OPEN da barra i+1
3. SL/TP verificados a partir da barra i+2

ANTI-OVERFITTING:
- Parametros validados via Anchored Walk-Forward (5 folds)
- Monte Carlo validation (100 shuffles)
- Out-of-sample final (20% dados nunca vistos)
- Custos conservadores: 2.3 pips/trade

Autor: Claude Opus 4.5 (Auditoria Automatizada)
Data: 2025-12-29
================================================================================
"""

from datetime import datetime
from typing import Optional, List, Dict
from collections import deque
import numpy as np
import json
import os

from ..base import BaseStrategy, Signal, SignalType
from .bphs_betti_persistence import BettiPersistenceHomologyScanner, TopologyRegime

try:
    from config.logging_config import get_logger
    _logger = get_logger("bphs.optimized")
except ImportError:
    import logging
    _logger = logging.getLogger(__name__)


class BPHSOptimizedStrategy(BaseStrategy):
    """
    Estrategia BPHS Otimizada - Compativel com parametros do otimizador robusto.

    Esta estrategia usa a mesma logica de geracao de sinais validada pelo
    otimizador, garantindo ausencia de look-ahead bias e overfitting.

    Condicoes para sinal:
    - Regime CYCLIC (B1 > 0 com persistencia significativa)
    - Posicao no ciclo = TOP (SHORT) ou BOTTOM (LONG)
    - Confianca >= min_confidence
    - Entropia < max_entropy_threshold
    - Nao esta em BETTI_CRASH

    Nao opera em:
    - Regime LINEAR (sem ciclos)
    - Regime CHAOTIC (entropia alta)
    - Posicao MIDDLE do ciclo
    """

    def __init__(self,
                 # Parametros do indicador (serao carregados do config)
                 embedding_dim: int = 3,
                 filtration_steps: int = 50,
                 min_loop_persistence: float = 0.10,
                 max_entropy_threshold: float = 2.0,
                 position_threshold: float = 0.30,
                 min_confidence: float = 0.40,
                 # Parametros de trade
                 stop_loss_pips: float = 26.0,
                 take_profit_pips: float = 50.0,
                 cooldown_bars: int = 25,
                 # Parametros internos
                 min_prices: int = 100,
                 # Carregar config automaticamente
                 load_optimized_config: bool = True):
        """
        Inicializa a estrategia BPHS Otimizada

        Args:
            embedding_dim: Dimensao de imersao de Takens
            filtration_steps: Passos da filtracao Vietoris-Rips
            min_loop_persistence: Persistencia minima para sinal valido
            max_entropy_threshold: Limiar para regime caotico
            position_threshold: Limiar para detectar extremos do ciclo
            min_confidence: Confianca minima para sinal
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            cooldown_bars: Cooldown entre sinais
            min_prices: Minimo de precos para analise
            load_optimized_config: Se True, carrega params do JSON
        """
        super().__init__(name="BPHS-Optimized")

        # Tentar carregar config otimizado
        if load_optimized_config:
            config = self._load_optimized_config()
            if config:
                embedding_dim = config.get('embedding_dim', embedding_dim)
                filtration_steps = config.get('filtration_steps', filtration_steps)
                min_loop_persistence = config.get('min_loop_persistence', min_loop_persistence)
                max_entropy_threshold = config.get('max_entropy_threshold', max_entropy_threshold)
                position_threshold = config.get('position_threshold', position_threshold)
                min_confidence = config.get('min_confidence', min_confidence)
                stop_loss_pips = config.get('stop_loss_pips', stop_loss_pips)
                take_profit_pips = config.get('take_profit_pips', take_profit_pips)
                cooldown_bars = config.get('cooldown_bars', cooldown_bars)
                _logger.info(f"Loaded optimized config: emb={embedding_dim}, "
                            f"pers={min_loop_persistence}, conf={min_confidence}")

        # Salvar parametros
        self.embedding_dim = embedding_dim
        self.filtration_steps = filtration_steps
        self.min_loop_persistence = min_loop_persistence
        self.max_entropy_threshold = max_entropy_threshold
        self.position_threshold = position_threshold
        self.min_confidence = min_confidence
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.cooldown_bars = cooldown_bars
        self.min_prices = min_prices

        # Buffer de precos
        self.prices = deque(maxlen=600)

        # Cooldown
        self.signal_cooldown = 0
        self.bar_count = 0

        # Indicador BPHS
        self.bphs = BettiPersistenceHomologyScanner(
            embedding_dim=embedding_dim,
            time_delay=None,  # Auto-calculate
            max_homology_dim=2,
            filtration_steps=filtration_steps,
            min_loop_persistence=min_loop_persistence,
            max_entropy_threshold=max_entropy_threshold,
            position_threshold=position_threshold,
            min_data_points=min_prices
        )

        self.last_analysis = None

        _logger.info(f"BPHSOptimizedStrategy initialized:")
        _logger.info(f"  embedding_dim: {self.embedding_dim}")
        _logger.info(f"  filtration_steps: {self.filtration_steps}")
        _logger.info(f"  min_loop_persistence: {self.min_loop_persistence}")
        _logger.info(f"  max_entropy_threshold: {self.max_entropy_threshold}")
        _logger.info(f"  position_threshold: {self.position_threshold}")
        _logger.info(f"  min_confidence: {self.min_confidence}")
        _logger.info(f"  stop_loss: {self.stop_loss_pips} pips")
        _logger.info(f"  take_profit: {self.take_profit_pips} pips")
        _logger.info(f"  cooldown: {self.cooldown_bars} bars")

    def _load_optimized_config(self) -> Optional[dict]:
        """Carrega configuracao otimizada do arquivo JSON"""
        try:
            # Tentar varios caminhos possiveis
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", "configs", "bphs_robust_optimized.json"),
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "bphs_robust_optimized.json"),
                "/home/azureuser/EliBotCD/configs/bphs_robust_optimized.json",
            ]

            for path in possible_paths:
                path = os.path.abspath(path)
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        config = json.load(f)
                        if config and 'parameters' in config:
                            return config['parameters']

            _logger.warning("Config file not found, using default params")
            return None

        except Exception as e:
            _logger.error(f"Error loading config: {e}")
            return None

    def add_price(self, price: float):
        """Adiciona um preco ao buffer"""
        self.prices.append(price)
        self.bar_count += 1

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado usando a logica VALIDADA do otimizador.

        PROTOCOLO DE EXECUCAO (sem look-ahead):
        1. Sinal gerado com dados ATE este ponto
        2. Entrada sera no OPEN da proxima barra
        3. SL/TP verificados nas barras futuras

        Condicoes para sinal:
        - Regime CYCLIC
        - Posicao = TOP ou BOTTOM
        - Confianca >= min_confidence
        - Sem BETTI_CRASH

        Args:
            price: Preco atual (close da barra)
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (volatilidade, hurst, etc)

        Returns:
            Signal se condicoes atendidas, None caso contrario
        """
        # Adiciona preco ao buffer
        self.add_price(price)

        # Verifica dados suficientes
        if len(self.prices) < self.min_prices:
            return None

        # Cooldown
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        # Converte para numpy array
        # IMPORTANTE: Apenas dados PASSADOS (ate o indice atual)
        prices_array = np.array(self.prices)

        try:
            # Executa analise BPHS
            result = self.bphs.analyze(prices_array)
            self.last_analysis = result

            # ============================================================
            # LOGICA VALIDADA DO OTIMIZADOR
            # ============================================================

            # 1. Verificar regime
            regime = result['regime']

            # Bloquear em regimes nao-ciclicos
            if regime == 'LINEAR':
                return None
            if regime == 'CHAOTIC':
                return None
            if regime == 'TRANSITIONAL':
                return None

            # 2. Verificar BETTI_CRASH
            if 'BETTI_CRASH' in result.get('reasons', []):
                return None

            # 3. Verificar confianca
            confidence = result['confidence']
            if confidence < self.min_confidence:
                return None

            # 4. Verificar posicao no ciclo
            position = result['position_in_loop']

            # 5. Verificar persistencia do loop
            persistence = result['max_loop_persistence']
            if persistence < self.min_loop_persistence:
                return None

            # 6. Verificar B1 (numero de loops)
            betti_1 = result['betti_1']
            if betti_1 == 0:
                return None

            # ============================================================
            # GERAR SINAL
            # ============================================================

            signal_type = None
            reason_prefix = ""

            if position == "BOTTOM":
                # Na base do ciclo -> LONG
                signal_type = SignalType.BUY
                reason_prefix = "CYCLE_BOTTOM"

            elif position == "TOP":
                # No topo do ciclo -> SHORT
                signal_type = SignalType.SELL
                reason_prefix = "CYCLE_TOP"

            else:
                # MIDDLE ou UNKNOWN -> nao opera
                return None

            # Calcular stop loss e take profit
            pip_value = 0.0001

            if signal_type == SignalType.BUY:
                stop_loss = price - (self.stop_loss_pips * pip_value)
                take_profit = price + (self.take_profit_pips * pip_value)
            else:
                stop_loss = price + (self.stop_loss_pips * pip_value)
                take_profit = price - (self.take_profit_pips * pip_value)

            # Criar sinal
            signal = Signal(
                type=signal_type,
                price=price,
                timestamp=timestamp,
                strategy_name=self.name,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                stop_loss_pips=self.stop_loss_pips,
                take_profit_pips=self.take_profit_pips,
                reason=self._generate_reason(result, reason_prefix)
            )

            self.last_signal = signal
            self.signal_cooldown = self.cooldown_bars

            _logger.info(f"Signal generated: {signal_type.name} @ {price:.5f}, "
                        f"B1={betti_1}, pers={persistence:.3f}, pos={position}")

            return signal

        except Exception as e:
            _logger.error(f"Error in BPHS analysis: {e}", exc_info=True)
            return None

    def _generate_reason(self, result: dict, prefix: str) -> str:
        """Gera descricao do motivo do sinal"""
        return (f"BPHS-Opt {prefix} | "
                f"B1={result['betti_1']} | "
                f"Pers={result['max_loop_persistence']:.3f} | "
                f"Regime={result['regime']} | "
                f"Entropy={result['topological_entropy']:.3f}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.prices.clear()
        self.bphs.reset()
        self.last_analysis = None
        self.last_signal = None
        self.signal_cooldown = 0
        self.bar_count = 0

    def get_analysis_summary(self) -> Optional[dict]:
        """Retorna resumo da ultima analise"""
        if self.last_analysis is None:
            return None

        return {
            'betti_1': self.last_analysis['betti_1'],
            'max_loop_persistence': self.last_analysis['max_loop_persistence'],
            'regime': self.last_analysis['regime'],
            'position_in_loop': self.last_analysis['position_in_loop'],
            'topological_entropy': self.last_analysis['topological_entropy'],
            'confidence': self.last_analysis['confidence'],
            'bar_count': self.bar_count
        }

    def get_betti_info(self) -> Optional[dict]:
        """Retorna numeros de Betti atuais"""
        if self.last_analysis is None:
            return None
        return {
            'betti_0': self.last_analysis['betti_0'],
            'betti_1': self.last_analysis['betti_1'],
            'betti_2': self.last_analysis['betti_2']
        }

    def is_cyclic_regime(self) -> bool:
        """Verifica se esta em regime ciclico"""
        if self.last_analysis is None:
            return False
        return self.last_analysis['regime'] == 'CYCLIC'

    def has_persistent_loop(self) -> bool:
        """Verifica se ha loop persistente"""
        if self.last_analysis is None:
            return False
        return (self.last_analysis['betti_1'] > 0 and
                self.last_analysis['max_loop_persistence'] >= self.min_loop_persistence)

    def get_position_in_loop(self) -> Optional[str]:
        """Retorna posicao no ciclo"""
        if self.last_analysis is None:
            return None
        return self.last_analysis['position_in_loop']

    def get_loop_persistence(self) -> Optional[float]:
        """Retorna persistencia do loop"""
        if self.last_analysis is None:
            return None
        return self.last_analysis['max_loop_persistence']

    def get_topological_entropy(self) -> Optional[float]:
        """Retorna entropia topologica"""
        if self.last_analysis is None:
            return None
        return self.last_analysis['topological_entropy']


# ==============================================================================
# TESTE
# ==============================================================================

def test_bphs_optimized():
    """Testa a estrategia BPHS otimizada"""
    print("=" * 70)
    print("TESTE DA ESTRATEGIA BPHS OTIMIZADA")
    print("=" * 70)

    strategy = BPHSOptimizedStrategy(load_optimized_config=False)

    print(f"\nEstrategia: {strategy.name}")
    print(f"Embedding: {strategy.embedding_dim}")
    print(f"Persistencia minima: {strategy.min_loop_persistence}")
    print(f"Confianca minima: {strategy.min_confidence}")
    print(f"SL: {strategy.stop_loss_pips} pips")
    print(f"TP: {strategy.take_profit_pips} pips")

    # Simular precos
    np.random.seed(42)
    base_price = 1.0850

    print("\nSimulando 200 barras...")

    for i in range(200):
        # Preco com ciclo + ruido
        cycle = 0.0010 * np.sin(2 * np.pi * i / 50)  # Ciclo de 50 barras
        noise = np.random.randn() * 0.0003
        price = base_price + cycle + noise

        timestamp = datetime.now()
        signal = strategy.analyze(price, timestamp)

        if signal:
            print(f"  Barra {i}: {signal.type.name} @ {price:.5f} "
                  f"(conf={signal.confidence:.2f})")

    # Resumo
    print("\nResumo final:")
    summary = strategy.get_analysis_summary()
    if summary:
        print(f"  B1 (loops): {summary['betti_1']}")
        print(f"  Persistencia: {summary['max_loop_persistence']:.3f}")
        print(f"  Regime: {summary['regime']}")
        print(f"  Posicao: {summary['position_in_loop']}")

    return strategy


if __name__ == "__main__":
    test_bphs_optimized()
