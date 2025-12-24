"""
================================================================================
Adaptador de Estratégia para o Detector de Tunelamento Topológico
================================================================================

VERSÃO V3.0 - CORREÇÕES DE FUNDAMENTOS TEÓRICOS 24/12/2025:
1. Múltiplos modos de direção (momentum, quantum, topology, hybrid)
2. Opção de exigir consenso entre métodos
3. Logging detalhado de qual direção foi usada
4. Integração com calibração automática de parâmetros quânticos

Integra o indicador DTT com o sistema de trading.
================================================================================
"""
import logging
from datetime import datetime
from typing import Optional
from collections import deque
from enum import Enum
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .dtt_tunelamento_topologico import DetectorTunelamentoTopologico

# Importar módulo compartilhado de direção
try:
    from backtesting.common.direction_calculator import (
        calculate_direction_from_closes,
        DEFAULT_DIRECTION_LOOKBACK,
        DIRECTION_LONG,
        DIRECTION_SHORT,
        DIRECTION_NEUTRAL
    )
    USE_SHARED_DIRECTION = True
except ImportError:
    USE_SHARED_DIRECTION = False
    DEFAULT_DIRECTION_LOOKBACK = 12
    DIRECTION_LONG = 1
    DIRECTION_SHORT = -1
    DIRECTION_NEUTRAL = 0

# Configurar logger
logger = logging.getLogger(__name__)


class DirectionMode(Enum):
    """
    Modo de cálculo de direção V3.0

    - MOMENTUM: Momentum simples de barras fechadas (conservador, sem look-ahead)
    - QUANTUM: Direção do gradiente da função de onda ψ (experimental)
    - TOPOLOGY: Baseado em posição relativa às barreiras de potencial
    - HYBRID: Consenso de múltiplos métodos (mais conservador)
    """
    MOMENTUM = "momentum"
    QUANTUM = "quantum"
    TOPOLOGY = "topology"
    HYBRID = "hybrid"


class DTTStrategy(BaseStrategy):
    """
    Estratégia baseada no Detector de Tunelamento Topológico

    VERSÃO V3.0 - FUNDAMENTOS TEÓRICOS CORRIGIDOS:

    Modos de Direção:
    - momentum: Usa diferença de preços fechados (mais seguro)
    - quantum: Usa gradiente da função de onda ψ
    - topology: Usa posição relativa às barreiras de potencial
    - hybrid: Exige consenso entre métodos (mais conservador)

    Parâmetros Quânticos:
    - Com auto_calibrate_quantum=True, ℏ, m, kT são calibrados
      automaticamente baseado na volatilidade do ativo
    """

    def __init__(self,
                 min_prices: int = 150,
                 stop_loss_pips: float = 25.0,
                 take_profit_pips: float = 50.0,
                 persistence_entropy_threshold: float = 0.5,
                 tunneling_probability_threshold: float = 0.15,
                 min_signal_strength: float = 0.3,
                 direction_lookback: int = DEFAULT_DIRECTION_LOOKBACK,
                 # V3.0: Novos parâmetros de direção
                 direction_mode: str = "hybrid",
                 require_consensus: bool = True,
                 quantum_weight: float = 0.3,
                 # V3.0: Calibração de parâmetros quânticos
                 auto_calibrate_quantum: bool = True,
                 hbar: float = None,
                 particle_mass: float = None,
                 kT: float = None):
        """
        Inicializa a estratégia DTT V3.0

        Args:
            min_prices: Mínimo de preços necessários para análise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            persistence_entropy_threshold: Limiar de entropia para mercado não-trivial
            tunneling_probability_threshold: Limiar de probabilidade de tunelamento
            min_signal_strength: Força mínima do sinal para disparo
            direction_lookback: Barras para calcular direção momentum

            # V3.0 Novos:
            direction_mode: "momentum", "quantum", "topology", ou "hybrid"
            require_consensus: Se True, no modo hybrid exige que todos concordem
            quantum_weight: Peso da direção quântica/topológica no modo hybrid (0-1)
            auto_calibrate_quantum: Calibrar parâmetros ℏ, m, kT automaticamente
            hbar: Constante de Planck reduzida (None = auto calibrar)
            particle_mass: Massa da partícula quântica (None = auto calibrar)
            kT: Temperatura do mercado (None = auto calibrar)
        """
        super().__init__(name="DTT-V3.0")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.min_signal_strength = min_signal_strength
        self.direction_lookback = direction_lookback

        # V3.0: Configuração de direção
        try:
            self.direction_mode = DirectionMode(direction_mode)
        except ValueError:
            logger.warning(f"Modo de direção inválido: {direction_mode}, usando 'hybrid'")
            self.direction_mode = DirectionMode.HYBRID

        self.require_consensus = require_consensus
        self.quantum_weight = np.clip(quantum_weight, 0.0, 1.0)

        # Buffer de preços
        self.prices = deque(maxlen=600)
        self.closes = deque(maxlen=600)

        # Indicador DTT V3.0 com calibração
        self.dtt = DetectorTunelamentoTopologico(
            max_points=150,
            use_dimensionality_reduction=True,
            reduction_method='pca',
            persistence_entropy_threshold=persistence_entropy_threshold,
            tunneling_probability_threshold=tunneling_probability_threshold,
            hbar=hbar,
            particle_mass=particle_mass,
            kT=kT,
            auto_calibrate_quantum=auto_calibrate_quantum
        )

        # Estado
        self.last_analysis = None
        self.last_direction_source = None
        self.signal_cooldown = 0
        self.error_count = 0
        self.max_errors_to_log = 10

        logger.info(f"DTTStrategy V3.0 inicializada: mode={direction_mode}, "
                    f"consensus={require_consensus}, quantum_weight={quantum_weight}, "
                    f"auto_calibrate={auto_calibrate_quantum}")

    def add_price(self, price: float):
        """Adiciona um preço ao buffer"""
        self.prices.append(price)
        self.closes.append(price)

    # =========================================================================
    # MÉTODOS DE CÁLCULO DE DIREÇÃO V3.0
    # =========================================================================

    def _calculate_direction_momentum(self) -> int:
        """
        Direção via momentum simples (sem look-ahead)

        REGRAS ANTI LOOK-AHEAD:
        - closes[-1] = barra atual (momento do sinal) - NÃO USAR
        - closes[-2] = última barra completamente fechada - USAR
        - closes[-(lookback+1)] = barra de comparação - USAR

        Returns:
            1 para LONG, -1 para SHORT, 0 para NEUTRAL
        """
        if len(self.closes) < self.direction_lookback + 2:
            return DIRECTION_NEUTRAL

        if USE_SHARED_DIRECTION:
            return calculate_direction_from_closes(
                list(self.closes),
                self.direction_lookback
            )
        else:
            recent_close = self.closes[-2]
            past_close = self.closes[-(self.direction_lookback + 1)]
            trend = recent_close - past_close
            return DIRECTION_LONG if trend > 0 else DIRECTION_SHORT

    def _calculate_direction_quantum(self) -> int:
        """
        Direção via gradiente da função de onda ψ

        Usa o momentum_direction calculado pelo módulo Schrödinger,
        que indica o fluxo de probabilidade no preço atual.

        NOTA: O KDE foi corrigido para excluir o preço atual,
        então não há look-ahead direto. Porém, esta direção ainda
        é mais "experimental" que o momentum simples.

        Returns:
            1 para LONG, -1 para SHORT, 0 para NEUTRAL
        """
        if self.last_analysis is None:
            return DIRECTION_NEUTRAL

        tunneling = self.last_analysis.get('tunneling', {})
        mom_dir = tunneling.get('momentum_direction', 0)

        if mom_dir is None or np.isnan(mom_dir) or mom_dir == 0:
            return DIRECTION_NEUTRAL

        return DIRECTION_LONG if mom_dir > 0 else DIRECTION_SHORT

    def _calculate_direction_topology(self) -> int:
        """
        Direção baseada na estrutura topológica

        Lógica:
        - Se há ciclos significativos (estrutura topológica não-trivial)
        - Direção = posição do preço relativa ao centro do "poço de potencial"
        - Preço acima do centro + alta prob. tunelamento → LONG (rompendo para cima)
        - Preço abaixo do centro + alta prob. tunelamento → SHORT (rompendo para baixo)

        Returns:
            1 para LONG, -1 para SHORT, 0 para NEUTRAL
        """
        if self.last_analysis is None:
            return DIRECTION_NEUTRAL

        entropy_info = self.last_analysis.get('entropy', {})
        tunneling = self.last_analysis.get('tunneling', {})

        # Se não há ciclos significativos, sem direção topológica
        if entropy_info.get('n_significant_cycles', 0) == 0:
            return DIRECTION_NEUTRAL

        current_price = self.last_analysis.get('current_price', 0)
        left_barrier = tunneling.get('left_barrier', current_price)
        right_barrier = tunneling.get('right_barrier', current_price)

        # Centro do "poço" de potencial
        center = (left_barrier + right_barrier) / 2

        # Se preço acima do centro → LONG (rompendo para cima)
        # Se preço abaixo do centro → SHORT (rompendo para baixo)
        return DIRECTION_LONG if current_price > center else DIRECTION_SHORT

    def _calculate_direction_hybrid(self) -> int:
        """
        Combina múltiplas fontes de direção

        Modos:
        1. require_consensus=True: Só opera se todas as direções válidas concordam
        2. require_consensus=False: Média ponderada das direções

        Returns:
            1 para LONG, -1 para SHORT, 0 para NEUTRAL/sem consenso
        """
        dir_momentum = self._calculate_direction_momentum()
        dir_quantum = self._calculate_direction_quantum()
        dir_topology = self._calculate_direction_topology()

        directions = {
            'momentum': dir_momentum,
            'quantum': dir_quantum,
            'topology': dir_topology
        }

        # Filtrar direções válidas (não-neutras)
        valid = {k: v for k, v in directions.items() if v != DIRECTION_NEUTRAL}

        if not valid:
            self.last_direction_source = "no_valid_directions"
            return DIRECTION_NEUTRAL

        if self.require_consensus:
            # Exigir que todas as direções válidas concordem
            values = list(valid.values())
            if len(set(values)) == 1:
                self.last_direction_source = f"consensus({','.join(valid.keys())})"
                return values[0]
            else:
                self.last_direction_source = f"no_consensus({','.join(f'{k}={v}' for k,v in valid.items())})"
                return DIRECTION_NEUTRAL
        else:
            # Média ponderada
            weights = {
                'momentum': 1.0 - self.quantum_weight,
                'quantum': self.quantum_weight * 0.5,
                'topology': self.quantum_weight * 0.5
            }

            weighted_sum = sum(directions[k] * weights[k] for k in valid.keys())
            weight_total = sum(weights[k] for k in valid.keys())

            if weight_total > 0:
                avg = weighted_sum / weight_total
                result = DIRECTION_LONG if avg > 0 else DIRECTION_SHORT
                self.last_direction_source = f"weighted({','.join(valid.keys())})"
                return result

            self.last_direction_source = "weighted_zero"
            return DIRECTION_NEUTRAL

    def _calculate_direction(self) -> int:
        """
        Calcula direção baseado no modo configurado

        Returns:
            1 para LONG, -1 para SHORT, 0 para NEUTRAL
        """
        if self.direction_mode == DirectionMode.MOMENTUM:
            self.last_direction_source = "momentum"
            return self._calculate_direction_momentum()
        elif self.direction_mode == DirectionMode.QUANTUM:
            self.last_direction_source = "quantum"
            return self._calculate_direction_quantum()
        elif self.direction_mode == DirectionMode.TOPOLOGY:
            self.last_direction_source = "topology"
            return self._calculate_direction_topology()
        elif self.direction_mode == DirectionMode.HYBRID:
            return self._calculate_direction_hybrid()
        else:
            self.last_direction_source = "fallback_momentum"
            return self._calculate_direction_momentum()

    # =========================================================================
    # ANÁLISE PRINCIPAL
    # =========================================================================

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal se houver tunnelling event

        VERSÃO V3.0:
        - Direção via modo configurado (momentum/quantum/topology/hybrid)
        - Parâmetros quânticos calibrados automaticamente
        - Tratamento de erros com logging

        Args:
            price: Preço atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais

        Returns:
            Signal se tunnelling detectado, None caso contrário
        """
        # Adiciona preço ao buffer
        self.add_price(price)

        # Verifica se temos dados suficientes
        if len(self.prices) < self.min_prices:
            return None

        # Cooldown para evitar sinais em sequência
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        # Converte para numpy array
        prices_array = np.array(self.prices)

        try:
            # Executa análise DTT V3.0
            result = self.dtt.analyze(prices_array)
            self.last_analysis = result

            # Verifica se há tunnelling event
            if result['trade_on']:
                # Verifica força mínima do sinal
                if result['signal_strength'] < self.min_signal_strength:
                    return None

                # V3.0: Calcula direção via modo configurado
                direction_num = self._calculate_direction()

                if direction_num == DIRECTION_NEUTRAL:
                    logger.debug(f"Direção indeterminada: {self.last_direction_source}")
                    return None

                direction = SignalType.BUY if direction_num == DIRECTION_LONG else SignalType.SELL

                # Calcula níveis de stop e take profit
                pip_value = 0.0001

                if direction == SignalType.BUY:
                    stop_loss = price - (self.stop_loss_pips * pip_value)
                    take_profit = price + (self.take_profit_pips * pip_value)
                else:
                    stop_loss = price + (self.stop_loss_pips * pip_value)
                    take_profit = price - (self.take_profit_pips * pip_value)

                # Cria sinal
                signal = Signal(
                    type=direction,
                    price=price,
                    timestamp=timestamp,
                    strategy_name=self.name,
                    confidence=result['signal_strength'],
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    stop_loss_pips=self.stop_loss_pips,
                    take_profit_pips=self.take_profit_pips,
                    reason=self._generate_reason(result)
                )

                self.last_signal = signal
                self.signal_cooldown = 15

                logger.info(f"Sinal DTT: {direction.value} via {self.last_direction_source} | "
                            f"strength={result['signal_strength']:.3f}")

                return signal

        except Exception as e:
            self.error_count += 1
            if self.error_count <= self.max_errors_to_log:
                logger.warning(f"DTT análise falhou [{self.error_count}]: {e}")
            elif self.error_count == self.max_errors_to_log + 1:
                logger.warning("DTT: Erros subsequentes serão suprimidos...")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descrição do motivo do sinal"""
        entropy = result['entropy']
        tunneling = result['tunneling']

        # V3.0: Incluir fonte da direção
        calibration = result.get('quantum_calibration', {})
        calibrated = calibration.get('calibrated', False)

        return (f"Tunnelling V3.0 | dir_src={self.last_direction_source} | "
                f"Entropia: {entropy['persistence_entropy']:.3f} | "
                f"P(Tunnel): {tunneling['tunneling_probability']:.3f} | "
                f"Betti_1: {result['topology']['betti_1']} | "
                f"calibrated={calibrated}")

    def reset(self):
        """Reseta o estado da estratégia"""
        self.prices.clear()
        self.closes.clear()
        self.last_analysis = None
        self.last_signal = None
        self.last_direction_source = None
        self.signal_cooldown = 0
        self.error_count = 0

    def get_analysis_summary(self) -> Optional[dict]:
        """Retorna resumo da última análise"""
        if self.last_analysis is None:
            return None

        calibration = self.last_analysis.get('quantum_calibration', {})

        return {
            'trade_on': self.last_analysis['trade_on'],
            'direction': self.last_analysis['direction'],
            'direction_source': self.last_direction_source,
            'direction_mode': self.direction_mode.value,
            'signal_strength': self.last_analysis['signal_strength'],
            'persistence_entropy': self.last_analysis['entropy']['persistence_entropy'],
            'tunneling_probability': self.last_analysis['tunneling']['tunneling_probability'],
            'betti_1': self.last_analysis['topology']['betti_1'],
            'tda_backend': self.last_analysis['tda_backend'],
            # V3.0: Info de calibração
            'quantum_calibration': calibration
        }

    def get_error_count(self) -> int:
        """Retorna contagem de erros para diagnóstico"""
        return self.error_count

    def get_direction_mode(self) -> str:
        """Retorna modo de direção atual"""
        return self.direction_mode.value

    def set_direction_mode(self, mode: str):
        """Altera modo de direção em runtime"""
        try:
            self.direction_mode = DirectionMode(mode)
            logger.info(f"Modo de direção alterado para: {mode}")
        except ValueError:
            logger.warning(f"Modo de direção inválido: {mode}")
