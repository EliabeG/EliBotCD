"""
================================================================================
PRM OPTIMIZED STRATEGY - Compatible with Optimizer's Logic
================================================================================

IMPORTANTE: Esta estratégia usa a mesma lógica do otimizador (optimizer.py).
O PRMStrategy original usa lógica diferente que não é compatível com os
parâmetros otimizados.

Diferenças críticas:
1. Lyapunov: Otimizador verifica >= threshold, PRM original verifica 0 < x < threshold
2. Curvatura: Otimizador IGNORA, PRM original REQUER
3. HMM: Lógica similar mas otimizador é menos restritivo

Esta versão implementa a lógica EXATA do otimizador para garantir que os
parâmetros otimizados funcionem corretamente.
"""

from datetime import datetime
from typing import Optional, List
from collections import deque
import numpy as np
import json
import os

from ..base import BaseStrategy, Signal, SignalType
from .prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot

try:
    from config.logging_config import get_logger
    _logger = get_logger("prm.optimized")
except ImportError:
    import logging
    _logger = logging.getLogger(__name__)


class PRMOptimizedStrategy(BaseStrategy):
    """
    Estratégia PRM Otimizada - Compatível com parâmetros do otimizador

    Esta estratégia usa a mesma lógica de geração de sinais do otimizador:
    - hmm_prob >= hmm_threshold
    - lyapunov >= lyapunov_threshold (NÃO 0 < lyap < threshold!)
    - hmm_state in hmm_states_allowed
    - direction != 0 (baseado em tendência de barras fechadas)

    NÃO usa a condição de curvatura (o otimizador também não usa).
    """

    def __init__(self,
                 # Parâmetros otimizados (do prm_robust_top10.json)
                 hmm_threshold: float = 0.7,
                 lyapunov_threshold: float = 0.04,
                 hmm_states_allowed: List[int] = None,
                 stop_loss_pips: float = 30.5,
                 take_profit_pips: float = 64.6,
                 # Parâmetros internos do PRM
                 min_prices: int = 100,
                 hmm_training_window: int = 200,
                 hmm_min_training_samples: int = 50,
                 trend_lookback: int = 10,
                 signal_cooldown: int = 10,
                 # Carregar config automaticamente
                 load_optimized_config: bool = True):
        """
        Inicializa a estratégia PRM Otimizada

        Args:
            hmm_threshold: Threshold para probabilidade HMM (prob >= threshold)
            lyapunov_threshold: Threshold para Lyapunov (lyap >= threshold)
            hmm_states_allowed: Estados HMM permitidos (ex: [1, 2])
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            min_prices: Mínimo de preços para análise
            hmm_training_window: Janela de treino do HMM
            hmm_min_training_samples: Mínimo de amostras para HMM
            trend_lookback: Barras para calcular tendência
            signal_cooldown: Cooldown entre sinais
            load_optimized_config: Se True, carrega params de prm_robust_top10.json
        """
        super().__init__(name="PRM-Optimized")

        # Tentar carregar config otimizado
        if load_optimized_config:
            config = self._load_optimized_config()
            if config:
                hmm_threshold = config.get('hmm_threshold', hmm_threshold)
                lyapunov_threshold = config.get('lyapunov_threshold', lyapunov_threshold)
                hmm_states_allowed = config.get('hmm_states_allowed', hmm_states_allowed)
                stop_loss_pips = config.get('stop_loss_pips', stop_loss_pips)
                take_profit_pips = config.get('take_profit_pips', take_profit_pips)
                _logger.info(f"Loaded optimized config: hmm={hmm_threshold}, lyap={lyapunov_threshold}")

        # Parâmetros da estratégia
        self.hmm_threshold = hmm_threshold
        self.lyapunov_threshold = lyapunov_threshold
        self.hmm_states_allowed = hmm_states_allowed if hmm_states_allowed else [1, 2]
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        self.min_prices = min_prices
        self.trend_lookback = trend_lookback
        self.signal_cooldown_max = signal_cooldown
        self.signal_cooldown = 0

        # Buffer de preços
        self.prices = deque(maxlen=500)
        self.volumes = deque(maxlen=500)
        self.closes_history = deque(maxlen=500)

        # Indicador PRM com thresholds BAIXOS para pre-computar todos os valores
        # O filtro real é feito no método analyze() usando os thresholds otimizados
        self.prm = ProtocoloRiemannMandelbrot(
            n_states=3,
            hmm_threshold=0.1,  # Baixo para pegar todos os sinais
            lyapunov_threshold_k=0.001,  # Baixo para pegar todos os sinais
            curvature_threshold=0.0001,  # Baixo (mas não usamos curvatura)
            lookback_window=100,
            hmm_training_window=hmm_training_window,
            hmm_min_training_samples=hmm_min_training_samples
        )

        self.last_analysis = None
        self.bar_count = 0

        _logger.info(f"PRMOptimizedStrategy initialized:")
        _logger.info(f"  hmm_threshold: {self.hmm_threshold}")
        _logger.info(f"  lyapunov_threshold: {self.lyapunov_threshold}")
        _logger.info(f"  hmm_states_allowed: {self.hmm_states_allowed}")
        _logger.info(f"  stop_loss: {self.stop_loss_pips} pips")
        _logger.info(f"  take_profit: {self.take_profit_pips} pips")

    def _load_optimized_config(self) -> Optional[dict]:
        """Carrega configuração otimizada do arquivo JSON"""
        try:
            # Tentar vários caminhos possíveis
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", "configs", "prm_robust_top10.json"),
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "prm_robust_top10.json"),
                "/home/azureuser/EliBotCD/configs/prm_robust_top10.json",
            ]

            for path in possible_paths:
                path = os.path.abspath(path)
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        configs = json.load(f)
                        if configs and len(configs) > 0:
                            return configs[0]['params']

            _logger.warning("Config file not found, using default params")
            return None
        except Exception as e:
            _logger.error(f"Error loading config: {e}")
            return None

    def add_price(self, price: float, volume: float = None):
        """Adiciona um preço ao buffer"""
        self.prices.append(price)
        self.volumes.append(volume if volume else 1.0)

    def _calculate_direction(self) -> int:
        """
        Calcula direção baseada em tendência de barras FECHADAS
        Usa a MESMA lógica do otimizador.

        Returns:
            1 = LONG, -1 = SHORT, 0 = HOLD
        """
        min_bars = self.trend_lookback + 2
        if len(self.closes_history) < min_bars:
            return 0

        # closes_history[-1] = barra atual (NÃO usar)
        # closes_history[-2] = última barra fechada
        recent_close = self.closes_history[-2]
        past_close = self.closes_history[-(self.trend_lookback + 2)]

        trend = recent_close - past_close

        return 1 if trend > 0 else -1

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado usando a lógica do OTIMIZADOR (não do PRM original)

        Lógica do otimizador:
        - hmm_prob >= hmm_threshold
        - lyapunov >= lyapunov_threshold
        - hmm_state in hmm_states_allowed
        - direction != 0

        NÃO verifica curvatura (o otimizador não verifica).
        """
        # Adiciona preço ao buffer
        volume = indicators.get('volume', 1.0)
        self.add_price(price, volume)
        self.closes_history.append(price)
        self.bar_count += 1

        # Verifica se temos dados suficientes
        if len(self.prices) < self.min_prices:
            return None

        # Cooldown
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        # Converte para numpy arrays
        prices_array = np.array(self.prices)
        volumes_array = np.array(self.volumes)

        try:
            # Executa análise PRM
            result = self.prm.analyze(prices_array, volumes_array)
            self.last_analysis = result

            # ============================================================
            # LÓGICA DO OTIMIZADOR (NÃO do PRM original!)
            # ============================================================

            # 1. Prob_HMM >= threshold
            hmm_prob = result['Prob_HMM']
            if hmm_prob < self.hmm_threshold:
                return None

            # 2. Lyapunov >= threshold (DIFERENTE do PRM original que usa 0 < lyap < threshold)
            lyapunov = result['Lyapunov_Score']
            if lyapunov < self.lyapunov_threshold:
                return None

            # 3. Estado HMM permitido
            hmm_state = result['hmm_analysis']['current_state']
            if hmm_state not in self.hmm_states_allowed:
                return None

            # 4. Direção válida
            direction = self._calculate_direction()
            if direction == 0:
                return None

            # NÃO verifica curvatura (o otimizador não verifica)

            # ============================================================
            # GERAR SINAL
            # ============================================================

            signal_type = SignalType.BUY if direction == 1 else SignalType.SELL

            # Confiança baseada nos scores
            confidence = self._calculate_confidence(result)

            signal = Signal(
                type=signal_type,
                price=price,
                timestamp=timestamp,
                strategy_name=self.name,
                confidence=confidence,
                stop_loss=None,
                take_profit=None,
                stop_loss_pips=self.stop_loss_pips,
                take_profit_pips=self.take_profit_pips,
                reason=self._generate_reason(result, direction)
            )

            self.last_signal = signal
            self.signal_cooldown = self.signal_cooldown_max

            _logger.info(f"Signal generated: {signal_type.name} @ {price:.5f}, "
                        f"hmm_prob={hmm_prob:.3f}, lyap={lyapunov:.4f}, state={hmm_state}")

            return signal

        except Exception as e:
            _logger.error(f"Error in PRM analysis: {e}", exc_info=True)
            return None

    def _calculate_confidence(self, result: dict) -> float:
        """Calcula confiança do sinal"""
        hmm_prob = result['Prob_HMM']
        lyap_score = result['Lyapunov_Score']

        # Normalizar Lyapunov (0-1 baseado em range típico 0-0.2)
        lyap_normalized = min(lyap_score / 0.2, 1.0)

        # Média ponderada
        confidence = 0.6 * hmm_prob + 0.4 * lyap_normalized

        return min(max(confidence, 0.0), 1.0)

    def _generate_reason(self, result: dict, direction: int) -> str:
        """Gera descrição do motivo do sinal"""
        hmm = result['hmm_analysis']
        lyap = result['lyapunov_analysis']

        state_names = ['Consolidação', 'Alta Vol. Direcional', 'Choque de Vol.']
        state_name = state_names[hmm['current_state']]
        dir_name = "LONG" if direction == 1 else "SHORT"

        return (f"PRM-Opt {dir_name} | Estado: {state_name} "
                f"(P={result['Prob_HMM']:.2f}) | "
                f"Lyap: {lyap['lyapunov_max']:.4f}")

    def reset(self):
        """Reseta o estado da estratégia"""
        self.prices.clear()
        self.volumes.clear()
        self.closes_history.clear()
        self.last_analysis = None
        self.last_signal = None
        self.signal_cooldown = 0
        self.bar_count = 0

        # Recriar PRM
        self.prm = ProtocoloRiemannMandelbrot(
            n_states=3,
            hmm_threshold=0.1,
            lyapunov_threshold_k=0.001,
            curvature_threshold=0.0001,
            lookback_window=100,
            hmm_training_window=self.prm.hmm_training_window,
            hmm_min_training_samples=self.prm.hmm_min_training_samples
        )

    def get_analysis_summary(self) -> Optional[dict]:
        """Retorna resumo da última análise"""
        if self.last_analysis is None:
            return None

        return {
            'prob_hmm': self.last_analysis['Prob_HMM'],
            'lyapunov': self.last_analysis['Lyapunov_Score'],
            'hmm_state': self.last_analysis['hmm_analysis']['current_state'],
            'bar_count': self.bar_count
        }
