# strategies/optimized/optimized_sma.py
"""
Estratégia SMA Crossover Otimizada - Melhor Profit Factor no backtest.

Resultados do backtest (17 dias, 3000 barras M5):
- Win Rate: 50.0%
- Profit Factor: 3.51
- Max Drawdown: 0.70%
- Net Profit: $96.70 (0.97%)
"""
import numpy as np
import talib
from typing import Dict, Optional, Any
from datetime import datetime, timezone

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from config.settings import CONFIG


class OptimizedSMAStrategy(BaseStrategy):
    """
    Estratégia de cruzamento de médias móveis otimizada.

    Parâmetros otimizados:
    - Fast MA: 15 períodos
    - Slow MA: 50 períodos
    - Stop Loss: 35 pips
    - Take Profit: 70 pips (2:1 R:R)
    """

    def __init__(self):
        super().__init__("OptimizedSMA_TrendFollow")
        self.suitable_regimes = [
            MarketRegime.TREND,
            MarketRegime.HIGH_VOLATILITY
        ]
        self.min_time_between_signals_sec = 600  # 10 minutos (estratégia de tendência)

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Moving Average Parameters (Otimizados)
            'fast_period': 15,
            'slow_period': 50,
            'ma_type': 'SMA',  # Pode ser 'SMA' ou 'EMA'

            # Risk Management (Otimizados - 2:1 R:R)
            'stop_loss_pips': 35,
            'take_profit_pips': 70,

            # Filtros de tendência
            'use_trend_filter': True,
            'trend_ma_period': 100,  # MA longa para confirmar tendência geral

            # Filtros de volatilidade
            'use_atr_filter': True,
            'atr_period': 14,
            'min_atr_pips': 8,
            'max_atr_pips': 60,

            # Confirmação de momentum
            'require_momentum_confirmation': True,
            'momentum_bars': 3,  # Barras de confirmação após cruzamento

            # Position sizing
            'confidence_base': 0.70,
            'confidence_boost_trend_aligned': 0.15,
        }

    async def calculate_indicators(self, market_context: Dict[str, Any]) -> None:
        """Calcula médias móveis e indicadores de suporte."""
        recent_ticks = market_context.get('recent_ticks', [])

        min_data = self.parameters['slow_period'] + 10
        if self.parameters['use_trend_filter']:
            min_data = max(min_data, self.parameters['trend_ma_period'] + 5)

        if not recent_ticks or len(recent_ticks) < min_data:
            self.current_indicators = {}
            return

        close_prices = self._get_prices_from_context(market_context, 'mid')
        high_prices = self._get_prices_from_context(market_context, 'high')
        low_prices = self._get_prices_from_context(market_context, 'low')

        if len(close_prices) < min_data:
            self.current_indicators = {}
            return

        # Calcular médias móveis
        if self.parameters['ma_type'] == 'EMA':
            fast_ma = talib.EMA(close_prices, timeperiod=self.parameters['fast_period'])
            slow_ma = talib.EMA(close_prices, timeperiod=self.parameters['slow_period'])
        else:
            fast_ma = talib.SMA(close_prices, timeperiod=self.parameters['fast_period'])
            slow_ma = talib.SMA(close_prices, timeperiod=self.parameters['slow_period'])

        current_fast = fast_ma[-1] if not np.isnan(fast_ma[-1]) else None
        current_slow = slow_ma[-1] if not np.isnan(slow_ma[-1]) else None
        prev_fast = fast_ma[-2] if len(fast_ma) > 1 and not np.isnan(fast_ma[-2]) else None
        prev_slow = slow_ma[-2] if len(slow_ma) > 1 and not np.isnan(slow_ma[-2]) else None

        if current_fast is None or current_slow is None or prev_fast is None or prev_slow is None:
            self.current_indicators = {}
            return

        # MA de tendência longa
        trend_ma_value = None
        trend_direction = None
        if self.parameters['use_trend_filter']:
            trend_ma = talib.SMA(close_prices, timeperiod=self.parameters['trend_ma_period'])
            trend_ma_value = trend_ma[-1] if not np.isnan(trend_ma[-1]) else None
            if trend_ma_value is not None:
                trend_direction = 'up' if close_prices[-1] > trend_ma_value else 'down'

        # ATR para filtro de volatilidade
        atr_value = 0.0
        if self.parameters['use_atr_filter'] and len(high_prices) >= self.parameters['atr_period']:
            atr_values = talib.ATR(high_prices, low_prices, close_prices,
                                   timeperiod=self.parameters['atr_period'])
            atr_value = atr_values[-1] if not np.isnan(atr_values[-1]) else 0.0

        symbol = market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL)
        pip_size = 0.0001 if 'JPY' not in symbol.upper() else 0.01
        atr_pips = atr_value / pip_size if atr_value > 0 else 0.0

        # Detectar cruzamentos
        prev_diff = prev_fast - prev_slow
        curr_diff = current_fast - current_slow

        bullish_cross = prev_diff < 0 and curr_diff > 0
        bearish_cross = prev_diff > 0 and curr_diff < 0

        # Calcular distância entre MAs (força do sinal)
        ma_spread = abs(curr_diff)
        ma_spread_pips = ma_spread / pip_size

        # Momentum: verificar se o cruzamento tem força
        momentum_confirmed = True
        if self.parameters['require_momentum_confirmation']:
            bars_to_check = self.parameters['momentum_bars']
            if len(fast_ma) > bars_to_check:
                if bullish_cross:
                    # Fast MA deve estar subindo
                    momentum_confirmed = fast_ma[-1] > fast_ma[-bars_to_check]
                elif bearish_cross:
                    # Fast MA deve estar caindo
                    momentum_confirmed = fast_ma[-1] < fast_ma[-bars_to_check]

        self.current_indicators = {
            'fast_ma': current_fast,
            'slow_ma': current_slow,
            'prev_fast_ma': prev_fast,
            'prev_slow_ma': prev_slow,
            'trend_ma': trend_ma_value,
            'trend_direction': trend_direction,
            'atr_pips': atr_pips,
            'bullish_cross': bullish_cross,
            'bearish_cross': bearish_cross,
            'ma_spread_pips': ma_spread_pips,
            'momentum_confirmed': momentum_confirmed,
            'current_price': close_prices[-1],
            'pip_size': pip_size
        }

    async def generate_signal(self, market_context: Dict[str, Any]) -> Optional[Signal]:
        """Gera sinal de entrada baseado em cruzamento de MAs."""
        if not self.current_indicators:
            return None

        ind = self.current_indicators

        # Verificar volatilidade
        if self.parameters['use_atr_filter']:
            if ind['atr_pips'] < self.parameters['min_atr_pips']:
                self.logger.debug(f"ATR muito baixo: {ind['atr_pips']:.1f} pips")
                return None
            if ind['atr_pips'] > self.parameters['max_atr_pips']:
                self.logger.debug(f"ATR muito alto: {ind['atr_pips']:.1f} pips")
                return None

        # Verificar momentum
        if self.parameters['require_momentum_confirmation'] and not ind['momentum_confirmed']:
            return None

        current_price = ind['current_price']
        pip_size = ind['pip_size']
        sl_pips = self.parameters['stop_loss_pips']
        tp_pips = self.parameters['take_profit_pips']

        # Sinal de COMPRA (cruzamento bullish)
        if ind['bullish_cross']:
            confidence = self.parameters['confidence_base']

            # Boost se alinhado com tendência de longo prazo
            if self.parameters['use_trend_filter'] and ind['trend_direction'] == 'up':
                confidence += self.parameters['confidence_boost_trend_aligned']

            stop_loss = current_price - (sl_pips * pip_size)
            take_profit = current_price + (tp_pips * pip_size)

            return Signal(
                strategy_name=self.name,
                side='buy',
                confidence=min(confidence, 0.95),
                symbol=market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"Bullish MA Cross (Fast={ind['fast_ma']:.5f} > Slow={ind['slow_ma']:.5f})",
                metadata={
                    'fast_ma': ind['fast_ma'],
                    'slow_ma': ind['slow_ma'],
                    'ma_spread_pips': ind['ma_spread_pips'],
                    'trend_aligned': ind['trend_direction'] == 'up'
                }
            )

        # Sinal de VENDA (cruzamento bearish)
        if ind['bearish_cross']:
            confidence = self.parameters['confidence_base']

            if self.parameters['use_trend_filter'] and ind['trend_direction'] == 'down':
                confidence += self.parameters['confidence_boost_trend_aligned']

            stop_loss = current_price + (sl_pips * pip_size)
            take_profit = current_price - (tp_pips * pip_size)

            return Signal(
                strategy_name=self.name,
                side='sell',
                confidence=min(confidence, 0.95),
                symbol=market_context.get('tick', {}).get('symbol', CONFIG.SYMBOL),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"Bearish MA Cross (Fast={ind['fast_ma']:.5f} < Slow={ind['slow_ma']:.5f})",
                metadata={
                    'fast_ma': ind['fast_ma'],
                    'slow_ma': ind['slow_ma'],
                    'ma_spread_pips': ind['ma_spread_pips'],
                    'trend_aligned': ind['trend_direction'] == 'down'
                }
            )

        return None

    async def evaluate_exit_conditions(self, open_position: Position,
                                       market_context: Dict[str, Any]) -> Optional[ExitSignal]:
        """Avalia condições de saída para posições abertas."""
        if not self.current_indicators:
            return None

        ind = self.current_indicators
        current_price = ind.get('current_price')

        if current_price is None:
            return None

        # Saída no cruzamento oposto
        if open_position.side.lower() == 'buy':
            # Sair se houver cruzamento bearish
            if ind['bearish_cross']:
                return ExitSignal(
                    position_id_to_close=open_position.id,
                    reason="Bearish MA crossover",
                    exit_price=current_price
                )

        elif open_position.side.lower() == 'sell':
            # Sair se houver cruzamento bullish
            if ind['bullish_cross']:
                return ExitSignal(
                    position_id_to_close=open_position.id,
                    reason="Bullish MA crossover",
                    exit_price=current_price
                )

        return None
