# strategies/optimized/optimized_rsi.py
"""
Estratégia RSI Otimizada - Melhor performance no backtest.

Resultados do backtest (17 dias, 3000 barras M5):
- Win Rate: 72.7%
- Profit Factor: 2.09
- Max Drawdown: 0.87%
- Net Profit: $179.50 (1.79%)
"""
import numpy as np
import talib
from typing import Dict, Optional, Any
from datetime import datetime, timezone

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from config.settings import CONFIG


class OptimizedRSIStrategy(BaseStrategy):
    """
    Estratégia de reversão à média baseada em RSI otimizado.

    Parâmetros otimizados:
    - RSI Period: 14
    - Oversold: 30
    - Overbought: 70
    - Stop Loss: 25 pips
    - Take Profit: 25 pips (1:1 R:R)
    """

    def __init__(self):
        super().__init__("OptimizedRSI_Conservative")
        self.suitable_regimes = [
            MarketRegime.RANGE,
            MarketRegime.LOW_VOLUME,
            MarketRegime.TREND  # Funciona bem em pullbacks de tendência
        ]
        self.min_time_between_signals_sec = 300  # 5 minutos entre sinais

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # RSI Parameters (Otimizados)
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,

            # Risk Management (Otimizados)
            'stop_loss_pips': 25,
            'take_profit_pips': 25,

            # Filtros adicionais
            'use_atr_filter': True,
            'atr_period': 14,
            'min_atr_pips': 5,  # Volatilidade mínima para entrar
            'max_atr_pips': 50,  # Volatilidade máxima

            # Confirmações
            'require_price_momentum': True,
            'momentum_lookback': 3,  # Barras para verificar momentum
            'min_momentum_pips': 2,  # Movimento mínimo antes do sinal

            # Position sizing
            'confidence_base': 0.75,  # Confiança base do sinal
            'confidence_boost_extreme': 0.10,  # Boost para RSI extremo
        }

    async def calculate_indicators(self, market_context: Dict[str, Any]) -> None:
        """Calcula RSI e indicadores de suporte."""
        recent_ticks = market_context.get('recent_ticks', [])

        min_data = self.parameters['rsi_period'] + 10

        if not recent_ticks or len(recent_ticks) < min_data:
            self.current_indicators = {}
            return

        # Extrair preços
        close_prices = self._get_prices_from_context(market_context, 'mid')
        high_prices = self._get_prices_from_context(market_context, 'high')
        low_prices = self._get_prices_from_context(market_context, 'low')

        if len(close_prices) < min_data:
            self.current_indicators = {}
            return

        # Calcular RSI
        rsi_values = talib.RSI(close_prices, timeperiod=self.parameters['rsi_period'])
        current_rsi = rsi_values[-1] if not np.isnan(rsi_values[-1]) else 50.0
        prev_rsi = rsi_values[-2] if len(rsi_values) > 1 and not np.isnan(rsi_values[-2]) else 50.0

        # Calcular ATR
        atr_value = 0.0
        if self.parameters['use_atr_filter'] and len(high_prices) >= self.parameters['atr_period']:
            atr_values = talib.ATR(high_prices, low_prices, close_prices,
                                   timeperiod=self.parameters['atr_period'])
            atr_value = atr_values[-1] if not np.isnan(atr_values[-1]) else 0.0

        # Pip size
        symbol = self._get_symbol_from_context(market_context)
        pip_size = 0.0001 if 'JPY' not in symbol.upper() else 0.01
        atr_pips = atr_value / pip_size if atr_value > 0 else 0.0

        # Momentum (variação de preço nas últimas N barras)
        momentum_pips = 0.0
        lookback = self.parameters['momentum_lookback']
        if len(close_prices) >= lookback + 1:
            price_change = close_prices[-1] - close_prices[-lookback-1]
            momentum_pips = abs(price_change) / pip_size

        # Detectar condições extremas de RSI
        is_oversold = current_rsi < self.parameters['rsi_oversold']
        is_overbought = current_rsi > self.parameters['rsi_overbought']
        is_extreme_oversold = current_rsi < 20
        is_extreme_overbought = current_rsi > 80

        # Divergência de RSI (simplificada)
        rsi_rising = current_rsi > prev_rsi
        price_rising = close_prices[-1] > close_prices[-2] if len(close_prices) > 1 else False
        has_bullish_divergence = is_oversold and rsi_rising and not price_rising
        has_bearish_divergence = is_overbought and not rsi_rising and price_rising

        self.current_indicators = {
            'rsi': current_rsi,
            'prev_rsi': prev_rsi,
            'atr_pips': atr_pips,
            'momentum_pips': momentum_pips,
            'is_oversold': is_oversold,
            'is_overbought': is_overbought,
            'is_extreme_oversold': is_extreme_oversold,
            'is_extreme_overbought': is_extreme_overbought,
            'has_bullish_divergence': has_bullish_divergence,
            'has_bearish_divergence': has_bearish_divergence,
            'current_price': close_prices[-1],
            'pip_size': pip_size
        }

    async def generate_signal(self, market_context: Dict[str, Any]) -> Optional[Signal]:
        """Gera sinal de entrada baseado em RSI."""
        if not self.current_indicators:
            return None

        ind = self.current_indicators

        # Verificar volatilidade (filtro ATR)
        if self.parameters['use_atr_filter']:
            if ind['atr_pips'] < self.parameters['min_atr_pips']:
                self.logger.debug(f"ATR muito baixo: {ind['atr_pips']:.1f} pips")
                return None
            if ind['atr_pips'] > self.parameters['max_atr_pips']:
                self.logger.debug(f"ATR muito alto: {ind['atr_pips']:.1f} pips")
                return None

        # Verificar momentum
        if self.parameters['require_price_momentum']:
            if ind['momentum_pips'] < self.parameters['min_momentum_pips']:
                return None

        current_price = ind['current_price']
        pip_size = ind['pip_size']
        sl_pips = self.parameters['stop_loss_pips']
        tp_pips = self.parameters['take_profit_pips']

        # Sinal de COMPRA (RSI oversold)
        if ind['is_oversold']:
            confidence = self.parameters['confidence_base']
            if ind['is_extreme_oversold']:
                confidence += self.parameters['confidence_boost_extreme']
            if ind['has_bullish_divergence']:
                confidence += 0.05

            stop_loss = current_price - (sl_pips * pip_size)
            take_profit = current_price + (tp_pips * pip_size)

            return Signal(
                strategy_name=self.name,
                side='buy',
                confidence=min(confidence, 0.95),
                symbol=self._get_symbol_from_context(market_context),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"RSI Oversold ({ind['rsi']:.1f})",
                metadata={
                    'rsi': ind['rsi'],
                    'atr_pips': ind['atr_pips'],
                    'divergence': ind['has_bullish_divergence']
                }
            )

        # Sinal de VENDA (RSI overbought)
        if ind['is_overbought']:
            confidence = self.parameters['confidence_base']
            if ind['is_extreme_overbought']:
                confidence += self.parameters['confidence_boost_extreme']
            if ind['has_bearish_divergence']:
                confidence += 0.05

            stop_loss = current_price + (sl_pips * pip_size)
            take_profit = current_price - (tp_pips * pip_size)

            return Signal(
                strategy_name=self.name,
                side='sell',
                confidence=min(confidence, 0.95),
                symbol=self._get_symbol_from_context(market_context),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"RSI Overbought ({ind['rsi']:.1f})",
                metadata={
                    'rsi': ind['rsi'],
                    'atr_pips': ind['atr_pips'],
                    'divergence': ind['has_bearish_divergence']
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

        rsi = ind.get('rsi', 50)

        # Saída quando RSI volta ao neutro (50)
        # Para compra: sair quando RSI cruza acima de 50
        # Para venda: sair quando RSI cruza abaixo de 50

        if open_position.side.lower() == 'buy':
            # Posição de compra: sair se RSI > 50 (voltou ao neutro/bullish)
            if rsi > 55:  # Um pouco acima de 50 para evitar whipsaws
                return ExitSignal(
                    position_id_to_close=open_position.id,
                    reason=f"RSI recovered to neutral ({rsi:.1f})",
                    exit_price=current_price
                )

        elif open_position.side.lower() == 'sell':
            # Posição de venda: sair se RSI < 45
            if rsi < 45:
                return ExitSignal(
                    position_id_to_close=open_position.id,
                    reason=f"RSI recovered to neutral ({rsi:.1f})",
                    exit_price=current_price
                )

        return None
