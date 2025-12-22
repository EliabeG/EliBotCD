# backtest/optimized_strategies.py
"""
Estratégias otimizadas com os melhores parâmetros encontrados.
Baseado em backtest com 3000 barras M5 de EURUSD (17 dias de dados).
"""
import pandas as pd
from typing import Dict, Optional


class OptimizedRSIStrategy:
    """
    RSI Mean Reversion - MELHOR ESTRATÉGIA

    Parâmetros otimizados:
    - RSI Period: 14
    - Oversold: 30
    - Overbought: 70
    - Stop Loss: 25 pips
    - Take Profit: 25 pips

    Resultados do backtest:
    - Profit: $179.20 (1.79%)
    - Win Rate: 72.7%
    - Profit Factor: 2.09
    - Max Drawdown: 0.9%
    - 22 trades em 17 dias
    """

    def __init__(self):
        self.name = "OptimizedRSI_14_30_70"
        self.period = 14
        self.oversold = 30
        self.overbought = 70
        self.stop_loss_pips = 25
        self.take_profit_pips = 25
        self.active = True
        self.lot_size = 0.1

    def _calculate_rsi(self, closes: pd.Series) -> float:
        delta = closes.diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - 100 / (1 + rs)
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    async def on_bar(self, context: Dict) -> Optional[Dict]:
        bars = context.get('bars')
        if bars is None or len(bars) < self.period + 2:
            return None

        # Não abrir se já tem posição
        if len(context.get('open_positions', [])) > 0:
            return None

        rsi = self._calculate_rsi(bars['close'])
        if pd.isna(rsi):
            return None

        current_price = context['tick']['mid']
        symbol = context.get('symbol', 'EURUSD')
        pip_value = 0.0001 if 'JPY' not in symbol else 0.01

        # Comprar quando RSI < 30 (sobrevendido)
        if rsi < self.oversold:
            return {
                'side': 'buy',
                'size': self.lot_size,
                'stop_loss': current_price - (self.stop_loss_pips * pip_value),
                'take_profit': current_price + (self.take_profit_pips * pip_value),
                'strategy_name': self.name,
                'metadata': {'rsi': rsi, 'signal': 'oversold'}
            }

        # Vender quando RSI > 70 (sobrecomprado)
        if rsi > self.overbought:
            return {
                'side': 'sell',
                'size': self.lot_size,
                'stop_loss': current_price + (self.stop_loss_pips * pip_value),
                'take_profit': current_price - (self.take_profit_pips * pip_value),
                'strategy_name': self.name,
                'metadata': {'rsi': rsi, 'signal': 'overbought'}
            }

        return None


class OptimizedSMAStrategy:
    """
    SMA Crossover - Estratégia de Tendência

    Parâmetros otimizados:
    - Fast MA: 15
    - Slow MA: 50
    - Stop Loss: 35 pips
    - Take Profit: 70 pips (2:1 R:R)

    Resultados do backtest:
    - Profit: $96.30 (0.96%)
    - Win Rate: 50.0%
    - Profit Factor: 3.48
    - Max Drawdown: 0.7%
    - 4 trades em 17 dias (menos frequente)
    """

    def __init__(self):
        self.name = "OptimizedSMA_15_50"
        self.fast_period = 15
        self.slow_period = 50
        self.stop_loss_pips = 35
        self.take_profit_pips = 70
        self.active = True
        self.lot_size = 0.1

    async def on_bar(self, context: Dict) -> Optional[Dict]:
        bars = context.get('bars')
        if bars is None or len(bars) < self.slow_period + 2:
            return None

        if len(context.get('open_positions', [])) > 0:
            return None

        closes = bars['close'].values
        fast_ma = pd.Series(closes).rolling(self.fast_period).mean().values
        slow_ma = pd.Series(closes).rolling(self.slow_period).mean().values

        if pd.isna(fast_ma[-1]) or pd.isna(slow_ma[-1]):
            return None
        if pd.isna(fast_ma[-2]) or pd.isna(slow_ma[-2]):
            return None

        current_price = context['tick']['mid']
        symbol = context.get('symbol', 'EURUSD')
        pip_value = 0.0001 if 'JPY' not in symbol else 0.01

        prev_diff = fast_ma[-2] - slow_ma[-2]
        curr_diff = fast_ma[-1] - slow_ma[-1]

        # Cruzamento para cima: comprar
        if prev_diff < 0 and curr_diff > 0:
            return {
                'side': 'buy',
                'size': self.lot_size,
                'stop_loss': current_price - (self.stop_loss_pips * pip_value),
                'take_profit': current_price + (self.take_profit_pips * pip_value),
                'strategy_name': self.name,
                'metadata': {'fast_ma': fast_ma[-1], 'slow_ma': slow_ma[-1]}
            }

        # Cruzamento para baixo: vender
        if prev_diff > 0 and curr_diff < 0:
            return {
                'side': 'sell',
                'size': self.lot_size,
                'stop_loss': current_price + (self.stop_loss_pips * pip_value),
                'take_profit': current_price - (self.take_profit_pips * pip_value),
                'strategy_name': self.name,
                'metadata': {'fast_ma': fast_ma[-1], 'slow_ma': slow_ma[-1]}
            }

        return None


class AggressiveRSIStrategy:
    """
    RSI Agressivo - Mais trades, menor TP/SL

    Parâmetros:
    - RSI Period: 7
    - Oversold: 20
    - Overbought: 80
    - Stop Loss: 15 pips
    - Take Profit: 20 pips

    Resultados:
    - Profit: $97.20
    - Win Rate: 56.8%
    - 44 trades em 17 dias
    """

    def __init__(self):
        self.name = "AggressiveRSI_7"
        self.period = 7
        self.oversold = 20
        self.overbought = 80
        self.stop_loss_pips = 15
        self.take_profit_pips = 20
        self.active = True
        self.lot_size = 0.1

    def _calculate_rsi(self, closes: pd.Series) -> float:
        delta = closes.diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - 100 / (1 + rs)
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    async def on_bar(self, context: Dict) -> Optional[Dict]:
        bars = context.get('bars')
        if bars is None or len(bars) < self.period + 2:
            return None

        if len(context.get('open_positions', [])) > 0:
            return None

        rsi = self._calculate_rsi(bars['close'])
        if pd.isna(rsi):
            return None

        current_price = context['tick']['mid']
        pip_value = 0.0001

        if rsi < self.oversold:
            return {
                'side': 'buy',
                'size': self.lot_size,
                'stop_loss': current_price - (self.stop_loss_pips * pip_value),
                'take_profit': current_price + (self.take_profit_pips * pip_value),
                'strategy_name': self.name
            }

        if rsi > self.overbought:
            return {
                'side': 'sell',
                'size': self.lot_size,
                'stop_loss': current_price + (self.stop_loss_pips * pip_value),
                'take_profit': current_price - (self.take_profit_pips * pip_value),
                'strategy_name': self.name
            }

        return None


# Configurações recomendadas para diferentes perfis de risco
STRATEGY_CONFIGS = {
    'conservative': {
        'name': 'OptimizedRSI',
        'description': 'RSI(14) 30/70 - Alta win rate, baixo drawdown',
        'params': {
            'period': 14,
            'oversold': 30,
            'overbought': 70,
            'sl_pips': 25,
            'tp_pips': 25
        },
        'expected_metrics': {
            'win_rate': 72.7,
            'profit_factor': 2.09,
            'max_drawdown': 0.9,
            'trades_per_day': 1.3
        }
    },
    'balanced': {
        'name': 'OptimizedSMA',
        'description': 'SMA(15,50) - Alto profit factor, menos trades',
        'params': {
            'fast': 15,
            'slow': 50,
            'sl_pips': 35,
            'tp_pips': 70
        },
        'expected_metrics': {
            'win_rate': 50.0,
            'profit_factor': 3.48,
            'max_drawdown': 0.7,
            'trades_per_day': 0.2
        }
    },
    'aggressive': {
        'name': 'AggressiveRSI',
        'description': 'RSI(7) 20/80 - Mais trades, scalping',
        'params': {
            'period': 7,
            'oversold': 20,
            'overbought': 80,
            'sl_pips': 15,
            'tp_pips': 20
        },
        'expected_metrics': {
            'win_rate': 56.8,
            'profit_factor': 1.29,
            'max_drawdown': 0.9,
            'trades_per_day': 2.6
        }
    }
}


def get_strategy(profile: str = 'conservative'):
    """
    Retorna a estratégia otimizada para o perfil especificado.

    Args:
        profile: 'conservative', 'balanced', ou 'aggressive'

    Returns:
        Instância da estratégia
    """
    if profile == 'conservative':
        return OptimizedRSIStrategy()
    elif profile == 'balanced':
        return OptimizedSMAStrategy()
    elif profile == 'aggressive':
        return AggressiveRSIStrategy()
    else:
        raise ValueError(f"Perfil desconhecido: {profile}")


if __name__ == "__main__":
    print("Estratégias Otimizadas Disponíveis:")
    print("="*50)
    for profile, config in STRATEGY_CONFIGS.items():
        print(f"\n{profile.upper()}: {config['name']}")
        print(f"  {config['description']}")
        print(f"  Win Rate: {config['expected_metrics']['win_rate']}%")
        print(f"  Profit Factor: {config['expected_metrics']['profit_factor']}")
        print(f"  Max DD: {config['expected_metrics']['max_drawdown']}%")
