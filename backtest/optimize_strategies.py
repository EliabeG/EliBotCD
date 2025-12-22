# backtest/optimize_strategies.py
"""
Otimizador de parâmetros para estratégias de trading.
Testa múltiplas combinações e encontra os melhores parâmetros.
"""
import asyncio
import sys
from pathlib import Path
from itertools import product
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import pandas as pd

# Adicionar diretório raiz ao path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import importlib.util

def _load_module_direct(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Carregar módulos
_hist_module = _load_module_direct('historical_data', f'{root_dir}/backtest/historical_data.py')
_engine_module = _load_module_direct('bar_backtest_engine', f'{root_dir}/backtest/bar_backtest_engine.py')

HistoricalDataClient = _hist_module.HistoricalDataClient
BarBacktestEngine = _engine_module.BarBacktestEngine

from utils.logger import setup_logger
from datetime import datetime, timedelta

logger = setup_logger("optimizer")


@dataclass
class OptimizationResult:
    """Resultado de uma otimização."""
    strategy_name: str
    params: Dict[str, Any]
    net_profit: float
    return_pct: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    score: float  # Score combinado para ranking


class SimpleMAStrategy:
    """Estratégia de cruzamento de MAs com parâmetros configuráveis."""

    def __init__(self, fast_period: int = 10, slow_period: int = 30,
                 stop_loss_pips: float = 20, take_profit_pips: float = 40):
        self.name = f"SMA_{fast_period}_{slow_period}"
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.active = True

    async def on_bar(self, context: dict):
        bars = context.get('bars')
        if bars is None or len(bars) < self.slow_period + 2:
            return None

        closes = bars['close'].values
        fast_ma = pd.Series(closes).rolling(self.fast_period).mean().values
        slow_ma = pd.Series(closes).rolling(self.slow_period).mean().values

        if pd.isna(fast_ma[-1]) or pd.isna(slow_ma[-1]):
            return None
        if pd.isna(fast_ma[-2]) or pd.isna(slow_ma[-2]):
            return None

        current_price = context['tick']['mid']
        pip_value = 0.0001 if 'JPY' not in context.get('symbol', 'EURUSD') else 0.01

        open_positions = context.get('open_positions', [])
        if len(open_positions) > 0:
            return None

        prev_diff = fast_ma[-2] - slow_ma[-2]
        curr_diff = fast_ma[-1] - slow_ma[-1]

        if prev_diff < 0 and curr_diff > 0:
            return {
                'side': 'buy',
                'size': 0.1,
                'stop_loss': current_price - (self.stop_loss_pips * pip_value),
                'take_profit': current_price + (self.take_profit_pips * pip_value),
                'strategy_name': self.name
            }

        if prev_diff > 0 and curr_diff < 0:
            return {
                'side': 'sell',
                'size': 0.1,
                'stop_loss': current_price + (self.stop_loss_pips * pip_value),
                'take_profit': current_price - (self.take_profit_pips * pip_value),
                'strategy_name': self.name
            }

        return None


class RSIMeanReversionStrategy:
    """Estratégia RSI com parâmetros configuráveis."""

    def __init__(self, rsi_period: int = 14,
                 oversold: float = 30, overbought: float = 70,
                 stop_loss_pips: float = 25, take_profit_pips: float = 25):
        self.name = f"RSI_{rsi_period}_{int(oversold)}_{int(overbought)}"
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.active = True

    def _calculate_rsi(self, closes: pd.Series) -> float:
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    async def on_bar(self, context: dict):
        bars = context.get('bars')
        if bars is None or len(bars) < self.rsi_period + 2:
            return None

        open_positions = context.get('open_positions', [])
        if len(open_positions) > 0:
            return None

        closes = bars['close']
        rsi = self._calculate_rsi(closes)

        current_price = context['tick']['mid']
        pip_value = 0.0001 if 'JPY' not in context.get('symbol', 'EURUSD') else 0.01

        if rsi < self.oversold:
            return {
                'side': 'buy',
                'size': 0.1,
                'stop_loss': current_price - (self.stop_loss_pips * pip_value),
                'take_profit': current_price + (self.take_profit_pips * pip_value),
                'strategy_name': self.name
            }

        if rsi > self.overbought:
            return {
                'side': 'sell',
                'size': 0.1,
                'stop_loss': current_price + (self.stop_loss_pips * pip_value),
                'take_profit': current_price - (self.take_profit_pips * pip_value),
                'strategy_name': self.name
            }

        return None


class BollingerBandStrategy:
    """Estratégia de Bollinger Bands - compra na banda inferior, vende na superior."""

    def __init__(self, period: int = 20, std_dev: float = 2.0,
                 stop_loss_pips: float = 20, take_profit_pips: float = 30):
        self.name = f"BB_{period}_{std_dev}"
        self.period = period
        self.std_dev = std_dev
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.active = True

    async def on_bar(self, context: dict):
        bars = context.get('bars')
        if bars is None or len(bars) < self.period + 2:
            return None

        open_positions = context.get('open_positions', [])
        if len(open_positions) > 0:
            return None

        closes = bars['close']
        sma = closes.rolling(self.period).mean()
        std = closes.rolling(self.period).std()

        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)

        current_price = context['tick']['mid']
        pip_value = 0.0001 if 'JPY' not in context.get('symbol', 'EURUSD') else 0.01

        # Preço tocou banda inferior - comprar
        if current_price <= lower_band.iloc[-1]:
            return {
                'side': 'buy',
                'size': 0.1,
                'stop_loss': current_price - (self.stop_loss_pips * pip_value),
                'take_profit': current_price + (self.take_profit_pips * pip_value),
                'strategy_name': self.name
            }

        # Preço tocou banda superior - vender
        if current_price >= upper_band.iloc[-1]:
            return {
                'side': 'sell',
                'size': 0.1,
                'stop_loss': current_price + (self.stop_loss_pips * pip_value),
                'take_profit': current_price - (self.take_profit_pips * pip_value),
                'strategy_name': self.name
            }

        return None


class MACDStrategy:
    """Estratégia baseada em MACD."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9,
                 stop_loss_pips: float = 25, take_profit_pips: float = 35):
        self.name = f"MACD_{fast}_{slow}_{signal}"
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.active = True

    async def on_bar(self, context: dict):
        bars = context.get('bars')
        if bars is None or len(bars) < self.slow + self.signal_period + 2:
            return None

        open_positions = context.get('open_positions', [])
        if len(open_positions) > 0:
            return None

        closes = bars['close']

        # Calcular MACD
        ema_fast = closes.ewm(span=self.fast, adjust=False).mean()
        ema_slow = closes.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        current_price = context['tick']['mid']
        pip_value = 0.0001 if 'JPY' not in context.get('symbol', 'EURUSD') else 0.01

        # Cruzamento MACD acima da linha de sinal - comprar
        if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
            return {
                'side': 'buy',
                'size': 0.1,
                'stop_loss': current_price - (self.stop_loss_pips * pip_value),
                'take_profit': current_price + (self.take_profit_pips * pip_value),
                'strategy_name': self.name
            }

        # Cruzamento MACD abaixo da linha de sinal - vender
        if macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
            return {
                'side': 'sell',
                'size': 0.1,
                'stop_loss': current_price + (self.stop_loss_pips * pip_value),
                'take_profit': current_price - (self.take_profit_pips * pip_value),
                'strategy_name': self.name
            }

        return None


def calculate_score(result) -> float:
    """
    Calcula score combinado para ranking.
    Prioriza: Profit Factor, Sharpe, Win Rate, minimiza Drawdown.
    """
    m = result.metrics

    # Penalizar se poucos trades
    if m.total_trades < 5:
        return -1000

    # Score baseado em múltiplos fatores
    score = 0

    # Profit factor (peso alto)
    if m.profit_factor > 0:
        score += min(m.profit_factor, 5) * 20  # Max 100 pontos

    # Sharpe ratio
    score += max(min(m.sharpe_ratio, 3), -3) * 15  # -45 a +45 pontos

    # Win rate
    score += m.win_rate * 0.5  # 0 a 50 pontos

    # Return
    score += min(m.return_pct, 20) * 2  # Max 40 pontos

    # Penalidade por drawdown
    score -= m.max_drawdown_pct * 2  # Cada % de DD reduz 2 pontos

    # Penalidade por poucos trades
    if m.total_trades < 10:
        score -= (10 - m.total_trades) * 5

    return score


async def download_data(symbol: str, timeframe: str, bars_count: int) -> pd.DataFrame:
    """Baixa dados históricos."""
    logger.info(f"Baixando {bars_count} barras de {symbol} {timeframe}...")

    client = HistoricalDataClient()

    if not await client.connect():
        logger.error("Falha ao conectar à API")
        return pd.DataFrame()

    try:
        all_bars = []
        remaining = bars_count
        end_time = None

        while remaining > 0:
            batch_size = min(remaining, 1000)
            df = await client.get_bars(symbol, timeframe, count=batch_size, end_time=end_time)

            if df.empty:
                break

            all_bars.append(df)
            remaining -= len(df)

            if 'timestamp' in df.columns:
                period_minutes = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
                                 'H1': 60, 'H4': 240, 'D1': 1440}.get(timeframe, 5)
                end_time = df['timestamp'].min() - timedelta(minutes=period_minutes)

            await asyncio.sleep(0.2)

        if all_bars:
            result = pd.concat(all_bars, ignore_index=True)
            result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            return result

        return pd.DataFrame()
    finally:
        await client.disconnect()


async def optimize_sma_strategy(bars: pd.DataFrame, symbol: str, timeframe: str) -> List[OptimizationResult]:
    """Otimiza parâmetros da estratégia SMA."""
    print("\n" + "="*60)
    print("OTIMIZANDO: Simple Moving Average Crossover")
    print("="*60)

    # Parâmetros a testar
    fast_periods = [5, 8, 10, 12, 15]
    slow_periods = [20, 25, 30, 40, 50]
    stop_losses = [15, 20, 25, 30]
    take_profits = [20, 30, 40, 50, 60]

    results = []
    total_combinations = len(fast_periods) * len(slow_periods) * len(stop_losses) * len(take_profits)
    tested = 0

    for fast in fast_periods:
        for slow in slow_periods:
            if fast >= slow:
                continue  # Fast deve ser menor que slow

            for sl in stop_losses:
                for tp in take_profits:
                    tested += 1
                    strategy = SimpleMAStrategy(
                        fast_period=fast,
                        slow_period=slow,
                        stop_loss_pips=sl,
                        take_profit_pips=tp
                    )

                    engine = BarBacktestEngine(initial_balance=10000)
                    result = await engine.run(strategy, bars, symbol, timeframe)

                    score = calculate_score(result)

                    opt_result = OptimizationResult(
                        strategy_name=strategy.name,
                        params={'fast': fast, 'slow': slow, 'sl': sl, 'tp': tp},
                        net_profit=result.metrics.net_profit,
                        return_pct=result.metrics.return_pct,
                        win_rate=result.metrics.win_rate,
                        profit_factor=result.metrics.profit_factor,
                        sharpe_ratio=result.metrics.sharpe_ratio,
                        max_drawdown=result.metrics.max_drawdown_pct,
                        total_trades=result.metrics.total_trades,
                        score=score
                    )
                    results.append(opt_result)

                    if tested % 20 == 0:
                        print(f"  Testados: {tested} combinações...")

    print(f"  Total testado: {tested} combinações")
    return sorted(results, key=lambda x: x.score, reverse=True)


async def optimize_rsi_strategy(bars: pd.DataFrame, symbol: str, timeframe: str) -> List[OptimizationResult]:
    """Otimiza parâmetros da estratégia RSI."""
    print("\n" + "="*60)
    print("OTIMIZANDO: RSI Mean Reversion")
    print("="*60)

    # Parâmetros a testar
    rsi_periods = [7, 10, 14, 21]
    oversolds = [20, 25, 30, 35]
    overboughts = [65, 70, 75, 80]
    stop_losses = [15, 20, 25, 30, 35]
    take_profits = [15, 20, 25, 30, 40]

    results = []
    tested = 0

    for period in rsi_periods:
        for oversold in oversolds:
            for overbought in overboughts:
                if oversold >= overbought - 20:
                    continue  # Precisa de gap mínimo

                for sl in stop_losses:
                    for tp in take_profits:
                        tested += 1
                        strategy = RSIMeanReversionStrategy(
                            rsi_period=period,
                            oversold=oversold,
                            overbought=overbought,
                            stop_loss_pips=sl,
                            take_profit_pips=tp
                        )

                        engine = BarBacktestEngine(initial_balance=10000)
                        result = await engine.run(strategy, bars, symbol, timeframe)

                        score = calculate_score(result)

                        opt_result = OptimizationResult(
                            strategy_name=strategy.name,
                            params={'period': period, 'oversold': oversold,
                                   'overbought': overbought, 'sl': sl, 'tp': tp},
                            net_profit=result.metrics.net_profit,
                            return_pct=result.metrics.return_pct,
                            win_rate=result.metrics.win_rate,
                            profit_factor=result.metrics.profit_factor,
                            sharpe_ratio=result.metrics.sharpe_ratio,
                            max_drawdown=result.metrics.max_drawdown_pct,
                            total_trades=result.metrics.total_trades,
                            score=score
                        )
                        results.append(opt_result)

                        if tested % 50 == 0:
                            print(f"  Testados: {tested} combinações...")

    print(f"  Total testado: {tested} combinações")
    return sorted(results, key=lambda x: x.score, reverse=True)


async def optimize_bollinger_strategy(bars: pd.DataFrame, symbol: str, timeframe: str) -> List[OptimizationResult]:
    """Otimiza parâmetros da estratégia Bollinger Bands."""
    print("\n" + "="*60)
    print("OTIMIZANDO: Bollinger Bands")
    print("="*60)

    periods = [15, 20, 25, 30]
    std_devs = [1.5, 2.0, 2.5, 3.0]
    stop_losses = [15, 20, 25, 30]
    take_profits = [20, 30, 40, 50]

    results = []
    tested = 0

    for period in periods:
        for std in std_devs:
            for sl in stop_losses:
                for tp in take_profits:
                    tested += 1
                    strategy = BollingerBandStrategy(
                        period=period,
                        std_dev=std,
                        stop_loss_pips=sl,
                        take_profit_pips=tp
                    )

                    engine = BarBacktestEngine(initial_balance=10000)
                    result = await engine.run(strategy, bars, symbol, timeframe)

                    score = calculate_score(result)

                    opt_result = OptimizationResult(
                        strategy_name=strategy.name,
                        params={'period': period, 'std': std, 'sl': sl, 'tp': tp},
                        net_profit=result.metrics.net_profit,
                        return_pct=result.metrics.return_pct,
                        win_rate=result.metrics.win_rate,
                        profit_factor=result.metrics.profit_factor,
                        sharpe_ratio=result.metrics.sharpe_ratio,
                        max_drawdown=result.metrics.max_drawdown_pct,
                        total_trades=result.metrics.total_trades,
                        score=score
                    )
                    results.append(opt_result)

                    if tested % 20 == 0:
                        print(f"  Testados: {tested} combinações...")

    print(f"  Total testado: {tested} combinações")
    return sorted(results, key=lambda x: x.score, reverse=True)


async def optimize_macd_strategy(bars: pd.DataFrame, symbol: str, timeframe: str) -> List[OptimizationResult]:
    """Otimiza parâmetros da estratégia MACD."""
    print("\n" + "="*60)
    print("OTIMIZANDO: MACD")
    print("="*60)

    fasts = [8, 12, 16]
    slows = [21, 26, 30]
    signals = [7, 9, 12]
    stop_losses = [20, 25, 30, 35]
    take_profits = [25, 35, 45, 55]

    results = []
    tested = 0

    for fast in fasts:
        for slow in slows:
            if fast >= slow:
                continue

            for signal in signals:
                for sl in stop_losses:
                    for tp in take_profits:
                        tested += 1
                        strategy = MACDStrategy(
                            fast=fast,
                            slow=slow,
                            signal=signal,
                            stop_loss_pips=sl,
                            take_profit_pips=tp
                        )

                        engine = BarBacktestEngine(initial_balance=10000)
                        result = await engine.run(strategy, bars, symbol, timeframe)

                        score = calculate_score(result)

                        opt_result = OptimizationResult(
                            strategy_name=strategy.name,
                            params={'fast': fast, 'slow': slow, 'signal': signal, 'sl': sl, 'tp': tp},
                            net_profit=result.metrics.net_profit,
                            return_pct=result.metrics.return_pct,
                            win_rate=result.metrics.win_rate,
                            profit_factor=result.metrics.profit_factor,
                            sharpe_ratio=result.metrics.sharpe_ratio,
                            max_drawdown=result.metrics.max_drawdown_pct,
                            total_trades=result.metrics.total_trades,
                            score=score
                        )
                        results.append(opt_result)

                        if tested % 20 == 0:
                            print(f"  Testados: {tested} combinações...")

    print(f"  Total testado: {tested} combinações")
    return sorted(results, key=lambda x: x.score, reverse=True)


def print_top_results(results: List[OptimizationResult], top_n: int = 5):
    """Imprime os melhores resultados."""
    print(f"\n{'='*80}")
    print(f"TOP {top_n} MELHORES CONFIGURAÇÕES")
    print('='*80)

    for i, r in enumerate(results[:top_n], 1):
        print(f"\n#{i} - Score: {r.score:.1f}")
        print(f"   Parâmetros: {r.params}")
        print(f"   Net Profit: ${r.net_profit:.2f} ({r.return_pct:.2f}%)")
        print(f"   Win Rate: {r.win_rate:.1f}% | Profit Factor: {r.profit_factor:.2f}")
        print(f"   Sharpe: {r.sharpe_ratio:.2f} | Max DD: {r.max_drawdown:.2f}%")
        print(f"   Trades: {r.total_trades}")


async def main():
    """Função principal de otimização."""
    print("\n" + "="*80)
    print("OTIMIZADOR DE ESTRATÉGIAS - Trading Bot FX")
    print("="*80)

    # Configuração
    symbol = "EURUSD"
    timeframe = "M5"
    bars_count = 5000  # Mais dados para otimização

    # Baixar dados
    print(f"\nBaixando {bars_count} barras de {symbol} {timeframe}...")
    bars = await download_data(symbol, timeframe, bars_count)

    if bars.empty:
        logger.error("Sem dados para otimização")
        return

    print(f"Dados carregados: {len(bars)} barras")
    print(f"Período: {bars['timestamp'].iloc[0]} a {bars['timestamp'].iloc[-1]}")

    # Otimizar cada estratégia
    all_results = {}

    # 1. SMA
    sma_results = await optimize_sma_strategy(bars, symbol, timeframe)
    all_results['SMA'] = sma_results
    print_top_results(sma_results, 3)

    # 2. RSI
    rsi_results = await optimize_rsi_strategy(bars, symbol, timeframe)
    all_results['RSI'] = rsi_results
    print_top_results(rsi_results, 3)

    # 3. Bollinger Bands
    bb_results = await optimize_bollinger_strategy(bars, symbol, timeframe)
    all_results['Bollinger'] = bb_results
    print_top_results(bb_results, 3)

    # 4. MACD
    macd_results = await optimize_macd_strategy(bars, symbol, timeframe)
    all_results['MACD'] = macd_results
    print_top_results(macd_results, 3)

    # Resumo final - melhor de cada
    print("\n" + "="*80)
    print("RESUMO: MELHOR CONFIGURAÇÃO DE CADA ESTRATÉGIA")
    print("="*80)

    best_overall = []
    for name, results in all_results.items():
        if results:
            best = results[0]
            best_overall.append((name, best))
            print(f"\n{name}:")
            print(f"  Params: {best.params}")
            print(f"  Profit: ${best.net_profit:.2f} | Win Rate: {best.win_rate:.1f}%")
            print(f"  PF: {best.profit_factor:.2f} | Sharpe: {best.sharpe_ratio:.2f}")

    # Ranking geral
    best_overall.sort(key=lambda x: x[1].score, reverse=True)

    print("\n" + "="*80)
    print("RANKING GERAL (por Score)")
    print("="*80)
    for i, (name, r) in enumerate(best_overall, 1):
        print(f"{i}. {name}: Score={r.score:.1f} | Profit=${r.net_profit:.2f} | WR={r.win_rate:.1f}%")

    # Salvar resultados
    results_dir = Path("data/optimization_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    for name, results in all_results.items():
        if results:
            df = pd.DataFrame([{
                'params': str(r.params),
                'net_profit': r.net_profit,
                'return_pct': r.return_pct,
                'win_rate': r.win_rate,
                'profit_factor': r.profit_factor,
                'sharpe': r.sharpe_ratio,
                'max_dd': r.max_drawdown,
                'trades': r.total_trades,
                'score': r.score
            } for r in results])
            filename = results_dir / f"{name.lower()}_optimization_{timestamp_str}.csv"
            df.to_csv(filename, index=False)
            print(f"\nResultados {name} salvos em: {filename}")

    print("\n" + "="*80)
    print("OTIMIZAÇÃO CONCLUÍDA!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
