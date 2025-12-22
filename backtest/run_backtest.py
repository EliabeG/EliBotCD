# backtest/run_backtest.py
"""
Script para executar backtest com dados reais da API.
"""
import asyncio
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import pandas as pd
from datetime import datetime, timezone

# Importar diretamente dos arquivos para evitar circular imports
import importlib.util

def _load_module_direct(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Carregar módulos diretamente
_hist_module = _load_module_direct('historical_data', f'{root_dir}/backtest/historical_data.py')
_engine_module = _load_module_direct('bar_backtest_engine', f'{root_dir}/backtest/bar_backtest_engine.py')

HistoricalDataClient = _hist_module.HistoricalDataClient
BarBacktestEngine = _engine_module.BarBacktestEngine

from utils.logger import setup_logger

logger = setup_logger("backtest_runner")


class SimpleMAStrategy:
    """
    Estratégia simples de cruzamento de médias móveis para teste.
    Compra quando MA rápida cruza acima da MA lenta.
    Vende quando MA rápida cruza abaixo da MA lenta.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 30,
                 stop_loss_pips: float = 20, take_profit_pips: float = 40):
        self.name = f"SMA_{fast_period}_{slow_period}"
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.active = True
        self.last_position = None  # 'long', 'short', ou None

    async def on_bar(self, context: dict):
        """Gera sinal baseado em cruzamento de MAs."""
        bars = context.get('bars')
        if bars is None or len(bars) < self.slow_period + 2:
            return None

        # Calcular médias móveis
        closes = bars['close'].values
        fast_ma = pd.Series(closes).rolling(self.fast_period).mean().values
        slow_ma = pd.Series(closes).rolling(self.slow_period).mean().values

        # Precisamos de pelo menos 2 valores para detectar cruzamento
        if pd.isna(fast_ma[-1]) or pd.isna(slow_ma[-1]):
            return None
        if pd.isna(fast_ma[-2]) or pd.isna(slow_ma[-2]):
            return None

        current_price = context['tick']['mid']
        pip_value = 0.0001 if 'JPY' not in context.get('symbol', 'EURUSD') else 0.01

        # Verificar cruzamentos
        prev_diff = fast_ma[-2] - slow_ma[-2]
        curr_diff = fast_ma[-1] - slow_ma[-1]

        # Verificar se já tem posição aberta
        open_positions = context.get('open_positions', [])
        has_position = len(open_positions) > 0

        if has_position:
            return None

        # Cruzamento para cima: sinal de compra
        if prev_diff < 0 and curr_diff > 0:
            return {
                'side': 'buy',
                'size': 0.1,
                'stop_loss': current_price - (self.stop_loss_pips * pip_value),
                'take_profit': current_price + (self.take_profit_pips * pip_value),
                'strategy_name': self.name
            }

        # Cruzamento para baixo: sinal de venda
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
    """
    Estratégia de reversão à média baseada em RSI.
    Compra quando RSI está sobrevendido e venda quando sobrecomprado.
    """

    def __init__(self, rsi_period: int = 14,
                 oversold: float = 30, overbought: float = 70,
                 stop_loss_pips: float = 25, take_profit_pips: float = 25):
        self.name = f"RSI_MeanReversion_{rsi_period}"
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.active = True

    def _calculate_rsi(self, closes: pd.Series) -> float:
        """Calcula RSI."""
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    async def on_bar(self, context: dict):
        """Gera sinal baseado em RSI."""
        bars = context.get('bars')
        if bars is None or len(bars) < self.rsi_period + 2:
            return None

        # Verificar se já tem posição
        open_positions = context.get('open_positions', [])
        if len(open_positions) > 0:
            return None

        closes = bars['close']
        rsi = self._calculate_rsi(closes)

        current_price = context['tick']['mid']
        pip_value = 0.0001 if 'JPY' not in context.get('symbol', 'EURUSD') else 0.01

        # RSI sobrevendido: comprar
        if rsi < self.oversold:
            return {
                'side': 'buy',
                'size': 0.1,
                'stop_loss': current_price - (self.stop_loss_pips * pip_value),
                'take_profit': current_price + (self.take_profit_pips * pip_value),
                'strategy_name': self.name
            }

        # RSI sobrecomprado: vender
        if rsi > self.overbought:
            return {
                'side': 'sell',
                'size': 0.1,
                'stop_loss': current_price + (self.stop_loss_pips * pip_value),
                'take_profit': current_price - (self.take_profit_pips * pip_value),
                'strategy_name': self.name
            }

        return None


async def download_data(symbol: str = "EURUSD",
                        timeframe: str = "M5",
                        bars_count: int = 2000) -> pd.DataFrame:
    """Baixa dados históricos da API em múltiplos requests."""
    logger.info(f"Baixando {bars_count} barras de {symbol} {timeframe}...")

    client = HistoricalDataClient()

    if not await client.connect():
        logger.error("Falha ao conectar à API")
        return pd.DataFrame()

    try:
        all_bars = []
        remaining = bars_count
        end_time = None  # Começar do presente

        while remaining > 0:
            batch_size = min(remaining, 1000)  # API limita a 1000
            df = await client.get_bars(symbol, timeframe, count=batch_size,
                                       end_time=end_time)

            if df.empty:
                break

            all_bars.append(df)
            remaining -= len(df)

            # Próximo request começa antes da primeira barra obtida
            if 'timestamp' in df.columns:
                from datetime import timedelta
                period_minutes = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
                                 'H1': 60, 'H4': 240, 'D1': 1440}.get(timeframe, 5)
                end_time = df['timestamp'].min() - timedelta(minutes=period_minutes)

            logger.info(f"Progresso: {bars_count - remaining}/{bars_count} barras")
            await asyncio.sleep(0.3)  # Rate limiting

        if all_bars:
            result = pd.concat(all_bars, ignore_index=True)
            result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            logger.info(f"Total: {len(result)} barras baixadas")
            return result

        return pd.DataFrame()
    finally:
        await client.disconnect()


async def main():
    """Função principal do backtest."""
    print("\n" + "="*60)
    print("SISTEMA DE BACKTEST - Trading Bot FX")
    print("="*60 + "\n")

    # 1. Baixar dados
    symbol = "EURUSD"
    timeframe = "M5"
    bars_count = 3000

    df = await download_data(symbol, timeframe, bars_count)

    if df.empty:
        logger.error("Sem dados para backtest")
        return

    print(f"\nDados carregados:")
    print(f"  Período: {df['timestamp'].iloc[0]} a {df['timestamp'].iloc[-1]}")
    print(f"  Total de barras: {len(df)}")
    print(f"  Timeframe: {timeframe}")

    # 2. Criar motor de backtest
    engine = BarBacktestEngine(
        initial_balance=10000.0,
        commission_per_lot=7.0,
        slippage_pips=0.5,
        spread_pips=1.0
    )

    # 3. Testar estratégia SMA
    print("\n" + "-"*60)
    print("Testando: Simple Moving Average Crossover")
    print("-"*60)

    sma_strategy = SimpleMAStrategy(fast_period=10, slow_period=30,
                                    stop_loss_pips=20, take_profit_pips=40)

    result_sma = await engine.run(sma_strategy, df, symbol, timeframe)
    engine.print_summary(result_sma)

    # 4. Testar estratégia RSI
    print("\n" + "-"*60)
    print("Testando: RSI Mean Reversion")
    print("-"*60)

    rsi_strategy = RSIMeanReversionStrategy(rsi_period=14,
                                            oversold=30, overbought=70,
                                            stop_loss_pips=25, take_profit_pips=25)

    result_rsi = await engine.run(rsi_strategy, df, symbol, timeframe)
    engine.print_summary(result_rsi)

    # 5. Comparação
    print("\n" + "="*60)
    print("COMPARAÇÃO DE ESTRATÉGIAS")
    print("="*60)
    print(f"{'Métrica':<25} {'SMA Crossover':>15} {'RSI Mean Rev':>15}")
    print("-"*60)
    print(f"{'Net Profit ($)':<25} {result_sma.metrics.net_profit:>15.2f} {result_rsi.metrics.net_profit:>15.2f}")
    print(f"{'Return (%)':<25} {result_sma.metrics.return_pct:>15.2f} {result_rsi.metrics.return_pct:>15.2f}")
    print(f"{'Win Rate (%)':<25} {result_sma.metrics.win_rate:>15.1f} {result_rsi.metrics.win_rate:>15.1f}")
    print(f"{'Profit Factor':<25} {result_sma.metrics.profit_factor:>15.2f} {result_rsi.metrics.profit_factor:>15.2f}")
    print(f"{'Max Drawdown (%)':<25} {result_sma.metrics.max_drawdown_pct:>15.2f} {result_rsi.metrics.max_drawdown_pct:>15.2f}")
    print(f"{'Sharpe Ratio':<25} {result_sma.metrics.sharpe_ratio:>15.2f} {result_rsi.metrics.sharpe_ratio:>15.2f}")
    print(f"{'Total Trades':<25} {result_sma.metrics.total_trades:>15} {result_rsi.metrics.total_trades:>15}")
    print("="*60)

    # 6. Salvar resultados
    results_dir = Path("data/backtest_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Salvar trades em CSV
    if result_sma.trades:
        trades_df = pd.DataFrame([t.to_dict() for t in result_sma.trades])
        trades_file = results_dir / f"sma_trades_{timestamp_str}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"\nTrades SMA salvos em: {trades_file}")

    if result_rsi.trades:
        trades_df = pd.DataFrame([t.to_dict() for t in result_rsi.trades])
        trades_file = results_dir / f"rsi_trades_{timestamp_str}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"Trades RSI salvos em: {trades_file}")

    print("\nBacktest concluído com sucesso!")


if __name__ == "__main__":
    asyncio.run(main())
