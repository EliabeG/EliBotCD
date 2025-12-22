# backtest/bar_backtest_engine.py
"""
Motor de backtesting baseado em barras (OHLCV) sem lookahead bias.
Projetado para trabalhar com dados históricos da API TickTrader.
"""
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import time

from config.settings import CONFIG
from utils.logger import setup_logger

logger = setup_logger("bar_backtest")


@dataclass
class BacktestPosition:
    """Representa uma posição aberta no backtest."""
    id: str
    strategy_name: str
    symbol: str
    side: str  # 'buy' ou 'sell'
    entry_time: datetime
    entry_price: float
    size_lots: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestTrade:
    """Trade fechado no backtest."""
    id: str
    strategy_name: str
    symbol: str
    side: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    size_lots: float
    pnl_gross: float  # PnL bruto (antes de comissões)
    pnl_net: float    # PnL líquido (após comissões)
    pnl_pips: float
    commission: float
    exit_reason: str
    duration_seconds: int
    mfe: float = 0.0  # Maximum Favorable Excursion
    mae: float = 0.0  # Maximum Adverse Excursion
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'side': self.side,
            'entry_time': self.entry_time.isoformat(),
            'entry_price': self.entry_price,
            'exit_time': self.exit_time.isoformat(),
            'exit_price': self.exit_price,
            'size_lots': self.size_lots,
            'pnl_gross': self.pnl_gross,
            'pnl_net': self.pnl_net,
            'pnl_pips': self.pnl_pips,
            'commission': self.commission,
            'exit_reason': self.exit_reason,
            'duration_seconds': self.duration_seconds,
            'mfe': self.mfe,
            'mae': self.mae
        }


@dataclass
class BacktestMetrics:
    """Métricas do backtest calculadas."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    total_commission: float = 0.0

    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    expectancy: float = 0.0

    max_drawdown_pct: float = 0.0
    max_drawdown_abs: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    total_pips: float = 0.0
    avg_pips_per_trade: float = 0.0

    initial_balance: float = 0.0
    final_balance: float = 0.0
    return_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class BacktestResult:
    """Resultado completo do backtest."""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime

    metrics: BacktestMetrics
    trades: List[BacktestTrade]
    equity_curve: List[Tuple[datetime, float]]

    run_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'metrics': self.metrics.to_dict(),
            'trades_count': len(self.trades),
            'run_time_seconds': self.run_time_seconds
        }


class BarBacktestEngine:
    """
    Motor de backtest baseado em barras OHLCV.

    Características:
    - Sem lookahead bias: cada barra é processada sequencialmente
    - Suporta múltiplas estratégias simultâneas
    - Simula slippage e comissões
    - Calcula métricas de performance completas
    """

    def __init__(self,
                 initial_balance: float = 10000.0,
                 commission_per_lot: float = 7.0,  # USD por lote
                 slippage_pips: float = 0.5,
                 spread_pips: float = 1.0,
                 contract_size: int = 100000):

        self.initial_balance = initial_balance
        self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips
        self.spread_pips = spread_pips
        self.contract_size = contract_size

        # Estado do backtest
        self.balance = initial_balance
        self.equity = initial_balance
        self.open_positions: Dict[str, BacktestPosition] = {}
        self.closed_trades: List[BacktestTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

        self._trade_counter = 0

    def reset(self):
        """Reseta o estado do backtest."""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.open_positions.clear()
        self.closed_trades.clear()
        self.equity_curve.clear()
        self._trade_counter = 0

    def _get_pip_value(self, symbol: str) -> float:
        """Retorna o valor de 1 pip para o símbolo."""
        if 'JPY' in symbol.upper():
            return 0.01
        return 0.0001

    def _price_to_pips(self, price_diff: float, symbol: str) -> float:
        """Converte diferença de preço para pips."""
        return price_diff / self._get_pip_value(symbol)

    def _calculate_pnl(self, position: BacktestPosition, exit_price: float) -> Tuple[float, float]:
        """
        Calcula PnL bruto e líquido para uma posição.
        Retorna: (pnl_gross, pnl_net)
        """
        pip_value = self._get_pip_value(position.symbol)

        if position.side == 'buy':
            price_diff = exit_price - position.entry_price
        else:
            price_diff = position.entry_price - exit_price

        # PnL em valor monetário (assumindo conta USD)
        # Para EURUSD: 1 pip = $10 por lote padrão
        pips = price_diff / pip_value
        pnl_per_pip = 10.0 * position.size_lots  # $10 por pip por lote

        pnl_gross = pips * pnl_per_pip

        # Comissão: entrada + saída
        commission = self.commission_per_lot * position.size_lots * 2

        pnl_net = pnl_gross - commission

        return pnl_gross, pnl_net, pips, commission

    def open_position(self,
                      bar_time: datetime,
                      symbol: str,
                      side: str,
                      size_lots: float,
                      entry_price: float,
                      stop_loss: Optional[float] = None,
                      take_profit: Optional[float] = None,
                      strategy_name: str = "unknown",
                      metadata: Optional[Dict] = None) -> str:
        """
        Abre uma nova posição.
        O preço de entrada já deve incluir spread/slippage.
        """
        self._trade_counter += 1
        position_id = f"BT_{self._trade_counter}"

        # Aplicar slippage ao preço
        pip_value = self._get_pip_value(symbol)
        slippage_price = self.slippage_pips * pip_value

        if side == 'buy':
            # Comprando: pagamos o ask + slippage
            actual_entry = entry_price + slippage_price
        else:
            # Vendendo: recebemos o bid - slippage
            actual_entry = entry_price - slippage_price

        position = BacktestPosition(
            id=position_id,
            strategy_name=strategy_name,
            symbol=symbol,
            side=side,
            entry_time=bar_time,
            entry_price=actual_entry,
            size_lots=size_lots,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata or {}
        )

        self.open_positions[position_id] = position
        logger.debug(f"Posição aberta: {position_id} {side.upper()} {size_lots} {symbol} @ {actual_entry:.5f}")

        return position_id

    def close_position(self,
                       position_id: str,
                       bar_time: datetime,
                       exit_price: float,
                       exit_reason: str,
                       mfe: float = 0.0,
                       mae: float = 0.0) -> Optional[BacktestTrade]:
        """Fecha uma posição aberta."""
        if position_id not in self.open_positions:
            logger.warning(f"Posição {position_id} não encontrada para fechar")
            return None

        position = self.open_positions.pop(position_id)

        # Aplicar slippage ao preço de saída
        pip_value = self._get_pip_value(position.symbol)
        slippage_price = self.slippage_pips * pip_value

        if position.side == 'buy':
            # Fechando compra: vendemos no bid - slippage
            actual_exit = exit_price - slippage_price
        else:
            # Fechando venda: compramos no ask + slippage
            actual_exit = exit_price + slippage_price

        pnl_gross, pnl_net, pips, commission = self._calculate_pnl(position, actual_exit)

        # Atualizar balanço
        self.balance += pnl_net

        duration = int((bar_time - position.entry_time).total_seconds())

        trade = BacktestTrade(
            id=position.id,
            strategy_name=position.strategy_name,
            symbol=position.symbol,
            side=position.side,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            exit_time=bar_time,
            exit_price=actual_exit,
            size_lots=position.size_lots,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            pnl_pips=pips,
            commission=commission,
            exit_reason=exit_reason,
            duration_seconds=duration,
            mfe=mfe,
            mae=mae,
            metadata=position.metadata
        )

        self.closed_trades.append(trade)
        logger.debug(f"Posição fechada: {position_id} @ {actual_exit:.5f} | PnL: ${pnl_net:.2f} ({pips:.1f} pips) | {exit_reason}")

        return trade

    def check_stops(self, bar: pd.Series, bar_time: datetime):
        """
        Verifica SL/TP para todas as posições abertas usando os preços da barra.
        Usa HIGH/LOW para verificação mais realista.
        """
        for pos_id in list(self.open_positions.keys()):
            position = self.open_positions.get(pos_id)
            if not position:
                continue

            high_price = bar['high']
            low_price = bar['low']

            exit_price = None
            exit_reason = None

            if position.side == 'buy':
                # Para compra: SL se low <= SL, TP se high >= TP
                if position.stop_loss and low_price <= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = "Stop Loss"
                elif position.take_profit and high_price >= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "Take Profit"
            else:
                # Para venda: SL se high >= SL, TP se low <= TP
                if position.stop_loss and high_price >= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = "Stop Loss"
                elif position.take_profit and low_price <= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "Take Profit"

            if exit_price and exit_reason:
                self.close_position(pos_id, bar_time, exit_price, exit_reason)

    def update_equity(self, current_price: float, bar_time: datetime):
        """Atualiza a equity curve com o valor atual."""
        unrealized_pnl = 0.0

        for position in self.open_positions.values():
            pnl_gross, _, _, _ = self._calculate_pnl(position, current_price)
            unrealized_pnl += pnl_gross

        self.equity = self.balance + unrealized_pnl
        self.equity_curve.append((bar_time, self.equity))

    def calculate_metrics(self) -> BacktestMetrics:
        """Calcula todas as métricas do backtest."""
        metrics = BacktestMetrics()
        metrics.initial_balance = self.initial_balance
        metrics.final_balance = self.balance
        metrics.total_trades = len(self.closed_trades)

        if not self.closed_trades:
            return metrics

        # Estatísticas básicas
        wins = [t for t in self.closed_trades if t.pnl_net > 0]
        losses = [t for t in self.closed_trades if t.pnl_net <= 0]

        metrics.winning_trades = len(wins)
        metrics.losing_trades = len(losses)
        metrics.win_rate = len(wins) / len(self.closed_trades) * 100

        # PnL
        metrics.gross_profit = sum(t.pnl_net for t in wins)
        metrics.gross_loss = abs(sum(t.pnl_net for t in losses))
        metrics.net_profit = metrics.gross_profit - metrics.gross_loss
        metrics.total_commission = sum(t.commission for t in self.closed_trades)

        # Médias
        metrics.avg_win = metrics.gross_profit / len(wins) if wins else 0
        metrics.avg_loss = metrics.gross_loss / len(losses) if losses else 0
        metrics.avg_trade = metrics.net_profit / len(self.closed_trades)

        # Profit Factor
        metrics.profit_factor = metrics.gross_profit / metrics.gross_loss if metrics.gross_loss > 0 else float('inf')

        # Expectancy
        win_prob = len(wins) / len(self.closed_trades)
        loss_prob = len(losses) / len(self.closed_trades)
        metrics.expectancy = (win_prob * metrics.avg_win) - (loss_prob * metrics.avg_loss)

        # Pips
        metrics.total_pips = sum(t.pnl_pips for t in self.closed_trades)
        metrics.avg_pips_per_trade = metrics.total_pips / len(self.closed_trades)

        # Consecutive wins/losses
        current_streak = 0
        max_wins = 0
        max_losses = 0
        last_was_win = None

        for trade in self.closed_trades:
            is_win = trade.pnl_net > 0
            if is_win:
                if last_was_win == True:
                    current_streak += 1
                else:
                    current_streak = 1
                max_wins = max(max_wins, current_streak)
            else:
                if last_was_win == False:
                    current_streak += 1
                else:
                    current_streak = 1
                max_losses = max(max_losses, current_streak)
            last_was_win = is_win

        metrics.max_consecutive_wins = max_wins
        metrics.max_consecutive_losses = max_losses

        # Drawdown
        if self.equity_curve:
            equity_values = [e for _, e in self.equity_curve]
            peak = equity_values[0]
            max_dd = 0
            max_dd_abs = 0

            for eq in equity_values:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak
                dd_abs = peak - eq
                max_dd = max(max_dd, dd)
                max_dd_abs = max(max_dd_abs, dd_abs)

            metrics.max_drawdown_pct = max_dd * 100
            metrics.max_drawdown_abs = max_dd_abs

        # Retorno
        metrics.return_pct = ((self.balance - self.initial_balance) / self.initial_balance) * 100

        # Sharpe Ratio (simplificado, assumindo returns diários)
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev_eq = self.equity_curve[i-1][1]
                curr_eq = self.equity_curve[i][1]
                ret = (curr_eq - prev_eq) / prev_eq if prev_eq > 0 else 0
                returns.append(ret)

            if returns:
                avg_ret = np.mean(returns)
                std_ret = np.std(returns)
                if std_ret > 0:
                    # Anualizado (assumindo ~252 dias de trading)
                    metrics.sharpe_ratio = (avg_ret * 252) / (std_ret * np.sqrt(252))

                    # Sortino (usa apenas retornos negativos para downside)
                    neg_returns = [r for r in returns if r < 0]
                    if neg_returns:
                        downside_std = np.std(neg_returns)
                        if downside_std > 0:
                            metrics.sortino_ratio = (avg_ret * 252) / (downside_std * np.sqrt(252))

        # Calmar Ratio
        if metrics.max_drawdown_pct > 0:
            annual_return = metrics.return_pct  # Simplificado
            metrics.calmar_ratio = annual_return / metrics.max_drawdown_pct

        return metrics

    async def run(self,
                  strategy,
                  bars: pd.DataFrame,
                  symbol: str = "EURUSD",
                  timeframe: str = "M1") -> BacktestResult:
        """
        Executa o backtest para uma estratégia.

        Args:
            strategy: Instância da estratégia a ser testada
            bars: DataFrame com colunas timestamp, open, high, low, close, volume
            symbol: Símbolo do instrumento
            timeframe: Timeframe das barras

        Returns:
            BacktestResult com todas as métricas e trades
        """
        self.reset()
        start_time = time.perf_counter()

        # Garantir que temos timestamp como índice
        df = bars.copy()
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')

        if df.empty:
            logger.error("DataFrame de barras está vazio")
            return self._create_empty_result(strategy.name, symbol, timeframe)

        logger.info(f"Iniciando backtest de {strategy.name} em {symbol} {timeframe}")
        logger.info(f"Período: {df.index[0]} a {df.index[-1]} ({len(df)} barras)")

        # Inicializar equity curve
        self.equity_curve.append((df.index[0], self.initial_balance))

        total_bars = len(df)

        # Loop principal: processar cada barra sequencialmente
        for i, (bar_time, bar) in enumerate(df.iterrows()):
            # 1. Verificar stops nas posições abertas (usando high/low da barra)
            self.check_stops(bar, bar_time)

            # 2. Construir contexto de mercado SEM dados futuros
            # A estratégia só vê barras até a atual (índice 0 até i, inclusive)
            historical_slice = df.iloc[:i+1]

            market_context = self._build_market_context(
                historical_slice, bar, bar_time, symbol, timeframe
            )

            # 3. Verificar saídas discricionárias da estratégia
            await self._check_strategy_exits(strategy, market_context, bar_time, bar)

            # 4. Verificar novos sinais de entrada
            await self._check_strategy_entries(strategy, market_context, bar_time, bar, symbol)

            # 5. Atualizar equity curve
            self.update_equity(bar['close'], bar_time)

            # Log de progresso
            if (i + 1) % (total_bars // 10 or 1) == 0:
                pct = ((i + 1) / total_bars) * 100
                logger.info(f"Progresso: {pct:.0f}% | Trades: {len(self.closed_trades)} | Balance: ${self.balance:.2f}")

        # Fechar posições abertas ao final
        if self.open_positions:
            logger.info(f"Fechando {len(self.open_positions)} posições abertas ao final do backtest")
            last_bar = df.iloc[-1]
            last_time = df.index[-1]
            for pos_id in list(self.open_positions.keys()):
                self.close_position(pos_id, last_time, last_bar['close'], "Fim do Backtest")

        # Calcular métricas finais
        metrics = self.calculate_metrics()

        run_time = time.perf_counter() - start_time

        result = BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=df.index[0],
            end_date=df.index[-1],
            metrics=metrics,
            trades=self.closed_trades.copy(),
            equity_curve=self.equity_curve.copy(),
            run_time_seconds=run_time
        )

        logger.info(f"Backtest concluído em {run_time:.2f}s")
        logger.info(f"Trades: {metrics.total_trades} | Win Rate: {metrics.win_rate:.1f}%")
        logger.info(f"Net Profit: ${metrics.net_profit:.2f} | Max DD: {metrics.max_drawdown_pct:.1f}%")

        return result

    def _build_market_context(self,
                              historical_bars: pd.DataFrame,
                              current_bar: pd.Series,
                              bar_time: datetime,
                              symbol: str,
                              timeframe: str) -> Dict[str, Any]:
        """
        Constrói o contexto de mercado para a estratégia.
        IMPORTANTE: Não inclui dados futuros.
        """
        # Calcular spread simulado
        spread = self.spread_pips * self._get_pip_value(symbol)

        # Calcular bid/ask a partir do close
        mid = current_bar['close']
        bid = mid - spread / 2
        ask = mid + spread / 2

        # Criar tick simulado
        tick_data = {
            'symbol': symbol,
            'timestamp': bar_time,
            'bid': bid,
            'ask': ask,
            'mid': mid,
            'spread': spread,
            'volume': current_bar.get('volume', 0)
        }

        # Volatilidade simples (ATR aproximado)
        volatility = 0.0
        if len(historical_bars) >= 14:
            high = historical_bars['high'].iloc[-14:]
            low = historical_bars['low'].iloc[-14:]
            close = historical_bars['close'].iloc[-14:]
            tr = np.maximum(high - low,
                           np.maximum(abs(high - close.shift(1)),
                                     abs(low - close.shift(1))))
            volatility = tr.mean() / mid if mid > 0 else 0

        return {
            'tick': tick_data,
            'current_bar': current_bar.to_dict(),
            'bars': historical_bars,
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': bar_time,
            'spread': spread,
            'volatility': volatility,
            'balance': self.balance,
            'equity': self.equity,
            'open_positions': list(self.open_positions.values())
        }

    async def _check_strategy_exits(self, strategy, context: Dict, bar_time: datetime, bar: pd.Series):
        """Verifica sinais de saída da estratégia para posições abertas."""
        for pos_id, position in list(self.open_positions.items()):
            # Converter posição para formato que a estratégia espera
            position_dict = {
                'id': position.id,
                'side': position.side,
                'entry_price': position.entry_price,
                'size_lots': position.size_lots,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit
            }

            # Chamar método de verificação de saída se existir
            if hasattr(strategy, 'check_exit'):
                try:
                    exit_signal = await strategy.check_exit(position_dict, context)
                    if exit_signal:
                        exit_price = exit_signal.get('price', bar['close'])
                        exit_reason = exit_signal.get('reason', 'Strategy Exit')
                        self.close_position(pos_id, bar_time, exit_price, exit_reason)
                except Exception as e:
                    logger.debug(f"Erro ao verificar saída: {e}")

    async def _check_strategy_entries(self, strategy, context: Dict, bar_time: datetime,
                                       bar: pd.Series, symbol: str):
        """Verifica sinais de entrada da estratégia."""
        # Verificar se estratégia está ativa
        if hasattr(strategy, 'active') and not strategy.active:
            return

        # Chamar método de geração de sinal
        signal = None
        try:
            if hasattr(strategy, 'on_bar'):
                signal = await strategy.on_bar(context)
            elif hasattr(strategy, 'on_tick'):
                signal = await strategy.on_tick(context)
            elif hasattr(strategy, 'generate_signal'):
                signal = await strategy.generate_signal(context)
        except Exception as e:
            logger.debug(f"Erro ao gerar sinal: {e}")
            return

        if not signal:
            return

        # Processar sinal
        side = signal.side if hasattr(signal, 'side') else signal.get('side')
        if not side:
            return

        side = side.lower()
        if side not in ('buy', 'sell'):
            return

        # Obter parâmetros do sinal
        stop_loss = signal.stop_loss if hasattr(signal, 'stop_loss') else signal.get('stop_loss')
        take_profit = signal.take_profit if hasattr(signal, 'take_profit') else signal.get('take_profit')
        size = signal.size if hasattr(signal, 'size') else signal.get('size', 0.1)
        strategy_name = signal.strategy_name if hasattr(signal, 'strategy_name') else signal.get('strategy_name', strategy.name if hasattr(strategy, 'name') else 'Unknown')

        # Preço de entrada (close da barra)
        entry_price = bar['close']

        # Abrir posição
        self.open_position(
            bar_time=bar_time,
            symbol=symbol,
            side=side,
            size_lots=size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=strategy_name
        )

    def _create_empty_result(self, strategy_name: str, symbol: str, timeframe: str) -> BacktestResult:
        """Cria resultado vazio para casos de erro."""
        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc),
            metrics=BacktestMetrics(),
            trades=[],
            equity_curve=[]
        )

    def print_summary(self, result: BacktestResult):
        """Imprime um resumo formatado do backtest."""
        m = result.metrics
        print("\n" + "="*60)
        print(f"BACKTEST SUMMARY: {result.strategy_name}")
        print("="*60)
        print(f"Symbol: {result.symbol} | Timeframe: {result.timeframe}")
        print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
        print("-"*60)
        print(f"Initial Balance: ${m.initial_balance:,.2f}")
        print(f"Final Balance:   ${m.final_balance:,.2f}")
        print(f"Net Profit:      ${m.net_profit:,.2f} ({m.return_pct:.2f}%)")
        print("-"*60)
        print(f"Total Trades:    {m.total_trades}")
        print(f"Win Rate:        {m.win_rate:.1f}%")
        print(f"Profit Factor:   {m.profit_factor:.2f}")
        print(f"Avg Trade:       ${m.avg_trade:.2f}")
        print(f"Expectancy:      ${m.expectancy:.2f}")
        print("-"*60)
        print(f"Total Pips:      {m.total_pips:.1f}")
        print(f"Avg Pips/Trade:  {m.avg_pips_per_trade:.1f}")
        print("-"*60)
        print(f"Max Drawdown:    {m.max_drawdown_pct:.2f}% (${m.max_drawdown_abs:.2f})")
        print(f"Sharpe Ratio:    {m.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:   {m.sortino_ratio:.2f}")
        print(f"Calmar Ratio:    {m.calmar_ratio:.2f}")
        print("-"*60)
        print(f"Max Consec Wins:   {m.max_consecutive_wins}")
        print(f"Max Consec Losses: {m.max_consecutive_losses}")
        print(f"Total Commission:  ${m.total_commission:.2f}")
        print("="*60)


# Função auxiliar para executar backtest
async def run_backtest(strategy,
                       bars: pd.DataFrame,
                       symbol: str = "EURUSD",
                       timeframe: str = "M1",
                       initial_balance: float = 10000.0,
                       commission_per_lot: float = 7.0,
                       slippage_pips: float = 0.5) -> BacktestResult:
    """
    Função auxiliar para executar backtest.

    Args:
        strategy: Estratégia a ser testada
        bars: DataFrame OHLCV
        symbol: Símbolo
        timeframe: Timeframe
        initial_balance: Balanço inicial
        commission_per_lot: Comissão por lote
        slippage_pips: Slippage em pips

    Returns:
        BacktestResult
    """
    engine = BarBacktestEngine(
        initial_balance=initial_balance,
        commission_per_lot=commission_per_lot,
        slippage_pips=slippage_pips
    )

    return await engine.run(strategy, bars, symbol, timeframe)
