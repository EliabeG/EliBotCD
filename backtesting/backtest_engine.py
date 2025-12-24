"""
================================================================================
BACKTEST ENGINE
Motor de Backtesting para Estrategias de Trading
================================================================================

IMPORTANTE: Este motor usa APENAS dados REAIS do mercado.
Nenhuma simulacao ou dados sinteticos sao permitidos.
Isso envolve dinheiro real, entao a precisao e crucial.

Funcionalidades:
- Execucao tick-by-tick ou barra-por-barra
- Calculo de metricas de performance
- Simulacao realista de execucao
- Gerenciamento de posicoes
- Relatorios detalhados
"""

import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from api.fxopen_historical_ws import Bar, get_historical_data_sync as get_historical_data
from strategies.base import BaseStrategy, Signal, SignalType


class PositionType(Enum):
    """Tipo de posicao"""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    """Representa uma posicao aberta"""
    type: PositionType
    entry_price: float
    entry_time: datetime
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_name: str = ""
    signal_confidence: float = 0.0


@dataclass
class Trade:
    """Representa um trade completo (entrada + saida)"""
    position_type: PositionType
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    size: float
    pnl: float
    pnl_pips: float
    exit_reason: str  # 'stop_loss', 'take_profit', 'signal', 'end_of_data'
    strategy_name: str
    signal_confidence: float

    @property
    def duration(self) -> timedelta:
        return self.exit_time - self.entry_time

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestResult:
    """Resultado completo do backtest"""
    # Identificacao
    strategy_name: str
    symbol: str
    periodicity: str
    start_time: datetime
    end_time: datetime

    # Dados processados
    total_bars: int
    total_signals: int

    # Trades
    trades: List[Trade] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Performance
    total_pnl: float = 0.0
    total_pnl_pips: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pips: float = 0.0
    max_drawdown_duration: timedelta = field(default_factory=lambda: timedelta(0))

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Outros
    avg_trade_duration: timedelta = field(default_factory=lambda: timedelta(0))
    longest_trade: timedelta = field(default_factory=lambda: timedelta(0))
    shortest_trade: timedelta = field(default_factory=lambda: timedelta(0))


class BacktestEngine:
    """
    Motor de Backtesting

    IMPORTANTE: Usa APENAS dados REAIS do mercado.
    """

    def __init__(self,
                 initial_capital: float = 10000.0,
                 position_size: float = 0.01,  # Lotes
                 pip_value: float = 0.0001,
                 spread_pips: float = 1.0,
                 commission_per_lot: float = 0.0,
                 slippage_pips: float = 0.5):
        """
        Inicializa o motor de backtest

        Args:
            initial_capital: Capital inicial em USD
            position_size: Tamanho da posicao em lotes
            pip_value: Valor de 1 pip (0.0001 para EURUSD)
            spread_pips: Spread em pips
            commission_per_lot: Comissao por lote
            slippage_pips: Slippage medio em pips
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.pip_value = pip_value
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips

        # Estado
        self.capital = initial_capital
        self.equity_curve: List[float] = []
        self.current_position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.signals_generated: List[Signal] = []

    def run(self,
            strategy: BaseStrategy,
            symbol: str,
            periodicity: str,
            start_time: datetime,
            end_time: datetime = None,
            verbose: bool = True) -> BacktestResult:
        """
        Executa backtest de uma estrategia

        IMPORTANTE: Usa dados REAIS do mercado Forex.

        Args:
            strategy: Estrategia a testar
            symbol: Simbolo (ex: 'EURUSD')
            periodicity: Periodicidade (M1, H1, D1, etc)
            start_time: Inicio do periodo
            end_time: Fim do periodo
            verbose: Se True, mostra progresso

        Returns:
            BacktestResult com metricas de performance
        """
        if verbose:
            print("\n" + "=" * 70)
            print("  BACKTEST - DADOS REAIS DO MERCADO")
            print("=" * 70)
            print(f"  Estrategia: {strategy.name}")
            print(f"  Simbolo: {symbol}")
            print(f"  Periodicidade: {periodicity}")
            print(f"  Capital inicial: ${self.initial_capital:,.2f}")
            print(f"  Tamanho posicao: {self.position_size} lotes")
            print("=" * 70 + "\n")

        # Reset estado
        self._reset()
        strategy.reset()

        # Baixa dados REAIS
        bars = get_historical_data(symbol, periodicity, start_time, end_time)

        if not bars:
            print("ERRO: Nenhum dado historico disponivel!")
            return BacktestResult(
                strategy_name=strategy.name,
                symbol=symbol,
                periodicity=periodicity,
                start_time=start_time,
                end_time=end_time or datetime.now(timezone.utc),
                total_bars=0,
                total_signals=0
            )

        if verbose:
            print(f"Dados carregados: {len(bars)} barras")
            print(f"Periodo: {bars[0].timestamp} a {bars[-1].timestamp}")
            print("-" * 70)

        # Processa cada barra
        for i, bar in enumerate(bars):
            # Atualiza equity curve
            self._update_equity(bar.close)

            # Verifica stop/take da posicao atual
            if self.current_position:
                self._check_position_exit(bar)

            # Gera sinal da estrategia
            signal = strategy.analyze(
                price=bar.close,
                timestamp=bar.timestamp,
                volume=bar.volume,
                high=bar.high,
                low=bar.low,
                open=bar.open
            )

            if signal and signal.type != SignalType.HOLD:
                self.signals_generated.append(signal)

                # Processa sinal
                self._process_signal(signal, bar)

            # Progress
            if verbose and (i + 1) % 100 == 0:
                print(f"  Processado: {i+1}/{len(bars)} barras | "
                      f"Trades: {len(self.trades)} | "
                      f"Capital: ${self.capital:,.2f}")

        # Fecha posicao aberta no final
        if self.current_position:
            self._close_position(bars[-1].close, bars[-1].timestamp, "end_of_data")

        # Calcula metricas
        result = self._calculate_metrics(
            strategy_name=strategy.name,
            symbol=symbol,
            periodicity=periodicity,
            start_time=bars[0].timestamp,
            end_time=bars[-1].timestamp,
            total_bars=len(bars)
        )

        if verbose:
            self._print_results(result)

        return result

    def _reset(self):
        """Reseta estado do backtest"""
        self.capital = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.current_position = None
        self.trades = []
        self.signals_generated = []

    def _update_equity(self, current_price: float):
        """Atualiza curva de equity"""
        if self.current_position:
            # Calcula PnL nao realizado
            if self.current_position.type == PositionType.LONG:
                unrealized_pnl = (current_price - self.current_position.entry_price) / self.pip_value
            else:
                unrealized_pnl = (self.current_position.entry_price - current_price) / self.pip_value

            unrealized_pnl *= self.position_size * 100000 * self.pip_value  # Em USD
            equity = self.capital + unrealized_pnl
        else:
            equity = self.capital

        self.equity_curve.append(equity)

    def _check_position_exit(self, bar: Bar):
        """Verifica se posicao deve ser fechada por stop/take"""
        if not self.current_position:
            return

        pos = self.current_position

        # Verifica stop loss
        if pos.stop_loss:
            if pos.type == PositionType.LONG and bar.low <= pos.stop_loss:
                self._close_position(pos.stop_loss, bar.timestamp, "stop_loss")
                return
            elif pos.type == PositionType.SHORT and bar.high >= pos.stop_loss:
                self._close_position(pos.stop_loss, bar.timestamp, "stop_loss")
                return

        # Verifica take profit
        if pos.take_profit:
            if pos.type == PositionType.LONG and bar.high >= pos.take_profit:
                self._close_position(pos.take_profit, bar.timestamp, "take_profit")
                return
            elif pos.type == PositionType.SHORT and bar.low <= pos.take_profit:
                self._close_position(pos.take_profit, bar.timestamp, "take_profit")
                return

    def _process_signal(self, signal: Signal, bar: Bar):
        """Processa um sinal de trading"""
        # Se tem posicao aberta, verifica se deve fechar
        if self.current_position:
            # Fecha se sinal e oposto
            if (self.current_position.type == PositionType.LONG and signal.type == SignalType.SELL) or \
               (self.current_position.type == PositionType.SHORT and signal.type == SignalType.BUY):
                self._close_position(bar.close, bar.timestamp, "signal")

        # Abre nova posicao se nao tem
        if not self.current_position:
            self._open_position(signal, bar)

    def _open_position(self, signal: Signal, bar: Bar):
        """Abre uma nova posicao"""
        # Aplica slippage
        slippage = self.slippage_pips * self.pip_value
        spread = self.spread_pips * self.pip_value

        if signal.type == SignalType.BUY:
            entry_price = bar.close + spread / 2 + slippage  # Ask + slippage
            pos_type = PositionType.LONG
        else:
            entry_price = bar.close - spread / 2 - slippage  # Bid - slippage
            pos_type = PositionType.SHORT

        self.current_position = Position(
            type=pos_type,
            entry_price=entry_price,
            entry_time=bar.timestamp,
            size=self.position_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy_name=signal.strategy_name,
            signal_confidence=signal.confidence
        )

    def _close_position(self, exit_price: float, exit_time: datetime, reason: str):
        """Fecha posicao atual"""
        if not self.current_position:
            return

        pos = self.current_position

        # Aplica slippage na saida
        slippage = self.slippage_pips * self.pip_value

        if pos.type == PositionType.LONG:
            actual_exit = exit_price - slippage  # Bid - slippage
            pnl_pips = (actual_exit - pos.entry_price) / self.pip_value
        else:
            actual_exit = exit_price + slippage  # Ask + slippage
            pnl_pips = (pos.entry_price - actual_exit) / self.pip_value

        # Calcula PnL em USD (1 pip = $10 por lote padrao para EURUSD)
        pnl_usd = pnl_pips * pos.size * 10  # 10 USD por pip por lote

        # Subtrai comissao
        pnl_usd -= self.commission_per_lot * pos.size

        # Atualiza capital
        self.capital += pnl_usd

        # Registra trade
        trade = Trade(
            position_type=pos.type,
            entry_price=pos.entry_price,
            entry_time=pos.entry_time,
            exit_price=actual_exit,
            exit_time=exit_time,
            size=pos.size,
            pnl=pnl_usd,
            pnl_pips=pnl_pips,
            exit_reason=reason,
            strategy_name=pos.strategy_name,
            signal_confidence=pos.signal_confidence
        )
        self.trades.append(trade)

        # Limpa posicao
        self.current_position = None

    def _calculate_metrics(self, strategy_name: str, symbol: str, periodicity: str,
                          start_time: datetime, end_time: datetime, total_bars: int) -> BacktestResult:
        """Calcula todas as metricas de performance"""
        result = BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            periodicity=periodicity,
            start_time=start_time,
            end_time=end_time,
            total_bars=total_bars,
            total_signals=len(self.signals_generated),
            trades=self.trades
        )

        if not self.trades:
            return result

        # Metricas basicas
        result.total_trades = len(self.trades)
        result.winning_trades = sum(1 for t in self.trades if t.is_winner)
        result.losing_trades = result.total_trades - result.winning_trades

        result.total_pnl = sum(t.pnl for t in self.trades)
        result.total_pnl_pips = sum(t.pnl_pips for t in self.trades)

        # Win rate
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Medias
        winners = [t.pnl for t in self.trades if t.pnl > 0]
        losers = [t.pnl for t in self.trades if t.pnl < 0]

        result.avg_win = np.mean(winners) if winners else 0
        result.avg_loss = np.mean(losers) if losers else 0
        result.avg_trade = np.mean([t.pnl for t in self.trades])

        result.max_win = max(winners) if winners else 0
        result.max_loss = min(losers) if losers else 0

        # Drawdown - CORRIGIDO
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        # Evita divisão por zero
        drawdown = np.where(peak > 0, (peak - equity) / peak, 0)
        result.max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # Drawdown em USD (não em pips - cálculo anterior estava errado)
        drawdown_usd = peak - equity
        result.max_drawdown_pips = float(np.max(drawdown_usd)) if len(drawdown_usd) > 0 else 0.0

        # Determinar fator de anualização baseado na periodicidade
        # Períodos por ano aproximados para diferentes timeframes
        periods_per_year = self._get_periods_per_year(periodicity)

        # Sharpe Ratio (anualizado) - CORRIGIDO
        returns = np.diff(equity) / np.maximum(equity[:-1], 1.0)
        if len(returns) > 1:
            returns_std = np.std(returns)
            if returns_std > 0:
                # Anualiza usando o fator correto para a periodicidade
                result.sharpe_ratio = np.sqrt(periods_per_year) * np.mean(returns) / returns_std
            else:
                result.sharpe_ratio = 0.0

        # Sortino Ratio - CORRIGIDO
        # Sortino usa downside deviation (sqrt(mean(min(returns, 0)^2)))
        if len(returns) > 1:
            downside_returns = np.minimum(returns, 0)
            downside_std = np.sqrt(np.mean(downside_returns ** 2))
            if downside_std > 0:
                result.sortino_ratio = np.sqrt(periods_per_year) * np.mean(returns) / downside_std
            else:
                result.sortino_ratio = 0.0

        # Calmar Ratio - CORRIGIDO (agora anualizado corretamente)
        if result.max_drawdown > 0 and len(self.trades) > 0:
            # Calcular duração do backtest em dias
            total_days = (end_time - start_time).days
            if total_days > 0:
                # Anualizar o retorno
                total_return = result.total_pnl / self.initial_capital
                annual_return = total_return * (365.0 / total_days)
                result.calmar_ratio = annual_return / result.max_drawdown
            else:
                result.calmar_ratio = 0.0

        # Duracao dos trades
        durations = [t.duration for t in self.trades]
        if durations:
            result.avg_trade_duration = sum(durations, timedelta()) / len(durations)
            result.longest_trade = max(durations)
            result.shortest_trade = min(durations)

        return result

    def _get_periods_per_year(self, periodicity: str) -> float:
        """
        Retorna o número aproximado de períodos por ano para cada timeframe.

        Usado para anualizar métricas como Sharpe Ratio.
        """
        # Trading days per year ~252, hours per day ~24, etc.
        periods_map = {
            'M1': 252 * 24 * 60,      # 362880 (1-minute bars)
            'M5': 252 * 24 * 12,      # 72576 (5-minute bars)
            'M15': 252 * 24 * 4,      # 24192 (15-minute bars)
            'M30': 252 * 24 * 2,      # 12096 (30-minute bars)
            'H1': 252 * 24,           # 6048 (hourly bars)
            'H4': 252 * 6,            # 1512 (4-hour bars)
            'D1': 252,                # 252 (daily bars)
            'W1': 52,                 # 52 (weekly bars)
            'MN': 12,                 # 12 (monthly bars)
        }
        return periods_map.get(periodicity.upper(), 252)  # Default to daily

    def _print_results(self, result: BacktestResult):
        """Imprime resultados do backtest"""
        print("\n" + "=" * 70)
        print("  RESULTADOS DO BACKTEST")
        print("=" * 70)

        print(f"\n  RESUMO:")
        print(f"    Estrategia: {result.strategy_name}")
        print(f"    Simbolo: {result.symbol}")
        print(f"    Periodo: {result.start_time.strftime('%Y-%m-%d')} a {result.end_time.strftime('%Y-%m-%d')}")
        print(f"    Total de barras: {result.total_bars}")
        print(f"    Total de sinais: {result.total_signals}")

        print(f"\n  TRADES:")
        print(f"    Total de trades: {result.total_trades}")
        print(f"    Trades vencedores: {result.winning_trades}")
        print(f"    Trades perdedores: {result.losing_trades}")
        print(f"    Win Rate: {result.win_rate:.1%}")

        print(f"\n  PERFORMANCE:")
        print(f"    PnL Total: ${result.total_pnl:,.2f} ({result.total_pnl_pips:.1f} pips)")
        print(f"    Profit Factor: {result.profit_factor:.2f}")
        print(f"    Media por trade: ${result.avg_trade:,.2f}")
        print(f"    Media ganho: ${result.avg_win:,.2f}")
        print(f"    Media perda: ${result.avg_loss:,.2f}")
        print(f"    Maior ganho: ${result.max_win:,.2f}")
        print(f"    Maior perda: ${result.max_loss:,.2f}")

        print(f"\n  RISCO:")
        print(f"    Max Drawdown: {result.max_drawdown:.1%} ({result.max_drawdown_pips:.1f} pips)")
        print(f"    Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"    Sortino Ratio: {result.sortino_ratio:.2f}")
        print(f"    Calmar Ratio: {result.calmar_ratio:.2f}")

        print(f"\n  TEMPO:")
        print(f"    Duracao media trade: {result.avg_trade_duration}")
        print(f"    Trade mais longo: {result.longest_trade}")
        print(f"    Trade mais curto: {result.shortest_trade}")

        print("\n" + "=" * 70)

        # Detalhes dos trades
        if result.trades:
            print("\n  ULTIMOS 10 TRADES:")
            print("  " + "-" * 66)
            for trade in result.trades[-10:]:
                direction = "LONG " if trade.position_type == PositionType.LONG else "SHORT"
                result_str = "WIN " if trade.is_winner else "LOSS"
                print(f"    {direction} | {trade.entry_time.strftime('%Y-%m-%d %H:%M')} | "
                      f"Entry: {trade.entry_price:.5f} | Exit: {trade.exit_price:.5f} | "
                      f"PnL: {trade.pnl_pips:+.1f} pips | {result_str} | {trade.exit_reason}")


def run_backtest(strategy: BaseStrategy,
                symbol: str = "EURUSD",
                days: int = 30,
                periodicity: str = "H1",
                initial_capital: float = 10000.0,
                verbose: bool = True) -> BacktestResult:
    """
    Funcao de conveniencia para executar backtest

    Args:
        strategy: Estrategia a testar
        symbol: Simbolo
        days: Numero de dias de historico
        periodicity: Periodicidade
        initial_capital: Capital inicial
        verbose: Mostrar progresso

    Returns:
        BacktestResult
    """
    engine = BacktestEngine(initial_capital=initial_capital)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    return engine.run(
        strategy=strategy,
        symbol=symbol,
        periodicity=periodicity,
        start_time=start_time,
        end_time=end_time,
        verbose=verbose
    )
