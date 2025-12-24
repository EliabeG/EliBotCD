"""
================================================================================
BACKTEST ENGINE V2.0 - PRONTO PARA DINHEIRO REAL
Motor de Backtesting para Estrategias de Trading
================================================================================

IMPORTANTE: Este motor usa APENAS dados REAIS do mercado.
Nenhuma simulacao ou dados sinteticos sao permitidos.
Isso envolve dinheiro real, entao a precisao e crucial.

VERSÃO V2.0 - EXECUÇÃO REALISTA COM CUSTOS REAIS
================================================
Correções aplicadas:
1. Sinais são executados no OPEN da próxima barra (não no close atual)
2. Stop Loss considera gaps (execução no open se houver gap)
3. Take Profit considera gaps (execução no open se houver gap)
4. Lógica conservadora: stop tem prioridade sobre take em caso de ambiguidade
5. Slippage aplicado corretamente em todas as situações
6. NOVO V2.0: Spread realista de 1.5 pips
7. NOVO V2.0: Slippage realista de 0.8 pips

Funcionalidades:
- Execucao barra-por-barra com execução realista
- Calculo de metricas de performance
- Simulacao realista de execucao com gaps
- Gerenciamento de posicoes
- Relatorios detalhados
- Custos de execução realistas
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
    
    # NOVO: Informações adicionais para análise
    had_gap: bool = False  # Se houve gap na execução
    intended_exit_price: float = 0.0  # Preço pretendido (stop/take)

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
    
    # NOVO: Estatísticas de execução
    trades_with_gap: int = 0
    total_slippage_pips: float = 0.0


class BacktestEngine:
    """
    Motor de Backtesting - VERSÃO CORRIGIDA

    IMPORTANTE: Usa APENAS dados REAIS do mercado.
    
    CORREÇÕES IMPLEMENTADAS:
    1. Sinais executados no OPEN da próxima barra
    2. Stop/Take consideram gaps
    3. Execução realista com slippage
    """

    def __init__(self,
                 initial_capital: float = 10000.0,
                 position_size: float = 0.01,  # Lotes
                 pip_value: float = 0.0001,
                 spread_pips: float = 1.5,     # V2.0: Spread realista
                 commission_per_lot: float = 0.0,
                 slippage_pips: float = 0.8):  # V2.0: Slippage realista
        """
        Inicializa o motor de backtest

        V2.0: Custos realistas para dinheiro real

        Args:
            initial_capital: Capital inicial em USD
            position_size: Tamanho da posicao em lotes
            pip_value: Valor de 1 pip (0.0001 para EURUSD)
            spread_pips: Spread em pips (V2.0: 1.5 pips realista)
            commission_per_lot: Comissao por lote
            slippage_pips: Slippage medio em pips (V2.0: 0.8 pips realista)
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

        # NOVO: Sinal pendente para execução na próxima barra
        self.pending_signal: Optional[Signal] = None

        # CORREÇÃO #1: Fechamento pendente para executar no OPEN da próxima barra
        self.pending_close_signal: bool = False

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
        
        FLUXO CORRIGIDO:
        1. Para cada barra:
           a. Primeiro: Executar sinal PENDENTE da barra anterior (no OPEN desta barra)
           b. Segundo: Verificar stop/take profit (usando high/low desta barra)
           c. Terceiro: Gerar novo sinal (será executado na PRÓXIMA barra)

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
            print("  Versão Corrigida - Execução Realista")
            print("=" * 70)
            print(f"  Estrategia: {strategy.name}")
            print(f"  Simbolo: {symbol}")
            print(f"  Periodicidade: {periodicity}")
            print(f"  Capital inicial: ${self.initial_capital:,.2f}")
            print(f"  Tamanho posicao: {self.position_size} lotes")
            print(f"  Spread: {self.spread_pips} pips")
            print(f"  Slippage: {self.slippage_pips} pips")
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
            # ================================================================
            # PASSO 0: CORREÇÃO #1 - Fechar posição PENDENTE no OPEN desta barra
            # (Quando sinal oposto foi gerado na barra anterior)
            # ================================================================
            if self.pending_close_signal and self.current_position:
                self._close_position(bar.open, bar.timestamp, "signal")
                self.pending_close_signal = False

            # ================================================================
            # PASSO 1: Executar sinal PENDENTE no OPEN desta barra
            # ================================================================
            if self.pending_signal is not None:
                self._execute_pending_signal(self.pending_signal, bar)
                self.pending_signal = None

            # ================================================================
            # PASSO 2: Verificar stop/take da posição atual
            # (usando OPEN, HIGH, LOW desta barra - ordem importa!)
            # ================================================================
            if self.current_position:
                self._check_position_exit_realistic(bar)

            # ================================================================
            # PASSO 3: Atualizar equity curve
            # ================================================================
            self._update_equity(bar.close)

            # ================================================================
            # PASSO 4: Gerar sinal da estratégia
            # (será executado na PRÓXIMA barra, não agora!)
            # ================================================================
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

                # Verificar se devemos armazenar como pendente ou processar
                # Se tem posição aberta e sinal é oposto, marcar para fechar na PRÓXIMA barra
                if self.current_position:
                    should_close = (
                        (self.current_position.type == PositionType.LONG and signal.type == SignalType.SELL) or
                        (self.current_position.type == PositionType.SHORT and signal.type == SignalType.BUY)
                    )
                    if should_close:
                        # CORREÇÃO #1: Marcar para fechar no OPEN da PRÓXIMA barra
                        # (antes era: self._close_position(bar.close, ...) - ERRADO!)
                        self.pending_close_signal = True

                # Armazenar sinal para executar na PRÓXIMA barra
                if not self.current_position:
                    self.pending_signal = signal

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
        self.pending_signal = None
        self.pending_close_signal = False  # CORREÇÃO #1

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

    def _execute_pending_signal(self, signal: Signal, bar: Bar):
        """
        NOVO: Executa um sinal pendente no OPEN da barra atual

        Esta é a forma CORRETA de executar sinais:
        - O sinal foi gerado no CLOSE da barra anterior
        - A execução acontece no OPEN desta barra
        - Isso reflete a realidade do trading

        CORREÇÃO #5: Stop/Take são recalculados baseados no entry_price REAL
        """
        if self.current_position:
            # Já tem posição, não abrir outra
            return

        # Aplicar spread e slippage ao OPEN
        slippage = self.slippage_pips * self.pip_value
        spread = self.spread_pips * self.pip_value

        if signal.type == SignalType.BUY:
            # Compra no Ask (Open + spread/2) + slippage
            entry_price = bar.open + spread / 2 + slippage
            pos_type = PositionType.LONG
        else:
            # Venda no Bid (Open - spread/2) - slippage
            entry_price = bar.open - spread / 2 - slippage
            pos_type = PositionType.SHORT

        # CORREÇÃO #5: Recalcular stop/take baseado no entry_price REAL
        # Se o sinal tem stop_loss_pips/take_profit_pips, usar esses valores
        # para calcular os níveis reais baseados na entrada
        if signal.stop_loss_pips is not None and signal.take_profit_pips is not None:
            # Calcular níveis baseados no entry_price REAL
            if pos_type == PositionType.LONG:
                actual_stop_loss = entry_price - (signal.stop_loss_pips * self.pip_value)
                actual_take_profit = entry_price + (signal.take_profit_pips * self.pip_value)
            else:  # SHORT
                actual_stop_loss = entry_price + (signal.stop_loss_pips * self.pip_value)
                actual_take_profit = entry_price - (signal.take_profit_pips * self.pip_value)
        else:
            # Compatibilidade: usar valores fixos se não tiver pips
            actual_stop_loss = signal.stop_loss
            actual_take_profit = signal.take_profit

        self.current_position = Position(
            type=pos_type,
            entry_price=entry_price,
            entry_time=bar.timestamp,
            size=self.position_size,
            stop_loss=actual_stop_loss,      # CORREÇÃO #5: Baseado na entrada real
            take_profit=actual_take_profit,  # CORREÇÃO #5: Baseado na entrada real
            strategy_name=signal.strategy_name,
            signal_confidence=signal.confidence
        )

    def _check_position_exit_realistic(self, bar: Bar):
        """
        CORRIGIDO: Verifica se posição deve ser fechada por stop/take
        
        Considera GAPS e ordem realista de execução:
        1. Primeiro verifica se o OPEN já atingiu stop ou take (gap)
        2. Se não, verifica se HIGH/LOW atingiram durante a barra
        3. Stop loss tem PRIORIDADE em caso de ambiguidade (conservador)
        
        REGRAS DE GAP:
        - Se LONG e bar.open < stop_loss: executa no OPEN (gap down)
        - Se SHORT e bar.open > stop_loss: executa no OPEN (gap up)
        - Se LONG e bar.open > take_profit: executa no OPEN (gap up favorável)
        - Se SHORT e bar.open < take_profit: executa no OPEN (gap down favorável)
        """
        if not self.current_position:
            return

        pos = self.current_position
        had_gap = False
        intended_price = 0.0

        # ====================================================================
        # VERIFICAÇÃO DE GAP NO OPEN
        # ====================================================================
        
        if pos.type == PositionType.LONG:
            # LONG: Stop se preço CAI, Take se preço SOBE
            
            # Gap Down - Stop Loss atingido no OPEN
            if pos.stop_loss and bar.open <= pos.stop_loss:
                had_gap = True
                intended_price = pos.stop_loss
                # Executa no OPEN (pior que o stop)
                exit_price = bar.open
                self._close_position_with_details(
                    exit_price, bar.timestamp, "stop_loss", 
                    had_gap=True, intended_price=intended_price
                )
                return
                
            # Gap Up - Take Profit atingido no OPEN
            if pos.take_profit and bar.open >= pos.take_profit:
                had_gap = True
                intended_price = pos.take_profit
                # Executa no OPEN (melhor ou igual ao take)
                exit_price = bar.open
                self._close_position_with_details(
                    exit_price, bar.timestamp, "take_profit",
                    had_gap=True, intended_price=intended_price
                )
                return
                
        else:  # SHORT
            # SHORT: Stop se preço SOBE, Take se preço CAI
            
            # Gap Up - Stop Loss atingido no OPEN
            if pos.stop_loss and bar.open >= pos.stop_loss:
                had_gap = True
                intended_price = pos.stop_loss
                # Executa no OPEN (pior que o stop)
                exit_price = bar.open
                self._close_position_with_details(
                    exit_price, bar.timestamp, "stop_loss",
                    had_gap=True, intended_price=intended_price
                )
                return
                
            # Gap Down - Take Profit atingido no OPEN
            if pos.take_profit and bar.open <= pos.take_profit:
                had_gap = True
                intended_price = pos.take_profit
                # Executa no OPEN (melhor ou igual ao take)
                exit_price = bar.open
                self._close_position_with_details(
                    exit_price, bar.timestamp, "take_profit",
                    had_gap=True, intended_price=intended_price
                )
                return

        # ====================================================================
        # VERIFICAÇÃO DURANTE A BARRA (HIGH/LOW)
        # Usando lógica CONSERVADORA: Stop tem prioridade
        # ====================================================================
        
        if pos.type == PositionType.LONG:
            # Verificar Stop Loss primeiro (conservador)
            if pos.stop_loss and bar.low <= pos.stop_loss:
                self._close_position_with_details(
                    pos.stop_loss, bar.timestamp, "stop_loss",
                    had_gap=False, intended_price=pos.stop_loss
                )
                return
                
            # Verificar Take Profit
            if pos.take_profit and bar.high >= pos.take_profit:
                self._close_position_with_details(
                    pos.take_profit, bar.timestamp, "take_profit",
                    had_gap=False, intended_price=pos.take_profit
                )
                return
                
        else:  # SHORT
            # Verificar Stop Loss primeiro (conservador)
            if pos.stop_loss and bar.high >= pos.stop_loss:
                self._close_position_with_details(
                    pos.stop_loss, bar.timestamp, "stop_loss",
                    had_gap=False, intended_price=pos.stop_loss
                )
                return
                
            # Verificar Take Profit
            if pos.take_profit and bar.low <= pos.take_profit:
                self._close_position_with_details(
                    pos.take_profit, bar.timestamp, "take_profit",
                    had_gap=False, intended_price=pos.take_profit
                )
                return

    def _close_position_with_details(self, exit_price: float, exit_time: datetime, 
                                      reason: str, had_gap: bool = False, 
                                      intended_price: float = 0.0):
        """
        NOVO: Fecha posição com detalhes adicionais sobre execução
        """
        if not self.current_position:
            return

        pos = self.current_position

        # Aplica slippage na saída (sempre desfavorável)
        slippage = self.slippage_pips * self.pip_value

        if pos.type == PositionType.LONG:
            # Vende no Bid - slippage
            actual_exit = exit_price - slippage
            pnl_pips = (actual_exit - pos.entry_price) / self.pip_value
        else:
            # Compra no Ask + slippage
            actual_exit = exit_price + slippage
            pnl_pips = (pos.entry_price - actual_exit) / self.pip_value

        # Calcula PnL em USD (1 pip = $10 por lote padrao para EURUSD)
        pnl_usd = pnl_pips * pos.size * 10  # 10 USD por pip por lote

        # Subtrai comissao
        pnl_usd -= self.commission_per_lot * pos.size

        # Atualiza capital
        self.capital += pnl_usd

        # Registra trade com detalhes
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
            signal_confidence=pos.signal_confidence,
            had_gap=had_gap,
            intended_exit_price=intended_price if intended_price > 0 else exit_price
        )
        self.trades.append(trade)

        # Limpa posicao
        self.current_position = None

    def _close_position(self, exit_price: float, exit_time: datetime, reason: str):
        """
        Fecha posição (compatibilidade com código existente)
        """
        self._close_position_with_details(exit_price, exit_time, reason)

    def _open_position(self, signal: Signal, bar: Bar):
        """
        DEPRECATED: Usar _execute_pending_signal ao invés
        
        Mantido para compatibilidade, mas não deve ser chamado diretamente.
        """
        # Redireciona para o novo método
        self._execute_pending_signal(signal, bar)

    def _process_signal(self, signal: Signal, bar: Bar):
        """
        DEPRECATED: A lógica de processamento foi movida para o loop principal
        
        Mantido para compatibilidade.
        """
        pass

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

        # Drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        result.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        drawdown_pips = (peak - equity) / self.pip_value
        result.max_drawdown_pips = np.max(drawdown_pips) if len(drawdown_pips) > 0 else 0

        # Sharpe Ratio (anualizado)
        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            result.sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)

        # Sortino Ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 1 and np.std(negative_returns) > 0:
            result.sortino_ratio = np.sqrt(252) * np.mean(returns) / np.std(negative_returns)

        # Calmar Ratio
        if result.max_drawdown > 0:
            annual_return = result.total_pnl / self.initial_capital
            result.calmar_ratio = annual_return / result.max_drawdown

        # Duracao dos trades
        durations = [t.duration for t in self.trades]
        if durations:
            result.avg_trade_duration = sum(durations, timedelta()) / len(durations)
            result.longest_trade = max(durations)
            result.shortest_trade = min(durations)

        # NOVO: Estatísticas de execução
        result.trades_with_gap = sum(1 for t in self.trades if t.had_gap)
        
        # Calcular slippage total (diferença entre preço pretendido e executado)
        total_slippage = 0.0
        for t in self.trades:
            if t.intended_exit_price > 0:
                slippage = abs(t.exit_price - t.intended_exit_price) / self.pip_value
                total_slippage += slippage
        result.total_slippage_pips = total_slippage

        return result

    def _print_results(self, result: BacktestResult):
        """Imprime resultados do backtest"""
        print("\n" + "=" * 70)
        print("  RESULTADOS DO BACKTEST (Execução Realista)")
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
        
        # NOVO: Estatísticas de execução
        print(f"\n  EXECUCAO:")
        print(f"    Trades com gap: {result.trades_with_gap}")
        print(f"    Slippage total: {result.total_slippage_pips:.1f} pips")
        if result.total_trades > 0:
            print(f"    Slippage medio por trade: {result.total_slippage_pips/result.total_trades:.2f} pips")

        print("\n" + "=" * 70)

        # Detalhes dos trades
        if result.trades:
            print("\n  ULTIMOS 10 TRADES:")
            print("  " + "-" * 80)
            for trade in result.trades[-10:]:
                direction = "LONG " if trade.position_type == PositionType.LONG else "SHORT"
                result_str = "WIN " if trade.is_winner else "LOSS"
                gap_str = " [GAP]" if trade.had_gap else ""
                print(f"    {direction} | {trade.entry_time.strftime('%Y-%m-%d %H:%M')} | "
                      f"Entry: {trade.entry_price:.5f} | Exit: {trade.exit_price:.5f} | "
                      f"PnL: {trade.pnl_pips:+.1f} pips | {result_str} | {trade.exit_reason}{gap_str}")


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
