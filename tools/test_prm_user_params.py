#!/usr/bin/env python3
"""
================================================================================
BACKTEST PRM COM PARAMETROS DO USUARIO
================================================================================

Parametros solicitados:
- hmm_threshold = 0.9
- lyapunov_threshold = 0.1
- stop_loss_pips = 10
- take_profit_pips = 10
- signal_cooldown = 10
- hmm_states_allowed = [1, 2]
- trend_lookback = 10
- min_prices (warmup) = 6624
- periodicity = M5
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from api.fxopen_historical_ws import Bar, get_historical_data_with_spread_sync
from strategies.alta_volatilidade.prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot
from strategies.base import Signal, SignalType

# Custos de execucao
SPREAD_PIPS = 0.2
SLIPPAGE_PIPS = 0.0

# ===============================================================================
# PARAMETROS DO USUARIO
# ===============================================================================
HMM_THRESHOLD = 0.9
LYAPUNOV_THRESHOLD = 0.1
STOP_LOSS_PIPS = 10.0
TAKE_PROFIT_PIPS = 10.0
SIGNAL_COOLDOWN = 10
HMM_STATES_ALLOWED = [1, 2]
TREND_LOOKBACK = 10
MIN_PRICES_WARMUP = 6624  # Barras pre-carregadas antes de comecar a operar
# ===============================================================================


class PositionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    type: PositionType
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float


@dataclass
class Trade:
    type: PositionType
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    pnl_pips: float
    exit_reason: str


class PRMUserParamsStrategy:
    """
    Estrategia PRM com parametros do usuario
    """

    def __init__(self):
        self.name = "PRM-UserParams"

        # Parametros do usuario
        self.hmm_threshold = HMM_THRESHOLD
        self.lyapunov_threshold = LYAPUNOV_THRESHOLD
        self.stop_loss_pips = STOP_LOSS_PIPS
        self.take_profit_pips = TAKE_PROFIT_PIPS
        self.signal_cooldown_max = SIGNAL_COOLDOWN
        self.hmm_states_allowed = HMM_STATES_ALLOWED
        self.trend_lookback = TREND_LOOKBACK

        self.signal_cooldown = 0

        # Indicador PRM com thresholds BAIXOS para capturar todos os dados
        # O filtro real e feito no metodo analyze()
        self.prm = ProtocoloRiemannMandelbrot(
            n_states=3,
            hmm_threshold=0.1,  # Baixo para pegar tudo
            lyapunov_threshold_k=0.001,  # Baixo para pegar tudo
            curvature_threshold=0.0001,
            lookback_window=100,
            hmm_training_window=200,
            hmm_min_training_samples=50
        )

        self.last_analysis = None

    def calculate_direction(self, closes: List[float]) -> int:
        """
        Calcula direcao baseada em tendencia
        """
        min_bars = self.trend_lookback + 2
        if len(closes) < min_bars:
            return 0

        # closes[-1] = barra atual (NAO usar)
        # closes[-2] = ultima barra fechada
        recent_close = closes[-2]
        past_close = closes[-(self.trend_lookback + 2)]

        trend = recent_close - past_close

        return 1 if trend > 0 else -1

    def analyze(self, prices: np.ndarray, closes: List[float],
                timestamp: datetime) -> Optional[Signal]:
        """
        Analisa usando logica do otimizador
        """
        # Cooldown
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        try:
            # Analisar com PRM
            result = self.prm.analyze(prices)
            self.last_analysis = result

            # Filtrar por parametros do usuario
            hmm_prob = result['Prob_HMM']
            if hmm_prob < self.hmm_threshold:
                return None

            lyapunov = result['Lyapunov_Score']
            if lyapunov < self.lyapunov_threshold:
                return None

            hmm_state = result['hmm_analysis']['current_state']
            if hmm_state not in self.hmm_states_allowed:
                return None

            direction = self.calculate_direction(closes)
            if direction == 0:
                return None

            # Gerar sinal
            signal_type = SignalType.BUY if direction == 1 else SignalType.SELL

            signal = Signal(
                type=signal_type,
                price=prices[-1],
                timestamp=timestamp,
                strategy_name=self.name,
                confidence=hmm_prob,
                stop_loss_pips=self.stop_loss_pips,
                take_profit_pips=self.take_profit_pips
            )

            self.signal_cooldown = self.signal_cooldown_max

            return signal

        except Exception as e:
            return None


def run_backtest_with_warmup(bars: List[Bar], warmup_bars: int) -> List[Trade]:
    """
    Backtest com warmup - pre-carrega barras antes de comecar a operar
    """
    trades = []
    position: Optional[Position] = None
    pending_signal: Optional[Signal] = None
    pip = 0.0001

    strategy = PRMUserParamsStrategy()

    # Acumular precos e closes
    prices_buffer = []
    closes_buffer = []

    total_bars = len(bars)
    trading_start = warmup_bars

    print(f"\n  Total de barras: {total_bars}")
    print(f"  Barras de warmup: {warmup_bars}")
    print(f"  Barras de trading: {total_bars - warmup_bars}")

    # Fase de WARMUP - alimentar indicador sem operar
    print(f"\n  [WARMUP] Processando {warmup_bars} barras...")
    for i in range(warmup_bars):
        bar = bars[i]
        prices_buffer.append(bar.close)
        closes_buffer.append(bar.close)

        # Progresso de warmup
        if (i + 1) % 1000 == 0:
            print(f"    Warmup: {i+1}/{warmup_bars} barras")

    # Fazer uma analise inicial para treinar HMM
    if len(prices_buffer) >= 100:
        prices_array = np.array(prices_buffer)
        try:
            _ = strategy.prm.analyze(prices_array)
            print(f"  [WARMUP] HMM treinado com {len(prices_buffer)} barras")
        except Exception as e:
            print(f"  [WARMUP] Erro no treino inicial: {e}")

    # Fase de TRADING
    print(f"\n  [TRADING] Iniciando operacoes...")

    for i in range(warmup_bars, total_bars):
        bar = bars[i]

        # Adicionar ao buffer
        prices_buffer.append(bar.close)
        closes_buffer.append(bar.close)

        # 1. Executar sinal pendente no OPEN desta barra
        if pending_signal and position is None:
            if bar.has_spread_data:
                spread = bar.spread_pips * pip
            else:
                spread = SPREAD_PIPS * pip
            slippage = SLIPPAGE_PIPS * pip

            if pending_signal.type == SignalType.BUY:
                if bar.ask_open:
                    entry_price = bar.ask_open + slippage
                else:
                    entry_price = bar.open + spread / 2 + slippage
                pos_type = PositionType.LONG
                sl = entry_price - (strategy.stop_loss_pips * pip)
                tp = entry_price + (strategy.take_profit_pips * pip)
            else:
                if bar.bid_open:
                    entry_price = bar.bid_open - slippage
                else:
                    entry_price = bar.open - spread / 2 - slippage
                pos_type = PositionType.SHORT
                sl = entry_price + (strategy.stop_loss_pips * pip)
                tp = entry_price - (strategy.take_profit_pips * pip)

            position = Position(
                type=pos_type,
                entry_price=entry_price,
                entry_time=bar.timestamp,
                stop_loss=sl,
                take_profit=tp
            )
            pending_signal = None

        # 2. Verificar stop/take para posicao aberta
        if position:
            exit_price = None
            exit_reason = None

            if bar.has_spread_data:
                bid_close = bar.bid_close if bar.bid_close else bar.close
                ask_close = bar.ask_close if bar.ask_close else bar.close
                bid_low = bar.bid_low if bar.bid_low else bar.low
                ask_high = bar.ask_high if bar.ask_high else bar.high
            else:
                bid_close = bar.close
                ask_close = bar.close
                bid_low = bar.low
                ask_high = bar.high

            if position.type == PositionType.LONG:
                if bid_low <= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = "stop_loss"
                elif ask_high >= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "take_profit"
            else:
                if ask_high >= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = "stop_loss"
                elif bid_low <= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "take_profit"

            if exit_price:
                if position.type == PositionType.LONG:
                    pnl_pips = (exit_price - position.entry_price) / pip
                else:
                    pnl_pips = (position.entry_price - exit_price) / pip

                trades.append(Trade(
                    type=position.type,
                    entry_price=position.entry_price,
                    entry_time=position.entry_time,
                    exit_price=exit_price,
                    exit_time=bar.timestamp,
                    pnl_pips=pnl_pips,
                    exit_reason=exit_reason
                ))
                position = None

        # 3. Gerar novo sinal (se nao ha posicao)
        if position is None and pending_signal is None:
            prices_array = np.array(prices_buffer)
            signal = strategy.analyze(prices_array, closes_buffer, bar.timestamp)
            if signal:
                pending_signal = signal

        # Progresso
        trading_bar = i - warmup_bars + 1
        total_trading = total_bars - warmup_bars
        if trading_bar % 500 == 0:
            print(f"    Trading: {trading_bar}/{total_trading} barras, {len(trades)} trades")

    # Fechar posicao aberta no final
    if position:
        last_bar = bars[-1]
        if position.type == PositionType.LONG:
            exit_price = last_bar.bid_close if last_bar.bid_close else last_bar.close
            pnl_pips = (exit_price - position.entry_price) / pip
        else:
            exit_price = last_bar.ask_close if last_bar.ask_close else last_bar.close
            pnl_pips = (position.entry_price - exit_price) / pip

        trades.append(Trade(
            type=position.type,
            entry_price=position.entry_price,
            entry_time=position.entry_time,
            exit_price=exit_price,
            exit_time=last_bar.timestamp,
            pnl_pips=pnl_pips,
            exit_reason="end_of_data"
        ))

    return trades


def main():
    print("=" * 70)
    print("  BACKTEST PRM COM PARAMETROS DO USUARIO")
    print("=" * 70)

    print(f"\n  PARAMETROS:")
    print(f"    hmm_threshold: {HMM_THRESHOLD}")
    print(f"    lyapunov_threshold: {LYAPUNOV_THRESHOLD}")
    print(f"    stop_loss_pips: {STOP_LOSS_PIPS}")
    print(f"    take_profit_pips: {TAKE_PROFIT_PIPS}")
    print(f"    signal_cooldown: {SIGNAL_COOLDOWN}")
    print(f"    hmm_states_allowed: {HMM_STATES_ALLOWED}")
    print(f"    trend_lookback: {TREND_LOOKBACK}")
    print(f"    warmup_bars: {MIN_PRICES_WARMUP}")

    # Calcular periodo necessario
    # 6624 barras M5 = ~23 dias
    # Queremos mais para ter dados de trading
    total_days = 60  # 60 dias = ~17000 barras M5

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=total_days)

    print(f"\n  Periodo: {start_time.date()} a {end_time.date()}")
    print(f"  Par: EURUSD M5")

    # Baixar dados
    print("\n  Baixando dados...")
    bars = get_historical_data_with_spread_sync(
        symbol="EURUSD",
        periodicity="M5",
        start_time=start_time,
        end_time=end_time
    )

    if not bars:
        print("  ERRO: Falha ao baixar dados")
        return

    print(f"  Barras baixadas: {len(bars)}")

    if len(bars) < MIN_PRICES_WARMUP + 100:
        print(f"  ERRO: Barras insuficientes. Precisa de pelo menos {MIN_PRICES_WARMUP + 100}")
        return

    # Estatisticas de spread
    spreads = [bar.spread_pips for bar in bars if bar.has_spread_data]
    if spreads:
        print(f"\n  Spread (pips):")
        print(f"    Min: {min(spreads):.2f}")
        print(f"    Max: {max(spreads):.2f}")
        print(f"    Media: {np.mean(spreads):.2f}")

    # Executar backtest
    print("\n" + "=" * 70)
    print("  EXECUTANDO BACKTEST")
    print("=" * 70)

    trades = run_backtest_with_warmup(bars, MIN_PRICES_WARMUP)

    # Resultados
    print("\n" + "=" * 70)
    print("  RESULTADOS")
    print("=" * 70)

    if not trades:
        print("\n  NENHUM TRADE GERADO!")
        print("  Possivel causa: thresholds muito restritivos")
        print(f"    hmm_threshold = {HMM_THRESHOLD} (muito alto?)")
        print(f"    lyapunov_threshold = {LYAPUNOV_THRESHOLD}")
        return

    # Metricas
    pnls = [t.pnl_pips for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_pnl_pips = sum(pnls)

    usd_per_pip = 0.10  # 0.01 lotes EURUSD
    total_pnl_usd = total_pnl_pips * usd_per_pip

    win_rate = len(wins) / len(trades) if trades else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss

    # Breakeven para TP=SL=10
    breakeven_wr = STOP_LOSS_PIPS / (TAKE_PROFIT_PIPS + STOP_LOSS_PIPS)

    print(f"\n  TRADES:")
    print(f"    Total: {len(trades)}")
    print(f"    Vencedores: {len(wins)}")
    print(f"    Perdedores: {len(losses)}")
    print(f"    Win Rate: {win_rate:.1%}")
    print(f"    Breakeven WR: {breakeven_wr:.1%}")

    print(f"\n  PERFORMANCE:")
    print(f"    PnL Total: ${total_pnl_usd:.2f} ({total_pnl_pips:.1f} pips)")
    print(f"    Profit Factor: {profit_factor:.2f}")
    print(f"    Avg Trade: {np.mean(pnls):.1f} pips")
    print(f"    Max Win: {max(pnls):.1f} pips")
    print(f"    Max Loss: {min(pnls):.1f} pips")

    # Drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    drawdowns = peak - equity
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

    print(f"\n  RISCO:")
    print(f"    Max Drawdown: {max_dd:.1f} pips")

    # Por tipo de saida
    stops = [t for t in trades if t.exit_reason == "stop_loss"]
    takes = [t for t in trades if t.exit_reason == "take_profit"]
    print(f"\n  SAIDAS:")
    print(f"    Stop Loss: {len(stops)} ({len(stops)/len(trades)*100:.0f}%)")
    print(f"    Take Profit: {len(takes)} ({len(takes)/len(trades)*100:.0f}%)")

    # Ultimos trades
    print(f"\n  ULTIMOS 10 TRADES:")
    for trade in trades[-10:]:
        pnl_sign = "+" if trade.pnl_pips > 0 else ""
        print(f"    {trade.entry_time.strftime('%Y-%m-%d %H:%M')} | "
              f"{trade.type.name:5} | "
              f"Entry: {trade.entry_price:.5f} | "
              f"Exit: {trade.exit_price:.5f} | "
              f"PnL: {pnl_sign}{trade.pnl_pips:.1f} pips | "
              f"{trade.exit_reason}")

    # Verificar se lucrativo
    print("\n" + "=" * 70)
    print("  CONCLUSAO")
    print("=" * 70)

    if total_pnl_pips > 0:
        print(f"\n  ESTRATEGIA LUCRATIVA!")
        print(f"    Lucro: ${total_pnl_usd:.2f} ({total_pnl_pips:.1f} pips)")
    else:
        print(f"\n  ESTRATEGIA NAO LUCRATIVA")
        print(f"    Prejuizo: ${total_pnl_usd:.2f} ({total_pnl_pips:.1f} pips)")
        if win_rate < breakeven_wr:
            print(f"    Win Rate ({win_rate:.1%}) abaixo do breakeven ({breakeven_wr:.1%})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
