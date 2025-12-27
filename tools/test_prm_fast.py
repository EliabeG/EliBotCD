#!/usr/bin/env python3
"""
================================================================================
BACKTEST PRM RAPIDO - HMM treinado apenas uma vez no warmup
================================================================================

Parametros solicitados pelo usuario:
- hmm_threshold = 0.9
- lyapunov_threshold = 0.1
- stop_loss_pips = 10
- take_profit_pips = 10
- signal_cooldown = 10
- hmm_states_allowed = [1, 2]
- trend_lookback = 10
- min_prices (warmup) = 6624
- periodicity = M5

OTIMIZACAO: O HMM e treinado APENAS uma vez com as barras de warmup.
Depois disso, apenas faz inferencia (sem re-treinar).
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
from collections import deque

from api.fxopen_historical_ws import Bar, get_historical_data_with_spread_sync

# Importar apenas o necessario do PRM
try:
    from hmmlearn import hmm
except ImportError:
    print("ERRO: hmmlearn nao instalado. Execute: pip install hmmlearn")
    sys.exit(1)

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


class FastPRMIndicator:
    """
    Indicador PRM otimizado - treina HMM apenas uma vez
    """

    def __init__(self, n_states=3, hmm_training_window=200):
        self.n_states = n_states
        self.hmm_training_window = hmm_training_window
        self.hmm_model = None
        self.is_trained = False

    def train_hmm(self, prices: np.ndarray):
        """Treina o HMM com os dados fornecidos"""
        if len(prices) < self.hmm_training_window:
            print(f"  Dados insuficientes para treino: {len(prices)} < {self.hmm_training_window}")
            return False

        # Calcular retornos
        returns = np.diff(np.log(prices))
        returns = returns.reshape(-1, 1)

        try:
            # Criar e treinar modelo HMM
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            self.hmm_model.fit(returns)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"  Erro ao treinar HMM: {e}")
            return False

    def compute_lyapunov(self, prices: np.ndarray, window: int = 50) -> float:
        """Calcula expoente de Lyapunov simplificado"""
        if len(prices) < window + 1:
            return 0.0

        returns = np.diff(np.log(prices[-window-1:]))
        if len(returns) < 2:
            return 0.0

        # Calculo simplificado do Lyapunov
        abs_returns = np.abs(returns)
        abs_returns = abs_returns[abs_returns > 1e-10]
        if len(abs_returns) == 0:
            return 0.0

        lyapunov = np.mean(np.log(abs_returns))
        return max(0, lyapunov + 10)  # Normalizar para valor positivo

    def analyze(self, prices: np.ndarray) -> dict:
        """Analisa usando HMM pre-treinado"""
        result = {
            'hmm_prob': 0.0,
            'hmm_state': 0,
            'lyapunov': 0.0,
            'valid': False
        }

        if not self.is_trained or self.hmm_model is None:
            return result

        if len(prices) < 10:
            return result

        try:
            # Calcular retornos
            returns = np.diff(np.log(prices[-self.hmm_training_window:]))
            returns = returns.reshape(-1, 1)

            # Inferir estado (sem re-treinar)
            probs = self.hmm_model.predict_proba(returns)
            current_probs = probs[-1]

            # Probabilidade maxima dos estados 1 e 2
            hmm_prob = max(current_probs[1] if len(current_probs) > 1 else 0,
                          current_probs[2] if len(current_probs) > 2 else 0)

            # Estado atual
            hmm_state = np.argmax(current_probs)

            # Lyapunov
            lyapunov = self.compute_lyapunov(prices)

            result = {
                'hmm_prob': hmm_prob,
                'hmm_state': hmm_state,
                'lyapunov': lyapunov,
                'valid': True
            }

        except Exception:
            pass

        return result


def calculate_direction(closes: List[float], lookback: int) -> int:
    """Calcula direcao baseada em tendencia"""
    min_bars = lookback + 2
    if len(closes) < min_bars:
        return 0

    recent_close = closes[-2]
    past_close = closes[-(lookback + 2)]

    trend = recent_close - past_close

    return 1 if trend > 0 else -1


def run_fast_backtest(bars: List[Bar], warmup_bars: int) -> List[Trade]:
    """
    Backtest rapido - HMM treinado apenas uma vez
    """
    trades = []
    position: Optional[Position] = None
    pip = 0.0001
    signal_cooldown = 0

    # Indicador PRM otimizado
    prm = FastPRMIndicator(n_states=3, hmm_training_window=200)

    total_bars = len(bars)
    trading_start = warmup_bars

    print(f"\n  Total de barras: {total_bars}")
    print(f"  Barras de warmup: {warmup_bars}")
    print(f"  Barras de trading: {total_bars - warmup_bars}")

    # Fase de WARMUP - treinar HMM
    print(f"\n  [WARMUP] Preparando {warmup_bars} barras...")
    warmup_prices = np.array([bar.close for bar in bars[:warmup_bars]])
    warmup_closes = [bar.close for bar in bars[:warmup_bars]]

    print(f"  [WARMUP] Treinando HMM...")
    success = prm.train_hmm(warmup_prices)
    if not success:
        print("  ERRO: Falha ao treinar HMM")
        return []
    print(f"  [WARMUP] HMM treinado com sucesso!")

    # Buffer de precos (inicia com warmup)
    prices_buffer = list(warmup_prices)
    closes_buffer = list(warmup_closes)

    # Fase de TRADING
    print(f"\n  [TRADING] Iniciando operacoes...")

    pending_signal = None

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

            if pending_signal == 'BUY':
                if bar.ask_open:
                    entry_price = bar.ask_open + slippage
                else:
                    entry_price = bar.open + spread / 2 + slippage
                pos_type = PositionType.LONG
                sl = entry_price - (STOP_LOSS_PIPS * pip)
                tp = entry_price + (TAKE_PROFIT_PIPS * pip)
            else:
                if bar.bid_open:
                    entry_price = bar.bid_open - slippage
                else:
                    entry_price = bar.open - spread / 2 - slippage
                pos_type = PositionType.SHORT
                sl = entry_price + (STOP_LOSS_PIPS * pip)
                tp = entry_price - (TAKE_PROFIT_PIPS * pip)

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
                bid_low = bar.bid_low if bar.bid_low else bar.low
                ask_high = bar.ask_high if bar.ask_high else bar.high
            else:
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
            if signal_cooldown > 0:
                signal_cooldown -= 1
            else:
                # Usar ultimas 500 barras para analise
                prices_array = np.array(prices_buffer[-500:])
                result = prm.analyze(prices_array)

                if result['valid']:
                    # Filtrar por parametros do usuario
                    if (result['hmm_prob'] >= HMM_THRESHOLD and
                        result['lyapunov'] >= LYAPUNOV_THRESHOLD and
                        result['hmm_state'] in HMM_STATES_ALLOWED):

                        direction = calculate_direction(closes_buffer, TREND_LOOKBACK)
                        if direction != 0:
                            pending_signal = 'BUY' if direction == 1 else 'SELL'
                            signal_cooldown = SIGNAL_COOLDOWN

        # Progresso
        trading_bar = i - warmup_bars + 1
        total_trading = total_bars - warmup_bars
        if trading_bar % 1000 == 0:
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
    print("  BACKTEST PRM RAPIDO - HMM treinado uma vez")
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
    total_days = 90  # 90 dias = ~26000 barras M5

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

    trades = run_fast_backtest(bars, MIN_PRICES_WARMUP)

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
