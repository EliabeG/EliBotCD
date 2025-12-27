#!/usr/bin/env python3
"""
================================================================================
BACKTEST PRM INCREMENTAL - Calculo otimizado sem recalcular tudo
================================================================================

Otimizacoes:
1. HMM treinado uma vez no warmup
2. Buffer de retornos cresce incrementalmente
3. A cada barra, apenas adiciona novo retorno (nao recalcula historico)
4. Lyapunov calculado apenas sobre ultimas 50 barras

Parametros:
- hmm_threshold = 0.6
- stop_loss_pips = 10
- take_profit_pips = 20
- Breakeven WR = 10/(10+20) = 33.3%
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
from collections import deque
import numpy as np

from api.fxopen_historical_ws import Bar, get_historical_data_with_spread_sync

try:
    from hmmlearn import hmm
except ImportError:
    print("ERRO: hmmlearn nao instalado")
    sys.exit(1)

# ===============================================================================
# PARAMETROS DO USUARIO
# ===============================================================================
# NOTA: hmm_threshold=0.6 eh IMPOSSIVEL (max eh 0.4451)
# Usando 0.44 que corresponde ao P90 (10% das barras passam)
HMM_THRESHOLD = 0.44
LYAPUNOV_THRESHOLD = 0.5  # Ajustado para gerar sinais
STOP_LOSS_PIPS = 10.0
TAKE_PROFIT_PIPS = 20.0
SIGNAL_COOLDOWN = 10
TREND_LOOKBACK = 10
MIN_PRICES_WARMUP = 6624
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


class IncrementalPRMIndicator:
    """
    PRM com calculo incremental - nao recalcula historico completo
    """

    def __init__(self, n_states=3, hmm_window=200, lyap_window=50):
        self.n_states = n_states
        self.hmm_window = hmm_window
        self.lyap_window = lyap_window

        # HMM model
        self.hmm_model = None
        self.is_trained = False

        # Buffer incremental de retornos (ultimos hmm_window retornos)
        self.returns_buffer = deque(maxlen=hmm_window)

        # Buffer de precos para Lyapunov (ultimos lyap_window + 1 precos)
        self.prices_buffer = deque(maxlen=lyap_window + 10)

        # Cache do ultimo resultado
        self.last_hmm_prob = 0.0
        self.last_lyapunov = 0.0

    def initialize_buffers(self, prices: np.ndarray):
        """Inicializa buffers com dados de warmup"""
        # Calcular todos os retornos do warmup
        returns = np.diff(np.log(prices))

        # Preencher buffer de retornos (ultimos hmm_window)
        self.returns_buffer.clear()
        for r in returns[-self.hmm_window:]:
            self.returns_buffer.append(r)

        # Preencher buffer de precos (ultimos lyap_window + 10)
        self.prices_buffer.clear()
        for p in prices[-(self.lyap_window + 10):]:
            self.prices_buffer.append(p)

    def train_hmm(self, prices: np.ndarray):
        """Treina HMM com dados de warmup"""
        if len(prices) < self.hmm_window:
            return False

        # Calcular retornos
        returns = np.diff(np.log(prices))
        returns = returns.reshape(-1, 1)

        try:
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            self.hmm_model.fit(returns)
            self.is_trained = True

            # Inicializar buffers
            self.initialize_buffers(prices)

            return True
        except Exception as e:
            print(f"  Erro ao treinar HMM: {e}")
            return False

    def add_price(self, price: float):
        """
        Adiciona novo preco e atualiza buffers INCREMENTALMENTE
        - Nao recalcula historico
        - Apenas adiciona nova informacao
        """
        if len(self.prices_buffer) > 0:
            # Calcular novo retorno
            last_price = self.prices_buffer[-1]
            if last_price > 0 and price > 0:
                new_return = np.log(price / last_price)
                self.returns_buffer.append(new_return)

        # Adicionar preco ao buffer
        self.prices_buffer.append(price)

    def compute_hmm_prob(self) -> float:
        """Calcula probabilidade HMM usando buffer de retornos"""
        if not self.is_trained or len(self.returns_buffer) < 10:
            return 0.0

        try:
            # Converter buffer para array
            returns = np.array(list(self.returns_buffer)).reshape(-1, 1)

            # Inferir probabilidades (sem re-treinar)
            probs = self.hmm_model.predict_proba(returns)
            current_probs = probs[-1]

            # Prob max de estados 1 e 2
            hmm_prob = max(
                current_probs[1] if len(current_probs) > 1 else 0,
                current_probs[2] if len(current_probs) > 2 else 0
            )

            self.last_hmm_prob = hmm_prob
            return hmm_prob

        except Exception:
            return self.last_hmm_prob

    def compute_lyapunov(self) -> float:
        """Calcula Lyapunov usando apenas ultimos precos do buffer"""
        if len(self.prices_buffer) < self.lyap_window + 1:
            return 0.0

        try:
            # Usar apenas ultimos precos do buffer
            prices = list(self.prices_buffer)[-self.lyap_window - 1:]
            returns = np.diff(np.log(prices))

            abs_returns = np.abs(returns)
            abs_returns = abs_returns[abs_returns > 1e-10]

            if len(abs_returns) == 0:
                return 0.0

            lyapunov = np.mean(np.log(abs_returns)) + 10
            lyapunov = max(0, lyapunov)

            self.last_lyapunov = lyapunov
            return lyapunov

        except Exception:
            return self.last_lyapunov

    def analyze(self) -> dict:
        """Analisa estado atual usando buffers incrementais"""
        hmm_prob = self.compute_hmm_prob()
        lyapunov = self.compute_lyapunov()

        return {
            'hmm_prob': hmm_prob,
            'lyapunov': lyapunov,
            'valid': self.is_trained and len(self.returns_buffer) >= 10
        }


def calculate_direction(closes: deque, lookback: int) -> int:
    """Calcula direcao baseada em tendencia"""
    min_bars = lookback + 2
    if len(closes) < min_bars:
        return 0

    closes_list = list(closes)
    recent = closes_list[-2]
    past = closes_list[-(lookback + 2)]
    trend = recent - past

    return 1 if trend > 0 else -1


def run_incremental_backtest(bars: List[Bar], warmup_bars: int) -> List[Trade]:
    """Backtest com calculo incremental"""
    trades = []
    position: Optional[Position] = None
    pip = 0.0001
    signal_cooldown = 0

    # Indicador incremental
    prm = IncrementalPRMIndicator(n_states=3, hmm_window=200, lyap_window=50)

    total_bars = len(bars)

    print(f"\n  Total de barras: {total_bars}")
    print(f"  Barras de warmup: {warmup_bars}")
    print(f"  Barras de trading: {total_bars - warmup_bars}")

    # =========================================================================
    # WARMUP - Treinar HMM uma vez
    # =========================================================================
    print(f"\n  [WARMUP] Treinando HMM com {warmup_bars} barras...")
    warmup_prices = np.array([bar.close for bar in bars[:warmup_bars]])

    if not prm.train_hmm(warmup_prices):
        print("  ERRO: Falha ao treinar HMM")
        return []

    print(f"  [WARMUP] HMM treinado!")
    print(f"  [WARMUP] Buffers inicializados (retornos: {len(prm.returns_buffer)}, precos: {len(prm.prices_buffer)})")

    # Buffer de closes para direcao
    closes_buffer = deque(maxlen=500)
    for bar in bars[:warmup_bars]:
        closes_buffer.append(bar.close)

    # =========================================================================
    # TRADING - Calculo incremental
    # =========================================================================
    print(f"\n  [TRADING] Iniciando com calculo incremental...")
    print(f"  [TRADING] Breakeven WR = {STOP_LOSS_PIPS/(STOP_LOSS_PIPS+TAKE_PROFIT_PIPS)*100:.1f}%")
    print()

    pending_signal = None
    signals_generated = 0
    start_time = datetime.now()

    # Estatisticas em tempo real
    running_pnl = 0.0
    running_wins = 0
    running_losses = 0

    for i in range(warmup_bars, total_bars):
        bar = bars[i]

        # INCREMENTAL: Adicionar apenas o novo preco
        prm.add_price(bar.close)
        closes_buffer.append(bar.close)

        # 1. Executar sinal pendente
        if pending_signal and position is None:
            spread = bar.spread_pips * pip if bar.has_spread_data else 0.2 * pip

            if pending_signal == 'BUY':
                entry_price = (bar.ask_open if bar.ask_open else bar.open + spread/2)
                pos_type = PositionType.LONG
                sl = entry_price - (STOP_LOSS_PIPS * pip)
                tp = entry_price + (TAKE_PROFIT_PIPS * pip)
            else:
                entry_price = (bar.bid_open if bar.bid_open else bar.open - spread/2)
                pos_type = PositionType.SHORT
                sl = entry_price + (STOP_LOSS_PIPS * pip)
                tp = entry_price - (TAKE_PROFIT_PIPS * pip)

            position = Position(pos_type, entry_price, bar.timestamp, sl, tp)
            pending_signal = None

        # 2. Verificar stop/take
        if position:
            exit_price = None
            exit_reason = None

            bid_low = bar.bid_low if bar.bid_low else bar.low
            ask_high = bar.ask_high if bar.ask_high else bar.high

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
                    position.type, position.entry_price, position.entry_time,
                    exit_price, bar.timestamp, pnl_pips, exit_reason
                ))

                # Atualizar estatisticas
                running_pnl += pnl_pips
                if pnl_pips > 0:
                    running_wins += 1
                else:
                    running_losses += 1

                position = None

        # 3. Gerar sinal (calculo incremental - muito rapido)
        if position is None and pending_signal is None:
            if signal_cooldown > 0:
                signal_cooldown -= 1
            else:
                result = prm.analyze()

                if result['valid']:
                    if (result['hmm_prob'] >= HMM_THRESHOLD and
                        result['lyapunov'] >= LYAPUNOV_THRESHOLD):

                        direction = calculate_direction(closes_buffer, TREND_LOOKBACK)
                        if direction != 0:
                            # INVERTIDO: contra-tendencia
                            pending_signal = 'SELL' if direction == 1 else 'BUY'
                            signal_cooldown = SIGNAL_COOLDOWN
                            signals_generated += 1

        # Progresso em tempo real
        trading_bar = i - warmup_bars + 1
        total_trading = total_bars - warmup_bars
        if trading_bar % 2000 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            bars_per_sec = trading_bar / elapsed if elapsed > 0 else 0
            total_trades = running_wins + running_losses
            win_rate = running_wins / total_trades * 100 if total_trades > 0 else 0

            print(f"    Barra {trading_bar}/{total_trading} | "
                  f"{bars_per_sec:.0f} barras/s | "
                  f"Trades: {total_trades} | "
                  f"WR: {win_rate:.1f}% | "
                  f"PnL: {running_pnl:+.1f} pips")

    # Fechar posicao aberta
    if position:
        last_bar = bars[-1]
        if position.type == PositionType.LONG:
            exit_price = last_bar.bid_close if last_bar.bid_close else last_bar.close
            pnl_pips = (exit_price - position.entry_price) / pip
        else:
            exit_price = last_bar.ask_close if last_bar.ask_close else last_bar.close
            pnl_pips = (position.entry_price - exit_price) / pip

        trades.append(Trade(
            position.type, position.entry_price, position.entry_time,
            exit_price, last_bar.timestamp, pnl_pips, "end_of_data"
        ))

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n  [TRADING] Concluido em {elapsed:.1f} segundos")
    print(f"  [TRADING] Sinais gerados: {signals_generated}")

    return trades


def main():
    print("=" * 70)
    print("  BACKTEST PRM INCREMENTAL")
    print("  Calculo otimizado - nao recalcula historico")
    print("=" * 70)

    breakeven = STOP_LOSS_PIPS / (STOP_LOSS_PIPS + TAKE_PROFIT_PIPS) * 100

    print(f"\n  PARAMETROS:")
    print(f"    hmm_threshold: {HMM_THRESHOLD}")
    print(f"    lyapunov_threshold: {LYAPUNOV_THRESHOLD}")
    print(f"    stop_loss_pips: {STOP_LOSS_PIPS}")
    print(f"    take_profit_pips: {TAKE_PROFIT_PIPS}")
    print(f"    signal_cooldown: {SIGNAL_COOLDOWN}")
    print(f"    trend_lookback: {TREND_LOOKBACK}")
    print(f"    warmup_bars: {MIN_PRICES_WARMUP}")
    print(f"    BREAKEVEN WR: {breakeven:.1f}%")

    # Periodo: 2025-07-01 ate hoje
    end_time = datetime.now(timezone.utc)
    start_time = datetime(2025, 7, 1, tzinfo=timezone.utc)

    print(f"\n  Periodo: {start_time.date()} a {end_time.date()}")

    # Baixar dados
    print("\n  Baixando dados com spread real...")
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
        print(f"\n  Spread (pips): Min={min(spreads):.2f}, Max={max(spreads):.2f}, Media={np.mean(spreads):.2f}")

    # Backtest
    print("\n" + "=" * 70)
    print("  EXECUTANDO BACKTEST INCREMENTAL")
    print("=" * 70)

    trades = run_incremental_backtest(bars, MIN_PRICES_WARMUP)

    # Resultados
    print("\n" + "=" * 70)
    print("  RESULTADOS FINAIS")
    print("=" * 70)

    if not trades:
        print("\n  NENHUM TRADE!")
        return

    pnls = [t.pnl_pips for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)

    win_rate = len(wins) / len(trades) * 100
    profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')

    print(f"\n  TRADES:")
    print(f"    Total: {len(trades)}")
    print(f"    Vencedores: {len(wins)}")
    print(f"    Perdedores: {len(losses)}")
    print(f"    Win Rate: {win_rate:.1f}%")
    print(f"    Breakeven: {breakeven:.1f}%")

    print(f"\n  PERFORMANCE:")
    print(f"    PnL Total: ${total_pnl * 0.10:.2f} ({total_pnl:.1f} pips)")
    print(f"    Profit Factor: {profit_factor:.2f}")
    print(f"    Avg Trade: {np.mean(pnls):.1f} pips")
    print(f"    Max Win: {max(pnls):.1f} pips")
    print(f"    Max Loss: {min(pnls):.1f} pips")

    # Drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    max_dd = np.max(peak - equity)

    print(f"\n  RISCO:")
    print(f"    Max Drawdown: {max_dd:.1f} pips (${max_dd * 0.10:.2f})")

    stops = len([t for t in trades if t.exit_reason == "stop_loss"])
    takes = len([t for t in trades if t.exit_reason == "take_profit"])
    print(f"\n  SAIDAS:")
    print(f"    Stop Loss: {stops} ({stops/len(trades)*100:.0f}%)")
    print(f"    Take Profit: {takes} ({takes/len(trades)*100:.0f}%)")

    # Analise por mes
    print(f"\n  PERFORMANCE POR MES:")
    trades_by_month = {}
    for t in trades:
        month_key = t.entry_time.strftime("%Y-%m")
        if month_key not in trades_by_month:
            trades_by_month[month_key] = []
        trades_by_month[month_key].append(t.pnl_pips)

    for month in sorted(trades_by_month.keys()):
        month_pnls = trades_by_month[month]
        month_wins = len([p for p in month_pnls if p > 0])
        month_total = len(month_pnls)
        month_wr = month_wins / month_total * 100 if month_total > 0 else 0
        month_pnl = sum(month_pnls)
        print(f"    {month}: {month_total} trades, WR={month_wr:.0f}%, PnL={month_pnl:+.0f} pips")

    # Ultimos trades
    print(f"\n  ULTIMOS 10 TRADES:")
    for t in trades[-10:]:
        sign = "+" if t.pnl_pips > 0 else ""
        print(f"    {t.entry_time.strftime('%Y-%m-%d %H:%M')} | "
              f"{t.type.name:5} | {sign}{t.pnl_pips:.0f} pips | {t.exit_reason}")

    # Conclusao
    print("\n" + "=" * 70)
    print("  CONCLUSAO")
    print("=" * 70)

    if win_rate > breakeven:
        edge = win_rate - breakeven
        print(f"\n  ESTRATEGIA LUCRATIVA!")
        print(f"    Win Rate ({win_rate:.1f}%) > Breakeven ({breakeven:.1f}%)")
        print(f"    Edge: +{edge:.1f}%")
        print(f"    Lucro: ${total_pnl * 0.10:.2f} ({total_pnl:.0f} pips)")
    else:
        deficit = breakeven - win_rate
        print(f"\n  ESTRATEGIA NAO LUCRATIVA")
        print(f"    Win Rate ({win_rate:.1f}%) < Breakeven ({breakeven:.1f}%)")
        print(f"    Deficit: -{deficit:.1f}%")
        print(f"    Prejuizo: ${total_pnl * 0.10:.2f} ({total_pnl:.0f} pips)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
