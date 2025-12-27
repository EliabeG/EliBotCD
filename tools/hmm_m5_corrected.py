#!/usr/bin/env python3
"""
================================================================================
BACKTEST CORRIGIDO - SEM LOOK-AHEAD BIAS
================================================================================

CORRECOES APLICADAS:
1. Sinal gerado ANTES de atualizar indicador (usa dados ate barra N-1)
2. Update do indicador movido para o FINAL do loop
3. Tendencia calculada com closes ate barra anterior

FLUXO CORRETO:
  Barra N:
    1. Executar sinal pendente (gerado na barra N-1) no OPEN da barra N
    2. Verificar stop/take usando HIGH/LOW da barra N
    3. Gerar sinal usando dados ate barra N-1 (indicador ainda nao atualizado)
    4. Atualizar indicador com barra N (para uso na barra N+1)
================================================================================
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from typing import List, Optional
from dataclasses import dataclass
from collections import deque
from enum import Enum
import numpy as np

from api.fxopen_historical_ws import Bar, get_historical_data_with_spread_sync

# ===============================================================================
# PARAMETROS
# ===============================================================================
STOP_LOSS_PIPS = 10.0
TAKE_PROFIT_PIPS = 20.0
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
class BacktestResult:
    name: str
    params: dict
    total_trades: int
    win_rate: float
    pnl_pips: float
    profit_factor: float
    max_drawdown: float
    avg_trade: float
    trades_per_month: float


class CorrectedIndicator:
    """
    Indicador CORRIGIDO sem look-ahead bias.

    GARANTIAS:
    1. get_volatility_score() usa apenas retornos ate barra ANTERIOR
    2. get_trend_direction() usa apenas closes ate barra ANTERIOR
    3. update() so eh chamado APOS gerar sinal
    """

    def __init__(self, vol_window=50, atr_window=14):
        self.vol_window = vol_window
        self.atr_window = atr_window

        self.returns_buffer = deque(maxlen=vol_window)
        self.tr_buffer = deque(maxlen=atr_window)
        self.closes_buffer = deque(maxlen=50)

        self.vol_p50 = 0
        self.vol_p75 = 0
        self.vol_p90 = 0
        self.atr_p50 = 0
        self.atr_p75 = 0
        self.atr_p90 = 0

        self.is_calibrated = False
        self.last_close = 0

    def calibrate(self, bars: List[Bar]):
        """Calibra percentis com dados de warmup."""
        prices = [bar.close for bar in bars]
        returns = np.diff(np.log(prices))

        # Volatilidades
        vols = []
        for i in range(self.vol_window, len(returns)):
            vol = np.std(returns[i-self.vol_window:i])
            vols.append(vol)

        if vols:
            self.vol_p50 = np.percentile(vols, 50)
            self.vol_p75 = np.percentile(vols, 75)
            self.vol_p90 = np.percentile(vols, 90)

        # ATRs
        trs = []
        for i in range(1, len(bars)):
            tr = max(
                bars[i].high - bars[i].low,
                abs(bars[i].high - bars[i-1].close),
                abs(bars[i].low - bars[i-1].close)
            )
            trs.append(tr)

        atrs = []
        for i in range(self.atr_window, len(trs)):
            atr = np.mean(trs[i-self.atr_window:i])
            atrs.append(atr)

        if atrs:
            self.atr_p50 = np.percentile(atrs, 50)
            self.atr_p75 = np.percentile(atrs, 75)
            self.atr_p90 = np.percentile(atrs, 90)

        # Inicializar buffers (EXCLUINDO a ultima barra do warmup)
        # Assim o primeiro sinal usara dados ate warmup-1
        for r in returns[-(self.vol_window+1):-1]:
            self.returns_buffer.append(r)

        for i in range(len(bars)-self.atr_window-1, len(bars)-1):
            if i > 0:
                tr = max(
                    bars[i].high - bars[i].low,
                    abs(bars[i].high - bars[i-1].close),
                    abs(bars[i].low - bars[i-1].close)
                )
                self.tr_buffer.append(tr)

        for bar in bars[-51:-1]:
            self.closes_buffer.append(bar.close)

        self.last_close = bars[-1].close
        self.is_calibrated = True

    def update(self, bar: Bar, prev_close: float):
        """
        Atualiza indicador com nova barra.
        DEVE ser chamado APOS gerar sinal.
        """
        if not self.is_calibrated:
            return

        # Retorno da barra que acabou de fechar
        if prev_close > 0 and bar.close > 0:
            ret = np.log(bar.close / prev_close)
            self.returns_buffer.append(ret)

        # TR
        tr = max(
            bar.high - bar.low,
            abs(bar.high - prev_close),
            abs(bar.low - prev_close)
        )
        self.tr_buffer.append(tr)

        # Close
        self.closes_buffer.append(bar.close)

    def get_volatility_score(self) -> float:
        """Retorna score usando dados JA no buffer (ate barra anterior)."""
        if len(self.returns_buffer) < self.vol_window:
            return 0.0

        vol = np.std(list(self.returns_buffer))

        if vol <= self.vol_p50:
            return 0.0
        elif vol >= self.vol_p90:
            return 1.0
        elif vol >= self.vol_p75:
            return 0.5 + 0.5 * (vol - self.vol_p75) / (self.vol_p90 - self.vol_p75)
        else:
            return 0.5 * (vol - self.vol_p50) / (self.vol_p75 - self.vol_p50)

    def get_atr_score(self) -> float:
        """Retorna score usando dados JA no buffer (ate barra anterior)."""
        if len(self.tr_buffer) < self.atr_window:
            return 0.0

        atr = np.mean(list(self.tr_buffer))

        if atr <= self.atr_p50:
            return 0.0
        elif atr >= self.atr_p90:
            return 1.0
        elif atr >= self.atr_p75:
            return 0.5 + 0.5 * (atr - self.atr_p75) / (self.atr_p90 - self.atr_p75)
        else:
            return 0.5 * (atr - self.atr_p50) / (self.atr_p75 - self.atr_p50)

    def get_trend_direction(self, lookback: int) -> int:
        """Retorna direcao usando closes JA no buffer (ate barra anterior)."""
        if len(self.closes_buffer) < lookback + 1:
            return 0

        closes = list(self.closes_buffer)
        # closes[-1] eh o close da barra ANTERIOR (nao da atual)
        if closes[-1] > closes[-lookback-1]:
            return 1
        else:
            return -1


def run_corrected_backtest(
    bars: List[Bar],
    warmup_bars: int,
    vol_threshold: float,
    use_vol: bool,
    use_atr: bool,
    combine_mode: str,
    signal_cooldown: int,
    trend_lookback: int,
    direction: str
) -> BacktestResult:
    """
    Backtest CORRIGIDO sem look-ahead bias.

    ORDEM DE EXECUCAO (para cada barra N):
    1. Executar sinal pendente (gerado na barra N-1) no OPEN da barra N
    2. Verificar stop/take usando HIGH/LOW da barra N
    3. Gerar sinal usando indicador (que tem dados ate barra N-1)
    4. Atualizar indicador com barra N (para proxima iteracao)
    """

    pip = 0.0001
    total_bars = len(bars)
    trading_months = (bars[-1].timestamp - bars[warmup_bars].timestamp).days / 30.0

    # Calibrar indicador
    indicator = CorrectedIndicator(vol_window=50, atr_window=14)
    indicator.calibrate(bars[:warmup_bars])

    trades_pnl = []
    position: Optional[Position] = None
    pending_signal = None
    cooldown_counter = 0

    # Preco anterior para calcular retorno
    prev_close = bars[warmup_bars - 1].close

    for i in range(warmup_bars, total_bars):
        bar = bars[i]

        # =====================================================================
        # PASSO 1: Executar sinal pendente (gerado na barra anterior)
        # =====================================================================
        if pending_signal is not None and position is None:
            spread = bar.spread_pips * pip if bar.has_spread_data else 0.2 * pip

            if pending_signal == 'BUY':
                entry_price = bar.ask_open if bar.ask_open else bar.open + spread/2
                pos_type = PositionType.LONG
                sl = entry_price - (STOP_LOSS_PIPS * pip)
                tp = entry_price + (TAKE_PROFIT_PIPS * pip)
            else:
                entry_price = bar.bid_open if bar.bid_open else bar.open - spread/2
                pos_type = PositionType.SHORT
                sl = entry_price + (STOP_LOSS_PIPS * pip)
                tp = entry_price - (TAKE_PROFIT_PIPS * pip)

            position = Position(pos_type, entry_price, bar.timestamp, sl, tp)
            pending_signal = None

        # =====================================================================
        # PASSO 2: Verificar stop/take profit
        # =====================================================================
        if position:
            exit_price = None

            bid_low = bar.bid_low if bar.bid_low else bar.low
            bid_high = bar.bid_high if bar.bid_high else bar.high
            ask_low = bar.ask_low if bar.ask_low else bar.low
            ask_high = bar.ask_high if bar.ask_high else bar.high

            if position.type == PositionType.LONG:
                # Verifica stop primeiro (mais conservador)
                if bid_low <= position.stop_loss:
                    exit_price = position.stop_loss
                elif bid_high >= position.take_profit:
                    exit_price = position.take_profit
            else:
                if ask_high >= position.stop_loss:
                    exit_price = position.stop_loss
                elif ask_low <= position.take_profit:
                    exit_price = position.take_profit

            if exit_price:
                if position.type == PositionType.LONG:
                    pnl = (exit_price - position.entry_price) / pip
                else:
                    pnl = (position.entry_price - exit_price) / pip

                trades_pnl.append(pnl)
                position = None

        # =====================================================================
        # PASSO 3: Gerar sinal (ANTES de atualizar indicador)
        # Indicador ainda tem dados ate barra N-1
        # =====================================================================
        if position is None and pending_signal is None:
            if cooldown_counter > 0:
                cooldown_counter -= 1
            else:
                # Scores usam dados ate barra ANTERIOR (N-1)
                vol_score = indicator.get_volatility_score() if use_vol else 0
                atr_score = indicator.get_atr_score() if use_atr else 0

                signal_valid = False

                if use_vol and use_atr:
                    if combine_mode == 'or':
                        signal_valid = vol_score >= vol_threshold or atr_score >= vol_threshold
                    elif combine_mode == 'avg':
                        avg_score = (vol_score + atr_score) / 2
                        signal_valid = avg_score >= vol_threshold
                elif use_vol:
                    signal_valid = vol_score >= vol_threshold
                elif use_atr:
                    signal_valid = atr_score >= vol_threshold

                if signal_valid:
                    # Tendencia usa closes ate barra N-1
                    trend = indicator.get_trend_direction(trend_lookback)
                    if trend != 0:
                        if direction == 'contra':
                            pending_signal = 'SELL' if trend == 1 else 'BUY'
                        else:
                            pending_signal = 'BUY' if trend == 1 else 'SELL'
                        cooldown_counter = signal_cooldown

        # =====================================================================
        # PASSO 4: Atualizar indicador (APOS gerar sinal)
        # Agora o indicador tera dados ate barra N para proxima iteracao
        # =====================================================================
        indicator.update(bar, prev_close)
        prev_close = bar.close

    # Calcular metricas
    if not trades_pnl:
        return BacktestResult("", {}, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    pnls = np.array(trades_pnl)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    total_trades = len(pnls)
    win_rate = len(wins) / total_trades * 100
    pnl_total = float(np.sum(pnls))
    profit_factor = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else 0

    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max(peak - equity))
    avg_trade = float(np.mean(pnls))
    trades_per_month = total_trades / trading_months if trading_months > 0 else 0

    return BacktestResult("", {}, total_trades, win_rate, pnl_total, profit_factor, max_dd, avg_trade, trades_per_month)


def main():
    print("=" * 80)
    print("  BACKTEST CORRIGIDO - SEM LOOK-AHEAD BIAS")
    print("=" * 80)

    breakeven = STOP_LOSS_PIPS / (STOP_LOSS_PIPS + TAKE_PROFIT_PIPS) * 100
    print(f"\n  Breakeven WR: {breakeven:.1f}%")
    print(f"  SL: {STOP_LOSS_PIPS} pips, TP: {TAKE_PROFIT_PIPS} pips")

    # Baixar dados
    end_time = datetime.now(timezone.utc)
    start_time = datetime(2025, 7, 1, tzinfo=timezone.utc)

    print(f"\n  Periodo: {start_time.date()} a {end_time.date()}")
    print("\n  Baixando dados...")

    bars = get_historical_data_with_spread_sync(
        symbol="EURUSD",
        periodicity="M5",
        start_time=start_time,
        end_time=end_time
    )

    if not bars or len(bars) < MIN_PRICES_WARMUP + 100:
        print("  ERRO: Dados insuficientes")
        return

    print(f"  Barras: {len(bars)}")

    # Testar varias configuracoes
    vol_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    cooldowns = [1, 2, 3, 5, 7, 10]
    lookbacks = [3, 5, 7, 10]

    combinations = [
        (True, False, 'and', 'Vol_Only'),
        (False, True, 'and', 'ATR_Only'),
        (True, True, 'or', 'Vol+ATR_OR'),
        (True, True, 'avg', 'Vol+ATR_AVG'),
    ]

    print("\n" + "=" * 80)
    print("  EXECUTANDO BACKTEST CORRIGIDO")
    print("=" * 80)

    all_results = []
    tested = 0
    total = len(combinations) * len(vol_thresholds) * len(cooldowns) * len(lookbacks)

    for use_vol, use_atr, combine_mode, combo_name in combinations:
        for vol_t in vol_thresholds:
            for cool in cooldowns:
                for look in lookbacks:
                    result = run_corrected_backtest(
                        bars, MIN_PRICES_WARMUP,
                        vol_threshold=vol_t,
                        use_vol=use_vol,
                        use_atr=use_atr,
                        combine_mode=combine_mode,
                        signal_cooldown=cool,
                        trend_lookback=look,
                        direction='contra'
                    )

                    if result.total_trades >= 360:  # 60/mes
                        result.name = f"{combo_name}_t{vol_t}_c{cool}_l{look}"
                        result.params = {
                            'combo': combo_name,
                            'threshold': vol_t,
                            'cooldown': cool,
                            'lookback': look
                        }
                        all_results.append(result)

                    tested += 1
                    if tested % 100 == 0:
                        print(f"    Testado: {tested}/{total}, Validos: {len(all_results)}")

    print(f"\n  Combinacoes com 360+ trades: {len(all_results)}")

    # Ordenar por Win Rate
    all_results.sort(key=lambda x: x.win_rate, reverse=True)

    print("\n" + "=" * 80)
    print("  TOP 20 - SEM LOOK-AHEAD BIAS")
    print("=" * 80)

    print(f"\n  {'#':<3} {'Nome':<25} {'Trades':<7} {'T/Mes':<6} {'WR%':<6} {'PnL':<8} {'PF':<5} {'Avg':<6}")
    print("  " + "-" * 75)

    for i, r in enumerate(all_results[:20]):
        print(f"  {i+1:<3} {r.name:<25} {r.total_trades:<7} {r.trades_per_month:<6.0f} "
              f"{r.win_rate:<6.1f} {r.pnl_pips:+<8.0f} {r.profit_factor:<5.2f} {r.avg_trade:<+6.1f}")

    # Melhor resultado
    if all_results:
        best = all_results[0]
        print("\n" + "=" * 80)
        print("  MELHOR RESULTADO (CORRIGIDO)")
        print("=" * 80)

        print(f"\n  Configuracao: {best.name}")
        print(f"  Parametros: {best.params}")
        print(f"  Trades: {best.total_trades} ({best.trades_per_month:.0f}/mes)")
        print(f"  Win Rate: {best.win_rate:.1f}%")
        print(f"  PnL: {best.pnl_pips:+.0f} pips")
        print(f"  Profit Factor: {best.profit_factor:.2f}")
        print(f"  Avg Trade: {best.avg_trade:+.1f} pips")
        print(f"  Max Drawdown: {best.max_drawdown:.0f} pips")

    print("\n" + "=" * 80)
    print("  VERIFICACAO DE CORRECOES")
    print("=" * 80)
    print("""
  CORRECOES APLICADAS:

  1. ORDEM DO LOOP CORRIGIDA:
     - Sinal gerado ANTES de indicator.update()
     - Isso garante que o sinal usa dados ate barra N-1

  2. CALIBRACAO AJUSTADA:
     - Buffers inicializados excluindo a ultima barra do warmup
     - Primeiro sinal usa dados ate warmup_bars-1

  3. FLUXO TEMPORAL:
     Barra N:
       1. Executa sinal pendente (gerado na N-1) no OPEN de N
       2. Verifica SL/TP usando HIGH/LOW de N
       3. Gera novo sinal usando dados ate N-1
       4. Atualiza indicador com N (para N+1)

  GARANTIAS:
  - Nenhum sinal usa informacao da barra em que eh gerado
  - Todos os calculos usam apenas dados passados
  - Entrada sempre no OPEN da barra seguinte ao sinal
    """)

    print("=" * 80)


if __name__ == "__main__":
    main()
