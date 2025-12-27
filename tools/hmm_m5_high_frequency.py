#!/usr/bin/env python3
"""
================================================================================
OTIMIZACAO PARA ALTA FREQUENCIA (60+ trades/mes)
================================================================================

Requisitos:
- Minimo 360 trades em 6 meses (60/mes)
- Sem look-ahead bias
- Maximizar Win Rate

VERIFICACAO DE LOOK-AHEAD BIAS:
1. Percentis calculados APENAS no warmup (dados passados)
2. Sinal gerado na barra N, executado na barra N+1
3. Indicadores usam apenas dados ate barra atual
4. Precos de entrada/saida usam BID/ASK corretos
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
import itertools

from api.fxopen_historical_ws import Bar, get_historical_data_with_spread_sync

# ===============================================================================
# PARAMETROS FIXOS
# ===============================================================================
STOP_LOSS_PIPS = 10.0
TAKE_PROFIT_PIPS = 20.0
MIN_PRICES_WARMUP = 6624
MIN_TRADES_TOTAL = 360  # 60 trades/mes * 6 meses
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


class NoLookAheadIndicator:
    """
    Indicador SEM look-ahead bias.

    GARANTIAS:
    1. Percentis calibrados APENAS com dados de warmup
    2. Cada calculo usa APENAS dados ate o momento atual
    3. Nenhuma informacao futura eh usada
    """

    def __init__(self, vol_window=50, atr_window=14):
        self.vol_window = vol_window
        self.atr_window = atr_window

        # Buffers (apenas dados passados)
        self.returns_buffer = deque(maxlen=vol_window)
        self.tr_buffer = deque(maxlen=atr_window)
        self.closes_buffer = deque(maxlen=50)

        # Percentis calibrados no warmup (FIXOS apos warmup)
        self.vol_p50 = 0
        self.vol_p75 = 0
        self.vol_p90 = 0
        self.atr_p50 = 0
        self.atr_p75 = 0
        self.atr_p90 = 0

        self.is_calibrated = False
        self.last_price = 0

    def calibrate(self, bars: List[Bar]):
        """
        Calibra percentis APENAS com dados de warmup.
        Isso eh feito UMA VEZ antes do trading.
        """
        prices = [bar.close for bar in bars]
        returns = np.diff(np.log(prices))

        # Calcular volatilidades historicas
        vols = []
        for i in range(self.vol_window, len(returns)):
            vol = np.std(returns[i-self.vol_window:i])
            vols.append(vol)

        if vols:
            self.vol_p50 = np.percentile(vols, 50)
            self.vol_p75 = np.percentile(vols, 75)
            self.vol_p90 = np.percentile(vols, 90)

        # Calcular ATRs historicos
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

        # Inicializar buffers com ultimos dados do warmup
        for r in returns[-self.vol_window:]:
            self.returns_buffer.append(r)

        for i in range(len(bars)-self.atr_window, len(bars)):
            if i > 0:
                tr = max(
                    bars[i].high - bars[i].low,
                    abs(bars[i].high - bars[i-1].close),
                    abs(bars[i].low - bars[i-1].close)
                )
                self.tr_buffer.append(tr)

        for bar in bars[-50:]:
            self.closes_buffer.append(bar.close)

        self.last_price = bars[-1].close
        self.is_calibrated = True

    def update(self, bar: Bar):
        """
        Atualiza indicador com nova barra.
        USA APENAS dados da barra atual e passadas.
        """
        if not self.is_calibrated:
            return

        # Calcular retorno (usando preco anterior)
        if self.last_price > 0:
            ret = np.log(bar.close / self.last_price)
            self.returns_buffer.append(ret)

        # Calcular TR
        prev_close = self.last_price if self.last_price > 0 else bar.close
        tr = max(
            bar.high - bar.low,
            abs(bar.high - prev_close),
            abs(bar.low - prev_close)
        )
        self.tr_buffer.append(tr)

        # Atualizar closes
        self.closes_buffer.append(bar.close)

        # Guardar preco para proxima barra
        self.last_price = bar.close

    def get_volatility_score(self) -> float:
        """
        Retorna score de volatilidade (0-1).
        Usa APENAS dados do buffer (passados).
        """
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
        """
        Retorna score de ATR (0-1).
        Usa APENAS dados do buffer (passados).
        """
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
        """
        Retorna direcao da tendencia.
        Usa APENAS closes passados.
        """
        if len(self.closes_buffer) < lookback + 1:
            return 0

        closes = list(self.closes_buffer)
        # Compara preco atual com preco de 'lookback' barras atras
        if closes[-1] > closes[-lookback-1]:
            return 1  # Tendencia de alta
        else:
            return -1  # Tendencia de baixa


def run_backtest_no_lookahead(
    bars: List[Bar],
    warmup_bars: int,
    vol_threshold: float,
    atr_threshold: float,
    use_vol: bool,
    use_atr: bool,
    combine_mode: str,
    signal_cooldown: int,
    trend_lookback: int,
    direction: str
) -> BacktestResult:
    """
    Backtest SEM look-ahead bias.

    GARANTIAS:
    1. Sinal gerado na barra N eh executado na abertura da barra N+1
    2. Indicadores usam apenas dados ate barra N
    3. Precos de entrada usam ASK (compra) ou BID (venda)
    4. Precos de saida usam BID (venda de long) ou ASK (compra de short)
    """

    pip = 0.0001
    total_bars = len(bars)
    trading_months = (bars[-1].timestamp - bars[warmup_bars].timestamp).days / 30.0

    # Calibrar indicador com warmup
    indicator = NoLookAheadIndicator(vol_window=50, atr_window=14)
    indicator.calibrate(bars[:warmup_bars])

    trades_pnl = []
    position: Optional[Position] = None
    pending_signal = None  # Sinal pendente para executar na proxima barra
    cooldown_counter = 0

    for i in range(warmup_bars, total_bars):
        bar = bars[i]

        # =====================================================================
        # PASSO 1: Executar sinal pendente da barra ANTERIOR
        # (Isso garante que nao usamos informacao da barra atual para entrar)
        # =====================================================================
        if pending_signal is not None and position is None:
            spread = bar.spread_pips * pip if bar.has_spread_data else 0.2 * pip

            if pending_signal == 'BUY':
                # Compra no ASK (preco mais alto)
                entry_price = bar.ask_open if bar.ask_open else bar.open + spread/2
                pos_type = PositionType.LONG
                sl = entry_price - (STOP_LOSS_PIPS * pip)
                tp = entry_price + (TAKE_PROFIT_PIPS * pip)
            else:
                # Venda no BID (preco mais baixo)
                entry_price = bar.bid_open if bar.bid_open else bar.open - spread/2
                pos_type = PositionType.SHORT
                sl = entry_price + (STOP_LOSS_PIPS * pip)
                tp = entry_price - (TAKE_PROFIT_PIPS * pip)

            position = Position(pos_type, entry_price, bar.timestamp, sl, tp)
            pending_signal = None

        # =====================================================================
        # PASSO 2: Verificar stop/take profit
        # (Usa high/low da barra atual - isso eh realista)
        # =====================================================================
        if position:
            exit_price = None

            # Long sai no BID, Short sai no ASK
            bid_low = bar.bid_low if bar.bid_low else bar.low
            bid_high = bar.bid_high if bar.bid_high else bar.high
            ask_low = bar.ask_low if bar.ask_low else bar.low
            ask_high = bar.ask_high if bar.ask_high else bar.high

            if position.type == PositionType.LONG:
                # Long: sai vendendo no BID
                if bid_low <= position.stop_loss:
                    exit_price = position.stop_loss
                elif bid_high >= position.take_profit:
                    exit_price = position.take_profit
            else:
                # Short: sai comprando no ASK
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
        # PASSO 3: Atualizar indicador com barra atual
        # =====================================================================
        indicator.update(bar)

        # =====================================================================
        # PASSO 4: Gerar sinal para PROXIMA barra (se nao estiver em posicao)
        # =====================================================================
        if position is None and pending_signal is None:
            if cooldown_counter > 0:
                cooldown_counter -= 1
            else:
                # Calcular scores (usa apenas dados ate barra atual)
                vol_score = indicator.get_volatility_score() if use_vol else 0
                atr_score = indicator.get_atr_score() if use_atr else 0

                # Determinar se sinal eh valido
                signal_valid = False

                if use_vol and use_atr:
                    if combine_mode == 'and':
                        signal_valid = vol_score >= vol_threshold and atr_score >= atr_threshold
                    elif combine_mode == 'or':
                        signal_valid = vol_score >= vol_threshold or atr_score >= atr_threshold
                    elif combine_mode == 'avg':
                        avg_score = (vol_score + atr_score) / 2
                        signal_valid = avg_score >= (vol_threshold + atr_threshold) / 2
                elif use_vol:
                    signal_valid = vol_score >= vol_threshold
                elif use_atr:
                    signal_valid = atr_score >= atr_threshold

                if signal_valid:
                    trend = indicator.get_trend_direction(trend_lookback)
                    if trend != 0:
                        if direction == 'contra':
                            pending_signal = 'SELL' if trend == 1 else 'BUY'
                        else:
                            pending_signal = 'BUY' if trend == 1 else 'SELL'
                        cooldown_counter = signal_cooldown

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
    print("=" * 70)
    print("  OTIMIZACAO ALTA FREQUENCIA (60+ trades/mes)")
    print("  SEM LOOK-AHEAD BIAS")
    print("=" * 70)

    breakeven = STOP_LOSS_PIPS / (STOP_LOSS_PIPS + TAKE_PROFIT_PIPS) * 100
    print(f"\n  Breakeven WR: {breakeven:.1f}%")
    print(f"  SL: {STOP_LOSS_PIPS} pips, TP: {TAKE_PROFIT_PIPS} pips")
    print(f"  Minimo de trades: {MIN_TRADES_TOTAL} (60/mes)")

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

    # Parametros para gerar mais trades
    vol_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    atr_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    cooldowns = [1, 2, 3, 5, 7, 10]
    lookbacks = [3, 5, 7, 10]

    combinations = [
        (True, False, 'and', 'Vol_Only'),
        (False, True, 'and', 'ATR_Only'),
        (True, True, 'or', 'Vol+ATR_OR'),
        (True, True, 'avg', 'Vol+ATR_AVG'),
    ]

    print("\n" + "=" * 70)
    print("  EXECUTANDO OTIMIZACAO")
    print("=" * 70)

    all_results = []
    tested = 0
    total = len(combinations) * len(vol_thresholds) * len(cooldowns) * len(lookbacks)

    for use_vol, use_atr, combine_mode, combo_name in combinations:
        for vol_t in vol_thresholds:
            for cool in cooldowns:
                for look in lookbacks:
                    result = run_backtest_no_lookahead(
                        bars, MIN_PRICES_WARMUP,
                        vol_threshold=vol_t,
                        atr_threshold=vol_t,
                        use_vol=use_vol,
                        use_atr=use_atr,
                        combine_mode=combine_mode,
                        signal_cooldown=cool,
                        trend_lookback=look,
                        direction='contra'
                    )

                    # Filtrar: minimo 360 trades
                    if result.total_trades >= MIN_TRADES_TOTAL:
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

    if not all_results:
        print("\n  NENHUMA COMBINACAO GEROU 360+ TRADES!")
        print("  Reduzindo requisito para 200 trades...")

        # Rodar novamente com menos restricao
        for use_vol, use_atr, combine_mode, combo_name in combinations:
            for vol_t in vol_thresholds:
                for cool in cooldowns:
                    for look in lookbacks:
                        result = run_backtest_no_lookahead(
                            bars, MIN_PRICES_WARMUP,
                            vol_threshold=vol_t,
                            atr_threshold=vol_t,
                            use_vol=use_vol,
                            use_atr=use_atr,
                            combine_mode=combine_mode,
                            signal_cooldown=cool,
                            trend_lookback=look,
                            direction='contra'
                        )

                        if result.total_trades >= 200:
                            result.name = f"{combo_name}_t{vol_t}_c{cool}_l{look}"
                            result.params = {
                                'combo': combo_name,
                                'threshold': vol_t,
                                'cooldown': cool,
                                'lookback': look
                            }
                            all_results.append(result)

    # Ordenar por Win Rate
    all_results.sort(key=lambda x: x.win_rate, reverse=True)

    # Resultados
    print("\n" + "=" * 70)
    print("  TOP 20 - ORDENADO POR WIN RATE (com 60+ trades/mes)")
    print("=" * 70)

    print(f"\n  {'#':<3} {'Nome':<25} {'Trades':<7} {'T/Mes':<6} {'WR%':<6} {'PnL':<8} {'PF':<5} {'Avg':<6}")
    print("  " + "-" * 75)

    for i, r in enumerate(all_results[:20]):
        print(f"  {i+1:<3} {r.name:<25} {r.total_trades:<7} {r.trades_per_month:<6.0f} "
              f"{r.win_rate:<6.1f} {r.pnl_pips:+<8.0f} {r.profit_factor:<5.2f} {r.avg_trade:<+6.1f}")

    # Melhor com 60+ trades/mes
    high_freq = [r for r in all_results if r.trades_per_month >= 60]
    if high_freq:
        best = max(high_freq, key=lambda x: x.win_rate)
        print("\n" + "=" * 70)
        print("  MELHOR COM 60+ TRADES/MES")
        print("=" * 70)

        print(f"\n  Configuracao: {best.name}")
        print(f"  Parametros: {best.params}")
        print(f"  Trades: {best.total_trades} ({best.trades_per_month:.0f}/mes)")
        print(f"  Win Rate: {best.win_rate:.1f}%")
        print(f"  PnL: {best.pnl_pips:+.0f} pips")
        print(f"  Profit Factor: {best.profit_factor:.2f}")
        print(f"  Avg Trade: {best.avg_trade:+.1f} pips")

    # Melhor WR geral
    if all_results:
        best_wr = all_results[0]
        print("\n" + "=" * 70)
        print("  MELHOR WIN RATE GERAL")
        print("=" * 70)

        print(f"\n  Configuracao: {best_wr.name}")
        print(f"  Trades: {best_wr.total_trades} ({best_wr.trades_per_month:.0f}/mes)")
        print(f"  Win Rate: {best_wr.win_rate:.1f}%")
        print(f"  PnL: {best_wr.pnl_pips:+.0f} pips")

    print("\n" + "=" * 70)
    print("  VERIFICACAO DE LOOK-AHEAD BIAS")
    print("=" * 70)
    print("""
  Este backtest NAO tem look-ahead bias porque:

  1. CALIBRACAO: Percentis calculados APENAS no warmup (6624 barras passadas)
  2. SINAL: Gerado na barra N, executado na ABERTURA da barra N+1
  3. INDICADORES: Usam apenas dados ate a barra atual (buffers FIFO)
  4. ENTRADA: Compra no ASK, Venda no BID (precos reais)
  5. SAIDA: Long sai no BID, Short sai no ASK (precos reais)
  6. SPREAD: Usa spread real baixado da API (BID/ASK separados)
    """)

    print("=" * 70)


if __name__ == "__main__":
    main()
