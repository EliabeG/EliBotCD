#!/usr/bin/env python3
"""
================================================================================
BENCHMARK TEST: DTT vs Filtros Simples
================================================================================

VERSÃO V3.0 - 24/12/2025

Objetivo: Determinar se a complexidade do DTT (Topologia + Quântica) adiciona
valor real comparado a filtros de volatilidade simples.

Filtros Testados:
1. ATR Filter - Baseado em Average True Range
2. Bollinger Filter - Baseado em desvio padrão
3. Volatility Percentile - Baseado em percentil histórico
4. Donchian Breakout - Baseado em canal de preços
5. DTT Completo - Topologia + Schrödinger

Métricas:
- Total de trades
- Win rate
- Profit factor
- Total PnL (pips)
- Max drawdown
- Sharpe ratio

Uso:
    python -m backtesting.dtt.benchmark_test
================================================================================
"""

import sys
import os
import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import deque

# Adiciona diretorio raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.fxopen_historical_ws import Bar, download_historical_data


@dataclass
class BenchmarkResult:
    """Resultado de um benchmark"""
    name: str
    total_trades: int
    win_rate: float
    profit_factor: float
    total_pnl_pips: float
    max_drawdown: float
    sharpe_ratio: float
    signals_generated: int = 0

    def __str__(self):
        return (f"{self.name:<35} | Trades: {self.total_trades:>4} | "
                f"WR: {self.win_rate:>5.1%} | PF: {self.profit_factor:>5.2f} | "
                f"PnL: {self.total_pnl_pips:>8.1f} pips | "
                f"Sharpe: {self.sharpe_ratio:>5.2f}")


class SimpleBenchmarks:
    """Filtros simples para comparação com DTT"""

    @staticmethod
    def atr_filter(bars: List[Bar], period: int = 14,
                   threshold_mult: float = 1.5) -> List[bool]:
        """
        Filtro ATR: Trade ON quando ATR atual > média * threshold

        Lógica: Alta volatilidade recente indica possível movimento forte
        """
        signals = [False] * len(bars)

        if len(bars) < period + 1:
            return signals

        for i in range(period, len(bars)):
            # Calcular ATR
            atrs = []
            for j in range(i - period, i):
                tr = max(
                    bars[j].high - bars[j].low,
                    abs(bars[j].high - bars[j-1].close),
                    abs(bars[j].low - bars[j-1].close)
                )
                atrs.append(tr)

            atr_current = atrs[-1]
            atr_avg = np.mean(atrs)

            signals[i] = atr_current > atr_avg * threshold_mult

        return signals

    @staticmethod
    def bollinger_filter(bars: List[Bar], period: int = 20,
                         std_mult: float = 2.0) -> List[bool]:
        """
        Filtro Bollinger: Trade ON quando preço fora das bandas

        Lógica: Preço fora das bandas indica movimento extremo
        """
        signals = [False] * len(bars)

        if len(bars) < period + 1:
            return signals

        for i in range(period, len(bars)):
            closes = [bars[j].close for j in range(i - period, i)]
            mean = np.mean(closes)
            std = np.std(closes)

            upper = mean + std_mult * std
            lower = mean - std_mult * std

            current = bars[i].close
            signals[i] = current > upper or current < lower

        return signals

    @staticmethod
    def volatility_percentile_filter(bars: List[Bar],
                                      lookback: int = 100,
                                      percentile: float = 75) -> List[bool]:
        """
        Filtro de Percentil: Trade ON quando volatilidade > percentil histórico

        Lógica: Volatilidade no percentil alto indica regime de alta volatilidade
        """
        signals = [False] * len(bars)

        if len(bars) < lookback + 1:
            return signals

        for i in range(lookback, len(bars)):
            # Volatilidade recente (últimas 20 barras)
            recent_returns = []
            for j in range(max(1, i - 20), i):
                ret = abs(bars[j].close - bars[j-1].close) / bars[j-1].close
                recent_returns.append(ret)

            if not recent_returns:
                continue

            recent_vol = np.std(recent_returns)

            # Volatilidade histórica
            hist_vols = []
            for k in range(i - lookback, max(0, i - 20)):
                window_returns = []
                for j in range(k, min(k + 20, i)):
                    if j > 0:
                        ret = abs(bars[j].close - bars[j-1].close) / bars[j-1].close
                        window_returns.append(ret)
                if window_returns:
                    hist_vols.append(np.std(window_returns))

            if hist_vols:
                threshold = np.percentile(hist_vols, percentile)
                signals[i] = recent_vol > threshold

        return signals

    @staticmethod
    def donchian_breakout_filter(bars: List[Bar],
                                  period: int = 20) -> List[Tuple[bool, int]]:
        """
        Filtro Donchian: Trade ON + direção quando preço rompe canal

        Lógica: Breakout de canal de N períodos indica início de tendência

        Returns: List of (trade_on, direction) where direction is 1 or -1
        """
        signals = [(False, 0)] * len(bars)

        if len(bars) < period + 1:
            return signals

        for i in range(period, len(bars)):
            highs = [bars[j].high for j in range(i - period, i)]
            lows = [bars[j].low for j in range(i - period, i)]

            upper = max(highs)
            lower = min(lows)

            current = bars[i].close

            if current > upper:
                signals[i] = (True, 1)  # Breakout para cima
            elif current < lower:
                signals[i] = (True, -1)  # Breakout para baixo

        return signals


class DTTBenchmarkRunner:
    """Executor de benchmarks comparativos"""

    # Custos realistas
    SPREAD_PIPS = 1.5
    SLIPPAGE_PIPS = 0.8

    def __init__(self, bars: List[Bar], pip: float = 0.0001):
        self.bars = bars
        self.pip = pip

    def run_with_filter(self,
                        filter_signals: List,
                        name: str,
                        direction_lookback: int = 12,
                        sl_pips: float = 30,
                        tp_pips: float = 60) -> BenchmarkResult:
        """
        Executa backtest com um filtro específico

        Args:
            filter_signals: Lista de sinais (bool ou tuple(bool, int))
            name: Nome do filtro
            direction_lookback: Barras para calcular direção momentum
            sl_pips: Stop loss em pips
            tp_pips: Take profit em pips

        Returns:
            BenchmarkResult com métricas
        """
        pnls = []
        last_exit = -1
        signals_generated = sum(1 for s in filter_signals if (isinstance(s, tuple) and s[0]) or (isinstance(s, bool) and s))

        for i in range(direction_lookback + 1, len(self.bars) - 1):
            # Verificar se filtro está ON
            if isinstance(filter_signals[i], tuple):
                trade_on, direction = filter_signals[i]
            else:
                trade_on = filter_signals[i]
                # Calcular direção por momentum
                if i >= direction_lookback + 1:
                    recent = self.bars[i-1].close
                    past = self.bars[i-direction_lookback].close
                    direction = 1 if recent > past else -1
                else:
                    direction = 0

            if not trade_on or direction == 0:
                continue

            if i <= last_exit:
                continue

            # Entrada no OPEN da próxima barra (sem look-ahead)
            if i + 1 >= len(self.bars):
                continue

            entry_bar = self.bars[i + 1]
            entry_price = entry_bar.open

            # Aplicar custos de entrada
            cost = (self.SPREAD_PIPS + self.SLIPPAGE_PIPS) * self.pip
            if direction == 1:
                entry_price += cost
                sl_price = entry_price - sl_pips * self.pip
                tp_price = entry_price + tp_pips * self.pip
            else:
                entry_price -= cost
                sl_price = entry_price + sl_pips * self.pip
                tp_price = entry_price - tp_pips * self.pip

            # Simular trade
            exit_price = None
            for j in range(i + 2, min(i + 200, len(self.bars))):
                bar = self.bars[j]

                if direction == 1:
                    if bar.low <= sl_price:
                        exit_price = sl_price - self.SLIPPAGE_PIPS * self.pip
                        last_exit = j
                        break
                    if bar.high >= tp_price:
                        exit_price = tp_price - self.SLIPPAGE_PIPS * self.pip
                        last_exit = j
                        break
                else:
                    if bar.high >= sl_price:
                        exit_price = sl_price + self.SLIPPAGE_PIPS * self.pip
                        last_exit = j
                        break
                    if bar.low <= tp_price:
                        exit_price = tp_price + self.SLIPPAGE_PIPS * self.pip
                        last_exit = j
                        break

            if exit_price is None:
                last_exit = min(i + 200, len(self.bars) - 1)
                exit_price = self.bars[last_exit].close

            # Calcular PnL
            if direction == 1:
                pnl = (exit_price - entry_price) / self.pip
            else:
                pnl = (entry_price - exit_price) / self.pip

            pnls.append(pnl)

        return self._calculate_metrics(pnls, name, signals_generated)

    def _calculate_metrics(self, pnls: List[float], name: str,
                           signals_generated: int = 0) -> BenchmarkResult:
        """Calcula métricas do benchmark"""
        if not pnls:
            return BenchmarkResult(name, 0, 0, 0, 0, 1.0, 0, signals_generated)

        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls)

        gross_profit = sum(p for p in pnls if p > 0) or 0.001
        gross_loss = abs(sum(p for p in pnls if p < 0)) or 0.001
        profit_factor = gross_profit / gross_loss

        total_pnl = sum(pnls)

        # Drawdown
        equity = np.cumsum([0] + pnls)
        peak = np.maximum.accumulate(equity + 1000)
        drawdown = (peak - (equity + 1000)) / peak
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

        # Sharpe ratio anualizado
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        else:
            sharpe = 0

        return BenchmarkResult(
            name=name,
            total_trades=len(pnls),
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_pnl_pips=total_pnl,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            signals_generated=signals_generated
        )


async def run_complete_benchmark(symbol: str = 'EURUSD',
                                  timeframe: str = 'H1',
                                  start_date: datetime = None,
                                  end_date: datetime = None):
    """
    Executa benchmark completo: DTT vs Alternativas

    Args:
        symbol: Par de moedas
        timeframe: Timeframe
        start_date: Data inicial
        end_date: Data final
    """
    print("=" * 80)
    print("  BENCHMARK V3.0: DTT vs FILTROS SIMPLES")
    print("  Objetivo: Validar se complexidade do DTT adiciona valor")
    print("=" * 80)

    # Datas padrão
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    if start_date is None:
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # Carregar dados
    print(f"\nCarregando dados históricos...")
    print(f"  Símbolo: {symbol}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Período: {start_date.date()} a {end_date.date()}")

    bars = await download_historical_data(
        symbol, timeframe,
        start_date, end_date
    )
    print(f"  Barras carregadas: {len(bars)}")

    if len(bars) < 500:
        print("ERRO: Dados insuficientes para benchmark (mínimo 500 barras)")
        return []

    # Inicializar runner
    runner = DTTBenchmarkRunner(bars)
    benchmarks = SimpleBenchmarks()

    results = []

    # 1. ATR Filter
    print("\n[1/5] Testando ATR Filter...")
    atr_signals = benchmarks.atr_filter(bars, period=14, threshold_mult=1.5)
    result_atr = runner.run_with_filter(atr_signals, "ATR Filter (14, 1.5x)",
                                         sl_pips=30, tp_pips=60)
    results.append(result_atr)
    print(f"      Sinais: {result_atr.signals_generated}, Trades: {result_atr.total_trades}")

    # 2. Bollinger Filter
    print("[2/5] Testando Bollinger Filter...")
    boll_signals = benchmarks.bollinger_filter(bars, period=20, std_mult=2.0)
    result_boll = runner.run_with_filter(boll_signals, "Bollinger Filter (20, 2σ)",
                                          sl_pips=30, tp_pips=60)
    results.append(result_boll)
    print(f"      Sinais: {result_boll.signals_generated}, Trades: {result_boll.total_trades}")

    # 3. Volatility Percentile
    print("[3/5] Testando Volatility Percentile...")
    vol_signals = benchmarks.volatility_percentile_filter(bars, lookback=100, percentile=75)
    result_vol = runner.run_with_filter(vol_signals, "Vol Percentile (75%)",
                                         sl_pips=30, tp_pips=60)
    results.append(result_vol)
    print(f"      Sinais: {result_vol.signals_generated}, Trades: {result_vol.total_trades}")

    # 4. Donchian Breakout
    print("[4/5] Testando Donchian Breakout...")
    donch_signals = benchmarks.donchian_breakout_filter(bars, period=20)
    result_donch = runner.run_with_filter(donch_signals, "Donchian Breakout (20)",
                                           sl_pips=30, tp_pips=60)
    results.append(result_donch)
    print(f"      Sinais: {result_donch.signals_generated}, Trades: {result_donch.total_trades}")

    # 5. DTT Completo
    print("[5/5] Testando DTT Completo (Topologia + Quântico)...")
    try:
        from strategies.alta_volatilidade.dtt_tunelamento_topologico import DetectorTunelamentoTopologico

        dtt = DetectorTunelamentoTopologico(
            max_points=150,
            persistence_entropy_threshold=0.5,
            tunneling_probability_threshold=0.15,
            auto_calibrate_quantum=True  # V3.0
        )

        prices_buf = deque(maxlen=500)
        dtt_signals = [False] * len(bars)
        dtt_signal_count = 0

        print("      Processando DTT (pode demorar)...")
        for i, bar in enumerate(bars):
            prices_buf.append(bar.close)
            if len(prices_buf) >= 150:
                try:
                    result = dtt.analyze(np.array(prices_buf))
                    if result['trade_on']:
                        dtt_signals[i] = True
                        dtt_signal_count += 1
                except Exception:
                    pass

            # Progress
            if i > 0 and i % 1000 == 0:
                print(f"      Processado: {i}/{len(bars)} barras...")

        result_dtt = runner.run_with_filter(dtt_signals, "DTT Completo (Topo+Quântico)",
                                             sl_pips=30, tp_pips=60)
        result_dtt.signals_generated = dtt_signal_count
        results.append(result_dtt)
        print(f"      Sinais: {result_dtt.signals_generated}, Trades: {result_dtt.total_trades}")

    except Exception as e:
        print(f"  AVISO: Não foi possível testar DTT: {e}")

    # Resultados
    print("\n" + "=" * 80)
    print("  RESULTADOS DO BENCHMARK")
    print("=" * 80)
    print()

    # Header
    print(f"{'Filtro':<35} | {'Trades':>6} | {'WR':>6} | {'PF':>6} | {'PnL (pips)':>10} | {'Sharpe':>6}")
    print("-" * 80)

    # Ordenar por Profit Factor
    results.sort(key=lambda x: x.profit_factor, reverse=True)

    for r in results:
        print(r)

    print("\n" + "-" * 80)

    # Análise comparativa
    if len(results) >= 2:
        best = results[0]
        dtt_result = next((r for r in results if "DTT" in r.name), None)

        print(f"\n  ANÁLISE COMPARATIVA:")
        print(f"    Melhor filtro: {best.name}")
        print(f"    Melhor PF: {best.profit_factor:.2f}")

        if dtt_result:
            print(f"\n    DTT:")
            print(f"      Profit Factor: {dtt_result.profit_factor:.2f}")
            print(f"      Trades: {dtt_result.total_trades}")
            print(f"      Win Rate: {dtt_result.win_rate:.1%}")

            if dtt_result.profit_factor > best.profit_factor * 1.1:
                improvement = (dtt_result.profit_factor/best.profit_factor - 1) * 100
                print(f"\n  ✅ DTT adiciona valor significativo (+{improvement:.1f}% vs {best.name})")
            elif dtt_result.profit_factor > best.profit_factor:
                improvement = (dtt_result.profit_factor/best.profit_factor - 1) * 100
                print(f"\n  ⚠️ DTT marginalmente melhor (+{improvement:.1f}%)")
            else:
                pct_worse = (1 - dtt_result.profit_factor/best.profit_factor) * 100
                print(f"\n  ❌ DTT NÃO adiciona valor (-{pct_worse:.1f}% vs {best.name})")
                print(f"     Considere simplificar para {best.name}")
        else:
            print("\n  ⚠️ DTT não foi testado (erro na execução)")

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark DTT vs Filtros Simples')
    parser.add_argument('--symbol', default='EURUSD', help='Par de moedas')
    parser.add_argument('--timeframe', default='H1', help='Timeframe')
    parser.add_argument('--days', type=int, default=365, help='Dias de histórico')

    args = parser.parse_args()

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.days)

    asyncio.run(run_complete_benchmark(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=start_date,
        end_date=end_date
    ))
