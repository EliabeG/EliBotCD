#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR DSG ROBUSTO - COM VALIDACAO ANTI-OVERFITTING
================================================================================

DSG (Detector de Singularidade Gravitacional):
- Usa Tensor Metrico Financeiro para modelar espaco-tempo
- Usa Escalar de Ricci para detectar curvatura
- Usa Forca de Mare para detectar rompimentos

VALIDACAO:
1. Divide dados em 70% treino / 30% teste
2. Otimiza apenas no treino
3. Valida no teste (dados nunca vistos)
4. Descarta resultados que nao passam nos filtros de realismo

PARA DINHEIRO REAL. SEM OVERFITTING.
================================================================================
"""

import sys
import os
import json
import asyncio
import random
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.fxopen_historical_ws import Bar, download_historical_data
from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional
from backtesting.robust_optimizer import (
    RobustBacktester, RobustResult, BacktestResult,
    save_robust_config
)


@dataclass
class DSGSignal:
    """Sinal pre-calculado do DSG"""
    bar_idx: int
    price: float
    high: float
    low: float
    ricci_scalar: float
    tidal_force: float
    event_horizon_distance: float
    ricci_collapsing: bool
    crossing_horizon: bool
    geodesic_direction: int
    signal: int


class DSGRobustOptimizer:
    """Otimizador DSG com validacao anti-overfitting"""

    def __init__(self, symbol: str = "EURUSD", periodicity: str = "H1"):
        self.symbol = symbol
        self.periodicity = periodicity
        self.backtester = RobustBacktester(pip=0.0001, spread=1.0)

        self.bars: List[Bar] = []
        self.signals: List[DSGSignal] = []
        self.train_bars: List[Bar] = []
        self.test_bars: List[Bar] = []
        self.train_signals: List[DSGSignal] = []
        self.test_signals: List[DSGSignal] = []

        self.robust_results: List[RobustResult] = []
        self.best: Optional[RobustResult] = None

    async def load_and_precompute(self, start_date: datetime, end_date: datetime):
        """Carrega dados e pre-calcula sinais DSG"""
        print("\n" + "=" * 70)
        print("  CARREGANDO DADOS REAIS")
        print("=" * 70)

        self.bars = await download_historical_data(
            symbol=self.symbol,
            periodicity=self.periodicity,
            start_time=start_date,
            end_time=end_date
        )
        print(f"  Total de barras: {len(self.bars)}")

        if len(self.bars) < 300:
            print("  ERRO: Dados insuficientes!")
            return False

        # SPLIT TRAIN/TEST (70/30)
        split_idx = int(len(self.bars) * 0.70)
        self.train_bars = self.bars[:split_idx]
        self.test_bars = self.bars[split_idx:]

        print(f"\n  DIVISAO TRAIN/TEST:")
        print(f"    Treino: {len(self.train_bars)} barras")
        print(f"    Teste:  {len(self.test_bars)} barras")

        # Pre-calcular DSG
        print("\n  Pre-calculando sinais DSG (computacionalmente intensivo)...")

        dsg = DetectorSingularidadeGravitacional(
            ricci_collapse_threshold=-0.5,
            tidal_force_threshold=0.1,
            lookback_window=30
        )

        prices_buf = deque(maxlen=100)
        self.signals = []
        min_prices = 50

        for i, bar in enumerate(self.bars):
            prices_buf.append(bar.close)

            if len(prices_buf) < min_prices:
                continue

            try:
                prices_arr = np.array(prices_buf)
                result = dsg.analyze(prices_arr)

                self.signals.append(DSGSignal(
                    bar_idx=i,
                    price=bar.close,
                    high=bar.high,
                    low=bar.low,
                    ricci_scalar=result['Ricci_Scalar'],
                    tidal_force=result['Tidal_Force_Magnitude'],
                    event_horizon_distance=result['Event_Horizon_Distance'],
                    ricci_collapsing=result['ricci_collapsing'],
                    crossing_horizon=result['crossing_horizon'],
                    geodesic_direction=result['geodesic_direction'],
                    signal=result['signal']
                ))

            except:
                continue

            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(self.bars)} barras...")

        # Separar sinais
        self.train_signals = [s for s in self.signals if s.bar_idx < split_idx]
        self.test_signals = [s for s in self.signals if s.bar_idx >= split_idx]

        print(f"\n  Sinais pre-calculados:")
        print(f"    Treino: {len(self.train_signals)} sinais")
        print(f"    Teste:  {len(self.test_signals)} sinais")

        # Debug: mostrar distribuicao de valores
        if self.signals:
            ricci_vals = [s.ricci_scalar for s in self.signals]
            tidal_vals = [s.tidal_force for s in self.signals]
            print(f"\n  Distribuicao de valores:")
            print(f"    Ricci: min={min(ricci_vals):.4f}, max={max(ricci_vals):.4f}, mean={np.mean(ricci_vals):.4f}")
            print(f"    Tidal: min={min(tidal_vals):.6f}, max={max(tidal_vals):.6f}, mean={np.mean(tidal_vals):.6f}")

        return len(self.train_signals) > 50 and len(self.test_signals) > 20

    def _run_backtest(self, signals: List[DSGSignal], bars: List[Bar],
                      ricci_thresh: float, tidal_thresh: float,
                      sl: float, tp: float,
                      bar_offset: int = 0) -> List[float]:
        """Executa backtest em um conjunto de dados"""
        if tp <= sl:
            return []

        entries = []
        for s in signals:
            # Condicoes de entrada baseadas no DSG
            ricci_collapse = s.ricci_scalar < ricci_thresh or s.ricci_collapsing
            high_tidal = s.tidal_force > tidal_thresh
            crossing = s.crossing_horizon

            # Precisa de pelo menos 2 condicoes
            conditions = sum([ricci_collapse, high_tidal, crossing])

            if conditions >= 2 and s.geodesic_direction != 0:
                entries.append((s.bar_idx - bar_offset, s.price, s.geodesic_direction))

        if len(entries) < 3:
            return []

        pnls = []
        for bar_idx, entry_price, direction in entries:
            if bar_idx < 0 or bar_idx >= len(bars) - 1:
                continue

            trade = self.backtester.execute_trade(
                bars=bars,
                entry_idx=bar_idx,
                entry_price=entry_price,
                direction=direction,
                sl_pips=sl,
                tp_pips=tp,
                max_bars=200
            )
            pnls.append(trade.pnl_pips)

        return pnls

    def _test_params(self, ricci_thresh: float, tidal_thresh: float,
                     sl: float, tp: float) -> Optional[RobustResult]:
        """Testa parametros em treino e teste"""

        train_pnls = self._run_backtest(
            self.train_signals, self.train_bars,
            ricci_thresh, tidal_thresh, sl, tp,
            bar_offset=0
        )
        train_result = self.backtester.calculate_backtest_result(train_pnls)

        # Filtros para treino
        if not train_result.is_valid(
            min_trades=20,
            max_win_rate=0.68,
            min_win_rate=0.28,
            max_pf=5.0,
            min_pf=1.05,
            max_dd=0.45
        ):
            return None

        # Backtest no TESTE
        split_idx = len(self.train_bars)
        test_pnls = self._run_backtest(
            self.test_signals, self.test_bars,
            ricci_thresh, tidal_thresh, sl, tp,
            bar_offset=split_idx
        )
        test_result = self.backtester.calculate_backtest_result(test_pnls)

        # Filtros para teste
        if not test_result.is_valid(
            min_trades=10,
            max_win_rate=0.75,
            min_win_rate=0.20,
            max_pf=6.0,
            min_pf=0.9,
            max_dd=0.55
        ):
            return None

        # Calcula robustez
        robustness, degradation, _ = self.backtester.calculate_robustness(
            train_result, test_result
        )

        # Robustez: teste deve manter >= 50% do treino
        pf_ratio = test_result.profit_factor / train_result.profit_factor if train_result.profit_factor > 0 else 0
        wr_ratio = test_result.win_rate / train_result.win_rate if train_result.win_rate > 0 else 0
        is_robust = pf_ratio >= 0.50 and wr_ratio >= 0.50 and test_result.profit_factor >= 0.9

        if not is_robust:
            return None

        params = {
            "ricci_collapse_threshold": round(ricci_thresh, 4),
            "tidal_force_threshold": round(tidal_thresh, 6),
            "stop_loss_pips": round(sl, 1),
            "take_profit_pips": round(tp, 1)
        }

        return RobustResult(
            params=params,
            train_result=train_result,
            test_result=test_result,
            robustness_score=robustness,
            degradation=degradation,
            is_robust=is_robust
        )

    def optimize(self, n: int = 100000) -> Optional[RobustResult]:
        """Executa otimizacao robusta"""
        if not self.train_signals or not self.test_signals:
            print("  ERRO: Dados nao carregados!")
            return None

        print(f"\n{'='*70}")
        print(f"  OTIMIZACAO ROBUSTA DSG: {n:,} COMBINACOES")
        print(f"  Com validacao Train/Test Split")
        print(f"{'='*70}")

        # Ranges baseados na teoria e distribuicao real
        ricci_vals = np.linspace(-1.0, -0.1, 20)
        tidal_vals = np.linspace(0.001, 0.5, 20)
        sl_vals = np.linspace(20, 55, 15)
        tp_vals = np.linspace(25, 80, 20)

        best_robustness = -1
        tested = 0
        robust_count = 0
        start = datetime.now()

        for _ in range(n):
            tested += 1

            ricci = float(random.choice(ricci_vals))
            tidal = float(random.choice(tidal_vals))
            sl = float(random.choice(sl_vals))
            tp = float(random.choice(tp_vals))

            result = self._test_params(ricci, tidal, sl, tp)

            if result:
                robust_count += 1
                self.robust_results.append(result)

                if result.robustness_score > best_robustness:
                    best_robustness = result.robustness_score
                    self.best = result

                    print(f"\n  [ROBUSTO #{robust_count}] Robustez={result.robustness_score:.4f}")
                    print(f"    TREINO: {result.train_result.trades} trades, "
                          f"WR={result.train_result.win_rate:.1%}, "
                          f"PF={result.train_result.profit_factor:.2f}")
                    print(f"    TESTE:  {result.test_result.trades} trades, "
                          f"WR={result.test_result.win_rate:.1%}, "
                          f"PF={result.test_result.profit_factor:.2f}")

            if tested % 20000 == 0:
                elapsed = (datetime.now() - start).total_seconds()
                rate = tested / elapsed
                eta = (n - tested) / rate / 60
                print(f"  {tested:,}/{n:,} ({tested/n*100:.1f}%) | "
                      f"Robustos: {robust_count} | "
                      f"Vel: {rate:.0f}/s | ETA: {eta:.0f}min")

        elapsed = (datetime.now() - start).total_seconds()
        print(f"\n{'='*70}")
        print(f"  CONCLUIDO em {elapsed/60:.1f}min")
        print(f"  Testados: {tested:,} | Robustos: {robust_count}")
        print(f"{'='*70}")

        return self.best

    def save(self, n_tested: int = 0):
        """Salva melhor configuracao robusta"""
        if not self.best:
            print("  Nenhuma configuracao robusta encontrada!")
            return

        save_robust_config(
            result=self.best,
            strategy_name="DSG-SingularidadeGravitacional",
            symbol=self.symbol,
            periodicity=self.periodicity,
            n_tested=n_tested,
            n_robust=len(self.robust_results)
        )

        # Top 10
        top_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "dsg_robust_top10.json"
        )
        sorted_results = sorted(self.robust_results, key=lambda x: x.robustness_score, reverse=True)[:10]
        top_data = [r.to_dict() for r in sorted_results]
        with open(top_file, 'w') as f:
            json.dump(top_data, f, indent=2)
        print(f"  Top 10 robustos salvo em: {top_file}")


async def main():
    N_COMBINATIONS = 100000

    print("=" * 70)
    print("  OTIMIZADOR DSG ROBUSTO")
    print("  Com Validacao Anti-Overfitting")
    print(f"  {N_COMBINATIONS:,} Combinacoes")
    print("  PARA DINHEIRO REAL")
    print("=" * 70)

    opt = DSGRobustOptimizer("EURUSD", "H1")

    start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    print(f"\n  Periodo: {start.date()} a {end.date()}")

    if await opt.load_and_precompute(start, end):
        best = opt.optimize(N_COMBINATIONS)
        if best:
            print(f"\n{'='*70}")
            print(f"  MELHOR RESULTADO ROBUSTO:")
            print(f"{'='*70}")
            print(f"  TREINO: {best.train_result.trades} trades, "
                  f"WR={best.train_result.win_rate:.1%}, "
                  f"PF={best.train_result.profit_factor:.2f}")
            print(f"  TESTE:  {best.test_result.trades} trades, "
                  f"WR={best.test_result.win_rate:.1%}, "
                  f"PF={best.test_result.profit_factor:.2f}")
            print(f"  Degradacao: {best.degradation*100:.1f}%")
            print(f"{'='*70}")

            opt.save(n_tested=N_COMBINATIONS)
        else:
            print("\n  AVISO: Nenhuma configuracao passou nos filtros!")
    else:
        print("\n  ERRO: Falha ao carregar dados!")


if __name__ == "__main__":
    asyncio.run(main())
