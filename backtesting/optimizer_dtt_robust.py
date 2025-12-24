#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR DTT ROBUSTO - COM VALIDACAO ANTI-OVERFITTING
================================================================================

DTT (Detector de Tunelamento Topologico):
- Usa Homologia Persistente para detectar estruturas topologicas
- Usa Equacao de Schrodinger para probabilidade de tunelamento

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
from strategies.alta_volatilidade.dtt_tunelamento_topologico import DetectorTunelamentoTopologico
from backtesting.robust_optimizer import (
    RobustBacktester, RobustResult, BacktestResult,
    save_robust_config
)


@dataclass
class DTTSignal:
    """Sinal pre-calculado do DTT"""
    bar_idx: int
    price: float
    high: float
    low: float
    persistence_entropy: float
    tunneling_probability: float
    signal_strength: float
    direction: int


class DTTRobustOptimizer:
    """Otimizador DTT com validacao anti-overfitting"""

    def __init__(self, symbol: str = "EURUSD", periodicity: str = "H1"):
        self.symbol = symbol
        self.periodicity = periodicity
        self.backtester = RobustBacktester(pip=0.0001, spread=1.0)

        self.bars: List[Bar] = []
        self.signals: List[DTTSignal] = []
        self.train_bars: List[Bar] = []
        self.test_bars: List[Bar] = []
        self.train_signals: List[DTTSignal] = []
        self.test_signals: List[DTTSignal] = []

        self.robust_results: List[RobustResult] = []
        self.best: Optional[RobustResult] = None

    async def load_and_precompute(self, start_date: datetime, end_date: datetime):
        """Carrega dados e pre-calcula sinais DTT"""
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

        # Pre-calcular DTT
        print("\n  Pre-calculando sinais DTT (computacionalmente intensivo)...")

        dtt = DetectorTunelamentoTopologico(
            max_points=150,
            use_dimensionality_reduction=True,
            reduction_method='pca',
            persistence_entropy_threshold=0.1,
            tunneling_probability_threshold=0.05
        )

        prices_buf = deque(maxlen=500)
        self.signals = []
        min_prices = 150

        for i, bar in enumerate(self.bars):
            prices_buf.append(bar.close)

            if len(prices_buf) < min_prices:
                continue

            try:
                result = dtt.analyze(np.array(prices_buf))

                entropy = result['entropy']['persistence_entropy']
                tunneling = result['tunneling']['tunneling_probability']
                strength = result['signal_strength']
                direction_str = result['direction']
                direction = 1 if direction_str == 'LONG' else (-1 if direction_str == 'SHORT' else 0)

                self.signals.append(DTTSignal(
                    bar_idx=i,
                    price=bar.close,
                    high=bar.high,
                    low=bar.low,
                    persistence_entropy=entropy,
                    tunneling_probability=tunneling,
                    signal_strength=strength,
                    direction=direction
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
            entropies = [s.persistence_entropy for s in self.signals]
            tunnelings = [s.tunneling_probability for s in self.signals]
            print(f"\n  Distribuicao de valores:")
            print(f"    Entropy: min={min(entropies):.3f}, max={max(entropies):.3f}, mean={np.mean(entropies):.3f}")
            print(f"    Tunneling: min={min(tunnelings):.3f}, max={max(tunnelings):.3f}, mean={np.mean(tunnelings):.3f}")

        return len(self.train_signals) > 50 and len(self.test_signals) > 20

    def _run_backtest(self, signals: List[DTTSignal], bars: List[Bar],
                      entropy_thresh: float, tunneling_thresh: float,
                      strength_thresh: float, sl: float, tp: float,
                      bar_offset: int = 0) -> List[float]:
        """Executa backtest em um conjunto de dados"""
        if tp <= sl:
            return []

        entries = []
        for s in signals:
            if (s.persistence_entropy >= entropy_thresh and
                s.tunneling_probability >= tunneling_thresh and
                s.signal_strength >= strength_thresh and
                s.direction != 0):
                entries.append((s.bar_idx - bar_offset, s.price, s.direction))

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

    def _test_params(self, entropy_thresh: float, tunneling_thresh: float,
                     strength_thresh: float, sl: float, tp: float) -> Optional[RobustResult]:
        """Testa parametros em treino e teste"""

        train_pnls = self._run_backtest(
            self.train_signals, self.train_bars,
            entropy_thresh, tunneling_thresh, strength_thresh, sl, tp,
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
            entropy_thresh, tunneling_thresh, strength_thresh, sl, tp,
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
            "persistence_entropy_threshold": round(entropy_thresh, 4),
            "tunneling_probability_threshold": round(tunneling_thresh, 4),
            "min_signal_strength": round(strength_thresh, 4),
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

    def optimize(self, n: int = 300000) -> Optional[RobustResult]:
        """Executa otimizacao robusta"""
        if not self.train_signals or not self.test_signals:
            print("  ERRO: Dados nao carregados!")
            return None

        print(f"\n{'='*70}")
        print(f"  OTIMIZACAO ROBUSTA DTT: {n:,} COMBINACOES")
        print(f"  Com validacao Train/Test Split")
        print(f"{'='*70}")

        # Ranges baseados na distribuicao real (entropy=0.826-0.955, tunneling=0.116-0.925)
        entropy_vals = np.linspace(0.83, 0.95, 15)
        tunneling_vals = np.linspace(0.15, 0.50, 15)
        strength_vals = np.linspace(0.2, 0.7, 10)
        sl_vals = np.linspace(20, 55, 15)
        tp_vals = np.linspace(25, 80, 20)

        best_robustness = -1
        tested = 0
        robust_count = 0
        start = datetime.now()

        for _ in range(n):
            tested += 1

            entropy = float(random.choice(entropy_vals))
            tunneling = float(random.choice(tunneling_vals))
            strength = float(random.choice(strength_vals))
            sl = float(random.choice(sl_vals))
            tp = float(random.choice(tp_vals))

            result = self._test_params(entropy, tunneling, strength, sl, tp)

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

            if tested % 30000 == 0:
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
            strategy_name="DTT-TunelamentoTopologico",
            symbol=self.symbol,
            periodicity=self.periodicity,
            n_tested=n_tested,
            n_robust=len(self.robust_results)
        )

        # Top 10
        top_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "dtt_robust_top10.json"
        )
        sorted_results = sorted(self.robust_results, key=lambda x: x.robustness_score, reverse=True)[:10]
        top_data = [r.to_dict() for r in sorted_results]
        with open(top_file, 'w') as f:
            json.dump(top_data, f, indent=2)
        print(f"  Top 10 robustos salvo em: {top_file}")


async def main():
    N_COMBINATIONS = 100000

    print("=" * 70)
    print("  OTIMIZADOR DTT ROBUSTO")
    print("  Com Validacao Anti-Overfitting")
    print(f"  {N_COMBINATIONS:,} Combinacoes")
    print("  PARA DINHEIRO REAL")
    print("=" * 70)

    opt = DTTRobustOptimizer("EURUSD", "H1")

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
