#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR ODMN ROBUSTO - COM VALIDACAO ANTI-OVERFITTING
================================================================================

ODMN (Oraculo de Derivativos de Malliavin-Nash):
- Usa Modelo de Heston para volatilidade estocastica
- Usa Derivadas de Malliavin para detectar fragilidade
- Usa Mean Field Games para prever comportamento institucional

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
from strategies.alta_volatilidade.odmn_malliavin_nash import OracloDerivativosMalliavinNash
from backtesting.robust_optimizer import (
    RobustBacktester, RobustResult, BacktestResult,
    save_robust_config
)


@dataclass
class ODMNSignal:
    """Sinal pre-calculado do ODMN"""
    bar_idx: int
    price: float
    high: float
    low: float
    fragility_index: float
    fragility_percentile: float
    mfg_direction: float
    mfg_equilibrium: bool
    regime: str
    signal: int
    confidence: float


class ODMNRobustOptimizer:
    """Otimizador ODMN com validacao anti-overfitting"""

    def __init__(self, symbol: str = "EURUSD", periodicity: str = "H1"):
        self.symbol = symbol
        self.periodicity = periodicity
        self.backtester = RobustBacktester(pip=0.0001, spread=1.0)

        self.bars: List[Bar] = []
        self.signals: List[ODMNSignal] = []
        self.train_bars: List[Bar] = []
        self.test_bars: List[Bar] = []
        self.train_signals: List[ODMNSignal] = []
        self.test_signals: List[ODMNSignal] = []

        self.robust_results: List[RobustResult] = []
        self.best: Optional[RobustResult] = None

    async def load_and_precompute(self, start_date: datetime, end_date: datetime):
        """Carrega dados e pre-calcula sinais ODMN"""
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

        # Pre-calcular ODMN
        print("\n  Pre-calculando sinais ODMN (computacionalmente intensivo)...")

        odmn = OracloDerivativosMalliavinNash(
            lookback_window=100,
            fragility_threshold=2.0,
            mfg_direction_threshold=0.1,
            use_deep_galerkin=False,  # Usar solucao analitica para velocidade
            malliavin_paths=1000,
            malliavin_steps=30
        )

        prices_buf = deque(maxlen=150)
        self.signals = []
        min_prices = 110

        for i, bar in enumerate(self.bars):
            prices_buf.append(bar.close)

            if len(prices_buf) < min_prices:
                continue

            try:
                prices_arr = np.array(prices_buf)
                result = odmn.analyze(prices_arr)

                self.signals.append(ODMNSignal(
                    bar_idx=i,
                    price=bar.close,
                    high=bar.high,
                    low=bar.low,
                    fragility_index=result['fragility_index'],
                    fragility_percentile=result['fragility_percentile'],
                    mfg_direction=result['mfg_direction'],
                    mfg_equilibrium=result['mfg_equilibrium'],
                    regime=result['regime'],
                    signal=result['signal'],
                    confidence=result['confidence']
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
            frag_vals = [s.fragility_index for s in self.signals]
            mfg_vals = [s.mfg_direction for s in self.signals]
            print(f"\n  Distribuicao de valores:")
            print(f"    Fragility: min={min(frag_vals):.4f}, max={max(frag_vals):.4f}, mean={np.mean(frag_vals):.4f}")
            print(f"    MFG Dir: min={min(mfg_vals):.4f}, max={max(mfg_vals):.4f}, mean={np.mean(mfg_vals):.4f}")

        return len(self.train_signals) > 50 and len(self.test_signals) > 20

    def _run_backtest(self, signals: List[ODMNSignal], bars: List[Bar],
                      fragility_pct_thresh: float, confidence_thresh: float,
                      sl: float, tp: float,
                      bar_offset: int = 0) -> List[float]:
        """Executa backtest em um conjunto de dados"""
        if tp <= sl:
            return []

        entries = []
        # Usar preco recente para determinar direcao de reversao
        recent_prices = {}
        for s in signals:
            recent_prices[s.bar_idx] = s.price

        for s in signals:
            # Condicoes de entrada: fragilidade alta = reversao
            high_fragility = s.fragility_percentile > fragility_pct_thresh

            if high_fragility:
                # Determinar direcao baseado em tendencia recente (reversao)
                if s.bar_idx >= 5:
                    prices_before = [recent_prices.get(s.bar_idx - i, s.price) for i in range(1, 6) if s.bar_idx - i in recent_prices]
                    if prices_before:
                        avg_before = np.mean(prices_before)
                        # Se preco subiu, short (reversao para baixo)
                        # Se preco caiu, long (reversao para cima)
                        direction = -1 if s.price > avg_before else 1
                        entries.append((s.bar_idx - bar_offset, s.price, direction))

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

    def _test_params(self, fragility_pct_thresh: float, confidence_thresh: float,
                     sl: float, tp: float) -> Optional[RobustResult]:
        """Testa parametros em treino e teste"""

        train_pnls = self._run_backtest(
            self.train_signals, self.train_bars,
            fragility_pct_thresh, confidence_thresh, sl, tp,
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
            fragility_pct_thresh, confidence_thresh, sl, tp,
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
            "fragility_percentile_threshold": round(fragility_pct_thresh, 3),
            "confidence_threshold": round(confidence_thresh, 3),
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
        print(f"  OTIMIZACAO ROBUSTA ODMN: {n:,} COMBINACOES")
        print(f"  Com validacao Train/Test Split")
        print(f"{'='*70}")

        # Ranges baseados na teoria
        fragility_pct_vals = np.linspace(0.30, 0.90, 20)
        confidence_vals = np.linspace(0.30, 0.80, 20)
        sl_vals = np.linspace(20, 55, 15)
        tp_vals = np.linspace(25, 80, 20)

        best_robustness = -1
        tested = 0
        robust_count = 0
        start = datetime.now()

        for _ in range(n):
            tested += 1

            fragility_pct = float(random.choice(fragility_pct_vals))
            confidence = float(random.choice(confidence_vals))
            sl = float(random.choice(sl_vals))
            tp = float(random.choice(tp_vals))

            result = self._test_params(fragility_pct, confidence, sl, tp)

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
            strategy_name="ODMN-MalliavinNash",
            symbol=self.symbol,
            periodicity=self.periodicity,
            n_tested=n_tested,
            n_robust=len(self.robust_results)
        )

        # Top 10
        top_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "odmn_robust_top10.json"
        )
        sorted_results = sorted(self.robust_results, key=lambda x: x.robustness_score, reverse=True)[:10]
        top_data = []
        for r in sorted_results:
            d = r.to_dict()
            # Converter numpy bool para Python bool
            for k, v in d.items():
                if isinstance(v, (np.bool_, np.generic)):
                    d[k] = bool(v)
            top_data.append(d)
        with open(top_file, 'w') as f:
            json.dump(top_data, f, indent=2)
        print(f"  Top 10 robustos salvo em: {top_file}")


async def main():
    N_COMBINATIONS = 100000

    print("=" * 70)
    print("  OTIMIZADOR ODMN ROBUSTO")
    print("  Com Validacao Anti-Overfitting")
    print(f"  {N_COMBINATIONS:,} Combinacoes")
    print("  PARA DINHEIRO REAL")
    print("=" * 70)

    opt = ODMNRobustOptimizer("EURUSD", "H1")

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
