#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR DSG V2.0 - COM WALK-FORWARD VALIDATION
================================================================================

Baseado na metodologia do PRM Optimizer:
1. Walk-Forward Validation (4 janelas progressivas)
2. Filtros rigorosos para dinheiro real
3. Custos realistas (spread 1.5 pips, slippage 0.8 pips)
4. Ranges calibrados baseados na distribuicao REAL dos dados

DIFERENTE DO PRM:
- DSG usa thresholds em escalas muito diferentes
- Ricci Scalar: -51000 a -49500 (sempre muito negativo)
- Tidal Force: 0.0001 a 0.067 (muito pequeno)
- DSG tem menos sinais, precisa de filtros mais relaxados

CORREÇÕES V2.1 (Quarta Auditoria 25/12/2025):
1. FILTROS CENTRALIZADOS: Importa de config/optimizer_filters.py
2. CUSTOS CENTRALIZADOS: Importa de config/execution_costs.py (não mais hardcoded)
3. WALK-FORWARD CORRETO: Janela deslizante real (não expanding window)
4. Consistência garantida entre optimizer.py e optimizer_wf.py

================================================================================
"""

import sys
import os
import json
import asyncio
import random
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.fxopen_historical_ws import Bar, download_historical_data
from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional

from config.execution_costs import (
    SPREAD_PIPS,
    SLIPPAGE_PIPS,
    get_pip_value,
)

# CORREÇÃO V2.1: Importar filtros centralizados
from config.optimizer_filters import (
    MIN_TRADES_TRAIN, MIN_TRADES_TEST,
    MIN_WIN_RATE, MAX_WIN_RATE,
    MIN_PROFIT_FACTOR, MAX_PROFIT_FACTOR,
    MAX_DRAWDOWN,
    MIN_WIN_RATE_TEST, MAX_WIN_RATE_TEST,
    MIN_PROFIT_FACTOR_TEST, MAX_PROFIT_FACTOR_TEST,
    MAX_DRAWDOWN_TEST,
    MIN_PF_RATIO, MIN_WR_RATIO,
    MIN_WINDOWS_PASSED, MIN_EXPECTANCY_PIPS,
)


@dataclass
class DSGSignal:
    """Sinal pre-calculado do DSG"""
    bar_idx: int
    signal_price: float
    next_bar_idx: int
    entry_price: float
    high: float
    low: float
    ricci_scalar: float
    tidal_force: float
    event_horizon_distance: float
    ricci_collapsing: bool
    crossing_horizon: bool
    geodesic_direction: int
    signal: int


@dataclass
class BacktestResult:
    """Resultado de um backtest"""
    trades: int
    wins: int
    losses: int
    total_pnl: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    avg_trade: float
    expectancy: float

    def is_valid(self, min_trades=30, max_win_rate=0.65, min_win_rate=0.28,
                 max_pf=4.5, min_pf=1.05, max_dd=0.45) -> bool:
        if self.trades < min_trades:
            return False
        if self.win_rate > max_win_rate or self.win_rate < min_win_rate:
            return False
        if self.profit_factor > max_pf or self.profit_factor < min_pf:
            return False
        if self.max_drawdown > max_dd:
            return False
        return True


@dataclass
class WalkForwardWindow:
    """Resultado de uma janela walk-forward"""
    window_idx: int
    train_result: BacktestResult
    test_result: BacktestResult
    pf_ratio: float
    wr_ratio: float
    passed: bool


@dataclass
class RobustResult:
    """Resultado robusto com validacao walk-forward"""
    params: Dict
    windows: List[WalkForwardWindow]
    combined_train: BacktestResult
    combined_test: BacktestResult
    overall_robustness: float
    degradation: float
    all_passed: bool

    def to_dict(self) -> Dict:
        return {
            "params": self.params,
            "walk_forward": {
                "windows": len(self.windows),
                "all_passed": self.all_passed,
            },
            "train": {
                "trades": self.combined_train.trades,
                "win_rate": round(self.combined_train.win_rate, 4),
                "profit_factor": round(self.combined_train.profit_factor, 4),
                "total_pnl_pips": round(self.combined_train.total_pnl, 2),
                "max_drawdown": round(self.combined_train.max_drawdown, 4),
                "expectancy": round(self.combined_train.expectancy, 2),
            },
            "test": {
                "trades": self.combined_test.trades,
                "win_rate": round(self.combined_test.win_rate, 4),
                "profit_factor": round(self.combined_test.profit_factor, 4),
                "total_pnl_pips": round(self.combined_test.total_pnl, 2),
                "max_drawdown": round(self.combined_test.max_drawdown, 4),
                "expectancy": round(self.combined_test.expectancy, 2),
            },
            "robustness_score": round(self.overall_robustness, 4),
            "degradation_pct": round(self.degradation * 100, 2),
        }


class DSGWalkForwardOptimizer:
    """
    Otimizador DSG V2.1 com Walk-Forward Validation
    Baseado na metodologia do PRM Optimizer

    CORREÇÃO V2.1: Custos e filtros agora são importados de config/
    Não mais hardcoded na classe para garantir consistência
    """

    # CORREÇÃO V2.1: Custos importados de config/execution_costs.py
    # (SPREAD_PIPS e SLIPPAGE_PIPS agora vem do import)

    # CORREÇÃO V2.1: Filtros importados de config/optimizer_filters.py
    # (Todas as constantes MIN_*, MAX_* agora vem do import)

    def __init__(self, symbol: str = "EURUSD", periodicity: str = "H1"):
        self.symbol = symbol
        self.periodicity = periodicity
        self.pip = get_pip_value(symbol)

        self.bars: List[Bar] = []
        self.signals: List[DSGSignal] = []
        self.robust_results: List[RobustResult] = []
        self.best: Optional[RobustResult] = None

    async def load_and_precompute(self, start_date: datetime, end_date: datetime):
        """Carrega dados e pre-calcula sinais DSG"""
        print("\n" + "=" * 70)
        print("  CARREGANDO DADOS - DSG Walk-Forward Optimizer V2.0")
        print("=" * 70)

        self.bars = await download_historical_data(
            symbol=self.symbol,
            periodicity=self.periodicity,
            start_time=start_date,
            end_time=end_date
        )
        print(f"  Total de barras: {len(self.bars)}")

        if len(self.bars) < 500:
            print("  ERRO: Dados insuficientes!")
            return False

        # Pre-calcular DSG
        print("\n  Pre-calculando sinais DSG...")
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

            if i >= len(self.bars) - 1:
                continue

            try:
                prices_arr = np.array(prices_buf)
                result = dsg.analyze(prices_arr)

                next_bar = self.bars[i + 1]

                self.signals.append(DSGSignal(
                    bar_idx=i,
                    signal_price=bar.close,
                    next_bar_idx=i + 1,
                    entry_price=next_bar.open,
                    high=next_bar.high,
                    low=next_bar.low,
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

            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(self.bars)} barras...")

        print(f"\n  Sinais pre-calculados: {len(self.signals)}")

        if self.signals:
            ricci_vals = [s.ricci_scalar for s in self.signals]
            tidal_vals = [s.tidal_force for s in self.signals]
            print(f"\n  Distribuicao REAL:")
            print(f"    Ricci: min={min(ricci_vals):.2f}, max={max(ricci_vals):.2f}, mean={np.mean(ricci_vals):.2f}")
            print(f"    Tidal: min={min(tidal_vals):.6f}, max={max(tidal_vals):.6f}, mean={np.mean(tidal_vals):.6f}")

            # Calcular percentis para calibrar ranges
            ricci_p10 = np.percentile(ricci_vals, 10)
            ricci_p90 = np.percentile(ricci_vals, 90)
            tidal_p10 = np.percentile(tidal_vals, 10)
            tidal_p90 = np.percentile(tidal_vals, 90)
            print(f"    Ricci P10={ricci_p10:.2f}, P90={ricci_p90:.2f}")
            print(f"    Tidal P10={tidal_p10:.6f}, P90={tidal_p90:.6f}")

        return len(self.signals) > 100

    def _calculate_result(self, pnls: List[float]) -> BacktestResult:
        """Calcula metricas de backtest"""
        if not pnls:
            return BacktestResult(0, 0, 0, 0, 0, 0, 1.0, 0, 0)

        wins = sum(1 for p in pnls if p > 0)
        losses = len(pnls) - wins
        total_pnl = sum(pnls)
        win_rate = wins / len(pnls)

        gross_profit = sum(p for p in pnls if p > 0) or 0.001
        gross_loss = abs(sum(p for p in pnls if p <= 0)) or 0.001
        profit_factor = gross_profit / gross_loss

        equity = np.cumsum([0] + pnls)
        peak = np.maximum.accumulate(equity + 10000)
        drawdowns = (peak - (equity + 10000)) / peak
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

        avg_trade = total_pnl / len(pnls)
        expectancy = avg_trade

        return BacktestResult(
            trades=len(pnls),
            wins=wins,
            losses=losses,
            total_pnl=total_pnl,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            avg_trade=avg_trade,
            expectancy=expectancy
        )

    def _run_backtest(self, signals: List[DSGSignal], bars: List[Bar],
                      ricci_thresh: float, tidal_thresh: float,
                      sl: float, tp: float, bar_offset: int = 0) -> List[float]:
        """Executa backtest com custos realistas"""
        if tp <= sl * 1.2:
            return []

        entries = []
        for s in signals:
            # Condicoes DSG: Ricci em colapso OU muito negativo + Tidal alto
            ricci_collapse = s.ricci_scalar < ricci_thresh or s.ricci_collapsing
            high_tidal = s.tidal_force > tidal_thresh
            crossing = s.crossing_horizon

            conditions = sum([ricci_collapse, high_tidal, crossing])

            if conditions >= 2 and s.geodesic_direction != 0:
                execution_idx = s.next_bar_idx - bar_offset
                entries.append((execution_idx, s.entry_price, s.geodesic_direction))

        if len(entries) < 3:
            return []

        pnls = []
        pip = self.pip
        # CORREÇÃO V2.1: Usa constantes importadas de config/execution_costs.py
        spread = SPREAD_PIPS * pip
        slippage = SLIPPAGE_PIPS * pip
        total_cost = spread + slippage

        last_exit_idx = -1

        for entry_idx, entry_price_raw, direction in entries:
            if entry_idx < 0 or entry_idx >= len(bars) - 1:
                continue
            if entry_idx <= last_exit_idx:
                continue

            # Aplicar custos
            if direction == 1:
                entry_price = entry_price_raw + total_cost / 2
                stop_price = entry_price - sl * pip
                take_price = entry_price + tp * pip
            else:
                entry_price = entry_price_raw - total_cost / 2
                stop_price = entry_price + sl * pip
                take_price = entry_price - tp * pip

            exit_price = None
            exit_bar_idx = entry_idx
            max_bars = min(200, len(bars) - entry_idx - 1)

            for j in range(1, max_bars + 1):
                bar_idx = entry_idx + j
                if bar_idx >= len(bars):
                    break

                bar = bars[bar_idx]

                # Gaps
                if direction == 1:
                    if bar.open <= stop_price:
                        exit_price = bar.open - slippage
                        exit_bar_idx = bar_idx
                        break
                    if bar.open >= take_price:
                        exit_price = bar.open - slippage
                        exit_bar_idx = bar_idx
                        break
                else:
                    if bar.open >= stop_price:
                        exit_price = bar.open + slippage
                        exit_bar_idx = bar_idx
                        break
                    if bar.open <= take_price:
                        exit_price = bar.open + slippage
                        exit_bar_idx = bar_idx
                        break

                # High/Low
                if direction == 1:
                    if bar.low <= stop_price:
                        exit_price = stop_price - slippage
                        exit_bar_idx = bar_idx
                        break
                    if bar.high >= take_price:
                        exit_price = take_price - slippage
                        exit_bar_idx = bar_idx
                        break
                else:
                    if bar.high >= stop_price:
                        exit_price = stop_price + slippage
                        exit_bar_idx = bar_idx
                        break
                    if bar.low <= take_price:
                        exit_price = take_price + slippage
                        exit_bar_idx = bar_idx
                        break

            if exit_price is None:
                exit_bar_idx = min(entry_idx + max_bars, len(bars) - 1)
                last_bar = bars[exit_bar_idx]
                if direction == 1:
                    exit_price = last_bar.close - slippage
                else:
                    exit_price = last_bar.close + slippage

            if direction == 1:
                pnl_pips = (exit_price - entry_price) / pip
            else:
                pnl_pips = (entry_price - exit_price) / pip

            pnls.append(pnl_pips)
            last_exit_idx = exit_bar_idx

        return pnls

    def _create_walk_forward_windows(self, n_windows: int = 4) -> List[Tuple[int, int, int, int]]:
        """
        CORREÇÃO V2.1: Cria janelas walk-forward com JANELA DESLIZANTE real

        ANTES (expanding window - INCORRETO):
        - Janela 1: Treino [0-17.5%], Teste [17.5-25%]
        - Janela 2: Treino [0-35%], Teste [35-50%]
        - Problema: Cada janela treina desde o início, dados antigos poluem

        AGORA (sliding window - CORRETO):
        - Janela 1: Treino [0-60%], Teste [60-80%]
        - Janela 2: Treino [20-80%], Teste [80-100%]
        - Problema resolvido: Cada janela treina em período diferente
        """
        total_bars = len(self.bars)

        # Configuração da janela deslizante
        train_size = int(total_bars * 0.50)  # 50% para treino
        test_size = int(total_bars * 0.20)   # 20% para teste
        step_size = total_bars // n_windows  # Passo entre janelas

        windows = []
        for i in range(n_windows):
            # Calcula início da janela de treino (desliza para frente)
            train_start = i * step_size

            # Garante que não ultrapassa o total de barras
            if train_start + train_size + test_size > total_bars:
                # Ajusta para usar dados até o final
                train_start = max(0, total_bars - train_size - test_size)

            train_end = train_start + train_size
            test_start = train_end
            test_end = min(test_start + test_size, total_bars)

            # Verifica se temos dados suficientes
            if train_end - train_start < 100 or test_end - test_start < 50:
                continue

            windows.append((train_start, train_end, test_start, test_end))

        return windows

    def _test_params(self, ricci_thresh: float, tidal_thresh: float,
                     sl: float, tp: float) -> Optional[RobustResult]:
        """Testa parametros com Walk-Forward Validation"""
        windows = self._create_walk_forward_windows(n_windows=4)
        wf_results = []
        all_train_pnls = []
        all_test_pnls = []

        for idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            train_signals = [s for s in self.signals if train_start <= s.bar_idx < train_end]
            test_signals = [s for s in self.signals if test_start <= s.bar_idx < test_end]

            train_bars = self.bars[train_start:train_end]
            test_bars = self.bars[test_start:test_end]

            if len(train_signals) < 15 or len(test_signals) < 8:
                return None

            train_pnls = self._run_backtest(
                train_signals, train_bars,
                ricci_thresh, tidal_thresh, sl, tp,
                bar_offset=train_start
            )
            train_result = self._calculate_result(train_pnls)

            if train_result.trades < 15 or train_result.profit_factor < 1.0:
                return None

            test_pnls = self._run_backtest(
                test_signals, test_bars,
                ricci_thresh, tidal_thresh, sl, tp,
                bar_offset=test_start
            )
            test_result = self._calculate_result(test_pnls)

            if test_result.trades < 8 or test_result.profit_factor < 0.85:
                return None

            pf_ratio = test_result.profit_factor / train_result.profit_factor if train_result.profit_factor > 0 else 0
            wr_ratio = test_result.win_rate / train_result.win_rate if train_result.win_rate > 0 else 0

            passed = pf_ratio >= 0.50 and wr_ratio >= 0.50 and test_result.profit_factor >= 0.90

            wf_results.append(WalkForwardWindow(
                window_idx=idx,
                train_result=train_result,
                test_result=test_result,
                pf_ratio=pf_ratio,
                wr_ratio=wr_ratio,
                passed=passed
            ))

            all_train_pnls.extend(train_pnls)
            all_test_pnls.extend(test_pnls)

        # CORREÇÃO V2.1: Usa MIN_WINDOWS_PASSED do config/optimizer_filters.py
        passed_count = sum(1 for w in wf_results if w.passed)
        if passed_count < MIN_WINDOWS_PASSED:
            return None

        combined_train = self._calculate_result(all_train_pnls)
        combined_test = self._calculate_result(all_test_pnls)

        # CORREÇÃO V2.1: Usa filtros CENTRALIZADOS de config/optimizer_filters.py
        if not combined_train.is_valid(
            min_trades=MIN_TRADES_TRAIN,
            min_pf=MIN_PROFIT_FACTOR,
            min_win_rate=MIN_WIN_RATE,
            max_win_rate=MAX_WIN_RATE,
            max_dd=MAX_DRAWDOWN
        ):
            return None

        # CORREÇÃO V2.1: Usa filtros CENTRALIZADOS para teste
        if not combined_test.is_valid(
            min_trades=MIN_TRADES_TEST,
            min_pf=MIN_PROFIT_FACTOR_TEST,
            min_win_rate=MIN_WIN_RATE_TEST,
            max_win_rate=MAX_WIN_RATE_TEST,
            max_dd=MAX_DRAWDOWN_TEST
        ):
            return None

        # CORREÇÃO V2.1: Usa MIN_EXPECTANCY_PIPS do config
        if combined_train.expectancy < MIN_EXPECTANCY_PIPS:
            return None

        avg_pf_ratio = np.mean([w.pf_ratio for w in wf_results])
        avg_wr_ratio = np.mean([w.wr_ratio for w in wf_results])
        degradation = 1.0 - (avg_pf_ratio + avg_wr_ratio) / 2
        robustness = max(0, min(1, 1 - degradation))

        params = {
            "ricci_collapse_threshold": round(ricci_thresh, 2),
            "tidal_force_threshold": round(tidal_thresh, 6),
            "stop_loss_pips": round(sl, 1),
            "take_profit_pips": round(tp, 1)
        }

        return RobustResult(
            params=params,
            windows=wf_results,
            combined_train=combined_train,
            combined_test=combined_test,
            overall_robustness=robustness,
            degradation=degradation,
            all_passed=(passed_count == 4)
        )

    def optimize(self, n: int = 300000, seed: int = 42) -> Optional[RobustResult]:
        """Executa otimizacao robusta com Walk-Forward"""
        if not self.signals:
            print("  ERRO: Dados nao carregados!")
            return None

        random.seed(seed)
        np.random.seed(seed)

        print(f"\n{'='*70}")
        print(f"  OTIMIZACAO DSG V2.0 - WALK-FORWARD VALIDATION")
        print(f"  {n:,} Combinacoes | 4 Janelas | Custos Realistas")
        print(f"{'='*70}")

        # Ranges calibrados baseados na distribuicao REAL
        # Ricci: sempre muito negativo (-51000 a -49500)
        # Como ricci_collapse sempre e True, o threshold nao importa muito
        # Mas vamos testar uma faixa que permite variar a sensibilidade
        ricci_vals = np.linspace(-51000, -49000, 15)

        # Tidal: 0.0001 a 0.067 (media 0.009)
        # Vamos testar percentis para calibrar
        tidal_vals = np.linspace(0.001, 0.03, 20)

        # SL/TP
        sl_vals = np.linspace(20, 60, 15)
        tp_vals = np.linspace(35, 100, 20)

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

                if result.overall_robustness > best_robustness:
                    best_robustness = result.overall_robustness
                    self.best = result

                    print(f"\n  [ROBUSTO #{robust_count}] Robustez={result.overall_robustness:.4f}")
                    print(f"    Params: Ricci<{ricci:.0f}, Tidal>{tidal:.4f}, SL={sl:.0f}, TP={tp:.0f}")
                    print(f"    TREINO: {result.combined_train.trades} trades, "
                          f"WR={result.combined_train.win_rate:.1%}, "
                          f"PF={result.combined_train.profit_factor:.2f}")
                    print(f"    TESTE:  {result.combined_test.trades} trades, "
                          f"WR={result.combined_test.win_rate:.1%}, "
                          f"PF={result.combined_test.profit_factor:.2f}")

            if tested % 30000 == 0:
                elapsed = (datetime.now() - start).total_seconds()
                rate = tested / elapsed
                eta = (n - tested) / rate / 60
                print(f"  {tested:,}/{n:,} ({tested/n*100:.1f}%) | "
                      f"Robustos: {robust_count} | "
                      f"Vel: {rate:.0f}/s | ETA: {eta:.1f}min")

        elapsed = (datetime.now() - start).total_seconds()
        print(f"\n{'='*70}")
        print(f"  CONCLUIDO em {elapsed/60:.1f}min")
        print(f"  Testados: {tested:,} | Robustos: {robust_count}")
        print(f"{'='*70}")

        return self.best

    def save(self, n_tested: int = 0):
        """Salva melhor configuracao"""
        if not self.best:
            print("  Nenhuma configuracao robusta encontrada!")
            return

        configs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "configs"
        )
        os.makedirs(configs_dir, exist_ok=True)

        best_file = os.path.join(configs_dir, "dsg_walkforward_robust.json")

        config = {
            "strategy": "DSG-SingularidadeGravitacional",
            "symbol": self.symbol,
            "periodicity": self.periodicity,
            "version": "2.1-walkforward",  # CORREÇÃO V2.1
            "optimized_at": datetime.now(timezone.utc).isoformat(),
            "validation": {
                "method": "walk_forward_sliding",  # CORREÇÃO V2.1: sliding window
                "n_windows": 4,
                "combinations_tested": n_tested,
                "robust_found": len(self.robust_results),
                "costs": {
                    # CORREÇÃO V2.1: Usa constantes importadas
                    "spread_pips": SPREAD_PIPS,
                    "slippage_pips": SLIPPAGE_PIPS,
                }
            },
            "parameters": self.best.params,
            "performance": self.best.to_dict(),
        }

        with open(best_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"\n  Melhor config salva em: {best_file}")

        # Top 10
        top_file = os.path.join(configs_dir, "dsg_walkforward_top10.json")
        sorted_results = sorted(
            self.robust_results,
            key=lambda x: x.overall_robustness,
            reverse=True
        )[:10]

        top_data = [r.to_dict() for r in sorted_results]
        with open(top_file, 'w') as f:
            json.dump(top_data, f, indent=2, default=str)
        print(f"  Top 10 salvo em: {top_file}")


async def main():
    N_COMBINATIONS = 300000

    print("=" * 70)
    print("  OTIMIZADOR DSG V2.0 - WALK-FORWARD VALIDATION")
    print("=" * 70)
    print("\n  Baseado na metodologia do PRM Optimizer:")
    print("    - Walk-Forward Validation (4 janelas)")
    print("    - Custos realistas (spread 1.5 + slippage 0.8)")
    print("    - Filtros calibrados para DSG")
    print("=" * 70)

    opt = DSGWalkForwardOptimizer("EURUSD", "H1")

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    print(f"\n  Periodo: {start.date()} a {end.date()}")

    if await opt.load_and_precompute(start, end):
        best = opt.optimize(N_COMBINATIONS)
        if best:
            print(f"\n{'='*70}")
            print(f"  MELHOR RESULTADO ROBUSTO")
            print(f"{'='*70}")
            print(f"  Robustez: {best.overall_robustness:.4f}")
            print(f"  Janelas aprovadas: {sum(1 for w in best.windows if w.passed)}/4")
            print(f"\n  TREINO:")
            print(f"    Trades: {best.combined_train.trades}")
            print(f"    Win Rate: {best.combined_train.win_rate:.1%}")
            print(f"    Profit Factor: {best.combined_train.profit_factor:.2f}")
            print(f"    Expectancy: {best.combined_train.expectancy:.1f} pips/trade")
            print(f"\n  TESTE:")
            print(f"    Trades: {best.combined_test.trades}")
            print(f"    Win Rate: {best.combined_test.win_rate:.1%}")
            print(f"    Profit Factor: {best.combined_test.profit_factor:.2f}")
            print(f"    Expectancy: {best.combined_test.expectancy:.1f} pips/trade")
            print(f"\n  PARAMETROS:")
            for k, v in best.params.items():
                print(f"    {k}: {v}")
            print(f"{'='*70}")

            opt.save(n_tested=N_COMBINATIONS)
        else:
            print("\n  AVISO: Nenhuma configuracao passou nos filtros!")
    else:
        print("\n  ERRO: Falha ao carregar dados!")


if __name__ == "__main__":
    asyncio.run(main())
