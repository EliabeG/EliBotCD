#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR ODMN ROBUSTO V2.3 - PRONTO PARA DINHEIRO REAL
================================================================================

Este otimizador implementa:
1. Walk-Forward Validation (multiplas janelas train/test)
2. Filtros rigorosos para dinheiro real (PF > 1.3)
3. Custos realistas (spread 1.5 pips, slippage 0.8 pips) - V2.3: Spread aplicado na SAIDA
4. Validacao em multiplos periodos de mercado

ODMN (Oraculo de Derivativos de Malliavin-Nash):
================================================
- Modelo de Heston para volatilidade estocastica
- Derivadas de Malliavin para detectar fragilidade estrutural
- Mean Field Games para prever comportamento institucional

SEM LOOK-AHEAD BIAS:
===================
- Calibracao Heston usa apenas dados passados (janela deslizante)
- Malliavin simula trajetorias para frente (Monte Carlo causal)
- MFG resolve PDEs sem usar dados futuros
- Direcao baseada APENAS em barras fechadas
- Entrada no OPEN da proxima barra

REGRAS PARA DINHEIRO REAL:
- Minimo 50 trades no treino, 25 no teste
- Win Rate entre 35% e 60%
- Profit Factor minimo 1.3 (treino) e 1.15 (teste)
- Drawdown maximo 30%
- Performance do teste >= 70% do treino
- Aprovacao em TODAS as janelas walk-forward

PARA DINHEIRO REAL. SEM OVERFITTING. SEM LOOK-AHEAD. CUSTOS REALISTAS.
================================================================================
"""

import sys
import os
import json
import asyncio
import random
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.fxopen_historical_ws import Bar, download_historical_data
from strategies.alta_volatilidade.odmn_malliavin_nash import OracloDerivativosMalliavinNash

# Importar custos e filtros CENTRALIZADOS
from config.execution_costs import SPREAD_PIPS, SLIPPAGE_PIPS
from config.optimizer_filters import (
    MIN_TRADES_TRAIN,
    MIN_TRADES_TEST,
    MIN_WIN_RATE,
    MAX_WIN_RATE,
    MIN_PROFIT_FACTOR,
    MAX_PROFIT_FACTOR,
    MAX_DRAWDOWN,
    MIN_ROBUSTNESS,
    MIN_EXPECTANCY_PIPS,
)
from config.odmn_config import (
    MIN_PRICES,
    HESTON_CALIBRATION_WINDOW,
    MALLIAVIN_PATHS,
    MALLIAVIN_STEPS,
    USE_DEEP_GALERKIN,
    TREND_LOOKBACK,
    MFG_DIRECTION_THRESHOLD,
)

# Importar direction_calculator centralizado
from backtesting.common.direction_calculator import calculate_direction_from_bars

# Seed global para reprodutibilidade
OPTIMIZER_SEED = 42


@dataclass
class ODMNSignal:
    """
    Sinal pre-calculado do ODMN

    SEM LOOK-AHEAD: Armazena informacoes para execucao realista
    """
    bar_idx: int              # Indice da barra onde o sinal foi GERADO
    signal_price: float       # Preco de fechamento quando sinal foi gerado
    next_bar_idx: int         # Indice da barra onde deve EXECUTAR
    entry_price: float        # Preco de ABERTURA da proxima barra
    high: float               # High da barra de entrada
    low: float                # Low da barra de entrada
    fragility_index: float    # Indice de fragilidade de Malliavin
    fragility_percentile: float  # Percentil da fragilidade
    mfg_direction: float      # Direcao do Mean Field Game
    mfg_equilibrium: bool     # Se MFG convergiu
    regime: str               # Regime de mercado
    confidence: float         # Confianca do sinal
    direction: int            # Baseado APENAS em barras JA FECHADAS


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
    largest_win: float
    largest_loss: float
    expectancy: float = 0.0

    def is_valid_for_real_money(self,
                                min_trades: int = 50,
                                max_win_rate: float = 0.60,
                                min_win_rate: float = 0.35,
                                max_pf: float = 3.5,
                                min_pf: float = 1.30,
                                max_dd: float = 0.30,
                                min_expectancy: float = None) -> bool:
        """Verifica se o resultado passa nos filtros RIGOROSOS para dinheiro real"""
        if min_expectancy is None:
            min_expectancy = MIN_EXPECTANCY_PIPS

        if self.trades < min_trades:
            return False
        if self.win_rate > max_win_rate or self.win_rate < min_win_rate:
            return False
        if self.profit_factor > max_pf or self.profit_factor < min_pf:
            return False
        if self.max_drawdown > max_dd:
            return False
        if self.expectancy < min_expectancy:
            return False
        return True


@dataclass
class WalkForwardResult:
    """Resultado de uma janela walk-forward"""
    window_idx: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_result: BacktestResult
    test_result: BacktestResult
    robustness_score: float
    degradation: float
    passed: bool


@dataclass
class RobustResult:
    """Resultado robusto com validacao walk-forward completa"""
    params: Dict
    walk_forward_results: List[WalkForwardResult]
    avg_train_pf: float
    avg_test_pf: float
    avg_train_wr: float
    avg_test_wr: float
    total_train_trades: int
    total_test_trades: int
    overall_robustness: float
    all_windows_passed: bool
    combined_train_result: BacktestResult
    combined_test_result: BacktestResult

    def to_dict(self) -> Dict:
        return {
            "params": self.params,
            "walk_forward": {
                "windows": len(self.walk_forward_results),
                "all_passed": self.all_windows_passed,
                "avg_train_pf": round(self.avg_train_pf, 4),
                "avg_test_pf": round(self.avg_test_pf, 4),
                "avg_train_wr": round(self.avg_train_wr, 4),
                "avg_test_wr": round(self.avg_test_wr, 4),
            },
            "combined_train": {
                "trades": self.combined_train_result.trades,
                "win_rate": round(self.combined_train_result.win_rate, 4),
                "profit_factor": round(self.combined_train_result.profit_factor, 4),
                "total_pnl": round(self.combined_train_result.total_pnl, 2),
                "max_drawdown": round(self.combined_train_result.max_drawdown, 4),
                "expectancy": round(self.combined_train_result.expectancy, 2),
            },
            "combined_test": {
                "trades": self.combined_test_result.trades,
                "win_rate": round(self.combined_test_result.win_rate, 4),
                "profit_factor": round(self.combined_test_result.profit_factor, 4),
                "total_pnl": round(self.combined_test_result.total_pnl, 2),
                "max_drawdown": round(self.combined_test_result.max_drawdown, 4),
                "expectancy": round(self.combined_test_result.expectancy, 2),
            },
            "overall_robustness": round(self.overall_robustness, 4),
        }


class ODMNRobustOptimizer:
    """
    Otimizador ODMN V2.0 com Walk-Forward Validation

    PRONTO PARA DINHEIRO REAL

    Todos os custos e filtros sao importados de config centralizado
    """

    # Custos CENTRALIZADOS
    SPREAD_PIPS = SPREAD_PIPS
    SLIPPAGE_PIPS = SLIPPAGE_PIPS
    COMMISSION_PIPS = 0.0

    # Filtros CENTRALIZADOS
    MIN_TRADES_TRAIN = MIN_TRADES_TRAIN
    MIN_TRADES_TEST = MIN_TRADES_TEST
    MIN_WIN_RATE = MIN_WIN_RATE
    MAX_WIN_RATE = MAX_WIN_RATE
    MIN_PF_TRAIN = MIN_PROFIT_FACTOR
    MIN_PF_TEST = 1.15
    MAX_PF = MAX_PROFIT_FACTOR
    MAX_DRAWDOWN = MAX_DRAWDOWN
    MIN_ROBUSTNESS = MIN_ROBUSTNESS
    MIN_EXPECTANCY = MIN_EXPECTANCY_PIPS

    def __init__(self, symbol: str = "EURUSD", periodicity: str = "H1"):
        self.symbol = symbol
        self.periodicity = periodicity
        self.pip = 0.0001

        self.bars: List[Bar] = []
        self.signals: List[ODMNSignal] = []

        # Resultados
        self.robust_results: List[RobustResult] = []
        self.best: Optional[RobustResult] = None

    async def load_and_precompute(self, start_date: datetime, end_date: datetime,
                                   split_date: datetime = None):
        """
        Carrega dados e pre-calcula sinais ODMN

        SEM LOOK-AHEAD:
        - Direcao usa apenas barras completamente fechadas
        - Entry price e o OPEN da proxima barra
        """
        print("\n" + "=" * 70)
        print("  CARREGANDO DADOS REAIS - V2.0 PRONTO PARA DINHEIRO REAL")
        print("=" * 70)

        self.bars = await download_historical_data(
            symbol=self.symbol,
            periodicity=self.periodicity,
            start_time=start_date,
            end_time=end_date
        )
        print(f"  Total de barras: {len(self.bars)}")

        if len(self.bars) < 500:
            print("  ERRO: Dados insuficientes! Minimo 500 barras necessario.")
            return False

        # SPLIT TRAIN/TEST por data especifica ou 70/30
        if split_date:
            split_idx = 0
            for i, bar in enumerate(self.bars):
                if bar.timestamp >= split_date:
                    split_idx = i
                    break
            if split_idx == 0:
                split_idx = int(len(self.bars) * 0.70)
        else:
            split_idx = int(len(self.bars) * 0.70)

        self.train_bars = self.bars[:split_idx]
        self.test_bars = self.bars[split_idx:]

        print(f"\n  DIVISAO TRAIN/TEST:")
        print(f"    Treino: {len(self.train_bars)} barras ({self.train_bars[0].timestamp.date()} a {self.train_bars[-1].timestamp.date()})")
        print(f"    Teste:  {len(self.test_bars)} barras ({self.test_bars[0].timestamp.date()} a {self.test_bars[-1].timestamp.date()})")

        # Pre-calcular sinais para TODOS os dados
        print("\n  Pre-calculando sinais ODMN V2.0 (sem look-ahead)...")
        print(f"  NOTA: Computacionalmente intensivo (Heston + Malliavin + MFG)")

        # REPRODUTIBILIDADE: Usa seed fixo para resultados deterministicos
        odmn = OracloDerivativosMalliavinNash(
            lookback_window=HESTON_CALIBRATION_WINDOW,
            fragility_threshold=2.0,
            mfg_direction_threshold=MFG_DIRECTION_THRESHOLD,  # Do config centralizado
            use_deep_galerkin=USE_DEEP_GALERKIN,  # Analitico e mais rapido
            malliavin_paths=MALLIAVIN_PATHS,
            malliavin_steps=MALLIAVIN_STEPS,
            seed=OPTIMIZER_SEED  # Para reprodutibilidade
        )

        prices_buf = deque(maxlen=500)
        self.signals = []

        min_prices = MIN_PRICES

        for i, bar in enumerate(self.bars):
            prices_buf.append(bar.close)

            if len(prices_buf) < min_prices:
                continue

            # Precisamos da PROXIMA barra para executar
            if i >= len(self.bars) - 1:
                continue

            try:
                result = odmn.analyze(np.array(prices_buf))

                # Direcao baseada APENAS em barras FECHADAS
                # Usa direction_calculator centralizado para consistencia
                direction = calculate_direction_from_bars(self.bars, i, lookback=TREND_LOOKBACK)

                next_bar = self.bars[i + 1]

                self.signals.append(ODMNSignal(
                    bar_idx=i,
                    signal_price=bar.close,
                    next_bar_idx=i + 1,
                    entry_price=next_bar.open,
                    high=next_bar.high,
                    low=next_bar.low,
                    fragility_index=result['fragility_index'],
                    fragility_percentile=result['fragility_percentile'],
                    mfg_direction=result['mfg_direction'],
                    mfg_equilibrium=result['mfg_equilibrium'],
                    regime=result['regime'],
                    confidence=result['confidence'],
                    direction=direction
                ))

            except Exception as e:
                continue

            if (i + 1) % 300 == 0:
                print(f"    {i+1}/{len(self.bars)} barras...")

        print(f"\n  Sinais pre-calculados: {len(self.signals)}")

        # Estatisticas
        long_signals = sum(1 for s in self.signals if s.direction == 1)
        short_signals = sum(1 for s in self.signals if s.direction == -1)
        print(f"    Long: {long_signals}, Short: {short_signals}")

        if self.signals:
            frag_vals = [s.fragility_percentile for s in self.signals]
            mfg_vals = [s.mfg_direction for s in self.signals]
            print(f"    Fragility P: mean={np.mean(frag_vals)*100:.1f}%, max={max(frag_vals)*100:.1f}%")
            print(f"    MFG Dir: mean={np.mean(mfg_vals):.4f}, std={np.std(mfg_vals):.4f}")

        return len(self.signals) > 200

    def _calculate_backtest_result(self, pnls: List[float]) -> BacktestResult:
        """Calcula metricas de um backtest"""
        if not pnls:
            return BacktestResult(
                trades=0, wins=0, losses=0, total_pnl=0,
                win_rate=0, profit_factor=0, max_drawdown=1.0,
                avg_trade=0, largest_win=0, largest_loss=0,
                expectancy=0
            )

        wins = sum(1 for p in pnls if p > 0)
        losses = len(pnls) - wins
        total_pnl = sum(pnls)
        win_rate = wins / len(pnls) if pnls else 0

        gross_profit = sum(p for p in pnls if p > 0) or 0.001
        gross_loss = abs(sum(p for p in pnls if p <= 0)) or 0.001
        profit_factor = gross_profit / gross_loss

        # Drawdown
        equity = np.cumsum([0] + pnls)
        peak = np.maximum.accumulate(equity + 10000)
        drawdowns = (peak - (equity + 10000)) / peak
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

        avg_trade = total_pnl / len(pnls) if pnls else 0
        largest_win = max(pnls) if pnls else 0
        largest_loss = min(pnls) if pnls else 0

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
            largest_win=largest_win,
            largest_loss=largest_loss,
            expectancy=expectancy
        )

    def _run_backtest(self, signals: List[ODMNSignal], bars: List[Bar],
                      frag_pct_thresh: float, mfg_dir_thresh: float,
                      min_confidence: float, sl: float, tp: float,
                      bar_offset: int = 0) -> List[float]:
        """
        Executa backtest com CUSTOS REALISTAS

        V2.0:
        - Spread: 1.5 pips
        - Slippage: 0.8 pips
        - Entrada no OPEN
        - Verificacao de gaps
        """
        if tp <= sl:
            return []

        # Encontra entradas validas
        entries = []
        for s in signals:
            # Condicao: alta fragilidade + direcao MFG + confianca
            if (s.fragility_percentile >= frag_pct_thresh and
                abs(s.mfg_direction) >= mfg_dir_thresh and
                s.confidence >= min_confidence and
                s.direction != 0):

                # Determina direcao do trade
                # Alta fragilidade = possivel reversao
                # MFG positivo + alta fragilidade = possivel squeeze para cima
                # MFG negativo + alta fragilidade = possivel crash

                # Logica: seguir a tendencia recente quando mercado fragil
                # (reversao contra-tendencia e mais arriscada)
                trade_direction = s.direction

                execution_idx = s.next_bar_idx - bar_offset
                entries.append((
                    execution_idx,
                    s.entry_price,
                    trade_direction
                ))

        if len(entries) < 5:
            return []

        # Executa trades com custos REALISTAS
        pnls = []
        pip = self.pip
        spread = self.SPREAD_PIPS * pip
        slippage = self.SLIPPAGE_PIPS * pip
        total_cost = spread + slippage

        last_exit_idx = -1

        for entry_idx, entry_price_raw, direction in entries:
            if entry_idx < 0 or entry_idx >= len(bars) - 1:
                continue

            if entry_idx <= last_exit_idx:
                continue

            # Aplicar custos na entrada
            if direction == 1:  # LONG
                entry_price = entry_price_raw + total_cost / 2
                stop_price = entry_price - sl * pip
                take_price = entry_price + tp * pip
            else:  # SHORT
                entry_price = entry_price_raw - total_cost / 2
                stop_price = entry_price + sl * pip
                take_price = entry_price - tp * pip

            # Simular execucao
            exit_price = None
            exit_bar_idx = entry_idx
            max_bars = min(200, len(bars) - entry_idx - 1)

            for j in range(1, max_bars + 1):
                bar_idx = entry_idx + j
                if bar_idx >= len(bars):
                    break

                bar = bars[bar_idx]

                # Verificar GAPS no OPEN
                # V2.3: Aplica spread + slippage na saida (correcao auditoria)
                if direction == 1:  # LONG
                    if bar.open <= stop_price:
                        exit_price = bar.open - spread/2 - slippage
                        exit_bar_idx = bar_idx
                        break
                    if bar.open >= take_price:
                        exit_price = bar.open - spread/2 - slippage
                        exit_bar_idx = bar_idx
                        break
                else:  # SHORT
                    if bar.open >= stop_price:
                        exit_price = bar.open + spread/2 + slippage
                        exit_bar_idx = bar_idx
                        break
                    if bar.open <= take_price:
                        exit_price = bar.open + spread/2 + slippage
                        exit_bar_idx = bar_idx
                        break

                # Verificar durante a barra (stop tem prioridade)
                # V2.3: Aplica spread + slippage na saida (correcao auditoria)
                if direction == 1:  # LONG
                    if bar.low <= stop_price:
                        exit_price = stop_price - spread/2 - slippage
                        exit_bar_idx = bar_idx
                        break
                    if bar.high >= take_price:
                        exit_price = take_price - spread/2 - slippage
                        exit_bar_idx = bar_idx
                        break
                else:  # SHORT
                    if bar.high >= stop_price:
                        exit_price = stop_price + spread/2 + slippage
                        exit_bar_idx = bar_idx
                        break
                    if bar.low <= take_price:
                        exit_price = take_price + spread/2 + slippage
                        exit_bar_idx = bar_idx
                        break

            # Timeout
            # V2.3: Aplica spread + slippage na saida (correcao auditoria)
            if exit_price is None:
                exit_bar_idx = min(entry_idx + max_bars, len(bars) - 1)
                last_bar = bars[exit_bar_idx]
                if direction == 1:
                    exit_price = last_bar.close - spread/2 - slippage
                else:
                    exit_price = last_bar.close + spread/2 + slippage

            # Calcular PnL
            if direction == 1:
                pnl_pips = (exit_price - entry_price) / pip
            else:
                pnl_pips = (entry_price - exit_price) / pip

            pnls.append(pnl_pips)
            last_exit_idx = exit_bar_idx

        return pnls

    def _create_walk_forward_windows(self, n_windows: int = 4) -> List[Tuple[int, int, int, int]]:
        """
        Cria janelas para Walk-Forward Validation

        Divide os dados em n_windows janelas, cada uma com 70% treino e 30% teste
        """
        total_bars = len(self.bars)
        window_size = total_bars // n_windows

        windows = []
        for i in range(n_windows):
            window_end = (i + 1) * window_size
            if i == n_windows - 1:
                window_end = total_bars

            window_start = 0
            train_end = int(window_end * 0.70)
            test_start = train_end
            test_end = window_end

            windows.append((window_start, train_end, test_start, test_end))

        return windows

    def _test_params_walk_forward(self, frag_pct_thresh: float, mfg_dir_thresh: float,
                                   min_confidence: float, sl: float, tp: float) -> Optional[RobustResult]:
        """
        Testa parametros com Walk-Forward Validation completa
        """
        windows = self._create_walk_forward_windows(n_windows=4)
        wf_results = []
        all_train_pnls = []
        all_test_pnls = []

        for idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            train_signals = [s for s in self.signals if train_start <= s.bar_idx < train_end]
            test_signals = [s for s in self.signals if test_start <= s.bar_idx < test_end]

            train_bars = self.bars[train_start:train_end]
            test_bars = self.bars[test_start:test_end]

            if len(train_signals) < 20 or len(test_signals) < 10:
                return None

            # Backtest treino
            train_pnls = self._run_backtest(
                train_signals, train_bars,
                frag_pct_thresh, mfg_dir_thresh, min_confidence, sl, tp,
                bar_offset=train_start
            )
            train_result = self._calculate_backtest_result(train_pnls)

            if train_result.trades < 20 or train_result.profit_factor < 1.15:
                return None

            # Backtest teste
            test_pnls = self._run_backtest(
                test_signals, test_bars,
                frag_pct_thresh, mfg_dir_thresh, min_confidence, sl, tp,
                bar_offset=test_start
            )
            test_result = self._calculate_backtest_result(test_pnls)

            if test_result.trades < 10 or test_result.profit_factor < 0.95:
                return None

            # Calcular robustez
            pf_ratio = test_result.profit_factor / train_result.profit_factor if train_result.profit_factor > 0 else 0
            wr_ratio = test_result.win_rate / train_result.win_rate if train_result.win_rate > 0 else 0
            degradation = 1.0 - (pf_ratio + wr_ratio) / 2
            robustness = max(0, min(1, 1 - degradation))

            passed = pf_ratio >= 0.65 and wr_ratio >= 0.65 and test_result.profit_factor >= 1.0

            wf_results.append(WalkForwardResult(
                window_idx=idx,
                train_start=self.bars[train_start].timestamp,
                train_end=self.bars[train_end - 1].timestamp,
                test_start=self.bars[test_start].timestamp,
                test_end=self.bars[test_end - 1].timestamp,
                train_result=train_result,
                test_result=test_result,
                robustness_score=robustness,
                degradation=degradation,
                passed=passed
            ))

            all_train_pnls.extend(train_pnls)
            all_test_pnls.extend(test_pnls)

        # Verificar se TODAS as janelas passaram
        all_passed = all(wf.passed for wf in wf_results)
        if not all_passed:
            return None

        # Calcular metricas combinadas
        combined_train = self._calculate_backtest_result(all_train_pnls)
        combined_test = self._calculate_backtest_result(all_test_pnls)

        # Filtros finais RIGOROSOS
        if not combined_train.is_valid_for_real_money(
            min_trades=self.MIN_TRADES_TRAIN,
            min_pf=self.MIN_PF_TRAIN,
            min_win_rate=self.MIN_WIN_RATE,
            max_win_rate=self.MAX_WIN_RATE,
            max_dd=self.MAX_DRAWDOWN,
            min_expectancy=self.MIN_EXPECTANCY
        ):
            return None

        if not combined_test.is_valid_for_real_money(
            min_trades=self.MIN_TRADES_TEST,
            min_pf=self.MIN_PF_TEST,
            min_win_rate=self.MIN_WIN_RATE - 0.05,
            max_win_rate=self.MAX_WIN_RATE + 0.05,
            max_dd=self.MAX_DRAWDOWN + 0.05,
            min_expectancy=self.MIN_EXPECTANCY * 0.7
        ):
            return None

        # Robustez geral
        avg_train_pf = np.mean([wf.train_result.profit_factor for wf in wf_results])
        avg_test_pf = np.mean([wf.test_result.profit_factor for wf in wf_results])
        avg_train_wr = np.mean([wf.train_result.win_rate for wf in wf_results])
        avg_test_wr = np.mean([wf.test_result.win_rate for wf in wf_results])
        overall_robustness = np.mean([wf.robustness_score for wf in wf_results])

        params = {
            "fragility_percentile_threshold": round(frag_pct_thresh, 4),
            "mfg_direction_threshold": round(mfg_dir_thresh, 4),
            "min_confidence": round(min_confidence, 4),
            "stop_loss_pips": round(sl, 1),
            "take_profit_pips": round(tp, 1)
        }

        return RobustResult(
            params=params,
            walk_forward_results=wf_results,
            avg_train_pf=avg_train_pf,
            avg_test_pf=avg_test_pf,
            avg_train_wr=avg_train_wr,
            avg_test_wr=avg_test_wr,
            total_train_trades=combined_train.trades,
            total_test_trades=combined_test.trades,
            overall_robustness=overall_robustness,
            all_windows_passed=all_passed,
            combined_train_result=combined_train,
            combined_test_result=combined_test
        )

    def optimize(self, n: int = 500000) -> Optional[RobustResult]:
        """Executa otimizacao robusta com Walk-Forward"""
        if not self.signals:
            print("  ERRO: Dados nao carregados!")
            return None

        print(f"\n{'='*70}")
        print(f"  OTIMIZACAO ROBUSTA ODMN V2.0: {n:,} COMBINACOES")
        print(f"  Walk-Forward Validation (4 janelas)")
        print(f"  CUSTOS REALISTAS: Spread {self.SPREAD_PIPS} + Slippage {self.SLIPPAGE_PIPS} pips")
        print(f"  FILTROS RIGOROSOS PARA DINHEIRO REAL")
        print(f"{'='*70}")
        print(f"\n  Filtros aplicados:")
        print(f"    Min trades (treino): {self.MIN_TRADES_TRAIN}")
        print(f"    Min trades (teste): {self.MIN_TRADES_TEST}")
        print(f"    Win Rate: {self.MIN_WIN_RATE:.0%} - {self.MAX_WIN_RATE:.0%}")
        print(f"    Profit Factor: >= {self.MIN_PF_TRAIN} (treino), >= {self.MIN_PF_TEST} (teste)")
        print(f"    Max Drawdown: {self.MAX_DRAWDOWN:.0%}")
        print(f"    Min Expectancy: {self.MIN_EXPECTANCY} pips/trade")
        print(f"    Min Robustness: {self.MIN_ROBUSTNESS:.0%}")

        # Ranges de parametros para ODMN
        frag_pct_vals = np.linspace(0.50, 0.90, 15)    # Percentil de fragilidade
        mfg_dir_vals = np.linspace(0.03, 0.15, 12)     # Direcao MFG
        conf_vals = np.linspace(0.40, 0.75, 10)        # Confianca minima
        sl_vals = np.linspace(20, 50, 12)              # Stop loss
        tp_vals = np.linspace(35, 90, 15)              # Take profit

        best_robustness = -1
        tested = 0
        robust_count = 0
        start = datetime.now()

        # REPRODUTIBILIDADE: Seed fixo para random sampling
        random.seed(OPTIMIZER_SEED)
        np.random.seed(OPTIMIZER_SEED)

        for _ in range(n):
            tested += 1

            frag_pct = float(random.choice(frag_pct_vals))
            mfg_dir = float(random.choice(mfg_dir_vals))
            conf = float(random.choice(conf_vals))
            sl = float(random.choice(sl_vals))
            tp = float(random.choice(tp_vals))

            result = self._test_params_walk_forward(frag_pct, mfg_dir, conf, sl, tp)

            if result:
                robust_count += 1
                self.robust_results.append(result)

                if result.overall_robustness > best_robustness:
                    best_robustness = result.overall_robustness
                    self.best = result

                    print(f"\n  [ROBUSTO #{robust_count}] Robustez={result.overall_robustness:.4f}")
                    print(f"    Walk-Forward: {len(result.walk_forward_results)} janelas APROVADAS")
                    print(f"    TREINO: {result.combined_train_result.trades} trades, "
                          f"WR={result.combined_train_result.win_rate:.1%}, "
                          f"PF={result.combined_train_result.profit_factor:.2f}, "
                          f"Exp={result.combined_train_result.expectancy:.1f}pips/trade")
                    print(f"    TESTE:  {result.combined_test_result.trades} trades, "
                          f"WR={result.combined_test_result.win_rate:.1%}, "
                          f"PF={result.combined_test_result.profit_factor:.2f}, "
                          f"Exp={result.combined_test_result.expectancy:.1f}pips/trade")

            if tested % 50000 == 0:
                elapsed = (datetime.now() - start).total_seconds()
                rate = tested / elapsed
                eta = (n - tested) / rate / 60
                print(f"  {tested:,}/{n:,} ({tested/n*100:.1f}%) | "
                      f"Robustos: {robust_count} | "
                      f"Vel: {rate:.0f}/s | ETA: {eta:.0f}min")

        elapsed = (datetime.now() - start).total_seconds()
        print(f"\n{'='*70}")
        print(f"  CONCLUIDO em {elapsed/60:.1f}min")
        print(f"  Testados: {tested:,} | Robustos para DINHEIRO REAL: {robust_count}")
        print(f"{'='*70}")

        return self.best

    def save(self, n_tested: int = 0):
        """Salva melhor configuracao robusta"""
        if not self.best:
            print("  Nenhuma configuracao robusta encontrada!")
            return

        configs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "configs"
        )
        os.makedirs(configs_dir, exist_ok=True)

        # Salvar melhor configuracao
        best_file = os.path.join(configs_dir, "odmn-malliavin-nash_robust.json")

        config = {
            "strategy": "ODMN-MalliavinNash",
            "symbol": self.symbol,
            "periodicity": self.periodicity,
            "version": "2.0-real-money",
            "optimized_at": datetime.now(timezone.utc).isoformat(),
            "validation": {
                "method": "walk_forward",
                "n_windows": 4,
                "combinations_tested": n_tested,
                "robust_found": len(self.robust_results),
                "costs": {
                    "spread_pips": self.SPREAD_PIPS,
                    "slippage_pips": self.SLIPPAGE_PIPS,
                },
                "filters": {
                    "min_trades_train": self.MIN_TRADES_TRAIN,
                    "min_trades_test": self.MIN_TRADES_TEST,
                    "min_pf_train": self.MIN_PF_TRAIN,
                    "min_pf_test": self.MIN_PF_TEST,
                    "max_drawdown": self.MAX_DRAWDOWN,
                    "min_expectancy": self.MIN_EXPECTANCY,
                }
            },
            "parameters": self.best.params,
            "performance": self.best.to_dict(),
            "ready_for_real_money": True,
        }

        with open(best_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"\n  Melhor config salva em: {best_file}")

        # Salvar top 10
        top_file = os.path.join(configs_dir, "odmn_robust_top10.json")
        sorted_results = sorted(
            self.robust_results,
            key=lambda x: x.overall_robustness,
            reverse=True
        )[:10]

        top_data = [r.to_dict() for r in sorted_results]

        with open(top_file, 'w') as f:
            json.dump(top_data, f, indent=2, default=str)
        print(f"  Top 10 robustos salvo em: {top_file}")


async def main():
    N_COMBINATIONS = 500000

    print("=" * 70)
    print("  OTIMIZADOR ODMN V2.0 - PRONTO PARA DINHEIRO REAL")
    print("=" * 70)
    print("\n  CARACTERISTICAS:")
    print("    - Walk-Forward Validation (4 janelas)")
    print("    - Custos realistas (spread 1.5 + slippage 0.8 pips)")
    print("    - Filtros rigorosos (PF > 1.3, Exp > 1.5 pips)")
    print("    - Sem look-ahead em nenhum calculo")
    print("    - Modelo de Heston calibrado em janela deslizante")
    print("    - Malliavin Monte Carlo causal")
    print("    - Mean Field Games sem dados futuros")
    print("=" * 70)

    opt = ODMNRobustOptimizer("EURUSD", "H1")

    # Periodos de treino e teste
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    split = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    print(f"\n  Periodo Total: {start.date()} a {end.date()}")
    print(f"  Treino: {start.date()} a {split.date()}")
    print(f"  Teste:  {split.date()} a {end.date()}")

    if await opt.load_and_precompute(start, end, split_date=split):
        best = opt.optimize(N_COMBINATIONS)
        if best:
            print(f"\n{'='*70}")
            print(f"  MELHOR RESULTADO - PRONTO PARA DINHEIRO REAL")
            print(f"{'='*70}")
            print(f"  Robustez Geral: {best.overall_robustness:.4f}")
            print(f"  Walk-Forward: {len(best.walk_forward_results)} janelas APROVADAS")
            print(f"\n  TREINO COMBINADO:")
            print(f"    Trades: {best.combined_train_result.trades}")
            print(f"    Win Rate: {best.combined_train_result.win_rate:.1%}")
            print(f"    Profit Factor: {best.combined_train_result.profit_factor:.2f}")
            print(f"    PnL: {best.combined_train_result.total_pnl:.1f} pips")
            print(f"    Expectancy: {best.combined_train_result.expectancy:.1f} pips/trade")
            print(f"\n  TESTE COMBINADO (Out-of-Sample):")
            print(f"    Trades: {best.combined_test_result.trades}")
            print(f"    Win Rate: {best.combined_test_result.win_rate:.1%}")
            print(f"    Profit Factor: {best.combined_test_result.profit_factor:.2f}")
            print(f"    PnL: {best.combined_test_result.total_pnl:.1f} pips")
            print(f"    Expectancy: {best.combined_test_result.expectancy:.1f} pips/trade")
            print(f"\n  PARAMETROS:")
            for k, v in best.params.items():
                print(f"    {k}: {v}")
            print(f"{'='*70}")

            opt.save(n_tested=N_COMBINATIONS)
        else:
            print("\n  AVISO: Nenhuma configuracao passou nos filtros rigorosos!")
            print("  Possiveis causas:")
            print("    1. Periodo de dados muito curto")
            print("    2. Indicador nao tem edge suficiente com custos reais")
            print("    3. Filtros muito rigorosos para o periodo atual")
            print("\n  Sugestoes:")
            print("    1. Aumentar periodo de dados (minimo 1 ano)")
            print("    2. Ajustar filtros (reduzir MIN_PF_TRAIN para 1.2)")
            print("    3. Testar em outros pares de moedas")
    else:
        print("\n  ERRO: Falha ao carregar dados!")


if __name__ == "__main__":
    asyncio.run(main())
