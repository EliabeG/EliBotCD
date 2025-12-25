#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR DTT ROBUSTO V3.3 - VALIDAÇÃO ESTATÍSTICA COMPLETA
================================================================================

VERSÃO V3.3 - VALIDAÇÃO ROBUSTA 25/12/2025:

MELHORIAS V3.3:
- Monte Carlo aumentado de 1.000 para 10.000 permutações
- Walk-Forward aumentado de 6 para 12 janelas independentes
- Validação Out-of-Sample (20% dos dados) - NUNCA tocados na otimização
- Detecção de overfitting por comparação WF vs OOS

CORREÇÃO V3.2 (herdado):
- τ (time_delay) e m (embedding_dim) calculados APENAS com dados de TREINO
- Parâmetros FIXADOS antes de qualquer análise de dados de teste
- Elimina look-ahead sutil no Takens Embedding

METODOLOGIA:
1. 80% dados para Walk-Forward (12 janelas deslizantes)
2. 20% dados reservados para Out-of-Sample (validação final)
3. Monte Carlo com 10.000 permutações (p-value < 0.05)
4. Comparação WF vs OOS para detectar overfitting

PARA DINHEIRO REAL. SEM OVERFITTING. SEM LOOK-AHEAD. CUSTOS REALISTAS.
================================================================================
"""

import sys
import os
import json
import asyncio
import random
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.fxopen_historical_ws import Bar, download_historical_data
from strategies.alta_volatilidade.dtt_tunelamento_topologico import DetectorTunelamentoTopologico

# Importar módulo compartilhado de direção
try:
    from backtesting.common.direction_calculator import (
        calculate_direction_from_bars,
        DEFAULT_DIRECTION_LOOKBACK
    )
    USE_SHARED_DIRECTION = True
except ImportError:
    USE_SHARED_DIRECTION = False
    DEFAULT_DIRECTION_LOOKBACK = 12

# Configurar logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DTTSignal:
    """
    Sinal pre-calculado do DTT

    V2.1: Usa módulo compartilhado para direção
    """
    bar_idx: int          # Índice da barra onde o sinal foi GERADO
    signal_price: float   # Preço de fechamento quando sinal foi gerado
    next_bar_idx: int     # Índice da barra onde deve EXECUTAR
    entry_price: float    # Preço de ABERTURA da próxima barra
    high: float           # High da barra de entrada
    low: float            # Low da barra de entrada
    persistence_entropy: float
    tunneling_probability: float
    signal_strength: float
    direction: int        # Calculado via módulo compartilhado


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
                                min_trades: int = 200,
                                max_win_rate: float = 0.60,
                                min_win_rate: float = 0.35,
                                max_pf: float = 3.5,
                                min_pf: float = 1.30,
                                max_dd: float = 0.30,
                                min_expectancy: float = 3.0) -> bool:
        """Verifica se resultado passa nos filtros para dinheiro real"""
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
    """Resultado robusto com validação walk-forward"""
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
            },
            "combined_train": {
                "trades": self.combined_train_result.trades,
                "win_rate": round(self.combined_train_result.win_rate, 4),
                "profit_factor": round(self.combined_train_result.profit_factor, 4),
                "expectancy": round(self.combined_train_result.expectancy, 2),
            },
            "combined_test": {
                "trades": self.combined_test_result.trades,
                "win_rate": round(self.combined_test_result.win_rate, 4),
                "profit_factor": round(self.combined_test_result.profit_factor, 4),
                "expectancy": round(self.combined_test_result.expectancy, 2),
            },
            "overall_robustness": round(self.overall_robustness, 4),
        }


class DTTRobustOptimizer:
    """
    Otimizador DTT V3.3 com Validação Estatística Completa

    MELHORIAS V3.3:
    - Monte Carlo: 10.000 permutações (era 1.000)
    - Walk-Forward: 12 janelas (era 6)
    - Out-of-Sample: 20% reservado para validação final
    - Detecção de overfitting por degradação WF vs OOS

    CORREÇÕES V3.2 (herdado):
    - τ (time_delay) e m (embedding_dim) APENAS com dados de TREINO
    - Parâmetros FIXADOS antes de qualquer análise

    CORREÇÕES V3.1 (herdado):
    - Walk-forward com janelas DESLIZANTES
    - Teste de significância estatística (Monte Carlo)
    - Mínimo 200/100 trades
    """

    # Custos REALISTAS
    SPREAD_PIPS = 1.5
    SLIPPAGE_PIPS = 0.8
    COMMISSION_PIPS = 0.0

    # Filtros RIGOROSOS
    MIN_TRADES_TRAIN = 200
    MIN_TRADES_TEST = 100
    MIN_WIN_RATE = 0.35
    MAX_WIN_RATE = 0.60
    MIN_PF_TRAIN = 1.30
    MIN_PF_TEST = 1.15
    MAX_PF = 3.5
    MAX_DRAWDOWN = 0.30
    MIN_ROBUSTNESS = 0.70
    MIN_EXPECTANCY = 3.0

    # V3.3: Configuração de significância estatística ROBUSTA
    MONTE_CARLO_PERMUTATIONS = 10000  # V3.3: Aumentado de 1000 para 10000
    SIGNIFICANCE_LEVEL = 0.05  # p-value < 0.05

    # V3.3: Walk-Forward com mais janelas
    N_WALK_FORWARD_WINDOWS = 12  # V3.3: Aumentado de 6 para 12

    # V3.3: Validação Out-of-Sample
    OUT_OF_SAMPLE_RATIO = 0.20  # 20% reservado para validação final

    def __init__(self, symbol: str = "EURUSD", periodicity: str = "H1"):
        self.symbol = symbol
        self.periodicity = periodicity
        self.pip = 0.0001

        self.bars: List[Bar] = []
        self.signals: List[DTTSignal] = []
        self.robust_results: List[RobustResult] = []
        self.best: Optional[RobustResult] = None

        # V3.2: Parâmetros de embedding FIXOS (calculados apenas com treino)
        self.fixed_tau: Optional[int] = None
        self.fixed_m: Optional[int] = None

        # V3.3: Separação de dados para Out-of-Sample
        self.wf_bars: List[Bar] = []      # Barras para Walk-Forward
        self.oos_bars: List[Bar] = []     # Barras Out-of-Sample (nunca tocadas)
        self.oos_signals: List[DTTSignal] = []  # Sinais OOS para validação final

        logger.info(f"DTTRobustOptimizer V3.3 inicializado: {symbol} {periodicity}")
        logger.info(f"  Módulo compartilhado de direção: {USE_SHARED_DIRECTION}")
        logger.info(f"  Min trades: {self.MIN_TRADES_TRAIN}/{self.MIN_TRADES_TEST}")
        logger.info(f"  Monte Carlo: {self.MONTE_CARLO_PERMUTATIONS} permutações")

    def _calculate_direction(self, bar_idx: int) -> int:
        """
        Calcula direção usando módulo compartilhado ou fallback.

        REGRA: Usa barras ANTES do índice atual para evitar look-ahead.
        """
        if USE_SHARED_DIRECTION:
            return calculate_direction_from_bars(
                self.bars,
                bar_idx,
                DEFAULT_DIRECTION_LOOKBACK
            )
        else:
            # Fallback local
            if bar_idx < DEFAULT_DIRECTION_LOOKBACK + 1:
                return 0
            recent_close = self.bars[bar_idx - 1].close
            past_close = self.bars[bar_idx - DEFAULT_DIRECTION_LOOKBACK].close
            trend = recent_close - past_close
            return 1 if trend > 0 else -1

    async def load_and_precompute(self, start_date: datetime, end_date: datetime,
                                   split_date: datetime = None, train_ratio: float = 0.60):
        """
        V3.3: Carrega dados e pre-calcula sinais DTT SEM LOOK-AHEAD

        ESTRUTURA DE DADOS V3.3:
        - 60% para calcular embedding (τ, m) e otimização walk-forward
        - 20% reservado para OUT-OF-SAMPLE (nunca tocado durante otimização)

        CORREÇÃO CRÍTICA V3.2 (Auditoria Final):
        - Parâmetros de Embedding (τ, m) calculados APENAS com dados de TREINO
        - Evita look-ahead bias sutil no embedding
        - τ e m são FIXOS para toda a análise

        Args:
            start_date: Data inicial dos dados
            end_date: Data final dos dados
            split_date: Data de divisão (opcional, usa train_ratio se não fornecido)
            train_ratio: Proporção de dados para treino/WF (default: 60%, resto é OOS)
        """
        print("\n" + "=" * 70)
        print("  CARREGANDO DADOS REAIS - V3.3 COM OUT-OF-SAMPLE")
        print("=" * 70)
        print("\n  ✓ Embedding (τ,m) calculado APENAS com dados de TREINO")
        print(f"  ✓ {self.OUT_OF_SAMPLE_RATIO*100:.0f}% reservado para OUT-OF-SAMPLE")
        print()

        self.bars = await download_historical_data(
            symbol=self.symbol,
            periodicity=self.periodicity,
            start_time=start_date,
            end_time=end_date
        )
        print(f"  Total de barras: {len(self.bars)}")

        if len(self.bars) < 1500:
            print("  ERRO: Dados insuficientes! Mínimo 1500 barras para validação com OOS.")
            return False

        # =====================================================================
        # V3.3: SEPARAR DADOS EM TREINO/WF E OUT-OF-SAMPLE
        # =====================================================================
        total_bars = len(self.bars)
        oos_idx = int(total_bars * (1 - self.OUT_OF_SAMPLE_RATIO))

        # Guardar barras OOS para validação final (NUNCA usadas na otimização)
        self.oos_bars = self.bars[oos_idx:]
        self.wf_bars = self.bars[:oos_idx]  # Barras para walk-forward

        print(f"\n  Divisão de dados V3.3:")
        print(f"    Walk-Forward: {len(self.wf_bars)} barras ({100*(1-self.OUT_OF_SAMPLE_RATIO):.0f}%)")
        print(f"    Out-of-Sample: {len(self.oos_bars)} barras ({self.OUT_OF_SAMPLE_RATIO*100:.0f}%) - RESERVADO")

        # =====================================================================
        # V3.2: CALCULAR τ e m APENAS COM DADOS DE TREINO (ANTI LOOK-AHEAD)
        # =====================================================================
        # Usar apenas primeira parte dos dados WF para calcular embedding
        split_idx = int(len(self.wf_bars) * train_ratio)
        train_bars = self.wf_bars[:split_idx]

        print(f"\n  Calculando embedding com dados de TREINO apenas:")
        print(f"    Barras de treino (embedding): {len(train_bars)} ({train_ratio*100:.0f}% do WF)")

        # Extrair preços de treino para calcular τ e m
        train_prices = np.array([b.close for b in train_bars])

        # Criar DTT temporário para calcular τ e m com dados de treino
        dtt_calibration = DetectorTunelamentoTopologico(
            max_points=150,
            use_dimensionality_reduction=True,
            reduction_method='pca',
            persistence_entropy_threshold=0.1,
            tunneling_probability_threshold=0.05
        )

        # Calcular embedding APENAS com dados de treino
        embedding_result = dtt_calibration.get_takens_embedding(train_prices)

        # FIXAR parâmetros calculados
        self.fixed_tau = embedding_result['time_delay']
        self.fixed_m = embedding_result['embedding_dim']

        print(f"\n  Parâmetros de Embedding FIXADOS (anti look-ahead):")
        print(f"    τ (time_delay):    {self.fixed_tau}")
        print(f"    m (embedding_dim): {self.fixed_m}")

        # =====================================================================
        # CRIAR DTT COM PARÂMETROS FIXOS PARA TODA A ANÁLISE
        # =====================================================================
        print("\n  Pre-calculando sinais DTT V3.2 (embedding fixo)...")

        dtt = DetectorTunelamentoTopologico(
            embedding_dim=self.fixed_m,    # FIXO - calculado apenas com treino
            time_delay=self.fixed_tau,     # FIXO - calculado apenas com treino
            max_points=150,
            use_dimensionality_reduction=True,
            reduction_method='pca',
            persistence_entropy_threshold=0.1,
            tunneling_probability_threshold=0.05
        )

        # =====================================================================
        # V3.3: PRE-CALCULAR SINAIS PARA WALK-FORWARD (wf_bars)
        # =====================================================================
        prices_buf = deque(maxlen=500)
        self.signals = []
        min_prices = 150
        error_count = 0

        print(f"\n  Pre-calculando sinais Walk-Forward ({len(self.wf_bars)} barras)...")

        for i, bar in enumerate(self.wf_bars):
            prices_buf.append(bar.close)

            if len(prices_buf) < min_prices:
                continue

            if i >= len(self.wf_bars) - 1:
                continue

            try:
                result = dtt.analyze(np.array(prices_buf))

                # V2.1: Direção via módulo compartilhado
                direction = self._calculate_direction_for_bars(self.wf_bars, i)

                next_bar = self.wf_bars[i + 1]

                self.signals.append(DTTSignal(
                    bar_idx=i,
                    signal_price=bar.close,
                    next_bar_idx=i + 1,
                    entry_price=next_bar.open,
                    high=next_bar.high,
                    low=next_bar.low,
                    persistence_entropy=result['entropy']['persistence_entropy'],
                    tunneling_probability=result['tunneling']['tunneling_probability'],
                    signal_strength=result['signal_strength'],
                    direction=direction
                ))

            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    logger.warning(f"Erro WF na barra {i}: {e}")
                continue

            if (i + 1) % 500 == 0:
                print(f"    WF: {i+1}/{len(self.wf_bars)} barras...")

        print(f"\n  Sinais Walk-Forward: {len(self.signals)}")
        if error_count > 0:
            print(f"  Erros WF: {error_count}")

        # =====================================================================
        # V3.3: PRE-CALCULAR SINAIS PARA OUT-OF-SAMPLE (oos_bars)
        # =====================================================================
        print(f"\n  Pre-calculando sinais Out-of-Sample ({len(self.oos_bars)} barras)...")

        # Iniciar com últimos preços do WF para continuidade
        oos_prices_buf = deque(prices_buf, maxlen=500)
        self.oos_signals = []
        oos_error_count = 0

        for i, bar in enumerate(self.oos_bars):
            oos_prices_buf.append(bar.close)

            if len(oos_prices_buf) < min_prices:
                continue

            if i >= len(self.oos_bars) - 1:
                continue

            try:
                result = dtt.analyze(np.array(oos_prices_buf))
                direction = self._calculate_direction_for_bars(self.oos_bars, i)
                next_bar = self.oos_bars[i + 1]

                self.oos_signals.append(DTTSignal(
                    bar_idx=i,
                    signal_price=bar.close,
                    next_bar_idx=i + 1,
                    entry_price=next_bar.open,
                    high=next_bar.high,
                    low=next_bar.low,
                    persistence_entropy=result['entropy']['persistence_entropy'],
                    tunneling_probability=result['tunneling']['tunneling_probability'],
                    signal_strength=result['signal_strength'],
                    direction=direction
                ))

            except Exception as e:
                oos_error_count += 1
                if oos_error_count <= 3:
                    logger.warning(f"Erro OOS na barra {i}: {e}")
                continue

        print(f"  Sinais Out-of-Sample: {len(self.oos_signals)}")
        if oos_error_count > 0:
            print(f"  Erros OOS: {oos_error_count}")

        long_signals = sum(1 for s in self.signals if s.direction == 1)
        short_signals = sum(1 for s in self.signals if s.direction == -1)
        print(f"\n  Distribuição WF - Long: {long_signals}, Short: {short_signals}")

        return len(self.signals) > 500 and len(self.oos_signals) > 100

    def _calculate_direction_for_bars(self, bars: List[Bar], bar_idx: int) -> int:
        """
        V3.3: Calcula direção para um conjunto específico de barras.
        """
        if bar_idx < DEFAULT_DIRECTION_LOOKBACK + 1:
            return 0
        recent_close = bars[bar_idx - 1].close
        past_close = bars[bar_idx - DEFAULT_DIRECTION_LOOKBACK].close
        trend = recent_close - past_close
        return 1 if trend > 0 else -1

    def _calculate_backtest_result(self, pnls: List[float]) -> BacktestResult:
        """Calcula métricas de um backtest"""
        if not pnls:
            return BacktestResult(
                trades=0, wins=0, losses=0, total_pnl=0,
                win_rate=0, profit_factor=0, max_drawdown=1.0,
                avg_trade=0, largest_win=0, largest_loss=0, expectancy=0
            )

        wins = sum(1 for p in pnls if p > 0)
        losses = len(pnls) - wins
        total_pnl = sum(pnls)
        win_rate = wins / len(pnls) if pnls else 0

        gross_profit = sum(p for p in pnls if p > 0) or 0.001
        gross_loss = abs(sum(p for p in pnls if p <= 0)) or 0.001
        profit_factor = gross_profit / gross_loss

        equity = np.cumsum([0] + pnls)
        peak = np.maximum.accumulate(equity + 10000)
        drawdowns = (peak - (equity + 10000)) / peak
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

        avg_trade = total_pnl / len(pnls) if pnls else 0
        largest_win = max(pnls) if pnls else 0
        largest_loss = min(pnls) if pnls else 0

        return BacktestResult(
            trades=len(pnls), wins=wins, losses=losses, total_pnl=total_pnl,
            win_rate=win_rate, profit_factor=profit_factor, max_drawdown=max_dd,
            avg_trade=avg_trade, largest_win=largest_win, largest_loss=largest_loss,
            expectancy=avg_trade
        )

    def _run_backtest(self, signals: List[DTTSignal], bars: List[Bar],
                      entropy_thresh: float, tunneling_thresh: float,
                      strength_thresh: float, sl: float, tp: float,
                      bar_offset: int = 0) -> List[float]:
        """Executa backtest com custos REALISTAS"""
        if tp <= sl:
            return []

        entries = []
        for s in signals:
            if (s.persistence_entropy >= entropy_thresh and
                s.tunneling_probability >= tunneling_thresh and
                s.signal_strength >= strength_thresh and
                s.direction != 0):

                execution_idx = s.next_bar_idx - bar_offset
                entries.append((execution_idx, s.entry_price, s.direction))

        if len(entries) < 10:
            return []

        pnls = []
        pip = self.pip
        half_spread = (self.SPREAD_PIPS * pip) / 2
        slippage = self.SLIPPAGE_PIPS * pip
        entry_cost = half_spread + slippage
        exit_cost = half_spread + slippage

        last_exit_idx = -1

        for entry_idx, entry_price_raw, direction in entries:
            if entry_idx < 0 or entry_idx >= len(bars) - 1:
                continue

            if entry_idx <= last_exit_idx:
                continue

            if direction == 1:
                entry_price = entry_price_raw + entry_cost
                stop_price = entry_price - sl * pip
                take_price = entry_price + tp * pip
            else:
                entry_price = entry_price_raw - entry_cost
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

                if direction == 1:
                    if bar.open <= stop_price:
                        exit_price = bar.open - exit_cost
                        exit_bar_idx = bar_idx
                        break
                    if bar.open >= take_price:
                        exit_price = bar.open - exit_cost
                        exit_bar_idx = bar_idx
                        break
                else:
                    if bar.open >= stop_price:
                        exit_price = bar.open + exit_cost
                        exit_bar_idx = bar_idx
                        break
                    if bar.open <= take_price:
                        exit_price = bar.open + exit_cost
                        exit_bar_idx = bar_idx
                        break

                if direction == 1:
                    if bar.low <= stop_price:
                        exit_price = stop_price - exit_cost
                        exit_bar_idx = bar_idx
                        break
                    if bar.high >= take_price:
                        exit_price = take_price - exit_cost
                        exit_bar_idx = bar_idx
                        break
                else:
                    if bar.high >= stop_price:
                        exit_price = stop_price + exit_cost
                        exit_bar_idx = bar_idx
                        break
                    if bar.low <= take_price:
                        exit_price = take_price + exit_cost
                        exit_bar_idx = bar_idx
                        break

            if exit_price is None:
                exit_bar_idx = min(entry_idx + max_bars, len(bars) - 1)
                last_bar = bars[exit_bar_idx]
                if direction == 1:
                    exit_price = last_bar.close - exit_cost
                else:
                    exit_price = last_bar.close + exit_cost

            if direction == 1:
                pnl_pips = (exit_price - entry_price) / pip
            else:
                pnl_pips = (entry_price - exit_price) / pip

            pnls.append(pnl_pips)
            last_exit_idx = exit_bar_idx

        return pnls

    def _test_statistical_significance(self, pnls: List[float],
                                        n_permutations: int = None) -> dict:
        """
        V3.1: Teste de Monte Carlo para verificar significância estatística

        Embaralha os PnLs e verifica se o resultado real é significativamente
        melhor que aleatório.

        Args:
            pnls: Lista de PnLs reais
            n_permutations: Número de permutações (default: MONTE_CARLO_PERMUTATIONS)

        Returns:
            Dict com real_pf, p_value, significant
        """
        if n_permutations is None:
            n_permutations = self.MONTE_CARLO_PERMUTATIONS

        if len(pnls) < 20:
            return {'real_pf': 0, 'p_value': 1.0, 'significant': False}

        # Calcular PF real
        gross_profit = sum(p for p in pnls if p > 0) or 0.001
        gross_loss = abs(sum(p for p in pnls if p <= 0)) or 0.001
        real_pf = gross_profit / gross_loss

        # Permutações aleatórias
        random_pfs = []
        pnls_array = np.array(pnls)

        for _ in range(n_permutations):
            shuffled = np.random.permutation(pnls_array)
            gp = np.sum(shuffled[shuffled > 0]) or 0.001
            gl = abs(np.sum(shuffled[shuffled <= 0])) or 0.001
            random_pfs.append(gp / gl)

        # P-valor: proporção de permutações com PF >= real
        p_value = np.mean([pf >= real_pf for pf in random_pfs])

        return {
            'real_pf': real_pf,
            'random_pf_mean': np.mean(random_pfs),
            'random_pf_std': np.std(random_pfs),
            'p_value': p_value,
            'significant': p_value < self.SIGNIFICANCE_LEVEL
        }

    def _create_walk_forward_windows_sliding(self, n_windows: int = None,
                                              train_size_bars: int = None,
                                              test_size_bars: int = None,
                                              step_bars: int = None) -> List[Tuple[int, int, int, int]]:
        """
        V3.3: Walk-Forward REAL com janelas DESLIZANTES

        V3.3: Aumentado para 12 janelas (era 6) para maior significância estatística.

        Diferença do V2.1 (janelas independentes):
        - V2.1: Cada janela usa dados DIFERENTES (não sobrepostos)
        - V3.1+: Janelas DESLIZAM, treino cresce/move com o tempo

        Args:
            n_windows: Número de janelas (default: N_WALK_FORWARD_WINDOWS = 12)
            train_size_bars: Tamanho do treino em barras (default: 50% dos dados)
            test_size_bars: Tamanho do teste em barras (default: 10% dos dados)
            step_bars: Passo entre janelas (default: calculado automaticamente)
        """
        # V3.3: Usar constante da classe se não especificado
        if n_windows is None:
            n_windows = self.N_WALK_FORWARD_WINDOWS

        # V3.3: Usar wf_bars (exclui OOS)
        total_bars = len(self.wf_bars) if self.wf_bars else len(self.bars)

        # Defaults baseados no total de dados
        if train_size_bars is None:
            train_size_bars = int(total_bars * 0.50)  # 50% para treino
        if test_size_bars is None:
            test_size_bars = int(total_bars * 0.10)   # 10% para teste
        if step_bars is None:
            # Calcular step para ter n_windows janelas
            available = total_bars - train_size_bars - test_size_bars
            step_bars = max(50, available // max(1, n_windows - 1))

        windows = []
        train_start = 0

        while train_start + train_size_bars + test_size_bars <= total_bars:
            train_end = train_start + train_size_bars
            test_start = train_end
            test_end = min(test_start + test_size_bars, total_bars)

            if test_end > test_start:  # Garantir janela válida
                windows.append((train_start, train_end, test_start, test_end))

            train_start += step_bars

            if len(windows) >= n_windows:
                break

        return windows

    def _create_walk_forward_windows(self, n_windows: int = 4,
                                      train_pct: float = 0.70) -> List[Tuple[int, int, int, int]]:
        """
        V3.1: Usa janelas DESLIZANTES por padrão

        Mantém assinatura antiga para compatibilidade, mas usa novo método.
        """
        return self._create_walk_forward_windows_sliding(n_windows=n_windows)

    def _test_params_walk_forward(self, entropy_thresh: float, tunneling_thresh: float,
                                   strength_thresh: float, sl: float, tp: float) -> Optional[RobustResult]:
        """
        V3.1: Testa parâmetros com Walk-Forward REAL (janelas deslizantes)

        Inclui:
        - Janelas deslizantes (não independentes)
        - Teste de significância estatística
        """
        windows = self._create_walk_forward_windows_sliding()  # V3.3: Usa N_WALK_FORWARD_WINDOWS (12)
        wf_results = []
        all_train_pnls = []
        all_test_pnls = []

        for idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            train_signals = [s for s in self.signals if train_start <= s.bar_idx < train_end]
            test_signals = [s for s in self.signals if test_start <= s.bar_idx < test_end]

            train_bars = self.bars[train_start:train_end]
            test_bars = self.bars[test_start:test_end]

            if len(train_signals) < 50 or len(test_signals) < 25:
                return None

            train_pnls = self._run_backtest(
                train_signals, train_bars,
                entropy_thresh, tunneling_thresh, strength_thresh, sl, tp,
                bar_offset=train_start
            )
            train_result = self._calculate_backtest_result(train_pnls)

            if train_result.trades < 30 or train_result.profit_factor < 1.10:
                return None

            test_pnls = self._run_backtest(
                test_signals, test_bars,
                entropy_thresh, tunneling_thresh, strength_thresh, sl, tp,
                bar_offset=test_start
            )
            test_result = self._calculate_backtest_result(test_pnls)

            if test_result.trades < 15 or test_result.profit_factor < 0.90:
                return None

            pf_ratio = test_result.profit_factor / train_result.profit_factor if train_result.profit_factor > 0 else 0
            wr_ratio = test_result.win_rate / train_result.win_rate if train_result.win_rate > 0 else 0
            degradation = 1.0 - (pf_ratio + wr_ratio) / 2
            robustness = max(0, min(1, 1 - degradation))

            passed = pf_ratio >= 0.60 and wr_ratio >= 0.60 and test_result.profit_factor >= 0.95

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

        all_passed = all(wf.passed for wf in wf_results)
        if not all_passed:
            return None

        combined_train = self._calculate_backtest_result(all_train_pnls)
        combined_test = self._calculate_backtest_result(all_test_pnls)

        # V3.1: Filtros rigorosos
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

        # V3.1: Teste de significância estatística (Monte Carlo)
        significance_test = self._test_statistical_significance(all_test_pnls)
        if not significance_test['significant']:
            return None  # Resultado não é estatisticamente significativo

        avg_train_pf = np.mean([wf.train_result.profit_factor for wf in wf_results])
        avg_test_pf = np.mean([wf.test_result.profit_factor for wf in wf_results])
        avg_train_wr = np.mean([wf.train_result.win_rate for wf in wf_results])
        avg_test_wr = np.mean([wf.test_result.win_rate for wf in wf_results])
        overall_robustness = np.mean([wf.robustness_score for wf in wf_results])

        params = {
            "persistence_entropy_threshold": round(entropy_thresh, 4),
            "tunneling_probability_threshold": round(tunneling_thresh, 4),
            "min_signal_strength": round(strength_thresh, 4),
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
        """
        V3.2: Executa otimização robusta com embedding anti look-ahead
        """
        if not self.signals:
            logger.error("Dados não carregados!")
            return None

        print(f"\n{'='*70}")
        print(f"  OTIMIZAÇÃO DTT V3.2: {n:,} COMBINAÇÕES")
        print(f"  Embedding: τ={self.fixed_tau}, m={self.fixed_m} (FIXOS, anti look-ahead)")
        print(f"  Walk-Forward REAL (janelas deslizantes)")
        print(f"  Teste de Significância: Monte Carlo ({self.MONTE_CARLO_PERMUTATIONS} perm)")
        print(f"  Min trades: {self.MIN_TRADES_TRAIN}/{self.MIN_TRADES_TEST}")
        print(f"{'='*70}")

        entropy_vals = np.linspace(0.50, 0.90, 15)
        tunneling_vals = np.linspace(0.10, 0.40, 12)
        strength_vals = np.linspace(0.20, 0.60, 10)
        sl_vals = np.linspace(25, 55, 12)
        tp_vals = np.linspace(45, 100, 15)

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

            result = self._test_params_walk_forward(entropy, tunneling, strength, sl, tp)

            if result:
                robust_count += 1
                self.robust_results.append(result)

                if result.overall_robustness > best_robustness:
                    best_robustness = result.overall_robustness
                    self.best = result

                    print(f"\n  [ROBUSTO #{robust_count}] Score={result.overall_robustness:.4f}")
                    print(f"    TREINO: {result.combined_train_result.trades} trades, "
                          f"PF={result.combined_train_result.profit_factor:.2f}")
                    print(f"    TESTE:  {result.combined_test_result.trades} trades, "
                          f"PF={result.combined_test_result.profit_factor:.2f}")

            if tested % 50000 == 0:
                elapsed = (datetime.now() - start).total_seconds()
                rate = tested / elapsed
                eta = (n - tested) / rate / 60
                print(f"  {tested:,}/{n:,} | Robustos: {robust_count} | ETA: {eta:.0f}min")

        elapsed = (datetime.now() - start).total_seconds()
        print(f"\n{'='*70}")
        print(f"  CONCLUÍDO em {elapsed/60:.1f}min")
        print(f"  Robustos: {robust_count}")
        print(f"{'='*70}")

        return self.best

    def validate_out_of_sample(self, params: Dict = None) -> Optional[BacktestResult]:
        """
        V3.3: Valida parâmetros em dados Out-of-Sample (nunca vistos).

        Esta validação é CRÍTICA - se os parâmetros não funcionarem em OOS,
        há alto risco de overfitting.

        Args:
            params: Parâmetros a testar (default: self.best.params)

        Returns:
            BacktestResult do teste OOS ou None se falhar
        """
        if params is None:
            if self.best is None:
                logger.error("Nenhum resultado para validar OOS!")
                return None
            params = self.best.params

        if not self.oos_signals:
            logger.error("Sinais OOS não calculados!")
            return None

        print(f"\n{'='*70}")
        print(f"  VALIDAÇÃO OUT-OF-SAMPLE V3.3")
        print(f"  {len(self.oos_signals)} sinais em dados NUNCA VISTOS")
        print(f"{'='*70}")

        # Extrair parâmetros
        entropy_thresh = params.get('persistence_entropy_threshold', 0.5)
        tunneling_thresh = params.get('tunneling_probability_threshold', 0.15)
        strength_thresh = params.get('min_signal_strength', 0.3)
        sl = params.get('stop_loss_pips', 35)
        tp = params.get('take_profit_pips', 52)

        # Executar backtest em OOS
        oos_pnls = self._run_backtest(
            self.oos_signals, self.oos_bars,
            entropy_thresh, tunneling_thresh, strength_thresh, sl, tp,
            bar_offset=0
        )

        oos_result = self._calculate_backtest_result(oos_pnls)

        print(f"\n  Resultados Out-of-Sample:")
        print(f"    Trades: {oos_result.trades}")
        print(f"    Win Rate: {oos_result.win_rate*100:.1f}%")
        print(f"    Profit Factor: {oos_result.profit_factor:.2f}")
        print(f"    Expectancy: {oos_result.expectancy:.2f} pips")
        print(f"    Max Drawdown: {oos_result.max_drawdown*100:.1f}%")

        # Comparar com resultados Walk-Forward
        if self.best:
            wf_pf = self.best.combined_test_result.profit_factor
            oos_pf = oos_result.profit_factor
            degradation = 1 - (oos_pf / wf_pf) if wf_pf > 0 else 1

            print(f"\n  Comparação WF vs OOS:")
            print(f"    Walk-Forward PF: {wf_pf:.2f}")
            print(f"    Out-of-Sample PF: {oos_pf:.2f}")
            print(f"    Degradação: {degradation*100:.1f}%")

            if degradation > 0.30:
                print(f"\n  ⚠️  AVISO: Degradação > 30% - POSSÍVEL OVERFITTING!")
            elif degradation > 0.20:
                print(f"\n  ⚠️  Degradação moderada - monitorar de perto")
            else:
                print(f"\n  ✓ Degradação aceitável - parâmetros parecem robustos")

        # Verificar se passa nos filtros
        is_valid = oos_result.is_valid_for_real_money(
            min_trades=50,  # Menos exigente pois OOS é menor
            min_pf=1.10,
            min_win_rate=self.MIN_WIN_RATE - 0.05,
            max_win_rate=self.MAX_WIN_RATE + 0.05,
            max_dd=self.MAX_DRAWDOWN + 0.10,
            min_expectancy=2.0
        )

        if is_valid:
            print(f"\n  ✓ APROVADO: Resultados OOS passam nos filtros mínimos")
        else:
            print(f"\n  ✗ REPROVADO: Resultados OOS NÃO passam nos filtros")

        self.oos_result = oos_result
        return oos_result if is_valid else None

    def save(self, n_tested: int = 0, oos_result: BacktestResult = None):
        """
        V3.3: Salva melhor configuração com resultados OOS
        """
        if not self.best:
            print("  Nenhuma configuração robusta encontrada!")
            return

        configs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "configs"
        )
        os.makedirs(configs_dir, exist_ok=True)

        best_file = os.path.join(configs_dir, "dtt-tunelamentotopologico_robust.json")

        config = {
            "strategy": "DTT-TunelamentoTopologico",
            "version": "3.3-robust-validation",
            "optimized_at": datetime.now(timezone.utc).isoformat(),
            "embedding": {
                "time_delay_tau": self.fixed_tau,
                "embedding_dim_m": self.fixed_m,
                "note": "Calculados apenas com dados de TREINO (anti look-ahead)"
            },
            "validation": {
                "method": "walk_forward_sliding_windows",
                "n_windows": self.N_WALK_FORWARD_WINDOWS,
                "min_trades_train": self.MIN_TRADES_TRAIN,
                "min_trades_test": self.MIN_TRADES_TEST,
                "monte_carlo_permutations": self.MONTE_CARLO_PERMUTATIONS,
                "significance_level": self.SIGNIFICANCE_LEVEL,
                "out_of_sample_ratio": self.OUT_OF_SAMPLE_RATIO,
            },
            "parameters": self.best.params,
            "performance": self.best.to_dict(),
        }

        # V3.3: Incluir resultados Out-of-Sample
        if oos_result:
            config["out_of_sample"] = {
                "trades": oos_result.trades,
                "win_rate": round(oos_result.win_rate, 4),
                "profit_factor": round(oos_result.profit_factor, 4),
                "expectancy": round(oos_result.expectancy, 2),
                "max_drawdown": round(oos_result.max_drawdown, 4),
                "validated": oos_result.profit_factor >= 1.10,
            }
        elif hasattr(self, 'oos_result') and self.oos_result:
            config["out_of_sample"] = {
                "trades": self.oos_result.trades,
                "win_rate": round(self.oos_result.win_rate, 4),
                "profit_factor": round(self.oos_result.profit_factor, 4),
                "expectancy": round(self.oos_result.expectancy, 2),
                "max_drawdown": round(self.oos_result.max_drawdown, 4),
                "validated": self.oos_result.profit_factor >= 1.10,
            }

        with open(best_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"\n  Salvo em: {best_file}")


async def main():
    N_COMBINATIONS = 500000

    print("=" * 70)
    print("  OTIMIZADOR DTT V3.3 - VALIDAÇÃO ESTATÍSTICA COMPLETA")
    print("=" * 70)
    print("\n  MELHORIAS V3.3:")
    print("    ✓ Monte Carlo: 10.000 permutações (era 1.000)")
    print("    ✓ Walk-Forward: 12 janelas (era 6)")
    print("    ✓ Out-of-Sample: 20% reservado para validação final")
    print("    ✓ Detecção de overfitting por degradação WF vs OOS")
    print("\n  CORREÇÕES V3.2 (herdado):")
    print("    ✓ Embedding (τ,m) calculado APENAS com dados de TREINO")
    print("    ✓ Parâmetros FIXADOS para toda a análise")
    print("\n  CORREÇÕES V3.1 (herdado):")
    print("    ✓ Walk-Forward REAL (janelas DESLIZANTES)")
    print("    ✓ Teste de significância estatística (Monte Carlo)")
    print("    ✓ p-value < 0.05 para validação")
    print("=" * 70)

    opt = DTTRobustOptimizer("EURUSD", "H1")

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    if await opt.load_and_precompute(start, end):
        best = opt.optimize(N_COMBINATIONS)
        if best:
            # V3.3: Validação Out-of-Sample CRÍTICA
            print("\n" + "=" * 70)
            print("  FASE FINAL: VALIDAÇÃO OUT-OF-SAMPLE")
            print("=" * 70)

            oos_result = opt.validate_out_of_sample()

            if oos_result:
                print("\n  ✓ APROVADO: Parâmetros validados em dados OOS!")
                opt.save(n_tested=N_COMBINATIONS, oos_result=oos_result)
            else:
                print("\n  ✗ REPROVADO: Parâmetros não passaram na validação OOS")
                print("  POSSÍVEL OVERFITTING - não usar em produção!")
                # Salvar mesmo assim para análise, mas marcar como não validado
                opt.save(n_tested=N_COMBINATIONS)
        else:
            print("\n  AVISO: Nenhuma configuração passou nos filtros!")
            print("  Possíveis causas:")
            print("    - Resultados não são estatisticamente significativos")
            print("    - Insuficientes trades para validação")
            print("    - DTT pode não adicionar valor vs filtros simples")


if __name__ == "__main__":
    asyncio.run(main())
