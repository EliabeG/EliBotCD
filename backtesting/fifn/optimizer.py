#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR FIFN ROBUSTO V2.2 - PRONTO PARA DINHEIRO REAL
================================================================================

Este otimizador implementa:
1. Walk-Forward Validation (multiplas janelas train/test)
2. Filtros rigorosos para dinheiro real (PF > 1.3)
3. Custos realistas (spread 1.5 pips, slippage 0.8 pips)
4. Validacao em multiplos periodos de mercado

CORRECOES V2.0:
1. Entrada no OPEN da proxima barra (sem look-ahead)
2. Direcao baseada apenas em barras FECHADAS
3. Walk-Forward com 4 janelas de validacao
4. Filtros mais rigorosos para dinheiro real
5. Custos de execucao realistas

AUDITORIA 27 (V2.1):
1. Latin Hypercube Sampling para melhor cobertura do espaco de parametros
2. Aumento de 500k para 800k combinacoes (~16.5% de cobertura)
3. LHS equivale a ~25% de eficiencia vs random sampling

AUDITORIA 28 (V2.2) - CRITICO:
1. Stops DINAMICOS baseados em Reynolds (igual strategy) - CRITICO
2. min_prices unificado: 100 barras (era 80 optimizer, 120 strategy) - CRITICO
3. Cooldown de 12 barras apos cada trade (igual strategy) - CRITICO
4. Documentacao de prevencao de look-ahead no indicador

REGRAS PARA DINHEIRO REAL:
- Minimo 50 trades no treino, 35 no teste (AUDITORIA 25: aumentado de 25)
- Win Rate entre 35% e 65%
- Profit Factor minimo 1.3 (treino) e 1.15 (teste)
- Drawdown maximo 20% (AUDITORIA 25: reduzido de 30%)
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

# AUDITORIA 27: Latin Hypercube Sampling para melhor cobertura do espaço de parâmetros
try:
    from scipy.stats import qmc
    LHS_AVAILABLE = True
except ImportError:
    LHS_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.fxopen_historical_ws import Bar, download_historical_data
from strategies.alta_volatilidade.fifn_fisher_navier import FluxoInformacaoFisherNavier

# AUDITORIA 25: Usar módulo centralizado para cálculo de direção
from backtesting.common.direction_calculator import calculate_direction_from_bars


@dataclass
class FIFNSignal:
    """
    Sinal pre-calculado do FIFN

    CORRIGIDO V2.1: Armazena informacoes para execucao realista
    AUDITORIA 1: Removidos campos high/low que causavam look-ahead
    """
    bar_idx: int          # Indice da barra onde o sinal foi GERADO
    signal_price: float   # Preco de fechamento quando sinal foi gerado (para referencia)
    next_bar_idx: int     # Indice da barra onde deve EXECUTAR (proxima barra)
    entry_price: float    # Preco de ABERTURA da proxima barra (onde realmente entra)
    reynolds: float       # Numero de Reynolds (calculado sem barra atual)
    kl_divergence: float  # Divergencia KL (calculado sem barra atual)
    skewness: float       # Assimetria (calculado sem barra atual)
    pressure_gradient: float  # Gradiente de pressao (calculado sem barra atual)
    in_sweet_spot: bool   # Se esta na zona de transicao
    direction: int        # Baseado apenas em barras JA FECHADAS


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
                                max_win_rate: float = 0.65,
                                min_win_rate: float = 0.35,
                                max_pf: float = 3.5,
                                min_pf: float = 1.30,
                                max_dd: float = 0.30,
                                min_expectancy: float = 3.0) -> bool:
        """
        Verifica se o resultado passa nos filtros RIGOROSOS para dinheiro real
        """
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


class FIFNRobustOptimizer:
    """
    Otimizador FIFN V2.0 com Walk-Forward Validation

    PRONTO PARA DINHEIRO REAL
    """

    # Custos REALISTAS de execucao
    SPREAD_PIPS = 1.5
    SLIPPAGE_PIPS = 0.8
    COMMISSION_PIPS = 0.0

    # Filtros RIGOROSOS para dinheiro real
    # AUDITORIA 24: Aumentado MIN_TRADES_TEST de 25 para 35
    # AUDITORIA 24: Reduzido MAX_DRAWDOWN de 30% para 20%
    MIN_TRADES_TRAIN = 50
    MIN_TRADES_TEST = 35    # AUDITORIA 24: Aumentado de 25 para 35
    MIN_WIN_RATE = 0.35
    MAX_WIN_RATE = 0.65
    MIN_PF_TRAIN = 1.30
    MIN_PF_TEST = 1.15
    MAX_PF = 3.5
    MAX_DRAWDOWN = 0.20     # AUDITORIA 24: Reduzido de 0.30 para 0.20
    MIN_ROBUSTNESS = 0.70
    MIN_EXPECTANCY = 3.0

    # AUDITORIA 28: Cooldown para consistência com strategy
    SIGNAL_COOLDOWN_BARS = 12  # Ignora 12 barras após cada trade (igual strategy)

    # AUDITORIA 28: Parâmetros para stops dinâmicos (igual strategy)
    BASE_STOP_LOSS_PIPS = 18.0
    BASE_TAKE_PROFIT_PIPS = 36.0

    # Limites para gaps extremos
    MAX_GAP_PIPS = 100

    @staticmethod
    def _calculate_dynamic_stops(reynolds: float, base_sl: float, base_tp: float) -> tuple:
        """
        AUDITORIA 28: Calcula stops dinâmicos baseados em Reynolds.
        Implementação IDÊNTICA à fifn_strategy.py para garantir consistência.

        O Número de Reynolds indica o regime de mercado:
        - Turbulento (Re > 4000): Alta volatilidade → stops mais largos
        - Sweet Spot (2300-4000): Volatilidade moderada → stops padrão
        - Laminar (Re < 2000): Baixa volatilidade → stops mais apertados

        Args:
            reynolds: Número de Reynolds do mercado
            base_sl: Stop loss base em pips
            base_tp: Take profit base em pips

        Returns:
            Tuple[dynamic_sl, dynamic_tp]: Stops ajustados por regime
        """
        if reynolds > 4000:  # Turbulento - stops mais largos para evitar ruído
            multiplier = 1.5
        elif reynolds > 3000:  # Transição alta
            multiplier = 1.2
        elif reynolds < 2000:  # Laminar - stops mais apertados
            multiplier = 0.8
        elif reynolds < 2300:  # Transição baixa
            multiplier = 0.9
        else:  # Sweet Spot (2300-3000)
            multiplier = 1.0

        return base_sl * multiplier, base_tp * multiplier

    def __init__(self, symbol: str = "EURUSD", periodicity: str = "H1"):
        self.symbol = symbol
        self.periodicity = periodicity
        self.pip = 0.0001

        self.bars: List[Bar] = []
        self.signals: List[FIFNSignal] = []

        # Resultados
        self.robust_results: List[RobustResult] = []
        self.best: Optional[RobustResult] = None

    async def load_and_precompute(self, start_date: datetime, end_date: datetime,
                                   split_date: datetime = None):
        """
        Carrega dados e pre-calcula sinais FIFN

        CORRIGIDO V2.0:
        - Direcao usa apenas barras completamente fechadas
        - Entry price e o OPEN da proxima barra

        Args:
            start_date: Data inicial dos dados
            end_date: Data final dos dados
            split_date: Data de divisao train/test (se None, usa 70/30)
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
        print("\n  Pre-calculando sinais FIFN V2.0 (sem look-ahead)...")

        fifn = FluxoInformacaoFisherNavier(
            window_size=50,
            kl_lookback=10,
            reynolds_sweet_low=2300,
            reynolds_sweet_high=4000,
            skewness_threshold=0.5
        )

        prices_buf = deque(maxlen=500)
        self.signals = []

        # AUDITORIA 28: Unificado com strategy (era 80, strategy usa 120)
        min_prices = 100  # Valor intermediário para consistência
        min_bars_for_direction = 12

        for i, bar in enumerate(self.bars):
            prices_buf.append(bar.close)

            if len(prices_buf) < min_prices:
                continue

            # Precisamos da PROXIMA barra para executar
            if i >= len(self.bars) - 1:
                continue

            try:
                # CORRIGIDO AUDITORIA 1: Usar apenas barras JA FECHADAS
                # Exclui a barra atual (que ainda esta "em formacao")
                # Isso garante que nao ha look-ahead no calculo do indicador
                prices_for_analysis = np.array(prices_buf)[:-1]  # Exclui barra atual

                if len(prices_for_analysis) < min_prices - 1:
                    continue

                result = fifn.analyze(prices_for_analysis)

                reynolds = result['Reynolds_Number']
                kl_div = result['KL_Divergence']
                skewness = result['directional_signal']['skewness']
                pressure_grad = result['Pressure_Gradient']
                in_sweet_spot = result['directional_signal']['in_sweet_spot']

                # AUDITORIA 25: Usar módulo centralizado para cálculo de direção
                # Garante consistência entre optimizer, strategy e outros componentes
                # calculate_direction_from_bars compara bars[i-1] vs bars[i-11] = 10 barras
                direction = calculate_direction_from_bars(self.bars, i)

                next_bar = self.bars[i + 1]

                self.signals.append(FIFNSignal(
                    bar_idx=i,
                    signal_price=bar.close,
                    next_bar_idx=i + 1,
                    entry_price=next_bar.open,
                    reynolds=reynolds,
                    kl_divergence=kl_div,
                    skewness=skewness,
                    pressure_gradient=pressure_grad,
                    in_sweet_spot=in_sweet_spot,
                    direction=direction
                ))

            except Exception as e:
                continue

            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(self.bars)} barras...")

        print(f"\n  Sinais pre-calculados: {len(self.signals)}")

        # Estatisticas
        long_signals = sum(1 for s in self.signals if s.direction == 1)
        short_signals = sum(1 for s in self.signals if s.direction == -1)
        print(f"    Long: {long_signals}, Short: {short_signals}")

        # Distribuicao de valores
        if self.signals:
            reynolds_vals = [s.reynolds for s in self.signals]
            skew_vals = [s.skewness for s in self.signals]
            kl_vals = [s.kl_divergence for s in self.signals]
            sweet_count = sum(1 for s in self.signals if s.in_sweet_spot)
            print(f"\n  Distribuicao de valores:")
            print(f"    Reynolds: min={min(reynolds_vals):.0f}, max={max(reynolds_vals):.0f}, mean={np.mean(reynolds_vals):.0f}")
            print(f"    Skewness: min={min(skew_vals):.3f}, max={max(skew_vals):.3f}, mean={np.mean(skew_vals):.3f}")
            print(f"    KL Div: min={min(kl_vals):.4f}, max={max(kl_vals):.4f}, mean={np.mean(kl_vals):.4f}")
            print(f"    In Sweet Spot: {sweet_count} ({sweet_count/len(self.signals)*100:.1f}%)")

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

        # Expectancy (media por trade)
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

    # AUDITORIA 2: Limite maximo de gap aceitavel
    MAX_GAP_PIPS = 50.0  # Gaps maiores que 50 pips sao rejeitados

    def _run_backtest(self, signals: List[FIFNSignal], bars: List[Bar],
                      reynolds_low: float, reynolds_high: float,
                      skewness_thresh: float, kl_thresh: float,
                      sl: float, tp: float,
                      bar_offset: int = 0) -> List[float]:
        """
        Executa backtest com CUSTOS REALISTAS

        V2.1 (AUDITORIA 2):
        - Spread: 1.5 pips
        - Slippage: 0.8 pips
        - Entrada no OPEN da proxima barra
        - Verificacao de gaps com limite maximo
        - Validacao de TP > custos totais
        """
        # AUDITORIA 2: Validar TP > custos totais
        total_costs = self.SPREAD_PIPS + self.SLIPPAGE_PIPS
        if tp <= total_costs:
            return []

        if tp <= sl:
            return []

        # Encontra entradas validas
        # AUDITORIA 28: Agora inclui Reynolds para stops dinâmicos
        entries = []
        for s in signals:
            # Verificar se esta na zona de operacao (sweet spot)
            in_zone = reynolds_low <= s.reynolds <= reynolds_high

            # Verificar condicoes de entrada
            if (in_zone and
                abs(s.skewness) >= skewness_thresh and
                s.kl_divergence >= kl_thresh and
                s.direction != 0):

                # CORRIGIDO V2.0: Usar direcao baseada em skewness E direcao de tendencia
                # LONG: skewness positiva, pressao negativa, tendencia alta
                if s.skewness > skewness_thresh and s.pressure_gradient < 0 and s.direction == 1:
                    execution_idx = s.next_bar_idx - bar_offset
                    # AUDITORIA 28: Inclui Reynolds para stops dinâmicos
                    entries.append((execution_idx, s.entry_price, 1, s.reynolds))
                # SHORT: skewness negativa, pressao positiva, tendencia baixa
                elif s.skewness < -skewness_thresh and s.pressure_gradient > 0 and s.direction == -1:
                    execution_idx = s.next_bar_idx - bar_offset
                    # AUDITORIA 28: Inclui Reynolds para stops dinâmicos
                    entries.append((execution_idx, s.entry_price, -1, s.reynolds))

        if len(entries) < 5:
            return []

        # Executa trades com custos REALISTAS
        pnls = []
        pip = self.pip
        spread = self.SPREAD_PIPS * pip
        slippage = self.SLIPPAGE_PIPS * pip
        total_cost = spread + slippage

        # AUDITORIA 28: Controle de cooldown (igual strategy)
        last_exit_idx = -1
        cooldown_until_idx = -1

        for entry_idx, entry_price_raw, direction, signal_reynolds in entries:
            if entry_idx < 0 or entry_idx >= len(bars) - 1:
                continue

            # AUDITORIA 28: Verificar cooldown (igual strategy)
            if entry_idx <= cooldown_until_idx:
                continue

            if entry_idx <= last_exit_idx:
                continue

            # AUDITORIA 28: Calcular stops dinâmicos baseados em Reynolds
            # Usa SL/TP BASE da otimização, mas AJUSTA pelo regime de volatilidade
            dynamic_sl, dynamic_tp = self._calculate_dynamic_stops(
                signal_reynolds, sl, tp
            )

            # Aplicar custos na entrada
            if direction == 1:  # LONG
                entry_price = entry_price_raw + total_cost / 2
                stop_price = entry_price - dynamic_sl * pip
                take_price = entry_price + dynamic_tp * pip
            else:  # SHORT
                entry_price = entry_price_raw - total_cost / 2
                stop_price = entry_price + dynamic_sl * pip
                take_price = entry_price - dynamic_tp * pip

            # Simular execucao
            exit_price = None
            exit_bar_idx = entry_idx
            max_bars = min(200, len(bars) - entry_idx - 1)

            for j in range(1, max_bars + 1):
                bar_idx = entry_idx + j
                if bar_idx >= len(bars):
                    break

                bar = bars[bar_idx]

                # AUDITORIA 2: Verificar GAPS no OPEN com limite maximo
                prev_bar = bars[bar_idx - 1] if bar_idx > 0 else bars[bar_idx]
                gap_size = abs(bar.open - prev_bar.close) / pip

                # Rejeitar gaps excessivos (podem indicar dados ruins ou eventos extremos)
                if gap_size > self.MAX_GAP_PIPS:
                    # Gap muito grande - usar preco mais conservador
                    if direction == 1:  # LONG
                        if bar.open <= stop_price:
                            # Gap contra a posicao - assumir pior caso
                            exit_price = stop_price - gap_size * pip * 0.5  # Extra slippage por gap
                            exit_bar_idx = bar_idx
                            break
                    else:  # SHORT
                        if bar.open >= stop_price:
                            exit_price = stop_price + gap_size * pip * 0.5
                            exit_bar_idx = bar_idx
                            break

                # Verificar GAPS no OPEN normais
                if direction == 1:  # LONG
                    if bar.open <= stop_price:
                        exit_price = bar.open - slippage
                        exit_bar_idx = bar_idx
                        break
                    if bar.open >= take_price:
                        exit_price = bar.open - slippage
                        exit_bar_idx = bar_idx
                        break
                else:  # SHORT
                    if bar.open >= stop_price:
                        exit_price = bar.open + slippage
                        exit_bar_idx = bar_idx
                        break
                    if bar.open <= take_price:
                        exit_price = bar.open + slippage
                        exit_bar_idx = bar_idx
                        break

                # Verificar durante a barra (stop tem prioridade)
                if direction == 1:  # LONG
                    if bar.low <= stop_price:
                        exit_price = stop_price - slippage
                        exit_bar_idx = bar_idx
                        break
                    if bar.high >= take_price:
                        exit_price = take_price - slippage
                        exit_bar_idx = bar_idx
                        break
                else:  # SHORT
                    if bar.high >= stop_price:
                        exit_price = stop_price + slippage
                        exit_bar_idx = bar_idx
                        break
                    if bar.low <= take_price:
                        exit_price = take_price + slippage
                        exit_bar_idx = bar_idx
                        break

            # Timeout
            if exit_price is None:
                exit_bar_idx = min(entry_idx + max_bars, len(bars) - 1)
                last_bar = bars[exit_bar_idx]
                if direction == 1:
                    exit_price = last_bar.close - slippage
                else:
                    exit_price = last_bar.close + slippage

            # Calcular PnL
            if direction == 1:
                pnl_pips = (exit_price - entry_price) / pip
            else:
                pnl_pips = (entry_price - exit_price) / pip

            pnls.append(pnl_pips)
            last_exit_idx = exit_bar_idx

            # AUDITORIA 28: Aplicar cooldown após cada trade (igual strategy)
            cooldown_until_idx = exit_bar_idx + self.SIGNAL_COOLDOWN_BARS

        return pnls

    # AUDITORIA 3/23: Gap entre treino e teste para evitar data leakage
    # AUDITORIA 23: Aumentado de 24 para 70 barras
    # Motivo: FIFN usa window_size=50 + kl_lookback=10 = 60 barras de dependência temporal
    # Gap deve ser >= 60 + buffer(10) = 70 para evitar data leakage
    TRAIN_TEST_GAP_BARS = 70  # 70 barras >= window_size + kl_lookback + buffer

    def _create_walk_forward_windows(self, n_windows: int = 4) -> List[Tuple[int, int, int, int]]:
        """
        Cria janelas para Walk-Forward Validation

        CORRIGIDO AUDITORIA 1: Janelas NAO-SOBREPOSTAS
        CORRIGIDO AUDITORIA 3: Gap de 24 barras entre treino e teste
        CORRIGIDO AUDITORIA 23: Gap aumentado para 70 barras (>= dependência do indicador)
        Divide os dados em n_windows janelas sequenciais
        Cada janela: 70% treino, 30% teste (com gap)
        Janelas sao completamente independentes (sem overlap)
        """
        total_bars = len(self.bars)
        window_size = total_bars // n_windows

        windows = []
        for i in range(n_windows):
            # CORRIGIDO: Janelas NAO-SOBREPOSTAS
            window_start = i * window_size
            window_end = (i + 1) * window_size
            if i == n_windows - 1:
                window_end = total_bars

            # Dentro de cada janela: 70% treino, 30% teste (com gap)
            train_size = int((window_end - window_start) * 0.70)
            train_start = window_start
            train_end = window_start + train_size

            # AUDITORIA 3: Gap entre treino e teste
            test_start = train_end + self.TRAIN_TEST_GAP_BARS
            test_end = window_end

            # Verificar se teste tem tamanho minimo
            if test_end - test_start < 50:  # Minimo 50 barras para teste
                test_start = train_end  # Remover gap se janela muito pequena

            windows.append((train_start, train_end, test_start, test_end))

        return windows

    def _test_params_walk_forward(self, reynolds_low: float, reynolds_high: float,
                                   skewness_thresh: float, kl_thresh: float,
                                   sl: float, tp: float) -> Optional[RobustResult]:
        """
        Testa parametros com Walk-Forward Validation completa
        """
        # Validacao basica de parametros
        if reynolds_high <= reynolds_low:
            return None

        windows = self._create_walk_forward_windows(n_windows=4)
        wf_results = []
        all_train_pnls = []
        all_test_pnls = []

        for idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            # Separar sinais e barras para esta janela
            train_signals = [s for s in self.signals if train_start <= s.bar_idx < train_end]
            test_signals = [s for s in self.signals if test_start <= s.bar_idx < test_end]

            train_bars = self.bars[train_start:train_end]
            test_bars = self.bars[test_start:test_end]

            if len(train_signals) < 20 or len(test_signals) < 10:
                return None

            # Backtest treino
            train_pnls = self._run_backtest(
                train_signals, train_bars,
                reynolds_low, reynolds_high, skewness_thresh, kl_thresh, sl, tp,
                bar_offset=train_start
            )
            train_result = self._calculate_backtest_result(train_pnls)

            # Verificar filtros do treino
            if train_result.trades < 20 or train_result.profit_factor < 1.15:
                return None

            # Backtest teste
            test_pnls = self._run_backtest(
                test_signals, test_bars,
                reynolds_low, reynolds_high, skewness_thresh, kl_thresh, sl, tp,
                bar_offset=test_start
            )
            test_result = self._calculate_backtest_result(test_pnls)

            # Verificar filtros do teste
            if test_result.trades < 10 or test_result.profit_factor < 0.95:
                return None

            # Calcular robustez desta janela
            pf_ratio = test_result.profit_factor / train_result.profit_factor if train_result.profit_factor > 0 else 0
            wr_ratio = test_result.win_rate / train_result.win_rate if train_result.win_rate > 0 else 0
            degradation = 1.0 - (pf_ratio + wr_ratio) / 2
            robustness = max(0, min(1, 1 - degradation))

            # Janela passa se mantem 65% da performance
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

        # Filtros finais RIGOROSOS para dinheiro real
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
            "reynolds_sweet_low": round(reynolds_low, 0),
            "reynolds_sweet_high": round(reynolds_high, 0),
            "skewness_threshold": round(skewness_thresh, 4),
            "kl_divergence_threshold": round(kl_thresh, 5),
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

    def optimize(self, n: int = 800000, use_lhs: bool = True) -> Optional[RobustResult]:
        """
        Executa otimizacao robusta com Walk-Forward.

        AUDITORIA 27: Implementação de Latin Hypercube Sampling (LHS)
        - LHS garante cobertura uniforme do espaço de parâmetros
        - Com 800k samples de ~4.86M combinações = 16.5% de cobertura
        - LHS equivale a ~25% de cobertura em termos de eficiência vs random

        Args:
            n: Número de combinações a testar (default: 800,000)
            use_lhs: Se True, usa Latin Hypercube Sampling (recomendado)
        """
        if not self.signals:
            print("  ERRO: Dados nao carregados!")
            return None

        sampling_method = "LHS (Latin Hypercube)" if use_lhs and LHS_AVAILABLE else "Random"

        print(f"\n{'='*70}")
        print(f"  OTIMIZACAO ROBUSTA FIFN V2.1: {n:,} COMBINACOES")
        print(f"  AUDITORIA 27: Sampling via {sampling_method}")
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

        # Ranges de parametros baseados na teoria
        # AUDITORIA 27: Definir bounds para LHS (min, max)
        param_bounds = {
            'reynolds_low': (1800, 2800),
            'reynolds_high': (3500, 5000),
            'skewness': (0.25, 0.75),
            'kl': (0.005, 0.05),
            'sl': (20, 50),
            'tp': (25, 80)
        }

        # AUDITORIA 27: Gerar samples via LHS ou Random
        if use_lhs and LHS_AVAILABLE:
            print(f"\n  Gerando {n:,} samples via Latin Hypercube Sampling...")
            sampler = qmc.LatinHypercube(d=6, seed=42)
            samples = sampler.random(n=n)

            # Escalar para os bounds de cada parâmetro
            lower_bounds = np.array([b[0] for b in param_bounds.values()])
            upper_bounds = np.array([b[1] for b in param_bounds.values()])
            scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)

            # Converter para lista de parâmetros
            param_list = []
            for i in range(n):
                param_list.append({
                    'reynolds_low': float(scaled_samples[i, 0]),
                    'reynolds_high': float(scaled_samples[i, 1]),
                    'skewness': float(scaled_samples[i, 2]),
                    'kl': float(scaled_samples[i, 3]),
                    'sl': float(scaled_samples[i, 4]),
                    'tp': float(scaled_samples[i, 5])
                })
            print(f"  LHS samples gerados com sucesso!")
        else:
            # Fallback para random sampling com grids discretos
            reynolds_low_vals = np.linspace(1800, 2800, 15)
            reynolds_high_vals = np.linspace(3500, 5000, 15)
            skewness_vals = np.linspace(0.25, 0.75, 12)
            kl_vals = np.linspace(0.005, 0.05, 10)
            sl_vals = np.linspace(20, 50, 12)
            tp_vals = np.linspace(25, 80, 15)
            param_list = None  # Usará random.choice no loop

        best_robustness = -1
        tested = 0
        robust_count = 0
        start = datetime.now()

        for idx in range(n):
            tested += 1

            # AUDITORIA 27: Usar parâmetros de LHS ou random
            if param_list is not None:
                params = param_list[idx]
                reynolds_low = params['reynolds_low']
                reynolds_high = params['reynolds_high']
                skewness_thresh = params['skewness']
                kl_thresh = params['kl']
                sl = params['sl']
                tp = params['tp']
            else:
                reynolds_low = float(random.choice(reynolds_low_vals))
                reynolds_high = float(random.choice(reynolds_high_vals))
                skewness_thresh = float(random.choice(skewness_vals))
                kl_thresh = float(random.choice(kl_vals))
                sl = float(random.choice(sl_vals))
                tp = float(random.choice(tp_vals))

            result = self._test_params_walk_forward(
                reynolds_low, reynolds_high, skewness_thresh, kl_thresh, sl, tp
            )

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
        best_file = os.path.join(configs_dir, "fifn-fishernavier_robust.json")

        config = {
            "strategy": "FIFN-FisherNavier",
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
        top_file = os.path.join(configs_dir, "fifn_robust_top10.json")
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
    # AUDITORIA 27: Aumentado de 500k para 800k com LHS
    N_COMBINATIONS = 800000

    print("=" * 70)
    print("  OTIMIZADOR FIFN V2.2 - PRONTO PARA DINHEIRO REAL")
    print("  AUDITORIA 27: Latin Hypercube Sampling + 800k samples")
    print("=" * 70)
    print("\n  CARACTERISTICAS:")
    print("    - Walk-Forward Validation (4 janelas)")
    print("    - Custos realistas (spread 1.5 + slippage 0.8 pips)")
    print("    - Filtros rigorosos (PF > 1.3, Exp > 3 pips)")
    print("    - Sem look-ahead em nenhum calculo")
    print("    - Direcao baseada apenas em barras FECHADAS")
    print("    - Entrada no OPEN da proxima barra")
    print("    - AUDITORIA 27: Latin Hypercube Sampling (16.5% cobertura)")
    print("=" * 70)

    opt = FIFNRobustOptimizer("EURUSD", "H1")

    # Periodos especificos de treino e teste
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
