#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR ROBUSTO V2.0 - PRONTO PARA DINHEIRO REAL
================================================================================

Este otimizador implementa:
1. Train/Test Split (70/30) - treina em 70%, valida em 30%
2. Filtros RIGOROSOS para dinheiro real
3. Score de Robustez - compara performance train vs test
4. Custos de execução REALISTAS

CORREÇÕES V2.0:
1. Verificação de GAPS no OPEN antes de HIGH/LOW
2. Stop Loss tem prioridade (conservador)
3. Spread realista: 1.5 pips
4. Slippage realista: 0.8 pips
5. Filtros mais rigorosos (PF > 1.3)

REGRAS PARA DINHEIRO REAL:
- Minimo 50 trades no treino, 25 no teste
- Win Rate entre 35% e 60% (realista)
- Profit Factor mínimo 1.3 (treino), 1.15 (teste)
- Drawdown maximo 30%
- Performance do teste deve ser >= 70% do treino
- Expectativa mínima: 3 pips por trade

IMPORTANTE: Isso é dinheiro real. Custos realistas aplicados.
================================================================================
"""

import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import os

# CORREÇÃO C4: Importar custos centralizados
from config.execution_costs import (
    SPREAD_PIPS,
    SLIPPAGE_PIPS,
)


@dataclass
class TradeResult:
    """Resultado de um trade individual"""
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    direction: int  # 1=LONG, -1=SHORT
    pnl_pips: float
    hit_sl: bool
    hit_tp: bool
    had_gap: bool = False  # NOVO: Se houve gap na execução


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

    def is_valid(self, min_trades: int = 30,
                 max_win_rate: float = 0.65,
                 min_win_rate: float = 0.30,
                 max_pf: float = 4.0,
                 min_pf: float = 1.1,
                 max_dd: float = 0.40) -> bool:
        """Verifica se o resultado passa nos filtros de realismo"""
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
class RobustResult:
    """Resultado robusto com train e test"""
    params: Dict
    train_result: BacktestResult
    test_result: BacktestResult
    robustness_score: float
    degradation: float  # % de degradacao do treino para teste
    is_robust: bool

    def to_dict(self) -> Dict:
        return {
            "params": self.params,
            "train": {
                "trades": self.train_result.trades,
                "win_rate": round(self.train_result.win_rate, 4),
                "profit_factor": round(self.train_result.profit_factor, 4),
                "total_pnl": round(self.train_result.total_pnl, 2),
                "max_drawdown": round(self.train_result.max_drawdown, 4)
            },
            "test": {
                "trades": self.test_result.trades,
                "win_rate": round(self.test_result.win_rate, 4),
                "profit_factor": round(self.test_result.profit_factor, 4),
                "total_pnl": round(self.test_result.total_pnl, 2),
                "max_drawdown": round(self.test_result.max_drawdown, 4)
            },
            "robustness_score": round(self.robustness_score, 4),
            "degradation_pct": round(self.degradation * 100, 2),
            "is_robust": self.is_robust
        }


class RobustBacktester:
    """
    Backtester robusto V2.0 - Pronto para dinheiro real

    Custos REALISTAS e filtros RIGOROSOS
    """

    # Constantes RIGOROSAS para dinheiro real
    MIN_TRADES_TRAIN = 50
    MIN_TRADES_TEST = 25
    MIN_WIN_RATE = 0.35
    MAX_WIN_RATE = 0.60
    MIN_PROFIT_FACTOR = 1.30   # V2.0: Mais rigoroso
    MAX_PROFIT_FACTOR = 3.5
    MAX_DRAWDOWN = 0.30        # V2.0: Mais conservador
    MIN_ROBUSTNESS = 0.70      # V2.0: Teste deve ter >= 70% da performance do treino
    MIN_EXPECTANCY = 3.0       # V2.0: Mínimo 3 pips por trade

    def __init__(self, pip: float = 0.0001, spread: float = None, slippage: float = None):
        """
        CORREÇÃO C4: Usa custos centralizados por padrão

        Args:
            pip: Valor de 1 pip (ex: 0.0001 para EURUSD)
            spread: Spread em pips (None = usa config centralizado)
            slippage: Slippage em pips (None = usa config centralizado)
        """
        self.pip = pip
        # CORREÇÃO C4: Usar custos centralizados se não especificados
        self.spread = spread if spread is not None else SPREAD_PIPS
        self.slippage = slippage if slippage is not None else SLIPPAGE_PIPS

    def split_data(self, data: List, train_ratio: float = 0.70) -> Tuple[List, List]:
        """Divide dados em treino e teste"""
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:]

    def calculate_backtest_result(self, pnls: List[float]) -> BacktestResult:
        """Calcula metricas de um backtest"""
        if not pnls:
            return BacktestResult(
                trades=0, wins=0, losses=0, total_pnl=0,
                win_rate=0, profit_factor=0, max_drawdown=1.0,
                avg_trade=0, largest_win=0, largest_loss=0
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
            largest_loss=largest_loss
        )

    def calculate_robustness(self, train: BacktestResult,
                             test: BacktestResult) -> Tuple[float, float, bool]:
        """
        Calcula score de robustez comparando treino vs teste

        Retorna: (robustness_score, degradation, is_robust)
        """
        # Metricas de comparacao
        if train.profit_factor > 0:
            pf_ratio = test.profit_factor / train.profit_factor
        else:
            pf_ratio = 0

        if train.win_rate > 0:
            wr_ratio = test.win_rate / train.win_rate
        else:
            wr_ratio = 0

        # Degradacao media
        degradation = 1.0 - (pf_ratio + wr_ratio) / 2

        # Score de robustez (0 a 1)
        # Penaliza alta degradacao
        robustness = max(0, min(1, 1 - degradation))

        # E' robusto se teste manteve >= 60% da performance
        is_robust = (
            pf_ratio >= self.MIN_ROBUSTNESS and
            wr_ratio >= self.MIN_ROBUSTNESS and
            test.profit_factor >= 1.0  # Teste deve ser lucrativo
        )

        return robustness, degradation, is_robust

    def execute_trade(self, bars: List, entry_idx: int,
                      entry_price: float, direction: int,
                      sl_pips: float, tp_pips: float,
                      max_bars: int = 200) -> TradeResult:
        """
        Executa um trade com SL/TP
        
        VERSÃO CORRIGIDA:
        1. Verifica GAPS no OPEN antes de HIGH/LOW
        2. Stop tem prioridade (conservador)
        3. Slippage aplicado corretamente

        IMPORTANTE: Entrada na barra SEGUINTE ao sinal (entry_idx é onde executa)
        O entry_price já deve ser o OPEN da barra de entrada
        """
        # Aplicar spread e slippage na entrada
        slippage_value = self.slippage * self.pip
        spread_value = self.spread * self.pip
        
        if direction == 1:  # LONG
            actual_entry = entry_price + spread_value / 2 + slippage_value
            sl_price = actual_entry - sl_pips * self.pip
            tp_price = actual_entry + tp_pips * self.pip
        else:  # SHORT
            actual_entry = entry_price - spread_value / 2 - slippage_value
            sl_price = actual_entry + sl_pips * self.pip
            tp_price = actual_entry - tp_pips * self.pip

        exit_idx = entry_idx
        exit_price = actual_entry
        hit_sl = False
        hit_tp = False
        had_gap = False

        # Percorre barras futuras (a partir da PROXIMA barra após entrada)
        for j in range(entry_idx + 1, min(entry_idx + max_bars, len(bars))):
            bar = bars[j]

            # ================================================================
            # CORREÇÃO: VERIFICAR GAPS NO OPEN PRIMEIRO
            # ================================================================
            
            if direction == 1:  # LONG
                # Gap Down - Stop Loss atingido no OPEN
                if bar.open <= sl_price:
                    exit_price = bar.open - slippage_value  # Pior execução
                    exit_idx = j
                    hit_sl = True
                    had_gap = True
                    break
                    
                # Gap Up - Take Profit atingido no OPEN
                if bar.open >= tp_price:
                    exit_price = bar.open - slippage_value  # Execução no open
                    exit_idx = j
                    hit_tp = True
                    had_gap = True
                    break
                    
            else:  # SHORT
                # Gap Up - Stop Loss atingido no OPEN
                if bar.open >= sl_price:
                    exit_price = bar.open + slippage_value  # Pior execução
                    exit_idx = j
                    hit_sl = True
                    had_gap = True
                    break
                    
                # Gap Down - Take Profit atingido no OPEN
                if bar.open <= tp_price:
                    exit_price = bar.open + slippage_value  # Execução no open
                    exit_idx = j
                    hit_tp = True
                    had_gap = True
                    break

            # ================================================================
            # VERIFICAR DURANTE A BARRA (HIGH/LOW)
            # Stop tem prioridade (conservador)
            # ================================================================
            
            if direction == 1:  # LONG
                # Verifica SL primeiro (conservador)
                if bar.low <= sl_price:
                    exit_price = sl_price - slippage_value
                    exit_idx = j
                    hit_sl = True
                    break
                # Verifica TP
                if bar.high >= tp_price:
                    exit_price = tp_price - slippage_value
                    exit_idx = j
                    hit_tp = True
                    break
            else:  # SHORT
                # Verifica SL primeiro (conservador)
                if bar.high >= sl_price:
                    exit_price = sl_price + slippage_value
                    exit_idx = j
                    hit_sl = True
                    break
                # Verifica TP
                if bar.low <= tp_price:
                    exit_price = tp_price + slippage_value
                    exit_idx = j
                    hit_tp = True
                    break

        # Se nao bateu SL nem TP, fecha no final
        if not hit_sl and not hit_tp:
            exit_idx = min(entry_idx + max_bars, len(bars) - 1)
            if direction == 1:
                exit_price = bars[exit_idx].close - slippage_value
            else:
                exit_price = bars[exit_idx].close + slippage_value

        # Calcula PnL (já inclui spread na entrada)
        if direction == 1:
            pnl = (exit_price - actual_entry) / self.pip
        else:
            pnl = (actual_entry - exit_price) / self.pip

        return TradeResult(
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            entry_price=actual_entry,
            exit_price=exit_price,
            direction=direction,
            pnl_pips=pnl,
            hit_sl=hit_sl,
            hit_tp=hit_tp,
            had_gap=had_gap
        )


def save_robust_config(result: RobustResult, strategy_name: str,
                       symbol: str, periodicity: str,
                       n_tested: int, n_robust: int,
                       filename: str = None):
    """Salva configuracao robusta validada"""

    if filename is None:
        filename = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", f"{strategy_name.lower()}_robust.json"
        )

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    config = {
        "strategy": strategy_name,
        "symbol": symbol,
        "periodicity": periodicity,
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "version": "2.0-corrected",  # NOVO: Indicar versão corrigida
        "validation": {
            "method": "train_test_split",
            "train_ratio": 0.70,
            "test_ratio": 0.30,
            "combinations_tested": n_tested,
            "robust_found": n_robust,
            "anti_overfitting": True,
            "gap_handling": True,  # NOVO
            "realistic_execution": True  # NOVO
        },
        "parameters": result.params,
        "performance": {
            "train": {
                "trades": result.train_result.trades,
                "win_rate": round(result.train_result.win_rate, 4),
                "profit_factor": round(result.train_result.profit_factor, 4),
                "total_pnl_pips": round(result.train_result.total_pnl, 2),
                "max_drawdown": round(result.train_result.max_drawdown, 4)
            },
            "test": {
                "trades": result.test_result.trades,
                "win_rate": round(result.test_result.win_rate, 4),
                "profit_factor": round(result.test_result.profit_factor, 4),
                "total_pnl_pips": round(result.test_result.total_pnl, 2),
                "max_drawdown": round(result.test_result.max_drawdown, 4)
            },
            "robustness_score": round(result.robustness_score, 4),
            "degradation_pct": round(result.degradation * 100, 2)
        },
        "filters_applied": {
            "min_trades_train": RobustBacktester.MIN_TRADES_TRAIN,
            "min_trades_test": RobustBacktester.MIN_TRADES_TEST,
            "win_rate_range": [RobustBacktester.MIN_WIN_RATE, RobustBacktester.MAX_WIN_RATE],
            "profit_factor_range": [RobustBacktester.MIN_PROFIT_FACTOR, RobustBacktester.MAX_PROFIT_FACTOR],
            "max_drawdown": RobustBacktester.MAX_DRAWDOWN,
            "min_robustness": RobustBacktester.MIN_ROBUSTNESS
        }
    }

    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n  Configuracao robusta salva em: {filename}")
    return filename
