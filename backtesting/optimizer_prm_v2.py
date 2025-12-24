#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR PRM V2 - Versao Otimizada com Cache
================================================================================

Abordagem mais eficiente:
1. Gera sinais UMA vez com parametros padrao
2. Otimiza apenas SL/TP sobre esses sinais
3. Depois testa variações do indicador se necessário

IMPORTANTE: Dados REAIS do mercado FXOpen.
"""

import sys
import os
import json
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.fxopen_historical_ws import Bar, download_historical_data
from strategies.alta_volatilidade import PRMStrategy
from strategies.base import SignalType


@dataclass
class CachedSignal:
    """Sinal cacheado"""
    bar_idx: int
    timestamp: datetime
    price: float
    direction: str
    confidence: float


@dataclass
class BacktestResult:
    """Resultado do backtest"""
    stop_loss: float
    take_profit: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl_pips: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    score: float


class PRMOptimizerV2:
    """Otimizador PRM com cache de sinais"""

    def __init__(self, symbol: str = "EURUSD", periodicity: str = "M15"):
        self.symbol = symbol
        self.periodicity = periodicity
        self.pip_value = 0.0001
        self.spread_pips = 1.0
        self.bars: List[Bar] = []
        self.cached_signals: List[CachedSignal] = []
        self.best_result: Optional[BacktestResult] = None
        self.detection_params = {}

    async def load_data(self, start_date: datetime, end_date: datetime):
        """Carrega dados REAIS"""
        print("\n  Carregando dados REAIS da FXOpen...")
        self.bars = await download_historical_data(
            symbol=self.symbol,
            periodicity=self.periodicity,
            start_time=start_date,
            end_time=end_date
        )
        print(f"  Barras carregadas: {len(self.bars)}")

    def generate_signals(self,
                        min_prices: int = 50,
                        hmm_threshold: float = 0.80,
                        lyapunov_threshold: float = 0.5,
                        curvature_threshold: float = 0.10,
                        verbose: bool = True):
        """Gera e cacheia sinais do PRM"""
        if verbose:
            print(f"\n  Gerando sinais PRM...")
            print(f"    min_prices={min_prices}, hmm={hmm_threshold}")
            print(f"    lyap={lyapunov_threshold}, curv={curvature_threshold}")

        self.detection_params = {
            'min_prices': min_prices,
            'hmm_threshold': hmm_threshold,
            'lyapunov_threshold': lyapunov_threshold,
            'curvature_threshold': curvature_threshold
        }

        strategy = PRMStrategy(
            min_prices=min_prices,
            stop_loss_pips=20.0,
            take_profit_pips=40.0,
            hmm_threshold=hmm_threshold,
            lyapunov_threshold=lyapunov_threshold,
            curvature_threshold=curvature_threshold
        )

        self.cached_signals = []
        last_print = 0

        for i, bar in enumerate(self.bars):
            # Progress
            if verbose and i > 0 and i % 500 == 0:
                print(f"    Processado: {i}/{len(self.bars)} barras, {len(self.cached_signals)} sinais")

            signal = strategy.analyze(
                price=bar.close,
                timestamp=bar.timestamp,
                volume=bar.volume,
                high=bar.high,
                low=bar.low,
                open=bar.open
            )

            if signal and signal.type != SignalType.HOLD:
                self.cached_signals.append(CachedSignal(
                    bar_idx=i,
                    timestamp=bar.timestamp,
                    price=bar.close,
                    direction='BUY' if signal.type == SignalType.BUY else 'SELL',
                    confidence=signal.confidence
                ))

        if verbose:
            print(f"\n  Total de sinais gerados: {len(self.cached_signals)}")

    def backtest_sltp(self, stop_loss: float, take_profit: float) -> BacktestResult:
        """Backtest rapido usando sinais cacheados"""
        trades_pnl = []

        for signal in self.cached_signals:
            entry_price = signal.price
            entry_idx = signal.bar_idx

            # Calcula niveis
            if signal.direction == 'BUY':
                sl_price = entry_price - (stop_loss * self.pip_value)
                tp_price = entry_price + (take_profit * self.pip_value)
            else:
                sl_price = entry_price + (stop_loss * self.pip_value)
                tp_price = entry_price - (take_profit * self.pip_value)

            # Busca saida
            pnl = 0.0
            for i in range(entry_idx + 1, min(entry_idx + 1000, len(self.bars))):
                bar = self.bars[i]

                if signal.direction == 'BUY':
                    if bar.low <= sl_price:
                        pnl = -stop_loss - self.spread_pips
                        break
                    if bar.high >= tp_price:
                        pnl = take_profit - self.spread_pips
                        break
                else:
                    if bar.high >= sl_price:
                        pnl = -stop_loss - self.spread_pips
                        break
                    if bar.low <= tp_price:
                        pnl = take_profit - self.spread_pips
                        break

            if pnl == 0:  # Nao fechou
                exit_bar = self.bars[min(entry_idx + 100, len(self.bars) - 1)]
                if signal.direction == 'BUY':
                    pnl = (exit_bar.close - entry_price) / self.pip_value - self.spread_pips
                else:
                    pnl = (entry_price - exit_bar.close) / self.pip_value - self.spread_pips

            trades_pnl.append(pnl)

        if not trades_pnl:
            return BacktestResult(
                stop_loss=stop_loss, take_profit=take_profit,
                total_trades=0, winning_trades=0, losing_trades=0,
                total_pnl_pips=0, win_rate=0, profit_factor=0,
                max_drawdown=0, score=0
            )

        # Metricas
        winners = [p for p in trades_pnl if p > 0]
        losers = [p for p in trades_pnl if p <= 0]
        total_pnl = sum(trades_pnl)

        win_rate = len(winners) / len(trades_pnl) if trades_pnl else 0
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)

        # Drawdown
        equity = np.cumsum([0] + trades_pnl)
        peak = np.maximum.accumulate(equity + 10000)
        dd = (peak - (equity + 10000)) / peak
        max_dd = np.max(dd)

        # Score
        score = 0
        if len(trades_pnl) >= 5 and pf > 1.0:
            score = (0.35 * min(pf, 5) / 5 +
                    0.30 * win_rate +
                    0.20 * max(0, 1 - max_dd * 5) +
                    0.15 * min(len(trades_pnl) / 50, 1))

        return BacktestResult(
            stop_loss=stop_loss,
            take_profit=take_profit,
            total_trades=len(trades_pnl),
            winning_trades=len(winners),
            losing_trades=len(losers),
            total_pnl_pips=total_pnl,
            win_rate=win_rate,
            profit_factor=pf,
            max_drawdown=max_dd,
            score=score
        )

    def optimize_sltp(self, verbose: bool = True) -> BacktestResult:
        """Otimiza SL/TP sobre sinais cacheados"""
        if not self.cached_signals:
            print("  ERRO: Nenhum sinal cacheado!")
            return None

        if verbose:
            print(f"\n  Otimizando SL/TP sobre {len(self.cached_signals)} sinais...")

        sl_range = [10, 15, 20, 25, 30, 35, 40, 50]
        tp_range = [20, 30, 40, 50, 60, 80, 100, 120]

        best_score = -1
        best_result = None

        for sl in sl_range:
            for tp in tp_range:
                if tp < sl:  # Risk/reward minimo
                    continue

                result = self.backtest_sltp(sl, tp)

                if result.score > best_score:
                    best_score = result.score
                    best_result = result

                    if verbose:
                        print(f"  [MELHOR] SL={sl} TP={tp} | "
                              f"Trades={result.total_trades} | "
                              f"WR={result.win_rate:.1%} | "
                              f"PF={result.profit_factor:.2f} | "
                              f"PnL={result.total_pnl_pips:.1f} pips")

        self.best_result = best_result
        return best_result

    def save_config(self, filename: str = None):
        """Salva configuracao otimizada"""
        if not self.best_result:
            return

        if filename is None:
            filename = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs",
                "prm_optimized.json"
            )

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        config = {
            "strategy": "PRM-RiemannMandelbrot",
            "symbol": self.symbol,
            "periodicity": self.periodicity,
            "optimized_at": datetime.now(timezone.utc).isoformat(),
            "data_period": {
                "start": self.bars[0].timestamp.isoformat() if self.bars else None,
                "end": self.bars[-1].timestamp.isoformat() if self.bars else None,
                "total_bars": len(self.bars)
            },
            "parameters": {
                "min_prices": self.detection_params.get('min_prices', 50),
                "stop_loss_pips": self.best_result.stop_loss,
                "take_profit_pips": self.best_result.take_profit,
                "hmm_threshold": self.detection_params.get('hmm_threshold', 0.80),
                "lyapunov_threshold": self.detection_params.get('lyapunov_threshold', 0.5),
                "curvature_threshold": self.detection_params.get('curvature_threshold', 0.10)
            },
            "performance": {
                "total_signals": len(self.cached_signals),
                "total_trades": self.best_result.total_trades,
                "winning_trades": self.best_result.winning_trades,
                "losing_trades": self.best_result.losing_trades,
                "win_rate": self.best_result.win_rate,
                "profit_factor": self.best_result.profit_factor,
                "total_pnl_pips": self.best_result.total_pnl_pips,
                "max_drawdown": self.best_result.max_drawdown,
                "score": self.best_result.score
            }
        }

        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n  Configuracao salva em: {filename}")

    def print_results(self):
        """Imprime resultados"""
        if not self.best_result:
            return

        r = self.best_result
        print("\n" + "=" * 60)
        print("  RESULTADO DA OTIMIZACAO PRM")
        print("=" * 60)
        print(f"\n  PARAMETROS:")
        print(f"    min_prices: {self.detection_params.get('min_prices', 50)}")
        print(f"    hmm_threshold: {self.detection_params.get('hmm_threshold', 0.80)}")
        print(f"    lyapunov_threshold: {self.detection_params.get('lyapunov_threshold', 0.5)}")
        print(f"    curvature_threshold: {self.detection_params.get('curvature_threshold', 0.10)}")
        print(f"    stop_loss_pips: {r.stop_loss}")
        print(f"    take_profit_pips: {r.take_profit}")
        print(f"\n  PERFORMANCE:")
        print(f"    Sinais: {len(self.cached_signals)}")
        print(f"    Trades: {r.total_trades}")
        print(f"    Win Rate: {r.win_rate:.1%}")
        print(f"    Profit Factor: {r.profit_factor:.2f}")
        print(f"    PnL Total: {r.total_pnl_pips:.1f} pips")
        print(f"    Max Drawdown: {r.max_drawdown:.1%}")
        print(f"    Score: {r.score:.4f}")
        print("=" * 60)


async def main():
    """Funcao principal"""
    print("=" * 60)
    print("  OTIMIZADOR PRM V2")
    print("  Dados REAIS do Mercado Forex")
    print("=" * 60)

    # Configuracao
    symbol = "EURUSD"
    periodicity = "M15"
    start_date = datetime(2025, 7, 1, tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)

    print(f"\n  Simbolo: {symbol}")
    print(f"  Periodicidade: {periodicity}")
    print(f"  Periodo: {start_date.date()} a {end_date.date()}")

    optimizer = PRMOptimizerV2(symbol=symbol, periodicity=periodicity)

    # Carrega dados
    await optimizer.load_data(start_date, end_date)

    if not optimizer.bars:
        print("  ERRO: Nenhum dado carregado!")
        return

    # Testa diferentes configuracoes de deteccao
    detection_configs = [
        {'min_prices': 30, 'hmm_threshold': 0.75, 'lyapunov_threshold': 0.4, 'curvature_threshold': 0.08},
        {'min_prices': 50, 'hmm_threshold': 0.80, 'lyapunov_threshold': 0.5, 'curvature_threshold': 0.10},
        {'min_prices': 75, 'hmm_threshold': 0.85, 'lyapunov_threshold': 0.5, 'curvature_threshold': 0.12},
    ]

    all_results = []

    for i, config in enumerate(detection_configs):
        print(f"\n{'='*60}")
        print(f"  TESTE {i+1}/{len(detection_configs)}")
        print(f"{'='*60}")

        optimizer.generate_signals(**config, verbose=True)

        if optimizer.cached_signals:
            result = optimizer.optimize_sltp(verbose=True)
            if result and result.score > 0:
                all_results.append((config, result))

    if all_results:
        # Encontra melhor
        best_config, best_result = max(all_results, key=lambda x: x[1].score)

        print("\n" + "=" * 60)
        print("  MELHOR CONFIGURACAO ENCONTRADA")
        print("=" * 60)

        # Regenera com melhor config para salvar
        optimizer.generate_signals(**best_config, verbose=False)
        optimizer.best_result = best_result
        optimizer.detection_params = best_config

        optimizer.print_results()
        optimizer.save_config()
    else:
        print("\n  Nenhuma configuracao lucrativa encontrada!")


if __name__ == "__main__":
    asyncio.run(main())
