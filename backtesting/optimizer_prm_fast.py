#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR PRM RAPIDO - Protocolo Riemann-Mandelbrot
Otimizacao em 2 Fases para Eficiencia
================================================================================

Este script otimiza os parametros do indicador PRM de forma eficiente:

FASE 1: Otimiza parametros do indicador (hmm, lyapunov, curvature)
        com SL/TP fixos para encontrar melhor configuracao de deteccao

FASE 2: Com os melhores parametros de deteccao, otimiza SL/TP

IMPORTANTE: Nenhuma simulacao - apenas dados reais do mercado FXOpen.
"""

import sys
import os
import json
import asyncio
import random
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.fxopen_historical_ws import Bar, download_historical_data
from strategies.alta_volatilidade import PRMStrategy
from strategies.base import SignalType


@dataclass
class Signal:
    """Sinal gerado pela estrategia"""
    timestamp: datetime
    price: float
    direction: str  # 'BUY' ou 'SELL'
    confidence: float
    bar_index: int


@dataclass
class TradeResult:
    """Resultado de um trade"""
    entry_bar: int
    entry_price: float
    exit_bar: int
    exit_price: float
    direction: str
    pnl_pips: float
    exit_reason: str


@dataclass
class OptimizationResult:
    """Resultado de uma combinacao de parametros"""
    # Parametros
    min_prices: int
    hmm_threshold: float
    lyapunov_threshold: float
    curvature_threshold: float
    stop_loss_pips: float
    take_profit_pips: float

    # Metricas
    total_signals: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_pips: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    avg_trade_pips: float = 0.0
    score: float = 0.0


class PRMFastOptimizer:
    """
    Otimizador rapido de parametros para o PRM

    Usa abordagem em 2 fases para maior eficiencia
    """

    def __init__(self,
                 symbol: str = "EURUSD",
                 periodicity: str = "M15",
                 pip_value: float = 0.0001,
                 spread_pips: float = 1.0):
        """
        Inicializa o otimizador
        """
        self.symbol = symbol
        self.periodicity = periodicity
        self.pip_value = pip_value
        self.spread_pips = spread_pips

        self.bars: List[Bar] = []
        self.best_result: Optional[OptimizationResult] = None
        self.all_results: List[OptimizationResult] = []

    async def load_data(self, start_date: datetime, end_date: datetime):
        """Carrega dados historicos REAIS da FXOpen"""
        print("\n" + "=" * 70)
        print("  CARREGANDO DADOS HISTORICOS REAIS")
        print("=" * 70)

        self.bars = await download_historical_data(
            symbol=self.symbol,
            periodicity=self.periodicity,
            start_time=start_date,
            end_time=end_date
        )

        print(f"\n  Total de barras carregadas: {len(self.bars)}")

    def _generate_signals(self,
                         min_prices: int,
                         hmm_threshold: float,
                         lyapunov_threshold: float,
                         curvature_threshold: float) -> List[Signal]:
        """
        Gera sinais com uma configuracao especifica do PRM
        """
        strategy = PRMStrategy(
            min_prices=min_prices,
            stop_loss_pips=20.0,  # Valores fixos, nao importa aqui
            take_profit_pips=40.0,
            hmm_threshold=hmm_threshold,
            lyapunov_threshold=lyapunov_threshold,
            curvature_threshold=curvature_threshold
        )

        signals = []

        for i, bar in enumerate(self.bars):
            signal = strategy.analyze(
                price=bar.close,
                timestamp=bar.timestamp,
                volume=bar.volume,
                high=bar.high,
                low=bar.low,
                open=bar.open
            )

            if signal and signal.type != SignalType.HOLD:
                signals.append(Signal(
                    timestamp=bar.timestamp,
                    price=bar.close,
                    direction='BUY' if signal.type == SignalType.BUY else 'SELL',
                    confidence=signal.confidence,
                    bar_index=i
                ))

        return signals

    def _backtest_signals(self,
                         signals: List[Signal],
                         stop_loss_pips: float,
                         take_profit_pips: float) -> List[TradeResult]:
        """
        Executa backtest de sinais com SL/TP especificos
        """
        trades = []

        for signal in signals:
            entry_bar = signal.bar_index
            entry_price = signal.price

            # Aplica spread na entrada
            if signal.direction == 'BUY':
                entry_price += self.spread_pips * self.pip_value / 2
            else:
                entry_price -= self.spread_pips * self.pip_value / 2

            # Calcula niveis de SL/TP
            if signal.direction == 'BUY':
                stop_price = entry_price - (stop_loss_pips * self.pip_value)
                take_price = entry_price + (take_profit_pips * self.pip_value)
            else:
                stop_price = entry_price + (stop_loss_pips * self.pip_value)
                take_price = entry_price - (take_profit_pips * self.pip_value)

            # Procura saida
            exit_bar = None
            exit_price = None
            exit_reason = None

            for i in range(entry_bar + 1, len(self.bars)):
                bar = self.bars[i]

                if signal.direction == 'BUY':
                    # Check stop loss
                    if bar.low <= stop_price:
                        exit_bar = i
                        exit_price = stop_price
                        exit_reason = 'stop_loss'
                        break
                    # Check take profit
                    if bar.high >= take_price:
                        exit_bar = i
                        exit_price = take_price
                        exit_reason = 'take_profit'
                        break
                else:  # SELL
                    if bar.high >= stop_price:
                        exit_bar = i
                        exit_price = stop_price
                        exit_reason = 'stop_loss'
                        break
                    if bar.low <= take_price:
                        exit_bar = i
                        exit_price = take_price
                        exit_reason = 'take_profit'
                        break

            # Se nao encontrou saida, usa ultimo bar
            if exit_bar is None:
                exit_bar = len(self.bars) - 1
                exit_price = self.bars[-1].close
                exit_reason = 'end_of_data'

            # Calcula PnL
            if signal.direction == 'BUY':
                pnl_pips = (exit_price - entry_price) / self.pip_value
            else:
                pnl_pips = (entry_price - exit_price) / self.pip_value

            trades.append(TradeResult(
                entry_bar=entry_bar,
                entry_price=entry_price,
                exit_bar=exit_bar,
                exit_price=exit_price,
                direction=signal.direction,
                pnl_pips=pnl_pips,
                exit_reason=exit_reason
            ))

        return trades

    def _calculate_metrics(self, trades: List[TradeResult]) -> Dict:
        """Calcula metricas de um conjunto de trades"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl_pips': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_pips': 0.0,
                'max_drawdown': 0.0
            }

        winners = [t for t in trades if t.pnl_pips > 0]
        losers = [t for t in trades if t.pnl_pips <= 0]

        total_pnl = sum(t.pnl_pips for t in trades)
        gross_profit = sum(t.pnl_pips for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl_pips for t in losers)) if losers else 0

        # Calcula drawdown
        equity = [0]
        for t in trades:
            equity.append(equity[-1] + t.pnl_pips)
        equity = np.array(equity)
        peak = np.maximum.accumulate(equity + 10000)  # Base 10000
        dd = (peak - (equity + 10000)) / peak
        max_dd = np.max(dd) if len(dd) > 0 else 0

        return {
            'total_trades': len(trades),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'total_pnl_pips': total_pnl,
            'win_rate': len(winners) / len(trades) if trades else 0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0),
            'avg_trade_pips': total_pnl / len(trades) if trades else 0,
            'max_drawdown': max_dd
        }

    def _calculate_score(self, metrics: Dict) -> float:
        """Calcula score para ranking"""
        if metrics['total_trades'] < 5:
            return 0
        if metrics['profit_factor'] <= 1.0:
            return 0

        pf_score = min(metrics['profit_factor'], 5.0) / 5.0
        wr_score = metrics['win_rate']
        dd_score = max(0, 1 - metrics['max_drawdown'] * 5)
        trades_score = min(metrics['total_trades'] / 50, 1.0)

        return 0.35 * pf_score + 0.30 * wr_score + 0.20 * dd_score + 0.15 * trades_score

    def optimize_phase1(self, verbose: bool = True) -> Tuple[int, float, float, float, List[Signal]]:
        """
        FASE 1: Otimiza parametros de deteccao do PRM

        Testa diferentes combinacoes de min_prices, hmm, lyapunov, curvature
        com SL/TP fixos para encontrar qual configuracao gera os melhores sinais.
        """
        print("\n" + "=" * 70)
        print("  FASE 1: OTIMIZACAO DE PARAMETROS DE DETECCAO")
        print("=" * 70)

        # Grid de parametros de deteccao
        param_grid = {
            'min_prices': [30, 50, 75, 100],
            'hmm_threshold': [0.70, 0.80, 0.85],
            'lyapunov_threshold': [0.3, 0.5, 0.7],
            'curvature_threshold': [0.05, 0.10, 0.15]
        }

        # SL/TP fixos para fase 1
        fixed_sl = 20.0
        fixed_tp = 40.0

        best_score = -1
        best_params = None
        best_signals = None
        total_tests = (len(param_grid['min_prices']) *
                      len(param_grid['hmm_threshold']) *
                      len(param_grid['lyapunov_threshold']) *
                      len(param_grid['curvature_threshold']))

        print(f"  Combinacoes a testar: {total_tests}")
        print(f"  SL/TP fixos: {fixed_sl}/{fixed_tp} pips")
        print("-" * 70)

        test_num = 0
        for mp in param_grid['min_prices']:
            for hmm in param_grid['hmm_threshold']:
                for lyap in param_grid['lyapunov_threshold']:
                    for curv in param_grid['curvature_threshold']:
                        test_num += 1

                        if verbose and test_num % 10 == 0:
                            print(f"  Testando {test_num}/{total_tests}...")

                        try:
                            # Gera sinais
                            signals = self._generate_signals(mp, hmm, lyap, curv)

                            if len(signals) < 3:
                                continue

                            # Backtest com SL/TP fixos
                            trades = self._backtest_signals(signals, fixed_sl, fixed_tp)
                            metrics = self._calculate_metrics(trades)
                            score = self._calculate_score(metrics)

                            if score > best_score:
                                best_score = score
                                best_params = (mp, hmm, lyap, curv)
                                best_signals = signals

                                if verbose:
                                    print(f"\n  [NOVO MELHOR] Test {test_num}")
                                    print(f"    Score: {score:.4f}")
                                    print(f"    Sinais: {len(signals)} | Trades: {metrics['total_trades']}")
                                    print(f"    WR: {metrics['win_rate']:.1%} | PF: {metrics['profit_factor']:.2f}")
                                    print(f"    Params: mp={mp}, hmm={hmm}, lyap={lyap}, curv={curv}")

                        except Exception as e:
                            continue

        if best_params:
            print(f"\n  Melhores parametros de deteccao:")
            print(f"    min_prices: {best_params[0]}")
            print(f"    hmm_threshold: {best_params[1]}")
            print(f"    lyapunov_threshold: {best_params[2]}")
            print(f"    curvature_threshold: {best_params[3]}")
            print(f"    Sinais gerados: {len(best_signals)}")

        return best_params[0], best_params[1], best_params[2], best_params[3], best_signals

    def optimize_phase2(self, signals: List[Signal], verbose: bool = True) -> OptimizationResult:
        """
        FASE 2: Otimiza SL/TP usando os melhores sinais
        """
        print("\n" + "=" * 70)
        print("  FASE 2: OTIMIZACAO DE STOP LOSS / TAKE PROFIT")
        print("=" * 70)

        # Grid de SL/TP
        sl_values = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        tp_values = [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0]

        print(f"  Sinais para testar: {len(signals)}")
        print(f"  Combinacoes SL/TP: {len(sl_values) * len(tp_values)}")
        print("-" * 70)

        best_score = -1
        best_sl = None
        best_tp = None
        best_metrics = None

        for sl in sl_values:
            for tp in tp_values:
                # Ignora combinacoes onde TP < SL (risk/reward ruim)
                if tp < sl:
                    continue

                trades = self._backtest_signals(signals, sl, tp)
                metrics = self._calculate_metrics(trades)
                score = self._calculate_score(metrics)

                if score > best_score:
                    best_score = score
                    best_sl = sl
                    best_tp = tp
                    best_metrics = metrics

                    if verbose:
                        print(f"  [MELHOR] SL={sl} TP={tp} | "
                              f"WR={metrics['win_rate']:.1%} | "
                              f"PF={metrics['profit_factor']:.2f} | "
                              f"PnL={metrics['total_pnl_pips']:.1f} pips | "
                              f"Score={score:.4f}")

        print(f"\n  Melhor SL/TP: {best_sl}/{best_tp} pips")

        return best_sl, best_tp, best_metrics, best_score

    def optimize(self, verbose: bool = True) -> OptimizationResult:
        """
        Executa otimizacao completa em 2 fases
        """
        if not self.bars:
            print("ERRO: Dados nao carregados!")
            return None

        print("\n" + "=" * 70)
        print("  OTIMIZADOR PRM RAPIDO - 2 FASES")
        print("=" * 70)
        print(f"\n  Total de barras: {len(self.bars)}")
        print(f"  Periodo: {self.bars[0].timestamp.date()} a {self.bars[-1].timestamp.date()}")

        # FASE 1: Otimiza parametros de deteccao
        mp, hmm, lyap, curv, signals = self.optimize_phase1(verbose)

        if not signals:
            print("\n  ERRO: Nenhum sinal gerado na Fase 1!")
            return None

        # FASE 2: Otimiza SL/TP
        best_sl, best_tp, best_metrics, best_score = self.optimize_phase2(signals, verbose)

        # Cria resultado final
        self.best_result = OptimizationResult(
            min_prices=mp,
            hmm_threshold=hmm,
            lyapunov_threshold=lyap,
            curvature_threshold=curv,
            stop_loss_pips=best_sl,
            take_profit_pips=best_tp,
            total_signals=len(signals),
            total_trades=best_metrics['total_trades'],
            winning_trades=best_metrics['winning_trades'],
            losing_trades=best_metrics['losing_trades'],
            total_pnl_pips=best_metrics['total_pnl_pips'],
            win_rate=best_metrics['win_rate'],
            profit_factor=best_metrics['profit_factor'],
            max_drawdown=best_metrics['max_drawdown'],
            avg_trade_pips=best_metrics['avg_trade_pips'],
            score=best_score
        )

        return self.best_result

    def save_config(self, filename: str = None):
        """Salva melhor configuracao"""
        if not self.best_result:
            print("ERRO: Nenhum resultado!")
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
                "start": self.bars[0].timestamp.isoformat(),
                "end": self.bars[-1].timestamp.isoformat(),
                "total_bars": len(self.bars)
            },
            "parameters": {
                "min_prices": self.best_result.min_prices,
                "stop_loss_pips": self.best_result.stop_loss_pips,
                "take_profit_pips": self.best_result.take_profit_pips,
                "hmm_threshold": self.best_result.hmm_threshold,
                "lyapunov_threshold": self.best_result.lyapunov_threshold,
                "curvature_threshold": self.best_result.curvature_threshold
            },
            "performance": {
                "total_signals": self.best_result.total_signals,
                "total_trades": self.best_result.total_trades,
                "winning_trades": self.best_result.winning_trades,
                "losing_trades": self.best_result.losing_trades,
                "win_rate": self.best_result.win_rate,
                "profit_factor": self.best_result.profit_factor,
                "total_pnl_pips": self.best_result.total_pnl_pips,
                "max_drawdown": self.best_result.max_drawdown,
                "avg_trade_pips": self.best_result.avg_trade_pips,
                "score": self.best_result.score
            }
        }

        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n  Configuracao salva em: {filename}")
        return filename

    def print_results(self):
        """Imprime resultados da otimizacao"""
        if not self.best_result:
            return

        r = self.best_result

        print("\n" + "=" * 70)
        print("  RESULTADO FINAL DA OTIMIZACAO PRM")
        print("=" * 70)

        print(f"\n  PARAMETROS OTIMIZADOS:")
        print(f"    min_prices: {r.min_prices}")
        print(f"    hmm_threshold: {r.hmm_threshold}")
        print(f"    lyapunov_threshold: {r.lyapunov_threshold}")
        print(f"    curvature_threshold: {r.curvature_threshold}")
        print(f"    stop_loss_pips: {r.stop_loss_pips}")
        print(f"    take_profit_pips: {r.take_profit_pips}")

        print(f"\n  PERFORMANCE:")
        print(f"    Total Sinais: {r.total_signals}")
        print(f"    Total Trades: {r.total_trades}")
        print(f"    Trades Vencedores: {r.winning_trades}")
        print(f"    Trades Perdedores: {r.losing_trades}")
        print(f"    Win Rate: {r.win_rate:.1%}")
        print(f"    Profit Factor: {r.profit_factor:.2f}")
        print(f"    PnL Total: {r.total_pnl_pips:.1f} pips")
        print(f"    Media por Trade: {r.avg_trade_pips:.1f} pips")
        print(f"    Max Drawdown: {r.max_drawdown:.1%}")
        print(f"    Score: {r.score:.4f}")

        print("\n" + "=" * 70)


async def run_fast_optimization(
    symbol: str = "EURUSD",
    periodicity: str = "M15",
    start_date: datetime = None,
    end_date: datetime = None
):
    """Funcao principal"""
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    if start_date is None:
        start_date = datetime(2025, 7, 1, tzinfo=timezone.utc)

    print("=" * 70)
    print("  OTIMIZADOR PRM RAPIDO")
    print("  Dados Historicos REAIS do Mercado Forex")
    print("=" * 70)
    print(f"\n  Simbolo: {symbol}")
    print(f"  Periodicidade: {periodicity}")
    print(f"  Periodo: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")

    optimizer = PRMFastOptimizer(symbol=symbol, periodicity=periodicity)

    await optimizer.load_data(start_date, end_date)

    if not optimizer.bars:
        print("\nERRO: Nao foi possivel carregar dados!")
        return None

    result = optimizer.optimize(verbose=True)

    if result:
        optimizer.print_results()
        optimizer.save_config()

    return result


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Otimizador PRM Rapido")
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--periodicity", type=str, default="M15")
    parser.add_argument("--start", type=str, default="2025-07-01")

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)

    asyncio.run(run_fast_optimization(
        symbol=args.symbol,
        periodicity=args.periodicity,
        start_date=start_date,
        end_date=end_date
    ))


if __name__ == "__main__":
    main()
