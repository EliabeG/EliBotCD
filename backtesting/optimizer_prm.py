#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR PRM - Protocolo Riemann-Mandelbrot
Grid Search para Encontrar Melhores Parametros
================================================================================

Este script otimiza os parametros do indicador PRM usando dados historicos REAIS.

IMPORTANTE: Nenhuma simulacao - apenas dados reais do mercado FXOpen.

Parametros otimizados:
- min_prices: Minimo de precos para analise
- stop_loss_pips: Stop loss em pips
- take_profit_pips: Take profit em pips
- hmm_threshold: Threshold para ativacao do HMM
- lyapunov_threshold: Threshold K para Lyapunov
- curvature_threshold: Threshold para aceleracao da curvatura

Uso:
    python -m backtesting.optimizer_prm
"""

import sys
import os
import json
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from itertools import product
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.fxopen_historical_ws import Bar, download_historical_data, FXOpenHistoricalClient
from strategies.alta_volatilidade import PRMStrategy
from strategies.base import SignalType


@dataclass
class OptimizationResult:
    """Resultado de uma combinacao de parametros"""
    # Parametros testados
    min_prices: int
    stop_loss_pips: float
    take_profit_pips: float
    hmm_threshold: float
    lyapunov_threshold: float
    curvature_threshold: float

    # Metricas
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_pips: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_trade: float = 0.0

    # Score combinado
    score: float = 0.0


@dataclass
class Trade:
    """Representa um trade"""
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    direction: str
    pnl_pips: float
    exit_reason: str


class PRMOptimizer:
    """
    Otimizador de parametros para o indicador PRM

    Usa Grid Search para encontrar a melhor combinacao de parametros
    """

    # Parametros a otimizar e seus ranges (reduzido para eficiencia)
    PARAM_GRID = {
        'min_prices': [30, 50, 75, 100],
        'stop_loss_pips': [15.0, 20.0, 25.0, 30.0],
        'take_profit_pips': [30.0, 40.0, 50.0, 60.0],
        'hmm_threshold': [0.70, 0.80, 0.85, 0.90],
        'lyapunov_threshold': [0.3, 0.5, 0.7],
        'curvature_threshold': [0.05, 0.10, 0.15]
    }

    def __init__(self,
                 symbol: str = "EURUSD",
                 periodicity: str = "M15",
                 pip_value: float = 0.0001,
                 spread_pips: float = 1.0,
                 slippage_pips: float = 0.5):
        """
        Inicializa o otimizador

        Args:
            symbol: Par de moedas
            periodicity: Periodicidade (M15, H1, etc)
            pip_value: Valor de 1 pip
            spread_pips: Spread em pips
            slippage_pips: Slippage em pips
        """
        self.symbol = symbol
        self.periodicity = periodicity
        self.pip_value = pip_value
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips

        self.bars: List[Bar] = []
        self.results: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None

    async def load_data(self, start_date: datetime, end_date: datetime):
        """
        Carrega dados historicos REAIS da FXOpen

        Args:
            start_date: Data inicial
            end_date: Data final
        """
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
        if self.bars:
            print(f"  Periodo: {self.bars[0].timestamp} a {self.bars[-1].timestamp}")

    def _run_backtest(self,
                     min_prices: int,
                     stop_loss_pips: float,
                     take_profit_pips: float,
                     hmm_threshold: float,
                     lyapunov_threshold: float,
                     curvature_threshold: float) -> OptimizationResult:
        """
        Executa backtest com parametros especificos
        """
        result = OptimizationResult(
            min_prices=min_prices,
            stop_loss_pips=stop_loss_pips,
            take_profit_pips=take_profit_pips,
            hmm_threshold=hmm_threshold,
            lyapunov_threshold=lyapunov_threshold,
            curvature_threshold=curvature_threshold
        )

        # Cria estrategia com parametros
        strategy = PRMStrategy(
            min_prices=min_prices,
            stop_loss_pips=stop_loss_pips,
            take_profit_pips=take_profit_pips,
            hmm_threshold=hmm_threshold,
            lyapunov_threshold=lyapunov_threshold,
            curvature_threshold=curvature_threshold
        )

        # Estado do backtest
        trades: List[Trade] = []
        current_position = None
        equity_curve = [10000.0]  # Capital inicial
        capital = 10000.0

        # Processa cada barra
        for bar in self.bars:
            # Verifica stop/take da posicao atual
            if current_position:
                should_close, exit_price, exit_reason = self._check_exit(
                    current_position, bar, stop_loss_pips, take_profit_pips
                )

                if should_close:
                    # Calcula PnL
                    if current_position['direction'] == 'LONG':
                        pnl_pips = (exit_price - current_position['entry_price']) / self.pip_value
                    else:
                        pnl_pips = (current_position['entry_price'] - exit_price) / self.pip_value

                    pnl_pips -= self.spread_pips + self.slippage_pips

                    # Registra trade
                    trade = Trade(
                        entry_time=current_position['entry_time'],
                        entry_price=current_position['entry_price'],
                        exit_time=bar.timestamp,
                        exit_price=exit_price,
                        direction=current_position['direction'],
                        pnl_pips=pnl_pips,
                        exit_reason=exit_reason
                    )
                    trades.append(trade)

                    # Atualiza capital
                    pnl_usd = pnl_pips * 0.01 * 10  # 0.01 lote, $10/pip
                    capital += pnl_usd

                    current_position = None

            # Gera sinal
            signal = strategy.analyze(
                price=bar.close,
                timestamp=bar.timestamp,
                volume=bar.volume,
                high=bar.high,
                low=bar.low,
                open=bar.open
            )

            # Abre posicao se nao tem
            if signal and signal.type != SignalType.HOLD and current_position is None:
                current_position = {
                    'entry_time': bar.timestamp,
                    'entry_price': bar.close,
                    'direction': 'LONG' if signal.type == SignalType.BUY else 'SHORT',
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit
                }

            # Atualiza equity curve
            equity_curve.append(capital)

        # Fecha posicao aberta no final
        if current_position:
            last_bar = self.bars[-1]
            if current_position['direction'] == 'LONG':
                pnl_pips = (last_bar.close - current_position['entry_price']) / self.pip_value
            else:
                pnl_pips = (current_position['entry_price'] - last_bar.close) / self.pip_value

            pnl_pips -= self.spread_pips + self.slippage_pips

            trade = Trade(
                entry_time=current_position['entry_time'],
                entry_price=current_position['entry_price'],
                exit_time=last_bar.timestamp,
                exit_price=last_bar.close,
                direction=current_position['direction'],
                pnl_pips=pnl_pips,
                exit_reason='end_of_data'
            )
            trades.append(trade)

        # Calcula metricas
        result = self._calculate_metrics(result, trades, equity_curve)

        return result

    def _check_exit(self, position: dict, bar: Bar,
                   stop_loss_pips: float, take_profit_pips: float) -> Tuple[bool, float, str]:
        """Verifica se deve fechar posicao"""
        entry = position['entry_price']

        if position['direction'] == 'LONG':
            stop_price = entry - (stop_loss_pips * self.pip_value)
            take_price = entry + (take_profit_pips * self.pip_value)

            if bar.low <= stop_price:
                return True, stop_price, 'stop_loss'
            if bar.high >= take_price:
                return True, take_price, 'take_profit'
        else:
            stop_price = entry + (stop_loss_pips * self.pip_value)
            take_price = entry - (take_profit_pips * self.pip_value)

            if bar.high >= stop_price:
                return True, stop_price, 'stop_loss'
            if bar.low <= take_price:
                return True, take_price, 'take_profit'

        return False, 0.0, ''

    def _calculate_metrics(self, result: OptimizationResult,
                          trades: List[Trade],
                          equity_curve: List[float]) -> OptimizationResult:
        """Calcula todas as metricas"""
        result.total_trades = len(trades)

        if not trades:
            return result

        # Trades vencedores/perdedores
        winners = [t for t in trades if t.pnl_pips > 0]
        losers = [t for t in trades if t.pnl_pips <= 0]

        result.winning_trades = len(winners)
        result.losing_trades = len(losers)
        result.total_pnl_pips = sum(t.pnl_pips for t in trades)
        result.total_pnl = result.total_pnl_pips * 0.01 * 10  # USD

        # Win rate
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0

        # Profit factor
        gross_profit = sum(t.pnl_pips for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl_pips for t in losers)) if losers else 0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)

        # Media por trade
        result.avg_trade = result.total_pnl_pips / result.total_trades if result.total_trades > 0 else 0

        # Drawdown
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        result.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Sharpe Ratio simplificado
        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            result.sharpe_ratio = np.sqrt(252 * 24 * 4) * np.mean(returns) / np.std(returns)  # M15

        # Score combinado para ranking
        result.score = self._calculate_score(result)

        return result

    def _calculate_score(self, result: OptimizationResult) -> float:
        """
        Calcula score combinado para ranking

        Prioriza:
        1. Profit Factor > 1.5
        2. Win Rate > 50%
        3. Minimo de trades
        4. Baixo drawdown
        5. Sharpe positivo
        """
        if result.total_trades < 10:
            return 0  # Precisa de minimo de trades

        if result.profit_factor <= 1.0:
            return 0  # Deve ser lucrativo

        # Componentes do score
        pf_score = min(result.profit_factor, 5.0) / 5.0  # Normaliza PF (max 5)
        wr_score = result.win_rate
        dd_score = max(0, 1 - result.max_drawdown * 5)  # Penaliza drawdown
        sharpe_score = min(max(result.sharpe_ratio, 0), 3) / 3  # Normaliza Sharpe (max 3)
        trades_score = min(result.total_trades / 100, 1.0)  # Bonus para mais trades

        # Score ponderado
        score = (
            0.30 * pf_score +
            0.25 * wr_score +
            0.20 * dd_score +
            0.15 * sharpe_score +
            0.10 * trades_score
        )

        return score

    def optimize(self,
                max_combinations: int = None,
                min_trades: int = 10,
                verbose: bool = True) -> OptimizationResult:
        """
        Executa otimizacao Grid Search

        Args:
            max_combinations: Limite de combinacoes (None = todas)
            min_trades: Minimo de trades para considerar valido
            verbose: Mostrar progresso

        Returns:
            Melhor resultado encontrado
        """
        if not self.bars:
            print("ERRO: Dados nao carregados!")
            return None

        print("\n" + "=" * 70)
        print("  OTIMIZACAO PRM - GRID SEARCH")
        print("=" * 70)

        # Gera todas as combinacoes
        param_names = list(self.PARAM_GRID.keys())
        param_values = list(self.PARAM_GRID.values())
        all_combinations = list(product(*param_values))

        if max_combinations:
            all_combinations = all_combinations[:max_combinations]

        total_combinations = len(all_combinations)
        print(f"\n  Total de combinacoes: {total_combinations}")
        print(f"  Barras por teste: {len(self.bars)}")
        print("-" * 70)

        self.results = []
        best_score = -1

        for i, combo in enumerate(all_combinations):
            params = dict(zip(param_names, combo))

            # Executa backtest
            try:
                result = self._run_backtest(**params)
                self.results.append(result)

                # Verifica se e melhor
                if result.score > best_score and result.total_trades >= min_trades:
                    best_score = result.score
                    self.best_result = result

                    if verbose:
                        print(f"\n  [NOVO MELHOR] Combinacao {i+1}/{total_combinations}")
                        print(f"    Score: {result.score:.4f}")
                        print(f"    Trades: {result.total_trades} | WR: {result.win_rate:.1%}")
                        print(f"    PF: {result.profit_factor:.2f} | PnL: {result.total_pnl_pips:.1f} pips")
                        print(f"    Params: min_prices={params['min_prices']}, "
                              f"SL={params['stop_loss_pips']}, TP={params['take_profit_pips']}")

                # Progress
                if verbose and (i + 1) % 100 == 0:
                    print(f"  Testado: {i+1}/{total_combinations} | Melhor Score: {best_score:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Erro na combinacao {i+1}: {e}")
                continue

        print("\n" + "=" * 70)
        print("  OTIMIZACAO CONCLUIDA")
        print("=" * 70)

        return self.best_result

    def save_best_config(self, filename: str = None):
        """
        Salva a melhor configuracao em arquivo JSON

        Args:
            filename: Nome do arquivo (default: configs/prm_optimized.json)
        """
        if not self.best_result:
            print("ERRO: Nenhum resultado disponivel!")
            return

        if filename is None:
            filename = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs",
                "prm_optimized.json"
            )

        # Cria diretorio se nao existe
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
                "min_prices": self.best_result.min_prices,
                "stop_loss_pips": self.best_result.stop_loss_pips,
                "take_profit_pips": self.best_result.take_profit_pips,
                "hmm_threshold": self.best_result.hmm_threshold,
                "lyapunov_threshold": self.best_result.lyapunov_threshold,
                "curvature_threshold": self.best_result.curvature_threshold
            },
            "performance": {
                "total_trades": self.best_result.total_trades,
                "winning_trades": self.best_result.winning_trades,
                "losing_trades": self.best_result.losing_trades,
                "win_rate": self.best_result.win_rate,
                "profit_factor": self.best_result.profit_factor,
                "total_pnl_pips": self.best_result.total_pnl_pips,
                "total_pnl_usd": self.best_result.total_pnl,
                "max_drawdown": self.best_result.max_drawdown,
                "sharpe_ratio": self.best_result.sharpe_ratio,
                "avg_trade_pips": self.best_result.avg_trade,
                "score": self.best_result.score
            }
        }

        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n  Configuracao salva em: {filename}")

        return filename

    def save_all_results(self, filename: str = None):
        """Salva todos os resultados da otimizacao"""
        if not self.results:
            print("ERRO: Nenhum resultado disponivel!")
            return

        if filename is None:
            filename = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs",
                "prm_optimization_results.json"
            )

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Ordena por score
        sorted_results = sorted(self.results, key=lambda x: x.score, reverse=True)

        data = {
            "symbol": self.symbol,
            "periodicity": self.periodicity,
            "optimized_at": datetime.now(timezone.utc).isoformat(),
            "total_combinations": len(self.results),
            "results": [asdict(r) for r in sorted_results[:50]]  # Top 50
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"  Resultados salvos em: {filename}")

        return filename

    def print_top_results(self, n: int = 10):
        """Imprime os N melhores resultados"""
        if not self.results:
            print("Nenhum resultado disponivel!")
            return

        sorted_results = sorted(self.results, key=lambda x: x.score, reverse=True)

        print("\n" + "=" * 70)
        print(f"  TOP {n} MELHORES CONFIGURACOES")
        print("=" * 70)

        for i, r in enumerate(sorted_results[:n], 1):
            print(f"\n  #{i} - Score: {r.score:.4f}")
            print(f"      Trades: {r.total_trades} | WR: {r.win_rate:.1%} | PF: {r.profit_factor:.2f}")
            print(f"      PnL: {r.total_pnl_pips:.1f} pips (${r.total_pnl:.2f})")
            print(f"      DD: {r.max_drawdown:.1%} | Sharpe: {r.sharpe_ratio:.2f}")
            print(f"      Params: min_prices={r.min_prices}, SL={r.stop_loss_pips}, TP={r.take_profit_pips}")
            print(f"              HMM={r.hmm_threshold}, Lyap={r.lyapunov_threshold}, Curv={r.curvature_threshold}")


async def run_optimization(
    symbol: str = "EURUSD",
    periodicity: str = "M15",
    start_date: datetime = None,
    end_date: datetime = None,
    max_combinations: int = None,
    save_config: bool = True
):
    """
    Funcao principal para executar otimizacao

    Args:
        symbol: Par de moedas
        periodicity: Periodicidade
        start_date: Data inicial
        end_date: Data final
        max_combinations: Limite de combinacoes
        save_config: Se deve salvar configuracao
    """
    # Datas padrao
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    if start_date is None:
        start_date = datetime(2025, 7, 1, tzinfo=timezone.utc)

    print("=" * 70)
    print("  OTIMIZADOR PRM - PROTOCOLO RIEMANN-MANDELBROT")
    print("  Dados Historicos REAIS do Mercado Forex")
    print("=" * 70)
    print(f"\n  Simbolo: {symbol}")
    print(f"  Periodicidade: {periodicity}")
    print(f"  Periodo: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")

    # Cria otimizador
    optimizer = PRMOptimizer(
        symbol=symbol,
        periodicity=periodicity
    )

    # Carrega dados
    await optimizer.load_data(start_date, end_date)

    if not optimizer.bars:
        print("\nERRO: Nao foi possivel carregar dados!")
        return None

    # Executa otimizacao
    best = optimizer.optimize(max_combinations=max_combinations)

    if best:
        # Mostra melhores resultados
        optimizer.print_top_results(10)

        # Salva configuracao
        if save_config:
            optimizer.save_best_config()
            optimizer.save_all_results()

        print("\n" + "=" * 70)
        print("  MELHOR CONFIGURACAO ENCONTRADA")
        print("=" * 70)
        print(f"\n  PARAMETROS:")
        print(f"    min_prices: {best.min_prices}")
        print(f"    stop_loss_pips: {best.stop_loss_pips}")
        print(f"    take_profit_pips: {best.take_profit_pips}")
        print(f"    hmm_threshold: {best.hmm_threshold}")
        print(f"    lyapunov_threshold: {best.lyapunov_threshold}")
        print(f"    curvature_threshold: {best.curvature_threshold}")
        print(f"\n  PERFORMANCE:")
        print(f"    Total Trades: {best.total_trades}")
        print(f"    Win Rate: {best.win_rate:.1%}")
        print(f"    Profit Factor: {best.profit_factor:.2f}")
        print(f"    PnL Total: {best.total_pnl_pips:.1f} pips (${best.total_pnl:.2f})")
        print(f"    Max Drawdown: {best.max_drawdown:.1%}")
        print(f"    Sharpe Ratio: {best.sharpe_ratio:.2f}")
        print(f"    Score: {best.score:.4f}")
    else:
        print("\n  Nenhuma configuracao lucrativa encontrada!")

    return best


def main():
    """Funcao principal"""
    import argparse

    parser = argparse.ArgumentParser(description="Otimizador PRM")
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--periodicity", type=str, default="M15")
    parser.add_argument("--start", type=str, default="2025-07-01")
    parser.add_argument("--max-combinations", type=int, default=None)

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)

    asyncio.run(run_optimization(
        symbol=args.symbol,
        periodicity=args.periodicity,
        start_date=start_date,
        end_date=end_date,
        max_combinations=args.max_combinations
    ))


if __name__ == "__main__":
    main()
