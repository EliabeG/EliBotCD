#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR PRM MASSIVO - 2 MILHOES DE COMBINACOES
================================================================================

Testa 2.000.000 de combinacoes de parametros usando:
- Amostragem aleatoria eficiente
- Deteccao relaxada (HMM + Lyapunov, sem curvatura)
- Otimizacao paralela de SL/TP

IMPORTANTE: Dados REAIS do mercado FXOpen.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.fxopen_historical_ws import Bar, download_historical_data
from strategies.alta_volatilidade.prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot


@dataclass
class TestResult:
    """Resultado de um teste"""
    # Parametros de deteccao
    min_prices: int
    hmm_threshold: float
    lyapunov_threshold: float
    hmm_states_allowed: List[int]

    # Parametros de trade
    stop_loss_pips: float
    take_profit_pips: float

    # Resultados
    total_signals: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl_pips: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    score: float


class MassiveOptimizer:
    """
    Otimizador massivo com 2 milhoes de combinacoes
    """

    # Ranges de parametros expandidos
    PARAM_RANGES = {
        'min_prices': (20, 150),          # Range continuo
        'hmm_threshold': (0.50, 0.99),    # Range continuo
        'lyapunov_threshold': (0.01, 0.20), # Range continuo
        'stop_loss_pips': (5, 60),        # Range continuo
        'take_profit_pips': (10, 150),    # Range continuo
    }

    def __init__(self, symbol: str = "EURUSD", periodicity: str = "H1"):
        self.symbol = symbol
        self.periodicity = periodicity
        self.pip_value = 0.0001
        self.spread_pips = 1.0

        self.bars: List[Bar] = []
        self.prices: List[float] = []
        self.volumes: List[float] = []

        self.best_result: Optional[TestResult] = None
        self.all_results: List[TestResult] = []
        self.tested_count = 0

    async def load_data(self, start_date: datetime, end_date: datetime):
        """Carrega dados REAIS"""
        print("\n  Carregando dados REAIS da FXOpen...")
        self.bars = await download_historical_data(
            symbol=self.symbol,
            periodicity=self.periodicity,
            start_time=start_date,
            end_time=end_date
        )

        if self.bars:
            self.prices = [b.close for b in self.bars]
            self.volumes = [b.volume for b in self.bars]

        print(f"  Barras carregadas: {len(self.bars)}")

    def _generate_random_params(self) -> Dict:
        """Gera parametros aleatorios"""
        return {
            'min_prices': random.randint(self.PARAM_RANGES['min_prices'][0],
                                         self.PARAM_RANGES['min_prices'][1]),
            'hmm_threshold': round(random.uniform(self.PARAM_RANGES['hmm_threshold'][0],
                                                  self.PARAM_RANGES['hmm_threshold'][1]), 3),
            'lyapunov_threshold': round(random.uniform(self.PARAM_RANGES['lyapunov_threshold'][0],
                                                       self.PARAM_RANGES['lyapunov_threshold'][1]), 4),
            'stop_loss_pips': round(random.uniform(self.PARAM_RANGES['stop_loss_pips'][0],
                                                   self.PARAM_RANGES['stop_loss_pips'][1]), 1),
            'take_profit_pips': round(random.uniform(self.PARAM_RANGES['take_profit_pips'][0],
                                                     self.PARAM_RANGES['take_profit_pips'][1]), 1),
            'hmm_states': random.choice([[1], [1, 2], [0, 1], [0, 1, 2]])
        }

    def _run_fast_backtest(self, params: Dict) -> Optional[TestResult]:
        """
        Executa backtest rapido com parametros especificos

        Usa deteccao relaxada: HMM + Lyapunov (ignora curvatura que e' sempre 0)
        """
        min_prices = params['min_prices']
        hmm_threshold = params['hmm_threshold']
        lyapunov_threshold = params['lyapunov_threshold']
        stop_loss = params['stop_loss_pips']
        take_profit = params['take_profit_pips']
        hmm_states = params['hmm_states']

        # Ignora se TP < SL (risk/reward ruim)
        if take_profit < stop_loss:
            return None

        # Cria PRM com parametros
        prm = ProtocoloRiemannMandelbrot(
            n_states=3,
            hmm_threshold=hmm_threshold,
            lyapunov_threshold_k=lyapunov_threshold,
            curvature_threshold=0.001,  # Muito baixo para ignorar
            lookback_window=100
        )

        # Gera sinais com deteccao relaxada
        signals = []  # (bar_idx, direction, price)
        prices_buffer = deque(maxlen=500)
        volumes_buffer = deque(maxlen=500)

        for i, bar in enumerate(self.bars):
            prices_buffer.append(bar.close)
            volumes_buffer.append(bar.volume)

            if len(prices_buffer) < min_prices:
                continue

            try:
                prices_arr = np.array(prices_buffer)
                volumes_arr = np.array(volumes_buffer)

                result = prm.analyze(prices_arr, volumes_arr)

                # Deteccao relaxada: HMM prob + Lyapunov + Estado correto
                hmm_prob = result['Prob_HMM']
                lyapunov = result['Lyapunov_Score']
                hmm_state = result['hmm_analysis']['current_state']

                if (hmm_prob >= hmm_threshold and
                    lyapunov >= lyapunov_threshold and
                    hmm_state in hmm_states):

                    # Determina direcao baseado no estado
                    if hmm_state == 1:  # Alta volatilidade direcional
                        # Usa tendencia recente
                        if len(prices_buffer) >= 10:
                            trend = prices_buffer[-1] - prices_buffer[-10]
                            direction = 'BUY' if trend > 0 else 'SELL'
                        else:
                            direction = 'BUY'
                    else:
                        direction = 'BUY'  # Default

                    signals.append((i, direction, bar.close))

            except:
                continue

        if len(signals) < 3:
            return None

        # Executa trades sobre os sinais
        trades_pnl = []

        for sig_idx, (bar_idx, direction, entry_price) in enumerate(signals):
            # Calcula niveis
            if direction == 'BUY':
                sl_price = entry_price - (stop_loss * self.pip_value)
                tp_price = entry_price + (take_profit * self.pip_value)
            else:
                sl_price = entry_price + (stop_loss * self.pip_value)
                tp_price = entry_price - (take_profit * self.pip_value)

            # Busca saida
            pnl = 0.0
            for i in range(bar_idx + 1, min(bar_idx + 500, len(self.bars))):
                bar = self.bars[i]

                if direction == 'BUY':
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

            if pnl == 0:  # Nao fechou, usa preco atual
                exit_price = self.bars[min(bar_idx + 100, len(self.bars) - 1)].close
                if direction == 'BUY':
                    pnl = (exit_price - entry_price) / self.pip_value - self.spread_pips
                else:
                    pnl = (entry_price - exit_price) / self.pip_value - self.spread_pips

            trades_pnl.append(pnl)

        if not trades_pnl:
            return None

        # Calcula metricas
        winners = [p for p in trades_pnl if p > 0]
        losers = [p for p in trades_pnl if p <= 0]
        total_pnl = sum(trades_pnl)

        win_rate = len(winners) / len(trades_pnl) if trades_pnl else 0
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0.001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Drawdown
        equity = np.cumsum([0] + trades_pnl)
        peak = np.maximum.accumulate(equity + 10000)
        dd = (peak - (equity + 10000)) / peak
        max_dd = np.max(dd) if len(dd) > 0 else 0

        # Sharpe
        if len(trades_pnl) > 1 and np.std(trades_pnl) > 0:
            sharpe = np.mean(trades_pnl) / np.std(trades_pnl) * np.sqrt(252)
        else:
            sharpe = 0

        # Score
        score = 0
        if len(trades_pnl) >= 5 and profit_factor > 1.0:
            score = (
                0.30 * min(profit_factor, 5) / 5 +
                0.25 * win_rate +
                0.20 * max(0, 1 - max_dd * 5) +
                0.15 * min(max(sharpe, 0), 3) / 3 +
                0.10 * min(len(trades_pnl) / 100, 1)
            )

        return TestResult(
            min_prices=min_prices,
            hmm_threshold=hmm_threshold,
            lyapunov_threshold=lyapunov_threshold,
            hmm_states_allowed=hmm_states,
            stop_loss_pips=stop_loss,
            take_profit_pips=take_profit,
            total_signals=len(signals),
            total_trades=len(trades_pnl),
            winning_trades=len(winners),
            losing_trades=len(losers),
            total_pnl_pips=total_pnl,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            score=score
        )

    def optimize(self, n_combinations: int = 2000000, verbose: bool = True):
        """
        Executa otimizacao massiva com N combinacoes
        """
        if not self.bars:
            print("  ERRO: Dados nao carregados!")
            return None

        print(f"\n{'='*70}")
        print(f"  OTIMIZACAO MASSIVA - {n_combinations:,} COMBINACOES")
        print(f"{'='*70}")
        print(f"  Barras: {len(self.bars)}")
        print(f"  Periodo: {self.bars[0].timestamp.date()} a {self.bars[-1].timestamp.date()}")
        print(f"{'-'*70}")

        best_score = -1
        self.tested_count = 0
        valid_count = 0
        profitable_count = 0

        start_time = datetime.now()
        last_print = 0

        for i in range(n_combinations):
            self.tested_count += 1

            # Gera parametros aleatorios
            params = self._generate_random_params()

            # Executa backtest
            result = self._run_fast_backtest(params)

            if result:
                valid_count += 1

                if result.profit_factor > 1.0:
                    profitable_count += 1
                    self.all_results.append(result)

                    if result.score > best_score:
                        best_score = result.score
                        self.best_result = result

                        if verbose:
                            print(f"\n  [NOVO MELHOR #{valid_count}] Score: {result.score:.4f}")
                            print(f"    Trades: {result.total_trades} | WR: {result.win_rate:.1%} | PF: {result.profit_factor:.2f}")
                            print(f"    PnL: {result.total_pnl_pips:.1f} pips | DD: {result.max_drawdown:.1%}")
                            print(f"    Params: mp={result.min_prices}, hmm={result.hmm_threshold:.3f}, "
                                  f"lyap={result.lyapunov_threshold:.4f}")
                            print(f"    SL={result.stop_loss_pips:.1f}, TP={result.take_profit_pips:.1f}, "
                                  f"states={result.hmm_states_allowed}")

            # Progress
            if verbose and (i + 1) % 10000 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = (i + 1) / elapsed
                remaining = (n_combinations - i - 1) / rate / 60
                print(f"  Testado: {i+1:,}/{n_combinations:,} ({(i+1)/n_combinations*100:.1f}%) | "
                      f"Validos: {valid_count:,} | Lucrativos: {profitable_count:,} | "
                      f"ETA: {remaining:.0f}min")

        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"\n{'='*70}")
        print(f"  OTIMIZACAO CONCLUIDA")
        print(f"{'='*70}")
        print(f"  Tempo total: {elapsed/60:.1f} minutos")
        print(f"  Combinacoes testadas: {self.tested_count:,}")
        print(f"  Combinacoes validas: {valid_count:,}")
        print(f"  Combinacoes lucrativas: {profitable_count:,}")
        print(f"  Taxa de sucesso: {profitable_count/max(valid_count,1)*100:.1f}%")

        return self.best_result

    def save_config(self, filename: str = None):
        """Salva melhor configuracao"""
        if not self.best_result:
            print("  ERRO: Nenhum resultado!")
            return

        if filename is None:
            filename = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs",
                "prm_optimized.json"
            )

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        r = self.best_result
        config = {
            "strategy": "PRM-RiemannMandelbrot",
            "symbol": self.symbol,
            "periodicity": self.periodicity,
            "optimized_at": datetime.now(timezone.utc).isoformat(),
            "optimization_info": {
                "combinations_tested": self.tested_count,
                "profitable_found": len(self.all_results),
                "method": "random_search_2M"
            },
            "data_period": {
                "start": self.bars[0].timestamp.isoformat(),
                "end": self.bars[-1].timestamp.isoformat(),
                "total_bars": len(self.bars)
            },
            "parameters": {
                "min_prices": r.min_prices,
                "stop_loss_pips": r.stop_loss_pips,
                "take_profit_pips": r.take_profit_pips,
                "hmm_threshold": r.hmm_threshold,
                "lyapunov_threshold": r.lyapunov_threshold,
                "curvature_threshold": 0.01,
                "hmm_states_allowed": r.hmm_states_allowed
            },
            "performance": {
                "total_signals": r.total_signals,
                "total_trades": r.total_trades,
                "winning_trades": r.winning_trades,
                "losing_trades": r.losing_trades,
                "win_rate": round(r.win_rate, 4),
                "profit_factor": round(r.profit_factor, 4),
                "total_pnl_pips": round(r.total_pnl_pips, 2),
                "max_drawdown": round(r.max_drawdown, 4),
                "sharpe_ratio": round(r.sharpe_ratio, 4),
                "score": round(r.score, 4)
            }
        }

        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n  Configuracao salva em: {filename}")
        return filename

    def save_top_results(self, n: int = 100, filename: str = None):
        """Salva top N resultados"""
        if not self.all_results:
            return

        if filename is None:
            filename = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs",
                "prm_top_results.json"
            )

        # Ordena por score
        sorted_results = sorted(self.all_results, key=lambda x: x.score, reverse=True)[:n]

        results_data = []
        for r in sorted_results:
            results_data.append({
                "score": round(r.score, 4),
                "trades": r.total_trades,
                "win_rate": round(r.win_rate, 3),
                "profit_factor": round(r.profit_factor, 3),
                "pnl_pips": round(r.total_pnl_pips, 1),
                "params": {
                    "min_prices": r.min_prices,
                    "hmm_threshold": r.hmm_threshold,
                    "lyapunov_threshold": r.lyapunov_threshold,
                    "stop_loss": r.stop_loss_pips,
                    "take_profit": r.take_profit_pips,
                    "hmm_states": r.hmm_states_allowed
                }
            })

        with open(filename, 'w') as f:
            json.dump({"top_results": results_data}, f, indent=2)

        print(f"  Top {n} resultados salvos em: {filename}")

    def print_results(self):
        """Imprime melhor resultado"""
        if not self.best_result:
            return

        r = self.best_result
        print(f"\n{'='*70}")
        print(f"  MELHOR CONFIGURACAO ENCONTRADA")
        print(f"{'='*70}")
        print(f"\n  PARAMETROS:")
        print(f"    min_prices: {r.min_prices}")
        print(f"    hmm_threshold: {r.hmm_threshold}")
        print(f"    lyapunov_threshold: {r.lyapunov_threshold}")
        print(f"    hmm_states_allowed: {r.hmm_states_allowed}")
        print(f"    stop_loss_pips: {r.stop_loss_pips}")
        print(f"    take_profit_pips: {r.take_profit_pips}")
        print(f"\n  PERFORMANCE:")
        print(f"    Total Sinais: {r.total_signals}")
        print(f"    Total Trades: {r.total_trades}")
        print(f"    Winning: {r.winning_trades} | Losing: {r.losing_trades}")
        print(f"    Win Rate: {r.win_rate:.1%}")
        print(f"    Profit Factor: {r.profit_factor:.2f}")
        print(f"    PnL Total: {r.total_pnl_pips:.1f} pips")
        print(f"    Max Drawdown: {r.max_drawdown:.1%}")
        print(f"    Sharpe Ratio: {r.sharpe_ratio:.2f}")
        print(f"    Score: {r.score:.4f}")
        print(f"{'='*70}")


async def main():
    """Funcao principal"""
    print("=" * 70)
    print("  OTIMIZADOR PRM MASSIVO")
    print("  2.000.000 de Combinacoes de Parametros")
    print("  Dados REAIS do Mercado Forex")
    print("=" * 70)

    # Configuracao
    symbol = "EURUSD"
    periodicity = "H1"
    start_date = datetime(2025, 7, 1, tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)

    print(f"\n  Simbolo: {symbol}")
    print(f"  Periodicidade: {periodicity}")
    print(f"  Periodo: {start_date.date()} a {end_date.date()}")

    optimizer = MassiveOptimizer(symbol=symbol, periodicity=periodicity)

    # Carrega dados
    await optimizer.load_data(start_date, end_date)

    if not optimizer.bars:
        print("  ERRO: Nenhum dado carregado!")
        return

    # Executa otimizacao massiva
    best = optimizer.optimize(n_combinations=2000000, verbose=True)

    if best:
        optimizer.print_results()
        optimizer.save_config()
        optimizer.save_top_results(100)
    else:
        print("\n  Nenhuma configuracao lucrativa encontrada!")


if __name__ == "__main__":
    asyncio.run(main())
