#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR PRM ULTRA-RAPIDO - 2M COMBINACOES
================================================================================

Estrategia de 2 fases para velocidade maxima:
1. Pre-calcula TODOS os valores PRM uma vez (lento mas uma vez so)
2. Testa 2M combinacoes de thresholds/SL/TP sobre dados pre-calculados (ultra rapido)
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
from strategies.alta_volatilidade.prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot


@dataclass
class PRMData:
    """Dados pre-calculados do PRM para uma barra"""
    bar_idx: int
    timestamp: datetime
    price: float
    high: float
    low: float
    hmm_prob: float
    lyapunov: float
    hmm_state: int


@dataclass
class TestResult:
    """Resultado de um teste"""
    min_prices: int
    hmm_threshold: float
    lyapunov_threshold: float
    hmm_states: List[int]
    stop_loss: float
    take_profit: float
    trades: int
    wins: int
    losses: int
    pnl: float
    win_rate: float
    pf: float
    dd: float
    score: float


class UltraOptimizer:
    """Otimizador ultra-rapido com pre-calculo"""

    def __init__(self, symbol: str = "EURUSD", periodicity: str = "H1"):
        self.symbol = symbol
        self.periodicity = periodicity
        self.pip = 0.0001
        self.spread = 1.0

        self.bars: List[Bar] = []
        self.prm_data: List[PRMData] = []
        self.best: Optional[TestResult] = None
        self.best_by_trades: Optional[TestResult] = None  # Maior numero de trades
        self.top_results: List[TestResult] = []

    async def load_and_precompute(self, start_date: datetime, end_date: datetime):
        """Carrega dados e pre-calcula valores PRM"""
        print("\n  Carregando dados REAIS...")
        self.bars = await download_historical_data(
            symbol=self.symbol,
            periodicity=self.periodicity,
            start_time=start_date,
            end_time=end_date
        )
        print(f"  Barras: {len(self.bars)}")

        if not self.bars:
            return False

        print("\n  Pre-calculando valores PRM (isso leva alguns minutos)...")

        prm = ProtocoloRiemannMandelbrot(
            n_states=3,
            hmm_threshold=0.1,
            lyapunov_threshold_k=0.001,
            curvature_threshold=0.0001,
            lookback_window=100
        )

        prices_buf = deque(maxlen=500)
        volumes_buf = deque(maxlen=500)
        self.prm_data = []

        for i, bar in enumerate(self.bars):
            prices_buf.append(bar.close)
            volumes_buf.append(bar.volume)

            if len(prices_buf) < 50:
                continue

            try:
                result = prm.analyze(np.array(prices_buf), np.array(volumes_buf))

                self.prm_data.append(PRMData(
                    bar_idx=i,
                    timestamp=bar.timestamp,
                    price=bar.close,
                    high=bar.high,
                    low=bar.low,
                    hmm_prob=result['Prob_HMM'],
                    lyapunov=result['Lyapunov_Score'],
                    hmm_state=result['hmm_analysis']['current_state']
                ))
            except:
                continue

            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(self.bars)} barras...")

        print(f"  Pre-calculado: {len(self.prm_data)} pontos de dados")
        return True

    def _fast_backtest(self, hmm_thresh: float, lyap_thresh: float,
                       states: List[int], sl: float, tp: float,
                       min_idx: int = 50) -> Optional[TestResult]:
        """Backtest ultra-rapido sobre dados pre-calculados"""
        if tp < sl:
            return None

        # Encontra sinais
        signals = []
        for d in self.prm_data:
            if d.bar_idx < min_idx:
                continue
            if d.hmm_prob >= hmm_thresh and d.lyapunov >= lyap_thresh and d.hmm_state in states:
                # Direcao baseada em tendencia
                if d.bar_idx >= 10:
                    trend = d.price - self.bars[d.bar_idx - 10].close
                    direction = 1 if trend > 0 else -1
                else:
                    direction = 1
                signals.append((d.bar_idx, d.price, direction))

        if len(signals) < 3:
            return None

        # Executa trades
        pnls = []
        for bar_idx, entry, direction in signals:
            sl_price = entry - direction * sl * self.pip
            tp_price = entry + direction * tp * self.pip

            pnl = 0
            for j in range(bar_idx + 1, min(bar_idx + 500, len(self.bars))):
                b = self.bars[j]
                if direction == 1:  # LONG
                    if b.low <= sl_price:
                        pnl = -sl - self.spread
                        break
                    if b.high >= tp_price:
                        pnl = tp - self.spread
                        break
                else:  # SHORT
                    if b.high >= sl_price:
                        pnl = -sl - self.spread
                        break
                    if b.low <= tp_price:
                        pnl = tp - self.spread
                        break

            if pnl == 0:
                exit_p = self.bars[min(bar_idx + 100, len(self.bars) - 1)].close
                pnl = direction * (exit_p - entry) / self.pip - self.spread

            pnls.append(pnl)

        if not pnls:
            return None

        wins = sum(1 for p in pnls if p > 0)
        losses = len(pnls) - wins
        total = sum(pnls)
        wr = wins / len(pnls)
        gp = sum(p for p in pnls if p > 0) or 0.001
        gl = abs(sum(p for p in pnls if p <= 0)) or 0.001
        pf = gp / gl

        # Drawdown
        eq = np.cumsum([0] + pnls)
        peak = np.maximum.accumulate(eq + 10000)
        dd = np.max((peak - (eq + 10000)) / peak)

        # Score
        score = 0
        if len(pnls) >= 5 and pf > 1.0:
            score = (0.30 * min(pf, 5) / 5 + 0.25 * wr +
                    0.20 * max(0, 1 - dd * 5) + 0.25 * min(len(pnls) / 50, 1))

        return TestResult(
            min_prices=min_idx,
            hmm_threshold=hmm_thresh,
            lyapunov_threshold=lyap_thresh,
            hmm_states=states,
            stop_loss=sl,
            take_profit=tp,
            trades=len(pnls),
            wins=wins,
            losses=losses,
            pnl=total,
            win_rate=wr,
            pf=pf,
            dd=dd,
            score=score
        )

    def optimize(self, n: int = 2000000) -> Optional[TestResult]:
        """Executa otimizacao com N combinacoes"""
        if not self.prm_data:
            print("  ERRO: Dados nao pre-calculados!")
            return None

        print(f"\n{'='*70}")
        print(f"  OTIMIZACAO: {n:,} COMBINACOES")
        print(f"{'='*70}")

        # Ranges
        hmm_vals = np.linspace(0.5, 0.99, 50)
        lyap_vals = np.linspace(0.01, 0.15, 40)
        sl_vals = np.linspace(5, 50, 30)
        tp_vals = np.linspace(10, 120, 40)
        states_opts = [[1], [0, 1], [1, 2], [0, 1, 2]]
        min_prices_vals = [30, 50, 75, 100]

        best_score = -1
        best_trades = 0
        tested = 0
        profitable = 0
        start = datetime.now()

        for _ in range(n):
            tested += 1

            # Parametros aleatorios
            hmm = float(random.choice(hmm_vals))
            lyap = float(random.choice(lyap_vals))
            sl = float(random.choice(sl_vals))
            tp = float(random.choice(tp_vals))
            states = random.choice(states_opts)
            mp = random.choice(min_prices_vals)

            result = self._fast_backtest(hmm, lyap, states, sl, tp, mp)

            if result and result.pf > 1.0:
                profitable += 1
                self.top_results.append(result)

                # Melhor por score
                if result.score > best_score:
                    best_score = result.score
                    self.best = result
                    print(f"\n  [MELHOR SCORE #{profitable}] Score={result.score:.4f} "
                          f"Trades={result.trades} WR={result.win_rate:.1%} "
                          f"PF={result.pf:.2f} PnL={result.pnl:.0f}pips")
                    print(f"    hmm={result.hmm_threshold:.3f} lyap={result.lyapunov_threshold:.4f} "
                          f"SL={result.stop_loss:.0f} TP={result.take_profit:.0f}")

                # Melhor por numero de trades (com PF > 1.5 para ser lucrativo)
                if result.trades > best_trades and result.pf > 1.5:
                    best_trades = result.trades
                    self.best_by_trades = result
                    print(f"\n  [MAIS TRADES #{profitable}] Trades={result.trades} "
                          f"Score={result.score:.4f} WR={result.win_rate:.1%} "
                          f"PF={result.pf:.2f} PnL={result.pnl:.0f}pips")
                    print(f"    hmm={result.hmm_threshold:.3f} lyap={result.lyapunov_threshold:.4f} "
                          f"SL={result.stop_loss:.0f} TP={result.take_profit:.0f}")

            if tested % 100000 == 0:
                elapsed = (datetime.now() - start).total_seconds()
                rate = tested / elapsed
                eta = (n - tested) / rate / 60
                print(f"  {tested:,}/{n:,} ({tested/n*100:.1f}%) | "
                      f"Lucrativos: {profitable:,} | "
                      f"Vel: {rate:.0f}/s | ETA: {eta:.0f}min")

        elapsed = (datetime.now() - start).total_seconds()
        print(f"\n{'='*70}")
        print(f"  CONCLUIDO em {elapsed/60:.1f}min")
        print(f"  Testados: {tested:,} | Lucrativos: {profitable:,}")
        print(f"{'='*70}")

        return self.best

    def save(self, filename: str = None, n_tested: int = 0):
        """Salva melhor configuracao"""
        if not self.best:
            return

        if filename is None:
            filename = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs", "prm_optimized.json"
            )

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        r = self.best
        config = {
            "strategy": "PRM-RiemannMandelbrot",
            "symbol": self.symbol,
            "periodicity": self.periodicity,
            "optimized_at": datetime.now(timezone.utc).isoformat(),
            "optimization": {
                "method": "ultra_fast_random_search",
                "combinations_tested": n_tested,
                "profitable_found": len(self.top_results)
            },
            "parameters": {
                "min_prices": r.min_prices,
                "stop_loss_pips": round(r.stop_loss, 1),
                "take_profit_pips": round(r.take_profit, 1),
                "hmm_threshold": round(r.hmm_threshold, 3),
                "lyapunov_threshold": round(r.lyapunov_threshold, 4),
                "curvature_threshold": 0.01,
                "hmm_states_allowed": r.hmm_states
            },
            "performance": {
                "total_trades": r.trades,
                "winning_trades": r.wins,
                "losing_trades": r.losses,
                "win_rate": round(r.win_rate, 4),
                "profit_factor": round(r.pf, 4),
                "total_pnl_pips": round(r.pnl, 2),
                "max_drawdown": round(r.dd, 4),
                "score": round(r.score, 4)
            }
        }

        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n  Melhor Score salvo em: {filename}")

        # Salva configuracao com mais trades
        if self.best_by_trades:
            r2 = self.best_by_trades
            trades_file = filename.replace('.json', '_max_trades.json')
            config_trades = {
                "strategy": "PRM-RiemannMandelbrot",
                "symbol": self.symbol,
                "periodicity": self.periodicity,
                "optimized_at": datetime.now(timezone.utc).isoformat(),
                "optimization": {
                    "method": "ultra_fast_random_search",
                    "combinations_tested": n_tested,
                    "criteria": "maximum_trades_with_pf_above_1.5"
                },
                "parameters": {
                    "min_prices": r2.min_prices,
                    "stop_loss_pips": round(r2.stop_loss, 1),
                    "take_profit_pips": round(r2.take_profit, 1),
                    "hmm_threshold": round(r2.hmm_threshold, 3),
                    "lyapunov_threshold": round(r2.lyapunov_threshold, 4),
                    "curvature_threshold": 0.01,
                    "hmm_states_allowed": r2.hmm_states
                },
                "performance": {
                    "total_trades": r2.trades,
                    "winning_trades": r2.wins,
                    "losing_trades": r2.losses,
                    "win_rate": round(r2.win_rate, 4),
                    "profit_factor": round(r2.pf, 4),
                    "total_pnl_pips": round(r2.pnl, 2),
                    "max_drawdown": round(r2.dd, 4),
                    "score": round(r2.score, 4)
                }
            }
            with open(trades_file, 'w') as f:
                json.dump(config_trades, f, indent=2)
            print(f"  Mais Trades salvo em: {trades_file}")

        # Salva top 100 por score
        top_file = filename.replace('.json', '_top100.json')
        sorted_top = sorted(self.top_results, key=lambda x: x.score, reverse=True)[:100]
        top_data = [{"score": round(t.score, 4), "trades": t.trades,
                    "wr": round(t.win_rate, 3), "pf": round(t.pf, 3),
                    "pnl": round(t.pnl, 1),
                    "params": {"hmm": t.hmm_threshold, "lyap": t.lyapunov_threshold,
                              "sl": t.stop_loss, "tp": t.take_profit, "states": t.hmm_states}}
                   for t in sorted_top]
        with open(top_file, 'w') as f:
            json.dump(top_data, f, indent=2)
        print(f"  Top 100 (score) salvo em: {top_file}")

        # Salva top 100 por numero de trades
        top_trades_file = filename.replace('.json', '_top100_by_trades.json')
        sorted_by_trades = sorted([t for t in self.top_results if t.pf > 1.5],
                                  key=lambda x: x.trades, reverse=True)[:100]
        top_trades_data = [{"trades": t.trades, "score": round(t.score, 4),
                          "wr": round(t.win_rate, 3), "pf": round(t.pf, 3),
                          "pnl": round(t.pnl, 1),
                          "params": {"hmm": t.hmm_threshold, "lyap": t.lyapunov_threshold,
                                    "sl": t.stop_loss, "tp": t.take_profit, "states": t.hmm_states}}
                         for t in sorted_by_trades]
        with open(top_trades_file, 'w') as f:
            json.dump(top_trades_data, f, indent=2)
        print(f"  Top 100 (trades) salvo em: {top_trades_file}")


async def main():
    N_COMBINATIONS = 20000000  # 20 milhoes

    print("=" * 70)
    print("  OTIMIZADOR PRM ULTRA-RAPIDO")
    print(f"  {N_COMBINATIONS:,} Combinacoes")
    print("  Dados REAIS do Mercado Forex")
    print("=" * 70)

    opt = UltraOptimizer("EURUSD", "H1")

    start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    print(f"\n  Periodo: {start.date()} a {end.date()}")

    if await opt.load_and_precompute(start, end):
        best = opt.optimize(N_COMBINATIONS)
        if best:
            print(f"\n  MELHOR RESULTADO (SCORE):")
            print(f"    Trades: {best.trades}")
            print(f"    Win Rate: {best.win_rate:.1%}")
            print(f"    Profit Factor: {best.pf:.2f}")
            print(f"    PnL: {best.pnl:.1f} pips")
            print(f"    Score: {best.score:.4f}")

            if opt.best_by_trades:
                print(f"\n  MELHOR RESULTADO (MAIS TRADES):")
                print(f"    Trades: {opt.best_by_trades.trades}")
                print(f"    Win Rate: {opt.best_by_trades.win_rate:.1%}")
                print(f"    Profit Factor: {opt.best_by_trades.pf:.2f}")
                print(f"    PnL: {opt.best_by_trades.pnl:.1f} pips")
                print(f"    Score: {opt.best_by_trades.score:.4f}")

            opt.save(n_tested=N_COMBINATIONS)


if __name__ == "__main__":
    asyncio.run(main())
