#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR DTT ULTRA-RAPIDO - 20M COMBINACOES
================================================================================

Estrategia de 2 fases para velocidade maxima:
1. Pre-calcula TODOS os valores DTT uma vez (lento mas uma vez so)
2. Testa 20M combinacoes de thresholds/SL/TP sobre dados pre-calculados (ultra rapido)

DTT (Detector de Tunelamento Topologico):
- Usa Homologia Persistente para detectar estruturas topologicas
- Usa Equacao de Schrodinger para calcular probabilidade de tunelamento
- Combina ambos para gerar sinais de breakout
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
from strategies.alta_volatilidade.dtt_tunelamento_topologico import DetectorTunelamentoTopologico


@dataclass
class DTTData:
    """Dados pre-calculados do DTT para uma barra"""
    bar_idx: int
    timestamp: datetime
    price: float
    high: float
    low: float
    persistence_entropy: float
    tunneling_probability: float
    signal_strength: float
    direction: int  # 1=LONG, -1=SHORT, 0=NEUTRAL
    trade_on: bool
    n_cycles: int
    energy_ratio: float


@dataclass
class TestResult:
    """Resultado de um teste"""
    entropy_threshold: float
    tunneling_threshold: float
    min_signal_strength: float
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


class DTTUltraOptimizer:
    """Otimizador ultra-rapido com pre-calculo para DTT"""

    def __init__(self, symbol: str = "EURUSD", periodicity: str = "H1"):
        self.symbol = symbol
        self.periodicity = periodicity
        self.pip = 0.0001
        self.spread = 1.0

        self.bars: List[Bar] = []
        self.dtt_data: List[DTTData] = []
        self.best: Optional[TestResult] = None
        self.best_by_trades: Optional[TestResult] = None
        self.top_results: List[TestResult] = []

    async def load_and_precompute(self, start_date: datetime, end_date: datetime):
        """Carrega dados e pre-calcula valores DTT"""
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

        print("\n  Pre-calculando valores DTT (isso leva alguns minutos)...")
        print("  NOTA: DTT e' computacionalmente intensivo (Homologia Persistente)")

        # DTT com thresholds baixos para capturar todos os dados
        dtt = DetectorTunelamentoTopologico(
            max_points=150,
            use_dimensionality_reduction=True,
            reduction_method='pca',
            reduction_components=3,
            persistence_entropy_threshold=0.1,  # Baixo para capturar tudo
            tunneling_probability_threshold=0.05,  # Baixo para capturar tudo
            hbar=1.0,
            particle_mass=1.0,
            n_eigenstates=10
        )

        prices_buf = deque(maxlen=500)
        self.dtt_data = []

        min_prices_needed = 150  # DTT precisa de mais dados

        for i, bar in enumerate(self.bars):
            prices_buf.append(bar.close)

            if len(prices_buf) < min_prices_needed:
                continue

            try:
                prices_arr = np.array(prices_buf)
                result = dtt.analyze(prices_arr)

                # Extrair metricas
                entropy = result['entropy']['persistence_entropy']
                tunneling = result['tunneling']['tunneling_probability']
                signal_strength = result['signal_strength']
                direction_str = result['direction']
                direction = 1 if direction_str == 'LONG' else (-1 if direction_str == 'SHORT' else 0)
                trade_on = result['trade_on']
                n_cycles = result['entropy']['n_significant_cycles']
                energy_ratio = result['tunneling']['energy_ratio']

                self.dtt_data.append(DTTData(
                    bar_idx=i,
                    timestamp=bar.timestamp,
                    price=bar.close,
                    high=bar.high,
                    low=bar.low,
                    persistence_entropy=entropy,
                    tunneling_probability=tunneling,
                    signal_strength=signal_strength,
                    direction=direction,
                    trade_on=trade_on,
                    n_cycles=n_cycles,
                    energy_ratio=energy_ratio
                ))

            except Exception as e:
                continue

            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(self.bars)} barras... ({len(self.dtt_data)} pontos validos)")

        print(f"  Pre-calculado: {len(self.dtt_data)} pontos de dados")
        return len(self.dtt_data) > 100

    def _fast_backtest(self, entropy_thresh: float, tunneling_thresh: float,
                       min_strength: float, sl: float, tp: float) -> Optional[TestResult]:
        """Backtest ultra-rapido sobre dados pre-calculados"""
        if tp < sl:
            return None

        # Encontra sinais
        signals = []
        for d in self.dtt_data:
            # Condicoes para sinal
            if (d.persistence_entropy >= entropy_thresh and
                d.tunneling_probability >= tunneling_thresh and
                d.signal_strength >= min_strength and
                d.direction != 0):

                signals.append((d.bar_idx, d.price, d.direction))

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
            entropy_threshold=entropy_thresh,
            tunneling_threshold=tunneling_thresh,
            min_signal_strength=min_strength,
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

    def optimize(self, n: int = 20000000) -> Optional[TestResult]:
        """Executa otimizacao com N combinacoes"""
        if not self.dtt_data:
            print("  ERRO: Dados nao pre-calculados!")
            return None

        print(f"\n{'='*70}")
        print(f"  OTIMIZACAO DTT: {n:,} COMBINACOES")
        print(f"{'='*70}")

        # Ranges para DTT
        entropy_vals = np.linspace(0.1, 0.9, 40)
        tunneling_vals = np.linspace(0.05, 0.5, 40)
        strength_vals = np.linspace(0.1, 0.8, 30)
        sl_vals = np.linspace(5, 60, 35)
        tp_vals = np.linspace(10, 150, 45)

        best_score = -1
        best_trades = 0
        tested = 0
        profitable = 0
        start = datetime.now()

        for _ in range(n):
            tested += 1

            # Parametros aleatorios
            entropy = float(random.choice(entropy_vals))
            tunneling = float(random.choice(tunneling_vals))
            strength = float(random.choice(strength_vals))
            sl = float(random.choice(sl_vals))
            tp = float(random.choice(tp_vals))

            result = self._fast_backtest(entropy, tunneling, strength, sl, tp)

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
                    print(f"    entropy={result.entropy_threshold:.3f} "
                          f"tunneling={result.tunneling_threshold:.3f} "
                          f"strength={result.min_signal_strength:.2f} "
                          f"SL={result.stop_loss:.0f} TP={result.take_profit:.0f}")

                # Melhor por numero de trades
                if result.trades > best_trades and result.pf > 1.5:
                    best_trades = result.trades
                    self.best_by_trades = result
                    print(f"\n  [MAIS TRADES #{profitable}] Trades={result.trades} "
                          f"Score={result.score:.4f} WR={result.win_rate:.1%} "
                          f"PF={result.pf:.2f} PnL={result.pnl:.0f}pips")
                    print(f"    entropy={result.entropy_threshold:.3f} "
                          f"tunneling={result.tunneling_threshold:.3f}")

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
            print("  Nenhuma configuracao lucrativa encontrada!")
            return

        if filename is None:
            filename = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs", "dtt_optimized.json"
            )

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        r = self.best
        config = {
            "strategy": "DTT-TunelamentoTopologico",
            "symbol": self.symbol,
            "periodicity": self.periodicity,
            "optimized_at": datetime.now(timezone.utc).isoformat(),
            "optimization": {
                "method": "ultra_fast_random_search",
                "combinations_tested": n_tested,
                "profitable_found": len(self.top_results)
            },
            "parameters": {
                "persistence_entropy_threshold": round(r.entropy_threshold, 4),
                "tunneling_probability_threshold": round(r.tunneling_threshold, 4),
                "min_signal_strength": round(r.min_signal_strength, 4),
                "stop_loss_pips": round(r.stop_loss, 1),
                "take_profit_pips": round(r.take_profit, 1),
                "max_points": 150,
                "use_dimensionality_reduction": True,
                "reduction_method": "pca",
                "reduction_components": 3
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
                "strategy": "DTT-TunelamentoTopologico",
                "symbol": self.symbol,
                "periodicity": self.periodicity,
                "optimized_at": datetime.now(timezone.utc).isoformat(),
                "optimization": {
                    "method": "ultra_fast_random_search",
                    "combinations_tested": n_tested,
                    "criteria": "maximum_trades_with_pf_above_1.5"
                },
                "parameters": {
                    "persistence_entropy_threshold": round(r2.entropy_threshold, 4),
                    "tunneling_probability_threshold": round(r2.tunneling_threshold, 4),
                    "min_signal_strength": round(r2.min_signal_strength, 4),
                    "stop_loss_pips": round(r2.stop_loss, 1),
                    "take_profit_pips": round(r2.take_profit, 1),
                    "max_points": 150,
                    "use_dimensionality_reduction": True,
                    "reduction_method": "pca",
                    "reduction_components": 3
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
                    "params": {"entropy": t.entropy_threshold,
                              "tunneling": t.tunneling_threshold,
                              "strength": t.min_signal_strength,
                              "sl": t.stop_loss, "tp": t.take_profit}}
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
                          "params": {"entropy": t.entropy_threshold,
                                    "tunneling": t.tunneling_threshold,
                                    "strength": t.min_signal_strength,
                                    "sl": t.stop_loss, "tp": t.take_profit}}
                         for t in sorted_by_trades]
        with open(top_trades_file, 'w') as f:
            json.dump(top_trades_data, f, indent=2)
        print(f"  Top 100 (trades) salvo em: {top_trades_file}")


async def main():
    N_COMBINATIONS = 2000000  # 2 milhoes

    print("=" * 70)
    print("  OTIMIZADOR DTT ULTRA-RAPIDO")
    print("  Detector de Tunelamento Topologico")
    print(f"  {N_COMBINATIONS:,} Combinacoes")
    print("  Dados REAIS do Mercado Forex")
    print("=" * 70)

    opt = DTTUltraOptimizer("EURUSD", "H1")

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
        else:
            print("\n  AVISO: Nenhuma configuracao lucrativa encontrada!")
            print("  O DTT pode ser muito seletivo neste periodo.")
    else:
        print("\n  ERRO: Falha ao pre-calcular dados DTT!")


if __name__ == "__main__":
    asyncio.run(main())
