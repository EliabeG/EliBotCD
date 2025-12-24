#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR FIFN ULTRA-RAPIDO - 2M COMBINACOES
================================================================================

Estrategia de 2 fases para velocidade maxima:
1. Pre-calcula TODOS os valores FIFN uma vez (lento mas uma vez so)
2. Testa 2M combinacoes de thresholds/SL/TP sobre dados pre-calculados (ultra rapido)

FIFN (Fluxo de Informacao Fisher-Navier):
- Usa Numero de Reynolds para detectar estado do mercado
- Usa KL Divergence e Skewness para direcao
- "Sweet Spot" e' a zona ideal de operacao (Re entre 2300-4000)
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
from strategies.alta_volatilidade.fifn_fisher_navier import FluxoInformacaoFisherNavier


@dataclass
class FIFNData:
    """Dados pre-calculados do FIFN para uma barra"""
    bar_idx: int
    timestamp: datetime
    price: float
    high: float
    low: float
    reynolds: float
    kl_divergence: float
    skewness: float
    pressure_gradient: float
    signal: int  # 1=LONG, -1=SHORT, 0=NEUTRAL
    confidence: float
    in_sweet_spot: bool


@dataclass
class TestResult:
    """Resultado de um teste"""
    reynolds_low: float
    reynolds_high: float
    skewness_threshold: float
    min_confidence: float
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


class FIFNUltraOptimizer:
    """Otimizador ultra-rapido com pre-calculo para FIFN"""

    def __init__(self, symbol: str = "EURUSD", periodicity: str = "H1"):
        self.symbol = symbol
        self.periodicity = periodicity
        self.pip = 0.0001
        self.spread = 1.0

        self.bars: List[Bar] = []
        self.fifn_data: List[FIFNData] = []
        self.best: Optional[TestResult] = None
        self.best_by_trades: Optional[TestResult] = None
        self.top_results: List[TestResult] = []

    async def load_and_precompute(self, start_date: datetime, end_date: datetime):
        """Carrega dados e pre-calcula valores FIFN"""
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

        print("\n  Pre-calculando valores FIFN (isso leva alguns minutos)...")
        print("  NOTA: FIFN resolve Navier-Stokes - computacionalmente intensivo")

        # FIFN com thresholds baixos para capturar todos os dados
        fifn = FluxoInformacaoFisherNavier(
            window_size=50,
            kl_lookback=10,
            reynolds_laminar=1000,  # Baixo para capturar mais
            reynolds_turbulent=6000,  # Alto para capturar mais
            reynolds_sweet_low=1500,
            reynolds_sweet_high=5500,
            skewness_threshold=0.1  # Baixo para capturar mais sinais
        )

        prices_buf = deque(maxlen=500)
        self.fifn_data = []

        min_prices_needed = 100  # FIFN precisa de dados suficientes

        for i, bar in enumerate(self.bars):
            prices_buf.append(bar.close)

            if len(prices_buf) < min_prices_needed:
                continue

            try:
                prices_arr = np.array(prices_buf)
                result = fifn.analyze(prices_arr)

                # Extrair metricas
                reynolds = result['Reynolds_Number']
                kl_div = result['KL_Divergence']
                signal_info = result['directional_signal']

                self.fifn_data.append(FIFNData(
                    bar_idx=i,
                    timestamp=bar.timestamp,
                    price=bar.close,
                    high=bar.high,
                    low=bar.low,
                    reynolds=reynolds,
                    kl_divergence=kl_div,
                    skewness=signal_info['skewness'],
                    pressure_gradient=signal_info['pressure_gradient'],
                    signal=signal_info['signal'],
                    confidence=signal_info['confidence'],
                    in_sweet_spot=signal_info['in_sweet_spot']
                ))

            except Exception as e:
                continue

            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(self.bars)} barras... ({len(self.fifn_data)} pontos validos)")

        print(f"  Pre-calculado: {len(self.fifn_data)} pontos de dados")
        return len(self.fifn_data) > 100

    def _fast_backtest(self, re_low: float, re_high: float,
                       skew_thresh: float, min_conf: float,
                       sl: float, tp: float) -> Optional[TestResult]:
        """Backtest ultra-rapido sobre dados pre-calculados"""
        if tp < sl:
            return None

        # Encontra sinais
        signals = []
        for d in self.fifn_data:
            # Condicoes para sinal baseado em Reynolds Sweet Spot
            in_zone = re_low <= d.reynolds <= re_high

            # Direcao baseada em skewness e pressure gradient
            if in_zone and d.confidence >= min_conf:
                # Long: skewness positivo e pressure gradient negativo
                if d.skewness > skew_thresh and d.pressure_gradient < 0:
                    signals.append((d.bar_idx, d.price, 1))
                # Short: skewness negativo e pressure gradient positivo
                elif d.skewness < -skew_thresh and d.pressure_gradient > 0:
                    signals.append((d.bar_idx, d.price, -1))

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
            reynolds_low=re_low,
            reynolds_high=re_high,
            skewness_threshold=skew_thresh,
            min_confidence=min_conf,
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
        if not self.fifn_data:
            print("  ERRO: Dados nao pre-calculados!")
            return None

        print(f"\n{'='*70}")
        print(f"  OTIMIZACAO FIFN: {n:,} COMBINACOES")
        print(f"{'='*70}")

        # Ranges para FIFN
        re_low_vals = np.linspace(1500, 3000, 25)
        re_high_vals = np.linspace(3500, 6000, 25)
        skew_vals = np.linspace(0.1, 1.5, 30)
        conf_vals = np.linspace(0.0, 0.5, 20)
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
            re_low = float(random.choice(re_low_vals))
            re_high = float(random.choice(re_high_vals))
            if re_high <= re_low:
                re_high = re_low + 1000

            skew = float(random.choice(skew_vals))
            conf = float(random.choice(conf_vals))
            sl = float(random.choice(sl_vals))
            tp = float(random.choice(tp_vals))

            result = self._fast_backtest(re_low, re_high, skew, conf, sl, tp)

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
                    print(f"    Re=[{result.reynolds_low:.0f}-{result.reynolds_high:.0f}] "
                          f"skew={result.skewness_threshold:.2f} "
                          f"conf={result.min_confidence:.2f} "
                          f"SL={result.stop_loss:.0f} TP={result.take_profit:.0f}")

                # Melhor por numero de trades
                if result.trades > best_trades and result.pf > 1.5:
                    best_trades = result.trades
                    self.best_by_trades = result
                    print(f"\n  [MAIS TRADES #{profitable}] Trades={result.trades} "
                          f"Score={result.score:.4f} WR={result.win_rate:.1%} "
                          f"PF={result.pf:.2f} PnL={result.pnl:.0f}pips")
                    print(f"    Re=[{result.reynolds_low:.0f}-{result.reynolds_high:.0f}] "
                          f"skew={result.skewness_threshold:.2f}")

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
                "configs", "fifn_optimized.json"
            )

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        r = self.best
        config = {
            "strategy": "FIFN-FisherNavier",
            "symbol": self.symbol,
            "periodicity": self.periodicity,
            "optimized_at": datetime.now(timezone.utc).isoformat(),
            "optimization": {
                "method": "ultra_fast_random_search",
                "combinations_tested": n_tested,
                "profitable_found": len(self.top_results)
            },
            "parameters": {
                "reynolds_sweet_low": round(r.reynolds_low, 0),
                "reynolds_sweet_high": round(r.reynolds_high, 0),
                "skewness_threshold": round(r.skewness_threshold, 4),
                "min_confidence": round(r.min_confidence, 4),
                "stop_loss_pips": round(r.stop_loss, 1),
                "take_profit_pips": round(r.take_profit, 1),
                "window_size": 50,
                "kl_lookback": 10
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
                "strategy": "FIFN-FisherNavier",
                "symbol": self.symbol,
                "periodicity": self.periodicity,
                "optimized_at": datetime.now(timezone.utc).isoformat(),
                "optimization": {
                    "method": "ultra_fast_random_search",
                    "combinations_tested": n_tested,
                    "criteria": "maximum_trades_with_pf_above_1.5"
                },
                "parameters": {
                    "reynolds_sweet_low": round(r2.reynolds_low, 0),
                    "reynolds_sweet_high": round(r2.reynolds_high, 0),
                    "skewness_threshold": round(r2.skewness_threshold, 4),
                    "min_confidence": round(r2.min_confidence, 4),
                    "stop_loss_pips": round(r2.stop_loss, 1),
                    "take_profit_pips": round(r2.take_profit, 1),
                    "window_size": 50,
                    "kl_lookback": 10
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
                    "params": {"re_low": t.reynolds_low, "re_high": t.reynolds_high,
                              "skew": t.skewness_threshold, "conf": t.min_confidence,
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
                          "params": {"re_low": t.reynolds_low, "re_high": t.reynolds_high,
                                    "skew": t.skewness_threshold, "conf": t.min_confidence,
                                    "sl": t.stop_loss, "tp": t.take_profit}}
                         for t in sorted_by_trades]
        with open(top_trades_file, 'w') as f:
            json.dump(top_trades_data, f, indent=2)
        print(f"  Top 100 (trades) salvo em: {top_trades_file}")


async def main():
    N_COMBINATIONS = 2000000  # 2 milhoes

    print("=" * 70)
    print("  OTIMIZADOR FIFN ULTRA-RAPIDO")
    print("  Fluxo de Informacao Fisher-Navier")
    print(f"  {N_COMBINATIONS:,} Combinacoes")
    print("  Dados REAIS do Mercado Forex")
    print("=" * 70)

    opt = FIFNUltraOptimizer("EURUSD", "H1")

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
            print("  O FIFN pode ser muito seletivo neste periodo.")
    else:
        print("\n  ERRO: Falha ao pre-calcular dados FIFN!")


if __name__ == "__main__":
    asyncio.run(main())
