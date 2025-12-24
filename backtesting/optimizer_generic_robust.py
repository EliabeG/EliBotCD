#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR GENERICO ROBUSTO - PARA MULTIPLOS INDICADORES
================================================================================

Otimiza qualquer indicador que tenha metodo analyze() retornando 'signal'.

Indicadores suportados:
- PHM (Projetor Holografico Maldacena)
- RHHF (Ressonador Hilbert-Huang Fractal)
- SEED (Sintetizador Evolutivo Estruturas Dissipativas)
- SEMA (Sincronizador Espectral)
- STGK (Sintetizador Topos Grothendieck-Kolmogorov)

PARA DINHEIRO REAL. SEM OVERFITTING.
================================================================================
"""

import sys
import os
import json
import asyncio
import random
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.fxopen_historical_ws import Bar, download_historical_data
from backtesting.robust_optimizer import (
    RobustBacktester, RobustResult, BacktestResult,
    save_robust_config
)


@dataclass
class GenericSignal:
    """Sinal pre-calculado generico"""
    bar_idx: int
    price: float
    high: float
    low: float
    signal: int
    confidence: float
    extra: dict


INDICATORS = {
    'PHM': {
        'module': 'strategies.alta_volatilidade.phm_projetor_holografico',
        'class': 'ProjetorHolograficoMaldacena',
        'name': 'PHM-ProjetorHolografico'
    },
    'RHHF': {
        'module': 'strategies.alta_volatilidade.rhhf_ressonador_hilbert_huang',
        'class': 'RessonadorHilbertHuangFractal',
        'name': 'RHHF-HilbertHuangFractal'
    },
    'SEED': {
        'module': 'strategies.alta_volatilidade.seed_sintetizador_evolutivo',
        'class': 'SintetizadorEvolutivoEstruturasDissipativas',
        'name': 'SEED-SintetizadorEvolutivo'
    },
    'SEMA': {
        'module': 'strategies.alta_volatilidade.sema_sincronizador_espectral',
        'class': 'SincronizadorEspectral',
        'name': 'SEMA-SincronizadorEspectral'
    },
    'STGK': {
        'module': 'strategies.alta_volatilidade.stgk_sintetizador_topos',
        'class': 'SintetizadorToposGrothendieckKolmogorov',
        'name': 'STGK-ToposGrothendieck'
    }
}


class GenericRobustOptimizer:
    """Otimizador generico com validacao anti-overfitting"""

    def __init__(self, indicator_name: str, symbol: str = "EURUSD", periodicity: str = "H1"):
        self.indicator_name = indicator_name
        self.symbol = symbol
        self.periodicity = periodicity
        self.backtester = RobustBacktester(pip=0.0001, spread=1.0)

        self.bars: List[Bar] = []
        self.signals: List[GenericSignal] = []
        self.train_bars: List[Bar] = []
        self.test_bars: List[Bar] = []
        self.train_signals: List[GenericSignal] = []
        self.test_signals: List[GenericSignal] = []

        self.robust_results: List[RobustResult] = []
        self.best: Optional[RobustResult] = None

        # Load indicator class
        if indicator_name not in INDICATORS:
            raise ValueError(f"Indicador desconhecido: {indicator_name}")

        ind_info = INDICATORS[indicator_name]
        module = __import__(ind_info['module'], fromlist=[ind_info['class']])
        self.indicator_class = getattr(module, ind_info['class'])
        self.strategy_name = ind_info['name']

    async def load_and_precompute(self, start_date: datetime, end_date: datetime):
        """Carrega dados e pre-calcula sinais"""
        print("\n" + "=" * 70)
        print(f"  CARREGANDO DADOS REAIS - {self.indicator_name}")
        print("=" * 70)

        self.bars = await download_historical_data(
            symbol=self.symbol,
            periodicity=self.periodicity,
            start_time=start_date,
            end_time=end_date
        )
        print(f"  Total de barras: {len(self.bars)}")

        if len(self.bars) < 300:
            print("  ERRO: Dados insuficientes!")
            return False

        # SPLIT TRAIN/TEST (70/30)
        split_idx = int(len(self.bars) * 0.70)
        self.train_bars = self.bars[:split_idx]
        self.test_bars = self.bars[split_idx:]

        print(f"\n  DIVISAO TRAIN/TEST:")
        print(f"    Treino: {len(self.train_bars)} barras")
        print(f"    Teste:  {len(self.test_bars)} barras")

        # Pre-calcular sinais
        print(f"\n  Pre-calculando sinais {self.indicator_name}...")

        indicator = self.indicator_class()
        prices_buf = deque(maxlen=200)
        self.signals = []
        min_prices = 100

        for i, bar in enumerate(self.bars):
            prices_buf.append(bar.close)

            if len(prices_buf) < min_prices:
                continue

            try:
                prices_arr = np.array(prices_buf)
                result = indicator.analyze(prices_arr)

                signal = result.get('signal', 0)
                confidence = result.get('confidence', 0.5)

                self.signals.append(GenericSignal(
                    bar_idx=i,
                    price=bar.close,
                    high=bar.high,
                    low=bar.low,
                    signal=signal,
                    confidence=confidence,
                    extra={}
                ))

            except:
                continue

            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(self.bars)} barras...")

        # Separar sinais
        self.train_signals = [s for s in self.signals if s.bar_idx < split_idx]
        self.test_signals = [s for s in self.signals if s.bar_idx >= split_idx]

        print(f"\n  Sinais pre-calculados:")
        print(f"    Treino: {len(self.train_signals)} sinais")
        print(f"    Teste:  {len(self.test_signals)} sinais")

        # Estatisticas de sinais
        if self.signals:
            signals_buy = sum(1 for s in self.signals if s.signal == 1)
            signals_sell = sum(1 for s in self.signals if s.signal == -1)
            signals_hold = sum(1 for s in self.signals if s.signal == 0)
            print(f"\n  Distribuicao de sinais:")
            print(f"    BUY: {signals_buy}, SELL: {signals_sell}, HOLD: {signals_hold}")

        return len(self.train_signals) > 50 and len(self.test_signals) > 20

    def _run_backtest(self, signals: List[GenericSignal], bars: List[Bar],
                      confidence_thresh: float,
                      sl: float, tp: float,
                      bar_offset: int = 0) -> List[float]:
        """Executa backtest usando sinais do indicador"""
        if tp <= sl:
            return []

        entries = []
        for s in signals:
            # Usar sinal do indicador se confianca for alta o suficiente
            if s.signal != 0 and s.confidence >= confidence_thresh:
                entries.append((s.bar_idx - bar_offset, s.price, s.signal))

        if len(entries) < 3:
            return []

        pnls = []
        for bar_idx, entry_price, direction in entries:
            if bar_idx < 0 or bar_idx >= len(bars) - 1:
                continue

            trade = self.backtester.execute_trade(
                bars=bars,
                entry_idx=bar_idx,
                entry_price=entry_price,
                direction=direction,
                sl_pips=sl,
                tp_pips=tp,
                max_bars=200
            )
            pnls.append(trade.pnl_pips)

        return pnls

    def _test_params(self, confidence_thresh: float,
                     sl: float, tp: float) -> Optional[RobustResult]:
        """Testa parametros em treino e teste"""

        train_pnls = self._run_backtest(
            self.train_signals, self.train_bars,
            confidence_thresh, sl, tp,
            bar_offset=0
        )
        train_result = self.backtester.calculate_backtest_result(train_pnls)

        # Filtros para treino
        if not train_result.is_valid(
            min_trades=15,
            max_win_rate=0.70,
            min_win_rate=0.25,
            max_pf=5.0,
            min_pf=1.03,
            max_dd=0.50
        ):
            return None

        # Backtest no TESTE
        split_idx = len(self.train_bars)
        test_pnls = self._run_backtest(
            self.test_signals, self.test_bars,
            confidence_thresh, sl, tp,
            bar_offset=split_idx
        )
        test_result = self.backtester.calculate_backtest_result(test_pnls)

        # Filtros para teste
        if not test_result.is_valid(
            min_trades=8,
            max_win_rate=0.80,
            min_win_rate=0.18,
            max_pf=6.0,
            min_pf=0.85,
            max_dd=0.60
        ):
            return None

        # Calcula robustez
        robustness, degradation, _ = self.backtester.calculate_robustness(
            train_result, test_result
        )

        # Robustez: teste deve manter >= 45% do treino
        pf_ratio = test_result.profit_factor / train_result.profit_factor if train_result.profit_factor > 0 else 0
        wr_ratio = test_result.win_rate / train_result.win_rate if train_result.win_rate > 0 else 0
        is_robust = pf_ratio >= 0.45 and wr_ratio >= 0.45 and test_result.profit_factor >= 0.85

        if not is_robust:
            return None

        params = {
            "confidence_threshold": round(confidence_thresh, 3),
            "stop_loss_pips": round(sl, 1),
            "take_profit_pips": round(tp, 1)
        }

        return RobustResult(
            params=params,
            train_result=train_result,
            test_result=test_result,
            robustness_score=robustness,
            degradation=degradation,
            is_robust=is_robust
        )

    def optimize(self, n: int = 50000) -> Optional[RobustResult]:
        """Executa otimizacao robusta"""
        if not self.train_signals or not self.test_signals:
            print("  ERRO: Dados nao carregados!")
            return None

        print(f"\n{'='*70}")
        print(f"  OTIMIZACAO ROBUSTA {self.indicator_name}: {n:,} COMBINACOES")
        print(f"  Com validacao Train/Test Split")
        print(f"{'='*70}")

        # Ranges
        confidence_vals = np.linspace(0.20, 0.85, 20)
        sl_vals = np.linspace(18, 60, 18)
        tp_vals = np.linspace(22, 85, 22)

        best_robustness = -1
        tested = 0
        robust_count = 0
        start = datetime.now()

        for _ in range(n):
            tested += 1

            confidence = float(random.choice(confidence_vals))
            sl = float(random.choice(sl_vals))
            tp = float(random.choice(tp_vals))

            result = self._test_params(confidence, sl, tp)

            if result:
                robust_count += 1
                self.robust_results.append(result)

                if result.robustness_score > best_robustness:
                    best_robustness = result.robustness_score
                    self.best = result

                    print(f"\n  [ROBUSTO #{robust_count}] Robustez={result.robustness_score:.4f}")
                    print(f"    TREINO: {result.train_result.trades} trades, "
                          f"WR={result.train_result.win_rate:.1%}, "
                          f"PF={result.train_result.profit_factor:.2f}")
                    print(f"    TESTE:  {result.test_result.trades} trades, "
                          f"WR={result.test_result.win_rate:.1%}, "
                          f"PF={result.test_result.profit_factor:.2f}")

            if tested % 10000 == 0:
                elapsed = (datetime.now() - start).total_seconds()
                rate = tested / elapsed if elapsed > 0 else 0
                eta = (n - tested) / rate / 60 if rate > 0 else 0
                print(f"  {tested:,}/{n:,} ({tested/n*100:.1f}%) | "
                      f"Robustos: {robust_count} | "
                      f"Vel: {rate:.0f}/s | ETA: {eta:.0f}min")

        elapsed = (datetime.now() - start).total_seconds()
        print(f"\n{'='*70}")
        print(f"  CONCLUIDO em {elapsed/60:.1f}min")
        print(f"  Testados: {tested:,} | Robustos: {robust_count}")
        print(f"{'='*70}")

        return self.best

    def save(self, n_tested: int = 0):
        """Salva melhor configuracao robusta"""
        if not self.best:
            print("  Nenhuma configuracao robusta encontrada!")
            return

        save_robust_config(
            result=self.best,
            strategy_name=self.strategy_name,
            symbol=self.symbol,
            periodicity=self.periodicity,
            n_tested=n_tested,
            n_robust=len(self.robust_results)
        )

        # Top 10
        top_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", f"{self.indicator_name.lower()}_robust_top10.json"
        )
        sorted_results = sorted(self.robust_results, key=lambda x: x.robustness_score, reverse=True)[:10]
        top_data = []
        for r in sorted_results:
            d = r.to_dict()
            for k, v in d.items():
                if isinstance(v, (np.bool_, np.generic)):
                    d[k] = bool(v) if isinstance(v, np.bool_) else float(v)
            top_data.append(d)
        with open(top_file, 'w') as f:
            json.dump(top_data, f, indent=2)
        print(f"  Top 10 robustos salvo em: {top_file}")


async def optimize_indicator(indicator_name: str, n_combinations: int = 50000):
    """Otimiza um unico indicador"""
    print("=" * 70)
    print(f"  OTIMIZADOR {indicator_name} ROBUSTO")
    print("  Com Validacao Anti-Overfitting")
    print(f"  {n_combinations:,} Combinacoes")
    print("  PARA DINHEIRO REAL")
    print("=" * 70)

    opt = GenericRobustOptimizer(indicator_name, "EURUSD", "H1")

    start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    print(f"\n  Periodo: {start.date()} a {end.date()}")

    if await opt.load_and_precompute(start, end):
        best = opt.optimize(n_combinations)
        if best:
            print(f"\n{'='*70}")
            print(f"  MELHOR RESULTADO ROBUSTO:")
            print(f"{'='*70}")
            print(f"  TREINO: {best.train_result.trades} trades, "
                  f"WR={best.train_result.win_rate:.1%}, "
                  f"PF={best.train_result.profit_factor:.2f}")
            print(f"  TESTE:  {best.test_result.trades} trades, "
                  f"WR={best.test_result.win_rate:.1%}, "
                  f"PF={best.test_result.profit_factor:.2f}")
            print(f"  Degradacao: {best.degradation*100:.1f}%")
            print(f"{'='*70}")

            opt.save(n_tested=n_combinations)
            return True
        else:
            print("\n  AVISO: Nenhuma configuracao passou nos filtros!")
            return False
    else:
        print("\n  ERRO: Falha ao carregar dados!")
        return False


async def main():
    """Otimiza todos os indicadores restantes"""
    indicators = ['PHM', 'RHHF', 'SEED', 'SEMA', 'STGK']

    for ind in indicators:
        print("\n" + "=" * 80)
        print(f"  INICIANDO OTIMIZACAO: {ind}")
        print("=" * 80)

        try:
            await optimize_indicator(ind, n_combinations=50000)
        except Exception as e:
            print(f"  ERRO ao otimizar {ind}: {e}")
            import traceback
            traceback.print_exc()

        print("\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Otimizar indicador especifico
        asyncio.run(optimize_indicator(sys.argv[1].upper()))
    else:
        # Otimizar todos
        asyncio.run(main())
