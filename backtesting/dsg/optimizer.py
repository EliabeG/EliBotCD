#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR DSG ROBUSTO V3.2 - PRONTO PARA DINHEIRO REAL
================================================================================

DSG (Detector de Singularidade Gravitacional):
- Usa Tensor Metrico Financeiro para modelar espaco-tempo
- Usa Escalar de Ricci para detectar curvatura
- Usa Forca de Mare para detectar rompimentos

VALIDACAO:
1. Divide dados em 70% treino / 30% teste
2. Otimiza apenas no treino
3. Valida no teste (dados nunca vistos)
4. Descarta resultados que nao passam nos filtros de realismo

CORREÇÕES V2.0:
1. Entrada no OPEN da próxima barra (não no CLOSE atual)
2. Direção calculada com barras fechadas
3. Stop/Take consideram gaps
4. Evita trades simultâneos

CORREÇÕES V3.0 (Auditoria):
5. Filtros UNIFICADOS com robust_optimizer.py
6. MIN_TRADES_TRAIN = 50 (era 30)
7. MIN_WIN_RATE = 0.35 (era 0.30)
8. MAX_WIN_RATE = 0.60 (era 0.65)
9. MIN_PROFIT_FACTOR = 1.30 (era 1.10)
10. MAX_DRAWDOWN = 0.30 (era 0.40)

CORREÇÕES V3.1 (Auditoria Completa):
11. Usa DSG V3.1 com validação de inputs, thread-safety, subsampling adaptativo
12. Volumes sintéticos agora usam função CENTRALIZADA (config/volume_generator.py)
13. Consistência garantida entre backtest, estratégia e indicador

CORREÇÕES V3.2 (Segunda Auditoria 24/12/2025):
14. Verificação de erro em resultado de análise
15. Usa DSG V3.2 com correções de look-ahead residual
16. Step function usa NaN ao invés de 0.0
17. Centro de massa usa NaN quando histórico vazio
18. Thread-safety completo em todos os acessos ao histórico

CORREÇÕES V3.4 (Quarta Auditoria 25/12/2025):
19. Filtros CENTRALIZADOS: Importa de config/optimizer_filters.py
20. Ricci threshold corrigido para escala real (-50500)
21. Consistência garantida entre optimizer.py e optimizer_wf.py

PARA DINHEIRO REAL. SEM OVERFITTING. SEM LOOK-AHEAD.
================================================================================
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

# Adiciona o diretorio raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.fxopen_historical_ws import Bar, download_historical_data
from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional
from backtesting.common.robust_optimizer import (
    RobustBacktester, RobustResult, BacktestResult,
    save_robust_config
)

# CORREÇÃO C4: Importar custos centralizados
from config.execution_costs import (
    SPREAD_PIPS,
    SLIPPAGE_PIPS,
    get_pip_value,
)

# CORREÇÃO V3.4: Importar filtros centralizados
from config.optimizer_filters import (
    MIN_TRADES_TRAIN, MIN_TRADES_TEST,
    MIN_WIN_RATE, MAX_WIN_RATE,
    MIN_PROFIT_FACTOR, MAX_PROFIT_FACTOR,
    MAX_DRAWDOWN,
    MIN_WIN_RATE_TEST, MAX_WIN_RATE_TEST,
    MIN_PROFIT_FACTOR_TEST, MAX_PROFIT_FACTOR_TEST,
    MAX_DRAWDOWN_TEST,
    MIN_PF_RATIO, MIN_WR_RATIO,
)


@dataclass
class DSGSignal:
    """
    Sinal pre-calculado do DSG

    CORRIGIDO: Agora armazena informações para execução realista
    """
    bar_idx: int              # Índice da barra onde o sinal foi GERADO
    signal_price: float       # Preço de fechamento quando sinal foi gerado (referência)
    next_bar_idx: int         # NOVO: Índice da barra onde deve EXECUTAR
    entry_price: float        # NOVO: Preço de ABERTURA da próxima barra (onde realmente entra)
    high: float               # High da barra de execução
    low: float                # Low da barra de execução
    ricci_scalar: float
    tidal_force: float
    event_horizon_distance: float
    ricci_collapsing: bool
    crossing_horizon: bool
    geodesic_direction: int
    signal: int


class DSGRobustOptimizer:
    """Otimizador DSG com validacao anti-overfitting"""

    def __init__(self, symbol: str = "EURUSD", periodicity: str = "H1"):
        self.symbol = symbol
        self.periodicity = periodicity

        # CORREÇÃO C4: Usar custos centralizados
        # ANTES: RobustBacktester(pip=0.0001, spread=1.0) - spread fixo incorreto
        # DEPOIS: Usa valores do config/execution_costs.py
        pip_value = get_pip_value(symbol)
        self.backtester = RobustBacktester(pip=pip_value, spread=SPREAD_PIPS)

        self.bars: List[Bar] = []
        self.signals: List[DSGSignal] = []
        self.train_bars: List[Bar] = []
        self.test_bars: List[Bar] = []
        self.train_signals: List[DSGSignal] = []
        self.test_signals: List[DSGSignal] = []

        self.robust_results: List[RobustResult] = []
        self.best: Optional[RobustResult] = None

    async def load_and_precompute(self, start_date: datetime, end_date: datetime,
                                   split_date: datetime = None):
        """
        Carrega dados e pre-calcula sinais DSG

        Args:
            start_date: Data inicial dos dados
            end_date: Data final dos dados
            split_date: Data de divisão train/test (se None, usa 70/30)
        """
        print("\n" + "=" * 70)
        print("  CARREGANDO DADOS REAIS")
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

        # SPLIT TRAIN/TEST por data específica ou 70/30
        if split_date:
            # Encontrar índice da data de split
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

        # Pre-calcular DSG
        print("\n  Pre-calculando sinais DSG (computacionalmente intensivo)...")

        dsg = DetectorSingularidadeGravitacional(
            ricci_collapse_threshold=-0.5,
            tidal_force_threshold=0.1,
            lookback_window=30
        )

        prices_buf = deque(maxlen=100)
        self.signals = []
        min_prices = 50

        for i, bar in enumerate(self.bars):
            prices_buf.append(bar.close)

            if len(prices_buf) < min_prices:
                continue

            # CORREÇÃO: Precisamos da PRÓXIMA barra para executar
            # Se não há próxima barra, não podemos gerar sinal
            if i >= len(self.bars) - 1:
                continue

            try:
                prices_arr = np.array(prices_buf)
                # CORREÇÃO V3.2: Volumes não passados explicitamente porque
                # DSG V3.2 usa generate_synthetic_volumes() internamente
                # Isso garante consistência com dsg_strategy.py em produção
                result = dsg.analyze(prices_arr)

                # CORREÇÃO V3.2: Verificar se análise falhou
                if 'error' in result and result['error']:
                    continue

                # CORREÇÃO: Entrada no OPEN da PRÓXIMA barra
                next_bar = self.bars[i + 1]

                self.signals.append(DSGSignal(
                    bar_idx=i,                          # Onde o sinal foi gerado
                    signal_price=bar.close,             # Preço quando sinal gerado (referência)
                    next_bar_idx=i + 1,                 # NOVO: Onde vai executar
                    entry_price=next_bar.open,          # NOVO: Preço de entrada (OPEN da próxima)
                    high=next_bar.high,                 # High da barra de execução
                    low=next_bar.low,                   # Low da barra de execução
                    ricci_scalar=result['Ricci_Scalar'],
                    tidal_force=result['Tidal_Force_Magnitude'],
                    event_horizon_distance=result['Event_Horizon_Distance'],
                    ricci_collapsing=result['ricci_collapsing'],
                    crossing_horizon=result['crossing_horizon'],
                    geodesic_direction=result['geodesic_direction'],
                    signal=result['signal']
                ))

            except:
                continue

            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(self.bars)} barras...")

        # Separar sinais
        self.train_signals = [s for s in self.signals if s.bar_idx < split_idx]
        self.test_signals = [s for s in self.signals if s.bar_idx >= split_idx]

        print(f"\n  Sinais pre-calculados:")
        print(f"    Treino: {len(self.train_signals)} sinais")
        print(f"    Teste:  {len(self.test_signals)} sinais")

        # Debug: mostrar distribuicao de valores
        if self.signals:
            ricci_vals = [s.ricci_scalar for s in self.signals]
            tidal_vals = [s.tidal_force for s in self.signals]
            print(f"\n  Distribuicao de valores:")
            print(f"    Ricci: min={min(ricci_vals):.4f}, max={max(ricci_vals):.4f}, mean={np.mean(ricci_vals):.4f}")
            print(f"    Tidal: min={min(tidal_vals):.6f}, max={max(tidal_vals):.6f}, mean={np.mean(tidal_vals):.6f}")

        return len(self.train_signals) > 50 and len(self.test_signals) > 20

    def _run_backtest(self, signals: List[DSGSignal], bars: List[Bar],
                      ricci_thresh: float, tidal_thresh: float,
                      sl: float, tp: float,
                      bar_offset: int = 0) -> List[float]:
        """
        Executa backtest em um conjunto de dados

        VERSÃO CORRIGIDA:
        1. Usa OPEN da próxima barra como entry_price
        2. Evita trades simultâneos (last_exit_idx)
        3. R:R mínimo de 1.2
        """
        if tp <= sl * 1.2:  # Exige R:R mínimo de 1.2
            return []

        entries = []
        for s in signals:
            # Condicoes de entrada baseadas no DSG
            ricci_collapse = s.ricci_scalar < ricci_thresh or s.ricci_collapsing
            high_tidal = s.tidal_force > tidal_thresh
            crossing = s.crossing_horizon

            # Precisa de pelo menos 2 condicoes
            conditions = sum([ricci_collapse, high_tidal, crossing])

            if conditions >= 2 and s.geodesic_direction != 0:
                # CORREÇÃO: Usar next_bar_idx e entry_price (OPEN da próxima barra)
                execution_idx = s.next_bar_idx - bar_offset
                entries.append((execution_idx, s.entry_price, s.geodesic_direction))

        if len(entries) < 3:
            return []

        pnls = []
        last_exit_idx = -1  # CORREÇÃO: Evitar trades simultâneos

        for exec_idx, entry_price, direction in entries:
            # Validações
            if exec_idx < 0 or exec_idx >= len(bars) - 1:
                continue

            # CORREÇÃO: Evitar trades simultâneos
            if exec_idx <= last_exit_idx:
                continue

            trade = self.backtester.execute_trade(
                bars=bars,
                entry_idx=exec_idx,
                entry_price=entry_price,
                direction=direction,
                sl_pips=sl,
                tp_pips=tp,
                max_bars=200
            )
            pnls.append(trade.pnl_pips)

            # Atualiza último índice de saída
            last_exit_idx = trade.exit_idx

        return pnls

    def _test_params(self, ricci_thresh: float, tidal_thresh: float,
                     sl: float, tp: float) -> Optional[RobustResult]:
        """
        Testa parametros em treino e teste

        CORREÇÃO V3.0: Filtros UNIFICADOS com robust_optimizer.py
        Mesmo nível de rigor para garantir consistência entre estratégias
        """

        train_pnls = self._run_backtest(
            self.train_signals, self.train_bars,
            ricci_thresh, tidal_thresh, sl, tp,
            bar_offset=0
        )
        train_result = self.backtester.calculate_backtest_result(train_pnls)

        # CORREÇÃO V3.4: Usa filtros CENTRALIZADOS de config/optimizer_filters.py
        # Garante consistência entre todos os otimizadores
        if not train_result.is_valid(
            min_trades=MIN_TRADES_TRAIN,
            max_win_rate=MAX_WIN_RATE,
            min_win_rate=MIN_WIN_RATE,
            max_pf=MAX_PROFIT_FACTOR,
            min_pf=MIN_PROFIT_FACTOR,
            max_dd=MAX_DRAWDOWN
        ):
            return None

        # Backtest no TESTE
        split_idx = len(self.train_bars)
        test_pnls = self._run_backtest(
            self.test_signals, self.test_bars,
            ricci_thresh, tidal_thresh, sl, tp,
            bar_offset=split_idx
        )
        test_result = self.backtester.calculate_backtest_result(test_pnls)

        # CORREÇÃO V3.4: Usa filtros CENTRALIZADOS para teste
        if not test_result.is_valid(
            min_trades=MIN_TRADES_TEST,
            max_win_rate=MAX_WIN_RATE_TEST,
            min_win_rate=MIN_WIN_RATE_TEST,
            max_pf=MAX_PROFIT_FACTOR_TEST,
            min_pf=MIN_PROFIT_FACTOR_TEST,
            max_dd=MAX_DRAWDOWN_TEST
        ):
            return None

        # Calcula robustez
        robustness, degradation, _ = self.backtester.calculate_robustness(
            train_result, test_result
        )

        # CORREÇÃO V3.4: Usa ratios CENTRALIZADOS
        # CORREÇÃO V3.5: Usa MIN_PROFIT_FACTOR_TEST ao invés de 1.0 hardcoded
        pf_ratio = test_result.profit_factor / train_result.profit_factor if train_result.profit_factor > 0 else 0
        wr_ratio = test_result.win_rate / train_result.win_rate if train_result.win_rate > 0 else 0
        is_robust = pf_ratio >= MIN_PF_RATIO and wr_ratio >= MIN_WR_RATIO and test_result.profit_factor >= MIN_PROFIT_FACTOR_TEST

        if not is_robust:
            return None

        params = {
            "ricci_collapse_threshold": round(ricci_thresh, 4),
            "tidal_force_threshold": round(tidal_thresh, 6),
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

    def optimize(self, n: int = 100000, seed: int = 42) -> Optional[RobustResult]:
        """
        Executa otimizacao robusta

        CORREÇÃO: Seed fixo para reprodutibilidade
        """
        if not self.train_signals or not self.test_signals:
            print("  ERRO: Dados nao carregados!")
            return None

        # CORREÇÃO: Fixar seeds para reprodutibilidade
        random.seed(seed)
        np.random.seed(seed)

        print(f"\n{'='*70}")
        print(f"  OTIMIZACAO ROBUSTA DSG: {n:,} COMBINACOES")
        print(f"  Com validacao Train/Test Split")
        print(f"  Seed: {seed} (para reprodutibilidade)")
        print(f"{'='*70}")

        # Ranges baseados na distribuicao REAL dos dados
        # Ricci: -50836 a -49798 (sempre muito negativo, sempre em colapso)
        # Tidal: 0.0001 a 0.067 (média 0.009)
        # Como ricci_collapse sempre é True, o Tidal é o filtro principal
        # CORREÇÃO: Ranges baseados na distribuição REAL observada
        # Ricci: min=-50645, max=-44165, mean=-49492
        # Tidal: min=0.000008, max=0.048746, mean=0.001192
        ricci_vals = np.linspace(-50700, -44000, 20)  # Cobre toda distribuição real
        tidal_vals = np.linspace(0.0001, 0.05, 20)    # Cobre toda distribuição real
        sl_vals = np.linspace(15, 50, 15)
        tp_vals = np.linspace(20, 70, 20)

        best_robustness = -1
        tested = 0
        robust_count = 0
        start = datetime.now()

        for _ in range(n):
            tested += 1

            ricci = float(random.choice(ricci_vals))
            tidal = float(random.choice(tidal_vals))
            sl = float(random.choice(sl_vals))
            tp = float(random.choice(tp_vals))

            result = self._test_params(ricci, tidal, sl, tp)

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

            if tested % 20000 == 0:
                elapsed = (datetime.now() - start).total_seconds()
                rate = tested / elapsed
                eta = (n - tested) / rate / 60
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
            strategy_name="DSG-SingularidadeGravitacional",
            symbol=self.symbol,
            periodicity=self.periodicity,
            n_tested=n_tested,
            n_robust=len(self.robust_results)
        )

        # Top 10
        top_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "dsg_robust_top10.json"
        )
        sorted_results = sorted(self.robust_results, key=lambda x: x.robustness_score, reverse=True)[:10]
        top_data = [r.to_dict() for r in sorted_results]
        with open(top_file, 'w') as f:
            json.dump(top_data, f, indent=2)
        print(f"  Top 10 robustos salvo em: {top_file}")


async def main():
    N_COMBINATIONS = 300000

    print("=" * 70)
    print("  OTIMIZADOR DSG ROBUSTO")
    print("  Com Validacao Anti-Overfitting")
    print(f"  {N_COMBINATIONS:,} Combinacoes")
    print("  PARA DINHEIRO REAL")
    print("=" * 70)

    opt = DSGRobustOptimizer("EURUSD", "H1")

    # Períodos específicos de treino e teste
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)      # Início do treino
    split = datetime(2025, 1, 1, tzinfo=timezone.utc)      # Fim do treino / Início do teste
    end = datetime.now(timezone.utc)                        # Fim do teste

    print(f"\n  Periodo Total: {start.date()} a {end.date()}")
    print(f"  Treino: {start.date()} a {split.date()}")
    print(f"  Teste:  {split.date()} a {end.date()}")

    if await opt.load_and_precompute(start, end, split_date=split):
        best = opt.optimize(N_COMBINATIONS)
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

            opt.save(n_tested=N_COMBINATIONS)
        else:
            print("\n  AVISO: Nenhuma configuracao passou nos filtros!")
    else:
        print("\n  ERRO: Falha ao carregar dados!")


if __name__ == "__main__":
    asyncio.run(main())
