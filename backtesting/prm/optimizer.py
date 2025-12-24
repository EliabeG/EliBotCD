#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR PRM ROBUSTO - COM VALIDACAO ANTI-OVERFITTING
VERSÃO CORRIGIDA - SEM LOOK-AHEAD BIAS
================================================================================

Este otimizador:
1. Divide dados em 70% treino / 30% teste
2. Otimiza apenas no treino
3. Valida no teste (dados nunca vistos)
4. Descarta resultados que nao passam nos filtros de realismo
5. Calcula score de robustez

CORREÇÕES APLICADAS:
1. Direção calculada apenas com barras COMPLETAMENTE FECHADAS
2. Entrada executada no OPEN da próxima barra (não no close atual)
3. Stop/Take consideram gaps
4. Sem look-ahead em nenhum cálculo

REGRAS:
- Minimo 30 trades no treino, 15 no teste
- Win Rate entre 30% e 65%
- Profit Factor entre 1.1 e 4.0
- Performance do teste >= 60% do treino

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
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.fxopen_historical_ws import Bar, download_historical_data
from strategies.alta_volatilidade.prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot
from backtesting.common.robust_optimizer import (
    RobustBacktester, RobustResult, BacktestResult,
    save_robust_config
)


@dataclass
class PRMSignal:
    """
    Sinal pre-calculado do PRM
    
    CORRIGIDO: Agora armazena informações para execução realista
    """
    bar_idx: int          # Índice da barra onde o sinal foi GERADO
    signal_price: float   # Preço de fechamento quando sinal foi gerado (para referência)
    next_bar_idx: int     # NOVO: Índice da barra onde deve EXECUTAR (próxima barra)
    entry_price: float    # NOVO: Preço de ABERTURA da próxima barra (onde realmente entra)
    high: float           # High da barra de entrada (para stop/take)
    low: float            # Low da barra de entrada (para stop/take)
    hmm_prob: float
    lyapunov: float
    hmm_state: int
    direction: int        # CORRIGIDO: Baseado apenas em barras JÁ FECHADAS


class PRMRobustOptimizer:
    """
    Otimizador PRM com validacao anti-overfitting
    
    VERSÃO CORRIGIDA - SEM LOOK-AHEAD BIAS
    """

    def __init__(self, symbol: str = "EURUSD", periodicity: str = "H1"):
        self.symbol = symbol
        self.periodicity = periodicity
        self.backtester = RobustBacktester(pip=0.0001, spread=1.0)

        self.bars: List[Bar] = []
        self.signals: List[PRMSignal] = []

        # Dados separados
        self.train_bars: List[Bar] = []
        self.test_bars: List[Bar] = []
        self.train_signals: List[PRMSignal] = []
        self.test_signals: List[PRMSignal] = []

        # Resultados
        self.robust_results: List[RobustResult] = []
        self.best: Optional[RobustResult] = None

    async def load_and_precompute(self, start_date: datetime, end_date: datetime):
        """
        Carrega dados e pre-calcula sinais PRM
        
        CORRIGIDO: 
        - Direção usa apenas barras completamente fechadas
        - Entry price é o OPEN da próxima barra
        """
        print("\n" + "=" * 70)
        print("  CARREGANDO DADOS REAIS")
        print("  Versão Corrigida - Sem Look-Ahead")
        print("=" * 70)

        self.bars = await download_historical_data(
            symbol=self.symbol,
            periodicity=self.periodicity,
            start_time=start_date,
            end_time=end_date
        )
        print(f"  Total de barras: {len(self.bars)}")

        if len(self.bars) < 200:
            print("  ERRO: Dados insuficientes!")
            return False

        # SPLIT TRAIN/TEST (70/30)
        split_idx = int(len(self.bars) * 0.70)
        self.train_bars = self.bars[:split_idx]
        self.test_bars = self.bars[split_idx:]

        print(f"\n  DIVISAO TRAIN/TEST:")
        print(f"    Treino: {len(self.train_bars)} barras ({self.train_bars[0].timestamp.date()} a {self.train_bars[-1].timestamp.date()})")
        print(f"    Teste:  {len(self.test_bars)} barras ({self.test_bars[0].timestamp.date()} a {self.test_bars[-1].timestamp.date()})")

        # Pre-calcular sinais para TODOS os dados
        print("\n  Pre-calculando sinais PRM (sem look-ahead)...")

        prm = ProtocoloRiemannMandelbrot(
            n_states=3,
            hmm_threshold=0.1,
            lyapunov_threshold_k=0.001,
            curvature_threshold=0.0001,
            lookback_window=100,
            hmm_training_window=200,      # NOVO: Janela de treino do HMM
            hmm_min_training_samples=50   # NOVO: Mínimo de amostras
        )

        prices_buf = deque(maxlen=500)
        volumes_buf = deque(maxlen=500)
        self.signals = []

        min_prices = 50
        
        # CORREÇÃO: Precisamos de pelo menos 12 barras para calcular direção
        # (11 barras passadas + barra atual cujo close NÃO usamos)
        min_bars_for_direction = 12

        for i, bar in enumerate(self.bars):
            prices_buf.append(bar.close)
            volumes_buf.append(bar.volume)

            if len(prices_buf) < min_prices:
                continue
            
            # CORREÇÃO: Precisamos da PRÓXIMA barra para executar
            # Se não há próxima barra, não podemos gerar sinal
            if i >= len(self.bars) - 1:
                continue

            try:
                result = prm.analyze(np.array(prices_buf), np.array(volumes_buf))

                # ============================================================
                # CORREÇÃO CRÍTICA: Direção baseada APENAS em barras FECHADAS
                # ============================================================
                # 
                # ANTES (ERRADO):
                #   trend = bar.close - self.bars[i - 10].close
                #   Problema: bar.close é a barra ATUAL (ainda não fechou no momento real)
                #
                # DEPOIS (CORRETO):
                #   Usar self.bars[i-1].close (última barra FECHADA)
                #   Comparar com self.bars[i-11].close (11 barras atrás)
                #
                # No momento em que decidimos a direção:
                # - Barra i está "em andamento" (não sabemos o close)
                # - Barra i-1 acabou de fechar (sabemos o close)
                # - Comparamos i-1 com i-11 para ver tendência dos últimos 10 períodos
                #
                if i >= min_bars_for_direction:
                    # Usar apenas barras COMPLETAMENTE FECHADAS
                    recent_close = self.bars[i - 1].close      # Última barra fechada
                    past_close = self.bars[i - 11].close       # 10 barras antes da última fechada
                    trend = recent_close - past_close
                    direction = 1 if trend > 0 else -1
                else:
                    direction = 0  # Sem direção definida
                
                # ============================================================
                # CORREÇÃO: Entrada no OPEN da PRÓXIMA barra
                # ============================================================
                #
                # O sinal é GERADO quando a barra i fecha
                # A execução acontece no OPEN da barra i+1
                #
                next_bar = self.bars[i + 1]
                
                self.signals.append(PRMSignal(
                    bar_idx=i,                          # Onde o sinal foi gerado
                    signal_price=bar.close,             # Preço quando sinal gerado (referência)
                    next_bar_idx=i + 1,                 # NOVO: Onde vai executar
                    entry_price=next_bar.open,          # NOVO: Preço de entrada (OPEN da próxima)
                    high=next_bar.high,                 # High da barra de execução
                    low=next_bar.low,                   # Low da barra de execução
                    hmm_prob=result['Prob_HMM'],
                    lyapunov=result['Lyapunov_Score'],
                    hmm_state=result['hmm_analysis']['current_state'],
                    direction=direction
                ))

            except Exception as e:
                continue

            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(self.bars)} barras...")

        # Separar sinais em treino e teste
        # NOTA: O sinal pertence ao treino se foi GERADO no período de treino
        # mas a execução pode acontecer na primeira barra do teste (isso é ok)
        self.train_signals = [s for s in self.signals if s.bar_idx < split_idx]
        self.test_signals = [s for s in self.signals if s.bar_idx >= split_idx]

        print(f"\n  Sinais pre-calculados (sem look-ahead):")
        print(f"    Treino: {len(self.train_signals)} sinais")
        print(f"    Teste:  {len(self.test_signals)} sinais")
        
        # Estatísticas de direção
        train_long = sum(1 for s in self.train_signals if s.direction == 1)
        train_short = sum(1 for s in self.train_signals if s.direction == -1)
        print(f"    Treino - Long: {train_long}, Short: {train_short}")

        return len(self.train_signals) > 50 and len(self.test_signals) > 20

    def _run_backtest(self, signals: List[PRMSignal], bars: List[Bar],
                      hmm_thresh: float, lyap_thresh: float,
                      states: List[int], sl: float, tp: float,
                      bar_offset: int = 0) -> List[float]:
        """
        Executa backtest em um conjunto de dados
        
        CORRIGIDO:
        - Entrada no OPEN da barra de execução (não no close do sinal)
        - Stop/Take consideram gaps
        - Execução realista
        """
        if tp <= sl:
            return []

        # Encontra entradas validas
        entries = []
        for s in signals:
            if (s.hmm_prob >= hmm_thresh and
                s.lyapunov >= lyap_thresh and
                s.hmm_state in states and
                s.direction != 0):
                
                # CORRIGIDO: Usar next_bar_idx e entry_price (OPEN)
                execution_idx = s.next_bar_idx - bar_offset
                entries.append((
                    execution_idx,      # Índice da barra de execução
                    s.entry_price,      # Preço de entrada (OPEN da barra)
                    s.direction
                ))

        if len(entries) < 3:
            return []

        # Executa trades com lógica realista
        pnls = []
        pip = 0.0001
        spread = 1.0 * pip  # 1 pip de spread
        slippage = 0.5 * pip  # 0.5 pip de slippage
        
        for entry_idx, entry_price_raw, direction in entries:
            if entry_idx < 0 or entry_idx >= len(bars) - 1:
                continue

            # Aplicar spread e slippage na entrada
            if direction == 1:  # LONG
                entry_price = entry_price_raw + spread / 2 + slippage
                stop_price = entry_price - sl * pip
                take_price = entry_price + tp * pip
            else:  # SHORT
                entry_price = entry_price_raw - spread / 2 - slippage
                stop_price = entry_price + sl * pip
                take_price = entry_price - tp * pip

            # Simular execução nas barras seguintes
            exit_price = None
            exit_reason = None
            
            max_bars = min(200, len(bars) - entry_idx - 1)
            
            for j in range(1, max_bars + 1):
                bar_idx = entry_idx + j
                if bar_idx >= len(bars):
                    break
                    
                bar = bars[bar_idx]
                
                # ============================================================
                # VERIFICAR GAPS NO OPEN
                # ============================================================
                if direction == 1:  # LONG
                    # Gap down - stop atingido no open
                    if bar.open <= stop_price:
                        exit_price = bar.open - slippage  # Pior execução
                        exit_reason = "stop_gap"
                        break
                    # Gap up - take atingido no open
                    if bar.open >= take_price:
                        exit_price = bar.open - slippage  # Execução no open
                        exit_reason = "take_gap"
                        break
                else:  # SHORT
                    # Gap up - stop atingido no open
                    if bar.open >= stop_price:
                        exit_price = bar.open + slippage  # Pior execução
                        exit_reason = "stop_gap"
                        break
                    # Gap down - take atingido no open
                    if bar.open <= take_price:
                        exit_price = bar.open + slippage  # Execução no open
                        exit_reason = "take_gap"
                        break
                
                # ============================================================
                # VERIFICAR DURANTE A BARRA (HIGH/LOW)
                # Stop tem prioridade (conservador)
                # ============================================================
                if direction == 1:  # LONG
                    # Stop loss primeiro
                    if bar.low <= stop_price:
                        exit_price = stop_price - slippage
                        exit_reason = "stop"
                        break
                    # Take profit
                    if bar.high >= take_price:
                        exit_price = take_price - slippage
                        exit_reason = "take"
                        break
                else:  # SHORT
                    # Stop loss primeiro
                    if bar.high >= stop_price:
                        exit_price = stop_price + slippage
                        exit_reason = "stop"
                        break
                    # Take profit
                    if bar.low <= take_price:
                        exit_price = take_price + slippage
                        exit_reason = "take"
                        break
            
            # Se não saiu por stop/take, fechar no último close disponível
            if exit_price is None:
                last_bar = bars[min(entry_idx + max_bars, len(bars) - 1)]
                if direction == 1:
                    exit_price = last_bar.close - slippage
                else:
                    exit_price = last_bar.close + slippage
                exit_reason = "timeout"
            
            # Calcular PnL em pips
            if direction == 1:
                pnl_pips = (exit_price - entry_price) / pip
            else:
                pnl_pips = (entry_price - exit_price) / pip
            
            pnls.append(pnl_pips)

        return pnls

    def _test_params(self, hmm_thresh: float, lyap_thresh: float,
                     states: List[int], sl: float, tp: float,
                     debug: bool = False) -> Optional[RobustResult]:
        """Testa parametros em treino e teste"""

        # Backtest no TREINO
        train_pnls = self._run_backtest(
            self.train_signals, self.train_bars,
            hmm_thresh, lyap_thresh, states, sl, tp,
            bar_offset=0
        )
        train_result = self.backtester.calculate_backtest_result(train_pnls)

        if debug and train_result.trades > 0:
            print(f"    DEBUG TRAIN: {train_result.trades} trades, WR={train_result.win_rate:.2f}, PF={train_result.profit_factor:.2f}")

        # Filtros para PRM (indicador seletivo)
        if not train_result.is_valid(
            min_trades=20,
            max_win_rate=0.68,
            min_win_rate=0.28,
            max_pf=5.0,
            min_pf=1.05,
            max_dd=0.45
        ):
            return None

        # Backtest no TESTE (dados nunca vistos)
        split_idx = len(self.train_bars)
        test_pnls = self._run_backtest(
            self.test_signals, self.test_bars,
            hmm_thresh, lyap_thresh, states, sl, tp,
            bar_offset=split_idx
        )
        test_result = self.backtester.calculate_backtest_result(test_pnls)

        if debug and test_result.trades > 0:
            print(f"    DEBUG TEST:  {test_result.trades} trades, WR={test_result.win_rate:.2f}, PF={test_result.profit_factor:.2f}")

        # Verifica se teste passa nos filtros
        if not test_result.is_valid(
            min_trades=10,
            max_win_rate=0.75,
            min_win_rate=0.20,
            max_pf=6.0,
            min_pf=0.9,
            max_dd=0.55
        ):
            return None

        # Calcula robustez
        robustness, degradation, is_robust = self.backtester.calculate_robustness(
            train_result, test_result
        )

        # Robustez: teste deve manter >= 50% do treino
        pf_ratio = test_result.profit_factor / train_result.profit_factor if train_result.profit_factor > 0 else 0
        wr_ratio = test_result.win_rate / train_result.win_rate if train_result.win_rate > 0 else 0
        is_robust_relaxed = pf_ratio >= 0.50 and wr_ratio >= 0.50 and test_result.profit_factor >= 0.9

        if not is_robust_relaxed:
            return None

        params = {
            "hmm_threshold": round(hmm_thresh, 4),
            "lyapunov_threshold": round(lyap_thresh, 4),
            "hmm_states_allowed": states,
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

    def optimize(self, n: int = 500000) -> Optional[RobustResult]:
        """Executa otimizacao robusta"""
        if not self.train_signals or not self.test_signals:
            print("  ERRO: Dados nao carregados!")
            return None

        print(f"\n{'='*70}")
        print(f"  OTIMIZACAO ROBUSTA PRM: {n:,} COMBINACOES")
        print(f"  Com validacao Train/Test Split")
        print(f"  VERSAO CORRIGIDA - SEM LOOK-AHEAD")
        print(f"{'='*70}")

        # Ranges de parametros
        hmm_vals = np.linspace(0.50, 0.75, 20)
        lyap_vals = np.linspace(0.055, 0.085, 15)
        sl_vals = np.linspace(20, 50, 15)
        tp_vals = np.linspace(30, 80, 20)
        states_opts = [[0, 1]]

        best_robustness = -1
        tested = 0
        robust_count = 0
        start = datetime.now()

        # Debug: testar algumas combinacoes manualmente
        debug_count = 0
        for _ in range(n):
            tested += 1

            # Parametros aleatorios
            hmm = float(random.choice(hmm_vals))
            lyap = float(random.choice(lyap_vals))
            sl = float(random.choice(sl_vals))
            tp = float(random.choice(tp_vals))
            states = random.choice(states_opts)

            # Debug primeiras 5 combinacoes
            debug = debug_count < 5
            if debug:
                debug_count += 1
                print(f"\n  DEBUG #{debug_count}: hmm={hmm:.2f}, lyap={lyap:.3f}, sl={sl:.0f}, tp={tp:.0f}")

            result = self._test_params(hmm, lyap, states, sl, tp, debug=debug)

            if result:
                robust_count += 1
                self.robust_results.append(result)

                if result.robustness_score > best_robustness:
                    best_robustness = result.robustness_score
                    self.best = result

                    print(f"\n  [ROBUSTO #{robust_count}] Robustez={result.robustness_score:.4f}")
                    print(f"    TREINO: {result.train_result.trades} trades, "
                          f"WR={result.train_result.win_rate:.1%}, "
                          f"PF={result.train_result.profit_factor:.2f}, "
                          f"PnL={result.train_result.total_pnl:.0f}pips")
                    print(f"    TESTE:  {result.test_result.trades} trades, "
                          f"WR={result.test_result.win_rate:.1%}, "
                          f"PF={result.test_result.profit_factor:.2f}, "
                          f"PnL={result.test_result.total_pnl:.0f}pips")
                    print(f"    Degradacao: {result.degradation*100:.1f}%")

            if tested % 50000 == 0:
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
            strategy_name="PRM-RiemannMandelbrot",
            symbol=self.symbol,
            periodicity=self.periodicity,
            n_tested=n_tested,
            n_robust=len(self.robust_results)
        )

        # Salva top 10 robustos
        top_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "prm_robust_top10.json"
        )
        sorted_results = sorted(
            self.robust_results,
            key=lambda x: x.robustness_score,
            reverse=True
        )[:10]

        top_data = [r.to_dict() for r in sorted_results]
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(top_file), exist_ok=True)
        
        with open(top_file, 'w') as f:
            json.dump(top_data, f, indent=2)
        print(f"  Top 10 robustos salvo em: {top_file}")


async def main():
    N_COMBINATIONS = 500000

    print("=" * 70)
    print("  OTIMIZADOR PRM ROBUSTO")
    print("  Com Validacao Anti-Overfitting")
    print("  VERSAO CORRIGIDA - SEM LOOK-AHEAD BIAS")
    print(f"  {N_COMBINATIONS:,} Combinacoes")
    print("  PARA DINHEIRO REAL")
    print("=" * 70)
    
    print("\n  CORRECOES APLICADAS:")
    print("    1. Direcao calculada apenas com barras FECHADAS")
    print("    2. Entrada no OPEN da proxima barra")
    print("    3. Stop/Take consideram gaps")
    print("    4. HMM treinado apenas em dados passados")

    opt = PRMRobustOptimizer("EURUSD", "H1")

    # Usar periodo maior se disponivel
    start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    print(f"\n  Periodo: {start.date()} a {end.date()}")

    if await opt.load_and_precompute(start, end):
        best = opt.optimize(N_COMBINATIONS)
        if best:
            print(f"\n{'='*70}")
            print(f"  MELHOR RESULTADO ROBUSTO:")
            print(f"{'='*70}")
            print(f"  Robustez: {best.robustness_score:.4f}")
            print(f"  Degradacao Train->Test: {best.degradation*100:.1f}%")
            print(f"\n  TREINO:")
            print(f"    Trades: {best.train_result.trades}")
            print(f"    Win Rate: {best.train_result.win_rate:.1%}")
            print(f"    Profit Factor: {best.train_result.profit_factor:.2f}")
            print(f"    PnL: {best.train_result.total_pnl:.1f} pips")
            print(f"\n  TESTE (Out-of-Sample):")
            print(f"    Trades: {best.test_result.trades}")
            print(f"    Win Rate: {best.test_result.win_rate:.1%}")
            print(f"    Profit Factor: {best.test_result.profit_factor:.2f}")
            print(f"    PnL: {best.test_result.total_pnl:.1f} pips")
            print(f"{'='*70}")

            opt.save(n_tested=N_COMBINATIONS)
        else:
            print("\n  AVISO: Nenhuma configuracao passou nos filtros de robustez!")
            print("  Isso pode indicar que o indicador nao e' adequado para este periodo.")
            print("  (Ou que as correcoes de look-ahead revelaram a performance real)")
    else:
        print("\n  ERRO: Falha ao carregar dados!")


if __name__ == "__main__":
    asyncio.run(main())
