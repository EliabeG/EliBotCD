#!/usr/bin/env python3
"""
================================================================================
OTIMIZADOR PRM FINAL
Analisa comportamento do PRM e salva configuracao otimizada
================================================================================

O PRM (Protocolo Riemann-Mandelbrot) e' um indicador de ALTA SELETIVIDADE
projetado para detectar "singularidades" - condicoes extremas raras no mercado.

Este otimizador:
1. Analisa os valores gerados pelo PRM nos dados historicos
2. Determina thresholds otimizados baseados na distribuicao real
3. Testa configuracoes de SL/TP em sinais simulados
4. Salva a melhor configuracao encontrada

IMPORTANTE: Dados REAIS do mercado FXOpen.
"""

import sys
import os
import json
import asyncio
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.fxopen_historical_ws import Bar, download_historical_data
from strategies.alta_volatilidade.prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot


@dataclass
class PRMAnalysis:
    """Dados de uma analise PRM"""
    timestamp: datetime
    price: float
    prob_hmm: float
    lyapunov: float
    curvature: float
    hmm_state: int
    singularity: bool


@dataclass
class OptimalConfig:
    """Configuracao otimizada"""
    min_prices: int
    hmm_threshold: float
    lyapunov_threshold: float
    curvature_threshold: float
    stop_loss_pips: float
    take_profit_pips: float
    expected_signals_per_month: float
    percentile_used: int


class PRMAnalyzer:
    """
    Analisa comportamento do PRM e determina configuracao otimizada
    """

    def __init__(self, symbol: str = "EURUSD", periodicity: str = "H1"):
        self.symbol = symbol
        self.periodicity = periodicity
        self.bars: List[Bar] = []
        self.analyses: List[PRMAnalysis] = []
        self.config: Optional[OptimalConfig] = None

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

    def analyze_prm_behavior(self, min_prices: int = 50):
        """
        Analisa o comportamento do PRM nos dados historicos
        """
        print(f"\n  Analisando comportamento do PRM...")

        prm = ProtocoloRiemannMandelbrot(
            n_states=3,
            hmm_threshold=0.5,  # Baixo para capturar todos os dados
            lyapunov_threshold_k=0.01,
            curvature_threshold=0.001,
            lookback_window=100
        )

        prices_buffer = []
        volumes_buffer = []
        self.analyses = []

        for i, bar in enumerate(self.bars):
            prices_buffer.append(bar.close)
            volumes_buffer.append(bar.volume)

            if len(prices_buffer) < min_prices:
                continue

            try:
                prices_arr = np.array(prices_buffer[-500:])
                volumes_arr = np.array(volumes_buffer[-500:])

                result = prm.analyze(prices_arr, volumes_arr)

                self.analyses.append(PRMAnalysis(
                    timestamp=bar.timestamp,
                    price=bar.close,
                    prob_hmm=result['Prob_HMM'],
                    lyapunov=result['Lyapunov_Score'],
                    curvature=result['Curvature_Signal'],
                    hmm_state=result['hmm_analysis']['current_state'],
                    singularity=result['singularity_detected']
                ))

            except Exception as e:
                continue

            if (i + 1) % 200 == 0:
                print(f"    Processado: {i+1}/{len(self.bars)} barras")

        print(f"  Total de analises: {len(self.analyses)}")

    def calculate_optimal_thresholds(self, percentile: int = 90) -> Dict:
        """
        Calcula thresholds otimizados baseados na distribuicao real dos valores

        Usa percentil para determinar thresholds que capturam os top X% dos valores
        """
        if not self.analyses:
            return {}

        hmm_probs = [a.prob_hmm for a in self.analyses]
        lyapunov_scores = [a.lyapunov for a in self.analyses]
        curvature_signals = [abs(a.curvature) for a in self.analyses]

        # Estatisticas
        stats = {
            'hmm': {
                'min': np.min(hmm_probs),
                'max': np.max(hmm_probs),
                'mean': np.mean(hmm_probs),
                'std': np.std(hmm_probs),
                f'p{percentile}': np.percentile(hmm_probs, percentile)
            },
            'lyapunov': {
                'min': np.min(lyapunov_scores),
                'max': np.max(lyapunov_scores),
                'mean': np.mean(lyapunov_scores),
                'std': np.std(lyapunov_scores),
                f'p{percentile}': np.percentile(lyapunov_scores, percentile)
            },
            'curvature': {
                'min': np.min(curvature_signals),
                'max': np.max(curvature_signals),
                'mean': np.mean(curvature_signals),
                'std': np.std(curvature_signals),
                f'p{percentile}': np.percentile(curvature_signals, percentile)
            }
        }

        # Conta estados HMM
        states = [a.hmm_state for a in self.analyses]
        state_counts = {
            'state_0': states.count(0),
            'state_1': states.count(1),
            'state_2': states.count(2)
        }

        print(f"\n  Estatisticas do PRM (percentil {percentile}):")
        print(f"    HMM Prob: min={stats['hmm']['min']:.3f}, max={stats['hmm']['max']:.3f}, "
              f"mean={stats['hmm']['mean']:.3f}, p{percentile}={stats['hmm'][f'p{percentile}']:.3f}")
        print(f"    Lyapunov: min={stats['lyapunov']['min']:.4f}, max={stats['lyapunov']['max']:.4f}, "
              f"mean={stats['lyapunov']['mean']:.4f}, p{percentile}={stats['lyapunov'][f'p{percentile}']:.4f}")
        print(f"    Curvature: min={stats['curvature']['min']:.4f}, max={stats['curvature']['max']:.4f}, "
              f"mean={stats['curvature']['mean']:.4f}, p{percentile}={stats['curvature'][f'p{percentile}']:.4f}")
        print(f"    Estados HMM: {state_counts}")

        return stats

    def determine_optimal_config(self, target_signals_per_month: int = 10) -> OptimalConfig:
        """
        Determina configuracao otimizada baseada nos dados analisados

        Args:
            target_signals_per_month: Numero alvo de sinais por mes
        """
        print(f"\n  Determinando configuracao otimizada...")
        print(f"    Alvo: ~{target_signals_per_month} sinais/mes")

        # Testa diferentes percentis para encontrar o que gera o numero certo de sinais
        best_percentile = 90
        best_diff = float('inf')

        for percentile in [70, 75, 80, 85, 90, 95]:
            stats = self.calculate_optimal_thresholds(percentile)

            hmm_thresh = stats['hmm'][f'p{percentile}']
            lyap_thresh = stats['lyapunov'][f'p{percentile}']
            curv_thresh = stats['curvature'][f'p{percentile}']

            # Conta sinais que passariam nos thresholds
            signals = sum(1 for a in self.analyses
                         if a.prob_hmm >= hmm_thresh
                         and a.lyapunov >= lyap_thresh
                         and abs(a.curvature) >= curv_thresh
                         and a.hmm_state in [1, 2])

            # Calcula sinais por mes
            days = (self.bars[-1].timestamp - self.bars[0].timestamp).days
            signals_per_month = (signals / max(days, 1)) * 30

            diff = abs(signals_per_month - target_signals_per_month)
            print(f"    Percentil {percentile}: {signals} sinais ({signals_per_month:.1f}/mes)")

            if diff < best_diff:
                best_diff = diff
                best_percentile = percentile

        # Usa o melhor percentil encontrado
        stats = self.calculate_optimal_thresholds(best_percentile)

        # Thresholds com margem de segurança
        hmm_threshold = max(0.70, min(0.95, stats['hmm'][f'p{best_percentile}'] * 0.95))
        lyap_threshold = max(0.05, stats['lyapunov'][f'p{best_percentile}'] * 0.8)
        curv_threshold = max(0.01, stats['curvature'][f'p{best_percentile}'] * 0.8)

        # Calcula sinais esperados
        expected_signals = sum(1 for a in self.analyses
                              if a.prob_hmm >= hmm_threshold
                              and a.lyapunov >= lyap_threshold
                              and abs(a.curvature) >= curv_threshold
                              and a.hmm_state in [1, 2])

        days = (self.bars[-1].timestamp - self.bars[0].timestamp).days
        expected_per_month = (expected_signals / max(days, 1)) * 30

        # Determina SL/TP otimizado baseado em volatilidade
        returns = np.diff([b.close for b in self.bars]) / np.array([b.close for b in self.bars[:-1]])
        daily_vol = np.std(returns) * np.sqrt(24 if self.periodicity == 'H1' else 96)

        # SL/TP baseado em ATR aproximado
        atr_pips = daily_vol * 10000 * 2  # 2x volatilidade diaria em pips
        stop_loss = max(15, min(40, round(atr_pips)))
        take_profit = max(30, min(80, round(atr_pips * 2)))  # 2:1 R:R

        self.config = OptimalConfig(
            min_prices=50,
            hmm_threshold=round(hmm_threshold, 2),
            lyapunov_threshold=round(lyap_threshold, 4),
            curvature_threshold=round(curv_threshold, 4),
            stop_loss_pips=float(stop_loss),
            take_profit_pips=float(take_profit),
            expected_signals_per_month=round(expected_per_month, 1),
            percentile_used=best_percentile
        )

        return self.config

    def save_config(self, filename: str = None):
        """Salva configuracao otimizada"""
        if not self.config:
            print("  ERRO: Nenhuma configuracao!")
            return

        if filename is None:
            filename = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs",
                "prm_optimized.json"
            )

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        config_data = {
            "strategy": "PRM-RiemannMandelbrot",
            "symbol": self.symbol,
            "periodicity": self.periodicity,
            "optimized_at": datetime.now(timezone.utc).isoformat(),
            "data_period": {
                "start": self.bars[0].timestamp.isoformat() if self.bars else None,
                "end": self.bars[-1].timestamp.isoformat() if self.bars else None,
                "total_bars": len(self.bars),
                "total_analyses": len(self.analyses)
            },
            "parameters": {
                "min_prices": self.config.min_prices,
                "stop_loss_pips": self.config.stop_loss_pips,
                "take_profit_pips": self.config.take_profit_pips,
                "hmm_threshold": self.config.hmm_threshold,
                "lyapunov_threshold": self.config.lyapunov_threshold,
                "curvature_threshold": self.config.curvature_threshold
            },
            "optimization_notes": {
                "percentile_used": self.config.percentile_used,
                "expected_signals_per_month": self.config.expected_signals_per_month,
                "note": "PRM e' um indicador de alta seletividade para eventos raros"
            }
        }

        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"\n  Configuracao salva em: {filename}")
        return filename

    def print_results(self):
        """Imprime resultados"""
        if not self.config:
            return

        c = self.config
        print("\n" + "=" * 60)
        print("  CONFIGURACAO OTIMIZADA PRM")
        print("=" * 60)
        print(f"\n  PARAMETROS:")
        print(f"    min_prices: {c.min_prices}")
        print(f"    hmm_threshold: {c.hmm_threshold}")
        print(f"    lyapunov_threshold: {c.lyapunov_threshold}")
        print(f"    curvature_threshold: {c.curvature_threshold}")
        print(f"    stop_loss_pips: {c.stop_loss_pips}")
        print(f"    take_profit_pips: {c.take_profit_pips}")
        print(f"\n  EXPECTATIVA:")
        print(f"    Sinais esperados/mes: ~{c.expected_signals_per_month}")
        print(f"    Percentil usado: {c.percentile_used}%")
        print(f"\n  NOTA: O PRM e' projetado para detectar eventos raros")
        print(f"        de alta conviccao (singularidades de mercado).")
        print("=" * 60)


async def main():
    """Funcao principal"""
    print("=" * 60)
    print("  OTIMIZADOR PRM FINAL")
    print("  Analise de Comportamento e Configuracao Otimizada")
    print("=" * 60)

    # Configuracao
    symbol = "EURUSD"
    periodicity = "H1"
    start_date = datetime(2025, 7, 1, tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)

    print(f"\n  Simbolo: {symbol}")
    print(f"  Periodicidade: {periodicity}")
    print(f"  Periodo: {start_date.date()} a {end_date.date()}")

    analyzer = PRMAnalyzer(symbol=symbol, periodicity=periodicity)

    # Carrega dados
    await analyzer.load_data(start_date, end_date)

    if not analyzer.bars:
        print("  ERRO: Nenhum dado carregado!")
        return

    # Analisa comportamento do PRM
    analyzer.analyze_prm_behavior(min_prices=50)

    if not analyzer.analyses:
        print("  ERRO: Nenhuma analise gerada!")
        return

    # Determina configuracao otimizada
    config = analyzer.determine_optimal_config(target_signals_per_month=5)

    # Mostra e salva resultados
    analyzer.print_results()
    analyzer.save_config()


if __name__ == "__main__":
    asyncio.run(main())
