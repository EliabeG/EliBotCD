#!/usr/bin/env python3
"""
================================================================================
DIAGNOSTICO PRM - Verifica por que o indicador nao gera sinais
================================================================================
"""

import asyncio
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
import numpy as np
from api.fxopen_historical_ws import get_historical_data_with_spread_sync
from strategies.alta_volatilidade.prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot


def diagnose_prm(bars: list, use_optimized: bool = True):
    """
    Executa diagnostico do PRM em dados historicos

    Args:
        bars: Lista de barras com dados
        use_optimized: Se True, usa parametros otimizados do prm_robust_top10.json
    """
    print("=" * 70)
    print("  DIAGNOSTICO DO INDICADOR PRM")
    print("=" * 70)

    # Carregar parametros otimizados
    if use_optimized:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "prm_robust_top10.json"
        )
        with open(config_path, 'r') as f:
            configs = json.load(f)
            best_config = configs[0]['params']

        print(f"\n  Usando parametros OTIMIZADOS:")
        print(f"    hmm_threshold: {best_config.get('hmm_threshold', 0.7)}")
        print(f"    lyapunov_threshold: {best_config.get('lyapunov_threshold', 0.04)}")
        print(f"    hmm_states_allowed: {best_config.get('hmm_states_allowed', [1, 2])}")
        print(f"    stop_loss_pips: {best_config.get('stop_loss_pips', 30.5)}")
        print(f"    take_profit_pips: {best_config.get('take_profit_pips', 64.6)}")

        prm = ProtocoloRiemannMandelbrot(
            n_states=3,
            hmm_threshold=best_config.get('hmm_threshold', 0.7),
            lyapunov_threshold_k=best_config.get('lyapunov_threshold', 0.04),
            curvature_threshold=0.1,  # Nao esta no config, usar default
            hmm_training_window=200,
            hmm_min_training_samples=50
        )
    else:
        print(f"\n  Usando parametros DEFAULT:")
        print(f"    hmm_threshold: 0.85")
        print(f"    lyapunov_threshold: 0.5")
        prm = ProtocoloRiemannMandelbrot()

    # Preparar precos
    prices = np.array([bar.close for bar in bars])
    print(f"\n  Barras: {len(bars)}")
    print(f"  Primeiro preco: {prices[0]:.5f}")
    print(f"  Ultimo preco: {prices[-1]:.5f}")
    print(f"  Min preco: {prices.min():.5f}")
    print(f"  Max preco: {prices.max():.5f}")

    # Analisar em janelas deslizantes
    min_window = 150  # Minimo de barras para analise
    singularities = []
    hmm_activations = []
    lyap_entries = []
    curv_exceeds = []

    print(f"\n  Analisando {len(bars) - min_window} janelas...")

    for i in range(min_window, len(bars)):
        window_prices = prices[:i+1]

        try:
            result = prm.analyze(window_prices)

            # Coletar estatisticas
            if result['singularity_detected']:
                singularities.append(i)
            if result['hmm_analysis']['hmm_activated']:
                hmm_activations.append(i)
            if result['lyapunov_analysis']['is_entry_point']:
                lyap_entries.append(i)
            if result['curvature_analysis']['exceeds_threshold']:
                curv_exceeds.append(i)

            # Mostrar progresso a cada 500 barras
            if i % 500 == 0:
                print(f"    Barra {i}: HMM={result['hmm_analysis']['hmm_activated']}, "
                      f"Lyap={result['lyapunov_analysis']['is_entry_point']}, "
                      f"Curv={result['curvature_analysis']['exceeds_threshold']}, "
                      f"Sing={result['singularity_detected']}")
        except Exception as e:
            if i % 1000 == 0:
                print(f"    Barra {i}: Erro - {e}")

    # Resumo
    print("\n" + "=" * 70)
    print("  RESUMO DA ANALISE")
    print("=" * 70)
    print(f"\n  Total de janelas analisadas: {len(bars) - min_window}")
    print(f"\n  Condicoes atendidas:")
    print(f"    HMM ativado (P > threshold): {len(hmm_activations)} vezes ({len(hmm_activations)/(len(bars)-min_window)*100:.1f}%)")
    print(f"    Lyapunov entrada (0 < L < K): {len(lyap_entries)} vezes ({len(lyap_entries)/(len(bars)-min_window)*100:.1f}%)")
    print(f"    Curvatura excede threshold: {len(curv_exceeds)} vezes ({len(curv_exceeds)/(len(bars)-min_window)*100:.1f}%)")
    print(f"\n  SINGULARIDADES DETECTADAS: {len(singularities)}")

    if len(singularities) > 0:
        print(f"\n  Primeiras 10 singularidades (indice de barra):")
        for idx in singularities[:10]:
            bar = bars[idx]
            print(f"    Barra {idx}: {bar.timestamp} @ {bar.close:.5f}")
    else:
        print("\n  PROBLEMA: Nenhuma singularidade detectada!")
        print("\n  Analise detalhada da ultima barra:")

        # Analisar ultima barra com detalhes
        result = prm.analyze(prices)

        print(f"\n  HMM Analysis:")
        hmm = result['hmm_analysis']
        print(f"    Estado atual: {hmm['current_state']}")
        print(f"    P(Estado 0): {hmm['prob_state_0']:.4f}")
        print(f"    P(Estado 1): {hmm['prob_state_1']:.4f}")
        print(f"    P(Estado 2): {hmm['prob_state_2']:.4f}")
        print(f"    Prob_HMM (max P1,P2): {hmm['Prob_HMM']:.4f}")
        print(f"    Threshold: {prm.hmm_threshold}")
        print(f"    HMM Ativado: {hmm['hmm_activated']}")
        print(f"    High Vol State: {hmm['high_volatility_state']}")

        print(f"\n  Lyapunov Analysis:")
        lyap = result['lyapunov_analysis']
        print(f"    Lambda max: {lyap['lyapunov_max']:.6f}")
        print(f"    Threshold K: {lyap['threshold_K']}")
        print(f"    Classificacao: {lyap['classification']}")
        print(f"    Is Entry Point: {lyap['is_entry_point']}")

        print(f"\n  Curvature Analysis:")
        curv = result['curvature_analysis']
        print(f"    Curvatura atual: {curv['current_curvature']:.6f}")
        print(f"    Aceleracao atual: {curv['current_acceleration']:.6f}")
        print(f"    Threshold: {curv['threshold']}")
        print(f"    Excede: {curv['exceeds_threshold']}")

    return singularities


def main():
    """Funcao principal"""
    print("\n" + "=" * 70)
    print("  BAIXANDO DADOS HISTORICOS COM SPREAD REAL")
    print("=" * 70)

    # Baixar dados
    end_time = datetime.now(timezone.utc)
    start_time = datetime(2025, 7, 1, tzinfo=timezone.utc)

    print(f"\n  Periodo: {start_time.date()} a {end_time.date()}")
    print(f"  Par: EURUSD M5")

    bars = get_historical_data_with_spread_sync(
        symbol="EURUSD",
        periodicity="M5",
        start_time=start_time,
        end_time=end_time
    )

    if not bars:
        print("  ERRO: Falha ao baixar dados")
        return

    print(f"\n  Barras baixadas: {len(bars)}")

    # Estatisticas de spread
    spreads = [bar.spread_pips for bar in bars if bar.has_spread_data]
    if spreads:
        print(f"\n  Spread (pips):")
        print(f"    Min: {min(spreads):.2f}")
        print(f"    Max: {max(spreads):.2f}")
        print(f"    Media: {np.mean(spreads):.2f}")

    # Diagnostico com parametros otimizados
    print("\n\n")
    diagnose_prm(bars, use_optimized=True)

    # Diagnostico com parametros default
    print("\n\n")
    diagnose_prm(bars, use_optimized=False)


if __name__ == "__main__":
    main()
