#!/usr/bin/env python3
"""Debug script para entender o comportamento do PRM"""

import sys
import os
# Adiciona diretorio raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
import numpy as np
from datetime import datetime, timezone
from collections import deque

from api.fxopen_historical_ws import download_historical_data
from strategies.alta_volatilidade.prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot


async def main():
    print("=" * 60)
    print("  DEBUG PRM - Analise de Distribuicao de Sinais")
    print("=" * 60)

    # Carregar dados
    bars = await download_historical_data(
        'EURUSD', 'H1',
        datetime(2025, 7, 1, tzinfo=timezone.utc),
        datetime.now(timezone.utc)
    )
    print(f"\nBarras carregadas: {len(bars)}")

    # Calcular PRM
    prm = ProtocoloRiemannMandelbrot(
        n_states=3,
        hmm_threshold=0.1,
        lyapunov_threshold_k=0.001
    )

    prices_buf = deque(maxlen=500)
    volumes_buf = deque(maxlen=500)

    hmm_probs = []
    lyapunov_scores = []
    hmm_states = []

    print("\nCalculando valores PRM...")
    for i, bar in enumerate(bars):
        prices_buf.append(bar.close)
        volumes_buf.append(bar.volume)

        if len(prices_buf) < 50:
            continue

        try:
            result = prm.analyze(np.array(prices_buf), np.array(volumes_buf))
            hmm_probs.append(result['Prob_HMM'])
            lyapunov_scores.append(result['Lyapunov_Score'])
            hmm_states.append(result['hmm_analysis']['current_state'])
        except:
            pass

    print(f"Pontos calculados: {len(hmm_probs)}")

    # Estatisticas
    print("\n" + "=" * 60)
    print("  ESTATISTICAS DOS VALORES PRM")
    print("=" * 60)

    print(f"\nHMM Prob:")
    print(f"  Min: {min(hmm_probs):.4f}")
    print(f"  Max: {max(hmm_probs):.4f}")
    print(f"  Mean: {np.mean(hmm_probs):.4f}")
    print(f"  Median: {np.median(hmm_probs):.4f}")

    print(f"\nLyapunov Score:")
    print(f"  Min: {min(lyapunov_scores):.4f}")
    print(f"  Max: {max(lyapunov_scores):.4f}")
    print(f"  Mean: {np.mean(lyapunov_scores):.4f}")
    print(f"  Median: {np.median(lyapunov_scores):.4f}")

    print(f"\nHMM States:")
    for s in [0, 1, 2]:
        count = hmm_states.count(s)
        pct = count / len(hmm_states) * 100
        print(f"  State {s}: {count} ({pct:.1f}%)")

    # Contar sinais com diferentes thresholds
    print("\n" + "=" * 60)
    print("  CONTAGEM DE SINAIS POR THRESHOLD")
    print("=" * 60)

    print("\n  HMM >= X e Lyapunov >= Y:")
    print(f"  {'HMM':<8} | {'Lyap=0.02':<12} | {'Lyap=0.05':<12} | {'Lyap=0.08':<12} | {'Lyap=0.10':<12}")
    print("  " + "-" * 60)

    for hmm_t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        counts = []
        for lyap_t in [0.02, 0.05, 0.08, 0.10]:
            count = sum(1 for h, l in zip(hmm_probs, lyapunov_scores)
                       if h >= hmm_t and l >= lyap_t)
            counts.append(count)
        print(f"  {hmm_t:<8.2f} | {counts[0]:<12} | {counts[1]:<12} | {counts[2]:<12} | {counts[3]:<12}")

    # Filtrar por states tambem
    print("\n  Incluindo filtro de HMM States [0,1] ou [1,2]:")
    states_filter = [0, 1]
    for hmm_t in [0.55, 0.60, 0.65]:
        for lyap_t in [0.03, 0.05, 0.07]:
            count = sum(1 for h, l, s in zip(hmm_probs, lyapunov_scores, hmm_states)
                       if h >= hmm_t and l >= lyap_t and s in states_filter)
            print(f"    hmm>={hmm_t}, lyap>={lyap_t}, states={states_filter}: {count} sinais")


if __name__ == "__main__":
    asyncio.run(main())
