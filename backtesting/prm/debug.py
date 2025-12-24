#!/usr/bin/env python3
"""
Debug script para entender o comportamento do PRM

VERSÃO CORRIGIDA - Usa PRM com parâmetros corretos
"""

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
    print("  VERSAO CORRIGIDA - Sem Look-Ahead")
    print("=" * 60)

    # Carregar dados
    bars = await download_historical_data(
        'EURUSD', 'H1',
        datetime(2025, 7, 1, tzinfo=timezone.utc),
        datetime.now(timezone.utc)
    )
    print(f"\nBarras carregadas: {len(bars)}")

    # Calcular PRM com parâmetros CORRIGIDOS
    prm = ProtocoloRiemannMandelbrot(
        n_states=3,
        hmm_threshold=0.1,
        lyapunov_threshold_k=0.001,
        hmm_training_window=200,        # NOVO: Janela de treino
        hmm_min_training_samples=50     # NOVO: Mínimo de amostras
    )

    prices_buf = deque(maxlen=500)
    volumes_buf = deque(maxlen=500)

    hmm_probs = []
    lyapunov_scores = []
    hmm_states = []
    
    # NOVO: Estatísticas de direção
    directions = []
    min_bars_for_direction = 12

    print("\nCalculando valores PRM (com HMM retreinando em janela deslizante)...")
    print("NOTA: Isso pode ser mais lento que a versão anterior devido ao retreino do HMM")
    
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
            
            # CORRIGIDO: Calcular direção apenas com barras FECHADAS
            if i >= min_bars_for_direction:
                recent_close = bars[i - 1].close   # Última barra FECHADA
                past_close = bars[i - 11].close    # 10 barras antes
                trend = recent_close - past_close
                direction = 1 if trend > 0 else -1
            else:
                direction = 0
            directions.append(direction)
            
        except Exception as e:
            if i < 55:  # Só mostrar erros nas primeiras barras
                print(f"  Barra {i}: {e}")
            continue

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(bars)} barras...")

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
    print(f"  Std: {np.std(hmm_probs):.4f}")

    print(f"\nLyapunov Score:")
    print(f"  Min: {min(lyapunov_scores):.4f}")
    print(f"  Max: {max(lyapunov_scores):.4f}")
    print(f"  Mean: {np.mean(lyapunov_scores):.4f}")
    print(f"  Median: {np.median(lyapunov_scores):.4f}")
    print(f"  Std: {np.std(lyapunov_scores):.4f}")

    print(f"\nHMM States:")
    for s in [0, 1, 2]:
        count = hmm_states.count(s)
        pct = count / len(hmm_states) * 100 if hmm_states else 0
        print(f"  State {s}: {count} ({pct:.1f}%)")

    # NOVO: Estatísticas de direção
    print(f"\nDirecao (baseada em barras FECHADAS):")
    long_count = directions.count(1)
    short_count = directions.count(-1)
    neutral_count = directions.count(0)
    total = len(directions)
    print(f"  Long:    {long_count} ({long_count/total*100:.1f}%)")
    print(f"  Short:   {short_count} ({short_count/total*100:.1f}%)")
    print(f"  Neutral: {neutral_count} ({neutral_count/total*100:.1f}%)")

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

    # NOVO: Sinais com direção
    print("\n" + "=" * 60)
    print("  SINAIS COM DIRECAO (usando barras FECHADAS)")
    print("=" * 60)
    
    for hmm_t in [0.55, 0.60, 0.65]:
        for lyap_t in [0.05, 0.06, 0.07]:
            long_signals = sum(1 for h, l, s, d in zip(hmm_probs, lyapunov_scores, hmm_states, directions)
                              if h >= hmm_t and l >= lyap_t and s in [0, 1] and d == 1)
            short_signals = sum(1 for h, l, s, d in zip(hmm_probs, lyapunov_scores, hmm_states, directions)
                               if h >= hmm_t and l >= lyap_t and s in [0, 1] and d == -1)
            total_signals = long_signals + short_signals
            print(f"    hmm>={hmm_t}, lyap>={lyap_t}: {total_signals} sinais (L:{long_signals}, S:{short_signals})")


if __name__ == "__main__":
    asyncio.run(main())
