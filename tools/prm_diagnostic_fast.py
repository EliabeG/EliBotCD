#!/usr/bin/env python3
"""
================================================================================
DIAGNOSTICO RAPIDO PRM - Analisa amostras para entender comportamento
================================================================================
"""

import asyncio
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
import numpy as np
from api.fxopen_historical_ws import get_historical_data_with_spread_sync
from strategies.alta_volatilidade.prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot


def analyze_sample(prm, prices, idx, total):
    """Analisa uma amostra e retorna resultados"""
    try:
        result = prm.analyze(prices)
        return {
            'idx': idx,
            'hmm_activated': result['hmm_analysis']['hmm_activated'],
            'hmm_state': result['hmm_analysis']['current_state'],
            'prob_hmm': result['hmm_analysis']['Prob_HMM'],
            'lyap_entry': result['lyapunov_analysis']['is_entry_point'],
            'lyap_max': result['lyapunov_analysis']['lyapunov_max'],
            'curv_exceeds': result['curvature_analysis']['exceeds_threshold'],
            'curv_acc': result['curvature_analysis']['current_acceleration'],
            'singularity': result['singularity_detected'],
            'high_vol_state': result['hmm_analysis']['high_volatility_state']
        }
    except Exception as e:
        return {'idx': idx, 'error': str(e)}


def diagnose_prm_fast(bars, use_optimized=True):
    """
    Diagnostico rapido - analisa apenas algumas amostras
    """
    print("=" * 70)
    print("  DIAGNOSTICO RAPIDO DO PRM")
    print("=" * 70)

    # Carregar parametros
    if use_optimized:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "prm_robust_top10.json"
        )
        with open(config_path, 'r') as f:
            configs = json.load(f)
            best_config = configs[0]['params']

        hmm_threshold = best_config.get('hmm_threshold', 0.7)
        lyap_threshold = best_config.get('lyapunov_threshold', 0.04)

        print(f"\n  Parametros OTIMIZADOS:")
        print(f"    hmm_threshold: {hmm_threshold}")
        print(f"    lyapunov_threshold: {lyap_threshold}")

        prm = ProtocoloRiemannMandelbrot(
            n_states=3,
            hmm_threshold=hmm_threshold,
            lyapunov_threshold_k=lyap_threshold,
            curvature_threshold=0.1,
            hmm_training_window=200,
            hmm_min_training_samples=50
        )
    else:
        print(f"\n  Parametros DEFAULT:")
        print(f"    hmm_threshold: 0.85")
        print(f"    lyapunov_threshold: 0.5")
        prm = ProtocoloRiemannMandelbrot()

    # Preparar precos
    prices = np.array([bar.close for bar in bars])
    print(f"\n  Total de barras: {len(bars)}")

    # Analisar amostras esparsas (a cada 100 barras + ultimas 10)
    min_window = 200
    sample_indices = list(range(min_window, len(bars), 100))  # A cada 100 barras
    sample_indices.extend(range(max(min_window, len(bars)-10), len(bars)))  # Ultimas 10
    sample_indices = sorted(set(sample_indices))

    print(f"  Analisando {len(sample_indices)} amostras...")

    results = []
    for i in sample_indices:
        window_prices = prices[:i+1]
        res = analyze_sample(prm, window_prices, i, len(bars))
        results.append(res)

    # Contagem de cada condicao
    hmm_activated = sum(1 for r in results if r.get('hmm_activated', False))
    lyap_entries = sum(1 for r in results if r.get('lyap_entry', False))
    curv_exceeds = sum(1 for r in results if r.get('curv_exceeds', False))
    singularities = sum(1 for r in results if r.get('singularity', False))
    high_vol = sum(1 for r in results if r.get('high_vol_state', False))

    print("\n" + "=" * 70)
    print("  RESUMO")
    print("=" * 70)
    print(f"\n  Amostras analisadas: {len(results)}")
    print(f"\n  Condicoes atendidas:")
    print(f"    HMM ativado (P > threshold):   {hmm_activated}/{len(results)} ({hmm_activated/len(results)*100:.1f}%)")
    print(f"    High Vol State (estado 1/2):   {high_vol}/{len(results)} ({high_vol/len(results)*100:.1f}%)")
    print(f"    Lyapunov entrada (0 < L < K):  {lyap_entries}/{len(results)} ({lyap_entries/len(results)*100:.1f}%)")
    print(f"    Curvatura excede threshold:    {curv_exceeds}/{len(results)} ({curv_exceeds/len(results)*100:.1f}%)")
    print(f"\n  SINGULARIDADES: {singularities}/{len(results)}")

    # Analise detalhada de distribuicao
    print("\n" + "=" * 70)
    print("  DISTRIBUICAO DOS VALORES")
    print("=" * 70)

    prob_hmms = [r['prob_hmm'] for r in results if 'prob_hmm' in r]
    lyap_vals = [r['lyap_max'] for r in results if 'lyap_max' in r]
    curv_vals = [r['curv_acc'] for r in results if 'curv_acc' in r]
    states = [r['hmm_state'] for r in results if 'hmm_state' in r]

    if prob_hmms:
        print(f"\n  Prob_HMM (max P1, P2):")
        print(f"    Min: {min(prob_hmms):.4f}")
        print(f"    Max: {max(prob_hmms):.4f}")
        print(f"    Media: {np.mean(prob_hmms):.4f}")
        print(f"    Threshold: {prm.hmm_threshold}")
        print(f"    Acima threshold: {sum(1 for p in prob_hmms if p > prm.hmm_threshold)}/{len(prob_hmms)}")

    if lyap_vals:
        print(f"\n  Lyapunov Max:")
        print(f"    Min: {min(lyap_vals):.6f}")
        print(f"    Max: {max(lyap_vals):.6f}")
        print(f"    Media: {np.mean(lyap_vals):.6f}")
        print(f"    Threshold K: {prm.lyapunov_threshold_k}")
        print(f"    Entre 0 e K: {sum(1 for l in lyap_vals if 0 < l < prm.lyapunov_threshold_k)}/{len(lyap_vals)}")

    if curv_vals:
        print(f"\n  Curvatura Aceleracao:")
        print(f"    Min: {min(curv_vals):.6f}")
        print(f"    Max: {max(curv_vals):.6f}")
        print(f"    Media: {np.mean(curv_vals):.6f}")
        print(f"    Threshold: {prm.curvature_threshold}")
        print(f"    Abs > threshold: {sum(1 for c in curv_vals if abs(c) > prm.curvature_threshold)}/{len(curv_vals)}")

    if states:
        print(f"\n  Estados HMM (0=Consol, 1=AltaVol, 2=Choque):")
        for s in [0, 1, 2]:
            count = sum(1 for st in states if st == s)
            print(f"    Estado {s}: {count}/{len(states)} ({count/len(states)*100:.1f}%)")

    # Mostrar exemplos de amostras
    print("\n" + "=" * 70)
    print("  EXEMPLOS DE AMOSTRAS")
    print("=" * 70)

    print("\n  Ultimas 5 amostras:")
    for r in results[-5:]:
        if 'error' in r:
            print(f"    Idx {r['idx']}: ERRO - {r['error']}")
        else:
            print(f"    Idx {r['idx']}: HMM={r['prob_hmm']:.3f}|{r['hmm_activated']}, "
                  f"Lyap={r['lyap_max']:.4f}|{r['lyap_entry']}, "
                  f"Curv={r['curv_acc']:.4f}|{r['curv_exceeds']}, "
                  f"State={r['hmm_state']}, Sing={r['singularity']}")

    # Diagnostico do problema
    print("\n" + "=" * 70)
    print("  DIAGNOSTICO DO PROBLEMA")
    print("=" * 70)

    if singularities == 0:
        problems = []

        if hmm_activated < len(results) * 0.1:
            problems.append("HMM raramente ativado - threshold muito alto?")

        if high_vol == 0:
            problems.append("Nunca em High Volatility State - HMM nao detecta volatilidade")

        if lyap_entries < len(results) * 0.1:
            problems.append("Lyapunov raramente sinaliza entrada - threshold_K muito baixo?")

        if curv_exceeds < len(results) * 0.1:
            problems.append("Curvatura raramente excede threshold - threshold muito alto?")

        if problems:
            print("\n  Problemas identificados:")
            for p in problems:
                print(f"    - {p}")
        else:
            print("\n  Condicoes sao atendidas individualmente mas nao simultaneamente")
            print("  Isso e normal para um indicador rigoroso")

    return results


def main():
    """Funcao principal"""
    print("\n" + "=" * 70)
    print("  BAIXANDO DADOS HISTORICOS")
    print("=" * 70)

    # Baixar dados
    end_time = datetime.now(timezone.utc)
    start_time = datetime(2025, 7, 1, tzinfo=timezone.utc)

    print(f"\n  Periodo: {start_time.date()} a {end_time.date()}")

    bars = get_historical_data_with_spread_sync(
        symbol="EURUSD",
        periodicity="M5",
        start_time=start_time,
        end_time=end_time
    )

    if not bars:
        print("  ERRO: Falha ao baixar dados")
        return

    print(f"  Barras: {len(bars)}")

    # Diagnostico com parametros otimizados
    print("\n")
    diagnose_prm_fast(bars, use_optimized=True)

    # Comparar com parametros default
    print("\n\n")
    diagnose_prm_fast(bars, use_optimized=False)


if __name__ == "__main__":
    main()
