#!/usr/bin/env python3
"""
================================================================================
DIAGNOSTICO PRM - Analisa distribuicao de valores para ajustar thresholds
================================================================================
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
import numpy as np

from api.fxopen_historical_ws import get_historical_data_with_spread_sync

try:
    from hmmlearn import hmm
except ImportError:
    print("ERRO: hmmlearn nao instalado")
    sys.exit(1)


def analyze_thresholds():
    """Analisa distribuicao de valores para recomendar thresholds"""
    print("=" * 70)
    print("  DIAGNOSTICO DE THRESHOLDS PRM")
    print("=" * 70)

    # Baixar dados
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=90)

    print(f"\n  Baixando dados de {start_time.date()} a {end_time.date()}...")
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

    # Preparar dados
    prices = np.array([bar.close for bar in bars])

    # Treinar HMM com warmup
    warmup = 6624
    warmup_prices = prices[:warmup]

    print(f"\n  Treinando HMM com {warmup} barras de warmup...")

    returns = np.diff(np.log(warmup_prices)).reshape(-1, 1)

    hmm_model = hmm.GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    hmm_model.fit(returns)

    # Analisar distribuicao de probabilidades em todas as barras
    print(f"\n  Analisando distribuicao de probabilidades...")

    hmm_probs = []
    lyap_values = []
    states = []

    for i in range(warmup, len(prices)):
        # Janela de analise
        window_prices = prices[i-200:i+1]
        window_returns = np.diff(np.log(window_prices)).reshape(-1, 1)

        try:
            probs = hmm_model.predict_proba(window_returns)
            current_probs = probs[-1]

            # Prob max de estados 1 e 2
            hmm_prob = max(current_probs[1] if len(current_probs) > 1 else 0,
                          current_probs[2] if len(current_probs) > 2 else 0)

            hmm_probs.append(hmm_prob)
            states.append(np.argmax(current_probs))

            # Lyapunov simplificado
            abs_returns = np.abs(window_returns[-50:])
            abs_returns = abs_returns[abs_returns > 1e-10]
            if len(abs_returns) > 0:
                lyap = np.mean(np.log(abs_returns)) + 10
                lyap = max(0, lyap)
            else:
                lyap = 0
            lyap_values.append(lyap)

        except Exception:
            pass

        if (i - warmup) % 2000 == 0:
            print(f"    Processado: {i - warmup}/{len(prices) - warmup}")

    hmm_probs = np.array(hmm_probs)
    lyap_values = np.array(lyap_values)
    states = np.array(states)

    # Estatisticas
    print("\n" + "=" * 70)
    print("  DISTRIBUICAO DE PROBABILIDADES HMM")
    print("=" * 70)

    percentiles = [50, 75, 80, 85, 90, 95, 99]
    print(f"\n  Percentis de Prob_HMM (max P1, P2):")
    for p in percentiles:
        val = np.percentile(hmm_probs, p)
        count_above = np.sum(hmm_probs >= val)
        pct_above = count_above / len(hmm_probs) * 100
        print(f"    P{p}: {val:.4f} ({count_above} barras acima, {pct_above:.1f}%)")

    print(f"\n  Estatisticas:")
    print(f"    Min: {np.min(hmm_probs):.4f}")
    print(f"    Max: {np.max(hmm_probs):.4f}")
    print(f"    Media: {np.mean(hmm_probs):.4f}")
    print(f"    Mediana: {np.median(hmm_probs):.4f}")

    print("\n" + "=" * 70)
    print("  DISTRIBUICAO DE LYAPUNOV")
    print("=" * 70)

    print(f"\n  Percentis de Lyapunov:")
    for p in percentiles:
        val = np.percentile(lyap_values, p)
        count_above = np.sum(lyap_values >= val)
        pct_above = count_above / len(lyap_values) * 100
        print(f"    P{p}: {val:.4f} ({count_above} barras acima, {pct_above:.1f}%)")

    print(f"\n  Estatisticas:")
    print(f"    Min: {np.min(lyap_values):.4f}")
    print(f"    Max: {np.max(lyap_values):.4f}")
    print(f"    Media: {np.mean(lyap_values):.4f}")
    print(f"    Mediana: {np.median(lyap_values):.4f}")

    print("\n" + "=" * 70)
    print("  ESTADOS HMM")
    print("=" * 70)

    for s in [0, 1, 2]:
        count = np.sum(states == s)
        pct = count / len(states) * 100
        print(f"    Estado {s}: {count} ({pct:.1f}%)")

    # Combinacoes de thresholds
    print("\n" + "=" * 70)
    print("  SINAIS POR COMBINACAO DE THRESHOLDS")
    print("=" * 70)

    hmm_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    lyap_thresholds = [0.0, 0.05, 0.1, 0.15, 0.2]

    print(f"\n  Contagem de barras que passam nos filtros (estados 1 e 2):")
    print(f"\n  {'HMM_TH':<10}", end="")
    for lt in lyap_thresholds:
        print(f"{'LYAP>'+str(lt):<12}", end="")
    print()
    print("  " + "-" * 70)

    for ht in hmm_thresholds:
        print(f"  {ht:<10}", end="")
        for lt in lyap_thresholds:
            mask = (hmm_probs >= ht) & (lyap_values >= lt) & ((states == 1) | (states == 2))
            count = np.sum(mask)
            pct = count / len(hmm_probs) * 100
            print(f"{count} ({pct:.1f}%){'':<4}", end="")
        print()

    # Recomendacoes
    print("\n" + "=" * 70)
    print("  RECOMENDACOES")
    print("=" * 70)

    # Encontrar thresholds que geram ~5-10% de sinais
    best_combo = None
    for ht in hmm_thresholds:
        for lt in lyap_thresholds:
            mask = (hmm_probs >= ht) & (lyap_values >= lt) & ((states == 1) | (states == 2))
            pct = np.sum(mask) / len(hmm_probs) * 100
            if 3 <= pct <= 10:
                if best_combo is None or pct > best_combo[2]:
                    best_combo = (ht, lt, pct)

    if best_combo:
        print(f"\n  Recomendacao (3-10% de sinais):")
        print(f"    hmm_threshold: {best_combo[0]}")
        print(f"    lyapunov_threshold: {best_combo[1]}")
        print(f"    Sinais esperados: {best_combo[2]:.1f}%")
    else:
        print(f"\n  Nenhuma combinacao gera sinais suficientes.")
        print(f"  Considere usar hmm_threshold mais baixo.")

    # Com hmm_threshold = 0.9
    mask_09 = (hmm_probs >= 0.9)
    print(f"\n  Com hmm_threshold = 0.9:")
    print(f"    Barras que passam: {np.sum(mask_09)} ({np.sum(mask_09)/len(hmm_probs)*100:.2f}%)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    analyze_thresholds()
