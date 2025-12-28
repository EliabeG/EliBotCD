#!/usr/bin/env python3
"""
Debug específico do cálculo de dimensão fractal no RHHF
"""

import sys
import os
import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_box_counting(curve: np.ndarray, n_scales: int = 20, eps: float = 1e-10) -> float:
    """Versão com debug do box counting"""
    n = len(curve)
    print(f"\n=== BOX COUNTING DEBUG ===")
    print(f"Curva: n={n}")

    if n < 10:
        print("  RETORNO PREMATURO: n < 10")
        return 1.5

    curve_min = np.min(curve)
    curve_max = np.max(curve)
    curve_range = curve_max - curve_min

    print(f"  Min: {curve_min:.10f}")
    print(f"  Max: {curve_max:.10f}")
    print(f"  Range: {curve_range:.10f}")
    print(f"  eps: {eps:.10f}")

    if curve_range < eps:
        print(f"  [BUG] curve_range ({curve_range}) < eps ({eps})")
        print(f"  RETORNANDO 1.0 - CURVA CONSIDERADA CONSTANTE!")
        return 1.0

    curve_norm = (curve - curve_min) / curve_range

    min_scale = 1
    max_scale = n // 4
    scales = np.logspace(np.log10(min_scale), np.log10(max(max_scale, 2)), n_scales)
    scales = np.unique(scales.astype(int))
    scales = scales[scales > 0]

    print(f"  Scales: {scales[:5]}...{scales[-3:]}")
    print(f"  Num scales: {len(scales)}")

    if len(scales) < 3:
        print("  RETORNO PREMATURO: len(scales) < 3")
        return 1.5

    counts = []

    for scale in scales:
        n_boxes_t = int(np.ceil(n / scale))
        n_boxes_y = int(np.ceil(1.0 / (scale / n)))

        if n_boxes_y < 1:
            n_boxes_y = 1

        occupied = set()

        for i in range(n):
            box_t = int(i / scale)
            box_y = int(curve_norm[i] * n_boxes_y)
            box_y = min(box_y, n_boxes_y - 1)
            occupied.add((box_t, box_y))

        counts.append(len(occupied))

    log_scales = np.log(scales)
    log_counts = np.log(np.array(counts) + 1)

    try:
        coeffs = np.polyfit(log_scales, log_counts, 1)
        dimension = -coeffs[0]
    except:
        dimension = 1.5

    print(f"  Coeffs: slope={coeffs[0]:.4f}, intercept={coeffs[1]:.4f}")
    print(f"  Dimension (raw): {dimension:.4f}")

    dimension = np.clip(dimension, 1.0, 2.0)
    print(f"  Dimension (clipped): {dimension:.4f}")

    return dimension


def main():
    print("=" * 70)
    print("  DEBUG DO CALCULO FRACTAL")
    print("=" * 70)

    # Simular dados de preço
    np.random.seed(42)
    n = 300
    prices = 1.1000 + np.cumsum(np.random.randn(n) * 0.0005)

    print(f"\nPreços: n={n}, range={prices.max()-prices.min():.6f}")

    # Calcular frequência como o RHHF faz
    print("\n--- Calculando frequência instantânea ---")

    # Simular IMF (diferença do preço suavizado)
    imf = np.diff(prices)
    imf = np.append(imf, imf[-1])  # Manter tamanho

    print(f"IMF: n={len(imf)}, std={np.std(imf):.10f}")

    # Transformada de Hilbert
    analytic_signal = hilbert(imf)
    phase = np.unwrap(np.angle(analytic_signal))
    frequency = np.gradient(phase)
    frequency = np.abs(frequency)

    print(f"Frequência (raw): min={frequency.min():.6f}, max={frequency.max():.6f}, std={np.std(frequency):.6f}")

    # Suavização
    smoothing_sigma = 3
    frequency_smooth = gaussian_filter1d(frequency, sigma=smoothing_sigma)

    print(f"Frequência (smooth): min={frequency_smooth.min():.10f}, max={frequency_smooth.max():.10f}")
    print(f"Frequência (smooth): range={frequency_smooth.max()-frequency_smooth.min():.10f}")

    # Testar box counting
    print("\n--- Testando box counting com eps=1e-10 ---")
    dim = debug_box_counting(frequency_smooth, eps=1e-10)

    print("\n--- Testando box counting com eps=1e-15 ---")
    dim2 = debug_box_counting(frequency_smooth, eps=1e-15)

    print("\n--- Testando com frequência NÃO suavizada ---")
    dim3 = debug_box_counting(frequency, eps=1e-10)

    # Proposta de fix
    print("\n" + "=" * 70)
    print("  PROPOSTA DE CORRECAO")
    print("=" * 70)

    print("""
    O PROBLEMA:
    1. A frequência suavizada tem range muito pequeno (~1e-6)
    2. O eps=1e-10 não é atingido, MAS o range pequeno causa
       problemas no cálculo de n_boxes_y
    3. n_boxes_y = ceil(1.0 / (scale/n)) pode ser MUITO grande
       quando scale é pequeno, distorcendo o cálculo

    A CORREÇÃO:
    1. Normalizar a frequência ANTES de calcular box counting
    2. Usar um número fixo de boxes em Y (ex: 100)
    3. Ou usar Hurst exponent em vez de box counting
    """)

    # Testar com frequência normalizada
    print("\n--- Testando com frequência normalizada manualmente ---")
    freq_normalized = (frequency_smooth - frequency_smooth.min()) / (frequency_smooth.max() - frequency_smooth.min() + 1e-15)
    # Escalar para range [0, 100] para ter variação numérica significativa
    freq_scaled = freq_normalized * 100

    dim4 = debug_box_counting(freq_scaled, eps=1e-10)

    print("\n" + "=" * 70)
    print("  CONCLUSAO")
    print("=" * 70)

    print(f"""
    Dimensão com freq smooth (eps=1e-10): {dim:.4f}
    Dimensão com freq smooth (eps=1e-15): {dim2:.4f}
    Dimensão com freq raw: {dim3:.4f}
    Dimensão com freq scaled [0,100]: {dim4:.4f}

    O problema é que a frequência suavizada tem valores muito próximos,
    e o cálculo de box counting não está tratando isso corretamente.

    SOLUÇÃO RECOMENDADA:
    - Usar Hurst Exponent (R/S analysis) em vez de box counting
    - Ou normalizar/escalar a frequência antes do box counting
    """)


if __name__ == "__main__":
    main()
