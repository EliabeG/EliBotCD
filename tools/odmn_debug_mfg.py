#!/usr/bin/env python3
"""
Debug MFG calculation in ODMN
"""

import sys
import os
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simular dados
np.random.seed(42)
n = 500
prices = 1.1000 + np.cumsum(np.random.randn(n) * 0.0005)

print("=" * 60)
print("DEBUG MFG CALCULATION")
print("=" * 60)

# Parametros
price_level = prices[-1]
log_price = np.log(price_level)

# Calcular mean_log_price
prices_for_mean = prices[-100:]
mean_log_price = np.mean(np.log(prices_for_mean))

print(f"\nPrices:")
print(f"  Current price: {price_level:.5f}")
print(f"  Log price: {log_price:.6f}")
print(f"  Mean log price: {mean_log_price:.6f}")
print(f"  Difference: {log_price - mean_log_price:.8f}")

# Simular volatilidade do Heston
returns = np.diff(np.log(prices))
var_r = np.var(returns)
sigma = np.sqrt(var_r * 252)  # Volatilidade anualizada

print(f"\nVolatility:")
print(f"  Daily variance: {var_r:.10f}")
print(f"  Annualized vol (sigma): {sigma:.6f}")

# Calcular A_0
alpha = 0.1
T = 1.0
A_0 = sigma**2 / (4 * alpha * T)

print(f"\nMFG parameters:")
print(f"  alpha (cost): {alpha}")
print(f"  T (horizon): {T}")
print(f"  A_0 (Riccati): {A_0:.10f}")

# Calcular optimal_direction
optimal_direction = -2 * A_0 * (log_price - mean_log_price)

print(f"\nOptimal direction calculation:")
print(f"  -2 * A_0 = {-2 * A_0:.10f}")
print(f"  log_price - mean_log_price = {log_price - mean_log_price:.10f}")
print(f"  optimal_direction = {optimal_direction:.10f}")

# O problema e que A_0 e muito pequeno!
# Vamos ver qual seria um valor razoavel
print(f"\n" + "=" * 60)
print("ANALISE DO PROBLEMA")
print("=" * 60)

print(f"\nO problema: A_0 = sigma^2 / (4 * alpha * T)")
print(f"  Com sigma = {sigma:.6f}, A_0 = {A_0:.10f}")
print(f"  Isso e MUITO pequeno!")

# Proposta de fix: usar escala diferente
print(f"\nProposta de correcao:")
print(f"  1. Aumentar escala de A_0 (multiplicar por fator)")
print(f"  2. Ou usar formula diferente para optimal_direction")

# Testar com escala
for scale in [1, 10, 100, 1000, 10000]:
    A_0_scaled = A_0 * scale
    opt_dir = -2 * A_0_scaled * (log_price - mean_log_price)
    print(f"  Scale {scale:>5}x: A_0={A_0_scaled:.6f}, dir={opt_dir:+.6f}")

# Alternativa: usar diferenca percentual diretamente
pct_diff = (price_level - np.mean(prices_for_mean)) / np.mean(prices_for_mean)
print(f"\nAlternativa: usar diferenca percentual do preco")
print(f"  Diff %: {pct_diff*100:+.4f}%")
print(f"  Sinal: {'BUY' if pct_diff < -0.01 else 'SELL' if pct_diff > 0.01 else 'HOLD'}")
