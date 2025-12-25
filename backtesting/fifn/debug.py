#!/usr/bin/env python3
"""
================================================================================
DEBUG FIFN - Analise de Distribuicao de Sinais
================================================================================

Script de debug para entender o comportamento do FIFN e analisar
a distribuicao de sinais com diferentes thresholds.

VERSAO V2.0 - Usa FIFN sem look-ahead bias
==============================================
- Direcao baseada em barras fechadas
- Todos os parametros corrigidos
- Analise de Reynolds, Skewness, KL Divergence

Uso:
    python -m backtesting.fifn.debug
"""

import sys
import os
# Adiciona diretorio raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import deque

from api.fxopen_historical_ws import download_historical_data
from strategies.alta_volatilidade.fifn_fisher_navier import FluxoInformacaoFisherNavier


async def main():
    print("=" * 70)
    print("  DEBUG FIFN - Analise de Distribuicao de Sinais")
    print("  VERSAO V2.0 - Sem Look-Ahead")
    print("=" * 70)

    # Carregar dados - AUDITORIA 4: Usar data relativa
    print("\nCarregando dados historicos...")
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=180)  # Ultimos 6 meses
    bars = await download_historical_data(
        'EURUSD', 'H1',
        start_time,
        end_time
    )
    print(f"Barras carregadas: {len(bars)}")

    if len(bars) < 100:
        print("ERRO: Dados insuficientes!")
        return

    # Calcular FIFN
    fifn = FluxoInformacaoFisherNavier(
        window_size=50,
        kl_lookback=10,
        reynolds_sweet_low=2300,
        reynolds_sweet_high=4000,
        skewness_threshold=0.5
    )

    prices_buf = deque(maxlen=500)
    # AUDITORIA 29: Unificado com optimizer e strategy
    min_prices = 100  # window_size(50) + kl_lookback(10) + buffer(40)

    reynolds_values = []
    kl_divergences = []
    skewness_values = []
    pressure_gradients = []
    in_sweet_spots = []
    fisher_values = []

    # Estatisticas de direcao (baseada em barras FECHADAS)
    directions = []
    min_bars_for_direction = 12

    print("\nCalculando valores FIFN...")
    print("NOTA: Pode ser mais lento devido aos calculos complexos")

    errors = 0
    for i, bar in enumerate(bars):
        prices_buf.append(bar.close)

        if len(prices_buf) < min_prices:
            continue

        try:
            # AUDITORIA 4: Excluir barra atual para consistencia com optimizer
            prices_for_analysis = np.array(prices_buf)[:-1] if len(prices_buf) > min_prices else np.array(prices_buf)
            result = fifn.analyze(prices_for_analysis)

            reynolds_values.append(result['Reynolds_Number'])
            kl_divergences.append(result['KL_Divergence'])
            skewness_values.append(result['directional_signal']['skewness'])
            pressure_gradients.append(result['Pressure_Gradient'])
            in_sweet_spots.append(result['directional_signal']['in_sweet_spot'])
            fisher_values.append(result['fisher_series'][-1])

            # CORRIGIDO V2.0: Calcular direcao apenas com barras FECHADAS
            if i >= min_bars_for_direction:
                recent_close = bars[i - 1].close   # Ultima barra FECHADA
                past_close = bars[i - 11].close    # 10 barras antes
                trend = recent_close - past_close
                direction = 1 if trend > 0 else -1
            else:
                direction = 0
            directions.append(direction)

        except Exception as e:
            errors += 1
            if errors <= 3:  # So mostrar primeiros erros
                print(f"  Erro na barra {i}: {e}")
            continue

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(bars)} barras processadas...")

    print(f"\nPontos calculados: {len(reynolds_values)}")
    if errors > 0:
        print(f"Erros encontrados: {errors}")

    if len(reynolds_values) == 0:
        print("ERRO: Nenhum ponto calculado!")
        return

    # Estatisticas
    print("\n" + "=" * 70)
    print("  ESTATISTICAS DOS VALORES FIFN")
    print("=" * 70)

    print(f"\nNumero de Reynolds (Discriminador):")
    print(f"  Min:    {min(reynolds_values):.0f}")
    print(f"  Max:    {max(reynolds_values):.0f}")
    print(f"  Mean:   {np.mean(reynolds_values):.0f}")
    print(f"  Median: {np.median(reynolds_values):.0f}")
    print(f"  Std:    {np.std(reynolds_values):.0f}")

    # Distribuicao por zonas de Reynolds
    laminar = sum(1 for r in reynolds_values if r < 2000)
    transition = sum(1 for r in reynolds_values if 2000 <= r < 2300)
    sweet_spot = sum(1 for r in reynolds_values if 2300 <= r <= 4000)
    turbulent = sum(1 for r in reynolds_values if r > 4000)

    print(f"\n  Distribuicao por zona:")
    print(f"    Laminar (Re < 2000): {laminar} ({laminar/len(reynolds_values)*100:.1f}%) - NAO OPERAR")
    print(f"    Transicao (2000-2300): {transition} ({transition/len(reynolds_values)*100:.1f}%) - AGUARDAR")
    print(f"    Sweet Spot (2300-4000): {sweet_spot} ({sweet_spot/len(reynolds_values)*100:.1f}%) - OPERAR")
    print(f"    Turbulento (Re > 4000): {turbulent} ({turbulent/len(reynolds_values)*100:.1f}%) - PERIGO")

    print(f"\nKL Divergence (Gatilho Direcional):")
    print(f"  Min:    {min(kl_divergences):.6f}")
    print(f"  Max:    {max(kl_divergences):.6f}")
    print(f"  Mean:   {np.mean(kl_divergences):.6f}")
    print(f"  Median: {np.median(kl_divergences):.6f}")
    print(f"  Std:    {np.std(kl_divergences):.6f}")

    print(f"\nSkewness (Assimetria):")
    print(f"  Min:    {min(skewness_values):.4f}")
    print(f"  Max:    {max(skewness_values):.4f}")
    print(f"  Mean:   {np.mean(skewness_values):.4f}")
    print(f"  Median: {np.median(skewness_values):.4f}")
    print(f"  Std:    {np.std(skewness_values):.4f}")

    # Distribuicao de skewness
    pos_skew = sum(1 for s in skewness_values if s > 0.5)
    neg_skew = sum(1 for s in skewness_values if s < -0.5)
    neutral_skew = sum(1 for s in skewness_values if -0.5 <= s <= 0.5)

    print(f"\n  Distribuicao:")
    print(f"    Positiva (>0.5): {pos_skew} ({pos_skew/len(skewness_values)*100:.1f}%) - TENDENCIA ALTA")
    print(f"    Negativa (<-0.5): {neg_skew} ({neg_skew/len(skewness_values)*100:.1f}%) - TENDENCIA BAIXA")
    print(f"    Neutra (-0.5 a 0.5): {neutral_skew} ({neutral_skew/len(skewness_values)*100:.1f}%) - SEM DIRECAO")

    print(f"\nGradiente de Pressao:")
    print(f"  Min:    {min(pressure_gradients):.6f}")
    print(f"  Max:    {max(pressure_gradients):.6f}")
    print(f"  Mean:   {np.mean(pressure_gradients):.6f}")
    print(f"  Median: {np.median(pressure_gradients):.6f}")
    print(f"  Std:    {np.std(pressure_gradients):.6f}")

    print(f"\nFisher Information:")
    print(f"  Min:    {min(fisher_values):.4f}")
    print(f"  Max:    {max(fisher_values):.4f}")
    print(f"  Mean:   {np.mean(fisher_values):.4f}")
    print(f"  Median: {np.median(fisher_values):.4f}")
    print(f"  Std:    {np.std(fisher_values):.4f}")

    print(f"\nIn Sweet Spot:")
    sweet_count = sum(in_sweet_spots)
    print(f"  Sim: {sweet_count} ({sweet_count/len(in_sweet_spots)*100:.1f}%)")
    print(f"  Nao: {len(in_sweet_spots) - sweet_count} ({(1-sweet_count/len(in_sweet_spots))*100:.1f}%)")

    # Estatisticas de direcao
    if directions:
        print(f"\nDirecao (baseada em barras FECHADAS):")
        long_count = directions.count(1)
        short_count = directions.count(-1)
        neutral_count = directions.count(0)
        total = len(directions)
        print(f"  Long:    {long_count} ({long_count/total*100:.1f}%)")
        print(f"  Short:   {short_count} ({short_count/total*100:.1f}%)")
        print(f"  Neutral: {neutral_count} ({neutral_count/total*100:.1f}%)")

    # Contar sinais com diferentes thresholds
    print("\n" + "=" * 70)
    print("  CONTAGEM DE SINAIS POR THRESHOLD")
    print("=" * 70)

    print("\n  Reynolds Sweet Spot e Skewness >= X:")
    print(f"  {'Re_Low':<10} | {'Re_High':<10} | {'Skew=0.3':<10} | {'Skew=0.4':<10} | {'Skew=0.5':<10} | {'Skew=0.6':<10}")
    print("  " + "-" * 70)

    for re_low in [2000, 2200, 2300, 2500]:
        re_high = 4000
        counts = []
        for skew_t in [0.3, 0.4, 0.5, 0.6]:
            count = sum(1 for r, s in zip(reynolds_values, skewness_values)
                       if re_low <= r <= re_high and abs(s) >= skew_t)
            counts.append(count)
        print(f"  {re_low:<10} | {re_high:<10} | {counts[0]:<10} | {counts[1]:<10} | {counts[2]:<10} | {counts[3]:<10}")

    # Filtrar por KL divergence tambem
    print("\n  Com filtro de KL Divergence:")
    for re_low in [2200, 2300, 2500]:
        for skew_t in [0.3, 0.4, 0.5]:
            for kl_t in [0.01, 0.02, 0.03]:
                count = sum(1 for r, s, k in zip(reynolds_values, skewness_values, kl_divergences)
                           if re_low <= r <= 4000 and abs(s) >= skew_t and k >= kl_t)
                print(f"    Re>={re_low}, Skew>={skew_t}, KL>={kl_t}: {count} sinais")

    # Sinais com direcao
    if directions:
        print("\n" + "=" * 70)
        print("  SINAIS COM DIRECAO (usando barras FECHADAS)")
        print("=" * 70)

        for re_low in [2200, 2300, 2500]:
            for skew_t in [0.3, 0.4, 0.5]:
                # Long signals: skewness positiva, pressao negativa, direcao = 1
                long_signals = sum(1 for r, s, p, d in zip(reynolds_values, skewness_values, pressure_gradients, directions)
                                  if re_low <= r <= 4000 and s > skew_t and p < 0 and d == 1)
                # Short signals: skewness negativa, pressao positiva, direcao = -1
                short_signals = sum(1 for r, s, p, d in zip(reynolds_values, skewness_values, pressure_gradients, directions)
                                   if re_low <= r <= 4000 and s < -skew_t and p > 0 and d == -1)
                total_signals = long_signals + short_signals
                print(f"    Re>={re_low}, Skew>={skew_t}: {total_signals} sinais (L:{long_signals}, S:{short_signals})")

    # Distribuicao temporal dos sinais
    print("\n" + "=" * 70)
    print("  DISTRIBUICAO TEMPORAL DOS SINAIS")
    print("=" * 70)

    # Usar threshold medio para analise
    re_low = 2300
    skew_t = 0.4

    # Contar sinais por periodo
    n_periods = 10
    period_size = len(reynolds_values) // n_periods

    print(f"\n  Sinais por periodo (Re>={re_low}, |Skew|>={skew_t}):")
    for p in range(n_periods):
        start_idx = p * period_size
        end_idx = start_idx + period_size if p < n_periods - 1 else len(reynolds_values)

        period_count = sum(1 for i in range(start_idx, end_idx)
                          if re_low <= reynolds_values[i] <= 4000 and abs(skewness_values[i]) >= skew_t)

        print(f"    Periodo {p+1}: {period_count} sinais")

    # Correlacoes
    print("\n" + "=" * 70)
    print("  CORRELACOES ENTRE METRICAS")
    print("=" * 70)

    if len(reynolds_values) > 10:
        corr_re_skew = np.corrcoef(reynolds_values, skewness_values)[0, 1]
        corr_re_kl = np.corrcoef(reynolds_values, kl_divergences)[0, 1]
        corr_skew_kl = np.corrcoef(skewness_values, kl_divergences)[0, 1]
        corr_re_fisher = np.corrcoef(reynolds_values, fisher_values)[0, 1]

        print(f"\n  Reynolds vs Skewness: {corr_re_skew:.4f}")
        print(f"  Reynolds vs KL Div:   {corr_re_kl:.4f}")
        print(f"  Skewness vs KL Div:   {corr_skew_kl:.4f}")
        print(f"  Reynolds vs Fisher:   {corr_re_fisher:.4f}")

    print("\n" + "=" * 70)
    print("  FIM DA ANALISE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
