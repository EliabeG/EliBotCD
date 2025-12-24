#!/usr/bin/env python3
"""
================================================================================
DEBUG DSG - Analise de Distribuicao de Sinais
================================================================================

Script de debug para entender o comportamento do DSG (Detector de Singularidade
Gravitacional) e analisar a distribuição de sinais com diferentes thresholds.

VERSÃO CORRIGIDA - Usa DSG sem look-ahead bias
==============================================
- EMA causal (não gaussian_filter1d)
- Direção baseada em barras fechadas
- Todos os parâmetros corrigidos

O DSG usa geometria pseudo-Riemanniana para detectar:
1. Escalar de Ricci (R) - Curvatura do espaço-tempo
2. Força de Maré - Desvio geodésico
3. Horizonte de Eventos - Ponto de não-retorno

Uso:
    python -m backtesting.dsg.debug
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
from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional


async def main():
    print("=" * 70)
    print("  DEBUG DSG - Detector de Singularidade Gravitacional")
    print("  Analise de Distribuicao de Sinais")
    print("  VERSAO CORRIGIDA - Sem Look-Ahead")
    print("=" * 70)

    # Carregar dados
    print("\nCarregando dados historicos...")
    bars = await download_historical_data(
        'EURUSD', 'H1',
        datetime(2025, 7, 1, tzinfo=timezone.utc),
        datetime.now(timezone.utc)
    )
    print(f"Barras carregadas: {len(bars)}")

    if len(bars) < 100:
        print("ERRO: Dados insuficientes!")
        return

    # Calcular DSG com parâmetros CORRIGIDOS
    dsg = DetectorSingularidadeGravitacional(
        ricci_collapse_threshold=-0.5,
        tidal_force_threshold=0.1,
        event_horizon_threshold=0.001,
        lookback_window=30
    )

    prices_buf = deque(maxlen=500)

    # Armazenar resultados
    ricci_scalars = []
    tidal_forces = []
    event_horizon_distances = []
    curvature_classes = []
    geodesic_directions = []
    ricci_collapsing_flags = []
    crossing_horizon_flags = []
    signals = []

    print("\nCalculando valores DSG...")
    print("NOTA: DSG usa cálculo tensorial intensivo (pode ser mais lento)")

    errors = 0
    for i, bar in enumerate(bars):
        prices_buf.append(bar.close)

        if len(prices_buf) < 50:
            continue

        try:
            result = dsg.analyze(np.array(prices_buf))

            ricci_scalars.append(result['Ricci_Scalar'])
            tidal_forces.append(result['Tidal_Force_Magnitude'])
            event_horizon_distances.append(result['Event_Horizon_Distance'])
            curvature_classes.append(result['curvature_class']['class'])
            geodesic_directions.append(result['geodesic_direction'])
            ricci_collapsing_flags.append(result['ricci_collapsing'])
            crossing_horizon_flags.append(result['crossing_horizon'])
            signals.append(result['signal'])

        except Exception as e:
            errors += 1
            if errors <= 3:  # Só mostrar primeiros erros
                print(f"  Erro na barra {i}: {e}")
            continue

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(bars)} barras processadas...")

    print(f"\nPontos calculados: {len(ricci_scalars)}")
    if errors > 0:
        print(f"Erros encontrados: {errors}")

    if len(ricci_scalars) == 0:
        print("ERRO: Nenhum ponto calculado!")
        return

    # Estatisticas
    print("\n" + "=" * 70)
    print("  ESTATISTICAS DOS VALORES DSG")
    print("=" * 70)

    print(f"\nEscalar de Ricci (R):")
    print(f"  Min:    {min(ricci_scalars):.6f}")
    print(f"  Max:    {max(ricci_scalars):.6f}")
    print(f"  Mean:   {np.mean(ricci_scalars):.6f}")
    print(f"  Median: {np.median(ricci_scalars):.6f}")
    print(f"  Std:    {np.std(ricci_scalars):.6f}")

    print(f"\nForca de Mare (Tidal Force):")
    print(f"  Min:    {min(tidal_forces):.6f}")
    print(f"  Max:    {max(tidal_forces):.6f}")
    print(f"  Mean:   {np.mean(tidal_forces):.6f}")
    print(f"  Median: {np.median(tidal_forces):.6f}")
    print(f"  Std:    {np.std(tidal_forces):.6f}")

    print(f"\nDistancia ao Horizonte de Eventos:")
    print(f"  Min:    {min(event_horizon_distances):.6f}")
    print(f"  Max:    {max(event_horizon_distances):.6f}")
    print(f"  Mean:   {np.mean(event_horizon_distances):.6f}")
    print(f"  Median: {np.median(event_horizon_distances):.6f}")
    print(f"  Std:    {np.std(event_horizon_distances):.6f}")

    # Classes de curvatura
    print(f"\nClasses de Curvatura:")
    unique_classes = set(curvature_classes)
    for c in sorted(unique_classes):
        count = curvature_classes.count(c)
        pct = count / len(curvature_classes) * 100
        print(f"  {c}: {count} ({pct:.1f}%)")

    # Direções geodésicas (baseada em barras FECHADAS)
    print(f"\nDirecao Geodesica (baseada em barras FECHADAS):")
    long_count = geodesic_directions.count(1)
    short_count = geodesic_directions.count(-1)
    neutral_count = geodesic_directions.count(0)
    total = len(geodesic_directions)
    print(f"  Long (1):    {long_count} ({long_count/total*100:.1f}%)")
    print(f"  Short (-1):  {short_count} ({short_count/total*100:.1f}%)")
    print(f"  Neutral (0): {neutral_count} ({neutral_count/total*100:.1f}%)")

    # Flags
    print(f"\nFlags de Singularidade:")
    ricci_col_count = sum(ricci_collapsing_flags)
    crossing_count = sum(crossing_horizon_flags)
    print(f"  Ricci Colapsando: {ricci_col_count} ({ricci_col_count/total*100:.1f}%)")
    print(f"  Cruzando Horizonte: {crossing_count} ({crossing_count/total*100:.1f}%)")

    # Sinais gerados
    print(f"\nSinais Gerados:")
    signal_long = signals.count(1)
    signal_short = signals.count(-1)
    signal_neutral = signals.count(0)
    print(f"  Long:    {signal_long} ({signal_long/total*100:.1f}%)")
    print(f"  Short:   {signal_short} ({signal_short/total*100:.1f}%)")
    print(f"  Neutral: {signal_neutral} ({signal_neutral/total*100:.1f}%)")

    # Contar sinais com diferentes thresholds
    print("\n" + "=" * 70)
    print("  CONTAGEM DE SINAIS POR THRESHOLD")
    print("=" * 70)

    print("\n  Ricci < X e Tidal > Y:")
    print(f"  {'Ricci':<10} | {'Tidal=0.01':<12} | {'Tidal=0.05':<12} | {'Tidal=0.10':<12} | {'Tidal=0.15':<12}")
    print("  " + "-" * 65)

    for ricci_t in [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]:
        counts = []
        for tidal_t in [0.01, 0.05, 0.10, 0.15]:
            count = sum(1 for r, t in zip(ricci_scalars, tidal_forces)
                       if r < ricci_t and t > tidal_t)
            counts.append(count)
        print(f"  {ricci_t:<10.2f} | {counts[0]:<12} | {counts[1]:<12} | {counts[2]:<12} | {counts[3]:<12}")

    # Filtrar por condições combinadas
    print("\n  Condições combinadas (Ricci OU Tidal OU Crossing):")
    for ricci_t in [-0.3, -0.5, -0.7]:
        for tidal_t in [0.05, 0.10, 0.15]:
            count_2cond = 0
            count_3cond = 0
            for r, t, c in zip(ricci_scalars, tidal_forces, crossing_horizon_flags):
                conditions = sum([r < ricci_t, t > tidal_t, c])
                if conditions >= 2:
                    count_2cond += 1
                if conditions >= 3:
                    count_3cond += 1
            print(f"    ricci<{ricci_t}, tidal>{tidal_t}: 2+ cond={count_2cond}, 3 cond={count_3cond}")

    # Sinais com direção
    print("\n" + "=" * 70)
    print("  SINAIS COM DIRECAO (usando barras FECHADAS)")
    print("=" * 70)

    for ricci_t in [-0.3, -0.5, -0.7]:
        for tidal_t in [0.05, 0.10, 0.15]:
            long_signals = 0
            short_signals = 0
            for r, t, c, d in zip(ricci_scalars, tidal_forces, crossing_horizon_flags, geodesic_directions):
                conditions = sum([r < ricci_t, t > tidal_t, c])
                if conditions >= 2:
                    if d == 1:
                        long_signals += 1
                    elif d == -1:
                        short_signals += 1
            total_signals = long_signals + short_signals
            print(f"    ricci<{ricci_t}, tidal>{tidal_t}: {total_signals} sinais (L:{long_signals}, S:{short_signals})")

    # Distribuição temporal dos sinais
    print("\n" + "=" * 70)
    print("  DISTRIBUICAO TEMPORAL DOS SINAIS")
    print("=" * 70)

    # Usar threshold médio para análise
    ricci_t = -0.5
    tidal_t = 0.10

    # Contar sinais por período
    n_periods = 10
    period_size = len(ricci_scalars) // n_periods

    print(f"\n  Sinais por periodo (ricci<{ricci_t}, tidal>{tidal_t}, 2+ cond):")
    for p in range(n_periods):
        start_idx = p * period_size
        end_idx = start_idx + period_size if p < n_periods - 1 else len(ricci_scalars)

        period_count = 0
        for i in range(start_idx, end_idx):
            conditions = sum([
                ricci_scalars[i] < ricci_t,
                tidal_forces[i] > tidal_t,
                crossing_horizon_flags[i]
            ])
            if conditions >= 2:
                period_count += 1

        print(f"    Periodo {p+1}: {period_count} sinais")

    # Análise de correlação entre indicadores
    print("\n" + "=" * 70)
    print("  CORRELACAO ENTRE INDICADORES")
    print("=" * 70)

    ricci_arr = np.array(ricci_scalars)
    tidal_arr = np.array(tidal_forces)
    distance_arr = np.array(event_horizon_distances)

    corr_ricci_tidal = np.corrcoef(ricci_arr, tidal_arr)[0, 1]
    corr_ricci_dist = np.corrcoef(ricci_arr, distance_arr)[0, 1]
    corr_tidal_dist = np.corrcoef(tidal_arr, distance_arr)[0, 1]

    print(f"\n  Correlacoes:")
    print(f"    Ricci vs Tidal:    {corr_ricci_tidal:.4f}")
    print(f"    Ricci vs Distance: {corr_ricci_dist:.4f}")
    print(f"    Tidal vs Distance: {corr_tidal_dist:.4f}")

    # Percentis
    print(f"\n  Percentis do Escalar de Ricci:")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        val = np.percentile(ricci_arr, p)
        print(f"    P{p}: {val:.6f}")

    print(f"\n  Percentis da Forca de Mare:")
    for p in percentiles:
        val = np.percentile(tidal_arr, p)
        print(f"    P{p}: {val:.6f}")

    # Sugestão de thresholds
    print("\n" + "=" * 70)
    print("  SUGESTAO DE THRESHOLDS")
    print("=" * 70)

    # Baseado nos percentis
    ricci_suggested = np.percentile(ricci_arr, 10)  # P10 (muito negativo)
    tidal_suggested = np.percentile(tidal_arr, 90)  # P90 (alto)

    print(f"\n  Baseado nos percentis:")
    print(f"    Ricci (P10): {ricci_suggested:.4f}")
    print(f"    Tidal (P90): {tidal_suggested:.6f}")

    # Testar com sugestão
    suggested_count = 0
    for r, t, c in zip(ricci_scalars, tidal_forces, crossing_horizon_flags):
        conditions = sum([r < ricci_suggested, t > tidal_suggested, c])
        if conditions >= 2:
            suggested_count += 1

    print(f"\n  Sinais com thresholds sugeridos: {suggested_count}")
    print(f"  Taxa de sinais: {suggested_count/len(ricci_scalars)*100:.2f}%")

    print("\n" + "=" * 70)
    print("  FIM DA ANALISE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
