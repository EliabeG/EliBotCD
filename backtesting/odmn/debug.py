#!/usr/bin/env python3
"""
================================================================================
DEBUG ODMN - Analise de Distribuicao de Sinais
================================================================================

Script de debug para entender o comportamento do ODMN e analisar
a distribuicao de sinais com diferentes thresholds.

VERSAO SEM LOOK-AHEAD BIAS
==========================
- Calibracao Heston usa apenas dados passados
- Malliavin simula trajetorias para frente (causal)
- MFG resolve PDEs sem usar dados futuros
- Direcao baseada em barras fechadas

COMPONENTES DO ODMN:
====================
1. Modelo de Heston: kappa, theta, sigma, rho, v0
2. Indice de Fragilidade de Malliavin: sensibilidade ao ruido
3. Direcao MFG: pressao institucional (Mean Field Games)
4. Regime de Mercado: NORMAL, HIGH_VOL, HIGH_FRAGILITY, CRITICAL

Uso:
    python -m backtesting.odmn.debug
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
from strategies.alta_volatilidade.odmn_malliavin_nash import OracloDerivativosMalliavinNash

# Importar config centralizado
try:
    from config.odmn_config import (
        MIN_PRICES,
        HESTON_CALIBRATION_WINDOW,
        FRAGILITY_PERCENTILE_THRESHOLD,
        MFG_DIRECTION_THRESHOLD,
        MIN_CONFIDENCE,
        USE_DEEP_GALERKIN,
        MALLIAVIN_PATHS,
        MALLIAVIN_STEPS,
        TREND_LOOKBACK,
    )
except ImportError:
    MIN_PRICES = 150
    HESTON_CALIBRATION_WINDOW = 100
    FRAGILITY_PERCENTILE_THRESHOLD = 0.80
    MFG_DIRECTION_THRESHOLD = 0.1
    MIN_CONFIDENCE = 0.60
    USE_DEEP_GALERKIN = False
    MALLIAVIN_PATHS = 1000
    MALLIAVIN_STEPS = 30
    TREND_LOOKBACK = 10

# Importar direction_calculator centralizado
try:
    from backtesting.common.direction_calculator import calculate_direction_from_bars
except ImportError:
    calculate_direction_from_bars = None


async def main():
    print("=" * 70)
    print("  DEBUG ODMN - Analise de Distribuicao de Sinais")
    print("  VERSAO SEM LOOK-AHEAD BIAS")
    print("=" * 70)
    print(f"\n  Config Centralizado:")
    print(f"    MIN_PRICES: {MIN_PRICES}")
    print(f"    HESTON_CALIBRATION_WINDOW: {HESTON_CALIBRATION_WINDOW}")
    print(f"    FRAGILITY_THRESHOLD: P{FRAGILITY_PERCENTILE_THRESHOLD*100:.0f}")
    print(f"    MFG_DIRECTION_THRESHOLD: {MFG_DIRECTION_THRESHOLD}")

    # Carregar dados
    print("\nCarregando dados historicos...")
    bars = await download_historical_data(
        'EURUSD', 'H1',
        datetime(2025, 7, 1, tzinfo=timezone.utc),
        datetime.now(timezone.utc)
    )
    print(f"Barras carregadas: {len(bars)}")

    if len(bars) < 200:
        print("ERRO: Dados insuficientes!")
        return

    # Criar ODMN com config centralizado
    odmn = OracloDerivativosMalliavinNash(
        lookback_window=HESTON_CALIBRATION_WINDOW,
        fragility_threshold=2.0,
        mfg_direction_threshold=MFG_DIRECTION_THRESHOLD,
        use_deep_galerkin=USE_DEEP_GALERKIN,  # Analitico e mais rapido
        malliavin_paths=MALLIAVIN_PATHS,
        malliavin_steps=MALLIAVIN_STEPS
    )

    prices_buf = deque(maxlen=500)

    # Coleta de valores
    fragility_indices = []
    fragility_percentiles = []
    mfg_directions = []
    mfg_equilibria = []
    regimes = []
    signals = []
    confidences = []
    implied_vols = []

    # Parametros Heston calibrados
    heston_kappas = []
    heston_thetas = []
    heston_sigmas = []
    heston_rhos = []

    # Estatisticas de direcao (baseada em barras FECHADAS)
    directions = []
    # Usa TREND_LOOKBACK do config para consistencia
    min_bars_for_direction = TREND_LOOKBACK + 2

    print("\nCalculando valores ODMN...")
    print("NOTA: Computacionalmente intensivo (Monte Carlo + Heston)")

    errors = 0
    for i, bar in enumerate(bars):
        prices_buf.append(bar.close)

        if len(prices_buf) < MIN_PRICES:
            continue

        try:
            result = odmn.analyze(np.array(prices_buf))

            fragility_indices.append(result['fragility_index'])
            fragility_percentiles.append(result['fragility_percentile'])
            mfg_directions.append(result['mfg_direction'])
            mfg_equilibria.append(result['mfg_equilibrium'])
            regimes.append(result['regime'])
            signals.append(result['signal'])
            confidences.append(result['confidence'])
            implied_vols.append(result['implied_vol'])

            # Heston params
            if result['heston_params']:
                heston_kappas.append(result['heston_params']['kappa'])
                heston_thetas.append(result['heston_params']['theta'])
                heston_sigmas.append(result['heston_params']['sigma'])
                heston_rhos.append(result['heston_params']['rho'])

            # Direcao baseada em barras FECHADAS - usando direction_calculator centralizado
            if calculate_direction_from_bars is not None:
                direction = calculate_direction_from_bars(bars, i, lookback=TREND_LOOKBACK)
            elif i >= min_bars_for_direction:
                # Fallback: calculo manual consistente com direction_calculator
                recent_close = bars[i - 1].close   # Ultima barra FECHADA
                past_close = bars[i - TREND_LOOKBACK - 1].close  # TREND_LOOKBACK barras antes
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

    print(f"\nPontos calculados: {len(fragility_indices)}")
    if errors > 0:
        print(f"Erros encontrados: {errors}")

    if len(fragility_indices) == 0:
        print("ERRO: Nenhum ponto calculado!")
        return

    # Estatisticas
    print("\n" + "=" * 70)
    print("  ESTATISTICAS DOS VALORES ODMN")
    print("=" * 70)

    print(f"\nFragilidade de Malliavin (Indice):")
    print(f"  Min:    {min(fragility_indices):.6f}")
    print(f"  Max:    {max(fragility_indices):.6f}")
    print(f"  Mean:   {np.mean(fragility_indices):.6f}")
    print(f"  Median: {np.median(fragility_indices):.6f}")
    print(f"  Std:    {np.std(fragility_indices):.6f}")

    print(f"\nFragilidade de Malliavin (Percentil):")
    print(f"  Min:    {min(fragility_percentiles)*100:.1f}%")
    print(f"  Max:    {max(fragility_percentiles)*100:.1f}%")
    print(f"  Mean:   {np.mean(fragility_percentiles)*100:.1f}%")
    print(f"  Median: {np.median(fragility_percentiles)*100:.1f}%")

    print(f"\nDirecao MFG (Mean Field Games):")
    print(f"  Min:    {min(mfg_directions):.6f}")
    print(f"  Max:    {max(mfg_directions):.6f}")
    print(f"  Mean:   {np.mean(mfg_directions):.6f}")
    print(f"  Median: {np.median(mfg_directions):.6f}")
    print(f"  Std:    {np.std(mfg_directions):.6f}")

    print(f"\nVolatilidade Implicita (Heston):")
    print(f"  Min:    {min(implied_vols)*100:.2f}%")
    print(f"  Max:    {max(implied_vols)*100:.2f}%")
    print(f"  Mean:   {np.mean(implied_vols)*100:.2f}%")
    print(f"  Median: {np.median(implied_vols)*100:.2f}%")

    print(f"\nEquilibrio MFG:")
    equilibrium_count = sum(1 for e in mfg_equilibria if e)
    print(f"  Convergiu: {equilibrium_count} ({equilibrium_count/len(mfg_equilibria)*100:.1f}%)")
    print(f"  Nao convergiu: {len(mfg_equilibria) - equilibrium_count}")

    print(f"\nRegimes de Mercado:")
    regime_counts = {}
    for r in regimes:
        regime_counts[r] = regime_counts.get(r, 0) + 1
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        pct = count / len(regimes) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")

    print(f"\nSinais Gerados:")
    buy_signals = sum(1 for s in signals if s == 1)
    sell_signals = sum(1 for s in signals if s == -1)
    hold_signals = sum(1 for s in signals if s == 0)
    print(f"  BUY:  {buy_signals} ({buy_signals/len(signals)*100:.1f}%)")
    print(f"  SELL: {sell_signals} ({sell_signals/len(signals)*100:.1f}%)")
    print(f"  HOLD: {hold_signals} ({hold_signals/len(signals)*100:.1f}%)")

    print(f"\nConfianca:")
    print(f"  Min:    {min(confidences):.2f}")
    print(f"  Max:    {max(confidences):.2f}")
    print(f"  Mean:   {np.mean(confidences):.2f}")

    # Parametros Heston
    if heston_kappas:
        print(f"\n" + "=" * 70)
        print("  PARAMETROS HESTON CALIBRADOS")
        print("=" * 70)

        print(f"\nkappa (velocidade de reversao):")
        print(f"  Mean: {np.mean(heston_kappas):.4f}, Std: {np.std(heston_kappas):.4f}")

        print(f"\ntheta (variancia media):")
        print(f"  Mean: {np.mean(heston_thetas):.6f}, Std: {np.std(heston_thetas):.6f}")

        print(f"\nsigma (vol da vol):")
        print(f"  Mean: {np.mean(heston_sigmas):.4f}, Std: {np.std(heston_sigmas):.4f}")

        print(f"\nrho (correlacao):")
        print(f"  Mean: {np.mean(heston_rhos):.4f}, Std: {np.std(heston_rhos):.4f}")
        print(f"  Nota: rho tipicamente negativo indica 'leverage effect'")

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

    # Contagem de sinais por threshold
    print("\n" + "=" * 70)
    print("  CONTAGEM DE SINAIS POR THRESHOLD")
    print("=" * 70)

    print("\n  Fragilidade >= X% (Percentil) e |MFG| >= Y:")
    print(f"  {'Frag%':<10} | {'MFG=0.05':<12} | {'MFG=0.08':<12} | {'MFG=0.10':<12} | {'MFG=0.15':<12}")
    print("  " + "-" * 70)

    for frag_t in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
        counts = []
        for mfg_t in [0.05, 0.08, 0.10, 0.15]:
            count = sum(1 for fp, md in zip(fragility_percentiles, mfg_directions)
                       if fp >= frag_t and abs(md) >= mfg_t)
            counts.append(count)
        print(f"  P{frag_t*100:<7.0f} | {counts[0]:<12} | {counts[1]:<12} | {counts[2]:<12} | {counts[3]:<12}")

    # Sinais com direcao
    if directions:
        print("\n" + "=" * 70)
        print("  SINAIS COM DIRECAO (usando barras FECHADAS)")
        print("=" * 70)

        for frag_t in [0.70, 0.75, 0.80]:
            for mfg_t in [0.08, 0.10, 0.12]:
                long_signals = sum(1 for fp, md, d in zip(fragility_percentiles, mfg_directions, directions)
                                  if fp >= frag_t and abs(md) >= mfg_t and d == 1)
                short_signals = sum(1 for fp, md, d in zip(fragility_percentiles, mfg_directions, directions)
                                   if fp >= frag_t and abs(md) >= mfg_t and d == -1)
                total_signals = long_signals + short_signals
                print(f"    frag>=P{frag_t*100:.0f}, |mfg|>={mfg_t}: {total_signals} sinais (L:{long_signals}, S:{short_signals})")

    # Sinais com alta confianca
    print("\n" + "=" * 70)
    print("  SINAIS COM ALTA CONFIANCA")
    print("=" * 70)

    for conf_t in [0.50, 0.60, 0.70, 0.80]:
        buy_high_conf = sum(1 for s, c in zip(signals, confidences) if s == 1 and c >= conf_t)
        sell_high_conf = sum(1 for s, c in zip(signals, confidences) if s == -1 and c >= conf_t)
        print(f"    Confianca >= {conf_t:.0%}: BUY={buy_high_conf}, SELL={sell_high_conf}, TOTAL={buy_high_conf + sell_high_conf}")

    # Distribuicao temporal dos sinais
    print("\n" + "=" * 70)
    print("  DISTRIBUICAO TEMPORAL DOS SINAIS")
    print("=" * 70)

    # Usar threshold configurado para analise
    frag_t = FRAGILITY_PERCENTILE_THRESHOLD
    mfg_t = MFG_DIRECTION_THRESHOLD

    # Contar sinais por periodo
    n_periods = 10
    period_size = len(fragility_percentiles) // n_periods

    print(f"\n  Sinais por periodo (frag>=P{frag_t*100:.0f}, |mfg|>={mfg_t}):")
    for p in range(n_periods):
        start_idx = p * period_size
        end_idx = start_idx + period_size if p < n_periods - 1 else len(fragility_percentiles)

        period_count = sum(1 for i in range(start_idx, end_idx)
                          if fragility_percentiles[i] >= frag_t and abs(mfg_directions[i]) >= mfg_t)

        print(f"    Periodo {p+1}: {period_count} sinais")

    # Analise de regimes vs sinais
    print("\n" + "=" * 70)
    print("  SINAIS POR REGIME DE MERCADO")
    print("=" * 70)

    for regime in set(regimes):
        regime_indices = [i for i, r in enumerate(regimes) if r == regime]
        if regime_indices:
            buy_count = sum(1 for i in regime_indices if signals[i] == 1)
            sell_count = sum(1 for i in regime_indices if signals[i] == -1)
            avg_conf = np.mean([confidences[i] for i in regime_indices])
            print(f"  {regime}:")
            print(f"    Ocorrencias: {len(regime_indices)}")
            print(f"    BUY: {buy_count}, SELL: {sell_count}")
            print(f"    Confianca media: {avg_conf:.2f}")

    print("\n" + "=" * 70)
    print("  FIM DA ANALISE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
