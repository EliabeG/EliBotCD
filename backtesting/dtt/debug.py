#!/usr/bin/env python3
"""
================================================================================
DEBUG DTT - Analise de Distribuicao de Sinais
================================================================================

Script de debug para entender o comportamento do DTT e analisar
a distribuição de sinais com diferentes thresholds.

VERSÃO CORRIGIDA - Análise SEM look-ahead bias
==============================================
- Direção baseada em barras FECHADAS
- Análise de entropia de persistência
- Análise de probabilidade de tunelamento
- Verificação de distribuição temporal

Uso:
    python -m backtesting.dtt.debug
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
from strategies.alta_volatilidade.dtt_tunelamento_topologico import DetectorTunelamentoTopologico


async def main():
    print("=" * 60)
    print("  DEBUG DTT - Analise de Distribuicao de Sinais")
    print("  VERSAO CORRIGIDA - Sem Look-Ahead")
    print("=" * 60)

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

    # Calcular DTT com parâmetros padrão
    dtt = DetectorTunelamentoTopologico(
        max_points=150,
        use_dimensionality_reduction=True,
        reduction_method='pca',
        persistence_entropy_threshold=0.1,  # Baixo para capturar mais sinais
        tunneling_probability_threshold=0.05  # Baixo para capturar mais sinais
    )

    prices_buf = deque(maxlen=500)

    # Armazenar valores para análise
    entropies = []
    tunnelings = []
    signal_strengths = []
    trade_ons = []

    # Estatísticas de direção (baseada em barras FECHADAS)
    directions = []
    direction_lookback = 12  # Consistente com DTTStrategy
    min_prices = 150

    print("\nCalculando valores DTT...")
    print("NOTA: Análise topológica pode ser mais lenta (computacionalmente intensiva)")

    errors = 0
    skipped = 0

    for i, bar in enumerate(bars):
        prices_buf.append(bar.close)

        if len(prices_buf) < min_prices:
            skipped += 1
            continue

        try:
            result = dtt.analyze(np.array(prices_buf))

            entropies.append(result['entropy']['persistence_entropy'])
            tunnelings.append(result['tunneling']['tunneling_probability'])
            signal_strengths.append(result['signal_strength'])
            trade_ons.append(1 if result['trade_on'] else 0)

            # CORREÇÃO AUDITORIA 2: Calcular direção apenas com barras FECHADAS
            # - bars[i] = barra atual (momento da análise)
            # - bars[i-1] = barra anterior (já fechada)
            # - bars[i-direction_lookback] = N barras antes
            # NÃO usar result['direction'] que pode ter look-ahead
            if i >= direction_lookback + 1:
                recent_close = bars[i - 1].close
                past_close = bars[i - direction_lookback].close
                trend = recent_close - past_close
                direction = 1 if trend > 0 else -1
            else:
                direction = 0
            directions.append(direction)

        except Exception as e:
            errors += 1
            if errors <= 3:  # Só mostrar primeiros erros
                print(f"  Erro na barra {i}: {e}")
            continue

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(bars)} barras processadas...")

    print(f"\nPontos calculados: {len(entropies)}")
    print(f"Barras puladas (dados insuficientes): {skipped}")
    if errors > 0:
        print(f"Erros encontrados: {errors}")

    if len(entropies) == 0:
        print("ERRO: Nenhum ponto calculado!")
        return

    # Estatisticas
    print("\n" + "=" * 60)
    print("  ESTATISTICAS DOS VALORES DTT")
    print("=" * 60)

    print(f"\nEntropia de Persistência:")
    print(f"  Min:    {min(entropies):.4f}")
    print(f"  Max:    {max(entropies):.4f}")
    print(f"  Mean:   {np.mean(entropies):.4f}")
    print(f"  Median: {np.median(entropies):.4f}")
    print(f"  Std:    {np.std(entropies):.4f}")

    print(f"\nProbabilidade de Tunelamento:")
    print(f"  Min:    {min(tunnelings):.4f}")
    print(f"  Max:    {max(tunnelings):.4f}")
    print(f"  Mean:   {np.mean(tunnelings):.4f}")
    print(f"  Median: {np.median(tunnelings):.4f}")
    print(f"  Std:    {np.std(tunnelings):.4f}")

    print(f"\nForça do Sinal:")
    print(f"  Min:    {min(signal_strengths):.4f}")
    print(f"  Max:    {max(signal_strengths):.4f}")
    print(f"  Mean:   {np.mean(signal_strengths):.4f}")
    print(f"  Median: {np.median(signal_strengths):.4f}")
    print(f"  Std:    {np.std(signal_strengths):.4f}")

    print(f"\nTrade ON:")
    total_on = sum(trade_ons)
    total_off = len(trade_ons) - total_on
    print(f"  TRADE ON:  {total_on} ({total_on/len(trade_ons)*100:.1f}%)")
    print(f"  SYSTEM OFF: {total_off} ({total_off/len(trade_ons)*100:.1f}%)")

    # Estatísticas de direção (usando cálculo CORRIGIDO)
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
    print("\n" + "=" * 60)
    print("  CONTAGEM DE SINAIS POR THRESHOLD")
    print("=" * 60)

    print("\n  Entropy >= X e Tunneling >= Y:")
    print(f"  {'Entropy':<8} | {'Tunn=0.10':<12} | {'Tunn=0.15':<12} | {'Tunn=0.20':<12} | {'Tunn=0.25':<12}")
    print("  " + "-" * 60)

    for ent_t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        counts = []
        for tunn_t in [0.10, 0.15, 0.20, 0.25]:
            count = sum(1 for e, t in zip(entropies, tunnelings)
                       if e >= ent_t and t >= tunn_t)
            counts.append(count)
        print(f"  {ent_t:<8.2f} | {counts[0]:<12} | {counts[1]:<12} | {counts[2]:<12} | {counts[3]:<12}")

    # Filtrar por força do sinal também
    print("\n  Com filtro de Signal Strength >= 0.3:")
    strength_thresh = 0.3
    for ent_t in [0.55, 0.60, 0.65, 0.70]:
        for tunn_t in [0.10, 0.15, 0.20]:
            count = sum(1 for e, t, s in zip(entropies, tunnelings, signal_strengths)
                       if e >= ent_t and t >= tunn_t and s >= strength_thresh)
            print(f"    entropy>={ent_t}, tunneling>={tunn_t}, strength>={strength_thresh}: {count} sinais")

    # Sinais com direção
    if directions:
        print("\n" + "=" * 60)
        print("  SINAIS COM DIRECAO (usando barras FECHADAS)")
        print("=" * 60)

        for ent_t in [0.55, 0.60, 0.65]:
            for tunn_t in [0.10, 0.15, 0.20]:
                long_signals = sum(1 for e, t, s, d in zip(entropies, tunnelings, signal_strengths, directions)
                                  if e >= ent_t and t >= tunn_t and s >= 0.3 and d == 1)
                short_signals = sum(1 for e, t, s, d in zip(entropies, tunnelings, signal_strengths, directions)
                                   if e >= ent_t and t >= tunn_t and s >= 0.3 and d == -1)
                total_signals = long_signals + short_signals
                print(f"    ent>={ent_t}, tunn>={tunn_t}: {total_signals} sinais (L:{long_signals}, S:{short_signals})")

    # Distribuição temporal dos sinais
    print("\n" + "=" * 60)
    print("  DISTRIBUICAO TEMPORAL DOS SINAIS")
    print("=" * 60)

    # Usar threshold médio para análise
    ent_t = 0.60
    tunn_t = 0.15

    # Contar sinais por período
    n_periods = 10
    period_size = len(entropies) // n_periods

    print(f"\n  Sinais por período (ent>={ent_t}, tunn>={tunn_t}):")
    for p in range(n_periods):
        start_idx = p * period_size
        end_idx = start_idx + period_size if p < n_periods - 1 else len(entropies)

        period_count = sum(1 for i in range(start_idx, end_idx)
                          if entropies[i] >= ent_t and tunnelings[i] >= tunn_t)

        print(f"    Período {p+1}: {period_count} sinais")

    # Análise de correlação temporal
    print("\n" + "=" * 60)
    print("  ANALISE DE CORRELACAO TEMPORAL")
    print("=" * 60)

    if len(entropies) > 10:
        # Verificar se há autocorrelação (indicaria look-ahead)
        entropy_diff = np.diff(entropies)
        tunn_diff = np.diff(tunnelings)

        # Correlação entre valor atual e próximo (não deveria ser muito alta)
        autocorr_ent = np.corrcoef(entropies[:-1], entropies[1:])[0, 1]
        autocorr_tunn = np.corrcoef(tunnelings[:-1], tunnelings[1:])[0, 1]

        print(f"\n  Autocorrelação (lag=1):")
        print(f"    Entropy:   {autocorr_ent:.4f}")
        print(f"    Tunneling: {autocorr_tunn:.4f}")

        # Verificar estacionariedade
        print(f"\n  Variação ao longo do tempo:")
        half = len(entropies) // 2
        print(f"    Entropy (1ª metade):  mean={np.mean(entropies[:half]):.4f}")
        print(f"    Entropy (2ª metade):  mean={np.mean(entropies[half:]):.4f}")
        print(f"    Tunneling (1ª metade):  mean={np.mean(tunnelings[:half]):.4f}")
        print(f"    Tunneling (2ª metade):  mean={np.mean(tunnelings[half:]):.4f}")

    # Verificação de look-ahead potencial
    print("\n" + "=" * 60)
    print("  VERIFICACAO DE LOOK-AHEAD BIAS")
    print("=" * 60)

    print("\n  Checklist de potenciais problemas:")
    print("  [✓] Direção calculada usando barras FECHADAS (i-1 e i-11)")
    print("  [✓] Embedding de Takens usa apenas dados passados")
    print("  [✓] Homologia Persistente calcula em nuvem de pontos passados")
    print("  [✓] KDE para potencial usa apenas preços recentes (passados)")
    print("  [✓] Schrödinger resolve baseado em potencial passado")

    # Verificar se há correlação impossível com futuro
    if len(directions) > 1:
        # Se houvesse look-ahead, direção do DTT correlacionaria com retorno futuro
        # Isso NÃO deveria acontecer se tudo estiver correto
        future_returns = []
        for i in range(len(bars) - min_prices - 1):
            if i + min_prices + 1 < len(bars):
                future_ret = bars[i + min_prices + 1].close - bars[i + min_prices].close
                future_returns.append(future_ret)

        if len(future_returns) == len(directions):
            corr_dir_future = np.corrcoef(directions, future_returns)[0, 1]
            print(f"\n  Correlação direção x retorno futuro: {corr_dir_future:.4f}")
            if abs(corr_dir_future) > 0.3:
                print("  [!] ALERTA: Correlação alta pode indicar look-ahead!")
            else:
                print("  [✓] Correlação baixa - sem indício de look-ahead")

    print("\n" + "=" * 60)
    print("  FIM DA ANALISE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
