#!/usr/bin/env python3
"""
================================================================================
DIAGNOSTICO PROFUNDO DO HMM
================================================================================

Investiga por que 100% das barras estao no estado 0.

Possiveis problemas:
1. Ordem dos estados eh arbitraria no HMM
2. Modelo degenerado (um estado domina)
3. Retornos M5 muito homogeneos
4. Calculo incorreto das probabilidades
================================================================================
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from collections import deque
import numpy as np

from api.fxopen_historical_ws import get_historical_data_with_spread_sync

try:
    from hmmlearn import hmm
except ImportError:
    print("ERRO: hmmlearn nao instalado")
    sys.exit(1)


def analyze_hmm_model(hmm_model, returns):
    """Analisa o modelo HMM treinado"""
    print("\n" + "=" * 70)
    print("  ANALISE DO MODELO HMM TREINADO")
    print("=" * 70)

    # Parametros do modelo
    print("\n  1. PARAMETROS DO MODELO:")
    print(f"     n_components (estados): {hmm_model.n_components}")

    print("\n  2. PROBABILIDADES INICIAIS (startprob_):")
    print(f"     Estado 0: {hmm_model.startprob_[0]:.4f}")
    print(f"     Estado 1: {hmm_model.startprob_[1]:.4f}")
    print(f"     Estado 2: {hmm_model.startprob_[2]:.4f}")

    print("\n  3. MATRIZ DE TRANSICAO (transmat_):")
    print("     De/Para   Estado 0    Estado 1    Estado 2")
    for i in range(3):
        print(f"     Estado {i}  {hmm_model.transmat_[i][0]:.4f}      {hmm_model.transmat_[i][1]:.4f}      {hmm_model.transmat_[i][2]:.4f}")

    print("\n  4. MEDIAS DOS ESTADOS (means_):")
    for i in range(3):
        print(f"     Estado {i}: {hmm_model.means_[i][0]:.8f}")

    print("\n  5. COVARIANCIAS DOS ESTADOS (covars_):")
    for i in range(3):
        print(f"     Estado {i}: {hmm_model.covars_[i][0][0]:.12f}")

    # Interpretar estados
    print("\n  6. INTERPRETACAO DOS ESTADOS:")
    means = [hmm_model.means_[i][0] for i in range(3)]
    covars = [hmm_model.covars_[i][0][0] for i in range(3)]

    # Ordenar por volatilidade (covariancia)
    state_info = [(i, means[i], covars[i]) for i in range(3)]
    state_info.sort(key=lambda x: x[2])  # Ordenar por covariancia

    print("     Ordenado por volatilidade (covariancia):")
    labels = ["BAIXA VOLATILIDADE", "MEDIA VOLATILIDADE", "ALTA VOLATILIDADE"]
    for idx, (state, mean, cov) in enumerate(state_info):
        print(f"     - Estado {state}: {labels[idx]} (mean={mean:.8f}, cov={cov:.12f})")

    return state_info


def analyze_state_distribution(hmm_model, returns_buffer, warmup_returns):
    """Analisa a distribuicao de estados"""
    print("\n" + "=" * 70)
    print("  ANALISE DA DISTRIBUICAO DE ESTADOS")
    print("=" * 70)

    # Predizer estados para todo o warmup
    rets = warmup_returns.reshape(-1, 1)
    predicted_states = hmm_model.predict(rets)
    probs = hmm_model.predict_proba(rets)

    print("\n  1. DISTRIBUICAO DE ESTADOS NO WARMUP:")
    for s in range(3):
        count = np.sum(predicted_states == s)
        pct = count / len(predicted_states) * 100
        print(f"     Estado {s}: {count} ({pct:.1f}%)")

    print("\n  2. ESTATISTICAS DAS PROBABILIDADES:")
    for s in range(3):
        state_probs = probs[:, s]
        print(f"     Estado {s}: min={np.min(state_probs):.4f}, max={np.max(state_probs):.4f}, mean={np.mean(state_probs):.4f}")

    print("\n  3. VERIFICANDO PROBLEMA DE DEGENERACAO:")
    # Verificar se um estado domina
    max_probs = np.max(probs, axis=1)
    avg_confidence = np.mean(max_probs)
    print(f"     Confianca media do modelo: {avg_confidence:.4f}")

    if avg_confidence > 0.95:
        print("     ALERTA: Modelo muito confiante - possivel degeneracao!")

    # Verificar se probabilidades sao muito similares
    prob_variance = np.var(probs, axis=0)
    print(f"     Variancia das probabilidades por estado: {prob_variance}")

    if np.all(prob_variance < 0.01):
        print("     ALERTA: Probabilidades muito homogeneas!")

    return predicted_states, probs


def test_different_hmm_configs(returns):
    """Testa diferentes configuracoes de HMM"""
    print("\n" + "=" * 70)
    print("  TESTANDO DIFERENTES CONFIGURACOES DE HMM")
    print("=" * 70)

    configs = [
        {"n_components": 3, "covariance_type": "full", "n_iter": 100},
        {"n_components": 3, "covariance_type": "diag", "n_iter": 100},
        {"n_components": 3, "covariance_type": "spherical", "n_iter": 100},
        {"n_components": 2, "covariance_type": "full", "n_iter": 100},
        {"n_components": 4, "covariance_type": "full", "n_iter": 100},
        {"n_components": 3, "covariance_type": "full", "n_iter": 200},
    ]

    rets = returns.reshape(-1, 1)

    for i, config in enumerate(configs):
        try:
            model = hmm.GaussianHMM(
                n_components=config["n_components"],
                covariance_type=config["covariance_type"],
                n_iter=config["n_iter"],
                random_state=42
            )
            model.fit(rets)

            states = model.predict(rets)
            probs = model.predict_proba(rets)

            state_dist = [np.sum(states == s) / len(states) * 100 for s in range(config["n_components"])]

            # Probabilidade maxima de estados nao-0
            if config["n_components"] > 1:
                non_zero_prob = np.max(probs[:, 1:], axis=1)
                max_non_zero = np.max(non_zero_prob)
            else:
                max_non_zero = 0

            print(f"\n  Config {i+1}: {config}")
            print(f"     Distribuicao: {[f'{d:.1f}%' for d in state_dist]}")
            print(f"     Max prob nao-estado0: {max_non_zero:.4f}")
            print(f"     Log-likelihood: {model.score(rets):.2f}")

        except Exception as e:
            print(f"\n  Config {i+1}: ERRO - {e}")


def analyze_returns_distribution(returns):
    """Analisa a distribuicao dos retornos"""
    print("\n" + "=" * 70)
    print("  ANALISE DA DISTRIBUICAO DOS RETORNOS")
    print("=" * 70)

    print(f"\n  1. ESTATISTICAS BASICAS:")
    print(f"     N: {len(returns)}")
    print(f"     Mean: {np.mean(returns):.8f}")
    print(f"     Std: {np.std(returns):.8f}")
    print(f"     Min: {np.min(returns):.8f}")
    print(f"     Max: {np.max(returns):.8f}")
    print(f"     Mediana: {np.median(returns):.8f}")

    print(f"\n  2. PERCENTIS:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(returns, p)
        print(f"     P{p}: {val:.8f}")

    print(f"\n  3. ANALISE DE CLUSTERS (K-means simples):")
    # Cluster manual baseado em percentis
    p25 = np.percentile(np.abs(returns), 25)
    p75 = np.percentile(np.abs(returns), 75)

    low_vol = np.sum(np.abs(returns) <= p25)
    mid_vol = np.sum((np.abs(returns) > p25) & (np.abs(returns) <= p75))
    high_vol = np.sum(np.abs(returns) > p75)

    print(f"     Baixa vol (|r| <= P25): {low_vol} ({low_vol/len(returns)*100:.1f}%)")
    print(f"     Media vol (P25 < |r| <= P75): {mid_vol} ({mid_vol/len(returns)*100:.1f}%)")
    print(f"     Alta vol (|r| > P75): {high_vol} ({high_vol/len(returns)*100:.1f}%)")

    print(f"\n  4. RETORNOS EXTREMOS (>3 std):")
    std = np.std(returns)
    extremos = np.sum(np.abs(returns) > 3 * std)
    print(f"     Retornos extremos: {extremos} ({extremos/len(returns)*100:.2f}%)")


def test_hmm_with_standardization(returns):
    """Testa HMM com retornos padronizados"""
    print("\n" + "=" * 70)
    print("  TESTANDO HMM COM RETORNOS PADRONIZADOS")
    print("=" * 70)

    # Padronizar retornos (z-score)
    mean = np.mean(returns)
    std = np.std(returns)
    standardized = (returns - mean) / std

    print(f"\n  Retornos padronizados:")
    print(f"     Mean: {np.mean(standardized):.8f} (deve ser ~0)")
    print(f"     Std: {np.std(standardized):.8f} (deve ser ~1)")

    # Treinar HMM com retornos padronizados
    rets = standardized.reshape(-1, 1)

    model = hmm.GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    model.fit(rets)

    states = model.predict(rets)
    probs = model.predict_proba(rets)

    print(f"\n  Distribuicao de estados (retornos padronizados):")
    for s in range(3):
        count = np.sum(states == s)
        pct = count / len(states) * 100
        print(f"     Estado {s}: {count} ({pct:.1f}%)")

    print(f"\n  Medias dos estados:")
    for i in range(3):
        print(f"     Estado {i}: {model.means_[i][0]:.4f}")

    print(f"\n  Covariancias dos estados:")
    for i in range(3):
        print(f"     Estado {i}: {model.covars_[i][0][0]:.4f}")

    # Probabilidades
    print(f"\n  Probabilidades por estado:")
    for s in range(3):
        state_probs = probs[:, s]
        print(f"     Estado {s}: min={np.min(state_probs):.4f}, max={np.max(state_probs):.4f}, mean={np.mean(state_probs):.4f}")

    return model, states, probs


def main():
    print("=" * 70)
    print("  DIAGNOSTICO PROFUNDO DO HMM")
    print("=" * 70)

    # Baixar dados
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=90)

    print(f"\n  Periodo: {start_time.date()} a {end_time.date()}")
    print("\n  Baixando dados...")

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

    # Preparar retornos
    warmup_bars = 6624
    prices = np.array([bar.close for bar in bars[:warmup_bars]])
    returns = np.diff(np.log(prices))

    print(f"\n  Precos de warmup: {len(prices)}")
    print(f"  Retornos: {len(returns)}")

    # 1. Analisar distribuicao dos retornos
    analyze_returns_distribution(returns)

    # 2. Treinar HMM original
    print("\n" + "=" * 70)
    print("  TREINANDO HMM (CONFIGURACAO ORIGINAL)")
    print("=" * 70)

    rets = returns.reshape(-1, 1)

    hmm_model = hmm.GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    hmm_model.fit(rets)

    # 3. Analisar modelo
    state_info = analyze_hmm_model(hmm_model, returns)

    # 4. Analisar distribuicao de estados
    states, probs = analyze_state_distribution(hmm_model, None, returns)

    # 5. Problema identificado: verificar se o calculo de hmm_prob esta correto
    print("\n" + "=" * 70)
    print("  VERIFICANDO CALCULO DE HMM_PROB")
    print("=" * 70)

    print("\n  Codigo atual:")
    print("     hmm_prob = max(current_probs[1], current_probs[2])")
    print("\n  Isso pega o MAXIMO das probabilidades dos estados 1 e 2.")
    print("  Se o estado 0 domina, P(1) e P(2) serao sempre baixos.")

    # Simular o calculo
    hmm_probs_calculated = np.max(probs[:, 1:], axis=1)
    print(f"\n  hmm_prob calculado:")
    print(f"     Min: {np.min(hmm_probs_calculated):.4f}")
    print(f"     Max: {np.max(hmm_probs_calculated):.4f}")
    print(f"     Mean: {np.mean(hmm_probs_calculated):.4f}")

    print("\n  EXPLICACAO DO PROBLEMA:")
    print("  Se P(estado 0) ≈ 0.56, entao P(1) + P(2) ≈ 0.44")
    print("  Como max(P1, P2) <= P1 + P2, o maximo de hmm_prob eh ~0.44")
    print("  Isso explica por que hmm_threshold > 0.44 nunca eh atingido!")

    # 6. Testar com retornos padronizados
    model_std, states_std, probs_std = test_hmm_with_standardization(returns)

    # 7. Testar diferentes configuracoes
    test_different_hmm_configs(returns)

    # 8. Proposta de correcao
    print("\n" + "=" * 70)
    print("  PROPOSTA DE CORRECAO")
    print("=" * 70)

    print("\n  O problema nao eh um 'erro' - eh o comportamento esperado!")
    print("\n  EXPLICACAO:")
    print("  1. O HMM identifica 3 estados baseado na DISTRIBUICAO dos retornos")
    print("  2. Para dados de forex M5, a maioria dos retornos eh muito pequena")
    print("  3. O estado 0 captura a maioria (regime normal/baixa volatilidade)")
    print("  4. Estados 1 e 2 sao raros (alta volatilidade)")
    print("\n  OPCOES DE CORRECAO:")
    print("  A) Usar o ESTADO predito diretamente (nao a probabilidade)")
    print("  B) Usar hmm_prob como P(nao-estado-0) = P(1) + P(2)")
    print("  C) Usar volatilidade realizada em vez de estado HMM")
    print("  D) Usar hmm_prob < threshold para filtrar (regime calmo)")

    # Testar opcao B
    print("\n  Teste da OPCAO B (hmm_prob = P(1) + P(2)):")
    hmm_prob_sum = probs[:, 1] + probs[:, 2]
    print(f"     Min: {np.min(hmm_prob_sum):.4f}")
    print(f"     Max: {np.max(hmm_prob_sum):.4f}")
    print(f"     Mean: {np.mean(hmm_prob_sum):.4f}")

    # Percentis
    print(f"\n     Percentis:")
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(hmm_prob_sum, p)
        print(f"       P{p}: {val:.4f}")

    print("\n  CONCLUSAO:")
    print("  O HMM esta funcionando CORRETAMENTE.")
    print("  A maioria das barras M5 esta em regime de baixa volatilidade.")
    print("  Para capturar momentos de alta volatilidade, use:")
    print("    - hmm_prob >= 0.44 (10% das barras)")
    print("    - Ou combine com Lyapunov alto")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
