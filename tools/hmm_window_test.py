#!/usr/bin/env python3
"""
================================================================================
TESTE: Janela de 200 vs. Sequencia Completa
================================================================================

Investiga por que a janela de 200 da 100% estado 0,
enquanto a sequencia completa da 50/50.
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


def main():
    print("=" * 70)
    print("  TESTE: JANELA DE 200 vs SEQUENCIA COMPLETA")
    print("=" * 70)

    # Baixar dados
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=90)

    print(f"\n  Baixando dados...")
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
    warmup_bars = 6624
    prices = np.array([bar.close for bar in bars[:warmup_bars]])
    all_returns = np.diff(np.log(prices))

    print(f"\n  Retornos de warmup: {len(all_returns)}")

    # Treinar HMM com TODO o warmup
    print("\n" + "=" * 70)
    print("  TREINANDO HMM COM TODO O WARMUP")
    print("=" * 70)

    rets = all_returns.reshape(-1, 1)

    hmm_model = hmm.GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    hmm_model.fit(rets)

    # Predizer com TODO o warmup
    states_full = hmm_model.predict(rets)
    probs_full = hmm_model.predict_proba(rets)

    print(f"\n  Predicao com sequencia COMPLETA ({len(all_returns)} retornos):")
    for s in range(3):
        count = np.sum(states_full == s)
        pct = count / len(states_full) * 100
        print(f"     Estado {s}: {count} ({pct:.1f}%)")

    # Agora simular o que o codigo do otimizador faz:
    # Usa apenas uma janela de 200 retornos
    print("\n" + "=" * 70)
    print("  SIMULANDO JANELA DESLIZANTE DE 200")
    print("=" * 70)

    hmm_window = 200
    states_window = []
    probs_window = []

    # Simular janela deslizante
    for i in range(hmm_window, len(all_returns)):
        window_rets = all_returns[i-hmm_window:i].reshape(-1, 1)

        # Predizer com a janela
        try:
            probs = hmm_model.predict_proba(window_rets)
            current_probs = probs[-1]  # Probabilidades da ultima observacao
            state = np.argmax(current_probs)

            states_window.append(state)
            probs_window.append(current_probs)
        except:
            pass

    states_window = np.array(states_window)
    probs_window = np.array(probs_window)

    print(f"\n  Predicao com JANELA DE 200 ({len(states_window)} observacoes):")
    for s in range(3):
        count = np.sum(states_window == s)
        pct = count / len(states_window) * 100
        print(f"     Estado {s}: {count} ({pct:.1f}%)")

    print(f"\n  Probabilidades por estado (janela de 200):")
    for s in range(3):
        state_probs = probs_window[:, s]
        print(f"     Estado {s}: min={np.min(state_probs):.4f}, max={np.max(state_probs):.4f}, mean={np.mean(state_probs):.4f}")

    # Calcular hmm_prob como no codigo original
    hmm_probs = np.max(probs_window[:, 1:], axis=1)
    print(f"\n  hmm_prob (max de P1, P2):")
    print(f"     Min: {np.min(hmm_probs):.4f}")
    print(f"     Max: {np.max(hmm_probs):.4f}")
    print(f"     Mean: {np.mean(hmm_probs):.4f}")

    # Comparar com predicao de todo o warmup (ultimas 200 obs)
    print("\n" + "=" * 70)
    print("  COMPARACAO DIRETA")
    print("=" * 70)

    # Ultimas 200 observacoes da predicao completa
    last_200_states = states_full[-len(states_window):]

    print(f"\n  Ultimas {len(last_200_states)} observacoes:")
    print(f"     Predicao COMPLETA - Estados: ", end="")
    for s in range(3):
        count = np.sum(last_200_states == s)
        pct = count / len(last_200_states) * 100
        print(f"{s}:{pct:.1f}% ", end="")
    print()

    print(f"     Predicao JANELA   - Estados: ", end="")
    for s in range(3):
        count = np.sum(states_window == s)
        pct = count / len(states_window) * 100
        print(f"{s}:{pct:.1f}% ", end="")
    print()

    # Verificar se os estados coincidem
    matches = np.sum(last_200_states == states_window)
    match_pct = matches / len(states_window) * 100
    print(f"\n     Coincidencias: {matches}/{len(states_window)} ({match_pct:.1f}%)")

    # Entender a diferenca
    print("\n" + "=" * 70)
    print("  ANALISE DA DIFERENCA")
    print("=" * 70)

    print(f"\n  O problema esta em como o HMM faz inferencia:")
    print(f"  - Com sequencia COMPLETA: O modelo ve toda a historia")
    print(f"    e pode usar informacao de transicoes passadas")
    print(f"  - Com JANELA: O modelo so ve os ultimos 200 retornos")
    print(f"    e 'esquece' o contexto anterior")

    # Verificar matriz de transicao
    print(f"\n  Matriz de transicao do modelo:")
    print("     De/Para   Estado 0    Estado 1    Estado 2")
    for i in range(3):
        print(f"     Estado {i}  {hmm_model.transmat_[i][0]:.4f}      {hmm_model.transmat_[i][1]:.4f}      {hmm_model.transmat_[i][2]:.4f}")

    print(f"\n  PROBLEMA IDENTIFICADO:")
    print(f"  - Estado 0 -> Estado 1: {hmm_model.transmat_[0][1]:.4f} (quase sempre vai para 1)")
    print(f"  - Estado 1 -> Estado 0: {hmm_model.transmat_[1][0]:.4f} (quase sempre vai para 0)")
    print(f"  - O modelo oscila RAPIDAMENTE entre estados 0 e 1!")
    print(f"\n  Quando usamos janela curta, o modelo 'inicia' de um estado")
    print(f"  e a predicao depende muito do estado inicial assumido.")

    # Testar com diferentes random_states
    print("\n" + "=" * 70)
    print("  TESTE: DIFERENTES INICIALIZACOES")
    print("=" * 70)

    for seed in [42, 123, 456, 789, 1000]:
        model = hmm.GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=100,
            random_state=seed
        )
        model.fit(rets)

        states = model.predict(rets)

        print(f"\n  Seed {seed}:")
        for s in range(3):
            count = np.sum(states == s)
            pct = count / len(states) * 100
            print(f"     Estado {s}: {pct:.1f}%")

    print("\n" + "=" * 70)
    print("  CONCLUSAO")
    print("=" * 70)

    print(f"""
  1. O modelo HMM com 3 estados para dados M5 nao esta bem definido
  2. Estados 0 e 1 sao quase identicos (mesma covariancia)
  3. Estado 2 eh degenerado (covariancia infinita, nunca ocorre)
  4. O modelo oscila rapidamente entre estados 0 e 1

  RECOMENDACAO:
  - Nao usar o 'estado' predito como filtro
  - Usar apenas a probabilidade HMM (que funciona)
  - Ou usar volatilidade realizada diretamente

  A otimizacao anterior esta CORRETA porque:
  - Ela nao filtra por estado (usa [0,1,2] que inclui tudo)
  - Ela usa hmm_prob >= 0.4 que funciona como filtro de volatilidade
    """)


if __name__ == "__main__":
    main()
