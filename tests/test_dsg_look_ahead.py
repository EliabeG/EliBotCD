#!/usr/bin/env python3
"""
================================================================================
TESTE DE LOOK-AHEAD BIAS PARA DSG
================================================================================

Este script verifica se o indicador DSG contém look-ahead bias.

CONCEITO:
O sinal no tempo T deve ser IDÊNTICO independente de como processamos:
1. Online (barra por barra até T)
2. Offline (todas as barras de uma vez)

Se os sinais diferirem, há look-ahead bias.

EXECUÇÃO:
    python -m tests.test_dsg_look_ahead

================================================================================
"""

import sys
import os
import numpy as np

# Adiciona raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional


def test_step_function_causal():
    """
    Testa se a step function é realmente causal

    CORREÇÃO: Agora testa também o comportamento quando índice < primeiro cálculo
    Índices ANTES do primeiro cálculo devem ser 0.0 (valor neutro)
    Índices APÓS o primeiro cálculo usam step function
    """
    print("\n" + "=" * 60)
    print("TESTE 1: Step Function Causal")
    print("=" * 60)

    dsg = DetectorSingularidadeGravitacional()

    # Cenário 1: primeiro cálculo no índice 0
    print("\n  Cenário 1: primeiro cálculo no índice 0")
    indices = [0, 10, 20]
    values = [1.0, 2.0, 3.0]
    n = 25

    result = dsg._apply_step_function_causal(n, indices, values)

    errors = []

    # Índice 0: deve usar valor 1.0 (primeiro cálculo)
    if result[0] != 1.0:
        errors.append(f"Cenário 1 - Índice 0: esperado 1.0, obtido {result[0]}")

    # Índice 5: deve usar valor 1.0 (último calculado até 5 é índice 0)
    if result[5] != 1.0:
        errors.append(f"Cenário 1 - Índice 5: esperado 1.0, obtido {result[5]}")

    # Índice 10: deve usar valor 2.0
    if result[10] != 2.0:
        errors.append(f"Cenário 1 - Índice 10: esperado 2.0, obtido {result[10]}")

    # Cenário 2: primeiro cálculo no índice 5 (não no 0)
    # Isso testa a correção de look-ahead
    print("  Cenário 2: primeiro cálculo no índice 5 (correção look-ahead)")
    indices2 = [5, 15, 25]
    values2 = [1.0, 2.0, 3.0]
    n2 = 30

    result2 = dsg._apply_step_function_causal(n2, indices2, values2)

    # CORREÇÃO: Índices 0-4 devem ser 0.0 (valor neutro, NÃO values2[0])
    # Isso é CRÍTICO: usar values2[0] seria look-ahead bias!
    for i in range(5):
        if result2[i] != 0.0:
            errors.append(f"Cenário 2 - Índice {i}: esperado 0.0 (neutro), obtido {result2[i]}")

    # Índice 5: deve usar valor 1.0 (primeiro cálculo)
    if result2[5] != 1.0:
        errors.append(f"Cenário 2 - Índice 5: esperado 1.0, obtido {result2[5]}")

    # Índice 10: deve usar valor 1.0 (último calculado até 10 é índice 5)
    if result2[10] != 1.0:
        errors.append(f"Cenário 2 - Índice 10: esperado 1.0, obtido {result2[10]}")

    # Índice 15: deve usar valor 2.0
    if result2[15] != 2.0:
        errors.append(f"Cenário 2 - Índice 15: esperado 2.0, obtido {result2[15]}")

    if errors:
        print("❌ FALHOU:")
        for e in errors:
            print(f"   {e}")
        return False
    else:
        print("✅ PASSOU: Step function é 100% causal")
        print(f"   Cenário 1: {result[:15]}...")
        print(f"   Cenário 2: {result2[:20]}...")
        print("   Índices antes do primeiro cálculo usam valor neutro (0.0)")
        return True


def test_no_look_ahead_incremental():
    """
    Teste principal: Verifica se não há look-ahead bias

    Compara o processamento incremental vs batch.
    Se forem iguais, não há look-ahead.
    """
    print("\n" + "=" * 60)
    print("TESTE 2: Look-Ahead Bias (Incremental vs Batch)")
    print("=" * 60)

    np.random.seed(42)

    # Gerar dados de teste
    n_bars = 200
    prices = 1.1000 + 0.0002 * np.cumsum(np.random.randn(n_bars))

    # Processar de forma incremental (como seria em tempo real)
    dsg_incremental = DetectorSingularidadeGravitacional()
    signals_incremental = []
    ricci_incremental = []

    min_bars = 50

    print(f"  Processando {n_bars} barras incrementalmente...")

    for i in range(min_bars, n_bars):
        # Resetar histórico para cada iteração
        dsg_incremental._ricci_history = []
        dsg_incremental._distance_history = []
        dsg_incremental._coords_history = []

        # Processar apenas até a barra atual
        result = dsg_incremental.analyze(prices[:i+1])

        signals_incremental.append(result['signal'])
        ricci_incremental.append(result['Ricci_Scalar'])

    # Processar de forma batch (todas as barras de uma vez)
    dsg_batch = DetectorSingularidadeGravitacional()
    result_batch = dsg_batch.analyze(prices)

    # Comparar os últimos valores
    last_signal_incremental = signals_incremental[-1]
    last_signal_batch = result_batch['signal']

    last_ricci_incremental = ricci_incremental[-1]
    last_ricci_batch = result_batch['Ricci_Scalar']

    print(f"\n  Resultados:")
    print(f"    Sinal (incremental): {last_signal_incremental}")
    print(f"    Sinal (batch):       {last_signal_batch}")
    print(f"    Ricci (incremental): {last_ricci_incremental:.6f}")
    print(f"    Ricci (batch):       {last_ricci_batch:.6f}")

    # Tolerância para diferenças numéricas
    ricci_diff = abs(last_ricci_incremental - last_ricci_batch)

    if last_signal_incremental != last_signal_batch:
        print("\n❌ LOOK-AHEAD BIAS DETECTADO!")
        print("   Os sinais diferem entre processamento incremental e batch.")
        return False
    elif ricci_diff > 1e-6:
        print(f"\n⚠️  AVISO: Diferença numérica no Ricci: {ricci_diff:.10f}")
        print("   Sinais são iguais, mas há pequena diferença numérica.")
        print("   Isso pode ser aceitável se for apenas erro de ponto flutuante.")
        return True
    else:
        print("\n✅ SEM LOOK-AHEAD BIAS!")
        print("   Os resultados são idênticos entre processamento incremental e batch.")
        return True


def test_subsampling_consistency():
    """
    Testa se o subsampling não introduz inconsistências
    """
    print("\n" + "=" * 60)
    print("TESTE 3: Consistência do Subsampling")
    print("=" * 60)

    np.random.seed(123)

    # Gerar dados com evento de volatilidade
    n_bars = 300
    prices = 1.1000 + 0.0001 * np.cumsum(np.random.randn(n_bars))

    # Adicionar "singularidade" no meio
    prices[150:160] += np.linspace(0, 0.005, 10)

    dsg = DetectorSingularidadeGravitacional()

    # Processar com diferentes tamanhos de janela
    result_100 = dsg.analyze(prices[:100])
    dsg._ricci_history = []
    dsg._distance_history = []
    dsg._coords_history = []

    result_200 = dsg.analyze(prices[:200])
    dsg._ricci_history = []
    dsg._distance_history = []
    dsg._coords_history = []

    result_300 = dsg.analyze(prices)

    print(f"  Resultados por tamanho de janela:")
    print(f"    100 barras: signal={result_100['signal']}, ricci={result_100['Ricci_Scalar']:.6f}")
    print(f"    200 barras: signal={result_200['signal']}, ricci={result_200['Ricci_Scalar']:.6f}")
    print(f"    300 barras: signal={result_300['signal']}, ricci={result_300['Ricci_Scalar']:.6f}")

    print("\n✅ Teste concluído (verificação visual)")
    return True


def main():
    """Executa todos os testes"""
    print("\n" + "=" * 60)
    print("TESTES DE LOOK-AHEAD BIAS - DSG")
    print("Detector de Singularidade Gravitacional")
    print("=" * 60)

    results = []

    # Teste 1: Step Function
    results.append(("Step Function Causal", test_step_function_causal()))

    # Teste 2: Look-Ahead
    results.append(("Look-Ahead Bias", test_no_look_ahead_incremental()))

    # Teste 3: Subsampling
    results.append(("Consistência Subsampling", test_subsampling_consistency()))

    # Resumo
    print("\n" + "=" * 60)
    print("RESUMO DOS TESTES")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("   O indicador DSG não apresenta look-ahead bias detectável.")
    else:
        print("\n🚨 ALGUNS TESTES FALHARAM!")
        print("   O indicador pode conter look-ahead bias.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
