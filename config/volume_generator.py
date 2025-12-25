"""
================================================================================
GERADOR CENTRALIZADO DE VOLUMES SINTETICOS - V3.1
================================================================================

IMPORTANTE: Este arquivo define a geração de volumes sintéticos para TODO o sistema.
Qualquer componente (indicador, estratégia, backtest, otimizador) DEVE usar esta
função para garantir CONSISTÊNCIA entre backtest e produção.

PROBLEMA RESOLVIDO (Auditoria 24/12/2025):
- dsg_detector_singularidade.py usava: |prices[i-1] - prices[i-2]| * 1000 + 50
- dsg_strategy.py usava: |prices[-1] - prices[-2]| * 50000 + 50
- Diferença de 50x causava resultados inconsistentes!

SOLUÇÃO: Função única usada por TODOS os componentes.

Última Atualização: Dezembro 2024
Versão: 3.1 (Correção de auditoria - unificação de volumes)
"""

import numpy as np
from typing import Tuple

# =============================================================================
# CONSTANTES DE CONFIGURAÇÃO
# =============================================================================

# Multiplicador para conversão de variação de preço em volume
# Escolhido como valor intermediário entre 1000 e 50000
VOLUME_MULTIPLIER: float = 10000.0

# Volume base mínimo (evita volumes zero)
VOLUME_BASE: float = 50.0


# =============================================================================
# FUNÇÃO PRINCIPAL - USAR EM TODOS OS COMPONENTES
# =============================================================================

def generate_synthetic_volumes(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera volumes sintéticos DETERMINÍSTICOS baseados na variação de preço.

    IMPORTANTE: Usar esta função em TODOS os componentes:
    - dsg_detector_singularidade.py (método analyze)
    - dsg_strategy.py (método add_price)
    - optimizer.py
    - backtest.py

    REGRA ANTI LOOK-AHEAD:
    - Volume[i] é calculado com base em prices[i-1] e prices[i-2]
    - NUNCA usa prices[i] (a barra atual) no cálculo
    - Isso garante que em tempo real, o volume está disponível ANTES do close

    Args:
        prices: Array de preços de fechamento (np.ndarray)

    Returns:
        Tuple (bid_volumes, ask_volumes) - ambos np.ndarray do mesmo tamanho

    Raises:
        ValueError: Se prices for None ou vazio
    """
    if prices is None or len(prices) == 0:
        raise ValueError("Array de preços vazio ou None")

    n = len(prices)
    bid_volumes = np.zeros(n, dtype=np.float64)
    ask_volumes = np.zeros(n, dtype=np.float64)

    # Valores base para primeiras barras (sem histórico suficiente)
    bid_volumes[0] = VOLUME_BASE
    ask_volumes[0] = VOLUME_BASE

    if n > 1:
        bid_volumes[1] = VOLUME_BASE
        ask_volumes[1] = VOLUME_BASE

    # Para i >= 2: volume baseado em variação de barras JÁ FECHADAS
    # Volume[i] = |prices[i-1] - prices[i-2]| * MULTIPLIER + BASE
    # Isso é 100% causal: só usa dados passados
    for i in range(2, n):
        change = np.abs(prices[i-1] - prices[i-2])
        vol = change * VOLUME_MULTIPLIER + VOLUME_BASE
        bid_volumes[i] = vol
        ask_volumes[i] = vol

    return bid_volumes, ask_volumes


def generate_single_volume(price_current: float, price_prev: float) -> float:
    """
    Gera volume sintético para uma única barra (uso em tempo real).

    REGRA ANTI LOOK-AHEAD:
    - price_current é o preço da barra ANTERIOR (já fechada)
    - price_prev é o preço duas barras atrás
    - O resultado é o volume para a barra ATUAL

    Uso típico em add_price():
        if len(prices) >= 2:
            vol = generate_single_volume(prices[-2], prices[-3])
        else:
            vol = VOLUME_BASE

    Args:
        price_current: Preço da barra anterior (i-1)
        price_prev: Preço de duas barras atrás (i-2)

    Returns:
        Volume sintético (float)
    """
    change = np.abs(price_current - price_prev)
    return change * VOLUME_MULTIPLIER + VOLUME_BASE


def get_volume_base() -> float:
    """Retorna o volume base para barras sem histórico suficiente."""
    return VOLUME_BASE


def get_volume_multiplier() -> float:
    """Retorna o multiplicador de volume."""
    return VOLUME_MULTIPLIER


# =============================================================================
# VALIDAÇÃO
# =============================================================================

def validate_volumes(volumes: np.ndarray) -> bool:
    """
    Valida que os volumes gerados estão dentro de ranges razoáveis.

    Args:
        volumes: Array de volumes a validar

    Returns:
        True se válido, False caso contrário
    """
    if volumes is None or len(volumes) == 0:
        return False

    # Verificar NaN e Inf
    if np.any(np.isnan(volumes)) or np.any(np.isinf(volumes)):
        return False

    # Verificar valores negativos
    if np.any(volumes < 0):
        return False

    # Verificar volume mínimo
    if np.any(volumes < VOLUME_BASE * 0.5):  # Tolerância de 50%
        return False

    return True


if __name__ == "__main__":
    # Teste básico
    print("=" * 60)
    print("TESTE DO GERADOR DE VOLUMES SINTÉTICOS")
    print("=" * 60)

    # Dados de teste
    test_prices = np.array([1.1000, 1.1005, 1.1010, 1.1008, 1.1015])

    bid_vols, ask_vols = generate_synthetic_volumes(test_prices)

    print(f"\nPreços: {test_prices}")
    print(f"Bid Volumes: {bid_vols}")
    print(f"Ask Volumes: {ask_vols}")

    print(f"\nConstantes:")
    print(f"  VOLUME_MULTIPLIER: {VOLUME_MULTIPLIER}")
    print(f"  VOLUME_BASE: {VOLUME_BASE}")

    # Verificar cálculo manual
    print(f"\nVerificação manual:")
    for i in range(2, len(test_prices)):
        change = np.abs(test_prices[i-1] - test_prices[i-2])
        expected = change * VOLUME_MULTIPLIER + VOLUME_BASE
        print(f"  i={i}: |{test_prices[i-1]:.4f} - {test_prices[i-2]:.4f}| * {VOLUME_MULTIPLIER} + {VOLUME_BASE} = {expected:.2f}")
        assert np.isclose(bid_vols[i], expected), f"Erro no índice {i}"

    print(f"\nValidação: {validate_volumes(bid_vols)}")
    print("\n✅ Teste concluído com sucesso!")
