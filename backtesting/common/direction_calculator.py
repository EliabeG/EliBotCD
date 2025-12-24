"""
================================================================================
MÓDULO COMPARTILHADO: Cálculo de Direção
================================================================================

Este módulo centraliza o cálculo de direção para TODOS os componentes do sistema.
Isso garante consistência entre:
- DTTStrategy
- DTT Optimizer
- DTT Debug
- Qualquer outro componente que precise calcular direção

REGRAS ANTI LOOK-AHEAD:
1. NUNCA usar o preço atual (índice -1 ou i) para decisão
2. Usar apenas barras COMPLETAMENTE FECHADAS
3. Manter consistência de índices em todos os arquivos

VERSÃO: 2.1
DATA: 24/12/2025
================================================================================
"""

from typing import List, Union
import numpy as np


# Configuração padrão
DEFAULT_DIRECTION_LOOKBACK = 12


def calculate_direction_from_closes(
    closes: Union[List[float], np.ndarray],
    lookback: int = DEFAULT_DIRECTION_LOOKBACK
) -> int:
    """
    Calcula direção baseada APENAS em barras FECHADAS.

    ESTA É A FUNÇÃO OFICIAL - use em TODOS os componentes.

    Lógica:
    - closes[-1] = barra atual (momento do sinal) - NÃO USAR
    - closes[-2] = última barra completamente fechada - USAR
    - closes[-(lookback+1)] = barra de comparação - USAR

    A direção é baseada na tendência das últimas 'lookback' barras ANTES
    do momento atual.

    Args:
        closes: Lista ou array de preços de fechamento
        lookback: Número de barras para calcular tendência (default: 12)

    Returns:
        1 para LONG (tendência de alta)
        -1 para SHORT (tendência de baixa)
        0 para NEUTRAL (dados insuficientes)

    Exemplo:
        closes = [1.1000, 1.1010, 1.1020, ..., 1.1150]  # 20 barras
        direction = calculate_direction_from_closes(closes, lookback=12)
        # Compara closes[-2] (1.1140) com closes[-13] (1.1010)
        # Retorna 1 (LONG) pois 1.1140 > 1.1010
    """
    # Validação de entrada
    if closes is None or len(closes) < lookback + 2:
        return 0

    # Converter para lista se necessário
    if isinstance(closes, np.ndarray):
        closes = closes.tolist()

    # Índices:
    # -1 = barra atual (NÃO USAR - pode ter look-ahead)
    # -2 = última barra completamente fechada
    # -(lookback + 1) = barra de comparação
    recent_close = closes[-2]
    past_close = closes[-(lookback + 1)]

    # Calcular tendência
    trend = recent_close - past_close

    return 1 if trend > 0 else -1


def calculate_direction_from_bars(
    bars: list,  # List[Bar]
    current_idx: int,
    lookback: int = DEFAULT_DIRECTION_LOOKBACK
) -> int:
    """
    Calcula direção a partir de uma lista de barras e índice atual.

    Esta função é equivalente a calculate_direction_from_closes,
    mas recebe barras e índice em vez de lista de closes.

    Lógica:
    - bars[current_idx] = barra atual (momento do sinal) - NÃO USAR
    - bars[current_idx - 1] = última barra fechada - USAR
    - bars[current_idx - lookback] = barra de comparação - USAR

    Args:
        bars: Lista de objetos Bar com atributo .close
        current_idx: Índice da barra atual
        lookback: Número de barras para calcular tendência (default: 12)

    Returns:
        1 para LONG, -1 para SHORT, 0 para NEUTRAL
    """
    # Validação
    if bars is None or current_idx < lookback + 1:
        return 0

    if current_idx >= len(bars):
        return 0

    # Índices:
    # current_idx = barra atual (NÃO USAR)
    # current_idx - 1 = última barra fechada
    # current_idx - lookback = barra de comparação
    recent_close = bars[current_idx - 1].close
    past_close = bars[current_idx - lookback].close

    trend = recent_close - past_close

    return 1 if trend > 0 else -1


def validate_direction_consistency(
    closes_result: int,
    bars_result: int,
    context: str = ""
) -> bool:
    """
    Valida se os dois métodos de cálculo retornam o mesmo resultado.

    Use em testes para garantir consistência.

    Args:
        closes_result: Resultado de calculate_direction_from_closes
        bars_result: Resultado de calculate_direction_from_bars
        context: Descrição do contexto para debug

    Returns:
        True se consistente, False caso contrário

    Raises:
        ValueError se inconsistente (em modo de validação)
    """
    if closes_result != bars_result:
        msg = f"Inconsistência de direção detectada! closes={closes_result}, bars={bars_result}"
        if context:
            msg += f" | Contexto: {context}"
        raise ValueError(msg)

    return True


# Constantes exportadas para uso em outros módulos
DIRECTION_LONG = 1
DIRECTION_SHORT = -1
DIRECTION_NEUTRAL = 0
