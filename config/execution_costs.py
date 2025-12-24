"""
================================================================================
CUSTOS DE EXECUÇÃO PADRONIZADOS - CONFIGURAÇÃO CENTRALIZADA
================================================================================

IMPORTANTE: Este arquivo define os custos de execução para TODO o sistema.
Qualquer componente (backtest, otimizador, produção) DEVE usar estes valores
para garantir consistência e resultados confiáveis.

NUNCA modifique estes valores em arquivos individuais!
Se precisar ajustar, modifique APENAS aqui.

Última Atualização: Dezembro 2024
Versão: 2.1 (Correção de auditoria)
"""

from typing import Dict

# =============================================================================
# CUSTOS DE EXECUÇÃO PADRÃO
# =============================================================================

# Spread médio realista em pips
# Baseado em condições normais de mercado para corretoras ECN/STP
SPREAD_PIPS: float = 1.5

# Slippage médio realista em pips
# Considera latência e liquidez típica
SLIPPAGE_PIPS: float = 0.8

# Comissão por lote (se aplicável)
# Para corretoras com spread puro, usar 0.0
COMMISSION_PER_LOT: float = 0.0


# =============================================================================
# VALOR DO PIP POR PAR (em unidades da moeda cotada)
# =============================================================================

PIP_VALUES: Dict[str, float] = {
    # Pares com USD como moeda cotada
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "AUDUSD": 0.0001,
    "NZDUSD": 0.0001,

    # Pares com JPY como moeda cotada
    "USDJPY": 0.01,
    "EURJPY": 0.01,
    "GBPJPY": 0.01,
    "AUDJPY": 0.01,

    # Crosses
    "EURGBP": 0.0001,
    "EURAUD": 0.0001,
    "GBPAUD": 0.0001,
    "AUDNZD": 0.0001,

    # Outros
    "USDCHF": 0.0001,
    "USDCAD": 0.0001,
    "EURCHF": 0.0001,
}


# =============================================================================
# USD POR PIP POR LOTE PADRÃO (100,000 unidades)
# =============================================================================

# NOTA: Para pares XXX/USD, o valor é fixo em $10 por pip por lote.
# Para outros pares, o valor depende da taxa de câmbio atual.
# Os valores abaixo são aproximações. Para precisão, calcular em tempo real.

USD_PER_PIP_PER_LOT: Dict[str, float] = {
    # Pares XXX/USD - valor fixo
    "EURUSD": 10.0,
    "GBPUSD": 10.0,
    "AUDUSD": 10.0,
    "NZDUSD": 10.0,

    # Pares USD/XXX - depende da taxa (valores aproximados)
    "USDJPY": 6.70,    # 10 / ~1.49 (taxa USDJPY ~149)
    "USDCHF": 11.50,   # 10 / ~0.87 (taxa USDCHF ~0.87)
    "USDCAD": 7.50,    # 10 / ~1.33 (taxa USDCAD ~1.33)

    # Crosses - dependem das taxas cruzadas (valores aproximados)
    "EURJPY": 6.70,    # Mesmo que USDJPY
    "GBPJPY": 6.70,    # Mesmo que USDJPY
    "AUDJPY": 6.70,    # Mesmo que USDJPY
    "EURGBP": 12.50,   # 10 / ~0.80 (taxa EURGBP ~0.80)
    "EURAUD": 6.50,    # 10 / ~1.54 (taxa EURAUD ~1.54)
    "GBPAUD": 6.50,    # Mesmo que EURAUD
    "AUDNZD": 9.20,    # 10 / ~1.09 (taxa AUDNZD ~1.09)
    "EURCHF": 11.50,   # Mesmo que USDCHF
}


# =============================================================================
# SPREADS ESPECÍFICOS POR PAR (opcional, para mais precisão)
# =============================================================================

# Alguns pares têm spreads maiores que outros
SPREAD_BY_PAIR: Dict[str, float] = {
    "EURUSD": 1.5,
    "GBPUSD": 1.8,
    "USDJPY": 1.5,
    "USDCHF": 2.0,
    "USDCAD": 2.0,
    "AUDUSD": 1.8,
    "NZDUSD": 2.0,
    "EURJPY": 2.0,
    "GBPJPY": 3.0,  # Mais volátil, spread maior
    "EURGBP": 1.5,
    "AUDNZD": 2.5,
}


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def get_pip_value(symbol: str) -> float:
    """
    Retorna o valor de 1 pip para o símbolo especificado.

    Args:
        symbol: Par de moedas (ex: 'EURUSD')

    Returns:
        Valor do pip (ex: 0.0001 para EURUSD)
    """
    return PIP_VALUES.get(symbol.upper(), 0.0001)


def get_usd_per_pip(symbol: str) -> float:
    """
    Retorna o valor em USD de 1 pip por lote padrão.

    Args:
        symbol: Par de moedas

    Returns:
        Valor em USD por pip por lote
    """
    return USD_PER_PIP_PER_LOT.get(symbol.upper(), 10.0)


def get_spread(symbol: str = None) -> float:
    """
    Retorna o spread em pips para o símbolo.
    Se não especificado ou não encontrado, retorna o spread padrão.

    Args:
        symbol: Par de moedas (opcional)

    Returns:
        Spread em pips
    """
    if symbol:
        return SPREAD_BY_PAIR.get(symbol.upper(), SPREAD_PIPS)
    return SPREAD_PIPS


def calculate_pnl_usd(pnl_pips: float, position_size: float, symbol: str = "EURUSD") -> float:
    """
    Calcula o PnL em USD baseado no par de moedas.

    Args:
        pnl_pips: PnL em pips
        position_size: Tamanho da posição em lotes
        symbol: Par de moedas

    Returns:
        PnL em USD
    """
    usd_per_pip = get_usd_per_pip(symbol)
    return pnl_pips * position_size * usd_per_pip


# =============================================================================
# VALIDAÇÃO
# =============================================================================

def validate_costs():
    """
    Valida que os custos estão dentro de ranges razoáveis.
    Chame esta função durante inicialização para verificar configuração.
    """
    assert 0.5 <= SPREAD_PIPS <= 5.0, f"Spread fora do range: {SPREAD_PIPS}"
    assert 0.0 <= SLIPPAGE_PIPS <= 3.0, f"Slippage fora do range: {SLIPPAGE_PIPS}"
    assert COMMISSION_PER_LOT >= 0, f"Comissão negativa: {COMMISSION_PER_LOT}"

    print(f"[CONFIG] Custos de execução validados:")
    print(f"         Spread: {SPREAD_PIPS} pips")
    print(f"         Slippage: {SLIPPAGE_PIPS} pips")
    print(f"         Comissão: ${COMMISSION_PER_LOT}/lote")


if __name__ == "__main__":
    validate_costs()
