"""
================================================================================
FILTROS DE OTIMIZAÇÃO CENTRALIZADOS - DSG
================================================================================

IMPORTANTE: Este arquivo define os filtros de validação para TODOS os otimizadores.
Qualquer otimizador (train/test split, walk-forward) DEVE usar estes valores
para garantir consistência e resultados comparáveis.

NUNCA modifique estes valores em arquivos individuais!
Se precisar ajustar, modifique APENAS aqui.

Última Atualização: Dezembro 2025
Versão: 1.1 (V2.3 - MIN_EXPECTANCY aumentado para 3.0)
"""

# =============================================================================
# FILTROS PARA PERÍODO DE TREINO
# =============================================================================

# Número mínimo de trades no período de treino
# RELAXADO V2: 30 trades ainda é estatisticamente significativo
MIN_TRADES_TRAIN: int = 30

# Win rate mínimo no treino (para evitar estratégias muito ruins)
# RELAXADO V2: DSG corrigido pode ter WR muito baixo (~18-25%)
MIN_WIN_RATE: float = 0.18

# Win rate máximo no treino (para evitar overfitting)
MAX_WIN_RATE: float = 0.65

# Profit factor mínimo no treino
# RELAXADO V2: Break-even é aceitável se consistente
MIN_PROFIT_FACTOR: float = 1.00

# Profit factor máximo no treino (evita overfitting)
MAX_PROFIT_FACTOR: float = 4.0

# Drawdown máximo permitido no treino
# RELAXADO V2: Estratégias de alta volatilidade têm DD maior
MAX_DRAWDOWN: float = 0.40


# =============================================================================
# FILTROS PARA PERÍODO DE TESTE
# =============================================================================

# Número mínimo de trades no teste
# RELAXADO V2: 15 trades mínimo no teste
MIN_TRADES_TEST: int = 15

# Win rate mínimo no teste (ligeiramente mais permissivo)
# RELAXADO V2: Aceita WR muito baixo se PF compensar
MIN_WIN_RATE_TEST: float = 0.15

# Win rate máximo no teste (ligeiramente mais permissivo)
MAX_WIN_RATE_TEST: float = 0.70

# Profit factor mínimo no teste
# RELAXADO V2: Aceita pequena perda no teste se robusto
MIN_PROFIT_FACTOR_TEST: float = 0.90

# Profit factor máximo no teste
MAX_PROFIT_FACTOR_TEST: float = 5.0

# Drawdown máximo permitido no teste (ligeiramente mais permissivo)
# RELAXADO V2: DD maior permitido em testes
MAX_DRAWDOWN_TEST: float = 0.45


# =============================================================================
# FILTROS DE ROBUSTEZ
# =============================================================================

# Robustez mínima (teste deve manter X% do treino)
# RELAXADO V2: Aceita maior degradação
MIN_ROBUSTNESS: float = 0.35

# Razão mínima PF teste/treino
# RELAXADO V2: Teste pode ter até 65% de degradação do PF
MIN_PF_RATIO: float = 0.35

# Razão mínima WR teste/treino
# RELAXADO V2: WR pode degradar mais
MIN_WR_RATIO: float = 0.35


# =============================================================================
# FILTROS ESPECÍFICOS PARA WALK-FORWARD
# =============================================================================

# Número mínimo de janelas que devem passar (de 4)
MIN_WINDOWS_PASSED: int = 3

# Expectancy mínima por trade (em pips)
# V2.3: Aumentado de 1.5 para 3.0 para cobrir custos de 4.6 pips + margem
MIN_EXPECTANCY_PIPS: float = 3.0


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def validate_train_result(trades: int, win_rate: float, profit_factor: float,
                          max_drawdown: float) -> bool:
    """
    Valida resultado de treino usando filtros padronizados.

    Args:
        trades: Número de trades
        win_rate: Taxa de acerto (0-1)
        profit_factor: Fator de lucro
        max_drawdown: Drawdown máximo (0-1)

    Returns:
        True se passar em todos os filtros
    """
    if trades < MIN_TRADES_TRAIN:
        return False
    if win_rate < MIN_WIN_RATE or win_rate > MAX_WIN_RATE:
        return False
    if profit_factor < MIN_PROFIT_FACTOR or profit_factor > MAX_PROFIT_FACTOR:
        return False
    if max_drawdown > MAX_DRAWDOWN:
        return False
    return True


def validate_test_result(trades: int, win_rate: float, profit_factor: float,
                         max_drawdown: float) -> bool:
    """
    Valida resultado de teste usando filtros padronizados.

    Args:
        trades: Número de trades
        win_rate: Taxa de acerto (0-1)
        profit_factor: Fator de lucro
        max_drawdown: Drawdown máximo (0-1)

    Returns:
        True se passar em todos os filtros
    """
    if trades < MIN_TRADES_TEST:
        return False
    if win_rate < MIN_WIN_RATE_TEST or win_rate > MAX_WIN_RATE_TEST:
        return False
    if profit_factor < MIN_PROFIT_FACTOR_TEST or profit_factor > MAX_PROFIT_FACTOR_TEST:
        return False
    if max_drawdown > MAX_DRAWDOWN_TEST:
        return False
    return True


def validate_robustness(pf_train: float, pf_test: float,
                        wr_train: float, wr_test: float) -> bool:
    """
    Valida robustez entre treino e teste.

    Args:
        pf_train: Profit factor do treino
        pf_test: Profit factor do teste
        wr_train: Win rate do treino
        wr_test: Win rate do teste

    Returns:
        True se robustez for aceitável
    """
    if pf_train <= 0 or wr_train <= 0:
        return False

    pf_ratio = pf_test / pf_train
    wr_ratio = wr_test / wr_train

    return (pf_ratio >= MIN_PF_RATIO and
            wr_ratio >= MIN_WR_RATIO and
            pf_test >= MIN_PROFIT_FACTOR_TEST)


def print_filters():
    """Imprime todos os filtros configurados."""
    print(f"[FILTROS DSG] Configuração centralizada V1.0")
    print(f"  TREINO:")
    print(f"    Min Trades: {MIN_TRADES_TRAIN}")
    print(f"    Win Rate: {MIN_WIN_RATE:.0%} - {MAX_WIN_RATE:.0%}")
    print(f"    Profit Factor: {MIN_PROFIT_FACTOR:.2f} - {MAX_PROFIT_FACTOR:.2f}")
    print(f"    Max Drawdown: {MAX_DRAWDOWN:.0%}")
    print(f"  TESTE:")
    print(f"    Min Trades: {MIN_TRADES_TEST}")
    print(f"    Win Rate: {MIN_WIN_RATE_TEST:.0%} - {MAX_WIN_RATE_TEST:.0%}")
    print(f"    Profit Factor: {MIN_PROFIT_FACTOR_TEST:.2f} - {MAX_PROFIT_FACTOR_TEST:.2f}")
    print(f"    Max Drawdown: {MAX_DRAWDOWN_TEST:.0%}")
    print(f"  ROBUSTEZ:")
    print(f"    Min PF Ratio: {MIN_PF_RATIO:.0%}")
    print(f"    Min WR Ratio: {MIN_WR_RATIO:.0%}")


if __name__ == "__main__":
    print_filters()
