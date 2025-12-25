"""
================================================================================
CONFIGURACAO CENTRALIZADA - INDICADOR PRM (Protocolo Riemann-Mandelbrot)
================================================================================

IMPORTANTE: Este arquivo define TODOS os parametros do PRM para TODO o sistema.
Qualquer componente (indicador, estrategia, backtest, otimizador) DEVE usar
estes valores para garantir CONSISTENCIA entre desenvolvimento e producao.

NUNCA modifique estes valores em arquivos individuais!
Se precisar ajustar, modifique APENAS aqui.

Criado: 25 de Dezembro de 2025
Versao: 1.0 (Correcao de auditoria - unificacao de parametros)
"""

# =============================================================================
# CUSTOS DE EXECUCAO - IMPORTAR DE execution_costs.py
# =============================================================================
# NOTA: Os custos de execucao devem ser importados de config/execution_costs.py
# Aqui apenas re-exportamos para conveniencia
from config.execution_costs import (
    SPREAD_PIPS,
    SLIPPAGE_PIPS,
    COMMISSION_PER_LOT,
    get_spread,
    get_pip_value,
)


# =============================================================================
# PARAMETROS DO INDICADOR PRM
# =============================================================================

# Numero minimo de precos necessarios para analise
# UNIFICADO: Todos os componentes devem usar este valor
MIN_PRICES: int = 100

# Janela de treino do HMM (excluindo barra atual)
HMM_TRAINING_WINDOW: int = 200

# Minimo de amostras para treinar o HMM
HMM_MIN_TRAINING_SAMPLES: int = 50

# Numero de estados do HMM
HMM_N_STATES: int = 3

# Threshold para ativacao do HMM (probabilidade posterior)
HMM_THRESHOLD_DEFAULT: float = 0.85

# Threshold K para expoente de Lyapunov
LYAPUNOV_THRESHOLD_DEFAULT: float = 0.5

# Threshold para aceleracao da curvatura
CURVATURE_THRESHOLD_DEFAULT: float = 0.1

# Janela de lookback para calculos deslizantes
LOOKBACK_WINDOW: int = 100

# Numero de barras para calcular tendencia (direcao do trade)
TREND_LOOKBACK: int = 10


# =============================================================================
# PARAMETROS GARCH
# =============================================================================

GARCH_OMEGA: float = 0.00001
GARCH_ALPHA: float = 0.1
GARCH_BETA: float = 0.85

# Janela de inicializacao do GARCH (evita look-ahead)
GARCH_INIT_WINDOW: int = 20


# =============================================================================
# PARAMETROS DE TRADE
# =============================================================================

# Stop loss padrao em pips
DEFAULT_STOP_LOSS_PIPS: float = 20.0

# Take profit padrao em pips
DEFAULT_TAKE_PROFIT_PIPS: float = 40.0

# Cooldown entre sinais (em barras)
SIGNAL_COOLDOWN: int = 10


# =============================================================================
# FUNCOES AUXILIARES
# =============================================================================

def get_prm_config() -> dict:
    """
    Retorna todos os parametros do PRM como um dicionario.
    Util para logging e debugging.
    """
    return {
        "min_prices": MIN_PRICES,
        "hmm_training_window": HMM_TRAINING_WINDOW,
        "hmm_min_training_samples": HMM_MIN_TRAINING_SAMPLES,
        "hmm_n_states": HMM_N_STATES,
        "hmm_threshold": HMM_THRESHOLD_DEFAULT,
        "lyapunov_threshold": LYAPUNOV_THRESHOLD_DEFAULT,
        "curvature_threshold": CURVATURE_THRESHOLD_DEFAULT,
        "lookback_window": LOOKBACK_WINDOW,
        "trend_lookback": TREND_LOOKBACK,
        "garch_omega": GARCH_OMEGA,
        "garch_alpha": GARCH_ALPHA,
        "garch_beta": GARCH_BETA,
        "garch_init_window": GARCH_INIT_WINDOW,
        "default_stop_loss_pips": DEFAULT_STOP_LOSS_PIPS,
        "default_take_profit_pips": DEFAULT_TAKE_PROFIT_PIPS,
        "signal_cooldown": SIGNAL_COOLDOWN,
        "spread_pips": SPREAD_PIPS,
        "slippage_pips": SLIPPAGE_PIPS,
    }


def validate_config():
    """
    Valida que os parametros estao dentro de ranges razoaveis.
    Chame esta funcao durante inicializacao para verificar configuracao.
    """
    assert MIN_PRICES >= 50, f"MIN_PRICES muito baixo: {MIN_PRICES}"
    assert HMM_TRAINING_WINDOW >= 50, f"HMM_TRAINING_WINDOW muito baixo: {HMM_TRAINING_WINDOW}"
    assert HMM_MIN_TRAINING_SAMPLES >= 20, f"HMM_MIN_TRAINING_SAMPLES muito baixo"
    assert 0.0 < HMM_THRESHOLD_DEFAULT <= 1.0, f"HMM_THRESHOLD invalido"
    assert LYAPUNOV_THRESHOLD_DEFAULT > 0, f"LYAPUNOV_THRESHOLD invalido"
    assert GARCH_INIT_WINDOW >= 5, f"GARCH_INIT_WINDOW muito baixo"
    assert DEFAULT_STOP_LOSS_PIPS > 0, f"STOP_LOSS deve ser positivo"
    assert DEFAULT_TAKE_PROFIT_PIPS > 0, f"TAKE_PROFIT deve ser positivo"

    print(f"[CONFIG PRM] Configuracao validada:")
    print(f"  Min Prices: {MIN_PRICES}")
    print(f"  HMM Training Window: {HMM_TRAINING_WINDOW}")
    print(f"  HMM Min Samples: {HMM_MIN_TRAINING_SAMPLES}")
    print(f"  Spread: {SPREAD_PIPS} pips")
    print(f"  Slippage: {SLIPPAGE_PIPS} pips")


if __name__ == "__main__":
    validate_config()
    print("\nConfig completa:")
    for k, v in get_prm_config().items():
        print(f"  {k}: {v}")
