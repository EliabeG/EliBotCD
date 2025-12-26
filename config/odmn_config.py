"""
================================================================================
CONFIGURACAO CENTRALIZADA - INDICADOR ODMN (Oraculo Derivativos Malliavin-Nash)
================================================================================

IMPORTANTE: Este arquivo define TODOS os parametros do ODMN para TODO o sistema.
Qualquer componente (indicador, estrategia, backtest, otimizador) DEVE usar
estes valores para garantir CONSISTENCIA entre desenvolvimento e producao.

NUNCA modifique estes valores em arquivos individuais!
Se precisar ajustar, modifique APENAS aqui.

Criado: 25 de Dezembro de 2025
Versao: 1.0 (Alinhamento com sistema PRM - sem look-ahead bias)

FUNDAMENTOS TEORICOS DO ODMN:
============================
1. Modelo de Heston: Volatilidade estocastica (dv_t = k(theta-v_t)dt + sigma*sqrt(v_t)*dW)
2. Calculo de Malliavin: Derivadas estocasticas para medir fragilidade estrutural
3. Mean Field Games: Equilibrio de Nash em jogos com infinitos jogadores (institucionais)

SEM LOOK-AHEAD BIAS:
===================
- Calibracao Heston usa apenas dados passados (janela deslizante)
- Malliavin simula trajetorias para frente (Monte Carlo causal)
- MFG resolve PDEs sem usar dados futuros
- Direcao baseada APENAS em barras fechadas
"""

# =============================================================================
# CUSTOS DE EXECUCAO - IMPORTAR DE execution_costs.py
# =============================================================================
from config.execution_costs import (
    SPREAD_PIPS,
    SLIPPAGE_PIPS,
    COMMISSION_PER_LOT,
    get_spread,
    get_pip_value,
)


# =============================================================================
# PARAMETROS DO INDICADOR ODMN
# =============================================================================

# =============================================================================
# V2.6: CONFIGURACAO DE TIMEFRAME
# =============================================================================
# CRITICO: O modelo de Heston usa anualizacao de variancia.
# O fator de anualizacao DEVE corresponder ao timeframe dos dados.
#
# Timeframes suportados e seus fatores:
#   D1  (Diario)    -> 252 (dias de trading por ano)
#   H4  (4 horas)   -> 252 * 6 = 1512
#   H1  (1 hora)    -> 252 * 24 = 6048
#   M30 (30 min)    -> 252 * 48 = 12096
#   M15 (15 min)    -> 252 * 96 = 24192
#   M5  (5 min)     -> 252 * 288 = 72576
#
# IMPORTANTE: Ajuste TIMEFRAME para corresponder aos dados que voce esta usando!

TIMEFRAME: str = "D1"  # Timeframe dos dados de entrada

# Mapeamento de timeframe para fator de anualizacao
_ANNUALIZATION_FACTORS = {
    "D1": 252,
    "H4": 252 * 6,      # 1512
    "H1": 252 * 24,     # 6048
    "M30": 252 * 48,    # 12096
    "M15": 252 * 96,    # 24192
    "M5": 252 * 288,    # 72576
    "M1": 252 * 1440,   # 362880
}

# Fator de anualizacao (calculado automaticamente a partir do TIMEFRAME)
ANNUALIZATION_FACTOR: int = _ANNUALIZATION_FACTORS.get(TIMEFRAME, 252)

# Horizonte Malliavin em fracao de ano (dinamico por timeframe)
# Para D1: 1/252 = 1 dia | Para H1: 4/6048 = 4 horas | etc.
_MALLIAVIN_HORIZONS = {
    "D1": 1/252,        # 1 dia
    "H4": 4/1512,       # 4 horas (1 barra)
    "H1": 4/6048,       # 4 horas (4 barras)
    "M30": 2/12096,     # 2 horas (4 barras)
    "M15": 1/24192,     # 1 hora (4 barras)
    "M5": 0.5/72576,    # 30 min (6 barras)
    "M1": 0.25/362880,  # 15 min (15 barras)
}

MALLIAVIN_HORIZON: float = _MALLIAVIN_HORIZONS.get(TIMEFRAME, 1/252)

# =============================================================================
# V2.6: WARMUP (PROTECAO CONTRA COLD START)
# =============================================================================
# Numero minimo de barras antes de gerar sinais reais.
# Durante warmup, o sistema coleta dados de fragilidade para calcular percentis.
# Sem warmup, as primeiras barras teriam percentil distorcido (ex: sempre 100%).

MIN_WARMUP_BARS: int = 100  # Barras minimas antes de operar

# Numero minimo de precos necessarios para analise
# UNIFICADO: Todos os componentes devem usar este valor
MIN_PRICES: int = 150

# Janela de calibracao do Heston (apenas dados passados)
HESTON_CALIBRATION_WINDOW: int = 100

# Numero de trajetorias Monte Carlo para Malliavin
MALLIAVIN_PATHS: int = 2000

# Numero de passos temporais na simulacao
MALLIAVIN_STEPS: int = 30

# Threshold de fragilidade (percentil)
# Acima deste percentil = mercado fragil
FRAGILITY_PERCENTILE_THRESHOLD: float = 0.80

# Threshold para direcao do MFG
# |direcao| > threshold indica pressao institucional
MFG_DIRECTION_THRESHOLD: float = 0.1

# Confianca minima para gerar sinal
MIN_CONFIDENCE: float = 0.60

# Usar Deep Galerkin para MFG (True = redes neurais, False = analitico)
USE_DEEP_GALERKIN: bool = False  # Analitico e mais rapido e sem look-ahead


# =============================================================================
# PARAMETROS DO MODELO DE HESTON
# =============================================================================

# Valores default (serao recalibrados com dados reais)
HESTON_KAPPA_DEFAULT: float = 2.0      # Velocidade de reversao a media
HESTON_THETA_DEFAULT: float = 0.04     # Nivel medio de variancia
HESTON_SIGMA_DEFAULT: float = 0.3      # Vol da vol
HESTON_RHO_DEFAULT: float = -0.7       # Correlacao (tipicamente negativa)
HESTON_V0_DEFAULT: float = 0.04        # Variancia inicial


# =============================================================================
# PARAMETROS DE TRADE
# =============================================================================

# Stop loss padrao em pips
DEFAULT_STOP_LOSS_PIPS: float = 25.0

# Take profit padrao em pips
DEFAULT_TAKE_PROFIT_PIPS: float = 50.0

# Cooldown entre sinais (em barras)
SIGNAL_COOLDOWN: int = 25  # Maior que PRM devido a complexidade do calculo

# Numero de barras para calcular tendencia (direcao do trade)
TREND_LOOKBACK: int = 10


# =============================================================================
# PARAMETROS DE VALIDACAO (DINHEIRO REAL)
# =============================================================================

# Numero minimo de trades para validacao
MIN_TRADES_VALIDATION: int = 30

# Profit factor minimo para dinheiro real
MIN_PF_REAL_MONEY: float = 1.2

# Expectancy minima em pips
MIN_EXPECTANCY_PIPS: float = 2.0


# =============================================================================
# FUNCOES AUXILIARES
# =============================================================================

def get_odmn_config() -> dict:
    """
    Retorna todos os parametros do ODMN como um dicionario.
    Util para logging e debugging.
    """
    return {
        # V2.6: Configuracoes de timeframe
        "timeframe": TIMEFRAME,
        "annualization_factor": ANNUALIZATION_FACTOR,
        "malliavin_horizon": MALLIAVIN_HORIZON,
        "min_warmup_bars": MIN_WARMUP_BARS,
        # Parametros principais
        "min_prices": MIN_PRICES,
        "heston_calibration_window": HESTON_CALIBRATION_WINDOW,
        "malliavin_paths": MALLIAVIN_PATHS,
        "malliavin_steps": MALLIAVIN_STEPS,
        "fragility_percentile_threshold": FRAGILITY_PERCENTILE_THRESHOLD,
        "mfg_direction_threshold": MFG_DIRECTION_THRESHOLD,
        "min_confidence": MIN_CONFIDENCE,
        "use_deep_galerkin": USE_DEEP_GALERKIN,
        "heston_kappa_default": HESTON_KAPPA_DEFAULT,
        "heston_theta_default": HESTON_THETA_DEFAULT,
        "heston_sigma_default": HESTON_SIGMA_DEFAULT,
        "heston_rho_default": HESTON_RHO_DEFAULT,
        "heston_v0_default": HESTON_V0_DEFAULT,
        "default_stop_loss_pips": DEFAULT_STOP_LOSS_PIPS,
        "default_take_profit_pips": DEFAULT_TAKE_PROFIT_PIPS,
        "signal_cooldown": SIGNAL_COOLDOWN,
        "trend_lookback": TREND_LOOKBACK,
        "spread_pips": SPREAD_PIPS,
        "slippage_pips": SLIPPAGE_PIPS,
    }


def validate_config():
    """
    Valida que os parametros estao dentro de ranges razoaveis.
    Chame esta funcao durante inicializacao para verificar configuracao.
    """
    # V2.6: Validar timeframe
    assert TIMEFRAME in _ANNUALIZATION_FACTORS, f"TIMEFRAME invalido: {TIMEFRAME}"
    assert MIN_WARMUP_BARS >= 50, f"MIN_WARMUP_BARS muito baixo: {MIN_WARMUP_BARS}"

    # Validacoes originais
    assert MIN_PRICES >= 100, f"MIN_PRICES muito baixo: {MIN_PRICES}"
    assert HESTON_CALIBRATION_WINDOW >= 50, f"HESTON_CALIBRATION_WINDOW muito baixo"
    assert MALLIAVIN_PATHS >= 500, f"MALLIAVIN_PATHS muito baixo para Monte Carlo"
    assert MALLIAVIN_STEPS >= 10, f"MALLIAVIN_STEPS muito baixo"
    assert 0.0 < FRAGILITY_PERCENTILE_THRESHOLD <= 1.0, f"FRAGILITY threshold invalido"
    assert MFG_DIRECTION_THRESHOLD > 0, f"MFG_DIRECTION_THRESHOLD invalido"
    assert 0.0 < MIN_CONFIDENCE <= 1.0, f"MIN_CONFIDENCE invalido"
    assert DEFAULT_STOP_LOSS_PIPS > 0, f"STOP_LOSS deve ser positivo"
    assert DEFAULT_TAKE_PROFIT_PIPS > 0, f"TAKE_PROFIT deve ser positivo"

    print(f"[CONFIG ODMN] Configuracao validada (V2.6):")
    print(f"  Timeframe: {TIMEFRAME}")
    print(f"  Annualization Factor: {ANNUALIZATION_FACTOR}")
    print(f"  Malliavin Horizon: {MALLIAVIN_HORIZON:.6f}")
    print(f"  Min Warmup Bars: {MIN_WARMUP_BARS}")
    print(f"  Min Prices: {MIN_PRICES}")
    print(f"  Heston Calibration Window: {HESTON_CALIBRATION_WINDOW}")
    print(f"  Malliavin Paths: {MALLIAVIN_PATHS}")
    print(f"  Fragility Threshold: P{FRAGILITY_PERCENTILE_THRESHOLD*100:.0f}")
    print(f"  MFG Direction Threshold: {MFG_DIRECTION_THRESHOLD}")
    print(f"  Use Deep Galerkin: {USE_DEEP_GALERKIN}")
    print(f"  Spread: {SPREAD_PIPS} pips")
    print(f"  Slippage: {SLIPPAGE_PIPS} pips")


if __name__ == "__main__":
    validate_config()
    print("\nConfig completa:")
    for k, v in get_odmn_config().items():
        print(f"  {k}: {v}")
