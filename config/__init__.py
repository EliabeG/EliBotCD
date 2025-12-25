# Importar custos de execução primeiro (não dependem de dotenv)
from .execution_costs import (
    SPREAD_PIPS,
    SLIPPAGE_PIPS,
    COMMISSION_PER_LOT,
    PIP_VALUES,
    USD_PER_PIP_PER_LOT,
    SPREAD_BY_PAIR,
    get_pip_value,
    get_usd_per_pip,
    get_spread,
    calculate_pnl_usd,
)

# Importar gerador de volumes sintéticos (V3.1)
from .volume_generator import (
    generate_synthetic_volumes,
    generate_single_volume,
    get_volume_base,
    get_volume_multiplier,
    validate_volumes,
    VOLUME_MULTIPLIER,
    VOLUME_BASE,
)

# Tentar importar settings (pode falhar se dotenv não estiver instalado)
try:
    from .settings import *
except ImportError:
    pass  # Ignora se dotenv não estiver disponível
