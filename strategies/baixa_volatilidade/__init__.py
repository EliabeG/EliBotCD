"""
Estrategias para Baixa Volatilidade

Quando o mercado esta em BAIXA volatilidade (< 0.238 pips em 5s):
- Mercado em consolidacao/range
- Menor risco, menor potencial
- Bom para estrategias de reversao a media
- Aguardar breakout pode ser mais lucrativo

Indicadores disponiveis:
- GJFCP: Granular Jamming & Force Chain Percolator
- PRSBD: Parisi Replica Symmetry Breaking Detector
"""

from .gjfcp_strategy import GJFCPStrategy
from .gjfcp_granular_jamming import GranularJammingForceChainPercolator

from .prsbd_strategy import PRSBDStrategy
from .prsbd_parisi_replica import ParisiReplicaSymmetryBreakingDetector

__all__ = [
    'GJFCPStrategy',
    'GranularJammingForceChainPercolator',
    'PRSBDStrategy',
    'ParisiReplicaSymmetryBreakingDetector',
]
