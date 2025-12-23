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
- QZD-LDT: Quantum Zeno Discord & Lindblad Decoherence Timer
- RZ-CID: Riemann-Zeta Cryptanalytic Iceberg Decompiler
"""

from .gjfcp_strategy import GJFCPStrategy
from .gjfcp_granular_jamming import GranularJammingForceChainPercolator

from .prsbd_strategy import PRSBDStrategy
from .prsbd_parisi_replica import ParisiReplicaSymmetryBreakingDetector

from .qzdldt_strategy import QZDLDTStrategy
from .qzdldt_quantum_zeno import QuantumZenoDiscordLindbladTimer

from .rzcid_strategy import RZCIDStrategy
from .rzcid_riemann_zeta import RiemannZetaCryptanalyticIcebergDecompiler

__all__ = [
    'GJFCPStrategy',
    'GranularJammingForceChainPercolator',
    'PRSBDStrategy',
    'ParisiReplicaSymmetryBreakingDetector',
    'QZDLDTStrategy',
    'QuantumZenoDiscordLindbladTimer',
    'RZCIDStrategy',
    'RiemannZetaCryptanalyticIcebergDecompiler',
]
