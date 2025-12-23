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
- NS-PPS: Neuromorphic Spiking Synaptic Pre-Potentiation Scanner
- IP-CBM: Invasive Percolation & Capillary Breakthrough Monitor
- GL-DMD: Gravitational Lensing & Dark Matter Detector
- RD-ME: Turing-Gray-Scott Reaction-Diffusion Morphogenesis Engine
- GMS-CS: Global Macro Spectral Coherence Scanner
"""

from .gjfcp_strategy import GJFCPStrategy
from .gjfcp_granular_jamming import GranularJammingForceChainPercolator

from .prsbd_strategy import PRSBDStrategy
from .prsbd_parisi_replica import ParisiReplicaSymmetryBreakingDetector

from .qzdldt_strategy import QZDLDTStrategy
from .qzdldt_quantum_zeno import QuantumZenoDiscordLindbladTimer

from .rzcid_strategy import RZCIDStrategy
from .rzcid_riemann_zeta import RiemannZetaCryptanalyticIcebergDecompiler

from .nspps_strategy import NSPPSStrategy
from .nspps_neuromorphic_spiking import NeuromorphicSpikingPrePotentiationScanner

from .ipcbm_strategy import IPCBMStrategy
from .ipcbm_invasive_percolation import InvasivePercolationCapillaryBreakthroughMonitor

from .gldmd_strategy import GLDMDStrategy
from .gldmd_gravitational_lensing import GravitationalLensingDarkMatterDetector

from .rdme_strategy import RDMEStrategy
from .rdme_reaction_diffusion import TuringGrayScottReactionDiffusionMorphogenesisEngine

from .gmscs_strategy import GMSCSStrategy
from .gmscs_spectral_coherence import GlobalMacroSpectralCoherenceScanner

__all__ = [
    'GJFCPStrategy',
    'GranularJammingForceChainPercolator',
    'PRSBDStrategy',
    'ParisiReplicaSymmetryBreakingDetector',
    'QZDLDTStrategy',
    'QuantumZenoDiscordLindbladTimer',
    'RZCIDStrategy',
    'RiemannZetaCryptanalyticIcebergDecompiler',
    'NSPPSStrategy',
    'NeuromorphicSpikingPrePotentiationScanner',
    'IPCBMStrategy',
    'InvasivePercolationCapillaryBreakthroughMonitor',
    'GLDMDStrategy',
    'GravitationalLensingDarkMatterDetector',
    'RDMEStrategy',
    'TuringGrayScottReactionDiffusionMorphogenesisEngine',
    'GMSCSStrategy',
    'GlobalMacroSpectralCoherenceScanner',
]
