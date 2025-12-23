"""
Estrategias para Media Volatilidade

Quando o mercado esta em MEDIA volatilidade (0.238 - 0.385 pips em 5s):
- Condicoes normais de mercado
- Bom para estrategias de tendencia
- Stops e alvos moderados
- Indicadores classicos funcionam bem

Indicadores disponiveis:
- LSQPC: Langevin-Schrodinger Quantum Probability Cloud
- H2PLO: Hilbert-Huang Phase-Lock Oscillator
- RCTF: Riemannian Curvature Tensor Flow
- BPHS: Betti-Persistence Homology Scanner
- KdVSH: Korteweg-de Vries Soliton Hunter
- FSIGE: Fisher-Shannon Information Gravity Engine
- HJBNES: Hamilton-Jacobi-Bellman Nash Equilibrium Solver
- MPSDEO: Marchenko-Pastur Spectral De-Noiser & Eigen-Entropic Oscillator
- MVGKSD: Multiplex Visibility Graph & Kuramoto Synchronization Detector
- FKQPIP: Feynman-Kac Quantum Path Integral Propagator
"""

from .lsqpc_strategy import LSQPCStrategy
from .lsqpc_langevin_schrodinger import LangevinSchrodingerQuantumIndicator

from .h2plo_strategy import H2PLOStrategy
from .h2plo_hilbert_huang import HilbertHuangPhaseLockOscillator

from .rctf_strategy import RCTFStrategy
from .rctf_riemannian_curvature import RiemannianCurvatureTensorFlow

from .bphs_strategy import BPHSStrategy
from .bphs_betti_persistence import BettiPersistenceHomologyScanner

from .kdvsh_strategy import KdVSHStrategy
from .kdvsh_korteweg_devries import KdVSolitonHunter

from .fsige_strategy import FSIGEStrategy
from .fsige_fisher_shannon import FisherShannonInformationGravityEngine

from .hjbnes_strategy import HJBNESStrategy
from .hjbnes_hamilton_jacobi import HJBNashEquilibriumSolver

from .mpsdeo_strategy import MPSDEOStrategy
from .mpsdeo_marchenko_pastur import MarchenkoPasturSpectralDeNoiser

from .mvgksd_strategy import MVGKSDStrategy
from .mvgksd_visibility_kuramoto import MultiplexVisibilityKuramotoDetector

from .fkqpip_strategy import FKQPIPStrategy
from .fkqpip_feynman_kac import FeynmanKacQuantumPathIntegralPropagator

__all__ = [
    'LSQPCStrategy',
    'LangevinSchrodingerQuantumIndicator',
    'H2PLOStrategy',
    'HilbertHuangPhaseLockOscillator',
    'RCTFStrategy',
    'RiemannianCurvatureTensorFlow',
    'BPHSStrategy',
    'BettiPersistenceHomologyScanner',
    'KdVSHStrategy',
    'KdVSolitonHunter',
    'FSIGEStrategy',
    'FisherShannonInformationGravityEngine',
    'HJBNESStrategy',
    'HJBNashEquilibriumSolver',
    'MPSDEOStrategy',
    'MarchenkoPasturSpectralDeNoiser',
    'MVGKSDStrategy',
    'MultiplexVisibilityKuramotoDetector',
    'FKQPIPStrategy',
    'FeynmanKacQuantumPathIntegralPropagator',
]
