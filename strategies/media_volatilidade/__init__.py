"""
Estratégias para Média Volatilidade

Quando o mercado está em MÉDIA volatilidade (0.238 - 0.385 pips em 5s):
- Condições normais de mercado
- Bom para estratégias de tendência
- Stops e alvos moderados
- Indicadores clássicos funcionam bem

Indicadores disponíveis:
- LSQPC: Langevin-Schrödinger Quantum Probability Cloud
- H2PLO: Hilbert-Huang Phase-Lock Oscillator
- RCTF: Riemannian Curvature Tensor Flow
"""

from .lsqpc_strategy import LSQPCStrategy
from .lsqpc_langevin_schrodinger import LangevinSchrodingerQuantumIndicator

from .h2plo_strategy import H2PLOStrategy
from .h2plo_hilbert_huang import HilbertHuangPhaseLockOscillator

from .rctf_strategy import RCTFStrategy
from .rctf_riemannian_curvature import RiemannianCurvatureTensorFlow

__all__ = [
    'LSQPCStrategy',
    'LangevinSchrodingerQuantumIndicator',
    'H2PLOStrategy',
    'HilbertHuangPhaseLockOscillator',
    'RCTFStrategy',
    'RiemannianCurvatureTensorFlow',
]
