"""
Estratégias para Alta Volatilidade

Quando o mercado está em ALTA volatilidade (> 0.385 pips em 5s):
- Movimentos rápidos e amplos
- Maior risco, maior potencial de lucro
- Requer stops mais largos
- Indicadores de momentum e breakout funcionam melhor

Indicadores disponíveis:
1. PRM (Protocolo Riemann-Mandelbrot) - Detecção de singularidade de preço
2. DTT (Detector de Tunelamento Topológico) - Análise topológica + quântica
"""

from .prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot, plot_prm_analysis
from .prm_strategy import PRMStrategy
from .dtt_tunelamento_topologico import DetectorTunelamentoTopologico
from .dtt_strategy import DTTStrategy

__all__ = [
    # PRM
    'ProtocoloRiemannMandelbrot',
    'plot_prm_analysis',
    'PRMStrategy',
    # DTT
    'DetectorTunelamentoTopologico',
    'DTTStrategy',
]
