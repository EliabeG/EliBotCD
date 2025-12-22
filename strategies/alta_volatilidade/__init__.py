"""
Estratégias para Alta Volatilidade

Quando o mercado está em ALTA volatilidade (> 0.385 pips em 5s):
- Movimentos rápidos e amplos
- Maior risco, maior potencial de lucro
- Requer stops mais largos
- Indicadores de momentum e breakout funcionam melhor

Indicadores disponíveis:
1. PRM (Protocolo Riemann-Mandelbrot) - Detecção de singularidade de preço
"""

from .prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot, plot_prm_analysis
from .prm_strategy import PRMStrategy

__all__ = [
    'ProtocoloRiemannMandelbrot',
    'plot_prm_analysis',
    'PRMStrategy',
]
