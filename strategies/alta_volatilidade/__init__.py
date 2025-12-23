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
3. FIFN (Fluxo de Informação Fisher-Navier) - Navier-Stokes + Reynolds
4. DSG (Detector de Singularidade Gravitacional) - Relatividade geral + Schwarzschild
5. ODMN (Otimizador Dinâmico Malliavin-Nash) - Cálculo de Malliavin + Equilíbrio de Nash
6. PHM (Projetor Holográfico de Mercado) - Princípio holográfico AdS/CFT
7. RHHF (Ressonador Hilbert-Huang Fracionário) - EMD + Transformada de Hilbert
8. SEED (Sintetizador Evolutivo de Estruturas Dissipativas) - Termodinâmica + Dinâmica evolutiva
9. SEMA (Sincronizador Espectral de Mercados) - RMT + Teoria Espectral de Grafos
10. STGK (Sintetizador de Topos Grothendieck-Kolmogorov) - Teoria das Categorias + Complexidade de Kolmogorov
"""

# PRM - Protocolo Riemann-Mandelbrot
from .prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot, plot_prm_analysis
from .prm_strategy import PRMStrategy

# DTT - Detector de Tunelamento Topológico
from .dtt_tunelamento_topologico import DetectorTunelamentoTopologico
from .dtt_strategy import DTTStrategy

# FIFN - Fluxo de Informação Fisher-Navier
from .fifn_fisher_navier import FluxoInformacaoFisherNavier
from .fifn_strategy import FIFNStrategy

# DSG - Detector de Singularidade Gravitacional
from .dsg_detector_singularidade import DetectorSingularidadeGravitacional
from .dsg_strategy import DSGStrategy

# ODMN - Oráculo Derivativos Malliavin-Nash
from .odmn_malliavin_nash import OracloDerivativosMalliavinNash
from .odmn_strategy import ODMNStrategy

# PHM - Projetor Holográfico Maldacena
from .phm_projetor_holografico import ProjetorHolograficoMaldacena
from .phm_strategy import PHMStrategy

# RHHF - Ressonador Hilbert-Huang Fractal
from .rhhf_ressonador_hilbert_huang import RessonadorHilbertHuangFractal
from .rhhf_strategy import RHHFStrategy

# SEED - Sintetizador Evolutivo de Estruturas Dissipativas
from .seed_sintetizador_evolutivo import SintetizadorEvolutivoEstruturasDissipativas
from .seed_strategy import SEEDStrategy

# SEMA - Sincronizador Espectral de Mercados
from .sema_sincronizador_espectral import SincronizadorEspectral
from .sema_strategy import SEMAStrategy

# STGK - Sintetizador de Topos Grothendieck-Kolmogorov
from .stgk_sintetizador_topos import SintetizadorToposGrothendieckKolmogorov
from .stgk_strategy import STGKStrategy

__all__ = [
    # PRM
    'ProtocoloRiemannMandelbrot',
    'plot_prm_analysis',
    'PRMStrategy',
    # DTT
    'DetectorTunelamentoTopologico',
    'DTTStrategy',
    # FIFN
    'FluxoInformacaoFisherNavier',
    'FIFNStrategy',
    # DSG
    'DetectorSingularidadeGravitacional',
    'DSGStrategy',
    # ODMN
    'OracloDerivativosMalliavinNash',
    'ODMNStrategy',
    # PHM
    'ProjetorHolograficoMaldacena',
    'PHMStrategy',
    # RHHF
    'RessonadorHilbertHuangFractal',
    'RHHFStrategy',
    # SEED
    'SintetizadorEvolutivoEstruturasDissipativas',
    'SEEDStrategy',
    # SEMA
    'SincronizadorEspectral',
    'SEMAStrategy',
    # STGK
    'SintetizadorToposGrothendieckKolmogorov',
    'STGKStrategy',
]
