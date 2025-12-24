"""
================================================================================
STRATEGY FACTORY
Fabrica de Estrategias Reais
================================================================================

Este modulo cria instancias de todas as estrategias reais do sistema,
organizadas por nivel de volatilidade.

Alta Volatilidade (10 estrategias):
- PRMStrategy: Protocolo Riemann-Mandelbrot
- DTTStrategy: Detector Tunelamento Topologico
- FIFNStrategy: Fluxo Informacao Fisher-Navier
- DSGStrategy: Detector de Singularidade Gravitacional
- ODMNStrategy: Otimizador Dinamico Malliavin-Nash
- PHMStrategy: Projetor Holografico de Mercado
- RHHFStrategy: Ressonador Hilbert-Huang Fractal
- SEEDStrategy: Sintetizador Evolutivo de Estrategias Dinamicas
- SEMAStrategy: Sincronizador Espectral Multi-Ativo
- STGKStrategy: Sintetizador Topos Geometrico Kahler

Media Volatilidade (10 estrategias):
- LSQPCStrategy: Langevin-Schrodinger Quantum Probability Cloud
- H2PLOStrategy: Hilbert-Huang Phase-Lock Oscillator
- RCTFStrategy: Riemannian Curvature Tensor Flow
- BPHSStrategy: Betti-Persistence Homology Scanner
- KdVSHStrategy: Korteweg-de Vries Soliton Hunter
- FSIGEStrategy: Fisher-Shannon Information Gravity Engine
- HJBNESStrategy: Hamilton-Jacobi-Bellman Nash Equilibrium Solver
- MPSDEOStrategy: Marchenko-Pastur Spectral De-Noiser
- MVGKSDStrategy: Multiplex Visibility Graph Kuramoto Sync Detector
- FKQPIPStrategy: Feynman-Kac Quantum Path Integral Propagator

Baixa Volatilidade (10 estrategias):
- GJFCPStrategy: Granular Jamming Force Chain Percolator
- PRSBDStrategy: Parisi Replica Symmetry Breaking Detector
- QZDLDTStrategy: Quantum Zeno Discord Lindblad Timer
- RZCIDStrategy: Riemann-Zeta Cryptanalytic Iceberg Decompiler
- NSPPSStrategy: Neuromorphic Spiking Pre-Potentiation Scanner
- IPCBMStrategy: Invasive Percolation Capillary Breakthrough Monitor
- GLDMDStrategy: Gravitational Lensing Dark Matter Detector
- RDMEStrategy: Reaction-Diffusion Morphogenesis Engine
- GMSCSStrategy: Global Macro Spectral Coherence Scanner
- HBBPStrategy: Holographic AdS/CFT Bulk-Boundary Projector
"""

from typing import Dict, List
from .base import BaseStrategy

# Alta Volatilidade (10 estrategias)
from .alta_volatilidade import (
    PRMStrategy, DTTStrategy, FIFNStrategy,
    DSGStrategy, ODMNStrategy, PHMStrategy,
    RHHFStrategy, SEEDStrategy, SEMAStrategy, STGKStrategy
)

# Media Volatilidade
from .media_volatilidade import (
    LSQPCStrategy, H2PLOStrategy, RCTFStrategy, BPHSStrategy,
    KdVSHStrategy, FSIGEStrategy, HJBNESStrategy, MPSDEOStrategy,
    MVGKSDStrategy, FKQPIPStrategy
)

# Baixa Volatilidade
from .baixa_volatilidade import (
    GJFCPStrategy, PRSBDStrategy, QZDLDTStrategy, RZCIDStrategy,
    NSPPSStrategy, IPCBMStrategy, GLDMDStrategy, RDMEStrategy,
    GMSCSStrategy, HBBPStrategy
)


def create_alta_volatilidade_strategies() -> List[BaseStrategy]:
    """
    Cria estrategias para alta volatilidade (> 0.385 pips em 5s)

    Estas estrategias sao otimizadas para mercados com movimentos
    rapidos e amplos. Usam stops mais largos e buscam capturar
    grandes movimentos direcionais.

    Returns:
        Lista de estrategias para alta volatilidade
    """
    strategies = [
        # PRM - Detecta singularidades de preco usando HMM, Lyapunov e curvatura
        PRMStrategy(
            min_prices=100,
            stop_loss_pips=20.0,
            take_profit_pips=40.0,
            hmm_threshold=0.85,
            lyapunov_threshold=0.5,
            curvature_threshold=0.1
        ),

        # DTT - Analise topologica + tunelamento quantico
        DTTStrategy(
            min_prices=100,
            stop_loss_pips=20.0,
            take_profit_pips=40.0
        ),

        # FIFN - Navier-Stokes + Fisher Information
        FIFNStrategy(
            min_prices=100,
            stop_loss_pips=20.0,
            take_profit_pips=40.0
        ),

        # DSG - Detector de Singularidade Gravitacional
        DSGStrategy(
            min_prices=100,
            stop_loss_pips=20.0,
            take_profit_pips=40.0
        ),

        # ODMN - Otimizador Dinamico Malliavin-Nash
        ODMNStrategy(
            min_prices=100,
            stop_loss_pips=20.0,
            take_profit_pips=40.0
        ),

        # PHM - Projetor Holografico de Mercado
        PHMStrategy(
            min_prices=100,
            stop_loss_pips=20.0,
            take_profit_pips=40.0
        ),

        # RHHF - Ressonador Hilbert-Huang Fractal
        RHHFStrategy(
            min_prices=100,
            stop_loss_pips=20.0,
            take_profit_pips=40.0
        ),

        # SEED - Sintetizador Evolutivo de Estrategias Dinamicas
        SEEDStrategy(
            min_prices=100,
            stop_loss_pips=20.0,
            take_profit_pips=40.0
        ),

        # SEMA - Sincronizador Espectral Multi-Ativo
        SEMAStrategy(
            min_prices=100,
            stop_loss_pips=20.0,
            take_profit_pips=40.0
        ),

        # STGK - Sintetizador Topos Geometrico Kahler
        STGKStrategy(
            min_prices=100,
            stop_loss_pips=20.0,
            take_profit_pips=40.0
        ),
    ]

    return strategies


def create_media_volatilidade_strategies() -> List[BaseStrategy]:
    """
    Cria estrategias para media volatilidade (0.238 - 0.385 pips em 5s)

    Estas estrategias sao otimizadas para condicoes normais de mercado.
    Boas para tendencias e swings, com stops e alvos moderados.

    Returns:
        Lista de estrategias para media volatilidade
    """
    strategies = [
        # LSQPC - Fisica estatistica quantica
        LSQPCStrategy(
            min_prices=100,
            stop_loss_pips=20.0,
            take_profit_pips=40.0,
            n_trajectories=3000,
            forecast_horizon=15
        ),

        # H2PLO - Hilbert-Huang Phase Lock
        H2PLOStrategy(
            min_prices=100,
            stop_loss_pips=18.0,
            take_profit_pips=36.0
        ),

        # RCTF - Curvatura Riemanniana
        RCTFStrategy(
            min_prices=100,
            stop_loss_pips=18.0,
            take_profit_pips=36.0
        ),

        # BPHS - Homologia Persistente
        BPHSStrategy(
            min_prices=100,
            stop_loss_pips=18.0,
            take_profit_pips=36.0
        ),

        # KdVSH - Solitons
        KdVSHStrategy(
            min_prices=100,
            stop_loss_pips=18.0,
            take_profit_pips=36.0
        ),

        # FSIGE - Fisher-Shannon Gravity
        FSIGEStrategy(
            min_prices=100,
            stop_loss_pips=18.0,
            take_profit_pips=36.0
        ),

        # HJBNES - Nash Equilibrium
        HJBNESStrategy(
            min_prices=100,
            stop_loss_pips=18.0,
            take_profit_pips=36.0
        ),

        # MPSDEO - Marchenko-Pastur Spectral
        MPSDEOStrategy(
            min_prices=100,
            stop_loss_pips=18.0,
            take_profit_pips=36.0
        ),

        # MVGKSD - Visibility Graph Kuramoto
        MVGKSDStrategy(
            min_prices=100,
            stop_loss_pips=18.0,
            take_profit_pips=36.0
        ),

        # FKQPIP - Feynman-Kac Path Integral
        FKQPIPStrategy(
            min_prices=100,
            stop_loss_pips=18.0,
            take_profit_pips=36.0
        ),
    ]

    return strategies


def create_baixa_volatilidade_strategies() -> List[BaseStrategy]:
    """
    Cria estrategias para baixa volatilidade (< 0.238 pips em 5s)

    Estas estrategias sao otimizadas para mercados em consolidacao/range.
    Focam em reversao a media e deteccao de breakouts iminentes.

    Returns:
        Lista de estrategias para baixa volatilidade
    """
    strategies = [
        # GJFCP - Granular Jamming (detecta transicoes solido->liquido)
        GJFCPStrategy(
            min_prices=50,
            stop_loss_pips=15.0,
            take_profit_pips=30.0,
            particles_per_bar=5,
            compactness_critical=0.64
        ),

        # PRSBD - Parisi Replica (quebra de simetria)
        PRSBDStrategy(
            min_prices=50,
            stop_loss_pips=15.0,
            take_profit_pips=30.0
        ),

        # QZDLDT - Quantum Zeno Discord
        QZDLDTStrategy(
            min_prices=50,
            stop_loss_pips=15.0,
            take_profit_pips=30.0
        ),

        # RZCID - Riemann-Zeta
        RZCIDStrategy(
            min_prices=50,
            stop_loss_pips=15.0,
            take_profit_pips=30.0
        ),

        # NSPPS - Neuromorphic Spiking
        NSPPSStrategy(
            min_prices=50,
            stop_loss_pips=15.0,
            take_profit_pips=30.0
        ),

        # IPCBM - Invasive Percolation
        IPCBMStrategy(
            min_prices=50,
            stop_loss_pips=15.0,
            take_profit_pips=30.0
        ),

        # GLDMD - Gravitational Lensing Dark Matter
        GLDMDStrategy(
            min_prices=50,
            stop_loss_pips=15.0,
            take_profit_pips=30.0
        ),

        # RDME - Reaction-Diffusion Morphogenesis
        RDMEStrategy(
            min_prices=50,
            stop_loss_pips=15.0,
            take_profit_pips=30.0
        ),

        # GMSCS - Global Macro Spectral Coherence
        GMSCSStrategy(
            min_prices=50,
            stop_loss_pips=15.0,
            take_profit_pips=30.0
        ),

        # HBBP - Holographic AdS/CFT
        HBBPStrategy(
            min_prices=50,
            stop_loss_pips=15.0,
            take_profit_pips=30.0
        ),
    ]

    return strategies


def create_all_strategies() -> Dict[str, List[BaseStrategy]]:
    """
    Cria todas as estrategias do sistema organizadas por volatilidade

    Returns:
        Dict com niveis de volatilidade e suas estrategias:
        {
            'ALTA': [...],
            'MEDIA': [...],
            'BAIXA': [...]
        }
    """
    return {
        'ALTA': create_alta_volatilidade_strategies(),
        'MEDIA': create_media_volatilidade_strategies(),
        'BAIXA': create_baixa_volatilidade_strategies()
    }


def get_strategy_count() -> Dict[str, int]:
    """
    Retorna contagem de estrategias por nivel

    Returns:
        Dict com contagem por nivel
    """
    return {
        'ALTA': 10,
        'MEDIA': 10,
        'BAIXA': 10,
        'TOTAL': 30
    }


def get_strategy_names() -> Dict[str, List[str]]:
    """
    Retorna nomes das estrategias por nivel

    Returns:
        Dict com nomes das estrategias
    """
    return {
        'ALTA': [
            'PRM-RiemannMandelbrot',
            'DTT-TunelamentoTopologico',
            'FIFN-FisherNavier',
            'DSG-Singularidade',
            'ODMN-MalliavinNash',
            'PHM-Holografico',
            'RHHF-HilbertHuang',
            'SEED-Evolutivo',
            'SEMA-Espectral',
            'STGK-ToposKahler'
        ],
        'MEDIA': [
            'LSQPC-QuantumProbability',
            'H2PLO-HilbertHuang',
            'RCTF-RiemannCurvature',
            'BPHS-BettiPersistence',
            'KdVSH-Soliton',
            'FSIGE-FisherShannon',
            'HJBNES-NashEquilibrium',
            'MPSDEO-MarchenkoPastur',
            'MVGKSD-VisibilityKuramoto',
            'FKQPIP-FeynmanKac'
        ],
        'BAIXA': [
            'GJFCP-Granular',
            'PRSBD-ParisiReplica',
            'QZDLDT-QuantumZeno',
            'RZCID-RiemannZeta',
            'NSPPS-Neuromorphic',
            'IPCBM-Percolation',
            'GLDMD-GravitationalLensing',
            'RDME-ReactionDiffusion',
            'GMSCS-SpectralCoherence',
            'HBBP-Holographic'
        ]
    }


if __name__ == "__main__":
    # Teste de criacao
    print("=" * 60)
    print("  STRATEGY FACTORY TEST")
    print("=" * 60)

    strategies = create_all_strategies()

    for level, strats in strategies.items():
        print(f"\n{level} VOLATILIDADE ({len(strats)} estrategias):")
        for s in strats:
            print(f"  - {s.name}")

    print(f"\nTotal: {sum(len(s) for s in strategies.values())} estrategias")
