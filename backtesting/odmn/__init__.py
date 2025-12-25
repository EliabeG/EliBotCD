"""
================================================================================
ODMN-MalliavinNash - Backtesting e Otimizacao V2.0
================================================================================

Oraculo de Derivativos de Malliavin-Nash (ODMN)

FUNDAMENTOS TEORICOS:
====================
1. Modelo de Heston: Volatilidade estocastica calibrada em janela deslizante
2. Calculo de Malliavin: Derivadas estocasticas para fragilidade estrutural
3. Mean Field Games: Equilibrio Nash para comportamento institucional

COMPONENTES:
============
- backtest.py: Backtest simples com dados reais
- debug.py: Analise de distribuicao de sinais
- optimizer.py: Otimizador V2.0 com Walk-Forward Validation
- verify_real_money_ready.py: Verificacao de prontidao para producao

SEM LOOK-AHEAD BIAS:
===================
- Calibracao Heston usa apenas dados passados (janela deslizante)
- Malliavin simula trajetorias para frente (Monte Carlo causal)
- MFG resolve PDEs sem usar dados futuros
- Direcao baseada APENAS em barras fechadas
- Entrada no OPEN da proxima barra

VERSAO PRONTA PARA DINHEIRO REAL
"""

from .optimizer import ODMNRobustOptimizer
from .backtest import run_odmn_backtest, create_odmn_strategy

__all__ = [
    'ODMNRobustOptimizer',
    'run_odmn_backtest',
    'create_odmn_strategy',
]
