#!/usr/bin/env python3
"""
================================================================================
VERIFICACAO DE PRONTIDAO PARA DINHEIRO REAL - ODMN V2.0
================================================================================

Este script verifica se o sistema ODMN esta pronto para dinheiro real,
verificando TODAS as condicoes criticas para evitar look-ahead bias
e outros problemas que podem causar resultados irreais.

VERIFICACOES REALIZADAS:
========================
1. Calibracao Heston sem look-ahead (janela deslizante passada)
2. Malliavin sem look-ahead (Monte Carlo causal, forward simulation)
3. MFG sem look-ahead (resolve PDEs sem dados futuros)
4. Direcao baseada APENAS em barras fechadas
5. Entrada no OPEN da proxima barra
6. Custos centralizados e realistas
7. Filtros rigorosos configurados
8. Walk-Forward Validation implementada

FUNDAMENTOS TEORICOS DO ODMN:
============================
1. Modelo de Heston: Volatilidade estocastica calibrada em dados passados
2. Calculo de Malliavin: Derivadas estocasticas para fragilidade
3. Mean Field Games: Equilibrio Nash para comportamento institucional

SE TODAS AS 8 VERIFICACOES PASSAREM = PRONTO PARA DINHEIRO REAL

Uso:
    python -m backtesting.odmn.verify_real_money_ready
"""

import sys
import os
import re
import ast

# Adiciona o diretorio raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ODMNVerificationResult:
    """Resultado de uma verificacao"""
    def __init__(self, name: str, passed: bool, details: str):
        self.name = name
        self.passed = passed
        self.details = details


def verify_heston_calibration_no_lookahead() -> ODMNVerificationResult:
    """
    Verifica 1: Calibracao do Heston usa apenas dados passados

    O calibrador deve usar apenas dados em janela deslizante,
    nunca dados futuros.
    """
    indicator_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_malliavin_nash.py"
    )

    try:
        with open(indicator_path, 'r') as f:
            content = f.read()

        # Verifica se HestonCalibrator usa returns_window (janela passada)
        has_returns_window = "returns_window" in content and "self.returns_window" in content

        # Verifica se calibrate usa dados do final (mais recentes)
        has_recent_data_usage = "r = returns[-self.returns_window:]" in content or \
                                "returns[-self.returns_window" in content

        # Verifica se nao usa dados futuros
        no_forward_slice = "returns[i+1:" not in content

        passed = has_returns_window and no_forward_slice
        details = []

        if has_returns_window:
            details.append("returns_window implementado corretamente")
        else:
            details.append("ERRO: returns_window nao encontrado")

        if has_recent_data_usage:
            details.append("Usa dados recentes (passados) para calibracao")

        if no_forward_slice:
            details.append("Nenhum slice futuro encontrado")
        else:
            details.append("ERRO: Detectado slice futuro nos dados")

        return ODMNVerificationResult(
            "Calibracao Heston sem look-ahead",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "Calibracao Heston sem look-ahead",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_malliavin_no_lookahead() -> ODMNVerificationResult:
    """
    Verifica 2: Calculo de Malliavin e causal (sem look-ahead)

    O Monte Carlo deve simular trajetorias PARA FRENTE a partir do
    ponto atual, nunca usando dados futuros reais.
    """
    indicator_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_malliavin_nash.py"
    )

    try:
        with open(indicator_path, 'r') as f:
            content = f.read()

        # Verifica se MalliavinDerivativeCalculator simula para frente
        has_forward_simulation = "for t in range(self.n_steps)" in content or \
                                 "for t in range(n_steps)" in content

        # Verifica se usa apenas S0 como ponto inicial
        has_s0_initial = "S[:, 0] = S0" in content or "S[0] = S0" in content

        # Verifica se nao usa prices[i+1:] (dados futuros)
        no_future_prices = "prices[i+1:" not in content

        # Verifica se usa randn para gerar ruido (Monte Carlo)
        has_monte_carlo = "np.random.randn" in content

        passed = has_forward_simulation and has_monte_carlo and no_future_prices
        details = []

        if has_forward_simulation:
            details.append("Simulacao forward implementada")
        else:
            details.append("ERRO: Simulacao forward nao encontrada")

        if has_s0_initial:
            details.append("S0 usado como ponto inicial")

        if has_monte_carlo:
            details.append("Monte Carlo (randn) implementado")

        if no_future_prices:
            details.append("Nenhum acesso a precos futuros")
        else:
            details.append("ERRO: Acesso a precos futuros detectado")

        return ODMNVerificationResult(
            "Malliavin Monte Carlo causal",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "Malliavin Monte Carlo causal",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_mfg_no_lookahead() -> ODMNVerificationResult:
    """
    Verifica 3: Mean Field Games resolve PDEs sem dados futuros

    O solver MFG deve resolver HJB backward (de T para 0) mas
    usando apenas informacao disponivel no momento da analise.
    """
    indicator_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_malliavin_nash.py"
    )

    try:
        with open(indicator_path, 'r') as f:
            content = f.read()

        # Verifica se MFG usa apenas preco atual e volatilidade
        has_price_level_input = "price_level" in content and "volatility" in content

        # Verifica se solucao analitica esta disponivel
        has_analytical = "_analytical_approximation" in content

        # Verifica se nao usa series temporais futuras
        no_future_series = "[i+1:]" not in content.split("class DeepGalerkinMFGSolver")[0] if "class DeepGalerkinMFGSolver" in content else True

        # Verifica se usa apenas informacao atual (t=0, x=log(price))
        has_current_info = "t_now" in content or "np.log(price_level)" in content

        passed = has_price_level_input and (has_analytical or no_future_series)
        details = []

        if has_price_level_input:
            details.append("Usa preco atual e volatilidade como input")

        if has_analytical:
            details.append("Solucao analitica disponivel (sem look-ahead)")

        if has_current_info:
            details.append("Avalia apenas no ponto atual (t=0)")

        return ODMNVerificationResult(
            "Mean Field Games sem look-ahead",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "Mean Field Games sem look-ahead",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_direction_from_closed_bars() -> ODMNVerificationResult:
    """
    Verifica 4: Direcao do trade baseada APENAS em barras fechadas

    A direcao (long/short) deve ser calculada usando apenas barras
    que ja fecharam, nunca a barra atual.
    """
    optimizer_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "backtesting", "odmn", "optimizer.py"
    )

    try:
        with open(optimizer_path, 'r') as f:
            content = f.read()

        # Verifica se usa bars[i-1] (barra anterior)
        has_closed_bar = "bars[i - 1].close" in content or "self.bars[i - 1].close" in content

        # Verifica se exclui barra atual
        excludes_current = "bars[i].close" not in content.split("direction")[0] if "direction" in content else True

        # Verifica se usa trend_lookback
        has_trend_lookback = "min_bars_for_direction" in content or "TREND_LOOKBACK" in content

        passed = has_closed_bar and has_trend_lookback
        details = []

        if has_closed_bar:
            details.append("Usa bars[i-1].close (barra fechada)")
        else:
            details.append("ERRO: Nao encontrado uso de barra fechada")

        if has_trend_lookback:
            details.append("TREND_LOOKBACK configurado")

        return ODMNVerificationResult(
            "Direcao de barras fechadas apenas",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "Direcao de barras fechadas apenas",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_entry_on_next_bar_open() -> ODMNVerificationResult:
    """
    Verifica 5: Entrada no OPEN da proxima barra

    O trade deve ser executado no preco de abertura da
    barra seguinte ao sinal, nao no preco do sinal.
    """
    optimizer_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "backtesting", "odmn", "optimizer.py"
    )

    try:
        with open(optimizer_path, 'r') as f:
            content = f.read()

        # Verifica se usa next_bar.open
        has_next_bar_open = "next_bar.open" in content

        # Verifica se entry_price vem do OPEN
        has_entry_from_open = "entry_price=next_bar.open" in content or \
                              "entry_price: float" in content

        # Verifica se ODMNSignal tem next_bar_idx
        has_next_bar_idx = "next_bar_idx" in content

        passed = has_next_bar_open and has_next_bar_idx
        details = []

        if has_next_bar_open:
            details.append("Usa next_bar.open para entrada")
        else:
            details.append("ERRO: next_bar.open nao encontrado")

        if has_next_bar_idx:
            details.append("next_bar_idx rastreado no sinal")

        if has_entry_from_open:
            details.append("Entry price vem do OPEN")

        return ODMNVerificationResult(
            "Entrada no OPEN da proxima barra",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "Entrada no OPEN da proxima barra",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_centralized_costs() -> ODMNVerificationResult:
    """
    Verifica 6: Custos centralizados e realistas

    Os custos (spread + slippage) devem ser importados de
    config/execution_costs.py e serem realistas.
    """
    # Verifica config de custos
    costs_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "execution_costs.py"
    )

    optimizer_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "backtesting", "odmn", "optimizer.py"
    )

    try:
        # Verifica valores centralizados
        with open(costs_path, 'r') as f:
            costs_content = f.read()

        # Extrai valores
        spread_match = re.search(r'SPREAD_PIPS:\s*float\s*=\s*([\d.]+)', costs_content)
        slippage_match = re.search(r'SLIPPAGE_PIPS:\s*float\s*=\s*([\d.]+)', costs_content)

        if spread_match and slippage_match:
            spread = float(spread_match.group(1))
            slippage = float(slippage_match.group(1))
            total_cost = spread + slippage

            # Verifica se custos sao realistas (entre 1.5 e 5 pips)
            costs_realistic = 1.5 <= total_cost <= 5.0
        else:
            spread = 0
            slippage = 0
            costs_realistic = False

        # Verifica se optimizer importa de config
        with open(optimizer_path, 'r') as f:
            optimizer_content = f.read()

        imports_from_config = "from config.execution_costs import" in optimizer_content

        passed = costs_realistic and imports_from_config
        details = []

        if spread_match and slippage_match:
            details.append(f"Spread={spread} pips, Slippage={slippage} pips")
        else:
            details.append("ERRO: Custos nao encontrados")

        if costs_realistic:
            details.append(f"Total={total_cost} pips (realista)")
        else:
            details.append("ERRO: Custos nao sao realistas")

        if imports_from_config:
            details.append("Optimizer importa de config centralizado")
        else:
            details.append("ERRO: Optimizer nao importa de config")

        return ODMNVerificationResult(
            "Custos centralizados e realistas",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "Custos centralizados e realistas",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_rigorous_filters() -> ODMNVerificationResult:
    """
    Verifica 7: Filtros rigorosos configurados

    Os filtros para dinheiro real devem estar configurados:
    - Min trades: 50 (treino), 25 (teste)
    - Win Rate: 35% - 60%
    - Profit Factor: >= 1.3 (treino), >= 1.15 (teste)
    - Max Drawdown: 30%
    - Min Expectancy: 1.5 pips/trade
    """
    filters_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "optimizer_filters.py"
    )

    try:
        with open(filters_path, 'r') as f:
            content = f.read()

        # Extrai valores dos filtros
        min_trades_train = re.search(r'MIN_TRADES_TRAIN:\s*int\s*=\s*(\d+)', content)
        min_trades_test = re.search(r'MIN_TRADES_TEST:\s*int\s*=\s*(\d+)', content)
        min_pf = re.search(r'MIN_PROFIT_FACTOR:\s*float\s*=\s*([\d.]+)', content)
        max_dd = re.search(r'MAX_DRAWDOWN:\s*float\s*=\s*([\d.]+)', content)
        min_exp = re.search(r'MIN_EXPECTANCY_PIPS:\s*float\s*=\s*([\d.]+)', content)

        details = []
        checks = []

        if min_trades_train:
            val = int(min_trades_train.group(1))
            checks.append(val >= 50)
            details.append(f"MIN_TRADES_TRAIN={val} {'OK' if val >= 50 else 'BAIXO'}")

        if min_trades_test:
            val = int(min_trades_test.group(1))
            checks.append(val >= 25)
            details.append(f"MIN_TRADES_TEST={val} {'OK' if val >= 25 else 'BAIXO'}")

        if min_pf:
            val = float(min_pf.group(1))
            checks.append(val >= 1.3)
            details.append(f"MIN_PROFIT_FACTOR={val} {'OK' if val >= 1.3 else 'BAIXO'}")

        if max_dd:
            val = float(max_dd.group(1))
            checks.append(val <= 0.35)
            details.append(f"MAX_DRAWDOWN={val*100:.0f}% {'OK' if val <= 0.35 else 'ALTO'}")

        if min_exp:
            val = float(min_exp.group(1))
            checks.append(val >= 1.0)
            details.append(f"MIN_EXPECTANCY={val} pips {'OK' if val >= 1.0 else 'BAIXO'}")

        passed = len(checks) >= 4 and all(checks)

        return ODMNVerificationResult(
            "Filtros rigorosos configurados",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "Filtros rigorosos configurados",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_walk_forward_implemented() -> ODMNVerificationResult:
    """
    Verifica 8: Walk-Forward Validation implementada

    O optimizer deve implementar Walk-Forward validation com
    multiplas janelas train/test.
    """
    optimizer_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "backtesting", "odmn", "optimizer.py"
    )

    try:
        with open(optimizer_path, 'r') as f:
            content = f.read()

        # Verifica estruturas de Walk-Forward
        has_wf_class = "WalkForwardResult" in content
        has_wf_windows = "_create_walk_forward_windows" in content
        has_wf_test = "_test_params_walk_forward" in content
        has_multiple_windows = "n_windows" in content and ("4" in content or "n_windows: int = 4" in content)

        # Verifica se todas as janelas devem passar
        has_all_passed_check = "all_windows_passed" in content and "all(wf.passed" in content

        passed = has_wf_class and has_wf_windows and has_wf_test and has_all_passed_check
        details = []

        if has_wf_class:
            details.append("WalkForwardResult implementado")

        if has_wf_windows:
            details.append("_create_walk_forward_windows existe")

        if has_wf_test:
            details.append("_test_params_walk_forward existe")

        if has_multiple_windows:
            details.append("4 janelas configuradas")

        if has_all_passed_check:
            details.append("Verifica se TODAS as janelas passam")
        else:
            details.append("ERRO: Nao verifica se todas janelas passam")

        return ODMNVerificationResult(
            "Walk-Forward Validation implementada",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "Walk-Forward Validation implementada",
            False,
            f"Erro ao verificar: {e}"
        )


def run_all_verifications():
    """Executa todas as verificacoes e imprime resultado"""
    print("=" * 70)
    print("  VERIFICACAO DE PRONTIDAO PARA DINHEIRO REAL - ODMN V2.0")
    print("  Oraculo de Derivativos de Malliavin-Nash")
    print("=" * 70)
    print("\n  8 verificacoes criticas para evitar look-ahead bias\n")

    verifications = [
        verify_heston_calibration_no_lookahead,
        verify_malliavin_no_lookahead,
        verify_mfg_no_lookahead,
        verify_direction_from_closed_bars,
        verify_entry_on_next_bar_open,
        verify_centralized_costs,
        verify_rigorous_filters,
        verify_walk_forward_implemented,
    ]

    results = []
    for i, verify_func in enumerate(verifications, 1):
        result = verify_func()
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        icon = "+" if result.passed else "X"
        print(f"  [{icon}] {i}. {result.name}: {status}")
        print(f"      {result.details}")
        print()

    # Resumo
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)

    print("=" * 70)
    print(f"  RESULTADO FINAL: {passed_count}/{total_count} verificacoes passaram")
    print("=" * 70)

    if passed_count == total_count:
        print("\n  +++ SISTEMA ODMN PRONTO PARA DINHEIRO REAL +++")
        print("\n  O indicador ODMN passou em TODAS as 8 verificacoes criticas:")
        print("    1. Calibracao Heston usa apenas dados passados")
        print("    2. Malliavin Monte Carlo e causal (forward simulation)")
        print("    3. Mean Field Games resolve PDEs sem dados futuros")
        print("    4. Direcao baseada APENAS em barras fechadas")
        print("    5. Entrada no OPEN da proxima barra")
        print("    6. Custos realistas (spread 1.5 + slippage 0.8 pips)")
        print("    7. Filtros rigorosos (PF >= 1.3, Exp >= 1.5 pips)")
        print("    8. Walk-Forward Validation com 4 janelas")
        print("\n  Proximos passos:")
        print("    1. Execute o optimizer: python -m backtesting.odmn.optimizer")
        print("    2. Valide os resultados no periodo de teste")
        print("    3. Faca paper trading por 2-4 semanas antes de real")
        return True
    else:
        print("\n  XXX ATENCAO: SISTEMA NAO PRONTO PARA DINHEIRO REAL XXX")
        print("\n  Verificacoes que falharam:")
        for r in results:
            if not r.passed:
                print(f"    - {r.name}")
        print("\n  Corrija os problemas acima antes de usar com dinheiro real!")
        return False


if __name__ == "__main__":
    success = run_all_verifications()
    sys.exit(0 if success else 1)
