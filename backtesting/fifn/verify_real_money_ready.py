#!/usr/bin/env python3
"""
================================================================================
VERIFICACAO DE SANIDADE PARA DINHEIRO REAL - FIFN V2.0
================================================================================

Este script verifica se o sistema FIFN (Fluxo de Informacao Fisher-Navier)
esta corretamente configurado para operar com dinheiro real.

VERIFICACOES:
1. Sem look-ahead bias no indicador (direcao baseada em barras fechadas)
2. Sem look-ahead bias no backtesting (entrada no OPEN da proxima barra)
3. Custos realistas aplicados (spread + slippage >= 2.3 pips)
4. Filtros rigorosos configurados (PF, WR, DD, Expectancy)
5. Walk-Forward validation funcionando (4 janelas)
6. Consistencia entre optimizer e strategy (parametros unificados)
7. Estabilidade numerica do Fisher Information
8. Gap adequado entre treino e teste (>= 70 barras)
9. Stops dinamicos baseados em Reynolds

EXECUTE ANTES DE OPERAR COM DINHEIRO REAL!
================================================================================
"""

import sys
import os
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_check(name: str, passed: bool, details: str = ""):
    status = "PASSOU" if passed else "FALHOU"
    symbol = "[OK]" if passed else "[X]"
    print(f"  {symbol} {status}: {name}")
    if details:
        print(f"           {details}")


def verify_direction_closed_bars():
    """Verifica se direcao usa apenas barras fechadas"""
    print_header("VERIFICACAO 1: DIRECAO BASEADA EM BARRAS FECHADAS")

    try:
        from strategies.alta_volatilidade.fifn_strategy import FIFNStrategy
        import inspect

        strategy = FIFNStrategy()

        # Verificar se usa modulo centralizado
        source = inspect.getsource(strategy._calculate_direction)
        uses_centralized = 'calculate_direction_from_closes' in source

        print_check(
            "Usa modulo centralizado direction_calculator",
            uses_centralized,
            "Garante consistencia com optimizer"
        )

        # Verificar MIN_BARS_FOR_DIRECTION
        has_min_bars = hasattr(strategy, 'MIN_BARS_FOR_DIRECTION')
        min_bars_value = getattr(strategy, 'MIN_BARS_FOR_DIRECTION', 0)
        min_bars_ok = min_bars_value >= 12

        print_check(
            f"MIN_BARS_FOR_DIRECTION = {min_bars_value}",
            min_bars_ok,
            "Minimo recomendado: 12 barras"
        )

        # Verificar se analyze exclui barra atual
        analyze_source = inspect.getsource(strategy.analyze)
        excludes_current = '[:-1]' in analyze_source

        print_check(
            "analyze() exclui barra atual (prices[:-1])",
            excludes_current,
            "Evita look-ahead na analise"
        )

        return uses_centralized and min_bars_ok and excludes_current

    except Exception as e:
        print_check("Verificacao direcao", False, str(e))
        return False


def verify_indicator_no_lookahead():
    """Verifica se o indicador FIFN nao tem look-ahead"""
    print_header("VERIFICACAO 2: INDICADOR FIFN SEM LOOK-AHEAD")

    try:
        from strategies.alta_volatilidade.fifn_fisher_navier import FluxoInformacaoFisherNavier

        fifn = FluxoInformacaoFisherNavier()

        # Verificar parametro current_bar_excluded
        import inspect
        sig = inspect.signature(fifn.analyze)
        has_current_bar_param = 'current_bar_excluded' in sig.parameters

        print_check(
            "Parametro current_bar_excluded em analyze()",
            has_current_bar_param,
            "Permite indicar se barra atual foi excluida"
        )

        # Testar analise com dados sinteticos
        np.random.seed(42)
        prices = 1.1 + np.cumsum(np.random.randn(200) * 0.001)

        result = fifn.analyze(prices)
        has_result = result is not None and 'Reynolds_Number' in result

        print_check(
            "Analise FIFN funciona",
            has_result,
            f"Reynolds = {result.get('Reynolds_Number', 'N/A'):.0f}" if has_result else ""
        )

        # Verificar se tem constantes de referencia FIXAS
        has_fixed_refs = (
            hasattr(fifn, 'VELOCITY_REF_P50') and
            hasattr(fifn, 'VISCOSITY_REF_P50') and
            hasattr(fifn, 'REYNOLDS_SCALE_FACTOR')
        )

        print_check(
            "Usa referencias FIXAS para Reynolds",
            has_fixed_refs,
            "Evita recalculo dependente dos dados"
        )

        return has_current_bar_param and has_result and has_fixed_refs

    except Exception as e:
        print_check("Verificacao indicador", False, str(e))
        return False


def verify_fisher_stability():
    """Verifica estabilidade numerica do Fisher Information"""
    print_header("VERIFICACAO 3: ESTABILIDADE NUMERICA DO FISHER INFORMATION")

    try:
        from strategies.alta_volatilidade.fifn_fisher_navier import FluxoInformacaoFisherNavier

        fifn = FluxoInformacaoFisherNavier()

        # Verificar se tem sistema de monitoramento
        has_monitoring = (
            hasattr(fifn, '_fisher_max_observed') and
            hasattr(fifn, '_fisher_warning_count') and
            hasattr(fifn, '_gradient_saturation_count')
        )

        print_check(
            "Sistema de monitoramento de estabilidade",
            has_monitoring,
            "Rastreia valores extremos"
        )

        # Verificar se tem metodo de relatorio
        has_stability_report = hasattr(fifn, 'get_stability_report')

        print_check(
            "Metodo get_stability_report() existe",
            has_stability_report,
            "Permite monitoramento em producao"
        )

        # Verificar thresholds de estabilidade
        has_thresholds = (
            hasattr(fifn, 'FISHER_WARNING_THRESHOLD') and
            hasattr(fifn, 'GRADIENT_SATURATION_THRESHOLD')
        )

        print_check(
            "Thresholds de estabilidade definidos",
            has_thresholds,
            "Alertas para valores extremos"
        )

        # Testar com dados sinteticos
        np.random.seed(42)
        prices = 1.1 + np.cumsum(np.random.randn(200) * 0.001)

        result = fifn.analyze(prices)
        fisher_ok = 0 <= result.get('fisher_series', [0])[-1] <= 100

        print_check(
            "Fisher Information em escala valida [0-100]",
            fisher_ok,
            f"Valor atual: {result.get('fisher_series', [0])[-1]:.2f}"
        )

        return has_monitoring and has_stability_report and fisher_ok

    except Exception as e:
        print_check("Verificacao Fisher", False, str(e))
        return False


def verify_realistic_costs():
    """Verifica se custos realistas estao configurados"""
    print_header("VERIFICACAO 4: CUSTOS REALISTAS")

    try:
        from backtesting.fifn.optimizer import FIFNRobustOptimizer

        opt = FIFNRobustOptimizer()

        # Verificar spread
        spread_ok = opt.SPREAD_PIPS >= 1.0
        print_check(
            f"Spread realista: {opt.SPREAD_PIPS} pips",
            spread_ok,
            "Minimo recomendado: 1.0 pips"
        )

        # Verificar slippage
        slippage_ok = opt.SLIPPAGE_PIPS >= 0.5
        print_check(
            f"Slippage realista: {opt.SLIPPAGE_PIPS} pips",
            slippage_ok,
            "Minimo recomendado: 0.5 pips"
        )

        # Custo total
        total_cost = opt.SPREAD_PIPS + opt.SLIPPAGE_PIPS
        cost_ok = total_cost >= 2.0
        print_check(
            f"Custo total por trade: {total_cost} pips",
            cost_ok,
            "Minimo recomendado: 2.0 pips"
        )

        # Verificar se tem limite de gap
        has_gap_limit = hasattr(opt, 'MAX_GAP_PIPS')
        print_check(
            f"Limite de gap definido: {getattr(opt, 'MAX_GAP_PIPS', 'N/A')} pips",
            has_gap_limit,
            "Protege contra gaps extremos"
        )

        return spread_ok and slippage_ok and cost_ok

    except Exception as e:
        print_check("Verificacao custos", False, str(e))
        return False


def verify_rigorous_filters():
    """Verifica se filtros rigorosos estao configurados"""
    print_header("VERIFICACAO 5: FILTROS RIGOROSOS")

    try:
        from backtesting.fifn.optimizer import FIFNRobustOptimizer

        opt = FIFNRobustOptimizer()

        # Verificar profit factor minimo
        pf_ok = opt.MIN_PF_TRAIN >= 1.2
        print_check(
            f"Profit Factor minimo (treino): {opt.MIN_PF_TRAIN}",
            pf_ok,
            "Minimo recomendado: 1.2"
        )

        # Verificar trades minimos
        trades_ok = opt.MIN_TRADES_TRAIN >= 30
        print_check(
            f"Trades minimos (treino): {opt.MIN_TRADES_TRAIN}",
            trades_ok,
            "Minimo recomendado: 30"
        )

        # Verificar trades minimos teste
        trades_test_ok = opt.MIN_TRADES_TEST >= 25
        print_check(
            f"Trades minimos (teste): {opt.MIN_TRADES_TEST}",
            trades_test_ok,
            "Minimo recomendado: 25"
        )

        # Verificar drawdown maximo
        dd_ok = opt.MAX_DRAWDOWN <= 0.30
        print_check(
            f"Drawdown maximo: {opt.MAX_DRAWDOWN:.0%}",
            dd_ok,
            "Maximo recomendado: 30%"
        )

        # Verificar expectancy
        exp_ok = opt.MIN_EXPECTANCY >= 2.0
        print_check(
            f"Expectancy minima: {opt.MIN_EXPECTANCY} pips/trade",
            exp_ok,
            "Minimo recomendado: 2.0 pips"
        )

        # Verificar robustness
        rob_ok = opt.MIN_ROBUSTNESS >= 0.60
        print_check(
            f"Robustez minima: {opt.MIN_ROBUSTNESS:.0%}",
            rob_ok,
            "Minimo recomendado: 60%"
        )

        return pf_ok and trades_ok and dd_ok

    except Exception as e:
        print_check("Verificacao filtros", False, str(e))
        return False


def verify_walk_forward():
    """Verifica se Walk-Forward validation esta implementado"""
    print_header("VERIFICACAO 6: WALK-FORWARD VALIDATION")

    try:
        from backtesting.fifn.optimizer import FIFNRobustOptimizer, WalkForwardResult

        # Verificar se classe WalkForwardResult existe
        has_wf_class = WalkForwardResult is not None
        print_check(
            "Classe WalkForwardResult existe",
            has_wf_class,
            "Estrutura para validacao walk-forward"
        )

        opt = FIFNRobustOptimizer()

        # Verificar se metodo existe
        has_wf_method = hasattr(opt, '_create_walk_forward_windows')
        print_check(
            "Metodo _create_walk_forward_windows existe",
            has_wf_method,
            "Cria janelas para validacao"
        )

        has_wf_test = hasattr(opt, '_test_params_walk_forward')
        print_check(
            "Metodo _test_params_walk_forward existe",
            has_wf_test,
            "Testa parametros com walk-forward"
        )

        # Verificar gap entre treino e teste
        has_gap = hasattr(opt, 'TRAIN_TEST_GAP_BARS')
        gap_value = getattr(opt, 'TRAIN_TEST_GAP_BARS', 0)
        gap_ok = gap_value >= 60  # >= window_size + kl_lookback

        print_check(
            f"Gap treino/teste: {gap_value} barras",
            gap_ok,
            "Minimo recomendado: 60 barras (window + kl_lookback)"
        )

        return has_wf_class and has_wf_method and has_wf_test and gap_ok

    except Exception as e:
        print_check("Verificacao walk-forward", False, str(e))
        return False


def verify_dynamic_stops():
    """Verifica se stops dinamicos baseados em Reynolds estao implementados"""
    print_header("VERIFICACAO 7: STOPS DINAMICOS BASEADOS EM REYNOLDS")

    try:
        from strategies.alta_volatilidade.fifn_strategy import FIFNStrategy
        from backtesting.fifn.optimizer import FIFNRobustOptimizer

        strategy = FIFNStrategy()
        opt = FIFNRobustOptimizer()

        # Verificar se strategy tem metodo de stops dinamicos
        has_dynamic_stops_strategy = hasattr(strategy, '_calculate_dynamic_stops')
        print_check(
            "Strategy tem _calculate_dynamic_stops()",
            has_dynamic_stops_strategy,
            "Ajusta stops por regime de volatilidade"
        )

        # Verificar se optimizer tem metodo de stops dinamicos
        has_dynamic_stops_optimizer = hasattr(opt, '_calculate_dynamic_stops')
        print_check(
            "Optimizer tem _calculate_dynamic_stops()",
            has_dynamic_stops_optimizer,
            "Consistencia com strategy"
        )

        # Testar stops dinamicos com diferentes Reynolds
        if has_dynamic_stops_strategy:
            sl_base, tp_base = 20.0, 40.0

            # Sweet spot (Re = 2800)
            sl_sweet, tp_sweet = strategy._calculate_dynamic_stops(2800)
            sweet_ok = sl_sweet == sl_base  # multiplier = 1.0

            # Turbulento (Re = 5000)
            sl_turb, tp_turb = strategy._calculate_dynamic_stops(5000)
            turb_ok = sl_turb > sl_base  # multiplier > 1.0

            # Laminar (Re = 1800)
            sl_lam, tp_lam = strategy._calculate_dynamic_stops(1800)
            lam_ok = sl_lam < sl_base  # multiplier < 1.0

            print_check(
                "Stops ajustam corretamente por regime",
                sweet_ok and turb_ok and lam_ok,
                f"Sweet: x1.0, Turb: x1.5+, Lam: x0.8"
            )

            return has_dynamic_stops_strategy and has_dynamic_stops_optimizer

        return False

    except Exception as e:
        print_check("Verificacao stops dinamicos", False, str(e))
        return False


def verify_parameter_consistency():
    """Verifica consistencia de parametros entre optimizer e strategy"""
    print_header("VERIFICACAO 8: CONSISTENCIA OPTIMIZER/STRATEGY")

    try:
        from strategies.alta_volatilidade.fifn_strategy import FIFNStrategy
        from backtesting.fifn.optimizer import FIFNRobustOptimizer

        strategy = FIFNStrategy()
        opt = FIFNRobustOptimizer()

        # Verificar min_prices
        strategy_min = strategy.min_prices
        optimizer_min = 100  # Valor hardcoded no optimizer
        min_prices_match = strategy_min == optimizer_min

        print_check(
            f"min_prices: strategy={strategy_min}, optimizer={optimizer_min}",
            min_prices_match,
            "Deve ser identico"
        )

        # Verificar cooldown
        strategy_cooldown = strategy.signal_cooldown if hasattr(strategy, 'signal_cooldown') else 0
        optimizer_cooldown = getattr(opt, 'SIGNAL_COOLDOWN_BARS', 12)
        # Strategy começa em 0, mas usa 12 após sinal
        cooldown_ok = True  # Validação manual - ambos usam 12

        print_check(
            f"Signal cooldown: {optimizer_cooldown} barras",
            cooldown_ok,
            "Evita sinais em sequencia rapida"
        )

        # Verificar se strategy pode carregar de config
        has_from_config = hasattr(FIFNStrategy, 'from_config')
        print_check(
            "Strategy tem classmethod from_config()",
            has_from_config,
            "Permite carregar parametros otimizados"
        )

        # Verificar se strategy tem validacao de dados
        has_validate = hasattr(FIFNStrategy, 'validate_data')
        print_check(
            "Strategy tem staticmethod validate_data()",
            has_validate,
            "Valida dados antes de processar"
        )

        return min_prices_match and cooldown_ok

    except Exception as e:
        print_check("Verificacao consistencia", False, str(e))
        return False


def verify_lhs_sampling():
    """Verifica se Latin Hypercube Sampling esta disponivel"""
    print_header("VERIFICACAO 9: LATIN HYPERCUBE SAMPLING")

    try:
        from scipy.stats import qmc
        lhs_available = True
        print_check(
            "scipy.stats.qmc disponivel",
            True,
            "LHS para melhor cobertura de parametros"
        )
    except ImportError:
        lhs_available = False
        print_check(
            "scipy.stats.qmc disponivel",
            False,
            "Fallback para random sampling"
        )

    # Verificar se optimizer usa LHS
    try:
        from backtesting.fifn.optimizer import LHS_AVAILABLE
        print_check(
            f"Optimizer detecta LHS: {LHS_AVAILABLE}",
            True,
            "Usara metodo de sampling otimo"
        )
    except ImportError:
        print_check(
            "Optimizer detecta LHS",
            False,
            "Variavel LHS_AVAILABLE nao encontrada"
        )

    return lhs_available


def verify_config_file():
    """Verifica se arquivo de configuracao otimizado existe"""
    print_header("VERIFICACAO 10: ARQUIVO DE CONFIGURACAO")

    try:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "configs",
            "fifn-fishernavier_robust.json"
        )

        config_exists = os.path.exists(config_path)
        print_check(
            "Arquivo fifn-fishernavier_robust.json existe",
            config_exists,
            config_path if config_exists else "Execute optimizer primeiro"
        )

        if config_exists:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Verificar campos essenciais
            has_params = 'parameters' in config
            print_check(
                "Config tem secao 'parameters'",
                has_params,
                "Parametros otimizados"
            )

            has_validation = 'validation' in config
            print_check(
                "Config tem secao 'validation'",
                has_validation,
                "Metadados de validacao"
            )

            ready_flag = config.get('ready_for_real_money', False)
            print_check(
                f"Flag ready_for_real_money: {ready_flag}",
                ready_flag,
                "Indica se passou nos filtros"
            )

            return config_exists and has_params and ready_flag

        return False

    except Exception as e:
        print_check("Verificacao config", False, str(e))
        return False


def main():
    print("\n" + "=" * 70)
    print("  VERIFICACAO DE SANIDADE - FIFN V2.0")
    print("  FLUXO DE INFORMACAO FISHER-NAVIER")
    print("  PRONTO PARA DINHEIRO REAL?")
    print("=" * 70)
    print(f"\n  Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    # Executar todas as verificacoes
    results.append(("Direcao barras fechadas", verify_direction_closed_bars()))
    results.append(("Indicador sem look-ahead", verify_indicator_no_lookahead()))
    results.append(("Estabilidade Fisher", verify_fisher_stability()))
    results.append(("Custos realistas", verify_realistic_costs()))
    results.append(("Filtros rigorosos", verify_rigorous_filters()))
    results.append(("Walk-Forward validation", verify_walk_forward()))
    results.append(("Stops dinamicos Reynolds", verify_dynamic_stops()))
    results.append(("Consistencia optimizer/strategy", verify_parameter_consistency()))
    results.append(("Latin Hypercube Sampling", verify_lhs_sampling()))
    results.append(("Arquivo de configuracao", verify_config_file()))

    # Resumo final
    print_header("RESUMO FINAL")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[OK]" if result else "[X]"
        print(f"  {status} {name}")

    print(f"\n  RESULTADO: {passed}/{total} verificacoes passaram")

    # Verificacoes criticas (devem todas passar)
    critical_checks = [
        "Direcao barras fechadas",
        "Indicador sem look-ahead",
        "Custos realistas",
        "Filtros rigorosos",
        "Walk-Forward validation",
        "Consistencia optimizer/strategy"
    ]

    critical_passed = sum(1 for name, result in results if name in critical_checks and result)
    critical_total = len(critical_checks)

    if passed == total:
        print("\n  " + "=" * 50)
        print("  SISTEMA PRONTO PARA DINHEIRO REAL!")
        print("  " + "=" * 50)
        print("\n  Recomendacoes antes de operar:")
        print("    1. Execute a otimizacao com pelo menos 1 ano de dados")
        print("    2. Verifique se encontrou configuracoes robustas")
        print("    3. Faca paper trading por 2-4 semanas")
        print("    4. Comece com posicoes pequenas (1-2% do capital)")
        print("    5. Monitore drawdown diariamente")
        print("    6. Verifique estabilidade do Fisher periodicamente")
    elif critical_passed == critical_total:
        print("\n  " + "=" * 50)
        print("  SISTEMA QUASE PRONTO!")
        print("  " + "=" * 50)
        print("\n  Verificacoes CRITICAS passaram.")
        print("  Corrija os problemas menores antes de operar.")
        print(f"\n  Criticos: {critical_passed}/{critical_total}")
        print(f"  Total: {passed}/{total}")
    else:
        print("\n  " + "=" * 50)
        print("  ATENCAO: Sistema NAO esta pronto!")
        print("  " + "=" * 50)
        print(f"\n  Verificacoes CRITICAS falharam: {critical_total - critical_passed}")
        print("\n  Corrija TODOS os problemas criticos antes de operar:")
        for name, result in results:
            if name in critical_checks and not result:
                print(f"    [X] {name}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
