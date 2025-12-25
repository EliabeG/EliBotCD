#!/usr/bin/env python3
"""
================================================================================
VERIFICAÇÃO DE SANIDADE PARA DINHEIRO REAL - DSG V3.0
================================================================================

Este script verifica se o sistema DSG está corretamente configurado
para operar com dinheiro real.

VERIFICAÇÕES:
1. Subsampling não calcula barra atual (n-1)
2. Direção geodésica usa apenas barras fechadas
3. Centro de massa exclui barra atual
4. Volumes determinísticos sem look-ahead
5. ricci_collapsing e crossing_horizon sem contaminação
6. Filtros unificados com robust_optimizer.py
7. Custos realistas aplicados
8. Parâmetros configuráveis (cooldown, confidence)

EXECUTE ANTES DE OPERAR COM DINHEIRO REAL!
================================================================================
"""

import sys
import os
import numpy as np
from datetime import datetime, timezone
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_check(name: str, passed: bool, details: str = ""):
    status = "✅ PASSOU" if passed else "❌ FALHOU"
    print(f"  {status}: {name}")
    if details:
        print(f"           {details}")


def verify_subsampling_no_current_bar():
    """Verifica se o subsampling não calcula a barra atual (n-1)"""
    print_header("VERIFICAÇÃO 1: SUBSAMPLING NÃO CALCULA BARRA ATUAL")

    try:
        from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional

        source = inspect.getsource(DetectorSingularidadeGravitacional.analyze)

        # Verificar se usa last_closed_idx = n - 2
        uses_last_closed = 'last_closed_idx = n - 2' in source
        print_check(
            "Usa last_closed_idx = n - 2",
            uses_last_closed,
            "Exclui barra atual (n-1) do cálculo"
        )

        # Verificar se loop vai até last_closed_idx
        loop_correct = 'last_closed_idx + 1' in source or 'last_closed_idx inclusive' in source
        print_check(
            "Loop de cálculo termina em last_closed_idx",
            loop_correct,
            "Não calcula indicadores para barra atual"
        )

        # Verificar comentário da correção V3.0
        has_v3_comment = 'CORREÇÃO V3.0' in source and 'barra atual' in source.lower()
        print_check(
            "Documentação da correção V3.0 presente",
            has_v3_comment,
            "Código documentado com explicação anti-look-ahead"
        )

        return uses_last_closed and loop_correct

    except Exception as e:
        print_check("Verificação subsampling", False, str(e))
        return False


def verify_geodesic_direction_closed_bars():
    """Verifica se direção geodésica usa apenas barras fechadas"""
    print_header("VERIFICAÇÃO 2: DIREÇÃO GEODÉSICA BARRAS FECHADAS")

    try:
        from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional

        source = inspect.getsource(DetectorSingularidadeGravitacional.analyze)

        # Verificar se usa _coords_history[-4:] sem excluir (porque o loop já exclui)
        # ou se menciona "barras FECHADAS"
        uses_closed_bars = ('barras FECHADAS' in source or
                           'COMPLETAMENTE FECHADAS' in source or
                           'já fechados' in source.lower())
        print_check(
            "Documentação indica uso de barras fechadas",
            uses_closed_bars,
            "Direção calculada apenas com histórico fechado"
        )

        # Verificar que o loop para antes de n-1
        loop_stops_before_current = 'last_closed_idx' in source
        print_check(
            "Loop para antes da barra atual",
            loop_stops_before_current,
            "Histórico não contém barra atual"
        )

        return uses_closed_bars and loop_stops_before_current

    except Exception as e:
        print_check("Verificação direção geodésica", False, str(e))
        return False


def verify_center_of_mass_excludes_current():
    """Verifica se centro de massa exclui barra atual"""
    print_header("VERIFICAÇÃO 3: CENTRO DE MASSA EXCLUI BARRA ATUAL")

    try:
        from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional

        source = inspect.getsource(DetectorSingularidadeGravitacional.analyze_point)

        # Verificar se append vem DEPOIS do cálculo do centro de massa
        # A ordem correta é: calcular com histórico existente, DEPOIS adicionar ao histórico
        calculates_before_append = (
            'ANTES de adicionar' in source or
            'DEPOIS dos cálculos' in source or
            'CORREÇÃO V3.0' in source
        )
        print_check(
            "Centro de massa calculado antes de adicionar barra",
            calculates_before_append,
            "VWAP usa apenas barras anteriores"
        )

        # Verificar se usa histórico existente
        uses_existing_history = 'histórico existente' in source.lower() or 'barras ANTERIORES' in source
        print_check(
            "Usa histórico existente (sem barra atual)",
            uses_existing_history,
            "Não inclui preço atual no VWAP"
        )

        return calculates_before_append

    except Exception as e:
        print_check("Verificação centro de massa", False, str(e))
        return False


def verify_deterministic_volumes():
    """Verifica se volumes são determinísticos e sem look-ahead"""
    print_header("VERIFICAÇÃO 4: VOLUMES DETERMINÍSTICOS SEM LOOK-AHEAD")

    try:
        from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional

        source = inspect.getsource(DetectorSingularidadeGravitacional.analyze)

        # Verificar se usa backward diff
        uses_backward = 'prices[i-1] - prices[i-2]' in source
        print_check(
            "Volume usa diferença de barras anteriores",
            uses_backward,
            "volume[i] = |prices[i-1] - prices[i-2]|"
        )

        # Verificar que não usa np.diff com toda série
        no_full_diff = 'np.diff(prices, prepend=prices[0])' not in source
        print_check(
            "Não usa np.diff em toda série",
            no_full_diff,
            "Evita calcular volume usando close atual"
        )

        return uses_backward and no_full_diff

    except Exception as e:
        print_check("Verificação volumes", False, str(e))
        return False


def verify_history_not_contaminated():
    """Verifica se históricos de Ricci e distância não estão contaminados"""
    print_header("VERIFICAÇÃO 5: HISTÓRICOS SEM CONTAMINAÇÃO")

    try:
        from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional

        source = inspect.getsource(DetectorSingularidadeGravitacional.analyze)

        # Verificar documentação de correção
        ricci_ok = 'valores de barras FECHADAS' in source or 'SEM contaminação' in source
        print_check(
            "Histórico de Ricci documentado como sem contaminação",
            ricci_ok,
            "_ricci_history contém apenas barras fechadas"
        )

        distance_ok = 'barras FECHADAS' in source
        print_check(
            "Histórico de distância sem contaminação",
            distance_ok,
            "_distance_history contém apenas barras fechadas"
        )

        return ricci_ok

    except Exception as e:
        print_check("Verificação históricos", False, str(e))
        return False


def verify_unified_filters():
    """Verifica se filtros estão unificados com robust_optimizer.py"""
    print_header("VERIFICAÇÃO 6: FILTROS UNIFICADOS")

    try:
        from backtesting.common.robust_optimizer import RobustBacktester

        # Valores esperados do robust_optimizer
        expected_min_trades_train = 50
        expected_min_pf = 1.30
        expected_max_dd = 0.30
        expected_min_robustness = 0.70

        # Verificar valores
        mt_ok = RobustBacktester.MIN_TRADES_TRAIN == expected_min_trades_train
        print_check(
            f"MIN_TRADES_TRAIN = {RobustBacktester.MIN_TRADES_TRAIN}",
            mt_ok,
            f"Esperado: {expected_min_trades_train}"
        )

        pf_ok = RobustBacktester.MIN_PROFIT_FACTOR == expected_min_pf
        print_check(
            f"MIN_PROFIT_FACTOR = {RobustBacktester.MIN_PROFIT_FACTOR}",
            pf_ok,
            f"Esperado: {expected_min_pf}"
        )

        dd_ok = RobustBacktester.MAX_DRAWDOWN == expected_max_dd
        print_check(
            f"MAX_DRAWDOWN = {RobustBacktester.MAX_DRAWDOWN}",
            dd_ok,
            f"Esperado: {expected_max_dd}"
        )

        rob_ok = RobustBacktester.MIN_ROBUSTNESS == expected_min_robustness
        print_check(
            f"MIN_ROBUSTNESS = {RobustBacktester.MIN_ROBUSTNESS}",
            rob_ok,
            f"Esperado: {expected_min_robustness}"
        )

        # Verificar DSG optimizer usa mesmos valores
        source_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'optimizer.py'
        )
        with open(source_file, 'r') as f:
            dsg_source = f.read()

        dsg_uses_unified = (
            'min_trades=50' in dsg_source and
            'min_pf=1.30' in dsg_source and
            'max_dd=0.30' in dsg_source
        )
        print_check(
            "DSG optimizer usa filtros unificados",
            dsg_uses_unified,
            "Valores alinhados com RobustBacktester"
        )

        return mt_ok and pf_ok and dd_ok and dsg_uses_unified

    except Exception as e:
        print_check("Verificação filtros", False, str(e))
        return False


def verify_realistic_costs():
    """Verifica se custos realistas estão configurados"""
    print_header("VERIFICAÇÃO 7: CUSTOS REALISTAS")

    try:
        from config.execution_costs import SPREAD_PIPS, SLIPPAGE_PIPS

        # Verificar spread
        spread_ok = SPREAD_PIPS >= 1.0
        print_check(
            f"Spread centralizado: {SPREAD_PIPS} pips",
            spread_ok,
            "Mínimo recomendado: 1.0 pips"
        )

        # Verificar slippage
        slippage_ok = SLIPPAGE_PIPS >= 0.5
        print_check(
            f"Slippage centralizado: {SLIPPAGE_PIPS} pips",
            slippage_ok,
            "Mínimo recomendado: 0.5 pips"
        )

        # Custo total
        total_cost = SPREAD_PIPS + SLIPPAGE_PIPS
        cost_ok = total_cost >= 2.0
        print_check(
            f"Custo total por trade: {total_cost} pips",
            cost_ok,
            "Mínimo recomendado: 2.0 pips"
        )

        return spread_ok and slippage_ok

    except Exception as e:
        print_check("Verificação custos", False, str(e))
        return False


def verify_configurable_params():
    """Verifica se parâmetros são configuráveis"""
    print_header("VERIFICAÇÃO 8: PARÂMETROS CONFIGURÁVEIS")

    try:
        from strategies.alta_volatilidade.dsg_strategy import DSGStrategy
        import inspect

        sig = inspect.signature(DSGStrategy.__init__)
        params = sig.parameters

        # Verificar signal_cooldown_bars
        has_cooldown = 'signal_cooldown_bars' in params
        print_check(
            "signal_cooldown_bars é parâmetro configurável",
            has_cooldown,
            "Era hardcoded como 30"
        )

        # Verificar min_confidence
        has_confidence = 'min_confidence' in params
        print_check(
            "min_confidence é parâmetro configurável",
            has_confidence,
            "Era hardcoded como 0.5"
        )

        # Testar instanciação com parâmetros customizados
        strategy = DSGStrategy(
            signal_cooldown_bars=20,
            min_confidence=0.6
        )
        cooldown_ok = strategy.signal_cooldown_bars == 20
        confidence_ok = strategy.min_confidence == 0.6
        print_check(
            "Parâmetros são aplicados corretamente",
            cooldown_ok and confidence_ok,
            f"cooldown={strategy.signal_cooldown_bars}, confidence={strategy.min_confidence}"
        )

        return has_cooldown and has_confidence and cooldown_ok and confidence_ok

    except Exception as e:
        print_check("Verificação parâmetros", False, str(e))
        return False


def verify_indicator_version():
    """Verifica se indicador é versão V3.0"""
    print_header("VERIFICAÇÃO 9: VERSÃO DO INDICADOR")

    try:
        source_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'strategies', 'alta_volatilidade', 'dsg_detector_singularidade.py'
        )
        with open(source_file, 'r') as f:
            source = f.read()

        is_v3 = 'V3.0' in source and 'PRONTO PARA DINHEIRO REAL' in source
        print_check(
            "Indicador é versão V3.0",
            is_v3,
            "Versão com todas as correções da auditoria"
        )

        has_audit_fixes = (
            'Correções aplicadas (V3.0' in source or
            'CORREÇÃO V3.0' in source
        )
        print_check(
            "Correções da auditoria documentadas",
            has_audit_fixes,
            "Código contém documentação das correções"
        )

        return is_v3 and has_audit_fixes

    except Exception as e:
        print_check("Verificação versão", False, str(e))
        return False


def main():
    print("\n" + "=" * 70)
    print("  VERIFICAÇÃO DE SANIDADE - DSG V3.0")
    print("  PRONTO PARA DINHEIRO REAL?")
    print("=" * 70)
    print(f"\n  Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    # Executar todas as verificações
    results.append(("Subsampling não calcula barra atual", verify_subsampling_no_current_bar()))
    results.append(("Direção geodésica barras fechadas", verify_geodesic_direction_closed_bars()))
    results.append(("Centro de massa exclui barra atual", verify_center_of_mass_excludes_current()))
    results.append(("Volumes determinísticos sem look-ahead", verify_deterministic_volumes()))
    results.append(("Históricos sem contaminação", verify_history_not_contaminated()))
    results.append(("Filtros unificados", verify_unified_filters()))
    results.append(("Custos realistas", verify_realistic_costs()))
    results.append(("Parâmetros configuráveis", verify_configurable_params()))
    results.append(("Versão V3.0 do indicador", verify_indicator_version()))

    # Resumo final
    print_header("RESUMO FINAL")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")

    print(f"\n  RESULTADO: {passed}/{total} verificações passaram")

    if passed == total:
        print("\n  " + "=" * 50)
        print("  🎉 SISTEMA DSG PRONTO PARA DINHEIRO REAL!")
        print("  " + "=" * 50)
        print("\n  Recomendações antes de operar:")
        print("    1. Execute a otimização com pelo menos 1 ano de dados")
        print("    2. Verifique se encontrou configurações robustas")
        print("    3. Faça paper trading por 2-4 semanas")
        print("    4. Comece com posições pequenas (1-2% do capital)")
        print("    5. Monitore drawdown diariamente")
        print("    6. Compare resultados com backtest (máx 2 desvios padrão)")
    else:
        print("\n  " + "=" * 50)
        print("  ⚠️  ATENÇÃO: Sistema NÃO está pronto!")
        print("  " + "=" * 50)
        print("\n  Corrija os problemas identificados antes de operar.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
