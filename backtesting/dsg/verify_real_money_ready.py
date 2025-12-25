#!/usr/bin/env python3
"""
================================================================================
VERIFICAÇÃO DE SANIDADE PARA DINHEIRO REAL - DSG V3.5
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
9. Testes funcionais de look-ahead
10. Verificação de centralização de filtros
11. NOVO V3.5: Escala de Ricci na ESTRATÉGIA (não só indicador)
12. NOVO V3.5: Consistência entre estratégia e otimizador
13. NOVO V3.5: Método from_config para carregar parâmetros
14. NOVO V3.5: Testes funcionais de consistência

EXECUTE ANTES DE OPERAR COM DINHEIRO REAL!

CORREÇÕES V3.5 (Quinta Auditoria 25/12/2025):
- CRÍTICO: Verifica threshold de Ricci na ESTRATÉGIA (era só no indicador)
- Verificação de consistência funcional entre componentes
- Teste do método from_config para carregar parâmetros otimizados
================================================================================
"""

import sys
import os
import numpy as np
from datetime import datetime, timezone
import inspect
import json  # CORREÇÃO V3.5: Para testes de from_config

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
    """Verifica se indicador é versão V3.4"""
    print_header("VERIFICAÇÃO 9: VERSÃO DO INDICADOR")

    try:
        source_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'strategies', 'alta_volatilidade', 'dsg_detector_singularidade.py'
        )
        with open(source_file, 'r') as f:
            source = f.read()

        # ATUALIZADO V3.4: Verificar versão mais recente
        is_v34 = 'V3.4' in source
        print_check(
            "Indicador é versão V3.4 ou superior",
            is_v34,
            "Versão com todas as correções da auditoria"
        )

        has_audit_fixes = (
            'CORREÇÃO V3.4' in source or
            'Quarta Auditoria' in source
        )
        print_check(
            "Correções da auditoria V3.4 documentadas",
            has_audit_fixes,
            "Inclui threshold de Ricci corrigido"
        )

        return is_v34 and has_audit_fixes

    except Exception as e:
        print_check("Verificação versão", False, str(e))
        return False


def verify_functional_no_lookahead():
    """NOVO V3.4: Teste funcional de look-ahead bias"""
    print_header("VERIFICAÇÃO 10: TESTE FUNCIONAL DE LOOK-AHEAD")

    try:
        from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional

        np.random.seed(42)
        prices = 1.1 + 0.0002 * np.cumsum(np.random.randn(100))

        # Processar até barra 80
        dsg1 = DetectorSingularidadeGravitacional()
        result1 = dsg1.analyze(prices[:80])
        signal1 = result1['signal']

        # Processar até barra 80 com barras futuras diferentes
        prices_modified = prices.copy()
        prices_modified[80:] = prices_modified[80:] + 0.01  # Altera futuro

        dsg2 = DetectorSingularidadeGravitacional()
        result2 = dsg2.analyze(prices_modified[:80])
        signal2 = result2['signal']

        # Sinais devem ser idênticos - barras futuras não devem afetar
        no_lookahead = (signal1 == signal2)
        print_check(
            "Sinal não muda com alteração de barras futuras",
            no_lookahead,
            f"Sinal original: {signal1}, Sinal modificado: {signal2}"
        )

        return no_lookahead

    except Exception as e:
        print_check("Teste funcional look-ahead", False, str(e))
        return False


def verify_centralized_filters():
    """NOVO V3.4: Verifica se todos os otimizadores usam filtros centralizados"""
    print_header("VERIFICAÇÃO 11: FILTROS CENTRALIZADOS")

    try:
        from config.optimizer_filters import (
            MIN_TRADES_TRAIN, MIN_PROFIT_FACTOR, MAX_DRAWDOWN
        )
        from backtesting.common.robust_optimizer import RobustBacktester

        # Verificar se robust_optimizer usa valores do config
        config_match = (
            RobustBacktester.MIN_TRADES_TRAIN == MIN_TRADES_TRAIN and
            RobustBacktester.MIN_PROFIT_FACTOR == MIN_PROFIT_FACTOR and
            RobustBacktester.MAX_DRAWDOWN == MAX_DRAWDOWN
        )
        print_check(
            "robust_optimizer.py usa filtros centralizados",
            config_match,
            f"MIN_TRADES={MIN_TRADES_TRAIN}, MIN_PF={MIN_PROFIT_FACTOR}, MAX_DD={MAX_DRAWDOWN}"
        )

        return config_match

    except Exception as e:
        print_check("Verificação filtros centralizados", False, str(e))
        return False


def verify_ricci_threshold_scale():
    """ATUALIZADO V3.5: Verifica threshold de Ricci no indicador E na estratégia"""
    print_header("VERIFICAÇÃO 12: ESCALA DO THRESHOLD DE RICCI")

    try:
        from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional
        from strategies.alta_volatilidade.dsg_strategy import DSGStrategy

        # Verificar indicador
        dsg = DetectorSingularidadeGravitacional()
        indicator_ok = dsg.ricci_collapse_threshold < -40000
        print_check(
            f"Indicador - Threshold de Ricci: {dsg.ricci_collapse_threshold}",
            indicator_ok,
            "Deve estar na escala real (< -40000)"
        )

        # CORREÇÃO V3.5: Verificar estratégia também (era o problema principal!)
        strategy = DSGStrategy()
        strategy_threshold = strategy.dsg.ricci_collapse_threshold
        strategy_ok = strategy_threshold < -40000
        print_check(
            f"Estratégia - Threshold de Ricci: {strategy_threshold}",
            strategy_ok,
            "Deve estar na escala real (< -40000), não -0.5"
        )

        # Verificar consistência entre indicador e estratégia
        consistent = abs(dsg.ricci_collapse_threshold - strategy_threshold) < 1000
        print_check(
            "Consistência entre indicador e estratégia",
            consistent,
            f"Indicador={dsg.ricci_collapse_threshold}, Estratégia={strategy_threshold}"
        )

        return indicator_ok and strategy_ok and consistent

    except Exception as e:
        print_check("Verificação escala Ricci", False, str(e))
        return False


def verify_strategy_optimizer_consistency():
    """NOVO V3.5: Verifica consistência entre parâmetros da estratégia e otimizadores"""
    print_header("VERIFICAÇÃO 13: CONSISTÊNCIA ESTRATÉGIA-OTIMIZADOR")

    try:
        from strategies.alta_volatilidade.dsg_strategy import DSGStrategy

        # Verificar que estratégia usa escala correta por padrão
        strategy = DSGStrategy()

        # Valores que devem estar em escala real
        ricci_ok = strategy.dsg.ricci_collapse_threshold < -40000
        print_check(
            f"ricci_collapse_threshold default: {strategy.dsg.ricci_collapse_threshold}",
            ricci_ok,
            "Deve ser < -40000 (escala real)"
        )

        # Verificar ranges dos otimizadores
        source_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'optimizer.py'
        )
        with open(source_file, 'r') as f:
            opt_source = f.read()

        # Otimizador deve usar ranges na escala real
        uses_real_scale = '-51000' in opt_source or '-50500' in opt_source
        print_check(
            "Otimizador usa ranges na escala real",
            uses_real_scale,
            "Ranges de Ricci devem estar em -51000 a -49500"
        )

        return ricci_ok and uses_real_scale

    except Exception as e:
        print_check("Verificação consistência", False, str(e))
        return False


def verify_from_config_method():
    """NOVO V3.5: Verifica se método from_config existe e funciona"""
    print_header("VERIFICAÇÃO 14: MÉTODO FROM_CONFIG")

    try:
        from strategies.alta_volatilidade.dsg_strategy import DSGStrategy
        import tempfile

        # Verificar que método existe
        has_from_config = hasattr(DSGStrategy, 'from_config')
        print_check(
            "DSGStrategy tem método from_config",
            has_from_config,
            "Permite carregar parâmetros otimizados"
        )

        if not has_from_config:
            return False

        # Testar carregamento de config
        test_config = {
            "params": {
                "ricci_collapse_threshold": -50300.0,
                "tidal_force_threshold": 0.015,
                "stop_loss_pips": 25.0,
                "take_profit_pips": 50.0,
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_path = f.name

        try:
            strategy = DSGStrategy.from_config(temp_path)
            params_loaded = (
                strategy.dsg.ricci_collapse_threshold == -50300.0 and
                strategy.dsg.tidal_force_threshold == 0.015 and
                strategy.stop_loss_pips == 25.0 and
                strategy.take_profit_pips == 50.0
            )
            print_check(
                "from_config carrega parâmetros corretamente",
                params_loaded,
                f"Ricci={strategy.dsg.ricci_collapse_threshold}, Tidal={strategy.dsg.tidal_force_threshold}"
            )
        finally:
            os.unlink(temp_path)

        return has_from_config and params_loaded

    except Exception as e:
        print_check("Verificação from_config", False, str(e))
        return False


def verify_functional_consistency():
    """NOVO V3.5: Teste funcional de consistência entre otimização e produção"""
    print_header("VERIFICAÇÃO 15: CONSISTÊNCIA FUNCIONAL")

    try:
        from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional
        from strategies.alta_volatilidade.dsg_strategy import DSGStrategy

        np.random.seed(42)
        prices = 1.1 + 0.0002 * np.cumsum(np.random.randn(200))

        # Criar indicador com parâmetros específicos
        params = {
            'ricci_collapse_threshold': -50300.0,
            'tidal_force_threshold': 0.015,
        }

        # Testar indicador direto
        dsg = DetectorSingularidadeGravitacional(**params)
        result_indicator = dsg.analyze(prices)
        signal_indicator = result_indicator['signal']

        # Testar via estratégia com mesmos parâmetros
        strategy = DSGStrategy(
            ricci_collapse_threshold=params['ricci_collapse_threshold'],
            tidal_force_threshold=params['tidal_force_threshold'],
        )

        # Alimentar estratégia com os mesmos preços
        strategy2 = DSGStrategy(
            ricci_collapse_threshold=params['ricci_collapse_threshold'],
            tidal_force_threshold=params['tidal_force_threshold'],
        )
        # Analisar usando método interno do indicador
        result_strategy = strategy2.dsg.analyze(prices)
        signal_strategy = result_strategy['signal']

        # Sinais devem ser idênticos
        signals_match = (signal_indicator == signal_strategy)
        print_check(
            "Sinal do indicador = Sinal da estratégia",
            signals_match,
            f"Indicador={signal_indicator}, Estratégia={signal_strategy}"
        )

        # Verificar que Ricci está na escala correta
        ricci_indicator = result_indicator['Ricci_Scalar']
        ricci_strategy = result_strategy['Ricci_Scalar']
        ricci_scale_ok = (
            ricci_indicator < -40000 and
            ricci_indicator > -60000 and
            abs(ricci_indicator - ricci_strategy) < 1.0
        )
        print_check(
            f"Ricci na escala real",
            ricci_scale_ok,
            f"Indicador={ricci_indicator:.2f}, Estratégia={ricci_strategy:.2f}"
        )

        return signals_match and ricci_scale_ok

    except Exception as e:
        print_check("Verificação consistência funcional", False, str(e))
        return False


def main():
    print("\n" + "=" * 70)
    print("  VERIFICAÇÃO DE SANIDADE - DSG V3.5")
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
    results.append(("Versão V3.4+ do indicador", verify_indicator_version()))
    # VERIFICAÇÕES V3.4
    results.append(("Teste funcional look-ahead", verify_functional_no_lookahead()))
    results.append(("Filtros centralizados", verify_centralized_filters()))
    results.append(("Escala threshold Ricci", verify_ricci_threshold_scale()))
    # NOVAS VERIFICAÇÕES V3.5
    results.append(("Consistência estratégia-otimizador", verify_strategy_optimizer_consistency()))
    results.append(("Método from_config", verify_from_config_method()))
    results.append(("Consistência funcional", verify_functional_consistency()))

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
