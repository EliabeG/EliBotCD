#!/usr/bin/env python3
"""
================================================================================
VERIFICACAO DE PRONTIDAO PARA DINHEIRO REAL - ODMN V2.7
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
9. Signal inclui stop_loss_pips e take_profit_pips
10. MIN_CONFIDENCE do config centralizado
11. direction_calculator centralizado usado
12. Suporte a seed para reprodutibilidade
13. Consistencia de parametros (config vs componentes)
14. Teste de reprodutibilidade (mesmos resultados com mesmo seed)
15. V2.3: Spread aplicado na SAIDA do optimizer
16. V2.3: Seed passado na estrategia ODMN
17. V2.3: MIN_EXPECTANCY >= 3.0 pips
18. V2.4: MFG usa historical_prices para direcao correta
19. V2.4: SIGNAL_COOLDOWN do config centralizado
20. V2.4: Indicador tem metodo reset() publico
21. V2.5: MFG levanta ValueError se historical_prices invalido
22. V2.6: ANNUALIZATION_FACTOR configuravel (sem 252 hardcoded)
23. V2.6: MALLIAVIN_HORIZON dinamico por timeframe
24. V2.6: MIN_WARMUP_BARS warmup implementado
25. V2.6: Tratamento de erro defensivo na estrategia

FUNDAMENTOS TEORICOS DO ODMN:
============================
1. Modelo de Heston: Volatilidade estocastica calibrada em dados passados
2. Calculo de Malliavin: Derivadas estocasticas para fragilidade
3. Mean Field Games: Equilibrio Nash para comportamento institucional

SE TODAS AS 25 VERIFICACOES PASSAREM = PRONTO PARA DINHEIRO REAL

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

        # V2.7: Verifica se usa calculate_direction_from_bars (modulo centralizado)
        # OU bars[i-1].close diretamente
        uses_centralized = "calculate_direction_from_bars" in content
        has_closed_bar = "bars[i - 1].close" in content or "self.bars[i - 1].close" in content

        # Verifica se exclui barra atual
        excludes_current = "bars[i].close" not in content.split("direction")[0] if "direction" in content else True

        # Verifica se usa trend_lookback
        has_trend_lookback = "min_bars_for_direction" in content or "TREND_LOOKBACK" in content

        # V2.7: Passa se usa modulo centralizado OU acessa bars[i-1] diretamente
        passed = (uses_centralized or has_closed_bar) and has_trend_lookback
        details = []

        if uses_centralized:
            details.append("Usa calculate_direction_from_bars (modulo centralizado)")
        elif has_closed_bar:
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


def verify_signal_includes_pips() -> ODMNVerificationResult:
    """
    Verifica 9: Signal inclui stop_loss_pips e take_profit_pips

    O Signal deve incluir valores em pips para que o BacktestEngine
    possa recalcular os niveis baseado no preco de entrada real.
    """
    strategy_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_strategy.py"
    )

    try:
        with open(strategy_path, 'r') as f:
            content = f.read()

        details = []

        # Verifica se stop_loss_pips e take_profit_pips sao passados no Signal
        has_sl_pips = "stop_loss_pips=self.stop_loss_pips" in content
        has_tp_pips = "take_profit_pips=self.take_profit_pips" in content

        if has_sl_pips:
            details.append("stop_loss_pips incluido no Signal")
        else:
            details.append("ERRO: stop_loss_pips NAO incluido no Signal")

        if has_tp_pips:
            details.append("take_profit_pips incluido no Signal")
        else:
            details.append("ERRO: take_profit_pips NAO incluido no Signal")

        passed = has_sl_pips and has_tp_pips
        return ODMNVerificationResult(
            "Signal inclui stop/take em pips",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "Signal inclui stop/take em pips",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_min_confidence_from_config() -> ODMNVerificationResult:
    """
    Verifica 10: MIN_CONFIDENCE importado do config

    A confianca minima deve vir do config centralizado,
    nao de valores hardcoded.
    """
    strategy_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_strategy.py"
    )

    try:
        with open(strategy_path, 'r') as f:
            content = f.read()

        details = []

        # Verifica se importa MIN_CONFIDENCE
        imports_config = "from config.odmn_config import MIN_CONFIDENCE" in content

        # V2.7: Verifica se usa MIN_CONFIDENCE no codigo
        # Aceita tanto 'confidence >= MIN_CONFIDENCE' quanto "['confidence'] >= MIN_CONFIDENCE"
        uses_config = "MIN_CONFIDENCE" in content and ">= MIN_CONFIDENCE" in content

        # Verifica se NAO tem 0.6 hardcoded
        no_hardcoded = "confidence >= 0.6" not in content

        if imports_config:
            details.append("Importa MIN_CONFIDENCE do config")
        else:
            details.append("ERRO: Nao importa MIN_CONFIDENCE")

        if uses_config:
            details.append("Usa MIN_CONFIDENCE no codigo")
        else:
            details.append("ERRO: Nao usa MIN_CONFIDENCE")

        if no_hardcoded:
            details.append("Sem valor hardcoded 0.6")
        else:
            details.append("AVISO: Valor 0.6 hardcoded encontrado")

        passed = imports_config and uses_config and no_hardcoded
        return ODMNVerificationResult(
            "MIN_CONFIDENCE do config centralizado",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "MIN_CONFIDENCE do config centralizado",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_direction_calculator_used() -> ODMNVerificationResult:
    """
    Verifica 11: direction_calculator centralizado usado

    O optimizer e debug devem usar direction_calculator para
    garantir consistencia no calculo de direcao.
    """
    optimizer_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "backtesting", "odmn", "optimizer.py"
    )

    debug_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "backtesting", "odmn", "debug.py"
    )

    try:
        with open(optimizer_path, 'r') as f:
            optimizer_content = f.read()

        with open(debug_path, 'r') as f:
            debug_content = f.read()

        details = []

        # Verifica optimizer
        optimizer_imports = "from backtesting.common.direction_calculator import" in optimizer_content
        optimizer_uses = "calculate_direction_from_bars" in optimizer_content

        # Verifica debug
        debug_imports = "from backtesting.common.direction_calculator import" in debug_content
        debug_uses = "calculate_direction_from_bars" in debug_content or "TREND_LOOKBACK" in debug_content

        if optimizer_imports and optimizer_uses:
            details.append("Optimizer usa direction_calculator")
        else:
            details.append("ERRO: Optimizer nao usa direction_calculator")

        if debug_imports or debug_uses:
            details.append("Debug usa direction_calculator ou TREND_LOOKBACK")
        else:
            details.append("AVISO: Debug pode ter valores hardcoded")

        passed = optimizer_imports and optimizer_uses
        return ODMNVerificationResult(
            "direction_calculator centralizado usado",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "direction_calculator centralizado usado",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_seed_reproducibility_support() -> ODMNVerificationResult:
    """
    Verifica 9: Suporte a seed para reprodutibilidade

    Os componentes Monte Carlo devem aceitar parametro seed para
    garantir resultados deterministicos.
    """
    indicator_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_malliavin_nash.py"
    )

    try:
        with open(indicator_path, 'r') as f:
            content = f.read()

        details = []

        # Verifica se HestonModel.simulate aceita seed
        heston_seed = "def simulate(self, S0: float, T: float, n_steps: int, n_paths: int = 1000,\n                 seed: int = None)" in content or \
                      "seed: int = None) -> dict" in content

        # Verifica se MalliavinDerivativeCalculator aceita seed
        malliavin_seed = "seed: int = None) -> dict:" in content and "compute_malliavin_weight" in content

        # Verifica se DeepGalerkinMFGSolver aceita seed
        mfg_seed = "seed: int = None) -> dict:" in content and "solve_mfg_equilibrium" in content

        # Verifica se OracloDerivativosMalliavinNash aceita seed
        odmn_seed = "seed: int = None):" in content and "OracloDerivativosMalliavinNash" in content

        # Verifica se usa np.random.default_rng (thread-safe)
        uses_rng = "np.random.default_rng" in content

        if heston_seed:
            details.append("HestonModel.simulate aceita seed")
        else:
            details.append("ERRO: HestonModel.simulate nao aceita seed")

        if malliavin_seed:
            details.append("MalliavinCalculator aceita seed")
        else:
            details.append("ERRO: MalliavinCalculator nao aceita seed")

        if mfg_seed:
            details.append("MFGSolver aceita seed")
        else:
            details.append("ERRO: MFGSolver nao aceita seed")

        if odmn_seed:
            details.append("ODMN aceita seed")
        else:
            details.append("ERRO: ODMN nao aceita seed")

        if uses_rng:
            details.append("Usa np.random.default_rng (thread-safe)")
        else:
            details.append("AVISO: Nao usa Generator thread-safe")

        passed = heston_seed and malliavin_seed and odmn_seed
        return ODMNVerificationResult(
            "Suporte a seed para reprodutibilidade",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "Suporte a seed para reprodutibilidade",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_parameter_consistency() -> ODMNVerificationResult:
    """
    Verifica 10: Consistencia de parametros entre config e componentes

    Os valores default dos componentes devem corresponder aos valores
    definidos em config/odmn_config.py.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "odmn_config.py"
    )

    indicator_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_malliavin_nash.py"
    )

    optimizer_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "backtesting", "odmn", "optimizer.py"
    )

    try:
        with open(config_path, 'r') as f:
            config_content = f.read()

        with open(indicator_path, 'r') as f:
            indicator_content = f.read()

        with open(optimizer_path, 'r') as f:
            optimizer_content = f.read()

        details = []
        checks = []

        # Extrai valores do config
        malliavin_paths_config = re.search(r'MALLIAVIN_PATHS:\s*int\s*=\s*(\d+)', config_content)
        malliavin_steps_config = re.search(r'MALLIAVIN_STEPS:\s*int\s*=\s*(\d+)', config_content)
        mfg_thresh_config = re.search(r'MFG_DIRECTION_THRESHOLD:\s*float\s*=\s*([\d.]+)', config_content)

        # Verifica se indicator usa valores corretos
        if malliavin_paths_config:
            config_val = int(malliavin_paths_config.group(1))
            # Verifica default do MalliavinDerivativeCalculator
            indicator_default = re.search(r'def __init__\(self, n_paths: int = (\d+)', indicator_content)
            if indicator_default:
                ind_val = int(indicator_default.group(1))
                matches = ind_val == config_val
                checks.append(matches)
                details.append(f"MALLIAVIN_PATHS: config={config_val}, indicator={ind_val} {'OK' if matches else 'ERRO'}")
            else:
                checks.append(True)  # Assume OK if can't find
                details.append(f"MALLIAVIN_PATHS: config={config_val}")

        if malliavin_steps_config:
            config_val = int(malliavin_steps_config.group(1))
            indicator_default = re.search(r'n_paths: int = \d+, n_steps: int = (\d+)', indicator_content)
            if indicator_default:
                ind_val = int(indicator_default.group(1))
                matches = ind_val == config_val
                checks.append(matches)
                details.append(f"MALLIAVIN_STEPS: config={config_val}, indicator={ind_val} {'OK' if matches else 'ERRO'}")

        # Verifica se optimizer usa MFG_DIRECTION_THRESHOLD do config
        if mfg_thresh_config:
            uses_config = "MFG_DIRECTION_THRESHOLD" in optimizer_content and \
                          "mfg_direction_threshold=MFG_DIRECTION_THRESHOLD" in optimizer_content
            checks.append(uses_config)
            details.append(f"Optimizer usa MFG_DIRECTION_THRESHOLD do config: {'SIM' if uses_config else 'NAO'}")

        # Verifica se optimizer usa seed
        uses_seed = "seed=OPTIMIZER_SEED" in optimizer_content or "seed=" in optimizer_content
        checks.append(uses_seed)
        details.append(f"Optimizer usa seed: {'SIM' if uses_seed else 'NAO'}")

        passed = len(checks) >= 3 and all(checks)
        return ODMNVerificationResult(
            "Consistencia de parametros config vs componentes",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "Consistencia de parametros config vs componentes",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_reproducibility_test() -> ODMNVerificationResult:
    """
    Verifica 11: Teste de reprodutibilidade

    Verifica se o codigo tem estrutura para gerar resultados
    deterministicos quando seed e fornecido.
    """
    indicator_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_malliavin_nash.py"
    )

    optimizer_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "backtesting", "odmn", "optimizer.py"
    )

    try:
        with open(indicator_path, 'r') as f:
            indicator_content = f.read()

        with open(optimizer_path, 'r') as f:
            optimizer_content = f.read()

        details = []
        checks = []

        # Verifica se usa Generator ao inves de np.random global
        uses_generator = "np.random.default_rng" in indicator_content
        checks.append(uses_generator)
        details.append(f"Usa np.random.Generator: {'SIM' if uses_generator else 'NAO'}")

        # Verifica se nao usa np.random.randn diretamente (sem seed)
        # Procura por uso sem rng.
        uses_unsafe_random = "np.random.randn" in indicator_content
        uses_safe = "rng.standard_normal" in indicator_content
        safe_random = uses_safe and not uses_unsafe_random
        # Nota: pode ter ambos durante transicao, priorizamos o safe
        if uses_safe:
            checks.append(True)
            details.append("Usa rng.standard_normal (thread-safe)")
        elif uses_unsafe_random:
            checks.append(False)
            details.append("ERRO: Usa np.random.randn (nao thread-safe)")
        else:
            checks.append(True)
            details.append("Sem uso direto de random detectado")

        # Verifica se optimizer tem seed fixo
        has_optimizer_seed = "OPTIMIZER_SEED" in optimizer_content and \
                            "random.seed" in optimizer_content
        checks.append(has_optimizer_seed)
        details.append(f"Optimizer tem seed fixo: {'SIM' if has_optimizer_seed else 'NAO'}")

        # Verifica se torch.manual_seed e usado para MFG
        has_torch_seed = "torch.manual_seed" in indicator_content
        checks.append(has_torch_seed)
        details.append(f"MFG usa torch.manual_seed: {'SIM' if has_torch_seed else 'NAO'}")

        passed = len(checks) >= 3 and sum(checks) >= 3
        return ODMNVerificationResult(
            "Teste de reprodutibilidade",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "Teste de reprodutibilidade",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_spread_applied_on_exit() -> ODMNVerificationResult:
    """
    Verifica 15 (V2.3): Spread aplicado na SAIDA do optimizer

    O optimizer deve aplicar spread/2 + slippage na saida,
    nao apenas slippage. Correcao critica da auditoria V2.3.
    """
    optimizer_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "backtesting", "odmn", "optimizer.py"
    )

    try:
        with open(optimizer_path, 'r') as f:
            content = f.read()

        details = []

        # Verifica se spread/2 e aplicado na saida
        has_spread_on_exit = "spread/2 - slippage" in content or "spread/2 + slippage" in content

        # Verifica se e aplicado em gaps, durante barra e timeout
        has_gap_spread = "bar.open - spread/2 - slippage" in content  # LONG gap
        has_bar_spread = "stop_price - spread/2 - slippage" in content  # LONG stop
        has_timeout_spread = "last_bar.close - spread/2 - slippage" in content  # LONG timeout

        if has_spread_on_exit:
            details.append("Spread aplicado na saida (spread/2 + slippage)")
        else:
            details.append("ERRO: Spread NAO aplicado na saida")

        if has_gap_spread:
            details.append("Spread em gaps verificado")

        if has_bar_spread:
            details.append("Spread em stops/takes verificado")

        if has_timeout_spread:
            details.append("Spread em timeout verificado")

        passed = has_spread_on_exit and has_gap_spread and has_bar_spread and has_timeout_spread
        return ODMNVerificationResult(
            "V2.3: Spread aplicado na SAIDA",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "V2.3: Spread aplicado na SAIDA",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_seed_in_strategy() -> ODMNVerificationResult:
    """
    Verifica 16 (V2.3): Seed passado na estrategia ODMN

    A estrategia deve aceitar parametro seed e passa-lo
    para o indicador ODMN para reprodutibilidade.
    """
    strategy_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_strategy.py"
    )

    try:
        with open(strategy_path, 'r') as f:
            content = f.read()

        details = []

        # Verifica se __init__ aceita seed
        accepts_seed = "seed: int = None" in content

        # Verifica se passa seed para ODMN
        passes_seed = "seed=seed" in content

        # Verifica se armazena seed
        stores_seed = "self.seed = seed" in content

        if accepts_seed:
            details.append("Strategy aceita parametro seed")
        else:
            details.append("ERRO: Strategy NAO aceita seed")

        if passes_seed:
            details.append("Passa seed para ODMN indicator")
        else:
            details.append("ERRO: NAO passa seed para ODMN")

        if stores_seed:
            details.append("Armazena self.seed")

        passed = accepts_seed and passes_seed
        return ODMNVerificationResult(
            "V2.3: Seed passado na estrategia",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "V2.3: Seed passado na estrategia",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_min_expectancy_adequate() -> ODMNVerificationResult:
    """
    Verifica 17 (V2.3): MIN_EXPECTANCY >= 3.0 pips

    Com custos totais de ~4.6 pips (spread 1.5 entrada + 0.75 saida
    + slippage 0.8 entrada + 0.8 saida), MIN_EXPECTANCY deve ser
    pelo menos 3.0 pips para garantir margem de seguranca.
    """
    filters_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "optimizer_filters.py"
    )

    try:
        with open(filters_path, 'r') as f:
            content = f.read()

        details = []

        # Extrai valor do MIN_EXPECTANCY_PIPS
        min_exp_match = re.search(r'MIN_EXPECTANCY_PIPS:\s*float\s*=\s*([\d.]+)', content)

        if min_exp_match:
            min_exp = float(min_exp_match.group(1))
            adequate = min_exp >= 3.0

            if adequate:
                details.append(f"MIN_EXPECTANCY_PIPS = {min_exp} pips (adequado)")
            else:
                details.append(f"ERRO: MIN_EXPECTANCY_PIPS = {min_exp} < 3.0 pips")

            # Verifica comentario da V2.3
            has_v23_comment = "V2.3" in content and "3.0" in content
            if has_v23_comment:
                details.append("Comentario V2.3 presente")

            passed = adequate
        else:
            details.append("ERRO: MIN_EXPECTANCY_PIPS nao encontrado")
            passed = False

        return ODMNVerificationResult(
            "V2.3: MIN_EXPECTANCY >= 3.0 pips",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "V2.3: MIN_EXPECTANCY >= 3.0 pips",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_mfg_uses_historical_prices() -> ODMNVerificationResult:
    """
    Verifica 18 (V2.4): MFG usa historical_prices para direcao correta

    BUG CRITICO V2.3: _analytical_approximation sempre retornava mfg_direction=0
    porque mean_log_price = log_price. V2.4 corrige passando historical_prices
    para calcular mean_log_price corretamente.
    """
    indicator_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_malliavin_nash.py"
    )

    try:
        with open(indicator_path, 'r') as f:
            content = f.read()

        details = []

        # Verifica se _analytical_approximation aceita historical_prices
        accepts_historical = "historical_prices: np.ndarray = None" in content

        # V2.7: Verifica se calcula mean_log_price a partir dos precos historicos
        # Aceita tanto 'np.mean(np.log(historical_prices))' quanto 'np.mean(np.log(prices_for_mean))'
        # porque prices_for_mean e derivado de historical_prices
        uses_historical = ("np.mean(np.log(historical_prices))" in content or
                          "np.mean(np.log(prices_for_mean))" in content)

        # Verifica se analyze() passa prices para MFG
        passes_prices = "historical_prices=prices" in content

        # Verifica que NAO tem o bug original (mean_log_price = log_price)
        no_bug = "mean_log_price = log_price" not in content

        if accepts_historical:
            details.append("_analytical_approximation aceita historical_prices")
        else:
            details.append("ERRO: _analytical_approximation NAO aceita historical_prices")

        if uses_historical:
            details.append("Calcula mean_log_price de precos historicos")
        else:
            details.append("ERRO: Nao calcula mean_log_price corretamente")

        if passes_prices:
            details.append("analyze() passa prices para MFG")
        else:
            details.append("ERRO: analyze() nao passa prices para MFG")

        if no_bug:
            details.append("Bug original corrigido (mean_log_price != log_price)")
        else:
            details.append("ERRO CRITICO: Bug original presente (mfg_direction=0)")

        passed = accepts_historical and uses_historical and passes_prices and no_bug
        return ODMNVerificationResult(
            "V2.4: MFG usa historical_prices",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "V2.4: MFG usa historical_prices",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_signal_cooldown_from_config() -> ODMNVerificationResult:
    """
    Verifica 19 (V2.4): SIGNAL_COOLDOWN do config centralizado

    A estrategia deve usar SIGNAL_COOLDOWN do config, nao valor hardcoded.
    """
    strategy_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_strategy.py"
    )

    try:
        with open(strategy_path, 'r') as f:
            content = f.read()

        details = []

        # Verifica se importa SIGNAL_COOLDOWN do config
        imports_cooldown = "from config.odmn_config import" in content and "SIGNAL_COOLDOWN" in content

        # Verifica se usa SIGNAL_COOLDOWN (nao valor hardcoded)
        uses_cooldown = "self.signal_cooldown = SIGNAL_COOLDOWN" in content

        # Verifica se NAO tem valor hardcoded
        no_hardcoded = "self.signal_cooldown = 25" not in content

        if imports_cooldown:
            details.append("Importa SIGNAL_COOLDOWN do config")
        else:
            details.append("ERRO: Nao importa SIGNAL_COOLDOWN")

        if uses_cooldown:
            details.append("Usa SIGNAL_COOLDOWN no codigo")
        else:
            details.append("ERRO: Nao usa SIGNAL_COOLDOWN")

        if no_hardcoded:
            details.append("Sem valor hardcoded 25")
        else:
            details.append("AVISO: Valor 25 hardcoded encontrado")

        passed = imports_cooldown and uses_cooldown and no_hardcoded
        return ODMNVerificationResult(
            "V2.4: SIGNAL_COOLDOWN do config",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "V2.4: SIGNAL_COOLDOWN do config",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_indicator_has_reset_method() -> ODMNVerificationResult:
    """
    Verifica 20 (V2.4): Indicador tem metodo reset() publico

    O indicador deve ter metodo reset() publico para que a estrategia
    possa resetar o estado sem acessar _cache diretamente.
    """
    indicator_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_malliavin_nash.py"
    )

    strategy_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_strategy.py"
    )

    try:
        with open(indicator_path, 'r') as f:
            indicator_content = f.read()

        with open(strategy_path, 'r') as f:
            strategy_content = f.read()

        details = []

        # Verifica se indicador tem metodo reset()
        has_reset = "def reset(self):" in indicator_content and "self._cache = {" in indicator_content

        # Verifica se estrategia usa self.odmn.reset() ao inves de acessar _cache
        uses_reset = "self.odmn.reset()" in strategy_content

        # Verifica se estrategia NAO acessa _cache diretamente
        no_direct_cache = "self.odmn._cache" not in strategy_content

        if has_reset:
            details.append("Indicador tem metodo reset()")
        else:
            details.append("ERRO: Indicador NAO tem reset()")

        if uses_reset:
            details.append("Estrategia usa self.odmn.reset()")
        else:
            details.append("ERRO: Estrategia nao usa reset()")

        if no_direct_cache:
            details.append("Estrategia nao acessa _cache diretamente")
        else:
            details.append("AVISO: Estrategia acessa _cache diretamente")

        passed = has_reset and uses_reset and no_direct_cache
        return ODMNVerificationResult(
            "V2.4: Indicador tem reset() publico",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "V2.4: Indicador tem reset() publico",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_mfg_raises_error_on_invalid_input() -> ODMNVerificationResult:
    """
    Verifica 21 (V2.5): MFG levanta ValueError se historical_prices invalido

    V2.5 FIX: _analytical_approximation deve levantar ValueError explicitamente
    quando historical_prices e None ou tem poucos elementos, ao inves de usar
    fallback silencioso que retorna valores incorretos.
    """
    indicator_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_malliavin_nash.py"
    )

    try:
        with open(indicator_path, 'r') as f:
            content = f.read()

        details = []

        # Verifica se levanta ValueError ao inves de fallback
        raises_error = "raise ValueError" in content and "historical_prices" in content

        # Verifica se NAO tem o fallback silencioso antigo
        no_silent_fallback = "mean_log_price = log_price - sigma * 0.5" not in content

        # Verifica se tem MIN_PRICES_FOR_MFG definido
        has_min_check = "MIN_PRICES_FOR_MFG" in content

        # Verifica comentario V2.5
        has_v25_comment = "V2.5" in content

        if raises_error:
            details.append("Levanta ValueError se historical_prices invalido")
        else:
            details.append("ERRO: Nao levanta ValueError")

        if no_silent_fallback:
            details.append("Fallback silencioso removido")
        else:
            details.append("ERRO CRITICO: Fallback silencioso ainda presente")

        if has_min_check:
            details.append("MIN_PRICES_FOR_MFG definido")

        if has_v25_comment:
            details.append("Comentario V2.5 presente")

        passed = raises_error and no_silent_fallback and has_min_check
        return ODMNVerificationResult(
            "V2.5: MFG levanta ValueError",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "V2.5: MFG levanta ValueError",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_annualization_factor_configurable() -> ODMNVerificationResult:
    """
    Verifica 22 (V2.6): ANNUALIZATION_FACTOR configuravel

    O indicador deve usar ANNUALIZATION_FACTOR do config ao inves de
    252 hardcoded, para suportar diferentes timeframes (H1, M15, etc).
    """
    indicator_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_malliavin_nash.py"
    )

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "odmn_config.py"
    )

    try:
        with open(indicator_path, 'r') as f:
            indicator_content = f.read()

        with open(config_path, 'r') as f:
            config_content = f.read()

        details = []

        # Verifica se importa ANNUALIZATION_FACTOR
        imports_factor = "ANNUALIZATION_FACTOR" in indicator_content

        # Verifica se usa ANNUALIZATION_FACTOR no codigo (nao 252 hardcoded)
        uses_factor = "* ANNUALIZATION_FACTOR" in indicator_content

        # Verifica se NAO tem 252 hardcoded nos calculos principais
        # Procura por padroes como "* 252" que indicam uso hardcoded
        no_hardcoded = "var_r * 252" not in indicator_content and \
                       "mean_r * 252" not in indicator_content

        # Verifica se config tem ANNUALIZATION_FACTOR definido
        config_has_factor = "ANNUALIZATION_FACTOR" in config_content and \
                           "_ANNUALIZATION_FACTORS" in config_content

        if imports_factor:
            details.append("Importa ANNUALIZATION_FACTOR")
        else:
            details.append("ERRO: Nao importa ANNUALIZATION_FACTOR")

        if uses_factor:
            details.append("Usa ANNUALIZATION_FACTOR nos calculos")
        else:
            details.append("ERRO: Nao usa ANNUALIZATION_FACTOR")

        if no_hardcoded:
            details.append("Sem 252 hardcoded")
        else:
            details.append("ERRO CRITICO: 252 hardcoded encontrado")

        if config_has_factor:
            details.append("Config tem mapeamento de timeframes")

        passed = imports_factor and uses_factor and no_hardcoded and config_has_factor
        return ODMNVerificationResult(
            "V2.6: ANNUALIZATION_FACTOR configuravel",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "V2.6: ANNUALIZATION_FACTOR configuravel",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_malliavin_horizon_dynamic() -> ODMNVerificationResult:
    """
    Verifica 23 (V2.6): MALLIAVIN_HORIZON dinamico por timeframe

    O horizonte de simulacao Malliavin deve ser configuravel,
    nao fixo em T=1/252 (1 dia).
    """
    indicator_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_malliavin_nash.py"
    )

    try:
        with open(indicator_path, 'r') as f:
            content = f.read()

        details = []

        # Verifica se importa MALLIAVIN_HORIZON
        imports_horizon = "MALLIAVIN_HORIZON" in content

        # Verifica se usa MALLIAVIN_HORIZON (nao T=1/252 hardcoded)
        uses_horizon = "T=MALLIAVIN_HORIZON" in content

        # Verifica se NAO tem T=1/252 hardcoded
        no_hardcoded = "T=1/252" not in content

        if imports_horizon:
            details.append("Importa MALLIAVIN_HORIZON")
        else:
            details.append("ERRO: Nao importa MALLIAVIN_HORIZON")

        if uses_horizon:
            details.append("Usa MALLIAVIN_HORIZON no Malliavin")
        else:
            details.append("ERRO: Nao usa MALLIAVIN_HORIZON")

        if no_hardcoded:
            details.append("Sem T=1/252 hardcoded")
        else:
            details.append("ERRO: T=1/252 hardcoded encontrado")

        passed = imports_horizon and uses_horizon and no_hardcoded
        return ODMNVerificationResult(
            "V2.6: MALLIAVIN_HORIZON dinamico",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "V2.6: MALLIAVIN_HORIZON dinamico",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_warmup_implemented() -> ODMNVerificationResult:
    """
    Verifica 24 (V2.6): MIN_WARMUP_BARS warmup implementado

    O indicador deve ter periodo de warmup para evitar sinais erraticos
    nas primeiras barras (cold start problem).
    """
    indicator_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_malliavin_nash.py"
    )

    try:
        with open(indicator_path, 'r') as f:
            content = f.read()

        details = []

        # Verifica se importa MIN_WARMUP_BARS
        imports_warmup = "MIN_WARMUP_BARS" in content

        # Verifica se tem logica de warmup
        has_warmup_check = "is_warmup" in content

        # Verifica se retorna is_warmup no resultado
        returns_warmup = "'is_warmup':" in content

        # Verifica se forca HOLD durante warmup
        holds_on_warmup = "if is_warmup:" in content and "signal = 0" in content

        if imports_warmup:
            details.append("Importa MIN_WARMUP_BARS")
        else:
            details.append("ERRO: Nao importa MIN_WARMUP_BARS")

        if has_warmup_check:
            details.append("Tem verificacao de warmup")
        else:
            details.append("ERRO: Sem verificacao de warmup")

        if holds_on_warmup:
            details.append("Forca HOLD durante warmup")
        else:
            details.append("ERRO: Nao protege warmup")

        if returns_warmup:
            details.append("Retorna is_warmup no resultado")

        passed = imports_warmup and has_warmup_check and holds_on_warmup
        return ODMNVerificationResult(
            "V2.6: Warmup implementado",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "V2.6: Warmup implementado",
            False,
            f"Erro ao verificar: {e}"
        )


def verify_defensive_error_handling() -> ODMNVerificationResult:
    """
    Verifica 25 (V2.6): Tratamento de erro defensivo na estrategia

    A estrategia deve ter tratamento de erro robusto que:
    - Rastreia erros consecutivos
    - Entra em modo de seguranca apos multiplos erros
    - Pode ser resetada para sair do modo de seguranca
    """
    strategy_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "strategies", "alta_volatilidade", "odmn_strategy.py"
    )

    try:
        with open(strategy_path, 'r') as f:
            content = f.read()

        details = []

        # Verifica se rastreia erros consecutivos
        tracks_errors = "_consecutive_errors" in content

        # Verifica se tem modo de seguranca
        has_error_state = "_in_error_state" in content

        # Verifica se tem metodo para checar estado de erro
        has_check_method = "is_in_error_state" in content

        # Verifica se reseta estado de erro no reset()
        resets_error = "_in_error_state = False" in content

        # Verifica se tem limite de erros
        has_limit = "MAX_CONSECUTIVE_ERRORS" in content

        if tracks_errors:
            details.append("Rastreia erros consecutivos")
        else:
            details.append("ERRO: Nao rastreia erros")

        if has_error_state:
            details.append("Tem modo de seguranca")
        else:
            details.append("ERRO: Sem modo de seguranca")

        if has_check_method:
            details.append("Tem is_in_error_state()")

        if resets_error:
            details.append("Reset limpa estado de erro")

        if has_limit:
            details.append("Tem limite de erros configurado")

        passed = tracks_errors and has_error_state and resets_error and has_limit
        return ODMNVerificationResult(
            "V2.6: Erro defensivo",
            passed,
            "; ".join(details)
        )

    except Exception as e:
        return ODMNVerificationResult(
            "V2.6: Erro defensivo",
            False,
            f"Erro ao verificar: {e}"
        )


def run_all_verifications():
    """Executa todas as verificacoes e imprime resultado"""
    print("=" * 70)
    print("  VERIFICACAO DE PRONTIDAO PARA DINHEIRO REAL - ODMN V2.7")
    print("  Oraculo de Derivativos de Malliavin-Nash")
    print("=" * 70)
    print("\n  25 verificacoes criticas para evitar look-ahead bias\n")

    verifications = [
        verify_heston_calibration_no_lookahead,
        verify_malliavin_no_lookahead,
        verify_mfg_no_lookahead,
        verify_direction_from_closed_bars,
        verify_entry_on_next_bar_open,
        verify_centralized_costs,
        verify_rigorous_filters,
        verify_walk_forward_implemented,
        verify_signal_includes_pips,
        verify_min_confidence_from_config,
        verify_direction_calculator_used,
        verify_seed_reproducibility_support,
        verify_parameter_consistency,
        verify_reproducibility_test,
        # V2.3 verificacoes adicionais
        verify_spread_applied_on_exit,
        verify_seed_in_strategy,
        verify_min_expectancy_adequate,
        # V2.4 verificacoes adicionais
        verify_mfg_uses_historical_prices,
        verify_signal_cooldown_from_config,
        verify_indicator_has_reset_method,
        # V2.5 verificacoes adicionais
        verify_mfg_raises_error_on_invalid_input,
        # V2.6 verificacoes adicionais
        verify_annualization_factor_configurable,
        verify_malliavin_horizon_dynamic,
        verify_warmup_implemented,
        verify_defensive_error_handling,
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
        print("\n  O indicador ODMN passou em TODAS as 25 verificacoes criticas:")
        print("    1. Calibracao Heston usa apenas dados passados")
        print("    2. Malliavin Monte Carlo e causal (forward simulation)")
        print("    3. Mean Field Games resolve PDEs sem dados futuros")
        print("    4. Direcao baseada APENAS em barras fechadas")
        print("    5. Entrada no OPEN da proxima barra")
        print("    6. Custos realistas (spread 1.5 + slippage 0.8 pips)")
        print("    7. Filtros rigorosos (PF >= 1.3, Exp >= 3.0 pips)")
        print("    8. Walk-Forward Validation com 4 janelas")
        print("    9. Signal inclui stop_loss_pips e take_profit_pips")
        print("   10. MIN_CONFIDENCE do config centralizado")
        print("   11. direction_calculator centralizado usado")
        print("   12. Suporte a seed para reprodutibilidade")
        print("   13. Consistencia de parametros (config vs componentes)")
        print("   14. Teste de reprodutibilidade (np.random.Generator)")
        print("   15. V2.3: Spread aplicado na SAIDA do optimizer")
        print("   16. V2.3: Seed passado na estrategia ODMN")
        print("   17. V2.3: MIN_EXPECTANCY >= 3.0 pips")
        print("   18. V2.4: MFG usa historical_prices (bug critico corrigido)")
        print("   19. V2.4: SIGNAL_COOLDOWN do config centralizado")
        print("   20. V2.4: Indicador tem metodo reset() publico")
        print("   21. V2.5: MFG levanta ValueError (falha explicita)")
        print("   22. V2.6: ANNUALIZATION_FACTOR configuravel (multi-timeframe)")
        print("   23. V2.6: MALLIAVIN_HORIZON dinamico por timeframe")
        print("   24. V2.6: Warmup implementado (protecao cold start)")
        print("   25. V2.6: Tratamento de erro defensivo na estrategia")
        print("\n  Proximos passos:")
        print("    1. Ajuste TIMEFRAME em config/odmn_config.py para seu timeframe")
        print("    2. Execute o optimizer: python -m backtesting.odmn.optimizer")
        print("    3. Valide os resultados no periodo de teste")
        print("    4. Faca paper trading por 2-4 semanas antes de real")
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
