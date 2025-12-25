#!/usr/bin/env python3
"""
================================================================================
TESTES UNITÁRIOS - DSG CORE
================================================================================

Testes unitários para componentes principais do sistema DSG:
1. Indicador DetectorSingularidadeGravitacional
2. Estratégia DSGStrategy
3. Filtros centralizados

EXECUÇÃO:
    python -m pytest tests/unit/test_dsg_core.py -v
    python -m tests.unit.test_dsg_core

Criado: Dezembro 2025 (Auditoria V3.4)
================================================================================
"""

import sys
import os
import numpy as np
import unittest

# Adiciona raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestOptimizerFilters(unittest.TestCase):
    """Testes para config/optimizer_filters.py"""

    def test_filters_exist(self):
        """Verifica se os filtros centralizados existem"""
        from config.optimizer_filters import (
            MIN_TRADES_TRAIN, MIN_TRADES_TEST,
            MIN_WIN_RATE, MAX_WIN_RATE,
            MIN_PROFIT_FACTOR, MAX_PROFIT_FACTOR,
            MAX_DRAWDOWN, MIN_ROBUSTNESS
        )

        self.assertIsInstance(MIN_TRADES_TRAIN, int)
        self.assertIsInstance(MIN_TRADES_TEST, int)
        self.assertIsInstance(MIN_WIN_RATE, float)
        self.assertIsInstance(MAX_WIN_RATE, float)
        self.assertIsInstance(MIN_PROFIT_FACTOR, float)
        self.assertIsInstance(MAX_PROFIT_FACTOR, float)
        self.assertIsInstance(MAX_DRAWDOWN, float)
        self.assertIsInstance(MIN_ROBUSTNESS, float)

    def test_filter_values_reasonable(self):
        """Verifica se os valores dos filtros são razoáveis"""
        from config.optimizer_filters import (
            MIN_TRADES_TRAIN, MIN_TRADES_TEST,
            MIN_WIN_RATE, MAX_WIN_RATE,
            MIN_PROFIT_FACTOR, MAX_PROFIT_FACTOR,
            MAX_DRAWDOWN
        )

        # Trades mínimos
        self.assertGreater(MIN_TRADES_TRAIN, 10)
        self.assertLess(MIN_TRADES_TRAIN, 200)
        self.assertGreater(MIN_TRADES_TEST, 5)

        # Win rate deve estar entre 0 e 1
        self.assertGreater(MIN_WIN_RATE, 0)
        self.assertLess(MAX_WIN_RATE, 1)
        self.assertLess(MIN_WIN_RATE, MAX_WIN_RATE)

        # Profit factor mínimo deve ser > 1 para ser lucrativo
        self.assertGreater(MIN_PROFIT_FACTOR, 1.0)
        self.assertLess(MAX_PROFIT_FACTOR, 10.0)

        # Drawdown máximo deve ser < 1 (100%)
        self.assertGreater(MAX_DRAWDOWN, 0)
        self.assertLess(MAX_DRAWDOWN, 1)

    def test_validation_functions(self):
        """Testa funções de validação"""
        from config.optimizer_filters import (
            validate_train_result, validate_test_result, validate_robustness
        )

        # Resultado bom deve passar
        self.assertTrue(validate_train_result(
            trades=100, win_rate=0.45, profit_factor=1.5, max_drawdown=0.20
        ))

        # Poucos trades deve falhar
        self.assertFalse(validate_train_result(
            trades=10, win_rate=0.45, profit_factor=1.5, max_drawdown=0.20
        ))

        # PF muito alto (overfitting) deve falhar
        self.assertFalse(validate_train_result(
            trades=100, win_rate=0.45, profit_factor=5.0, max_drawdown=0.20
        ))

        # Robustez com boas métricas deve passar
        self.assertTrue(validate_robustness(
            pf_train=1.5, pf_test=1.4, wr_train=0.50, wr_test=0.48
        ))


class TestExecutionCosts(unittest.TestCase):
    """Testes para config/execution_costs.py"""

    def test_costs_exist(self):
        """Verifica se custos centralizados existem"""
        from config.execution_costs import SPREAD_PIPS, SLIPPAGE_PIPS

        self.assertIsInstance(SPREAD_PIPS, float)
        self.assertIsInstance(SLIPPAGE_PIPS, float)

    def test_costs_realistic(self):
        """Verifica se custos são realistas"""
        from config.execution_costs import SPREAD_PIPS, SLIPPAGE_PIPS

        # Spread deve estar entre 0.5 e 5 pips
        self.assertGreater(SPREAD_PIPS, 0.5)
        self.assertLess(SPREAD_PIPS, 5.0)

        # Slippage deve estar entre 0 e 3 pips
        self.assertGreaterEqual(SLIPPAGE_PIPS, 0)
        self.assertLess(SLIPPAGE_PIPS, 3.0)

    def test_pip_values(self):
        """Verifica valores de pip por par"""
        from config.execution_costs import get_pip_value

        # EURUSD deve ser 0.0001
        self.assertEqual(get_pip_value("EURUSD"), 0.0001)

        # USDJPY deve ser 0.01
        self.assertEqual(get_pip_value("USDJPY"), 0.01)


class TestDSGIndicator(unittest.TestCase):
    """Testes para o indicador DSG"""

    @classmethod
    def setUpClass(cls):
        """Setup executado uma vez para toda a classe"""
        from strategies.alta_volatilidade.dsg_detector_singularidade import (
            DetectorSingularidadeGravitacional
        )
        cls.DSG = DetectorSingularidadeGravitacional

    def test_initialization(self):
        """Testa inicialização do indicador"""
        dsg = self.DSG()

        # Verifica parâmetros padrão
        self.assertEqual(dsg.c_base, 1.0)
        self.assertEqual(dsg.gamma, 0.1)
        self.assertEqual(dsg.lookback_window, 50)

        # CORREÇÃO V3.4: Threshold de Ricci na escala correta
        self.assertLess(dsg.ricci_collapse_threshold, -50000)

    def test_analyze_returns_dict(self):
        """Testa se analyze retorna dicionário com campos esperados"""
        dsg = self.DSG()
        np.random.seed(42)
        prices = 1.1 + 0.0002 * np.cumsum(np.random.randn(100))

        result = dsg.analyze(prices)

        # Campos obrigatórios
        self.assertIn('Ricci_Scalar', result)
        self.assertIn('Tidal_Force_Magnitude', result)
        self.assertIn('Event_Horizon_Distance', result)
        self.assertIn('signal', result)
        self.assertIn('signal_name', result)
        self.assertIn('confidence', result)

    def test_signal_values(self):
        """Testa se sinais são válidos (-1, 0, 1)"""
        dsg = self.DSG()
        np.random.seed(42)
        prices = 1.1 + 0.0002 * np.cumsum(np.random.randn(100))

        result = dsg.analyze(prices)

        self.assertIn(result['signal'], [-1, 0, 1])
        self.assertIn(result['signal_name'], ['COMPRA', 'NEUTRO', 'VENDA'])

    def test_no_nan_in_output(self):
        """Testa se não há NaN nos outputs principais"""
        dsg = self.DSG()
        np.random.seed(42)
        prices = 1.1 + 0.0002 * np.cumsum(np.random.randn(100))

        result = dsg.analyze(prices)

        self.assertFalse(np.isnan(result['Ricci_Scalar']))
        self.assertFalse(np.isnan(result['Tidal_Force_Magnitude']))
        self.assertFalse(np.isnan(result['confidence']))

    def test_thread_safety(self):
        """Testa se há lock para thread-safety"""
        dsg = self.DSG()

        # Verifica se lock existe
        self.assertTrue(hasattr(dsg, '_lock'))

    def test_ricci_threshold_scale(self):
        """Testa se threshold de Ricci está na escala correta"""
        dsg = self.DSG()
        np.random.seed(42)
        prices = 1.1 + 0.0002 * np.cumsum(np.random.randn(100))

        result = dsg.analyze(prices)
        ricci = result['Ricci_Scalar']

        # Ricci deve estar na mesma ordem de grandeza que o threshold
        # Se threshold é -50500, Ricci deve estar perto disso
        self.assertLess(abs(ricci), 1e6)  # Não deve ser absurdamente grande


class TestDSGStrategy(unittest.TestCase):
    """Testes para a estratégia DSG"""

    @classmethod
    def setUpClass(cls):
        """Setup executado uma vez para toda a classe"""
        from strategies.alta_volatilidade.dsg_strategy import DSGStrategy
        cls.Strategy = DSGStrategy

    def test_initialization(self):
        """Testa inicialização da estratégia"""
        strategy = self.Strategy()

        self.assertEqual(strategy.name, "DSG-SingularidadeGravitacional")
        self.assertEqual(strategy.min_prices, 100)
        self.assertIsNotNone(strategy.dsg)

    def test_has_lock(self):
        """Testa se estratégia tem lock próprio"""
        strategy = self.Strategy()

        # CORREÇÃO V3.4: Lock próprio
        self.assertTrue(hasattr(strategy, '_strategy_lock'))

    def test_reset_clears_buffers(self):
        """Testa se reset limpa todos os buffers"""
        strategy = self.Strategy()

        # Adiciona alguns preços
        for i in range(10):
            strategy.add_price(1.1 + 0.0001 * i)

        self.assertGreater(len(strategy.prices), 0)

        # Reset
        strategy.reset()

        self.assertEqual(len(strategy.prices), 0)
        self.assertEqual(len(strategy.bid_volumes), 0)
        self.assertEqual(len(strategy.ask_volumes), 0)

    def test_add_price_generates_volumes(self):
        """Testa se add_price gera volumes automaticamente"""
        strategy = self.Strategy()

        # Adiciona preços sem volumes
        for i in range(5):
            strategy.add_price(1.1 + 0.0001 * i)

        # Volumes devem ser gerados automaticamente
        self.assertEqual(len(strategy.prices), 5)
        self.assertEqual(len(strategy.bid_volumes), 5)
        self.assertEqual(len(strategy.ask_volumes), 5)


class TestVolumeGenerator(unittest.TestCase):
    """Testes para o gerador de volumes"""

    def test_volume_generator_exists(self):
        """Testa se gerador de volumes existe"""
        from config.volume_generator import generate_synthetic_volumes

        np.random.seed(42)
        prices = 1.1 + 0.0002 * np.cumsum(np.random.randn(100))

        bid_vols, ask_vols = generate_synthetic_volumes(prices)

        self.assertEqual(len(bid_vols), len(prices))
        self.assertEqual(len(ask_vols), len(prices))

    def test_volumes_positive(self):
        """Testa se volumes são sempre positivos"""
        from config.volume_generator import generate_synthetic_volumes

        np.random.seed(42)
        prices = 1.1 + 0.0002 * np.cumsum(np.random.randn(100))

        bid_vols, ask_vols = generate_synthetic_volumes(prices)

        self.assertTrue(np.all(bid_vols >= 0))
        self.assertTrue(np.all(ask_vols >= 0))

    def test_volumes_no_lookahead(self):
        """Testa se volumes não usam look-ahead"""
        from config.volume_generator import generate_synthetic_volumes

        np.random.seed(42)
        prices = 1.1 + 0.0002 * np.cumsum(np.random.randn(100))

        bid_vols, _ = generate_synthetic_volumes(prices)

        # Volume[i] deve depender apenas de prices[i-1] e prices[i-2]
        # Volume[0] e Volume[1] devem ser valores base (sem histórico)
        # Verificar que volume não muda se alterarmos preços futuros
        prices_modified = prices.copy()
        prices_modified[50:] = prices_modified[50:] + 0.01

        bid_vols_modified, _ = generate_synthetic_volumes(prices_modified)

        # Volumes até índice 49 devem ser idênticos (não olham para frente)
        np.testing.assert_array_equal(bid_vols[:50], bid_vols_modified[:50])


def run_all_tests():
    """Executa todos os testes"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Adiciona todas as classes de teste
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizerFilters))
    suite.addTests(loader.loadTestsFromTestCase(TestExecutionCosts))
    suite.addTests(loader.loadTestsFromTestCase(TestDSGIndicator))
    suite.addTests(loader.loadTestsFromTestCase(TestDSGStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestVolumeGenerator))

    # Executa
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
