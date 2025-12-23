"""
================================================================================
TESTS MODULE
Suite de Testes do Sistema de Trading
================================================================================

Este modulo contem todos os testes do sistema.

Estrutura:
- tests/unit/          Testes unitarios
- tests/integration/   Testes de integracao
- tests/stress/        Testes de stress/carga

Ferramentas utilizadas:
- pytest: Framework de testes
- pytest-asyncio: Suporte para testes async
- pytest-cov: Cobertura de codigo
- pytest-benchmark: Benchmarking

Convencoes:
- Arquivos: test_*.py
- Classes: Test*
- Metodos: test_*
- Fixtures: conftest.py

Execucao:
  # Todos os testes
  pytest

  # Apenas unitarios
  pytest tests/unit/

  # Com cobertura
  pytest --cov=strategies --cov-report=html

  # Benchmark
  pytest tests/stress/ --benchmark-only

Metricas de qualidade:
- Cobertura minima: 80%
- Tempo maximo por teste: 5s
- Zero falhas permitidas em CI
"""

__all__ = []
