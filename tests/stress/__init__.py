"""
================================================================================
STRESS TESTS
Testes de Stress e Performance
================================================================================

Testes de carga e performance do sistema.

Categorias:
- test_throughput.py: Capacidade de processamento
- test_latency.py: Latencia de operacoes
- test_memory.py: Uso de memoria
- test_concurrency.py: Operacoes concorrentes
- test_recovery.py: Recuperacao sob carga

Metricas medidas:
- Ticks/segundo processados
- Latencia p50, p95, p99
- Uso de memoria (peak, avg)
- CPU utilization
- Tempo de recuperacao

Cenarios de stress:
- Alto volume de ticks (10k+/s)
- Multiplas estrategias simultaneas
- Picos de volatilidade
- Desconexao/reconexao
- Memoria limitada

Benchmarks:
- Baseline: Operacao normal
- Peak: 10x volume normal
- Sustained: 1 hora de carga alta
- Spike: Burst de 100x por 1 min

Execucao:
  pytest tests/stress/ --benchmark-only
  pytest tests/stress/ --benchmark-json=results.json
"""

__all__ = []
