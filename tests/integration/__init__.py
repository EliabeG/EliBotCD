"""
================================================================================
INTEGRATION TESTS
Testes de Integracao
================================================================================

Testes de integracao entre componentes do sistema.

Categorias:
- test_api_integration.py: Integracao com API do broker
- test_database_integration.py: Integracao com banco de dados
- test_strategy_execution.py: Estrategia + Execucao
- test_end_to_end.py: Fluxo completo

Cenarios testados:
- Conexao e autenticacao com broker
- Envio e confirmacao de ordens
- Recepcao de dados em tempo real
- Persistencia de trades
- Recovery apos falha

Ambiente:
- Usar ambiente de sandbox/demo
- Dados reais ou simulados
- Timeout adequado para operacoes de rede

Tempo esperado:
- < 30s por teste
- Suite completa < 5min
"""

__all__ = []
