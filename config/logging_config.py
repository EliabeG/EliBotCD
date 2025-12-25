"""
================================================================================
CONFIGURAÇÃO DE LOGGING CENTRALIZADA - DSG
================================================================================

Este módulo configura logging estruturado para todo o sistema DSG.
Substitui uso de print() por logging adequado para produção.

NÍVEIS DE LOG:
- DEBUG: Detalhes técnicos para desenvolvimento
- INFO: Eventos normais de operação
- WARNING: Situações inesperadas que não impedem operação
- ERROR: Erros que afetam funcionalidade
- CRITICAL: Erros que impedem operação

USO:
    from config.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Mensagem de informação")
    logger.error("Erro ocorreu", exc_info=True)

Última Atualização: Dezembro 2025
Versão: 1.0 (Criado para correção de auditoria V3.4)
"""

import logging
import os
from datetime import datetime
from typing import Optional


# =============================================================================
# CONFIGURAÇÕES PADRÃO
# =============================================================================

DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Diretório para arquivos de log
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


# =============================================================================
# CONFIGURAÇÃO DE HANDLERS
# =============================================================================

def _ensure_log_dir():
    """Cria diretório de logs se não existir."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)


def _get_file_handler(name: str = "dsg") -> logging.FileHandler:
    """
    Cria handler para arquivo de log.

    Args:
        name: Nome base do arquivo de log

    Returns:
        FileHandler configurado
    """
    _ensure_log_dir()
    date_str = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(LOG_DIR, f"{name}_{date_str}.log")

    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setLevel(logging.DEBUG)  # Arquivo captura tudo
    handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    return handler


def _get_console_handler() -> logging.StreamHandler:
    """
    Cria handler para console.

    Returns:
        StreamHandler configurado
    """
    handler = logging.StreamHandler()
    handler.setLevel(DEFAULT_LOG_LEVEL)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    return handler


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

_loggers = {}  # Cache de loggers


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Obtém logger configurado para o módulo especificado.

    Args:
        name: Nome do módulo (geralmente __name__)
        level: Nível de log opcional (usa DEFAULT_LOG_LEVEL se None)

    Returns:
        Logger configurado

    Exemplo:
        logger = get_logger(__name__)
        logger.info("Iniciando análise DSG")
        logger.error("Erro na análise", exc_info=True)
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level or DEFAULT_LOG_LEVEL)

    # Evita duplicação de handlers
    if not logger.handlers:
        logger.addHandler(_get_console_handler())
        # Adiciona file handler apenas para módulos DSG
        if 'dsg' in name.lower() or 'optimizer' in name.lower():
            try:
                logger.addHandler(_get_file_handler())
            except Exception:
                pass  # Ignora se não conseguir criar arquivo

    # Evita propagação para root logger
    logger.propagate = False

    _loggers[name] = logger
    return logger


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def set_log_level(level: int):
    """
    Define nível de log global para todos os loggers DSG.

    Args:
        level: Nível de log (logging.DEBUG, logging.INFO, etc.)
    """
    global DEFAULT_LOG_LEVEL
    DEFAULT_LOG_LEVEL = level

    for logger in _loggers.values():
        logger.setLevel(level)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(level)


def enable_debug():
    """Ativa modo debug (mostra todos os detalhes)."""
    set_log_level(logging.DEBUG)


def enable_quiet():
    """Ativa modo silencioso (apenas warnings e erros)."""
    set_log_level(logging.WARNING)


# =============================================================================
# LOGGERS PRÉ-CONFIGURADOS PARA MÓDULOS PRINCIPAIS
# =============================================================================

# Logger para o indicador DSG
dsg_indicator_logger = get_logger("dsg.indicator")

# Logger para a estratégia DSG
dsg_strategy_logger = get_logger("dsg.strategy")

# Logger para otimizadores
optimizer_logger = get_logger("dsg.optimizer")

# Logger para backtest
backtest_logger = get_logger("dsg.backtest")


if __name__ == "__main__":
    # Teste de logging
    logger = get_logger("test")
    logger.debug("Mensagem DEBUG")
    logger.info("Mensagem INFO")
    logger.warning("Mensagem WARNING")
    logger.error("Mensagem ERROR")
    logger.critical("Mensagem CRITICAL")

    print(f"\nLogs salvos em: {LOG_DIR}")
