# config/settings.py
"""Modulo de configuracoes globais e especificas do bot, usando Pydantic."""
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Literal

# Tentar importar load_dotenv, se falhar, continuar
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None # type: ignore

# Usar pydantic.Field para Pydantic v2
# Para Pydantic v1, os campos nao precisariam de Field(...) para defaults simples.
from pydantic import BaseModel, Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determinar o caminho base do projeto e o arquivo .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_file_path = PROJECT_ROOT / ".env"

if load_dotenv:
    if not env_file_path.exists():
        print(f"AVISO: Arquivo .env nao encontrado em {env_file_path}. Usando valores padrao ou variaveis de ambiente.")
    else:
        load_dotenv(dotenv_path=env_file_path)
        print(f"Carregado .env de {env_file_path}") # Este print e importante para debug
else:
    print("AVISO: python-dotenv nao instalado, nao foi possivel carregar .env. Usando valores padrao ou variaveis de ambiente.")


# ==============================================================================
# MODELOS DE CONFIGURACAO BASE (Pydantic)
# ==============================================================================
class RiskParamsConfig(BaseSettings):
    """Parametros de risco do sistema."""
    MIN_STOP_DISTANCE_PIPS: float = 5.0
    MAX_STOP_DISTANCE_PIPS: float = 50.0
    TRAILING_STOP_DISTANCE_PIPS: float = 8.0
    TRAILING_STOP_ACTIVATION_PIPS: float = 10.0
    COMMISSION_PER_LOT: float = 7.0
    EXPECTED_SLIPPAGE_PIPS: float = 0.1
    MAX_ALLOWED_SLIPPAGE_PIPS: float = 0.3

    model_config = SettingsConfigDict(
        env_prefix='RISK_',
        env_file=str(env_file_path) if env_file_path.exists() else None,
        env_file_encoding='utf-8',
        extra='ignore',
        validate_default=True
    )

class TradingConfig(BaseSettings):
    """Configuracoes principais do bot de trading."""
    APP_NAME: str = "TradingBotFX_Eliabe"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: str = "INFO"
    DEBUG_MODE: bool = False
    TRADING_MODE: str = "paper"  # live, paper, backtest

    SYMBOL: str = "EURUSD"
    LEVERAGE: int = 500
    MAX_SPREAD_PIP: float = 3.0
    MAX_SLIPPAGE_PIP: float = 1.5

    BASE_CURRENCY: str = "USD"
    QUOTE_CURRENCY: str = "EUR"

    # Credenciais diretas (lidas do .env com prefixo TRADING_)
    LOGIN: str = Field(default="", alias="TRADING_LOGIN")
    PASSWORD: str = Field(default="", alias="TRADING_PASSWORD")
    SERVER: str = Field(default="ttdemomarginal.fxopen.net", alias="TRADING_SERVER")

    # Web API Tokens
    WEB_API_TOKEN_ID: str = Field(default="", alias="TRADING_WEB_API_TOKEN_ID")
    WEB_API_TOKEN_KEY: str = Field(default="", alias="TRADING_WEB_API_TOKEN_KEY")
    WEB_API_TOKEN_SECRET: str = Field(default="", alias="TRADING_WEB_API_TOKEN_SECRET")
    WEB_API_AUTH_TYPE: str = Field(default="HMAC", alias="TRADING_WEB_API_AUTH_TYPE")

    # WebSocket URLs
    WS_FEED_URL: str = Field(default="wss://marginalttdemowebapi.fxopen.net/feed", alias="TRADING_WS_FEED_URL")
    WS_TRADE_URL: str = Field(default="wss://marginalttdemowebapi.fxopen.net/trade", alias="TRADING_WS_TRADE_URL")
    REST_API_URL: str = Field(default="https://marginalttdemowebapi.fxopen.net", alias="TRADING_REST_API_URL")

    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[SecretStr] = None
    REDIS_TTL_HOURS: int = 72

    PARQUET_PATH: str = str(PROJECT_ROOT / "data" / "parquet_data")
    TICK_HISTORY_YEARS: int = 1
    MAX_RECENT_TICKS: int = 20000

    NTP_SERVER: str = "a.st1.ntp.br"
    NTP_SYNC_INTERVAL_SECONDS: int = 1800
    NTP_SYNC_MINUTES: int = 5  # Para o main.py

    ORCHESTRATOR_LOOP_INTERVAL_SECONDS: float = 0.1
    HEARTBEAT_INTERVAL_SECONDS: int = 60

    # Limites de risco
    GLOBAL_MAX_DRAWDOWN_PERCENT: float = 15.0
    GLOBAL_MAX_DAILY_LOSS_PERCENT: float = 5.0
    TARGET_DAILY_PROFIT_PERCENT: float = 3.0
    DAILY_LOSS_LIMIT: float = 0.03  # 3% perda maxima diaria
    TARGET_DAILY_PROFIT: float = 0.05  # 5% meta diaria

    # Alertas
    ALERT_LATENCY_MS: int = 200
    ALERT_DD_WEEKLY: float = 0.10  # 10% drawdown semanal

    # Metricas
    METRICS_PORT: int = 9090

    MODELS_PATH: str = str(PROJECT_ROOT / "models")
    LOG_PATH: str = str(PROJECT_ROOT / "logs")
    DATA_PATH: str = str(PROJECT_ROOT / "data")

    DOM_LEVELS: int = 10

    REGIME_CONFIDENCE_THRESHOLD: float = 0.60
    REGIME_UPDATE_MS: int = 60000

    SCORE_UPDATE_MINUTES: int = 30
    SCORE_UPDATE_TRADES: int = 10
    MAX_ACTIVE_STRATEGIES: int = 3
    MIN_STRATEGY_SCORE_TO_ACTIVATE: float = 0.5

    ORDER_TIMEOUT_MS: int = 10000

    # Especificacoes de simbolos
    SYMBOL_SPECIFICATIONS: Dict[str, Dict[str, Any]] = {
        "EURUSD": {
            "min_lot": 0.01,
            "max_lot": 100.0,
            "lot_step": 0.01,
            "contract_size": 100000,
            "digits": 5,
            "pip_size": 0.0001
        },
        "XAUUSD": {
            "min_lot": 0.01,
            "max_lot": 50.0,
            "lot_step": 0.01,
            "contract_size": 100,
            "digits": 2,
            "pip_size": 0.01
        }
    }

    SESSION_CONFIG: Dict[str, Dict[str, int]] = {
        "ASIA": {"start_hour": 23, "end_hour": 8},
        "LONDON": {"start_hour": 7, "end_hour": 16},
        "NEWYORK": {"start_hour": 13, "end_hour": 22},
        "OVERLAP_ASIA_LONDON": {"start_hour": 7, "end_hour": 8},
        "OVERLAP_LONDON_NY": {"start_hour": 13, "end_hour": 16},
        "OVERLAP": {"start_hour": 13, "end_hour": 16}
    }
    FOREX_MARKET_CLOSE_HOUR_FRIDAY_UTC: int = 21
    FOREX_MARKET_OPEN_HOUR_SUNDAY_UTC: int = 21

    model_config = SettingsConfigDict(
        env_prefix='',
        env_file=str(env_file_path) if env_file_path.exists() else None,
        env_file_encoding='utf-8',
        extra='ignore',
        validate_default=True,
        populate_by_name=True  # Permite usar aliases
    )

class RegimeDetectionConfig(BaseSettings):
    """Configuracoes para o detector de regime de mercado."""
    TREND_WINDOW: int = Field(default=100, description="Janela para calculo de tendencia (ex: regressao linear, ADX).")
    VOLATILITY_WINDOW: int = Field(default=20, description="Janela para calculo de volatilidade (ex: ATR).")
    
    VOLATILITY_ATR_PERCENTILE: int = Field(default=75, description="Percentil de ATR para definir 'alta volatilidade'.")
    VOLATILITY_SPREAD_DELTA: float = Field(default=3.0, description="Spread em pips acima do qual pode indicar alta volatilidade.") 

    TREND_ADX_MIN: float = Field(default=25.0, description="Valor minimo do ADX para considerar tendencia.")
    TREND_R2_MIN: float = Field(default=0.3, description="Valor minimo do R2 da regressao linear para considerar tendencia.")
    
    RANGE_BB_LOW: float = Field(default=0.2, description="Limite inferior do %B da Bollinger Band para range (ex: 0.2).")
    RANGE_BB_HIGH: float = Field(default=0.8, description="Limite superior do %B da Bollinger Band para range (ex: 0.8).")

    RF_N_ESTIMATORS: int = Field(default=150, description="Numero de arvores no RandomForest do regime.")
    RF_MAX_DEPTH: Optional[int] = Field(default=15, description="Profundidade maxima das arvores.")
    RF_MIN_SAMPLES_SPLIT: int = Field(default=5, description="Minimo de amostras para dividir um no.")
    RF_MIN_SAMPLES_LEAF: int = Field(default=3, description="Minimo de amostras em uma folha.")

    model_config = SettingsConfigDict(env_prefix='REGIME_', env_file=str(env_file_path) if env_file_path.exists() else None, env_file_encoding='utf-8', extra='ignore', validate_default=True)

class DataManagerConfig(BaseSettings):
    RECENT_TICKS_LOOKBACK_MINUTES: int = Field(default=120, description="Minutos de lookback para buscar ticks recentes se nao estiverem no cache.")
    OHLC_LOOKBACK_DAYS: int = Field(default=30, description="Dias de lookback para calcular OHLC se os ticks nao forem fornecidos.")
    STRATEGY_PARAMS_TTL_DAYS: int = Field(default=7, description="TTL em dias para parametros de estrategia otimizados no cache Redis.")
    
    model_config = SettingsConfigDict(env_prefix='DM_', env_file=str(env_file_path) if env_file_path.exists() else None, env_file_encoding='utf-8', extra='ignore', validate_default=True)

# ==============================================================================
# INSTANCIAS DAS CONFIGURACOES (Carregar e exportar)
# ==============================================================================
try:
    CONFIG = TradingConfig()
    REGIME_CONFIG = RegimeDetectionConfig()
    DATA_MANAGER_CONFIG = DataManagerConfig()
    RISK_PARAMS = RiskParamsConfig()

    print(f"Configuracao Principal Carregada. APP_NAME: {CONFIG.APP_NAME}, SYMBOL: {CONFIG.SYMBOL}")
    print(f"  Login: {CONFIG.LOGIN}, Server: {CONFIG.SERVER}")
    print(f"  WS Feed URL: {CONFIG.WS_FEED_URL}")
    print(f"Configuracao de Regime Carregada. TREND_WINDOW: {REGIME_CONFIG.TREND_WINDOW}")
    print(f"Configuracao do Data Manager Carregada. RECENT_TICKS_LOOKBACK_MINUTES: {DATA_MANAGER_CONFIG.RECENT_TICKS_LOOKBACK_MINUTES}")
    print(f"Configuracao de Risk Params Carregada. COMMISSION_PER_LOT: {RISK_PARAMS.COMMISSION_PER_LOT}")

except Exception as e:
    print(f"ERRO CRITICO AO CARREGAR CONFIGURACOES em settings.py: {e}")
    print("   Verifique seu arquivo .env e as definicoes das classes de configuracao Pydantic.")
    import traceback
    traceback.print_exc()

    # Fallback apenas para permitir importacoes em outros modulos se settings falhar completamente
    class FallbackRiskParams(BaseModel):
        MIN_STOP_DISTANCE_PIPS: float = 5.0
        MAX_STOP_DISTANCE_PIPS: float = 50.0
        TRAILING_STOP_DISTANCE_PIPS: float = 8.0
        COMMISSION_PER_LOT: float = 7.0

    class FallbackConfig(BaseModel):
        SYMBOL: str = "EURUSD"
        PARQUET_PATH: str = "data"
        REDIS_HOST: str = "localhost"
        REDIS_PORT: int = 6379
        REDIS_DB: int = 0
        REDIS_TTL_HOURS: int = 1
        MAX_RECENT_TICKS: int = 1000
        LOGIN: str = ""
        SERVER: str = "ttdemomarginal.fxopen.net"
        WS_FEED_URL: str = "wss://marginalttdemowebapi.fxopen.net/feed"
        WS_TRADE_URL: str = "wss://marginalttdemowebapi.fxopen.net/trade"
        REST_API_URL: str = "https://marginalttdemowebapi.fxopen.net"
        WEB_API_TOKEN_ID: str = ""
        WEB_API_TOKEN_KEY: str = ""
        WEB_API_TOKEN_SECRET: str = ""
        WEB_API_AUTH_TYPE: str = "HMAC"
        LEVERAGE: int = 500
        DOM_LEVELS: int = 10
        TRADING_MODE: str = "paper"
        METRICS_PORT: int = 9090
        NTP_SYNC_MINUTES: int = 5
        ALERT_LATENCY_MS: int = 200
        ALERT_DD_WEEKLY: float = 0.10
        DAILY_LOSS_LIMIT: float = 0.03
        TARGET_DAILY_PROFIT: float = 0.05
        REGIME_CONFIDENCE_THRESHOLD: float = 0.60
        REGIME_UPDATE_MS: int = 60000
        SCORE_UPDATE_MINUTES: int = 30
        SCORE_UPDATE_TRADES: int = 10
        MAX_ACTIVE_STRATEGIES: int = 3
        ORDER_TIMEOUT_MS: int = 10000
        SYMBOL_SPECIFICATIONS: Dict[str, Dict[str, Any]] = {}
        SESSION_CONFIG: Dict[str, Dict[str, int]] = {}

    class FallbackRegime(BaseModel):
        TREND_WINDOW: int = 50

    class FallbackDMConfig(BaseModel):
        RECENT_TICKS_LOOKBACK_MINUTES: int = 60
        OHLC_LOOKBACK_DAYS: int = 10
        STRATEGY_PARAMS_TTL_DAYS: int = 7

    if 'CONFIG' not in locals() or CONFIG is None:
        CONFIG = FallbackConfig()  # type: ignore
    if 'REGIME_CONFIG' not in locals() or REGIME_CONFIG is None:
        REGIME_CONFIG = FallbackRegime()  # type: ignore
    if 'DATA_MANAGER_CONFIG' not in locals() or DATA_MANAGER_CONFIG is None:
        DATA_MANAGER_CONFIG = FallbackDMConfig()  # type: ignore
    if 'RISK_PARAMS' not in locals() or RISK_PARAMS is None:
        RISK_PARAMS = FallbackRiskParams()  # type: ignore
    print("   Usando configuracoes de fallback devido a erro.")