"""
Adaptador de Estratégia para o Detector de Singularidade Gravitacional
Integra o indicador DSG com o sistema de trading

VERSÃO V3.5 - CORREÇÕES DA QUINTA AUDITORIA (25/12/2025)
===========================================================
Correções aplicadas (V2.0):
1. Stop/Take passados em PIPS (não níveis de preço)
2. BacktestEngine recalcula níveis baseado no entry_price REAL
3. Removida dependência do close atual para níveis

Correções aplicadas (V3.0 - Auditoria):
4. Cooldown tornado CONFIGURÁVEL (era hardcoded 30)
5. Confidence threshold tornado CONFIGURÁVEL (era hardcoded 0.5)
6. Indicador DSG V3.0 sem look-ahead bias

Correções aplicadas (V3.1 - Auditoria Completa):
7. VOLUMES CENTRALIZADOS: Usa config/volume_generator.py para consistência
8. Indicador DSG V3.1 com validação de inputs, thread-safety, subsampling adaptativo

Correções aplicadas (V3.2 - Segunda Auditoria 24/12/2025):
9. VERIFICAÇÃO DE ERRO: Checa campo 'error' antes de usar resultado
10. Indicador DSG V3.2 com correções de look-ahead residual

Correções aplicadas (V3.3 - Terceira Auditoria 25/12/2025):
11. RESET THREAD-SAFE: Usa lock do indicador ao resetar históricos
12. Indicador DSG V3.3 com correções de eh_distance e _generate_signal

Correções aplicadas (V3.4 - Quarta Auditoria 25/12/2025):
13. LOCK PRÓPRIO: Estratégia tem lock próprio para proteger buffers
14. RESET COMPLETO: Todas as operações protegidas por lock
15. Indicador DSG V3.4 com Ricci threshold corrigido e percentil dinâmico

Correções aplicadas (V3.5 - Quinta Auditoria 25/12/2025):
16. THRESHOLD CORRIGIDO: ricci_collapse_threshold = -50500.0 (era -0.5)
17. ANALYZE THREAD-SAFE: Método analyze() protegido por lock
18. FROM_CONFIG: Método para carregar parâmetros otimizados
"""
from datetime import datetime
from typing import Optional, Dict
from collections import deque
import numpy as np
import threading  # CORREÇÃO V3.4: Lock para thread-safety
import json  # CORREÇÃO V3.5: Para carregar parâmetros de config
import os  # CORREÇÃO V3.5: Para verificar existência de arquivos

from ..base import BaseStrategy, Signal, SignalType
from .dsg_detector_singularidade import DetectorSingularidadeGravitacional

# CORREÇÃO V3.4: Usar logging estruturado ao invés de print()
try:
    from config.logging_config import get_logger
    logger = get_logger("dsg.strategy")
except ImportError:
    # Fallback para logging básico se módulo não disponível
    import logging
    logger = logging.getLogger(__name__)

# CORREÇÃO V3.1: Importar gerador de volumes centralizado
try:
    from config.volume_generator import generate_single_volume, get_volume_base
    VOLUME_GENERATOR_AVAILABLE = True
except ImportError:
    # Fallback se módulo não disponível
    VOLUME_GENERATOR_AVAILABLE = False
    VOLUME_MULTIPLIER = 10000.0
    VOLUME_BASE = 50.0

    def generate_single_volume(price_current: float, price_prev: float) -> float:
        """Fallback para geração de volume."""
        return np.abs(price_current - price_prev) * VOLUME_MULTIPLIER + VOLUME_BASE

    def get_volume_base() -> float:
        """Fallback para volume base."""
        return VOLUME_BASE


class DSGStrategy(BaseStrategy):
    """
    Estratégia baseada no Detector de Singularidade Gravitacional (DSG)

    Usa geometria pseudo-Riemanniana 4D para modelar o mercado como uma
    variedade onde o preço segue geodésicas. Detecta "singularidades"
    (pontos de alta volatilidade) através do Escalar de Ricci, Forças
    de Maré e Horizonte de Eventos.
    """

    def __init__(self,
                 min_prices: int = 100,
                 stop_loss_pips: float = 30.0,
                 take_profit_pips: float = 60.0,
                 ricci_collapse_threshold: float = -50500.0,  # CORREÇÃO V3.5: Escala real (era -0.5)
                 tidal_force_threshold: float = 0.1,
                 event_horizon_threshold: float = 0.001,
                 lookback_window: int = 50,
                 c_base: float = 1.0,
                 gamma: float = 0.1,
                 signal_cooldown_bars: int = 30,
                 min_confidence: float = 0.5):
        """
        Inicializa a estratégia DSG

        Args:
            min_prices: Mínimo de preços necessários para análise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            ricci_collapse_threshold: Limiar para colapso do escalar de Ricci
            tidal_force_threshold: Limiar para força de maré alta
            event_horizon_threshold: Limiar de distância ao horizonte
            lookback_window: Janela de lookback para cálculos
            c_base: Velocidade base da luz financeira
            gamma: Fator de acoplamento volume bid/ask
            signal_cooldown_bars: NOVO V3.0 - Barras de cooldown entre sinais (era hardcoded 30)
            min_confidence: NOVO V3.0 - Confiança mínima para gerar sinal (era hardcoded 0.5)
        """
        super().__init__(name="DSG-SingularidadeGravitacional")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # CORREÇÃO V3.0: Parâmetros agora configuráveis
        self.signal_cooldown_bars = signal_cooldown_bars
        self.min_confidence = min_confidence

        # Buffer de preços e volumes
        self.prices = deque(maxlen=600)
        self.bid_volumes = deque(maxlen=600)
        self.ask_volumes = deque(maxlen=600)

        # Indicador DSG V3.0
        self.dsg = DetectorSingularidadeGravitacional(
            c_base=c_base,
            gamma=gamma,
            ricci_collapse_threshold=ricci_collapse_threshold,
            tidal_force_threshold=tidal_force_threshold,
            event_horizon_threshold=event_horizon_threshold,
            lookback_window=lookback_window
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

        # CORREÇÃO V3.4: Lock próprio para proteger buffers da estratégia
        # Evita race conditions se reset() for chamado durante analyze()
        self._strategy_lock = threading.Lock()

    def add_price(self, price: float, bid_vol: float = None, ask_vol: float = None):
        """Adiciona um preço e volumes ao buffer"""
        self.prices.append(price)

        # CORREÇÃO V3.1: Gera volumes usando função CENTRALIZADA
        # Isso garante consistência entre estratégia, indicador, backtest e otimizador
        # ANTES: usava multiplicador 50000 (diferente do indicador que usava 1000)
        # AGORA: usa função centralizada com multiplicador 10000
        if bid_vol is not None:
            self.bid_volumes.append(bid_vol)
        else:
            # Volume determinístico usando função centralizada
            # REGRA ANTI LOOK-AHEAD: Volume[i] usa prices[i-1] e prices[i-2]
            if len(self.prices) >= 3:
                # Temos pelo menos 3 preços: usar i-1 e i-2
                vol = generate_single_volume(self.prices[-2], self.prices[-3])
                self.bid_volumes.append(vol)
            else:
                # Sem histórico suficiente: usar volume base
                self.bid_volumes.append(get_volume_base())

        if ask_vol is not None:
            self.ask_volumes.append(ask_vol)
        else:
            # Volume determinístico usando função centralizada
            if len(self.prices) >= 3:
                vol = generate_single_volume(self.prices[-2], self.prices[-3])
                self.ask_volumes.append(vol)
            else:
                self.ask_volumes.append(get_volume_base())

    def analyze(self, price: float, timestamp: datetime,
                bid_volume: float = None, ask_volume: float = None,
                **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal se Singularidade detectada

        CORREÇÃO V3.5: Método agora é thread-safe com lock

        Args:
            price: Preço atual
            timestamp: Timestamp do tick
            bid_volume: Volume de bid (opcional)
            ask_volume: Volume de ask (opcional)
            **indicators: Indicadores adicionais

        Returns:
            Signal se Singularidade Gravitacional detectada, None caso contrário
        """
        # CORREÇÃO V3.5: Proteger todo o método com lock para evitar race conditions
        with self._strategy_lock:
            # Adiciona preço e volumes ao buffer
            self.add_price(price, bid_volume, ask_volume)

            # Verifica se temos dados suficientes
            if len(self.prices) < self.min_prices:
                return None

            # Cooldown para evitar sinais em sequência
            if self.signal_cooldown > 0:
                self.signal_cooldown -= 1
                return None

            # Converte para numpy arrays
            prices_array = np.array(self.prices)
            bid_vols_array = np.array(self.bid_volumes)
            ask_vols_array = np.array(self.ask_volumes)

            try:
                # Executa análise DSG
                result = self.dsg.analyze(prices_array, bid_vols_array, ask_vols_array)
                self.last_analysis = result

                # CORREÇÃO V3.2: Verificar se análise falhou com erro
                if 'error' in result and result['error']:
                    # Análise falhou (dados inválidos, etc.)
                    # Não gerar sinal quando há erro
                    return None

                # CORREÇÃO V3.0: Usar min_confidence configurável (era hardcoded 0.5)
                if result['signal'] != 0 and result['confidence'] >= self.min_confidence:
                    # Determina direção
                    if result['signal'] == 1:
                        direction = SignalType.BUY
                    else:
                        direction = SignalType.SELL

                    # CORREÇÃO V2.0: NÃO calcular níveis de stop/take baseado no close atual!
                    # Passar apenas em PIPS - o BacktestEngine recalcula baseado no entry_price REAL
                    #
                    # ANTES (ERRADO):
                    # stop_loss = price - (self.stop_loss_pips * pip_value)  # Usa close!
                    #
                    # DEPOIS (CORRETO):
                    # Passa None para stop_loss/take_profit e usa stop_loss_pips/take_profit_pips

                    # Confiança
                    confidence = result['confidence']

                    # Cria sinal com PIPS ao invés de níveis
                    signal = Signal(
                        type=direction,
                        price=price,
                        timestamp=timestamp,
                        strategy_name=self.name,
                        confidence=confidence,
                        stop_loss=None,        # Será calculado pelo BacktestEngine
                        take_profit=None,      # Será calculado pelo BacktestEngine
                        reason=self._generate_reason(result),
                        stop_loss_pips=self.stop_loss_pips,     # Em PIPS
                        take_profit_pips=self.take_profit_pips   # Em PIPS
                    )

                    self.last_signal = signal
                    # CORREÇÃO V3.0: Usar signal_cooldown_bars configurável (era hardcoded 30)
                    self.signal_cooldown = self.signal_cooldown_bars

                    return signal

            except Exception as e:
                # CORREÇÃO V3.4: Usar logging estruturado
                logger.error(f"Erro na análise DSG: {e}", exc_info=True)

            return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descrição do motivo do sinal"""
        curvature = result['curvature_class']
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"DSG Singularity | "
                f"R={result['Ricci_Scalar']:.4f} | "
                f"Tidal={result['Tidal_Force_Magnitude']:.4f} | "
                f"Curv={curvature['class']} | "
                f"{reasons_str}")

    def reset(self):
        """
        Reseta o estado da estratégia

        CORREÇÃO V3.4: Usa lock PRÓPRIO para proteger TODOS os buffers
        ANTES: Apenas histórico do indicador era protegido
        AGORA: Buffers da estratégia também são protegidos

        Evita race conditions se reset() for chamado durante analyze()
        """
        # CORREÇÃO V3.4: Proteger TODAS as operações com lock
        with self._strategy_lock:
            self.prices.clear()
            self.bid_volumes.clear()
            self.ask_volumes.clear()
            self.last_analysis = None
            self.last_signal = None
            self.signal_cooldown = 0

        # CORREÇÃO V3.3/V3.4: Reseta histórico do indicador COM lock
        with self.dsg._lock:
            self.dsg._ricci_history = []
            self.dsg._distance_history = []
            self.dsg._coords_history = []

    def get_analysis_summary(self) -> Optional[dict]:
        """Retorna resumo da última análise"""
        if self.last_analysis is None:
            return None

        return {
            'signal': self.last_analysis['signal_name'],
            'confidence': self.last_analysis['confidence'],
            'ricci_scalar': self.last_analysis['Ricci_Scalar'],
            'tidal_force': self.last_analysis['Tidal_Force_Magnitude'],
            'event_horizon_distance': self.last_analysis['Event_Horizon_Distance'],
            'ricci_collapsing': self.last_analysis['ricci_collapsing'],
            'crossing_horizon': self.last_analysis['crossing_horizon'],
            'geodesic_direction': self.last_analysis['geodesic_direction'],
            'curvature_class': self.last_analysis['curvature_class']['class'],
            'curvature_volatility': self.last_analysis['curvature_class']['volatility'],
            'current_price': self.last_analysis['current_price'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_curvature_info(self) -> Optional[dict]:
        """Retorna informações sobre a curvatura do espaço-tempo"""
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('curvature_class')

    def get_ricci_series(self) -> Optional[np.ndarray]:
        """Retorna a série temporal do Escalar de Ricci"""
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('ricci_series')

    def get_tidal_series(self) -> Optional[np.ndarray]:
        """Retorna a série temporal da Força de Maré"""
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('tidal_series')

    def get_horizon_distance_series(self) -> Optional[np.ndarray]:
        """Retorna a série temporal da distância ao Horizonte de Eventos"""
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('distance_series')

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saída principal [R, F_tidal, d_EH]

        Returns:
            Lista com [Ricci_Scalar, Tidal_Force_Magnitude, Event_Horizon_Distance]
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('output_vector')

    def is_near_singularity(self) -> bool:
        """
        Verifica se estamos próximos de uma singularidade

        Returns:
            True se a curvatura indica volatilidade extrema
        """
        if self.last_analysis is None:
            return False

        curvature = self.last_analysis.get('curvature_class', {})
        return curvature.get('class') in ['HIPERBOLICO_LEVE', 'HIPERBOLICO_EXTREMO']

    def is_crossing_event_horizon(self) -> bool:
        """
        Verifica se estamos cruzando o horizonte de eventos

        Returns:
            True se cruzando o ponto de não-retorno
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis.get('crossing_horizon', False)

    @classmethod
    def from_config(cls, config_path: str) -> 'DSGStrategy':
        """
        CORREÇÃO V3.5: Carrega parâmetros otimizados de um arquivo JSON

        Isso garante que a estratégia em produção use EXATAMENTE os mesmos
        parâmetros que foram validados durante a otimização.

        Args:
            config_path: Caminho para arquivo JSON com parâmetros otimizados

        Returns:
            Instância de DSGStrategy com parâmetros do config

        Raises:
            FileNotFoundError: Se arquivo não existe
            ValueError: Se arquivo não tem parâmetros necessários

        Exemplo de uso:
            >>> strategy = DSGStrategy.from_config('configs/dsg_optimized.json')

        Exemplo de arquivo JSON:
            {
                "params": {
                    "ricci_collapse_threshold": -50300.0,
                    "tidal_force_threshold": 0.015,
                    "stop_loss_pips": 25.0,
                    "take_profit_pips": 50.0,
                    "event_horizon_threshold": 0.0015,
                    "lookback_window": 50,
                    "signal_cooldown_bars": 30,
                    "min_confidence": 0.5
                }
            }
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Suporta tanto formato direto quanto aninhado em 'params'
        if 'params' in config:
            params = config['params']
        else:
            params = config

        # Parâmetros obrigatórios para consistência
        required_params = ['ricci_collapse_threshold', 'tidal_force_threshold']
        for param in required_params:
            if param not in params:
                raise ValueError(f"Parâmetro obrigatório ausente: {param}")

        # Verificar escala do Ricci threshold
        if 'ricci_collapse_threshold' in params:
            threshold = params['ricci_collapse_threshold']
            if threshold > -1000:
                logger.warning(
                    f"ATENÇÃO: ricci_collapse_threshold={threshold} parece estar na escala errada. "
                    f"Valores reais estão entre -51000 e -49500."
                )

        # Extrair parâmetros com valores padrão
        return cls(
            min_prices=params.get('min_prices', 100),
            stop_loss_pips=params.get('stop_loss_pips', 30.0),
            take_profit_pips=params.get('take_profit_pips', 60.0),
            ricci_collapse_threshold=params.get('ricci_collapse_threshold', -50500.0),
            tidal_force_threshold=params.get('tidal_force_threshold', 0.1),
            event_horizon_threshold=params.get('event_horizon_threshold', 0.001),
            lookback_window=params.get('lookback_window', 50),
            c_base=params.get('c_base', 1.0),
            gamma=params.get('gamma', 0.1),
            signal_cooldown_bars=params.get('signal_cooldown_bars', 30),
            min_confidence=params.get('min_confidence', 0.5),
        )
