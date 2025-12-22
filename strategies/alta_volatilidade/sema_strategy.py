"""
Adaptador de Estratégia para o Sincronizador Espectral (SEMA)
Integra o indicador SEMA com o sistema de trading
"""
from datetime import datetime
from typing import Optional, Dict
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .sema_sincronizador_espectral import SincronizadorEspectral


class SEMAStrategy(BaseStrategy):
    """
    Estratégia baseada no Sincronizador Espectral (SEMA)

    Usa Teoria de Matrizes Aleatórias (RMT) e Teoria Espectral de Grafos
    para detectar Sincronização Global do mercado - momento de máxima
    alavancagem quando todos os ativos se movem em uníssono.
    """

    def __init__(self,
                 min_prices: int = 150,
                 stop_loss_pips: float = 25.0,
                 take_profit_pips: float = 50.0,
                 correlation_window: int = 120,
                 fiedler_percentile_threshold: float = 90,
                 entropy_drop_threshold: float = 0.3,
                 entropy_critical_threshold: float = 0.5,
                 asset_names: list = None):
        """
        Inicializa a estratégia SEMA

        Args:
            min_prices: Mínimo de preços necessários para análise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            correlation_window: Janela para matriz de correlação
            fiedler_percentile_threshold: Percentil do Fiedler para trigger
            entropy_drop_threshold: Limiar de queda de entropia
            entropy_critical_threshold: Limiar crítico de entropia
            asset_names: Lista de nomes dos ativos
        """
        super().__init__(name="SEMA-SincronizadorEspectral")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Nomes dos ativos
        if asset_names is None:
            self.asset_names = ['EURUSD', 'DXY', 'XAUUSD', 'US10Y', 'SPX500']
        else:
            self.asset_names = asset_names

        # Buffers de preços para cada ativo
        self.price_buffers: Dict[str, deque] = {
            asset: deque(maxlen=600) for asset in self.asset_names
        }

        # Indicador SEMA
        self.sema = SincronizadorEspectral(
            correlation_window=correlation_window,
            fiedler_percentile_threshold=fiedler_percentile_threshold,
            entropy_drop_threshold=entropy_drop_threshold,
            entropy_critical_threshold=entropy_critical_threshold,
            use_lanczos=True,
            asset_names=self.asset_names
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def add_price(self, asset: str, price: float):
        """Adiciona um preço ao buffer de um ativo específico"""
        if asset in self.price_buffers:
            self.price_buffers[asset].append(price)

    def add_prices(self, prices_dict: Dict[str, float]):
        """Adiciona preços de múltiplos ativos"""
        for asset, price in prices_dict.items():
            self.add_price(asset, price)

    def _get_data_dict(self) -> Dict[str, np.ndarray]:
        """Converte buffers para dicionário de arrays"""
        return {
            asset: np.array(buffer)
            for asset, buffer in self.price_buffers.items()
        }

    def _has_sufficient_data(self) -> bool:
        """Verifica se todos os ativos têm dados suficientes"""
        return all(
            len(buffer) >= self.min_prices
            for buffer in self.price_buffers.values()
        )

    def analyze(self, price: float, timestamp: datetime,
                asset_prices: Dict[str, float] = None, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal se Sincronização Global detectada

        Args:
            price: Preço atual do EURUSD (ativo principal)
            timestamp: Timestamp do tick
            asset_prices: Dicionário com preços de todos os ativos
            **indicators: Indicadores adicionais

        Returns:
            Signal se Sincronização Global detectada, None caso contrário
        """
        # Adiciona preços ao buffer
        if asset_prices:
            self.add_prices(asset_prices)
        else:
            # Se não fornecido, assume apenas EURUSD
            self.add_price('EURUSD', price)

        # Verifica se temos dados suficientes
        if not self._has_sufficient_data():
            return None

        # Cooldown para evitar sinais em sequência
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        try:
            # Prepara dados
            data = self._get_data_dict()

            # Executa análise SEMA
            result = self.sema.analyze(data)
            self.last_analysis = result

            # Verifica sinal - apenas quando há Sincronização Global
            if result['global_sync'] and result['signal'] != 0:
                # Determina direção
                if result['signal'] == 1:
                    direction = SignalType.BUY
                else:
                    direction = SignalType.SELL

                # Calcula níveis de stop e take profit
                pip_value = 0.0001

                if direction == SignalType.BUY:
                    stop_loss = price - (self.stop_loss_pips * pip_value)
                    take_profit = price + (self.take_profit_pips * pip_value)
                else:
                    stop_loss = price + (self.stop_loss_pips * pip_value)
                    take_profit = price - (self.take_profit_pips * pip_value)

                # Confiança
                confidence = result['confidence']

                # Cria sinal
                signal = Signal(
                    type=direction,
                    price=price,
                    timestamp=timestamp,
                    strategy_name=self.name,
                    confidence=confidence,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=self._generate_reason(result)
                )

                self.last_signal = signal
                self.signal_cooldown = 20  # Cooldown maior para SEMA

                return signal

        except Exception as e:
            print(f"Erro na análise SEMA: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descrição do motivo do sinal"""
        return (f"SEMA Global Sync | "
                f"Fiedler={result['fiedler_value']:.4f} (P{result['fiedler_percentile']:.0f}) | "
                f"Entropia={result['entropy']:.3f} | "
                f"Padrão={result['pattern']}")

    def reset(self):
        """Reseta o estado da estratégia"""
        for buffer in self.price_buffers.values():
            buffer.clear()
        self.last_analysis = None
        self.last_signal = None
        self.signal_cooldown = 0
        # Reseta cache do indicador
        self.sema._cache = {
            'last_correlation': None,
            'last_eigenvalues': None,
            'last_eigenvectors': None,
            'fiedler_history': [],
            'entropy_history': []
        }

    def get_analysis_summary(self) -> Optional[dict]:
        """Retorna resumo da última análise"""
        if self.last_analysis is None:
            return None

        return {
            'fiedler_value': self.last_analysis['fiedler_value'],
            'fiedler_percentile': self.last_analysis['fiedler_percentile'],
            'fiedler_condition': self.last_analysis['fiedler_condition'],
            'entropy': self.last_analysis['entropy'],
            'entropy_collapse': self.last_analysis['entropy_collapse'],
            'global_sync': self.last_analysis['global_sync'],
            'pattern': self.last_analysis['pattern'],
            'is_real_trend': self.last_analysis['is_real_trend'],
            'direction': self.last_analysis['direction'],
            'signal': self.last_analysis['signal_name'],
            'confidence': self.last_analysis['confidence'],
            'eigenvector_centrality': self.last_analysis['eigenvector_centrality']
        }

    def get_correlation_matrix(self) -> Optional[np.ndarray]:
        """Retorna a matriz de correlação limpa da última análise"""
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('correlation_clean')

    def get_eigenvector_centrality(self) -> Optional[dict]:
        """Retorna a centralidade de autovetor de cada ativo"""
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('eigenvector_centrality')
