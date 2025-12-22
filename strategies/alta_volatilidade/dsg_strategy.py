"""
Adaptador de Estratégia para o Detector de Singularidade Gravitacional
Integra o indicador DSG com o sistema de trading
"""
from datetime import datetime
from typing import Optional, Dict
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .dsg_detector_singularidade import DetectorSingularidadeGravitacional


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
                 ricci_collapse_threshold: float = -0.5,
                 tidal_force_threshold: float = 0.1,
                 event_horizon_threshold: float = 0.001,
                 lookback_window: int = 50,
                 c_base: float = 1.0,
                 gamma: float = 0.1):
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
        """
        super().__init__(name="DSG-SingularidadeGravitacional")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de preços e volumes
        self.prices = deque(maxlen=600)
        self.bid_volumes = deque(maxlen=600)
        self.ask_volumes = deque(maxlen=600)

        # Indicador DSG
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

    def add_price(self, price: float, bid_vol: float = None, ask_vol: float = None):
        """Adiciona um preço e volumes ao buffer"""
        self.prices.append(price)

        # Gera volumes sintéticos se não fornecidos
        if bid_vol is not None:
            self.bid_volumes.append(bid_vol)
        else:
            # Volume baseado na variação de preço
            if len(self.prices) > 1:
                change = abs(self.prices[-1] - self.prices[-2])
                self.bid_volumes.append(change * 50000 + np.random.rand() * 100)
            else:
                self.bid_volumes.append(100)

        if ask_vol is not None:
            self.ask_volumes.append(ask_vol)
        else:
            if len(self.prices) > 1:
                change = abs(self.prices[-1] - self.prices[-2])
                self.ask_volumes.append(change * 50000 + np.random.rand() * 100)
            else:
                self.ask_volumes.append(100)

    def analyze(self, price: float, timestamp: datetime,
                bid_volume: float = None, ask_volume: float = None,
                **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal se Singularidade detectada

        Args:
            price: Preço atual
            timestamp: Timestamp do tick
            bid_volume: Volume de bid (opcional)
            ask_volume: Volume de ask (opcional)
            **indicators: Indicadores adicionais

        Returns:
            Signal se Singularidade Gravitacional detectada, None caso contrário
        """
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

            # Verifica sinal - apenas quando há singularidade real
            if result['signal'] != 0 and result['confidence'] >= 0.5:
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
                self.signal_cooldown = 30  # Cooldown maior para DSG

                return signal

        except Exception as e:
            print(f"Erro na análise DSG: {e}")

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
        """Reseta o estado da estratégia"""
        self.prices.clear()
        self.bid_volumes.clear()
        self.ask_volumes.clear()
        self.last_analysis = None
        self.last_signal = None
        self.signal_cooldown = 0
        # Reseta histórico do indicador
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
