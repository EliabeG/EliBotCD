"""
Adaptador de Estratégia para o Riemannian Curvature Tensor Flow
Integra o indicador RCTF com o sistema de trading
"""
from datetime import datetime
from typing import Optional, Tuple
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .rctf_riemannian_curvature import RiemannianCurvatureTensorFlow


class RCTFStrategy(BaseStrategy):
    """
    Estratégia baseada no Riemannian Curvature Tensor Flow (RCTF)

    Usa Geometria Diferencial para modelar o mercado como uma variedade
    Riemanniana. Calcula a curvatura do espaço preço-tempo-volume e
    detecta reversões quando a geodésica se rompe.

    Conceitos-chave:
    - Tensor Métrico g_ij: Define distâncias ponderadas por volume
    - Símbolos de Christoffel: Conexão de Levi-Civita
    - Escalar de Ricci: Densidade de energia geométrica
    - Desvio Geodésico: Equação de Jacobi para convergência
    """

    def __init__(self,
                 min_prices: int = 100,
                 stop_loss_pips: float = 20.0,
                 take_profit_pips: float = 40.0,
                 volume_weight: float = 1.0,
                 ricci_positive_threshold: float = 0.5,
                 ricci_negative_threshold: float = -0.5,
                 jacobi_threshold: float = 0.1,
                 lookback_window: int = 20):
        """
        Inicializa a estratégia RCTF

        Args:
            min_prices: Mínimo de preços necessários para análise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            volume_weight: Peso do volume na métrica
            ricci_positive_threshold: Limiar para curvatura positiva
            ricci_negative_threshold: Limiar para curvatura negativa
            jacobi_threshold: Threshold para aceleração de Jacobi
            lookback_window: Janela de lookback para cálculos
        """
        super().__init__(name="RCTF-RiemannianCurvature")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de preços e volumes
        self.prices = deque(maxlen=600)
        self.volumes = deque(maxlen=600)

        # Indicador RCTF
        self.rctf = RiemannianCurvatureTensorFlow(
            volume_weight=volume_weight,
            ricci_positive_threshold=ricci_positive_threshold,
            ricci_negative_threshold=ricci_negative_threshold,
            jacobi_threshold=jacobi_threshold,
            lookback_window=lookback_window,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def add_price(self, price: float, volume: float = None):
        """Adiciona um preço ao buffer"""
        self.prices.append(price)
        if volume is not None:
            self.volumes.append(volume)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em RCTF

        Args:
            price: Preço atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (pode incluir 'volume')

        Returns:
            Signal se singularidade geométrica detectada, None caso contrário
        """
        # Extrai volume se disponível
        volume = indicators.get('volume', None)

        # Adiciona preço ao buffer
        self.add_price(price, volume)

        # Verifica se temos dados suficientes
        if len(self.prices) < self.min_prices:
            return None

        # Cooldown para evitar sinais em sequência
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        # Converte para numpy arrays
        prices_array = np.array(self.prices)
        volumes_array = np.array(self.volumes) if len(self.volumes) > 0 else None

        try:
            # Executa análise RCTF
            result = self.rctf.analyze(prices_array, volumes_array)
            self.last_analysis = result

            # Verifica sinal
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
                self.signal_cooldown = 15  # Cooldown para RCTF

                return signal

        except Exception as e:
            print(f"Erro na análise RCTF: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descrição do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"RCTF Riemann | "
                f"Geom={result['geometry_type']} | "
                f"R={result['ricci_normalized']:.3f} | "
                f"J={result['jacobi_acceleration']:.4f} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estratégia"""
        self.prices.clear()
        self.volumes.clear()
        self.rctf.reset()
        self.last_analysis = None
        self.last_signal = None
        self.signal_cooldown = 0

    def get_analysis_summary(self) -> Optional[dict]:
        """Retorna resumo da última análise"""
        if self.last_analysis is None:
            return None

        return {
            'signal': self.last_analysis['signal_name'],
            'confidence': self.last_analysis['confidence'],
            'geometry_type': self.last_analysis['geometry_type'],
            'ricci_scalar': self.last_analysis['ricci_scalar'],
            'ricci_normalized': self.last_analysis['ricci_normalized'],
            'geodesic_deviation': self.last_analysis['geodesic_deviation'],
            'jacobi_acceleration': self.last_analysis['jacobi_acceleration'],
            'metric_determinant': self.last_analysis['metric_determinant'],
            'convergence_rate': self.last_analysis['convergence_rate'],
            'price_trend': self.last_analysis['price_trend'],
            'current_price': self.last_analysis['current_price'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_curvature_info(self) -> Optional[dict]:
        """Retorna informações sobre a curvatura"""
        if self.last_analysis is None:
            return None
        return {
            'ricci_scalar': self.last_analysis['ricci_scalar'],
            'ricci_normalized': self.last_analysis['ricci_normalized'],
            'geometry_type': self.last_analysis['geometry_type']
        }

    def get_geodesic_info(self) -> Optional[dict]:
        """Retorna informações sobre a geodésica"""
        if self.last_analysis is None:
            return None
        return {
            'geodesic_deviation': self.last_analysis['geodesic_deviation'],
            'jacobi_acceleration': self.last_analysis['jacobi_acceleration'],
            'convergence_rate': self.last_analysis['convergence_rate']
        }

    def get_metric_info(self) -> Optional[dict]:
        """Retorna informações sobre o tensor métrico"""
        if self.last_analysis is None:
            return None
        return {
            'determinant': self.last_analysis['metric_determinant'],
            'eigenvalues': self.last_analysis.get('metric_eigenvalues', [])
        }

    def get_ricci_history(self) -> Optional[np.ndarray]:
        """Retorna histórico do escalar de Ricci"""
        return self.rctf.get_ricci_history()

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saída principal

        Returns:
            Lista com [ricci_norm, jacobi_accel, geodesic_dev, convergence]
        """
        if self.last_analysis is None:
            return None
        return [
            self.last_analysis['ricci_normalized'],
            self.last_analysis['jacobi_acceleration'],
            self.last_analysis['geodesic_deviation'],
            self.last_analysis['convergence_rate']
        ]

    def is_spherical_geometry(self) -> bool:
        """
        Verifica se a geometria é esférica (compressão)

        Returns:
            True se R > threshold (curvatura positiva)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['geometry_type'] == 'SPHERICAL'

    def is_hyperbolic_geometry(self) -> bool:
        """
        Verifica se a geometria é hiperbólica (expansão)

        Returns:
            True se R < threshold (curvatura negativa)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['geometry_type'] == 'HYPERBOLIC'

    def is_flat_geometry(self) -> bool:
        """
        Verifica se a geometria é plana (geodésica estável)

        Returns:
            True se |R| < banda neutra
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['geometry_type'] == 'FLAT'

    def get_ricci_scalar(self) -> Optional[float]:
        """
        Retorna o escalar de Ricci atual

        Returns:
            Valor de R (curvatura escalar)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['ricci_scalar']

    def get_ricci_normalized(self) -> Optional[float]:
        """
        Retorna o escalar de Ricci normalizado

        Returns:
            Valor de R / std(R)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['ricci_normalized']

    def get_jacobi_acceleration(self) -> Optional[float]:
        """
        Retorna a aceleração de Jacobi

        Returns:
            D²J/dt² do desvio geodésico
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['jacobi_acceleration']

    def is_geodesic_converging(self) -> bool:
        """
        Verifica se as geodésicas estão convergindo

        Returns:
            True se taxa de convergência > 0
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['convergence_rate'] > 0

    def is_geodesic_diverging(self) -> bool:
        """
        Verifica se as geodésicas estão divergindo

        Returns:
            True se taxa de convergência < 0
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['convergence_rate'] < 0

    def get_geometry_type(self) -> Optional[str]:
        """
        Retorna o tipo de geometria atual

        Returns:
            FLAT, SPHERICAL ou HYPERBOLIC
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['geometry_type']

