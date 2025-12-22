"""
Adaptador de Estratégia para o Projetor Holográfico de Maldacena
Integra o indicador PHM com o sistema de trading
"""
from datetime import datetime
from typing import Optional
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .phm_projetor_holografico import ProjetorHolograficoMaldacena


class PHMStrategy(BaseStrategy):
    """
    Estratégia baseada no Projetor Holográfico de Maldacena (PHM)

    Usa correspondência AdS/CFT e redes tensoriais MERA para modelar
    o mercado como um sistema quântico de muitos corpos. Detecta
    formação de "Buracos Negros" (volatilidade extrema) através da
    Entropia de Entrelaçamento de Ryu-Takayanagi.
    """

    def __init__(self,
                 min_prices: int = 80,
                 stop_loss_pips: float = 25.0,
                 take_profit_pips: float = 50.0,
                 window_size: int = 128,
                 bond_dim: int = 8,
                 n_layers: int = 4,
                 svd_cutoff: float = 1e-5):
        """
        Inicializa a estratégia PHM

        Args:
            min_prices: Mínimo de preços necessários para análise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            window_size: Tamanho da janela para análise
            bond_dim: Dimensão do bond para MPS/MERA
            n_layers: Número de camadas MERA
            svd_cutoff: Cutoff para truncamento SVD
        """
        super().__init__(name="PHM-ProjetorHolografico")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de preços
        self.prices = deque(maxlen=600)

        # Indicador PHM
        self.phm = ProjetorHolograficoMaldacena(
            window_size=window_size,
            bond_dim=bond_dim,
            n_layers=n_layers,
            svd_cutoff=svd_cutoff
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def add_price(self, price: float):
        """Adiciona um preço ao buffer"""
        self.prices.append(price)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em PHM

        Args:
            price: Preço atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais

        Returns:
            Signal se Horizonte + Fase adequada, None caso contrário
        """
        # Adiciona preço ao buffer
        self.add_price(price)

        # Verifica se temos dados suficientes
        if len(self.prices) < self.min_prices:
            return None

        # Cooldown para evitar sinais em sequência
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        # Converte para numpy array
        prices_array = np.array(self.prices)

        try:
            # Executa análise PHM
            result = self.phm.analyze(prices_array)
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
                self.signal_cooldown = 20  # Cooldown para PHM

                return signal

        except Exception as e:
            print(f"Erro na análise PHM: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descrição do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"PHM Holographic | "
                f"Phase={result['phase_type']} | "
                f"S={result['entropy']:.3f} | "
                f"M={result['magnetization']:.3f} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estratégia"""
        self.prices.clear()
        self.last_analysis = None
        self.last_signal = None
        self.signal_cooldown = 0
        # Reseta histórico do indicador
        self.phm._entropy_history = []
        self.phm._complexity_history = []

    def get_analysis_summary(self) -> Optional[dict]:
        """Retorna resumo da última análise"""
        if self.last_analysis is None:
            return None

        return {
            'signal': self.last_analysis['signal_name'],
            'confidence': self.last_analysis['confidence'],
            'entropy': self.last_analysis['entropy'],
            'bulk_entropy': self.last_analysis['bulk_entropy'],
            'horizon_forming': self.last_analysis['horizon_forming'],
            'spike_magnitude': self.last_analysis['spike_magnitude'],
            'complexity': self.last_analysis['complexity'],
            'computational_stress': self.last_analysis['computational_stress']['stress_detected'],
            'stress_level': self.last_analysis['computational_stress']['level'],
            'phase_type': self.last_analysis['phase_type'],
            'magnetization': self.last_analysis['magnetization'],
            'correlation': self.last_analysis['correlation'],
            'phase_direction': self.last_analysis['phase_direction'],
            'mera_layers': self.last_analysis['mera_layers'],
            'current_price': self.last_analysis['current_price'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_entropy_profile(self) -> Optional[np.ndarray]:
        """Retorna o perfil de entropia de Ryu-Takayanagi"""
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('entropy_profile')

    def get_entanglement_spectra(self) -> Optional[list]:
        """Retorna os espectros de entrelaçamento das camadas MERA"""
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('entanglement_spectra')

    def get_phase_info(self) -> Optional[dict]:
        """Retorna informações sobre a fase de Ising"""
        if self.last_analysis is None:
            return None
        return {
            'phase': self.last_analysis['phase_type'],
            'magnetization': self.last_analysis['magnetization'],
            'correlation': self.last_analysis['correlation'],
            'direction': self.last_analysis['phase_direction'],
            'description': self.last_analysis['phase']['description']
        }

    def is_horizon_forming(self) -> bool:
        """
        Verifica se um horizonte de eventos está se formando

        Returns:
            True se spike de entropia detectado (buraco negro formando)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis.get('horizon_forming', False)

    def is_ferromagnetic(self) -> bool:
        """
        Verifica se o mercado está em fase ferromagnética (tendência)

        Returns:
            True se magnetização alta e correlação forte
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis.get('phase_type') == 'FERROMAGNETICO'

    def is_paramagnetic(self) -> bool:
        """
        Verifica se o mercado está em fase paramagnética (ruído)

        Returns:
            True se desordem detectada
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis.get('phase_type') == 'PARAMAGNETICO'

    def is_critical(self) -> bool:
        """
        Verifica se o mercado está no ponto crítico (transição)

        Returns:
            True se no ponto crítico entre fases
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis.get('phase_type') == 'CRITICO'

    def get_bulk_geometry(self) -> Optional[dict]:
        """
        Retorna informações sobre a geometria do Bulk (AdS)

        Returns:
            Dict com métricas do espaço Bulk
        """
        if self.last_analysis is None:
            return None
        return {
            'mera_layers': self.last_analysis['mera_layers'],
            'complexity': self.last_analysis['complexity'],
            'bulk_entropy': self.last_analysis['bulk_entropy'],
            'n_spectra': len(self.last_analysis.get('entanglement_spectra', []))
        }
