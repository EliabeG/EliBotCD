"""
Adaptador de Estratégia para o Sintetizador de Topos Grothendieck-Kolmogorov
Integra o indicador STGK com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .stgk_sintetizador_topos import SintetizadorToposGrothendieckKolmogorov


class STGKStrategy(BaseStrategy):
    """
    Estratégia baseada no Sintetizador de Topos Grothendieck-Kolmogorov (STGK)

    Usa Teoria das Categorias (Feixes e Cohomologia de Čech) e Complexidade
    de Kolmogorov para detectar obstruções cohomológicas e movimentos
    artificiais vs orgânicos no mercado.
    """

    def __init__(self,
                 min_prices: int = 100,
                 stop_loss_pips: float = 25.0,
                 take_profit_pips: float = 50.0,
                 timeframes: List[int] = None,
                 complexity_window: int = 100):
        """
        Inicializa a estratégia STGK

        Args:
            min_prices: Mínimo de preços necessários para análise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            timeframes: Lista de timeframes para análise categórica
            complexity_window: Janela para complexidade de Kolmogorov
        """
        super().__init__(name="STGK-ToposGrothendieck")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de preços
        self.prices = deque(maxlen=800)

        # Indicador STGK
        if timeframes is None:
            timeframes = [1, 5, 15]
        self.stgk = SintetizadorToposGrothendieckKolmogorov(
            timeframes=timeframes,
            complexity_window=complexity_window
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def add_price(self, price: float):
        """Adiciona um preço ao buffer"""
        self.prices.append(price)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em STGK

        Args:
            price: Preço atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais

        Returns:
            Signal se H¹ colapsar + K adequado, None caso contrário
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
            # Executa análise STGK
            result = self.stgk.analyze(prices_array)
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
                self.signal_cooldown = 20  # Cooldown para STGK

                return signal

        except Exception as e:
            print(f"Erro na análise STGK: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descrição do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"STGK Topos | "
                f"B1={result['Betti_Number_H1']} | "
                f"K={result['Kolmogorov_Complexity_Index']:.3f} | "
                f"H1={'Obst' if result['has_obstruction'] else 'Livre'} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estratégia"""
        self.prices.clear()
        self.last_analysis = None
        self.last_signal = None
        self.signal_cooldown = 0
        # Reseta histórico do indicador
        self.stgk._H1_history = []
        self.stgk._K_history = []

    def get_analysis_summary(self) -> Optional[dict]:
        """Retorna resumo da última análise"""
        if self.last_analysis is None:
            return None

        return {
            'signal': self.last_analysis['signal_name'],
            'confidence': self.last_analysis['confidence'],
            'betti_number_H1': self.last_analysis['Betti_Number_H1'],
            'kolmogorov_complexity': self.last_analysis['Kolmogorov_Complexity_Index'],
            'solomonoff_prior': self.last_analysis['Solomonoff_Prior'],
            'has_obstruction': self.last_analysis['has_obstruction'],
            'unified_trend': self.last_analysis['unified_trend'],
            'H0_consistency': self.last_analysis['H0']['consistency'],
            'H1_obstruction': self.last_analysis['H1']['obstruction'],
            'is_artificial': self.last_analysis['is_artificial'],
            'is_organic': self.last_analysis['is_organic'],
            'diagram_commutes': self.last_analysis['diagram_commutes'],
            'gluing_error': self.last_analysis['gluing_error'],
            'current_price': self.last_analysis['current_price'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_cohomology_info(self) -> Optional[dict]:
        """Retorna informações sobre a cohomologia de Čech"""
        if self.last_analysis is None:
            return None
        return {
            'H0': self.last_analysis['H0'],
            'H1': self.last_analysis['H1'],
            'betti_number': self.last_analysis['Betti_Number_H1'],
            'has_obstruction': self.last_analysis['has_obstruction'],
            'unified_trend': self.last_analysis['unified_trend']
        }

    def get_complexity_info(self) -> Optional[dict]:
        """Retorna informações sobre a complexidade de Kolmogorov"""
        if self.last_analysis is None:
            return None
        return {
            'K_combined': self.last_analysis['Kolmogorov_Complexity_Index'],
            'M_solomonoff': self.last_analysis['Solomonoff_Prior'],
            'is_artificial': self.last_analysis['is_artificial'],
            'is_organic': self.last_analysis['is_organic'],
            'movement': self.last_analysis['movement_analysis']
        }

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saída principal [B1, K, M]

        Returns:
            Lista com [Betti_Number_H1, Kolmogorov_Complexity_Index, Solomonoff_Prior]
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('output_vector')

    def has_cohomological_obstruction(self) -> bool:
        """
        Verifica se há obstrução cohomológica (H¹ ≠ 0)

        Returns:
            True se o "tecido do mercado está se rasgando"
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis.get('has_obstruction', False)

    def is_movement_artificial(self) -> bool:
        """
        Verifica se o movimento é artificial (HFT)

        Returns:
            True se K(x) baixo indica manipulação
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis.get('is_artificial', False)

    def is_movement_organic(self) -> bool:
        """
        Verifica se o movimento é orgânico (sustentável)

        Returns:
            True se K(x) alto indica movimento de humanos heterogêneos
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis.get('is_organic', False)

    def diagram_is_commutative(self) -> bool:
        """
        Verifica se o diagrama categórico comuta

        Returns:
            True se não há oportunidade de arbitragem entre timeframes
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis.get('diagram_commutes', True)

    def get_market_sustainability(self) -> Optional[str]:
        """
        Retorna a sustentabilidade do movimento atual

        Returns:
            'ROBUSTO', 'FRAGIL', 'INCERTO' ou 'IRRELEVANTE'
        """
        if self.last_analysis is None:
            return None
        mov = self.last_analysis.get('movement_analysis', {})
        return mov.get('sustainability', 'UNKNOWN')
