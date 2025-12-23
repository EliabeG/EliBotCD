"""
Adaptador de Estrategia para o Fisher-Shannon Information Gravity Engine
Integra o indicador FSIGE com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List, Dict
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .fsige_fisher_shannon import FisherShannonInformationGravityEngine, ThermodynamicState


class FSIGEStrategy(BaseStrategy):
    """
    Estrategia baseada no Fisher-Shannon Information Gravity Engine (FSIGE)

    Usa Geometria da Informacao e Termodinamica Estatistica para prever
    movimentos de preco baseados na maximizacao de entropia.

    Conceitos-chave:
    - Variedade Estatistica: Mercado como manifold de parametros theta
    - Tensor de Fisher: Mede "distancia informacional" entre estados
    - Forca Entropica: F_e = T*grad(S) - forca emergente estatistica
    - Tensao Termodinamica: Preco vs Gradiente Entropico
    - 2a Lei: Mercado colapsa para maxima entropia
    """

    def __init__(self,
                 min_prices: int = 100,
                 stop_loss_pips: float = 20.0,
                 take_profit_pips: float = 40.0,
                 kde_window: int = 100,
                 n_simulations: int = 500,
                 entropy_horizon: int = 10,
                 temperature_stable_threshold: float = 0.3,
                 tension_threshold: float = 0.5):
        """
        Inicializa a estrategia FSIGE

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            kde_window: Janela para KDE
            n_simulations: Simulacoes Monte Carlo para forca entropica
            entropy_horizon: Horizonte de previsao entropica
            temperature_stable_threshold: Limiar para temperatura estavel
            tension_threshold: Limiar de tensao para trigger
        """
        super().__init__(name="FSIGE-InformationGravity")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de precos
        self.prices = deque(maxlen=600)

        # Indicador FSIGE
        self.fsige = FisherShannonInformationGravityEngine(
            kde_window=kde_window,
            n_parameters=4,
            regularization=1e-4,
            n_simulations=n_simulations,
            entropy_horizon=entropy_horizon,
            temperature_stable_threshold=temperature_stable_threshold,
            tension_threshold=tension_threshold,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def add_price(self, price: float):
        """Adiciona um preco ao buffer"""
        self.prices.append(price)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em FSIGE

        Args:
            price: Preco atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais

        Returns:
            Signal se colapso entropico detectado, None caso contrario
        """
        # Adiciona preco ao buffer
        self.add_price(price)

        # Verifica se temos dados suficientes
        if len(self.prices) < self.min_prices:
            return None

        # Cooldown para evitar sinais em sequencia
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        # Converte para numpy array
        prices_array = np.array(self.prices)

        try:
            # Executa analise FSIGE
            result = self.fsige.analyze(prices_array)
            self.last_analysis = result

            # Verifica sinal (ignora HIBERNATE e NEUTRAL)
            if result['signal'] != 0 and result['confidence'] >= 0.3:
                # Determina direcao
                if result['signal'] == 1:
                    direction = SignalType.BUY
                else:
                    direction = SignalType.SELL

                # Calcula niveis de stop e take profit
                pip_value = 0.0001

                if direction == SignalType.BUY:
                    stop_loss = price - (self.stop_loss_pips * pip_value)
                    take_profit = price + (self.take_profit_pips * pip_value)
                else:
                    stop_loss = price + (self.stop_loss_pips * pip_value)
                    take_profit = price - (self.take_profit_pips * pip_value)

                # Confianca
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
                self.signal_cooldown = 25  # Cooldown para FSIGE

                return signal

        except Exception as e:
            print(f"Erro na analise FSIGE: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"FSIGE InfoGrav | "
                f"State={result['thermodynamic_state']} | "
                f"Tension={result['thermodynamic_tension']:.3f} | "
                f"Fe={result['entropic_force_direction']} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.prices.clear()
        self.fsige.reset()
        self.last_analysis = None
        self.last_signal = None
        self.signal_cooldown = 0

    def get_analysis_summary(self) -> Optional[dict]:
        """Retorna resumo da ultima analise"""
        if self.last_analysis is None:
            return None

        return {
            'signal': self.last_analysis['signal_name'],
            'confidence': self.last_analysis['confidence'],
            'thermodynamic_state': self.last_analysis['thermodynamic_state'],
            'fisher_curvature': self.last_analysis['fisher_curvature'],
            'information_resistance': self.last_analysis['information_resistance'],
            'entropic_force_magnitude': self.last_analysis['entropic_force_magnitude'],
            'entropic_force_direction': self.last_analysis['entropic_force_direction'],
            'temperature': self.last_analysis['temperature'],
            'entropy': self.last_analysis['entropy'],
            'free_energy': self.last_analysis['free_energy'],
            'thermodynamic_tension': self.last_analysis['thermodynamic_tension'],
            'tension_derivative': self.last_analysis['tension_derivative'],
            'manipulation_score': self.last_analysis['manipulation_score'],
            'current_price': self.last_analysis['current_price'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_fisher_info(self) -> Optional[dict]:
        """Retorna informacoes do tensor de Fisher"""
        if self.last_analysis is None:
            return None
        return {
            'curvature': self.last_analysis['fisher_curvature'],
            'resistance': self.last_analysis['information_resistance']
        }

    def get_entropic_force_info(self) -> Optional[dict]:
        """Retorna informacoes da forca entropica"""
        if self.last_analysis is None:
            return None
        return {
            'magnitude': self.last_analysis['entropic_force_magnitude'],
            'direction': self.last_analysis['entropic_force_direction']
        }

    def get_thermodynamic_info(self) -> Optional[dict]:
        """Retorna informacoes termodinamicas"""
        if self.last_analysis is None:
            return None
        return {
            'state': self.last_analysis['thermodynamic_state'],
            'temperature': self.last_analysis['temperature'],
            'entropy': self.last_analysis['entropy'],
            'free_energy': self.last_analysis['free_energy'],
            'tension': self.last_analysis['thermodynamic_tension'],
            'tension_derivative': self.last_analysis['tension_derivative']
        }

    def get_manipulation_info(self) -> Optional[dict]:
        """Retorna informacoes de deteccao de manipulacao"""
        if self.last_analysis is None:
            return None
        return {
            'manipulation_score': self.last_analysis['manipulation_score']
        }

    def get_entropy_history(self) -> np.ndarray:
        """Retorna historico de entropia"""
        return self.fsige.get_entropy_history()

    def get_temperature_history(self) -> np.ndarray:
        """Retorna historico de temperatura"""
        return self.fsige.get_temperature_history()

    def get_tension_history(self) -> np.ndarray:
        """Retorna historico de tensao"""
        return self.fsige.get_tension_history()

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [tension, entropy, temperature, resistance]
        """
        if self.last_analysis is None:
            return None

        return [
            self.last_analysis['thermodynamic_tension'],
            self.last_analysis['entropy'],
            self.last_analysis['temperature'],
            self.last_analysis['information_resistance']
        ]

    def is_adiabatic(self) -> bool:
        """
        Verifica se o sistema esta em estado adiabatico

        Returns:
            True se temperatura estavel (media volatilidade)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['thermodynamic_state'] == 'ADIABATIC'

    def is_hibernating(self) -> bool:
        """
        Verifica se o indicador esta hibernando

        Returns:
            True se fora do regime adiabatico
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['signal_name'] == 'HIBERNATE'

    def is_heating(self) -> bool:
        """
        Verifica se o sistema esta aquecendo

        Returns:
            True se temperatura aumentando
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['thermodynamic_state'] == 'HEATING'

    def is_cooling(self) -> bool:
        """
        Verifica se o sistema esta esfriando

        Returns:
            True se temperatura diminuindo
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['thermodynamic_state'] == 'COOLING'

    def has_high_tension(self, threshold: float = 0.5) -> bool:
        """
        Verifica se ha alta tensao termodinamica

        Args:
            threshold: Limiar de tensao

        Returns:
            True se tensao > threshold
        """
        if self.last_analysis is None:
            return False
        return abs(self.last_analysis['thermodynamic_tension']) > threshold

    def is_manipulation_detected(self, threshold: float = 1.5) -> bool:
        """
        Verifica se manipulacao foi detectada

        Args:
            threshold: Limiar de score

        Returns:
            True se manipulation_score > threshold
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['manipulation_score'] > threshold

    def get_entropic_direction(self) -> Optional[str]:
        """
        Retorna a direcao da forca entropica

        Returns:
            BULLISH, BEARISH ou NEUTRAL
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['entropic_force_direction']

    def get_temperature(self) -> Optional[float]:
        """
        Retorna a temperatura atual do mercado

        Returns:
            Volatilidade latente
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['temperature']

    def get_entropy(self) -> Optional[float]:
        """
        Retorna a entropia de Shannon atual

        Returns:
            Entropia da distribuicao de retornos
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['entropy']

    def get_tension(self) -> Optional[float]:
        """
        Retorna a tensao termodinamica atual

        Returns:
            Tensao entre preco e gradiente entropico
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['thermodynamic_tension']

    def get_information_resistance(self) -> Optional[float]:
        """
        Retorna a resistencia informacional

        Returns:
            Resistencia a mudanca de preco (Fisher)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['information_resistance']

    def is_entropy_collapsing(self) -> bool:
        """
        Verifica se a entropia esta colapsando

        Returns:
            True se tensao revertendo com direcao definida
        """
        if self.last_analysis is None:
            return False

        tension = self.last_analysis['thermodynamic_tension']
        derivative = self.last_analysis['tension_derivative']
        direction = self.last_analysis['entropic_force_direction']

        return (abs(tension) > 0.3 and
                derivative < -0.1 and
                direction in ['BULLISH', 'BEARISH'])
