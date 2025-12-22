"""
Adaptador de Estratégia para o Sintetizador Evolutivo de Estruturas Dissipativas
Integra o indicador SEED com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .seed_sintetizador_evolutivo import SintetizadorEvolutivoEstruturasDissipativas


class SEEDStrategy(BaseStrategy):
    """
    Estratégia baseada no Sintetizador Evolutivo de Estruturas Dissipativas (SEED)

    Usa Termodinâmica Estatística e Biologia Evolutiva para modelar o mercado
    como um sistema aberto longe do equilíbrio. Detecta Estruturas Dissipativas
    (auto-organização) através da Taxa de Produção de Entropia e Dinâmica de
    Replicador com memória evolutiva.
    """

    def __init__(self,
                 min_prices: int = 50,
                 stop_loss_pips: float = 25.0,
                 take_profit_pips: float = 50.0,
                 critical_sigma: float = 1.5,
                 slaving_threshold: float = 0.15,
                 dominance_threshold: float = 0.5,
                 fitness_window: int = 20,
                 dx_threshold: float = 0.02):
        """
        Inicializa a estratégia SEED

        Args:
            min_prices: Mínimo de preços necessários para análise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            critical_sigma: Limiar crítico para σ (regime não-linear)
            slaving_threshold: Limiar para colapso de x₃
            dominance_threshold: Limiar de dominância de espécie
            fitness_window: Janela para cálculo de fitness
            dx_threshold: Limiar para crescimento exponencial
        """
        super().__init__(name="SEED-EstruturasDissipativas")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de preços e volumes
        self.prices = deque(maxlen=600)
        self.volumes = deque(maxlen=600)

        # Indicador SEED
        self.seed = SintetizadorEvolutivoEstruturasDissipativas(
            critical_sigma=critical_sigma,
            slaving_threshold=slaving_threshold,
            dominance_threshold=dominance_threshold,
            fitness_window=fitness_window,
            dx_threshold=dx_threshold
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def add_price(self, price: float, volume: float = None):
        """Adiciona um preço e volume ao buffer"""
        self.prices.append(price)

        # Gera volume sintético se não fornecido
        if volume is not None:
            self.volumes.append(volume)
        else:
            if len(self.prices) > 1:
                change = abs(self.prices[-1] - self.prices[-2])
                self.volumes.append(change * 50000 + np.random.rand() * 1000)
            else:
                self.volumes.append(1000)

    def analyze(self, price: float, timestamp: datetime,
                volume: float = None, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em SEED

        Args:
            price: Preço atual
            timestamp: Timestamp do tick
            volume: Volume (opcional)
            **indicators: Indicadores adicionais

        Returns:
            Signal se Estrutura Dissipativa detectada, None caso contrário
        """
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
        volumes_array = np.array(self.volumes)

        try:
            # Executa análise SEED
            result = self.seed.analyze(prices_array, volumes_array)
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
                self.signal_cooldown = 25  # Cooldown para SEED

                return signal

        except Exception as e:
            print(f"Erro na análise SEED: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descrição do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"SEED Dissipative | "
                f"σ={result['sigma']:.3f} | "
                f"x1={result['x1_bulls']:.2f} x2={result['x2_bears']:.2f} | "
                f"Slaving={'Y' if result['slaving_active'] else 'N'} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estratégia"""
        self.prices.clear()
        self.volumes.clear()
        self.last_analysis = None
        self.last_signal = None
        self.signal_cooldown = 0
        # Reseta memória evolutiva do indicador
        self.seed.reset_memory()

    def get_analysis_summary(self) -> Optional[dict]:
        """Retorna resumo da última análise"""
        if self.last_analysis is None:
            return None

        return {
            'signal': self.last_analysis['signal_name'],
            'confidence': self.last_analysis['confidence'],
            'sigma': self.last_analysis['sigma'],
            'is_nonlinear_regime': self.last_analysis['is_nonlinear_regime'],
            'structure_forming': self.last_analysis['structure_forming'],
            'x1_bulls': self.last_analysis['x1_bulls'],
            'x2_bears': self.last_analysis['x2_bears'],
            'x3_mean_reverters': self.last_analysis['x3_mean_reverters'],
            'dx1_bulls': self.last_analysis['dx1_bulls'],
            'dx2_bears': self.last_analysis['dx2_bears'],
            'dx3_mean_reverters': self.last_analysis['dx3_mean_reverters'],
            'dominant_species': self.last_analysis['dominant_species'],
            'slaving_active': self.last_analysis['slaving_active'],
            'order_mode': self.last_analysis['order_mode'],
            'noise_became_signal': self.last_analysis['noise_became_signal'],
            'phase_region': self.last_analysis['phase_region'],
            'boundary_crossed': self.last_analysis['boundary_crossed'],
            'is_ignition_point': self.last_analysis['is_ignition_point'],
            'current_price': self.last_analysis['current_price'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_entropy_info(self) -> Optional[dict]:
        """Retorna informações sobre a produção de entropia"""
        if self.last_analysis is None:
            return None
        return {
            'sigma': self.last_analysis['sigma'],
            'is_nonlinear': self.last_analysis['is_nonlinear_regime'],
            'structure_forming': self.last_analysis['structure_forming'],
            'J': self.last_analysis['J'][-1] if len(self.last_analysis['J']) > 0 else 0,
            'X': self.last_analysis['X'][-1] if len(self.last_analysis['X']) > 0 else 0
        }

    def get_population_info(self) -> Optional[dict]:
        """Retorna informações sobre as populações evolutivas"""
        if self.last_analysis is None:
            return None
        return {
            'populations': self.last_analysis['populations'],
            'x1_bulls': self.last_analysis['x1_bulls'],
            'x2_bears': self.last_analysis['x2_bears'],
            'x3_mean_reverters': self.last_analysis['x3_mean_reverters'],
            'dx': self.last_analysis['dx'],
            'fitness': self.last_analysis['fitness'],
            'dominant_species': self.last_analysis['dominant_species'],
            'bulls_growing_exp': self.last_analysis['bulls_growing_exp'],
            'bears_growing_exp': self.last_analysis['bears_growing_exp']
        }

    def get_slaving_info(self) -> Optional[dict]:
        """Retorna informações sobre o princípio de escravização"""
        if self.last_analysis is None:
            return None
        return {
            'slaving_active': self.last_analysis['slaving_active'],
            'order_mode': self.last_analysis['order_mode'],
            'noise_became_signal': self.last_analysis['noise_became_signal'],
            'signal_strength': self.last_analysis['signal_strength']
        }

    def get_phase_space_info(self) -> Optional[dict]:
        """Retorna informações sobre o espaço de fase"""
        if self.last_analysis is None:
            return None
        return {
            'phase_region': self.last_analysis['phase_region'],
            'boundary_crossed': self.last_analysis['boundary_crossed'],
            'is_ignition_point': self.last_analysis['is_ignition_point']
        }

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saída principal [σ, x1, x2, x3]

        Returns:
            Lista com [sigma, x1_bulls, x2_bears, x3_mean_reverters]
        """
        if self.last_analysis is None:
            return None
        return [
            self.last_analysis['sigma'],
            self.last_analysis['x1_bulls'],
            self.last_analysis['x2_bears'],
            self.last_analysis['x3_mean_reverters']
        ]

    def is_structure_forming(self) -> bool:
        """
        Verifica se uma Estrutura Dissipativa está se formando

        Returns:
            True se auto-organização detectada (Prigogine)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis.get('structure_forming', False)

    def is_nonlinear_regime(self) -> bool:
        """
        Verifica se estamos em regime não-linear

        Returns:
            True se σ ultrapassou limiar crítico
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis.get('is_nonlinear_regime', False)

    def is_slaving_active(self) -> bool:
        """
        Verifica se o Princípio de Escravização de Haken está ativo

        Returns:
            True se variáveis rápidas escravizadas pelas lentas
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis.get('slaving_active', False)

    def noise_became_signal(self) -> bool:
        """
        Verifica se o ruído se tornou sinal

        Returns:
            True se Mean Reverters foram escravizados
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis.get('noise_became_signal', False)

    def is_ignition_point(self) -> bool:
        """
        Verifica se estamos em um ponto de ignição

        Returns:
            True se cruzou fronteira de Voronoi com alta σ
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis.get('is_ignition_point', False)

    def get_dominant_species(self) -> Optional[str]:
        """
        Retorna a espécie dominante no ecossistema

        Returns:
            'Bulls', 'Bears' ou 'Mean Reverters'
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('dominant_species')

    def get_order_mode(self) -> Optional[str]:
        """
        Retorna o modo de ordem quando escravização ativa

        Returns:
            'BULLS', 'BEARS' ou 'NONE'
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('order_mode')

    def get_phase_region(self) -> Optional[str]:
        """
        Retorna a região atual no espaço de fase

        Returns:
            'bull_dominance', 'bear_dominance', 'equilibrium', etc.
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('phase_region')
