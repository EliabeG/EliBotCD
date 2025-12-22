"""
Adaptador de Estratégia para o Langevin-Schrödinger Quantum Probability Cloud
Integra o indicador LSQPC com o sistema de trading
"""
from datetime import datetime
from typing import Optional, Tuple
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .lsqpc_langevin_schrodinger import LangevinSchrodingerQuantumIndicator


class LSQPCStrategy(BaseStrategy):
    """
    Estratégia baseada no Langevin-Schrödinger Quantum Probability Cloud (LSQPC)

    Usa Física Estatística Avançada para modelar o mercado como um sistema
    quântico-estatístico. Calcula a Densidade de Probabilidade (PDF) da
    função de onda do preço via:
    - Equação de Langevin Generalizada (motor dinâmico)
    - Equação de Fokker-Planck (evolução da PDF)
    - Monte Carlo com ruído de Lévy (caudas gordas)
    """

    def __init__(self,
                 min_prices: int = 100,
                 stop_loss_pips: float = 20.0,
                 take_profit_pips: float = 40.0,
                 n_trajectories: int = 5000,
                 forecast_horizon: int = 15,
                 memory_alpha: float = 0.5,
                 levy_alpha: float = 1.7,
                 probability_threshold: float = 0.05):
        """
        Inicializa a estratégia LSQPC

        Args:
            min_prices: Mínimo de preços necessários para análise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            n_trajectories: Número de trajetórias Monte Carlo
            forecast_horizon: Horizonte de previsão em minutos
            memory_alpha: Expoente do kernel de memória
            levy_alpha: Índice de estabilidade de Lévy
            probability_threshold: Limiar de probabilidade para sinais
        """
        super().__init__(name="LSQPC-QuantumProbability")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de preços
        self.prices = deque(maxlen=600)

        # Indicador LSQPC
        self.lsqpc = LangevinSchrodingerQuantumIndicator(
            n_trajectories=n_trajectories,
            forecast_horizon=forecast_horizon,
            memory_alpha=memory_alpha,
            levy_alpha=levy_alpha,
            probability_threshold=probability_threshold,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def add_price(self, price: float):
        """Adiciona um preço ao buffer"""
        self.prices.append(price)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em LSQPC

        Args:
            price: Preço atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais

        Returns:
            Signal se probabilidade de breach detectada, None caso contrário
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
            # Executa análise LSQPC
            result = self.lsqpc.analyze(prices_array)
            self.last_analysis = result

            # Verifica sinal
            if result['signal'] != 0 and result['confidence'] >= 0.5:
                # Determina direção
                if result['signal'] == 1:
                    direction = SignalType.BUY
                else:
                    direction = SignalType.SELL

                # Usa níveis calculados pelo indicador ou calcula baseado em pips
                pip_value = 0.0001

                if result['stop_loss'] != price and result['take_profit'] != price:
                    stop_loss = result['stop_loss']
                    take_profit = result['take_profit']
                else:
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
                self.signal_cooldown = 20  # Cooldown para LSQPC

                return signal

        except Exception as e:
            print(f"Erro na análise LSQPC: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descrição do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"LSQPC Quantum | "
                f"λ={result['lambda_param']:.3f} | "
                f"σ_L={result['sigma_levy']:.5f} | "
                f"P={result['probability_breach']:.3f} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estratégia"""
        self.prices.clear()
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
            'lambda_param': self.last_analysis['lambda_param'],
            'sigma_levy': self.last_analysis['sigma_levy'],
            'probability_breach': self.last_analysis['probability_breach'],
            'upper_bound': self.last_analysis['upper_bound'],
            'lower_bound': self.last_analysis['lower_bound'],
            'entry_price': self.last_analysis['entry_price'],
            'stop_loss': self.last_analysis['stop_loss'],
            'take_profit': self.last_analysis['take_profit'],
            'current_price': self.last_analysis['current_price'],
            'n_trajectories': self.last_analysis['n_trajectories'],
            'forecast_horizon': self.last_analysis['forecast_horizon'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_langevin_params(self) -> Optional[dict]:
        """Retorna parâmetros calibrados da equação de Langevin"""
        if self.last_analysis is None:
            return None
        return {
            'lambda_param': self.last_analysis['lambda_param'],
            'sigma_levy': self.last_analysis['sigma_levy'],
            'n_trajectories': self.last_analysis['n_trajectories'],
            'forecast_horizon': self.last_analysis['forecast_horizon']
        }

    def get_fokker_planck_bounds(self) -> Optional[dict]:
        """Retorna limites da distribuição de probabilidade"""
        if self.last_analysis is None:
            return None
        return {
            'upper_95': self.last_analysis['upper_bound'],
            'lower_95': self.last_analysis['lower_bound'],
            'current_price': self.last_analysis['current_price'],
            'probability_breach': self.last_analysis['probability_breach']
        }

    def get_probability_heatmap(self) -> Optional[np.ndarray]:
        """Retorna o mapa de calor de probabilidade"""
        return self.lsqpc.get_probability_heatmap()

    def get_effective_potential(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retorna o potencial efetivo V(x)"""
        return self.lsqpc.get_effective_potential()

    def get_trajectories(self) -> Optional[np.ndarray]:
        """Retorna as trajetórias Monte Carlo simuladas"""
        return self.lsqpc.get_trajectories()

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saída principal [λ, σ_L, P, upper, lower]

        Returns:
            Lista com parâmetros principais
        """
        if self.last_analysis is None:
            return None
        return [
            self.last_analysis['lambda_param'],
            self.last_analysis['sigma_levy'],
            self.last_analysis['probability_breach'],
            self.last_analysis['upper_bound'],
            self.last_analysis['lower_bound']
        ]

    def is_at_upper_bound(self) -> bool:
        """
        Verifica se o preço está no limite superior 95%

        Returns:
            True se preço >= upper_95
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['current_price'] >= self.last_analysis['upper_bound']

    def is_at_lower_bound(self) -> bool:
        """
        Verifica se o preço está no limite inferior 95%

        Returns:
            True se preço <= lower_95
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['current_price'] <= self.last_analysis['lower_bound']

    def is_high_breach_probability(self, threshold: float = 0.95) -> bool:
        """
        Verifica se probabilidade de breach é alta

        Args:
            threshold: Limiar de probabilidade

        Returns:
            True se P >= threshold ou P <= (1-threshold)
        """
        if self.last_analysis is None:
            return False
        prob = self.last_analysis['probability_breach']
        return prob >= threshold or prob <= (1 - threshold)

    def get_mean_reversion_strength(self) -> Optional[float]:
        """
        Retorna a força de reversão à média (λ)

        Returns:
            Valor de λ (maior = reversão mais rápida)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['lambda_param']

    def get_levy_volatility(self) -> Optional[float]:
        """
        Retorna a volatilidade de Lévy (σ_L)

        Returns:
            Valor de σ_L (caudas gordas)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['sigma_levy']

    def get_price_bounds(self) -> Optional[Tuple[float, float]]:
        """
        Retorna os limites de preço (95% confidence)

        Returns:
            Tupla (lower_bound, upper_bound)
        """
        if self.last_analysis is None:
            return None
        return (self.last_analysis['lower_bound'],
                self.last_analysis['upper_bound'])
