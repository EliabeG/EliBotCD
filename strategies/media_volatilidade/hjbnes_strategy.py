"""
Adaptador de Estrategia para o Hamilton-Jacobi-Bellman Nash Equilibrium Solver
Integra o indicador HJB-NES com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List, Dict
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .hjbnes_hamilton_jacobi import HJBNashEquilibriumSolver, NashEquilibrium


class HJBNESStrategy(BaseStrategy):
    """
    Estrategia baseada no Hamilton-Jacobi-Bellman Nash Equilibrium Solver (HJB-NES)

    Usa Mean Field Games (MFG) para calcular o Preco de Equilibrio de Nash
    do mercado. Em media volatilidade, o mercado SEMPRE converge para o
    equilibrio de Nash.

    Conceitos-chave:
    - Equacao HJB: Estrategia otima do agente medio (backward in time)
    - Equacao FPK: Evolucao da distribuicao de traders (forward in time)
    - Equilibrio de Nash: Ponto onde todos os agentes estao satisfeitos
    - Arbitragem Estrutural: Diferenca entre preco real e equilibrio
    - Movimento Irracional: Preco movendo contra o drift de Nash
    """

    def __init__(self,
                 min_prices: int = 100,
                 stop_loss_pips: float = 20.0,
                 take_profit_pips: float = 40.0,
                 n_space: int = 50,
                 n_time: int = 25,
                 max_iterations: int = 50,
                 tolerance: float = 1e-4,
                 base_risk_aversion: float = 1.0,
                 discrepancy_threshold: float = 0.05):
        """
        Inicializa a estrategia HJB-NES

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            n_space: Pontos no grid espacial
            n_time: Pontos no grid temporal
            max_iterations: Maximo de iteracoes do solver
            tolerance: Tolerancia para convergencia
            base_risk_aversion: gamma - aversao ao risco base
            discrepancy_threshold: Limiar de discrepancia para sinal
        """
        super().__init__(name="HJBNES-NashEquilibrium")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffers de precos e volumes
        self.prices = deque(maxlen=600)
        self.volumes = deque(maxlen=600)

        # Indicador HJB-NES
        self.hjbnes = HJBNashEquilibriumSolver(
            n_space=n_space,
            n_time=n_time,
            max_iterations=max_iterations,
            tolerance=tolerance,
            base_risk_aversion=base_risk_aversion,
            discrepancy_threshold=discrepancy_threshold,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def add_price(self, price: float, volume: float = None):
        """Adiciona um preco e volume ao buffer"""
        self.prices.append(price)
        if volume is not None:
            self.volumes.append(volume)
        else:
            # Volume sintetico baseado na variacao de preco
            if len(self.prices) > 1:
                delta = abs(price - self.prices[-2])
                synthetic_vol = delta * 50000 + np.random.rand() * 1000 + 500
                self.volumes.append(synthetic_vol)
            else:
                self.volumes.append(1000.0)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em HJB-NES

        Args:
            price: Preco atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (pode incluir 'volume')

        Returns:
            Signal se discrepancia de Nash detectada, None caso contrario
        """
        # Extrai volume se disponivel
        volume = indicators.get('volume', None)

        # Adiciona preco ao buffer
        self.add_price(price, volume)

        # Verifica se temos dados suficientes
        if len(self.prices) < self.min_prices:
            return None

        # Cooldown para evitar sinais em sequencia
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        # Converte para numpy arrays
        prices_array = np.array(self.prices)
        volumes_array = np.array(self.volumes)

        try:
            # Executa analise HJB-NES
            result = self.hjbnes.analyze(prices_array, volumes_array)
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
                self.signal_cooldown = 30  # Cooldown para HJB-NES (solver pesado)

                return signal

        except Exception as e:
            print(f"Erro na analise HJB-NES: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"HJBNES MFG | "
                f"Regime={result['market_regime']} | "
                f"Nash={result['nash_price']:.5f} | "
                f"Disc={result['price_discrepancy']:.4f} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.prices.clear()
        self.volumes.clear()
        self.hjbnes.reset()
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
            'nash_price': self.last_analysis['nash_price'],
            'nash_drift': self.last_analysis['nash_drift'],
            'current_price': self.last_analysis['current_price'],
            'current_drift': self.last_analysis['current_drift'],
            'price_discrepancy': self.last_analysis['price_discrepancy'],
            'drift_discrepancy': self.last_analysis['drift_discrepancy'],
            'market_regime': self.last_analysis['market_regime'],
            'is_irrational': self.last_analysis['is_irrational'],
            'solver_converged': self.last_analysis['solver_converged'],
            'solver_iterations': self.last_analysis['solver_iterations'],
            'solver_error': self.last_analysis['solver_error'],
            'risk_aversion': self.last_analysis['risk_aversion'],
            'volatility': self.last_analysis['volatility'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_nash_info(self) -> Optional[dict]:
        """Retorna informacoes sobre o equilibrio de Nash"""
        if self.last_analysis is None:
            return None
        return {
            'nash_price': self.last_analysis['nash_price'],
            'nash_drift': self.last_analysis['nash_drift'],
            'current_price': self.last_analysis['current_price'],
            'current_drift': self.last_analysis['current_drift']
        }

    def get_discrepancy_info(self) -> Optional[dict]:
        """Retorna informacoes sobre discrepancia"""
        if self.last_analysis is None:
            return None
        return {
            'price_discrepancy': self.last_analysis['price_discrepancy'],
            'drift_discrepancy': self.last_analysis['drift_discrepancy'],
            'is_irrational': self.last_analysis['is_irrational'],
            'correction_direction': self.last_analysis['correction_direction']
        }

    def get_solver_info(self) -> Optional[dict]:
        """Retorna informacoes sobre o solver"""
        if self.last_analysis is None:
            return None
        return {
            'converged': self.last_analysis['solver_converged'],
            'iterations': self.last_analysis['solver_iterations'],
            'error': self.last_analysis['solver_error']
        }

    def get_market_regime(self) -> Optional[str]:
        """Retorna o regime de mercado atual"""
        if self.last_analysis is None:
            return None
        return self.last_analysis['market_regime']

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [price_discrepancy, drift_discrepancy, converged, regime_code]
        """
        if self.last_analysis is None:
            return None

        regime_codes = {
            'EQUILIBRIUM': 0.0,
            'OVERVALUED': 1.0,
            'UNDERVALUED': -1.0,
            'UNSTABLE': 0.5
        }
        regime_code = regime_codes.get(self.last_analysis['market_regime'], 0.0)

        return [
            self.last_analysis['price_discrepancy'],
            self.last_analysis['drift_discrepancy'],
            1.0 if self.last_analysis['solver_converged'] else 0.0,
            regime_code
        ]

    def is_in_equilibrium(self) -> bool:
        """
        Verifica se o mercado esta em equilibrio de Nash

        Returns:
            True se no equilibrio
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['market_regime'] == 'EQUILIBRIUM'

    def is_hibernating(self) -> bool:
        """
        Verifica se o indicador esta hibernando

        Returns:
            True se solver nao convergiu
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['signal_name'] == 'HIBERNATE'

    def is_overvalued(self) -> bool:
        """
        Verifica se o mercado esta sobrevalorizado

        Returns:
            True se preco acima do equilibrio
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['market_regime'] == 'OVERVALUED'

    def is_undervalued(self) -> bool:
        """
        Verifica se o mercado esta subvalorizado

        Returns:
            True se preco abaixo do equilibrio
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['market_regime'] == 'UNDERVALUED'

    def is_irrational_movement(self) -> bool:
        """
        Verifica se ha movimento irracional

        Returns:
            True se preco movendo contra o drift de Nash
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['is_irrational']

    def solver_converged(self) -> bool:
        """
        Verifica se o solver MFG convergiu

        Returns:
            True se convergencia atingida
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['solver_converged']

    def get_nash_price(self) -> Optional[float]:
        """
        Retorna o preco de equilibrio de Nash

        Returns:
            Preco de equilibrio
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['nash_price']

    def get_nash_drift(self) -> Optional[float]:
        """
        Retorna o drift de equilibrio de Nash

        Returns:
            Drift de equilibrio (v*)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['nash_drift']

    def get_price_discrepancy(self) -> Optional[float]:
        """
        Retorna a discrepancia de preco

        Returns:
            Diferenca entre preco atual e equilibrio
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['price_discrepancy']

    def get_drift_discrepancy(self) -> Optional[float]:
        """
        Retorna a discrepancia de drift

        Returns:
            Diferenca entre drift real e drift de Nash
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['drift_discrepancy']

    def get_risk_aversion(self) -> Optional[float]:
        """
        Retorna a aversao ao risco calibrada

        Returns:
            gamma - parametro de aversao ao risco
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['risk_aversion']

    def get_volatility(self) -> Optional[float]:
        """
        Retorna a volatilidade calculada

        Returns:
            sigma - volatilidade do mercado
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['volatility']

    def get_equilibrium_history(self) -> np.ndarray:
        """Retorna historico de precos de equilibrio"""
        return self.hjbnes.get_equilibrium_history()

    def get_drift_history(self) -> np.ndarray:
        """Retorna historico de drifts de equilibrio"""
        return self.hjbnes.get_drift_history()

    def has_structural_arbitrage(self, threshold: float = 0.1) -> bool:
        """
        Verifica se ha oportunidade de arbitragem estrutural

        Args:
            threshold: Limiar de discrepancia

        Returns:
            True se discrepancia significativa detectada
        """
        if self.last_analysis is None:
            return False

        if not self.last_analysis['solver_converged']:
            return False

        return abs(self.last_analysis['price_discrepancy']) > threshold
