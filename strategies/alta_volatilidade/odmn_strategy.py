"""
Adaptador de Estratégia para o Oráculo de Derivativos de Malliavin-Nash
Integra o indicador ODMN com o sistema de trading
"""
from datetime import datetime
from typing import Optional
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .odmn_malliavin_nash import OracloDerivativosMalliavinNash


class ODMNStrategy(BaseStrategy):
    """
    Estratégia baseada no Oráculo de Derivativos de Malliavin-Nash (ODMN)

    Usa Cálculo de Malliavin para detectar fragilidade estrutural e
    Mean Field Games para prever comportamento institucional e pontos
    de transição de fase no mercado.
    """

    def __init__(self,
                 min_prices: int = 150,
                 stop_loss_pips: float = 25.0,
                 take_profit_pips: float = 50.0,
                 lookback_window: int = 100,
                 fragility_threshold: float = 2.0,
                 mfg_direction_threshold: float = 0.1,
                 use_deep_galerkin: bool = True,
                 malliavin_paths: int = 2000,
                 malliavin_steps: int = 30):
        """
        Inicializa a estratégia ODMN

        Args:
            min_prices: Mínimo de preços necessários para análise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            lookback_window: Janela para calibração do Heston
            fragility_threshold: Limiar do índice de fragilidade
            mfg_direction_threshold: Limiar para direção do MFG
            use_deep_galerkin: Se True, usa redes neurais para MFG
            malliavin_paths: Número de trajetórias Monte Carlo
            malliavin_steps: Passos temporais na simulação
        """
        super().__init__(name="ODMN-MalliavinNash")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de preços
        self.prices = deque(maxlen=800)

        # Indicador ODMN
        self.odmn = OracloDerivativosMalliavinNash(
            lookback_window=lookback_window,
            fragility_threshold=fragility_threshold,
            mfg_direction_threshold=mfg_direction_threshold,
            use_deep_galerkin=use_deep_galerkin,
            malliavin_paths=malliavin_paths,
            malliavin_steps=malliavin_steps
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def add_price(self, price: float):
        """Adiciona um preço ao buffer"""
        self.prices.append(price)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em Malliavin + MFG

        Args:
            price: Preço atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais

        Returns:
            Signal se fragilidade + MFG indicarem direção, None caso contrário
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
            # Executa análise ODMN
            result = self.odmn.analyze(prices_array)
            self.last_analysis = result

            # Verifica sinal
            if result['signal'] != 0 and result['confidence'] >= 0.6:
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
                self.signal_cooldown = 25  # Cooldown maior para ODMN

                return signal

        except Exception as e:
            print(f"Erro na análise ODMN: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descrição do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"ODMN Oracle | "
                f"Frag_P{result['fragility_percentile']*100:.0f} | "
                f"MFG={result['mfg_direction']:.4f} | "
                f"Regime={result['regime']} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estratégia"""
        self.prices.clear()
        self.last_analysis = None
        self.last_signal = None
        self.signal_cooldown = 0
        # Reseta cache do indicador
        self.odmn._cache = {
            'heston_params': None,
            'malliavin_result': None,
            'mfg_result': None,
            'fragility_history': deque(maxlen=100),
            'direction_history': deque(maxlen=100)
        }

    def get_analysis_summary(self) -> Optional[dict]:
        """Retorna resumo da última análise"""
        if self.last_analysis is None:
            return None

        return {
            'signal': self.last_analysis['signal_name'],
            'confidence': self.last_analysis['confidence'],
            'fragility_index': self.last_analysis['fragility_index'],
            'fragility_percentile': self.last_analysis['fragility_percentile'],
            'fragility_trigger': self.last_analysis['fragility_trigger'],
            'mfg_direction': self.last_analysis['mfg_direction'],
            'mfg_equilibrium': self.last_analysis['mfg_equilibrium'],
            'regime': self.last_analysis['regime'],
            'implied_vol': self.last_analysis['implied_vol'],
            'current_price': self.last_analysis['current_price'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_heston_params(self) -> Optional[dict]:
        """Retorna os parâmetros calibrados do modelo de Heston"""
        if self.last_analysis is None:
            return None
        return self.last_analysis.get('heston_params')

    def get_malliavin_metrics(self) -> Optional[dict]:
        """Retorna as métricas de Malliavin"""
        if self.last_analysis is None:
            return None
        return {
            'fragility_index': self.last_analysis['fragility_index'],
            'fragility_percentile': self.last_analysis['fragility_percentile'],
            'malliavin_norm_S': self.last_analysis['malliavin_norm_S'],
            'malliavin_norm_v': self.last_analysis['malliavin_norm_v']
        }

    def get_mfg_metrics(self) -> Optional[dict]:
        """Retorna as métricas do Mean Field Game"""
        if self.last_analysis is None:
            return None
        return {
            'mfg_direction': self.last_analysis['mfg_direction'],
            'mfg_value': self.last_analysis['mfg_value'],
            'mfg_equilibrium': self.last_analysis['mfg_equilibrium']
        }

    def get_forecast(self, horizon: int = 20) -> Optional[dict]:
        """
        Gera previsão usando o modelo de Heston calibrado

        Args:
            horizon: Horizonte de previsão em dias

        Returns:
            Dicionário com estatísticas da distribuição esperada
        """
        if len(self.prices) < self.min_prices:
            return None

        prices_array = np.array(self.prices)
        return self.odmn.get_heston_forecast(prices_array, horizon=horizon)
