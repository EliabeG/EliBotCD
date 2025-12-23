"""
Adaptador de Estrategia para o Marchenko-Pastur Spectral De-Noiser
Integra o indicador MP-SDEO com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .mpsdeo_marchenko_pastur import MarchenkoPasturSpectralDeNoiser, EigenRegime


class MPSDEOStrategy(BaseStrategy):
    """
    Estrategia baseada no Marchenko-Pastur Spectral De-Noiser (MP-SDEO)

    Usa Random Matrix Theory (RMT) para cirurgicamente remover ruido
    e identificar tendencias estruturais genuinas atraves da decomposicao
    espectral da matriz de correlacao time-lagged.

    Conceitos-chave:
    - Matriz de Wishart: Autocorrelacao temporal da serie de precos
    - Lei de Marchenko-Pastur: Limites universais do ruido
    - Limpeza Espectral: Remove autovalores dentro da zona MP
    - Preco Fantasma: Sinal purificado sem ruido
    - IPR (Inverse Participation Ratio): Mede localizacao do autovetor
    - Eigen-Entropy: Entropia da distribuicao de autovalores

    Sinais de Sniper:
    - P_raw > P_clean + IPR Spike -> SHORT (ruido puxou para cima)
    - P_raw < P_clean + IPR Spike -> LONG (ruido puxou para baixo)
    """

    def __init__(self,
                 min_prices: int = 250,
                 stop_loss_pips: float = 20.0,
                 take_profit_pips: float = 40.0,
                 n_lags: int = 30,
                 window: int = 100,
                 ipr_threshold: float = 0.08,
                 entropy_threshold: float = 3.0,
                 ipr_spike_factor: float = 1.5):
        """
        Inicializa a estrategia MP-SDEO

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            n_lags: N - numero de lags da matriz de trajetoria
            window: T - janela de tempo da matriz
            ipr_threshold: Limiar para considerar IPR alto
            entropy_threshold: Limiar para considerar entropia baixa
            ipr_spike_factor: Fator para deteccao de spike de IPR
        """
        super().__init__(name="MPSDEO-RMT")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de precos
        self.prices = deque(maxlen=600)

        # Indicador MP-SDEO
        self.mpsdeo = MarchenkoPasturSpectralDeNoiser(
            n_lags=n_lags,
            window=window,
            ipr_threshold=ipr_threshold,
            entropy_threshold=entropy_threshold,
            ipr_spike_factor=ipr_spike_factor,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em MP-SDEO

        Args:
            price: Preco atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (nao utilizados)

        Returns:
            Signal se oportunidade espectral detectada, None caso contrario
        """
        # Adiciona preco ao buffer
        self.prices.append(price)

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
            # Executa analise MP-SDEO
            result = self.mpsdeo.analyze(prices_array)
            self.last_analysis = result

            # Verifica sinal (ignora WAIT e NEUTRAL)
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
                self.signal_cooldown = 20  # Cooldown para MP-SDEO

                return signal

        except Exception as e:
            print(f"Erro na analise MP-SDEO: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"MPSDEO RMT | "
                f"Regime={result['eigen_regime']} | "
                f"IPR={result['ipr']:.4f} | "
                f"SNR={result['snr']:.2f} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.prices.clear()
        self.mpsdeo.reset()
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
            'eigen_regime': self.last_analysis['eigen_regime'],
            'price_raw': self.last_analysis['price_raw'],
            'price_clean': self.last_analysis['price_clean'],
            'price_deviation': self.last_analysis['price_deviation'],
            'lambda_max': self.last_analysis['lambda_max'],
            'lambda_min': self.last_analysis['lambda_min'],
            'n_signal_components': self.last_analysis['n_signal_components'],
            'n_noise_components': self.last_analysis['n_noise_components'],
            'snr': self.last_analysis['snr'],
            'eigen_entropy': self.last_analysis['eigen_entropy'],
            'ipr': self.last_analysis['ipr'],
            'ipr_spike': self.last_analysis['ipr_spike'],
            'localization': self.last_analysis['localization'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_spectral_info(self) -> Optional[dict]:
        """Retorna informacoes espectrais"""
        if self.last_analysis is None:
            return None
        return {
            'lambda_max': self.last_analysis['lambda_max'],
            'lambda_min': self.last_analysis['lambda_min'],
            'n_signal': self.last_analysis['n_signal_components'],
            'n_noise': self.last_analysis['n_noise_components'],
            'snr': self.last_analysis['snr'],
            'signal_strength': self.last_analysis['signal_strength'],
            'noise_level': self.last_analysis['noise_level']
        }

    def get_entropy_info(self) -> Optional[dict]:
        """Retorna informacoes de entropia e IPR"""
        if self.last_analysis is None:
            return None
        return {
            'entropy': self.last_analysis['eigen_entropy'],
            'ipr': self.last_analysis['ipr'],
            'ipr_spike': self.last_analysis['ipr_spike'],
            'localization': self.last_analysis['localization']
        }

    def get_price_info(self) -> Optional[dict]:
        """Retorna informacoes de preco"""
        if self.last_analysis is None:
            return None
        return {
            'price_raw': self.last_analysis['price_raw'],
            'price_clean': self.last_analysis['price_clean'],
            'deviation': self.last_analysis['price_deviation'],
            'deviation_normalized': self.last_analysis.get('deviation_normalized', 0.0)
        }

    def get_eigen_regime(self) -> Optional[str]:
        """Retorna o regime de autovetor atual"""
        if self.last_analysis is None:
            return None
        return self.last_analysis['eigen_regime']

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [snr, ipr, localization, deviation_normalized]
        """
        if self.last_analysis is None:
            return None

        return [
            self.last_analysis['snr'],
            self.last_analysis['ipr'],
            self.last_analysis['localization'],
            self.last_analysis.get('deviation_normalized', 0.0)
        ]

    def is_localized(self) -> bool:
        """
        Verifica se o mercado esta em regime localizado

        Returns:
            True se autovetor dominante esta localizado
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['eigen_regime'] == 'LOCALIZED'

    def is_delocalized(self) -> bool:
        """
        Verifica se o mercado esta em regime delocalizado

        Returns:
            True se autovetor dominante esta espalhado
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['eigen_regime'] == 'DELOCALIZED'

    def is_transitional(self) -> bool:
        """
        Verifica se o mercado esta em transicao

        Returns:
            True se entre regimes
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['eigen_regime'] == 'TRANSITIONAL'

    def is_waiting(self) -> bool:
        """
        Verifica se o indicador esta em modo espera

        Returns:
            True se aguardando melhor oportunidade
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['signal_name'] == 'WAIT'

    def has_ipr_spike(self) -> bool:
        """
        Verifica se houve spike de IPR

        Returns:
            True se localizacao subita detectada
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['ipr_spike']

    def has_signal_components(self) -> bool:
        """
        Verifica se ha componentes de sinal (fora da zona MP)

        Returns:
            True se autovalores > lambda_max detectados
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['n_signal_components'] > 0

    def get_snr(self) -> Optional[float]:
        """
        Retorna a razao sinal-ruido

        Returns:
            SNR (Signal-to-Noise Ratio)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['snr']

    def get_ipr(self) -> Optional[float]:
        """
        Retorna o Inverse Participation Ratio

        Returns:
            IPR do autovetor dominante
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['ipr']

    def get_localization(self) -> Optional[float]:
        """
        Retorna o grau de localizacao

        Returns:
            Localizacao [0, 1]
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['localization']

    def get_clean_price(self) -> Optional[float]:
        """
        Retorna o preco limpo (fantasma)

        Returns:
            Preco purificado sem ruido
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['price_clean']

    def get_price_deviation(self) -> Optional[float]:
        """
        Retorna o desvio entre preco real e limpo

        Returns:
            P_raw - P_clean
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['price_deviation']

    def get_ipr_history(self) -> np.ndarray:
        """Retorna historico de IPR"""
        return self.mpsdeo.get_ipr_history()

    def get_entropy_history(self) -> np.ndarray:
        """Retorna historico de entropia"""
        return self.mpsdeo.get_entropy_history()

    def get_clean_price_history(self) -> np.ndarray:
        """Retorna historico de preco limpo"""
        return self.mpsdeo.get_clean_price_history()

    def is_price_above_clean(self) -> bool:
        """
        Verifica se preco real esta acima do limpo

        Returns:
            True se P_raw > P_clean (potencial SHORT)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['price_deviation'] > 0

    def is_price_below_clean(self) -> bool:
        """
        Verifica se preco real esta abaixo do limpo

        Returns:
            True se P_raw < P_clean (potencial LONG)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['price_deviation'] < 0

    def is_spectral_sniper_setup(self) -> bool:
        """
        Verifica se o setup de Sniper Espectral esta ativo

        Returns:
            True se IPR spike + desvio significativo
        """
        if self.last_analysis is None:
            return False

        has_spike = self.last_analysis['ipr_spike']
        dev_norm = abs(self.last_analysis.get('deviation_normalized', 0.0))

        return has_spike and dev_norm > 0.5
