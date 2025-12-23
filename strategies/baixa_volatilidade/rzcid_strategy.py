"""
Adaptador de Estrategia para o Riemann-Zeta Cryptanalytic Iceberg Decompiler
Integra o indicador RZ-CID com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .rzcid_riemann_zeta import (
    RiemannZetaCryptanalyticIcebergDecompiler,
    IcebergType,
    PRNGState
)


class RZCIDStrategy(BaseStrategy):
    """
    Estrategia baseada no Riemann-Zeta Cryptanalytic Iceberg Decompiler (RZ-CID)

    Usa Teoria dos Numeros e propriedades da funcao Zeta de Riemann para
    detectar ordens iceberg institucionais em baixa volatilidade.

    Conceitos-chave:
    - Serie de Dirichlet D(s): Transforma fluxo de ordens para dominio complexo
    - Zeros da Zeta: Correlacao com estrutura de numeros primos
    - Polos Fantasmas: Singularidades que indicam estrutura deterministica
    - PRNG Cracker: Quebra geradores pseudo-aleatorios via LLL
    - Transformada de Perron: Isola funcao acumulativa do iceberg
    - Materia Escura: Liquidez oculta no mercado

    Sinais:
    - LONG: Iceberg de compra detectado com alta % preenchimento
    - SHORT: Iceberg de venda detectado com alta % preenchimento
    - WAIT: Estrutura detectada, aguardando threshold
    """

    def __init__(self,
                 min_prices: int = 50,
                 stop_loss_pips: float = 15.0,
                 take_profit_pips: float = 30.0,
                 fill_threshold: float = 0.8,
                 n_zeta_zeros: int = 20,
                 min_sequence_length: int = 20,
                 perron_integration_points: int = 50):
        """
        Inicializa a estrategia RZ-CID

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            fill_threshold: % de preenchimento para gatilho
            n_zeta_zeros: Numero de zeros da Zeta para testar
            min_sequence_length: Comprimento minimo para PRNG
            perron_integration_points: Pontos de integracao de Perron
        """
        super().__init__(name="RZCID-Riemann")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de precos, volumes e timestamps
        self.prices = deque(maxlen=400)
        self.volumes = deque(maxlen=400)
        self.timestamps = deque(maxlen=400)

        # Indicador RZ-CID
        self.rzcid = RiemannZetaCryptanalyticIcebergDecompiler(
            fill_threshold=fill_threshold,
            n_zeta_zeros=n_zeta_zeros,
            min_sequence_length=min_sequence_length,
            perron_integration_points=perron_integration_points,
            min_ticks=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0
        self.tick_counter = 0

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em RZ-CID

        Args:
            price: Preco atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (volume opcional)

        Returns:
            Signal se iceberg detectado, None caso contrario
        """
        # Adiciona preco ao buffer
        self.prices.append(price)

        # Extrai volume e timestamp
        volume = indicators.get('volume', 1000.0)
        self.volumes.append(volume)

        # Converte timestamp para microsegundos
        ts_us = int(timestamp.timestamp() * 1_000_000)
        self.timestamps.append(ts_us)

        self.tick_counter += 1

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
        timestamps_array = np.array(self.timestamps)

        try:
            # Executa analise RZ-CID
            result = self.rzcid.analyze(prices_array, volumes_array, timestamps_array)
            self.last_analysis = result

            # Verifica sinal
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
                self.signal_cooldown = 30  # Cooldown maior para RZ-CID (computacao pesada)

                return signal

        except Exception as e:
            print(f"Erro na analise RZ-CID: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"RZCID Riemann | "
                f"Iceberg={result['iceberg_type'][:3]} | "
                f"Fill={result['iceberg_fill_pct']:.0%} | "
                f"PRNG={result['prng_state'][:4]} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.prices.clear()
        self.volumes.clear()
        self.timestamps.clear()
        self.rzcid.reset()
        self.last_analysis = None
        self.last_signal = None
        self.signal_cooldown = 0
        self.tick_counter = 0

    def get_analysis_summary(self) -> Optional[dict]:
        """Retorna resumo da ultima analise"""
        if self.last_analysis is None:
            return None

        return {
            'signal': self.last_analysis['signal_name'],
            'confidence': self.last_analysis['confidence'],
            'iceberg_type': self.last_analysis['iceberg_type'],
            'iceberg_fill_pct': self.last_analysis['iceberg_fill_pct'],
            'iceberg_size_estimated': self.last_analysis['iceberg_size_estimated'],
            'iceberg_executed': self.last_analysis['iceberg_executed'],
            'iceberg_remaining': self.last_analysis['iceberg_remaining'],
            'algo_signature': self.last_analysis['algo_signature'],
            'prng_state': self.last_analysis['prng_state'],
            'prediction_error': self.last_analysis['prediction_error'],
            'prime_music_detected': self.last_analysis['prime_music_detected'],
            'n_ghost_poles': self.last_analysis['n_ghost_poles'],
            'n_anomalous_poles': self.last_analysis['n_anomalous_poles'],
            'zeta_correlation': self.last_analysis['zeta_correlation'],
            'dark_matter_ratio': self.last_analysis['dark_matter_ratio'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_iceberg_info(self) -> Optional[dict]:
        """Retorna informacoes do iceberg detectado"""
        if self.last_analysis is None:
            return None
        return {
            'type': self.last_analysis['iceberg_type'],
            'fill_percentage': self.last_analysis['iceberg_fill_pct'],
            'size_estimated': self.last_analysis['iceberg_size_estimated'],
            'executed': self.last_analysis['iceberg_executed'],
            'remaining': self.last_analysis['iceberg_remaining'],
            'algo_signature': self.last_analysis['algo_signature']
        }

    def get_prng_info(self) -> Optional[dict]:
        """Retorna informacoes do PRNG detectado"""
        if self.last_analysis is None:
            return None
        return {
            'state': self.last_analysis['prng_state'],
            'prediction_error': self.last_analysis['prediction_error'],
            'lcg_a': self.last_analysis['lcg_a'],
            'lcg_c': self.last_analysis['lcg_c'],
            'lcg_m': self.last_analysis['lcg_m'],
            'lcg_confidence': self.last_analysis['lcg_confidence']
        }

    def get_dirichlet_info(self) -> Optional[dict]:
        """Retorna informacoes da Serie de Dirichlet"""
        if self.last_analysis is None:
            return None
        return {
            'sum_real': self.last_analysis['dirichlet_sum_real'],
            'sum_imag': self.last_analysis['dirichlet_sum_imag'],
            'n_ghost_poles': self.last_analysis['n_ghost_poles'],
            'n_anomalous_poles': self.last_analysis['n_anomalous_poles']
        }

    def get_zeta_info(self) -> Optional[dict]:
        """Retorna informacoes de correlacao com Zeta"""
        if self.last_analysis is None:
            return None
        return {
            'prime_music_detected': self.last_analysis['prime_music_detected'],
            'zeta_correlation': self.last_analysis['zeta_correlation']
        }

    def get_perron_info(self) -> Optional[dict]:
        """Retorna informacoes da Transformada de Perron"""
        if self.last_analysis is None:
            return None
        return {
            'accumulation_function': self.last_analysis['accumulation_function'],
            'accumulation_mass': self.last_analysis['accumulation_mass'],
            'dark_matter_ratio': self.last_analysis['dark_matter_ratio']
        }

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [iceberg_fill, zeta_corr, dark_matter, prediction_error]
        """
        if self.last_analysis is None:
            return None

        return [
            self.last_analysis['iceberg_fill_pct'],
            self.last_analysis['zeta_correlation'],
            self.last_analysis['dark_matter_ratio'],
            self.last_analysis['prediction_error']
        ]

    def is_buy_iceberg(self) -> bool:
        """
        Verifica se ha iceberg de compra

        Returns:
            True se iceberg de compra detectado
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['iceberg_type'] == 'BUY_ICEBERG'

    def is_sell_iceberg(self) -> bool:
        """
        Verifica se ha iceberg de venda

        Returns:
            True se iceberg de venda detectado
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['iceberg_type'] == 'SELL_ICEBERG'

    def is_no_iceberg(self) -> bool:
        """
        Verifica se nao ha iceberg

        Returns:
            True se nenhum iceberg detectado
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['iceberg_type'] == 'NO_ICEBERG'

    def is_prng_random(self) -> bool:
        """
        Verifica se PRNG e aleatorio (mercado varejo)

        Returns:
            True se sem estrutura deterministica
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['prng_state'] == 'RANDOM'

    def is_prng_deterministic(self) -> bool:
        """
        Verifica se PRNG e deterministico

        Returns:
            True se estrutura detectada
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['prng_state'] == 'DETERMINISTIC'

    def is_prng_breaking(self) -> bool:
        """
        Verifica se PRNG esta sendo quebrado

        Returns:
            True se semente quase descoberta
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['prng_state'] == 'BREAKING'

    def is_prng_broken(self) -> bool:
        """
        Verifica se PRNG foi quebrado

        Returns:
            True se previsao disponivel
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['prng_state'] == 'BROKEN'

    def is_prime_music_detected(self) -> bool:
        """
        Verifica se Musica dos Primos foi detectada

        Returns:
            True se estrutura de numeros primos encontrada
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['prime_music_detected']

    def is_waiting(self) -> bool:
        """
        Verifica se o indicador esta em modo espera

        Returns:
            True se aguardando melhor oportunidade
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['signal_name'] in ['WAIT', 'NEUTRAL', 'INSUFFICIENT_DATA']

    def has_anomalous_poles(self) -> bool:
        """
        Verifica se ha polos anomalos (>2sigma)

        Returns:
            True se polos fantasmas anomalos detectados
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['n_anomalous_poles'] > 0

    def get_iceberg_fill_percentage(self) -> Optional[float]:
        """
        Retorna a % de preenchimento do iceberg

        Returns:
            Percentual preenchido (0-1)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['iceberg_fill_pct']

    def get_prediction_error(self) -> Optional[float]:
        """
        Retorna o erro de previsao do PRNG

        Returns:
            Erro normalizado (0 = perfeito, 1 = aleatorio)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['prediction_error']

    def get_zeta_correlation(self) -> Optional[float]:
        """
        Retorna a correlacao com zeros da Zeta

        Returns:
            Correlacao media
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['zeta_correlation']

    def get_dark_matter_ratio(self) -> Optional[float]:
        """
        Retorna a razao de materia escura

        Returns:
            Liquidez oculta / visivel
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['dark_matter_ratio']

    def get_algo_signature(self) -> Optional[str]:
        """
        Retorna a assinatura do algoritmo detectado

        Returns:
            String com identificacao do algo
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['algo_signature']

    def get_iceberg_type(self) -> Optional[str]:
        """
        Retorna o tipo de iceberg

        Returns:
            BUY_ICEBERG, SELL_ICEBERG, NO_ICEBERG ou UNCERTAIN
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['iceberg_type']

    def get_prng_state(self) -> Optional[str]:
        """
        Retorna o estado do PRNG

        Returns:
            RANDOM, DETERMINISTIC, BREAKING ou BROKEN
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['prng_state']

    def is_iceberg_almost_complete(self, threshold: float = 0.8) -> bool:
        """
        Verifica se iceberg esta quase completo

        Args:
            threshold: Limiar de preenchimento

        Returns:
            True se fill >= threshold
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['iceberg_fill_pct'] >= threshold

    def is_structure_detected(self) -> bool:
        """
        Verifica se alguma estrutura foi detectada

        Returns:
            True se prime music, polos anomalos, ou PRNG nao-aleatorio
        """
        if self.last_analysis is None:
            return False

        return (self.last_analysis['prime_music_detected'] or
                self.last_analysis['n_anomalous_poles'] > 0 or
                self.last_analysis['prng_state'] != 'RANDOM')

    def is_entry_signal(self) -> bool:
        """
        Verifica se ha sinal de entrada

        Returns:
            True se iceberg quase completo + estrutura detectada
        """
        if self.last_analysis is None:
            return False

        iceberg_ready = self.last_analysis['iceberg_fill_pct'] >= 0.8
        has_structure = self.is_structure_detected()
        has_direction = self.last_analysis['iceberg_type'] in ['BUY_ICEBERG', 'SELL_ICEBERG']

        return iceberg_ready and has_structure and has_direction
