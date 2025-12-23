"""
Adaptador de Estrategia para o Quantum Zeno Discord & Lindblad Decoherence Timer
Integra o indicador QZD-LDT com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .qzdldt_quantum_zeno import (
    QuantumZenoDiscordLindbladTimer,
    QuantumState,
    ZenoState,
    BlochPosition
)


class QZDLDTStrategy(BaseStrategy):
    """
    Estrategia baseada no Quantum Zeno Discord & Lindblad Decoherence Timer (QZD-LDT)

    Usa informacao quantica e decoerencia para detectar quando o "congelamento"
    do preco vai derreter - o momento exato do rompimento.

    Conceitos-chave:
    - Matriz Densidade (rho): Estado quantico do mercado
    - Esfera de Bloch: Representacao geometrica do qubit
    - Efeito Zeno: Preco travado por "observacao" constante (HFTs)
    - Equacao de Lindblad: Evolucao dissipativa do sistema
    - Quantum Discord: Correlacoes quanticas nao-classicas
    - Fidelidade: Detecta transicoes reais vs ruido

    Sinais:
    - LONG: Bloch -> Polo Sul (|1>), decoerencia + Zeno quebrando
    - SHORT: Bloch -> Polo Norte (|0>), decoerencia + Zeno quebrando
    - WAIT: Zeno ativo ou Discord alto (transicao iminente)
    """

    def __init__(self,
                 min_prices: int = 50,
                 stop_loss_pips: float = 15.0,
                 take_profit_pips: float = 30.0,
                 decoherence_rate: float = 0.1,
                 relaxation_rate: float = 0.05,
                 zeno_threshold: float = 0.7,
                 measurement_window: int = 50,
                 evolution_steps: int = 50):
        """
        Inicializa a estrategia QZD-LDT

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            decoherence_rate: Taxa de decoerencia (gamma_d)
            relaxation_rate: Taxa de relaxacao (gamma_r)
            zeno_threshold: Limiar para Zeno ativo
            measurement_window: Janela para frequencia de medicao
            evolution_steps: Passos para evolucao de Lindblad
        """
        super().__init__(name="QZDLDT-Quantum")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de precos e volumes
        self.prices = deque(maxlen=400)
        self.volumes = deque(maxlen=400)

        # Indicador QZD-LDT
        self.qzdldt = QuantumZenoDiscordLindbladTimer(
            decoherence_rate=decoherence_rate,
            relaxation_rate=relaxation_rate,
            zeno_threshold=zeno_threshold,
            measurement_window=measurement_window,
            evolution_steps=evolution_steps,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em QZD-LDT

        Args:
            price: Preco atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (volume opcional)

        Returns:
            Signal se decoerencia detectada, None caso contrario
        """
        # Adiciona preco ao buffer
        self.prices.append(price)

        # Extrai volume se disponivel
        volume = indicators.get('volume', None)
        if volume is not None:
            self.volumes.append(volume)

        # Verifica se temos dados suficientes
        if len(self.prices) < self.min_prices:
            return None

        # Cooldown para evitar sinais em sequencia
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        # Converte para numpy arrays
        prices_array = np.array(self.prices)
        volumes_array = np.array(self.volumes) if len(self.volumes) >= len(self.prices) else None

        try:
            # Executa analise QZD-LDT
            result = self.qzdldt.analyze(prices_array, volumes_array)
            self.last_analysis = result

            # Verifica sinal (ignora WAIT, NEUTRAL e INSUFFICIENT_DATA)
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
                self.signal_cooldown = 20  # Cooldown para QZD-LDT

                return signal

        except Exception as e:
            print(f"Erro na analise QZD-LDT: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"QZDLDT Quantum | "
                f"State={result['quantum_state']} | "
                f"Zeno={result['zeno_state'][:4]} | "
                f"z={result['bloch_z']:.2f} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.prices.clear()
        self.volumes.clear()
        self.qzdldt.reset()
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
            'quantum_state': self.last_analysis['quantum_state'],
            'zeno_state': self.last_analysis['zeno_state'],
            'bloch_position': self.last_analysis['bloch_position'],
            'purity': self.last_analysis['purity'],
            'von_neumann_entropy': self.last_analysis['von_neumann_entropy'],
            'coherence': self.last_analysis['coherence'],
            'bloch_z': self.last_analysis['bloch_z'],
            'bloch_radius': self.last_analysis['bloch_radius'],
            'zeno_strength': self.last_analysis['zeno_strength'],
            'measurement_frequency': self.last_analysis['measurement_frequency'],
            'zeno_breaking': self.last_analysis['zeno_breaking'],
            'coherence_decay_rate': self.last_analysis['coherence_decay_rate'],
            'entropy_diverging': self.last_analysis['entropy_diverging'],
            'fidelity': self.last_analysis['fidelity'],
            'quantum_discord': self.last_analysis['quantum_discord'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_density_matrix_info(self) -> Optional[dict]:
        """Retorna informacoes da matriz densidade"""
        if self.last_analysis is None:
            return None
        return {
            'purity': self.last_analysis['purity'],
            'von_neumann_entropy': self.last_analysis['von_neumann_entropy'],
            'coherence': self.last_analysis['coherence']
        }

    def get_bloch_info(self) -> Optional[dict]:
        """Retorna informacoes da esfera de Bloch"""
        if self.last_analysis is None:
            return None
        return {
            'x': self.last_analysis['bloch_x'],
            'y': self.last_analysis['bloch_y'],
            'z': self.last_analysis['bloch_z'],
            'radius': self.last_analysis['bloch_radius'],
            'position': self.last_analysis['bloch_position']
        }

    def get_zeno_info(self) -> Optional[dict]:
        """Retorna informacoes do efeito Zeno"""
        if self.last_analysis is None:
            return None
        return {
            'strength': self.last_analysis['zeno_strength'],
            'measurement_frequency': self.last_analysis['measurement_frequency'],
            'state': self.last_analysis['zeno_state'],
            'breaking': self.last_analysis['zeno_breaking']
        }

    def get_lindblad_info(self) -> Optional[dict]:
        """Retorna informacoes da evolucao de Lindblad"""
        if self.last_analysis is None:
            return None
        return {
            'coherence_decay_rate': self.last_analysis['coherence_decay_rate'],
            'entropy_rate': self.last_analysis['entropy_rate'],
            'entropy_diverging': self.last_analysis['entropy_diverging'],
            'fidelity': self.last_analysis['fidelity'],
            'fidelity_dropping': self.last_analysis['fidelity_dropping']
        }

    def get_discord_info(self) -> Optional[dict]:
        """Retorna informacoes de discord quantico"""
        if self.last_analysis is None:
            return None
        return {
            'quantum_discord': self.last_analysis['quantum_discord'],
            'classical_correlation': self.last_analysis['classical_correlation']
        }

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [purity, bloch_z, zeno_strength, fidelity]
        """
        if self.last_analysis is None:
            return None

        return [
            self.last_analysis['purity'],
            self.last_analysis['bloch_z'],
            self.last_analysis['zeno_strength'],
            self.last_analysis['fidelity']
        ]

    def is_coherent(self) -> bool:
        """
        Verifica se o sistema esta no estado coerente

        Returns:
            True se superposicao coerente (Zeno ativo)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['quantum_state'] == 'COHERENT'

    def is_decohering(self) -> bool:
        """
        Verifica se o sistema esta perdendo coerencia

        Returns:
            True se decoerencia em andamento
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['quantum_state'] == 'DECOHERING'

    def is_collapsed(self) -> bool:
        """
        Verifica se o estado colapsou

        Returns:
            True se movimento iminente
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['quantum_state'] == 'COLLAPSED'

    def is_mixed(self) -> bool:
        """
        Verifica se o sistema esta em estado misto

        Returns:
            True se ruido classico dominante
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['quantum_state'] == 'MIXED'

    def is_zeno_active(self) -> bool:
        """
        Verifica se o Efeito Zeno esta ativo

        Returns:
            True se preco travado por observacao
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['zeno_state'] == 'ZENO_ACTIVE'

    def is_zeno_weakening(self) -> bool:
        """
        Verifica se o Efeito Zeno esta enfraquecendo

        Returns:
            True se protecao enfraquecendo
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['zeno_state'] == 'ZENO_WEAKENING'

    def is_zeno_broken(self) -> bool:
        """
        Verifica se o Efeito Zeno quebrou

        Returns:
            True se protecao quebrada
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['zeno_state'] == 'ZENO_BROKEN'

    def is_zeno_breaking(self) -> bool:
        """
        Verifica se o Zeno esta quebrando agora

        Returns:
            True se transicao em andamento
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['zeno_breaking']

    def is_waiting(self) -> bool:
        """
        Verifica se o indicador esta em modo espera

        Returns:
            True se aguardando melhor oportunidade
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['signal_name'] in ['WAIT', 'NEUTRAL', 'INSUFFICIENT_DATA']

    def is_at_north_pole(self) -> bool:
        """
        Verifica se Bloch esta no Polo Norte

        Returns:
            True se estado |0> (Ask dominant) - SHORT
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['bloch_position'] == 'NORTH_POLE'

    def is_at_south_pole(self) -> bool:
        """
        Verifica se Bloch esta no Polo Sul

        Returns:
            True se estado |1> (Bid dominant) - LONG
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['bloch_position'] == 'SOUTH_POLE'

    def is_at_equator(self) -> bool:
        """
        Verifica se Bloch esta no Equador

        Returns:
            True se superposicao maxima
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['bloch_position'] == 'EQUATOR'

    def is_entropy_diverging(self) -> bool:
        """
        Verifica se entropia esta divergindo

        Returns:
            True se entropia de Von Neumann crescendo
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['entropy_diverging']

    def is_fidelity_dropping(self) -> bool:
        """
        Verifica se fidelidade esta caindo

        Returns:
            True se transicao real (nao ruido)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['fidelity_dropping']

    def get_purity(self) -> Optional[float]:
        """
        Retorna a pureza Tr(rho^2)

        Returns:
            Pureza do estado (1 = puro, 0.5 = maximamente misto)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['purity']

    def get_von_neumann_entropy(self) -> Optional[float]:
        """
        Retorna a entropia de Von Neumann

        Returns:
            S = -Tr(rho ln rho)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['von_neumann_entropy']

    def get_coherence(self) -> Optional[float]:
        """
        Retorna a coerencia

        Returns:
            Soma dos elementos fora da diagonal
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['coherence']

    def get_bloch_z(self) -> Optional[float]:
        """
        Retorna componente z do vetor de Bloch

        Returns:
            z > 0 -> |0> (SHORT), z < 0 -> |1> (LONG)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['bloch_z']

    def get_zeno_strength(self) -> Optional[float]:
        """
        Retorna a forca do Efeito Zeno

        Returns:
            Proporcao de medicoes sem mudanca de preco
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['zeno_strength']

    def get_fidelity(self) -> Optional[float]:
        """
        Retorna a fidelidade F(rho_0, rho_t)

        Returns:
            Fidelidade entre estados
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['fidelity']

    def get_quantum_discord(self) -> Optional[float]:
        """
        Retorna o Discord Quantico

        Returns:
            Correlacoes quanticas nao-classicas
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['quantum_discord']

    def get_entropy_history(self) -> np.ndarray:
        """Retorna historico de entropia"""
        return self.qzdldt.get_entropy_history()

    def get_coherence_history(self) -> np.ndarray:
        """Retorna historico de coerencia"""
        return self.qzdldt.get_coherence_history()

    def get_fidelity_history(self) -> np.ndarray:
        """Retorna historico de fidelidade"""
        return self.qzdldt.get_fidelity_history()

    def is_decoherence_event(self) -> bool:
        """
        Verifica se ha evento de decoerencia

        Returns:
            True se entropia divergindo OU fidelidade caindo
        """
        if self.last_analysis is None:
            return False

        return (self.last_analysis['entropy_diverging'] or
                self.last_analysis['fidelity_dropping'])

    def is_breakout_imminent(self) -> bool:
        """
        Verifica se rompimento e iminente

        Returns:
            True se decoerencia + Zeno quebrando
        """
        if self.last_analysis is None:
            return False

        return (self.is_decoherence_event() and
                (self.is_zeno_breaking() or not self.is_zeno_active()))

    def is_false_breakout_likely(self) -> bool:
        """
        Verifica se rompimento falso e provavel

        Returns:
            True se Zeno ainda ativo mas preco moveu
        """
        if self.last_analysis is None:
            return False

        # Zeno ativo + alta fidelidade = movimento e apenas ruido
        return (self.is_zeno_active() and
                self.last_analysis['fidelity'] > 0.95)

    def get_quantum_state(self) -> Optional[str]:
        """
        Retorna o estado quantico atual

        Returns:
            COHERENT, DECOHERING, COLLAPSED ou MIXED
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['quantum_state']

    def get_zeno_state(self) -> Optional[str]:
        """
        Retorna o estado do Efeito Zeno

        Returns:
            ZENO_ACTIVE, ZENO_WEAKENING ou ZENO_BROKEN
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['zeno_state']

    def get_bloch_position(self) -> Optional[str]:
        """
        Retorna a posicao na Esfera de Bloch

        Returns:
            NORTH_POLE, SOUTH_POLE, EQUATOR ou MIXED_STATE
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['bloch_position']
