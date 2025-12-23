"""
Adaptador de Estrategia para o Turing-Gray-Scott Reaction-Diffusion Morphogenesis Engine
Integra o indicador RD-ME com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .rdme_reaction_diffusion import (
    TuringGrayScottReactionDiffusionMorphogenesisEngine,
    RDMESignalType,
    PhaseState,
    PatternType,
    ReactionType
)


class RDMEStrategy(BaseStrategy):
    """
    Estrategia baseada no Turing-Gray-Scott Reaction-Diffusion Morphogenesis Engine (RD-ME)

    Usa Quimica de Reacao-Difusao para detectar nucleacao de padroes de
    tendencia antes que eles sejam visiveis no preco.

    Conceitos-chave:
    - Substrato U (Liquidez): Ordens limitadas disponiveis
    - Reagente V (Toxicidade): Fluxo toxico/informed trading (VPIN)
    - Reacao Autocatalitica: 2V + U -> 3V (cascade de ordens)
    - Padrao de Turing: Concentracao de toxicidade em nivel de preco
    - Bifurcacao de Hopf: Transicao de estabilidade para instabilidade

    Sinais:
    - PADRAO CRITICO: Spot de Turing formando - entrada forte
    - REACAO EM CADEIA: Producao de V explosiva
    - FASE TURING + VPIN: Alta toxicidade em fase instavel
    - WAIT: Na borda da instabilidade
    """

    def __init__(self,
                 min_prices: int = 50,
                 stop_loss_pips: float = 15.0,
                 take_profit_pips: float = 30.0,
                 n_price_levels: int = 100,
                 Du: float = 0.16,
                 Dv: float = 0.08,
                 default_F: float = 0.035,
                 default_k: float = 0.065,
                 solver_steps: int = 500,
                 vpin_buckets: int = 50):
        """
        Inicializa a estrategia RD-ME

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            n_price_levels: Numero de niveis de preco no grid
            Du: Coeficiente de difusao do substrato (liquidez)
            Dv: Coeficiente de difusao do reagente (toxicidade)
            default_F: Feed rate padrao
            default_k: Kill rate padrao
            solver_steps: Passos do solver Gray-Scott
            vpin_buckets: Numero de buckets para VPIN
        """
        super().__init__(name="RDME-ReactionDiffusion")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        # Buffer de precos e volumes
        self.prices = deque(maxlen=500)
        self.volumes = deque(maxlen=500)

        # Indicador RD-ME
        self.rdme = TuringGrayScottReactionDiffusionMorphogenesisEngine(
            n_price_levels=n_price_levels,
            Du=Du,
            Dv=Dv,
            default_F=default_F,
            default_k=default_k,
            solver_steps=solver_steps,
            solver_method="crank_nicolson",
            vpin_buckets=vpin_buckets,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em RD-ME

        Args:
            price: Preco atual
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (volume opcional)

        Returns:
            Signal se padrao de Turing detectado, None caso contrario
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
            # Executa analise RD-ME
            result = self.rdme.analyze(prices_array, volumes_array)
            self.last_analysis = result

            # Verifica sinal
            if result['signal'] != 0 and result['confidence'] >= 0.3:
                # Determina direcao
                if result['signal'] == 1:
                    direction = SignalType.BUY
                else:
                    direction = SignalType.SELL

                # Usa stop/take do indicador ou calcula
                if result['stop_loss'] != result['entry_price']:
                    stop_loss = result['stop_loss']
                    take_profit = result['take_profit']
                else:
                    pip_value = 0.0001
                    if direction == SignalType.BUY:
                        stop_loss = price - (self.stop_loss_pips * pip_value)
                        take_profit = price + (self.take_profit_pips * pip_value)
                    else:
                        stop_loss = price + (self.stop_loss_pips * pip_value)
                        take_profit = price - (self.take_profit_pips * pip_value)

                confidence = result['confidence']

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
                self.signal_cooldown = 30  # Cooldown para RD-ME

                return signal

        except Exception as e:
            print(f"Erro na analise RD-ME: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"RDME Reaction-Diffusion | "
                f"Phase={result['phase_state']} | "
                f"Pattern={result['pattern_type']} | "
                f"VPIN={result['vpin']:.3f} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.prices.clear()
        self.volumes.clear()
        self.rdme.reset()
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
            'phase_state': self.last_analysis['phase_state'],
            'pattern_type': self.last_analysis['pattern_type'],
            'reaction_type': self.last_analysis['reaction_type'],
            'u_mean': self.last_analysis['u_mean'],
            'v_mean': self.last_analysis['v_mean'],
            'F': self.last_analysis['F'],
            'k': self.last_analysis['k'],
            'vpin': self.last_analysis['vpin'],
            'toxicity_level': self.last_analysis['toxicity_level'],
            'pattern_amplitude': self.last_analysis['pattern_amplitude'],
            'pattern_growth_rate': self.last_analysis['pattern_growth_rate'],
            'hopf_stability': self.last_analysis['hopf_stability'],
            'is_chain_reaction': self.last_analysis['is_chain_reaction'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_concentration_info(self) -> Optional[dict]:
        """Retorna informacoes de concentracao"""
        if self.last_analysis is None:
            return None
        return {
            'u_mean': self.last_analysis['u_mean'],
            'v_mean': self.last_analysis['v_mean'],
            'u_gradient': self.last_analysis['u_gradient'],
            'v_gradient': self.last_analysis['v_gradient']
        }

    def get_gray_scott_params(self) -> Optional[dict]:
        """Retorna parametros Gray-Scott"""
        if self.last_analysis is None:
            return None
        return {
            'F': self.last_analysis['F'],
            'k': self.last_analysis['k'],
            'Du': self.last_analysis['Du'],
            'Dv': self.last_analysis['Dv']
        }

    def get_vpin_info(self) -> Optional[dict]:
        """Retorna informacoes de VPIN"""
        if self.last_analysis is None:
            return None
        return {
            'vpin': self.last_analysis['vpin'],
            'toxicity_level': self.last_analysis['toxicity_level']
        }

    def get_pattern_info(self) -> Optional[dict]:
        """Retorna informacoes do padrao de Turing"""
        if self.last_analysis is None:
            return None
        return {
            'pattern_type': self.last_analysis['pattern_type'],
            'pattern_location': self.last_analysis['pattern_location'],
            'pattern_amplitude': self.last_analysis['pattern_amplitude'],
            'pattern_growth_rate': self.last_analysis['pattern_growth_rate'],
            'pattern_direction': self.last_analysis['pattern_direction'],
            'pattern_is_critical': self.last_analysis['pattern_is_critical']
        }

    def get_hopf_info(self) -> Optional[dict]:
        """Retorna informacoes de bifurcacao de Hopf"""
        if self.last_analysis is None:
            return None
        return {
            'hopf_stability': self.last_analysis['hopf_stability'],
            'hopf_eigenvalue_real': self.last_analysis['hopf_eigenvalue_real'],
            'hopf_eigenvalue_imag': self.last_analysis['hopf_eigenvalue_imag'],
            'is_at_bifurcation': self.last_analysis['is_at_bifurcation']
        }

    def get_reaction_info(self) -> Optional[dict]:
        """Retorna informacoes da reacao quimica"""
        if self.last_analysis is None:
            return None
        return {
            'reaction_type': self.last_analysis['reaction_type'],
            'reaction_rate': self.last_analysis['reaction_rate'],
            'production_rate': self.last_analysis['production_rate'],
            'is_chain_reaction': self.last_analysis['is_chain_reaction']
        }

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [vpin, v_mean, reaction_rate, pattern_amplitude]
        """
        if self.last_analysis is None:
            return None

        return [
            self.last_analysis['vpin'],
            self.last_analysis['v_mean'],
            self.last_analysis['reaction_rate'],
            self.last_analysis['pattern_amplitude']
        ]

    def is_trivial_homogeneous(self) -> bool:
        """
        Verifica se o sistema esta em estado trivial homogeneo

        Returns:
            True se mercado estavel em baixa volatilidade
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['phase_state'] == 'TRIVIAL_HOMOGENEOUS'

    def is_turing_unstable(self) -> bool:
        """
        Verifica se o sistema esta na borda da instabilidade de Turing

        Returns:
            True se na regiao de instabilidade
        """
        if self.last_analysis is None:
            return False
        return bool(self.last_analysis['at_turing_instability'])

    def is_spots_phase(self) -> bool:
        """
        Verifica se o sistema esta na fase de spots

        Returns:
            True se formando manchas
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['phase_state'] == 'SPOTS'

    def is_stripes_phase(self) -> bool:
        """
        Verifica se o sistema esta na fase de stripes

        Returns:
            True se formando listras (tendencia)
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['phase_state'] == 'STRIPES'

    def is_chaos_phase(self) -> bool:
        """
        Verifica se o sistema esta em regime caotico

        Returns:
            True se em caos
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['phase_state'] == 'CHAOS'

    def is_pattern_critical(self) -> bool:
        """
        Verifica se ha padrao critico detectado

        Returns:
            True se padrao de Turing critico
        """
        if self.last_analysis is None:
            return False
        return bool(self.last_analysis['pattern_is_critical'])

    def is_chain_reaction(self) -> bool:
        """
        Verifica se ha reacao em cadeia

        Returns:
            True se reacao explosiva
        """
        if self.last_analysis is None:
            return False
        return bool(self.last_analysis['is_chain_reaction'])

    def is_at_bifurcation(self) -> bool:
        """
        Verifica se esta no ponto de bifurcacao de Hopf

        Returns:
            True se no ponto de bifurcacao
        """
        if self.last_analysis is None:
            return False
        return bool(self.last_analysis['is_at_bifurcation'])

    def is_high_toxicity(self) -> bool:
        """
        Verifica se ha alta toxicidade (VPIN)

        Returns:
            True se toxicidade HIGH ou EXTREME
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['toxicity_level'] in ['HIGH', 'EXTREME']

    def is_waiting(self) -> bool:
        """
        Verifica se o indicador esta em modo espera

        Returns:
            True se aguardando melhor oportunidade
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['signal_name'] in ['WAIT', 'NEUTRAL', 'INSUFFICIENT_DATA']

    def get_phase_state(self) -> Optional[str]:
        """
        Retorna o estado de fase atual

        Returns:
            Estado no diagrama de Pearson
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['phase_state']

    def get_pattern_type(self) -> Optional[str]:
        """
        Retorna o tipo de padrao detectado

        Returns:
            Tipo de padrao de Turing
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['pattern_type']

    def get_reaction_type(self) -> Optional[str]:
        """
        Retorna o tipo de reacao quimica

        Returns:
            Tipo de reacao
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['reaction_type']

    def get_vpin(self) -> Optional[float]:
        """
        Retorna o VPIN atual

        Returns:
            Volume-Synchronized Probability of Informed Trading
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['vpin']

    def get_toxicity_level(self) -> Optional[str]:
        """
        Retorna o nivel de toxicidade

        Returns:
            LOW, MEDIUM, HIGH ou EXTREME
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['toxicity_level']

    def get_feed_rate(self) -> Optional[float]:
        """
        Retorna o feed rate F

        Returns:
            Taxa de alimentacao de liquidez
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['F']

    def get_kill_rate(self) -> Optional[float]:
        """
        Retorna o kill rate k

        Returns:
            Taxa de consumo/remocao
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['k']

    def get_liquidity_mean(self) -> Optional[float]:
        """
        Retorna a liquidez media (u)

        Returns:
            Concentracao media de substrato
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['u_mean']

    def get_toxicity_mean(self) -> Optional[float]:
        """
        Retorna a toxicidade media (v)

        Returns:
            Concentracao media de reagente
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['v_mean']

    def get_reaction_rate(self) -> Optional[float]:
        """
        Retorna a taxa de reacao uv²

        Returns:
            Taxa da reacao autocatalitica
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['reaction_rate']

    def get_pattern_amplitude(self) -> Optional[float]:
        """
        Retorna a amplitude do padrao

        Returns:
            Amplitude do padrao de Turing
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['pattern_amplitude']

    def get_pattern_growth_rate(self) -> Optional[float]:
        """
        Retorna a taxa de crescimento do padrao

        Returns:
            Taxa de crescimento
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['pattern_growth_rate']

    def get_distance_to_instability(self) -> Optional[float]:
        """
        Retorna a distancia ate a instabilidade de Turing

        Returns:
            Distancia no diagrama de Pearson
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['distance_to_instability']

    def is_flash_crash_risk(self) -> bool:
        """
        Verifica se ha risco de flash crash

        Returns:
            True se reacao em cadeia + alta toxicidade
        """
        if self.last_analysis is None:
            return False

        return bool(self.last_analysis['is_chain_reaction'] and
                    self.last_analysis['toxicity_level'] in ['HIGH', 'EXTREME'])

    def is_breakout_setup(self) -> bool:
        """
        Verifica se ha setup de breakout iminente

        Returns:
            True se padrao crescendo + fase instavel
        """
        if self.last_analysis is None:
            return False

        return bool(self.last_analysis['pattern_is_critical'] or
                    (self.last_analysis['pattern_type'] == 'GROWING_SPOT' and
                     self.last_analysis['at_turing_instability']))

    def is_high_confidence_setup(self) -> bool:
        """
        Verifica se e um setup de alta confianca

        Returns:
            True se padrao critico ou reacao em cadeia
        """
        if self.last_analysis is None:
            return False

        return bool(self.last_analysis['pattern_is_critical'] or
                    (self.last_analysis['is_chain_reaction'] and
                     self.last_analysis['confidence'] > 0.7))

    def get_rdme_stats(self) -> Optional[dict]:
        """
        Retorna estatisticas completas do RD-ME

        Returns:
            Dict com todas as metricas
        """
        if self.last_analysis is None:
            return None

        return {
            'phase_state': self.last_analysis['phase_state'],
            'pattern_type': self.last_analysis['pattern_type'],
            'reaction_type': self.last_analysis['reaction_type'],
            'vpin': self.last_analysis['vpin'],
            'toxicity_level': self.last_analysis['toxicity_level'],
            'F': self.last_analysis['F'],
            'k': self.last_analysis['k'],
            'u_mean': self.last_analysis['u_mean'],
            'v_mean': self.last_analysis['v_mean'],
            'reaction_rate': self.last_analysis['reaction_rate'],
            'pattern_amplitude': self.last_analysis['pattern_amplitude'],
            'is_chain_reaction': self.last_analysis['is_chain_reaction'],
            'at_turing_instability': self.last_analysis['at_turing_instability'],
            'distance_to_instability': self.last_analysis['distance_to_instability']
        }
