"""
Adaptador de Estratégia para o Protocolo Riemann-Mandelbrot
Integra o indicador PRM com o sistema de trading

VERSÃO V2.0 - PRONTO PARA DINHEIRO REAL
=======================================
Correções aplicadas:
1. Removido is_fitted (não existe mais no PRM corrigido)
2. Adicionados novos parâmetros hmm_training_window e hmm_min_training_samples
3. Direção baseada em tendência de barras FECHADAS
4. NOVO V2.0: Normalização do HMM sem look-ahead (exclude_last=True)
5. NOVO V2.0: Inicialização GARCH sem usar série completa
6. NOVO V2.0: Compatível com Walk-Forward Validation
"""
from datetime import datetime
from typing import Optional
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot


class PRMStrategy(BaseStrategy):
    """
    Estratégia baseada no Protocolo Riemann-Mandelbrot

    Detecta "Singularidades de Preço" - momentos onde todos os subsistemas
    (HMM, Lyapunov, Curvatura) concordam que há uma oportunidade de entrada.

    VERSÃO V2.0 - PRONTO PARA DINHEIRO REAL
    =======================================
    - Sem look-ahead bias em nenhum cálculo
    - Normalização incremental do HMM
    - Direção baseada apenas em barras fechadas
    - Compatível com Walk-Forward Validation
    """

    def __init__(self,
                 min_prices: int = 100,
                 stop_loss_pips: float = 15.0,
                 take_profit_pips: float = 30.0,
                 hmm_threshold: float = 0.85,
                 lyapunov_threshold: float = 0.5,
                 curvature_threshold: float = 0.1,
                 hmm_training_window: int = 200,
                 hmm_min_training_samples: int = 50,
                 trend_lookback: int = 10):
        """
        Inicializa a estratégia PRM

        Args:
            min_prices: Mínimo de preços necessários para análise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            hmm_threshold: Threshold para ativação do HMM
            lyapunov_threshold: Threshold K para Lyapunov
            curvature_threshold: Threshold para aceleração da curvatura
            hmm_training_window: NOVO - Janela de treino do HMM
            hmm_min_training_samples: NOVO - Mínimo de amostras para HMM
            trend_lookback: NOVO - Barras para calcular tendência
        """
        super().__init__(name="PRM-RiemannMandelbrot")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.trend_lookback = trend_lookback

        # Buffer de preços
        self.prices = deque(maxlen=500)
        self.volumes = deque(maxlen=500)
        
        # NOVO: Buffer de closes para calcular tendência sem look-ahead
        self.closes_history = deque(maxlen=500)

        # Indicador PRM com novos parâmetros
        self.prm = ProtocoloRiemannMandelbrot(
            n_states=3,
            hmm_threshold=hmm_threshold,
            lyapunov_threshold_k=lyapunov_threshold,
            curvature_threshold=curvature_threshold,
            lookback_window=100,
            hmm_training_window=hmm_training_window,          # NOVO
            hmm_min_training_samples=hmm_min_training_samples  # NOVO
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0  # Evita sinais em sequência
        self.bar_count = 0  # NOVO: Contador de barras processadas

    def add_price(self, price: float, volume: float = None):
        """Adiciona um preço ao buffer"""
        self.prices.append(price)
        self.volumes.append(volume if volume else 1.0)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal se houver singularidade

        NOTA: Este método é chamado com o CLOSE da barra atual.
        O sinal gerado será executado no OPEN da PRÓXIMA barra pelo BacktestEngine.

        Args:
            price: Preço atual (close da barra)
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (volume, high, low, open)

        Returns:
            Signal se singularidade detectada, None caso contrário
        """
        # Adiciona preço ao buffer
        volume = indicators.get('volume', 1.0)
        self.add_price(price, volume)
        
        # NOVO: Armazenar close no histórico para cálculo de tendência
        self.closes_history.append(price)
        self.bar_count += 1

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
            # Executa análise PRM
            result = self.prm.analyze(prices_array, volumes_array)
            self.last_analysis = result

            # Verifica se há singularidade
            if result['singularity_detected']:
                # CORRIGIDO: Determina direção baseada em barras FECHADAS
                direction = self._determine_direction_safe(result)

                if direction == SignalType.HOLD:
                    return None

                # CORREÇÃO #5: Não calcular níveis de stop/take aqui!
                # O BacktestEngine calculará baseado no entry_price REAL (OPEN da próxima barra)
                # Passamos apenas os valores em PIPS para o BacktestEngine recalcular
                pip_value = 0.0001  # Para EURUSD

                # ANTES (ERRADO): Calculava stop/take baseado no CLOSE atual
                # if direction == SignalType.BUY:
                #     stop_loss = price - (self.stop_loss_pips * pip_value)
                #     take_profit = price + (self.take_profit_pips * pip_value)
                # else:  # SELL
                #     stop_loss = price + (self.stop_loss_pips * pip_value)
                #     take_profit = price - (self.take_profit_pips * pip_value)

                # Calcula confiança baseada nos scores
                confidence = self._calculate_confidence(result)

                # CORREÇÃO #5: Cria sinal com stop/take em PIPS (não níveis de preço)
                # O BacktestEngine usará stop_loss_pips e take_profit_pips para
                # calcular os níveis reais baseados no entry_price (OPEN da próxima barra)
                signal = Signal(
                    type=direction,
                    price=price,
                    timestamp=timestamp,
                    strategy_name=self.name,
                    confidence=confidence,
                    stop_loss=None,        # CORREÇÃO #5: Será calculado pelo BacktestEngine
                    take_profit=None,      # CORREÇÃO #5: Será calculado pelo BacktestEngine
                    stop_loss_pips=self.stop_loss_pips,       # NOVO
                    take_profit_pips=self.take_profit_pips,   # NOVO
                    reason=self._generate_reason(result)
                )

                self.last_signal = signal
                self.signal_cooldown = 10  # Cooldown de 10 ticks

                return signal

        except Exception as e:
            # Log do erro mas continua operando
            print(f"Erro na análise PRM: {e}")

        return None

    def _determine_direction_safe(self, result: dict) -> SignalType:
        """
        CORRIGIDO: Determina a direção do trade baseado APENAS em dados passados
        
        Usa tendência calculada com barras JÁ FECHADAS (não inclui barra atual).
        
        IMPORTANTE: 
        - closes_history[-1] = close da barra ATUAL (acabou de ser adicionado)
        - closes_history[-2] = close da barra ANTERIOR (já fechada)
        - Para calcular tendência, usamos [-2] até [-(trend_lookback+2)]
        """
        # Verificar se temos histórico suficiente
        min_history = self.trend_lookback + 2  # +2 porque não usamos a barra atual
        if len(self.closes_history) < min_history:
            return SignalType.HOLD
        
        # ================================================================
        # CORREÇÃO: Calcular tendência APENAS com barras FECHADAS
        # ================================================================
        # closes_history[-1] = barra atual (NÃO usar)
        # closes_history[-2] = última barra fechada
        # closes_history[-(trend_lookback+2)] = barra de referência
        
        recent_close = self.closes_history[-2]  # Última barra FECHADA
        past_close = self.closes_history[-(self.trend_lookback + 2)]  # N barras antes
        
        trend = recent_close - past_close
        
        # Também considerar o estado do HMM e curvatura
        hmm_state = result['hmm_analysis']['current_state']
        curvature_acc = result['curvature_analysis']['current_acceleration']
        
        # ================================================================
        # LÓGICA DE DECISÃO
        # ================================================================
        
        # Estado 1 (Alta Vol. Direcional) - Seguir tendência
        if hmm_state == 1:
            if trend > 0 and curvature_acc > 0:
                return SignalType.BUY
            elif trend < 0 and curvature_acc < 0:
                return SignalType.SELL
                
        # Estado 2 (Choque de volatilidade) - Mais cautela
        elif hmm_state == 2:
            # Em choques, só entra se tendência e curvatura concordam fortemente
            if abs(curvature_acc) > self.prm.curvature_threshold * 1.5:
                if trend > 0 and curvature_acc > 0:
                    return SignalType.BUY
                elif trend < 0 and curvature_acc < 0:
                    return SignalType.SELL

        return SignalType.HOLD

    def _determine_direction(self, result: dict) -> SignalType:
        """
        DEPRECATED: Use _determine_direction_safe ao invés
        
        Mantido para compatibilidade, redireciona para versão segura.
        """
        return self._determine_direction_safe(result)

    def _calculate_confidence(self, result: dict) -> float:
        """Calcula nível de confiança do sinal (0.0 a 1.0)"""
        # Componentes de confiança
        hmm_prob = result['Prob_HMM']
        lyap_score = result['Lyapunov_Score']
        curv_signal = result['Curvature_Signal']

        # Normaliza Lyapunov para 0-1 (assume range 0-0.5)
        lyap_normalized = min(lyap_score / 0.5, 1.0) if lyap_score > 0 else 0

        # Média ponderada
        confidence = (
            0.4 * hmm_prob +           # HMM tem maior peso
            0.3 * lyap_normalized +    # Lyapunov
            0.3 * curv_signal          # Curvatura
        )

        return min(max(confidence, 0.0), 1.0)

    def _generate_reason(self, result: dict) -> str:
        """Gera descrição do motivo do sinal"""
        hmm = result['hmm_analysis']
        lyap = result['lyapunov_analysis']

        state_names = ['Consolidação', 'Alta Vol. Direcional', 'Choque de Vol.']
        state_name = state_names[hmm['current_state']]

        return (f"Singularidade PRM | Estado: {state_name} "
                f"(P={hmm['current_prob']:.2f}) | "
                f"Lyap: {lyap['lyapunov_max']:.4f} ({lyap['classification']})")

    def reset(self):
        """Reseta o estado da estratégia"""
        self.prices.clear()
        self.volumes.clear()
        self.closes_history.clear()  # NOVO
        self.last_analysis = None
        self.last_signal = None
        self.signal_cooldown = 0
        self.bar_count = 0  # NOVO
        
        # CORRIGIDO: Não usar is_fitted (não existe mais)
        # O PRM corrigido retreina automaticamente a cada chamada
        # Criar nova instância para garantir estado limpo
        self.prm = ProtocoloRiemannMandelbrot(
            n_states=3,
            hmm_threshold=self.prm.hmm_threshold,
            lyapunov_threshold_k=self.prm.lyapunov_threshold_k,
            curvature_threshold=self.prm.curvature_threshold,
            lookback_window=self.prm.lookback_window,
            hmm_training_window=self.prm.hmm_training_window,
            hmm_min_training_samples=self.prm.hmm_min_training_samples
        )

    def get_analysis_summary(self) -> Optional[dict]:
        """Retorna resumo da última análise"""
        if self.last_analysis is None:
            return None

        return {
            'singularity': self.last_analysis['singularity_detected'],
            'prob_hmm': self.last_analysis['Prob_HMM'],
            'lyapunov': self.last_analysis['Lyapunov_Score'],
            'curvature': self.last_analysis['Curvature_Signal'],
            'hmm_state': self.last_analysis['hmm_analysis']['current_state'],
            'lyap_class': self.last_analysis['lyapunov_analysis']['classification'],
            'bar_count': self.bar_count  # NOVO
        }
