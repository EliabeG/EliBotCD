"""
Adaptador de Estrategia para o Global Macro Spectral Coherence Scanner
Integra o indicador GMS-CS com o sistema de trading
"""
from datetime import datetime
from typing import Optional, List, Dict
from collections import deque
import numpy as np

from ..base import BaseStrategy, Signal, SignalType
from .gmscs_spectral_coherence import (
    GlobalMacroSpectralCoherenceScanner,
    GMSCSSignalType,
    TopologyState,
    MarketPhase
)


class GMSCSStrategy(BaseStrategy):
    """
    Estrategia baseada no Global Macro Spectral Coherence Scanner (GMS-CS)

    Usa Teoria Espectral de Grafos e MST para detectar colapso topologico
    no mercado global antes de explosoes de volatilidade.

    Conceitos-chave:
    - Matriz de Distancia Ultrametrica: d_ij = sqrt(2(1 - rho_ij))
    - MST (Minimum Spanning Tree): Filtra ruido, mantem estrutura
    - Laplaciano Espectral: L = D - A
    - Valor de Fiedler (lambda_2): Conectividade algebrica
    - Centralidade: Quem e o lider topologico?

    Sinais:
    - STAR + Leader UP: Sincronia alta, seguir o lider
    - CONTRACTING: Rede se acoplando, preparar para breakout
    - RELAXED: Baixa vol genuina, sem perigo
    - COLLAPSED: Tarde para entrar
    """

    def __init__(self,
                 min_prices: int = 50,
                 stop_loss_pips: float = 15.0,
                 take_profit_pips: float = 30.0,
                 correlation_window: int = 100,
                 fiedler_threshold: float = 0.3,
                 star_threshold: float = 0.4,
                 target_asset: str = "EURUSD"):
        """
        Inicializa a estrategia GMS-CS

        Args:
            min_prices: Minimo de precos necessarios para analise
            stop_loss_pips: Stop loss em pips
            take_profit_pips: Take profit em pips
            correlation_window: Janela para calculo de correlacao
            fiedler_threshold: Limiar para valor de Fiedler
            star_threshold: Limiar para topologia estrela
            target_asset: Ativo alvo para sinais
        """
        super().__init__(name="GMSCS-SpectralGraph")

        self.min_prices = min_prices
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.target_asset = target_asset

        # Buffer de retornos por ativo
        self.returns_buffers: Dict[str, deque] = {}
        self.buffer_maxlen = 500

        # Indicador GMS-CS
        self.gmscs = GlobalMacroSpectralCoherenceScanner(
            correlation_window=correlation_window,
            fiedler_threshold=fiedler_threshold,
            star_threshold=star_threshold,
            min_data_points=min_prices
        )

        # Estado
        self.last_analysis = None
        self.signal_cooldown = 0
        self.current_price = 0.0

    def add_asset_return(self, asset_name: str, return_value: float):
        """
        Adiciona retorno de um ativo ao buffer

        Args:
            asset_name: Nome do ativo (ex: "EURUSD", "SPX500")
            return_value: Retorno do periodo
        """
        if asset_name not in self.returns_buffers:
            self.returns_buffers[asset_name] = deque(maxlen=self.buffer_maxlen)
        self.returns_buffers[asset_name].append(return_value)

    def analyze(self, price: float, timestamp: datetime, **indicators) -> Optional[Signal]:
        """
        Analisa o mercado e retorna sinal baseado em GMS-CS

        Args:
            price: Preco atual do ativo alvo
            timestamp: Timestamp do tick
            **indicators: Indicadores adicionais (retornos de outros ativos)

        Returns:
            Signal se colapso topologico detectado, None caso contrario
        """
        self.current_price = price

        # Extrai retornos de outros ativos dos indicadores
        for key, value in indicators.items():
            if key.endswith('_return'):
                asset_name = key.replace('_return', '').upper()
                self.add_asset_return(asset_name, value)

        # Adiciona retorno do ativo alvo se fornecido
        if 'return' in indicators:
            self.add_asset_return(self.target_asset, indicators['return'])

        # Verifica se temos ativos suficientes
        if len(self.returns_buffers) < 5:
            return None

        # Verifica se temos dados suficientes em cada ativo
        min_len = min(len(buf) for buf in self.returns_buffers.values())
        if min_len < self.min_prices:
            return None

        # Cooldown para evitar sinais em sequencia
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
            return None

        # Converte buffers para dict de arrays
        returns_dict = {
            name: np.array(buf) for name, buf in self.returns_buffers.items()
        }

        try:
            # Executa analise GMS-CS
            result = self.gmscs.analyze(returns_dict, self.target_asset)
            self.last_analysis = result

            # Verifica sinal
            if result['signal'] != 0 and result['confidence'] >= 0.3:
                # Determina direcao
                if result['signal'] == 1:
                    direction = SignalType.BUY
                else:
                    direction = SignalType.SELL

                # Calcula stop/take
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
                self.signal_cooldown = 30  # Cooldown para GMS-CS

                return signal

        except Exception as e:
            print(f"Erro na analise GMS-CS: {e}")

        return None

    def _generate_reason(self, result: dict) -> str:
        """Gera descricao do motivo do sinal"""
        reasons = result.get('reasons', [])[:2]
        reasons_str = ', '.join(reasons) if reasons else 'N/A'

        return (f"GMSCS SpectralGraph | "
                f"Topology={result['topology_state']} | "
                f"Leader={result['central_asset']} | "
                f"Fiedler={result['fiedler_value']:.3f} | "
                f"{reasons_str}")

    def reset(self):
        """Reseta o estado da estrategia"""
        self.returns_buffers.clear()
        self.gmscs.reset()
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
            'topology_state': self.last_analysis['topology_state'],
            'market_phase': self.last_analysis['market_phase'],
            'ntl': self.last_analysis['ntl'],
            'mst_topology': self.last_analysis['mst_topology'],
            'fiedler_value': self.last_analysis['fiedler_value'],
            'central_asset': self.last_analysis['central_asset'],
            'leader_direction': self.last_analysis['leader_direction'],
            'mean_correlation': self.last_analysis['mean_correlation'],
            'star_index': self.last_analysis['star_index'],
            'reasons': self.last_analysis.get('reasons', [])
        }

    def get_mst_info(self) -> Optional[dict]:
        """Retorna informacoes da MST"""
        if self.last_analysis is None:
            return None
        return {
            'ntl': self.last_analysis['ntl'],
            'ntl_change': self.last_analysis['ntl_change'],
            'mst_topology': self.last_analysis['mst_topology'],
            'diameter': self.last_analysis['diameter'],
            'max_degree': self.last_analysis['max_degree']
        }

    def get_laplacian_info(self) -> Optional[dict]:
        """Retorna informacoes do Laplaciano"""
        if self.last_analysis is None:
            return None
        return {
            'fiedler_value': self.last_analysis['fiedler_value'],
            'fiedler_change': self.last_analysis['fiedler_change'],
            'algebraic_connectivity': self.last_analysis['algebraic_connectivity']
        }

    def get_centrality_info(self) -> Optional[dict]:
        """Retorna informacoes de centralidade"""
        if self.last_analysis is None:
            return None
        return {
            'central_asset': self.last_analysis['central_asset'],
            'central_node': self.last_analysis['central_node'],
            'target_centrality': self.last_analysis['target_centrality'],
            'target_distance_to_center': self.last_analysis['target_distance_to_center'],
            'leader_direction': self.last_analysis['leader_direction']
        }

    def get_correlation_info(self) -> Optional[dict]:
        """Retorna informacoes de correlacao"""
        if self.last_analysis is None:
            return None
        return {
            'mean_correlation': self.last_analysis['mean_correlation'],
            'max_correlation': self.last_analysis['max_correlation'],
            'n_assets': self.last_analysis['n_assets']
        }

    def get_topology_info(self) -> Optional[dict]:
        """Retorna informacoes de topologia"""
        if self.last_analysis is None:
            return None
        return {
            'topology_state': self.last_analysis['topology_state'],
            'market_phase': self.last_analysis['market_phase'],
            'star_index': self.last_analysis['star_index'],
            'chain_index': self.last_analysis['chain_index']
        }

    def get_output_vector(self) -> Optional[list]:
        """
        Retorna o vetor de saida principal

        Returns:
            Lista com [fiedler_value, ntl, mean_correlation, star_index]
        """
        if self.last_analysis is None:
            return None

        return [
            self.last_analysis['fiedler_value'],
            self.last_analysis['ntl'],
            self.last_analysis['mean_correlation'],
            self.last_analysis['star_index']
        ]

    def is_relaxed(self) -> bool:
        """
        Verifica se a topologia esta relaxada

        Returns:
            True se baixa vol genuina
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['topology_state'] == 'RELAXED'

    def is_contracting(self) -> bool:
        """
        Verifica se a rede esta contraindo

        Returns:
            True se MST se contraindo
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['topology_state'] == 'CONTRACTING'

    def is_star(self) -> bool:
        """
        Verifica se a topologia e estrela

        Returns:
            True se alta sincronia
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['topology_state'] == 'STAR'

    def is_collapsed(self) -> bool:
        """
        Verifica se houve colapso topologico

        Returns:
            True se ja colapsou
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['topology_state'] == 'COLLAPSED'

    def is_synchronized(self) -> bool:
        """
        Verifica se o mercado esta sincronizado

        Returns:
            True se alta sincronia
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['market_phase'] == 'SYNCHRONIZED'

    def is_fragmented(self) -> bool:
        """
        Verifica se o mercado esta fragmentado

        Returns:
            True se cada ativo por si
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['market_phase'] == 'FRAGMENTED'

    def is_coupling(self) -> bool:
        """
        Verifica se os ativos estao se acoplando

        Returns:
            True se em processo de acoplamento
        """
        if self.last_analysis is None:
            return False
        return self.last_analysis['market_phase'] == 'COUPLING'

    def is_waiting(self) -> bool:
        """
        Verifica se o indicador esta em modo espera

        Returns:
            True se aguardando melhor oportunidade
        """
        if self.last_analysis is None:
            return True
        return self.last_analysis['signal_name'] in ['WAIT', 'NEUTRAL', 'INSUFFICIENT_DATA', 'INSUFFICIENT_ASSETS']

    def get_topology_state(self) -> Optional[str]:
        """
        Retorna o estado topologico

        Returns:
            Estado: RELAXED, CONTRACTING, STAR, COLLAPSED
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['topology_state']

    def get_market_phase(self) -> Optional[str]:
        """
        Retorna a fase do mercado

        Returns:
            Fase: FRAGMENTED, COUPLING, SYNCHRONIZED, DECOUPLING
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['market_phase']

    def get_fiedler_value(self) -> Optional[float]:
        """
        Retorna o valor de Fiedler (lambda_2)

        Returns:
            Conectividade algebrica
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['fiedler_value']

    def get_ntl(self) -> Optional[float]:
        """
        Retorna o Normalized Tree Length

        Returns:
            Comprimento normalizado da MST
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['ntl']

    def get_central_asset(self) -> Optional[str]:
        """
        Retorna o ativo central (lider topologico)

        Returns:
            Nome do ativo lider
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['central_asset']

    def get_leader_direction(self) -> Optional[str]:
        """
        Retorna a direcao do lider

        Returns:
            UP, DOWN ou NEUTRAL
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['leader_direction']

    def get_mean_correlation(self) -> Optional[float]:
        """
        Retorna a correlacao media

        Returns:
            Correlacao media entre ativos
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['mean_correlation']

    def get_star_index(self) -> Optional[float]:
        """
        Retorna o indice de estrela

        Returns:
            Quao proximo de topologia estrela (0-1)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['star_index']

    def get_chain_index(self) -> Optional[float]:
        """
        Retorna o indice de cadeia

        Returns:
            Quao proximo de topologia cadeia (0-1)
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['chain_index']

    def get_target_centrality(self) -> Optional[float]:
        """
        Retorna a centralidade do ativo alvo

        Returns:
            Centralidade de autovetor
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['target_centrality']

    def get_target_distance(self) -> Optional[int]:
        """
        Retorna a distancia do ativo alvo ao centro

        Returns:
            Numero de hops ate o lider
        """
        if self.last_analysis is None:
            return None
        return self.last_analysis['target_distance_to_center']

    def is_high_sync_risk(self) -> bool:
        """
        Verifica se ha risco de alta sincronia

        Returns:
            True se correlacao alta + topologia estrela
        """
        if self.last_analysis is None:
            return False

        return bool(self.last_analysis['star_index'] > 0.4 and
                    self.last_analysis['mean_correlation'] > 0.5)

    def is_breakout_imminent(self) -> bool:
        """
        Verifica se breakout e iminente

        Returns:
            True se contraindo + acoplando
        """
        if self.last_analysis is None:
            return False

        return bool(self.last_analysis['topology_state'] == 'CONTRACTING' and
                    self.last_analysis['market_phase'] == 'COUPLING')

    def is_safe_low_vol(self) -> bool:
        """
        Verifica se e baixa volatilidade segura

        Returns:
            True se relaxado + fragmentado
        """
        if self.last_analysis is None:
            return True

        return bool(self.last_analysis['topology_state'] == 'RELAXED' and
                    self.last_analysis['market_phase'] == 'FRAGMENTED')

    def get_gmscs_stats(self) -> Optional[dict]:
        """
        Retorna estatisticas completas do GMS-CS

        Returns:
            Dict com todas as metricas
        """
        if self.last_analysis is None:
            return None

        return {
            'topology_state': self.last_analysis['topology_state'],
            'market_phase': self.last_analysis['market_phase'],
            'ntl': self.last_analysis['ntl'],
            'fiedler_value': self.last_analysis['fiedler_value'],
            'central_asset': self.last_analysis['central_asset'],
            'leader_direction': self.last_analysis['leader_direction'],
            'mean_correlation': self.last_analysis['mean_correlation'],
            'star_index': self.last_analysis['star_index'],
            'chain_index': self.last_analysis['chain_index'],
            'diameter': self.last_analysis['diameter'],
            'n_assets': self.last_analysis['n_assets']
        }
