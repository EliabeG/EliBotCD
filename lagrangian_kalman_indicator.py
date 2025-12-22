#!/usr/bin/env python3
"""
The Lagrangian-Kalman Trajectory Estimator
==============================================
Sistema de previsão de preços para Scalping em EURUSD usando:
- Unscented Kalman Filter (UKF) para suavização
- Mecânica Lagrangiana para análise de momentum
- Kernel Density Estimation (KDE) para suporte/resistência
- Regressão Polinomial + Bayes para probabilidade

Autor: EliBotHFT
Integrado com: TickTrader WebSocket API
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from sklearn.neighbors import KernelDensity
import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
from collections import deque
import warnings
import sys
import os

# Adicionar diretório do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

# Importar cliente e configurações do botforex
from config.settings import CONFIG
from api.ticktrader_ws import TickTraderFeed, TickData
from api.ticktrader_rest import TickTraderREST
from utils.logger import setup_logger

logger = setup_logger("lagrangian_kalman")


# ============================================================================
# 2. O MOTOR MATEMÁTICO (THE CORE ENGINE) - MathCore
# ============================================================================

class UnscentedKalmanFilter:
    """
    Implementação do Unscented Kalman Filter (UKF)

    Estados estimados:
    - P_real: Preço Real (suavizado)
    - v: Velocidade Real (primeira derivada do preço)
    """

    def __init__(self, dim_x: int = 2, dim_z: int = 1,
                 alpha: float = 0.001, beta: float = 2.0, kappa: float = 0.0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.lambda_ = alpha**2 * (dim_x + kappa) - dim_x
        self.n_sigma = 2 * dim_x + 1

        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)

        self.Wm[0] = self.lambda_ / (dim_x + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

        for i in range(1, self.n_sigma):
            self.Wm[i] = 1.0 / (2 * (dim_x + self.lambda_))
            self.Wc[i] = self.Wm[i]

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x) * 0.001
        self.dt = 1.0

        self.F = np.array([
            [1.0, self.dt],
            [0.0, 1.0]
        ])

        self.H = np.array([[1.0, 0.0]])
        self.Q = np.array([[1e-9, 0], [0, 1e-8]])
        self.R = np.array([[1e-7]])

        self.price_history: List[float] = []
        self.velocity_history: List[float] = []
        self.filtered_price_history: List[float] = []

    def _compute_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        n = len(x)
        sigma_points = np.zeros((self.n_sigma, n))
        P_scaled = (n + self.lambda_) * P
        P_scaled = P_scaled + np.eye(n) * 1e-10
        P_scaled = (P_scaled + P_scaled.T) / 2

        try:
            sqrt_P = np.linalg.cholesky(P_scaled)
        except np.linalg.LinAlgError:
            U, S, Vt = np.linalg.svd(P_scaled)
            sqrt_P = U @ np.diag(np.sqrt(np.maximum(S, 1e-10)))

        sigma_points[0] = x
        for i in range(n):
            sigma_points[i + 1] = x + sqrt_P[i]
            sigma_points[n + i + 1] = x - sqrt_P[i]

        return sigma_points

    def _unscented_transform(self, sigma_points: np.ndarray,
                              func, noise_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_points, dim = sigma_points.shape
        transformed = np.array([func(sp) for sp in sigma_points])

        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)

        mean = np.sum(self.Wm.reshape(-1, 1) * transformed, axis=0)
        diff = transformed - mean
        cov = np.zeros((len(mean), len(mean)))

        for i, w in enumerate(self.Wc):
            cov += w * np.outer(diff[i], diff[i])
        cov += noise_cov

        return mean, cov

    def predict(self):
        sigma_points = self._compute_sigma_points(self.x, self.P)
        def state_transition(x):
            return self.F @ x
        self.x, self.P = self._unscented_transform(sigma_points, state_transition, self.Q)

    def update(self, z: float):
        z = np.array([z])
        sigma_points = self._compute_sigma_points(self.x, self.P)

        def observation(x):
            return self.H @ x

        z_pred = np.zeros(self.dim_z)
        for i, sp in enumerate(sigma_points):
            z_pred += self.Wm[i] * observation(sp)

        Pzz = self.R.copy()
        for i, sp in enumerate(sigma_points):
            diff = observation(sp) - z_pred
            Pzz += self.Wc[i] * np.outer(diff, diff)

        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i, sp in enumerate(sigma_points):
            x_diff = sp - self.x
            z_diff = observation(sp) - z_pred
            Pxz += self.Wc[i] * np.outer(x_diff, z_diff)

        K = Pxz @ np.linalg.inv(Pzz)
        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ Pzz @ K.T
        self.P = (self.P + self.P.T) / 2
        self.P = self.P + np.eye(self.dim_x) * 1e-10

        self.price_history.append(z[0])
        self.filtered_price_history.append(self.x[0])
        self.velocity_history.append(self.x[1])

    def process_candle(self, close_price: float) -> Tuple[float, float]:
        self.predict()
        self.update(close_price)
        return self.x[0], self.x[1]

    def initialize(self, initial_price: float):
        self.x = np.array([initial_price, 0.0])
        self.price_history = [initial_price]
        self.filtered_price_history = [initial_price]
        self.velocity_history = [0.0]


class LagrangianMechanics:
    """Análise de Energia Cinética e Potencial (Abordagem Lagrangiana)"""

    def __init__(self, lookback_regression: int = 50):
        self.lookback_regression = lookback_regression
        self.kinetic_energy: List[float] = []
        self.potential_energy: List[float] = []
        self.lagrangian: List[float] = []
        self.action: List[float] = []

    def calculate_mean_reversion_level(self, prices: np.ndarray) -> float:
        if len(prices) < self.lookback_regression:
            return np.mean(prices)
        recent_prices = prices[-self.lookback_regression:]
        x = np.arange(len(recent_prices))
        coeffs = np.polyfit(x, recent_prices, 1)
        mu = np.polyval(coeffs, len(recent_prices) - 1)
        return mu

    def calculate_elasticity_constant(self, prices: np.ndarray) -> float:
        if len(prices) < 2:
            return 1.0
        std = np.std(prices[-self.lookback_regression:])
        if std < 1e-10:
            return 1.0
        k = 1.0 / (std ** 2)
        return k

    def calculate_kinetic_energy(self, mass: float, velocity: float) -> float:
        return 0.5 * mass * (velocity ** 2)

    def calculate_potential_energy(self, price: float, prices: np.ndarray) -> float:
        mu = self.calculate_mean_reversion_level(prices)
        k = self.calculate_elasticity_constant(prices)
        return 0.5 * k * ((price - mu) ** 2)

    def calculate_lagrangian(self, T: float, V: float) -> float:
        return T - V

    def calculate_action_integral(self, dt: float = 1.0) -> float:
        if len(self.lagrangian) < 2:
            return 0.0
        return np.trapz(self.lagrangian, dx=dt)

    def process_candle(self, filtered_price: float, velocity: float,
                       volume: float, prices: np.ndarray) -> Dict[str, float]:
        mass = volume / 1000.0 if volume > 0 else 1.0
        T = self.calculate_kinetic_energy(mass, velocity)
        self.kinetic_energy.append(T)
        V = self.calculate_potential_energy(filtered_price, prices)
        self.potential_energy.append(V)
        L = self.calculate_lagrangian(T, V)
        self.lagrangian.append(L)
        S = self.calculate_action_integral()
        self.action.append(S)

        dS = self.action[-1] - self.action[-2] if len(self.action) >= 2 else 0.0
        dT = self.kinetic_energy[-1] - self.kinetic_energy[-2] if len(self.kinetic_energy) >= 2 else 0.0

        return {
            'kinetic_energy': T,
            'potential_energy': V,
            'lagrangian': L,
            'action': S,
            'action_derivative': dS,
            'kinetic_derivative': dT,
            'mean_reversion_level': self.calculate_mean_reversion_level(prices)
        }


class KDESupportResistance:
    """Cálculo de Suporte e Resistência usando Kernel Density Estimation"""

    def __init__(self, bandwidth: str = 'silverman'):
        self.bandwidth = bandwidth
        self.kde_model: Optional[KernelDensity] = None
        self.support_levels: List[float] = []
        self.resistance_levels: List[float] = []
        self.pdf_values: Optional[np.ndarray] = None
        self.price_grid: Optional[np.ndarray] = None

    def _calculate_bandwidth(self, data: np.ndarray) -> float:
        n = len(data)
        if n < 2:
            return 1e-4
        std = np.std(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        iqr_scaled = iqr / 1.34 if iqr > 0 else std
        A = min(std, iqr_scaled) if std > 0 else iqr_scaled
        if A < 1e-10:
            A = np.std(data) if np.std(data) > 0 else (data.max() - data.min()) / 10
            if A < 1e-10:
                A = 1e-4
        bandwidth = 0.9 * A * (n ** (-0.2))
        return max(bandwidth, 1e-6)

    def fit(self, prices: np.ndarray):
        if len(prices) < 10:
            return
        bw = self._calculate_bandwidth(prices)
        self.kde_model = KernelDensity(kernel='gaussian', bandwidth=bw)
        self.kde_model.fit(prices.reshape(-1, 1))

        std = np.std(prices)
        if std < 1e-10:
            std = (prices.max() - prices.min()) / 10
            if std < 1e-10:
                std = prices.mean() * 0.01

        price_min = prices.min() - 3 * std
        price_max = prices.max() + 3 * std
        if price_min >= price_max:
            price_min = prices.mean() - 0.01 * prices.mean()
            price_max = prices.mean() + 0.01 * prices.mean()

        self.price_grid = np.linspace(price_min, price_max, 1000)
        log_pdf = self.kde_model.score_samples(self.price_grid.reshape(-1, 1))
        self.pdf_values = np.exp(log_pdf)

    def find_support_resistance(self, current_price: float, n_levels: int = 3) -> Dict[str, List[float]]:
        if self.pdf_values is None:
            return {'supports': [], 'resistances': []}

        peaks, properties = find_peaks(self.pdf_values, height=np.mean(self.pdf_values), distance=20)
        if len(peaks) == 0:
            return {'supports': [], 'resistances': []}

        peak_prices = self.price_grid[peaks]
        peak_heights = self.pdf_values[peaks]
        sorted_idx = np.argsort(peak_heights)[::-1]
        peak_prices = peak_prices[sorted_idx]

        supports = peak_prices[peak_prices < current_price]
        resistances = peak_prices[peak_prices > current_price]
        supports = sorted(supports, reverse=True)[:n_levels]
        resistances = sorted(resistances)[:n_levels]

        return {'supports': supports, 'resistances': resistances}


class ProbabilityOracle:
    """Oráculo de Probabilidade usando Regressão Polinomial"""

    def __init__(self, lookback: int = 20, degree: int = 3):
        self.lookback = lookback
        self.degree = degree
        self.last_prediction: Optional[float] = None
        self.last_probability: Optional[float] = None
        self.prediction_history: List[float] = []

    def fit_polynomial(self, prices: np.ndarray) -> Optional[np.ndarray]:
        if len(prices) < self.lookback:
            return None
        recent = prices[-self.lookback:]
        if len(recent) <= self.degree:
            return None
        x = np.arange(len(recent))
        try:
            coeffs = np.polyfit(x, recent, self.degree)
        except (np.linalg.LinAlgError, ValueError):
            try:
                coeffs = np.polyfit(x, recent, 1)
            except Exception:
                return None
        return coeffs

    def extrapolate_next(self, prices: np.ndarray) -> float:
        coeffs = self.fit_polynomial(prices)
        if coeffs is None:
            return prices[-1] if len(prices) > 0 else 0.0
        next_x = self.lookback
        prediction = np.polyval(coeffs, next_x)
        self.last_prediction = prediction
        self.prediction_history.append(prediction)
        return prediction

    def calculate_z_score(self, prediction: float, prices: np.ndarray) -> float:
        if len(prices) < 2:
            return 0.0
        recent = prices[-self.lookback:]
        mean = np.mean(recent)
        std = np.std(recent)
        if std < 1e-10:
            return 0.0
        return (prediction - mean) / std

    def z_score_to_probability(self, z_score: float) -> float:
        return stats.norm.cdf(z_score) * 100.0

    def predict(self, prices: np.ndarray, current_price: float) -> Dict[str, float]:
        predicted_price = self.extrapolate_next(prices)
        z_score = self.calculate_z_score(predicted_price, prices)

        if predicted_price > current_price:
            direction = "UP"
        else:
            direction = "DOWN"

        confidence = min(99.0, 50 + abs(z_score) * 15)
        self.last_probability = confidence

        return {
            'predicted_price': predicted_price,
            'current_price': current_price,
            'price_change': predicted_price - current_price,
            'z_score': z_score,
            'probability': confidence,
            'direction': direction
        }


class SignalGenerator:
    """Sistema de Saída e Gatilho"""

    def __init__(self, probability_threshold: float = 85.0,
                 min_kinetic_energy: float = 1e-10,
                 min_price_change_pips: float = 1.0):
        self.probability_threshold = probability_threshold
        self.min_kinetic_energy = min_kinetic_energy
        self.min_price_change_pips = min_price_change_pips
        self.pip_value = 0.0001

    def generate_signal(self, oracle_result: Dict, lagrangian_result: Dict,
                        kde_result: Dict, current_price: float) -> Dict[str, Any]:
        predicted_price = oracle_result['predicted_price']
        probability = oracle_result['probability']
        direction = oracle_result['direction']
        price_change = oracle_result['price_change']
        kinetic_energy = lagrangian_result['kinetic_energy']
        action_derivative = lagrangian_result['action_derivative']

        nearest_support = kde_result['supports'][0] if kde_result['supports'] else current_price * 0.995
        nearest_resistance = kde_result['resistances'][0] if kde_result['resistances'] else current_price * 1.005

        condition_1 = direction == "UP"
        condition_2 = probability >= self.probability_threshold
        condition_3 = kinetic_energy >= self.min_kinetic_energy
        price_change_pips = abs(price_change) / self.pip_value
        condition_4 = price_change_pips >= self.min_price_change_pips

        if condition_1 and condition_2 and condition_3:
            signal = "COMPRA FORTE" if probability >= 90 else "COMPRA"
        elif not condition_1 and condition_2 and condition_3:
            signal = "VENDA FORTE" if probability >= 90 else "VENDA"
        else:
            signal = "NEUTRO"

        if "COMPRA" in signal:
            entry_price = nearest_support + (current_price - nearest_support) * 0.3
            entry_price = max(entry_price, current_price - 5 * self.pip_value)
        elif "VENDA" in signal:
            entry_price = nearest_resistance - (nearest_resistance - current_price) * 0.3
            entry_price = min(entry_price, current_price + 5 * self.pip_value)
        else:
            entry_price = current_price

        return {
            "Sinal": signal,
            "Probabilidade_Alta": f"{probability:.1f}%",
            "Preco_Entrada_Otimizado": round(entry_price, 5),
            "Suporte_Dinamico_KDE": round(nearest_support, 5),
            "Resistencia_Dinamica_KDE": round(nearest_resistance, 5),
            "Energia_Mercado_Joules": round(kinetic_energy * 1e9, 1),
            "_detalhes": {
                "preco_atual": round(current_price, 5),
                "preco_previsto": round(predicted_price, 5),
                "variacao_pips": round(price_change_pips, 1),
                "z_score": round(oracle_result['z_score'], 2),
                "energia_cinetica": kinetic_energy,
                "energia_potencial": lagrangian_result['potential_energy'],
                "lagrangiana": lagrangian_result['lagrangian'],
                "derivada_acao": action_derivative,
                "condicoes": {
                    "direcao_alta": condition_1,
                    "probabilidade_ok": condition_2,
                    "energia_ok": condition_3,
                    "movimento_ok": condition_4
                }
            }
        }


# ============================================================================
# CLASSE PRINCIPAL - MathCore Engine (Integrado com TickTrader)
# ============================================================================

class MathCore:
    """Motor Matemático Principal integrado com TickTrader API"""

    def __init__(self):
        # Módulos de análise
        self.ukf = UnscentedKalmanFilter()
        self.lagrangian = LagrangianMechanics()
        self.kde = KDESupportResistance()
        self.oracle = ProbabilityOracle()
        self.signal_generator = SignalGenerator()

        # Clientes de API
        self.feed: Optional[TickTraderFeed] = None
        self.rest: Optional[TickTraderREST] = None

        # Estado
        self.initialized = False
        self.current_data: Optional[pd.DataFrame] = None
        self.filtered_prices: List[float] = []
        self.tick_buffer: deque = deque(maxlen=500)

    async def initialize(self, symbol: str = "EURUSD"):
        """Inicializa o sistema conectando ao WebSocket feed"""
        logger.info("=" * 60)
        logger.info("  Lagrangian-Kalman Trajectory Estimator")
        logger.info("  Inicializando via WebSocket Feed...")
        logger.info("=" * 60)

        try:
            # Conectar ao feed WebSocket
            self.feed = TickTraderFeed()
            await self.feed.connect()
            await self.feed.subscribe_symbol(symbol)

            logger.info(f"[MathCore] Conectado ao WebSocket Feed para {symbol}")
            logger.info("[MathCore] Aguardando ticks para calibração inicial (30 segundos)...")

            # Coletar ticks por 30 segundos para calibração inicial
            start_time = datetime.utcnow()
            initial_prices = []

            while (datetime.utcnow() - start_time).total_seconds() < 30:
                tick = await self.feed.get_tick()
                if tick and tick.symbol == symbol and tick.mid > 0:
                    initial_prices.append(tick.mid)
                    if len(initial_prices) % 10 == 0:
                        logger.info(f"[MathCore] Coletados {len(initial_prices)} ticks...")
                else:
                    await asyncio.sleep(0.1)

            if len(initial_prices) < 10:
                logger.error(f"[ERRO] Insuficientes ticks coletados: {len(initial_prices)}")
                await self.feed.disconnect()
                return False

            logger.info(f"[MathCore] {len(initial_prices)} ticks coletados para calibração")

            # Inicializar UKF com primeiro preço
            initial_price = initial_prices[0]
            self.ukf.initialize(initial_price)

            # Processar ticks coletados pelo UKF
            logger.info("[MathCore] Processando ticks pelo UKF...")
            for price in initial_prices:
                P_real, v = self.ukf.process_candle(price)
                self.filtered_prices.append(P_real)

            # Ajustar KDE
            logger.info("[MathCore] Ajustando modelo KDE...")
            self.kde.fit(np.array(initial_prices))

            self.initialized = True
            logger.info("[MathCore] Sistema inicializado com sucesso!")
            logger.info(f"[MathCore] Preço atual: {initial_prices[-1]:.5f}")

            return True

        except Exception as e:
            logger.exception(f"[ERRO] Falha na inicialização: {e}")
            if self.feed:
                await self.feed.disconnect()
            return False

    def process_new_candle(self, candle: Dict) -> Dict:
        """Processa uma nova vela e gera análise completa"""
        if not self.initialized:
            raise RuntimeError("MathCore não inicializado")

        close = float(candle.get('close', candle.get('Close', 0)))
        volume = float(candle.get('volume', candle.get('Volume', 1000)))

        # 1. Processar pelo UKF
        P_real, velocity = self.ukf.process_candle(close)
        self.filtered_prices.append(P_real)

        # Manter array atualizado
        prices_array = np.array(self.filtered_prices[-500:])

        # 2. Calcular métricas Lagrangianas
        lagrangian_result = self.lagrangian.process_candle(
            filtered_price=P_real,
            velocity=velocity,
            volume=volume,
            prices=prices_array
        )

        # 3. Atualizar KDE
        if len(self.filtered_prices) >= 50:
            self.kde.fit(prices_array)

        kde_result = self.kde.find_support_resistance(close)

        # 4. Oráculo de Probabilidade
        oracle_result = self.oracle.predict(prices_array, close)

        # 5. Gerar Sinal
        signal = self.signal_generator.generate_signal(
            oracle_result=oracle_result,
            lagrangian_result=lagrangian_result,
            kde_result=kde_result,
            current_price=close
        )

        return signal

    def process_tick(self, tick: TickData) -> Optional[Dict]:
        """Processa um tick e retorna sinal se houver dados suficientes"""
        self.tick_buffer.append({
            'timestamp': tick.timestamp,
            'bid': tick.bid,
            'ask': tick.ask,
            'mid': tick.mid,
            'volume': tick.bid_volume + tick.ask_volume
        })

        # Criar vela sintética a partir do tick
        candle = {
            'close': tick.mid,
            'volume': tick.bid_volume + tick.ask_volume
        }

        return self.process_new_candle(candle)

    async def analyze_current(self, symbol: str = "EURUSD") -> Dict:
        """Analisa estado atual do mercado"""
        if not self.initialized:
            success = await self.initialize(symbol)
            if not success:
                return {"Sinal": "ERRO", "Mensagem": "Falha na inicialização"}

        # Obter tick mais recente
        if self.feed and self.feed.is_connected():
            tick = await self.feed.get_tick()
            if tick and tick.symbol == symbol and tick.mid > 0:
                return self.process_tick(tick)

        # Usar último preço filtrado se não houver tick novo
        if len(self.filtered_prices) > 0:
            last_price = self.filtered_prices[-1]
            candle = {'close': last_price, 'volume': 1000}
            return self.process_new_candle(candle)

        return {"Sinal": "ERRO", "Mensagem": "Sem dados disponíveis"}

    async def run_live(self, symbol: str = "EURUSD", interval_seconds: int = 60):
        """Executa análise em tempo real via WebSocket"""
        if not self.initialized:
            success = await self.initialize(symbol)
            if not success:
                logger.error("[ERRO] Não foi possível iniciar modo live")
                return

        # Feed já conectado durante initialize()

        logger.info("=" * 60)
        logger.info(f"  Iniciando análise em tempo real - {symbol}")
        logger.info(f"  Intervalo: {interval_seconds} segundos")
        logger.info("=" * 60)

        last_analysis_time = datetime.utcnow()
        consecutive_failures = 0

        try:
            while self.feed.is_connected():
                # Obter tick mais recente
                tick = await self.feed.get_tick()

                if tick and tick.symbol == symbol:
                    consecutive_failures = 0

                    # Verificar se é hora de fazer análise
                    now = datetime.utcnow()
                    if (now - last_analysis_time).total_seconds() >= interval_seconds:
                        signal = self.process_tick(tick)
                        last_analysis_time = now

                        # Exibir resultado
                        logger.info("-" * 40)
                        logger.info(f"[{now.strftime('%H:%M:%S')}] Análise:")
                        output = {k: v for k, v in signal.items() if not k.startswith('_')}
                        logger.info(json.dumps(output, indent=2, ensure_ascii=False))

                        if "FORTE" in signal['Sinal']:
                            logger.info("!" * 40)
                            logger.info(f"  SINAL FORTE: {signal['Sinal']}")
                            logger.info("!" * 40)
                else:
                    await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("[MathCore] Encerrando...")
        except Exception as e:
            logger.exception(f"[ERRO] {e}")
        finally:
            if self.feed:
                await self.feed.disconnect()
            if self.rest:
                await self.rest.disconnect()

    async def disconnect(self):
        """Desconecta dos serviços"""
        if self.feed:
            await self.feed.disconnect()
        if self.rest:
            await self.rest.disconnect()


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

async def main():
    """Função principal"""
    print("\n" + "=" * 70)
    print("  The Lagrangian-Kalman Trajectory Estimator")
    print("  Sistema de Previsão para Scalping EURUSD")
    print("=" * 70)

    engine = MathCore()

    try:
        # Inicializar com dados históricos
        success = await engine.initialize("EURUSD")

        if not success:
            print("[ERRO] Falha na inicialização")
            return

        # Analisar estado atual
        print("\n[INFO] Executando análise do momento atual...")
        signal = await engine.analyze_current()

        # Exibir resultado
        print("\n" + "=" * 50)
        print("  RESULTADO DA ANÁLISE")
        print("=" * 50)

        output = {k: v for k, v in signal.items() if not k.startswith('_')}
        print("\nFormato de Retorno (JSON):")
        print("-" * 30)
        print(json.dumps(output, indent=2, ensure_ascii=False))

        if '_detalhes' in signal:
            print("\n" + "-" * 50)
            print("Detalhes Técnicos:")
            print("-" * 30)
            details = signal['_detalhes']
            print(f"  Preço Atual:      {details['preco_atual']}")
            print(f"  Preço Previsto:   {details['preco_previsto']}")
            print(f"  Variação (pips):  {details['variacao_pips']}")
            print(f"  Z-Score:          {details['z_score']}")
            print(f"  Energia Cinética: {details['energia_cinetica']:.2e}")
            print(f"  Energia Potencial:{details['energia_potencial']:.2e}")
            print(f"  Lagrangiana:      {details['lagrangiana']:.2e}")

            print("\n  Condições de Gatilho:")
            conds = details['condicoes']
            print(f"    Direção Alta:     {'SIM' if conds['direcao_alta'] else 'NAO'}")
            print(f"    Probabilidade OK: {'SIM' if conds['probabilidade_ok'] else 'NAO'}")
            print(f"    Energia OK:       {'SIM' if conds['energia_ok'] else 'NAO'}")
            print(f"    Movimento OK:     {'SIM' if conds['movimento_ok'] else 'NAO'}")

        print("\n" + "=" * 50)
        print("  Análise concluída!")
        print("=" * 50)

    finally:
        await engine.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
