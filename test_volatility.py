#!/usr/bin/env python3
"""
Teste de Volatilidade em Tempo Real
Coleta ticks por 5 minutos e calcula volatilidade simplificada
"""
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from scipy.stats import entropy

from api import FXOpenClient, Tick
from config import settings


class TickCollector:
    """Coleta ticks e converte em candles OHLC"""

    def __init__(self, candle_seconds: int = 5):
        self.candle_seconds = candle_seconds
        self.ticks: list[Tick] = []
        self.candles: list[dict] = []
        self.current_candle: dict = None
        self.current_candle_start: datetime = None

    def add_tick(self, tick: Tick):
        """Adiciona um tick e atualiza candles"""
        self.ticks.append(tick)

        tick_time = tick.timestamp
        candle_start = tick_time.replace(
            second=(tick_time.second // self.candle_seconds) * self.candle_seconds,
            microsecond=0
        )

        if self.current_candle_start != candle_start:
            if self.current_candle is not None:
                self.candles.append(self.current_candle)

            self.current_candle_start = candle_start
            self.current_candle = {
                'timestamp': candle_start,
                'open': tick.mid,
                'high': tick.mid,
                'low': tick.mid,
                'close': tick.mid,
                'tick_count': 1
            }
        else:
            self.current_candle['high'] = max(self.current_candle['high'], tick.mid)
            self.current_candle['low'] = min(self.current_candle['low'], tick.mid)
            self.current_candle['close'] = tick.mid
            self.current_candle['tick_count'] += 1

    def get_ohlc_dataframe(self) -> pd.DataFrame:
        """Retorna DataFrame com candles OHLC"""
        all_candles = self.candles.copy()
        if self.current_candle is not None:
            all_candles.append(self.current_candle)

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(all_candles)
        df.set_index('timestamp', inplace=True)
        return df


class SimpleVolatilityIndicator:
    """
    Indicador de Volatilidade Simplificado para Tempo Real
    Usa Parkinson, Hurst simplificado e Entropia
    """

    def __init__(self, window: int = 10):
        self.window = window

    def calculate_parkinson_vol(self, high: np.ndarray, low: np.ndarray) -> float:
        """Volatilidade de Parkinson"""
        log_hl_sq = np.log(high / low) ** 2
        return np.sqrt(np.sum(log_hl_sq) / (4 * len(high) * np.log(2)))

    def calculate_simple_hurst(self, prices: np.ndarray) -> float:
        """Hurst simplificado via autocorrelação"""
        if len(prices) < 10:
            return 0.5

        returns = np.diff(np.log(prices))
        if len(returns) < 5:
            return 0.5

        # Autocorrelação lag-1
        mean_ret = np.mean(returns)
        var_ret = np.var(returns)

        if var_ret == 0:
            return 0.5

        autocorr = np.sum((returns[:-1] - mean_ret) * (returns[1:] - mean_ret)) / (len(returns) - 1) / var_ret

        # Converte autocorr para estimativa de Hurst
        # autocorr positiva -> H > 0.5 (tendência)
        # autocorr negativa -> H < 0.5 (reversão)
        hurst = 0.5 + autocorr * 0.3
        return np.clip(hurst, 0.1, 0.9)

    def calculate_entropy(self, returns: np.ndarray, bins: int = 10) -> float:
        """Entropia de Shannon normalizada"""
        if len(returns) < 5:
            return 0.5

        hist, _ = np.histogram(returns, bins=bins, density=True)
        hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        hist = hist[hist > 0]

        if len(hist) == 0:
            return 0.5

        max_entropy = np.log2(bins)
        return entropy(hist, base=2) / max_entropy

    def calculate_realized_vol(self, returns: np.ndarray) -> float:
        """Volatilidade realizada"""
        return np.std(returns) if len(returns) > 1 else 0

    def analyze(self, df: pd.DataFrame) -> dict:
        """Analisa volatilidade do DataFrame OHLC"""

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # Log returns
        returns = np.diff(np.log(close))

        # Métricas
        parkinson_vol = self.calculate_parkinson_vol(high, low)
        realized_vol = self.calculate_realized_vol(returns)
        hurst = self.calculate_simple_hurst(close)
        entropy_norm = self.calculate_entropy(returns)

        # Score de volatilidade combinado
        # V_final = vol_realizada * e^(1-H) * (1 + entropy)
        hurst_factor = np.exp(1 - hurst)
        entropy_factor = 1 + entropy_norm
        v_combined = realized_vol * hurst_factor * entropy_factor

        # Classificação baseada em percentis típicos
        # Ajustado para dados de tick de curto prazo
        vol_pips = parkinson_vol * 10000  # Converte para pips

        if vol_pips < 0.5:
            classification = "BAIXA"
        elif vol_pips < 2.0:
            classification = "MEDIA"
        else:
            classification = "ALTA"

        # Ajuste pela entropia (incerteza alta aumenta classificação)
        if entropy_norm > 0.8 and classification == "MEDIA":
            classification = "ALTA"
        elif entropy_norm < 0.3 and classification == "MEDIA":
            classification = "BAIXA"

        return {
            'parkinson_vol': parkinson_vol,
            'parkinson_pips': vol_pips,
            'realized_vol': realized_vol,
            'hurst': hurst,
            'entropy': entropy_norm,
            'v_combined': v_combined,
            'classification': classification,
            'price_range_pips': (high.max() - low.min()) * 10000,
            'trend_direction': 'UP' if close[-1] > close[0] else 'DOWN'
        }


async def main():
    """Coleta ticks por 5 minutos e calcula volatilidade"""

    COLLECTION_TIME = 5 * 60  # 5 minutos
    CANDLE_SECONDS = 5

    print("=" * 60)
    print("  EliBotCD - Análise de Volatilidade em Tempo Real")
    print("=" * 60)
    print(f"Simbolo: {settings.SYMBOL}")
    print(f"Tempo de coleta: {COLLECTION_TIME // 60} minutos")
    print(f"Tamanho do candle: {CANDLE_SECONDS} segundos")
    print("=" * 60)
    print()

    collector = TickCollector(candle_seconds=CANDLE_SECONDS)

    async def on_tick(tick: Tick):
        collector.add_tick(tick)

    client = FXOpenClient()
    client.on_tick = on_tick

    try:
        print("Conectando à FX Open...")
        if not await client.connect():
            print("Falha ao conectar!")
            return

        await client.subscribe(settings.SYMBOL)
        print()

        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(seconds=COLLECTION_TIME)

        print(f"Coletando ticks de {start_time.strftime('%H:%M:%S')} até {end_time.strftime('%H:%M:%S')} UTC")
        print("-" * 60)

        last_update = 0
        while datetime.now(timezone.utc) < end_time and client.connected:
            await asyncio.sleep(1)

            elapsed = (datetime.now(timezone.utc) - start_time).seconds
            remaining = COLLECTION_TIME - elapsed

            if elapsed // 30 > last_update:
                last_update = elapsed // 30
                print(f"  [{elapsed//60}:{elapsed%60:02d}] Ticks: {len(collector.ticks)} | "
                      f"Candles: {len(collector.candles)} | "
                      f"Restam: {remaining//60}:{remaining%60:02d}")

        print("-" * 60)
        print(f"\nColeta finalizada!")
        print(f"  Total de ticks: {len(collector.ticks)}")
        print(f"  Total de candles: {len(collector.candles) + (1 if collector.current_candle else 0)}")

        await client.disconnect()

        df = collector.get_ohlc_dataframe()

        if len(df) < 5:
            print(f"\nDados insuficientes ({len(df)} candles). Mínimo: 5")
            return

        print(f"\n" + "=" * 60)
        print("ANÁLISE DE VOLATILIDADE")
        print("=" * 60)

        indicator = SimpleVolatilityIndicator(window=10)
        result = indicator.analyze(df)

        print(f"\n  Preço Inicial: {df['close'].iloc[0]:.5f}")
        print(f"  Preço Final:   {df['close'].iloc[-1]:.5f}")
        print(f"  Direção:       {result['trend_direction']}")

        print(f"\n  Volatilidade Parkinson: {result['parkinson_pips']:.2f} pips")
        print(f"  Volatilidade Realizada: {result['realized_vol']*10000:.4f}")
        print(f"  Range Total:            {result['price_range_pips']:.1f} pips")

        print(f"\n  Expoente Hurst: {result['hurst']:.3f}", end="")
        if result['hurst'] > 0.55:
            print(" (Tendência)")
        elif result['hurst'] < 0.45:
            print(" (Reversão)")
        else:
            print(" (Neutro)")

        print(f"  Entropia:       {result['entropy']:.3f}", end="")
        if result['entropy'] > 0.7:
            print(" (Alta incerteza)")
        elif result['entropy'] < 0.3:
            print(" (Baixa incerteza)")
        else:
            print(" (Moderada)")

        print("\n" + "=" * 60)

        classification = result['classification']

        if classification == "BAIXA":
            print("  ╔════════════════════════════════════════════╗")
            print("  ║     >>> VOLATILIDADE: BAIXA <<<            ║")
            print("  ╠════════════════════════════════════════════╣")
            print("  ║  Mercado em zona de acumulação/consolidação║")
            print("  ║  Recomendação: Aguardar breakout           ║")
            print("  ╚════════════════════════════════════════════╝")
        elif classification == "MEDIA":
            print("  ╔════════════════════════════════════════════╗")
            print("  ║     >>> VOLATILIDADE: MÉDIA <<<            ║")
            print("  ╠════════════════════════════════════════════╣")
            print("  ║  Mercado operável com movimento normal     ║")
            print("  ║  Recomendação: Operar normalmente          ║")
            print("  ╚════════════════════════════════════════════╝")
        else:
            print("  ╔════════════════════════════════════════════╗")
            print("  ║     >>> VOLATILIDADE: ALTA <<<             ║")
            print("  ╠════════════════════════════════════════════╣")
            print("  ║  Mercado em zona crítica/alta atividade    ║")
            print("  ║  Recomendação: Cautela, risco elevado      ║")
            print("  ╚════════════════════════════════════════════╝")

        print("=" * 60)

        print("\nÚLTIMOS 10 CANDLES:")
        print(df[['open', 'high', 'low', 'close', 'tick_count']].tail(10).to_string())

    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário")
        await client.disconnect()
    except Exception as e:
        print(f"\nErro: {e}")
        import traceback
        traceback.print_exc()
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
