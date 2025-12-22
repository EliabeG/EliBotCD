# backtest/quick_optimize.py
"""
Otimizador rápido de estratégias - versão simplificada.
"""
import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd

root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_hist = _load_module('historical_data', f'{root_dir}/backtest/historical_data.py')
_engine = _load_module('bar_backtest_engine', f'{root_dir}/backtest/bar_backtest_engine.py')

HistoricalDataClient = _hist.HistoricalDataClient
BarBacktestEngine = _engine.BarBacktestEngine

from datetime import timedelta


@dataclass
class Result:
    name: str
    params: Dict
    profit: float
    win_rate: float
    pf: float
    sharpe: float
    dd: float
    trades: int


class SMAStrategy:
    def __init__(self, fast=10, slow=30, sl=20, tp=40):
        self.name = f"SMA_{fast}_{slow}"
        self.fast, self.slow, self.sl, self.tp = fast, slow, sl, tp
        self.active = True

    async def on_bar(self, ctx):
        bars = ctx.get('bars')
        if bars is None or len(bars) < self.slow + 2:
            return None
        if len(ctx.get('open_positions', [])) > 0:
            return None

        closes = bars['close'].values
        fast_ma = pd.Series(closes).rolling(self.fast).mean().values
        slow_ma = pd.Series(closes).rolling(self.slow).mean().values

        if pd.isna(fast_ma[-1]) or pd.isna(slow_ma[-1]) or pd.isna(fast_ma[-2]) or pd.isna(slow_ma[-2]):
            return None

        price = ctx['tick']['mid']
        pip = 0.0001

        prev = fast_ma[-2] - slow_ma[-2]
        curr = fast_ma[-1] - slow_ma[-1]

        if prev < 0 and curr > 0:
            return {'side': 'buy', 'size': 0.1, 'stop_loss': price - self.sl*pip,
                    'take_profit': price + self.tp*pip, 'strategy_name': self.name}
        if prev > 0 and curr < 0:
            return {'side': 'sell', 'size': 0.1, 'stop_loss': price + self.sl*pip,
                    'take_profit': price - self.tp*pip, 'strategy_name': self.name}
        return None


class RSIStrategy:
    def __init__(self, period=14, oversold=30, overbought=70, sl=25, tp=25):
        self.name = f"RSI_{period}"
        self.period, self.oversold, self.overbought = period, oversold, overbought
        self.sl, self.tp = sl, tp
        self.active = True

    def _rsi(self, closes):
        delta = closes.diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / loss.replace(0, 1e-10)
        return (100 - 100 / (1 + rs)).iloc[-1]

    async def on_bar(self, ctx):
        bars = ctx.get('bars')
        if bars is None or len(bars) < self.period + 2:
            return None
        if len(ctx.get('open_positions', [])) > 0:
            return None

        rsi = self._rsi(bars['close'])
        if pd.isna(rsi):
            return None

        price = ctx['tick']['mid']
        pip = 0.0001

        if rsi < self.oversold:
            return {'side': 'buy', 'size': 0.1, 'stop_loss': price - self.sl*pip,
                    'take_profit': price + self.tp*pip, 'strategy_name': self.name}
        if rsi > self.overbought:
            return {'side': 'sell', 'size': 0.1, 'stop_loss': price + self.sl*pip,
                    'take_profit': price - self.tp*pip, 'strategy_name': self.name}
        return None


class BBStrategy:
    def __init__(self, period=20, std=2.0, sl=20, tp=30):
        self.name = f"BB_{period}"
        self.period, self.std, self.sl, self.tp = period, std, sl, tp
        self.active = True

    async def on_bar(self, ctx):
        bars = ctx.get('bars')
        if bars is None or len(bars) < self.period + 2:
            return None
        if len(ctx.get('open_positions', [])) > 0:
            return None

        closes = bars['close']
        sma = closes.rolling(self.period).mean()
        std = closes.rolling(self.period).std()
        upper = sma + self.std * std
        lower = sma - self.std * std

        price = ctx['tick']['mid']
        pip = 0.0001

        if price <= lower.iloc[-1]:
            return {'side': 'buy', 'size': 0.1, 'stop_loss': price - self.sl*pip,
                    'take_profit': price + self.tp*pip, 'strategy_name': self.name}
        if price >= upper.iloc[-1]:
            return {'side': 'sell', 'size': 0.1, 'stop_loss': price + self.sl*pip,
                    'take_profit': price - self.tp*pip, 'strategy_name': self.name}
        return None


class MACDStrategy:
    def __init__(self, fast=12, slow=26, signal=9, sl=25, tp=35):
        self.name = f"MACD_{fast}_{slow}"
        self.fast, self.slow, self.signal = fast, slow, signal
        self.sl, self.tp = sl, tp
        self.active = True

    async def on_bar(self, ctx):
        bars = ctx.get('bars')
        if bars is None or len(bars) < self.slow + self.signal + 2:
            return None
        if len(ctx.get('open_positions', [])) > 0:
            return None

        closes = bars['close']
        ema_fast = closes.ewm(span=self.fast, adjust=False).mean()
        ema_slow = closes.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.signal, adjust=False).mean()

        price = ctx['tick']['mid']
        pip = 0.0001

        if macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
            return {'side': 'buy', 'size': 0.1, 'stop_loss': price - self.sl*pip,
                    'take_profit': price + self.tp*pip, 'strategy_name': self.name}
        if macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
            return {'side': 'sell', 'size': 0.1, 'stop_loss': price + self.sl*pip,
                    'take_profit': price - self.tp*pip, 'strategy_name': self.name}
        return None


async def download_data(symbol, timeframe, count):
    client = HistoricalDataClient()
    if not await client.connect():
        return pd.DataFrame()

    try:
        all_bars = []
        remaining = count
        end_time = None

        while remaining > 0:
            batch = min(remaining, 1000)
            df = await client.get_bars(symbol, timeframe, count=batch, end_time=end_time)
            if df.empty:
                break
            all_bars.append(df)
            remaining -= len(df)
            if 'timestamp' in df.columns:
                mins = {'M1': 1, 'M5': 5, 'M15': 15, 'H1': 60}.get(timeframe, 5)
                end_time = df['timestamp'].min() - timedelta(minutes=mins)
            await asyncio.sleep(0.15)

        if all_bars:
            result = pd.concat(all_bars, ignore_index=True)
            return result.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        return pd.DataFrame()
    finally:
        await client.disconnect()


async def test_strategy(strategy, bars, symbol, timeframe):
    engine = BarBacktestEngine(initial_balance=10000, commission_per_lot=7.0, slippage_pips=0.5)
    result = await engine.run(strategy, bars, symbol, timeframe)
    m = result.metrics
    return Result(
        name=strategy.name,
        params=strategy.__dict__.copy(),
        profit=m.net_profit,
        win_rate=m.win_rate,
        pf=m.profit_factor,
        sharpe=m.sharpe_ratio,
        dd=m.max_drawdown_pct,
        trades=m.total_trades
    )


async def main():
    print("\n" + "="*70)
    print("OTIMIZAÇÃO RÁPIDA DE ESTRATÉGIAS")
    print("="*70)

    symbol, timeframe = "EURUSD", "M5"
    bars = await download_data(symbol, timeframe, 3000)

    if bars.empty:
        print("Erro: sem dados")
        return

    print(f"\nDados: {len(bars)} barras de {bars['timestamp'].iloc[0]} a {bars['timestamp'].iloc[-1]}")

    results = []

    # ===== SMA Optimization =====
    print("\n--- Otimizando SMA ---")
    sma_params = [
        (5, 20, 15, 30), (5, 20, 20, 40), (5, 20, 25, 50),
        (8, 21, 15, 30), (8, 21, 20, 40), (8, 21, 25, 50),
        (10, 30, 15, 30), (10, 30, 20, 40), (10, 30, 25, 50),
        (12, 40, 20, 40), (12, 40, 25, 50), (12, 40, 30, 60),
        (15, 50, 25, 50), (15, 50, 30, 60), (15, 50, 35, 70),
    ]
    for i, (f, s, sl, tp) in enumerate(sma_params):
        r = await test_strategy(SMAStrategy(f, s, sl, tp), bars, symbol, timeframe)
        results.append(('SMA', r))
        print(f"  [{i+1}/{len(sma_params)}] SMA({f},{s}) SL={sl} TP={tp}: ${r.profit:.0f} WR={r.win_rate:.0f}% PF={r.pf:.2f}")

    # ===== RSI Optimization =====
    print("\n--- Otimizando RSI ---")
    rsi_params = [
        (7, 25, 75, 15, 20), (7, 25, 75, 20, 25), (7, 25, 75, 25, 30),
        (7, 30, 70, 15, 20), (7, 30, 70, 20, 25), (7, 30, 70, 25, 30),
        (10, 25, 75, 20, 25), (10, 25, 75, 25, 30), (10, 30, 70, 20, 25),
        (14, 25, 75, 20, 25), (14, 25, 75, 25, 30), (14, 30, 70, 20, 25),
        (14, 30, 70, 25, 30), (14, 35, 65, 20, 25), (14, 35, 65, 25, 30),
        (21, 30, 70, 25, 30), (21, 30, 70, 30, 35), (21, 35, 65, 25, 30),
    ]
    for i, (p, os, ob, sl, tp) in enumerate(rsi_params):
        r = await test_strategy(RSIStrategy(p, os, ob, sl, tp), bars, symbol, timeframe)
        results.append(('RSI', r))
        print(f"  [{i+1}/{len(rsi_params)}] RSI({p}) {os}/{ob} SL={sl} TP={tp}: ${r.profit:.0f} WR={r.win_rate:.0f}% PF={r.pf:.2f}")

    # ===== Bollinger Bands Optimization =====
    print("\n--- Otimizando Bollinger Bands ---")
    bb_params = [
        (15, 1.5, 15, 25), (15, 2.0, 15, 25), (15, 2.0, 20, 30),
        (20, 1.5, 15, 25), (20, 2.0, 15, 25), (20, 2.0, 20, 30),
        (20, 2.5, 20, 30), (20, 2.5, 25, 35), (25, 2.0, 20, 30),
        (25, 2.0, 25, 35), (25, 2.5, 25, 35), (30, 2.0, 25, 35),
    ]
    for i, (p, std, sl, tp) in enumerate(bb_params):
        r = await test_strategy(BBStrategy(p, std, sl, tp), bars, symbol, timeframe)
        results.append(('BB', r))
        print(f"  [{i+1}/{len(bb_params)}] BB({p},{std}) SL={sl} TP={tp}: ${r.profit:.0f} WR={r.win_rate:.0f}% PF={r.pf:.2f}")

    # ===== MACD Optimization =====
    print("\n--- Otimizando MACD ---")
    macd_params = [
        (8, 21, 7, 20, 30), (8, 21, 9, 20, 30), (8, 21, 9, 25, 35),
        (12, 26, 7, 20, 30), (12, 26, 9, 20, 30), (12, 26, 9, 25, 35),
        (12, 26, 9, 30, 45), (12, 26, 12, 25, 35), (16, 30, 9, 25, 35),
        (16, 30, 9, 30, 45), (16, 30, 12, 30, 45),
    ]
    for i, (f, s, sig, sl, tp) in enumerate(macd_params):
        r = await test_strategy(MACDStrategy(f, s, sig, sl, tp), bars, symbol, timeframe)
        results.append(('MACD', r))
        print(f"  [{i+1}/{len(macd_params)}] MACD({f},{s},{sig}) SL={sl} TP={tp}: ${r.profit:.0f} WR={r.win_rate:.0f}% PF={r.pf:.2f}")

    # ===== Results Summary =====
    print("\n" + "="*70)
    print("MELHORES RESULTADOS POR ESTRATÉGIA")
    print("="*70)

    for strat_type in ['SMA', 'RSI', 'BB', 'MACD']:
        strat_results = [r for t, r in results if t == strat_type]
        if strat_results:
            # Ordenar por profit
            best = sorted(strat_results, key=lambda x: x.profit, reverse=True)[:3]
            print(f"\n{strat_type} - Top 3:")
            for i, r in enumerate(best, 1):
                print(f"  {i}. {r.name}: Profit=${r.profit:.2f} | WR={r.win_rate:.1f}% | PF={r.pf:.2f} | DD={r.dd:.1f}% | Trades={r.trades}")

    # Overall best
    all_results = [r for _, r in results]
    profitable = [r for r in all_results if r.profit > 0 and r.trades >= 5]

    print("\n" + "="*70)
    print("TOP 10 GERAL (Profit > 0, Trades >= 5)")
    print("="*70)

    if profitable:
        top10 = sorted(profitable, key=lambda x: x.profit, reverse=True)[:10]
        for i, r in enumerate(top10, 1):
            print(f"{i:2}. {r.name:15} | Profit: ${r.profit:8.2f} | WR: {r.win_rate:5.1f}% | PF: {r.pf:5.2f} | Sharpe: {r.sharpe:5.2f} | DD: {r.dd:4.1f}%")
    else:
        print("Nenhuma estratégia lucrativa encontrada com os parâmetros testados.")

    # Best overall by different metrics
    if profitable:
        print("\n" + "="*70)
        print("MELHORES POR MÉTRICA")
        print("="*70)

        best_profit = max(profitable, key=lambda x: x.profit)
        best_wr = max(profitable, key=lambda x: x.win_rate)
        best_pf = max(profitable, key=lambda x: x.pf if x.pf < 100 else 0)
        best_sharpe = max(profitable, key=lambda x: x.sharpe)
        lowest_dd = min(profitable, key=lambda x: x.dd)

        print(f"Maior Profit:    {best_profit.name} - ${best_profit.profit:.2f}")
        print(f"Maior Win Rate:  {best_wr.name} - {best_wr.win_rate:.1f}%")
        print(f"Maior PF:        {best_pf.name} - {best_pf.pf:.2f}")
        print(f"Maior Sharpe:    {best_sharpe.name} - {best_sharpe.sharpe:.2f}")
        print(f"Menor Drawdown:  {lowest_dd.name} - {lowest_dd.dd:.1f}%")

    print("\n" + "="*70)
    print("OTIMIZAÇÃO CONCLUÍDA!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
