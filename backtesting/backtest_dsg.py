#!/usr/bin/env python3
"""
================================================================================
BACKTEST DSG - Detector de Singularidade Gravitacional
================================================================================
Backtest com dados REAIS da FXOpen API - Sem look-ahead bias
"""

import sys
import os
import urllib.request
import ssl
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional
import time as time_module
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional


@dataclass
class Bar:
    """Representa uma barra/candle OHLCV"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def download_fxopen_data(symbol: str, periodicity: str,
                         start_date: datetime, end_date: datetime,
                         verbose: bool = True) -> List[Bar]:
    """Baixa dados historicos REAIS da API REST publica da FXOpen"""
    if verbose:
        print(f"\n  Baixando dados REAIS da FXOpen API...")
        print(f"  Simbolo: {symbol}")
        print(f"  Periodicidade: {periodicity}")
        print(f"  Periodo: {start_date.date()} a {end_date.date()}")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    all_bars = []
    current_ts = int(end_date.timestamp() * 1000)
    start_ts = int(start_date.timestamp() * 1000)
    batch = 0
    max_retries = 3

    while current_ts > start_ts:
        batch += 1
        url = (f"https://marginalttdemowebapi.fxopen.net/api/v2/public/"
               f"quotehistory/{symbol}/{periodicity}/bars/bid"
               f"?timestamp={current_ts}&count=-1000")

        for retry in range(max_retries):
            try:
                with urllib.request.urlopen(url, context=ctx, timeout=30) as response:
                    data = json.loads(response.read().decode())

                if "Bars" not in data or not data["Bars"]:
                    if verbose:
                        print(f"    Batch {batch}: Sem mais dados")
                    return _finalize_bars(all_bars, verbose)

                for bar_data in data["Bars"]:
                    if start_ts <= bar_data["Timestamp"] <= int(end_date.timestamp() * 1000):
                        ts = datetime.fromtimestamp(bar_data["Timestamp"] / 1000, tz=timezone.utc)
                        all_bars.append(Bar(
                            timestamp=ts,
                            open=float(bar_data["Open"]),
                            high=float(bar_data["High"]),
                            low=float(bar_data["Low"]),
                            close=float(bar_data["Close"]),
                            volume=float(bar_data.get("Volume", 0))
                        ))

                oldest = min(data["Bars"], key=lambda x: x["Timestamp"])
                new_ts = oldest["Timestamp"] - 1

                if new_ts >= current_ts:
                    return _finalize_bars(all_bars, verbose)
                current_ts = new_ts

                if verbose and batch % 5 == 0:
                    print(f"    Batch {batch}: {len(all_bars)} barras...")
                time_module.sleep(0.3)  # Rate limiting
                break

            except Exception as e:
                if retry < max_retries - 1:
                    if verbose:
                        print(f"    Retry {retry+1}/{max_retries} - {e}")
                    time_module.sleep(2)
                else:
                    if verbose:
                        print(f"  Erro no batch {batch}: {e}")
                    return _finalize_bars(all_bars, verbose)

    return _finalize_bars(all_bars, verbose)


def _finalize_bars(all_bars: List[Bar], verbose: bool = True) -> List[Bar]:
    """Remove duplicatas e ordena"""
    seen = set()
    unique_bars = []
    for bar in all_bars:
        ts_key = int(bar.timestamp.timestamp())
        if ts_key not in seen:
            seen.add(ts_key)
            unique_bars.append(bar)

    unique_bars.sort(key=lambda b: b.timestamp)
    if verbose:
        print(f"  Total de barras: {len(unique_bars)}")
        if unique_bars:
            print(f"  Periodo real: {unique_bars[0].timestamp} a {unique_bars[-1].timestamp}")
    return unique_bars


class DSGBacktester:
    """Engine de backtest para DSG - Sem look-ahead bias"""

    def __init__(self, symbol: str = "EURUSD"):
        self.symbol = symbol
        self.pip = 0.0001
        self.spread = 1.0  # pips
        self.bars = []

    def load_data(self, start_date: datetime, end_date: datetime,
                  periodicity: str = "H1", verbose: bool = True) -> bool:
        """Carrega dados historicos da FXOpen"""
        self.bars = download_fxopen_data(
            self.symbol, periodicity, start_date, end_date, verbose
        )
        return len(self.bars) > 0

    def run_backtest(self, config: Dict, verbose: bool = True) -> Dict:
        """
        Executa backtest com configuracao especifica

        IMPORTANTE: Sem look-ahead bias
        - Cada barra so usa dados ate aquele ponto
        - Entrada na PROXIMA barra apos sinal
        - Saida verificada barra a barra
        """
        if verbose:
            print(f"\n  Testando configuracao DSG...")
            print(f"  Params: Ricci<={config.get('ricci_collapse_threshold', -0.5):.2f}, "
                  f"Tidal>={config.get('tidal_force_threshold', 0.1):.3f}, "
                  f"EH<={config.get('event_horizon_threshold', 0.001):.4f}")
            print(f"  SL={config.get('stop_loss_pips', 30)}, TP={config.get('take_profit_pips', 60)}")

        # Cria detector DSG com configuracao
        dsg = DetectorSingularidadeGravitacional(
            c_base=config.get('c_base', 1.0),
            gamma=config.get('gamma', 0.1),
            ricci_collapse_threshold=config.get('ricci_collapse_threshold', -0.5),
            tidal_force_threshold=config.get('tidal_force_threshold', 0.1),
            event_horizon_threshold=config.get('event_horizon_threshold', 0.001),
            lookback_window=config.get('lookback_window', 50)
        )

        min_prices = config.get('min_prices', 100)
        sl_pips = config.get('stop_loss_pips', 30.0)
        tp_pips = config.get('take_profit_pips', 60.0)

        # Buffers para simulacao em tempo real
        prices_buf = deque(maxlen=600)
        bid_vols_buf = deque(maxlen=600)
        ask_vols_buf = deque(maxlen=600)

        # Calcula analise DSG para cada barra (sem look-ahead)
        dsg_data = []
        for i, bar in enumerate(self.bars):
            prices_buf.append(bar.close)

            # Volume sintetico baseado em range (sem usar dados futuros)
            vol = (bar.high - bar.low) * 1000000 + 100
            bid_vols_buf.append(vol * 0.5)
            ask_vols_buf.append(vol * 0.5)

            if len(prices_buf) < min_prices:
                continue

            try:
                result = dsg.analyze(
                    np.array(prices_buf),
                    np.array(bid_vols_buf),
                    np.array(ask_vols_buf)
                )

                dsg_data.append({
                    'idx': i,
                    'timestamp': bar.timestamp,
                    'price': bar.close,
                    'high': bar.high,
                    'low': bar.low,
                    'signal': result['signal'],
                    'confidence': result['confidence'],
                    'ricci': result['Ricci_Scalar'],
                    'tidal': result['Tidal_Force_Magnitude'],
                    'event_horizon': result['Event_Horizon_Distance'],
                    'ricci_collapsing': result['ricci_collapsing'],
                    'crossing_horizon': result['crossing_horizon'],
                    'geodesic_direction': result['geodesic_direction']
                })
            except Exception:
                continue

            if verbose and (i + 1) % 2000 == 0:
                print(f"    Processado: {i+1}/{len(self.bars)} barras...")

        if verbose:
            print(f"  Pontos DSG calculados: {len(dsg_data)}")

        # Encontra sinais validos (signal != 0 e confidence >= 0.5)
        signals = []
        min_confidence = config.get('min_confidence', 0.5)
        signal_cooldown = 0

        for d in dsg_data:
            if signal_cooldown > 0:
                signal_cooldown -= 1
                continue

            if d['signal'] != 0 and d['confidence'] >= min_confidence:
                signals.append({
                    'idx': d['idx'],
                    'price': d['price'],
                    'direction': d['signal'],  # 1 = LONG, -1 = SHORT
                    'timestamp': d['timestamp'],
                    'confidence': d['confidence'],
                    'ricci': d['ricci'],
                    'tidal': d['tidal']
                })
                signal_cooldown = config.get('signal_cooldown', 30)

        if verbose:
            print(f"  Sinais gerados: {len(signals)}")

        if len(signals) < 1:
            return {
                'config': config,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'pnl': 0,
                'win_rate': 0,
                'pf': 0,
                'max_dd': 0,
                'signals': 0,
                'trade_details': []
            }

        # Executa trades (entrada na proxima barra - sem look-ahead)
        pnls = []
        trade_details = []

        for sig in signals:
            bar_idx = sig['idx']

            # IMPORTANTE: Entrada na PROXIMA barra (sem look-ahead)
            if bar_idx + 1 >= len(self.bars):
                continue

            entry_bar = self.bars[bar_idx + 1]
            entry = entry_bar.open  # Entrada no OPEN da proxima barra
            direction = sig['direction']

            # Calcula niveis de SL e TP
            sl_price = entry - direction * sl_pips * self.pip
            tp_price = entry + direction * tp_pips * self.pip

            pnl = 0
            exit_reason = "timeout"
            exit_price = entry
            exit_time = entry_bar.timestamp

            # Verifica saida barra a barra (sem look-ahead)
            max_bars = config.get('max_bars_in_trade', 500)
            for j in range(bar_idx + 2, min(bar_idx + 2 + max_bars, len(self.bars))):
                b = self.bars[j]

                if direction == 1:  # LONG
                    # Verifica stop loss primeiro (conservador)
                    if b.low <= sl_price:
                        pnl = -sl_pips - self.spread
                        exit_reason = "stop_loss"
                        exit_price = sl_price
                        exit_time = b.timestamp
                        break
                    # Depois verifica take profit
                    if b.high >= tp_price:
                        pnl = tp_pips - self.spread
                        exit_reason = "take_profit"
                        exit_price = tp_price
                        exit_time = b.timestamp
                        break
                else:  # SHORT
                    if b.high >= sl_price:
                        pnl = -sl_pips - self.spread
                        exit_reason = "stop_loss"
                        exit_price = sl_price
                        exit_time = b.timestamp
                        break
                    if b.low <= tp_price:
                        pnl = tp_pips - self.spread
                        exit_reason = "take_profit"
                        exit_price = tp_price
                        exit_time = b.timestamp
                        break

            # Timeout: fecha no preco de fechamento
            if pnl == 0:
                exit_idx = min(bar_idx + 1 + max_bars, len(self.bars) - 1)
                exit_price = self.bars[exit_idx].close
                exit_time = self.bars[exit_idx].timestamp
                pnl = direction * (exit_price - entry) / self.pip - self.spread

            pnls.append(pnl)
            trade_details.append({
                'entry_time': entry_bar.timestamp,
                'exit_time': exit_time,
                'direction': 'LONG' if direction == 1 else 'SHORT',
                'entry': entry,
                'exit': exit_price,
                'pnl_pips': round(pnl, 1),
                'result': 'WIN' if pnl > 0 else 'LOSS',
                'reason': exit_reason,
                'confidence': sig['confidence'],
                'ricci': sig['ricci'],
                'tidal': sig['tidal']
            })

        # Calcula metricas
        if not pnls:
            return {
                'config': config,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'pnl': 0,
                'win_rate': 0,
                'pf': 0,
                'max_dd': 0,
                'signals': len(signals),
                'trade_details': []
            }

        wins = sum(1 for p in pnls if p > 0)
        losses = len(pnls) - wins
        total = sum(pnls)
        wr = wins / len(pnls)
        gp = sum(p for p in pnls if p > 0) or 0.001
        gl = abs(sum(p for p in pnls if p <= 0)) or 0.001
        pf = gp / gl

        # Drawdown
        eq = np.cumsum([0] + pnls)
        peak = np.maximum.accumulate(eq + 10000)
        dd = np.max((peak - (eq + 10000)) / peak) if len(peak) > 0 else 0

        return {
            'config': config,
            'trades': len(pnls),
            'wins': wins,
            'losses': losses,
            'pnl': round(total, 1),
            'win_rate': round(wr, 4),
            'pf': round(pf, 2),
            'max_dd': round(dd * 100, 2),
            'signals': len(signals),
            'trade_details': trade_details
        }


def print_results(result: Dict, period: str = ""):
    """Imprime resultados formatados"""
    print(f"\n{'='*70}")
    print(f"  RESULTADO DSG BACKTEST")
    if period:
        print(f"  Periodo: {period}")
    print(f"{'='*70}")
    print(f"  Sinais: {result['signals']}")
    print(f"  Trades: {result['trades']}")
    print(f"  Wins: {result['wins']} | Losses: {result['losses']}")
    print(f"  Win Rate: {result['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {result['pf']:.2f}")
    print(f"  PnL Total: {result['pnl']:.1f} pips")
    print(f"  Max Drawdown: {result['max_dd']:.1f}%")

    if result.get('trade_details'):
        print(f"\n  Ultimos 10 trades:")
        for t in result['trade_details'][-10:]:
            ts_str = t['entry_time'].strftime('%Y-%m-%d %H:%M')
            print(f"    {t['direction']:5s} | {ts_str} | "
                  f"PnL: {t['pnl_pips']:+7.1f} pips | {t['result']:4s} | {t['reason']}")


def main():
    """Funcao principal - teste basico do backtest DSG"""
    print("=" * 70)
    print("  BACKTEST DSG - Detector de Singularidade Gravitacional")
    print("  Dados REAIS FXOpen | Sem Look-Ahead Bias")
    print("=" * 70)

    # Periodo: ultimos 6 meses
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=180)

    print(f"\n  Periodo: {start.date()} a {end.date()} (~6 meses)")

    backtester = DSGBacktester("EURUSD")

    if not backtester.load_data(start, end, "H1"):
        print("  ERRO: Nao foi possivel carregar dados!")
        return

    # Configuracao padrao
    config = {
        "name": "DSG Padrao",
        "min_prices": 100,
        "stop_loss_pips": 30.0,
        "take_profit_pips": 60.0,
        "ricci_collapse_threshold": -0.5,
        "tidal_force_threshold": 0.1,
        "event_horizon_threshold": 0.001,
        "lookback_window": 50,
        "c_base": 1.0,
        "gamma": 0.1,
        "min_confidence": 0.5,
        "signal_cooldown": 30
    }

    result = backtester.run_backtest(config)
    print_results(result, f"{start.date()} a {end.date()}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
