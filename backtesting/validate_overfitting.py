#!/usr/bin/env python3
"""
================================================================================
VALIDACAO DE OVERFITTING - PRM
Testa as 2 melhores configuracoes em periodo diferente (out-of-sample)
================================================================================

Usa API REST publica da FXOpen para dados historicos reais.
Configuracoes otimizadas: 2025-07-01 ate hoje
Periodo de validacao: 2025-01-01 ate hoje (inclui 6 meses antes da otimizacao)
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
from typing import List
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.alta_volatilidade.prm_riemann_mandelbrot import ProtocoloRiemannMandelbrot


@dataclass
class Bar:
    """Representa uma barra/candle OHLCV"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


# Configuracao 1: Melhor Score (7 trades, 100% WR)
CONFIG_BEST_SCORE = {
    "name": "MELHOR SCORE (7 trades, 100% WR original)",
    "min_prices": 30,
    "stop_loss_pips": 46.9,
    "take_profit_pips": 117.2,
    "hmm_threshold": 0.68,
    "lyapunov_threshold": 0.1249,
    "hmm_states_allowed": [1, 2]
}

# Configuracao 2: Mais Trades (74 trades, 42% WR)
CONFIG_MAX_TRADES = {
    "name": "MAIS TRADES (74 trades, 42% WR original)",
    "min_prices": 50,
    "stop_loss_pips": 50.0,
    "take_profit_pips": 111.5,
    "hmm_threshold": 0.97,
    "lyapunov_threshold": 0.0495,
    "hmm_states_allowed": [0, 1]
}


def download_fxopen_data(symbol: str, periodicity: str,
                         start_date: datetime, end_date: datetime) -> List[Bar]:
    """
    Baixa dados historicos REAIS da API REST publica da FXOpen

    Args:
        symbol: Simbolo (ex: 'EURUSD')
        periodicity: Periodicidade (H1, D1, etc)
        start_date: Data inicio
        end_date: Data fim

    Returns:
        Lista de barras reais
    """
    print(f"\n  Baixando dados REAIS da FXOpen API...")
    print(f"  Simbolo: {symbol}")
    print(f"  Periodicidade: {periodicity}")
    print(f"  Periodo: {start_date.date()} a {end_date.date()}")

    # SSL context
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    all_bars = []
    current_ts = int(end_date.timestamp() * 1000)
    start_ts = int(start_date.timestamp() * 1000)
    batch = 0

    while current_ts > start_ts:
        batch += 1
        url = (f"https://marginalttdemowebapi.fxopen.net/api/v2/public/"
               f"quotehistory/{symbol}/{periodicity}/bars/bid"
               f"?timestamp={current_ts}&count=-1000")

        try:
            with urllib.request.urlopen(url, context=ctx, timeout=30) as response:
                data = json.loads(response.read().decode())

            if "Bars" not in data or not data["Bars"]:
                break

            for bar_data in data["Bars"]:
                ts = datetime.fromtimestamp(bar_data["Timestamp"] / 1000, tz=timezone.utc)
                if start_ts <= bar_data["Timestamp"] <= int(end_date.timestamp() * 1000):
                    all_bars.append(Bar(
                        timestamp=ts,
                        open=float(bar_data["Open"]),
                        high=float(bar_data["High"]),
                        low=float(bar_data["Low"]),
                        close=float(bar_data["Close"]),
                        volume=float(bar_data.get("Volume", 0))
                    ))

            # Proximo batch
            oldest = min(data["Bars"], key=lambda x: x["Timestamp"])
            new_ts = oldest["Timestamp"] - 1

            if new_ts >= current_ts:
                break
            current_ts = new_ts

            print(f"    Batch {batch}: {len(all_bars)} barras carregadas...")

        except Exception as e:
            print(f"  Erro no batch {batch}: {e}")
            break

    # Remove duplicatas e ordena
    seen = set()
    unique_bars = []
    for bar in all_bars:
        ts_key = int(bar.timestamp.timestamp())
        if ts_key not in seen:
            seen.add(ts_key)
            unique_bars.append(bar)

    unique_bars.sort(key=lambda b: b.timestamp)

    print(f"  Total de barras: {len(unique_bars)}")
    if unique_bars:
        print(f"  Periodo real: {unique_bars[0].timestamp.date()} a {unique_bars[-1].timestamp.date()}")

    return unique_bars


class OverfitValidator:
    """Validador de overfitting"""

    def __init__(self, symbol: str = "EURUSD"):
        self.symbol = symbol
        self.pip = 0.0001
        self.spread = 1.0
        self.bars: List[Bar] = []

    def load_data(self, start_date: datetime, end_date: datetime) -> bool:
        """Carrega dados historicos REAIS"""
        self.bars = download_fxopen_data(self.symbol, "H1", start_date, end_date)
        return len(self.bars) > 0

    def run_backtest(self, config: dict) -> dict:
        """Executa backtest com configuracao especifica"""
        print(f"\n  Testando: {config['name']}")
        print(f"  Params: HMM>={config['hmm_threshold']:.2f}, Lyap>={config['lyapunov_threshold']:.4f}, "
              f"States={config['hmm_states_allowed']}, SL={config['stop_loss_pips']}, TP={config['take_profit_pips']}")

        prm = ProtocoloRiemannMandelbrot(
            n_states=3,
            hmm_threshold=0.1,
            lyapunov_threshold_k=0.001,
            curvature_threshold=0.0001,
            lookback_window=100
        )

        prices_buf = deque(maxlen=500)
        volumes_buf = deque(maxlen=500)

        # Pre-calcula valores PRM
        prm_data = []
        for i, bar in enumerate(self.bars):
            prices_buf.append(bar.close)
            volumes_buf.append(bar.volume if bar.volume > 0 else 1.0)

            if len(prices_buf) < config['min_prices']:
                continue

            try:
                result = prm.analyze(np.array(prices_buf), np.array(volumes_buf))
                prm_data.append({
                    'idx': i,
                    'timestamp': bar.timestamp,
                    'price': bar.close,
                    'high': bar.high,
                    'low': bar.low,
                    'hmm_prob': result['Prob_HMM'],
                    'lyapunov': result['Lyapunov_Score'],
                    'hmm_state': result['hmm_analysis']['current_state']
                })
            except:
                continue

        print(f"  Pontos PRM calculados: {len(prm_data)}")

        # Encontra sinais
        signals = []
        for d in prm_data:
            if (d['hmm_prob'] >= config['hmm_threshold'] and
                d['lyapunov'] >= config['lyapunov_threshold'] and
                d['hmm_state'] in config['hmm_states_allowed']):

                # Direcao baseada em tendencia
                if d['idx'] >= 10:
                    trend = d['price'] - self.bars[d['idx'] - 10].close
                    direction = 1 if trend > 0 else -1
                else:
                    direction = 1
                signals.append({
                    'idx': d['idx'],
                    'price': d['price'],
                    'direction': direction,
                    'timestamp': d['timestamp']
                })

        print(f"  Sinais gerados: {len(signals)}")

        if len(signals) < 1:
            return {
                'config': config['name'],
                'trades': 0, 'wins': 0, 'losses': 0,
                'pnl': 0, 'win_rate': 0, 'pf': 0,
                'max_dd': 0, 'signals': 0, 'trade_details': []
            }

        # Executa trades
        pnls = []
        trade_details = []
        sl = config['stop_loss_pips']
        tp = config['take_profit_pips']

        for sig in signals:
            bar_idx = sig['idx']
            entry = sig['price']
            direction = sig['direction']

            sl_price = entry - direction * sl * self.pip
            tp_price = entry + direction * tp * self.pip

            pnl = 0
            exit_reason = "timeout"
            exit_price = entry
            exit_time = sig['timestamp']

            for j in range(bar_idx + 1, min(bar_idx + 500, len(self.bars))):
                b = self.bars[j]
                if direction == 1:  # LONG
                    if b.low <= sl_price:
                        pnl = -sl - self.spread
                        exit_reason = "stop_loss"
                        exit_price = sl_price
                        exit_time = b.timestamp
                        break
                    if b.high >= tp_price:
                        pnl = tp - self.spread
                        exit_reason = "take_profit"
                        exit_price = tp_price
                        exit_time = b.timestamp
                        break
                else:  # SHORT
                    if b.high >= sl_price:
                        pnl = -sl - self.spread
                        exit_reason = "stop_loss"
                        exit_price = sl_price
                        exit_time = b.timestamp
                        break
                    if b.low <= tp_price:
                        pnl = tp - self.spread
                        exit_reason = "take_profit"
                        exit_price = tp_price
                        exit_time = b.timestamp
                        break

            if pnl == 0:  # Timeout
                exit_idx = min(bar_idx + 100, len(self.bars) - 1)
                exit_price = self.bars[exit_idx].close
                exit_time = self.bars[exit_idx].timestamp
                pnl = direction * (exit_price - entry) / self.pip - self.spread

            pnls.append(pnl)
            trade_details.append({
                'entry_time': sig['timestamp'],
                'exit_time': exit_time,
                'direction': 'LONG' if direction == 1 else 'SHORT',
                'entry': entry,
                'exit': exit_price,
                'pnl_pips': round(pnl, 1),
                'result': 'WIN' if pnl > 0 else 'LOSS',
                'reason': exit_reason
            })

        # Calcula metricas
        wins = sum(1 for p in pnls if p > 0)
        losses = len(pnls) - wins
        total = sum(pnls)
        wr = wins / len(pnls) if pnls else 0
        gp = sum(p for p in pnls if p > 0) or 0.001
        gl = abs(sum(p for p in pnls if p <= 0)) or 0.001
        pf = gp / gl

        # Drawdown
        eq = np.cumsum([0] + pnls)
        peak = np.maximum.accumulate(eq + 10000)
        dd = np.max((peak - (eq + 10000)) / peak) if len(peak) > 0 else 0

        return {
            'config': config['name'],
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


def print_results(result: dict, period: str):
    """Imprime resultados formatados"""
    print(f"\n{'='*70}")
    print(f"  RESULTADO: {result['config']}")
    print(f"  Periodo: {period}")
    print(f"{'='*70}")
    print(f"  Sinais: {result['signals']}")
    print(f"  Trades: {result['trades']}")
    print(f"  Wins: {result['wins']} | Losses: {result['losses']}")
    print(f"  Win Rate: {result['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {result['pf']:.2f}")
    print(f"  PnL Total: {result['pnl']:.1f} pips")
    print(f"  Max Drawdown: {result['max_dd']:.1f}%")

    if result['trade_details']:
        print(f"\n  Ultimos 10 trades:")
        for t in result['trade_details'][-10:]:
            ts = t['entry_time']
            ts_str = ts.strftime('%Y-%m-%d %H:%M') if hasattr(ts, 'strftime') else str(ts)[:16]
            print(f"    {t['direction']:5s} | {ts_str} | "
                  f"PnL: {t['pnl_pips']:+7.1f} pips | {t['result']:4s} | {t['reason']}")


def compare_results(original: dict, validation: dict, config_name: str):
    """Compara resultado original vs validacao"""
    print(f"\n{'='*70}")
    print(f"  COMPARACAO: {config_name}")
    print(f"{'='*70}")
    print(f"  {'Metrica':<20} {'Original':>15} {'Validacao':>15} {'Diff':>15}")
    print(f"  {'-'*65}")

    metrics = [
        ('Trades', original.get('trades', 0), validation['trades']),
        ('Win Rate %', original.get('win_rate', 0)*100, validation['win_rate']*100),
        ('Profit Factor', original.get('pf', 0), validation['pf']),
        ('PnL (pips)', original.get('pnl', 0), validation['pnl']),
        ('Max DD %', original.get('max_dd', 0), validation['max_dd']),
    ]

    for name, orig, val in metrics:
        diff = val - orig if isinstance(orig, (int, float)) else 'N/A'
        diff_str = f"{diff:+.1f}" if isinstance(diff, (int, float)) else diff
        print(f"  {name:<20} {orig:>15.1f} {val:>15.1f} {diff_str:>15}")


def main():
    print("=" * 70)
    print("  VALIDACAO DE OVERFITTING - PRM")
    print("  Dados REAIS da API FXOpen")
    print("=" * 70)

    validator = OverfitValidator("EURUSD")

    # Periodo de validacao: 2025-01-01 ate hoje
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    print(f"\n  Periodo original de otimizacao: 2025-07-01 a 2025-12-23 (~6 meses)")
    print(f"  Periodo de validacao: {start.date()} a {end.date()} (~12 meses)")
    print(f"  Isso adiciona 6 meses de dados OUT-OF-SAMPLE")

    if not validator.load_data(start, end):
        print("  ERRO: Nao foi possivel carregar dados!")
        return

    # Testa Config 1: Melhor Score
    result1 = validator.run_backtest(CONFIG_BEST_SCORE)
    print_results(result1, f"{start.date()} a {end.date()}")

    # Dados originais da Config 1
    original1 = {
        'trades': 7, 'wins': 7, 'losses': 0,
        'win_rate': 1.0, 'pf': 813256.41,
        'pnl': 813.3, 'max_dd': 0.0
    }
    compare_results(original1, result1, "MELHOR SCORE")

    # Testa Config 2: Mais Trades
    result2 = validator.run_backtest(CONFIG_MAX_TRADES)
    print_results(result2, f"{start.date()} a {end.date()}")

    # Dados originais da Config 2
    original2 = {
        'trades': 74, 'wins': 31, 'losses': 43,
        'win_rate': 0.4189, 'pf': 1.56,
        'pnl': 1163.2, 'max_dd': 4.28
    }
    compare_results(original2, result2, "MAIS TRADES")

    # Conclusao
    print(f"\n{'='*70}")
    print("  CONCLUSAO")
    print(f"{'='*70}")

    print(f"\n  Config 'Melhor Score':")
    if result1['trades'] == 0:
        print(f"    OVERFITTED! Nenhum trade gerado no periodo estendido.")
    elif result1['pf'] < 1.0:
        print(f"    OVERFITTED! PF caiu para {result1['pf']:.2f} (era 813,256)")
        print(f"    Esta configuracao NAO e viavel para trading real.")
    elif result1['pf'] < 2.0:
        print(f"    POSSIVEL OVERFIT. PF degradou para {result1['pf']:.2f}")
    else:
        print(f"    Performance mantida. PF = {result1['pf']:.2f}")

    print(f"\n  Config 'Mais Trades':")
    if result2['trades'] == 0:
        print(f"    PROBLEMA! Nenhum trade gerado.")
    elif result2['pf'] < 1.0:
        print(f"    NAO LUCRATIVA em periodo estendido. PF = {result2['pf']:.2f}")
    elif result2['pf'] >= 1.3:
        print(f"    ROBUSTA! PF se manteve ({result2['pf']:.2f} vs {original2['pf']:.2f})")
        print(f"    Esta configuracao PODE ser viavel para trading real.")
    else:
        pf2_degradation = (result2['pf'] - original2['pf']) / original2['pf'] * 100
        print(f"    Degradacao de {abs(pf2_degradation):.0f}% no PF.")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
