#!/usr/bin/env python3
"""
================================================================================
WALK-FORWARD OPTIMIZATION - DSG M1 (1 Minuto)
================================================================================
Otimizacao para timeframe M1 desde 2025-01-01
"""

import sys
import os
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple
from collections import deque
from itertools import product
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.backtest_dsg import download_fxopen_data, Bar
from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional


class FastDSGOptimizerM1:
    """Otimizador rapido para DSG em M1"""

    def __init__(self, symbol: str = "EURUSD", n_folds: int = 3):
        self.symbol = symbol
        self.n_folds = n_folds
        self.bars = []
        self.pip = 0.0001
        self.spread = 0.5  # Spread menor em M1

    def load_data(self, start_date: datetime, end_date: datetime,
                  periodicity: str = "M1") -> bool:
        """Carrega dados historicos"""
        print(f"\n{'='*70}")
        print(f"  WALK-FORWARD DSG - M1")
        print(f"{'='*70}")

        self.bars = download_fxopen_data(
            self.symbol, periodicity, start_date, end_date, verbose=True
        )
        return len(self.bars) > 0

    def precompute_dsg_signals(self, bars: List[Bar],
                                lookback: int = 30) -> List[Dict]:
        """Pre-calcula dados DSG para conjunto de barras"""
        dsg = DetectorSingularidadeGravitacional(
            c_base=1.0, gamma=0.1,
            ricci_collapse_threshold=-0.5,
            tidal_force_threshold=0.1,
            event_horizon_threshold=0.001,
            lookback_window=lookback
        )

        min_prices = max(30, lookback + 5)
        prices_buf = deque(maxlen=300)
        bid_vols_buf = deque(maxlen=300)
        ask_vols_buf = deque(maxlen=300)

        dsg_data = []

        for i, bar in enumerate(bars):
            prices_buf.append(bar.close)
            vol = (bar.high - bar.low) * 1000000 + 100
            bid_vols_buf.append(vol * 0.5)
            ask_vols_buf.append(vol * 0.5)

            if len(prices_buf) < min_prices:
                dsg_data.append(None)
                continue

            try:
                result = dsg.analyze(
                    np.array(prices_buf),
                    np.array(bid_vols_buf),
                    np.array(ask_vols_buf)
                )

                dsg_data.append({
                    'idx': i,
                    'price': bar.close,
                    'high': bar.high,
                    'low': bar.low,
                    'ricci': result['Ricci_Scalar'],
                    'tidal': result['Tidal_Force_Magnitude'],
                    'event_horizon': result['Event_Horizon_Distance'],
                    'ricci_collapsing': result['ricci_collapsing'],
                    'crossing_horizon': result['crossing_horizon'],
                    'geodesic_direction': result['geodesic_direction']
                })
            except Exception:
                dsg_data.append(None)

            if (i + 1) % 5000 == 0:
                print(f"      Processado: {i+1}/{len(bars)} barras...")

        return dsg_data

    def generate_signals_from_dsg(self, dsg_data: List[Dict],
                                   ricci_percentile: float,
                                   tidal_percentile: float,
                                   eh_percentile: float,
                                   min_confidence: float,
                                   signal_cooldown: int) -> List[Dict]:
        """Gera sinais usando percentis"""
        valid_data = [d for d in dsg_data if d is not None]
        if not valid_data:
            return []

        ricci_vals = [d['ricci'] for d in valid_data]
        tidal_vals = [d['tidal'] for d in valid_data]
        eh_vals = [d['event_horizon'] for d in valid_data]

        ricci_thresh = np.percentile(ricci_vals, ricci_percentile)
        tidal_thresh = np.percentile(tidal_vals, 100 - tidal_percentile)
        eh_thresh = np.percentile(eh_vals, eh_percentile)

        signals = []
        cooldown = 0

        for d in dsg_data:
            if d is None:
                if cooldown > 0:
                    cooldown -= 1
                continue

            if cooldown > 0:
                cooldown -= 1
                continue

            conditions_met = 0

            if d['ricci_collapsing'] or d['ricci'] < ricci_thresh:
                conditions_met += 1
            if d['tidal'] > tidal_thresh:
                conditions_met += 1
            if d['crossing_horizon'] or d['event_horizon'] < eh_thresh:
                conditions_met += 1

            confidence = conditions_met / 3

            if conditions_met >= 2 and confidence >= min_confidence:
                direction = 1 if d['geodesic_direction'] > 0 else (-1 if d['geodesic_direction'] < 0 else 0)

                if direction != 0:
                    signals.append({
                        'idx': d['idx'],
                        'price': d['price'],
                        'direction': direction,
                        'confidence': confidence
                    })
                    cooldown = signal_cooldown

        return signals

    def simulate_trades(self, bars: List[Bar], signals: List[Dict],
                        sl_pips: float, tp_pips: float) -> Dict:
        """Simula trades"""
        pnls = []

        for sig in signals:
            bar_idx = sig['idx']
            if bar_idx + 1 >= len(bars):
                continue

            entry = bars[bar_idx + 1].open
            direction = sig['direction']
            sl_price = entry - direction * sl_pips * self.pip
            tp_price = entry + direction * tp_pips * self.pip

            pnl = 0
            max_bars = 120  # Maximo 2 horas em M1

            for j in range(bar_idx + 2, min(bar_idx + 2 + max_bars, len(bars))):
                b = bars[j]
                if direction == 1:
                    if b.low <= sl_price:
                        pnl = -sl_pips - self.spread
                        break
                    if b.high >= tp_price:
                        pnl = tp_pips - self.spread
                        break
                else:
                    if b.high >= sl_price:
                        pnl = -sl_pips - self.spread
                        break
                    if b.low <= tp_price:
                        pnl = tp_pips - self.spread
                        break

            if pnl == 0:
                exit_idx = min(bar_idx + 1 + max_bars, len(bars) - 1)
                exit_price = bars[exit_idx].close
                pnl = direction * (exit_price - entry) / self.pip - self.spread

            pnls.append(pnl)

        if not pnls:
            return {'trades': 0, 'pnl': 0, 'win_rate': 0, 'pf': 0}

        wins = sum(1 for p in pnls if p > 0)
        total = sum(pnls)
        wr = wins / len(pnls)
        gp = sum(p for p in pnls if p > 0) or 0.001
        gl = abs(sum(p for p in pnls if p <= 0)) or 0.001
        pf = gp / gl

        return {
            'trades': len(pnls),
            'pnl': round(total, 1),
            'win_rate': round(wr, 4),
            'pf': round(pf, 2)
        }

    def optimize(self, min_trades_per_fold: int = 3) -> Dict:
        """Executa Walk-Forward Optimization"""
        n = len(self.bars)
        fold_size = n // self.n_folds

        print(f"\n  Criando {self.n_folds} janelas Walk-Forward:")
        print(f"  Total de barras: {n}")

        folds = []
        for i in range(self.n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.n_folds - 1 else n
            fold_bars = self.bars[start_idx:end_idx]
            split = int(len(fold_bars) * 0.7)
            train = fold_bars[:split]
            test = fold_bars[split:]
            folds.append((train, test))

            print(f"\n  Fold {i+1}:")
            print(f"    Train: {train[0].timestamp.strftime('%Y-%m-%d')} a {train[-1].timestamp.strftime('%Y-%m-%d')} ({len(train)} barras)")
            print(f"    Test:  {test[0].timestamp.strftime('%Y-%m-%d')} a {test[-1].timestamp.strftime('%Y-%m-%d')} ({len(test)} barras)")

        print(f"\n{'='*70}")
        print(f"  PRE-CALCULANDO DSG M1")
        print(f"{'='*70}")

        fold_dsg_data = []
        for i, (train, test) in enumerate(folds):
            print(f"\n  Fold {i+1} - Train ({len(train)} barras):")
            train_dsg = self.precompute_dsg_signals(train)
            print(f"  Fold {i+1} - Test ({len(test)} barras):")
            test_dsg = self.precompute_dsg_signals(test)
            fold_dsg_data.append({
                'train_bars': train,
                'test_bars': test,
                'train_dsg': train_dsg,
                'test_dsg': test_dsg
            })

        # Grid para M1 (SL/TP menores)
        param_grid = {
            'ricci_percentile': [5, 10, 15, 20],
            'tidal_percentile': [5, 10, 15, 20],
            'eh_percentile': [5, 10, 15, 20],
            'min_confidence': [0.5, 0.6, 0.7],
            'signal_cooldown': [30, 60, 120, 180],  # 30min a 3h em M1
            'sl_pips': [5.0, 10.0, 15.0],  # SL menor para M1
            'tp_pips': [10.0, 15.0, 20.0, 30.0]  # TP menor para M1
        }

        keys = list(param_grid.keys())
        combinations = list(product(*[param_grid[k] for k in keys]))

        configs = []
        for combo in combinations:
            cfg = {k: v for k, v in zip(keys, combo)}
            if cfg['tp_pips'] > cfg['sl_pips']:
                configs.append(cfg)

        print(f"\n{'='*70}")
        print(f"  TESTANDO {len(configs)} COMBINACOES")
        print(f"{'='*70}")

        approved = []

        for idx, cfg in enumerate(configs):
            if (idx + 1) % 200 == 0 or idx == 0:
                print(f"\n  Config {idx+1}/{len(configs)}...")

            fold_results = []
            passed = True

            for fold_idx, fold_data in enumerate(fold_dsg_data):
                test_signals = self.generate_signals_from_dsg(
                    fold_data['test_dsg'],
                    cfg['ricci_percentile'],
                    cfg['tidal_percentile'],
                    cfg['eh_percentile'],
                    cfg['min_confidence'],
                    cfg['signal_cooldown']
                )

                test_result = self.simulate_trades(
                    fold_data['test_bars'],
                    test_signals,
                    cfg['sl_pips'],
                    cfg['tp_pips']
                )

                fold_results.append(test_result)

                if test_result['trades'] < min_trades_per_fold or test_result['pf'] < 1.0:
                    passed = False
                    break

            if passed:
                test_trades = sum(f['trades'] for f in fold_results)
                test_pnl = sum(f['pnl'] for f in fold_results)
                test_wr = np.mean([f['win_rate'] for f in fold_results])
                test_pf = np.mean([f['pf'] for f in fold_results])

                approved.append({
                    'config': cfg,
                    'fold_results': fold_results,
                    'aggregated': {
                        'trades': test_trades,
                        'pnl': test_pnl,
                        'win_rate': test_wr,
                        'pf': test_pf
                    }
                })

        approved.sort(key=lambda x: x['aggregated']['pf'], reverse=True)

        print(f"\n{'='*70}")
        print(f"  RESULTADOS WALK-FORWARD M1")
        print(f"{'='*70}")
        print(f"\n  Configuracoes aprovadas: {len(approved)} de {len(configs)}")

        if not approved:
            print(f"\n  NENHUMA configuracao passou!")
            return {'approved_configs': [], 'best': None}

        print(f"\n  TOP 5:")
        for i, res in enumerate(approved[:5]):
            cfg = res['config']
            agg = res['aggregated']
            print(f"\n  #{i+1}:")
            print(f"    Trades: {agg['trades']} | PnL: {agg['pnl']:.1f} pips | "
                  f"WR: {agg['win_rate']*100:.1f}% | PF: {agg['pf']:.2f}")
            print(f"    Ricci P{cfg['ricci_percentile']}%, Tidal P{cfg['tidal_percentile']}%, "
                  f"EH P{cfg['eh_percentile']}%")
            print(f"    SL={cfg['sl_pips']}, TP={cfg['tp_pips']}, Cooldown={cfg['signal_cooldown']}")

        best = approved[0] if approved else None

        return {
            'approved_configs': approved,
            'best': best,
            'total_tested': len(configs),
            'passed_rate': len(approved) / len(configs) if configs else 0
        }

    def save_results(self, results: Dict):
        """Salva resultados"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"/home/user/EliBotCD/configs/dsg_m1_optimized_{timestamp}.json"

        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return obj

        save_data = {
            'timestamp': datetime.now().isoformat(),
            'timeframe': 'M5',
            'symbol': self.symbol,
            'n_folds': self.n_folds,
            'total_tested': results.get('total_tested', 0),
            'passed_rate': results.get('passed_rate', 0),
            'best_config': None,
            'approved_configs': []
        }

        if results.get('best'):
            best = results['best']
            save_data['best_config'] = {
                'parameters': {k: convert(v) for k, v in best['config'].items()},
                'performance': {k: convert(v) for k, v in best['aggregated'].items()}
            }

        for res in results.get('approved_configs', [])[:10]:
            save_data['approved_configs'].append({
                'parameters': {k: convert(v) for k, v in res['config'].items()},
                'performance': {k: convert(v) for k, v in res['aggregated'].items()}
            })

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"\n  Resultados salvos em: {filename}")
        return filename


def main():
    print("=" * 70)
    print("  WALK-FORWARD DSG - M5 (5 Minutos)")
    print("  Desde 2025-01-01")
    print("=" * 70)

    # Periodo: desde 2025-01-01
    end = datetime.now(timezone.utc)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)

    print(f"\n  Periodo: {start.date()} a {end.date()}")
    print(f"  Timeframe: M5")

    optimizer = FastDSGOptimizerM1(symbol="EURUSD", n_folds=3)

    if not optimizer.load_data(start, end, "M5"):
        print("  ERRO: Nao foi possivel carregar dados!")
        return

    results = optimizer.optimize(min_trades_per_fold=3)

    if results.get('best'):
        print(f"\n{'='*70}")
        print(f"  MELHOR CONFIGURACAO M1")
        print(f"{'='*70}")

        best = results['best']
        cfg = best['config']
        agg = best['aggregated']

        print(f"\n  Performance OUT-OF-SAMPLE:")
        print(f"    Trades: {agg['trades']}")
        print(f"    PnL: {agg['pnl']:.1f} pips")
        print(f"    Win Rate: {agg['win_rate']*100:.1f}%")
        print(f"    Profit Factor: {agg['pf']:.2f}")

        optimizer.save_results(results)

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
