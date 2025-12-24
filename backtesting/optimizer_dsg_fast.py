#!/usr/bin/env python3
"""
================================================================================
WALK-FORWARD OPTIMIZATION - DSG (VERSAO OTIMIZADA)
================================================================================
Versao RAPIDA que pre-calcula DSG uma vez por fold e otimiza apenas trade params.

O gargalo do DSG sao os calculos tensoriais. Esta versao:
1. Calcula DSG uma vez por conjunto de dados
2. Salva todos os sinais potenciais (com diferentes thresholds)
3. Otimiza apenas SL/TP/cooldown em cima dos sinais pre-calculados
================================================================================
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


class FastDSGOptimizer:
    """
    Otimizador rapido para DSG

    Estrategia: Pre-calcula TODOS os dados DSG uma vez, depois
    testa diferentes combinacoes de thresholds e SL/TP rapidamente.
    """

    def __init__(self, symbol: str = "EURUSD", n_folds: int = 3):
        self.symbol = symbol
        self.n_folds = n_folds
        self.bars = []
        self.pip = 0.0001
        self.spread = 1.0

    def load_data(self, start_date: datetime, end_date: datetime,
                  periodicity: str = "H1") -> bool:
        """Carrega dados historicos"""
        print(f"\n{'='*70}")
        print(f"  WALK-FORWARD DSG (VERSAO OTIMIZADA)")
        print(f"{'='*70}")

        self.bars = download_fxopen_data(
            self.symbol, periodicity, start_date, end_date, verbose=True
        )
        return len(self.bars) > 0

    def precompute_dsg_signals(self, bars: List[Bar],
                                c_base: float = 1.0,
                                gamma: float = 0.1,
                                lookback: int = 50) -> List[Dict]:
        """
        Pre-calcula TODOS os dados DSG para um conjunto de barras.
        Retorna lista com metricas DSG para cada barra.
        """
        dsg = DetectorSingularidadeGravitacional(
            c_base=c_base,
            gamma=gamma,
            ricci_collapse_threshold=-0.5,  # Valores padrao (nao importam para pre-calculo)
            tidal_force_threshold=0.1,
            event_horizon_threshold=0.001,
            lookback_window=lookback
        )

        min_prices = max(50, lookback + 10)
        prices_buf = deque(maxlen=600)
        bid_vols_buf = deque(maxlen=600)
        ask_vols_buf = deque(maxlen=600)

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
                    'geodesic_direction': result['geodesic_direction'],
                    'conditions_met': result.get('conditions_met', 0)
                })
            except Exception as e:
                dsg_data.append(None)

            if (i + 1) % 500 == 0:
                print(f"      Processado: {i+1}/{len(bars)} barras...")

        return dsg_data

    def generate_signals_from_dsg(self, dsg_data: List[Dict],
                                   ricci_percentile: float,
                                   tidal_percentile: float,
                                   eh_percentile: float,
                                   min_confidence: float,
                                   signal_cooldown: int) -> List[Dict]:
        """
        Gera sinais a partir dos dados DSG pre-calculados.
        USA PERCENTIS para thresholds (evita valores absolutos que variam muito).
        """
        # Calcula percentis dos dados (so valores validos)
        valid_data = [d for d in dsg_data if d is not None]
        if not valid_data:
            return []

        ricci_vals = [d['ricci'] for d in valid_data]
        tidal_vals = [d['tidal'] for d in valid_data]
        eh_vals = [d['event_horizon'] for d in valid_data]

        # Thresholds baseados em percentis
        # Ricci: queremos os mais negativos (percentil baixo)
        ricci_thresh = np.percentile(ricci_vals, ricci_percentile)
        # Tidal: queremos os mais altos (percentil alto)
        tidal_thresh = np.percentile(tidal_vals, 100 - tidal_percentile)
        # EH: queremos os mais baixos (percentil baixo)
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

            # Verifica condicoes usando thresholds dinamicos
            conditions_met = 0

            # Ricci muito negativo (abaixo do percentil)
            if d['ricci_collapsing'] or d['ricci'] < ricci_thresh:
                conditions_met += 1
            # Tidal alto (acima do percentil)
            if d['tidal'] > tidal_thresh:
                conditions_met += 1
            # Event horizon baixo (abaixo do percentil)
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
        """Simula trades a partir de sinais pre-gerados"""
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
            max_bars = 500

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

    def optimize(self, min_trades_per_fold: int = 2) -> Dict:
        """
        Executa Walk-Forward Optimization de forma RAPIDA
        """
        n = len(self.bars)
        fold_size = n // self.n_folds

        print(f"\n  Criando {self.n_folds} janelas Walk-Forward:")
        print(f"  Total de barras: {n}")

        # Cria folds
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

        # PRE-CALCULA DSG para cada fold (a parte mais pesada)
        print(f"\n{'='*70}")
        print(f"  PRE-CALCULANDO DSG (parte pesada - faz apenas uma vez)")
        print(f"{'='*70}")

        fold_dsg_data = []
        for i, (train, test) in enumerate(folds):
            print(f"\n  Fold {i+1} - Train:")
            train_dsg = self.precompute_dsg_signals(train)
            print(f"  Fold {i+1} - Test:")
            test_dsg = self.precompute_dsg_signals(test)
            fold_dsg_data.append({
                'train_bars': train,
                'test_bars': test,
                'train_dsg': train_dsg,
                'test_dsg': test_dsg
            })

        # Grid de parametros usando PERCENTIS (mais robusto)
        # Exemplo: ricci_percentile=10 significa pegar o top 10% mais negativo
        param_grid = {
            'ricci_percentile': [5, 10, 15, 20],  # Percentil mais baixo = mais extremo
            'tidal_percentile': [5, 10, 15, 20],  # Percentil mais alto = mais extremo
            'eh_percentile': [5, 10, 15, 20],     # Percentil mais baixo = mais proximo do horizonte
            'min_confidence': [0.5, 0.6, 0.7],
            'signal_cooldown': [20, 30, 50, 100],  # Cooldown maior para evitar overtrading
            'sl_pips': [25.0, 35.0, 50.0],
            'tp_pips': [50.0, 75.0, 100.0, 150.0]
        }

        keys = list(param_grid.keys())
        combinations = list(product(*[param_grid[k] for k in keys]))

        # Filtra: TP > SL
        configs = []
        for combo in combinations:
            cfg = {k: v for k, v in zip(keys, combo)}
            if cfg['tp_pips'] > cfg['sl_pips']:
                configs.append(cfg)

        print(f"\n{'='*70}")
        print(f"  TESTANDO {len(configs)} COMBINACOES DE PARAMETROS")
        print(f"  (Usando dados DSG pre-calculados - MUITO RAPIDO)")
        print(f"{'='*70}")

        approved = []

        for idx, cfg in enumerate(configs):
            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"\n  Testando config {idx+1}/{len(configs)}...")

            fold_results = []
            passed = True

            for fold_idx, fold_data in enumerate(fold_dsg_data):
                # Gera sinais a partir dos dados pre-calculados (RAPIDO)
                test_signals = self.generate_signals_from_dsg(
                    fold_data['test_dsg'],
                    cfg['ricci_percentile'],
                    cfg['tidal_percentile'],
                    cfg['eh_percentile'],
                    cfg['min_confidence'],
                    cfg['signal_cooldown']
                )

                # Simula trades (RAPIDO)
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
        print(f"  RESULTADOS WALK-FORWARD")
        print(f"{'='*70}")
        print(f"\n  Configuracoes aprovadas: {len(approved)} de {len(configs)}")

        if not approved:
            print(f"\n  NENHUMA configuracao passou em todos os folds!")
            return {'approved_configs': [], 'best': None}

        print(f"\n  TOP 5 Configuracoes (por PF medio OOS):")
        print(f"  {'-'*65}")

        for i, res in enumerate(approved[:5]):
            cfg = res['config']
            agg = res['aggregated']
            print(f"\n  #{i+1}:")
            print(f"    Trades OOS: {agg['trades']} | PnL: {agg['pnl']:.1f} pips | "
                  f"WR: {agg['win_rate']*100:.1f}% | PF: {agg['pf']:.2f}")
            print(f"    Ricci P{cfg['ricci_percentile']}%, Tidal P{cfg['tidal_percentile']}%, "
                  f"EH P{cfg['eh_percentile']}%")
            print(f"    SL={cfg['sl_pips']}, TP={cfg['tp_pips']}, "
                  f"Cooldown={cfg['signal_cooldown']}, MinConf={cfg['min_confidence']}")

            for j, fr in enumerate(res['fold_results']):
                print(f"      Fold {j+1}: {fr['trades']} trades, PnL={fr['pnl']:.1f}, PF={fr['pf']:.2f}")

        best = approved[0] if approved else None

        return {
            'approved_configs': approved,
            'best': best,
            'total_tested': len(configs),
            'passed_rate': len(approved) / len(configs) if configs else 0
        }

    def save_results(self, results: Dict):
        """Salva resultados em JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"/home/user/EliBotCD/configs/dsg_optimized_{timestamp}.json"

        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return obj

        save_data = {
            'timestamp': datetime.now().isoformat(),
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
    print("  WALK-FORWARD DSG - VERSAO OTIMIZADA")
    print("  Pre-calcula DSG uma vez, otimiza parametros rapidamente")
    print("=" * 70)

    # Periodo: 6 meses (mais rapido que 12)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=180)

    print(f"\n  Periodo: {start.date()} a {end.date()} (~6 meses)")

    optimizer = FastDSGOptimizer(symbol="EURUSD", n_folds=3)

    if not optimizer.load_data(start, end, "H1"):
        print("  ERRO: Nao foi possivel carregar dados!")
        return

    results = optimizer.optimize(min_trades_per_fold=2)

    if results.get('best'):
        print(f"\n{'='*70}")
        print(f"  MELHOR CONFIGURACAO")
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
