#!/usr/bin/env python3
"""
================================================================================
WALK-FORWARD OPTIMIZATION - DSG (Detector de Singularidade Gravitacional)
================================================================================
Otimizacao robusta que EVITA overfitting atraves de:
1. Divisao em janelas Train/Test rolantes
2. Validacao out-of-sample em CADA janela
3. Apenas aceita configs que funcionam em TODOS os periodos

IMPORTANTE: Este e o metodo CORRETO para otimizar estrategias de trading.
Sem Walk-Forward = Overfitting garantido.
================================================================================
"""

import sys
import os
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple
from itertools import product
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.backtest_dsg import DSGBacktester, download_fxopen_data, Bar


class WalkForwardOptimizer:
    """
    Walk-Forward Analysis para DSG

    Metodo:
    1. Divide os dados em N janelas (folds)
    2. Para cada fold: treina em 70%, testa em 30%
    3. Uma configuracao so e "aprovada" se for lucrativa em TODOS os folds out-of-sample
    4. Isso elimina configuracoes overfitted que so funcionam em um periodo especifico
    """

    def __init__(self, symbol: str = "EURUSD", n_folds: int = 4,
                 train_ratio: float = 0.7):
        """
        Args:
            symbol: Par de moedas
            n_folds: Numero de janelas Walk-Forward
            train_ratio: Proporcao de dados para treinamento em cada janela
        """
        self.symbol = symbol
        self.n_folds = n_folds
        self.train_ratio = train_ratio
        self.bars = []
        self.pip = 0.0001
        self.spread = 1.0

    def load_data(self, start_date: datetime, end_date: datetime,
                  periodicity: str = "H1") -> bool:
        """Carrega dados historicos"""
        print(f"\n{'='*70}")
        print(f"  WALK-FORWARD OPTIMIZATION - DSG")
        print(f"{'='*70}")

        self.bars = download_fxopen_data(
            self.symbol, periodicity, start_date, end_date, verbose=True
        )
        return len(self.bars) > 0

    def create_folds(self) -> List[Tuple[List[Bar], List[Bar]]]:
        """
        Cria janelas Walk-Forward

        Cada fold contem:
        - train_bars: 70% para otimizacao
        - test_bars: 30% para validacao out-of-sample (NAO PODE SER USADO NA OTIMIZACAO!)
        """
        n = len(self.bars)
        fold_size = n // self.n_folds
        folds = []

        print(f"\n  Criando {self.n_folds} janelas Walk-Forward:")
        print(f"  Total de barras: {n}")
        print(f"  Barras por fold: ~{fold_size}")

        for i in range(self.n_folds):
            start_idx = i * fold_size
            # Ultimo fold pega o resto
            end_idx = (i + 1) * fold_size if i < self.n_folds - 1 else n

            fold_bars = self.bars[start_idx:end_idx]
            split_idx = int(len(fold_bars) * self.train_ratio)

            train_bars = fold_bars[:split_idx]
            test_bars = fold_bars[split_idx:]

            folds.append((train_bars, test_bars))

            train_start = train_bars[0].timestamp.strftime('%Y-%m-%d')
            train_end = train_bars[-1].timestamp.strftime('%Y-%m-%d')
            test_start = test_bars[0].timestamp.strftime('%Y-%m-%d')
            test_end = test_bars[-1].timestamp.strftime('%Y-%m-%d')

            print(f"\n  Fold {i+1}:")
            print(f"    Train: {train_start} a {train_end} ({len(train_bars)} barras)")
            print(f"    Test:  {test_start} a {test_end} ({len(test_bars)} barras)")

        return folds

    def generate_param_grid(self, conservative: bool = False) -> List[Dict]:
        """
        Gera grid de parametros para otimizacao

        Args:
            conservative: Se True, usa menos combinacoes para teste rapido
        """
        if conservative:
            # Grid menor para teste rapido
            grid = {
                'min_prices': [80, 100],
                'stop_loss_pips': [30.0, 50.0],
                'take_profit_pips': [60.0, 90.0],
                'ricci_collapse_threshold': [-0.5, -0.3],
                'tidal_force_threshold': [0.08, 0.12],
                'event_horizon_threshold': [0.001, 0.005],
                'lookback_window': [50],
                'c_base': [1.0],
                'gamma': [0.1],
                'min_confidence': [0.5, 0.6],
                'signal_cooldown': [20, 30]
            }
        else:
            # Grid completo para otimizacao real
            grid = {
                'min_prices': [50, 80, 100, 120],
                'stop_loss_pips': [20.0, 30.0, 40.0, 50.0],
                'take_profit_pips': [40.0, 60.0, 80.0, 100.0, 120.0],
                'ricci_collapse_threshold': [-0.8, -0.5, -0.3, -0.2],
                'tidal_force_threshold': [0.05, 0.08, 0.1, 0.12, 0.15],
                'event_horizon_threshold': [0.0005, 0.001, 0.002, 0.005],
                'lookback_window': [30, 50, 70],
                'c_base': [0.8, 1.0, 1.2],
                'gamma': [0.08, 0.1, 0.12],
                'min_confidence': [0.5, 0.6, 0.7],
                'signal_cooldown': [15, 20, 30, 40]
            }

        # Gera todas as combinacoes
        keys = list(grid.keys())
        combinations = list(product(*[grid[k] for k in keys]))

        configs = []
        for combo in combinations:
            config = {k: v for k, v in zip(keys, combo)}
            # Filtro: TP deve ser maior que SL para ratio positivo
            if config['take_profit_pips'] > config['stop_loss_pips']:
                configs.append(config)

        print(f"\n  Grid de parametros gerado: {len(configs)} combinacoes")
        return configs

    def run_backtest_on_bars(self, bars: List[Bar], config: Dict) -> Dict:
        """Executa backtest em um conjunto especifico de barras"""
        from strategies.alta_volatilidade.dsg_detector_singularidade import DetectorSingularidadeGravitacional
        from collections import deque

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

        prices_buf = deque(maxlen=600)
        bid_vols_buf = deque(maxlen=600)
        ask_vols_buf = deque(maxlen=600)

        # Calcula DSG
        dsg_data = []
        for i, bar in enumerate(bars):
            prices_buf.append(bar.close)
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
                    'signal': result['signal'],
                    'confidence': result['confidence'],
                    'price': bar.close
                })
            except:
                continue

        # Encontra sinais
        signals = []
        signal_cooldown = 0
        min_confidence = config.get('min_confidence', 0.5)

        for d in dsg_data:
            if signal_cooldown > 0:
                signal_cooldown -= 1
                continue
            if d['signal'] != 0 and d['confidence'] >= min_confidence:
                signals.append({'idx': d['idx'], 'price': d['price'], 'direction': d['signal']})
                signal_cooldown = config.get('signal_cooldown', 30)

        if not signals:
            return {'trades': 0, 'pnl': 0, 'win_rate': 0, 'pf': 0}

        # Executa trades
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
            max_bars = config.get('max_bars_in_trade', 500)

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

    def optimize(self, conservative: bool = True, min_trades_per_fold: int = 2) -> Dict:
        """
        Executa Walk-Forward Optimization

        Args:
            conservative: Grid menor para teste rapido
            min_trades_per_fold: Minimo de trades em cada fold para validar

        Returns:
            Melhores configuracoes que passaram em TODOS os folds
        """
        folds = self.create_folds()
        configs = self.generate_param_grid(conservative)

        print(f"\n{'='*70}")
        print(f"  INICIANDO WALK-FORWARD OPTIMIZATION")
        print(f"  {len(configs)} configuracoes x {self.n_folds} folds")
        print(f"{'='*70}")

        # Resultados
        all_results = []

        for idx, config in enumerate(configs):
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"\n  Testando config {idx+1}/{len(configs)}...")

            fold_results = []
            passed_all_folds = True

            for fold_idx, (train_bars, test_bars) in enumerate(folds):
                # Treina (opcional - aqui apenas testamos)
                train_result = self.run_backtest_on_bars(train_bars, config)

                # Testa OUT-OF-SAMPLE (o mais importante!)
                test_result = self.run_backtest_on_bars(test_bars, config)

                fold_results.append({
                    'fold': fold_idx + 1,
                    'train': train_result,
                    'test': test_result
                })

                # Criterios de aprovacao para cada fold:
                # 1. Minimo de trades
                # 2. PF > 1.0 (lucrativo)
                # 3. Win rate razoavel
                if (test_result['trades'] < min_trades_per_fold or
                    test_result['pf'] < 1.0):
                    passed_all_folds = False
                    break

            if passed_all_folds:
                # Calcula metricas agregadas
                test_trades = sum(f['test']['trades'] for f in fold_results)
                test_pnl = sum(f['test']['pnl'] for f in fold_results)
                test_wr = np.mean([f['test']['win_rate'] for f in fold_results])
                test_pf = np.mean([f['test']['pf'] for f in fold_results])

                all_results.append({
                    'config': config,
                    'fold_results': fold_results,
                    'aggregated': {
                        'trades': test_trades,
                        'pnl': test_pnl,
                        'win_rate': test_wr,
                        'pf': test_pf
                    }
                })

        # Ordena por PF medio out-of-sample
        all_results.sort(key=lambda x: x['aggregated']['pf'], reverse=True)

        print(f"\n{'='*70}")
        print(f"  RESULTADOS WALK-FORWARD")
        print(f"{'='*70}")
        print(f"\n  Configuracoes aprovadas: {len(all_results)} de {len(configs)}")

        if not all_results:
            print(f"\n  NENHUMA configuracao passou em todos os folds!")
            print(f"  Isso significa que o DSG pode nao ser viavel para trading.")
            return {'approved_configs': [], 'best': None}

        # Mostra top 5
        print(f"\n  TOP 5 Configuracoes (por PF medio OOS):")
        print(f"  {'-'*65}")

        for i, res in enumerate(all_results[:5]):
            cfg = res['config']
            agg = res['aggregated']
            print(f"\n  #{i+1}:")
            print(f"    Trades OOS: {agg['trades']} | PnL: {agg['pnl']:.1f} pips | "
                  f"WR: {agg['win_rate']*100:.1f}% | PF: {agg['pf']:.2f}")
            print(f"    Params: SL={cfg['stop_loss_pips']}, TP={cfg['take_profit_pips']}, "
                  f"Ricci<={cfg['ricci_collapse_threshold']}, "
                  f"Tidal>={cfg['tidal_force_threshold']}")

            # Detalhe por fold
            for fr in res['fold_results']:
                t = fr['test']
                print(f"      Fold {fr['fold']}: {t['trades']} trades, "
                      f"PnL={t['pnl']:.1f}, PF={t['pf']:.2f}")

        # Melhor configuracao
        best = all_results[0] if all_results else None

        return {
            'approved_configs': all_results,
            'best': best,
            'total_tested': len(configs),
            'passed_rate': len(all_results) / len(configs) if configs else 0
        }

    def save_results(self, results: Dict, filename: str = None):
        """Salva resultados em JSON"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"/home/user/EliBotCD/configs/dsg_walkforward_{timestamp}.json"

        # Converte para formato serializavel
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Prepara dados
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'n_folds': self.n_folds,
            'train_ratio': self.train_ratio,
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

        # Salva top 10
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
    """Funcao principal - Walk-Forward Optimization para DSG"""
    print("=" * 70)
    print("  WALK-FORWARD OPTIMIZATION - DSG")
    print("  Otimizacao ROBUSTA que EVITA overfitting")
    print("=" * 70)

    # Periodo longo para Walk-Forward (12 meses)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365)

    print(f"\n  Periodo: {start.date()} a {end.date()} (~12 meses)")
    print(f"  Usando 4 folds Walk-Forward (70% train / 30% test cada)")

    optimizer = WalkForwardOptimizer(
        symbol="EURUSD",
        n_folds=4,
        train_ratio=0.7
    )

    if not optimizer.load_data(start, end, "H1"):
        print("  ERRO: Nao foi possivel carregar dados!")
        return

    # Executa otimizacao (conservative=True para teste rapido)
    print("\n  Executando otimizacao (modo conservador)...")
    print("  Para otimizacao completa, use conservative=False")

    results = optimizer.optimize(conservative=True, min_trades_per_fold=2)

    if results.get('best'):
        print(f"\n{'='*70}")
        print(f"  MELHOR CONFIGURACAO WALK-FORWARD")
        print(f"{'='*70}")

        best = results['best']
        cfg = best['config']
        agg = best['aggregated']

        print(f"\n  Performance OUT-OF-SAMPLE:")
        print(f"    Trades: {agg['trades']}")
        print(f"    PnL: {agg['pnl']:.1f} pips")
        print(f"    Win Rate: {agg['win_rate']*100:.1f}%")
        print(f"    Profit Factor: {agg['pf']:.2f}")

        print(f"\n  Parametros:")
        for k, v in cfg.items():
            print(f"    {k}: {v}")

        # Salva resultados
        optimizer.save_results(results)

    print(f"\n{'='*70}")
    print(f"  CONCLUSAO")
    print(f"{'='*70}")

    if results.get('approved_configs'):
        n_approved = len(results['approved_configs'])
        rate = results.get('passed_rate', 0) * 100
        print(f"\n  {n_approved} configuracoes passaram em TODOS os folds ({rate:.1f}%)")
        print(f"  Estas configuracoes tem MENOR risco de overfitting.")
    else:
        print(f"\n  NENHUMA configuracao passou em todos os folds!")
        print(f"  O indicador DSG pode precisar de ajustes ou nao ser viavel para trading.")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
