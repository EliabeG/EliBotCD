#!/usr/bin/env python3
"""
================================================================================
BACKTEST PRM - Protocolo Riemann-Mandelbrot
Backtest com Dados Historicos REAIS
================================================================================

Este script executa backtest do indicador PRM (Protocolo Riemann-Mandelbrot)
usando dados historicos REAIS do mercado Forex.

IMPORTANTE: Nenhuma simulacao - apenas dados reais do mercado.
Isso envolve dinheiro real, entao a precisao e crucial.

O indicador PRM detecta "Singularidades de Preco" usando:
1. Hidden Markov Model (HMM) - Estados de volatilidade
2. Lyapunov Exponent - Chaos/Estabilidade
3. Curvatura Geometrica - Aceleracao de tendencia

Uso:
    python -m backtesting.backtest_prm [--days DAYS] [--symbol SYMBOL]
"""

import sys
import os
import argparse
from datetime import datetime, timezone, timedelta

# Adiciona diretorio raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.backtest_engine import BacktestEngine, run_backtest, BacktestResult
from strategies.alta_volatilidade import PRMStrategy


def create_prm_strategy(
    min_prices: int = 50,
    stop_loss_pips: float = 20.0,
    take_profit_pips: float = 40.0,
    hmm_threshold: float = 0.85,
    lyapunov_threshold: float = 0.5,
    curvature_threshold: float = 0.1
) -> PRMStrategy:
    """
    Cria instancia da estrategia PRM com parametros customizados

    Args:
        min_prices: Minimo de precos para analise
        stop_loss_pips: Stop loss em pips
        take_profit_pips: Take profit em pips
        hmm_threshold: Threshold para HMM
        lyapunov_threshold: Threshold para Lyapunov
        curvature_threshold: Threshold para curvatura

    Returns:
        Instancia de PRMStrategy
    """
    return PRMStrategy(
        min_prices=min_prices,
        stop_loss_pips=stop_loss_pips,
        take_profit_pips=take_profit_pips,
        hmm_threshold=hmm_threshold,
        lyapunov_threshold=lyapunov_threshold,
        curvature_threshold=curvature_threshold
    )


def run_prm_backtest(
    symbol: str = "EURUSD",
    days: int = 30,
    periodicity: str = "H1",
    initial_capital: float = 10000.0,
    min_prices: int = 50,
    stop_loss_pips: float = 20.0,
    take_profit_pips: float = 40.0,
    verbose: bool = True
) -> BacktestResult:
    """
    Executa backtest do PRM com dados REAIS

    IMPORTANTE: Usa dados REAIS do mercado Forex.

    Args:
        symbol: Par de moedas
        days: Numero de dias de historico
        periodicity: Periodicidade (M1, H1, D1)
        initial_capital: Capital inicial
        min_prices: Minimo de precos para analise
        stop_loss_pips: Stop loss em pips
        take_profit_pips: Take profit em pips
        verbose: Mostrar detalhes

    Returns:
        BacktestResult com metricas
    """
    print("\n" + "=" * 70)
    print("  BACKTEST PRM - PROTOCOLO RIEMANN-MANDELBROT")
    print("  Dados Historicos REAIS do Mercado Forex")
    print("=" * 70)

    # Cria estrategia
    strategy = create_prm_strategy(
        min_prices=min_prices,
        stop_loss_pips=stop_loss_pips,
        take_profit_pips=take_profit_pips
    )

    # Configura engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        position_size=0.01,  # Mini lote
        pip_value=0.0001,
        spread_pips=1.0,
        slippage_pips=0.5
    )

    # Define periodo
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    # Executa backtest
    result = engine.run(
        strategy=strategy,
        symbol=symbol,
        periodicity=periodicity,
        start_time=start_time,
        end_time=end_time,
        verbose=verbose
    )

    return result


def run_optimization(
    symbol: str = "EURUSD",
    days: int = 30,
    periodicity: str = "H1"
) -> dict:
    """
    Executa otimizacao de parametros do PRM

    Testa diferentes combinacoes de parametros para encontrar
    a melhor configuracao.

    Args:
        symbol: Par de moedas
        days: Dias de historico
        periodicity: Periodicidade

    Returns:
        Dict com melhores parametros e resultados
    """
    print("\n" + "=" * 70)
    print("  OTIMIZACAO PRM - BUSCA DE MELHORES PARAMETROS")
    print("=" * 70)

    # Parametros a testar
    stop_loss_values = [15.0, 20.0, 25.0, 30.0]
    take_profit_values = [30.0, 40.0, 50.0, 60.0]
    min_prices_values = [30, 50, 75, 100]

    best_result = None
    best_params = None
    best_profit_factor = 0

    results = []

    total_tests = len(stop_loss_values) * len(take_profit_values) * len(min_prices_values)
    current_test = 0

    for sl in stop_loss_values:
        for tp in take_profit_values:
            for mp in min_prices_values:
                current_test += 1
                print(f"\n  Teste {current_test}/{total_tests}: SL={sl}, TP={tp}, MinPrices={mp}")

                result = run_prm_backtest(
                    symbol=symbol,
                    days=days,
                    periodicity=periodicity,
                    stop_loss_pips=sl,
                    take_profit_pips=tp,
                    min_prices=mp,
                    verbose=False
                )

                results.append({
                    'stop_loss': sl,
                    'take_profit': tp,
                    'min_prices': mp,
                    'profit_factor': result.profit_factor,
                    'total_pnl': result.total_pnl,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades,
                    'max_drawdown': result.max_drawdown
                })

                # Verifica se e melhor
                if result.profit_factor > best_profit_factor and result.total_trades >= 5:
                    best_profit_factor = result.profit_factor
                    best_result = result
                    best_params = {
                        'stop_loss': sl,
                        'take_profit': tp,
                        'min_prices': mp
                    }

                print(f"    PF={result.profit_factor:.2f} | PnL=${result.total_pnl:.2f} | "
                      f"WR={result.win_rate:.1%} | Trades={result.total_trades}")

    print("\n" + "=" * 70)
    print("  RESULTADOS DA OTIMIZACAO")
    print("=" * 70)

    if best_params:
        print(f"\n  MELHORES PARAMETROS:")
        print(f"    Stop Loss: {best_params['stop_loss']} pips")
        print(f"    Take Profit: {best_params['take_profit']} pips")
        print(f"    Min Prices: {best_params['min_prices']}")
        print(f"\n  PERFORMANCE:")
        print(f"    Profit Factor: {best_result.profit_factor:.2f}")
        print(f"    PnL Total: ${best_result.total_pnl:.2f}")
        print(f"    Win Rate: {best_result.win_rate:.1%}")
        print(f"    Total Trades: {best_result.total_trades}")
        print(f"    Max Drawdown: {best_result.max_drawdown:.1%}")
    else:
        print("  Nenhuma combinacao rentavel encontrada.")

    return {
        'best_params': best_params,
        'best_result': best_result,
        'all_results': results
    }


def generate_report(result: BacktestResult, filename: str = None):
    """
    Gera relatorio detalhado do backtest em formato texto

    Args:
        result: Resultado do backtest
        filename: Nome do arquivo (opcional)
    """
    report = []
    report.append("=" * 70)
    report.append("  RELATORIO DE BACKTEST - PRM")
    report.append("  Protocolo Riemann-Mandelbrot")
    report.append("=" * 70)
    report.append(f"\nData do relatorio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n{'='*70}")
    report.append("  CONFIGURACAO")
    report.append(f"{'='*70}")
    report.append(f"  Estrategia: {result.strategy_name}")
    report.append(f"  Simbolo: {result.symbol}")
    report.append(f"  Periodicidade: {result.periodicity}")
    report.append(f"  Periodo: {result.start_time.strftime('%Y-%m-%d')} a {result.end_time.strftime('%Y-%m-%d')}")
    report.append(f"  Total de barras: {result.total_bars}")

    report.append(f"\n{'='*70}")
    report.append("  ESTATISTICAS DE TRADING")
    report.append(f"{'='*70}")
    report.append(f"  Total de sinais: {result.total_signals}")
    report.append(f"  Total de trades: {result.total_trades}")
    report.append(f"  Trades vencedores: {result.winning_trades}")
    report.append(f"  Trades perdedores: {result.losing_trades}")
    report.append(f"  Win Rate: {result.win_rate:.2%}")

    report.append(f"\n{'='*70}")
    report.append("  PERFORMANCE FINANCEIRA")
    report.append(f"{'='*70}")
    report.append(f"  PnL Total: ${result.total_pnl:,.2f}")
    report.append(f"  PnL em Pips: {result.total_pnl_pips:.1f}")
    report.append(f"  Profit Factor: {result.profit_factor:.2f}")
    report.append(f"  Media por trade: ${result.avg_trade:.2f}")
    report.append(f"  Media ganho: ${result.avg_win:.2f}")
    report.append(f"  Media perda: ${result.avg_loss:.2f}")
    report.append(f"  Maior ganho: ${result.max_win:.2f}")
    report.append(f"  Maior perda: ${result.max_loss:.2f}")

    report.append(f"\n{'='*70}")
    report.append("  METRICAS DE RISCO")
    report.append(f"{'='*70}")
    report.append(f"  Max Drawdown: {result.max_drawdown:.2%}")
    report.append(f"  Max Drawdown Pips: {result.max_drawdown_pips:.1f}")
    report.append(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    report.append(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
    report.append(f"  Calmar Ratio: {result.calmar_ratio:.2f}")

    report.append(f"\n{'='*70}")
    report.append("  LISTA DE TRADES")
    report.append(f"{'='*70}")

    for i, trade in enumerate(result.trades, 1):
        direction = "LONG" if trade.position_type.value == "LONG" else "SHORT"
        outcome = "WIN" if trade.is_winner else "LOSS"
        report.append(
            f"  {i:3d}. {direction:5s} | "
            f"{trade.entry_time.strftime('%Y-%m-%d %H:%M')} | "
            f"Entry: {trade.entry_price:.5f} | "
            f"Exit: {trade.exit_price:.5f} | "
            f"PnL: {trade.pnl_pips:+7.1f} pips | "
            f"{outcome:4s} | "
            f"{trade.exit_reason}"
        )

    report.append(f"\n{'='*70}")
    report.append("  FIM DO RELATORIO")
    report.append(f"{'='*70}")

    report_text = "\n".join(report)

    if filename:
        with open(filename, 'w') as f:
            f.write(report_text)
        print(f"\nRelatorio salvo em: {filename}")
    else:
        print(report_text)

    return report_text


def main():
    """Funcao principal"""
    parser = argparse.ArgumentParser(
        description="Backtest PRM - Protocolo Riemann-Mandelbrot com dados REAIS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
    python -m backtesting.backtest_prm                    # Backtest padrao
    python -m backtesting.backtest_prm --days 60          # 60 dias de historico
    python -m backtesting.backtest_prm --optimize         # Otimizacao de parametros
    python -m backtesting.backtest_prm --symbol GBPUSD    # Outro par
        """
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        help="Par de moedas (default: EURUSD)"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Dias de historico (default: 30)"
    )

    parser.add_argument(
        "--periodicity",
        type=str,
        default="H1",
        choices=['M5', 'M15', 'M30', 'H1', 'H4', 'D1'],
        help="Periodicidade (default: H1)"
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Capital inicial (default: 10000)"
    )

    parser.add_argument(
        "--stop-loss",
        type=float,
        default=20.0,
        help="Stop loss em pips (default: 20)"
    )

    parser.add_argument(
        "--take-profit",
        type=float,
        default=40.0,
        help="Take profit em pips (default: 40)"
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Executa otimizacao de parametros"
    )

    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Salva relatorio em arquivo"
    )

    args = parser.parse_args()

    if args.optimize:
        # Modo otimizacao
        run_optimization(
            symbol=args.symbol,
            days=args.days,
            periodicity=args.periodicity
        )
    else:
        # Backtest normal
        result = run_prm_backtest(
            symbol=args.symbol,
            days=args.days,
            periodicity=args.periodicity,
            initial_capital=args.capital,
            stop_loss_pips=args.stop_loss,
            take_profit_pips=args.take_profit
        )

        if args.report:
            generate_report(result, args.report)


if __name__ == "__main__":
    main()
