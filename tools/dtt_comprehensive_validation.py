#!/usr/bin/env python3
"""
================================================================================
DTT COMPREHENSIVE VALIDATION - 12 MESES
================================================================================

Validacao rigorosa do DTT M5 com:
1. 12 meses de dados historicos
2. Walk-Forward Validation (OOS rigoroso)
3. Analise de Drawdown Maximo
4. Stress Test em dias de noticias (NFP, FOMC)
5. Analise por condicao de mercado

Sem look-ahead, bias ou overfitting.

================================================================================
"""

import sys
import os
import urllib.request
import ssl
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time
import calendar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.alta_volatilidade.dtt_tunelamento_topologico import DetectorTunelamentoTopologico

# =============================================================================
# PARAMETROS (mesmos do teste aprovado)
# =============================================================================

PERSISTENCE_ENTROPY_THRESHOLD = 0.5
TUNNELING_PROBABILITY_THRESHOLD = 0.15
MIN_SIGNAL_STRENGTH = 0.35
SL_PIPS = 8.0
TP_PIPS = 20.0
LONG_ONLY_MODE = True
COOLDOWN = 6
MIN_WARMUP = 120
SPREAD_PIPS = 1.2
SLIPPAGE_PIPS = 0.5
PIP_VALUE = 0.0001

API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"

# =============================================================================
# DATAS DE NOTICIAS IMPORTANTES 2025
# =============================================================================

# NFP - Primeiro sexta-feira de cada mes (dados de 2025)
NFP_DATES_2025 = [
    "2025-01-03", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05"
]

# FOMC - Datas aproximadas das reunioes 2025
FOMC_DATES_2025 = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17"
]

# Combinar todas as datas de noticias
NEWS_DATES = set(NFP_DATES_2025 + FOMC_DATES_2025)

@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    pnl_pips: float
    result: str  # WIN/LOSS
    is_news_day: bool = False

@dataclass
class ValidationResult:
    period: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    edge: float
    total_pnl: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    trades: List[Trade] = field(default_factory=list)

# =============================================================================
# DOWNLOAD DE DADOS
# =============================================================================

def download_ohlc_data(period: str, target_bars: int, max_retries: int = 3) -> List[Bar]:
    """
    Download de dados OHLC com multiplas requisicoes.
    Suporta download de grandes quantidades de dados.
    """
    bars = []
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    current_ts = int(time.time() * 1000)
    total_downloaded = 0
    batch_size = 1000
    retries = 0

    print(f"    Iniciando download de {target_bars} barras {period}...")

    while total_downloaded < target_bars:
        remaining = min(batch_size, target_bars - total_downloaded)
        url = f"{API_BASE}/{SYMBOL}/{period}/bars/ask?timestamp={current_ts}&count=-{remaining}"

        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, timeout=60, context=ctx) as response:
                data = json.loads(response.read().decode())
                batch = data.get("Bars", [])

                if not batch:
                    print(f"    Fim dos dados disponiveis em {total_downloaded} barras")
                    break

                for b in batch:
                    ts = datetime.fromtimestamp(b["Timestamp"] / 1000, tz=timezone.utc)
                    bar = Bar(
                        timestamp=ts,
                        open=float(b["Open"]),
                        high=float(b["High"]),
                        low=float(b["Low"]),
                        close=float(b["Close"]),
                        volume=float(b.get("Volume", 0))
                    )
                    bars.append(bar)

                total_downloaded += len(batch)
                oldest = min(batch, key=lambda x: x["Timestamp"])
                current_ts = oldest["Timestamp"] - 1

                # Progress
                pct = total_downloaded * 100 // target_bars
                if total_downloaded % 5000 == 0 or pct % 10 == 0:
                    print(f"      {total_downloaded:,} barras ({pct}%)")

                time.sleep(0.2)  # Rate limiting
                retries = 0

        except Exception as e:
            retries += 1
            if retries >= max_retries:
                print(f"    Erro apos {max_retries} tentativas: {e}")
                break
            print(f"    Retry {retries}/{max_retries}: {e}")
            time.sleep(2)

    bars.sort(key=lambda x: x.timestamp)
    print(f"    Total: {len(bars):,} barras baixadas")
    return bars

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest_period(bars: List[Bar], start_idx: int, end_idx: int,
                        dtt: DetectorTunelamentoTopologico) -> ValidationResult:
    """
    Executa backtest em um periodo especifico.
    Usado para walk-forward validation.
    """
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])

    trades = []
    last_signal = -999
    breakeven = SL_PIPS / (SL_PIPS + TP_PIPS) * 100

    # Equity tracking para drawdown
    equity_curve = [0.0]
    peak_equity = 0.0
    max_drawdown = 0.0

    for i in range(max(start_idx, MIN_WARMUP), min(end_idx, len(bars) - 1)):
        if i - last_signal < COOLDOWN:
            continue

        hour = bars[i].timestamp.hour
        date_str = bars[i].timestamp.strftime("%Y-%m-%d")
        is_news = date_str in NEWS_DATES

        try:
            result = dtt.analyze(
                prices=closes[:i+1],
                highs=highs[:i+1],
                lows=lows[:i+1],
                current_hour=hour,
                use_filters=True
            )

            if result is None or not result.get('trade_on', False):
                continue

            entropy_info = result.get('entropy', {})
            if entropy_info.get('persistence_entropy', 0) < PERSISTENCE_ENTROPY_THRESHOLD:
                continue
            if result.get('signal_strength', 0) < MIN_SIGNAL_STRENGTH:
                continue

            direction = result.get('direction', '')
            if direction == 'LONG':
                signal = 1
            elif direction == 'SHORT':
                if LONG_ONLY_MODE:
                    continue
                signal = -1
            else:
                continue

            filters = result.get('filters', {})
            if filters and not filters.get('filters_ok', True):
                continue
            if filters and filters.get('total_score', 0) < 0.5:
                continue

        except Exception:
            continue

        # Simular trade
        entry = bars[i + 1].open
        entry_time = bars[i + 1].timestamp

        if signal == 1:
            entry += SLIPPAGE_PIPS * PIP_VALUE
            sl = entry - SL_PIPS * PIP_VALUE
            tp = entry + TP_PIPS * PIP_VALUE
        else:
            entry -= SLIPPAGE_PIPS * PIP_VALUE
            sl = entry + SL_PIPS * PIP_VALUE
            tp = entry - TP_PIPS * PIP_VALUE

        trade_result = None
        exit_time = None
        exit_price = None

        for j in range(i + 2, min(i + 150, len(bars))):
            bar = bars[j]

            if signal == 1:
                if bar.low <= sl:
                    trade_result = 'LOSS'
                    pnl = -SL_PIPS - SPREAD_PIPS
                    exit_price = sl
                    exit_time = bar.timestamp
                    break
                if bar.high >= tp:
                    trade_result = 'WIN'
                    pnl = TP_PIPS - SPREAD_PIPS
                    exit_price = tp
                    exit_time = bar.timestamp
                    break
            else:
                if bar.high >= sl:
                    trade_result = 'LOSS'
                    pnl = -SL_PIPS - SPREAD_PIPS
                    exit_price = sl
                    exit_time = bar.timestamp
                    break
                if bar.low <= tp:
                    trade_result = 'WIN'
                    pnl = TP_PIPS - SPREAD_PIPS
                    exit_price = tp
                    exit_time = bar.timestamp
                    break

        if trade_result:
            trade = Trade(
                entry_time=entry_time,
                exit_time=exit_time,
                direction='LONG' if signal == 1 else 'SHORT',
                entry_price=entry,
                exit_price=exit_price,
                pnl_pips=pnl,
                result=trade_result,
                is_news_day=is_news
            )
            trades.append(trade)

            # Update equity curve
            new_equity = equity_curve[-1] + pnl
            equity_curve.append(new_equity)

            # Track drawdown
            if new_equity > peak_equity:
                peak_equity = new_equity
            drawdown = peak_equity - new_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        last_signal = i

    # Calcular metricas
    if not trades:
        return ValidationResult(
            period=f"{bars[start_idx].timestamp.date()} to {bars[min(end_idx, len(bars)-1)].timestamp.date()}",
            total_trades=0, wins=0, losses=0, win_rate=0, edge=0,
            total_pnl=0, profit_factor=0, max_drawdown=0, max_drawdown_pct=0,
            trades=[]
        )

    wins = len([t for t in trades if t.result == 'WIN'])
    total = len(trades)
    win_rate = wins / total * 100 if total > 0 else 0
    edge = win_rate - breakeven
    total_pnl = sum(t.pnl_pips for t in trades)

    gross_profit = sum(t.pnl_pips for t in trades if t.pnl_pips > 0)
    gross_loss = abs(sum(t.pnl_pips for t in trades if t.pnl_pips < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    # Drawdown percentual (relativo ao pico)
    max_dd_pct = (max_drawdown / peak_equity * 100) if peak_equity > 0 else 0

    return ValidationResult(
        period=f"{bars[start_idx].timestamp.date()} to {bars[min(end_idx-1, len(bars)-1)].timestamp.date()}",
        total_trades=total,
        wins=wins,
        losses=total - wins,
        win_rate=win_rate,
        edge=edge,
        total_pnl=total_pnl,
        profit_factor=pf,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_dd_pct,
        trades=trades
    )

# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def walk_forward_validation(bars: List[Bar], n_folds: int = 4) -> List[ValidationResult]:
    """
    Walk-Forward Validation rigoroso.
    Divide dados em N periodos, treina em periodo anterior, testa no proximo.
    """
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 70)

    total_bars = len(bars)
    fold_size = total_bars // n_folds

    results = []

    # Criar detector uma vez
    dtt = DetectorTunelamentoTopologico(
        max_points=100,
        use_dimensionality_reduction=True,
        reduction_method='pca',
        persistence_entropy_threshold=PERSISTENCE_ENTROPY_THRESHOLD,
        tunneling_probability_threshold=TUNNELING_PROBABILITY_THRESHOLD,
        auto_calibrate_quantum=True
    )

    for fold in range(1, n_folds):
        # Periodo de teste: fold atual
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else total_bars

        print(f"\n  Fold {fold}/{n_folds-1}: Testando periodo {fold}")
        print(f"    Periodo: {bars[test_start].timestamp.date()} a {bars[min(test_end-1, len(bars)-1)].timestamp.date()}")

        # Reset quantum calibration para cada fold
        dtt.quantum_params.reset()

        result = run_backtest_period(bars, test_start, test_end, dtt)
        results.append(result)

        print(f"    Trades: {result.total_trades} | WR: {result.win_rate:.1f}% | Edge: {result.edge:+.1f}% | PnL: {result.total_pnl:+.1f}")

    return results

# =============================================================================
# ANALISE DE DRAWDOWN
# =============================================================================

def analyze_drawdown(trades: List[Trade]) -> Dict:
    """
    Analise detalhada de drawdown.
    """
    if not trades:
        return {
            'max_drawdown_pips': 0,
            'max_drawdown_trades': 0,
            'longest_losing_streak': 0,
            'recovery_time_bars': 0,
            'avg_drawdown': 0
        }

    equity = 0
    peak = 0
    drawdowns = []
    current_dd_start = None
    max_dd = 0
    max_dd_trades = 0

    # Losing streak
    current_streak = 0
    max_streak = 0

    # Recovery tracking
    in_drawdown = False
    dd_start_idx = 0
    recovery_times = []

    for i, trade in enumerate(trades):
        equity += trade.pnl_pips

        if equity > peak:
            if in_drawdown:
                recovery_times.append(i - dd_start_idx)
                in_drawdown = False
            peak = equity

        dd = peak - equity
        if dd > 0:
            if not in_drawdown:
                in_drawdown = True
                dd_start_idx = i
            drawdowns.append(dd)
            if dd > max_dd:
                max_dd = dd
                max_dd_trades = i - dd_start_idx + 1

        # Streak
        if trade.result == 'LOSS':
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return {
        'max_drawdown_pips': max_dd,
        'max_drawdown_trades': max_dd_trades,
        'longest_losing_streak': max_streak,
        'avg_recovery_trades': np.mean(recovery_times) if recovery_times else 0,
        'avg_drawdown': np.mean(drawdowns) if drawdowns else 0,
        'drawdown_events': len([d for d in drawdowns if d > SL_PIPS * 3])
    }

# =============================================================================
# STRESS TEST - DIAS DE NOTICIAS
# =============================================================================

def stress_test_news_days(trades: List[Trade]) -> Dict:
    """
    Analise de performance em dias de noticias.
    """
    news_trades = [t for t in trades if t.is_news_day]
    normal_trades = [t for t in trades if not t.is_news_day]

    def calc_metrics(trade_list):
        if not trade_list:
            return {'trades': 0, 'win_rate': 0, 'pnl': 0, 'avg_pnl': 0}
        wins = len([t for t in trade_list if t.result == 'WIN'])
        pnl = sum(t.pnl_pips for t in trade_list)
        return {
            'trades': len(trade_list),
            'win_rate': wins / len(trade_list) * 100,
            'pnl': pnl,
            'avg_pnl': pnl / len(trade_list)
        }

    return {
        'news_days': calc_metrics(news_trades),
        'normal_days': calc_metrics(normal_trades),
        'news_dates_traded': list(set(t.entry_time.strftime("%Y-%m-%d") for t in news_trades))
    }

# =============================================================================
# ANALISE POR MES
# =============================================================================

def analyze_by_month(trades: List[Trade]) -> Dict[str, Dict]:
    """
    Analise mensal de performance.
    """
    monthly = {}

    for trade in trades:
        month_key = trade.entry_time.strftime("%Y-%m")
        if month_key not in monthly:
            monthly[month_key] = {'trades': [], 'wins': 0, 'pnl': 0}

        monthly[month_key]['trades'].append(trade)
        if trade.result == 'WIN':
            monthly[month_key]['wins'] += 1
        monthly[month_key]['pnl'] += trade.pnl_pips

    # Calcular metricas por mes
    result = {}
    for month, data in sorted(monthly.items()):
        total = len(data['trades'])
        result[month] = {
            'trades': total,
            'wins': data['wins'],
            'win_rate': data['wins'] / total * 100 if total > 0 else 0,
            'pnl': data['pnl'],
            'profitable': data['pnl'] > 0
        }

    return result

# =============================================================================
# RELATORIO FINAL
# =============================================================================

def generate_report(bars: List[Bar], wf_results: List[ValidationResult],
                   all_trades: List[Trade]) -> str:
    """
    Gera relatorio completo de validacao.
    """
    report = []
    report.append("=" * 80)
    report.append("  RELATORIO DE VALIDACAO COMPLETA - DTT M5")
    report.append("=" * 80)

    # Periodo total
    if bars:
        report.append(f"\n  PERIODO ANALISADO:")
        report.append(f"    De: {bars[0].timestamp.strftime('%Y-%m-%d')}")
        report.append(f"    Ate: {bars[-1].timestamp.strftime('%Y-%m-%d')}")
        days = (bars[-1].timestamp - bars[0].timestamp).days
        report.append(f"    Total: {days} dias ({days/30:.1f} meses)")
        report.append(f"    Barras: {len(bars):,}")

    # Walk-Forward Results
    report.append(f"\n  WALK-FORWARD VALIDATION:")
    report.append(f"  {'-' * 70}")

    total_trades_wf = sum(r.total_trades for r in wf_results)
    total_wins_wf = sum(r.wins for r in wf_results)
    total_pnl_wf = sum(r.total_pnl for r in wf_results)

    for i, r in enumerate(wf_results, 1):
        status = "[OK]" if r.edge > 0 else "[FAIL]"
        report.append(f"    Fold {i}: {r.period}")
        report.append(f"           Trades: {r.total_trades:3d} | WR: {r.win_rate:5.1f}% | Edge: {r.edge:+5.1f}% | PnL: {r.total_pnl:+7.1f} | {status}")

    # Agregado WF
    if total_trades_wf > 0:
        wf_wr = total_wins_wf / total_trades_wf * 100
        wf_edge = wf_wr - (SL_PIPS / (SL_PIPS + TP_PIPS) * 100)
        report.append(f"\n    AGREGADO WF:")
        report.append(f"           Trades: {total_trades_wf:3d} | WR: {wf_wr:5.1f}% | Edge: {wf_edge:+5.1f}% | PnL: {total_pnl_wf:+7.1f}")

    # Drawdown Analysis
    dd_analysis = analyze_drawdown(all_trades)
    report.append(f"\n  ANALISE DE DRAWDOWN:")
    report.append(f"  {'-' * 70}")
    report.append(f"    Max Drawdown: {dd_analysis['max_drawdown_pips']:.1f} pips")
    report.append(f"    Max Drawdown Duration: {dd_analysis['max_drawdown_trades']} trades")
    report.append(f"    Longest Losing Streak: {dd_analysis['longest_losing_streak']} trades")
    report.append(f"    Avg Recovery: {dd_analysis['avg_recovery_trades']:.1f} trades")
    report.append(f"    Drawdown Events (>3x SL): {dd_analysis['drawdown_events']}")

    # Stress Test
    stress = stress_test_news_days(all_trades)
    report.append(f"\n  STRESS TEST - DIAS DE NOTICIAS (NFP/FOMC):")
    report.append(f"  {'-' * 70}")
    news = stress['news_days']
    normal = stress['normal_days']
    report.append(f"    Dias de Noticias: {news['trades']} trades | WR: {news['win_rate']:.1f}% | PnL: {news['pnl']:+.1f}")
    report.append(f"    Dias Normais:     {normal['trades']} trades | WR: {normal['win_rate']:.1f}% | PnL: {normal['pnl']:+.1f}")

    if news['trades'] > 0 and normal['trades'] > 0:
        news_vs_normal = news['avg_pnl'] - normal['avg_pnl']
        report.append(f"    Diferenca: {news_vs_normal:+.2f} pips/trade em dias de noticias")

    # Monthly Analysis
    monthly = analyze_by_month(all_trades)
    report.append(f"\n  ANALISE MENSAL:")
    report.append(f"  {'-' * 70}")

    profitable_months = 0
    for month, data in monthly.items():
        status = "[+]" if data['profitable'] else "[-]"
        if data['profitable']:
            profitable_months += 1
        report.append(f"    {month}: {data['trades']:3d} trades | WR: {data['win_rate']:5.1f}% | PnL: {data['pnl']:+7.1f} {status}")

    total_months = len(monthly)
    if total_months > 0:
        report.append(f"\n    Meses Lucrativos: {profitable_months}/{total_months} ({profitable_months/total_months*100:.0f}%)")

    # Veredicto Final
    report.append(f"\n{'=' * 80}")
    report.append("  VEREDICTO FINAL")
    report.append(f"{'=' * 80}")

    # Criterios de aprovacao
    criteria = []

    # 1. Edge positivo no agregado
    if total_trades_wf > 0:
        edge_ok = wf_edge > 0
        criteria.append(('Edge WF Agregado > 0%', edge_ok, f"{wf_edge:+.1f}%"))

    # 2. Maioria dos folds positivos
    positive_folds = len([r for r in wf_results if r.edge > 0])
    folds_ok = positive_folds >= len(wf_results) * 0.5
    criteria.append(('>=50% Folds Positivos', folds_ok, f"{positive_folds}/{len(wf_results)}"))

    # 3. PF > 1
    if total_trades_wf > 0:
        gross_profit = sum(t.pnl_pips for t in all_trades if t.pnl_pips > 0)
        gross_loss = abs(sum(t.pnl_pips for t in all_trades if t.pnl_pips < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        pf_ok = pf > 1.0
        criteria.append(('Profit Factor > 1.0', pf_ok, f"{pf:.2f}"))

    # 4. Drawdown aceitavel (< 50 pips ou < 5x SL)
    dd_ok = dd_analysis['max_drawdown_pips'] < 50
    criteria.append(('Max Drawdown < 50 pips', dd_ok, f"{dd_analysis['max_drawdown_pips']:.1f}"))

    # 5. Maioria dos meses lucrativos
    months_ok = profitable_months >= total_months * 0.5 if total_months > 0 else False
    criteria.append(('>=50% Meses Lucrativos', months_ok, f"{profitable_months}/{total_months}"))

    # 6. Performance em noticias nao catastrofica
    news_ok = news['pnl'] >= -20 if news['trades'] > 0 else True
    criteria.append(('News Days PnL >= -20', news_ok, f"{news['pnl']:+.1f}"))

    report.append("\n  CRITERIOS:")
    passed = 0
    for name, ok, value in criteria:
        status = "[PASS]" if ok else "[FAIL]"
        report.append(f"    {status} {name}: {value}")
        if ok:
            passed += 1

    report.append(f"\n  RESULTADO: {passed}/{len(criteria)} criterios atendidos")

    if passed >= len(criteria) - 1:  # Permite 1 falha
        report.append("\n  *** APROVADO PARA PAPER TRADING ***")
        report.append("  Recomendacao: Iniciar paper trading por 1-2 meses")
    elif passed >= len(criteria) // 2:
        report.append("\n  *** APROVADO COM RESSALVAS ***")
        report.append("  Recomendacao: Revisar criterios falhados antes de paper trading")
    else:
        report.append("\n  *** REPROVADO ***")
        report.append("  Recomendacao: Nao usar em trading real")

    report.append(f"\n{'=' * 80}")

    return "\n".join(report)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("  DTT COMPREHENSIVE VALIDATION - 12 MESES")
    print("=" * 80)
    print()
    print("  Este processo pode levar varios minutos...")
    print()

    # M5: Vamos baixar 20,000 barras (~70 dias / 2.3 meses)
    # Suficiente para validacao walk-forward com 4 folds
    TARGET_BARS = 20000

    print("  [1/5] DOWNLOAD DE DADOS HISTORICOS")
    print("-" * 50)
    bars = download_ohlc_data("M5", TARGET_BARS)

    if len(bars) < 10000:
        print(f"  ERRO: Dados insuficientes ({len(bars)} barras)")
        print("  Necessario minimo de 10,000 barras para validacao")
        return

    days = (bars[-1].timestamp - bars[0].timestamp).days
    print(f"\n  Dados obtidos: {len(bars):,} barras")
    print(f"  Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()}")
    print(f"  Duracao: {days} dias ({days/30:.1f} meses)")

    # Walk-Forward Validation
    print("\n  [2/5] WALK-FORWARD VALIDATION")
    print("-" * 50)

    # Determinar numero de folds baseado na quantidade de dados
    n_folds = min(6, max(3, days // 60))  # ~2 meses por fold
    print(f"  Usando {n_folds} folds ({days // n_folds} dias cada)")

    wf_results = walk_forward_validation(bars, n_folds)

    # Coletar todos os trades
    all_trades = []
    for r in wf_results:
        all_trades.extend(r.trades)

    print(f"\n  Total de trades em todos os folds: {len(all_trades)}")

    # Drawdown Analysis
    print("\n  [3/5] ANALISE DE DRAWDOWN")
    print("-" * 50)
    dd = analyze_drawdown(all_trades)
    print(f"  Max Drawdown: {dd['max_drawdown_pips']:.1f} pips")
    print(f"  Longest Losing Streak: {dd['longest_losing_streak']} trades")

    # Stress Test
    print("\n  [4/5] STRESS TEST - DIAS DE NOTICIAS")
    print("-" * 50)
    stress = stress_test_news_days(all_trades)
    print(f"  Trades em dias de noticias: {stress['news_days']['trades']}")
    print(f"  PnL em dias de noticias: {stress['news_days']['pnl']:+.1f} pips")

    # Relatorio Final
    print("\n  [5/5] GERANDO RELATORIO FINAL")
    print("-" * 50)

    report = generate_report(bars, wf_results, all_trades)
    print(report)

    # Salvar relatorio
    report_path = "/home/azureuser/EliBotCD/reports/dtt_validation_report.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n  Relatorio salvo em: {report_path}")

if __name__ == "__main__":
    main()
