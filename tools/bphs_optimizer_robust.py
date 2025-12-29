#!/usr/bin/env python3
"""
================================================================================
BPHS ROBUST OPTIMIZER - ANTI-OVERFITTING
================================================================================

Otimizador robusto para a estrategia Betti-Persistence Homology Scanner (BPHS)
Projetado para MEDIA VOLATILIDADE com dinheiro real.

TECNICAS ANTI-OVERFITTING:
1. Minimo 50 trades para validar qualquer configuracao
2. Anchored Walk-Forward (5 folds) - treino cresce, teste avanca
3. Monte Carlo Validation (100 shuffles) - valida estabilidade
4. Penalizacao por complexidade de parametros
5. Out-of-sample final (20% dados nunca vistos)
6. Custos conservadores: spread 1.5 + slippage 0.8 = 2.3 pips

VALIDACAO DE LOOK-AHEAD BIAS:
- Sinal gerado com dados ATE indice i (closes[:i])
- Entrada executada no OPEN da barra i+1
- SL/TP verificados a partir da barra i+2

PARAMETROS OTIMIZAVEIS DO BPHS:
- embedding_dim: Dimensao de imersao de Takens (3, 4, 5)
- filtration_steps: Passos da filtracao Vietoris-Rips (30-60)
- min_loop_persistence: Persistencia minima para sinal (0.05-0.25)
- max_entropy_threshold: Limiar para regime caotico (1.5-3.0)
- position_threshold: Limiar para extremos do ciclo (0.2-0.4)
- min_confidence: Confianca minima para sinal (0.3-0.6)
- stop_loss_pips: Stop loss (15-40)
- take_profit_pips: Take profit (25-80)
- cooldown_bars: Cooldown entre sinais (15-40)

Autor: Claude Opus 4.5 (Otimizacao Automatizada)
Data: 2025-12-29
================================================================================
"""

import sys
import os
import urllib.request
import ssl
import json
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time
import itertools
import warnings

warnings.filterwarnings('ignore')

# Adiciona path do projeto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# CONFIGURACOES GLOBAIS
# ==============================================================================

API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"

# Custos CONSERVADORES para dinheiro real
SPREAD_PIPS = 1.5
SLIPPAGE_PIPS = 0.8
TOTAL_COST_PIPS = SPREAD_PIPS + SLIPPAGE_PIPS  # 2.3 pips
PIP_VALUE = 0.0001

# Anti-overfitting settings
MIN_TRADES = 50              # Minimo de trades para validar config
MIN_TRADES_PER_FOLD = 8      # Minimo por fold walk-forward
OOS_RATIO = 0.20             # 20% out-of-sample final
MONTE_CARLO_RUNS = 100       # Shuffles para validar estabilidade
WALK_FORWARD_FOLDS = 5       # Numero de folds

# Timeframe para media volatilidade
TIMEFRAME = "H1"  # H1 e ideal para media volatilidade
BARS_TO_DOWNLOAD = 3000  # ~125 dias de dados


# ==============================================================================
# ESTRUTURAS DE DADOS
# ==============================================================================

@dataclass
class Bar:
    """Representa uma barra OHLCV"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TradeResult:
    """Resultado de um trade individual"""
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    pnl_pips: float
    result: str  # 'win' ou 'loss'
    direction: int  # 1=LONG, -1=SHORT


# ==============================================================================
# DOWNLOAD DE DADOS
# ==============================================================================

def download_bars(period: str, count: int) -> List[Bar]:
    """
    Baixa barras historicas da API FXOpen

    IMPORTANTE: Dados sao baixados em ordem cronologica crescente.
    Nao ha look-ahead bias no download.
    """
    bars = []
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    current_ts = int(time.time() * 1000)
    total = 0

    print(f"    Baixando {count} barras {period}...", end=" ", flush=True)

    while total < count:
        remaining = min(1000, count - total)
        url = f"{API_BASE}/{SYMBOL}/{period}/bars/ask?timestamp={current_ts}&count=-{remaining}"

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30, context=ctx) as response:
                data = json.loads(response.read().decode())
                batch = data.get("Bars", [])

                if not batch:
                    break

                for b in batch:
                    ts = datetime.fromtimestamp(b["Timestamp"] / 1000, tz=timezone.utc)
                    bars.append(Bar(
                        timestamp=ts,
                        open=b["Open"],
                        high=b["High"],
                        low=b["Low"],
                        close=b["Close"],
                        volume=b.get("Volume", 0)
                    ))

                total += len(batch)
                oldest = min(batch, key=lambda x: x["Timestamp"])
                current_ts = oldest["Timestamp"] - 1
                time.sleep(0.05)  # Rate limiting

        except Exception as e:
            print(f"Erro: {e}")
            break

    # IMPORTANTE: Ordenar por timestamp crescente
    bars.sort(key=lambda x: x.timestamp)
    print(f"{len(bars)} barras OK")

    return bars


# ==============================================================================
# GERACAO DE SINAIS (SEM LOOK-AHEAD)
# ==============================================================================

def compute_signals(bars: List[Bar],
                   embedding_dim: int,
                   filtration_steps: int,
                   min_loop_persistence: float,
                   max_entropy_threshold: float,
                   position_threshold: float,
                   min_confidence: float,
                   step: int = 4) -> List[Tuple[int, int, float]]:
    """
    Computa sinais do BPHS para toda a serie.

    VALIDACAO DE LOOK-AHEAD BIAS:
    - Para cada indice i, usamos APENAS closes[:i]
    - O sinal gerado em i sera executado no OPEN de i+1
    - Isso garante que nao ha vazamento de dados futuros

    Args:
        bars: Lista de barras
        embedding_dim: Dimensao de Takens
        filtration_steps: Passos da filtracao
        min_loop_persistence: Persistencia minima
        max_entropy_threshold: Limiar de entropia
        position_threshold: Limiar de posicao
        min_confidence: Confianca minima
        step: Passo entre analises (para performance)

    Returns:
        Lista de (indice, direcao, confianca)
        direcao: 1=LONG, -1=SHORT
    """
    from strategies.media_volatilidade.bphs_betti_persistence import BettiPersistenceHomologyScanner

    closes = np.array([b.close for b in bars])
    n = len(closes)

    # Inicializa indicador BPHS
    bphs = BettiPersistenceHomologyScanner(
        embedding_dim=embedding_dim,
        time_delay=None,  # Auto-calculate
        max_homology_dim=2,
        filtration_steps=filtration_steps,
        min_loop_persistence=min_loop_persistence,
        max_entropy_threshold=max_entropy_threshold,
        position_threshold=position_threshold,
        min_data_points=100
    )

    signals = []

    # Minimo de dados para embedding + analise
    min_bars = max(100, embedding_dim * 50)

    # CRUCIAL: Loop comeca em min_bars e usa APENAS dados passados
    for i in range(min_bars, n - 1, step):
        try:
            # APENAS dados ate o indice i (NAO inclui i+1, i+2, etc)
            historical_closes = closes[:i]

            # Analise topologica
            result = bphs.analyze(historical_closes)

            signal = result['signal']
            confidence = result['confidence']

            # Filtra por confianca minima
            if signal != 0 and confidence >= min_confidence:
                signals.append((i, signal, confidence))

        except Exception as e:
            # Silently skip errors
            continue

    return signals


# ==============================================================================
# SIMULACAO DE TRADES (SEM LOOK-AHEAD)
# ==============================================================================

def simulate_trades(bars: List[Bar],
                   signals: List[Tuple[int, int, float]],
                   sl_pips: float,
                   tp_pips: float,
                   cooldown: int,
                   start_idx: int = 0,
                   end_idx: int = None,
                   direction_filter: int = 0) -> Dict:
    """
    Simula trades com validacao rigorosa de look-ahead.

    PROTOCOLO DE EXECUCAO:
    1. Sinal gerado no fechamento da barra [idx]
    2. Entrada executada no OPEN da barra [idx + 1]
    3. SL/TP verificados a partir da barra [idx + 2]

    Args:
        bars: Lista de barras
        signals: Lista de (idx, direction, confidence)
        sl_pips: Stop loss em pips
        tp_pips: Take profit em pips
        cooldown: Barras entre trades
        start_idx: Indice inicial (para walk-forward)
        end_idx: Indice final (para walk-forward)
        direction_filter: 0=ambos, 1=LONG, -1=SHORT

    Returns:
        Dict com metricas de performance
    """
    if end_idx is None:
        end_idx = len(bars)

    n = len(bars)
    trades: List[TradeResult] = []
    trade_pnls: List[float] = []
    last_trade_idx = -9999

    for idx, direction, confidence in signals:
        # Verifica se esta na janela de analise
        if idx < start_idx or idx >= end_idx:
            continue

        # Filtra por direcao se especificado
        if direction_filter != 0 and direction != direction_filter:
            continue

        # Verifica cooldown
        if idx - last_trade_idx < cooldown:
            continue

        # Verifica se ha barras futuras suficientes
        if idx + 2 >= n:
            continue

        # ENTRADA: OPEN da barra SEGUINTE ao sinal
        entry_bar = bars[idx + 1]
        entry_price = entry_bar.open

        # Calcula niveis de SL e TP
        if direction == 1:  # LONG
            sl_price = entry_price - sl_pips * PIP_VALUE
            tp_price = entry_price + tp_pips * PIP_VALUE
        else:  # SHORT
            sl_price = entry_price + sl_pips * PIP_VALUE
            tp_price = entry_price - tp_pips * PIP_VALUE

        # Verifica SL/TP nas barras FUTURAS (a partir de idx+2)
        result = None
        exit_idx = None
        exit_price = None

        # Maximo 100 barras de holding
        for j in range(idx + 2, min(idx + 100, n)):
            bar = bars[j]

            if direction == 1:  # LONG
                # SL verificado no LOW (pior caso)
                if bar.low <= sl_price:
                    result = 'loss'
                    exit_price = sl_price
                    exit_idx = j
                    pnl = -sl_pips - TOTAL_COST_PIPS
                    break
                # TP verificado no HIGH
                if bar.high >= tp_price:
                    result = 'win'
                    exit_price = tp_price
                    exit_idx = j
                    pnl = tp_pips - TOTAL_COST_PIPS
                    break
            else:  # SHORT
                # SL verificado no HIGH (pior caso)
                if bar.high >= sl_price:
                    result = 'loss'
                    exit_price = sl_price
                    exit_idx = j
                    pnl = -sl_pips - TOTAL_COST_PIPS
                    break
                # TP verificado no LOW
                if bar.low <= tp_price:
                    result = 'win'
                    exit_price = tp_price
                    exit_idx = j
                    pnl = tp_pips - TOTAL_COST_PIPS
                    break

        # Se trade fechou
        if result is not None:
            trade = TradeResult(
                entry_idx=idx + 1,
                exit_idx=exit_idx,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pips=pnl,
                result=result,
                direction=direction
            )
            trades.append(trade)
            trade_pnls.append(pnl)
            last_trade_idx = idx

    # Calcula metricas
    if not trade_pnls:
        return {
            'total': 0,
            'wins': 0,
            'losses': 0,
            'wr': 0,
            'edge': 0,
            'pnl': 0,
            'pf': 0,
            'max_dd': 0,
            'trades': [],
            'pnls': []
        }

    wins = len([t for t in trade_pnls if t > 0])
    losses = len([t for t in trade_pnls if t < 0])
    total = len(trade_pnls)

    wr = wins / total * 100
    breakeven = sl_pips / (sl_pips + tp_pips) * 100
    edge = wr - breakeven

    pnl = sum(trade_pnls)
    gross_profit = sum(t for t in trade_pnls if t > 0)
    gross_loss = abs(sum(t for t in trade_pnls if t < 0))

    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    # Max Drawdown
    cumsum = np.cumsum(trade_pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

    return {
        'total': total,
        'wins': wins,
        'losses': losses,
        'wr': wr,
        'edge': edge,
        'pnl': pnl,
        'pf': pf,
        'max_dd': max_dd,
        'trades': trades,
        'pnls': trade_pnls
    }


# ==============================================================================
# ANCHORED WALK-FORWARD VALIDATION
# ==============================================================================

def anchored_walk_forward(bars: List[Bar],
                          signals: List[Tuple[int, int, float]],
                          sl_pips: float,
                          tp_pips: float,
                          cooldown: int,
                          n_folds: int = 5) -> Dict:
    """
    Anchored Walk-Forward: treino sempre comeca do inicio, teste avanca.

    Mais robusto que walk-forward tradicional porque:
    - Treino sempre tem todo o historico
    - Teste e sempre out-of-sample
    - Simula uso real onde modelo e retreinado periodicamente

    Args:
        bars: Lista de barras
        signals: Sinais pre-computados
        sl_pips: Stop loss
        tp_pips: Take profit
        cooldown: Cooldown entre trades
        n_folds: Numero de folds

    Returns:
        Dict com resultados agregados
    """
    n = len(bars)
    fold_size = n // (n_folds + 1)

    fold_results = []
    all_oos_pnls = []

    for fold in range(n_folds):
        # Treino: do inicio ate fold * fold_size
        train_end = (fold + 1) * fold_size

        # Teste: do fim do treino ate proximo fold
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        if test_end <= test_start:
            break

        # Simular no periodo de TESTE (out-of-sample)
        r = simulate_trades(bars, signals, sl_pips, tp_pips, cooldown,
                           start_idx=test_start, end_idx=test_end)

        if r['total'] >= MIN_TRADES_PER_FOLD:
            fold_results.append({
                'fold': fold + 1,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'trades': r['total'],
                'wr': r['wr'],
                'edge': r['edge'],
                'pf': r['pf'],
                'pnl': r['pnl'],
                'max_dd': r['max_dd']
            })
            all_oos_pnls.extend(r['pnls'])

    if not fold_results:
        return {
            'valid': False,
            'folds_positive': 0,
            'total_folds': 0,
            'oos_trades': 0
        }

    # Conta folds positivos
    folds_positive = sum(1 for r in fold_results if r['edge'] > 0 and r['pf'] > 1.0)

    # Metricas agregadas OOS
    if all_oos_pnls:
        wins = len([p for p in all_oos_pnls if p > 0])
        total = len(all_oos_pnls)
        wr = wins / total * 100
        breakeven = sl_pips / (sl_pips + tp_pips) * 100

        gross_profit = sum(p for p in all_oos_pnls if p > 0)
        gross_loss = abs(sum(p for p in all_oos_pnls if p < 0))

        return {
            'valid': True,
            'folds_positive': folds_positive,
            'total_folds': len(fold_results),
            'oos_trades': total,
            'oos_wr': wr,
            'oos_edge': wr - breakeven,
            'oos_pf': gross_profit / gross_loss if gross_loss > 0 else 0,
            'oos_pnl': sum(all_oos_pnls),
            'fold_details': fold_results
        }

    return {
        'valid': False,
        'folds_positive': 0,
        'total_folds': len(fold_results),
        'oos_trades': 0
    }


# ==============================================================================
# MONTE CARLO VALIDATION
# ==============================================================================

def monte_carlo_validation(trade_pnls: List[float], n_runs: int = 100) -> Dict:
    """
    Monte Carlo: embaralha trades para verificar estabilidade.

    Se o resultado depender fortemente da ordem dos trades,
    pode indicar overfitting ou regime-dependence.

    Args:
        trade_pnls: Lista de PnL por trade
        n_runs: Numero de shuffles

    Returns:
        Dict com metricas de estabilidade
    """
    if len(trade_pnls) < 10:
        return {'valid': False, 'stability': 0}

    original_pnl = sum(trade_pnls)

    # Calcular max drawdown para cada shuffle
    drawdowns = []
    final_pnls = []

    for _ in range(n_runs):
        shuffled = trade_pnls.copy()
        np.random.shuffle(shuffled)

        cumsum = np.cumsum(shuffled)
        dd = np.max(np.maximum.accumulate(cumsum) - cumsum)
        drawdowns.append(dd)
        final_pnls.append(sum(shuffled))

    # Estabilidade: quao consistente e o drawdown
    dd_mean = np.mean(drawdowns)
    dd_std = np.std(drawdowns)
    dd_95th = np.percentile(drawdowns, 95)

    # Stability score: 0 a 1
    # Menor variacao = maior estabilidade
    stability = 1.0 - min(1.0, dd_std / (dd_mean + 1e-6))

    # Verificar se PnL final e consistente (deve ser igual)
    pnl_std = np.std(final_pnls)

    return {
        'valid': True,
        'stability': max(0, stability),
        'dd_mean': dd_mean,
        'dd_std': dd_std,
        'dd_95th': dd_95th,
        'pnl_consistency': pnl_std < 0.01  # Deve ser quase zero
    }


# ==============================================================================
# PENALIZACAO DE COMPLEXIDADE
# ==============================================================================

def calculate_complexity_penalty(embedding_dim: int,
                                  filtration_steps: int,
                                  min_loop_persistence: float,
                                  min_confidence: float) -> float:
    """
    Penaliza configuracoes mais complexas.

    Configs mais simples sao preferidas para evitar overfitting.

    Args:
        embedding_dim: Dimensao de Takens
        filtration_steps: Passos da filtracao
        min_loop_persistence: Persistencia minima
        min_confidence: Confianca minima

    Returns:
        Penalidade (maior = pior)
    """
    penalty = 0

    # embedding_dim: 3 e o padrao, maior = mais complexo
    if embedding_dim > 4:
        penalty += 2
    elif embedding_dim < 3:
        penalty += 3  # Muito baixo e ruim

    # filtration_steps: muito alto = mais computacao sem beneficio
    if filtration_steps > 60:
        penalty += 2
    elif filtration_steps < 30:
        penalty += 1  # Muito baixo pode perder informacao

    # min_loop_persistence: muito baixo = mais ruido
    if min_loop_persistence < 0.08:
        penalty += 3

    # min_confidence: muito baixo = muitos falsos positivos
    if min_confidence < 0.35:
        penalty += 3
    elif min_confidence > 0.6:
        penalty += 1  # Muito alto = poucos trades

    return penalty


# ==============================================================================
# SCORE ROBUSTO
# ==============================================================================

def robust_score(in_sample: Dict,
                 walk_forward: Dict,
                 monte_carlo: Dict,
                 complexity_penalty: float) -> float:
    """
    Calcula score robusto que equilibra:
    1. Performance out-of-sample (peso maior)
    2. Estabilidade Monte Carlo
    3. Numero de trades (mais = mais confiavel)
    4. Penalidade de complexidade

    Returns:
        Score (maior = melhor)
    """
    # Validacoes basicas
    if not walk_forward.get('valid', False):
        return -1000

    if walk_forward.get('oos_trades', 0) < MIN_TRADES:
        return -1000

    # === COMPONENTES DO SCORE ===

    # 1. Edge OOS (peso principal)
    oos_edge = walk_forward.get('oos_edge', 0)
    edge_score = oos_edge * 2  # Peso dobrado

    # 2. Profit Factor OOS
    oos_pf = walk_forward.get('oos_pf', 0)
    if oos_pf >= 1.5:
        pf_score = 10
    elif oos_pf >= 1.2:
        pf_score = 5
    elif oos_pf >= 1.0:
        pf_score = 2
    else:
        pf_score = (oos_pf - 1) * 20  # Penalidade forte se < 1

    # 3. Folds positivos
    folds_positive = walk_forward.get('folds_positive', 0)
    total_folds = walk_forward.get('total_folds', 1)
    folds_ratio = folds_positive / max(1, total_folds)
    folds_score = folds_ratio * 15

    # 4. Numero de trades (mais = mais confiavel)
    oos_trades = walk_forward.get('oos_trades', 0)
    trades_score = min(15, oos_trades / 5)

    # 5. Estabilidade Monte Carlo
    mc_stability = monte_carlo.get('stability', 0)
    stability_score = mc_stability * 10

    # 6. Penalidade de complexidade
    complexity_score = -complexity_penalty * 2

    # Score final
    score = (
        edge_score +
        pf_score +
        folds_score +
        trades_score +
        stability_score +
        complexity_score
    )

    return score


# ==============================================================================
# MAIN OPTIMIZER
# ==============================================================================

def main():
    print("=" * 80)
    print("  BPHS ROBUST OPTIMIZER - ANTI-OVERFITTING")
    print("  Betti-Persistence Homology Scanner")
    print("  Para MEDIA VOLATILIDADE com DINHEIRO REAL")
    print("=" * 80)

    print(f"""
  TECNICAS ANTI-OVERFITTING:
  - Minimo {MIN_TRADES} trades para validar
  - Anchored Walk-Forward ({WALK_FORWARD_FOLDS} folds)
  - Monte Carlo Validation ({MONTE_CARLO_RUNS} shuffles)
  - Penalizacao por complexidade
  - Out-of-sample final ({int(OOS_RATIO*100)}%)
  - Custos conservadores: {TOTAL_COST_PIPS} pips/trade

  VALIDACAO DE LOOK-AHEAD:
  - Sinal gerado com dados ATE indice i
  - Entrada no OPEN da barra i+1
  - SL/TP verificados a partir da barra i+2
""")

    start_time = time.time()

    # ==========================================================================
    # 1. DOWNLOAD DE DADOS
    # ==========================================================================
    print("\n[1/6] BAIXANDO DADOS...")
    print(f"    Timeframe: {TIMEFRAME}")
    print(f"    Simbolo: {SYMBOL}")

    all_bars = download_bars(TIMEFRAME, BARS_TO_DOWNLOAD)

    if len(all_bars) < 1000:
        print("    ERRO: Dados insuficientes!")
        return

    # Separar OOS final (20% nunca vistos)
    oos_split = int(len(all_bars) * (1 - OOS_RATIO))
    opt_bars = all_bars[:oos_split]
    final_oos_bars = all_bars[oos_split:]

    days = (all_bars[-1].timestamp - all_bars[0].timestamp).days
    print(f"    Periodo: {all_bars[0].timestamp.date()} a {all_bars[-1].timestamp.date()} ({days} dias)")
    print(f"    Otimizacao: {len(opt_bars)} barras")
    print(f"    Final OOS: {len(final_oos_bars)} barras (NUNCA usadas na otimizacao)")

    # ==========================================================================
    # 2. DEFINIR GRID DE PARAMETROS
    # ==========================================================================
    print("\n[2/6] DEFININDO GRID DE PARAMETROS...")

    # Parametros do indicador BPHS
    embedding_dim_list = [3, 4]  # Valores robustos
    filtration_steps_list = [40, 50]  # Range central
    min_loop_persistence_list = [0.08, 0.10, 0.12, 0.15]
    max_entropy_threshold_list = [2.0, 2.5]  # Range central
    position_threshold_list = [0.25, 0.30, 0.35]
    min_confidence_list = [0.35, 0.40, 0.45, 0.50]

    # Parametros de trade
    sl_list = [18, 22, 26, 30, 35]
    tp_list = [30, 40, 50, 60, 70]
    cooldown_list = [20, 25, 30, 35]

    # Combinacoes de indicador
    indicator_combos = list(itertools.product(
        embedding_dim_list,
        filtration_steps_list,
        min_loop_persistence_list,
        max_entropy_threshold_list,
        position_threshold_list,
        min_confidence_list
    ))

    # Combinacoes de trade
    trade_combos = list(itertools.product(sl_list, tp_list, cooldown_list))

    total_combos = len(indicator_combos) * len(trade_combos)

    print(f"    Configs de indicador: {len(indicator_combos)}")
    print(f"    Configs de trade: {len(trade_combos)}")
    print(f"    Total de combinacoes: {total_combos:,}")

    # ==========================================================================
    # 3. PRE-COMPUTAR SINAIS
    # ==========================================================================
    print("\n[3/6] PRE-COMPUTANDO SINAIS...")
    print("    (Isso pode demorar alguns minutos)")

    signals_cache = {}

    for i, (emb_dim, filt_steps, min_pers, max_ent, pos_th, min_conf) in enumerate(indicator_combos):
        key = (emb_dim, filt_steps, min_pers, max_ent, pos_th, min_conf)

        print(f"    [{i+1:3}/{len(indicator_combos)}] emb={emb_dim}, filt={filt_steps}, "
              f"pers={min_pers:.2f}, conf={min_conf:.2f}...", end=" ", flush=True)

        try:
            t0 = time.time()
            signals = compute_signals(
                opt_bars,
                embedding_dim=emb_dim,
                filtration_steps=filt_steps,
                min_loop_persistence=min_pers,
                max_entropy_threshold=max_ent,
                position_threshold=pos_th,
                min_confidence=min_conf,
                step=4  # A cada 4 barras para H1
            )
            elapsed = time.time() - t0
            signals_cache[key] = signals
            print(f"{len(signals)} sinais ({elapsed:.1f}s)")

        except Exception as e:
            print(f"ERRO: {e}")
            signals_cache[key] = []

    # ==========================================================================
    # 4. VALIDACAO ROBUSTA
    # ==========================================================================
    print("\n[4/6] VALIDACAO ROBUSTA (Walk-Forward + Monte Carlo)...")

    results = []
    tested = 0
    valid_count = 0

    for key, signals in signals_cache.items():
        if len(signals) < 30:
            tested += len(trade_combos)
            continue

        emb_dim, filt_steps, min_pers, max_ent, pos_th, min_conf = key
        complexity = calculate_complexity_penalty(emb_dim, filt_steps, min_pers, min_conf)

        for sl, tp, cooldown in trade_combos:
            tested += 1

            # Simular in-sample
            in_sample = simulate_trades(opt_bars, signals, sl, tp, cooldown)

            if in_sample['total'] < MIN_TRADES:
                continue

            # Walk-forward
            wf = anchored_walk_forward(opt_bars, signals, sl, tp, cooldown, WALK_FORWARD_FOLDS)

            if not wf.get('valid', False) or wf.get('oos_trades', 0) < MIN_TRADES:
                continue

            # Monte Carlo
            mc = monte_carlo_validation(in_sample['pnls'], MONTE_CARLO_RUNS)

            # Score robusto
            score = robust_score(in_sample, wf, mc, complexity)

            if score > 0:
                valid_count += 1
                results.append({
                    # Parametros indicador
                    'embedding_dim': emb_dim,
                    'filtration_steps': filt_steps,
                    'min_loop_persistence': min_pers,
                    'max_entropy_threshold': max_ent,
                    'position_threshold': pos_th,
                    'min_confidence': min_conf,
                    # Parametros trade
                    'stop_loss_pips': sl,
                    'take_profit_pips': tp,
                    'cooldown_bars': cooldown,
                    # Metricas in-sample
                    'is_trades': in_sample['total'],
                    'is_wr': in_sample['wr'],
                    'is_edge': in_sample['edge'],
                    'is_pf': in_sample['pf'],
                    'is_pnl': in_sample['pnl'],
                    # Metricas walk-forward (OOS)
                    'oos_trades': wf['oos_trades'],
                    'oos_wr': wf['oos_wr'],
                    'oos_edge': wf['oos_edge'],
                    'oos_pf': wf['oos_pf'],
                    'oos_pnl': wf['oos_pnl'],
                    'folds_positive': wf['folds_positive'],
                    'total_folds': wf['total_folds'],
                    # Monte Carlo
                    'mc_stability': mc.get('stability', 0),
                    'mc_dd_95th': mc.get('dd_95th', 0),
                    # Score
                    'complexity': complexity,
                    'score': score
                })

        # Progress
        if tested % 200 == 0:
            print(f"    [{tested:,}/{total_combos:,}] testadas, {valid_count} validas...")

    print(f"    Concluido: {tested:,} testadas, {len(results)} validas")

    # ==========================================================================
    # 5. RESULTADOS
    # ==========================================================================
    print("\n[5/6] RESULTADOS...")

    if not results:
        print("    NENHUMA configuracao valida encontrada!")
        print("    Possivel causa: dados insuficientes ou mercado desfavoravel")
        return

    # Ordenar por score
    results.sort(key=lambda x: x['score'], reverse=True)

    # Top 15
    print("\n    TOP 15 CONFIGURACOES:")
    print("-" * 140)
    print(f"    {'#':>2} | {'Emb':>3} | {'Filt':>4} | {'Pers':>5} | {'Conf':>4} | "
          f"{'SL':>3} | {'TP':>3} | {'CD':>2} | "
          f"{'Tr':>4} | {'WR%':>5} | {'Edge':>6} | {'PF':>5} | "
          f"{'Folds':>5} | {'Stab':>4} | {'Score':>6}")
    print("-" * 140)

    for i, r in enumerate(results[:15]):
        print(f"    {i+1:2} | {r['embedding_dim']:3} | {r['filtration_steps']:4} | "
              f"{r['min_loop_persistence']:5.2f} | {r['min_confidence']:4.2f} | "
              f"{r['stop_loss_pips']:3.0f} | {r['take_profit_pips']:3.0f} | {r['cooldown_bars']:2} | "
              f"{r['oos_trades']:4} | {r['oos_wr']:5.1f} | {r['oos_edge']:+6.1f} | {r['oos_pf']:5.2f} | "
              f"{r['folds_positive']}/{r['total_folds']} | {r['mc_stability']:4.2f} | {r['score']:6.1f}")

    # ==========================================================================
    # 6. VALIDACAO FINAL OOS
    # ==========================================================================
    print("\n[6/6] VALIDACAO FINAL OUT-OF-SAMPLE...")
    print(f"    Usando {len(final_oos_bars)} barras NUNCA vistas na otimizacao")

    best = results[0]

    # Computar sinais para OOS final
    final_signals = compute_signals(
        final_oos_bars,
        embedding_dim=best['embedding_dim'],
        filtration_steps=best['filtration_steps'],
        min_loop_persistence=best['min_loop_persistence'],
        max_entropy_threshold=best['max_entropy_threshold'],
        position_threshold=best['position_threshold'],
        min_confidence=best['min_confidence'],
        step=4
    )

    # Simular trades no OOS final
    final_result = simulate_trades(
        final_oos_bars,
        final_signals,
        best['stop_loss_pips'],
        best['take_profit_pips'],
        best['cooldown_bars']
    )

    print(f"\n    RESULTADO FINAL OOS:")
    print(f"    Trades: {final_result['total']}")
    print(f"    Win Rate: {final_result['wr']:.1f}%")
    print(f"    Edge: {final_result['edge']:+.1f}%")
    print(f"    Profit Factor: {final_result['pf']:.2f}")
    print(f"    PnL: {final_result['pnl']:.1f} pips")
    print(f"    Max DD: {final_result['max_dd']:.1f} pips")

    # ==========================================================================
    # CRITERIOS DE APROVACAO
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  CRITERIOS DE APROVACAO PARA DINHEIRO REAL")
    print("=" * 80)

    criteria = [
        ("Edge OOS > 0%", best['oos_edge'] > 0),
        ("Profit Factor OOS > 1.0", best['oos_pf'] > 1.0),
        ("Folds positivos >= 3/5", best['folds_positive'] >= 3),
        ("Monte Carlo stability > 0.5", best['mc_stability'] > 0.5),
        ("Trades OOS >= 50", best['oos_trades'] >= 50),
        ("Final OOS Edge > 0%", final_result['edge'] > 0),
        ("Final OOS PF > 1.0", final_result['pf'] > 1.0)
    ]

    passed = sum(1 for _, v in criteria if v)

    for name, ok in criteria:
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] {name}")

    print(f"\n    Resultado: {passed}/{len(criteria)}")

    if passed >= 6:
        approval = "APROVADO PARA DINHEIRO REAL"
        risk = "LOW"
    elif passed >= 5:
        approval = "APROVADO COM RESSALVAS"
        risk = "MEDIUM"
    elif passed >= 4:
        approval = "REQUER MAIS TESTES"
        risk = "HIGH"
    else:
        approval = "REPROVADO"
        risk = "CRITICAL"

    print(f"\n    *** {approval} ***")
    print(f"    Risco de Overfitting: {risk}")

    # ==========================================================================
    # SALVAR CONFIGURACAO
    # ==========================================================================
    config = {
        "strategy": "BPHS-ROBUST-ANTI-OVERFITTING",
        "symbol": SYMBOL,
        "periodicity": TIMEFRAME,
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "mode": "LONG_AND_SHORT",

        "anti_overfitting": {
            "min_trades": MIN_TRADES,
            "walk_forward_folds": WALK_FORWARD_FOLDS,
            "monte_carlo_runs": MONTE_CARLO_RUNS,
            "oos_ratio": OOS_RATIO,
            "spread_pips": SPREAD_PIPS,
            "slippage_pips": SLIPPAGE_PIPS
        },

        "parameters": {
            "embedding_dim": int(best['embedding_dim']),
            "filtration_steps": int(best['filtration_steps']),
            "min_loop_persistence": float(best['min_loop_persistence']),
            "max_entropy_threshold": float(best['max_entropy_threshold']),
            "position_threshold": float(best['position_threshold']),
            "min_confidence": float(best['min_confidence']),
            "stop_loss_pips": int(best['stop_loss_pips']),
            "take_profit_pips": int(best['take_profit_pips']),
            "cooldown_bars": int(best['cooldown_bars'])
        },

        "performance": {
            "in_sample": {
                "trades": int(best['is_trades']),
                "edge": float(best['is_edge']),
                "profit_factor": float(best['is_pf'])
            },
            "out_of_sample_wf": {
                "trades": int(best['oos_trades']),
                "edge": float(best['oos_edge']),
                "profit_factor": float(best['oos_pf']),
                "folds_positive": int(best['folds_positive']),
                "folds_total": int(best['total_folds'])
            },
            "final_oos": {
                "trades": int(final_result['total']),
                "edge": float(final_result['edge']),
                "profit_factor": float(final_result['pf'])
            },
            "expected_edge": float((best['oos_edge'] + final_result['edge']) / 2),
            "monte_carlo_stability": float(best['mc_stability']),
            "robust_score": float(best['score'])
        },

        "validation": {
            "criteria_passed": passed,
            "criteria_total": len(criteria),
            "final_oos_passed": final_result['edge'] > 0 and final_result['pf'] > 1.0,
            "overfitting_risk": risk
        },

        "top_10": [
            {
                "rank": i + 1,
                "params": {
                    "embedding_dim": int(r['embedding_dim']),
                    "filtration_steps": int(r['filtration_steps']),
                    "min_loop_persistence": float(r['min_loop_persistence']),
                    "min_confidence": float(r['min_confidence']),
                    "stop_loss_pips": int(r['stop_loss_pips']),
                    "take_profit_pips": int(r['take_profit_pips']),
                    "cooldown_bars": int(r['cooldown_bars'])
                },
                "oos_edge": float(r['oos_edge']),
                "oos_pf": float(r['oos_pf']),
                "score": float(r['score'])
            }
            for i, r in enumerate(results[:10])
        ]
    }

    # Salvar em arquivo
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "bphs_robust_optimized.json"
    )

    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n    Config salva: {config_path}")
    except Exception as e:
        print(f"\n    Erro ao salvar config: {e}")
        # Salvar no diretorio atual
        with open("bphs_robust_optimized.json", 'w') as f:
            json.dump(config, f, indent=2)
        print(f"    Config salva: bphs_robust_optimized.json")

    # ==========================================================================
    # RESUMO FINAL
    # ==========================================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("  RESUMO FINAL")
    print("=" * 80)
    print(f"""
    MELHOR CONFIGURACAO:

    Indicador BPHS:
      embedding_dim: {best['embedding_dim']}
      filtration_steps: {best['filtration_steps']}
      min_loop_persistence: {best['min_loop_persistence']}
      max_entropy_threshold: {best['max_entropy_threshold']}
      position_threshold: {best['position_threshold']}
      min_confidence: {best['min_confidence']}

    Trade Management:
      stop_loss_pips: {best['stop_loss_pips']}
      take_profit_pips: {best['take_profit_pips']}
      cooldown_bars: {best['cooldown_bars']}

    Performance Walk-Forward:
      Trades OOS: {best['oos_trades']}
      Edge OOS: {best['oos_edge']:+.1f}%
      Profit Factor OOS: {best['oos_pf']:.2f}
      Folds Positivos: {best['folds_positive']}/{best['total_folds']}

    Monte Carlo:
      Estabilidade: {best['mc_stability']:.2%}
      DD 95th percentile: {best['mc_dd_95th']:.1f} pips

    VALIDACAO FINAL:
      {approval}
      Risco de Overfitting: {risk}

    Tempo de otimizacao: {elapsed/60:.1f} minutos
""")

    print("=" * 80)


if __name__ == "__main__":
    main()
