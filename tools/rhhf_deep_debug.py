#!/usr/bin/env python3
"""
RHHF Deep Debug - Análise profunda dos sinais e cálculos
"""

import sys
import os
import urllib.request
import ssl
import json
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.alta_volatilidade.rhhf_ressonador_hilbert_huang import RessonadorHilbertHuangFractal

API_BASE = "https://ttlivewebapi.fxopen.net:8443/api/v2/public/quotehistory"
SYMBOL = "EURUSD"

@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

def download_bars(period: str, count: int) -> List[Bar]:
    bars = []
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    current_ts = int(time.time() * 1000)
    total = 0
    print(f"Baixando {count} barras {period}...")
    while total < count:
        remaining = min(1000, count - total)
        url = f"{API_BASE}/{SYMBOL}/{period}/bars/ask?timestamp={current_ts}&count=-{remaining}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30, context=ctx) as response:
                data = json.loads(response.read().decode())
                batch = data.get("Bars", [])
                if not batch: break
                for b in batch:
                    ts = datetime.fromtimestamp(b["Timestamp"] / 1000, tz=timezone.utc)
                    bars.append(Bar(ts, b["Open"], b["High"], b["Low"], b["Close"], b.get("Volume", 0)))
                total += len(batch)
                oldest = min(batch, key=lambda x: x["Timestamp"])
                current_ts = oldest["Timestamp"] - 1
                time.sleep(0.1)
        except Exception as e:
            print(f"Erro: {e}")
            break
    bars.sort(key=lambda x: x.timestamp)
    return bars


def main():
    print("=" * 80)
    print("  RHHF DEEP DEBUG - ANALISE DE SINAIS E CALCULOS")
    print("=" * 80)

    bars = download_bars("H4", 1500)
    closes = np.array([b.close for b in bars])
    print(f"\nBarras: {len(bars)}")
    print(f"Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()}")

    rhhf = RessonadorHilbertHuangFractal(
        n_ensembles=20,
        noise_amplitude=0.2,
        mirror_extension=30,
        use_predictive_extension=True,
        chirp_threshold=0.0,
        fractal_threshold=1.2
    )

    # Analisar distribuição de métricas
    print("\n" + "=" * 80)
    print("  DISTRIBUICAO DE METRICAS")
    print("=" * 80)

    signals = []
    confidences = []
    conditions_list = []
    chirp_detected_list = []
    fractal_dims = []
    fractal_triggers = []
    energy_c2_list = []
    energy_c3_list = []
    price_above_cloud = []
    price_below_cloud = []
    momentums = []

    print("\nAnalisando pontos (pode demorar)...")

    for i in range(150, len(closes), 20):
        try:
            prices = closes[:i]
            result = rhhf.analyze(prices)

            signals.append(result['signal'])
            confidences.append(result['confidence'])
            conditions_list.append(result['signal_details']['conditions_met'])
            chirp_detected_list.append(result['chirp_detected'])
            fractal_dims.append(result['fractal_dimension'])
            fractal_triggers.append(result['fractal_trigger'])
            energy_c2_list.append(result['signal_details']['energy_in_c2'])
            energy_c3_list.append(result['signal_details']['energy_in_c3'])
            price_above_cloud.append(result['signal_details']['price_above_cloud'])
            price_below_cloud.append(result['signal_details']['price_below_cloud'])

            # Calcular momentum manualmente para debug
            if len(prices) >= 10:
                recent = np.mean(prices[-5:])
                older = np.mean(prices[-10:-5])
                mom = (recent - older) / older if older > 0 else 0
                momentums.append(mom)
            else:
                momentums.append(0)

        except Exception as e:
            print(f"Erro em i={i}: {e}")
            continue

    n_samples = len(signals)
    print(f"\nAmostras analisadas: {n_samples}")

    print(f"\n--- SINAIS GERADOS ---")
    print(f"  BUY (1):   {signals.count(1)} ({signals.count(1)/n_samples*100:.1f}%)")
    print(f"  SELL (-1): {signals.count(-1)} ({signals.count(-1)/n_samples*100:.1f}%)")
    print(f"  HOLD (0):  {signals.count(0)} ({signals.count(0)/n_samples*100:.1f}%)")

    print(f"\n--- CONDICOES ATENDIDAS ---")
    for c in [0, 1, 2, 3]:
        count = conditions_list.count(c)
        print(f"  {c} condições: {count} ({count/n_samples*100:.1f}%)")

    print(f"\n--- CHIRP ---")
    chirp_true = sum(chirp_detected_list)
    print(f"  Chirp detectado: {chirp_true} ({chirp_true/n_samples*100:.1f}%)")
    print(f"  Sem chirp: {n_samples - chirp_true} ({(n_samples-chirp_true)/n_samples*100:.1f}%)")

    print(f"\n--- DIMENSAO FRACTAL ---")
    print(f"  Min: {min(fractal_dims):.4f}")
    print(f"  Max: {max(fractal_dims):.4f}")
    print(f"  Mean: {np.mean(fractal_dims):.4f}")
    print(f"  Std: {np.std(fractal_dims):.4f}")

    frac_trigger_count = sum(fractal_triggers)
    print(f"  D < 1.2 (trigger): {frac_trigger_count} ({frac_trigger_count/n_samples*100:.1f}%)")

    print(f"\n--- ENERGIA NAS IMFs ---")
    e_c2 = sum(energy_c2_list)
    e_c3 = sum(energy_c3_list)
    print(f"  Energia em c2: {e_c2} ({e_c2/n_samples*100:.1f}%)")
    print(f"  Energia em c3: {e_c3} ({e_c3/n_samples*100:.1f}%)")

    print(f"\n--- POSICAO vs NUVEM (Momentum) ---")
    above = sum(price_above_cloud)
    below = sum(price_below_cloud)
    neutral = n_samples - above - below
    print(f"  Acima (momentum > 0.0003): {above} ({above/n_samples*100:.1f}%)")
    print(f"  Abaixo (momentum < -0.0003): {below} ({below/n_samples*100:.1f}%)")
    print(f"  Neutro: {neutral} ({neutral/n_samples*100:.1f}%)")

    print(f"\n  Momentum stats:")
    print(f"    Min: {min(momentums)*100:.4f}%")
    print(f"    Max: {max(momentums)*100:.4f}%")
    print(f"    Mean: {np.mean(momentums)*100:.4f}%")

    # DIAGNÓSTICO DO PROBLEMA
    print("\n" + "=" * 80)
    print("  DIAGNOSTICO DO PROBLEMA")
    print("=" * 80)

    # 1. Verificar por que poucos sinais são gerados
    print("\n1. GERACAO DE SINAIS:")

    # Quantos têm >= 2 condições?
    has_2_conds = sum(1 for c in conditions_list if c >= 2)
    print(f"   >= 2 condições: {has_2_conds} ({has_2_conds/n_samples*100:.1f}%)")

    # Desses, quantos têm momentum positivo ou negativo?
    potential_buys = 0
    potential_sells = 0
    for i in range(n_samples):
        if conditions_list[i] >= 2:
            if price_above_cloud[i]:
                potential_buys += 1
            elif price_below_cloud[i]:
                potential_sells += 1

    print(f"   >= 2 conds + above cloud: {potential_buys} ({potential_buys/n_samples*100:.1f}%)")
    print(f"   >= 2 conds + below cloud: {potential_sells} ({potential_sells/n_samples*100:.1f}%)")

    # 2. Problema do momentum threshold
    print("\n2. THRESHOLD DE MOMENTUM:")
    print(f"   Threshold atual: 0.0003 (0.03%)")

    for th in [0.0001, 0.0002, 0.0003, 0.0005, 0.001]:
        above_th = sum(1 for m in momentums if m > th)
        below_th = sum(1 for m in momentums if m < -th)
        print(f"   > {th*100:.2f}%: {above_th} acima, {below_th} abaixo")

    # 3. Análise de correlação sinal vs resultado futuro
    print("\n3. PROBLEMAS IDENTIFICADOS:")

    # Problema 1: Momentum threshold muito alto?
    mean_abs_mom = np.mean([abs(m) for m in momentums])
    print(f"   - Momentum médio absoluto: {mean_abs_mom*100:.4f}%")
    if mean_abs_mom < 0.0003:
        print(f"     [PROBLEMA] Threshold (0.03%) > Momentum médio!")
        print(f"     -> Muitos sinais potenciais são filtrados")

    # Problema 2: Chirp muito raro?
    if chirp_true / n_samples < 0.3:
        print(f"   - Chirp detectado em apenas {chirp_true/n_samples*100:.1f}%")
        print(f"     [PROBLEMA] Chirp é condição rara")

    # Problema 3: Fractal trigger muito raro?
    if frac_trigger_count / n_samples < 0.3:
        print(f"   - Fractal trigger em apenas {frac_trigger_count/n_samples*100:.1f}%")
        print(f"     [PROBLEMA] Fractal é condição rara")

    # Problema 4: Energia concentrada rara?
    energy_concentrated = sum(1 for i in range(n_samples) if energy_c2_list[i] or energy_c3_list[i])
    if energy_concentrated / n_samples < 0.5:
        print(f"   - Energia concentrada em apenas {energy_concentrated/n_samples*100:.1f}%")

    print("\n" + "=" * 80)
    print("  RECOMENDACOES")
    print("=" * 80)

    print("""
    1. REDUZIR THRESHOLD DE MOMENTUM
       - Atual: 0.0003 (0.03%)
       - Sugerido: 0.0001 (0.01%) ou usar comparação com nuvem real

    2. USAR NUVEM REAL EM VEZ DE MOMENTUM
       - Comparar preço atual vs cloud value (EEMD)
       - Não depender de momentum de 5 barras

    3. RELAXAR CONDIÇÕES
       - Aceitar 1 condição forte + confirmação
       - Ou dar peso diferente às condições

    4. MELHORAR DETECÇÃO DE CHIRP
       - Threshold de chirp pode estar muito restritivo
       - Verificar freq_trend e amp_trend separadamente
    """)

    # Mostrar alguns sinais de exemplo
    print("\n" + "=" * 80)
    print("  EXEMPLOS DE SINAIS")
    print("=" * 80)

    rhhf2 = RessonadorHilbertHuangFractal(n_ensembles=20)
    shown = 0

    for i in range(200, min(len(closes), 800), 30):
        if shown >= 10:
            break
        try:
            result = rhhf2.analyze(closes[:i])
            conds = result['signal_details']['conditions_met']

            if conds >= 1:
                sig = result['signal']
                sig_name = "BUY" if sig == 1 else "SELL" if sig == -1 else "HOLD"

                print(f"\n  Bar {i} ({bars[i].timestamp.date()}):")
                print(f"    Signal: {sig_name} | Conditions: {conds}/3")
                print(f"    Chirp: {'YES' if result['chirp_detected'] else 'NO'} (dir={result['chirp_direction']})")
                print(f"    Fractal D: {result['fractal_dimension']:.3f} (trigger={'YES' if result['fractal_trigger'] else 'NO'})")
                print(f"    Energy c2: {'YES' if result['signal_details']['energy_in_c2'] else 'NO'}")
                print(f"    Energy c3: {'YES' if result['signal_details']['energy_in_c3'] else 'NO'}")
                print(f"    Price vs Cloud: above={result['signal_details']['price_above_cloud']}, below={result['signal_details']['price_below_cloud']}")
                print(f"    Reasons: {', '.join(result['reasons'][:2])}")
                shown += 1

        except Exception as e:
            continue

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
