#!/usr/bin/env python3
"""
PHM Debug - Análise profunda dos sinais gerados
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

from strategies.alta_volatilidade.phm_projetor_holografico import ProjetorHolograficoMaldacena

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
                if not batch:
                    break
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
    print("=" * 70)
    print("PHM DEBUG - ANALISE DE SINAIS")
    print("=" * 70)

    bars = download_bars("H1", 2000)
    closes = np.array([b.close for b in bars])
    print(f"\nBarras: {len(bars)}")
    print(f"Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()}")

    phm = ProjetorHolograficoMaldacena(
        window_size=64,
        bond_dim=4,
        n_layers=3
    )

    # Analisar distribuição de métricas
    print("\n" + "=" * 70)
    print("DISTRIBUICAO DE METRICAS PHM")
    print("=" * 70)

    signals = []
    confidences = []
    entropies = []
    spike_mags = []
    magnetizations = []
    phases = {'FERROMAGNETICO': 0, 'PARAMAGNETICO': 0, 'CRITICO': 0, 'UNKNOWN': 0}
    horizons = 0

    for i in range(100, len(closes), 10):
        prices = closes[:i]
        result = phm.analyze(prices)

        signals.append(result['signal'])
        confidences.append(result['confidence'])
        entropies.append(result['entropy'])
        spike_mags.append(result['spike_magnitude'])
        magnetizations.append(result['magnetization'])
        phases[result['phase_type']] = phases.get(result['phase_type'], 0) + 1
        if result['horizon_forming']:
            horizons += 1

    print(f"\nSinais gerados:")
    print(f"  BUY (1):   {signals.count(1)} ({signals.count(1)/len(signals)*100:.1f}%)")
    print(f"  SELL (-1): {signals.count(-1)} ({signals.count(-1)/len(signals)*100:.1f}%)")
    print(f"  HOLD (0):  {signals.count(0)} ({signals.count(0)/len(signals)*100:.1f}%)")

    print(f"\nFases detectadas:")
    for p, count in phases.items():
        pct = count / len(signals) * 100 if len(signals) > 0 else 0
        print(f"  {p}: {count} ({pct:.1f}%)")

    print(f"\nHorizontes formando: {horizons} ({horizons/len(signals)*100:.1f}%)")

    print(f"\nEstatisticas:")
    print(f"  Confidence: min={min(confidences):.3f}, max={max(confidences):.3f}, mean={np.mean(confidences):.3f}")
    print(f"  Entropy:    min={min(entropies):.3f}, max={max(entropies):.3f}, mean={np.mean(entropies):.3f}")
    print(f"  Spike Mag:  min={min(spike_mags):.3f}, max={max(spike_mags):.3f}, mean={np.mean(spike_mags):.3f}")
    print(f"  Magnetization: min={min(magnetizations):.4f}, max={max(magnetizations):.4f}, mean={np.mean(magnetizations):.4f}")

    # Analisar threshold de spike
    print(f"\nAnalise de thresholds:")
    for th in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        count = sum(1 for s in spike_mags if s >= th)
        print(f"  Spike >= {th}: {count} ({count/len(spike_mags)*100:.1f}%)")

    for th in [0.3, 0.4, 0.5, 0.6]:
        count = sum(1 for c in confidences if c >= th)
        print(f"  Confidence >= {th}: {count} ({count/len(confidences)*100:.1f}%)")

    # Analisar magnetização
    print(f"\nAnalise de magnetizacao:")
    for th in [0.01, 0.02, 0.03, 0.04, 0.05]:
        bullish = sum(1 for m in magnetizations if m > th)
        bearish = sum(1 for m in magnetizations if m < -th)
        print(f"  |Mag| > {th}: Bullish={bullish}, Bearish={bearish}")

    # Mostrar primeiros 15 sinais não-HOLD
    print("\n" + "=" * 70)
    print("PRIMEIROS 15 SINAIS NAO-HOLD")
    print("=" * 70)

    phm2 = ProjetorHolograficoMaldacena(window_size=64, bond_dim=4, n_layers=3)
    signals_shown = 0

    for i in range(100, len(closes), 5):
        if signals_shown >= 15:
            break

        prices = closes[:i]
        result = phm2.analyze(prices)

        if result['signal'] != 0:
            sig = "BUY" if result['signal'] == 1 else "SELL"
            print(f"\n  {sig} @ {bars[i].timestamp}")
            print(f"    Confidence: {result['confidence']:.3f}")
            print(f"    Phase: {result['phase_type']} (Mag={result['magnetization']:.4f})")
            print(f"    Entropy: {result['entropy']:.4f}")
            print(f"    Horizon: {'YES' if result['horizon_forming'] else 'NO'} (Spike={result['spike_magnitude']:.3f})")
            print(f"    Reasons: {', '.join(result['reasons'][:2])}")
            signals_shown += 1

    print("\n" + "=" * 70)
    print("DIAGNOSTICO")
    print("=" * 70)

    # Diagnóstico
    avg_conf = np.mean(confidences)
    avg_spike = np.mean(spike_mags)
    ferro_pct = phases['FERROMAGNETICO'] / len(signals) * 100

    print(f"\n  1. Confidence média: {avg_conf:.3f}")
    if avg_conf < 0.5:
        print(f"     -> PROBLEMA: Confiança muito baixa")
    else:
        print(f"     -> OK: Confiança adequada")

    print(f"\n  2. Spike magnitude média: {avg_spike:.3f}")
    if avg_spike < 0.8:
        print(f"     -> PROBLEMA: Spikes muito fracos (threshold alto demais)")
    else:
        print(f"     -> OK: Spikes detectados")

    print(f"\n  3. Fase Ferromagnética: {ferro_pct:.1f}%")
    if ferro_pct < 30:
        print(f"     -> PROBLEMA: Muito pouca tendência detectada")
    else:
        print(f"     -> OK: Tendências sendo detectadas")

    print(f"\n  4. Horizontes: {horizons/len(signals)*100:.1f}%")
    if horizons / len(signals) < 0.05:
        print(f"     -> PROBLEMA: Poucos horizontes detectados")
    else:
        print(f"     -> OK: Horizontes sendo detectados")

    # Verificar se o problema está na combinação
    print("\n" + "=" * 70)
    print("RECOMENDACOES")
    print("=" * 70)

    if ferro_pct < 30 or avg_spike < 0.8:
        print("\n  1. Ajustar thresholds de fase Ising (magnetização)")
        print("  2. Reduzir threshold de spike para capturar mais sinais")
        print("  3. Considerar usar horizonte OU fase ferromagnética (não E)")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
