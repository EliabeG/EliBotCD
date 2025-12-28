#!/usr/bin/env python3
"""
ODMN Debug - Investigar sinais gerados
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

from strategies.alta_volatilidade.odmn_malliavin_nash import OracloDerivativosMalliavinNash

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
    print("ODMN DEBUG - INVESTIGANDO SINAIS")
    print("=" * 70)

    bars = download_bars("H1", 2000)
    closes = np.array([b.close for b in bars])
    print(f"\nBarras: {len(bars)}")
    print(f"Periodo: {bars[0].timestamp.date()} a {bars[-1].timestamp.date()}")

    odmn = OracloDerivativosMalliavinNash(
        lookback_window=100,
        fragility_threshold=2.0,
        mfg_direction_threshold=0.03,
        malliavin_paths=500,
        malliavin_steps=20,
        seed=42
    )

    # Analisar distribuicao de sinais
    print("\n" + "=" * 70)
    print("ANALISE DE SINAIS")
    print("=" * 70)

    buy_signals = 0
    sell_signals = 0
    hold_signals = 0
    warmup_count = 0

    confidences = []
    fragilities = []
    mfg_directions = []

    print("\nPrimeiros 20 sinais nao-warmup:")
    signals_shown = 0

    for i in range(150, len(closes), 10):
        prices = closes[:i]
        result = odmn.analyze(prices)

        if result.get('is_warmup', True):
            warmup_count += 1
            continue

        signal = result['signal']
        conf = result['confidence']
        frag = result['fragility_percentile']
        mfg = result['mfg_direction']

        confidences.append(conf)
        fragilities.append(frag)
        mfg_directions.append(mfg)

        if signal == 1:
            buy_signals += 1
            if signals_shown < 20:
                print(f"  BUY  | conf={conf:.3f} | frag={frag:.3f} | mfg={mfg:+.4f}")
                signals_shown += 1
        elif signal == -1:
            sell_signals += 1
            if signals_shown < 20:
                print(f"  SELL | conf={conf:.3f} | frag={frag:.3f} | mfg={mfg:+.4f}")
                signals_shown += 1
        else:
            hold_signals += 1

    print("\n" + "-" * 70)
    print(f"Resumo:")
    print(f"  Warmup: {warmup_count}")
    print(f"  BUY:    {buy_signals}")
    print(f"  SELL:   {sell_signals}")
    print(f"  HOLD:   {hold_signals}")

    if confidences:
        print(f"\nEstatisticas:")
        print(f"  Confidence: min={min(confidences):.3f}, max={max(confidences):.3f}, mean={np.mean(confidences):.3f}")
        print(f"  Fragility:  min={min(fragilities):.3f}, max={max(fragilities):.3f}, mean={np.mean(fragilities):.3f}")
        print(f"  MFG Dir:    min={min(mfg_directions):.4f}, max={max(mfg_directions):.4f}, mean={np.mean(mfg_directions):.4f}")

        # Contar quantos passam nos thresholds
        print(f"\nThreshold analysis:")
        for conf_th in [0.50, 0.55, 0.60, 0.65]:
            count = sum(1 for c in confidences if c >= conf_th)
            print(f"  Confidence >= {conf_th}: {count}/{len(confidences)} ({count/len(confidences)*100:.1f}%)")

        for frag_th in [0.60, 0.70, 0.80, 0.90]:
            count = sum(1 for f in fragilities if f >= frag_th)
            print(f"  Fragility >= {frag_th}: {count}/{len(fragilities)} ({count/len(fragilities)*100:.1f}%)")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
