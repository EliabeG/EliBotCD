# Medium Volatility Strategies Optimization Report
## Real Spread Validation with Walk-Forward and Monte Carlo

**Date:** 2025-12-29
**Data Period:** 2025-01-10 to 2025-12-29 (352 days)
**Symbol:** EURUSD H1
**Validation:** 70% Training / 30% OOS Split

---

## Summary

| Strategy | Status | WF Edge | OOS Edge | OOS Trades | OOS PF | MC Score |
|----------|--------|---------|----------|------------|--------|----------|
| **FSIGE** | ✅ APPROVED | +13.8% | +20.0% | 5 | 2.07 | 73% |
| **HJBNES** | ✅ APPROVED | +5.6% | +1.9% | 43 | 1.02 | 70% |
| BPHS | ❌ FAILED | +10.2% | -3.9% | 20 | 0.81 | 100% |
| FKQPIP | ❌ FAILED | +17.7% | -1.0% | 32 | - | 90% |
| H2PLO | ❌ FAILED | +22.3% | -6.7% | 34 | - | 100% |
| KDVSH | ❌ FAILED | +13.7% | -0.4% | 29 | - | 100% |
| RCTF | ❌ FAILED | - | - | - | - | - |
| MPSDEO | ❌ FAILED | +20.5% | -12.6% | 7 | - | 97% |
| LSQPC | ❌ FAILED | - | - | - | - | - |
| MVGKSD | ❌ FAILED | - | - | - | - | - |

---

## Approved Strategies Details

### 1. FSIGE (Fisher-Shannon Information Gravity Engine)
**Concept:** Uses information geometry and thermodynamic concepts to detect market regime changes.

**Optimized Parameters:**
- KDE Window: 80
- Entropy Horizon: 8
- Tension Threshold: 0.35
- Min Confidence: 0.45

**Risk Management:**
- Stop Loss: 30 pips
- Take Profit: 45 pips
- Cooldown: 40 bars

**Validation Results:**
- Walk-Forward: 26 trades, 53.8% WR, +13.8% edge
- OOS: 5 trades, 60.0% WR, +20.0% edge, PF 2.07
- Monte Carlo: 73%

---

### 2. HJBNES (Hamilton-Jacobi-Bellman Nonlinear Expectation Solver)
**Concept:** Dynamic programming approach using risk-adjusted expected returns.

**Optimized Parameters:**
- Horizon: 15 bars
- Discount Factor: 0.9
- Risk Aversion: 1.0
- Min Confidence: 0.40

**Risk Management:**
- Stop Loss: 30 pips
- Take Profit: 45 pips
- Cooldown: 40 bars

**Validation Results:**
- Walk-Forward: 90 trades, 45.6% WR, +5.6% edge
- OOS: 43 trades, 41.9% WR, +1.9% edge, PF 1.02
- Monte Carlo: 70%

---

## Approval Criteria

All strategies were evaluated against 6 criteria:
1. Walk-Forward Edge > 1%
2. OOS Edge > 0%
3. OOS Profit Factor > 1.0
4. OOS Trades >= 5
5. OOS Max Drawdown < 100 pips
6. Monte Carlo Score >= 60%

**Minimum for approval: 4/6 criteria passed**

---

## Methodology

### Real Spread Calculation
- Downloaded separate ASK and BID data from API
- Real spread = (Ask_close - Bid_close) / PIP per bar
- Entry: BUY at ASK open, SELL at BID open
- Exit: BUY at BID, SELL at ASK
- Added 0.5 pips slippage to each trade

### Walk-Forward Validation
- 5 folds with anchored training
- Required 3/5 folds with positive edge
- Consistency check (60%+ folds profitable)
- Buy/Sell ratio balance (20-80%)

### Monte Carlo Validation
- 30 shuffles of signal directions
- Original PnL must beat 60%+ of random shuffles
- Confirms edge is not due to chance

---

## Files Uploaded to GitHub

1. `strategies/media_volatilidade/fsige_robust_optimized.json`
2. `strategies/media_volatilidade/hjbnes_robust_optimized.json`

---

## Notes for Live Trading

1. **FSIGE** showed excellent OOS performance (+20% edge) but with only 5 trades. Monitor closely for more data.

2. **HJBNES** showed consistent performance with 43 OOS trades and marginal positive edge (+1.9%). Lower risk allocation recommended.

3. Both strategies use identical risk parameters (SL=30, TP=45), suggesting this risk profile is suitable for current market conditions.

4. Failed strategies showed strong in-sample performance but couldn't maintain edge out-of-sample, indicating overfitting to historical patterns.

---

*Report generated automatically by walk-forward optimization system*
