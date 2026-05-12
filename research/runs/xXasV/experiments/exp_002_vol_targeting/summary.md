# Exp 002 — Volatility Targeting

**Date**: 2026-05-11
**Hypothesis**: Scale equity exposure to target annual vol (15%). Lower vol periods = fully invested. Higher vol periods = partially in cash. Same picks as v3 baseline.

## Results

| vol_target | lookback | CAGR | Sharpe | MaxDD | SubSharpes |
|---|---|---|---|---|---|
| 0.10 | 3 | 24.5% | 0.817 | -41.7% | [0.65, 1.16, 0.84] |
| 0.10 | 6 | 22.6% | 0.803 | -27.9% | [0.83, 0.93, 0.79] |
| 0.10 | 12 | 20.8% | 0.834 | -32.0% | [0.88, 1.01, 0.74] |
| 0.12 | 6 | 24.1% | 0.815 | -30.3% | [0.86, 0.95, 0.79] |
| 0.15 | 6 | 26.2% | 0.835 | -33.8% | [0.87, 0.99, 0.80] |
| 0.18 | 6 | 28.1% | 0.852 | -36.5% | [0.89, 1.01, 0.80] |
| **0.20** | **6** | **29.1%** | **0.855** | **-37.8%** | [0.90, 1.02, 0.80] |
| **0.25** | **6** | **31.2%** | **0.860** | **-41.0%** | [0.91, 1.01, 0.80] |
| **Baseline** | n/a | **40.7%** | **0.863** | **-49.5%** | [0.86, 1.00, 0.93] |

## Conclusion

**FAILED**. Vol targeting uniformly reduces CAGR without improving Sharpe.
- Best case (vt=0.25, lb=6): -9.5pp CAGR, -0.003 Sharpe
- Vol targeting scales back the early-recovery months, which are the high-return periods

## Why It Failed

The v3 crash gate ALREADY handles the worst vol regime (goes to cash in crash).
In non-crash months, the strategy's returns are uniformly positive on average.
Reducing exposure during higher-vol periods just means missing the 2009 recovery rally.

The monthly vol of the portfolio is driven by single-stock concentration risk (K=3).
Vol targeting can't reduce this without also reducing expected returns proportionally.

## Disposition

DEAD for this architecture. Log to dead_ends.md.
