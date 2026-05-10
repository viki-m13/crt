# H3 — Acceleration overlay — KILLED (sample-of-2 win)

**Hypothesis**: filter top-K picks by `pred_1m ≥ 0.5 · (pred_3m + pred_6m)` — only buy when short-horizon model agrees the trend is currently accelerating. Mulvaney lesson: pyramid on winners.

**Surface result**:
| | CAGR | Sharpe | MaxDD |
|---|---:|---:|---:|
| baseline | 40.78% | 0.953 | -49.83% |
| accel overlay | 42.39% | 0.964 | -49.83% |

Looks like +1.61 pp CAGR, +0.011 Sharpe — a small clean win.

**Diagnosis**:
- 4/22 years differ from baseline; 18 years identical (filter never blocked a baseline pick).
- The +1.61 pp CAGR comes entirely from 2 outlier years:
  - **2020 (COVID rebound)**: +56.5 pp diff
  - **2022 (rate-hike bear)**: +30.4 pp diff
- Counterbalanced by losses in 2012 (-17.4 pp), 2019 (-23.2 pp), 2024 (-5 pp).
- Median yearly diff: 0.0 pp.
- Time underwater (months at DD < -25%) INCREASED from 13 to 21 — accel filter doesn't reduce drawdowns.

**Why it's noise**: the accel filter changed 40/248 months (16%). With ~600 prior v3-v7 hypothesis tests + 19 in YLOka session, finding one filter that improves CAGR by ~1.6 pp via 2 lucky years is exactly what you'd expect from random search. The 1m head is itself a noisy signal (exp_09 ml_136 lost 3.8 pp by adding it as a third equal-weight target).

**Don't repeat the same form**: the simple accel filter is portfolio decoration without a robust mechanism. A real test of "model agreement" would be:
- Compute IC of pred_1m, pred_3m, pred_6m on rolling 24m windows
- Weight ensemble by recent IC (H1)
- Score = `pred_3m + pred_6m + ic_weight_1m · pred_1m`

That's a different, more principled experiment. Deferred.
