# Session 3 — Feature-based scorers — KILLED

Tested 11 hand-designed scorers using runner-pattern features (multibagger_ratio_24m, fip_score, breakout_strength_60, idio_mom_12_1, cst_score, etc.). All UNDERPERFORM v3 baseline (40.78% CAGR) by 13-36 pp.

**Root cause**: production v3's GBM (`experiments/monthly_dca/v2/ml_strategy.py:177-179`) auto-detects ALL 79 cached features per asof, including the runner-pattern ones. The model has learned their non-linear interactions and weights them optimally for the 1m/3m/6m forward-rank target. Hand-designed linear composites are a strictly weaker representation.

| variant | CAGR | Δ vs base | killing factor |
|---|---:|---:|---|
| baseline ml_3plus6 | 40.78% | 0 | — |
| ml_plus_cst (+0.15 cst rank) | 27.22% | -13.6 | already captured by GBM |
| ml_plus_runner_weak | 25.35% | -15.4 | adds noise to top picks |
| runner_gated_ml | 22.66% | -18.1 | filter dilutes ML signal |
| ml_plus_runner | 19.93% | -20.9 | additive blend hurts |
| fip_gate_ml | 19.82% | -21.0 | fip filter too restrictive |
| ml_plus_breakout | 15.96% | -24.8 | breakout flag is laggy |
| idio_plus_ml | 12.96% | -27.8 | idio is weak standalone |
| **idio_mom alone** | **8.16%** | -32.6 | rank-based, low top-K capture |
| **runner_footprint** | **4.59%** | -36.2 | composite of low-signal ranks |

**Don't repeat**: Linear composites of features the production GBM already uses cannot beat the GBM. Future feature work should either:
1. Engineer NEW features the GBM cannot derive from current inputs (e.g., from OHLC if it becomes available).
2. Train the GBM with a DIFFERENT target (e.g., asymmetric loss; probability of right-tail outcome).
