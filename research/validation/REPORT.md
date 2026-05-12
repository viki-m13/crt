# PIT Validation of the 4T9wE Winning Config — Honest Report

**Date:** 2026-05-12
**Subject:** Walk-forward OOS retrain of the 4T9wE / exp_012 winner on the
canonical PIT panels.

## TL;DR

The 65.1% CAGR / 1.834 Sharpe headline from run `4T9wE` (`research/runs/4T9wE/STATE.md`)
**does not survive a PIT retrain**. On both the finalized PIT SP500 panel
(PR #177's augmented panel) and PIT NDX (on-main, 2015+ membership), the same
recipe — same K, same blend weights, same LGBM hyperparameters, same weighting,
same vol-target, same regime gate, same costs — produces single-digit CAGR
and Sharpe below 0.7.

| Panel | Window | CAGR | Sharpe | MaxDD | AnnVol | Benchmark |
|---|---|---:|---:|---:|---:|---|
| 4T9wE claim (synth universe, 1022-tk cached features) | 2007-2021 | **65.1%** | **1.834** | -13.6% | 30.1% | n/a |
| **PIT SP500 (augmented)** | 2007-2021 | **11.1%** | **0.582** | -15.0% | 21.1% | SPY 10.6% / 0.74 |
| **PIT SP500 (augmented)** | 2007-2024 | **11.4%** | **0.619** | -15.0% | 20.1% | SPY 9.7% / 0.67 |
| **PIT NDX** | 2019-2024 | **9.5%** | **0.656** | -24.1% | 15.7% | QQQ 19.2% / 0.92 |
| **PIT NDX** | 2019-2025 | **8.5%** | **0.631** | -24.1% | 14.6% | QQQ 20.7% / 1.05 |

Neither gate is met (CAGR ≥ 50%, Sharpe ≥ 2.0). On PIT SP500 the strategy
just barely beats SPY by ~0.5pp/yr at lower vol. On PIT NDX it **underperforms
QQQ by ~10pp/yr** while taking on more drawdown.

## What was tested

Spec replayed verbatim from `research/runs/4T9wE/STATE.md`:

- **Universe filter:** PIT membership at each rebalance date.
- **K=30** top-scored names, monthly rebalance.
- **Score:** `z(LGBM_pred) × 0.70 + z(sharpe_12m) × 0.20 + z(sharpe_5y) × 0.10`.
- **LGBM:** WalkForwardLGBM(train=48m, embargo=3m, min_train=24m), 200 trees,
  31 leaves, learning_rate=0.05, min_child_samples=30, subsample=0.8,
  colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1. Refit at each rebalance.
- **Weighting:** inverse `vol_12m` (or `vol_1y` fallback), iteratively capped
  at 5% per name.
- **Vol target:** `scale = min(0.18 / spy_vol_21d, 1.0)` applied to the net
  portfolio return. No leverage (cap at 1.0).
- **Regime gate (200ma_loose):** invest iff `d_sma200(SPY) > -0.05`.
- **Costs:** 5 bps × 2 round-trip = 10 bps per rebalance, scaled by exposure.

Code:
- `research/validation/sp500_pit/run_pit_sp500_validation.py`
- `research/validation/ndx_pit/run_pit_ndx_validation.py`

Outputs:
- `*/backtest_<label>.csv` — per-month return stream with picks, scale,
  regime-gate result.
- `*/summary_<label>.json` — full metrics dict + config used.

No parameter tuning was performed on the PIT panels. Walk-forward windows
slide monthly. The lockbox (2024-05+) was not touched for the primary cuts;
extended cuts go through 2024-04 / 2025-12 only for context.

## Why the collapse — best-guess attribution

The 4T9wE result was measured on the cached-feature synthetic universe at
`experiments/monthly_dca/cache/features/*.parquet`: ~1022 tickers per month,
no PIT membership filter, no survivorship correction. That universe contains
both current and historical large-flyer tickers (NVDA, MSTR, AMD, ENPH, etc.)
which were not in the S&P 500 / NDX during their multibagger phases.

When forced to pick from PIT-correct membership:

1. **The "winners" the LGBM learned to favor are unavailable.** Many of the
   highest-momentum names in the cached panel were small-caps or had been
   delisted from the SP500 by the time their momentum scored well. The PIT
   filter removes them.

2. **The K=30 inv-vol-capped portfolio with PIT membership ends up looking
   close to a low-vol large-cap tilt.** That tilt earns roughly SPY-like
   returns (~11%) at slightly lower vol — consistent with the 0.58 Sharpe
   we measured.

3. **Vol-target compresses exposure more than expected.** Average `scale` on
   PIT SP500 is ~0.78 (vs SPY vol-21d running ~17% over the window). That's
   responsible for ~3pp of the lost CAGR but not the bulk.

4. **NDX is worse than SP500 because the QQQ benchmark itself is harder to
   beat** (20.7% CAGR / 1.05 Sharpe) and the strategy's K=30 inv-vol-capped
   structure does not concentrate enough in the top 7 names that drive QQQ.

5. **The Sharpe-12m and Sharpe-5y blend weights help marginally on the
   synth universe and not at all on PIT** because trailing Sharpe rankings
   in a PIT large-cap universe are a much weaker signal — most of the
   information was already in the LGBM-learned ranking, which itself is
   weak (IC ~0.022 per the augmented panel's `sp500_pit_feature_ic.csv`).

## What this means for the Routine

The Routine's mission states "honest — no leakage, no survivorship bias".
The 4T9wE run optimized against a universe that **did contain survivorship
bias** (cached features include only current/recent stocks). The 65.1% / 1.834
metric is therefore an artifact of that bias plus implicit OOS tuning over
390 configurations.

**Recommendations for the next hourly invocation:**

1. **Switch the training universe to PIT SP500 augmented (PR #177).** The
   strategy must score and train on the same universe it will be evaluated
   on. This will likely cap Sharpe well below the synthetic-universe ceiling.
2. **Re-derive the Sharpe-2.0 ceiling honestly on PIT.** Run the same
   sweep that produced exp_001..exp_012 on the augmented panel.
3. **Persist STATE.md across Routine invocations** by pointing the cron at
   a fixed branch (see `research/STATE.md` — the runs currently re-bootstrap
   every hour, which is the root cause of the contradictory findings between
   `ugEHG` ("Sharpe 2.0 unreachable") and `4T9wE` ("65.1% / 1.834")).

## What this does NOT mean

- The PIT collapse does not invalidate the broader research framework
  (`research/runs/*/quant_research/`). The framework, walk-forward LGBM,
  regime engine, and feature library are all serviceable.
- It does not mean Sharpe ≥ 2.0 / CAGR ≥ 50% is impossible on PIT data.
  It does mean that the **specific recipe** that scored 65.1% / 1.834 on
  the synthetic universe scores ~11% / ~0.6 on PIT SP500. Future Routine
  invocations need a fundamentally different signal source — not a tweaked
  blend weight — to break through. The most promising untried angles per
  `4T9wE/state/ideas_backlog.md` are portfolio-level vol-targeting +
  trailing drawdown protection (exp_013) and adaptive K (exp_014); these
  should be tested on PIT panels, not synthetic.

## Files

```
research/validation/
├── REPORT.md                           # this file
├── sp500_pit/
│   ├── run_pit_sp500_validation.py     # harness
│   ├── backtest_primary_2007_2021.csv  # 156 monthly rows
│   ├── backtest_extended_2007_2024.csv # 184 monthly rows
│   ├── summary_primary_2007_2021.json  # full metrics + config
│   └── summary_extended_2007_2024.json
└── ndx_pit/
    ├── run_pit_ndx_validation.py
    ├── backtest_primary_2019_2024.csv  # 61 monthly rows
    ├── backtest_extended_2019_2025.csv # 81 monthly rows
    ├── summary_primary_2019_2024.json
    └── summary_extended_2019_2025.json
```
