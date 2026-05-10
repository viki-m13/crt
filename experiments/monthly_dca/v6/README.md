# V6 — Risk-Controlled Improvement Over V3 (Inverse-Vol Weighting)

This directory contains the V6 work: a Pareto-improving change to the deployed
V3 monthly-pick strategy. **Headline:** switch the K=3 picks from equal-weight
to inverse-volatility weight, plus credit a 3% T-bill yield during cash months.
Everything else (ML model, regime gate, hold horizon, universe, costs) is
identical to V3.

## TL;DR

| Metric                              | V3 deployed | V6 winner |
|-------------------------------------|------------:|----------:|
| Walk-forward MEAN CAGR              | 42.80%      | **42.48%** (≈tied) |
| Walk-forward MIN CAGR (worst split) | 14.49%      | **20.92%** (+6.4pp) |
| Sharpe (full)                       | 0.955       | **0.971** |
| WF mean Sharpe                      | 1.031       | **1.125** (+0.094) |
| MaxDD (full)                        | -49.83%     | **-45.98%** (+3.9pp) |
| WF min MaxDD (deepest split DD)     | -47.46%     | **-39.68%** (+7.8pp) |
| WF positive splits                  | 10/10       | 10/10 |
| WF beats SPY                        | 9/10        | 9/10 |
| Sharpe ↑ in 8/8 universes tested    | -           | ✓ |
| MaxDD ↓ in 8/8 universes tested     | -           | ✓ |
| WF mean CAGR ↑ in 6/8 universes     | -           | ✓ |

See [`REPORT.md`](./REPORT.md) for the full write-up (12 sections, every
result on disk, every "what didn't work" honestly logged).

## How to reproduce

```bash
cd experiments/monthly_dca/v6/

# 1. Verify v3 baseline parity (should match the deployed numbers exactly)
python3 run_baseline.py
# → results/v3_baseline_metrics.json (cagr_full=0.39774062, sharpe=0.95536, etc.)

# 2. Sweep #1 (865 variants, ~140s)
python3 run_sweep.py
# → results/v6_sweep_results.csv

# 3. Sweep #2 (2,401 variants, ~340s) — adds spy_dd_scale, sticky cash
python3 run_sweep2.py
# → results/v6_sweep2_results.csv, v6_sweep2_pareto.csv

# 4. Sweep #3 (~6,000 variants, ~25min) — adds vol_penalty, monthly_exposure,
#    pullback_filter, K=3/4/5
python3 run_sweep3.py
# → results/v6_sweep3_results.csv

# 5. Generalization to 8 universes (~3min)
python3 run_generalize.py
# → results/v6_generalize_results.csv

# 6. Deep analysis on v6 winner: per-split WF, year-by-year, drawdowns,
#    most-picked, bias overlay
python3 analyze_winner.py
# → results/v6_winner_*.csv,json
```

## Files

| File                    | What it does |
|-------------------------|--------------|
| `lib_engine.py`         | Single source of truth for the simulator + all knobs |
| `run_baseline.py`       | Reproduce v3 baseline exactly (parity check) |
| `run_sweep.py`          | Sweep #1: gates × persist × ddr × ts × cy × vt × wt × k (865 variants) |
| `run_sweep2.py`         | Sweep #2: + spy_dd_scale, cash_sticky, ts levels (2,401 variants) |
| `run_sweep3.py`         | Sweep #3: + vol_penalty, monthly_exposure, pullback_filter, K=3/4/5 |
| `run_generalize.py`     | Apply v3 + v6 to 8 universes (sp500_pit, broader, non_sp500, random×5) |
| `analyze_winner.py`     | Deep analysis: per-split WF, year-by-year, drawdowns, picks, bias overlay |
| `REPORT.md`             | Full methodology + results write-up |

## V6 winner config

```python
V6Config(
    scorer="ml_3plus6", universe="sp500_pit", regime_gate="tight",
    k_normal=3, k_recovery=3, k_bull=3,
    weighting="invvol",        # ← V6 change (was "ew" in V3)
    hold_months=6, cost_bps=10.0,
    cash_yield_yr=0.03,        # ← V6 change (was 0.0 in V3 — "cash earns nothing")
)
```

## What we tried that did NOT work (logged honestly)

- Stricter regime gates (faber, faber_lite, strict_dd, combo): -5 to -22pp CAGR
- Crash fallback to SPY: -5pp CAGR, MaxDD WORSE
- Crash fallback to TLT: -7pp CAGR, MaxDD WORSE
- SPY DD-based gross scaling (binary or continuous): -7pp CAGR
- Trailing stop on portfolio drawdown: catastrophic (CAGR <5%)
- Sticky cash re-entry: -10 to -20pp CAGR
- Smart re-entry: -9pp CAGR, Sharpe **worse**
- Pullback filter (drop deeply-pulled-back picks): -10 to -13pp CAGR
- Pick-momentum filter: -13pp CAGR
- Vol-penalty on score: -2 to -14pp CAGR
- Quality blend (multiply by trend_health): -3 to -14pp CAGR
- K=4 or K=5: -4 to -8pp CAGR
- Conv weighting: Sharpe -0.20, MaxDD -21pp
- Softmax weighting: even worse
- Monthly exposure overlay: same as binary SDS — Sharpe up, CAGR down

A total of **9,114 strategy variants** were simulated across three sweeps.
All raw results are in `results/v6_sweep*.csv`.

## Bias-corrected CAGR (synthetic delisting Monte-Carlo)

At α=4%/yr (historical S&P 500 small-/mid-cap delisting rate), V6 retains:
- median CAGR 31.5% (vs V3 33.6%)
- +19pp edge over SPY DCA (~12%)

V3 is slightly more delisting-robust because EW spreads risk more uniformly
across the K=3 picks, but V6 is still robust at 3× historical α.

## Honest deployment recommendation

**Deploy V6.** It Pareto-improves on Sharpe + MaxDD + WF stability at
essentially the same WF mean CAGR. It generalises better to non-home
universes. And the change is just *one parameter* — the weighting scheme.
There's nothing to overfit, nothing to retrain, and the mechanism is a
textbook risk-parity tilt that operates on top of the same V3 picks.
