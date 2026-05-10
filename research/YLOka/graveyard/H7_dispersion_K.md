# H7 — Dispersion-conditional K — KILLED

**Hypothesis**: when XS dispersion of momentum is wide, the model's top picks are most differentiated → trust them, concentrate (small K). When dispersion is narrow (everything moves together), picks are interchangeable → diversify (large K).

**Implementation**: precomputed per-asof XS std of `mom_12_1` across the universe (cached in `data/YLOka/xs_dispersion.parquet`). At each rebalance, override K based on whether current dispersion is above/below its historical-up-to-asof percentile.

**Results** (7 variants tested):

| variant | K (narrow) | K (wide) | threshold | CAGR | Sharpe | MaxDD | Δ vs base |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | — | — | — | 40.78% | 0.953 | -49.83% | 0 |
| disp_K35_p50 | 5 | 3 | p50 | 31.53% | 0.842 | -59.09% | -9.3 |
| disp_K310_p50 | 10 | 3 | p50 | 29.19% | 0.863 | -51.29% | -11.6 |
| disp_inv_K53_p50 | 3 | 5 | p50 | 39.05% | **0.956** | -49.83% | -1.7 |
| disp_K15_p50 | 5 | 1 | p50 | 30.33% | 0.707 | -60.40% | -10.5 |
| **disp_K23_p50** | 3 | 2 | p50 | **41.77%** | 0.932 | -49.83% | **+1.0** |
| disp_K35_p33 | 5 | 3 | p33 | 31.80% | 0.842 | -59.09% | -9.0 |
| disp_K35_p67 | 5 | 3 | p67 | 31.22% | 0.862 | -59.09% | -9.6 |

**Why H7 failed (despite the +1pp surface result on `disp_K23`)**:

`disp_K23_p50` (K=2 in wide-dispersion months, K=3 in narrow) shows +1.0pp CAGR. But the diff is a SAMPLE-OF-3 artifact:

| year | base | exp_43 | diff (pp) |
|---|---:|---:|---:|
| 2004 | 27.7% | 57.2% | **+29.4** |
| 2009 | 625.5% | 699.6% | **+74.1** |
| 2016 | 35.5% | 64.0% | **+28.5** |
| 2013 | 80.1% | 56.4% | -23.7 |
| 2021 | 65.8% | 42.9% | -22.9 |
| (15 other years) | | | -10 to +5 |

- 8/22 years beat baseline.
- Median yearly diff: **0.0 pp**.
- Std of yearly diff: **19.7 pp** — the +1pp mean is one standard error from zero.
- 3 outlier years (2004, 2009, 2016) drive the entire lift.

The mechanism is the **same K-shrinkage trick as H6 Donchian**: when K=2 fires, you get accidental NVDA-style single-name concentration. avg n_picks 2.56 vs baseline 2.95.

The inverted variant (K=3 narrow / K=5 wide) gives marginally higher Sharpe (0.956 vs 0.953) at -1.7pp CAGR — diversifying in wide-dispersion is mildly Sharpe-positive but CAGR-negative.

**Don't repeat in this form**: like H6, this is portfolio concentration disguised as a regime signal. To honestly test "dispersion conditioning", you need to hold K fixed and vary something other than basket size — e.g., shift weighting toward higher-conviction picks, or pull more or less cash, or change the score blend.
