# Monthly Stock-Pick Strategy v4 — Comprehensive Search Report

**Run date.** 2026-05-10.
**Goal.** Find a stock-selection strategy that materially exceeds the deployed
v3 PIT S&P 500 strategy (42.80% WF mean OOS CAGR) without overfitting,
without survivorship bias, and that generalises across regimes/universes.

**Honest headline.** After an extensive systematic search across:

- 4,000+ variant configurations (different scorers, K, holds, weightings,
  regime gates, stop-loss / take-profit, score thresholds, capacity caps)
- A fresh LightGBM ensemble (5-seed, 10y rolling, 3m+6m horizon)
- Multi-factor blends (ML × quality × momentum × idio-mom × breakout etc)
- Various ensemble strategies (filter-based, rank-blended, score-weighted)

**we did not find a robust, honestly-tested improvement over v3 on PIT S&P 500.**
The deployed v3 strategy (`ml_3plus6 K=3 EW tight h=6`) at 42.80% WF mean OOS
CAGR is at or near the realistic ceiling for a stock-picking strategy
restricted to PIT S&P 500 large caps.

We did identify a marginal robustness improvement: switching from equal-weight
to inverse-volatility weighting with a 40% per-pick cap raises WF min OOS CAGR
from 14.49% to 22.00% (a 7.5pp lift on the worst split) at the cost of 1.1pp
on full-window CAGR. This is **not** a CAGR improvement and we do not
recommend changing the production strategy on its basis.

---

## What we tried

### 1. Daily-resolution take-profit / stop-loss

We hypothesised that exiting individual positions on intra-month spikes
(+50% take-profit) or drawdowns (-30% stop-loss) would add real alpha.

A first version of the simulator (`simulator_v4.py`, original) appeared to
show **WF mean OOS CAGR of 64.56% with TP=+50%**.  After diagnostic
inspection we identified a **simulator bug**: when TP fired during month T,
the simulator capped that month's reported return at +50% (entry-relative),
but in subsequent months it continued to apply actual close-to-close monthly
returns from the post-TP price level, double-counting both the locked-in
gain and the continuation. This created a fictitious +21pp CAGR boost.

We fixed the simulator (corrected: post-TP cash is held flat for the rest of
the hold period). Under the corrected semantics:

| Variant            | Full CAGR | WF mean | WF min  | Beats SPY |
|--------------------|----------:|--------:|--------:|----------:|
| baseline_v3 (no TP)|    39.77% |  42.80% |  14.49% |       9/10|
| TP=+30% honest     |    24.32% |  24.91% |  10.91% |       8/10|
| TP=+50% honest     |    31.61% |  34.49% |   7.82% |       8/10|
| TP=+75% honest     |    37.01% |  40.61% |  10.89% |       8/10|
| TP=+100% honest    |    38.11% |  41.70% |  12.63% |       9/10|
| TP=+150% honest    |    39.15% |  42.14% |  15.94% |       9/10|

Honest take-profit *consistently underperforms baseline*.  Stocks that hit
+50% above entry tend to continue higher; locking in there cuts continuation
upside.  Loose TPs (+150%) approach baseline asymptotically.

Honest stop-loss similarly underperforms baseline. Tight SLs (-15% to -25%)
hurt by 6-23 pp; loose SLs (-30% to -40%) close to baseline.

**Conclusion:** position-level TP/SL on monthly granularity does not add
alpha for this strategy. The original headline win was a simulator artifact.

### 2. Fresh ML ensemble (v4 ML)

We trained a new 5-seed LightGBM ensemble on the broader 1,833-ticker
panel (2003-2026), with annual retrain, 7-month embargo, 10-year rolling
training window, predicting cross-sectional rank of 3m and 6m forward
returns separately. Saved to `cache/v2/sp500_pit/ml_preds_v4.parquet`.

Tested as a sole scorer:

| Variant                    | Full CAGR | WF mean | Beats SPY |
|----------------------------|----------:|--------:|----------:|
| v4_only k=3 h=6 EW         |    10.18% |  10.02% |       3/10|
| v4_only k=2 h=6 invvol     |    12.00% |  12.86% |       7/10|
| v4_6m k=2 h=6 EW           |    20.23% |  17.15% |       6/10|
| v4_3m k=3 h=6 EW           |     8.87% |   6.85% |       2/10|

Tested as ensembles with v2:

| Variant                    | Full CAGR | WF mean | Beats SPY |
|----------------------------|----------:|--------:|----------:|
| 50/50 ranks (stack_v2_v4)  |     8.57% |   8.02% |       3/10|
| 80% v2 + 20% v4 ranks      |    10.72% |  11.48% |       6/10|
| 70% v2 + 30% v4 ranks      |    12.33% |  13.35% |       6/10|
| v2 score, v4 rank ≥ 50% filter |  8.79% |  20.16% |       7/10|

The fresh v4 model is materially weaker than the existing v2 GBM and adding
it as an ensemble or filter degrades performance. Likely causes: rolling
10-year training window loses high-signal pre-2010 data, LightGBM
hyperparameters not tuned for this specific cross-section, single-horizon
training mis-aligned with v2's joint multi-horizon training.

**Conclusion:** v4 ML, as trained, does not improve on v2.  Future work
could re-train v4 with full-history training and more careful hyperparameter
tuning.

### 3. Factor blends and alternative scorers

| Scorer (k=3 h=6 EW)             | Full CAGR | WF mean | Beats SPY |
|---------------------------------|----------:|--------:|----------:|
| ml_3plus6_baseline (v3)         |    39.77% |  42.80% |       9/10|
| ml_136_blend (1m+3m+6m)         |    35.80% |  37.77% |       9/10|
| ml_36_qmom (50% ml + 20% mom + 15% sharpe + 15% trend) | 19.93% | 20.34% | 8/10 |
| ml_36_idio_winner (50% ml + 25% idio + 25% mom)         | 14.64% | 15.98% | 4/10 |
| ml_36_strict_winner (ml - 0.3 if below 200dma)          | 21.93% | 22.81% | 8/10 |
| ml_36_breakout (70% ml + 30% breakout_strength)         | 19.06% | 20.51% | 8/10 |
| ml_36_low_dd (70% ml + 30% (1-max_dd_5y))               | 22.31% | 22.95% | 8/10 |
| ml_36_dispersion_aware (ml / vol_1y)                    | 12.40% | 13.44% | 5/10 |

All multi-factor blends underperform the pure ml_3plus6 score. The v2 GBM
already captures the relevant momentum + quality + reversal signals; adding
hand-picked factors hurts via tilt away from the ML's optimal weights.

### 4. Wider K × hold sweep

| K | h | CAGR     | WF mean  | WF min  | Beats SPY | Sharpe | MaxDD |
|---|---|---------:|---------:|--------:|----------:|-------:|------:|
| **3** | **6** | **39.77%** | **42.80%** | 14.49% | 9/10 | **0.96** | **-49.8%** |
| 3 | 7  | 40.08%   | 40.75%   | 11.22%  | 9/10      | 0.97   | -64.9% |
| 3 | 9  | 33.81%   | 37.18%   | 15.03%  | 10/10     | 0.85   | -76.7% |
| 3 | 12 | 35.57%   | 38.91%   | 10.62%  | 9/10      | 0.99   | -58.8% |
| 2 | 6  | 34.50%   | 37.43%   | 13.21%  | 9/10      | 0.79   | -69.1% |
| 2 | 7  | **45.48%** | **52.21%** | 6.70% | 9/10  | 0.94 | **-81.2%** |
| 2 | 8  | 14.53%   | 26.96%   | 6.08%   | 7/10      | 0.50   | -97.0% |
| 2 | 9  | 27.63%   | 29.34%   | 0.19%   | 7/10      | 0.69   | -85.8% |
| 2 |10  | 5.57%    | 16.71%   | -3.71%  | 4/10      | 0.35   | -97.7% |
| 2 |11  | 30.27%   | 29.54%   | 10.24%  | 7/10      | 0.76   | -59.4% |
| 2 |12  | 32.60%   | 40.38%   | 3.85%   | 8/10      | 0.86   | -72.4% |

K=2 h=7 hits an apparent peak at 45.48% CAGR / 52.21% WF mean — but
the surrounding cells (K=2 h=8: 14.53% CAGR; K=2 h=9: 27.63%; K=2 h=10:
5.57%) collapse violently.  This is a clear **overfit spike** to the
specific 10-split window, not a robust improvement. The MaxDD of -81% on
K=2 h=7 confirms it: the strategy bets concentrated on 2 names for 7
months at a time and gets lucky on the historical NVDA-class winners.

**Conclusion:** K=3 h=6 sits on a robust plateau. K=2 is overconcentrated;
h≠6 introduces irregular cycle artifacts.

### 5. Weighting and capacity caps

| Variant                     | CAGR     | WF mean  | WF min  | Sharpe | MaxDD |
|-----------------------------|---------:|---------:|--------:|-------:|------:|
| v3 baseline (EW)            |   39.77% |   42.80% |  14.49% |   0.96 | -49.8%|
| invvol no cap               |   38.14% |   42.41% |  20.82% |   0.97 | -46.4%|
| **invvol cap=0.40**         |   38.66% |   42.84% |**22.00%**|  0.98 | -48.4%|
| invvol cap=0.50             |   38.10% |   42.35% |  21.28% |   0.97 | -46.9%|
| invvol cap=0.60             |   38.16% |   42.43% |  20.92% |   0.97 | -46.4%|
| EW with conv weighting      |   30+%   |  ~30%    |    -    |    -   |    -  |
| EW with softmax weighting   |   25+%   |  ~25%    |    -    |    -   |    -  |

Inverse-volatility weighting with cap=0.40 produces **the same WF mean
CAGR (42.84% vs 42.80%) with materially better robustness**: WF min
22.00% (+7.51pp), Sharpe 0.98 (+0.02), MaxDD -48.4% (-1.4pp).  This is the
only honest improvement we found.

We do not recommend changing production on its basis: it sacrifices ~1pp
of full-window CAGR for the robustness lift, and the WF mean is essentially
unchanged.  Documented here for future reference.

### 6. Regime gate variants

| Variant                       | WF mean  | n_cash months |
|-------------------------------|---------:|--------------:|
| v3 tight (deployed)           |   42.80% |             4 |
| v4 (added vol-spike + DD-recovery) | 42.80% |          4 |
| breadth_tight (cross-section breadth) | tested in `sweep_simulator_knobs.py` | – |
| multi (SPY+breadth+DD)        | tested in same sweep                 | – |

Adding vol-spike or breadth signals to the regime gate did not improve
performance. The v3 'tight' gate (4 cash months in 22 years) is already
optimal — cash discipline at 21d ≤ -8% catches the GFC and COVID crashes
without false-positives in the long bull markets.

---

## Generalisation

The deployed v3 strategy generalises across universes (already documented
in `SP500_PIT_V3_REPORT.md`):

| Universe                 | n_pool | WF mean | WF min | Beats SPY |
|--------------------------|-------:|--------:|-------:|----------:|
| PIT S&P 500              |   587  |   42.8% |  14.5% |       9/10|
| Broader 1,833-ticker     | 1,811  |   51.8% |  13.7% |      10/10|
| Non-S&P 500 PIT (Russ−SP)| 1,579  |   51.0% |  21.4% |      10/10|
| Random 500 subsets (avg) |   497  |   56.4% |   8.4% |       9/10|

The strategy delivers higher CAGR on broader universes (51-56% WF mean) but
the user requirement is **PIT S&P 500 only**.  On the constrained universe,
42.8% WF mean OOS CAGR is the realistic ceiling.

---

## Why higher CAGR is not honestly available on PIT S&P 500

1. **Lower cross-sectional dispersion.**  S&P 500 large caps have lower
   return dispersion than small/mid caps.  The information ratio of any
   stock-picking signal is mechanically capped by the dispersion of the
   universe.

2. **Mega-cap concentration.**  NVDA dominates the historical picks (45%
   of months) due to its 2017-2024 run.  Forward-looking performance
   depends on whether new mega-cap winners emerge — by definition we
   cannot anticipate them.

3. **The v2 GBM is well-tuned.**  Trained on 1m/3m/6m horizons with
   annual retrain and 7-month embargo on the broader 1,833 panel, then
   restricted at predict time to S&P 500 PIT members, it captures
   essentially all of the available cross-sectional alpha.

4. **Tradeoff between CAGR and robustness is real.**  More aggressive
   K=2 or h≠6 configurations can hit higher CAGR in specific historical
   windows but at the cost of higher MaxDD and WF min — these are
   overfit to the past.

To genuinely target higher CAGR, one of the following changes is needed:

- **Expand universe** (already documented: broader 1,833 → 51.8% WF mean)
- **Add new asset classes** (crypto, futures, options)
- **Use leverage** (e.g. 1.5× or 2×, raises both CAGR and risk linearly)
- **Add information beyond price** (fundamentals, news, alternative data)
- **Allow more risk per name** (e.g. K=1, K=2 — possible but raises MaxDD)

---

## Files in this directory

- `simulator_v4.py` — extended simulator with daily-resolution TP/SL.
  Used during the bug-discovery phase; corrected to honest semantics.
- `v4_engine.py` — pre-existing engine (monthly TP/SL, breadth gates).
- `train_v4_ml.py` — fresh LightGBM 5-seed walk-forward training.
- `build_pit_panel.py` — rebuilds the PIT S&P 500 cross-section panel.
- `run_v4_sweep.py` — initial sweep including buggy TP variants.
- `run_v4_stage_a2.py` — deeper sweep (also based on buggy TP).
- `run_v4_honest_sweep.py` — honest (post-fix) TP/SL sweep.
- `run_v4_blend_sweep.py` — multi-factor blend sweep.
- `run_v4_advanced.py` — regime-conditional K and weighting tests.
- `validate_v4_winner.py` — full validation harness (WF, generalisation,
  bias overlay, drawdowns).
- `sweep_simulator_knobs.py` — pre-existing wide knob sweep.
- `sanity_check.py` — verifies the v4 engine reproduces v3 baseline exactly.
- `REPORT.md` — this file.

## Result files (in `cache/v2/sp500_pit/`)

- `v4_sweep_results.csv` — initial sweep (buggy TP — flagged)
- `v4_stage_a2_results.csv` — deeper buggy TP sweep
- `v4_honest_sweep_results.csv` — honest sweep (use this for conclusions)
- `v4_blend_sweep_results.csv` — blend sweep
- `v4_advanced_sweep_results.csv` — advanced sweep (cap, K, h variants)
- `v4_simulator_sweep.csv` — pre-existing knob sweep (1,920 variants)
- `ml_preds_v4.parquet` — fresh v4 LightGBM predictions

---

## Bottom line

**No production change recommended.**

The deployed v3 PIT S&P 500 strategy (`ml_3plus6 K=3 EW tight h=6`) at
42.80% WF mean OOS CAGR remains the best honest answer.  Extensive search
confirms it is at the local optimum of the strategy space we explored.

The honest improvement available is invvol weighting with cap=0.40, which
trades 1pp of full-window CAGR for a 7.5pp lift in WF min CAGR (worst-case
robustness).  This is a robustness/return trade-off, not a CAGR improvement.

To target higher CAGR, the user must accept either a broader universe
(51.8% WF mean on 1,833-ticker panel) or different asset classes.
