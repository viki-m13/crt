# 05 — Walk-forward parameter selection (the critical fairness test)

This is the single most decisive experiment in the review.

## Why this matters

Each agent claims an "always-on" parameter choice:
- A: always invvol
- B: always kb=2 in bull regime
- C: always q=0.4 chronos filter

These choices were selected by the agent **on the same dataset they then
evaluated on**. That's a multiple-comparisons / sweep-overfit problem:
the +Xpp lift they report may be partially or entirely an artifact of
having chosen the best cell from a sweep.

The fair test is: **let a WF selector also choose the parameter, using
only training data**. For each test split:

1. Compute candidate parameter's train CAGR (on all months before the split's start).
2. Pick the parameter that won train.
3. Measure the chosen parameter's CAGR on the test split.

If the agent's "always-on" parameter is robust, the WF selector picks it on
most/all splits and the WF-honest lift matches the published claim. If
the parameter was sweep-overfit, the selector picks something else and the
lift collapses.

## Results

### Option A: WF weighting selection (`run_wf_A_selection.py`)

Candidate variants: {`v3_ew`, `ew_cy3` (just add cash yield), `invvol_cy3`}.

**Selector trained on train-CAGR:**

| Split    | Best on train      | Test CAGR | v3 test CAGR | Lift  |
|----------|--------------------|----------:|-------------:|------:|
| A1       | `ew_cy3`           | 22.92%    | 22.88%       | +0.04 |
| A2       | `ew_cy3`           | 35.42%    | 35.37%       | +0.05 |
| A3       | `invvol_cy3`       | 33.77%    | 38.95%       | **−5.18** |
| R1_GFC   | `v3_ew`            | 108.79%   | 108.79%      | 0.00  |
| R2       | `ew_cy3`           | 43.13%    | 43.13%       | 0.00  |
| R3       | `ew_cy3`           | 14.49%    | 14.49%       | 0.00  |
| R4       | `ew_cy3`           | 19.70%    | 19.60%       | +0.10 |
| R5_COVID | `invvol_cy3`       | 53.70%    | 62.20%       | **−8.49** |
| R6_AI    | `ew_cy3`           | 40.85%    | 40.85%       | 0.00  |
| STRICT   | `ew_cy3`           | 41.75%    | 41.75%       | 0.00  |

**Mean test CAGR: 41.45% vs v3 42.80% → −1.35pp** | **beats v3 in 3/10 splits**

The selector picks `ew` (essentially v3) in 7 of 10 splits. When it picks
`invvol`, the variant *loses* on test in 2 of those 3 splits (A3 −5.18pp,
R5_COVID −8.49pp). Sharpe-objective version of the same selector gives
the same picture: −0.019 mean Sharpe lift, beats v3 in 3/10.

**Verdict for A**: the "always-on invvol" claim does not survive a WF
parameter selection. The agent's reported full-period Sharpe and MaxDD
lift is real, but it's a 22-year aggregation effect — *within* any
specific window, invvol is usually not the better choice.

### Option B: WF kb selection (`run_wf_selection.py`)

Candidate kb values: {1, 2, 3, 4, 5}, with invvol weighting + 3% cash yield.

| Split    | Best kb on train | Test CAGR | v3 test CAGR | Lift   |
|----------|-----------------:|----------:|-------------:|-------:|
| A1       | 1                | 23.52%    | 22.88%       | +0.64  |
| A2       | 1                | 41.66%    | 35.37%       | +6.29  |
| A3       | 1                | 28.03%    | 38.95%       | **−10.92** |
| R1_GFC   | 1                | 106.09%   | 108.79%      | −2.70  |
| R2       | 1                | 33.84%    | 43.13%       | **−9.29** |
| R3       | 1                | 33.86%    | 14.49%       | **+19.37** |
| R4       | 1                | 10.92%    | 19.60%       | **−8.68** |
| R5_COVID | 2                | 73.94%    | 62.20%       | +11.75 |
| R6_AI    | 2                | 37.10%    | 40.85%       | −3.74  |
| STRICT   | 1                | 26.27%    | 41.75%       | **−15.48** |

**Mean test CAGR: 41.52% vs v3 42.80% → −1.28pp** | **beats v3 in 4/10 splits**

The selector picks kb=**1** in 8 of 10 splits — *not* kb=2 as the agent
claimed. The agent's kb=2 is selected only on 2 of 10 train windows
(R5_COVID, R6_AI). When kb=1 wins train and is applied to test, it loses
badly 6 times (A3, R2, R4, STRICT alone cost ~−44pp combined).

**Verdict for B**: the published +3.35pp WF mean lift is sweep-overfit.
The WF-honest lift is **−1.28pp**, and the parameter the selector actually
prefers (kb=1) is fragile.

### Option C: WF q selection (`run_wf_selection.py`)

Candidate q values: {0.0 (no filter), 0.2, 0.3, 0.4, 0.5, 0.6}.

| Split    | Best q on train | Test CAGR | v3 test CAGR | Lift   |
|----------|----------------:|----------:|-------------:|-------:|
| A1       | 0.4             | 25.77%    | 22.88%       | +2.89  |
| A2       | 0.4             | 38.45%    | 35.37%       | +3.08  |
| A3       | 0.4             | 43.02%    | 38.95%       | +4.07  |
| R1_GFC   | 0.4             | 108.79%   | 108.79%      | 0.00   |
| R2       | 0.4             | 48.54%    | 43.13%       | +5.41  |
| R3       | 0.4             | 17.01%    | 14.49%       | +2.51  |
| R4       | 0.4             | 19.97%    | 19.60%       | +0.37  |
| R5_COVID | 0.4             | 63.41%    | 62.20%       | +1.21  |
| R6_AI    | 0.4             | 49.00%    | 40.85%       | **+8.16** |
| STRICT   | 0.4             | 44.64%    | 41.75%       | +2.89  |

**Mean test CAGR: 45.86% vs v3 42.80% → +3.06pp** | **beats v3 in 9/10 splits**

The WF selector picks **q=0.4 in every single training window**. The
"always-on q=0.4" claim is therefore robust to OOS parameter selection —
the same q wins on a wide range of training horizons (8 years up to 22
years of history). The +3.06pp WF-honest lift matches the published
+3.06pp.

**Verdict for C**: the only candidate of the three that survives the
WF parameter-selection test. The agent's q=0.4 choice is genuinely
robust to in-sample selection bias.

## Cross-summary

| Strategy | "Always-on" parameter | WF-honest lift CAGR | Selector picks agent's value? |
|---|---|---:|---|
| A (invvol+cy)    | invvol         | **−1.35pp** | No — picks ew in 7/10 splits |
| B (kb=2+invvol)  | kb=2           | **−1.28pp** | No — picks kb=1 in 8/10 splits |
| C (chr filter)   | q=0.4          | **+3.06pp** | **Yes — every single split** |

## Important nuance for A

A's full-period Sharpe (0.971) and MaxDD (−45.98%) are real improvements
over v3 (0.955, −49.83%). These come from aggregating 22 years of
returns where invvol's lower-vol-pick tilt smooths the basket's path. But
*within any given test window*, invvol can lose CAGR substantially
(A3 −5pp, R5_COVID −8pp). The WF selector — which only has access to
prior data — frequently picks the wrong horse.

For long-horizon deployment (5+ years), A's invvol mechanism is
defensible. For short windows the choice is closer to a coin flip.

## Why C survives where B doesn't

B's parameter (kb=2) is a regime-conditional concentration tweak. The
regime gate ("bull = SPY 12m mom > 10% AND > 200dma") fires on only ~15%
of months, so the "kb=2 effect" is sample-of-39-months — tiny. The
+3pp WF mean lift is driven by a handful of bull months happening to
favor 2-name concentration.

C's parameter (q=0.4) is a cross-sectional filter applied to *every*
month. It removes the bottom 40% of stocks by Chronos's forward-return
forecast at every rebalance. With 280 asofs × ~500 tickers × 6m hold,
this is a much larger effective sample than B's regime-conditional cell.
The signal is genuinely cross-sectional, not a regime artifact.

## What this means for shipping

- Do not ship B as a "kb=2 bull-regime upgrade" — the kb=2 is overfit.
- Do not ship A on the strength of WF CAGR — full-period CAGR is −1.6pp.
- A is acceptable if you value full-period Sharpe + MaxDD improvement and
  understand the within-window CAGR cost.
- C is the only candidate with a WF-honest CAGR lift. Ship as the
  primary alpha-add.
