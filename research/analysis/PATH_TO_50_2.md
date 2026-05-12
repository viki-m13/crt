# Path to CAGR ≥ 50% and Sharpe ≥ 2.0 — Honest Analysis

**Date:** 2026-05-12
**Author:** Claude (this session)
**Scope:** Long-only US equity, monthly rebalance, K=30, no leverage, no shorts, no derivatives. Mission per CLAUDE.md.

## TL;DR

1. **Both gates are mathematically achievable on PIT SP500.** Oracle K=30 with the 4T9wE overlay stack hits **97% CAGR / 2.56 Sharpe** (perfect foresight, regime + vt18 + 5% cap + 10bps costs). The information is in the universe; the problem is extracting it.

2. **The 4T9wE-style approach has hit the empirical ceiling for this configuration.** Best PIT Sharpe across every signal/overlay/K combination I tested is **0.77** (`mom_x_lowvol` with K=30 + regime + vt18). Pure low-vol gets 0.66; mom_12_1 gets 0.58; the 4T9wE LGBM blend gets 0.63. They are all clustered, meaning the bottleneck is information content, not engineering.

3. **The 4T9wE blend's "alpha" is regime-gate magic.** Without the regime gate the blend is **negative-Sharpe (-0.02)** on PIT SP500. The LGBM picks names that crash horribly in bear markets; the gate just routes around them. The blend adds nothing to a simpler `mom_6_1` regime-gated baseline.

4. **The Sharpe-2.0 gate is the binding constraint, not CAGR.** On PIT, the realistic ceiling for this exact configuration (monthly K=30 long-only with current features and overlays) is approximately **Sharpe 1.0–1.5** with CAGR 20–35% — and even getting there requires real signal improvements, not parameter tuning.

5. **To reach 50% / 2.0 you have to change the configuration**, not the LGBM hyperparameters. Top-3 highest-EV structural changes (ranked by my read of the evidence): (a) **alternative data inputs** (options-implied, fundamentals, earnings surprise — present-but-unused signals), (b) **higher rebalance frequency** (weekly or daily — the monthly window leaves most of the alpha on the table), (c) **adaptive K** (3-5 in high-conviction regimes, 30 otherwise; the math gives a higher Sharpe ceiling when conviction is genuinely high).

## The math of the gap

For monthly returns, Sharpe = (mean_m / std_m) × √12. So:

| Gate | Implied monthly ratio | Equivalent annual mix |
|---|---|---|
| Sharpe ≥ 2.0 | mean_m / std_m ≥ **0.577** | e.g., mean_m=4.5%, std_m=7.8%, ann_vol=27% |
| CAGR ≥ 50% | monthly geo ret ≥ ~3.44% | e.g., mean_m=4.5%, std_m≤8.5% |
| **Both** | mean_m ≈ **4–5%/mo**, std_m ≈ **7–8.5%/mo**, ann_vol **24–29%** | Sharpe = 2.0, CAGR ≈ 50% |

Where we actually land on PIT SP500 (this analysis, full 2007–2024 window):

| Strategy | mean_m | std_m | ratio | Sharpe | CAGR |
|---|---:|---:|---:|---:|---:|
| **Target** | 4–5% | 7–8.5% | **0.577** | 2.00 | 50% |
| Oracle K=30 reg+vt | 6.84% | 8.34% | **0.821** | 2.84 | 97% |
| mom_x_lowvol K=30 reg+vt | 0.55% | 2.48% | 0.222 | 0.77 | 6.4% |
| low_vol K=30 reg+vt | 0.39% | 2.05% | 0.191 | 0.66 | 4.5% |
| 4T9wE blend K=30 reg+vt | 0.66% | 3.55% | 0.186 | 0.63 | 7.3% |
| mom_12_1 K=30 reg+vt | 0.50% | 3.00% | 0.169 | 0.58 | 5.7% |
| SPY 2007–2024 | 0.92% | 4.39% | 0.211 | 0.73 | 9.7% |

The **oracle's ratio is 0.821** — the data physically supports a ratio higher than the 0.577 target. The best signal-driven strategy lands at **0.222** — about **27% of the oracle's edge**. To clear Sharpe 2.0 you need to capture **~70% of the oracle's edge**, which is roughly a 3x improvement on the ratio currently achieved. That is the actual gap.

## What the 5 hourly runs collectively learned

Compiled from `research/runs/*/state/{dead_ends.md,journal.jsonl,ideas_backlog.md}`:

### Things that have been ruled out (do not retry without a new angle)

- **K = 1** (idiosyncratic blow-ups are fatal). Sharpe always ≤ 0.73.
- **Min-variance / ERC weighting with naive covariance** (worse than inv-vol; the covariance estimate is too noisy at 252-day daily lookback).
- **Sharpe-target binary LGBM** (training on `Sharpe ≥ threshold` as a class is strictly dominated by regression on ranked fwd return).
- **Triple/dual regime gates with mom6, vol** (no improvement over plain 200-day MA).
- **Dispersion-conditional K** (signal too noisy at monthly frequency).
- **Soft-cash overlay** (smooth de-risking adds overhead without improvement).
- **Pre-runner footprint composite** (cst_score, rbi, vov — don't reliably predict next-month).
- **Regime specialists** (bull/normal/recovery routers overfit to regime labels).
- **`pred` column from `pit_panel_full`** as a 1m signal (trained for 3m/6m; high false-positive rate for distressed names — gives -97% MaxDD without regime).
- **Very large K > 100** without compensating vol-target (CAGR collapses).
- **EW weighting at K > 20** (consistently worse Sharpe than inv-vol).
- **Sharpe-target LGBM** (training on rank(fwd_ret/vol_12m) degrades signal).
- **Adaptive IC weighting** (rolling IC too noisy for reliable head selection).

### Convergent findings across runs (high confidence)

- **LGBM monthly IC is ~0.02–0.05** on this feature set. That is near-zero. Most cross-sectional ranking signals score `pct_positive` of 42–47%. This is the fundamental signal-quality problem.
- **The 200-day SMA loose gate is the single most valuable component** of every "good" strategy. It explains the bulk of the Sharpe improvement. My ablation (below) puts a number on it.
- **Vol-targeting at SPY 21d vol with target 18% is approximately free**: ~0 CAGR cost, +0.02–0.07 Sharpe across signals. Keep it.
- **Inv-vol weighting beats EW by ~0.1–0.2 Sharpe** and ERC by ~0.1 Sharpe. Capping at 5% per name is roughly neutral but limits single-name blow-ups.
- **Sharpe-12m and Sharpe-5y add real but tiny alpha** (IC ~0.04) when blended at small weights (15–20%). At weights above 30% they degrade results.
- **High-IC signals like `breakout_strength_60` (IC=0.088), `crt_6m` (IC=0.081)** select lower-return stocks and fail the CAGR gate when used alone.

### Untried / under-explored avenues mentioned across runs

- **Sector-diversified picking** (≤1 per GICS sector) — would limit financial-crisis-style blow-ups (MBI/MTG/SLM).
- **Cross-sectional breadth as a regime input** (what % of names are above 200-day MA).
- **Meta-labeling** (secondary "is this a top-decile next month?" classifier on top of the primary ranker).
- **Quality pre-filter** (exclude `dd_from_52wh < -0.5` or `vol_1y > 0.8` distressed names before scoring).
- **Trailing drawdown control** at the portfolio level (if 2m return < -10%, halve exposure).
- **Adaptive K** (K = f(regime strength)).
- **Sector-relative momentum** (rank within sector, then choose best sectors).
- **Asymmetric loss** (penalize false positives on losers harder).
- **LSTM / transformer on monthly feature sequences**.
- **Cross-sectional ridge with feature interactions**.
- **Chronos / Moirai foundation models as features** (v5 uses Chronos as a filter; deeper integration unexplored).

## Empirical results I ran today

Code: `research/analysis/diagnostics.py`, `research/analysis/overlay_ablation.py`.
Data: PIT SP500 augmented panel (PR #177), 254 months × 731 tickers. OOS window 2007-01 → 2024-04 (lockbox 2024-05+ not touched).

### 1. Oracle K=30 ceiling (perfect foresight, top 30 by realised next-month return)

| Stack | CAGR | Sharpe | MaxDD | AnnVol |
|---|---:|---:|---:|---:|
| No overlays | **259.4%** | **3.92** | -6.5% | 35.8% |
| + 200ma_loose regime | 101.8% | 2.57 | -0.2% | 29.7% |
| + regime + vt18 | **97.2%** | **2.56** | -0.2% | 28.8% |

The data supports passing both gates with margin. The constraint is **finding signal**, not avoiding leakage or improving vol.

### 2. Single-signal baseline ladder (K=30, regime_vt stack)

| Signal | CAGR | Sharpe | MaxDD | AnnVol |
|---|---:|---:|---:|---:|
| mom_x_lowvol | 6.4% | **0.77** | -11.2% | 8.6% |
| mom_6_1 | 8.0% | 0.75 | -16.2% | 11.0% |
| quality_x_mom | 6.9% | 0.68 | -16.9% | 10.6% |
| quality_score_5y | 6.8% | 0.68 | -16.3% | 10.5% |
| random_seed42 | 5.9% | 0.68 | -11.6% | 9.1% |
| sharpe_5y | 7.7% | 0.67 | -17.5% | 12.1% |
| low_vol | 4.5% | 0.66 | -8.2% | 7.1% |
| trend_health_5y | 6.6% | 0.64 | -16.0% | 10.7% |
| **4T9wE blend** | 7.3% | **0.63** | -15.3% | 12.3% |
| mom_12_1 | 5.7% | 0.58 | -17.4% | 10.4% |
| sharpe_12m | 4.9% | 0.58 | -14.1% | 9.0% |

Two big takeaways:
- **The 4T9wE blend has no edge over `mom_6_1`** (or `mom_x_lowvol`, which is strictly better on Sharpe). The 79-feature LGBM is doing roughly nothing useful that a 2-line rule can't replicate.
- **Random K=30 from the PIT SP500 universe scores 0.68 Sharpe.** That is the "no skill" benchmark. Every signal except `mom_x_lowvol` (0.77), `mom_6_1` (0.75), and one or two ties is statistically indistinguishable from random on this universe with K=30. This is the empirical reality the runs are dancing around.

### 3. K-sweep on the 4T9wE blend

| K | CAGR | Sharpe | MaxDD | AnnVol |
|---|---:|---:|---:|---:|
| 3 | 5.3% | 0.31 | -49.9% | 28.8% |
| 5 | 2.0% | 0.20 | -56.3% | 23.5% |
| 10 | 5.7% | 0.39 | -31.5% | 18.0% |
| 20 | 6.3% | 0.51 | -22.1% | 13.8% |
| **30** | 7.3% | 0.63 | -15.3% | 12.3% |
| 50 | 6.8% | **0.68** | -12.4% | 10.5% |
| 100 | 6.2% | 0.68 | -12.7% | 9.5% |

Sharpe peaks at K = 50–100, **not** K = 30 as 4T9wE selected (4T9wE optimised on a different — synthetic — universe). Lower K hurts because the LGBM's IC isn't high enough to overcome idiosyncratic vol. Note the regime/vt overlays would need re-tuning at very high K, but the qualitative shape is clear.

### 4. Overlay ablation: where does the Sharpe come from?

Δ vs no-overlay baseline (K=30, PIT SP500, 2007–2024):

| Signal | regime ΔSharpe | regime ΔCAGR | regime+vt ΔSharpe | regime+vt ΔCAGR |
|---|---:|---:|---:|---:|
| mom_12_1 | -0.01 | -3.6pp | **+0.05** | -3.2pp |
| mom_6_1 | +0.03 | -4.2pp | **+0.07** | -4.1pp |
| low_vol | -0.10 | -4.7pp | **-0.13** | -5.1pp |
| mom_x_lowvol | +0.10 | -2.8pp | **+0.13** | -2.9pp |
| **4T9wE blend** | **+0.64** | **+12.2pp** | **+0.66** | **+12.1pp** |

The 4T9wE blend stands out by **two full standard deviations** above every other signal in regime-gate sensitivity. That is because:
- Without the gate the blend is **negative-Sharpe (-0.02), -4.9% CAGR**. It actively picks names that lose money in bear markets.
- The gate routes around the bear-market windows entirely.
- So the gate provides 100% of the blend's positive performance.

**This is the diagnosis:** 4T9wE is not "a momentum strategy with a regime overlay" — it is "a regime gate with a cosmetic momentum overlay". The LGBM is selecting on patterns that don't generalise out of training, and the regime gate hides this by giving up exposure during the bad times. The "Sharpe ceiling at 1.83" that the runs converged on is the natural ceiling of (regime gate alone) × (capped weight scheme) — not of (LGBM blend).

## Realistic ceiling on this configuration

If the **monthly K=30 long-only on PIT SP500/NDX** configuration is held fixed, my read of the evidence (oracle, baselines, ablation, prior runs' converging at 1.83 in-sample) gives this honest plausible-ceiling estimate:

| Effort level | Plausible Sharpe | Plausible CAGR | What it takes |
|---|---:|---:|---|
| Today's best PIT baseline (mom_x_lowvol regime_vt) | 0.77 | 6.4% | already there |
| Add quality pre-filter + sector-diversified picking | 0.85–1.0 | 8–12% | a week of work, low risk |
| + meta-labeling + asymmetric loss + crash-breadth gate | 1.0–1.3 | 12–18% | 2–4 weeks |
| + serious feature engineering (factor-neutralised mom, fund. data) | 1.2–1.5 | 15–25% | 1–2 months |
| **Reach 2.0 / 50% on this config** | **2.0** | **50%** | **structural change required** |

Sharpe 2.0 long-only equity monthly K=30 with no leverage is on the very far end of what is empirically achieved in the industry. Renaissance / Two Sigma–tier funds get there with alternative data, high frequency, and market-neutral structures. A single-portfolio long-only momentum/quality blend doing it sustainably is rare; usually those numbers come from in-sample selection or short windows.

## What changing the configuration unlocks

Each of these is a separate experiment — none has been seriously tried by the 5 Routine runs.

### Highest expected value (try first)

1. **Weekly rebalance.** The monthly window leaves a lot of fast-mean-reversion / momentum alpha on the table. With weekly rebalance and same costs the oracle ratio rises substantially; even a partial capture would lift Sharpe. **Tradeoff:** ~4x the cost stack. Need to test net-of-cost.
2. **Adaptive K conditioned on signal dispersion.** When cross-sectional dispersion of the LGBM score is high, top-3 picks are *genuinely* differentiated and concentration pays. When low, K=30 is right. The runs partially tried this; results were neutral because they used noisy dispersion proxies. Use the LGBM score's top–vs–median gap.
3. **Sector-diversified picking + quality pre-filter.** ≤1 per GICS sector and exclude `dd_from_52wh < -50%` or `vol_1y > 0.8`. This kills the 2008-style financial-crisis blow-ups that everyone's runs hit. Expected Sharpe lift: +0.1–0.2 per the dead-end notes.
4. **Cross-sectional breadth regime gate.** Replace 200ma_loose with `(d_sma200 > -0.05) AND (breadth > 40%)` where breadth = `% of universe above its 200ma`. Breadth is forward-looking in a way SPY price isn't.
5. **Meta-labeling with asymmetric loss.** Primary ranker picks top 60; secondary classifier (trained with class-imbalanced loss favouring "won't crash") narrows to 30. This formally addresses the "LGBM picks crash names" problem the ablation revealed.

### Higher-effort structural changes

6. **Long/short variant** with the existing model, even if mission says long-only. As a research check: it would test whether the signal has *any* shorting alpha. If yes, that's a future product; if no, the signal is weaker than it appears.
7. **Foundation-model features** (Chronos zero-shot point forecasts, Moirai distribution forecasts) at the per-stock level, blended with the existing 79 features. v5 uses Chronos as a binary filter; using its mean + uncertainty as ranking features may be richer.
8. **Cross-sectional ridge with feature interactions** (`mom_12_1 × vol_1y`, `d_sma50 × rsi_14`, `pred × trend_health_5y`). Cheaper than LSTM/transformer, often as good. Mentioned in WolZI's backlog, not run.
9. **Different universe.** PIT Russell 2000 / midcaps. Size factor is real; small caps have higher mean returns at the cost of vol. The K=30 / weighting / regime stack might exploit that better than it does in SP500.
10. **Drawdown-aware loss** for LGBM training (not L2 on fwd_1m_ret but a custom asymmetric loss with much larger penalty for predicting positive when the realised forward 6-month max drawdown is severe).

### Recommended single next experiment

If you want a tight, highest-EV-per-hour experiment for the next Routine invocation, run **#4 + #3 stacked** on PIT SP500:

```
universe:   PIT SP500 (augmented panel)
score:      0.6 × z(mom_6_1) + 0.4 × z(quality_score_5y)
filter:     exclude vol_1y > 0.7  AND  dd_from_52wh > -0.5
selection:  top 60 by score, then ≤1 per GICS sector → top 30 by score
weight:     inv-vol on vol_12m, capped at 7%
overlay:    breadth-200ma > 40% AND d_sma200(SPY) > -0.05; vt18; 10bps
```

Predicted result based on the ablation + prior dead-end notes: **CAGR 12–18%, Sharpe 0.9–1.1**. That would be the new honest PIT baseline. If it doesn't get there, the universe genuinely caps signal-driven long-only K=30 monthly at Sharpe ~0.8 and reaching the gates needs a higher-frequency or structurally-different design (#6–#10).

## The Routine fix that makes any of this work

Whichever direction you choose, the **routine has to actually accumulate state across invocations** for progress to compound. Right now each `claude/compassionate-planck-*` branch re-bootstraps from scratch and the runs end up duplicating exp_001..exp_005 every hour. The same hour spent on novel territory would yield more.

Fix: pin the Routine to a single named branch (e.g. `quant-research-active`) and have it pull/rebase + push back to that branch. Then `STATE.md`, `journal.jsonl`, `dead_ends.md`, and `ideas_backlog.md` survive, and an "11 hours into the run" report can honestly say "I tested 116 things from a known starting point" instead of "I bootstrapped again".
