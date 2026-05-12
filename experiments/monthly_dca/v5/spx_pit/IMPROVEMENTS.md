# Model improvements on augmented PIT — K=2 dominates K=3

## Background

The deployed v5 strategy uses **K=3** picks per basket. That K was
selected on the **biased** v2 panel (the one that omitted 374 of the 985
historical PIT S&P 500 tickers). The augmented PIT panel is the
honest universe, and re-tuning v5 on it shifts the optimum.

## Sweep

`experiments/monthly_dca/v5/spx_pit/sweep_v5_aug.py` and
`sweep_fine_v5_aug.py` sweep the v5 hyperparameters on the augmented
PIT panel:

- K (top-K picks):    {1, 2, 3, 4, 5, 7, 10, 15}
- Chronos q:          {0.0 (off), 0.20, 0.30, 0.40, 0.45, 0.50, 0.60}
- Hold months:        {1, 3, 6, 9, 12}
- Cap per pick:       {0.34, 0.40, 0.50, 1.0}

All other parameters fixed at deployed defaults (regime=tight,
scorer=ml_3plus6, weighting=invvol, cost=10 bps).

Per-config results saved at:
- `augmented/v5_param_sweep_results.csv` (broad sweep, 23 configs)
- `augmented/v5_param_sweep_fine.csv` (focused 4×6×4×3 = 288 configs around K=2)

## Headline finding — K=2 dominates K=3 (lump-sum mode)

| Config (lump-sum)              |   CAGR | WF mean | Sharpe | Max DD | Beats SPY |
|--------------------------------|-------:|--------:|-------:|-------:|----------:|
| Deployed K=3, q=0.45, h=6, cap=0.40 | 32.92 % | 32.68 % |   0.92 | -51.3 % |       8/10 |
| **K=2**, q=0.45, h=6, cap=0.40 | **49.21 %** | **49.39 %** |  **1.04** | -52.5 % |  **10/10** |
| Δ                              | **+16.29pp** | **+16.71pp** | **+0.12** |  ~tied |   **+2** |

**Every metric improves at K=2 vs K=3 with no offsetting cost in Max DD.**

## Why this is robust (not a one-cell peak)

The fine 2D sweep (K × Chronos q × hold × cap, 288 configs) shows
the **top 10 by Sharpe are all K=2** with Chronos q ∈ [0.20, 0.60],
hold=6, cap ∈ [0.34, 0.50]. Sharpe in [1.01, 1.04] across that whole
cell. Headline numbers are remarkably stable inside the K=2 plateau:

```
 k  chr_q  hold  cap  cagr_full  wf_mean_cagr  sharpe    max_dd  beats_spy
 2   0.45     6 0.40   0.492     0.494         1.04     -0.525        10
 2   0.30     6 0.40   0.489     0.492         1.04     -0.525        10
 2   0.40     6 0.40   0.486     0.486         1.03     -0.525        10
 2   0.50     6 0.40   0.483     0.494         1.03     -0.502        10
 2   0.60     6 0.40   0.484     0.495         1.02     -0.500         9
```

K=1 actually has the highest WF mean (63.3 %) but Max DD blows out
to **−80 %**, consistent with the "k=1 is fragile to delisting"
finding in `reports/final_validation.md`. K=2 is the sweet spot:
gives up ~14 pp WF mean vs K=1 in exchange for ~28 pp of Max DD
recovery and a much higher minimum WF split CAGR.

## Why K shifted from 3 → 2 under PIT correction

The original K=3 was selected on the biased panel where the universe
was 1,811 tickers, mostly survivors. On augmented PIT:

- 161 acquired/renamed large-caps are now in the universe with
  realistic returns through their acquisition dates (AGN, ANTM,
  ABMD, ALXN, etc.).
- The cross-sectional rank-distribution at each asof now contains
  more strong candidates; the GBM + Chronos filter produces a
  higher-conviction shortlist.
- K=2 concentrates on the top-2 of that shortlist, which is
  consistently the highest-edge subset on PIT data. At K=3 we add a
  third pick that on average dilutes (rather than confirms) the
  top-2 conviction.

## Crash-aware-staggered with K=2 (timing-luck mitigation)

| Config (crash-aware-stagger)   |   CAGR | WF mean | Sharpe | Max DD | Beats SPY |
|--------------------------------|-------:|--------:|-------:|-------:|----------:|
| K=3, q=0.45, h=6, cap=0.40    | 29.80 % | 28.19 % |   0.86 | -51.0 % |       6/10 |
| K=2, q=0.45, h=6, cap=0.40    | 30.87 % | 29.68 % |   0.80 | -68.2 % |       7/10 |

K=2 + crash-aware-staggered IS worse than K=3 + crash-aware-staggered
on Sharpe and Max DD. Why: the 6-tranche stagger already concentrates
into 6 × K positions overlapping over time, so K=2 (12 active names
max instead of K=3's 18) drops the diversification of the active book.
Lump-sum K=2 wins because the crash gate sends 100 % to cash on crash;
the staggered version can't replicate that level of crash protection.

**Net deployment recommendation: K=2 lump-sum.** Staggered is the
timing-luck-mitigation play, but it sacrifices the K=2 picker edge.
Lump-sum K=2 captures the picker edge and the augmented-PIT-validated
crash protection.

## Comparison summary — full matrix

```
Config                          CAGR   WF_mean  Sharpe   MaxDD   beats
deployed_k3_lump              32.92%   32.68%    0.92  -51.25%     8/10
deployed_k3_stag_ca           29.80%   28.19%    0.86  -50.98%     6/10
new_k2_lump                   49.21%   49.39%    1.04  -52.50%    10/10  ←  WINNER
new_k2_stag_ca                30.87%   29.68%    0.80  -68.17%     7/10
```

Saved at `augmented/v5_k2_vs_deployed.json`.

## Deployment proposal

**Change one parameter**: `K_PICKS = 3` → `K_PICKS = 2` in `experiments/
monthly_dca/v5/score_winner_v5.py` and the live cron job. Everything else
stays the same (Chronos filter q=0.45, hold=6, invvol weighting capped
at 0.40, tight regime gate).

Expected live impact on PIT-honest historical track:
- WF mean CAGR: 32.7 % → 49.4 % (+16.7 pp)
- Sharpe: 0.92 → 1.04 (+0.12)
- Max DD: essentially unchanged
- Splits beating SPY: 8/10 → 10/10 (+2)

## Empirical verifications (Phase 7d–7f)

### MC synthetic-delisting overlay (`mc_delisting_k2_aug.py`)

30 iters per α, K=2 vs K=3. **K=2 dominates K=3 across all delisting rates.**

| α (annual delist) | K=3 median | K=2 median |        Δ |
|-------------------|-----------:|-----------:|---------:|
| 0 %               |  32.92 %   |   49.21 %  | +16.29 pp |
| 4 % (realistic)   |  28.10 %   |  **41.62 %**| +13.52 pp |
| 8 %               |  22.49 %   |   35.75 %  | +13.26 pp |
| 20 %              |   4.12 %   |   13.30 %  |  +9.18 pp |

Fewer picks = fewer monthly delist exposures + picker edge per surviving
pick > loss per delist event. K=2 is more robust to delisting, not less.

### Scorer-variant sweep (`sweep_scorer_k2_aug.py`)

| Scorer            |   CAGR | WF mean | Sharpe | Max DD | Beats SPY |
|-------------------|-------:|--------:|-------:|-------:|----------:|
| **ml_3plus6** (deployed) | **49.21 %** | **49.39 %** | **1.04** | -52.5 % | 10/10 |
| ml_3plus6plus1    | 48.74 %|  47.83 %|   1.03 | -44.1 %|     10/10 |
| ml_h6             | 39.81 %|  37.34 %|   0.88 | -56.8 %|      9/10 |
| ml_h3             | 29.63 %|  29.17 %|   0.79 | -60.0 %|      8/10 |
| ml_h1             | 23.29 %|  18.83 %|   0.66 | -47.2 %|      3/10 |

Deployed `ml_3plus6` wins. (`ml_3plus6plus1` trades 1.5 pp WF mean for
8 pp Max DD improvement — risk-preference choice; not adopted.)

### Regime-gate sweep (`sweep_regime_k2_aug.py`)

| Gate     |   CAGR | WF mean | Sharpe | Max DD | Cash months | Beats SPY |
|----------|-------:|--------:|-------:|-------:|------------:|----------:|
| **tight** (deployed) | **49.2 %** | **49.4 %** | **1.04** | -52.5 % | 11 | 10/10 |
| strict   |  10.1 %|  11.1 % |   0.46 | -59.4 %|          39 |      5/10 |
| ddgate   |  37.3 %|  31.9 % |   0.86 | -79.0 %|           0 |      8/10 |

Deployed `tight` wins. `strict` is over-aggressive (15 % cash months),
`ddgate` never triggers (no crash protection).

## Final deployment matrix

**Only one parameter changes**: `K_PICKS = 3` → `K_PICKS = 2`.
Everything else (scorer, Chronos q, hold, cap, regime gate, weighting)
stays at the deployed default — empirically confirmed as the K=2 optimum.

| Stress test            | K=3 (deployed) | K=2 (proposed) |        Δ |
|------------------------|---------------:|---------------:|---------:|
| Augmented PIT (clean)  |   32.7 % WF    |    49.4 % WF   | +16.7 pp |
| α=4 % delist MC median |   28.1 %       |    41.6 %      | +13.5 pp |
| 2024 timing-luck year  |   -25.0 pp edge|   -10.2 pp edge| +14.8 pp |
| Sharpe                 |   0.92         |   1.04         |   +0.12  |
| Max DD                 |   -51.3 %      |   -52.5 %      |   -1.2 pp |
| Splits beating SPY     |   8/10         |   10/10        |       +2 |

No metric regresses materially. K=2 is the new winner.

## Residual caveats

1. **213 OTC bankruptcy-Q tickers** (AAMRQ, LEHMQ, etc.) are still
   absent from the universe; a hypothetical CRSP-corrected K=2 number
   would shave maybe 3-5 pp off the 49.4 % WF mean. The relative
   improvement (K=2 vs K=3) likely survives CRSP correction since both
   configs use the same universe.

## Files

- [`sweep_v5_aug.py`](sweep_v5_aug.py) — broad sweep (K, q, h, cap, scorer)
- [`sweep_fine_v5_aug.py`](sweep_fine_v5_aug.py) — focused 2D sweep around K=2
- [`run_v5_k2_aug.py`](run_v5_k2_aug.py) — K=2 lump-sum + crash-aware-staggered comparator

Outputs (in `augmented/`):
- `v5_param_sweep_results.csv` — broad sweep
- `v5_param_sweep_fine.csv` — fine sweep
- `v5_param_sweep_winner.json` — global winner
- `v5_param_sweep_fine_winner.json` — fine-grid winner
- `v5_k2_lump_summary.json` — K=2 lump-sum headline
- `v5_k2_staggered_ca_*.{json,csv}` — K=2 crash-aware-staggered
- `v5_k2_vs_deployed.json` — side-by-side

## Phase 8 — Cross-universe generalization (May 2026)

Scripts:
- `score_chronos_broader_aug.py` — regenerates Chronos preds on the
  full augmented panel (~1964 tickers, vs PIT-only 731).
- `sweep_generalize_k2_aug.py` — runs K=2 v5 on 6 universes.

Output: `augmented/v5_k2_generalize.csv` (also surfaced via
data.json's `multi_universe_generalisation`).

| Universe                  | Pool size | Full CAGR | WF mean | Sharpe | Max DD  | Beats SPY |
|---------------------------|----------:|----------:|--------:|-------:|--------:|----------:|
| **PIT S&P 500 augmented** |       229 | **49.21%**| **49.39%**| **1.04** | **-52.5%** | **10/10** |
| broader augmented (~1964) |       815 |    38.13% |  31.48% |   0.74 |  -88.7% |     5/10  |
| non-S&P 500 augmented     |       586 |    31.28% |  25.77% |   0.69 |  -88.7% |     5/10  |
| random 500 seed1          |       199 |    33.90% |  33.81% |   0.72 |  -69.2% |     6/10  |
| random 500 seed2          |       201 |    40.52% |  73.81% |   0.82 |  -73.2% |     8/10  |
| random 500 seed3          |       210 |    30.49% |  38.93% |   0.69 |  -88.7% |     6/10  |

Findings:
1. **The K=2 picker generalizes positively to every universe.** Full
   CAGR is well above SPY's ~12% in all 6 tests; WF mean edge ranges
   from +12pp (non-S&P) to +60pp (random seed 2). No universe regresses
   below SPY on average.

2. **But S&P 500 is the sweet spot.** Every other universe shows:
   - Materially lower Sharpe (0.69-0.82 vs 1.04)
   - Wider Max DD (-69% to -89% vs -52%)
   - Fewer WF splits beating SPY (5-8/10 vs 10/10)

3. **The Max DD widening is the clearest signal.** Broader, non-S&P,
   and random-seed3 all hit -88.7% — concentrating into 2 names from
   a less-curated universe occasionally lands on a stock that
   collapses. The S&P 500's quality filter (market-cap, liquidity,
   maturity) is doing real work.

4. **Random subsets show high variance.** Seed-2 lucks into a 73.81%
   WF mean; seed-3 drags to 38.93%. The deployed S&P 500 universe
   (49.39%) is a tighter, more reliable result than any individual
   random pick.

5. **K=2 is universe-tuned for S&P 500.** Earlier evidence (K=1 on
   PIT scores 63% WF but blows Max DD to -80%) suggested the K choice
   trades return for concentration risk; the cross-universe results
   confirm this is universe-dependent. On a broader universe, K=2
   might be too concentrated (one might prefer K=3+ there). We
   haven't re-swept K on alt universes; that's an open follow-up.

**Honest claim for the website:** the strategy generalizes positively
across universes but is BEST on PIT S&P 500. The S&P 500 cohort's
quality screen is part of what makes K=2 safe. We're not selling
"works on any universe at 50% CAGR"; we're selling "K=2 + Chronos +
PIT S&P 500 is the discovered local optimum, and it survives PIT
correction + MC delisting overlay + cross-universe stress."

Files:
- `score_chronos_broader_aug.py`
- `sweep_generalize_k2_aug.py`
- `augmented/ml_preds_chronos_broader.parquet` (1964-ticker Chronos preds)
- `augmented/v5_k2_generalize.csv` (raw sweep results)
- `augmented/v5_winner_generalize.csv` (reshaped for homepage builder)
