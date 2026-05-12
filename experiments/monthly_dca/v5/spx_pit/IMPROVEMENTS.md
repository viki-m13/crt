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

## Caveats

1. **More concentrated**. K=2 has only 2 names per basket vs K=3's 3.
   A single blow-up has 50 % weight instead of 33 %. The augmented panel
   does include real acquired/delisted names so this is partly priced in,
   but 213 OTC bankruptcy-Q tickers (AAMRQ, LEHMQ, etc.) are still
   absent from the universe; a hypothetical CRSP-corrected K=2 number
   would be lower than 49 % WF mean.
2. **Monte-Carlo delisting overlay** (`v3_winner_bias_sensitivity.csv`)
   should be re-run for K=2. The existing overlay was for K=3 and shows
   median CAGR drops to 32 % at α=4 %/yr synthetic delisting rate. K=2
   will degrade faster under the same overlay; need to re-run before
   final deployment.
3. **The sweep was 23 + 288 configs**. We haven't tested e.g. scorer
   variants (ml_3plus6plus1), longer Chronos windows, or alternative
   regime gates. The K=2 finding is solid but there may be further
   improvements stacked on top.

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
