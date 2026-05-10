# exp_01-08 + K/h sweep — Phase-2 cheap experiments summary

**Branch**: `claude/rebuild-stock-selection-YLOka`
**Window**: research only (2003-09 → 2024-04, 248 months). Holdout (2024-05 → 2026-04) NOT touched.
**Baseline**: v3 `ml_3plus6` ew K=3 h=6 tight-gate cost=10bps round-trip → CAGR 40.78%, Sharpe 0.953, MaxDD -49.83%, 4 cash months.

## Results table

| run | hypothesis | scorer | K | hold | weighting | filter | other | CAGR | Sharpe | MaxDD | cash | verdict |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| exp_00 | baseline (repro) | ml_3plus6 | 3 | 6 | ew | none | — | **40.78%** | **0.953** | **-49.83%** | 4 | reference |
| exp_01 | H2 conv low | ml_3plus6 | 3 | 6 | conv λ=0.5 | none | — | 39.52% | 0.870 | -61.51% | 4 | KILL |
| exp_02 | H2 conv high | ml_3plus6 | 3 | 6 | conv λ=1.5 | none | — | 34.53% | 0.745 | -71.55% | 4 | KILL |
| exp_03 | H2 conv + cash floor | ml_3plus6 | 3 | 6 | conv λ=0.5 | none | floor q25 | 39.52% | 0.870 | -61.51% | 4 | KILL |
| exp_04 | H3 accel overlay | ml_3plus6 | 3 | 6 | ew | accel | — | 42.39% | 0.964 | -49.83% | 4 | **kill (sample-of-2)** |
| exp_05 | H6 Donchian-130 | ml_3plus6 | 3 | 6 | ew | donchian130 | — | 48.73% | 1.003 | -56.99% | 4 | **kill (sample-of-1)** |
| exp_06 | H4 soft-cash | ml_3plus6 | 3 | 6 | ew | none | soft_cash, no hard gate | 20.79% | 0.757 | -59.88% | 0 | KILL |
| exp_07 | H4+H2 combo | ml_3plus6 | 3 | 6 | conv λ=0.5 | none | soft_cash, no hard gate | 21.24% | 0.700 | -61.51% | 0 | KILL |
| exp_08 | cash yield 3% | ml_3plus6 | 3 | 6 | ew | none | cash_yield_apr=0.03 | 40.85% | 0.954 | **-49.46%** | 4 | **KEEP (free 0.07pp + tiny DD lift)** |
| exp_09 | ml_136 scorer | ml_136 | 3 | 6 | ew | none | — | 36.94% | 0.885 | -49.83% | 4 | KILL (1m head adds noise) |
| exp_10 | K=1 | ml_3plus6 | 1 | 6 | ew | none | — | 28.61% | 0.618 | -80.38% | 4 | KILL |
| exp_11 | K=2 | ml_3plus6 | 2 | 6 | ew | none | — | 37.10% | 0.813 | -69.07% | 4 | KILL |
| exp_12 | K=5 | ml_3plus6 | 5 | 6 | ew | none | — | 29.92% | 0.844 | -59.09% | 4 | KILL |
| exp_13 | h=3 | ml_3plus6 | 3 | 3 | ew | none | — | 31.54% | 0.834 | -56.72% | 5 | KILL |
| exp_14 | h=12 | ml_3plus6 | 3 | 12 | ew | none | — | 35.07% | **0.968** | -58.76% | 4 | KILL (CAGR -5.7pp; Sharpe +0.015 not enough) |
| exp_15 | K=2,h=3 | ml_3plus6 | 2 | 3 | ew | none | — | 33.73% | 0.804 | -69.07% | 5 | KILL |
| exp_16 | K=5,h=3 | ml_3plus6 | 5 | 3 | ew | none | — | 26.58% | 0.801 | -60.67% | 5 | KILL |
| exp_17 | K=1,h=3 | ml_3plus6 | 1 | 3 | ew | none | — | 32.34% | 0.727 | -79.83% | 5 | KILL |
| exp_18 | K=1,h=12 | ml_3plus6 | 1 | 12 | ew | none | — | 19.16% | 0.588 | -86.48% | 4 | KILL |

## Key findings

### Survivors (Phase-3 candidates)

- **exp_08 cash-yield variant**: +0.07 pp CAGR, +0.001 Sharpe, MaxDD -49.46% (vs -49.83%). Free improvement from 3% T-bill yield in 4 cash months. Trivial implementation.

### Fragile "wins" — kept in graveyard

- **exp_05 H6 Donchian-130**: surface CAGR +7.95 pp. **Driven entirely by 2016 (+237 pp single-year contribution)** when Donchian filter shrank the basket to a NVDA-heavy concentration during NVDA's 227% breakout year. Average pick count 2.17 vs baseline 2.95 — filter shrinks K. Worse MaxDD (-57% vs -50%). 4/22 years beat baseline; median yearly diff 0.0%. **Sample-of-1 alpha; will not survive holdout.**
- **exp_04 H3 accel overlay**: surface CAGR +1.61 pp. **Driven by 2020 (+56 pp) and 2022 (+30 pp) only**; 2/22 years beat baseline. Worst-month identical to baseline. Median yearly diff 0.0%. Time underwater INCREASED. **Sample-of-2 alpha.**

### Clean kills — what doesn't work

- **H2 conviction sizing (exp_01-03)**: model's #1 pick is not reliably best. Concentrating into top-score hurts CAGR (-1.3 pp at λ=0.5, -6.3 pp at λ=1.5) and devastates MaxDD (-12 to -22 pp). Equal-weight is the right baseline.
- **H4 soft-cash continuum (exp_06-07)**: -20 pp CAGR, -0.20 Sharpe, MaxDD WORSE. Continuous risk-off bleeds equity exposure during normal volatility. The hard tight-gate's sharp on/off is right.
- **K=1, K=2, K=5**: all worse than K=3 by 4-12 pp CAGR. K=3 is locally optimal.
- **h=3, h=12**: both worse than h=6 by 5-10 pp CAGR. h=6 is locally optimal.
- **ml_136 scorer**: 1m head adds noise, drags CAGR -3.8 pp.

## Why nothing dramatic worked

1. **The v3 GBM has already extracted ~80% of the signal in price-only features.** Cheap downstream tweaks (weights, K, hold, filters) re-arrange the same picks.
2. **The model's score is well-calibrated as a rank, not as a magnitude.** Conviction-weighting fails because the score gap doesn't predict relative outperformance better than the rank order.
3. **Apparent "wins" are sample-size 1-2 in 22 years.** With ≈600 v3-v7 prior variants + my 19 here = ~620 hypotheses tested on the same data, the multiple-testing exposure is huge. Anything that looks +5-8 pp on full-period CAGR but is driven by 1-2 outlier years is almost certainly noise.
4. **The cheap experiment menu is exhausted.** The remaining hypotheses (H1 multi-target ensemble with 12m and classifier heads; H8 overnight/intraday split; H9 volume thrust persistence; H10 sector residualization) all require either GBM retraining or new feature pipelines — properly investigated in a follow-up session.

## Honest read

The v3 baseline (40.78% CAGR 2003-09→2024-04, 39.77% over the full 268-month window) is **hard to beat** with cheap downstream rearrangements. The path to durable +pp on top of v3 requires:

1. **Different ML targets**: a 12m forward-rank head and a top-quintile classifier head, ensembled with 3m/6m by recent IC. Cheapest unexplored direction. Estimated cost: 1-2 hours of GBM training + harness wrapping.
2. **Sub-monthly information**: features computed from open/high/low (overnight vs intraday returns, true-range, gap behaviour) rather than only close-close monthly aggregates.
3. **Feature interactions the GBM didn't surface**: cross-asset pair signals, breadth-conditioned momentum, sector-residualized score.

These are deferred to the next session — see `research/YLOka/INDEX.md` for the Phase-2-continuation plan.

## What goes in graveyard

`research/YLOka/graveyard/` will hold a brief write-up for each kill so the experiments aren't repeated:
- `H2_conviction_sizing.md`
- `H4_soft_cash_continuum.md`
- `H6_donchian_130_breakout.md`  (with the 2016 NVDA-concentration explanation)
- `H3_accel_overlay.md`  (with the 2020/2022-only explanation)
- `K_hold_grid.md`  (showing K=3, h=6 is the local optimum)

## Files

- `strategy/YLOka/harness.py` — simulator + scorers + pickers + run logging
- `strategy/YLOka/run_experiments.py` — sweep driver
- `backtests/YLOka/runs/<ts>_<name>_<hash>/manifest.json` — per-run config + metrics + git SHA
- `backtests/YLOka/runs/<ts>_<name>_<hash>/equity.parquet` — per-run equity curve + picks
- `backtests/YLOka/experiment_log.csv` — append-only log of every run
