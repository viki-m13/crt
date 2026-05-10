# Review of 11 Agent Runs — Validation & Ship Decision

**Branch**: `claude/review-model-approaches-6O63E`
**Run date**: 2026-05-10
**Reviewer**: Claude (this branch)
**Mission**: Independently validate the 11 parallel agent submissions from 2026-05-10
and choose which (if any) to deploy in place of v3.

## Documents in this folder

| # | Document | Purpose |
|---|---|---|
| 00 | `00_INDEX.md` | This file — navigation |
| 01 | `01_review_of_agents.md` | What each of the 11 agents claimed and a skeptical first-pass verdict |
| 02 | `02_baseline_and_options.md` | v3 baseline definition, the three candidate options (A, B, C) selected for deep validation |
| 03 | `03_methodology.md` | Engine, walk-forward splits, universes, all the testing primitives |
| 04 | `04_reproduction_results.md` | Exact reproduction of published headline numbers for v3, A, B, C |
| 05 | `05_walk_forward_param_selection.md` | The critical WF-honest test — picks the parameter on training data only |
| 06 | `06_cross_universe_matrix.md` | Full 6-universe × 6-strategy result matrix (CAGR, Sharpe, MaxDD, WF, beats SPY) |
| 07 | `07_holdout_2024_2025.md` | 20-month frozen holdout per universe — the fairest test |
| 08 | `08_bias_sensitivity.md` | Synthetic delisting MC at α ∈ {0..20%}/yr per candidate × universe |
| 09 | `09_per_split_decomposition.md` | Per-split TEST CAGR / edge / Sharpe / MaxDD for every option × universe |
| 10 | `10_final_ship_decision.md` | The deployment recommendation, why, and what NOT to ship |
| 11 | `11_caveats_and_open_questions.md` | What I'd want to validate further before live capital |

## TL;DR (one paragraph)

Of the 11 agents, six independently concluded **v3 is near-optimal** on PIT S&P 500 with price-only signals. Five proposed lifts. After reproducing all of them and stress-testing on a walk-forward parameter selector (where the agent's chosen "always-on" parameter is *also* chosen on train data only), only **Option C (v3 + Chronos-bolt-tiny p70 filter at q=0.4)** survives — its q=0.4 choice is OOS-stable across every training window and delivers a **WF-honest +3.06pp lift** vs v3. Options A (invvol weighting) and B (kb=2 bull-regime concentration) are **rejected by the WF selector** (lifts of −1.35pp and −1.28pp respectively).

For the deployment target the user described — iShares tech ETFs + QQQ + Russell-3000-ish — the winning combination is **A+C on the `tech_broad` universe (212 names = QQQ ∪ IYW ∪ IGM extras)**:

- Full-period: CAGR 55.5%, Sharpe 1.43, MaxDD −41.5%, WF mean 51.7%, beats SPY 10/10
- 20-month holdout (2024-05 → 2025-12): +17.23pp edge, Sharpe 1.41
- vs v3 same universe: better Sharpe, better MaxDD, +17pp better recent edge, +1.2pp better WF mean

Russell-3000-style universes are too risky to deploy on (v3 had a 38pp loss to SPY in the recent holdout). QQQ alone is too tight (strategy adds ~zero alpha). World stocks need a separate data project.

## Artifacts

All Python runners that produced these numbers are under
`experiments/monthly_dca/v6/` (filenames begin with `run_*.py`) and
`experiments/monthly_dca/v5/`. All result CSVs are committed under
`experiments/monthly_dca/v6/results/`. The Chronos prediction parquets
(home and broader) are at
`experiments/monthly_dca/cache/v2/sp500_pit/ml_preds_chronos.parquet` and
`experiments/monthly_dca/cache/v2/ml_preds_chronos_broader.parquet`.

See `10_final_ship_decision.md` for the actionable recommendation.
