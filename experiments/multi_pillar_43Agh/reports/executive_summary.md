# Executive Summary — Multi-Pillar Stock Selection (43Agh)

Date: 2026-05-10. Branch: `claude/multi-pillar-stock-strategy-43Agh`.
Namespace: `multi_pillar_43Agh`. (One of several agents on this task — the
suffix uniquely identifies this run's outputs.)

## Headline

**The aggressive 100%+ CAGR / Sharpe 3+ target was not met on PIT S&P 500.**
The honest measured numbers from the multi-pillar architecture, the existing
~9,000-variant V6/V7 sweep, and a fresh sweep over Pillar combinations all
converge on the conclusion that the V3/V6 ML-based strategy has captured
the dominant exploitable edge on this universe at monthly rebalance.

## What was built (deliverables)

| Phase | Deliverable | Status |
|------|-------------|--------|
| 0 | `00_repo_map.md`, `01_engine_audit.md`, `02_invention.md` + parity test | ✅ |
| 1 | Forensic Study A (1,045 winner episodes), Study B (3,573 failures + 8 delistings), discriminating-feature ranking, archetype centroids | ✅ |
| 2 | Pillar 1 — Failure-avoidance filter (`strategy/failure_filter.py`) | ✅ |
| 2 | Pillar 2 — Trend & regime gate (`strategy/trend_regime.py`) | ✅ |
| 2 | Pillar 3 — Novel-math features (`strategy/novel_features_fast.py`) — 60d SPY-corr, lag-1 persistence, abs-skew | ✅ |
| 2 | Pillar 4 — Forensic archetype score (`strategy/archetype.py`) — combined distance + engineered | ✅ |
| 2 | Pillar 5 — Composite selection (`strategy/selection.py`) | ✅ |
| 3 | Pillar standalone tests + decomposition table | ✅ |
| 3 | Multi-pillar sweep over 17 combinations | ✅ |
| 4 | Hard-constraint test suite (`tests/test_pit_membership.py`, `tests/test_no_lookahead.py`) | ✅ |
| 5 | Leakage red-team (`tests/leakage_redteam.py`): PIT, shuffle-score, survivorship, generalisation | ✅ |
| 5 | Final reports | ✅ (this dir) |

## Headline metrics (sp500_pit, 2003-09 → 2025-12)

| Strategy | CAGR | Sharpe | MaxDD | WF mean CAGR | WF n_pos | WF beats SPY |
|----------|-----:|-------:|------:|-------------:|---------:|-------------:|
| **V3 deployed (baseline)** | **39.77%** | **0.955** | **-49.83%** | 42.80% | 10/10 | 9/10 |
| V6 winner (invvol+cy)      | 38.20% | 0.971 | -45.98% | 42.48% | 10/10 | 9/10 |
| V7 safer (sl+CDI+TLT)      | 29.57% | 1.105 | -28.97% | 32.64% | 10/10 | 9/10 |
| Pillar 1 only (drop 30%)   | 20.10% | 1.091 | -36.71% | 22.92% | 10/10 | 8/10 |
| Pillar 2 only (trend gate) | 19.25% | 0.904 | -48.28% | 20.71% | 10/10 | 9/10 |
| Pillar 3 only (novel 0.5)  | 28.30% | 0.802 | -54.92% | 27.04% | 10/10 | 8/10 |
| Pillar 4 only (arch 0.5)   | 29.55% | 0.826 | -52.68% | 31.03% | 10/10 | 8/10 |
| Pillars 1+2                | 18.46% | 1.059 | -40.61% | 21.84% | 10/10 | 8/10 |
| Pillars 1+2+4              | 16.01% | 0.843 | -33.97% | 19.00% | 10/10 | 7/10 |
| Pillars 1+2+3+4            | 18.99% | 0.987 | -43.10% | 24.25% | 10/10 | 7/10 |

## Mechanism (plain language — why each pillar helps OR hurts)

**Pillar 1 (failure filter)** — *helps Sharpe / hurts CAGR*. Removing the
universe's bottom 30% by failure-risk score does cut some catastrophic
picks (positive +9.3pp on MaxDD, +0.12 on Sharpe). But it also removes
deep-value rebound names that V3's ML model is selectively very good at
spotting at the bottom. The Sharpe gain is real but the CAGR cost is
larger. **Net effect on this universe at monthly rebal: not Pareto.**

**Pillar 2 (stock-level trend gate)** — *neutral-to-negative*. Requiring
positive multi-timeframe trend cuts the same deep-value rebounds that
Pillar 1 cuts. The gate is too tight at default values; relaxing helps
but never recovers. **Why**: in V3's worldview, the deep-value rebound
IS the edge.  Trend confirmation by definition arrives after the rebound
has started, by which time the ML model has already moved on.

**Pillar 3 (novel math features)** — *adds noise, not signal*. The 60-day
SPY correlation, lag-1 persistence, and abs-skew features are
short-horizon and weakly correlated with 6-month forward returns at
monthly rebal frequency. The full-fat versions (TDA, transfer entropy,
GPD tail-shape) were too computationally expensive to run end-to-end in
this session; the fast surrogates didn't capture the hypothesised edge.

**Pillar 4 (forensic archetype)** — *intuitive but redundant*. The
archetype centroid built from 1,045 winner episodes plausibly identifies
high-vol, deeply-pulled-back, base-building names. But the existing 67-
feature ML model already absorbs most of that signal — the archetype
score's information overlaps with `pullback_1y`, `vol_1y`, `mom_12_1`,
`vol_contraction`, etc., which feed the ML. **Net effect: dilutes ML
signal without adding orthogonal information.**

**Composite (Pillars 1+2+3+4)** — the costs compound, not the benefits.
The 1+2+3+4 combination has **lower CAGR** than any single pillar, with
only modest Sharpe improvement.

## What this means

The brief acknowledged this was an exploratory target. The thorough,
honest answer is:

1. **The dominant exploitable monthly-frequency edge on PIT S&P 500 has
   been captured by the V3/V6 ML scorer** with deep-value rebound capture.
   Most "obvious good ideas" (failure filtering, stronger trend gating,
   archetype matching, novel math) end up cutting this edge.

2. **What still adds value**: the v6 invvol weighting (Pareto improvement
   on Sharpe and MaxDD vs V3 EW). The v7 hedges (CDI overlay, daily
   per-pick stop-loss) materially reduce MaxDD at a CAGR cost. These are
   already deployed.

3. **Where edge might still hide** (not exhaustively explored here):
   - Lower-frequency rebal (quarterly) with stronger conviction filters
   - Cross-asset (gold, IEF, MUB) regime overlays not yet tested
   - Separate models for bear-market deep-value vs bull-market trend
   - True novel-math features (full TDA, full HMM) given more compute
   - Fundamentals — the engine has none right now (price-only)
   - Earnings-event causal-effect features (no fundamentals data)

## Validation gauntlet (Phase 5)

- **PIT membership**: every panel row passes (`tests/test_pit_membership.py`)
- **No-lookahead**: SPY mom_12_1 spot-check passes (`tests/test_no_lookahead.py`)
- **Shuffle-score**: random permutation of score → edge collapses to ≈ SPY
- **Survivorship-exclusion**: removing delisted tickers → small impact, NOT a dramatic improvement (would indicate leakage)
- **Generalisation (non-S&P)**: edge persists positive but smaller — characteristic of S&P-tuned ML model

The shuffle-score, survivorship-exclusion, and generalisation tests are
all consistent with the actual edge being real (small) rather than a
leakage artefact. Detailed numbers in `reports/leakage_redteam.json`.

## Recommendation

**Do not deploy multi-pillar as a replacement for V3/V6.** The deployed
strategy already delivers what the multi-pillar architecture aimed for,
on this universe.

**Useful pieces to keep**:
- Pillar 1's failure score is reusable as a *display field* for users
  ("this stock's failure-risk score = X") even if not as a hard filter.
- The forensic Study A/B episode catalogues are valuable research data
  (saved at `data/winners.parquet`, `data/failures.parquet`).
- The discriminating-feature analysis (`research/forensics/discriminating_features.md`)
  is a reference for any future feature-engineering work.

**Where a future agent could push**:
- Pull true SEC fundamentals; rebuild Beneish M, Altman Z, Sloan accruals;
  re-test failure filter with real fundamentals.
- Implement full TDA (gudhi/ripser) and rebuild Pillar 3 properly.
- Test on quarterly rebal frequency with stronger filters.
- Build a per-regime ML model (separate bear-rebound and bull-trend models).

## Files

```
experiments/multi_pillar_43Agh/
├── research/
│   ├── 00_repo_map.md
│   ├── 01_engine_audit.md
│   ├── 02_invention.md
│   ├── forensics/
│   │   ├── discriminating_features.md
│   │   └── archetypes.md
│   ├── pillar_1_failure_avoidance/...
│   ├── pillar_2_trend_regime/...
│   ├── pillar_3_novel_math/...
│   ├── pillar_4_archetype/...
│   ├── pillar_5_composite/...
│   └── graveyard/...
├── strategy/
│   ├── forensic_studies.py
│   ├── forensic_analysis.py
│   ├── failure_filter.py
│   ├── trend_regime.py
│   ├── novel_features.py
│   ├── novel_features_fast.py
│   ├── archetype.py
│   ├── selection.py
│   ├── run_multi_pillar.py
│   └── run_sweep.py
├── tests/
│   ├── test_pit_membership.py
│   ├── test_no_lookahead.py
│   └── leakage_redteam.py
├── data/
│   ├── winners.parquet (1,045 episodes)
│   ├── failures.parquet (3,573 episodes)
│   ├── winner_features.parquet
│   ├── failure_features.parquet
│   ├── winner_controls.parquet
│   ├── failure_controls.parquet
│   ├── discriminating_features_winners.csv
│   ├── discriminating_features_failures.csv
│   ├── winner_archetype_centroid.parquet
│   ├── failure_archetype_centroid.parquet
│   └── novel_features/{asof}.parquet (353 files)
├── backtests/
│   ├── runs/<ts>_<name>/{equity.csv, metrics.json}
│   ├── pillar_decomposition.csv
│   ├── pillar_decomposition.parquet
│   ├── experiment_log.csv
│   └── sweep_results.csv
└── reports/
    ├── executive_summary.md       (this file)
    ├── final_validation.md
    ├── leakage_redteam.md
    ├── mechanism.md
    ├── decomposition.md
    └── leakage_redteam.json
```
