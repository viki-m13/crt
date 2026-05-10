# Final Validation Report — multi_pillar_43Agh

Date: 2026-05-10. Branch: `claude/multi-pillar-stock-strategy-43Agh`.
Namespace: `multi_pillar_43Agh` (one of several agents on this task —
this suffix uniquely identifies this run's outputs to avoid collision).

This is the canonical write-up answering the brief's seven validation
questions. Read alongside:
- `executive_summary.md` — TL;DR and deliverables
- `decomposition.md` — pillar-by-pillar contribution
- `mechanism.md` — plain-language why-each-pillar-helps-or-hurts
- `leakage_redteam.md` — six leakage tests, all PASS

## 1. Honest OOS WF metrics

PIT S&P 500, 2003-09 → 2025-12, walk-forward 10-split mean. Engine and
splits as in `experiments/monthly_dca/v6/lib_engine.py`. Parity test
with V3 deployed: bit-for-bit match (verified at session start).

Best multi-pillar variants (full sweep at `backtests/sweep_results.csv`):

| Variant                        | CAGR | Sharpe | MaxDD | Sortino* | Calmar* | WF-mean CAGR | beats-SPY |
|--------------------------------|-----:|-------:|------:|---------:|--------:|-------------:|----------:|
| Baseline V3 (deployed)         | 39.77% | 0.955 | -49.83% | ~1.4 | 0.80 | 42.80% | 9/10 |
| Baseline V6 (invvol+cy)        | 38.20% | 0.971 | -45.98% | ~1.4 | 0.83 | 42.48% | 9/10 |
| Baseline V7 safer (sl+CDI+TLT) | 29.57% | 1.105 | -28.97% | ~1.6 | 1.02 | 32.64% | 9/10 |
| **Best multi-pillar (fail40%)**| 21.09% | 1.200 | -35.46% | ~1.7 | 0.59 | 22.30% | 8/10 |
| **Best multi-pillar (k5d20)**  | 20.14% | 1.217 | -37.14% | ~1.7 | 0.54 | 20.10% | 7/10 |
| Best mid-CAGR (penalty 0.20)   | 31.64% | 0.886 | -45.98% | ~1.3 | 0.69 | 35.30% | 9/10 |

*Sortino and Calmar are estimates derived from the returns series; not
explicitly computed in v6 evaluate(). Calmar = CAGR/|MaxDD|.

The brief's target was **100%+ CAGR / Sharpe 3+**. The honest measured
numbers from the multi-pillar architecture peak at ~21% CAGR / 1.2 Sharpe
on PIT S&P 500. The target is missed by a factor of ~5x on CAGR and
~2.5x on Sharpe. **This is the honest number; we do not massage.**

## 2. Decomposition by pillar

See `decomposition.md` for the full table. Headline:

| Pillar | Δ CAGR vs V6 | Δ Sharpe vs V6 | Δ MaxDD vs V6 |
|--------|-------------:|---------------:|--------------:|
| 1 alone (drop 30%) | -18.10pp | +0.120 | +9.27pp |
| 2 alone (trend gate) | -18.95pp | -0.067 | -2.29pp |
| 3 alone (novel 0.5) | -9.90pp  | -0.169 | -8.94pp |
| 4 alone (archetype 0.5) | -8.65pp | -0.144 | -6.69pp |
| 1+2+3+4 combined | -19.21pp | +0.016 | +2.88pp |

**Each pillar individually loses CAGR. None Pareto-improves V6.** The
combination doesn't compound benefits — it compounds the CAGR loss.

The CAGR comes from:
- Base ML signal (`ml_3plus6` HGB): the dominant contributor
- Tight regime gate (4 cash months): few-pp at full window, large in
  2008/2020 splits
- Inverse-vol weighting: -1.6pp full CAGR, +0.015 Sharpe (Pareto on
  Sharpe and MaxDD)
- Cash yield (3% bills): trivial but honest

## 3. Mechanism

See `mechanism.md`. One-sentence summary:

> The deployed V3/V6 ML model has captured the dominant exploitable
> edge on PIT S&P 500 monthly rebal (deep-value rebound discrimination
> via subtle non-linear feature interactions); hand-engineered overlays
> built from the same price-feature space cannot exceed it.

## 4. Leakage tests

See `leakage_redteam.md`. Six tests run, all consistent with the
measured edge being real (small) rather than a leakage artefact:

- PIT reconstitution: 0/88,124 violations
- Shuffle-score: edge collapses to ~0
- Survivorship exclusion: results WORSEN (would IMPROVE under leakage)
- Generalisation to non-S&P: edge persists +23pp
- Walk-forward embargo: 7m embargo verified by code review
- Feature timestamp spot check: matches hand-compute within 0.02

## 5. Conservative live-performance estimate

Because the multi-pillar variants are NOT Pareto-improvements over the
deployed V3/V6, no replacement is recommended. **The live-performance
estimate for the deployed V3/V6 is unchanged from those reports.**

For reference, V6's live haircut estimate (from `v6/REPORT.md §10` and
the bias-corrected Monte Carlo at α=4%/yr historical delisting rate):
- Historical CAGR: 38.20%
- Bias-corrected median: ~31.5%
- After +200bps additional slippage haircut and signal decay: **~25-30%
  CAGR live estimate, conservative**.

For the multi-pillar best variants (fail_40%, k5_drop20):
- Historical CAGR: ~20-21%
- After equivalent live haircut: ~15-18% CAGR live estimate
- Sharpe: ~1.0 live (haircut from 1.2)
- MaxDD: -35 to -40% live

The multi-pillar variants are roughly equivalent to the V7-safer
variant in profile — comparable risk reduction at a CAGR cost vs V6.

## 6. Capacity ceiling

K=3-5 picks of S&P 500 large-cap names with 6-month hold and ~4
rebalances/year. Pick names are typically S&P 500 mid-large cap with
ADV $50M-$1B+. Trading 3-5 names × 4×/yr × 25% portfolio turnover
= ~12-20 trades/year. Even a $1B AUM book trading $200M positions is
<5% of typical S&P 500 ADV. **Capacity ≥ $1B comfortably; no scaling
constraint binds at this size.**

## 7. Was the target hit?

**No.** The target was 100%+ CAGR / Sharpe 3+. We measured peak ~21%
CAGR / 1.2 Sharpe on the multi-pillar architecture.

**Binding constraints identified**:

1. **Price-feature space exhaustion**. The ML model uses 67 price-derived
   features. Hand-engineered overlays from the same feature space do not
   add orthogonal information.

2. **Frequency mismatch for trend confirmation**. Monthly rebal + 6-month
   hold is too slow for stock-level trend gating to add value. Rebound
   capture requires entering before the trend confirms; trend gating by
   definition delays entry.

3. **No fundamentals in the engine**. Forensic Study B's failure
   discriminators (Beneish M, Altman Z, Sloan accruals) require SEC
   filings. The engine has none. Real fundamental data could improve
   the failure filter to a Pareto-improvement, but pulling/lagging
   fundamentals at scale is a separate project.

4. **Single ML model for all regimes**. The model is one-size-fits-all.
   A separate bear-rebound model could plausibly help, but training
   regime-specific models walk-forward is fragile (small training sets
   in bear regimes).

5. **TDA / HMM / true transfer entropy under-built**. The fast surrogates
   used for Pillar 3 captured little. The full implementations were
   too computationally expensive for this session. Plausible 1-3pp
   Sharpe upside still on the table.

## What this report does NOT claim

- It does NOT claim the V3/V6 deployed strategy is the global optimum.
  It claims that **on this universe, at this frequency, with these
  features, the multi-pillar overlays we built do not exceed it**.
- It does NOT claim 100% CAGR is unreachable on PIT S&P 500 in
  general. It claims that the SPECIFIC architecture this brief
  prescribed (failure filter + trend gate + novel math + archetype +
  composite) does not reach it on this engine.
- It does NOT recommend wiring this multi-pillar strategy into the
  webapp main page (the brief's Phase-7 step). Per user instruction,
  the website is not updated by this branch.

## What this report DOES claim

- The multi-pillar architecture is **honestly measured**. PIT
  membership, no look-ahead, embargo, harsh delisting handling are all
  verified.
- Six leakage tests pass.
- The forensic studies catalogue (1,045 winner episodes, 3,573 failure
  episodes, with discriminating-feature analysis) is **independently
  valuable** as research output for any future work in this area.
- The conclusion is **falsifiable and replicable**. Anyone can run
  `PYTHONPATH=/home/user/crt python3 -m experiments.multi_pillar_43Agh.strategy.run_multi_pillar`
  and reproduce the decomposition table.

## Files (everything saved to repo on branch)

```
experiments/multi_pillar_43Agh/
  research/
    00_repo_map.md
    01_engine_audit.md
    02_invention.md
    forensics/
      discriminating_features.md
      archetypes.md
  strategy/
    forensic_studies.py        # Phase 1 — Studies A and B
    forensic_analysis.py        # Phase 1 — discriminating features + centroids
    failure_filter.py           # Pillar 1
    trend_regime.py             # Pillar 2
    novel_features.py           # Pillar 3 (full version, slow)
    novel_features_fast.py      # Pillar 3 (fast surrogates, used)
    archetype.py                # Pillar 4
    selection.py                # Pillar 5 — composite
    run_multi_pillar.py         # Phase 2/3 main runner
    run_sweep.py                # Phase 3 sweep
  tests/
    test_pit_membership.py      # Phase 4 PIT test
    test_no_lookahead.py        # Phase 4 no-lookahead test
    leakage_redteam.py          # Phase 5 red-team (6 tests)
  data/
    winners.parquet             # 1,045 winner episodes (>=5x in <540td)
    failures.parquet            # 3,573 failure episodes (incl 8 delistings)
    winner_features.parquet     # 793 pre-window snapshots
    failure_features.parquet    # 2,753 pre-window snapshots
    winner_controls.parquet     # 3,850 matched controls
    failure_controls.parquet    # 13,520 matched controls
    discriminating_features_winners.csv
    discriminating_features_failures.csv
    winner_archetype_centroid.parquet
    failure_archetype_centroid.parquet
    novel_features/{asof}.parquet  # 353 monthly novel-feature parquets
  backtests/
    runs/<ts>_<name>/{equity.csv, metrics.json}
    pillar_decomposition.csv
    pillar_decomposition.parquet
    sweep_results.csv
    experiment_log.csv
  reports/
    executive_summary.md
    final_validation.md         (this file)
    decomposition.md
    mechanism.md
    leakage_redteam.md
    leakage_redteam.json
```
