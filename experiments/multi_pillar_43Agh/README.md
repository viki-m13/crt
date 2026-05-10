# multi_pillar_43Agh

Multi-pillar stock selection strategy research. Branch:
`claude/multi-pillar-stock-strategy-43Agh`. Namespace: `multi_pillar_43Agh`
(unique identifier — multiple agents are working on this task in parallel
on different branches; this suffix tags this run's outputs).

## Quick read

- **Headline**: Aggressive 100%+ CAGR / Sharpe 3+ target was NOT met on
  PIT S&P 500. Best multi-pillar variant: ~21% CAGR / 1.20 Sharpe / -35%
  MaxDD. The deployed V3/V6 strategy (CAGR 38-40%, Sharpe 0.97) is not
  Pareto-improved by this architecture on this universe.
- **All 6 leakage tests PASS**. The measured edge is real, just smaller
  than the deployed baseline.
- **Forensic studies catalogue** (1,045 winners, 3,573 failures,
  discriminating-feature analysis) is independently valuable research
  output.
- **Website NOT updated** per user instruction.

## Read order

1. `reports/executive_summary.md` — TL;DR
2. `research/00_repo_map.md` — what's in the existing repo
3. `research/01_engine_audit.md` — engine honesty checklist
4. `research/02_invention.md` — multi-pillar architecture design
5. `research/forensics/discriminating_features.md` — Phase 1 winners/failures analysis
6. `research/forensics/archetypes.md` — Phase 1 winner/failure archetypes
7. `reports/decomposition.md` — pillar-by-pillar contribution
8. `reports/mechanism.md` — plain-language why each pillar helps/hurts
9. `reports/leakage_redteam.md` — six leakage tests, all PASS
10. `reports/final_validation.md` — canonical write-up

## Reproducibility

```
# Phase 0 parity (verifies engine reproduces V3 deployed bit-for-bit)
cd experiments/monthly_dca/v6 && python3 run_baseline.py

# Phase 1 forensic studies (~5 min)
PYTHONPATH=/home/user/crt python3 experiments/multi_pillar_43Agh/strategy/forensic_studies.py
PYTHONPATH=/home/user/crt python3 experiments/multi_pillar_43Agh/strategy/forensic_analysis.py

# Phase 2 — pillars built standalone (smoke tests)
python3 experiments/multi_pillar_43Agh/strategy/failure_filter.py
python3 experiments/multi_pillar_43Agh/strategy/trend_regime.py
python3 experiments/multi_pillar_43Agh/strategy/archetype.py
python3 experiments/multi_pillar_43Agh/strategy/novel_features_fast.py  # ~30s

# Phase 3 — multi-pillar runner (10 variants, ~3-5 min)
PYTHONPATH=/home/user/crt python3 -m experiments.multi_pillar_43Agh.strategy.run_multi_pillar

# Phase 3b — sweep (17 variants, ~6-7 min)
PYTHONPATH=/home/user/crt python3 -m experiments.multi_pillar_43Agh.strategy.run_sweep

# Phase 4 — hard-constraint tests
PYTHONPATH=/home/user/crt python3 experiments/multi_pillar_43Agh/tests/test_pit_membership.py
PYTHONPATH=/home/user/crt python3 experiments/multi_pillar_43Agh/tests/test_no_lookahead.py

# Phase 5 — leakage red-team
PYTHONPATH=/home/user/crt python3 -m experiments.multi_pillar_43Agh.tests.leakage_redteam
```

## What's in here

| Path | Purpose |
|------|---------|
| `research/` | Plans, analyses, write-ups |
| `strategy/` | Code modules — pillars, runners, sweeps |
| `tests/` | Hard-constraint + leakage tests |
| `data/` | Episode catalogues, centroids, novel-feature parquets |
| `backtests/` | Per-run equity curves and metrics; sweep results |
| `reports/` | Markdown reports |

## What is NOT in here

- The website. Per user instruction, `docs/` was not modified.
- A re-implemented engine. We built on top of `experiments/monthly_dca/v6/lib_engine.py`
  (parity-tested with V3 deployed).
- Fundamentals data. The engine is price-only.
- Full TDA / HMM implementations. The Pillar 3 fast surrogates were used
  due to compute budget; the full versions are stubbed in
  `strategy/novel_features.py`.
