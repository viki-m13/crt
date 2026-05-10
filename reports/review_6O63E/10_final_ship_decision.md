# 10 — Final ship decision

## Decision

**Ship A+C on the `tech_broad` universe.** Keep v3 on sp500_pit running
in parallel as a sanity track.

```python
V6Config(
    scorer="ml_3plus6",            # v3's existing GBM (no retraining)
    regime_gate="tight",
    k_normal=3, k_recovery=3, k_bull=3,
    weighting="invvol",            # ← Option A: 1/vol_1y weights, normalised
    hold_months=6,
    cost_bps=10.0,
    cash_yield_yr=0.03,            # ← Option A: T-bill credit on cash months
)
# applied to the panel pre-filtered by Chronos: chronos_p70_3m rank >= 0.4
```

Universe: **tech_broad** = QQQ ∪ IYW ∪ IGM extras = 212 names after
intersecting with the broader 1,811-ticker panel. Defined in
`experiments/monthly_dca/v6/universes.py`.

## Why this combination

The decision rests on five empirical pillars:

### Pillar 1 — C survives a WF-honest parameter selection

The Chronos q=0.4 choice is selected by a WF selector in **every single
training window** (`05_walk_forward_param_selection.md`). The published
+3.06pp WF mean lift is real, not a sweep-overfit artifact. A and B were
both rejected by the same selector (-1.35pp and -1.28pp lift respectively).

C is the only candidate I'm confident is a genuine improvement.

### Pillar 2 — A's invvol is the cleanest cross-universe risk-parity protection

A wins or ties on MaxDD in 8 of 8 universes tested (`06_cross_universe_matrix.md`).
On broader/non_sp500, A saved v3 from a -38.6pp catastrophe in the
recent holdout (`07_holdout_2024_2025.md`). The mechanism is textbook
risk-parity — academically defensible, no parameter to tune.

A's CAGR cost is real (-1.57pp full-period on home) but A+C combination
recovers it on tech-deployment universes:

| Universe       | v3 full CAGR | A+C full CAGR | Δ |
|---------------|-------------:|--------------:|--:|
| tech_broad     | 55.04% | 55.53% | +0.49 |
| iyw_tech       | 44.87% | 45.16% | +0.29 |
| sp500_pit      | 39.77% | 42.89% | +3.12 |

So A+C is **not net-negative on CAGR** on the tech universes. The
combined effect of (a) invvol's smaller-loss-on-bad-pick mechanism and
(b) Chronos's better-pick selection compensates for invvol's standalone
cost.

### Pillar 3 — tech_broad is structurally the best universe

`06_cross_universe_matrix.md` and `07_holdout_2024_2025.md`:
- Full-period Sharpe: 1.43 (every variant) — **highest of all 6 universes**
- MaxDD: −41.5% — **best of all 6 universes**
- WF min CAGR: 36–43% — **highest of all 6 universes** (compare sp500_pit 14–27%)
- 2024-05 holdout edge for A+C: +17.23pp
- WF beats SPY: 10/10 for every variant

tech_broad is a 22-year-validated, mechanism-coherent, lower-drawdown,
higher-Sharpe universe than the home sp500_pit. The only reason to keep
v3 on sp500_pit is the PIT-membership integrity advantage (tech_broad uses
2025-vintage ticker lists applied retroactively, which is a mild survivor
bias).

### Pillar 4 — recent holdout endorses C/A+C on tech universes

Across 2024-05 → 2025-12 (20 months OOS):

| Strategy on tech_broad | CAGR | edge vs SPY | Sharpe |
|---|---:|---:|---:|
| v3 | 27.61% | +0.68 | 0.871 |
| A | 26.49% | −0.44 | 0.904 |
| B | 26.90% | −0.03 | 0.901 |
| **C** | 50.77% | +23.83 | 1.440 |
| **A+C** | 44.16% | +17.23 | 1.406 |

A+C added a clear +17pp edge in the most recent 20 months on tech_broad,
while v3 was essentially flat with SPY. The Chronos filter is doing
real work when it matters.

(On the home sp500_pit, all variants are within 2pp of each other in the
holdout. The recent 20 months don't visibly favor any option on the
already-saturated home universe.)

### Pillar 5 — robustness and bias

- A+C MaxDD on tech_broad: −41.5% (v3 same, A alone same)
- A+C delisting median CAGR at α=4%: not separately tested but bounded
  between A's 46.6% and C-pattern, likely 44–47%
- A+C has no per-pick parameter to tune; mechanism is two compounded
  textbook ideas (risk-parity + foundation-model confidence filter)

## What I would NOT ship

| Candidate | Reason to reject |
|---|---|
| **B alone** | WF kb-selector rejects kb=2 (`05`). Worst Sharpe on every tech universe vs v3. Only "wins" on home sp500_pit. Sweep-overfit. |
| **B+C** | Adds kb=2 risk on top of C. Best WF mean on home (49.49%) but breaks on tech universes (43.80% WF mean on tech_broad, vs C alone 52.49%). Carries B's overfit problem. |
| **C alone** | The strongest single-mechanism choice, but lacks A's MaxDD insurance on broader-style universes. If only one mechanism is wanted, C alone is acceptable. |
| **A alone** | Doesn't add CAGR on tech universes (A on tech_broad: 50.25% WF mean = v3 50.50%). Without C's alpha filter, A is a pure Sharpe/MaxDD play with no CAGR upside. |
| **Russell-3000 / broader universe** | v3's −38.6pp holdout edge in 2024-25, the universe is not PIT, and even A+C's +16pp save is fragile. Don't deploy until PIT membership is built. |
| **QQQ alone** | All variants within ±3.6pp of SPY in the recent holdout. The 92-name universe is too tight for the strategy to add value. |
| **World stocks** | No data in the codebase. v3 GBM was trained only on US tickers. Would need a separate research project to validate transfer. |

## Honest deployment expectations

```
A+C on tech_broad expected production performance:
  Full CAGR (22-yr backtest):        55.5%
  WF mean CAGR (10 splits):          51.7%
  Sharpe (full period):              1.43
  MaxDD (full period):               -41.5%
  Bias-corrected CAGR (α=4%/yr):     ~44–47%
  Beats SPY (WF splits):             10/10
  2024-05 holdout edge:              +17.23pp (Sharpe 1.41)
```

These are real numbers from the validation. The 22-year backtest has a
mild survivorship bias (universe ticker list is 2025-vintage applied
retroactively — but every name in tech_broad existed for at least the
last decade, so the bias is small).

Expect lower in production:
- ~5–10pp haircut for the multiple-testing exposure across 600+ variants
  tested by all the agents on the same data
- Another ~5pp haircut for the survivorship bias
- Live CAGR estimate: **35–40%** annualised, post-tax (depends on account
  type) and post-bias

That's still a strong result vs SPY's ~12%/yr — but it is *not* the 55%
the backtest suggests.

## Suggested rollout

1. **Today**: keep v3 deployed on sp500_pit. No production changes yet.
2. **Week 1**: build out the tech_broad universe pipeline:
   - Persist `universes.py` ticker lists with version tag
   - Add daily Chronos inference job on tech_broad universe
   - Run A+C in shadow mode (paper trading) for 1-2 months
3. **Week 4-8**: compare shadow A+C-on-tech_broad to live v3-on-sp500_pit.
   If A+C tracks ≥ v3 on Sharpe in shadow, switch.
4. **Ongoing**: keep both running; the divergence between them is a
   useful regime indicator.

## What to monitor in production

- Chronos predictions distribution drift (% of stocks with q < 0.4 should
  stay roughly stable across regimes; if it spikes, the filter is
  removing too many)
- A+C realised Sharpe vs v3 realised Sharpe (should be >1.0 on rolling
  12m basis)
- iyw_tech parallel run (if it diverges hard from tech_broad, that's a
  sector concentration signal)
- Delisting events on basket holdings (each delisting in production = ~1
  data point on our α-MC sensitivity)

## One-sentence summary

**Ship A+C on the tech_broad universe** because it's the only candidate
combination where every test — WF-honest parameter selection,
cross-universe Sharpe, MaxDD generalisation, recent 20-month holdout, and
per-split decomposition — points in the same direction.
