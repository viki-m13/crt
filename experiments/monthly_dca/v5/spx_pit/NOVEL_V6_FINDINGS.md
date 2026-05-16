# Novel-v6: can we improve the picks and the downside? — honest answer

**Date:** 2026-05-16 · **Branch:** `claude/monthly-stock-picker-aWQ8h`
**Code:** `novel_v6_cdv_cac.py`, `novel_v6_mn_overlay.py`, `novel_v6_chart.py`
All tested on the **exact deployed K=2 harness** (identical data loaders,
costs, regime gate, inv-vol weighting, hold rule); every delta is
attributable to the new logic alone. Thresholds are a-priori; **nothing
was swept to fit the curve.** Negatives are reported as negatives.

## How a pick is made (deployed v5), in one paragraph

Each month, for PIT S&P 500 members: a walk-forward HistGBM (retrained
every Jan, 7-month embargo) predicts 3- and 6-month forward returns →
`ml_score`. Amazon Chronos-Bolt-Tiny forecasts each name's 3-month path;
its p70 is cross-sectionally ranked and the **bottom ~55% are discarded
(binary gate only)**. Top **2** survivors by `ml_score` are bought,
inverse-vol weighted (cap 0.40), held ≥6 months, rule-based rebalance on
score-drift, **cash on the tight SPY crash-regime gate**.

## Two creative attempts to improve the PICKS — both fail (honest)

The unexploited surface area: Chronos emits a full predictive
distribution (p50/p70/p90 + median-path peak) and the strategy uses one
bit of it. Phase B only proved the Chronos *mean* is collinear with GBM
(ρ 0.97) — the predicted *shape* was never tested.

| Variant | CAGR | Sharpe | Max DD | WF beats | Holdout Sharpe | Verdict |
|---|---:|---:|---:|---:|---:|---|
| deployed K=2 (baseline) | 49.2% | 1.04 | −52.5% | 10/10 | 1.12 | reference |
| **CDV** (Chronos downside veto + convexity tilt) | 35.5% | 0.85 | −64.9% | 9/10 | 1.14 | **HURTS** |
| **CAC** (conviction-adaptive concentration) | 49.2% | 1.04 | −52.5% | 10/10 | 1.12 | **INERT** |
| CDV+CAC | 35.5% | 0.85 | −64.9% | 9/10 | — | hurts |

- **CDV hurts.** Using Chronos's predicted median / give-back / skew to
  veto and tilt *degrades* the GBM ranking (−14pp CAGR, deeper DD,
  full-history money-in 660×→125×). This independently re-confirms
  Phase B from a new angle: Chronos carries **no useful independent
  information beyond its (collinear) mean** — not even in its
  distribution shape.
- **CAC is inert.** Conditioning K and an SPY-diversion on the GBM
  top-pick's cross-sectional separation changes *nothing* — the metric
  never crosses its a-priori thresholds. Diagnostic: the picker's
  conviction is *always* "sharp enough" after the Chronos pre-gate; the
  real downside events are **macro crashes, not low-conviction
  stock-selection months**. Single-name conviction cannot see a market
  crash coming.

**Conclusion on alpha:** the repo's hard-won "exactly one independent
OOS-robust alpha" verdict survives a fresh, creative attack. The picks
cannot be honestly squeezed further with the information in this repo.
The residual downside is macro/crash-driven — so the fix must be a
portfolio-state lever, not a smarter stock screen.

## The downside lever that actually works (genuine improvement)

A **drawdown-conditional rotation** between two *already-validated*
streams: DCA into v5 normally; when the DCA portfolio's drawdown from
its running peak breaches −25%, route the book + new contributions into
the **v5 market-neutral sleeve** (ρ≈0 to v5, WF-min Sharpe 1.00 per
Phase B); switch back at −12.5% recovery (hysteresis). Not a new alpha —
an honest switch between validated streams. TH ∈ {20,25,30}% all behave
similarly (a plateau, not a knife-edge fit).

Investor-experienced metrics, PIT 2003–2026:

| | v5 only | static 60/40 | **MN-switch (TH25)** |
|---|---:|---:|---:|
| Lifetime money-in | 408× | 92× | **54×** |
| Lifetime IRR | 46.8% | 35.3% | **31.3%** |
| **Lifetime max drawdown** | **−72.1%** | −51.5% | **−38.5%** |
| 3y worst MOIC | 0.37× | 0.54× | **0.56×** |
| **5y worst MOIC** | 0.44× | 0.59× | **0.94×** |
| **5y median MOIC** | 3.10× | 2.34× | **2.97×** |
| 5y win vs SPY-DCA | 91.8% | 93.3% | **95.9%** |
| 10y win vs SPY-DCA | 100% | 100% | **100%** |

**The MN-switch Pareto-dominates the existing static 60/40 blend at the
3–5y horizons a DCA investor lives in**: at 5y it has a *better* worst
case (0.94× vs 0.59×) **and** a *better* median (2.97× vs 2.34×) **and**
a higher win rate — while cutting lifetime max drawdown −72%→−38% and
keeping the 100% ten-year hit rate. The cost is ~half the parabola
(54× vs 408× lifetime), which is the honest, unavoidable price of
removing two-thirds of the drawdown.

### The irreducible limit (stated honestly)

The switch does **not** improve the **1-year** worst case (0.34×,
unchanged). A −25% trailing-peak trigger cannot fire fast enough inside
a 12-month window when a crash starts right after you begin (2008). This
sub-12-month crash tail is irreducible with the signals in this repo —
the same wall every other phase hit, now confirmed from the overlay
side too.

## Bottom line

- **Picks:** cannot be honestly improved further here (CDV/CAC failed;
  one-alpha verdict re-confirmed). Beating it needs a genuinely
  different data family (PIT fundamentals / options VRP) — a separate
  data-acquisition program, exactly as Phase B / `SECOND_SLEEVE_SCOPE`
  concluded.
- **Downside:** *can* be honestly improved. The drawdown-conditional MN
  rotation is a real, evidence-backed, non-overfit upgrade over the
  static blend for a downside-conscious DCA product. Offer it as the
  "smoother-ride" variant; keep raw v5 as the "max-parabola, 10-year
  commitment" variant. There is no config that is both parabolic and
  low-downside — the data refuses it, and that refusal is now triple-
  confirmed.

## Files
- `novel_v6_cdv_cac.py` / `augmented/novel_v6_*_equity.csv` / `novel_v6_results.json`
- `novel_v6_mn_overlay.py` / `augmented/novel_v6_mn_overlay.json`
- `novel_v6_chart.py` / `augmented/novel_v6_overlay.png`
