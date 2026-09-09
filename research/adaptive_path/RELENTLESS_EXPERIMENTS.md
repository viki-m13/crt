# Relentless selective forecasting experiments

Objective remains one thing: issue a stock + immutable endpoint >=30 trading sessions only when evidence supports Price(endpoint) > Price(issue), targeting >95% precision. No pick quota. Abstention is expected.

Do not stop because a family fails. Do not turn exploratory overfit into a result either. Every candidate mechanism is discovered on earlier data and must survive later untouched/rolling-origin data, PIT membership, dead/delisted names, date clustering, and pessimistic missing-outcome scoring.

## Parallel invention frontier

1. **Intersection-of-rare-evidence (IRE):** issue only when independent evidence families agree: residual trend, volatility compression, breadth regime, drawdown geometry, volume/accumulation, and catalyst state. Learn reliability of intersections, not average classifier probability.
2. **Counterfactual twin test (CTT):** for every candidate find historical same-regime/same-sector near-neighbours differing primarily in the proposed signal. Require the treated cohort to dominate its twins at the chosen horizon; veto signals whose apparent precision is just market beta.
3. **Failure-first veto network (FFV):** model the probability of being lower rather than higher. Separate crash, dilution/distress, blow-off, earnings-gap-reversal, regime-break and delisting-risk experts. Buy only when *all* independently trained failure experts assign sufficiently low risk.
4. **Adaptive horizon survival curve (AHSC):** estimate P(P_t+h > P_t) over h=30..756 from matured analog cohorts. Select the earliest robust plateau rather than maximizing an in-sample horizon. Horizon is fixed when issued.
5. **Regime fingerprint recurrence (RFR):** condition on market breadth, dispersion, vol-of-vol proxy, rates/credit proxies where causally available, sector residual state and trend age. Require recurrence across multiple historical eras; unseen fingerprints abstain.
6. **Cross-sectional tournament (CST):** do not ask whether every stock rises. Rank all PIT-eligible names, then test only the extreme winner tail. Calibrate the top-1/top-k decision itself and compare with random same-date names.
7. **Information-shock clock (ISC):** timestamp SEC/earnings/capital-allocation/contract events and learn family-specific payoff clocks, with price/volume used as confirmation/veto rather than the sole predictor.
8. **Conformal risk-control selector (CRCS):** wrap scores in selective risk bounds; choose thresholds using only matured prior forecasts. If the upper bound on failure risk cannot clear the target, no pick.
9. **Multi-view unanimity (MVU):** independently fit price-only, cross-sectional, event/fundamental and historical-recurrence views. Agreement is useful only if each view has incremental conditional information; correlated duplicate models do not count as independent votes.
10. **Anti-crowding inversion (ACI):** explicitly test the Stocksonstocks quiet-LOW / loud-MEDIUM lifecycle hypothesis on the harsher CRT PIT panel, including negative controls and a fresh confirmatory segment.
11. **Path-shape veto (PSV):** endpoint probability is primary, but reject candidates whose historical conditional paths have excessive interim drawdown, unstable endpoint timing, or bimodal outcomes inconsistent with the selected horizon.
12. **Sequential evidence accumulation (SEA):** allow a candidate to remain on a watchlist as evidence arrives, but the buy decision is a new timestamped forecast. Never rewrite an earlier forecast or use post-entry evidence to justify it.

## Search discipline

Exploration may be broad and creative. Keep a machine-readable ledger of every family/threshold/horizon tried. Use nested walk-forward selection: inner history discovers/tunes; outer future evaluates the complete adaptive policy. Reserve a final temporal segment/prospective stream that is not used for invention. Report precision and coverage together, plus number of unique dates, sectors, tickers, non-overlapping horizon blocks, longest no-pick interval, return distribution and matched controls.

A result above 95% with tiny n is a lead, not success. The preferred tool can be extremely selective, including months with no recommendation. The goal is not to force a number; it is to keep inventing until a genuinely reproducible high-precision region is found or the available data establish the frontier.