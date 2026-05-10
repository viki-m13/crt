# H4 — Soft-cash continuum — KILLED

**Hypothesis**: replace the binary `tight` crash gate with a continuous `risk_off ∈ [0, 1]` score using SPY DD-from-52wh (-5%→0, -15%→1) and SPY vol_12m (20%→0, 35%→1). Allocate `1 − risk_off` to equity, `risk_off` to cash earning T-bill yield. Idea: de-risk earlier, smoother, avoid binary whipsaw.

**Result**:
| | CAGR | Sharpe | MaxDD | cash months |
|---|---:|---:|---:|---:|
| baseline (tight gate, 0% cash yield) | 40.78% | 0.953 | -49.83% | 4 |
| soft-cash + 3% cash yield | 20.79% | 0.757 | -59.88% | 0 |
| soft-cash + conviction sizing | 21.24% | 0.700 | -61.51% | 0 |

**Why it failed**:
- Continuous de-risking trades a small drawdown reduction for ENORMOUS CAGR drag — equity exposure averages ~70% across the period, so the strategy gives back ~30% of its compounding power.
- MaxDD got *worse* (-60% vs -50%) because the soft-cash overlay reduces upside in recoveries (when DD is still at -5% to -10% you're still half-cash) — so the rebound from a 2008-style crash is muted.
- The equity sleeve's vol comes mostly from idiosyncratic stock vol, not SPY-DD-driven vol. SPY DD ≤ -5% triggers a partial de-risk *every minor correction*, even when individual picks are uncorrelated to SPY.

**Don't repeat in this form**: the v3 hard `tight` gate fires 4 times in 22 years — extremely rare, only on real crashes (≤-8% in 21d). That sparsity is the reason it works. Replacing it with a "more sensitive" continuous version was always going to bleed equity in normal markets.

**A better attempt would be**: a *conditional* soft-cash that only fires AFTER the hard gate has already triggered AND breadth is still deteriorating — i.e., adds extra cash WHILE in crash regime to size up the de-risk. Or: a vol-targeted leverage = clip(vol_target / portfolio_vol, 0, 1) which aims for constant portfolio vol rather than constant equity exposure. Both deferred.
