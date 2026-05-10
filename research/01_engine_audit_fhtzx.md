# 01 — Engine Audit

The brief says "the engine must be honest before anything else". Here is the
audit, what's clean, what's broken, and what we're going to do about each.

## Findings, in order of severity

### 🟥 H1. Universe is materially survivorship-biased
**Evidence.** `cache/prices_extended.parquet` contains 1833 tickers spanning
1995–2026. Only **9 tickers** show no observations in the last 6 months —
i.e. only 9 names "died". Real S&P 500 turnover historically removes roughly
20–30 names per year, so a 30-year panel should contain **600+ delisted /
acquired / removed names**, not 9. The panel is ~the union of "names that
survived to or joined the index recently".

**Impact.** Every backtest CAGR on this universe is biased upward. The MC
overlay at α=4%/yr partially compensates and gives a more honest 28.6%
median CAGR (vs 35.4% raw). But the overlay is a model, not real data —
e.g. it can't reproduce specific patterns like "stocks that crashed
mid-pullback before recovering" because the dead names are just gone.

**Plan.** I'm not going to rebuild PIT constituents from scratch (multi-week
project). Instead:
1. Keep the existing engine.
2. Use the α=4%/yr MC overlay as the **headline bias-corrected number**.
3. Run a generalization test on a held-out universe slice (frozen ticker
   holdout) and a held-out time slice (last 18 months untouched).
4. State the limitation explicitly in the final report.

### 🟧 H2. No volume data in the cached panel
**Evidence.** `prices_extended.parquet` has only adjusted close. The panel
columns are tickers, the rows are dates, the values are floats. No
`volume_<ticker>` column exists.

**Impact.** Accumulation-footprint signals (Wyckoff-style up-vs-down volume
asymmetry, dollar-volume signatures, volume-weighted relative strength)
cannot be computed without volume. Many of my candidate inventions need
volume.

**Plan.** Pull volume from yfinance for the existing 1833 tickers, append
to the cache as `prices_extended_with_volume.parquet`. Cap to last 25 years
to keep size manageable. (Done in feasibility step.)

### 🟧 H3. Same-day execution leak (small but real)
**Evidence.** `compound_engine.py` line 273-298: scoring uses
`load_features(date_t)` (which uses prices ≤ T-close), and deployment
happens at `panel_arr[cur_panel_pos, ci]` (T's close).

**Impact.** Signal can use information from T's close to decide what to
buy at T's close. Real-world equivalent: signal at T close → execute at
T+1 open. The bias rewards strategies that latch onto the day's intraday
momentum. Likely small at monthly frequency but non-zero.

**Plan.** Build a strict "T+1 open" execution mode in the new selection
harness. Compare the new strategy at both T-close (legacy) and T+1-open
(strict) to estimate the bias size. Headline number = T+1-open.

### 🟨 M1. No embargo at WF split boundaries
**Evidence.** The 10 WF splits in REPORT.md split TRAIN ends at year T,
TEST starts at year T+1, with no embargo period. The 12-month features
straddle the boundary, so there is mild information leakage of training-set
returns into test-set features.

**Impact.** Inflates OOS metrics by perhaps 0.5-2pp.

**Plan.** New WF splits will use a 6-month embargo and use a *purged*
splitter à la López de Prado. Test OOS numbers will be on the embargoed
splitter.

### 🟨 M2. Round-trip cost may be optimistic for small/illiquid
**Evidence.** `cost_bps=5` per trade (10bp round-trip). Reasonable for
mega-cap liquid names; aggressive for smaller index names where bid-ask +
impact at any meaningful AUM is 25–80bp round-trip.

**Plan.** Sensitivity sweep at 5 / 10 / 25 / 50 bp round-trip. Headline
uses 10bp (one-way 5bp) but report all four. Capacity estimate downstream
will scale slippage with ADV.

### 🟩 OK
- Feature parquets are computed strictly from `panel.loc[panel.index <= asof]`.
  No look-ahead in feature definitions.
- ETF / benchmark exclusion is enforced (`SPY/QQQ/IWM/VTI/RSP/DIA/BTC/ETH`).
- Synthetic delisting overlay is documented and disclosed.
- Eligibility requires ≥ 252 trading days of prior history, killing the
  obvious "newly listed" leak.
- XIRR is correctly money-weighted; benchmarks use the same dates.

## Summary

The engine is good enough to test new ideas on, with these overlays:
1. **Headline CAGR** = α=4%/yr MC bias-corrected.
2. **OOS metric of record** = walk-forward with **6-month embargo**.
3. **Execution price** = next-day open (T+1) for the new strategy.
4. **Generalization** = frozen ticker holdout + last 18-month time holdout.
5. **Slippage sensitivity** = 5 / 10 / 25 / 50 bp round-trip.

These overlays apply equally to baseline and to the new strategy. The
*relative* edge is what we're measuring; absolute numbers are bias-haircut.

We proceed.
