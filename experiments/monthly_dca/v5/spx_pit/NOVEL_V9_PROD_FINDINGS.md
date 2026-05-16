# Novel-v9 consensus — PRODUCTION-harness re-validation (honest, mixed)

**Date:** 2026-05-16 · `novel_v9_prod_validate.py`. Real pipeline:
PIT membership + Chronos p70 filter + inv-vol cap + production tight
regime gate + rule-based min-6m/score-drift + 10bps, both scorers on
identical inputs. The `ml_3plus6` arm reproduces the canonical/live
numbers exactly (CAGR 40.4%, Sharpe 0.87, MaxDD −81%, 100% 10y,
2021–26 loses) — so the comparison is trustworthy.

## Result

| Metric | Deployed `ml_3plus6` | `consensus` (v9) | Verdict |
|---|--:|--:|---|
| Full CAGR | 40.4% | **47.3%** | ✅ +6.9pp |
| Sharpe | 0.87 | **0.94** | ✅ +0.07 |
| Max DD | −81.2% | **−69.1%** | ✅ +12pp better |
| WF splits beat SPY | 8/10 | 8/10 | ➖ tied |
| DCA win 3y / 5y | 80% / 86% | **85% / 89%** | ✅ better |
| DCA win **10y** | **100%** | **98.7%** | ❌ slightly worse |
| Eras beating S&P-DCA | 3/4 | 3/4 | ➖ tied |
| MC delist α=4% / α=8% | 24.9% / −9.6% | **29.4% / −0.9%** | ✅ more robust |

Era-by-era (money-weighted IRR vs S&P-DCA):

| Era | Deployed | Consensus | S&P-DCA |
|---|--:|--:|--:|
| 2003–2009 | +84% | **+152%** | ~0% |
| **2010–2015** | **+28%** | **+0%** ❌ | +11% |
| 2016–2020 | +40% | +37% | +16% |
| **2021–2026** | +12% (loses) | **+23% (beats)** ✅ | +17% |

## Honest read — the isolation test oversold it

In the simplified isolation harness consensus looked like a clean
uniform win (4/4 eras, 100% 10y). On the **real** pipeline it is a
**genuine net improvement but NOT a strict Pareto win**:

- **Wins:** materially higher CAGR & Sharpe, **−12pp shallower max
  drawdown**, much better synthetic-delisting robustness, better 3y/5y
  DCA win, and it **does fix the live-relevant recent era** (2021–26
  now beats S&P-DCA).
- **Honest costs:** it does not eliminate era weakness — it *moves* it
  from 2021–26 to **2010–2015**, where it regresses hard (+0% vs the
  deployed +28%, losing to S&P-DCA there). And the perfect
  10-year-DCA-win drops from a literal **100% → 98.7%** — material
  because the live homepage headlines the 100%.

## Recommendation

This is the strongest, most defensible improvement found — and it
genuinely targets the recent-era problem on the production harness,
with a big drawdown/robustness bonus. But it is a **risk-preference
trade**, not a free win: better aggregate + recent era, at the cost of
the 2010–15 era and the literal-100% headline. Deployment changes a
live financial product's public numbers (incl. the 100% claim → 98.7%),
so this is a user decision, not an automatic ship. No website /
data.json / STRATEGY_SPEC changes were made in this commit.
