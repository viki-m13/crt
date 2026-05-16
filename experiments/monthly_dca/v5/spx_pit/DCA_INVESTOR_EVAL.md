# The DCA-investor evaluation — the honest version of "100% hit rate"

**Date:** 2026-05-16 · **Branch:** `claude/monthly-stock-picker-aWQ8h`
**Code:** `dca_investor_eval.py`, `dca_investor_chart.py`
**Data:** PIT-correct, audited, **no parameter tuning of any kind**.
Streams: `v5_winner_equity.csv` (deployed K=2 rule-based picker, net of
10 bps), `v5_mn_sleeve_returns.csv` (market-neutral sleeve),
`monthly_returns_clean.parquet` SPY column. Window **2003-01 → 2026-02
(254 months)**.

## Why this report exists

Every other metric in this repo is **lump-sum** (CAGR / Sharpe /
walk-forward). None of them describe what the actual target user — a
person who contributes a fixed amount every month — experiences. For a
DCA investor the relevant "hit rate" is **not** the monthly win rate
(~55%, basically a coin flip). It is: *over a rolling H-month
accumulation, did identical monthly contributions into the picker end
up worth more than the same contributions into the S&P 500?* That had
never been measured here. Now it has.

## Headline — your "near-100% hit rate" intuition is correct, but only at long horizons

Rolling monthly-DCA, every possible start month, win = beat DCA-into-SPY:

| DCA horizon | v5 win rate | median v5 ÷ SPY | **worst v5 window ÷ SPY** | 60/40 win rate |
|---|---:|---:|---:|---:|
| 1 year   | 71.2% | 1.15× | 0.35× | 73.2% |
| 2 years  | 80.5% | 1.30× | 0.33× | 82.2% |
| 3 years  | 88.1% | 1.57× | 0.35× | 90.4% |
| 5 years  | 91.8% | 2.37× | 0.37× | 93.3% |
| **10 years** | **100.0% (135/135)** | **6.24×** | **2.89×** | **100.0%** |

**A disciplined 10-year monthly-DCA investor in the deployed picker beat
the equivalent S&P 500 DCA in literally every one of the 135 historical
10-year windows in PIT data — by a median of 6.2× and never by less than
2.9×.** That is the honest, validated, survivorship-bias-free version of
"100% hit rate." It is real. It just requires a 10-year holding
commitment, which is exactly what a DCA accumulator is for.

## The parabola is also real

Full-history: contribute \$1 every month from 2003-01 to 2026-02
(\$254 total in):

| Stream | Ends worth | × money in | Money-weighted IRR | Max portfolio drawdown | Worst point vs cash in |
|---|---:|---:|---:|---:|---:|
| **v5 picker** | **\$103,584** | **407.8×** | **46.8%** | **−72.1%** | **−54.8%** |
| 60/40 v5+SPY | \$23,492 | 92.5× | 35.3% | −51.5% | −38.0% |
| SPY | \$1,133 | 4.46× | 12.6% | −38.6% | −33.3% |
| MN sleeve | \$1,346 | 5.30× | 13.9% | **−17.7%** | **−6.2%** |

10-year rolling: median **12.9× money-in**, p05 **7.3×**, worst-ever
**5.7×**, median IRR **48.8%**. This is genuinely parabolic and it
survives PIT correction.

## The honest part you cannot engineer away: "almost no downside" is false for the picker

The same data that vindicates the long-horizon hit rate **refutes**
"almost no downside" at short/medium horizons:

- Worst **1-year** DCA window ends at **0.34× money in** (a −66% loss on
  contributions) while the SPY-DCA investor was roughly flat.
- The accumulating portfolio had a **−72% peak-to-trough** and sat
  **−55% underwater on contributed cash** at the 2008 / 2022 troughs.
- **Every** catastrophic 1–5y window terminates at the **2008 GFC
  bottom**. The single enemy of a DCA picker investor is being forced to
  measure (or stop contributing) into a crash.

### Stretch test — "contribution shield" (DOCUMENTED NEGATIVE RESULT)

I tested an honest overlay: route each month's *new* contribution to SPY
instead of the picker whenever the picker's own, already-audited,
trailing-SPY `crash` regime label is on (no new alpha, no look-ahead).
Result: **it does essentially nothing** (1y win 71.2%→70.4%, worst MOIC
unchanged at 0.34×, lifetime 408×→370×). Reason: the regime label fires
only 11 months total and only 3 in 2008; the worst DCA windows are
destroyed by K=2 picks falling while the regime still reads
`recovery`/`normal`. **The existing crash gate does not protect a
short-horizon DCA investor's left tail, and a contribution-routing
overlay on it cannot either.** This negative result is the honest answer
to "can we just bolt on downside protection" — on this data, not with
the signals in this repo.

## The unavoidable trade-off (this is the real deliverable)

| You want… | Honest product | Cost of the honest version |
|---|---|---|
| **Parabolic** + 100% long-horizon hit | **v5 picker** | −72% drawdowns; brutal 1–3y windows; needs ≥10y commitment |
| **Almost no downside** | **MN sleeve** | only 5.3× vs SPY 4.5× lifetime — does **not** "substantially outperform" |
| **Most of both** | **60/40 v5+SPY** | 92× money-in, 100% 10y win, but still −51% drawdown |

You cannot have parabolic **and** no-downside on this data. The repo
already proved (Phases 11/A/B) there is exactly **one** independent
OOS-robust alpha here; this DCA analysis shows the same wall from the
investor's side. The right product decision is to **sell the 10-year
DCA promise honestly** (100% historical win, 6× S&P, with explicit
−70% interim-drawdown disclosure) — not to claim no-downside.

## Stretch exploration — what a genuine improvement would actually require

The honesty bar was set to "honest + stretch exploration." None of these
are claimed as results; each is scoped with its real dependency.

1. **Regime-conditional concentration.** Worst windows are GFC-terminal.
   A *forward-looking* crash detector good enough to de-risk new
   contributions before the drawdown (not the current lagging gate)
   would directly attack the only real DCA failure mode. Dependency: a
   crash-timing signal with OOS-positive lead — the repo's own evidence
   (regime sweeps in `IMPROVEMENTS.md`) says trailing price gates don't
   provide it. This is a research program, not a knob.
2. **Options collar on the accumulating book.** A rolling protective put
   would cap the −72% DD near −30%. Dependency: per-name (K=2) historical
   options-chain data, which is **not in this repo** (price-only). Honest
   estimate: ~2–4%/yr premium drag → trims the parabola by roughly a
   third while removing most of the left tail. Scope only.
3. **Second independent alpha (PIT fundamentals / earnings revisions).**
   Phase B's verdict stands: the only honest path past the ~1.2-Sharpe
   ceiling is a genuinely uncorrelated alpha family. Dependency: a
   point-in-time fundamentals source (Compustat/SimFin-class), a
   separate data acquisition. Two independent ~1-Sharpe sleeves combine
   to ~1.4, not 2.0 — even the stretch ceiling is honest, not parabolic.

## Files

- `dca_investor_eval.py` — simulator (rolling + full-history + shield)
- `dca_investor_chart.py` — `augmented/dca_investor_eval.png`
- `augmented/dca_investor_eval.json` — all metrics
- `augmented/dca_rolling_H{12,24,36,60,120}.csv` — every window
