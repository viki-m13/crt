# WaveTrend on PIT S&P 500 / NDX — findings

**Separate experiment**, not wired into anything deployed. Reproduce with
`python3 experiments/wavetrend/run_wavetrend_pit.py` (~6 min, CPU).

## What was tested

The repo-root `wavetrend` file is a WaveTrend (Pine `wt1/wt2`) pyramiding
strategy with an RSI exit, optimised by Bayesian search to **maximise trade
win-rate**, on a hand-picked yfinance universe.

Three problems were fixed before trusting any number:

1. **Look-ahead universe bug.** The original picks "underperformers vs SPY"
   using *full-sample final equity* — it cannot be known at trade time.
   Replaced with **PIT index membership** (S&P 500 and Nasdaq-100) from the
   repo's existing point-in-time data. A name is only tradable while it is an
   index member; it is force-exited on removal/delisting.
2. **Survivorship.** Runs on `prices_extended_pit.parquet` (1994 tickers,
   PIT-corrected). Caveat: panel is **adjusted close only**, so Pine's
   `ap = hlc3` is approximated by close. 10 bps round-trip cost applied.
3. **The win-rate objective is a trap.** It rewards "never sell at a loss",
   producing a strategy that barely trades and badly lags the market.
   Replaced with a risk-adjusted objective `Sharpe + 0.5·CAGR + 3·min(0,
   MDD+0.45)` with a ≥40-trade floor, **selected only on a train window**
   (S&P 500 2003–2013; NDX 2015–2020), holdout never used for tuning. The
   naive win-rate run is kept side-by-side as the control.

## Result 1 — the win-rate objective is empirically a trap (S&P 500 PIT)

| | Honest risk-adjusted | Naive win-rate (original) | SPY |
|---|---:|---:|---:|
| Full CAGR | **17.6%** | 4.3% | 11.7% |
| Sharpe | **1.06** | 0.78 | — |
| Holdout CAGR (2014+) | **22.8%** | 7.1% | — |
| Holdout Sharpe | **1.29** | 0.97 | — |
| Trades | 185 | 16 | — |
| Win rate | 69.7% | **87.5%** | — |
| WF splits beating SPY | **10/10** | 2/10 | — |

The 87.5%-win-rate config trades 16 times in 23 years and returns 4.3%/yr —
the exact failure the original objective invites. The honest config's
**holdout Sharpe (1.29) exceeds its train Sharpe** → not overfit.

## Result 2 — it does NOT beat the deployed strategy standalone, and NDX fails

- Honest WaveTrend SP500: 17.6% CAGR vs deployed **v5 ≈ 39.8%**. Not a
  replacement.
- NDX PIT (2015–2026, only 137 months): honest config holdout CAGR **2.4%**,
  Sharpe 0.46, **0/8** vs QQQ. A buy-the-dip mean-reverter structurally lags
  a momentum mega-cap index (QQQ 19.4%). **WaveTrend does not generalize to
  NDX** — reported honestly, not hidden.

## Result 3 — the real value: a near-orthogonal diversifying sleeve for the SP500 book

WaveTrend's monthly returns vs the deployed v5 stream
(`augmented/v5_winner_equity.csv`, 253 common months):

- **corr = 0.054** full; max |corr| across WF splits 0.38 (stable).
  This is a *lower* correlation than every existing diversifier the repo has
  banked — carry (0.23), quality second-sleeve (0.19), mn-composite (−0.19) —
  and unlike mn-composite it has a **positive standalone Sharpe (1.06)**.

Static v5 + WaveTrend blends (same metric code as the v5 winner):

| blend | CAGR | Sharpe | Max DD | WF mean Sharpe | WF min Sharpe |
|---|---:|---:|---:|---:|---:|
| v5 alone | 39.8% | 0.91 | −77% | 1.20 | 0.62 |
| 80% v5 + 20% wt | 37.5% | 1.00 | −69% | 1.35 | 0.68 |
| **70% v5 + 30% wt** | **35.9%** | **1.05** | **−65%** | **1.45** | **0.72** |
| **60% v5 + 40% wt** | **34.1%** | **1.11** | **−61%** | **1.56** | **0.77** |
| 50% v5 + 50% wt | 32.0% | 1.19 | −57% | 1.69 | 0.78 |

(Max DD is on the raw monthly stream; the deployed book reports the
accumulating-DCA DD which is shallower — directionally the same.)

## Recommendation

Do **not** deploy WaveTrend standalone and do **not** touch the deployed E2
strategy. The actionable finding is the sleeve: a **~30–40% WaveTrend /
60–70% v5** mix on the S&P 500 book lifts Sharpe 0.91 → 1.05–1.11, raises
WF-min Sharpe 0.62 → 0.77, and cuts the tail ~−77% → ~−61%, costing ~4–6pp
CAGR. It sits on the same orthogonal-sleeve lever the deployed E2 already
uses, and is the strongest low-correlation, positive-Sharpe diversifier
found so far. Natural next step (not done here): walk-forward-validate the
blend weight and stack it with the existing carry/quality sleeves rather
than v5 alone.

### Caveats
- Close-only proxy for `hlc3` (no intraday H/L in the PIT panel).
- 213 OTC bankruptcy-"Q" names still missing → residual tail-risk
  understatement (same limitation as the rest of the PIT work).
- NDX did not generalize — the edge is S&P-500-specific here.
- Blend weight shown is a full-sample read; treat the 30–40% band as
  indicative until walk-forward-validated.

---

# Part 2 — How high can win-rate go *honestly*? (creative trade filters)

`explore_winrate.py` adds a creative filter battery and maximises win-rate
**subject to hard floors** (train ≥50 trades, CAGR ≥ max(6%, 0.6·SPY),
MaxDD ≥ −55%), selected train-only, holdout never tuned. Filters: own-trend
SMA gate (no falling knives), market SMA gate, relative-strength gate,
WaveTrend turn-confirmation, per-lot profit-take / stop / time exits, and a
**no-pyramiding** mode (≤1 open lot per name — don't stack losers, the idea
from the user's variant scripts).

**Best honest high-win-rate recipe** (train-selected):
`no_pyramid=1, profit_take≈15%, stop≈48%, trend_sma=195, confirm=1,
shallow oversold≈−41, short WT 34/31`.

| | Train | **Holdout 2014+** | Full |
|---|---:|---:|---:|
| **Win rate** | 88.9% | **85.9%** | 87.0% |
| Trades | — | — | 2 994 |
| CAGR | — | 5.3% | 5.6% |
| Sharpe | — | 0.80 | 0.71 |
| Max DD | — | — | −30% |

Exit mix: 2 589 profit-takes, 260 stops, 128 delists. The **86% win-rate is
real out-of-sample** (holdout 85.9% ≈ train 88.9% → not overfit). The
filters that work: **no-pyramiding + a tight profit-take + an own-trend SMA
gate**. Market/RS gates and time-stops did *not* survive selection.

### The honest catch (the win-rate↔CAGR frontier)
A robust 87% win-rate is achievable — but it is a **deliberately
negative-skew** design: a 15% profit-take caps every winner while a 48%
stop lets losers run, so CAGR is only **5.6%** (loses to SPY's 11.7%).
Of 60 configs only 4 cleared the floor, and they trace a clean monotone
frontier — **you buy win-rate with CAGR**:

| config | win rate | CAGR | Sharpe |
|---|---:|---:|---:|
| max win-rate (no-pyr + 15% take + trend gate) | **87%** | 5.6% | 0.71 |
| mid | 60% | 5.5% | 0.83 |
| max CAGR feasible (50% take + mkt gate) | 58% | **12.9%** | 0.26 |

No feasible config has *both* high win-rate and high CAGR. This **confirms
Result 1 even with smart filters**: filters make a high win-rate *robust*
(OOS-stable), they cannot make it *profitable* — the asymmetric exit that
manufactures the win-rate is the same thing that caps return.

### Where it *is* useful
This high-win-rate stream is the **most orthogonal to deployed v5 found
anywhere**: corr **0.029** full, max split corr **0.18** (vs 0.05/0.38 for
the Sharpe-tuned WaveTrend, 0.23 carry, 0.19 quality). Low vol (8%),
shallow DD (−30%). As ultra-low-correlation ballast a 30/70 v5+wt blend
gives Sharpe 1.10 / MaxDD −41% — but because it contributes little return
it dilutes CAGR more than the Sharpe-tuned sleeve and its WF-min Sharpe
(0.68) is below that sleeve's (0.78).

**Bottom line:** "very high win-rate" is solved — `no_pyramid + ~15%
profit-take + trend-SMA gate` gives a robust ~86% OOS hit-rate. But it is
a low-return negative-skew profile by construction; the Part-1
Sharpe-tuned WaveTrend (≈70% win, 17.6% CAGR) is still the better
all-round sleeve. Use the win-rate config only as low-correlation ballast
or a "clip-the-bounce" overlay, never as the return engine.

---

# Part 3 — "Never sell": WaveTrend as a pure entry signal

Idea (user): never sell. WaveTrend only decides what/when to buy; every
equal-$ unit is held forever (a delisted/acquired name is frozen at its
last price — the repo's honest cash-payout convention, not a forced loss).
Never selling is itself the clean resolution of the win-rate/return
tension: winners run unbounded, so the entry hit-rate is structurally
high **without** a profit-take capping CAGR.

Optimised with a deliberately small parameter set (n1, n2, sig_len,
oversold, one optional trend-SMA gate — fewer knobs = less to overfit),
train-only on S&P 500 2003–2013, then put through a 5-test
generalisation gauntlet. Selected: `n1=99, n2=40, sig=8, oversold=−30,
trend_sma=210`.

| S&P 500 never-sell book | CAGR | Sharpe | Max DD | entry win-rate |
|---|---:|---:|---:|---:|
| Optimised — full | 13.3% | 0.93 | −47% | **79.9%** |
| Optimised — train 03–13 | 11.8% | 0.85 | −47% | — |
| **Optimised — holdout 14–26** | **14.6%** | **1.00** | **−23%** | — |
| Default WT — full | 13.2% | 0.51 | −93% | 73.2% |
| Default WT — holdout 14–26 | 2.2% | 0.54 | −85% | — |
| SPY — full | 11.7% | 0.84 | −51% | — |

The entry win-rate is **~80%** (every individual buy-and-hold-forever
unit, marked at last price) **while still keeping a 14.6% holdout CAGR
and Sharpe 1.00** — the opposite of Part 2's 87%-win / 5.6%-CAGR
negative-skew trap. This is the honest high-win-rate answer: *don't
engineer the exit, remove it.*

### Generalisation gauntlet (none of this was used for tuning)

- **G1 time-OOS:** holdout (Sharpe 1.00, DD −23%) is *better* than train
  (0.85, −47%) → not overfit in time.
- **G2 walk-forward:** **9/10** splits beat SPY.
- **G3 cross-universe (strongest test):** the S&P-500-trained params,
  **unchanged**, on the PIT Nasdaq-100 (2015+) → CAGR 14.5%, Sharpe 0.79,
  DD −33%, 66% entry win-rate. The params *transfer* to a completely
  different universe and still produce a robust positive, shallow-DD
  book — but they do **not** beat QQQ (19.4%, only 1/8 splits) because a
  buy-the-dip-and-hold book structurally lags a momentum mega-cap index.
  Honest read: the *strategy* generalises; the *edge over the benchmark*
  is S&P-500-specific (same structural finding as Parts 1–2).
- **G4 parameter plateau:** perturbing every parameter (±15 on n1, ±40 on
  n2/trend_sma, ±2 on sig_len, ±15 on oversold) keeps CAGR 10–13% and
  Sharpe 0.73–0.93 — a broad plateau, not a fragile spike → robust, not
  curve-fit.
- **G5 optimised vs default:** default WT never-sell is a disaster OOS
  (2% holdout CAGR, −85% DD). Optimisation — essentially discovering the
  **trend-SMA(210) gate** (only buy dips that are still above the long
  trend) — adds large, real OOS value, not fitted noise.

### Sleeve vs deployed v5
corr 0.071 (max-split 0.26). A 50/50 v5+never-sell blend → Sharpe **1.13**
(v5 alone 0.91), Max DD **−57%** (−77%), WF-min Sharpe **0.76** (0.62) —
on par with the Part-1 Sharpe-tuned sleeve and a better return
contributor than the Part-2 win-rate stream.

### Bottom line
**This is the best of the three approaches.** "Never sell" turns
WaveTrend into a robust, well-generalising **signal-timed accumulation**
of S&P 500 members: ~80% entry win-rate, 14.6% holdout CAGR, Sharpe 1.00,
−23% holdout DD — beating SPY on Sharpe and drawdown, passing time-OOS,
walk-forward, parameter-plateau and (for robustness, not edge)
cross-universe tests. The single load-bearing ingredient is the
long-trend gate; without it the never-sell book is uninvestable
(−85% DD). It is also a clean ~0.07-correlation diversifier for the
deployed book. Caveat: it does not beat a momentum index (NDX/QQQ), and
it is a return/Sharpe story, not the v5-level (~40% CAGR) engine.

## Files
- `wavetrend_pit.py` — leakage-free sim + indicators + metrics (Part 1 +
  filtered/no-pyramiding sim Part 2 + never-sell sim Part 3)
- `run_wavetrend_pit.py` — Part 1 train/holdout optimiser + sleeve study
- `explore_winrate.py` — Part 2 constrained win-rate explorer + frontier
- `hold_forever.py` — Part 3 never-sell optimiser + generalisation gauntlet
- `wavetrend_pit_results.json`, `winrate_explore_results.json`,
  `hold_forever_results.json` — metrics
- `winrate_frontier.csv` — every win-rate config (win-rate/CAGR frontier)
- `*_monthly_returns.csv`, `*_trades.csv`, `wt_equity.png`
- `run.log`, `winrate.log`, `holdfwd.log` — full optimisation traces
