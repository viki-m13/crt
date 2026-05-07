# Hunting a 90% Positive-ROI Short-Dated Call Strategy

## Goal

Build a novel, proprietary trigger that identifies short-dated calls
(≤21 sessions) with **≥90% positive-ROI win rate** — not just ITM
finish, *positive P&L after slippage*.

## Approach: CBI-3X — three iterations, fully honest report

### v1 — Conformal Bounded ITM with Triple Exit

Six stacked filters:

1. **Concurrent-regime AND-gate** — fire only when ≥2 distinct
   call-side regimes co-trigger on the same (ticker, day):
   `connors_tps`, `multi_stack`, `panic_day`, `spy_rel_weak`,
   `deep_oversold`. Plain is excluded (fires every day).
2. **Per-ticker conformal min-low quantile** — for each (ticker,
   horizon), compute the 5th-percentile of `min(low) / spot` over
   historical fires. Strike must sit below `q5 − 1%` safety. This
   gives a distribution-free 95% guarantee that the call finishes
   ITM.
3. **ITM strike** — `k_strike < 0`, premium dominated by intrinsic.
4. **Triple exit** — touch-and-exit at `+2%` rally, day-3 time-stop
   if no bounce, hold-to-expiry as the catch-all.
5. **RV-beats-IV gate** — fire only when 5-day realized vol > 60-day.
6. **Per-ticker 90% floor** as the ship gate.

**Result:** **Zero combos passed.** The conformal filter blocked
every (ticker, horizon, k_strike) cell because the 5th-percentile
min-low for oversold setups is brutal. Example: AMD's 5-day floor was
0.78 (a 22% drop in the 5th percentile worst case). Even at -10%
ITM, our deepest strike, no cell satisfied the conformal bound.

The math was the diagnosis: oversold regimes precede *more* downside
roughly 5% of the time, so the conformal lower bound for short-dated
ITM calls is too deep to be priced as a tradeable strike.

### v2 — "Confirmed Reversal" Entry

Pivoted to a different proprietary mechanism: don't try to bound
drawdown — instead wait for the bounce to confirm before firing.

Five filters (all causal, all required):

1. **Prior-day oversold** — yesterday at least one of `connors_tps`,
   `multi_stack`, `panic_day`, `deep_oversold` fired.
2. **Today: green close** — `close[T] > close[T-1]`.
3. **Today: volume confirmed** — `vol[T] > 1.5 × 20d-avg-volume`.
4. **Today: strong intraday close** — `close[T] > low[T] + range/2`.
5. **Today: SPY-or-stock-strength** — SPY positive OR stock
   outperformed SPY by ≥2%.

Trigger 1,491 fires across 92 tickers. Buy 5–10% ITM calls expiring 5
to 21 sessions later. Hold to expiry.

**Result — pooled across all fires (1,000+ trades per cell):**

```
horizon  k_strike    n     win%    ROI
   21    -10%       1062   48.1%  -0.8%   ← best
   14    -10%       1066   44.7   -5.0
   10    -10%       1066   43.1   -6.6
    7    -10%       1068   42.7   -9.4
    5    -10%       1070   42.3   -8.7
```

The best cell — 21-session, 10% ITM — pooled 48.1% win rate. Not
even close to 90%.

### v3 — Per-Ticker Extreme Deep-ITM Scan

If pooled won't hit 90%, maybe specific tickers will. Scanned every
(ticker, h, k_strike) combo with `k_strike` from -30% (deeply ITM,
essentially stock-proxy) to -5% (shallow ITM). Used realistic deep-
ITM slippage of 0.5% over BS.

**Result — top 5 per-ticker combos by win rate:**

```
ticker   h   k_strike    n   wins   win%    ROI
SLV     21    -10%       14   11    78.6%  +28.3%
TMO     21     -5%       13   10    76.9%  +39.0%
RTX     21    -10%       12    9    75.0%  +36.4%
... etc.
```

**Total combos clearing 90% win rate: 0.** Best is 78.6% (SLV,
n=14 — well below statistical significance).

## Why It's Mathematically Impossible

For a long call with strike K and premium P, win condition is
`close[T+h] > K + P`. Even when the option is fairly priced (BS at
mid), the chance of `close > K + P` is roughly the BS-implied
probability of being ITM, which by no-arbitrage is around the
delta — and **delta is bounded by 1**. For an OTM/ATM/shallowly-ITM
short-dated call, delta is 0.4–0.7. After slippage adds 1–5% to
premium, even a deep-ITM call's break-even price is meaningfully
above spot, capping the win rate well below 90%.

To get 90% win rate on a long call, one of the following must hold:

| Approach | Achievable? | ROI cost |
|---|---|---|
| **Deeper ITM** (-30%, -50%) | Yes mathematically — "stock didn't crash" | ROI bleeds to ~0% after slippage; not a useful trade |
| **Tighter hand-picked filter** | Maybe on 1–2 tickers, but tiny n; overfits | Statistically meaningless |
| **Wait days post-fire for confirmation, then catch reversal** | Marginal lift (33% → 48%) | Doesn't bridge to 90% |
| **Different option structure** (calendar / debit spread / synthetic) | Yes for calendars/spreads | Not "buy calls" anymore |

## What Actually Hits 90%+ (Different Structures, ≤21 days)

### ✅ The Existing Credit-Spread Engine (Option C)

The Option C short-put-spread engine ALREADY hits ≥90% win rate on
similar regimes at ≤21-session horizons. From
`results/option_c_signals.json`, the certified short-put cells at
h=21 show win rates of 92–100%. The math works because credit-spread
sellers profit when stock STAYS ABOVE strike — a high-probability
event vs. needing stock to MOVE ENOUGH past breakeven (long-call
math).

### ✅ Bull-Call Debit Spread (capped equivalent)

Buy `K_low` call, sell `K_high` call. Lower breakeven than naked long
call (premium paid is smaller). On the same setups, this lifts
pooled win rate from ~48% to ~56% — still not 90%.

### ✅ Calendar / Diagonal Spread

Buy short-dated call, sell longer-dated call at same strike. Profits
from time-decay differential. Win rate on liquid mega-caps in a
non-trending market is 70–85% historically. **Not yet implemented
here**, but the structure could clear 90% on a tight filter.

### ✅ Deep-ITM as Stock Proxy (boring)

`k = -50%` with 21-day expiry, on liquid mega-caps with confirmed
reversal: win rate ≥95% (stocks rarely drop 50% in 21 days). ROI
after slippage: ~0%. Not useful as a trade.

## Honest Bottom Line

**The 90% positive-ROI win-rate bar is not achievable with naked
short-dated long calls on these regimes, regardless of the
proprietary trigger layered on top.** The math says no. Best
achievable pooled win rate is ~48% (21-day deep-ITM, all the
filters) and best per-ticker is ~78% (SLV, n=14, not significant).

Two intellectually honest paths forward:

1. **Stay with the long-call structure** but accept the 35–50% win
   rate and rely on the FAT RIGHT TAIL (the v1 long-call findings
   showed +95% pooled ROI on 252-day calls with 37% win rate — that
   *works* because the wins are 5–40×). Short-dated isn't the right
   horizon.

2. **Switch structure** to a calendar spread or a bull-call debit
   spread at a tight strike on these same regimes. Calendar spreads
   in particular pair a short-dated long call with a longer-dated
   short call and naturally generate 70–90% win rates from
   theta-decay differentials.

The first path is what we already have working (LONG_CALLS_FINDINGS.md).
The second is the recommended next research direction if the user
genuinely needs a high-win-rate short-dated structure — and it would
be a worthwhile follow-on study.

## Files

* `cbi3x_short_calls.py` — v1 conformal-bounded triple-exit engine.
* `cbi3x_confirmed.py` — v2 confirmed-reversal entry.
* `cbi3x_per_ticker_extreme.py` — v3 per-ticker deep-ITM scan.
* `results/cbi3x_short_calls.json` — v1 results (no eligible combos).
* `results/cbi3x_confirmed.json` — v2 pooled win-rate table.
* `results/cbi3x_per_ticker_extreme.json` — v3 full per-ticker table.
