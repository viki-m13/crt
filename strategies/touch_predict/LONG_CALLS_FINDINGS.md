# Long-Call Variant of Option C — Walk-Forward Findings

## Question

The Option C engine (event-triggered conditional credit spreads on
touch-prediction regimes) is built around SELLING puts on oversold
fires. The call-side regimes (`connors_tps`, `multi_stack`,
`panic_day`, `spy_rel_weak`, `deep_oversold`, `plain`) identify stocks
that are oversold and historically rally — exactly the setup where
BUYING calls instead of selling puts could dominate.

This study replaces the short-put-spread payoff with a long-call
payoff on the same fires and measures pooled walk-forward
performance.

## Universe and Setup

* Universe: 94 liquid names (S&P 100 large-caps + major ETFs) — same
  as Option C.
* Fold years: 2020–2025 (one fold per calendar year).
* Pricing: Black-Scholes call premium with `IV = realized_vol(60d) *
  1.15`, post-slippage 1.05× (1.30× if BS price < $0.10), $0.02
  minimum tradable premium.
* Exit models tested:
  * **Hold-to-expiry** — payoff = `max(close[t+h] - K, 0) - premium`
  * **Touch-target-and-exit** — sell at first touch of `K +
    target_buffer`. Tested target buffers: 5/10/15/25/50%.
  * **Half-and-half** — sell half at touch of target, hold half to
    expiry.

## Headline Result — `connors_tps`, 252-day call, 20% OTM

Pooled across the entire universe, 2020–2025, hold-to-expiry:

| metric                              |     value |
| ----------------------------------- | --------: |
| Number of trades                    |     341   |
| Win rate                            |    37.2%  |
| % of trades ≥ +100% on premium      |    33.1%  |
| % of trades ≥ +300% on premium      |    25.2%  |
| Average premium paid per share      |    $9.62  |
| **Pooled ROI on premium**           | **+95.5%**|
| Median trade ROI on premium         |    -100%  |
| 90th-percentile trade ROI           |    +747%  |
| Max single-trade ROI                |  +3,812%  |
| Losing fold-years (out of 6)        |       2   |

In plain English: **most trades expire worthless, but the 1-in-3 that
hits delivers a 5–40× return, and the average $1 of premium turns
into $1.95 over the 6-year window.** This is a pure long-tail bet —
the median trade is a 100% loss on premium.

## The Right Tail Is the Strategy — Don't Cap It

The `touch-target-and-exit` exit model destroys the strategy:

| target buffer | ROI on premium |
| ------------: | -------------: |
|        +5%    |   -52.9%       |
|       +10%    |   -18.1%       |
|       +15%    |    +6.3%       |
|       +25%    |   +40.4%       |
|  hold-to-expiry | **+95.5%**   |

Selling into a small +5% rally clips the very fat-tail returns that
make the strategy work. The half-and-half exit at +25% buffer is a
reasonable compromise: **+67.9% ROI** with smoother realised P&L
than pure hold-to-expiry.

## Walk-Forward by Year (best rule)

```
year   n   ROI%    win%    notes
2020   78  +262%   55%     covid-rebound
2021  106   -47%   15%     mid-year stalls (ungated)
2022   14   -94%    0%     bear year — fires were mostly mistakes
2023   41  +304%   49%     post-2022 rebound
2024   81   +60%   42%     mixed
2025   21  +167%   67%     trend resumption
```

Two losing years out of six. One catastrophic (2022, -94%). The other
four years deliver +60% to +304% ROI on premium.

## Alpha vs. Unconditional ("plain") Calls

The mean-reversion regimes have to beat just buying calls every day
on every name. Top alpha rows (regime ROI minus plain ROI at same
horizon × OTM):

| regime         |   h | OTM% | regime ROI | plain ROI |  alpha |
| -------------- | --: | ---: | ---------: | --------: | -----: |
| connors_tps    | 252 | 20%  |    +95.5%  |   +39.3%  | +56.2% |
| connors_tps    | 252 | 15%  |    +91.5%  |   +44.6%  | +46.9% |
| multi_stack    | 252 | 20%  |    +68.7%  |   +39.3%  | +29.4% |
| connors_tps    | 252 | 10%  |    +86.1%  |   +48.5%  | +37.6% |
| multi_stack    | 180 | 20%  |    +55.1%  |   +15.5%  | +39.6% |
| connors_tps    | 180 | 20%  |    +51.9%  |   +15.5%  | +36.4% |

The conditional fires materially outperform unconditional buying. The
+39% baseline ROI on plain 252d 20%-OTM calls is largely just
universe beta — these are mega-caps over 2015-2026.

## What the Macro Gate Doesn't Buy You

Option C's existing macro gate (SPY ≥ 200-SMA on fire date) was
tested for the long-call variant. It strips out about 15% of fires,
modestly reduces the bull-year ROI, and barely helps in bear years:

```
rule                          ungated     gated
connors_tps-h252-otm20%        +95.5%    +73.5%
connors_tps-h252-otm10%        +86.1%    +68.8%
multi_stack-h180-otm10%        +55.2%    +46.9%
```

The gate strips good 2023 and 2025 fires (which fired AFTER SPY broke
the 200 SMA but were still profitable) without saving 2022. **Don't
use it for the long-call variant** — it just reduces sample size and
shaves alpha.

## What Doesn't Work

* **Short-dated calls** (≤21 sessions): consistently negative ROI —
  theta dominates the payoff window. Best rule at h=21 is plain at
  -1.5%; best h=14 is -0.6%.
* **Far-OTM short-dated lottery tickets** (panic_day h=7, OTM 20%):
  +103% alpha on paper, but median trade loses 100% on premium and
  the rule has 6 losing fold-years out of 7. Looks great pooled,
  unbearable per-fire.
* **`deep_oversold` and `panic_day`** at long horizons: only +33–41%
  ROI; weakest of the long-dated set.

## Concentration Risk

For the best rule, the per-ticker ROI is highly skewed:

```
ticker    n  ROI%      max-trade-PnL
GLD       9 +920%     $106
NVDA      1 +843%       $6
XLC       6 +746%      $14
QQQ       8 +492%      $61
AVGO      7 +328%     $174
```

Even within "1-in-3 hits," the magnitudes vary by 20×. A real
deployment must size every fire identically (e.g. 0.5–1% of capital)
and let the law of large numbers do the work. Sizing on conviction
will overweight a few names that may not repeat.

## Recommendations for a Production Variant

1. **Pick one rule to start.** `connors_tps` at 252-session horizon
   and 10–20% OTM strike is the most-validated cell. The 10% OTM
   version has +86% ROI with a 49% win rate (vs. 37% at 20% OTM) —
   that's a smoother ride and probably better psychological fit.

2. **Hold to expiry by default.** OR use the half-and-half +25%
   model: pre-set a GTC sell on half the position at strike + 25%.
   Either keeps the right tail; do not use a tight touch-target.

3. **Equal-size every fire.** Recommend 0.5–1% of capital per fire so
   the inevitable 50%+ wash-out trades don't destroy the account.
   Expect 10–30 fires per year on this rule across the universe.

4. **No macro gate.** It costs alpha without protecting drawdowns.

5. **Be honest about the variance.** With ~40 fires per year and a
   60%+ loss rate per fire, even +95% pooled ROI on premium will see
   YEARS of -100% (2022). Position-size for the 95th-percentile
   drawdown, not the average return.

## Files

* `option_c_long_calls.py` — full sweep across regime × horizon × OTM strike.
* `option_c_long_calls_v2.py` — top-12-rules deep-dive with multiple exit models and per-ticker breakdown.
* `option_c_long_calls_v3.py` — macro-gate test.
* `results/option_c_long_calls.json` — pooled rule results.
* `results/option_c_long_calls_v2.json` — exit-model + per-ticker results.
* `results/option_c_long_calls_v3.json` — gated vs. ungated.
