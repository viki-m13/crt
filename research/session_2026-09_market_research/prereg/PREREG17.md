# PRE-REGISTRATION v17 — VARIANCE HARVEST VIA LEVERAGED-ETF DECAY
Frozen 2026-09-05 before any v17 number is computed. Trial 47.

## The reasoning (the actual invention)
46 prior hypotheses tried to forecast RETURNS and failed. But this session
MEASURED that VARIANCE is forecastable: HAR-RV out-of-sample R2 = 0.43-0.56
(PREREG13/V1). Volatility clustering is the strongest regularity in finance.
So: stop forecasting returns. Find an instrument whose PAYOFF IS LINEAR IN
VARIANCE with zero directional exposure, and size it by a variance forecast.

Daily-rebalanced L-times ETF, log return over T:  L*R - L(L-1)/2 * sigma^2 * T
Short the 3x, long 3 units of the underlying:
   -(3R - 3*sigma^2*T) + 3R  =  +3*sigma^2*T
Beta cancels ALGEBRAICALLY. What remains is realized variance. Not a forecast
of direction, not an empirical regularity -- an identity of daily rebalancing.
No options. Alpaca-tradeable ETFs.

## Why this is not just "short leveraged ETFs"
Shorting a 3x alone is a levered SHORT BETA bet and loses in bull markets.
The hedged pair is delta-neutral. The novel increment is sizing the pair by a
VARIANCE FORECAST, converting the one forecastable quantity in finance into a
directionally-neutral P&L stream.

## Pairs (7 independent tests, fixed now)
 (TQQQ,QQQ,3) (QLD,QQQ,2) (SPXL,SPY,3) (UPRO,SPY,3) (SSO,SPY,2)
 (SOXL,SMH,3) (TNA,IWM,3)
TRAIN through 2019-12-31 | VALID 2020-01-01..2023-12-31 | TEST 2024+ LOCKED.

## THE COSTS THAT DECIDE IT (stated before results)
This trade's whole risk is that its costs eat the identity:
 C1 BORROW/SHORT FEE on the leveraged ETF. Real and variable. I do NOT have
    borrow-rate data. I will therefore report the BREAK-EVEN BORROW RATE --
    the annual fee at which the edge dies -- and compare it to plausible
    ranges (TQQQ ~0.3-3%/yr; less liquid 3x can be 5-20%/yr).
 C2 REBALANCING: the hedge drifts as prices move. Test daily and weekly
    rebalance; charge the measured spread each time.
 C3 The identity is an APPROXIMATION that degrades in large moves. Report the
    residual between realized pair return and 3*sigma^2*T.

## Pre-registered tests
 L1 Does the pair return match the theoretical L(L-1)/2 * sigma^2? Regress
    realized pair return on realized variance; slope should be ~L(L-1)/2,
    intercept ~0. If the identity does not hold empirically, everything else
    is void.
 L2 Gross Sharpe and return of the delta-neutral pair, per pair, TRAIN/VALID.
 L3 BREAK-EVEN BORROW RATE per pair. This is the headline number.
 L4 VARIANCE-TIMED version: scale exposure by HAR-RV forecast. Must beat the
    constant-size version, else the forecast adds nothing.
 L5 TAIL: worst day/week, max drawdown. A delta-neutral pair can still blow up
    on a gap or a borrow recall.

## Kill
L1 fails -> the mechanism is not real, dead.
Break-even borrow < 3%/yr -> not deployable, since real borrow can exceed it.

## RESULTS (2026-09-05)
L1 IDENTITY HOLDS. Regression of (L*R_under - R_lev) on realized variance:
   slopes 2.941 / 2.973 / 3.109 / 3.173 / 3.656 (theory 3.0) and 1.213 / 1.312
   (theory 1.0). Intercepts ALL POSITIVE, +1.54% to +8.16%/yr = the ETFs'
   expense ratio + embedded leverage financing, also captured by the short.
   7 pairs, 3,475-4,411 days each, 2006-2023. The mechanism is arithmetic.

L2 DELTA-NEUTRAL PORTFOLIO (equal weight, 7 pairs), per $1 short notional:
   TRAIN 2010-2019  ann +2.45%  vol 3.85%  Sharpe +0.64  maxDD -3.4%
   VALID 2020-2023  ann +5.95%  vol 2.72%  Sharpe +2.19  maxDD -1.5%
   Pairs are near-independent (TQQQ/SPXL corr 0.07, UPRO/TQQQ -0.08, SOXL ~0),
   which is why the portfolio Sharpe exceeds every individual leg.

L3 BREAK-EVEN BORROW: 2.45%/yr TRAIN, 5.95%/yr VALID (portfolio).
   Per pair 0.76%-13.49%. Typical real borrow on TQQQ/SPXL/UPRO is ~0.3-2%/yr,
   so the edge likely survives -- BUT I HAVE NO BORROW DATA AND CANNOT VERIFY.
   This is the single biggest unknown and it is a data gap, not a result.

L4 VARIANCE TIMING FAILED. Scaling by a HAR-RV forecast raised return
   (+5.95% -> +7.04%) but raised vol more: Sharpe 2.19 -> 1.14 (VALID),
   0.61 -> 0.43 (TRAIN). The constant-size version is better. My novel
   increment did not work; the identity does the work, not the forecast.

L5 TAIL: worst day -2.63%, worst month -1.28%, worst quarter -1.78%,
   only 15 of 4,411 days (0.34%) lose more than 1%. Exceptionally benign.

## CAPITAL EFFICIENCY -- THE REAL LIMITATION
Reg-T needs ~$2.9 of capital per $1 of short notional (150% short + 50% long L).
Return ON CAPITAL: +0.86%/yr TRAIN, +2.08%/yr VALID.
The Sharpe is excellent; the unlevered return on capital is small.
Also: it earns almost nothing in calm markets (+0.71%/yr in 2021) because it is
paid in variance. Returns are structurally lumpy.

## VERDICT: the best of 47 trials. Real, mechanical, delta-neutral, benign tail.
NOT "extremely profitable" unlevered. Whether it becomes so depends entirely on
(a) portfolio-margin treatment and (b) actual borrow cost -- neither verifiable
from the data I hold.
