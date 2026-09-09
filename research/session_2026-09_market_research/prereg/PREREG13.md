# PRE-REGISTRATION v13 — ENTROPY PRODUCTION AS A VOLATILITY-RISK-PREMIUM SIGNAL
Frozen 2026-09-05 before any v13 number is computed. Trial 40.

## Provenance: this hypothesis was pre-registered in PREREG8 and never run
PREREG8 H4 read: "ATI predicts next-day realized vol beyond rv itself
(information arrival)". It was never executed because PREREG8's N0 gate failed
on the RETURN-prediction side, which voided that batch. The volatility question
is untouched and is a different economic object.

## The idea (never done, as far as I can establish)
PREREG9 established that entropy production sigma measures INFORMATION ARRIVAL,
and that this makes it useless for predicting RETURNS -- arriving information is
already in the price (spearman(bound, achievable) = -0.394, wrong sign).
But information arrival is exactly what drives VOLATILITY (Clark 1973,
mixture-of-distributions). So sigma should forecast VOLATILITY precisely
BECAUSE it cannot forecast direction. Irreversibility has not been used as a
volatility predictor.
Economic use: options have FAT margins (whole vol points), not basis points.
39 prior hypotheses died on an 8-10 bp cost wall. This one does not face it.

## Data
Real option chains: 1,255 weekly snapshots 2019-02-09..2026-08-07, bid/ask/IV/
greeks, 63 tickers. 12 overlap with 1-min equity data:
AAPL AMD AMZN GOOGL META MSFT NFLX NVDA SPY TSLA XLE XLF.
Fills come from REAL quoted bid/ask, never a pricing model.
TRAIN 2019-2022 | VALID 2023-2024 | TEST 2025-01-01..2026-08-07 LOCKED.

## Construction (causal)
sigma_t : ordinal-pattern KL(fwd||rev) over the trailing 10 trading days of
          1-min returns, normalized per day. Constant window => constant
          finite-n bias => valid for ranking and regression. No look-ahead.
IV_t    : ATM implied vol, expiry nearest 30 days, mean of call and put IV at
          the strike closest to spot.
RV_fwd  : realized vol from t to that expiry (annualized, from daily closes).
VRP     : IV_t - RV_fwd, in vol points. Positive = option seller wins.

## Pre-registered tests, in order
 V1 sigma predicts RV_fwd BEYOND HAR-RV. Regress RV_fwd on HAR-RV terms
    (1d/5d/22d realized vol), then test whether sigma explains the residual.
    Requires t > 3 on VALID, same sign on TRAIN. If sigma adds nothing to a
    standard vol forecast, the whole idea is dead here.
 V2 THE DECIDING TEST: does sigma predict VRP itself? If IV already embeds
    sigma, V1 can pass and there is still NO EDGE. Requires t > 3 on VALID.
    I expect this to be where it dies, and say so in advance.
 V3 ECONOMICS: short a delta-neutral ATM straddle when sigma is in its bottom
    tercile vs unconditional. Entry at REAL bid, exit at REAL ask (crossing the
    spread both ways, the pessimistic convention). Report P&L per trade, and
    the conditional-minus-unconditional difference with clustered t.
 V4 CONTROL: sigma must beat the obvious rivals as a conditioner -- current
    IV level, IV minus trailing RV, and trailing RV itself. If ranking on
    "IV - trailing RV" does the same job, sigma is redundant and dead.
 V5 shuffle null on the V3 conditioner, 200 draws, real |t| > 99th pct.

## Kill
V1 fails -> sigma is not a vol forecaster, dead.
V2 fails -> sigma is priced into IV, dead (this is the likely outcome).
V4 fails -> redundant with a trivial rival, dead.
No re-tuning of tenor/window/tercile after seeing results; fixed now.

## V1 RESULT (2026-09-05): FAILED — sign flip
TRAIN beta_sigma=+0.0132 (clustered t=+3.32) | VALID beta_sigma=-0.0138 (t=-3.15)
Significant in BOTH directions = noise fitted to a window. dR2=+0.0013 vs HAR-RV
R2 of 0.43-0.56, economically negligible regardless of sign.
Per kill criterion: sigma is NOT a volatility forecaster. V2-V5 not run.
Options work discontinued at user's instruction (pricing-model risk), though this
build used REAL quoted bid/ask rather than a model IV.
TRIAL 40. TOTAL: 40 hypotheses, 0 tradable.
