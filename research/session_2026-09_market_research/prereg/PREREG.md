# PRE-REGISTRATION — Spectral Timbre / Overnight Impact Decomposition
Written BEFORE any result was computed. Frozen.

## Mechanism
Informed directional flow => PERMANENT price impact => spectral energy concentrated
at LOW frequency => "tonal" (low spectral flatness).
Uninformed liquidity churn => TRANSIENT impact that reverts => energy spread across
ALL frequencies => "flat/white" (high spectral flatness).
This is the Roll(1984)/Hasbrouck(1991) permanent-vs-transitory decomposition, but
measured over the FULL autocovariance structure by one SCALE-FREE scalar instead of
a 1-2 lag variance ratio. Scale-free => NOT a volatility proxy. That is the novelty.

## Signal (per stock-day, computed only from bars <= 15:55 ET)
r_t   = log returns of 1-min closes during RTH
P(f)  = periodogram of r_t (detrended, Hann window)
FLAT  = geometric_mean(P) / arithmetic_mean(P)      in (0,1]; 1 = white noise
CENT  = sum(f*P)/sum(P) normalised to Nyquist        "brightness"
DAYRET= log(close_1555 / open_0930)

## Hypotheses (directional, fixed now)
H1  Low FLAT (tonal)  -> overnight return CONTINUES the day's sign
H2  High FLAT (noisy) -> overnight return REVERSES the day's sign
H3  The interaction DAYRET x FLAT_rank predicts overnight return with a NEGATIVE
    coefficient (higher flatness => more reversal)

## Primary test
overnight = log(next_open / close_1600)
Regress overnight ~ a + b1*DAYRET + b2*(DAYRET * FLAT_rank_cross_sectional)
PASS if b2 < 0 with t < -2 in TRAIN and the SIGN REPLICATES in VALID.

## Splits (locked now)
TRAIN 2018-01-01..2021-12-31   design + all decisions
VALID 2022-01-01..2023-12-31   confirm sign only
TEST  2024-01-01..2026-07-31   OPENED ONCE, never tuned on

## Mandatory controls (a positive result is void without these)
C1 RANDOM NULL: shuffle FLAT across symbols within each date, 200 draws.
   Real |t| must exceed the 95th pct of the null.
C2 VOL CONTROL: rerun with FLAT replaced by realized-vol rank. If vol does the same
   job, FLAT is a vol proxy and the result is NOT novel.
C3 CONFOUND SCAN: corr(FLAT, {vol, volume, trade count, range, price}). If |corr|>0.7
   with any, declare FLAT contaminated by that variable.
C4 COST: charge the measured spread. Report net.

## Kill criteria (pre-committed)
- b2 sign flips between TRAIN and VALID  -> DEAD
- C1 fails (inside the null)             -> DEAD
- C2 shows vol rank does as well/better  -> NOT NOVEL, report as such
- Net of costs the book is negative      -> report as an unexploitable measurement
