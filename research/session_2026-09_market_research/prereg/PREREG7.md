# PRE-REGISTRATION v7 — WIDE UNIVERSE / BREADTH TEST
Frozen 2026-09-05 before any wide-universe result is computed.

## Why this exists (stated before results)
33 hypotheses failed on a 20-name cross-section. Grinold: IR = IC x sqrt(breadth).
At N=20 a real IC of 0.06 cannot reach significance and a L/S book cannot
diversify idiosyncratic risk. I CANNOT distinguish "no edge" from "no breadth"
on that data. This tests exactly that, and nothing else.

## Data
equity_daily_broad: open/close/dvol, 800 symbols, 2,666 days 2016-01-04..2026-08-11.
sp500_pit_members.csv: point-in-time S&P 500 membership (includes delisted
tickers -> survivorship-free). Universe on day D = PIT members as of D,
intersected with symbols having valid prices, filtered by TRAILING dvol only.

## Splits (TEST lock carried over unchanged)
TRAIN 2016-01-01..2021-12-31   VALID 2022-01-01..2023-12-31
TEST  2024-01-01..2026-08-11   LOCKED, NOT READ IN THIS BATCH.

## The four tests (no substitutions, no additions)
 W1 intraday reversal -> overnight:  signal = -(close_D/open_D - 1),
    target = open_{D+1}/close_D - 1.   (daily analogue of I1/I2)
 W2 W1 restricted to HIGH cross-sectional dispersion days (the I14 candidate,
    the only idea of 33 to clear edge/cost>1.0 on both splits).
 W3 overnight reversal -> intraday: signal = -(open_D/close_{D-1} - 1),
    target = close_D/open_D - 1.
 W4 NULL: W1 with the target permuted across names within each day, 200 draws.
    Real |t| must exceed the 99th percentile of the null.

## The breadth diagnostic (the actual point)
Run W1 at N = 20, 50, 100, 200, 400, full, by random subsampling the PIT
universe each day (20 seeds each). If t-stat scales as ~sqrt(N) and N=20 sits
near t~1, my 33 rejections were breadth-limited, not edge-limited.
If t is flat in N, the 20-name result was honest and the edge is simply absent.
BOTH OUTCOMES ARE INFORMATIVE AND WILL BE REPORTED.

## Costs
I have no measured spreads for 800 names and will NOT invent a cost model from
20 calibration points. Report GROSS edge and the BREAK-EVEN round-trip cost in
bp. The reader compares that to real S&P-500 spreads (~1-5 bp). No strategy is
called tradable on gross numbers alone.

## Pass criteria
W1/W2 significant on VALID with same sign on TRAIN, pass the W4 null, and a
break-even cost comfortably above plausible spreads. Otherwise: not tradable.
