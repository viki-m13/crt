# PRE-REGISTRATION v3 — RESONANCE TUNING
Frozen 2026-09-04, before any v3 result is computed.

## Debt disclosure (carried forward, honestly)
Trials already spent on TRAIN in this session:
  1. v1 spectral timbre (FLAT/CENT cross-section)         -> KILLED by white-noise null
  2. v2 cross-spectral lead-lag (band-resolved)            -> KILLED by own-oscillator control
  3. this is TRIAL 3. TEST (2024-01-01..2026-07-31) still LOCKED and never read.
Any v3 t-stat must be discounted for 3 trials.

## What the control forced
The surviving signal is the target's OWN band-passed oscillator:
    sig_t = EMA(c,3)_t - EMA(c,12)_t ,  c = cumsum(1-min log returns)
This is a band-pass with a FIXED passband (~2-20 min), chosen by convention.
IC(k=120) approx -0.19 to -0.22.  That is known intraday mean reversion.

## The v3 hypothesis (the only frequency claim left that is not decoration)
H1 (RESONANCE): the reversion period is NOT a constant. Each stock-day has a
dominant oscillation period tau* set by that day's microstructure. Tuning the
oscillator's passband to tau* beats the fixed 3/12 oscillator.

Mechanism if true: a market maker's inventory cycle / an execution algo's slice
interval imposes a characteristic period. A filter matched to that period has
higher SNR than a mismatched one -- the same reason a lock-in amplifier beats a
wideband one.

## Measurement (strictly causal, PIT)
For each stock-day:
  MORNING  = bars 09:31..12:30 ET   (signal-estimation half)
  AFTERNOON= bars 12:31..16:00 ET   (evaluation half)
  tau* := period of the max of the Welch/periodogram of MORNING detrended
          1-min returns, restricted to periods in [4, 60] minutes.
  tuned oscillator on AFTERNOON:  EMA(c, tau*/4) - EMA(c, tau*)
  fixed oscillator on AFTERNOON:  EMA(c, 3)      - EMA(c, 12)
  Both evaluated on the SAME afternoon bars, SAME target
  fwd_k = sum of next k 1-min log returns, k in {15,30,60}.
  IC computed per stock-day; aggregate = mean IC, clustered by DAY (all symbols
  on one day share market-wide shocks), t = mean / (std_daymeans/sqrt(n_days)).

## H1 pass criteria (ALL must hold on VALID 2022-01-01..2023-12-31)
  P1. |IC_tuned| > |IC_fixed| at k=60, day-clustered t(diff) > 2.5
  P2. same sign of improvement on TRAIN 2018-01-01..2021-12-31
  P3. improvement not explained by tau* merely proxying volatility:
      regress per-day IC-diff on log(realized vol); improvement must survive
  P4. improvement present in >= 12 of the 20 symbols

## Kill criteria (any one -> H1 dead, report and stop)
  K1. |IC_tuned| <= |IC_fixed| at k=60 on VALID
  K2. t(diff) < 2.5 on VALID
  K3. improvement flips sign TRAIN vs VALID
  K4. present in < 12 symbols

## Falsifiable null
If tau* carries no information, tuned should perform like a RANDOMLY chosen
passband. Control R1: replace tau* with a random draw from the same empirical
distribution of tau*, recompute. tuned must beat random-tau by t>2.5 as well.
R1 is REQUIRED, not optional. v1 and v2 both died at exactly this kind of null.

## No result may be reported before R1 has run.
