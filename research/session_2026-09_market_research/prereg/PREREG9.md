# PRE-REGISTRATION v9 — TUR CEILING, CORRECTED ACHIEVED-SIDE MEASUREMENT
Frozen 2026-09-05. Trial 36 of this session. TEST windows still never read.

## Why v8 was void (stated before v9 runs)
v8's T4 failed: surrogate oracle / real oracle = 0.949. Cause identified as a
DESIGN ERROR, not a market fact: the phase-randomized surrogate PRESERVES the
power spectrum, hence preserves linear autocorrelation (bid-ask bounce), which
is precisely what the pattern oracle exploits. A spectrum-preserving null cannot
null a spectrum-driven quantity. Second error: the oracle was fit IN-SAMPLE, so
its score mixes real predictability with 6-bucket fitting noise.
sigma (the BOUND side) is unaffected -- irreversibility is a phase property, so
the phase-randomized subtraction is the correct de-biasing there. Only the
ACHIEVED side is re-measured.

## Corrections
 (a) ACHIEVED CEILING is now strictly OUT OF SAMPLE: estimate mu_pi on the first
     half of each symbol's returns, evaluate sign(mu_pi) on the second half.
 (b) NULL is now an IID SHUFFLE of the return series (destroys spectrum AND
     phase). The OOS oracle on shuffled data measures pure fitting noise and
     must collapse to ~0.
 (c) BOUNCE CONTROL: report the OOS oracle on the phase-randomized surrogate
     too. The gap (real - surrogate) is the part of achieved edge that is NOT
     linear autocorrelation. This is the number the bound should track.

## Pre-registered tests (same bars as v8)
 T4' IID-shuffle oracle must be < 0.25 x real OOS oracle. Else achieved side is
     still noise and v9 is void as v8 was.
 T1' 0/20 violations of  oracle_OOS <= sqrt(sigma/2).
 T2' median ratio oracle_OOS / bound > 0.05.
 T3' spearman( bound, oracle_OOS ) > +0.5  AND
     spearman( bound, oracle_OOS - oracle_surrogate ) > +0.5.
     The second is the real test: the bound is a claim about irreversibility, so
     it should track the NON-bounce component of achievable edge specifically.

## Kill
T4' fails -> void, report and stop, no third attempt at this measurement.
T3' fails -> bound is real but not a usable asset pre-filter; report as such.
