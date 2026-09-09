# PRE-REGISTRATION v6 — THE ARROW OF TIME (ATI)
Frozen 2026-09-05 before any ATI value is computed.

## Debt
27 hypotheses already spent this session (v1-v3 frequency, batches 1-2).
This batch adds 6 -> 33. TEST 2024-01-01..2026-07-31 STILL LOCKED, never read.

## Why this is not another frequency trial
The power spectrum is invariant under time reversal: it is built from |FFT|^2
and discards phase. Every statistic in v1-v3 (flatness, centroid, rolloff,
coherence, tau*) returns an IDENTICAL value on a reversed price path. They were
structurally incapable of measuring directionality. Irreversibility is the exact
complement of what was measured, not a variation on it.

## The measure
For a day's 1-min returns r_1..r_n:
  - slide a window of length m=3, record the ORDINAL PATTERN (which of the m!
    orderings the window falls into). Histogram -> P_fwd.
  - repeat on the reversed series r_n..r_1 -> P_rev.
  - ATI_raw = 0.5 * sum_pi |P_fwd(pi) - P_rev(pi)|   (total variation distance)
ATI_raw is biased upward at finite n, so it is NEVER used raw.

## The mandatory null (the centerpiece)
Phase-randomized surrogate: FFT the day's returns, randomize phases, invert.
The surrogate has an IDENTICAL power spectrum and is time-reversible by
construction. 20 surrogates per stock-day.
  ATI = ATI_raw - mean(ATI_surrogate)
  z_ATI = (ATI_raw - mean_surr) / sd_surr
N0 (GO/NO-GO): mean z_ATI across all stock-days must exceed +3.
If N0 fails, markets are time-reversible at this resolution, the whole idea is
dead, and NOTHING below is run or reported as a result.

## Hypotheses (only tested if N0 passes)
 H1 ATI is not a repackaging: residualize ATI on (rv, |dayret|, skew1m, kurt1m,
    spectral flatness). The RESIDUAL must carry the effects below, not the raw.
 H2 CONDITIONER: overnight reversal (I1) is stronger on LOW-ATI days
    (noise-driven moves unwind) than HIGH-ATI days (information persists).
    Measured as NET P&L edge/cost, not IC.
 H3 CONTINUATION: high-ATI days show next-day continuation of dayret;
    low-ATI days show reversal. Sign flip across the ATI split.
 H4 ATI predicts next-day realized vol beyond rv itself (information arrival).
 H5 ATI is stable: sign consistent across all 6 calendar years.

## Pass criteria (unchanged from batch 2 -- money, not IC)
Cross-sectional quintile L/S, net of measured Corwin-Schultz round-trip spread,
day-clustered t. PASS = Holm-corrected significance on VALID + same sign on
TRAIN + edge/cost > 1.0 on VALID. IC is diagnostic only.

## Kill
N0 fails -> dead, report and stop.
H1 fails (effect lives in rv/skew, not the residual) -> it is a known moment
in disguise -> dead.
