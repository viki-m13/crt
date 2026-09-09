# PRE-REGISTRATION v4 — BATCH OF 12 IDEAS
Frozen 2026-09-04 before any v4 feature is computed or any v4 result is seen.

## Debt disclosure
Trials spent on TRAIN this session: v1 timbre (dead), v2 cross-spectral lead-lag
(dead), v3 resonance tuning (dead). This batch adds 12 more. Total 15.
TEST = 2024-01-01..2026-07-31 remains LOCKED and has never been read.

## Data / splits
Alpaca 1-min bars, 20 symbols, 2018-01-02..2023-12-31 (TEST excluded at load).
TRAIN 2018-01-01..2021-12-31.  VALID 2022-01-01..2023-12-31.
All signals built from bars <= 16:00 ET of day D. Targets are strictly after.

## Targets
  T_on   overnight:   next_open / close - 1
  T_oc   next day open->close
  T_cc   next day close->close
  T_i60  intraday: forward 60-min return within the same session

## Statistics (fixed in advance, applied identically to every idea)
- Cross-sectional ideas: per-day rank-IC (Spearman) of signal vs target across
  the 20 symbols; aggregate = mean of daily ICs; t = mean / (sd/sqrt(n_days)).
  This is day-clustered by construction.
- Time-series ideas: per-stock-day IC, aggregated with day-clustered t
  (cluster = calendar day; symbols on one day share market shocks).
- MULTIPLE TESTING: 12 ideas -> Holm-Bonferroni at family alpha 0.05 on VALID.
  A raw |t|>2 is NOT a pass. The Holm threshold is the pass bar.
- An idea passes only if: (a) survives Holm on VALID, AND (b) same sign on TRAIN.
- Economic bar: mean |IC| must imply edge/cost > 1.0 at measured Corwin-Schultz
  spread before anything is called tradable. Statistical pass alone is not a pass.

## THE 12 IDEAS (listed before any is computed; no substitutions allowed)
 I1  Overnight reversal: -ret_last30 predicts T_on  (late-day pressure unwinds)
 I2  Close-vs-VWAP dislocation: -(close-vwap)/close predicts T_on
 I3  Semivariance asymmetry: (semivar_up-semivar_dn)/rv predicts T_cc
 I4  Efficiency ratio |dayret|/(rv*sqrt(n)) predicts T_cc (trending days continue)
 I5  Amihud illiquidity |dayret|/dollarvol amplifies I1 (interaction, not level)
 I6  Trade-count surprise ntr/ma20(ntr) predicts |T_cc| then signed via -dayret
 I7  Cross-sectional dispersion of dayret as a REGIME GATE on I1 (high disp -> stronger)
 I8  Volume-clock reversion: intraday oscillator IC in volume time vs wall time (T_i60)
 I9  Slow-band oscillator EMA(12)-EMA(48) beats EMA(3)-EMA(12) on T_i60 net of cost
 I10 Time-of-day: reversion IC by hour bucket; is the U-shape exploitable as a filter
 I11 Intraday return skewness (1-min) predicts T_on (crash-risk premium)
 I12 Range-compression: rng/ma20(rng) low -> next-day breakout continuation on T_cc

## Required controls
 C-A  Every cross-sectional idea gets a SHUFFLE null: permute the signal across
      symbols within each day, 200 draws; the real |t| must exceed the 99th pct.
 C-B  Any idea that survives must be checked against the plain reversion signal
      (the only known-live effect) -- if it adds nothing beyond -ret_last30, it
      is not a new idea. Same control that killed v2.
 C-C  Sign stability across the 6 calendar years, reported, not gated.

## Kill criteria
Fails Holm on VALID, or sign flips TRAIN->VALID, or fails its shuffle null,
or edge/cost <= 1.0. Report the kill, do not re-tune and re-test.
