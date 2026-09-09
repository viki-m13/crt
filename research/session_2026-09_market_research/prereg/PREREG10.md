# PRE-REGISTRATION v10 — DISSIPATION PER UNIT ORDER FLOW
Frozen 2026-09-05 before any v10 quantity is computed. Trial 37+ of this session.

## The invention (what has not been done)
Kyle's lambda = PRICE IMPACT per unit signed flow. Universal, 40 years old.
This proposes: ENTROPY PRODUCTION per unit signed flow.
    D = sigma_local / |F|
Mechanism: informed flow moves price PERMANENTLY, and a permanent move is
IRREVERSIBLE -- the path betrays time's direction, so entropy production is high.
Uninformed / liquidity flow moves price TRANSIENTLY, and a transient move is
REVERSIBLE -- it comes back, so forward and reversed paths look alike and
entropy production is low.
=> D should classify WHICH moves revert. Low D = liquidity = fade it.
                                          High D = information = do not fade.

This also explains PREREG9's result rather than contradicting it: UNCONDITIONAL
sigma is dominated by the permanent component, which is why it ranked assets
backwards (spearman -0.394). Conditioning on flow is what separates the two.

## Why crypto perps
Not dissipation (measured: crypto sigma 0.00116 < equity 0.00303 -- that thesis
was refuted today and is discarded). Two real reasons:
 (a) signed order flow is directly observable (taker-buy volume). Equities never
     gave me this. It is the input the whole idea requires.
 (b) 5-min move / round-trip cost is 3.07 vs 2.04 for equities (1.5x headroom).

## Data / splits
Binance perps 1-min, 8 symbols, 2023-01-01..2026-08.
TRAIN 2023-01-01..2024-12-31   VALID 2025-01-01..2025-12-31
TEST  2026-01-01..2026-08-01   LOCKED, NOT READ IN THIS BATCH.

## Construction (all causal)
Every 5 min, using only bars <= t:
  F      = sum(2*tbv - v) over last 5 min, normalized by trailing-100-bar mean v
  dP     = 5-min log return
  lambda = |dP| / |F|                       (Kyle-like control)
  sigma_local = ordinal-pattern KL(fwd||rev) over the trailing 120 1-min returns.
       Uses the identity P_rev(pi) = P_fwd(reverse(pi)), so it is computed from
       the forward histogram alone via rolling counts. NOT surrogate-de-biased:
       the window length is constant so the finite-n bias is constant and cannot
       affect a RANKING across events. Stated in advance.
  D      = sigma_local / |F|                (THE NEW DISCRIMINATOR)
Target: reversal payoff = -sign(dP) * forward return over k in {15,30,60} min.

## Pre-registered tests
 H1 reversal payoff is monotonically DECREASING in D (fade low-D, not high-D).
    Measured as NET bp per trade after 10 bp round-trip taker cost.
 H2 ORTHOGONALITY (the control that killed v2): D must add beyond lambda,
    volume, trade size, and realized vol. Residualize D on all four; the
    residual must retain the effect.
 H3 the D-sorted spread (low-D minus high-D reversal payoff) must be positive
    on TRAIN and VALID with day-clustered t > 3 on VALID.
 H4 SHUFFLE NULL: permute D across events within each day, 200 draws; real |t|
    must exceed the 99th percentile.
 H5 per-symbol: effect present in >= 6 of 8 symbols.

## Pass = tradable
NET of 10 bp round-trip, day-clustered t > 3 on VALID, passes H2 and H4,
present in >=6/8 symbols. Anything less is reported as a failure, not a lead.

## Kill
H2 fails (D is lambda in disguise) -> dead. H4 fails -> dead.
Net <= 0 after cost -> not tradable, report and move on.
