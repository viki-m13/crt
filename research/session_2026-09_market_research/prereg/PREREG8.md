# PRE-REGISTRATION v8 — A THERMODYNAMIC CEILING ON ALPHA
Frozen 2026-09-05 before any bound is computed.

## Claim
Stochastic thermodynamics (Barato-Seifert 2015, Thermodynamic Uncertainty
Relation): for any current J in a NESS, Var(J)/<J>^2 >= 2/Sigma, Sigma =
entropy production. A strategy's cumulative P&L IS a current. With n steps,
Sigma = n*sigma and Sharpe_window = sqrt(n)*Sharpe_step, n cancels:

        Sharpe_per_step  <=  sqrt( sigma / 2 )

sigma = entropy production per step, estimated from the price path alone.
This BOUNDS alpha rather than searching for it. If true it is a pre-filter:
never search an asset whose ceiling is below your cost hurdle.

## Estimator (coarse-grained, honest about what it bounds)
sigma_hat = sum_pi P_fwd(pi) * ln[ P_fwd(pi) / P_rev(pi) ]
over ordinal patterns of length m=3 of 1-min returns (KL divergence between the
forward and time-reversed pattern distributions; Roldan-Parrondo-style
coarse-grained entropy production). De-biased with 20 phase-randomized
surrogates per symbol (identical power spectrum, reversible by construction):
sigma = max(0, sigma_hat_real - mean(sigma_hat_surrogate)).
Because m=3 patterns are a COARSE-GRAINING, sigma_hat is a LOWER bound on true
entropy production. Therefore sqrt(sigma/2) bounds the Sharpe achievable BY ANY
STRATEGY RESTRICTED TO THAT SAME INFORMATION. That is the honest claim, and it
is what will be tested. It is NOT a claim about strategies using more info.

## The matched empirical ceiling (what makes this falsifiable)
PATTERN ORACLE: the best possible strategy in exactly that information set.
For each of the 6 patterns pi, compute the IN-SAMPLE mean next-step return
mu_pi. Strategy: position = sign(mu_pi). Fit and evaluated on the SAME data, so
it is an upper bound on that information set BY CONSTRUCTION -- it cannot be
beaten, and it is deliberately look-ahead-contaminated. Its per-step Sharpe is
the EMPIRICAL ceiling.

## Pre-registered tests
 T1 VIOLATION TEST (falsifies the whole idea): for all 20 symbols,
    Sharpe_oracle <= sqrt(sigma/2) must hold. ANY violation kills it outright.
 T2 TIGHTNESS: if the bound holds only because it is 100x too large, it is
    true but useless. Report ratio = Sharpe_oracle / sqrt(sigma/2) per symbol.
    Useful requires median ratio > 0.05.
 T3 PREDICTIVE ORDERING (the real value): spearman( sqrt(sigma/2),
    Sharpe_oracle ) across the 20 symbols must be > +0.5. A bound that does not
    rank assets by their achievable edge is not a pre-filter.
 T4 SURROGATE CONTROL: on the phase-randomized surrogates, sigma ~ 0, so the
    bound must collapse toward 0 AND the surrogate pattern-oracle Sharpe must
    also collapse. If the surrogate oracle stays high, the oracle is measuring
    sampling noise, not structure, and T1-T3 are meaningless.

## Kill criteria
T1 violated -> dead. T3 < +0.5 -> not a usable pre-filter, report as such.
T4 fails -> the empirical ceiling is noise, everything above is void.
No result is reported before T4 has run.
