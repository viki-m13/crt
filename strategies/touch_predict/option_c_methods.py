"""Tier 2 / Tier 3 candidate methods if Tier 1 doesn't calibrate.

This file is a parking lot for novel approaches we'll plug in if the
walk-forward shows the basic 4-layer guard isn't enough to deliver
≥98% claim → ≥98% delivery.

Methods sketched here, to be activated as needed:

  M5: CONFORMAL STRIKE SELECTION (distribution-free guarantee)
      Replace "K_short ≥ worst_train + 1%" with a conformal quantile
      method. For each (ticker, side, h):
        - Compute the post-fire move distribution on training fires
        - The desired coverage q (e.g., 99%) determines the conformal
          quantile (1 − α/(N+1)) under exchangeability
        - K_short = quantile + safety
      Provides finite-sample distribution-free guarantee that the
      true win rate ≥ q with confidence 1 − δ.

  M6: PER-TRADE CONSENSUS (not per-ticker)
      Currently consensus uses (ticker, year, side). Strengthen to
      (ticker, expiry_date, strike_bucket) — require ≥ 2 distinct
      regime families to put forward overlapping spreads (within 5%
      strike) on the same Friday expiry. This eliminates "the same
      ticker passed two rules but on totally different trades."

  M7: REGIME-DISTANCE GATING (Mahalanobis)
      Compute the Mahalanobis distance of TODAY's feature vector
      (RSI2, volume z, dd252, mom_252, SPY-rel, vol20) from the
      mean of historical fires. If d > threshold, block: today is
      "out of distribution" relative to training.

  M8: QUARANTINE-ON-FAILURE (online learning)
      The live signal log already records every published signal.
      When a signal RESOLVES with a loss, immediately quarantine the
      (ticker, regime, horizon, k_short) combo permanently. The
      engine becomes anti-fragile — every loss tightens it.

  M9: BOOTSTRAP CONSENSUS
      For each (ticker, regime, h, k_short), do B bootstrap resamples
      of the training data. Compute K_short from each bootstrap.
      Use the MAX K_short across bootstraps (most conservative) as
      the production strike — robust to specific-sample luck.

  M10: TIME-DECAY WEIGHTED VOTING
      Recent fires weighted more heavily than ancient. The 99% win
      rate at t = 5 years ago counts less than 99% at t = 1 month.
      Detects regime shifts faster.

These aren't built yet — picked when results demand them.
"""

# Stub — methods are implemented inline in option_c_certified_v2.py
# when activated.
