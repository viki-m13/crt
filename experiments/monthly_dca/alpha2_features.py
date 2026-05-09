"""Alpha-2 feature engineering — additional high-IC features for the next-gen strategy.

VECTORIZED — no per-ticker Python loops.

Adds to per-month feature parquets:
  fip_score              — Frog-in-the-pan information discreteness (Da et al 2014)
                            sign(R_12_1) * (frac_neg_days - frac_pos_days)
                            More-negative => smoother diffuse trend => stronger continuation
  idio_mom_12_1          — Idiosyncratic 12m-1m return: residual after regressing
                            daily returns on SPY (alpha + beta*SPY).
                            Compounded residual over (-252,-21) days.
  beta_2y                — 2y rolling beta vs SPY
  mom_per_unit_vol_12    — mom_12_1 / vol_12m  (Sharpe-momentum)
  acceleration_2y        — mom_12_1 - mom_24_12 (accel of medium-term trend)
  quality_score_5y       — trend_health_5y * (1 + max_dd_5y) * sqrt(max(sharpe_5y,0))
  max_dd_5y              — max drawdown over 5y window (closer to 0 = better)
  sharpe_5y              — 5y annualized Sharpe ratio
  tight_consolidation_60 — fraction of last 60 days where 20-day-rolling range < 5%
  breakout_strength_60   — current_px / max(prior 60d close) - 1
  rsi_zone_score         — bell curve: max=1 at RSI=55, sigma=15, zero outside [25,80]
  min_dd_60d             — recent 60-day max drawdown
  earnings_drift_proxy   — drift after biggest 1-day move (PEAD proxy)

All are derivable strictly from price history; no fundamentals.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import CACHE, FEATURES_DIR, load_panel


SENTINEL_COL = "fip_score"


def compute_alpha2(panel: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    sub = panel.loc[panel.index <= asof]
    if len(sub) < 252:
        return pd.DataFrame()
    last_row = sub.iloc[-1]
    alive = last_row.notna()
    counts = sub.notna().sum()
    eligible = alive & (counts >= 252)
    cols = sub.columns[eligible]
    sub = sub[cols].astype("float64")
    out = pd.DataFrame(index=cols)

    rets = sub.pct_change()
    n = len(sub)

    # 1) FIP score
    if n >= 252:
        win = rets.iloc[-252:-21]
        pos = (win > 0).sum()
        neg = (win < 0).sum()
        denom = pos + neg
        px_12m = sub.iloc[-252]
        px_1m = sub.iloc[-21]
        r12_1 = (px_1m / px_12m - 1.0)
        sign = np.sign(r12_1.fillna(0))
        with np.errstate(invalid='ignore', divide='ignore'):
            id_score = (pos - neg) / denom.replace(0, np.nan)
        out["fip_score"] = (-sign * id_score).astype(float)

    # 2) beta_2y and idio_mom_12_1 — vectorized
    if "SPY" in cols and n >= 252:
        # Use last 504 days for beta (or n-1)
        beta_window = min(504, n - 1)
        rets_b = rets.iloc[-beta_window:]
        spy_b = rets_b["SPY"].to_numpy()
        # Center
        spy_centered = spy_b - np.nanmean(spy_b)
        spy_var = np.nanvar(spy_b)
        if spy_var > 0:
            # Compute beta = cov(stock, spy) / var(spy)
            stk = rets_b.to_numpy()
            stk_centered = stk - np.nanmean(stk, axis=0, keepdims=True)
            cov = np.nanmean(stk_centered * spy_centered[:, None], axis=0)
            beta = cov / spy_var
            out["beta_2y"] = pd.Series(beta, index=rets_b.columns)

        # Idio momentum: residual return over (-252, -21) using beta
        idio_window = rets.iloc[-252:-21]
        spy_window = idio_window["SPY"].to_numpy()
        stk_window = idio_window.to_numpy()
        beta_arr = out["beta_2y"].reindex(idio_window.columns).to_numpy()
        # residual = stk - beta * spy
        resid = stk_window - beta_arr[None, :] * spy_window[:, None]
        # cumulative (1 + r).prod() - 1
        with np.errstate(invalid='ignore'):
            log_resid = np.log1p(resid)
            cum = np.exp(np.nansum(log_resid, axis=0)) - 1.0
            # Mark NaN where most observations are NaN
            n_valid = np.sum(~np.isnan(resid), axis=0)
            cum = np.where(n_valid >= 60, cum, np.nan)
        out["idio_mom_12_1"] = pd.Series(cum, index=idio_window.columns)

    # 3) mom_per_unit_vol_12
    if n >= 252:
        vol12 = rets.iloc[-252:].std() * math.sqrt(252)
        px_12m = sub.iloc[-252]
        px_1m = sub.iloc[-21] if n > 21 else sub.iloc[-1]
        m12_1 = (px_1m / px_12m - 1.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            out["mom_per_unit_vol_12"] = (m12_1 / vol12).replace([np.inf, -np.inf], np.nan)

    # 4) acceleration_2y
    if n >= 504:
        px_24m = sub.iloc[-504]
        px_12m = sub.iloc[-252]
        px_1m = sub.iloc[-21]
        m_12_1 = (px_1m / px_12m - 1.0)
        m_24_12 = (px_12m / px_24m - 1.0)
        out["acceleration_2y"] = (m_12_1 - m_24_12).astype(float)

    # 5) Quality score 5y, max_dd_5y, sharpe_5y
    if n >= 252 * 5:
        rs5 = rets.iloc[-252 * 5:]
        sharpe5 = (rs5.mean() / rs5.std()) * math.sqrt(252)
        prices = sub.iloc[-252 * 5:]
        rolling_max = prices.cummax()
        dd_series = (prices / rolling_max - 1.0)
        max_dd = dd_series.min()
        sma200 = prices.rolling(200, min_periods=100).mean()
        above = (prices > sma200).sum()
        cnt = sma200.notna().sum()
        with np.errstate(invalid='ignore', divide='ignore'):
            trend_h = (above / cnt.replace(0, np.nan))
        with np.errstate(invalid='ignore'):
            qs = trend_h * (1 + max_dd) * np.sqrt(np.maximum(sharpe5, 0))
        out["quality_score_5y"] = qs.astype(float)
        out["max_dd_5y"] = max_dd.astype(float)
        out["sharpe_5y"] = sharpe5.astype(float)

    # 6) tight_consolidation_60
    if n >= 60:
        last60 = sub.iloc[-60:]
        rolling_max20 = last60.rolling(20, min_periods=10).max()
        rolling_min20 = last60.rolling(20, min_periods=10).min()
        rolling_mid = (rolling_max20 + rolling_min20) / 2
        with np.errstate(invalid='ignore', divide='ignore'):
            rng_pct = (rolling_max20 - rolling_min20) / rolling_mid.replace(0, np.nan)
        out["tight_consolidation_60"] = (rng_pct < 0.05).sum() / 60.0

    # 7) breakout_strength_60
    if n >= 60:
        prior60 = sub.iloc[-60:-1]
        max60 = prior60.max()
        cur = sub.iloc[-1]
        with np.errstate(invalid='ignore', divide='ignore'):
            out["breakout_strength_60"] = (cur / max60 - 1.0).astype(float)

    # 8) rsi_zone_score
    if n >= 15:
        delta = sub.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        with np.errstate(invalid='ignore', divide='ignore'):
            rs_ratio = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs_ratio))
        rsi_now = rsi.iloc[-1]
        zone = np.exp(-((rsi_now - 55) ** 2) / (2 * 15.0 ** 2))
        zone = zone.where((rsi_now >= 25) & (rsi_now <= 80), 0.0)
        out["rsi_zone_score"] = zone.astype(float)

    # 9) min_dd_60d
    if n >= 60:
        last60 = sub.iloc[-60:]
        rmax = last60.cummax()
        dd60 = (last60 / rmax - 1.0).min()
        out["min_dd_60d"] = dd60.astype(float)

    # 10) earnings_drift_proxy — vectorized
    if "SPY" in cols and n >= 252:
        # Find for each ticker the location (within last 252 days) of the
        # largest absolute return.  Then compute return 21->63 days after.
        win = rets.iloc[-252:-63]
        if len(win) >= 60:
            abs_win = win.abs()
            # Vectorized argmax. Replace NaN with -1 so they don't win.
            arr = abs_win.to_numpy()
            arr_nan = np.where(np.isnan(arr), -1.0, arr)
            argmax_in_win = np.argmax(arr_nan, axis=0)  # 0..len(win)-1
            big_pos = (n - 252) + argmax_in_win  # row idx in sub

            big_idx = big_pos
            start_idx = big_idx + 21
            end_idx = big_idx + 63
            valid_end = end_idx < n
            spy_arr = sub["SPY"].to_numpy()

            # Gather prices by index for each ticker column
            stk_arr = sub.to_numpy()  # shape (n, n_tickers)
            tcols = sub.columns
            n_t = stk_arr.shape[1]
            tk_idx = np.arange(n_t)

            # safe-clamp end_idx
            end_clamped = np.minimum(end_idx, n - 1)
            start_clamped = np.minimum(start_idx, n - 1)

            stk_end = stk_arr[end_clamped, tk_idx]
            stk_start = stk_arr[start_clamped, tk_idx]
            spy_end = spy_arr[end_clamped]
            spy_start = spy_arr[start_clamped]

            with np.errstate(invalid='ignore', divide='ignore'):
                stk_ret = stk_end / stk_start - 1.0
                spy_ret = spy_end / spy_start - 1.0

            # Sign of biggest move
            big_ret = arr[argmax_in_win, tk_idx]   # absolute value here
            # Get original (signed) return
            signed_arr = win.to_numpy()
            big_signed = signed_arr[argmax_in_win, tk_idx]
            sign = np.sign(big_signed)

            drift = sign * (stk_ret - spy_ret)
            drift = np.where(valid_end & np.isfinite(drift), drift, np.nan)
            out["earnings_drift_proxy"] = pd.Series(drift, index=tcols)

    return out


def main(force: bool = False):
    panel = load_panel()
    months = sorted([pd.Timestamp(f.stem) for f in FEATURES_DIR.glob("*.parquet")])
    print(f"Computing alpha2 for {len(months)} month-ends ...")
    for i, m in enumerate(months):
        path = FEATURES_DIR / f"{m.date()}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            if SENTINEL_COL in existing.columns and not force:
                continue
        else:
            existing = pd.DataFrame()
        new = compute_alpha2(panel, m)
        if new.empty:
            continue
        merged = existing.copy()
        for c in new.columns:
            merged[c] = new[c]
        merged.to_parquet(path)
        if i % 20 == 0:
            print(f"  {i+1}/{len(months)} {m.date()}: shape={merged.shape}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    main(force=force)
