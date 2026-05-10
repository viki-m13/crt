"""Build PIT S&P 500 membership at weekly cadence and SPY weekly features
for the regime gate.

Outputs:
  experiments/monthly_dca/v8/weekly/cache/sp500_membership_weekly.parquet
  experiments/monthly_dca/v8/weekly/cache/spy_features_weekly.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PIT = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit"
WEEKLY_CACHE = Path(__file__).resolve().parent / "cache"


def main():
    feat = pd.read_parquet(WEEKLY_CACHE / "features_weekly.parquet")
    weekly_asofs = sorted(feat["asof"].unique())
    print(f"[load] {len(weekly_asofs)} weekly asofs")

    # ------ PIT membership ------
    print("[membership] building weekly PIT membership from monthly")
    mem_m = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    mem_m["asof"] = pd.to_datetime(mem_m["asof"])
    monthly_asofs = sorted(mem_m["asof"].unique())
    rows = []
    # For each weekly asof, find the most recent monthly asof <= weekly asof
    monthly_idx = pd.DatetimeIndex(monthly_asofs)
    for w in weekly_asofs:
        pos = monthly_idx.searchsorted(w, side="right") - 1
        if pos < 0:
            continue
        m_match = monthly_idx[pos]
        members = mem_m[mem_m["asof"] == m_match]["ticker"].tolist()
        for tk in members:
            rows.append({"asof": w, "ticker": tk})
    mem_w = pd.DataFrame(rows)
    out = WEEKLY_CACHE / "sp500_membership_weekly.parquet"
    mem_w.to_parquet(out, index=False)
    print(f"  saved {len(mem_w)} rows -> {out}")

    # ------ SPY weekly features ------
    print("[spy] weekly SPY features")
    px = pd.read_parquet(ROOT / "experiments" / "monthly_dca" / "cache" / "prices_extended.parquet")
    if "SPY" not in px.columns:
        raise RuntimeError("SPY missing from prices_extended.parquet")
    spy = px["SPY"].dropna()

    log_spy = np.log(spy)
    ret_1d = log_spy.diff()
    ret_21d = log_spy - log_spy.shift(21)
    ret_63d = log_spy - log_spy.shift(63)
    ret_126d = log_spy - log_spy.shift(126)
    ret_252d = log_spy - log_spy.shift(252)
    mom_6_1 = ret_126d - ret_21d
    mom_12_1 = ret_252d - ret_21d
    sma200 = spy.rolling(200, min_periods=120).mean()
    d_sma200 = (spy / sma200) - 1.0
    high_252 = spy.rolling(252, min_periods=126).max()
    dd_52wh = (spy / high_252) - 1.0
    above_200 = (spy > sma200).astype(float)
    # Streak below-200 = number of recent consecutive days below 200
    below = (~above_200.astype(bool)).astype(int)
    streak_below = below.groupby((below == 0).cumsum()).cumcount() * below
    # rsi 14
    delta = spy.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    a_up = up.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    a_dn = down.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = a_up / a_dn.replace(0, np.nan)
    rsi_14 = 100 - 100/(1+rs)

    # Daily ret_21d simple from last 21 trading days
    ret_21d_simple = spy.pct_change(21)

    spy_df = pd.DataFrame({
        "spy_dsma200": d_sma200,
        "spy_rsi14": rsi_14,
        "spy_mom_12_1": mom_12_1,
        "spy_mom_6_1": mom_6_1,
        "spy_ret_21d": ret_21d_simple,
        "spy_below_200_streak": streak_below.reindex(spy.index, fill_value=0),
        "spy_dd_from_52wh": dd_52wh,
        "spy_vol_1y": ret_1d.rolling(252, min_periods=126).std() * np.sqrt(252),
    })
    # Take rows on weekly asofs
    spy_w = spy_df.loc[spy_df.index.isin(weekly_asofs)].copy()
    spy_w.index.name = "asof"
    out = WEEKLY_CACHE / "spy_features_weekly.parquet"
    spy_w.to_parquet(out)
    print(f"  saved {len(spy_w)} weekly SPY rows -> {out}")


if __name__ == "__main__":
    main()
