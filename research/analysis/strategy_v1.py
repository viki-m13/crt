"""
Strategy v1: PIT SP500 baseline with quality pre-filter + breadth regime.

Spec:
- Score:    0.60 * z(mom_6_1) + 0.40 * z(quality_score_5y)
- Filter:   exclude vol_1y > 0.7  OR  dd_from_52wh < -0.5  (distressed names)
- Select:   top 60 by score, then [optional] <=4 per IVV sector, top 30 by score
- Weight:   inv-vol on vol_12m, iteratively capped at 7% per name
- Regime:   d_sma200(SPY) > -0.05  AND  breadth_above_200ma > 0.40
- Vol-tgt:  scale = min(0.18 / spy_vol_21d, 1.0)
- Costs:    5 bps * 2 round-trip per rebalance (scaled by exposure)
- Sanity:   drop picks with |monthly ret| > 200% (PIT panel data errors)

Output: research/analysis/strategy_v1_results.json + backtest_v1.csv

Includes sector-diversified ablation. IVV sector map is current (slight
look-ahead for delisted historicals); marked accordingly.
"""
from __future__ import annotations
import io, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse helpers from diagnostics.py
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diagnostics import (
    load_panel, load_daily, load_membership, spy_regime, znorm,
    iterative_weight_cap, metrics,
    COST_BPS, TARGET_VOL, OOS_START, OOS_END,
)

OUT = HERE
TOP_K = 30
TOP_K_PRE = 60
SECTOR_CAP = 4
WEIGHT_CAP = 0.07
REGIME_DSMA = -0.05
BREADTH_THRESH = 0.40
EXCLUDE_VOL = 0.7
EXCLUDE_DD = -0.5

IVV_PATH = "/home/user/crt/experiments/monthly_dca/v5/ivv_holdings_latest.csv"


def load_sector_map():
    df = pd.read_csv(IVV_PATH)
    return dict(zip(df["ticker"], df["sector"]))


def breadth_above_200ma(daily: pd.DataFrame) -> pd.Series:
    """Cross-sectional fraction of tickers above their 200-day MA, monthly."""
    # Compute for each ticker: above_200 boolean daily
    rolling = daily.rolling(200, min_periods=100).mean()
    above = (daily > rolling).astype(float)
    # Per-month average over non-NaN tickers
    m = above.resample("ME").last()
    breadth = m.sum(axis=1) / m.notna().sum(axis=1).replace(0, np.nan)
    return breadth


def run_v1(panel, daily, membership, sector_map, use_sector_div: bool, label: str):
    monthly_px = daily.resample("ME").last().ffill(limit=5)
    sreg = spy_regime(daily)
    breadth = breadth_above_200ma(daily)
    mem_set = {(pd.Timestamp(r.asof), r.ticker) for r in membership.itertuples()}
    all_dates = sorted(panel["asof"].unique())
    rebalance = [d for d in all_dates if OOS_START <= d <= OOS_END]

    rows = []
    for i, date in enumerate(rebalance):
        idx_all = all_dates.index(date)
        if idx_all == len(all_dates) - 1: break

        snap = panel[panel["asof"] == date].copy()
        snap = snap[snap["ticker"].apply(lambda t: (date, t) in mem_set)]
        if snap.empty:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="no_mem"))
            continue

        # Regime gate: SPY 200ma_loose AND breadth > 40%
        reg = sreg.reindex([date]).iloc[0]
        d_sma200 = reg["d_sma200"]; spy_v = reg["vol_21d"]
        b = breadth.reindex([date]).iloc[0] if not breadth.empty else np.nan
        if not (np.isfinite(d_sma200) and d_sma200 > REGIME_DSMA):
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="regime_dsma"))
            continue
        if not (np.isfinite(b) and b > BREADTH_THRESH):
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="regime_breadth", breadth=b))
            continue

        # Quality pre-filter
        keep_mask = pd.Series(True, index=snap.index)
        if "vol_1y" in snap.columns:
            keep_mask &= snap["vol_1y"].fillna(0) <= EXCLUDE_VOL
        if "dd_from_52wh" in snap.columns:
            keep_mask &= snap["dd_from_52wh"].fillna(0) >= EXCLUDE_DD
        snap = snap[keep_mask]
        if snap.empty:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="empty_after_filter"))
            continue

        # Score: 0.6 * z(mom_6_1) + 0.4 * z(quality_score_5y)
        m6  = znorm(pd.Series(snap["mom_6_1"].values, index=snap["ticker"].values))
        qs5 = znorm(pd.Series(snap["quality_score_5y"].values, index=snap["ticker"].values))
        score = (0.60 * m6 + 0.40 * qs5).dropna()
        if score.empty:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="empty_score"))
            continue

        # Selection
        top_pre = score.sort_values(ascending=False).head(TOP_K_PRE)
        if use_sector_div:
            # Walk down top_pre, take name only if its sector still under cap
            sector_count = {}
            chosen = []
            for tk in top_pre.index:
                sec = sector_map.get(tk, "Unknown")
                if sector_count.get(sec, 0) < SECTOR_CAP:
                    chosen.append(tk)
                    sector_count[sec] = sector_count.get(sec, 0) + 1
                if len(chosen) >= TOP_K: break
            tickers = chosen
        else:
            tickers = top_pre.head(TOP_K).index.tolist()
        if not tickers:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="empty_after_selection"))
            continue

        # Prices
        next_date = all_dates[idx_all + 1]
        d0 = monthly_px.index[monthly_px.index.searchsorted(date, side="right") - 1]
        d1 = monthly_px.index[monthly_px.index.searchsorted(next_date, side="right") - 1]
        p0, p1 = monthly_px.loc[d0], monthly_px.loc[d1]
        common = [t for t in tickers
                  if t in monthly_px.columns
                  and np.isfinite(p0.get(t, np.nan)) and p0.get(t, 0) >= 1.0
                  and np.isfinite(p1.get(t, np.nan)) and p1.get(t, 0) >= 1.0]
        if not common:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="no_common"))
            continue

        # Weights: inv-vol cap 7%
        vmap = dict(zip(snap["ticker"].values, snap.get("vol_12m", snap["vol_1y"]).values))
        vols = np.array([max(float(vmap.get(t, 0.20)), 0.05)
                         if np.isfinite(vmap.get(t, np.nan)) else 0.20
                         for t in common])
        w = iterative_weight_cap(1.0 / vols, WEIGHT_CAP)

        rets = np.array([(p1[t] - p0[t]) / p0[t] for t in common])
        # Sanity (200% cap)
        sane = np.abs(rets) <= 2.0
        if not sane.all():
            if sane.sum() == 0:
                rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="all_extreme"))
                continue
            w = w[sane]; rets = rets[sane]; common = [c for c, s in zip(common, sane) if s]
            w = w / w.sum()

        raw_port = float((w * rets).sum())
        scale = min(TARGET_VOL / spy_v, 1.0) if np.isfinite(spy_v) and spy_v > 1e-6 else 1.0
        cost = COST_BPS / 10_000.0
        port = scale * raw_port - 2 * cost * scale
        rows.append(dict(date=date, ret_m=port, n_picks=len(common), scale=scale,
                         picks=",".join(common), reason="ok",
                         breadth=b, d_sma200=d_sma200, spy_vol=spy_v))

    df = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    return df, (metrics(df["ret_m"]) if not df.empty else {})


def main():
    print("Loading data ...")
    panel = load_panel(); daily = load_daily(); mem = load_membership()
    sector_map = load_sector_map()
    print(f"  panel {panel.shape}  prices {daily.shape}  members {mem.shape}  "
          f"sectors: {len(set(sector_map.values()))} distinct, {len(sector_map)} tickers")

    results = {}
    for label, use_sec in [("v1a_no_sector", False), ("v1b_sector_div", True)]:
        print(f"\n=== {label} (use_sector_div={use_sec}) ===")
        t0 = time.time()
        df, m = run_v1(panel, daily, mem, sector_map, use_sector_div=use_sec, label=label)
        print(f"  walltime: {time.time()-t0:.1f}s")
        if m:
            cash = int((df["ret_m"] == 0.0).sum())
            print(f"  CAGR={m['cagr']:>7.2%}  Sharpe={m['sharpe']:>5.2f}  "
                  f"MaxDD={m['max_dd']:>6.1%}  AnnVol={m['ann_vol']:>5.1%}  "
                  f"N={m['n_months']}  cash_months={cash}")
            results[label] = m
            df.to_csv(OUT / f"backtest_{label}.csv")
    json.dump(results, open(OUT / "strategy_v1_results.json", "w"), indent=2, default=str)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
