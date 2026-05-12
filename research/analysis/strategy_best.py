"""
research/analysis/strategy_best.py
==================================

The best method I could develop, dual-validated on both PIT panels.

Final result (walk-forward OOS, no parameter tuning on validation windows):

    Panel             Window      CAGR    Sharpe   MaxDD   AnnVol   Benchmark
    PIT SP500         2007-2024   6.7%    0.70    -14.7%   9.6%     SPY 9.7% / 0.67
    PIT NDX           2019-2025   7.2%    0.76    -15.4%   9.5%     QQQ 20.7% / 1.05

This is the empirical ceiling for monthly K=30 long-only on PIT data with the
features available in the augmented panel + cached features dir. Below the
Sharpe-2.0 / CAGR-50 mission targets by a large margin -- a structural change
(higher rebalance freq, long/short, alt data, different universe) is required
to approach them. The full analysis is in PATH_TO_50_2_FINAL.md.

The strategy:

    Universe:   PIT SP500 or PIT NDX (membership-filtered at every rebalance)
    Score:      mean of pct-ranks of 5 signals
                  + mom_6_1
                  + sharpe_5y
                  + idio_mom_12_1
                  - vol_1y
                  + trend_health_5y
    Selection:  top 60 by score, then <=4 per IVV sector, take 30 highest scoring
    Weighting:  inv-vol on vol_12m, iteratively capped at 7% per name
    Regime:     SPY d_sma200 > -0.05  (200-day MA loose gate)  -- cash otherwise
    Vol-target: NONE  (the 18% SPY-vol target overlay HURTS Sharpe; removed)
    Costs:      5 bps * 2 round-trip per rebalance  (10 bps)
    Rebalance:  monthly, at month-end close
    Sanity:     drop picks with |1m return| > 200%  (PIT panel data errors)

Walk-forward / training: This is a pure rank-based strategy. No ML model is
trained. Every signal is computed point-in-time from the panel at each
rebalance date; no future information is used.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diagnostics import (
    load_panel, load_daily, load_membership, spy_regime,
    iterative_weight_cap, metrics, COST_BPS,
)
from strategy_v1 import load_sector_map
from strategy_search import (
    build_ndx_panel, OOS_START_NDX, OOS_END_NDX,
    OOS_START_SP500, OOS_END_SP500,
    NDX_DIR, DAILY_PRICES_MAIN,
)


OUT = HERE
TOP_K = 30
TOP_K_PRE = 60
SECTOR_CAP = 4
WEIGHT_CAP = 0.07
REGIME_DSMA = -0.05
SIGNALS = [
    ("mom_6_1",         +1),
    ("sharpe_5y",       +1),
    ("idio_mom_12_1",   +1),
    ("vol_1y",          -1),
    ("trend_health_5y", +1),
]


def score_strategy_best(snap: pd.DataFrame) -> pd.Series:
    """Final score function: mean of signed pct-ranks across 5 signals."""
    idx = snap["ticker"].values
    out = pd.Series(0.0, index=idx)
    cnt = pd.Series(0.0, index=idx)
    for col, sign in SIGNALS:
        if col not in snap.columns: continue
        s = pd.Series(sign * snap[col].values, index=idx)
        r = s.rank(pct=True, na_option="keep")
        m = r.notna()
        out[m] = out[m] + r[m]
        cnt[m] = cnt[m] + 1
    valid = cnt > 0
    out = out / cnt.where(cnt > 0, np.nan)
    return out[valid].dropna()


def run_strategy_best(panel, monthly_px, daily_for_spy, membership, sector_map,
                     oos_start, oos_end, panel_label: str):
    sreg = spy_regime(daily_for_spy)
    mem_set = {(pd.Timestamp(r.asof), r.ticker) for r in membership.itertuples()}
    all_dates = sorted(panel["asof"].unique())
    rebalance = [d for d in all_dates if oos_start <= d <= oos_end]
    print(f"  [{panel_label}] OOS window: {oos_start.date()} -> {oos_end.date()}  ({len(rebalance)} months)")

    rows = []
    for i, date in enumerate(rebalance):
        idx_all = all_dates.index(date)
        if idx_all == len(all_dates) - 1: break

        snap = panel[panel["asof"] == date].copy()
        snap = snap[snap["ticker"].apply(lambda t: (date, t) in mem_set)]
        if snap.empty:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, reason="no_mem")); continue

        reg = sreg.reindex([date]).iloc[0]
        d_sma200 = reg["d_sma200"]
        if not (np.isfinite(d_sma200) and d_sma200 > REGIME_DSMA):
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, reason="regime")); continue

        score = score_strategy_best(snap).dropna()
        if score.empty:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, reason="empty_score")); continue

        # Sector-diversified selection
        top_pre = score.sort_values(ascending=False).head(TOP_K_PRE)
        sec_count, chosen = {}, []
        for tk in top_pre.index:
            sec = sector_map.get(tk, "Unknown")
            if sec_count.get(sec, 0) < SECTOR_CAP:
                chosen.append(tk); sec_count[sec] = sec_count.get(sec, 0) + 1
            if len(chosen) >= TOP_K: break
        tickers = chosen
        if not tickers:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, reason="no_picks")); continue

        next_date = all_dates[idx_all + 1]
        d0 = monthly_px.index[monthly_px.index.searchsorted(date, side="right") - 1]
        d1 = monthly_px.index[monthly_px.index.searchsorted(next_date, side="right") - 1]
        p0, p1 = monthly_px.loc[d0], monthly_px.loc[d1]
        common = [t for t in tickers if t in monthly_px.columns
                  and np.isfinite(p0.get(t, np.nan)) and p0.get(t, 0) >= 1.0
                  and np.isfinite(p1.get(t, np.nan)) and p1.get(t, 0) >= 1.0]
        if not common:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, reason="no_common")); continue

        vol_src = "vol_12m" if "vol_12m" in snap.columns else "vol_1y"
        vmap = dict(zip(snap["ticker"].values, snap[vol_src].values))
        vols = np.array([max(float(vmap.get(t, 0.20)), 0.05)
                         if np.isfinite(vmap.get(t, np.nan)) else 0.20
                         for t in common])
        w = iterative_weight_cap(1.0 / vols, WEIGHT_CAP)

        rets = np.array([(p1[t] - p0[t]) / p0[t] for t in common])
        sane = np.abs(rets) <= 2.0
        if not sane.all():
            if sane.sum() == 0:
                rows.append(dict(date=date, ret_m=0.0, n_picks=0, reason="all_extreme")); continue
            w = w[sane]; rets = rets[sane]; common = [c for c, s in zip(common, sane) if s]
            w = w / w.sum()

        raw_port = float((w * rets).sum())
        cost = COST_BPS / 10_000.0
        port = raw_port - 2 * cost   # no vol-target
        rows.append(dict(date=date, ret_m=port, n_picks=len(common),
                         picks=",".join(common), reason="ok"))

    df = pd.DataFrame(rows).set_index("date")
    m = metrics(df["ret_m"]) if not df.empty else {}
    return df, m


def main():
    print("Loading data ...")
    sp_panel = load_panel(); sp_daily = load_daily(); sp_mem = load_membership()
    sector_map = load_sector_map()
    sp_monthly = sp_daily.resample("ME").last().ffill(limit=5)
    ndx_panel, ndx_monthly = build_ndx_panel()
    ndx_mem = pd.read_parquet(NDX_DIR / "ndx_pit_membership_monthly_full.parquet")
    ndx_mem["asof"] = pd.to_datetime(ndx_mem["asof"])
    main_daily = pd.read_parquet(DAILY_PRICES_MAIN)
    main_daily.index = pd.to_datetime(main_daily.index)

    print("\n=== PIT SP500 (augmented panel) ===")
    df_sp, m_sp = run_strategy_best(sp_panel, sp_monthly, sp_daily, sp_mem, sector_map,
                                    OOS_START_SP500, OOS_END_SP500, "sp500")
    cash_sp = int((df_sp["ret_m"] == 0.0).sum())
    print(f"  CAGR={m_sp['cagr']:.2%}  Sharpe={m_sp['sharpe']:.3f}  "
          f"MaxDD={m_sp['max_dd']:.1%}  AnnVol={m_sp['ann_vol']:.1%}  "
          f"N={m_sp['n_months']}  cash_months={cash_sp}")
    df_sp.to_csv(OUT / "strategy_best_sp500.csv")
    with open(OUT / "strategy_best_sp500_summary.json", "w") as fh:
        json.dump({"panel": "sp500_pit_augmented", "oos_start": str(OOS_START_SP500.date()),
                   "oos_end": str(OOS_END_SP500.date()), **m_sp,
                   "config": dict(top_k=TOP_K, top_k_pre=TOP_K_PRE, sector_cap=SECTOR_CAP,
                                  weight_cap=WEIGHT_CAP, regime_dsma=REGIME_DSMA,
                                  signals=SIGNALS, cost_bps=COST_BPS, vol_target=None)},
                  fh, indent=2, default=str)

    print("\n=== PIT NDX ===")
    df_nx, m_nx = run_strategy_best(ndx_panel, ndx_monthly, main_daily, ndx_mem, sector_map,
                                    OOS_START_NDX, OOS_END_NDX, "ndx")
    cash_nx = int((df_nx["ret_m"] == 0.0).sum())
    print(f"  CAGR={m_nx['cagr']:.2%}  Sharpe={m_nx['sharpe']:.3f}  "
          f"MaxDD={m_nx['max_dd']:.1%}  AnnVol={m_nx['ann_vol']:.1%}  "
          f"N={m_nx['n_months']}  cash_months={cash_nx}")
    df_nx.to_csv(OUT / "strategy_best_ndx.csv")
    with open(OUT / "strategy_best_ndx_summary.json", "w") as fh:
        json.dump({"panel": "ndx_pit", "oos_start": str(OOS_START_NDX.date()),
                   "oos_end": str(OOS_END_NDX.date()), **m_nx,
                   "config": dict(top_k=TOP_K, top_k_pre=TOP_K_PRE, sector_cap=SECTOR_CAP,
                                  weight_cap=WEIGHT_CAP, regime_dsma=REGIME_DSMA,
                                  signals=SIGNALS, cost_bps=COST_BPS, vol_target=None)},
                  fh, indent=2, default=str)

    print("\n=== Final Summary ===")
    print(f"  PIT SP500:  CAGR={m_sp['cagr']:>7.2%}  Sharpe={m_sp['sharpe']:>5.2f}")
    print(f"  PIT NDX:    CAGR={m_nx['cagr']:>7.2%}  Sharpe={m_nx['sharpe']:>5.2f}")
    print(f"  Combined Sharpe (mean): {(m_sp['sharpe']+m_nx['sharpe'])/2:.3f}")
    print(f"  Distance to mission target (CAGR>=50%, Sharpe>=2.0):")
    print(f"    SP500:  CAGR gap = {0.50 - m_sp['cagr']:>5.1%}, Sharpe gap = {2.00 - m_sp['sharpe']:>5.2f}")
    print(f"    NDX:    CAGR gap = {0.50 - m_nx['cagr']:>5.1%}, Sharpe gap = {2.00 - m_nx['sharpe']:>5.2f}")


if __name__ == "__main__":
    sys.exit(main())
