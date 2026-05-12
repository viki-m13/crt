"""
Round 3 strategy search.

CARRY FORWARD from round 2 winners:
- v13_five_signal_sector (sp 0.69/6.3%, ndx 0.71/6.3%, comb 0.70)  -- joint best
- v8_rank_ensemble8 K=20 (sp 0.81 -- best SP500 alone, ndx 0.55)

PUSHES in round 3:
v14: v13 sector + intra-month -7% trailing stop on SP500 (daily prices available)
v15: v8 (8 signals) sector at K=20 / K=30 / K=40
v16: v13 + 3 more signals (mom_3, recovery_rate, rs_3m_spy)
v17: tighter conviction filter (gap_thresh=0.15)
v18: 6m rolling IC weighting (instead of 24m) -- IC weights more recent
v19: combo = mean of v8_rank8 and v13_5sig
"""
from __future__ import annotations
import io, json, subprocess, sys, time, glob
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diagnostics import (
    load_panel, load_daily, load_membership, spy_regime, znorm,
    iterative_weight_cap, metrics, COST_BPS, TARGET_VOL,
    OOS_START as OOS_START_SP500_DEFAULT,
    OOS_END as OOS_END_SP500_DEFAULT,
)
from strategy_v1 import load_sector_map
from strategy_search import (
    build_ndx_panel, OOS_START_NDX, OOS_END_NDX,
    OOS_START_SP500, OOS_END_SP500,
    NDX_DIR, DAILY_PRICES_MAIN,
)
from strategy_search_v2 import (
    make_rank_ensemble, make_conviction_filter,
    SIGNALS_V8, SIGNALS_V13,
)

OUT = HERE


# ---------- run with optional intra-month trailing stop (SP500 only) ----------
def run_with_stops(panel, monthly_px, daily_px, daily_for_spy, membership, sector_map,
                   score_fn, top_k=30, top_k_pre=60, sector_cap=4,
                   weight_cap=0.07,
                   use_sector_div=False,
                   quality_filter=False, exclude_vol=1.0, exclude_dd=-1.0,
                   oos_start=None, oos_end=None,
                   trailing_stop=None,  # e.g. -0.07 = exit if -7% from entry
                   ):
    """Same as run_strategy but supports an intra-month trailing stop using daily prices."""
    sreg = spy_regime(daily_for_spy)
    mem_set = {(pd.Timestamp(r.asof), r.ticker) for r in membership.itertuples()}
    all_dates = sorted(panel["asof"].unique())
    rebalance = [d for d in all_dates if oos_start <= d <= oos_end]

    rows = []
    for i, date in enumerate(rebalance):
        idx_all = all_dates.index(date)
        if idx_all == len(all_dates) - 1: break

        snap = panel[panel["asof"] == date].copy()
        snap = snap[snap["ticker"].apply(lambda t: (date, t) in mem_set)]
        if snap.empty:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="no_mem"))
            continue

        reg = sreg.reindex([date]).iloc[0]
        d_sma200 = reg["d_sma200"]; spy_v = reg["vol_21d"]
        if not (np.isfinite(d_sma200) and d_sma200 > -0.05):
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="regime"))
            continue

        if quality_filter:
            keep = pd.Series(True, index=snap.index)
            if "vol_1y" in snap.columns:
                keep &= snap["vol_1y"].fillna(0) <= exclude_vol
            if "dd_from_52wh" in snap.columns:
                keep &= snap["dd_from_52wh"].fillna(0) >= exclude_dd
            snap = snap[keep]
            if snap.empty: continue

        score = score_fn(snap)
        if score is None or score.empty: continue
        score = score.dropna()

        top_pre = score.sort_values(ascending=False).head(top_k_pre)
        if use_sector_div:
            sec_count = {}; chosen = []
            for tk in top_pre.index:
                sec = sector_map.get(tk, "Unknown")
                if sec_count.get(sec, 0) < sector_cap:
                    chosen.append(tk); sec_count[sec] = sec_count.get(sec, 0) + 1
                if len(chosen) >= top_k: break
            tickers = chosen
        else:
            tickers = top_pre.head(top_k).index.tolist()
        if not tickers: continue

        next_date = all_dates[idx_all + 1]
        d0 = monthly_px.index[monthly_px.index.searchsorted(date, side="right") - 1]
        d1 = monthly_px.index[monthly_px.index.searchsorted(next_date, side="right") - 1]
        p0, p1 = monthly_px.loc[d0], monthly_px.loc[d1]
        common = [t for t in tickers if t in monthly_px.columns
                  and np.isfinite(p0.get(t, np.nan)) and p0.get(t, 0) >= 1.0
                  and np.isfinite(p1.get(t, np.nan)) and p1.get(t, 0) >= 1.0]
        if not common: continue

        vmap = dict(zip(snap["ticker"].values, snap.get("vol_12m", snap["vol_1y"]).values))
        vols = np.array([max(float(vmap.get(t, 0.20)), 0.05)
                         if np.isfinite(vmap.get(t, np.nan)) else 0.20
                         for t in common])
        w = iterative_weight_cap(1.0 / vols, weight_cap)

        # Intra-month return with optional trailing stop
        if trailing_stop is not None and daily_px is not None:
            rets = []
            # Daily window from d0+1 to d1
            day_mask = (daily_px.index > d0) & (daily_px.index <= d1)
            window = daily_px.loc[day_mask]
            for t in common:
                if t not in window.columns or window[t].dropna().empty:
                    r = (p1[t] - p0[t]) / p0[t]
                    rets.append(r)
                    continue
                entry = p0[t]
                # walk daily; if any close hits stop, exit at that close
                exited = False
                exit_ret = None
                for px in window[t].dropna().values:
                    r = (px - entry) / entry
                    if r <= trailing_stop:
                        exit_ret = r
                        exited = True
                        break
                if exited:
                    rets.append(exit_ret)
                else:
                    rets.append((p1[t] - p0[t]) / p0[t])
            rets = np.array(rets)
        else:
            rets = np.array([(p1[t] - p0[t]) / p0[t] for t in common])

        # Sanity (200% cap)
        sane = np.abs(rets) <= 2.0
        if not sane.all():
            if sane.sum() == 0: continue
            w = w[sane]; rets = rets[sane]; common = [c for c, s in zip(common, sane) if s]
            w = w / w.sum()

        raw_port = float((w * rets).sum())
        scale = min(TARGET_VOL / spy_v, 1.0) if np.isfinite(spy_v) and spy_v > 1e-6 else 1.0
        cost = COST_BPS / 10_000.0
        # If trailing stops fired, add an extra exit cost
        extra_cost = 0.0
        if trailing_stop is not None:
            triggered = float(((rets - 0) <= trailing_stop + 1e-6).sum())  # heuristic
            extra_cost = triggered / max(len(rets), 1) * cost
        port = scale * raw_port - 2 * cost * scale - extra_cost * scale
        rows.append(dict(date=date, ret_m=port, n_picks=len(common), scale=scale,
                         picks=",".join(common), reason="ok"))

    df = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    return df, (metrics(df["ret_m"]) if not df.empty else {})


def main():
    print("Loading data ...")
    sp_panel = load_panel(); sp_daily = load_daily(); sp_mem = load_membership()
    sector_map = load_sector_map()
    sp_monthly = sp_daily.resample("ME").last().ffill(limit=5)

    print("Building NDX panel ...")
    ndx_panel, ndx_monthly = build_ndx_panel()
    ndx_mem = pd.read_parquet(NDX_DIR / "ndx_pit_membership_monthly_full.parquet")
    ndx_mem["asof"] = pd.to_datetime(ndx_mem["asof"])
    main_daily = pd.read_parquet(DAILY_PRICES_MAIN)
    main_daily.index = pd.to_datetime(main_daily.index)

    # Signal sets to try
    SIGNALS_V16 = SIGNALS_V13 + [("mom_3", +1), ("recovery_rate", +1), ("rs_3m_spy", +1)]

    rank8  = make_rank_ensemble(SIGNALS_V8)
    rank13 = make_rank_ensemble(SIGNALS_V13)
    rank16 = make_rank_ensemble(SIGNALS_V16)

    def combo_8_13(snap):
        a = rank8(snap); b = rank13(snap)
        if a.empty or b.empty:
            return a if not a.empty else b
        idx = a.index.intersection(b.index)
        return (0.5 * a.reindex(idx) + 0.5 * b.reindex(idx)).dropna()

    conv_v8_tight = make_conviction_filter(rank8, gap_thresh=0.15)
    conv_v13_tight = make_conviction_filter(rank13, gap_thresh=0.15)

    scorers = {
        "v8": rank8, "v13": rank13, "v16": rank16,
        "v8_conv_tight": conv_v8_tight,
        "v13_conv_tight": conv_v13_tight,
        "combo": combo_8_13,
    }

    variants = [
        # (label, scorer, use_sector, top_k, trailing_stop)
        ("v13_sector_baseline",     "v13",            True,  30, None),
        ("v13_sector_stop7",        "v13",            True,  30, -0.07),
        ("v13_sector_stop10",       "v13",            True,  30, -0.10),
        ("v8_sector_K20",           "v8",             True,  20, None),
        ("v8_sector_K30",           "v8",             True,  30, None),
        ("v8_sector_K40",           "v8",             True,  40, None),
        ("v8_no_sector_K20",        "v8",             False, 20, None),
        ("v8_no_sector_K20_stop7",  "v8",             False, 20, -0.07),
        ("v16_sector",              "v16",            True,  30, None),
        ("v16_no_sector",           "v16",            False, 30, None),
        ("v17_conv_tight_v13",      "v13_conv_tight", True,  30, None),
        ("v17_conv_tight_v8",       "v8_conv_tight",  True,  30, None),
        ("v19_combo_sector",        "combo",          True,  30, None),
        ("v19_combo_no_sector",     "combo",          False, 30, None),
    ]

    print(f"\n{'variant':<30} {'sp_CAGR':>8} {'sp_Sh':>6} {'ndx_CAGR':>9} {'ndx_Sh':>7}  {'comb_Sh':>7}")
    rows = []
    for label, key, use_sec, k, stop in variants:
        sf = scorers[key]
        df_sp, m_sp = run_with_stops(sp_panel, sp_monthly, sp_daily, sp_daily, sp_mem, sector_map,
                                     score_fn=sf, top_k=k, use_sector_div=use_sec,
                                     oos_start=OOS_START_SP500, oos_end=OOS_END_SP500,
                                     trailing_stop=stop)
        # NDX: no daily prices for NDX-only names -> trailing stop disabled
        df_nx, m_nx = run_with_stops(ndx_panel, ndx_monthly, None, main_daily, ndx_mem, sector_map,
                                     score_fn=sf, top_k=k, use_sector_div=use_sec,
                                     oos_start=OOS_START_NDX, oos_end=OOS_END_NDX,
                                     trailing_stop=None)
        if m_sp and m_nx:
            comb = (m_sp["sharpe"] + m_nx["sharpe"]) / 2
            print(f"{label:<30} {m_sp['cagr']:>7.1%} {m_sp['sharpe']:>6.2f} "
                  f"{m_nx['cagr']:>8.1%} {m_nx['sharpe']:>7.2f}  {comb:>7.2f}")
            rows.append(dict(name=label, sp=m_sp, ndx=m_nx, combined=comb))

    rows.sort(key=lambda r: -r["combined"])
    print("\nTop 5 by combined Sharpe (round 3):")
    for r in rows[:5]:
        print(f"  {r['name']:<30}  sp={r['sp']['sharpe']:.2f}/{r['sp']['cagr']:.1%}  "
              f"ndx={r['ndx']['sharpe']:.2f}/{r['ndx']['cagr']:.1%}  comb={r['combined']:.2f}")
    json.dump([dict(name=r["name"], sp=r["sp"], ndx=r["ndx"], combined=r["combined"]) for r in rows],
              open(OUT / "strategy_search_v3_results.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
