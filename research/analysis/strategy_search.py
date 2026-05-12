"""
Systematic strategy search on BOTH PIT panels.

Every variant is evaluated on PIT SP500 AND PIT NDX with the same config.
Winner = variant that improves over the v1 baseline on BOTH panels jointly.

Variants:
v2: mom_6_1 + quality_score_5y, no breadth gate
v3: walk-forward LGBM on rank_target_1m
v4: multi-horizon LGBM ensemble (1m + 3m + 6m)
v5: v4 + sector cap
v6: blend(multi-horizon LGBM, mom_6_1, quality_score_5y)
v6 K variants: K=10, 20, 50
v7: v6 + adaptive K
"""
from __future__ import annotations
import io, json, subprocess, sys, time, glob
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diagnostics import (
    load_panel, load_daily, load_membership, spy_regime, znorm,
    iterative_weight_cap, metrics,
    COST_BPS, TARGET_VOL,
    TRAIN_MONTHS, EMBARGO_MONTHS, MIN_TRAIN_MONTHS, LGB_PARAMS, FEATURE_COLS,
)
from strategy_v1 import load_sector_map, breadth_above_200ma

OUT = HERE
REPO = Path("/home/user/crt")
NDX_DIR = REPO / "experiments/monthly_dca/v5/qqq_pit"
FEAT_DIR = REPO / "experiments/monthly_dca/cache/features"
DAILY_PRICES_MAIN = REPO / "experiments/monthly_dca/cache/prices_extended.parquet"

OOS_START_SP500 = pd.Timestamp("2007-01-31")
OOS_END_SP500   = pd.Timestamp("2024-04-30")
OOS_START_NDX   = pd.Timestamp("2019-04-30")  # NDX membership starts 2015-01
OOS_END_NDX     = pd.Timestamp("2025-12-31")


# ---------------------------------------------------------------------------
# NDX panel builder (same as research/validation/ndx_pit/run_pit_ndx_validation.py)
# ---------------------------------------------------------------------------
def build_ndx_panel(start=pd.Timestamp("2010-01-31"), end=pd.Timestamp("2026-02-28")):
    ndx_mem = pd.read_parquet(NDX_DIR / "ndx_pit_membership_monthly_full.parquet")
    ndx_universe = set(ndx_mem["ticker"].unique())
    ndx_monthly_px = pd.read_parquet(NDX_DIR / "ndx_monthly_prices.parquet")
    feat_files = sorted(glob.glob(str(FEAT_DIR / "*.parquet")))
    frames = []
    for f in feat_files:
        date = pd.Timestamp(Path(f).stem)
        if date < start or date > end: continue
        df = pd.read_parquet(f)
        df = df[df.index.isin(ndx_universe)].copy()
        if df.empty: continue
        df["asof"] = date
        df.index.name = "ticker"
        df = df.reset_index()
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)

    # Forward returns from NDX monthly prices
    next_px = ndx_monthly_px.shift(-1)
    fwd_1m = (next_px - ndx_monthly_px) / ndx_monthly_px

    next3 = ndx_monthly_px.shift(-3)
    fwd_3m = (next3 / ndx_monthly_px) - 1
    next6 = ndx_monthly_px.shift(-6)
    fwd_6m = (next6 / ndx_monthly_px) - 1

    def get_fwd(df_fwd, row):
        d, t = row["asof"], row["ticker"]
        if d not in df_fwd.index or t not in df_fwd.columns: return np.nan
        v = df_fwd.at[d, t]
        return float(v) if np.isfinite(v) else np.nan

    panel["fwd_1m_ret"] = panel.apply(lambda r: get_fwd(fwd_1m, r), axis=1)
    panel["fwd_3m_ret"] = panel.apply(lambda r: get_fwd(fwd_3m, r), axis=1)
    panel["fwd_6m_ret"] = panel.apply(lambda r: get_fwd(fwd_6m, r), axis=1)
    return panel, ndx_monthly_px


# ---------------------------------------------------------------------------
# LGBM training
# ---------------------------------------------------------------------------
def fit_lgbm(panel, train_dates, feat_cols, target="fwd_1m_ret", params=None):
    sub = panel[panel["asof"].isin(train_dates)].copy().dropna(subset=[target])
    if sub.empty: return None
    X = sub[feat_cols].copy().fillna(sub[feat_cols].median(numeric_only=True))
    y = sub[target].values
    p = dict(LGB_PARAMS) if params is None else dict(params)
    m = lgb.LGBMRegressor(**p)
    m.fit(X, y)
    return m


def build_lgbm_cache(panel, oos_start, oos_end, target="fwd_1m_ret", params=None):
    feat_cols = [c for c in FEATURE_COLS if c in panel.columns]
    all_dates = sorted(panel["asof"].unique())
    fwd_dates = sorted(panel[panel[target].notna()]["asof"].unique())
    cache = {}
    last_key = None; last_model = None
    for i, date in enumerate(all_dates):
        if date < oos_start: continue
        if date > oos_end: break
        cutoff_idx = i - EMBARGO_MONTHS
        if cutoff_idx < 0: continue
        train_end = all_dates[cutoff_idx]
        pool = [d for d in fwd_dates if d <= train_end and d < date]
        if len(pool) < MIN_TRAIN_MONTHS: continue
        td = pool[-TRAIN_MONTHS:]
        key = (td[0], td[-1])
        if key == last_key:
            cache[date] = last_model; continue
        m = fit_lgbm(panel, td, feat_cols, target=target, params=params)
        cache[date] = m
        last_key = key; last_model = m
    return cache, feat_cols


# ---------------------------------------------------------------------------
# Score function makers (closures over panel + caches)
# ---------------------------------------------------------------------------
def make_1m_only(cache, feat_cols):
    def fn(snap):
        d = snap["asof"].iloc[0]
        m = cache.get(d)
        if m is None: return pd.Series(dtype=float)
        X = snap[feat_cols].copy().fillna(snap[feat_cols].median(numeric_only=True))
        return pd.Series(m.predict(X), index=snap["ticker"].values)
    return fn


def make_multi(cache_1, cache_3, cache_6, feat_cols):
    def fn(snap):
        d = snap["asof"].iloc[0]
        X = snap[feat_cols].copy().fillna(snap[feat_cols].median(numeric_only=True))
        total = None
        for cache, w in [(cache_1, 0.5), (cache_3, 0.3), (cache_6, 0.2)]:
            m = cache.get(d)
            if m is None: continue
            p = m.predict(X)
            z = znorm(pd.Series(p, index=snap["ticker"].values))
            total = z * w if total is None else total + z * w
        return total if total is not None else pd.Series(dtype=float)
    return fn


def make_blend(cache_1, cache_3, cache_6, feat_cols):
    multi = make_multi(cache_1, cache_3, cache_6, feat_cols)
    def fn(snap):
        ml = multi(snap)
        if ml.empty: return ml
        m6  = znorm(pd.Series(snap["mom_6_1"].values,          index=snap["ticker"].values))
        qs5 = znorm(pd.Series(snap["quality_score_5y"].values, index=snap["ticker"].values))
        return (0.50*ml + 0.30*m6 + 0.20*qs5).dropna()
    return fn


def make_mom_quality(snap_to_ticker=lambda s: s["ticker"].values):
    def fn(snap):
        m6  = znorm(pd.Series(snap["mom_6_1"].values,          index=snap_to_ticker(snap)))
        qs5 = znorm(pd.Series(snap["quality_score_5y"].values, index=snap_to_ticker(snap)))
        return (0.6*m6 + 0.4*qs5).dropna()
    return fn


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def run_strategy(panel, monthly_px, daily_for_spy, membership, sector_map,
                 score_fn, top_k=30, top_k_pre=60, sector_cap=4,
                 weight_cap=0.07,
                 use_breadth=False, breadth_thresh=0.40,
                 quality_filter=True, exclude_vol=0.7, exclude_dd=-0.5,
                 use_sector_div=False,
                 adaptive_k=False, adaptive_k_threshold=1.5,
                 oos_start=None, oos_end=None,
                 panel_label="?"):
    sreg = spy_regime(daily_for_spy)
    breadth = breadth_above_200ma(daily_for_spy) if use_breadth else None
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
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="regime_dsma"))
            continue
        if use_breadth:
            b = breadth.reindex([date]).iloc[0] if not breadth.empty else np.nan
            if not (np.isfinite(b) and b > breadth_thresh):
                rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="regime_breadth"))
                continue

        if quality_filter:
            keep = pd.Series(True, index=snap.index)
            if "vol_1y" in snap.columns:
                keep &= snap["vol_1y"].fillna(0) <= exclude_vol
            if "dd_from_52wh" in snap.columns:
                keep &= snap["dd_from_52wh"].fillna(0) >= exclude_dd
            snap = snap[keep]
            if snap.empty:
                rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="empty_filter"))
                continue

        score = score_fn(snap)
        if score is None or score.empty:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="empty_score"))
            continue
        score = score.dropna()

        if adaptive_k and len(score) > 5:
            sorted_sc = score.sort_values(ascending=False)
            top_avg = sorted_sc.head(10).mean()
            med = sorted_sc.median()
            gap = (top_avg - med) / (score.std() + 1e-9)
            k_now = 10 if gap > adaptive_k_threshold else top_k
        else:
            k_now = top_k

        top_pre = score.sort_values(ascending=False).head(top_k_pre)
        if use_sector_div:
            sec_count = {}; chosen = []
            for tk in top_pre.index:
                sec = sector_map.get(tk, "Unknown")
                if sec_count.get(sec, 0) < sector_cap:
                    chosen.append(tk); sec_count[sec] = sec_count.get(sec, 0) + 1
                if len(chosen) >= k_now: break
            tickers = chosen
        else:
            tickers = top_pre.head(k_now).index.tolist()
        if not tickers:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="no_picks"))
            continue

        next_date = all_dates[idx_all + 1]
        if date not in monthly_px.index or next_date not in monthly_px.index:
            d0 = monthly_px.index[monthly_px.index.searchsorted(date, side="right") - 1]
            d1 = monthly_px.index[monthly_px.index.searchsorted(next_date, side="right") - 1]
        else:
            d0, d1 = date, next_date
        p0, p1 = monthly_px.loc[d0], monthly_px.loc[d1]
        common = [t for t in tickers
                  if t in monthly_px.columns
                  and np.isfinite(p0.get(t, np.nan)) and p0.get(t, 0) >= 1.0
                  and np.isfinite(p1.get(t, np.nan)) and p1.get(t, 0) >= 1.0]
        if not common:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="no_common"))
            continue

        vmap = dict(zip(snap["ticker"].values, snap.get("vol_12m", snap["vol_1y"]).values))
        vols = np.array([max(float(vmap.get(t, 0.20)), 0.05)
                         if np.isfinite(vmap.get(t, np.nan)) else 0.20
                         for t in common])
        w = iterative_weight_cap(1.0 / vols, weight_cap)

        rets = np.array([(p1[t] - p0[t]) / p0[t] for t in common])
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
                         picks=",".join(common), reason="ok", k_used=k_now))

    df = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    return df, (metrics(df["ret_m"]) if not df.empty else {})


def main():
    print("Loading data ...")
    sp_panel = load_panel()
    sp_daily = load_daily()
    sp_mem = load_membership()
    sector_map = load_sector_map()
    sp_monthly = sp_daily.resample("ME").last().ffill(limit=5)

    print("Building NDX panel ...")
    ndx_panel, ndx_monthly = build_ndx_panel()
    ndx_mem = pd.read_parquet(NDX_DIR / "ndx_pit_membership_monthly_full.parquet")
    ndx_mem["asof"] = pd.to_datetime(ndx_mem["asof"])
    main_daily = pd.read_parquet(DAILY_PRICES_MAIN)
    main_daily.index = pd.to_datetime(main_daily.index)
    print(f"  sp500_panel: {sp_panel.shape}   ndx_panel: {ndx_panel.shape}")

    # ---- Build LGBM caches per panel ----
    print("\nLGBM caches: SP500 ...")
    t0 = time.time()
    sp_c1, sp_fc = build_lgbm_cache(sp_panel, OOS_START_SP500, OOS_END_SP500, target="fwd_1m_ret")
    sp_c3, _    = build_lgbm_cache(sp_panel, OOS_START_SP500, OOS_END_SP500, target="fwd_3m_ret")
    sp_c6, _    = build_lgbm_cache(sp_panel, OOS_START_SP500, OOS_END_SP500, target="fwd_6m_ret")
    print(f"  SP500 LGBMs done ({time.time()-t0:.0f}s)")

    print("LGBM caches: NDX ...")
    t0 = time.time()
    ndx_c1, ndx_fc = build_lgbm_cache(ndx_panel, OOS_START_NDX, OOS_END_NDX, target="fwd_1m_ret")
    ndx_c3, _      = build_lgbm_cache(ndx_panel, OOS_START_NDX, OOS_END_NDX, target="fwd_3m_ret")
    ndx_c6, _      = build_lgbm_cache(ndx_panel, OOS_START_NDX, OOS_END_NDX, target="fwd_6m_ret")
    print(f"  NDX LGBMs done ({time.time()-t0:.0f}s)")

    # ---- Variants ----
    variants = [
        ("v2_mom_qual_no_breadth",     "mq",     False, False, False, 30, True,  0.7),
        ("v2b_mom_qual_no_filter",     "mq",     False, False, False, 30, False, 1.0),
        ("v3_lgbm_1m",                 "lgbm1",  False, False, False, 30, True,  0.7),
        ("v4_multi_horizon",           "multi",  False, False, False, 30, True,  0.7),
        ("v5_multi_sector",            "multi",  False, True,  False, 30, True,  0.7),
        ("v6_blend",                   "blend",  False, False, False, 30, True,  0.7),
        ("v6_blend_sector",            "blend",  False, True,  False, 30, True,  0.7),
        ("v6_blend_K10",               "blend",  False, False, False, 10, True,  0.7),
        ("v6_blend_K20",               "blend",  False, False, False, 20, True,  0.7),
        ("v6_blend_K50",               "blend",  False, False, False, 50, True,  0.7),
        ("v7_blend_adaptive_k",        "blend",  False, False, True,  30, True,  0.7),
        ("v6_blend_no_filter",         "blend",  False, False, False, 30, False, 1.0),
        ("v6_blend_strict_quality",    "blend",  False, False, False, 30, True,  0.5),
        ("v6_blend_breadth",           "blend",  True,  False, False, 30, True,  0.7),
    ]

    sp_scorers = dict(
        mq=make_mom_quality(),
        lgbm1=make_1m_only(sp_c1, sp_fc),
        multi=make_multi(sp_c1, sp_c3, sp_c6, sp_fc),
        blend=make_blend(sp_c1, sp_c3, sp_c6, sp_fc),
    )
    ndx_scorers = dict(
        mq=make_mom_quality(),
        lgbm1=make_1m_only(ndx_c1, ndx_fc),
        multi=make_multi(ndx_c1, ndx_c3, ndx_c6, ndx_fc),
        blend=make_blend(ndx_c1, ndx_c3, ndx_c6, ndx_fc),
    )

    print(f"\n{'variant':<30} {'sp_CAGR':>8} {'sp_Sh':>6} {'ndx_CAGR':>9} {'ndx_Sh':>7}  {'comb_Sh':>7}")
    rows = []
    for name, fn_key, use_b, use_sec, adapt_k, k, qf, ev in variants:
        # SP500
        sf = sp_scorers[fn_key]
        df_sp, m_sp = run_strategy(sp_panel, sp_monthly, sp_daily, sp_mem, sector_map,
                                   score_fn=sf, top_k=k,
                                   use_breadth=use_b, use_sector_div=use_sec,
                                   adaptive_k=adapt_k, quality_filter=qf, exclude_vol=ev,
                                   oos_start=OOS_START_SP500, oos_end=OOS_END_SP500,
                                   panel_label="sp500")
        # NDX
        nf = ndx_scorers[fn_key]
        df_nx, m_nx = run_strategy(ndx_panel, ndx_monthly, main_daily, ndx_mem, sector_map,
                                   score_fn=nf, top_k=k,
                                   use_breadth=use_b, use_sector_div=use_sec,
                                   adaptive_k=adapt_k, quality_filter=qf, exclude_vol=ev,
                                   oos_start=OOS_START_NDX, oos_end=OOS_END_NDX,
                                   panel_label="ndx")
        if m_sp and m_nx:
            comb = (m_sp["sharpe"] + m_nx["sharpe"]) / 2
            print(f"{name:<30} {m_sp['cagr']:>7.1%} {m_sp['sharpe']:>6.2f} "
                  f"{m_nx['cagr']:>8.1%} {m_nx['sharpe']:>7.2f}  {comb:>7.2f}")
            rows.append(dict(name=name, sp=m_sp, ndx=m_nx, combined_sharpe=comb))
            df_sp.to_csv(OUT / f"backtest_{name}_sp500.csv")
            df_nx.to_csv(OUT / f"backtest_{name}_ndx.csv")

    rows.sort(key=lambda r: -r["combined_sharpe"])
    print("\nTop 5 by combined Sharpe:")
    for r in rows[:5]:
        print(f"  {r['name']:<30}  sp Sharpe={r['sp']['sharpe']:.2f}  ndx Sharpe={r['ndx']['sharpe']:.2f}  combined={r['combined_sharpe']:.2f}")
    json.dump([dict(name=r["name"], sp=r["sp"], ndx=r["ndx"], combined=r["combined_sharpe"]) for r in rows],
              open(OUT / "strategy_search_results.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
