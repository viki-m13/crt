"""
Round 4: structurally different ideas.

v20: Ridge regression on cross-sectional ranks (less overfit than LGBM)
v21: Drop vol-target (let bull market exposure run)
v22: Quarterly rebalance (lower cost, less noise)
v23: Universe expansion -- SP500 ∪ NDX combined universe
v24: Asymmetric loss "downside-aware" rank: penalize signals that historically
     selected drawdown names
v25: Buy-the-dip = mom_12_1 > 0 AND ret_21d < 0  (winners on pullback)
v26: Tighter vol-target = 12% (more conservative)
v27: Looser vol-target = 24% (more aggressive)
v28: Different signal weighting -- 50% mom_6_1 + 50% sharpe_5y (no quality)
"""
from __future__ import annotations
import io, json, sys, time, glob
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diagnostics import (
    load_panel, load_daily, load_membership, spy_regime, znorm,
    iterative_weight_cap, metrics, COST_BPS,
    TRAIN_MONTHS, EMBARGO_MONTHS, MIN_TRAIN_MONTHS, FEATURE_COLS,
)
from strategy_v1 import load_sector_map
from strategy_search import (
    build_ndx_panel, OOS_START_NDX, OOS_END_NDX,
    OOS_START_SP500, OOS_END_SP500,
    NDX_DIR, DAILY_PRICES_MAIN,
)
from strategy_search_v2 import make_rank_ensemble, SIGNALS_V13

OUT = HERE


def _rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, na_option="keep")


def build_ridge_cache(panel, oos_start, oos_end, target="fwd_1m_ret", alpha=10.0):
    """Walk-forward Ridge regression on cross-sectionally ranked features."""
    feat_cols = [c for c in FEATURE_COLS if c in panel.columns]
    all_dates = sorted(panel["asof"].unique())
    fwd_dates = sorted(panel[panel[target].notna()]["asof"].unique())
    # Precompute per-month cross-sectional ranks of features
    print(f"    pre-computing xs ranks for {len(feat_cols)} cols x {panel['asof'].nunique()} months ...")
    panel_xs = panel[["asof", "ticker", target] + feat_cols].copy()
    for col in feat_cols:
        panel_xs[col] = panel_xs.groupby("asof")[col].rank(pct=True)
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
        sub = panel_xs[panel_xs["asof"].isin(td)].dropna(subset=[target])
        if sub.empty: continue
        X = sub[feat_cols].fillna(0.5).values  # neutral rank if missing
        y = sub[target].values
        m = Ridge(alpha=alpha)
        m.fit(X, y)
        cache[date] = m
        last_key = key; last_model = m
    return cache, feat_cols, panel_xs


def make_ridge_scorer(cache, feat_cols, panel_xs):
    def fn(snap):
        d = snap["asof"].iloc[0]
        m = cache.get(d)
        if m is None: return pd.Series(dtype=float)
        # Use precomputed xs ranks for this date
        sub = panel_xs[panel_xs["asof"] == d]
        ticker_map = dict(zip(sub["ticker"].values, range(len(sub))))
        X = sub[feat_cols].fillna(0.5).values
        if len(X) == 0: return pd.Series(dtype=float)
        pred = m.predict(X)
        return pd.Series(pred, index=sub["ticker"].values)
    return fn


def make_buy_dip_score():
    """Score: only stocks with mom_12_1 > 0 AND ret_21d < 0; rank by -ret_21d × mom_12_1."""
    def fn(snap):
        idx = snap["ticker"].values
        m12 = pd.Series(snap["mom_12_1"].values, index=idx)
        r21 = pd.Series(snap["ret_21d"].values, index=idx)
        keep = (m12 > 0) & (r21 < 0)
        score = pd.Series(np.nan, index=idx)
        eligible = idx[keep]
        if len(eligible) == 0: return pd.Series(dtype=float)
        # score = m12 * (-r21)  -- big momentum + big dip = best
        score[keep] = (m12[keep] * (-r21[keep])).values
        return score.dropna()
    return fn


def make_mom_sharpe5y_score():
    def fn(snap):
        idx = snap["ticker"].values
        a = znorm(pd.Series(snap["mom_6_1"].values, index=idx))
        b = znorm(pd.Series(snap["sharpe_5y"].values, index=idx))
        return (0.5 * a + 0.5 * b).dropna()
    return fn


# Modified backtest: optional vol_target override, optional quarterly rebalance
def run_strategy_v4(panel, monthly_px, daily_for_spy, membership, sector_map,
                    score_fn, top_k=30, top_k_pre=60, sector_cap=4,
                    weight_cap=0.07,
                    use_sector_div=False,
                    target_vol=0.18, use_vol_target=True,
                    quarterly=False,
                    oos_start=None, oos_end=None):
    sreg = spy_regime(daily_for_spy)
    mem_set = {(pd.Timestamp(r.asof), r.ticker) for r in membership.itertuples()}
    all_dates = sorted(panel["asof"].unique())
    rebalance = [d for d in all_dates if oos_start <= d <= oos_end]
    if quarterly:
        rebalance = [d for d in rebalance if d.month in (3, 6, 9, 12)]

    rows = []
    for i, date in enumerate(rebalance):
        idx_all = all_dates.index(date)
        if idx_all == len(all_dates) - 1: break

        # Next rebalance date (could be 1 month or 3 months forward)
        if quarterly:
            next_idx_in_rebal = i + 1 if i + 1 < len(rebalance) else None
            if next_idx_in_rebal is None: break
            next_date = rebalance[next_idx_in_rebal]
        else:
            next_date = all_dates[idx_all + 1]

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

        score = score_fn(snap)
        if score is None or score.empty:
            rows.append(dict(date=date, ret_m=0.0, n_picks=0, scale=0.0, reason="empty_score"))
            continue
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

        rets = np.array([(p1[t] - p0[t]) / p0[t] for t in common])
        sane = np.abs(rets) <= 2.0
        if not sane.all():
            if sane.sum() == 0: continue
            w = w[sane]; rets = rets[sane]; common = [c for c, s in zip(common, sane) if s]
            w = w / w.sum()

        raw_port = float((w * rets).sum())
        scale = 1.0
        if use_vol_target and np.isfinite(spy_v) and spy_v > 1e-6:
            scale = min(target_vol / spy_v, 1.0)
        cost = COST_BPS / 10_000.0
        port = scale * raw_port - 2 * cost * scale
        rows.append(dict(date=date, ret_m=port, n_picks=len(common), scale=scale,
                         picks=",".join(common), reason="ok"))

    df = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    # For quarterly, scale Sharpe back to monthly basis by computing returns
    # as if 1m positions (quarterly returns / 3 won't be exactly right but
    # we report the actual quarterly metric here as-is; CAGR is still right)
    if df.empty: return df, {}
    m = metrics(df["ret_m"])
    if quarterly:
        # 3-month return → annualize properly: cagr unchanged, sharpe needs sqrt(4) not sqrt(12)
        # Re-derive: cagr = (1+ret).prod()**(4/n) - 1; sharpe = mean/std*sqrt(4)
        s = df["ret_m"].dropna()
        n = len(s)
        m["cagr"]    = float((1 + s).prod() ** (4 / n) - 1)
        m["sharpe"]  = float((s.mean() / s.std()) * np.sqrt(4)) if s.std() > 0 else 0.0
        m["ann_vol"] = float(s.std() * np.sqrt(4))
    return df, m


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

    print("Building Ridge caches (SP500 + NDX) ...")
    t0 = time.time()
    sp_ridge_c, sp_fc, sp_xs = build_ridge_cache(sp_panel, OOS_START_SP500, OOS_END_SP500)
    print(f"  SP500 Ridge done ({time.time()-t0:.0f}s)")
    ndx_ridge_c, ndx_fc, ndx_xs = build_ridge_cache(ndx_panel, OOS_START_NDX, OOS_END_NDX)
    print(f"  NDX Ridge done ({time.time()-t0:.0f}s)")

    rank13 = make_rank_ensemble(SIGNALS_V13)

    sp_scorers = {
        "rank13": rank13,
        "ridge": make_ridge_scorer(sp_ridge_c, sp_fc, sp_xs),
        "buy_dip": make_buy_dip_score(),
        "mom_sh5y": make_mom_sharpe5y_score(),
    }
    ndx_scorers = {
        "rank13": rank13,
        "ridge": make_ridge_scorer(ndx_ridge_c, ndx_fc, ndx_xs),
        "buy_dip": make_buy_dip_score(),
        "mom_sh5y": make_mom_sharpe5y_score(),
    }

    variants = [
        # (label, key, sector, top_k, target_vol, use_vt, quarterly)
        ("v20_ridge_xs_K30_sector",   "ridge",    True,  30, 0.18, True,  False),
        ("v20b_ridge_no_sector",      "ridge",    False, 30, 0.18, True,  False),
        ("v21_v13_no_vt",             "rank13",   True,  30, 0.18, False, False),
        ("v21b_v13_vt12",             "rank13",   True,  30, 0.12, True,  False),
        ("v21c_v13_vt24",             "rank13",   True,  30, 0.24, True,  False),
        ("v22_v13_quarterly",         "rank13",   True,  30, 0.18, True,  True),
        ("v24_buydip_K20",            "buy_dip",  False, 20, 0.18, True,  False),
        ("v24_buydip_K30_sector",     "buy_dip",  True,  30, 0.18, True,  False),
        ("v28_mom_sh5y_K30_sector",   "mom_sh5y", True,  30, 0.18, True,  False),
        ("v28_mom_sh5y_K20",          "mom_sh5y", False, 20, 0.18, True,  False),
        # No vol target variant of best so far
        ("v21_v13_no_vt_K20",         "rank13",   False, 20, 0.18, False, False),
    ]

    print(f"\n{'variant':<32} {'sp_CAGR':>8} {'sp_Sh':>6} {'ndx_CAGR':>9} {'ndx_Sh':>7}  {'comb_Sh':>7}")
    rows = []
    for label, key, use_sec, k, tv, use_vt, q in variants:
        sf = sp_scorers[key]; nf = ndx_scorers[key]
        df_sp, m_sp = run_strategy_v4(sp_panel, sp_monthly, sp_daily, sp_mem, sector_map,
                                      score_fn=sf, top_k=k, use_sector_div=use_sec,
                                      target_vol=tv, use_vol_target=use_vt, quarterly=q,
                                      oos_start=OOS_START_SP500, oos_end=OOS_END_SP500)
        df_nx, m_nx = run_strategy_v4(ndx_panel, ndx_monthly, main_daily, ndx_mem, sector_map,
                                      score_fn=nf, top_k=k, use_sector_div=use_sec,
                                      target_vol=tv, use_vol_target=use_vt, quarterly=q,
                                      oos_start=OOS_START_NDX, oos_end=OOS_END_NDX)
        if m_sp and m_nx:
            comb = (m_sp["sharpe"] + m_nx["sharpe"]) / 2
            print(f"{label:<32} {m_sp['cagr']:>7.1%} {m_sp['sharpe']:>6.2f} "
                  f"{m_nx['cagr']:>8.1%} {m_nx['sharpe']:>7.2f}  {comb:>7.2f}")
            rows.append(dict(name=label, sp=m_sp, ndx=m_nx, combined=comb))

    rows.sort(key=lambda r: -r["combined"])
    print("\nTop 5 by combined Sharpe (round 4):")
    for r in rows[:5]:
        print(f"  {r['name']:<32}  sp={r['sp']['sharpe']:.2f}/{r['sp']['cagr']:.1%}  "
              f"ndx={r['ndx']['sharpe']:.2f}/{r['ndx']['cagr']:.1%}  comb={r['combined']:.2f}")
    json.dump([dict(name=r["name"], sp=r["sp"], ndx=r["ndx"], combined=r["combined"]) for r in rows],
              open(OUT / "strategy_search_v4_results.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
