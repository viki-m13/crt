"""
Second round of search, building on findings from strategy_search.py:

WHAT WE LEARNED:
- Simple z(mom_6_1) + z(quality_score_5y) is the best signal (SP500 0.77, NDX 0.56)
- LGBM on these features overfits and degrades performance on both panels
- Sector cap helps NDX (more concentrated universe) but hurts SP500 slightly
- Breadth gate is too restrictive
- K=30 is fine; K=50 marginally better Sharpe but lower CAGR

WHAT WE TRY NEXT:
v8:  rank-ensemble of 8 hand-crafted signals (no LGBM)
v9:  v8 + conviction filter (skip when top-K aren't clearly differentiated)
v10: v8 + adaptive IC weighting (rolling 24m IC of each signal vs fwd_1m_ret)
v11: v2b + light LGBM weight (10% blend, not 50%)
v12: idio_mom_12_1 + quality_score_5y (idiosyncratic mom, removes market beta)
v13: ranks of [mom_6_1, sharpe_5y, idio_mom_12_1, -vol_1y, trend_health_5y]

All on PIT SP500 and PIT NDX with same overlays as v2b.
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
from strategy_v1 import load_sector_map
from strategy_search import (
    build_ndx_panel, build_lgbm_cache, run_strategy,
    OOS_START_SP500, OOS_END_SP500, OOS_START_NDX, OOS_END_NDX,
    NDX_DIR, DAILY_PRICES_MAIN,
)

OUT = HERE


def _rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, na_option="keep")


def make_rank_ensemble(feats_signs):
    """feats_signs: list of (column, sign). Score = sum of signed pct-ranks."""
    def fn(snap):
        idx = snap["ticker"].values
        out = pd.Series(0.0, index=idx)
        cnt = pd.Series(0, index=idx)
        for col, sign in feats_signs:
            if col not in snap.columns: continue
            vals = pd.Series(sign * snap[col].values, index=idx)
            r = _rank(vals)
            mask = r.notna()
            out[mask] = out[mask] + r[mask]
            cnt[mask] = cnt[mask] + 1
        valid = cnt > 0
        out = out / cnt.where(cnt > 0, np.nan)
        return out[valid].dropna()
    return fn


def make_conviction_filter(score_fn, gap_thresh=0.10):
    """Wrap a score fn so it returns empty when top-K isn't differentiated.
    gap = mean(top 30 scores) - median(all scores)."""
    def fn(snap):
        sc = score_fn(snap)
        if sc.empty: return sc
        top_avg = sc.sort_values(ascending=False).head(30).mean()
        med = sc.median()
        if (top_avg - med) < gap_thresh:
            return pd.Series(dtype=float)  # signals empty -> goes to cash
        return sc
    return fn


def make_light_lgbm_blend(cache_1, feat_cols, base_score_fn, w_lgbm=0.10):
    """LGBM at low weight + base score."""
    def fn(snap):
        d = snap["asof"].iloc[0]
        base = base_score_fn(snap)
        if base.empty: return base
        m = cache_1.get(d)
        if m is None: return base
        X = snap[feat_cols].copy().fillna(snap[feat_cols].median(numeric_only=True))
        p = m.predict(X)
        z = znorm(pd.Series(p, index=snap["ticker"].values))
        return ((1 - w_lgbm) * base + w_lgbm * z).dropna()
    return fn


def make_adaptive_ic_ensemble(panel, oos_start, oos_end, signals, lookback=24):
    """At each rebalance, weight each signal by its trailing 24m Spearman IC with fwd_1m_ret.
    Walk-forward: use only data with asof < current date."""
    all_dates = sorted(panel["asof"].unique())
    fwd_dates = sorted(panel[panel["fwd_1m_ret"].notna()]["asof"].unique())
    # Precompute monthly IC per signal
    ic_table = {}
    for col, sign in signals:
        ics = {}
        for d in fwd_dates:
            sub = panel[panel["asof"] == d]
            if col not in sub.columns or len(sub) < 30: continue
            try:
                ic = pd.Series(sign * sub[col].values).rank().corr(
                    pd.Series(sub["fwd_1m_ret"].values).rank())
                if np.isfinite(ic): ics[d] = ic
            except Exception:
                continue
        ic_table[col] = ics
    def fn(snap):
        d = snap["asof"].iloc[0]
        idx = snap["ticker"].values
        # Weights = mean trailing IC over last `lookback` months strictly before d
        prior = [pd.Timestamp(x) for x in fwd_dates if x < d]
        prior = prior[-lookback:]
        if not prior: return pd.Series(dtype=float)
        weights = {}
        for col, sign in signals:
            vals = [ic_table[col].get(p) for p in prior if p in ic_table[col]]
            vals = [v for v in vals if np.isfinite(v)]
            weights[col] = float(np.mean(vals)) if vals else 0.0
        # zero out negative-IC signals (anti-signals)
        w_pos = {c: max(0.0, w) for c, w in weights.items()}
        s = sum(w_pos.values())
        if s < 1e-6: return pd.Series(dtype=float)
        w_pos = {c: w / s for c, w in w_pos.items()}
        out = pd.Series(0.0, index=idx)
        cnt = pd.Series(0.0, index=idx)
        for col, sign in signals:
            if col not in snap.columns or w_pos.get(col, 0) == 0: continue
            r = _rank(pd.Series(sign * snap[col].values, index=idx))
            mask = r.notna()
            out[mask] = out[mask] + r[mask] * w_pos[col]
            cnt[mask] = cnt[mask] + w_pos[col]
        valid = cnt > 0
        out = out / cnt.where(cnt > 0, np.nan)
        return out[valid].dropna()
    return fn


# ---- Signal sets ----
SIGNALS_V8 = [
    ("mom_6_1", +1),
    ("mom_12_1", +1),
    ("sharpe_5y", +1),
    ("idio_mom_12_1", +1),
    ("quality_score_5y", +1),
    ("vol_1y", -1),
    ("trend_health_5y", +1),
    ("recovery_rate", +1),
]
SIGNALS_V13 = [
    ("mom_6_1", +1),
    ("sharpe_5y", +1),
    ("idio_mom_12_1", +1),
    ("vol_1y", -1),
    ("trend_health_5y", +1),
]


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

    # LGBM 1m caches (for v11)
    print("Building 1m LGBM caches (SP500 + NDX) ...")
    t0 = time.time()
    sp_c1, sp_fc = build_lgbm_cache(sp_panel, OOS_START_SP500, OOS_END_SP500, target="fwd_1m_ret")
    ndx_c1, ndx_fc = build_lgbm_cache(ndx_panel, OOS_START_NDX, OOS_END_NDX, target="fwd_1m_ret")
    print(f"  done ({time.time()-t0:.0f}s)")

    # Base scorers per panel (some closures need panel-specific data)
    def mq_score_fn():
        def fn(snap):
            idx = snap["ticker"].values
            m6  = znorm(pd.Series(snap["mom_6_1"].values, index=idx))
            qs5 = znorm(pd.Series(snap["quality_score_5y"].values, index=idx))
            return (0.6 * m6 + 0.4 * qs5).dropna()
        return fn

    # Adaptive IC ensembles need per-panel precomputation
    print("Building adaptive-IC tables ...")
    t0 = time.time()
    sp_adaptive = make_adaptive_ic_ensemble(sp_panel, OOS_START_SP500, OOS_END_SP500, SIGNALS_V8)
    ndx_adaptive = make_adaptive_ic_ensemble(ndx_panel, OOS_START_NDX, OOS_END_NDX, SIGNALS_V8)
    print(f"  done ({time.time()-t0:.0f}s)")

    sp_scorers = {
        "v2b_baseline":      mq_score_fn(),
        "v8_rank_ensemble8": make_rank_ensemble(SIGNALS_V8),
        "v8_conviction":     make_conviction_filter(make_rank_ensemble(SIGNALS_V8), gap_thresh=0.10),
        "v10_adaptive_ic":   sp_adaptive,
        "v11_light_lgbm":    make_light_lgbm_blend(sp_c1, sp_fc, mq_score_fn(), w_lgbm=0.10),
        "v12_idio_quality":  make_rank_ensemble([("idio_mom_12_1", +1), ("quality_score_5y", +1)]),
        "v13_five_signal":   make_rank_ensemble(SIGNALS_V13),
    }
    ndx_scorers = {
        "v2b_baseline":      mq_score_fn(),
        "v8_rank_ensemble8": make_rank_ensemble(SIGNALS_V8),
        "v8_conviction":     make_conviction_filter(make_rank_ensemble(SIGNALS_V8), gap_thresh=0.10),
        "v10_adaptive_ic":   ndx_adaptive,
        "v11_light_lgbm":    make_light_lgbm_blend(ndx_c1, ndx_fc, mq_score_fn(), w_lgbm=0.10),
        "v12_idio_quality":  make_rank_ensemble([("idio_mom_12_1", +1), ("quality_score_5y", +1)]),
        "v13_five_signal":   make_rank_ensemble(SIGNALS_V13),
    }

    variants = [
        # (label, key, use_sec, top_k, qf, exclude_vol)
        ("v2b_baseline",             "v2b_baseline",     False, 30, False, 1.0),
        ("v8_rank_ensemble8",        "v8_rank_ensemble8",False, 30, False, 1.0),
        ("v8_rank_ensemble8_qf",     "v8_rank_ensemble8",False, 30, True,  0.7),
        ("v8_rank_ensemble8_sector", "v8_rank_ensemble8",True,  30, False, 1.0),
        ("v9_v8_conviction",         "v8_conviction",    False, 30, False, 1.0),
        ("v9_conviction_sector",     "v8_conviction",    True,  30, False, 1.0),
        ("v10_adaptive_ic",          "v10_adaptive_ic",  False, 30, False, 1.0),
        ("v10_adaptive_sector",      "v10_adaptive_ic",  True,  30, False, 1.0),
        ("v11_light_lgbm",           "v11_light_lgbm",   False, 30, False, 1.0),
        ("v12_idio_quality",         "v12_idio_quality", False, 30, False, 1.0),
        ("v13_five_signal",          "v13_five_signal",  False, 30, False, 1.0),
        ("v13_five_signal_sector",   "v13_five_signal",  True,  30, False, 1.0),
        ("v8_K50",                   "v8_rank_ensemble8",False, 50, False, 1.0),
        ("v8_K20",                   "v8_rank_ensemble8",False, 20, False, 1.0),
    ]

    print(f"\n{'variant':<32} {'sp_CAGR':>8} {'sp_Sh':>6} {'ndx_CAGR':>9} {'ndx_Sh':>7}  {'comb_Sh':>7}")
    rows = []
    for label, key, use_sec, k, qf, ev in variants:
        sf = sp_scorers[key]; nf = ndx_scorers[key]
        df_sp, m_sp = run_strategy(sp_panel, sp_monthly, sp_daily, sp_mem, sector_map,
                                   score_fn=sf, top_k=k,
                                   use_breadth=False, use_sector_div=use_sec,
                                   adaptive_k=False, quality_filter=qf, exclude_vol=ev,
                                   oos_start=OOS_START_SP500, oos_end=OOS_END_SP500,
                                   panel_label="sp500")
        df_nx, m_nx = run_strategy(ndx_panel, ndx_monthly, main_daily, ndx_mem, sector_map,
                                   score_fn=nf, top_k=k,
                                   use_breadth=False, use_sector_div=use_sec,
                                   adaptive_k=False, quality_filter=qf, exclude_vol=ev,
                                   oos_start=OOS_START_NDX, oos_end=OOS_END_NDX,
                                   panel_label="ndx")
        if m_sp and m_nx:
            comb = (m_sp["sharpe"] + m_nx["sharpe"]) / 2
            print(f"{label:<32} {m_sp['cagr']:>7.1%} {m_sp['sharpe']:>6.2f} "
                  f"{m_nx['cagr']:>8.1%} {m_nx['sharpe']:>7.2f}  {comb:>7.2f}")
            rows.append(dict(name=label, sp=m_sp, ndx=m_nx, combined=comb))
            df_sp.to_csv(OUT / f"backtest_{label}_sp500.csv")
            df_nx.to_csv(OUT / f"backtest_{label}_ndx.csv")

    rows.sort(key=lambda r: -r["combined"])
    print("\nTop 5 by combined Sharpe (this round):")
    for r in rows[:5]:
        print(f"  {r['name']:<32}  sp={r['sp']['sharpe']:.2f}/{r['sp']['cagr']:.1%}  "
              f"ndx={r['ndx']['sharpe']:.2f}/{r['ndx']['cagr']:.1%}  "
              f"comb={r['combined']:.2f}")
    json.dump([dict(name=r["name"], sp=r["sp"], ndx=r["ndx"], combined=r["combined"]) for r in rows],
              open(OUT / "strategy_search_v2_results.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
