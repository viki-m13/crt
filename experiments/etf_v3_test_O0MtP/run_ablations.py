"""Run a focused ablation suite across all 4 universes (broad / levered /
combined ETFs + S&P 500 PIT) to find the best robust recipe.

Variants tested:
    - baseline:          K=3, EW,     no mid-hold, λ=0       (deployed v3)
    - k5_ew:             K=5, EW,     no mid-hold, λ=0
    - k5_ew_midhold:     K=5, EW,     mid-hold,    λ=0
    - k5_invvol:         K=5, invvol, no mid-hold, λ=0
    - k5_invvol_midhold: K=5, invvol, mid-hold,    λ=0
    - k3_ew_midhold:     K=3, EW,     mid-hold,    λ=0
    - k7_ew_midhold:     K=7, EW,     mid-hold,    λ=0   (more diversification)

Outputs results/ablations_<universe>.csv for each universe and
results/ablations_master.csv combining all rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = HERE / "cache"
RESULTS = HERE / "results"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from experiments.monthly_dca.v2.sp500_pit_extended_sweep import classify_regime_tight  # noqa: E402

from run_v3 import (  # noqa: E402
    evaluate_run, get_spy_features_per_month, load_panel,
)
from run_v3_robust import simulate_robust  # noqa: E402

V2 = ROOT / "experiments" / "monthly_dca" / "cache" / "v2"
FEATURES_DIR = ROOT / "experiments" / "monthly_dca" / "cache" / "features"


VARIANTS = [
    # (name, k, weighting, mid_hold, vol_lambda)
    ("baseline",         3, "ew",     False, 0.0),
    ("k5_ew",            5, "ew",     False, 0.0),
    ("k5_ew_midhold",    5, "ew",     True,  0.0),
    ("k5_invvol",        5, "invvol", False, 0.0),
    ("k5_invvol_midhold",5, "invvol", True,  0.0),
    ("k3_ew_midhold",    3, "ew",     True,  0.0),
    ("k7_ew_midhold",    7, "ew",     True,  0.0),
    ("k5_ew_volscore",   5, "ew",     False, 0.10),
    ("kitchen_sink",     5, "invvol", True,  0.10),
]


def run_etf(universe: str) -> pd.DataFrame:
    panel = load_panel(universe)
    feat = pd.read_parquet(CACHE / f"feat_{universe}.parquet")
    preds = pd.read_parquet(CACHE / f"preds_{universe}.parquet")
    spy = get_spy_features_per_month(feat)

    rows = []
    for name, k, weighting, midhold, lam in VARIANTS:
        # We use simulate_robust which is general; for "ew" it uses equal weights.
        # Override weighting='ew' inside simulate_robust requires a tweak —
        # see special-case below.
        eq = simulate_with_options(
            panel, preds, feat, spy,
            k=k, weighting=weighting,
            mid_hold_crash_check=midhold, vol_lambda=lam,
        )
        s = evaluate_run(eq, panel)
        rows.append({
            "universe": universe, "variant": name,
            "k": k, "weighting": weighting,
            "mid_hold": midhold, "vol_lambda": lam,
            "cagr_pct": s["cagr_full"] * 100,
            "spy_cagr_pct": s["spy_cagr_full"] * 100,
            "edge_pp": s["edge_full_pp"],
            "sharpe": s["sharpe"],
            "mdd_pct": s["max_dd"] * 100,
            "n_cash": s["n_cash"],
            "n_rebalances": s["n_rebalances"],
            "wf_mean_cagr_pct": (s["wf_mean_cagr"] or 0) * 100,
            "wf_mean_edge_pp": s["wf_mean_edge_pp"] or 0,
            "wf_n_beats_spy": s["wf_n_beats_spy"] or 0,
            "wf_n_splits": s["wf_n_splits"] or 0,
        })
        print(f"  {universe:8s} {name:22s} CAGR={s['cagr_full']*100:7.2f}%  "
              f"MDD={s['max_dd']*100:6.2f}%  Sharpe={s['sharpe']:.2f}  "
              f"WF={s['wf_n_beats_spy']}/{s['wf_n_splits']}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"ablations_{universe}.csv", index=False)
    return out


# Wrapper: simulate_robust always uses invvol unless lam=0 and weighting='ew'
# We need a unified simulator that respects 'ew' or 'invvol'. Add it.
def simulate_with_options(panel_prices, preds, feat_big, spy_features, *,
                          k, weighting, mid_hold_crash_check, vol_lambda):
    if weighting == "invvol":
        return simulate_robust(panel_prices, preds, feat_big, spy_features,
                               cost_bps=10.0, k=k, hold_months=6,
                               vol_lambda=vol_lambda,
                               mid_hold_crash_check=mid_hold_crash_check)
    # weighting == 'ew'
    monthly = panel_prices.resample("ME").last()
    mret = monthly.pct_change().clip(lower=-1.0, upper=2.0)
    mret_idx = mret.index
    excluded = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"}

    fb = feat_big.reset_index() if isinstance(feat_big.index, pd.MultiIndex) else feat_big.copy()
    fb["asof"] = pd.to_datetime(fb["asof"])
    p = preds.copy()
    p["asof"] = pd.to_datetime(p["asof"])
    if vol_lambda > 0 and "vol_1y" in fb.columns:
        p = p.merge(fb[["asof", "ticker", "vol_1y"]], on=["asof", "ticker"], how="left")
        p["vol_xs"] = p.groupby("asof")["vol_1y"].transform(lambda x: (x.rank(pct=True) - 0.5) * 2)
        p["score"] = p["pred"] - vol_lambda * p["vol_xs"].fillna(0.0)
    else:
        p["score"] = p["pred"]
    p = p[~p["ticker"].isin(excluded)].dropna(subset=["score"]).copy()

    by_asof = {pd.Timestamp(d): g.sort_values("score", ascending=False) for d, g in p.groupby("asof")}
    months = sorted(by_asof.keys())
    cf = 10.0 / 10000.0
    equity = 1.0
    cur_picks: list[str] = []
    cur_w = np.array([])
    held_for = 0
    cash = False
    rows = []

    for i, m in enumerate(months):
        s = spy_features.loc[m].to_dict() if m in spy_features.index else {}
        regime_now = classify_regime_tight(s)
        do_reb = (i == 0) or (held_for >= 6) or cash
        crash_force = (mid_hold_crash_check and regime_now == "crash" and not cash)
        do_reb = do_reb or crash_force

        if do_reb:
            if regime_now == "crash":
                cur_picks, cur_w, cash = [], np.array([]), True
                held_for = 0
            else:
                sub = by_asof.get(m)
                if sub is None or len(sub) < k:
                    cur_picks, cur_w, cash = [], np.array([]), True
                    held_for = 0
                else:
                    top = sub.head(k)
                    cur_picks = top["ticker"].tolist()
                    cur_w = np.ones(k) / k
                    cash = False
                    held_for = 0

        pos = mret_idx.searchsorted(m)
        cands = [(j, abs((mret_idx[j] - m).days)) for j in (pos - 1, pos) if 0 <= j < len(mret_idx)]
        cands.sort(key=lambda x: x[1])
        if cash or not cur_picks or not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mret_idx):
            ret_m = 0.0
        else:
            next_d = mret_idx[cands[0][0] + 1]
            picks_r = []
            for tk in cur_picks:
                if tk in mret.columns:
                    rv = mret.at[next_d, tk]
                    picks_r.append(-1.0 if pd.isna(rv) else float(rv))
                else:
                    picks_r.append(-1.0)
            ret_m = float((np.asarray(picks_r) * cur_w).sum())

        if not cash and cur_picks:
            equity *= (1 + ret_m) * (1 - cf if do_reb else 1.0)
        held_for += 1
        rows.append({
            "date": m, "equity": equity, "ret_m": ret_m,
            "regime": "cash" if cash else (regime_now if do_reb else "hold"),
            "n_picks": len(cur_picks),
            "picks": ",".join(cur_picks),
            "rebalance": int(do_reb),
            "crash_forced": int(crash_force),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SP500 ablations — use the same VARIANTS table.
# ---------------------------------------------------------------------------
def load_sp500_artifacts() -> dict:
    feat = pd.read_parquet(V2 / "sp500_pit" / "feature_panel_pit.parquet")
    feat["asof"] = pd.to_datetime(feat["asof"])
    preds = pd.read_parquet(V2 / "ml_preds_v2.parquet")
    preds["asof"] = pd.to_datetime(preds["asof"])
    membership = pd.read_parquet(V2 / "sp500_pit" / "sp500_membership_monthly.parquet")
    membership["asof"] = pd.to_datetime(membership["asof"])
    keys = set(zip(membership["asof"].to_numpy(), membership["ticker"].to_numpy()))
    pair_mask = [(a, t) in keys for a, t in zip(preds["asof"].to_numpy(), preds["ticker"].to_numpy())]
    preds = preds[pair_mask].copy()
    # Use ml_3plus6 score (matching deployed v3)
    if "pred_3m" in preds.columns and "pred_6m" in preds.columns:
        preds["pred"] = (preds["pred_3m"] + preds["pred_6m"]) / 2
    mret = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = _load_spy_feats_from_features_dir()
    return {"feat": feat, "preds": preds, "mret": mret, "spy_feats": spy_feats}


def _load_spy_feats_from_features_dir() -> pd.DataFrame:
    rows = []
    for f in sorted(FEATURES_DIR.glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "SPY" not in df.index:
            continue
        spy = df.loc["SPY"]
        rows.append({
            "asof": d,
            "spy_dsma200": float(spy.get("d_sma200", 0.0)),
            "spy_rsi14": float(spy.get("rsi_14", 50.0)),
            "spy_mom_12_1": float(spy.get("mom_12_1", 0.0)),
            "spy_mom_6_1": float(spy.get("mom_6_1", 0.0)),
            "spy_ret_21d": float(spy.get("ret_21d", 0.0)),
            "spy_below_200_streak": float(spy.get("max_below_200_streak", 0.0)),
            "spy_dd_from_52wh": float(spy.get("dd_from_52wh", 0.0)),
        })
    return pd.DataFrame(rows).set_index("asof")


def simulate_sp500_with_options(ds, *, k, weighting, mid_hold_crash_check, vol_lambda):
    """Simulator for SP500 using cached preds + monthly_returns_clean."""
    preds = ds["preds"].copy()
    feat = ds["feat"]
    mret = ds["mret"]
    spy_features = ds["spy_feats"]

    excluded = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
                "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
                "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD"}

    if vol_lambda > 0:
        preds = preds.merge(feat[["asof", "ticker", "vol_1y"]], on=["asof", "ticker"], how="left")
        preds["vol_xs"] = preds.groupby("asof")["vol_1y"].transform(lambda x: (x.rank(pct=True) - 0.5) * 2)
        preds["score"] = preds["pred"] - vol_lambda * preds["vol_xs"].fillna(0.0)
    else:
        preds["score"] = preds["pred"]
    preds = preds[~preds["ticker"].isin(excluded)].dropna(subset=["score"]).copy()
    by_asof = {pd.Timestamp(d): g.sort_values("score", ascending=False) for d, g in preds.groupby("asof")}
    months = sorted(by_asof.keys())
    mret_idx = mret.index
    cf = 10.0 / 10000.0

    feat_vol = feat[["asof", "ticker", "vol_1y"]].copy() if weighting == "invvol" else None

    equity = 1.0
    cur_picks: list[str] = []
    cur_w = np.array([])
    held_for = 0
    cash = False
    rows = []

    for i, m in enumerate(months):
        s = spy_features.loc[m].to_dict() if m in spy_features.index else {}
        regime_now = classify_regime_tight(s)
        do_reb = (i == 0) or (held_for >= 6) or cash
        crash_force = (mid_hold_crash_check and regime_now == "crash" and not cash)
        do_reb = do_reb or crash_force

        if do_reb:
            if regime_now == "crash":
                cur_picks, cur_w, cash = [], np.array([]), True
                held_for = 0
            else:
                sub = by_asof.get(m)
                if sub is None or len(sub) < k:
                    cur_picks, cur_w, cash = [], np.array([]), True
                    held_for = 0
                else:
                    top = sub.head(k)
                    cur_picks = top["ticker"].tolist()
                    if weighting == "invvol":
                        vsub = feat_vol[(feat_vol["asof"] == m) & (feat_vol["ticker"].isin(cur_picks))]
                        vol_map = vsub.set_index("ticker")["vol_1y"].to_dict()
                        vols = np.array([vol_map.get(t, 0.4) for t in cur_picks], dtype=float)
                        vols = np.where(np.isnan(vols) | (vols <= 0), 0.4, vols)
                        invv = 1.0 / vols
                        cur_w = invv / invv.sum()
                    else:
                        cur_w = np.ones(k) / k
                    cash = False
                    held_for = 0

        pos = mret_idx.searchsorted(m)
        cands = [(j, abs((mret_idx[j] - m).days)) for j in (pos - 1, pos) if 0 <= j < len(mret_idx)]
        cands.sort(key=lambda x: x[1])
        if cash or not cur_picks or not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mret_idx):
            ret_m = 0.0
        else:
            next_d = mret_idx[cands[0][0] + 1]
            picks_r = []
            for tk in cur_picks:
                if tk in mret.columns:
                    rv = mret.at[next_d, tk]
                    picks_r.append(-1.0 if pd.isna(rv) else float(rv))
                else:
                    picks_r.append(-1.0)
            ret_m = float((np.asarray(picks_r) * cur_w).sum())

        if not cash and cur_picks:
            equity *= (1 + ret_m) * (1 - cf if do_reb else 1.0)
        held_for += 1
        rows.append({
            "date": m, "equity": equity, "ret_m": ret_m,
            "regime": "cash" if cash else (regime_now if do_reb else "hold"),
            "n_picks": len(cur_picks),
            "picks": ",".join(cur_picks),
            "rebalance": int(do_reb),
            "crash_forced": int(crash_force),
        })
    return pd.DataFrame(rows)


def evaluate_sp500(eq_df: pd.DataFrame, mret: pd.DataFrame) -> dict:
    """Inline copy of run_sp500.evaluate (avoids import cycle on PIT-only setup)."""
    eq_df = eq_df.copy()
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    mret_idx = mret.index
    spy_m = mret["SPY"]

    aligned = []
    for d in eq_df["date"]:
        pos = mret_idx.searchsorted(pd.Timestamp(d))
        cands = [(j, abs((mret_idx[j] - d).days)) for j in (pos - 1, pos) if 0 <= j < len(mret_idx)]
        cands.sort(key=lambda x: x[1])
        if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mret_idx):
            aligned.append(0.0); continue
        v = spy_m.iloc[cands[0][0] + 1] if cands[0][0] + 1 < len(spy_m) else 0.0
        aligned.append(0.0 if pd.isna(v) else float(v))
    spy_s = pd.Series(aligned, index=eq_df["date"].values)

    r = eq_df["ret_m"].astype(float)
    def _cagr(x):
        if len(x) < 2: return 0.0
        return float(x.iloc[-1] ** (12.0 / len(x)) - 1.0)
    def _sh(rr):
        rs = rr.dropna()
        if len(rs) < 2 or rs.std() == 0: return 0.0
        return float((rs.mean() / rs.std()) * np.sqrt(12))
    def _mdd(rr):
        eq = (1 + rr).cumprod()
        if len(eq) == 0: return 0.0
        peak = eq.cummax()
        return float(((eq - peak) / peak).min())

    strat_cagr = _cagr((1 + r).cumprod()) if len(r) else 0.0
    spy_cagr = _cagr((1 + spy_s).cumprod())

    WF_SPLITS = [
        ("A1", "2011-01-01", "2018-12-31"),
        ("A2", "2015-01-01", "2021-12-31"),
        ("A3", "2018-01-01", "2024-12-31"),
        ("R1_GFC", "2008-01-01", "2010-12-31"),
        ("R2", "2011-01-01", "2013-12-31"),
        ("R3", "2014-01-01", "2016-12-31"),
        ("R4", "2017-01-01", "2019-12-31"),
        ("R5_COVID", "2020-01-01", "2022-12-31"),
        ("R6_AI", "2023-01-01", "2024-12-31"),
        ("STRICT", "2021-01-01", "2024-12-31"),
    ]
    wf_rows = []
    for split, lo, hi in WF_SPLITS:
        lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
        m = (eq_df["date"] >= lo_ts) & (eq_df["date"] <= hi_ts)
        if m.sum() < 12: continue
        rs = eq_df.loc[m, "ret_m"].astype(float)
        cv = _cagr((1 + rs).cumprod())
        spy_window = spy_s[m.values]
        scgr = _cagr((1 + spy_window).cumprod())
        wf_rows.append({"split": split, "cagr": cv, "spy_cagr": scgr, "edge_pp": (cv - scgr) * 100})
    wf = pd.DataFrame(wf_rows)
    return {
        "cagr_full": strat_cagr, "spy_cagr_full": spy_cagr,
        "edge_full_pp": (strat_cagr - spy_cagr) * 100,
        "sharpe": _sh(r), "max_dd": _mdd(r),
        "n_cash": int((eq_df["regime"] == "cash").sum()),
        "n_rebalances": int(eq_df["rebalance"].sum()),
        "wf_mean_cagr": float(wf["cagr"].mean()) if len(wf) else None,
        "wf_mean_edge_pp": float(wf["edge_pp"].mean()) if len(wf) else None,
        "wf_n_beats_spy": int((wf["cagr"] > wf["spy_cagr"]).sum()) if len(wf) else None,
        "wf_n_splits": int(len(wf)),
    }


def run_sp500() -> pd.DataFrame:
    print("\n=== SP500 ablations ===")
    ds = load_sp500_artifacts()
    rows = []
    for name, k, weighting, midhold, lam in VARIANTS:
        eq = simulate_sp500_with_options(ds, k=k, weighting=weighting,
                                          mid_hold_crash_check=midhold, vol_lambda=lam)
        s = evaluate_sp500(eq, ds["mret"])
        rows.append({
            "universe": "sp500_pit", "variant": name,
            "k": k, "weighting": weighting,
            "mid_hold": midhold, "vol_lambda": lam,
            "cagr_pct": s["cagr_full"] * 100,
            "spy_cagr_pct": s["spy_cagr_full"] * 100,
            "edge_pp": s["edge_full_pp"],
            "sharpe": s["sharpe"],
            "mdd_pct": s["max_dd"] * 100,
            "n_cash": s["n_cash"],
            "n_rebalances": s["n_rebalances"],
            "wf_mean_cagr_pct": (s["wf_mean_cagr"] or 0) * 100,
            "wf_mean_edge_pp": s["wf_mean_edge_pp"] or 0,
            "wf_n_beats_spy": s["wf_n_beats_spy"] or 0,
            "wf_n_splits": s["wf_n_splits"] or 0,
        })
        print(f"  sp500_pit {name:22s} CAGR={s['cagr_full']*100:7.2f}%  "
              f"MDD={s['max_dd']*100:6.2f}%  Sharpe={s['sharpe']:.2f}  "
              f"WF={s['wf_n_beats_spy']}/{s['wf_n_splits']}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ablations_sp500_pit.csv", index=False)
    return out


def main():
    parts = []
    for u in ("broad", "levered", "combined"):
        print(f"\n=== {u} ablations ===")
        parts.append(run_etf(u))
    parts.append(run_sp500())

    master = pd.concat(parts, axis=0, ignore_index=True)
    master.to_csv(RESULTS / "ablations_master.csv", index=False)
    print("\n=== ablations_master ===")
    cols = ["universe", "variant", "cagr_pct", "edge_pp", "sharpe", "mdd_pct",
            "wf_n_beats_spy", "wf_n_splits"]
    print(master[cols].to_string(index=False))


if __name__ == "__main__":
    main()
