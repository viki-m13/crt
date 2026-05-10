"""Apply v3-baseline and v3-robust simulators to the deployed S&P 500 PIT
universe — apples-to-apples with the ETF runs.

Reuses the production v3 walk-forward predictions (cache/v2/ml_preds_v2.parquet)
and the PIT feature panel (cache/v2/sp500_pit/feature_panel_pit.parquet) — these
are the actual deployed-model artifacts. Returns come from the production
monthly_returns_clean.parquet (data-error-cleaned monthly returns).

For SPY features (regime gate), we read SPY rows directly from the per-month
feature parquets in cache/features/.

Outputs (under experiments/etf_v3_test_O0MtP/results/):
    sp500_baseline_equity.csv
    sp500_baseline_picks.csv
    sp500_baseline_summary.json
    sp500_baseline_walkforward.csv
    sp500_baseline_yearly.csv
    sp500_robust_equity.csv     ... etc

Run:
    python3 experiments/etf_v3_test_O0MtP/run_sp500.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from experiments.monthly_dca.v2.sp500_pit_extended_sweep import classify_regime_tight  # noqa: E402

V2 = ROOT / "experiments" / "monthly_dca" / "cache" / "v2"
FEATURES_DIR = ROOT / "experiments" / "monthly_dca" / "cache" / "features"


# ---------------------------------------------------------------------------
def load_spy_features() -> pd.DataFrame:
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


def load_panel() -> dict:
    """Load preds + feature panel + monthly returns + SPY features.

    Predictions are filtered to PIT-S&P-500 membership at each asof — this
    matches the deployed v3 setup (build_panel_with_score in
    sp500_pit_extended_sweep.py), where picks are drawn only from
    point-in-time S&P 500 members.
    """
    feat = pd.read_parquet(V2 / "sp500_pit" / "feature_panel_pit.parquet")
    feat["asof"] = pd.to_datetime(feat["asof"])
    preds = pd.read_parquet(V2 / "ml_preds_v2.parquet")
    preds["asof"] = pd.to_datetime(preds["asof"])
    mret = pd.read_parquet(V2 / "monthly_returns_clean.parquet")

    membership = pd.read_parquet(V2 / "sp500_pit" / "sp500_membership_monthly.parquet")
    membership["asof"] = pd.to_datetime(membership["asof"])
    keys = set(zip(membership["asof"].to_numpy(), membership["ticker"].to_numpy()))
    print(f"  PIT-S&P-500 membership: {len(keys):,} (asof, ticker) pairs over {membership['asof'].nunique()} months")
    pre = len(preds)
    preds = preds[
        list(zip(preds["asof"].to_numpy(), preds["ticker"].to_numpy()))
        |  # type: ignore[operator]
        ()  # placeholder to satisfy mypy; replaced below
    ] if False else preds  # noqa  (we filter via apply below for clarity)
    pair_mask = [
        (a, t) in keys for a, t in zip(preds["asof"].to_numpy(), preds["ticker"].to_numpy())
    ]
    preds = preds[pair_mask].copy()
    print(f"  preds filtered to PIT members: {pre:,} -> {len(preds):,}")

    spy_feats = load_spy_features()
    return {"feat": feat, "preds": preds, "mret": mret, "spy_feats": spy_feats}


# ---------------------------------------------------------------------------
EXCLUDED = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
            "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
            "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD"}


def _next_month_idx(mret_idx: pd.DatetimeIndex, asof: pd.Timestamp) -> int:
    pos = mret_idx.searchsorted(asof)
    cands = [(j, abs((mret_idx[j] - asof).days)) for j in (pos - 1, pos)
             if 0 <= j < len(mret_idx)]
    cands.sort(key=lambda x: x[1])
    if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mret_idx):
        return -1
    return cands[0][0] + 1


def _attach_score_baseline(preds: pd.DataFrame) -> pd.DataFrame:
    """Baseline v3 score = ml_3plus6 (mean of pred_3m and pred_6m)."""
    p = preds.copy()
    if "pred_3m" in p.columns and "pred_6m" in p.columns:
        p["score"] = (p["pred_3m"] + p["pred_6m"]) / 2
    else:
        p["score"] = p["pred"]
    return p


def _attach_score_robust(preds: pd.DataFrame, feat: pd.DataFrame, lam: float = 0.10) -> pd.DataFrame:
    """Robust score = ml_3plus6 - lam * vol_xs."""
    p = preds.copy()
    if "pred_3m" in p.columns and "pred_6m" in p.columns:
        p["score"] = (p["pred_3m"] + p["pred_6m"]) / 2
    else:
        p["score"] = p["pred"]
    vol = feat[["asof", "ticker", "vol_1y"]].copy()
    p = p.merge(vol, on=["asof", "ticker"], how="left")
    p["vol_xs"] = p.groupby("asof")["vol_1y"].transform(lambda x: (x.rank(pct=True) - 0.5) * 2)
    p["score"] = p["score"] - lam * p["vol_xs"].fillna(0.0)
    return p


def simulate_v3(
    preds_with_score: pd.DataFrame,
    feat: pd.DataFrame,
    mret: pd.DataFrame,
    spy_feats: pd.DataFrame,
    *,
    k: int,
    hold_months: int,
    weighting: str,        # 'ew' or 'invvol'
    cost_bps: float,
    mid_hold_crash_check: bool,
) -> pd.DataFrame:
    """Generic v3 simulator (handles baseline and robust)."""
    p = preds_with_score[~preds_with_score["ticker"].isin(EXCLUDED)].dropna(subset=["score"]).copy()
    by_asof = {pd.Timestamp(d): g.sort_values("score", ascending=False)
               for d, g in p.groupby("asof")}
    months = sorted(by_asof.keys())
    mret_idx = mret.index
    cf = cost_bps / 10000.0

    # Vol lookup
    feat_vol = feat[["asof", "ticker", "vol_1y"]].copy() if weighting == "invvol" else None

    equity = 1.0
    cur_picks: list[str] = []
    cur_w = np.array([])
    held_for = 0
    cash = False
    rows = []

    for i, m in enumerate(months):
        s = spy_feats.loc[m].to_dict() if m in spy_feats.index else {}
        regime_now = classify_regime_tight(s)
        do_reb = (i == 0) or (held_for >= hold_months) or cash
        crash_force = (mid_hold_crash_check and regime_now == "crash" and not cash)
        do_reb = do_reb or crash_force
        regime_label = regime_now if do_reb else "hold"

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

        # Apply next-month return
        nxt_pos = _next_month_idx(mret_idx, m)
        if cash or not cur_picks or nxt_pos < 0:
            ret_m = 0.0
        else:
            next_d = mret_idx[nxt_pos]
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
            "regime": "cash" if cash else regime_label,
            "n_picks": len(cur_picks),
            "picks": ",".join(cur_picks),
            "weights": ",".join(f"{w:.3f}" for w in cur_w) if len(cur_w) else "",
            "rebalance": int(do_reb),
            "crash_forced": int(crash_force),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
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


def _cagr(eq: pd.Series) -> float:
    if len(eq) < 2:
        return 0.0
    return float(eq.iloc[-1] ** (12.0 / len(eq)) - 1.0)


def _sharpe_m(r: pd.Series) -> float:
    rs = r.dropna()
    if len(rs) < 2 or rs.std() == 0:
        return 0.0
    return float((rs.mean() / rs.std()) * np.sqrt(12))


def _max_dd(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    if len(eq) == 0:
        return 0.0
    peak = eq.cummax()
    return float(((eq - peak) / peak).min())


def evaluate(eq_df: pd.DataFrame, mret: pd.DataFrame) -> dict:
    eq_df = eq_df.copy()
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    mret_idx = mret.index
    spy_m = mret["SPY"]

    aligned = []
    for d in eq_df["date"]:
        nxt = _next_month_idx(mret_idx, pd.Timestamp(d))
        if nxt < 0:
            aligned.append(0.0)
            continue
        v = spy_m.iloc[nxt] if nxt < len(spy_m) else 0.0
        aligned.append(0.0 if pd.isna(v) else float(v))
    spy_s = pd.Series(aligned, index=eq_df["date"].values)

    r = eq_df["ret_m"].astype(float)
    strat_cagr = _cagr((1 + r).cumprod()) if len(r) else 0.0
    spy_cagr = _cagr((1 + spy_s).cumprod())
    strat_sh = _sharpe_m(r)
    strat_mdd = _max_dd(r)

    wf_rows = []
    for split, lo, hi in WF_SPLITS:
        lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
        m = (eq_df["date"] >= lo_ts) & (eq_df["date"] <= hi_ts)
        if m.sum() < 12:
            continue
        rs = eq_df.loc[m, "ret_m"].astype(float)
        cv = _cagr((1 + rs).cumprod())
        spy_window = spy_s[m.values]
        scgr = _cagr((1 + spy_window).cumprod())
        wf_rows.append({
            "split": split, "from": lo, "to": hi, "n_m": int(m.sum()),
            "cagr": cv, "spy_cagr": scgr, "edge_pp": (cv - scgr) * 100,
            "sharpe": _sharpe_m(rs), "max_dd": _max_dd(rs),
            "n_cash": int((eq_df.loc[m, "regime"] == "cash").sum()),
        })
    wf = pd.DataFrame(wf_rows)
    return {
        "n_months": int(len(r)),
        "first_month": str(eq_df["date"].min().date()) if len(eq_df) else None,
        "last_month": str(eq_df["date"].max().date()) if len(eq_df) else None,
        "cagr_full": strat_cagr, "spy_cagr_full": spy_cagr,
        "edge_full_pp": (strat_cagr - spy_cagr) * 100,
        "sharpe": strat_sh, "max_dd": strat_mdd,
        "n_cash": int((eq_df["regime"] == "cash").sum()),
        "n_rebalances": int(eq_df["rebalance"].sum()),
        "wf_mean_cagr": float(wf["cagr"].mean()) if len(wf) else None,
        "wf_median_cagr": float(wf["cagr"].median()) if len(wf) else None,
        "wf_min_cagr": float(wf["cagr"].min()) if len(wf) else None,
        "wf_max_cagr": float(wf["cagr"].max()) if len(wf) else None,
        "wf_mean_edge_pp": float(wf["edge_pp"].mean()) if len(wf) else None,
        "wf_n_pos": int((wf["cagr"] > 0).sum()) if len(wf) else None,
        "wf_n_beats_spy": int((wf["cagr"] > wf["spy_cagr"]).sum()) if len(wf) else None,
        "wf_n_splits": int(len(wf)),
        "wf_table": wf.to_dict("records"),
    }


def yearly(eq_df: pd.DataFrame, mret: pd.DataFrame) -> pd.DataFrame:
    eq = eq_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq["year"] = eq["date"].dt.year
    mret_idx = mret.index
    spy_aligned = []
    for d in eq["date"]:
        nxt = _next_month_idx(mret_idx, pd.Timestamp(d))
        v = mret["SPY"].iloc[nxt] if nxt >= 0 else 0.0
        spy_aligned.append(0.0 if pd.isna(v) else float(v))
    eq["spy_ret"] = spy_aligned
    yr = eq.groupby("year")["ret_m"].apply(lambda x: ((1 + x).prod() - 1)).rename("strat_year_ret")
    syr = eq.groupby("year")["spy_ret"].apply(lambda x: ((1 + x).prod() - 1)).rename("spy_year_ret")
    out = yr.to_frame().join(syr.to_frame(), how="left")
    out["edge_pp"] = (out["strat_year_ret"] - out["spy_year_ret"]) * 100
    return out.reset_index()


# ---------------------------------------------------------------------------
def run_variant(name: str, *, k: int, weighting: str, hold_months: int,
                cost_bps: float, mid_hold_crash_check: bool, robust_score: bool,
                ds: dict) -> dict:
    print(f"\n--- {name} ---")
    if robust_score:
        p = _attach_score_robust(ds["preds"], ds["feat"], lam=0.10)
    else:
        p = _attach_score_baseline(ds["preds"])

    eq = simulate_v3(
        p, ds["feat"], ds["mret"], ds["spy_feats"],
        k=k, hold_months=hold_months, weighting=weighting,
        cost_bps=cost_bps, mid_hold_crash_check=mid_hold_crash_check,
    )
    eq.to_csv(RESULTS / f"sp500_{name}_equity.csv", index=False)
    eq[eq["rebalance"] == 1][["date", "regime", "n_picks", "picks", "weights", "crash_forced"]].to_csv(
        RESULTS / f"sp500_{name}_picks.csv", index=False
    )
    yearly(eq, ds["mret"]).to_csv(RESULTS / f"sp500_{name}_yearly.csv", index=False)

    summary = evaluate(eq, ds["mret"])
    summary["universe"] = "sp500_pit"
    summary["variant"] = name
    summary["spec"] = {
        "k": k, "weighting": weighting, "hold_months": hold_months,
        "cost_bps": cost_bps, "mid_hold_crash_check": mid_hold_crash_check,
        "robust_score": robust_score,
    }
    with open(RESULTS / f"sp500_{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    pd.DataFrame(summary["wf_table"]).to_csv(RESULTS / f"sp500_{name}_walkforward.csv", index=False)

    print(f"  CAGR={summary['cagr_full']*100:.2f}%  SPY={summary['spy_cagr_full']*100:.2f}%  "
          f"edge={summary['edge_full_pp']:+.2f}pp  MDD={summary['max_dd']*100:.2f}%  "
          f"Sharpe={summary['sharpe']:.2f}  cash_m={summary['n_cash']}")
    if summary["wf_mean_cagr"] is not None:
        print(f"  WF mean CAGR={summary['wf_mean_cagr']*100:.2f}%  "
              f"+{summary['wf_mean_edge_pp']:+.2f}pp  "
              f"({summary['wf_n_beats_spy']}/{summary['wf_n_splits']} beat SPY)")
    return summary


def main():
    print("=== loading S&P 500 PIT v3 artifacts ===")
    ds = load_panel()
    print(f"  preds={ds['preds'].shape}  feat={ds['feat'].shape}  mret={ds['mret'].shape}")

    # v3 baseline (deployed): K=3 EW tight h=6 cost=10bps, NO mid-hold check
    base = run_variant(
        "baseline", k=3, weighting="ew", hold_months=6,
        cost_bps=10.0, mid_hold_crash_check=False, robust_score=False, ds=ds,
    )

    # v3 robust v1 (kitchen sink): K=5 invvol tight h=6 cost=10bps,
    # mid-hold crash check ON, vol-adjusted score. Initial guess.
    rob = run_variant(
        "robust", k=5, weighting="invvol", hold_months=6,
        cost_bps=10.0, mid_hold_crash_check=True, robust_score=True, ds=ds,
    )

    # v3-robust-v2 (the actual winner): just K=5 + inverse-vol weighting.
    # No mid-hold crash check, no vol-adjusted score. Ablations below
    # showed mid-hold and vol-score either neutral or harmful on the
    # production cross-section, while invvol-K5 is a clean Pareto
    # improvement (higher CAGR, same MDD, higher Sharpe, more WF beats).
    robust_v2 = run_variant(
        "robust_v2_invvol_k5", k=5, weighting="invvol", hold_months=6,
        cost_bps=10.0, mid_hold_crash_check=False, robust_score=False, ds=ds,
    )

    # Bonus ablations: which improvement matters most on S&P 500?
    abl_a = run_variant(
        "ablation_midhold_only", k=3, weighting="ew", hold_months=6,
        cost_bps=10.0, mid_hold_crash_check=True, robust_score=False, ds=ds,
    )
    abl_b = run_variant(
        "ablation_volscore_only", k=3, weighting="ew", hold_months=6,
        cost_bps=10.0, mid_hold_crash_check=False, robust_score=True, ds=ds,
    )
    abl_c = run_variant(
        "ablation_invvolk5_only", k=5, weighting="invvol", hold_months=6,
        cost_bps=10.0, mid_hold_crash_check=False, robust_score=False, ds=ds,
    )

    # Roll-up table
    rows = []
    for s in (base, rob, robust_v2, abl_a, abl_b, abl_c):
        rows.append({
            "variant": s["variant"],
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
    cmp = pd.DataFrame(rows)
    cmp.to_csv(RESULTS / "sp500_compare.csv", index=False)
    print("\n=== sp500_compare.csv ===")
    print(cmp.to_string(index=False))


if __name__ == "__main__":
    main()
