"""Generalization test of the v3 winner config on alternative universes.

The PIT S&P 500 winner config:
    scorer  = ml_filter (existing v2 GBM predictions)
    K       = 3 normal / 3 recovery / 3 bull
    weight  = EW
    gate    = tight
    hold    = 6 months
    cap     = 1.0 (no cap)
    cost    = 10 bp/month

We re-run this same config on multiple universes to check that the result is
not an artefact of the S&P 500 cohort:

    1. Full broader universe (1,833 tickers)
    2. PIT NON-S&P-500 large/mid (Russell 3000 minus S&P 500)
    3. PIT S&P 500 (the winner)
    4. Random subset (500 tickers, multiple seeds)
    5. Top-100 most-liquid (proxied by market-cap rank)

Saves comparative table to v3_generalization.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"

import sys
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v2"))
from sp500_pit_extended_sweep import (  # noqa: E402
    EXCLUDE, WF_SPLITS, REGIME_GATES, load_spy_features, simulate_variant,
    Variant,
)
from sp500_pit_v3_validate import per_split_eval  # noqa: E402


WINNER = Variant(
    name="ml_filter|k3_3_3|ew|tight|h6|cap1.0",
    scorer="ml_filter", k_normal=3, k_recovery=3, k_bull=3,
    weighting="ew", regime_gate="tight", hold_months=6, cap_per_pick=1.0,
)


def evaluate_simple(eq, spy_aligned):
    ret = eq["ret_m"].astype(float)
    eqc = (1 + ret).cumprod()
    cgr = (eqc.iloc[-1]) ** (12.0 / len(eqc)) - 1 if len(eqc) else 0
    sh = (ret.mean() / max(ret.std(), 1e-9)) * np.sqrt(12)
    peak = eqc.cummax()
    mdd = float(((eqc - peak) / peak).min())

    wf_rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy()
        if len(e) == 0:
            continue
        r = e["ret_m"].astype(float)
        ec = (1 + r).cumprod()
        cv = (ec.iloc[-1]) ** (12.0 / len(ec)) - 1
        spy = spy_aligned[(spy_aligned["date"] >= lo) & (spy_aligned["date"] <= hi)]
        sr = spy["spy_ret_m"].astype(float)
        sc = (1 + sr).cumprod()
        scgr = (sc.iloc[-1]) ** (12.0 / len(sc)) - 1
        wf_rows.append({"split": split, "cagr": cv, "spy_cagr": scgr,
                        "edge_pp": (cv - scgr) * 100})
    wf = pd.DataFrame(wf_rows)
    return {
        "cagr_full": cgr, "sharpe": sh, "max_dd": mdd,
        "wf_mean_cagr": float(wf["cagr"].mean()) if len(wf) else 0,
        "wf_min_cagr": float(wf["cagr"].min()) if len(wf) else 0,
        "wf_max_cagr": float(wf["cagr"].max()) if len(wf) else 0,
        "wf_mean_edge_pp": float(wf["edge_pp"].mean()) if len(wf) else 0,
        "wf_n_pos": int((wf["cagr"] > 0).sum()) if len(wf) else 0,
        "wf_n_beats": int((wf["cagr"] > wf["spy_cagr"]).sum()) if len(wf) else 0,
        "n_cash": int((eq["regime"] == "cash").sum()),
    }


def build_panel_for_universe(universe_tickers_per_month: dict[pd.Timestamp, set[str]] | None = None,
                              ticker_filter_set: set[str] | None = None) -> pd.DataFrame:
    """Build a scored panel from v2 ML predictions joined on ticker eligibility.

    universe_tickers_per_month: per-month asof -> set of eligible tickers
    ticker_filter_set: a fixed set of tickers (used across all months)

    Either / or.  Returns a DataFrame with [asof, ticker, score, vol_1y].
    """
    ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")
    ml["asof"] = pd.to_datetime(ml["asof"])
    ml = ml.rename(columns={"pred": "score"})

    # Need vol_1y per (asof,ticker) for invvol weighting (we use EW so this is
    # actually unused, but keep for completeness)
    panel = pd.read_parquet(PIT / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    # We'll re-merge with the broader panel; for non-S&P 500, the
    # cache/v2 panel doesn't have rows. So we'll work without the panel cache.

    full = ml.copy()
    full = full[~full["ticker"].isin(EXCLUDE)]
    if universe_tickers_per_month is not None:
        # Filter by per-month membership
        keep = []
        for d, sub in full.groupby("asof"):
            elig = universe_tickers_per_month.get(pd.Timestamp(d), set())
            keep.append(sub[sub["ticker"].isin(elig)])
        full = pd.concat(keep, axis=0, ignore_index=True)
    elif ticker_filter_set is not None:
        full = full[full["ticker"].isin(ticker_filter_set)]
    full["vol_1y"] = 0.3  # placeholder
    return full


def run_universe(name: str, panel: pd.DataFrame, monthly_returns: pd.DataFrame,
                 spy_features, spy_aligned) -> dict:
    eq = simulate_variant(panel, monthly_returns, spy_features, WINNER)
    metrics = evaluate_simple(eq, spy_aligned)
    metrics["universe"] = name
    metrics["n_picks_universe"] = int(panel["ticker"].nunique())
    eq.to_csv(PIT / f"v3_generalize_{name}_equity.csv", index=False)
    return metrics


def main():
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_features = load_spy_features()

    # SPY benchmark over panel asofs
    panel_dates = pd.DatetimeIndex(sorted(pd.read_parquet(PIT / "sp500_pit_panel.parquet")["asof"].unique()))
    next_month = panel_dates + pd.offsets.MonthEnd(1)
    spy_aligned = pd.DataFrame({
        "date": panel_dates,
        "spy_ret_m": [float(monthly_returns["SPY"].loc[nxt]) if nxt in monthly_returns["SPY"].index else 0.0
                      for nxt in next_month],
    })

    # PIT S&P 500 membership
    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set).to_dict()

    # All tickers ever in v2 panel
    all_tk = set(monthly_returns.columns) - EXCLUDE
    print(f"Universe sizes: SP500_PIT_unique={len(set(members['ticker']))}, "
          f"broader={len(all_tk)}")

    rows = []

    # 1. Full broader universe (1,833 tickers)
    panel_broad = build_panel_for_universe(ticker_filter_set=all_tk)
    print(f"\n[broader 1833] panel rows: {len(panel_broad)}")
    rows.append(run_universe("broader_1833", panel_broad, monthly_returns,
                              spy_features, spy_aligned))

    # 2. PIT non-S&P 500: broader minus SP500 PIT membership at each month
    nonsp_per_month = {d: all_tk - members_g.get(d, set()) for d in panel_dates}
    panel_nonsp = build_panel_for_universe(universe_tickers_per_month=nonsp_per_month)
    print(f"\n[non-SP500 PIT] panel rows: {len(panel_nonsp)}")
    rows.append(run_universe("non_sp500_pit", panel_nonsp, monthly_returns,
                              spy_features, spy_aligned))

    # 3. PIT S&P 500
    panel_sp = build_panel_for_universe(universe_tickers_per_month=members_g)
    print(f"\n[SP500 PIT] panel rows: {len(panel_sp)}")
    rows.append(run_universe("sp500_pit", panel_sp, monthly_returns,
                              spy_features, spy_aligned))

    # 4. Random subset 500 tickers, 5 seeds
    seeded = []
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.default_rng(seed)
        ts = list(all_tk)
        ts.sort()  # determinism
        sample = set(rng.choice(ts, size=500, replace=False))
        panel_rs = build_panel_for_universe(ticker_filter_set=sample)
        m = run_universe(f"random_500_seed{seed}", panel_rs, monthly_returns,
                         spy_features, spy_aligned)
        seeded.append(m)
        rows.append(m)
    seeded_avg = {
        "universe": "random_500_AVG_5seeds",
        "n_picks_universe": int(np.mean([s["n_picks_universe"] for s in seeded])),
        "cagr_full": float(np.mean([s["cagr_full"] for s in seeded])),
        "sharpe": float(np.mean([s["sharpe"] for s in seeded])),
        "max_dd": float(np.mean([s["max_dd"] for s in seeded])),
        "wf_mean_cagr": float(np.mean([s["wf_mean_cagr"] for s in seeded])),
        "wf_min_cagr": float(np.mean([s["wf_min_cagr"] for s in seeded])),
        "wf_max_cagr": float(np.mean([s["wf_max_cagr"] for s in seeded])),
        "wf_mean_edge_pp": float(np.mean([s["wf_mean_edge_pp"] for s in seeded])),
        "wf_n_pos": float(np.mean([s["wf_n_pos"] for s in seeded])),
        "wf_n_beats": float(np.mean([s["wf_n_beats"] for s in seeded])),
        "n_cash": int(np.mean([s["n_cash"] for s in seeded])),
    }
    rows.append(seeded_avg)

    out = pd.DataFrame(rows)
    out.to_csv(PIT / "v3_generalize.csv", index=False)
    print("\n=== Generalization table ===")
    pretty_cols = ["universe", "n_picks_universe", "cagr_full", "sharpe",
                   "max_dd", "wf_mean_cagr", "wf_min_cagr", "wf_max_cagr",
                   "wf_mean_edge_pp", "wf_n_pos", "wf_n_beats"]
    print(out[pretty_cols].round(3).to_string(index=False))

    # Honesty pass: report the percentage of picks that came from S&P 500
    # in the broader universe (to confirm picks aren't ALL S&P 500)
    print("\n=== Pick-distribution sanity (broader universe vs S&P 500) ===")
    eq_broad = pd.read_csv(PIT / "v3_generalize_broader_1833_equity.csv")
    eq_broad["date"] = pd.to_datetime(eq_broad["date"])
    sp_picks_count = 0
    total_picks = 0
    for _, r in eq_broad.iterrows():
        if r.get("regime") == "cash" or pd.isna(r.get("picks")) or not r["picks"]:
            continue
        for tk in str(r["picks"]).split(","):
            total_picks += 1
            if tk in members_g.get(pd.Timestamp(r["date"]), set()):
                sp_picks_count += 1
    print(f"  Of {total_picks} picks in broader-universe run, "
          f"{sp_picks_count} ({sp_picks_count/max(total_picks,1)*100:.1f}%) "
          f"were S&P 500 PIT members at the asof.")


if __name__ == "__main__":
    main()
