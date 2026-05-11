"""Save the supplementary analysis artifacts for the PIT NDX backtest.

Saves into experiments/monthly_dca/v5/qqq_pit/:
  ml_preds_v2_ndx.parquet      — GBM 3m+6m predictions restricted to PIT NDX tickers
  ml_preds_chronos_ndx.parquet — Chronos p70 predictions restricted to PIT NDX tickers
  qqq_monthly_prices.csv       — QQQ ETF month-end closes used as the benchmark
  spy_monthly_features.csv     — SPY-derived monthly features used for the crash gate
  ndx_rebalances.csv           — per-rebalance audit log:
       asof, regime, eligible_pool, chronos_filtered_pool, picks, weights,
       chronos_p70_ranks, ml_scores
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
QQQ_DIR = ROOT / "experiments" / "monthly_dca" / "v5" / "qqq_pit"
CACHE_V2 = ROOT / "experiments" / "monthly_dca" / "cache" / "v2"
SP500_PIT = CACHE_V2 / "sp500_pit"

sys.path.insert(0, str(QQQ_DIR))
from run_v5_on_qqq_pit import (
    fetch_qqq_prices, fetch_spy_prices, build_spy_features,
    calc_invvol_weights, regime_tight,
    CHRONOS_FILTER_Q, CAP_PER_PICK, HOLD_MONTHS, K_PICKS,
)


def main():
    print("=" * 60)
    print("Save supplementary PIT NDX analysis artifacts")
    print("=" * 60)

    # 1. Load PIT membership
    mem = pd.read_parquet(QQQ_DIR / "ndx_pit_membership_monthly.parquet")
    ndx_tickers_all = set(mem["ticker"].unique())
    print(f"[1] NDX universe: {len(ndx_tickers_all)} unique tickers, "
          f"{len(mem)} (asof, ticker) rows")

    # 2. ML preds restricted to NDX
    print("[2] Filtering ML preds to NDX...")
    ml = pd.read_parquet(CACHE_V2 / "ml_preds_v2.parquet")
    ml["asof"] = pd.to_datetime(ml["asof"])
    ml_ndx = ml[ml["ticker"].isin(ndx_tickers_all)].copy()
    ml_ndx.to_parquet(QQQ_DIR / "ml_preds_v2_ndx.parquet", index=False)
    print(f"    saved ml_preds_v2_ndx.parquet  shape={ml_ndx.shape}")

    # 3. Chronos preds restricted to NDX
    print("[3] Filtering Chronos preds to NDX...")
    ch = pd.read_parquet(SP500_PIT / "ml_preds_chronos_broader.parquet")
    ch["asof"] = pd.to_datetime(ch["asof"])
    ch_ndx = ch[ch["ticker"].isin(ndx_tickers_all)].copy()
    ch_ndx.to_parquet(QQQ_DIR / "ml_preds_chronos_ndx.parquet", index=False)
    print(f"    saved ml_preds_chronos_ndx.parquet  shape={ch_ndx.shape}")

    # 4. QQQ benchmark monthly closes
    print("[4] Fetching QQQ benchmark monthly closes...")
    qqq_m = fetch_qqq_prices()
    qqq_df = pd.DataFrame({"asof": qqq_m.index, "qqq_close": qqq_m.values})
    qqq_df.to_csv(QQQ_DIR / "qqq_monthly_prices.csv", index=False)
    print(f"    saved qqq_monthly_prices.csv  rows={len(qqq_df)}")

    # 5. SPY monthly features used by the crash gate
    print("[5] Building SPY monthly features used by the crash gate...")
    spy_daily = fetch_spy_prices()
    spy_feat_idx = build_spy_features(spy_daily).copy()
    spy_feat_idx.index = pd.to_datetime(spy_feat_idx.index)
    spy_feat_idx.index.name = "asof"
    spy_feat = spy_feat_idx.reset_index()
    spy_feat.to_csv(QQQ_DIR / "spy_monthly_features.csv", index=False)
    print(f"    saved spy_monthly_features.csv  rows={len(spy_feat)}")

    # 6. Per-rebalance audit log
    print("[6] Building per-rebalance audit log...")
    mret = pd.read_parquet(QQQ_DIR / "ndx_monthly_returns.parquet")
    if not isinstance(mret.index, pd.DatetimeIndex):
        mret.index = pd.to_datetime(mret.index)

    members_g = {asof: set(g["ticker"].tolist())
                 for asof, g in mem.groupby("asof")}
    chronos_at = {asof: dict(zip(g["ticker"], g["chronos_p70_3m"]))
                   for asof, g in ch_ndx.groupby("asof")}

    months = sorted(m for m in members_g.keys()
                    if m >= pd.Timestamp("2015-01-31")
                    and m <= mret.index.max())

    rows = []
    held = 0
    last_picks: list[str] = []
    for i, m in enumerate(months):
        spy_now = spy_feat_idx.loc[m].to_dict() if m in spy_feat_idx.index else {}
        regime = regime_tight(spy_now)
        do_reb = (i == 0) or (held >= HOLD_MONTHS)

        sub_ml = ml_ndx[ml_ndx["asof"] == m]
        sub_ml = sub_ml[sub_ml["ticker"].isin(members_g[m])].copy()
        eligible_pool = sorted(set(sub_ml["ticker"]))

        # Apply Chronos filter
        chronos_rk = {}
        chronos_filtered: list[str] = []
        if m in chronos_at and len(sub_ml) > 0:
            cr = chronos_at[m]
            sub_ml["chr"] = sub_ml["ticker"].map(cr)
            sub_ml["chr_rk"] = sub_ml["chr"].rank(pct=True)
            chronos_rk = dict(zip(sub_ml["ticker"], sub_ml["chr_rk"]))
            sub_ml = sub_ml[sub_ml["chr_rk"] >= CHRONOS_FILTER_Q]
            chronos_filtered = sorted(set(sub_ml["ticker"]))

        # Pick top-K and weights — but only audit on rebalance months
        picks: list[str] = []
        weights: list[float] = []
        if do_reb and regime != "crash" and len(sub_ml) >= K_PICKS:
            sub_ml["score"] = (sub_ml["pred_3m"] + sub_ml["pred_6m"]) / 2
            top = sub_ml.sort_values("score", ascending=False).head(K_PICKS)
            picks = top["ticker"].tolist()
            w = calc_invvol_weights(picks, mret, m, cap=CAP_PER_PICK)
            weights = list(w)
            last_picks = picks
            held = 0
        else:
            picks = last_picks
            held += 1

        # Per-pick details for picks made on this asof
        ml_scores = {t: float((sub_ml[sub_ml["ticker"] == t]["pred_3m"].iloc[0] +
                                sub_ml[sub_ml["ticker"] == t]["pred_6m"].iloc[0]) / 2)
                      for t in picks
                      if t in sub_ml["ticker"].values}

        rows.append({
            "asof": str(m.date()),
            "regime": regime,
            "rebalance": do_reb,
            "n_eligible": len(eligible_pool),
            "n_chronos_filtered": len(chronos_filtered),
            "picks": ",".join(picks),
            "weights": ",".join(f"{w:.4f}" for w in weights),
            "ml_scores": ",".join(f"{ml_scores.get(t, float('nan')):.4f}"
                                    for t in picks),
            "chronos_p70_ranks": ",".join(
                f"{chronos_rk.get(t, float('nan')):.4f}" for t in picks),
            "spy_ret_21d": spy_now.get("spy_ret_21d"),
            "spy_mom_6_1": spy_now.get("spy_mom_6_1"),
            "spy_mom_12_1": spy_now.get("spy_mom_12_1"),
            "spy_dsma200": spy_now.get("spy_dsma200"),
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(QQQ_DIR / "ndx_rebalances.csv", index=False)
    print(f"    saved ndx_rebalances.csv  rows={len(audit)} "
          f"(reb={int(audit['rebalance'].sum())}, "
          f"mid-cycle={int((~audit['rebalance']).sum())})")

    # 7. Year-by-year summary on PIT NDX
    print("[7] Building year-by-year summary on PIT NDX...")
    eq = pd.read_csv(QQQ_DIR / "qqq_pit_equity.csv")
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date").sort_index()
    yrs = []
    for year, g in eq.groupby(eq.index.year):
        strat_ret = (1 + g["ret_m"]).prod() - 1
        # QQQ return for the year (start vs end month closes)
        q_start = g["qqq_eq"].iloc[0]
        q_end = g["qqq_eq"].iloc[-1]
        qqq_ret = (q_end / q_start) - 1
        yrs.append({
            "year": int(year),
            "n_months": len(g),
            "strat_ret": float(strat_ret),
            "qqq_ret": float(qqq_ret),
            "edge_pp": float((strat_ret - qqq_ret) * 100),
        })
    yb = pd.DataFrame(yrs)
    yb.to_csv(QQQ_DIR / "qqq_pit_year_by_year.csv", index=False)
    print(f"    saved qqq_pit_year_by_year.csv  rows={len(yb)}")
    print()
    print(yb.to_string(index=False))


if __name__ == "__main__":
    main()
