"""Full validation of the PIT S&P 500 winner strategy.

Runs the winner config end-to-end:
  - Full equity curve
  - Per-split walk-forward TEST CAGR/Sharpe/MaxDD/edge
  - Year-by-year strategy vs SPY
  - Drawdown ledger
  - Most-picked tickers, turnover, cash months
  - IC stability over time, sub-decade WF
  - Bias overlay (synthetic delisting α=0..20%)

Usage:
  python3 sp500_pit_v3_validate.py [variant_name]

If variant_name is omitted, validates the composite winner from
sp500_pit_extended_winner.json (or sweep_winner.json as fallback).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"
FEATURES_DIR = CACHE / "features"

sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v2"))
from sp500_pit_extended_sweep import (  # noqa: E402
    REGIME_GATES, EXCLUDE, WF_SPLITS, load_spy_features,
    build_panel_with_score, simulate_variant, evaluate, Variant,
)


def parse_variant_name(name: str) -> Variant:
    """Parse a variant name like 'ml_filter|k3_3_3|ew|tight|h6|cap1.0'."""
    parts = name.split("|")
    scorer = parts[0]
    k = parts[1]  # k3_3_3
    if k.startswith("k"):
        knums = k[1:].split("_")
        kN, kR, kB = int(knums[0]), int(knums[1]), int(knums[2])
    else:
        kN = kR = kB = 3
    weighting = parts[2]
    gate = parts[3]
    hold = int(parts[4][1:])
    cap = float(parts[5][3:]) if len(parts) > 5 else 1.0
    return Variant(name=name, scorer=scorer,
                   k_normal=kN, k_recovery=kR, k_bull=kB,
                   weighting=weighting, regime_gate=gate,
                   hold_months=hold, cap_per_pick=cap)


def per_split_eval(eq, spy_aligned):
    rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy()
        if len(e) == 0: continue
        r = e["ret_m"].astype(float)
        ec = (1 + r).cumprod()
        cv = (ec.iloc[-1]) ** (12.0 / len(ec)) - 1
        sh = (r.mean() / max(r.std(), 1e-9)) * np.sqrt(12)
        peak = ec.cummax()
        mdd = float(((ec - peak) / peak).min())
        spy = spy_aligned[(spy_aligned["date"] >= lo) & (spy_aligned["date"] <= hi)]
        sr = spy["spy_ret_m"].astype(float)
        sc = (1 + sr).cumprod()
        scgr = (sc.iloc[-1]) ** (12.0 / len(sc)) - 1
        spy_sh = (sr.mean() / max(sr.std(), 1e-9)) * np.sqrt(12)
        rows.append({
            "split": split, "from": lo.date(), "to": hi.date(), "n_m": len(e),
            "cagr": cv, "spy_cagr": scgr, "edge_pp": (cv - scgr) * 100,
            "sharpe": sh, "spy_sharpe": spy_sh, "max_dd": mdd,
            "n_cash": int((e["regime"] == "cash").sum()),
        })
    return pd.DataFrame(rows)


def yearly_eval(eq, spy_aligned):
    eq = eq.copy()
    eq["year"] = eq["date"].dt.year
    yr = eq.groupby("year")["ret_m"].apply(lambda x: ((1 + x).prod() - 1)).rename("year_ret")
    sa = spy_aligned.copy()
    sa["year"] = sa["date"].dt.year
    syr = sa.groupby("year")["spy_ret_m"].apply(lambda x: ((1 + x).prod() - 1)).rename("spy_year_ret")
    out = yr.to_frame().join(syr.to_frame(), how="left")
    out["edge_pp"] = (out["year_ret"] - out["spy_year_ret"]) * 100
    return out


def drawdown_episodes(eq, threshold=-0.05):
    eq_idx = pd.Series(eq["equity"].values, index=pd.DatetimeIndex(eq["date"]))
    peak = eq_idx.cummax()
    dd = (eq_idx - peak) / peak
    episodes = []
    in_dd = False
    start, depth, trough = None, 0, None
    for d, ddv in dd.items():
        if not in_dd and ddv < threshold:
            in_dd, start, depth, trough = True, d, ddv, d
        elif in_dd:
            if ddv < depth:
                depth, trough = ddv, d
            if ddv >= -0.001:
                episodes.append({"start": start, "trough": trough, "end": d, "depth_pct": depth*100})
                in_dd = False
    if in_dd:
        episodes.append({"start": start, "trough": trough, "end": eq_idx.index[-1], "depth_pct": depth*100})
    return pd.DataFrame(episodes).sort_values("depth_pct")


def turnover_stats(eq):
    last = []
    sames = []
    for _, r in eq.iterrows():
        cur = r.get("picks", "")
        cur_set = set(str(cur).split(",")) if isinstance(cur, str) and cur else set()
        if last and cur_set:
            sames.append(len(set(last) & cur_set) / max(len(cur_set), 1))
        last = list(cur_set)
    sames = np.array(sames)
    return {
        "n_pairs": len(sames),
        "mean_overlap": float(sames.mean()) if len(sames) else 0.0,
        "approx_annl_turnover": float((1 - sames.mean()) * 12) if len(sames) else 0.0,
    }


def most_picked_table(eq, top_n=30):
    counts = {}
    for picks_str in eq["picks"].dropna():
        if not picks_str: continue
        for tk in picks_str.split(","):
            counts[tk] = counts.get(tk, 0) + 1
    return pd.DataFrame(counts.items(), columns=["ticker", "n_months_picked"]).sort_values(
        "n_months_picked", ascending=False).head(top_n)


def bias_overlay(eq, alphas=(0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20),
                 n_iters=30, cost_bps=10.0, seed=42) -> pd.DataFrame:
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    rng_master = np.random.default_rng(seed)
    rows = []

    # Reconstruct per-month per-pick returns from the eq dataframe
    per_month = []
    for _, r in eq.iterrows():
        d = pd.Timestamp(r["date"])
        if r.get("regime", "active") == "cash" or not r.get("picks"):
            per_month.append({"date": d, "active": False, "rets": [], "weights": []})
            continue
        picks = str(r["picks"]).split(",")
        # Find next month-end and pull returns
        pos = monthly_returns.index.searchsorted(d)
        cands = []
        for j in (pos - 1, pos):
            if 0 <= j < len(monthly_returns.index):
                cands.append((j, abs((monthly_returns.index[j] - d).days)))
        cands.sort(key=lambda x: x[1])
        if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(monthly_returns.index):
            per_month.append({"date": d, "active": False, "rets": [], "weights": []})
            continue
        next_d = monthly_returns.index[cands[0][0] + 1]
        rets = []
        for tk in picks:
            if tk in monthly_returns.columns:
                rr = monthly_returns.at[next_d, tk]
                rets.append(-1.0 if pd.isna(rr) else float(rr))
            else:
                rets.append(-1.0)
        per_month.append({"date": d, "active": True, "rets": rets,
                          "weights": [1/len(picks)] * len(picks)})  # EW assumed

    n_months = len(per_month)
    cf = cost_bps / 10000.0

    for alpha in alphas:
        p_month = 1 - (1 - alpha) ** (1/12) if alpha > 0 else 0
        finals = []
        for it in range(n_iters):
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            equity = 1.0
            for m in per_month:
                if not m["active"] or not m["rets"]:
                    continue
                rets = np.array(m["rets"], dtype=float)
                weights = np.array(m["weights"], dtype=float)
                if p_month > 0:
                    wipe = rng.random(len(rets)) < p_month
                    rets[wipe] = -1.0
                ret_m = float((rets * weights).sum())
                equity *= (1 + ret_m) * (1 - cf)
            finals.append(equity)
        finals = np.array(finals)
        years = n_months / 12.0
        cagrs = finals ** (1/years) - 1
        rows.append({
            "alpha_yr": alpha,
            "p10": float(np.percentile(cagrs, 10) * 100),
            "median": float(np.median(cagrs) * 100),
            "p90": float(np.percentile(cagrs, 90) * 100),
            "mean": float(np.mean(cagrs) * 100),
            "n_iters": n_iters,
        })
    return pd.DataFrame(rows)


def sub_period_wf(eq, spy_aligned):
    """Sub-decade walk-forward: 2003-2012, 2008-2017, 2013-2022, 2018-2025."""
    rows = []
    for label, lo, hi in [
        ("p1_03_12", "2003-09-30", "2012-12-31"),
        ("p2_08_17", "2008-01-01", "2017-12-31"),
        ("p3_13_22", "2013-01-01", "2022-12-31"),
        ("p4_18_25", "2018-01-01", "2025-12-31"),
        ("modern_10_25", "2010-01-01", "2025-12-31"),
    ]:
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
        rows.append({"period": label, "from": lo.date(), "to": hi.date(),
                     "cagr": cv, "spy_cagr": scgr, "edge_pp": (cv - scgr) * 100,
                     "n_m": len(e)})
    return pd.DataFrame(rows)


def main():
    if len(sys.argv) > 1:
        variant_name = sys.argv[1]
    else:
        # Pick from sweep winner json
        winner_file = (PIT / "sp500_pit_extended_winner.json")
        if not winner_file.exists():
            winner_file = PIT / "sp500_pit_sweep_winner.json"
        if winner_file.exists():
            variant_name = json.loads(winner_file.read_text())["name"]
        else:
            variant_name = "ml_filter|k3_3_3|ew|tight|h6|cap1.0"

    print(f"=== Validating: {variant_name} ===")
    v = parse_variant_name(variant_name)

    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_features = load_spy_features()

    panel = build_panel_with_score(v.scorer)
    eq = simulate_variant(panel, monthly_returns, spy_features, v)
    eq.to_csv(PIT / f"v3_{v.scorer}_{v.k_normal}{v.k_recovery}{v.k_bull}_{v.weighting}_{v.regime_gate}_h{v.hold_months}_equity.csv", index=False)

    # SPY benchmark over the panel window
    full_dates = pd.DatetimeIndex(eq["date"])
    next_month = full_dates + pd.offsets.MonthEnd(1)
    spy_aligned = pd.DataFrame({
        "date": full_dates,
        "spy_ret_m": [float(monthly_returns["SPY"].loc[nxt]) if nxt in monthly_returns["SPY"].index else 0.0
                      for nxt in next_month],
    })

    # Headline
    metrics = evaluate(eq, spy_aligned, variant_name)
    print(f"\n[headline]")
    for k, v_ in metrics.items():
        print(f"  {k}: {v_}")

    # Per-split
    splits = per_split_eval(eq, spy_aligned)
    print("\n[per-split walk-forward]")
    print(splits.round(3).to_string(index=False))
    splits.to_csv(PIT / "v3_winner_walkforward.csv", index=False)

    # Year-by-year
    yr = yearly_eval(eq, spy_aligned)
    yr["Strategy_pct"] = (yr["year_ret"]*100).round(1)
    yr["SPY_pct"] = (yr["spy_year_ret"]*100).round(1)
    yr["edge_pp_r"] = yr["edge_pp"].round(1)
    print("\n[year-by-year]")
    print(yr[["Strategy_pct", "SPY_pct", "edge_pp_r"]].to_string())
    yr.to_csv(PIT / "v3_winner_yearly.csv")
    n_pos_y = int((yr["year_ret"] > 0).sum())
    n_beat_y = int((yr["year_ret"] > yr["spy_year_ret"]).sum())
    cagr_full = ((yr["year_ret"]+1).prod()) ** (1/len(yr)) - 1
    cagr_ex_max = ((yr.drop(yr["year_ret"].idxmax())["year_ret"]+1).prod()) ** (1/(len(yr)-1)) - 1
    cagr_ex_min = ((yr.drop(yr["year_ret"].idxmin())["year_ret"]+1).prod()) ** (1/(len(yr)-1)) - 1
    print(f"  + years: {n_pos_y}/{len(yr)}, beats SPY: {n_beat_y}/{len(yr)}")
    print(f"  full cy CAGR: {cagr_full*100:.2f}%")
    print(f"  ex-best year CAGR: {cagr_ex_max*100:.2f}%")
    print(f"  ex-worst year CAGR: {cagr_ex_min*100:.2f}%")

    # Sub-period WF
    sp = sub_period_wf(eq, spy_aligned)
    print("\n[sub-period WF]")
    print(sp.round(3).to_string(index=False))
    sp.to_csv(PIT / "v3_winner_sub_periods.csv", index=False)

    # Drawdowns
    dd = drawdown_episodes(eq).head(10)
    print("\n[top drawdowns]")
    print(dd.round(2).to_string(index=False))
    dd.to_csv(PIT / "v3_winner_drawdowns.csv", index=False)

    # Most picked
    mp = most_picked_table(eq, top_n=30)
    print("\n[most picked tickers]")
    print(mp.head(20).to_string(index=False))
    mp.to_csv(PIT / "v3_winner_most_picked.csv", index=False)

    # Turnover
    to = turnover_stats(eq)
    print(f"\n[turnover] {to}")

    # Bias overlay
    print("\n=== Bias overlay (synthetic delisting MC) ===")
    bo = bias_overlay(eq)
    print(bo.round(2).to_string(index=False))
    bo.to_csv(PIT / "v3_winner_bias_sensitivity.csv", index=False)

    # Save summary JSON
    summary = {
        "variant_name": variant_name,
        "n_months": int(len(eq)),
        "final_equity": float(eq["equity"].iloc[-1]),
        "cagr_full": metrics["cagr_full"],
        "spy_cagr_full": metrics["spy_cagr_full"],
        "edge_full_pp": metrics["edge_full_pp"],
        "sharpe": metrics["sharpe"],
        "max_dd": metrics["max_dd"],
        "n_cash_months": metrics["n_cash"],
        "wf_mean_cagr": metrics["wf_mean_cagr"],
        "wf_median_cagr": metrics["wf_median_cagr"],
        "wf_min_cagr": metrics["wf_min_cagr"],
        "wf_max_cagr": metrics["wf_max_cagr"],
        "wf_mean_edge_pp": metrics["wf_mean_edge_pp"],
        "wf_n_positive": metrics["wf_n_pos"],
        "wf_n_beats_spy": metrics["wf_n_beats"],
        "n_positive_years": n_pos_y,
        "n_beats_spy_years": n_beat_y,
        "cy_cagr_full": float(cagr_full),
        "cy_cagr_ex_best": float(cagr_ex_max),
        "cy_cagr_ex_worst": float(cagr_ex_min),
        "turnover": to,
    }
    (PIT / f"v3_winner_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved -> v3_winner_summary.json")


if __name__ == "__main__":
    main()
