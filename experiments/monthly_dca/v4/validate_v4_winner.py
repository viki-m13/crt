"""Full validation of the v4 winner.

Inputs the winner config (variant); produces:
  - Full equity curve
  - Per-split walk-forward CAGR/Sharpe/MaxDD/edge
  - Year-by-year strategy vs SPY
  - Drawdown ledger
  - Most-picked tickers, turnover, cash months
  - Sub-period CAGR
  - Bias overlay (synthetic delisting α=0..20%)
  - Generalisation: same strategy on broader 1833-ticker universe + non-S&P 500
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from simulator_v4 import (
    Variant, simulate_variant_v4, evaluate, build_panel_with_score,
    load_spy_features, build_spy_aligned, _load_daily_prices,
    PIT, V2, CACHE, EXCLUDE, WF_SPLITS, REGIME_GATES,
)


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
        "n_pairs": int(len(sames)),
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


def sub_period_eval(eq, spy_aligned, periods):
    rows = []
    for label, lo, hi in periods:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)]
        if len(e) == 0: continue
        r = e["ret_m"].astype(float)
        ec = (1 + r).cumprod()
        cv = (ec.iloc[-1]) ** (12.0 / len(ec)) - 1 if len(ec) else 0
        spy = spy_aligned[(spy_aligned["date"] >= lo) & (spy_aligned["date"] <= hi)]
        sr = spy["spy_ret_m"].astype(float)
        sc = (1 + sr).cumprod()
        scgr = (sc.iloc[-1]) ** (12.0 / len(sc)) - 1 if len(sc) else 0
        rows.append({"period": label, "from": lo.date(), "to": hi.date(),
                     "n_m": len(e), "cagr": cv, "spy_cagr": scgr,
                     "edge_pp": (cv - scgr) * 100})
    return pd.DataFrame(rows)


def bias_overlay(panel, monthly_returns, spy_features, v: Variant,
                 alphas=(0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20),
                 n_iters=20, seed=42, daily_prices=None):
    rng = np.random.default_rng(seed)
    rows = []
    base_eq = simulate_variant_v4(panel, monthly_returns, spy_features, v, daily_prices=daily_prices)
    full_dates = pd.DatetimeIndex(base_eq["date"])
    spy_aligned = build_spy_aligned(panel)

    for alpha in alphas:
        if alpha == 0:
            cgr = (1 + base_eq["ret_m"]).cumprod().iloc[-1] ** (12.0/len(base_eq)) - 1
            spy_cgr = (1 + spy_aligned["spy_ret_m"]).cumprod().iloc[-1] ** (12.0/len(spy_aligned)) - 1
            rows.append({"alpha": alpha, "cagr_p10": cgr, "cagr_median": cgr, "cagr_p90": cgr,
                         "edge_median_pp": (cgr - spy_cgr) * 100})
            continue
        # Per-month delisting probability
        p_month = 1 - (1 - alpha) ** (1/12)
        cgrs = []
        for it in range(n_iters):
            eq_it = base_eq.copy()
            for i, row in eq_it.iterrows():
                if row.get("regime") == "cash" or not row.get("picks"):
                    continue
                picks = row["picks"].split(",")
                if not picks:
                    continue
                weights = np.ones(len(picks)) / len(picks)
                # Each pick has a probability p_month of being wiped this month
                wiped = rng.random(len(picks)) < p_month
                if wiped.any():
                    pick_rets = np.full(len(picks), float(row["ret_m"]))
                    pick_rets[wiped] = -1.0
                    new_ret = float((pick_rets * weights).sum())
                    eq_it.at[i, "ret_m"] = new_ret
            ec = (1 + eq_it["ret_m"]).cumprod()
            cgrs.append(ec.iloc[-1] ** (12.0/len(ec)) - 1)
        cgrs = np.array(cgrs)
        spy_cgr = (1 + spy_aligned["spy_ret_m"]).cumprod().iloc[-1] ** (12.0/len(spy_aligned)) - 1
        rows.append({"alpha": alpha, "cagr_p10": float(np.percentile(cgrs, 10)),
                     "cagr_median": float(np.median(cgrs)),
                     "cagr_p90": float(np.percentile(cgrs, 90)),
                     "edge_median_pp": float((np.median(cgrs) - spy_cgr) * 100)})
        print(f"  α={alpha*100:.1f}% → median CAGR {np.median(cgrs)*100:.2f}%  edge {(np.median(cgrs)-spy_cgr)*100:+.2f}pp",
              flush=True)
    return pd.DataFrame(rows)


def generalize_universes(scorer: str, v: Variant, monthly_returns, spy_features, daily_prices,
                          random_seeds=(1, 2, 3)):
    """Apply the same v4 winner config on alternative universes:
      - Broader 1833-ticker universe (no PIT filter)
      - Non-S&P 500 PIT (universe = all 1833 minus PIT S&P 500 members at each asof)
      - Random 500 subsets (3 seeds)

    Returns DataFrame summarising CAGR/sharpe/etc per universe.
    """
    # Build feature panel for each universe
    print(f"  building broader universes...", flush=True)

    # Borrow the broader-panel construction from v4 training
    feature_files = {pd.Timestamp(p.stem): p for p in (CACHE / "features").glob("*.parquet")}
    # canonical 67 feature cols
    ref_d = pd.Timestamp("2010-12-31")
    if ref_d not in feature_files: ref_d = sorted(feature_files.keys())[len(feature_files) // 2]
    feature_cols = list(pd.read_parquet(feature_files[ref_d]).columns)

    asofs = sorted(feature_files.keys())
    chunks_all = []  # broader 1833 panel (rank features within each asof)
    for d in asofs:
        feat = pd.read_parquet(feature_files[d])
        feat = feat[~feat.index.isin(EXCLUDE)]
        if not set(feature_cols).issubset(feat.columns):
            continue
        feat = feat[feature_cols]
        if len(feat) < 100:
            continue
        for c in feature_cols:
            r = feat[c].rank(pct=True)
            feat[c + "_xs"] = (r - 0.5) * 2
        feat = feat.reset_index().rename(columns={"index": "ticker"})
        feat["asof"] = d
        chunks_all.append(feat)
    broader = pd.concat(chunks_all, ignore_index=True)
    broader = broader[["asof", "ticker"] + [c + "_xs" for c in feature_cols] + feature_cols]

    # Attach the same scorer's predictions
    if scorer == "ml_3plus6":
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        broader = broader.merge(ml, on=["asof", "ticker"], how="left")
        broader["score"] = (broader["pred_3m"] + broader["pred_6m"]) / 2
    elif scorer == "stack_v2_v4":
        ml2 = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
        ml4 = pd.read_parquet(PIT / "ml_preds_v4.parquet")[["asof", "ticker", "pred_v4"]]
        ml2["asof"] = pd.to_datetime(ml2["asof"])
        ml4["asof"] = pd.to_datetime(ml4["asof"])
        broader = broader.merge(ml2, on=["asof", "ticker"], how="left")
        broader = broader.merge(ml4, on=["asof", "ticker"], how="left")
        v2 = (broader["pred_3m"] + broader["pred_6m"]) / 2
        broader["v2_rk"] = v2.groupby(broader["asof"]).rank(pct=True)
        broader["v4_rk"] = broader["pred_v4"].groupby(broader["asof"]).rank(pct=True)
        broader["score"] = 0.5 * broader["v2_rk"] + 0.5 * broader["v4_rk"]
    elif scorer == "v4_only":
        ml = pd.read_parquet(PIT / "ml_preds_v4.parquet")[["asof", "ticker", "pred_v4"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        broader = broader.merge(ml.rename(columns={"pred_v4": "score"}), on=["asof", "ticker"], how="left")
    else:
        raise ValueError(scorer)

    # PIT membership
    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    pit_set = members.groupby("asof")["ticker"].apply(set)

    panels = {
        "broader_1833": broader,
        "non_sp500_pit": broader[broader.apply(lambda r: r["ticker"] not in pit_set.get(r["asof"], set()), axis=1)],
    }

    # PIT
    pit_panel = build_panel_with_score(scorer)
    panels["sp500_pit"] = pit_panel

    # Random 500 seeds
    asofs_b = sorted(broader["asof"].unique())
    for s in random_seeds:
        rng = np.random.default_rng(s)
        # For each asof, pick 500 random tickers from those eligible at that asof
        chunks = []
        for d in asofs_b:
            sub = broader[broader["asof"] == d]
            if len(sub) <= 500:
                chunks.append(sub); continue
            tks = rng.choice(sub["ticker"].values, 500, replace=False)
            chunks.append(sub[sub["ticker"].isin(tks)])
        panels[f"random_500_seed{s}"] = pd.concat(chunks, ignore_index=True)

    # Simulate v4 winner on each
    rows = []
    for label, p in panels.items():
        spy_aligned = build_spy_aligned(p)
        eq = simulate_variant_v4(p, monthly_returns, spy_features, v, daily_prices=daily_prices)
        m = evaluate(eq, spy_aligned, label)
        rows.append({"universe": label, "n_unique": p["ticker"].nunique(), **m})
        print(f"  {label:20s}  CAGR={m['cagr_full']*100:6.2f}%  WF_mean={m['wf_mean_cagr']*100:6.2f}%  WF_min={m['wf_min_cagr']*100:6.2f}%  beats={m['wf_n_beats']}/{m['wf_n_splits']}",
              flush=True)
        eq.to_csv(PIT / f"v4_generalize_{label}_equity.csv", index=False)
    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 2:
        print("Usage: validate_v4_winner.py <variant_spec_json>")
        print('  e.g. \'{"name":"v4_winner","scorer":"ml_3plus6","k_normal":3,"k_recovery":3,"k_bull":3,"weighting":"ew","regime_gate":"tight","hold_months":6,"stop_loss_pct":0.0,"take_profit_pct":0.5,"score_threshold_pct":0.0,"cap_per_pick":1.0,"cost_bps":10.0}\'')
        sys.exit(1)
    spec = json.loads(sys.argv[1])
    v = Variant(**spec)
    print(f"=== Validating v4 winner: {v.name} ===")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_features = load_spy_features()
    daily_prices = _load_daily_prices()

    panel = build_panel_with_score(v.scorer)
    spy_aligned = build_spy_aligned(panel)

    print(f"  panel: {panel.shape}, asof {panel['asof'].min().date()}..{panel['asof'].max().date()}")

    eq = simulate_variant_v4(panel, monthly_returns, spy_features, v, daily_prices=daily_prices)
    eq.to_csv(PIT / f"v4_winner_equity.csv", index=False)
    main_metrics = evaluate(eq, spy_aligned, v.name)
    print(f"  full CAGR: {main_metrics['cagr_full']*100:.2f}%  WF mean: {main_metrics['wf_mean_cagr']*100:.2f}%  WF min: {main_metrics['wf_min_cagr']*100:.2f}%")

    print("\n--- 1. Per-split walk-forward ---")
    wf = per_split_eval(eq, spy_aligned)
    wf.to_csv(PIT / f"v4_winner_walkforward.csv", index=False)
    print(wf.to_string(index=False))

    print("\n--- 2. Yearly ---")
    yr = yearly_eval(eq, spy_aligned)
    yr.to_csv(PIT / f"v4_winner_yearly.csv")
    print(yr.to_string())

    print("\n--- 3. Drawdowns ---")
    dd = drawdown_episodes(eq)
    dd.to_csv(PIT / f"v4_winner_drawdowns.csv", index=False)
    print(dd.head(8).to_string(index=False))

    print("\n--- 4. Turnover & most-picked ---")
    turn = turnover_stats(eq)
    print(turn)
    mp = most_picked_table(eq, top_n=30)
    mp.to_csv(PIT / f"v4_winner_most_picked.csv", index=False)
    print(mp.head(15).to_string(index=False))

    print("\n--- 5. Sub-period ---")
    periods = [
        ("2003-09 to 2012-12", "2003-09-30", "2012-12-31"),
        ("2008-01 to 2017-12", "2008-01-31", "2017-12-31"),
        ("2013-01 to 2022-12", "2013-01-31", "2022-12-31"),
        ("2018-01 to 2025-12", "2018-01-31", "2025-12-31"),
        ("Modern 2010-2025",   "2010-01-31", "2025-12-31"),
    ]
    sp = sub_period_eval(eq, spy_aligned, periods)
    sp.to_csv(PIT / f"v4_winner_sub_periods.csv", index=False)
    print(sp.to_string(index=False))

    print("\n--- 6. Bias overlay ---")
    bias = bias_overlay(panel, monthly_returns, spy_features, v, daily_prices=daily_prices, n_iters=20)
    bias.to_csv(PIT / f"v4_winner_bias_sensitivity.csv", index=False)
    print(bias.to_string(index=False))

    print("\n--- 7. Generalisation ---")
    gen = generalize_universes(v.scorer, v, monthly_returns, spy_features, daily_prices)
    gen.to_csv(PIT / f"v4_generalize.csv", index=False)
    print(gen.to_string(index=False))

    summary = {
        "spec": spec,
        "headline": {
            "cagr_full": main_metrics["cagr_full"],
            "spy_cagr_full": main_metrics["spy_cagr_full"],
            "edge_full_pp": main_metrics["edge_full_pp"],
            "wf_mean_cagr": main_metrics["wf_mean_cagr"],
            "wf_median_cagr": main_metrics["wf_median_cagr"],
            "wf_min_cagr": main_metrics["wf_min_cagr"],
            "wf_max_cagr": main_metrics["wf_max_cagr"],
            "wf_mean_edge_pp": main_metrics["wf_mean_edge_pp"],
            "wf_n_pos": main_metrics["wf_n_pos"],
            "wf_n_beats": main_metrics["wf_n_beats"],
            "sharpe": main_metrics["sharpe"],
            "max_dd": main_metrics["max_dd"],
            "n_cash": main_metrics["n_cash"],
            "turnover": turn,
        },
    }
    with open(PIT / f"v4_winner_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to {PIT}/v4_winner_summary.json")


if __name__ == "__main__":
    main()
