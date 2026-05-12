"""Phase 10: rebalance-schedule sweep — fixed periods vs rule-based.

Tests several rebalance schedules on the augmented PIT panel with the
deployed K=2 v5 picker (Chronos q=0.45, cap=0.40, tight regime gate).

The motivation: the deployed h=6 schedule was the unlucky-year cost
in 2024 (-25pp edge). Earlier work showed staggered DCA mitigates this
at a -3pp cost to overall WF mean. This script tests two alternatives:

  A. FIXED quarterly (h=3)  — re-pick every 3 months instead of 6.
                              4 entries/year, less timing-date luck per
                              entry. Earlier broad sweep showed h=3 has
                              28.4% WF mean (vs h=6's 49.4%) — likely
                              too short for the 3m+6m GBM horizon.

  B. FIXED quarterly-mid (h=4) — between h=3 and h=6.

  C. RULE-BASED variants:
     C1: min_hold=6m, rebalance EARLIER if regime changes (already in
         deployed code) OR if top-2 picks aren't both in the new top-K
         eligible pool.
     C2: min_hold=3m, rebalance MORE OFTEN if either pick has dropped
         out of the top-decile by GBM score.
     C3: min_hold=6m, rebalance EARLIER if basket drawdown > X%
         (a stop-loss / "regime drift" trigger).
     C4: min_hold=3m, MAX_hold=9m, rebalance only on:
         - score-decile drift (a current pick fell below P50)
         - regime change
         - OR max_hold reached

  For each: full WF metrics, 2024 detail, Max DD.

Output:
  augmented/rebalance_rule_sweep.csv  — headline metrics per variant
  augmented/rebalance_rule_sweep_yearly.csv — year-by-year per variant
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_v5_aug import (  # noqa: E402
    AUG, PIT, EXCLUDE, COST_BPS, WF_SPLITS,
    classify_regime_tight, load_spy_features, calc_invvol_weights,
)

K = 2
CHR_Q = 0.45
CAP = 0.40


def run_v5_with_rule(rule, ml, chr_, spy, mr, members_g, months,
                    panel_by_asof, ml_by_asof, chr_by_asof,
                    k=K, chr_q=CHR_Q, cap=CAP) -> tuple[pd.DataFrame, list, dict]:
    """
    `rule` is a dict that decides when to rebalance:
      mode:          'fixed' | 'rule_based'
      hold:          int (months) for fixed mode  OR  min_hold for rule_based
      max_hold:      optional max-hold for rule_based (default infinity)
      triggers:      list of strings, any of:
        'score_drift'   — pick falls out of top-K eligible-pool
        'rank_drift'    — pick's GBM rank drops below P50 in current pool
        'drawdown'      — basket drawdown from peak > rule['dd_threshold']
        'regime_change' — regime transitions (always on for tight gate)
    """
    cf = COST_BPS / 1e4
    mode = rule.get("mode", "fixed")
    hold = rule.get("hold", 6)
    min_hold = rule.get("min_hold", hold)
    max_hold = rule.get("max_hold", 99)
    triggers = set(rule.get("triggers", []))
    dd_threshold = rule.get("dd_threshold", -0.15)

    cur_picks = []; cur_weights = np.array([])
    cur_basket_entry_equity = 1.0
    cur_basket_peak_equity = 1.0
    cash = False; held_for = 0; equity = 1.0
    rows = []; picks_log = []
    last_regime = None
    n_early_reb = 0

    for i, m in enumerate(months):
        regime = classify_regime_tight(spy.loc[m].to_dict() if m in spy.index else {})

        # Realize this month's return from the existing basket
        ret_m = 0.0
        if not cash and cur_picks:
            mr_pos = mr.index.searchsorted(m)
            if mr_pos + 1 < len(mr.index):
                next_d = mr.index[mr_pos + 1]
                pick_rets = [0.0 if pd.isna(mr.at[next_d, tk]) else float(mr.at[next_d, tk])
                              for tk in cur_picks if tk in mr.columns]
                if len(pick_rets) == len(cur_weights):
                    ret_m = float((np.array(pick_rets) * cur_weights).sum())
                    equity *= (1 + ret_m)
                    cur_basket_peak_equity = max(cur_basket_peak_equity, equity)

        # Decide rebalance
        do_reb = (i == 0) or (cash != (regime == "crash"))

        if mode == "fixed":
            do_reb = do_reb or (held_for >= hold)
        else:  # rule_based
            # Always honour min_hold and max_hold
            if held_for >= max_hold:
                do_reb = True
            elif held_for >= min_hold:
                # Check triggers
                if "regime_change" in triggers and last_regime is not None and regime != last_regime:
                    do_reb = True; n_early_reb += 1
                elif "score_drift" in triggers:
                    # Are current picks still in the top-K of the new eligible pool?
                    if cur_picks:
                        sub_ml = ml_by_asof.get(m)
                        sub_chr = chr_by_asof.get(m)
                        if sub_ml is not None:
                            sp_set = members_g.get(m, set())
                            sub = sub_ml[sub_ml["ticker"].isin(sp_set)]
                            sub = sub[~sub["ticker"].isin(EXCLUDE)]
                            sub = sub.dropna(subset=["ml_score"])
                            if chr_q > 0 and sub_chr is not None and not sub_chr.empty:
                                sub = sub.merge(sub_chr[["ticker", "chronos_p70_3m"]],
                                                on="ticker", how="left")
                                sub = sub.dropna(subset=["chronos_p70_3m"])
                                sub["chr_p70_rk"] = sub["chronos_p70_3m"].rank(pct=True)
                                sub = sub[sub["chr_p70_rk"] >= chr_q]
                            sub = sub.sort_values("ml_score", ascending=False)
                            new_top_k = set(sub.head(k)["ticker"])
                            # If neither of our picks is in the new top-K, rebalance
                            if not (set(cur_picks) & new_top_k):
                                do_reb = True; n_early_reb += 1
                elif "rank_drift" in triggers:
                    # If any current pick has GBM rank below P50, rebalance
                    if cur_picks:
                        sub_ml = ml_by_asof.get(m)
                        if sub_ml is not None:
                            sp_set = members_g.get(m, set())
                            sub = sub_ml[sub_ml["ticker"].isin(sp_set)]
                            sub = sub.dropna(subset=["ml_score"])
                            sub["score_rank"] = sub["ml_score"].rank(pct=True)
                            rank_lookup = dict(zip(sub["ticker"], sub["score_rank"]))
                            min_rank = min(rank_lookup.get(tk, 0) for tk in cur_picks)
                            if min_rank < 0.5:
                                do_reb = True; n_early_reb += 1
                elif "drawdown" in triggers:
                    cur_dd = (equity / cur_basket_peak_equity - 1.0)
                    if cur_dd <= dd_threshold:
                        do_reb = True; n_early_reb += 1

        if do_reb:
            equity *= (1 - cf)
            if regime == "crash":
                cur_picks = []; cur_weights = np.array([]); cash = True
            else:
                sub_panel = panel_by_asof.get(m); sub_ml = ml_by_asof.get(m)
                sub_chr = chr_by_asof.get(m)
                if sub_panel is None or sub_ml is None:
                    cur_picks = []; cur_weights = np.array([])
                else:
                    sp_set = members_g.get(m, set())
                    sub = sub_panel[sub_panel["ticker"].isin(sp_set)]
                    sub = sub[~sub["ticker"].isin(EXCLUDE)]
                    sub = sub.merge(sub_ml[["ticker", "ml_score"]], on="ticker", how="left")
                    sub = sub.dropna(subset=["ml_score"])
                    if chr_q > 0 and sub_chr is not None and not sub_chr.empty:
                        sub = sub.merge(sub_chr[["ticker", "chronos_p70_3m"]],
                                        on="ticker", how="left")
                        sub = sub.dropna(subset=["chronos_p70_3m"])
                        sub["chr_p70_rk"] = sub["chronos_p70_3m"].rank(pct=True)
                        sub = sub[sub["chr_p70_rk"] >= chr_q]
                    sub = sub.sort_values("ml_score", ascending=False)
                    top = sub.head(k)
                    if len(top) < k:
                        cur_picks = []; cur_weights = np.array([])
                    else:
                        cur_picks = top["ticker"].tolist()
                        cur_weights = calc_invvol_weights(cur_picks, mr, m, cap=cap)
                        for tk, w in zip(cur_picks, cur_weights):
                            picks_log.append({"asof": m, "ticker": tk, "weight": float(w),
                                              "regime": regime})
                cash = False
            held_for = 0
            cur_basket_entry_equity = equity
            cur_basket_peak_equity = equity
        else:
            held_for += 1
        last_regime = regime
        rows.append({"date": m, "regime": regime, "equity": equity, "ret_m": ret_m,
                     "cash": cash, "n_picks": len(cur_picks),
                     "picks": ",".join(cur_picks), "held_for": held_for})

    return pd.DataFrame(rows), picks_log, {"n_early_reb": n_early_reb}


def metrics(eq, mr) -> dict:
    n = len(eq)
    cagr = (eq["equity"].iloc[-1]) ** (12 / n) - 1
    r = eq["ret_m"].astype(float)
    sharpe = (r.mean() / max(r.std(), 1e-9)) * np.sqrt(12)
    peak = eq["equity"].cummax(); mdd = float(((eq["equity"] - peak) / peak).min())
    spy_ret = mr["SPY"].dropna()
    next_months = pd.DatetimeIndex(eq["date"]) + pd.offsets.MonthEnd(1)
    spy_aligned = [float(spy_ret.loc[nxt]) if nxt in spy_ret.index else 0.0 for nxt in next_months]
    spy_df = pd.DataFrame({"date": eq["date"], "spy_ret_m": spy_aligned})
    wf_rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy()
        if len(e) == 0:
            continue
        rr = e["ret_m"].astype(float); ec = (1 + rr).cumprod()
        cagr_v = (ec.iloc[-1]) ** (12.0 / len(ec)) - 1
        s = spy_df[(spy_df["date"] >= lo) & (spy_df["date"] <= hi)]
        sr = s["spy_ret_m"].astype(float); sc = (1 + sr).cumprod()
        scgr = (sc.iloc[-1]) ** (12.0 / len(sc)) - 1
        wf_rows.append({"cagr": cagr_v, "spy_cagr": scgr})
    wf = pd.DataFrame(wf_rows)
    # 2024-specific
    eq_yr = eq.copy(); eq_yr["year"] = pd.to_datetime(eq_yr["date"]).dt.year
    yr_strat = eq_yr.groupby("year")["ret_m"].apply(lambda r: (1+r).prod() - 1)
    spy_yr_strat = spy_ret.groupby(spy_ret.index.year).apply(lambda r: (1+r.dropna()).prod() - 1)
    yr2024 = yr_strat.get(2024, 0)
    spy2024 = spy_yr_strat.get(2024, 0)
    return {
        "cagr_full": float(cagr), "sharpe": float(sharpe), "max_dd": float(mdd),
        "wf_mean_cagr": float(wf["cagr"].mean()),
        "wf_min_cagr": float(wf["cagr"].min()),
        "wf_max_cagr": float(wf["cagr"].max()),
        "wf_mean_edge_pp": float((wf["cagr"] - wf["spy_cagr"]).mean() * 100),
        "wf_n_beats_spy": int((wf["cagr"] > wf["spy_cagr"]).sum()),
        "wf_n_positive": int((wf["cagr"] > 0).sum()),
        "wf_n_splits": int(len(wf)),
        "ret_2024": float(yr2024), "spy_2024": float(spy2024),
        "edge_2024_pp": float((yr2024 - spy2024) * 100),
    }


def main():
    t0 = time.time()
    print("Loading augmented data ...")
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    ml = pd.read_parquet(AUG / "ml_preds.parquet")
    ml["asof"] = pd.to_datetime(ml["asof"])
    ml["ml_score"] = (ml["pred_3m"] + ml["pred_6m"]) / 2
    chr_ = pd.read_parquet(AUG / "ml_preds_chronos.parquet")[["asof", "ticker", "chronos_p70_3m"]]
    chr_["asof"] = pd.to_datetime(chr_["asof"])
    spy = load_spy_features()
    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet").fillna(0.0)
    if not isinstance(mr.index, pd.DatetimeIndex):
        mr.index = pd.to_datetime(mr.index)
    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set).to_dict()
    panel_by_asof = {a: g for a, g in panel.groupby("asof")}
    ml_by_asof = {a: g for a, g in ml.groupby("asof")}
    chr_by_asof = {a: g for a, g in chr_.groupby("asof")}
    months = sorted(set(panel["asof"]).intersection(set(spy.index)))
    months = [pd.Timestamp(m) for m in months]

    base_variants = [
        ("h=6 fixed (deployed)",    {"mode": "fixed", "hold": 6}),
        ("h=3 fixed (quarterly)",   {"mode": "fixed", "hold": 3}),
        ("h=4 fixed",               {"mode": "fixed", "hold": 4}),
        ("h=2 fixed",               {"mode": "fixed", "hold": 2}),
        ("rule min=6, score_drift", {"mode": "rule_based", "min_hold": 6,
                                     "triggers": ["score_drift"]}),
        ("rule min=3, score_drift", {"mode": "rule_based", "min_hold": 3,
                                     "triggers": ["score_drift"]}),
        ("rule min=3, rank_drift",  {"mode": "rule_based", "min_hold": 3,
                                     "triggers": ["rank_drift"]}),
        ("rule min=6, drawdown 15%",{"mode": "rule_based", "min_hold": 6,
                                     "triggers": ["drawdown"], "dd_threshold": -0.15}),
        ("rule min=6, drawdown 25%",{"mode": "rule_based", "min_hold": 6,
                                     "triggers": ["drawdown"], "dd_threshold": -0.25}),
        ("rule min=3 max=9, score+regime", {"mode": "rule_based", "min_hold": 3,
                                     "max_hold": 9,
                                     "triggers": ["score_drift", "regime_change"]}),
        ("rule min=3 max=12, score+regime",{"mode": "rule_based", "min_hold": 3,
                                     "max_hold": 12,
                                     "triggers": ["score_drift", "regime_change"]}),
    ]
    # Test at both K=2 (deployed) and K=3 (prior deployed) for each variant.
    variants = []
    for k in (2, 3):
        for name, rule in base_variants:
            variants.append((f"K={k} | {name}", rule, k))

    results = []
    for name, rule, k in variants:
        eq, picks_log, info = run_v5_with_rule(
            rule, ml, chr_, spy, mr, members_g, months,
            panel_by_asof, ml_by_asof, chr_by_asof, k=k,
        )
        m = metrics(eq, mr)
        n_baskets = len(set([p["asof"] for p in picks_log]))
        n_early = info.get("n_early_reb", 0)
        print(f"  {name:>50s}  n_bask={n_baskets:>3d} n_early={n_early:>3d}  "
              f"CAGR {m['cagr_full']*100:>6.2f}% WF {m['wf_mean_cagr']*100:>6.2f}% "
              f"Sh {m['sharpe']:>4.2f} DD {m['max_dd']*100:>6.1f}% "
              f"beats {m['wf_n_beats_spy']}/{m['wf_n_splits']} "
              f"2024 {m['edge_2024_pp']:>+6.1f}pp  ({time.time()-t0:.0f}s)")
        m["variant"] = name; m["n_baskets"] = n_baskets; m["n_early_reb"] = n_early
        m["k"] = k
        results.append(m)

    df = pd.DataFrame(results)
    df.to_csv(AUG / "rebalance_rule_sweep.csv", index=False)
    print(f"\nSaved -> {AUG / 'rebalance_rule_sweep.csv'}")
    print("\n=== Summary ranked by WF mean CAGR ===")
    df_sorted = df.sort_values("wf_mean_cagr", ascending=False)
    cols = ["variant", "cagr_full", "wf_mean_cagr", "sharpe", "max_dd",
            "wf_n_beats_spy", "edge_2024_pp", "n_baskets"]
    print(df_sorted[cols].to_string(index=False))


if __name__ == "__main__":
    main()
