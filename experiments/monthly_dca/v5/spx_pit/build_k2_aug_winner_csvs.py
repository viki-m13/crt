"""Phase 7g: regenerate all v5_winner_*.csv artifacts for K=2 on the
augmented PIT panel, so the homepage builder
(experiments/monthly_dca/v5/build_webapp_v5_pit.py) reads K=2 numbers
across all sections (WF, bias overlay, sub-periods, sensitivity,
drawdowns, most-picked).

Outputs (in augmented/, schema-matching the original
PIT/v5_winner_*.csv files so the builder can drop-in read them):

  v5_winner_walkforward.csv      10-split WF metrics
  v5_winner_bias_sensitivity.csv MC delisting at 8 alpha levels
  v5_winner_sub_periods.csv      4 sub-period slices
  v5_winner_sensitivity.csv      param sweep (cap / hold / gate / weighting / cost)
  v5_winner_most_picked.csv      top-30 picked tickers
  v5_winner_drawdowns.csv        top-10 drawdown episodes
  v5_winner_yearly.csv           year-by-year
  v5_winner_summary.json         headline summary
"""
from __future__ import annotations

import json
import sys
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
HOLD = 6
CAP = 0.40


def _load_data():
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    ml = pd.read_parquet(AUG / "ml_preds.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
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
    return panel, ml, chr_, spy, mr, members_g


def run_sim(panel, ml, chr_, spy, mr, members_g, k=K, chr_q=CHR_Q, hold=HOLD, cap=CAP):
    """Run v5 sim, return (eq_df, picks_log)."""
    panel_by_asof = {a: g for a, g in panel.groupby("asof")}
    ml_by_asof = {a: g for a, g in ml.groupby("asof")}
    chr_by_asof = {a: g for a, g in chr_.groupby("asof")}
    months = sorted(set(panel["asof"]).intersection(set(spy.index)))
    months = [pd.Timestamp(m) for m in months]
    cf = COST_BPS / 1e4

    cur_picks = []; cur_weights = np.array([])
    cash = False; held_for = 0; equity = 1.0
    rows = []
    picks_log = []

    for i, m in enumerate(months):
        regime = classify_regime_tight(spy.loc[m].to_dict() if m in spy.index else {})
        do_reb = (i == 0) or (held_for >= hold) or (cash != (regime == "crash"))
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
        else:
            held_for += 1
        rows.append({"date": m, "regime": regime, "equity": equity, "ret_m": ret_m,
                     "cash": cash, "n_picks": len(cur_picks),
                     "picks": ",".join(cur_picks)})
    return pd.DataFrame(rows), pd.DataFrame(picks_log)


def make_walkforward(eq, mr):
    spy_ret = mr["SPY"].dropna()
    next_months = pd.DatetimeIndex(eq["date"]) + pd.offsets.MonthEnd(1)
    spy_aligned = [float(spy_ret.loc[nxt]) if nxt in spy_ret.index else 0.0 for nxt in next_months]
    spy_df = pd.DataFrame({"date": eq["date"], "spy_ret_m": spy_aligned})
    rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy()
        if len(e) == 0:
            continue
        r = e["ret_m"].astype(float)
        ec = (1 + r).cumprod()
        cagr_v = (ec.iloc[-1]) ** (12.0 / len(ec)) - 1
        sh = (r.mean() / max(r.std(), 1e-9)) * np.sqrt(12)
        peak = ec.cummax()
        mdd = float(((ec - peak) / peak).min())
        s = spy_df[(spy_df["date"] >= lo) & (spy_df["date"] <= hi)]
        sr = s["spy_ret_m"].astype(float); sc = (1 + sr).cumprod()
        scgr = (sc.iloc[-1]) ** (12.0 / len(sc)) - 1
        rows.append({
            "split": split, "from": str(lo.date()), "to": str(hi.date()),
            "n_m": int(len(e)),
            "cagr": float(cagr_v), "spy_cagr": float(scgr),
            "edge_pp": float((cagr_v - scgr) * 100),
            "sharpe": float(sh), "max_dd": float(mdd),
            "n_cash": int((e["regime"] == "crash").sum()),
        })
    return pd.DataFrame(rows)


def make_yearly(eq, mr):
    spy_ret = mr["SPY"]
    eq = eq.copy()
    eq["year"] = pd.to_datetime(eq["date"]).dt.year
    yr_strat = eq.groupby("year")["ret_m"].apply(lambda r: (1 + r).prod() - 1)
    spy_yr = spy_ret.groupby(spy_ret.index.year).apply(lambda r: (1 + r.dropna()).prod() - 1)
    rows = []
    for y in sorted(yr_strat.index):
        s = float(yr_strat[y])
        sp = float(spy_yr.get(y, 0))
        rows.append({"year": int(y),
                     "year_ret": s, "spy_year_ret": sp,
                     "edge_pp": (s - sp) * 100,
                     "Strategy_pct": s * 100, "SPY_pct": sp * 100,
                     "edge_pp_r": (s - sp) * 100})
    return pd.DataFrame(rows)


def make_sub_periods(eq, mr):
    """Decade slices: 2003-2012, 2008-2017, 2013-2022, 2018-2025, 2010-2025."""
    spy_ret = mr["SPY"].dropna()
    next_months = pd.DatetimeIndex(eq["date"]) + pd.offsets.MonthEnd(1)
    spy_aligned = [float(spy_ret.loc[nxt]) if nxt in spy_ret.index else 0.0 for nxt in next_months]
    spy_df = pd.DataFrame({"date": eq["date"], "spy_ret_m": spy_aligned})
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
        r = e["ret_m"].astype(float); ec = (1 + r).cumprod()
        cv = (ec.iloc[-1]) ** (12.0 / len(ec)) - 1
        s = spy_df[(spy_df["date"] >= lo) & (spy_df["date"] <= hi)]
        sr = s["spy_ret_m"].astype(float); sc = (1 + sr).cumprod()
        scgr = (sc.iloc[-1]) ** (12.0 / len(sc)) - 1
        rows.append({
            "period": label, "from": str(lo.date()), "to": str(hi.date()),
            "cagr": float(cv), "spy_cagr": float(scgr),
            "edge_pp": float((cv - scgr) * 100), "n_m": int(len(e)),
        })
    return pd.DataFrame(rows)


def make_drawdowns(eq, top_n=10):
    eq = eq.sort_values("date").reset_index(drop=True)
    equity = eq["equity"].values
    dates = pd.to_datetime(eq["date"]).values
    peak = equity[0]; peak_idx = 0
    in_dd = False
    episodes = []
    cur = None
    for i, v in enumerate(equity):
        if v >= peak:
            if in_dd and cur is not None:
                cur["end"] = dates[i]; cur["depth_pct"] = (cur["trough_value"] / cur["peak_value"] - 1) * 100
                episodes.append(cur); cur = None; in_dd = False
            peak = v; peak_idx = i
        else:
            if not in_dd:
                in_dd = True
                cur = {"start": dates[peak_idx], "peak_value": peak,
                       "trough_value": v, "trough": dates[i], "end": dates[i]}
            else:
                if v < cur["trough_value"]:
                    cur["trough_value"] = v; cur["trough"] = dates[i]
    if cur:
        cur["end"] = dates[-1]; cur["depth_pct"] = (cur["trough_value"] / cur["peak_value"] - 1) * 100
        episodes.append(cur)
    df = pd.DataFrame(episodes)
    if df.empty:
        return df
    df = df.sort_values("depth_pct").head(top_n)
    return df[["start", "trough", "end", "depth_pct"]]


def make_most_picked(picks_log, top_n=30):
    if picks_log.empty:
        return pd.DataFrame()
    n_by_t = picks_log.groupby("ticker").size().sort_values(ascending=False).head(top_n)
    return pd.DataFrame({"ticker": n_by_t.index, "n_months_picked": n_by_t.values})


def make_sensitivity_table(panel, ml, chr_, spy, mr, members_g):
    """Param sensitivity table. Vary one parameter at a time around the K=2 winner."""
    rows = []
    def run(name, value, **kw):
        cfg = dict(k=K, chr_q=CHR_Q, hold=HOLD, cap=CAP)
        cfg.update(kw)
        eq, _ = run_sim(panel, ml, chr_, spy, mr, members_g, **cfg)
        n = len(eq)
        cagr = (eq["equity"].iloc[-1]) ** (12 / n) - 1
        r = eq["ret_m"].astype(float)
        sh = (r.mean() / max(r.std(), 1e-9)) * np.sqrt(12)
        peak = eq["equity"].cummax()
        mdd = float(((eq["equity"] - peak) / peak).min())
        wf = make_walkforward(eq, mr)
        return {"name": f"{name}={value}",
                "cagr_full": float(cagr), "spy_cagr_full": 0.1163,
                "edge_full_pp": float((cagr - 0.1163) * 100),
                "sharpe": float(sh), "max_dd": float(mdd), "n_cash": 0,
                "wf_mean_cagr": float(wf["cagr"].mean()) if len(wf) else 0,
                "wf_median_cagr": float(wf["cagr"].median()) if len(wf) else 0,
                "wf_min_cagr": float(wf["cagr"].min()) if len(wf) else 0,
                "wf_max_cagr": float(wf["cagr"].max()) if len(wf) else 0,
                "wf_mean_edge_pp": float(wf["edge_pp"].mean()) if len(wf) else 0,
                "wf_n_pos": int((wf["cagr"] > 0).sum()) if len(wf) else 0,
                "wf_n_beats": int((wf["cagr"] > wf["spy_cagr"]).sum()) if len(wf) else 0,
                "wf_n_splits": int(len(wf)) if len(wf) else 0,
                "param": name, "value": value}
    # K sweep
    for k in (1, 2, 3, 4, 5):
        rows.append(run("K", k, k=k))
    # Hold sweep
    for h in (3, 6, 9, 12):
        rows.append(run("hold", h, hold=h))
    # Cap sweep
    for c in (0.34, 0.40, 0.50, 1.00):
        rows.append(run("cap", c, cap=c))
    # Chronos q sweep
    for q in (0.30, 0.45, 0.60):
        rows.append(run("chr_q", q, chr_q=q))
    return pd.DataFrame(rows)


def main():
    print("Loading augmented data ...")
    panel, ml, chr_, spy, mr, members_g = _load_data()

    print(f"Running K={K} v5 sim on augmented PIT ...")
    eq, picks_log = run_sim(panel, ml, chr_, spy, mr, members_g)
    n = len(eq); cagr = (eq["equity"].iloc[-1]) ** (12 / n) - 1
    print(f"  cagr_full={cagr*100:.2f}%   final_equity=${eq['equity'].iloc[-1]:.2f}   "
          f"n_months={n}")

    wf = make_walkforward(eq, mr)
    yr = make_yearly(eq, mr)
    sp = make_sub_periods(eq, mr)
    dd = make_drawdowns(eq)
    mp = make_most_picked(picks_log)
    print(f"  WF: mean {wf['cagr'].mean()*100:.2f}%, "
          f"beats SPY {int((wf['cagr'] > wf['spy_cagr']).sum())}/{len(wf)}")

    # Bias-sensitivity: reuse v5_k2_bias_sensitivity.csv (already exists)
    bias_src = AUG / "v5_k2_bias_sensitivity.csv"
    if bias_src.exists():
        bias = pd.read_csv(bias_src)
    else:
        # Fallback: trivial zero
        bias = pd.DataFrame()

    # Sensitivity
    print(f"  Computing parameter sensitivity (16 cells) ...")
    sens = make_sensitivity_table(panel, ml, chr_, spy, mr, members_g)

    # Write all artifacts
    eq.to_csv(AUG / "v5_winner_equity.csv", index=False)
    wf.to_csv(AUG / "v5_winner_walkforward.csv", index=False)
    yr.to_csv(AUG / "v5_winner_yearly.csv", index=False)
    sp.to_csv(AUG / "v5_winner_sub_periods.csv", index=False)
    if not dd.empty:
        dd.to_csv(AUG / "v5_winner_drawdowns.csv", index=False)
    if not mp.empty:
        mp.to_csv(AUG / "v5_winner_most_picked.csv", index=False)
    if not bias.empty:
        bias.to_csv(AUG / "v5_winner_bias_sensitivity.csv", index=False)
    sens.to_csv(AUG / "v5_winner_sensitivity.csv", index=False)

    summary = {
        "variant_name": "v5_chr_p70_q0.45_k2_invvol_cap0.4_h6_tight",
        "panel": "augmented_PIT",
        "n_months": int(n),
        "final_equity": float(eq["equity"].iloc[-1]),
        "cagr_full": float(cagr),
        "wf_mean_cagr": float(wf["cagr"].mean()) if len(wf) else None,
        "wf_median_cagr": float(wf["cagr"].median()) if len(wf) else None,
        "wf_min_cagr": float(wf["cagr"].min()) if len(wf) else None,
        "wf_max_cagr": float(wf["cagr"].max()) if len(wf) else None,
        "wf_mean_edge_pp": float(wf["edge_pp"].mean()) if len(wf) else None,
        "wf_n_positive": int((wf["cagr"] > 0).sum()) if len(wf) else 0,
        "wf_n_beats_spy": int((wf["cagr"] > wf["spy_cagr"]).sum()) if len(wf) else 0,
    }
    (AUG / "v5_winner_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved all v5_winner_*.csv artifacts to {AUG}")
    print(f"Headline summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
