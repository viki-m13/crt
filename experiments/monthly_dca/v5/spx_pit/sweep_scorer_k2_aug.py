"""Phase 7e: scorer-variant sweep at K=2 on augmented PIT.

The original Phase 7 sweep used scorer=ml_3plus6 throughout. The
final_validation.md exp_02 winner used ml_3plus6plus1 (avg of pred_1m
+ pred_3m + pred_6m, includes the 1-month horizon). Other variants
worth probing on K=2:

  ml_3plus6      avg(pred_3m, pred_6m)     deployed v5/v3 scorer
  ml_3plus6plus1 avg(pred_1m, pred_3m, pred_6m)   exp_02 winner
  ml_h3          pred_3m alone
  ml_h6          pred_6m alone
  ml_h1          pred_1m alone
  ml_avg         pred (the GBM's own ensemble)

Run all at K=2 with default Chronos q=0.45, hold=6, cap=0.40,
tight regime gate. Pick the highest-WF-mean.

Output: augmented/v5_k2_scorer_sweep.csv
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


def run_with_scorer(scorer_col, panel_by_asof, ml_by_asof, chr_by_asof,
                    members_g, mr, spy, months,
                    k=2, chr_q=0.45, hold=6, cap=0.40):
    cf = COST_BPS / 1e4
    cur_picks = []; cur_weights = np.array([])
    cash = False; held_for = 0; equity = 1.0
    rows = []

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
                    sub = sub.merge(sub_ml[["ticker", scorer_col]], on="ticker", how="left")
                    sub = sub.dropna(subset=[scorer_col])
                    if chr_q > 0 and sub_chr is not None and not sub_chr.empty:
                        sub = sub.merge(sub_chr[["ticker", "chronos_p70_3m"]],
                                        on="ticker", how="left")
                        sub = sub.dropna(subset=["chronos_p70_3m"])
                        sub["chr_p70_rk"] = sub["chronos_p70_3m"].rank(pct=True)
                        sub = sub[sub["chr_p70_rk"] >= chr_q]
                    sub = sub.sort_values(scorer_col, ascending=False)
                    top = sub.head(k)
                    if len(top) < k:
                        cur_picks = []; cur_weights = np.array([])
                    else:
                        cur_picks = top["ticker"].tolist()
                        cur_weights = calc_invvol_weights(cur_picks, mr, m, cap=cap)
                cash = False
            held_for = 0
        else:
            held_for += 1
        rows.append({"date": m, "regime": regime, "equity": equity, "ret_m": ret_m})

    eq = pd.DataFrame(rows)
    n_months = len(eq)
    cagr_full = (eq["equity"].iloc[-1]) ** (12 / n_months) - 1
    r = eq["ret_m"].astype(float)
    sharpe = (r.mean() / max(r.std(), 1e-9)) * np.sqrt(12)
    peak = eq["equity"].cummax()
    mdd = float(((eq["equity"] - peak) / peak).min())

    spy_ret = mr["SPY"].dropna()
    spy_full = (1 + spy_ret.loc[eq["date"].iloc[0]:eq["date"].iloc[-1]]).prod() ** (12 / n_months) - 1
    next_months = pd.DatetimeIndex(eq["date"]) + pd.offsets.MonthEnd(1)
    spy_aligned = [float(spy_ret.loc[nxt]) if nxt in spy_ret.index else 0.0 for nxt in next_months]
    spy_df = pd.DataFrame({"date": eq["date"], "spy_ret_m": spy_aligned})

    wf_rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy()
        if len(e) == 0:
            continue
        rr = e["ret_m"].astype(float)
        ec = (1 + rr).cumprod()
        cagr_v = (ec.iloc[-1]) ** (12.0 / len(ec)) - 1
        s = spy_df[(spy_df["date"] >= lo) & (spy_df["date"] <= hi)]
        sr = s["spy_ret_m"].astype(float)
        sc = (1 + sr).cumprod()
        scgr = (sc.iloc[-1]) ** (12.0 / len(sc)) - 1
        wf_rows.append({"cagr": cagr_v, "spy_cagr": scgr})
    wf = pd.DataFrame(wf_rows)
    return dict(scorer=scorer_col, k=k, chr_q=chr_q, hold=hold, cap=cap,
                cagr_full=float(cagr_full), spy_cagr_full=float(spy_full),
                edge_full_pp=float((cagr_full - spy_full) * 100),
                sharpe=float(sharpe), max_dd=float(mdd),
                wf_mean_cagr=float(wf["cagr"].mean()),
                wf_median_cagr=float(wf["cagr"].median()),
                wf_min_cagr=float(wf["cagr"].min()),
                wf_n_beats_spy=int((wf["cagr"] > wf["spy_cagr"]).sum()),
                wf_n_positive=int((wf["cagr"] > 0).sum()),
                wf_n_splits=int(len(wf)))


def main():
    t0 = time.time()
    print("Loading augmented data ...")
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    ml = pd.read_parquet(AUG / "ml_preds.parquet")
    ml["asof"] = pd.to_datetime(ml["asof"])

    # Build all scorer columns up-front
    ml["ml_3plus6"] = (ml["pred_3m"] + ml["pred_6m"]) / 2
    ml["ml_3plus6plus1"] = (ml["pred_1m"] + ml["pred_3m"] + ml["pred_6m"]) / 3
    ml["ml_h1"] = ml["pred_1m"]
    ml["ml_h3"] = ml["pred_3m"]
    ml["ml_h6"] = ml["pred_6m"]
    ml["ml_avg"] = ml["pred"]

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

    SCORERS = ["ml_3plus6", "ml_3plus6plus1", "ml_h1", "ml_h3", "ml_h6", "ml_avg"]
    results = []
    for s in SCORERS:
        r = run_with_scorer(s, panel_by_asof, ml_by_asof, chr_by_asof,
                             members_g, mr, spy, months)
        results.append(r)
        elapsed = time.time() - t0
        print(f"  {s:<18} cagr={r['cagr_full']*100:>6.1f}%  "
              f"wf_mean={r['wf_mean_cagr']*100:>6.1f}%  "
              f"sharpe={r['sharpe']:.2f}  "
              f"dd={r['max_dd']*100:>6.1f}%  "
              f"beats={r['wf_n_beats_spy']}/{r['wf_n_splits']}  ({elapsed:.0f}s)")

    df = pd.DataFrame(results)
    df.to_csv(AUG / "v5_k2_scorer_sweep.csv", index=False)

    df_top = df.sort_values("wf_mean_cagr", ascending=False)
    print("\nRanked by WF mean CAGR:")
    print(df_top[["scorer", "cagr_full", "wf_mean_cagr", "sharpe",
                  "max_dd", "wf_n_beats_spy"]].to_string(index=False))

    winner = df_top.iloc[0].to_dict()
    (AUG / "v5_k2_scorer_winner.json").write_text(
        json.dumps(winner, indent=2, default=str))


if __name__ == "__main__":
    main()
