"""Phase 7f: regime-gate sweep at K=2 on augmented PIT.

Tests whether deployed v5's 'tight' regime gate is still optimal at K=2,
or whether 'strict' (earlier crash trigger) or 'ddgate' (drawdown-based)
gives better numbers.

Three gates from sp500_pit_extended_sweep.py:
  tight    deployed v5/v3 default — crash if r21 ≤ -8% OR (r6m ≤ -5% AND r21 ≤ -3%)
  strict   earlier crash trigger — r21 ≤ -5% OR r6m ≤ -6% OR (dsma < -3% AND rsi < 42)
  ddgate   drawdown-based — dd_from_52wh ≤ -10% AND r21 < 0

Output: augmented/v5_k2_regime_sweep.csv
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
    load_spy_features, calc_invvol_weights,
)


def classify_regime_tight(s: dict) -> str:
    r21 = s.get("spy_ret_21d", 0.0); r6m = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0); mom12 = s.get("spy_mom_12_1", 0.0)
    if r21 <= -0.08 or (r6m <= -0.05 and r21 <= -0.03):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


def classify_regime_strict(s: dict) -> str:
    r21 = s.get("spy_ret_21d", 0.0); r6m = s.get("spy_mom_6_1", 0.0)
    dsma = s.get("spy_dsma200", 0.0); mom12 = s.get("spy_mom_12_1", 0.0)
    rsi = s.get("spy_rsi14", 50.0); streak = s.get("spy_below_200_streak", 0.0)
    if r21 <= -0.05 or r6m <= -0.06 or (dsma < -0.03 and rsi < 42):
        return "crash"
    if streak >= 30 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.08 and dsma > 0:
        return "bull"
    return "normal"


def load_spy_features_with_extras():
    """Extended SPY features including dd_from_52wh and rsi_14 for the
    strict / ddgate classifiers."""
    rows = []
    for f in sorted((AUG / "features").glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "SPY" not in df.index:
            continue
        spy = df.loc["SPY"]
        rows.append({
            "asof": d,
            "spy_dsma200": float(spy.get("d_sma200", 0.0)),
            "spy_mom_12_1": float(spy.get("mom_12_1", 0.0)),
            "spy_mom_6_1": float(spy.get("mom_6_1", 0.0)),
            "spy_ret_21d": float(spy.get("ret_21d", 0.0)),
            "spy_below_200_streak": float(spy.get("max_below_200_streak", 0.0)),
            "spy_rsi14": float(spy.get("rsi_14", 50.0)),
            "spy_dd_from_52wh": float(spy.get("dd_from_52wh", 0.0)),
        })
    return pd.DataFrame(rows).set_index("asof")


def classify_regime_ddgate(s: dict) -> str:
    dd = s.get("spy_dd_from_52wh", 0.0); r21 = s.get("spy_ret_21d", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0); streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    if dd <= -0.10 and r21 < 0:
        return "crash"
    if streak >= 30 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


GATES = {"tight": classify_regime_tight,
         "strict": classify_regime_strict,
         "ddgate": classify_regime_ddgate}


def run_with_gate(gate_fn, gate_name,
                  panel_by_asof, ml_by_asof, chr_by_asof,
                  members_g, mr, spy, months,
                  k=2, chr_q=0.45, hold=6, cap=0.40):
    cf = COST_BPS / 1e4
    cur_picks = []; cur_weights = np.array([])
    cash = False; held_for = 0; equity = 1.0
    rows = []

    for i, m in enumerate(months):
        regime = gate_fn(spy.loc[m].to_dict() if m in spy.index else {})
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
                cash = False
            held_for = 0
        else:
            held_for += 1
        rows.append({"date": m, "regime": regime, "equity": equity,
                     "ret_m": ret_m, "cash": cash})

    eq = pd.DataFrame(rows)
    n = len(eq)
    cagr_v = (eq["equity"].iloc[-1]) ** (12 / n) - 1
    r = eq["ret_m"].astype(float)
    sh = (r.mean() / max(r.std(), 1e-9)) * np.sqrt(12)
    peak = eq["equity"].cummax()
    mdd = float(((eq["equity"] - peak) / peak).min())

    spy_ret = mr["SPY"].dropna()
    spy_full = (1 + spy_ret.loc[eq["date"].iloc[0]:eq["date"].iloc[-1]]).prod() ** (12 / n) - 1
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
        cagr_w = (ec.iloc[-1]) ** (12.0 / len(ec)) - 1
        s = spy_df[(spy_df["date"] >= lo) & (spy_df["date"] <= hi)]
        sr = s["spy_ret_m"].astype(float); sc = (1 + sr).cumprod()
        scgr = (sc.iloc[-1]) ** (12.0 / len(sc)) - 1
        wf_rows.append({"cagr": cagr_w, "spy_cagr": scgr})
    wf = pd.DataFrame(wf_rows)
    return dict(regime_gate=gate_name, k=k,
                cagr_full=float(cagr_v), spy_cagr_full=float(spy_full),
                edge_full_pp=float((cagr_v - spy_full) * 100),
                sharpe=float(sh), max_dd=float(mdd),
                n_cash_months=int(eq["cash"].sum()),
                wf_mean_cagr=float(wf["cagr"].mean()),
                wf_min_cagr=float(wf["cagr"].min()),
                wf_n_beats_spy=int((wf["cagr"] > wf["spy_cagr"]).sum()),
                wf_n_splits=int(len(wf)))


def main():
    t0 = time.time()
    print("Loading augmented data ...")
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    ml = pd.read_parquet(AUG / "ml_preds.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
    ml["asof"] = pd.to_datetime(ml["asof"])
    ml["ml_score"] = (ml["pred_3m"] + ml["pred_6m"]) / 2
    chr_ = pd.read_parquet(AUG / "ml_preds_chronos.parquet")[["asof", "ticker", "chronos_p70_3m"]]
    chr_["asof"] = pd.to_datetime(chr_["asof"])
    spy = load_spy_features_with_extras()
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

    results = []
    for name, fn in GATES.items():
        r = run_with_gate(fn, name, panel_by_asof, ml_by_asof, chr_by_asof,
                          members_g, mr, spy, months)
        results.append(r)
        elapsed = time.time() - t0
        print(f"  {name:>8}  cagr={r['cagr_full']*100:>6.1f}%  "
              f"wf_mean={r['wf_mean_cagr']*100:>6.1f}%  "
              f"sharpe={r['sharpe']:.2f}  "
              f"dd={r['max_dd']*100:>6.1f}%  "
              f"cash_m={r['n_cash_months']}  "
              f"beats={r['wf_n_beats_spy']}/{r['wf_n_splits']}  ({elapsed:.0f}s)")

    df = pd.DataFrame(results)
    df.to_csv(AUG / "v5_k2_regime_sweep.csv", index=False)
    print(f"\nSaved -> {AUG / 'v5_k2_regime_sweep.csv'}")


if __name__ == "__main__":
    main()
