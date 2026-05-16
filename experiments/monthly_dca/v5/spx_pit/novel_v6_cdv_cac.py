"""Novel v6 research: Chronos Distributional Downside Veto (CDV) +
Conviction-Adaptive Concentration (CAC).

Tested against the EXACT deployed K=2 harness (sweep_v5_aug.run_one data
loaders + identical costs / regime gate / inv-vol weighting / hold rule),
so every delta is attributable to the new selection logic alone.

Honesty protocol (matches this repo's culture):
  - All thresholds are a-priori cross-sectional quantiles or economic
    constants. NOTHING is swept to fit the curve.
  - Reported walk-forward (10 splits) AND a true design/holdout cut
    (design <=2012, untouched holdout 2013+), exactly like Phase B.
  - Negative results are reported as negative. No cherry-picking.

Idea recap
----------
Chronos currently contributes ONE bit: is p70 rank >= 0.45. Its full
predictive distribution (p50/p70/p90 + median-path peak) is discarded.
The repo's collinearity proof (Phase B) covered only the Chronos *mean*
vs GBM (rho 0.97); the predicted *dispersion / skew / give-back* is a
different object and was never tested. CDV mines it for downside. CAC
makes concentration conditional on monthly edge sharpness.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sweep_v5_aug import (  # noqa: E402
    AUG, PIT, EXCLUDE, COST_BPS, WF_SPLITS,
    calc_invvol_weights,
)

CHR_Q = 0.45          # keep the deployed Chronos p70 pre-gate (comparability)
HOLD = 6
CAP = 0.40
BASE_K = 2


def load_data():
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    ml = pd.read_parquet(AUG / "ml_preds.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
    ml["asof"] = pd.to_datetime(ml["asof"])
    ml["ml_score"] = (ml["pred_3m"] + ml["pred_6m"]) / 2
    chr_ = pd.read_parquet(AUG / "ml_preds_chronos.parquet")
    chr_["asof"] = pd.to_datetime(chr_["asof"])
    # Canonical deployed regime sequence + month grid (identical gating to
    # v5_winner_equity.csv -> baseline mode reproduces the deployed sim).
    cano = pd.read_csv(AUG / "v5_winner_equity.csv", parse_dates=["date"])
    regime_by_m = dict(zip(pd.to_datetime(cano["date"]), cano["regime"]))
    months = [pd.Timestamp(d) for d in cano["date"]]
    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet").fillna(0.0)
    mp = pd.read_parquet(AUG / "monthly_prices_clean.parquet")
    if not isinstance(mr.index, pd.DatetimeIndex):
        mr.index = pd.to_datetime(mr.index)
        mp.index = pd.to_datetime(mp.index)
    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set).to_dict()
    return dict(
        panel_by_asof={a: g for a, g in panel.groupby("asof")},
        ml_by_asof={a: g for a, g in ml.groupby("asof")},
        chr_by_asof={a: g for a, g in chr_.groupby("asof")},
        regime_by_m=regime_by_m, mr=mr, members_g=members_g, months=months,
    )


def select(m, data, mode):
    """Return (picks, spy_fraction). mode in {'baseline','cdv','cac','cdv_cac'}."""
    sub_panel = data["panel_by_asof"].get(m)
    sub_ml = data["ml_by_asof"].get(m)
    sub_chr = data["chr_by_asof"].get(m)
    if sub_panel is None or sub_ml is None:
        return [], 0.0
    sp_set = data["members_g"].get(m, set())
    sub = sub_panel[sub_panel["ticker"].isin(sp_set)]
    sub = sub[~sub["ticker"].isin(EXCLUDE)]
    sub = sub.merge(sub_ml[["ticker", "ml_score"]], on="ticker", how="left")
    sub = sub.dropna(subset=["ml_score"])
    if sub_chr is not None and not sub_chr.empty:
        sub = sub.merge(
            sub_chr[["ticker", "chronos_p50_3m", "chronos_p70_3m",
                     "chronos_p90_3m", "chronos_p50_peak"]],
            on="ticker", how="left")
        sub = sub.dropna(subset=["chronos_p70_3m"])
        sub["chr_p70_rk"] = sub["chronos_p70_3m"].rank(pct=True)
        sub = sub[sub["chr_p70_rk"] >= CHR_Q]          # deployed pre-gate
    if len(sub) < BASE_K:
        return [], 0.0

    use_cdv = mode in ("cdv", "cdv_cac")
    use_cac = mode in ("cac", "cdv_cac")

    # ---- CDV: distributional downside veto + convexity tilt -------------
    if use_cdv and "chronos_p50_3m" in sub.columns and len(sub) >= 6:
        p50 = sub["chronos_p50_3m"]
        giveback = sub["chronos_p50_peak"] - sub["chronos_p50_3m"]
        # bottom-tercile median-forecast = falling-knife; top-tercile
        # give-back = spike-then-fade. Veto either.
        keep = (p50 > p50.quantile(1.0 / 3.0)) & \
               (giveback < giveback.quantile(2.0 / 3.0))
        if keep.sum() >= BASE_K:
            sub = sub[keep]
        # convexity = right-skew of the predicted distribution
        up = (sub["chronos_p90_3m"] - sub["chronos_p70_3m"]).clip(lower=0)
        lo = (sub["chronos_p70_3m"] - sub["chronos_p50_3m"]).abs() + 1e-6
        sub = sub.assign(_convex=(up / lo).rank(pct=True))

    sub = sub.sort_values("ml_score", ascending=False)

    # ---- CAC: conviction-adaptive concentration ------------------------
    spy_frac = 0.0
    k = BASE_K
    if use_cac and len(sub) >= 8:
        s = sub["ml_score"].to_numpy()
        # how separated is the marginal (K-th) pick from the pack?
        conv = (s[BASE_K - 1] - np.median(s)) / (s.std() + 1e-9)
        if conv >= 1.0:          # sharp edge -> concentrate, capture parabola
            k, spy_frac = 2, 0.0
        elif conv >= 0.5:        # moderate -> diversify the picks
            k, spy_frac = 4, 0.0
        else:                    # no edge this month -> de-risk to SPY
            k, spy_frac = 4, 0.5

    top = sub.head(k)
    if len(top) < min(k, BASE_K):
        return [], 0.0
    if use_cdv and "_convex" in top.columns and len(top) > 1:
        # tilt within picks toward the convex ones (bounded, then renorm)
        w = (0.6 + 0.8 * top["_convex"].to_numpy())
        w = w / w.sum()
        return list(zip(top["ticker"].tolist(), w)), spy_frac
    n = len(top)
    return list(zip(top["ticker"].tolist(), [1.0 / n] * n)), spy_frac


def simulate(data, mode):
    mr = data["mr"]; months = data["months"]; regime_by_m = data["regime_by_m"]
    cf = COST_BPS / 1e4
    cur, cur_w = [], np.array([])
    spy_frac = 0.0
    cash = False
    held = 0
    equity = 1.0
    rows = []
    for i, m in enumerate(months):
        regime = regime_by_m.get(m, "normal")
        do_reb = (i == 0) or (held >= HOLD) or (cash != (regime == "crash"))
        ret_m = 0.0
        if not cash and len(cur):
            pos = mr.index.searchsorted(m)
            if pos + 1 < len(mr.index):
                nxt = mr.index[pos + 1]
                pr = np.array([0.0 if (tk not in mr.columns or pd.isna(mr.at[nxt, tk]))
                               else float(mr.at[nxt, tk]) for tk in cur])
                picks_ret = float((pr * cur_w).sum())
                spy_ret = float(mr.at[nxt, "SPY"]) if "SPY" in mr.columns and nxt in mr.index else 0.0
                ret_m = (1.0 - spy_frac) * picks_ret + spy_frac * spy_ret
                equity *= (1 + ret_m)
        if do_reb:
            equity *= (1 - cf)
            if regime == "crash":
                cur, cur_w, spy_frac, cash = [], np.array([]), 0.0, True
            else:
                picks, spy_frac = select(m, data, mode)
                if not picks:
                    cur, cur_w = [], np.array([])
                else:
                    tickers = [p[0] for p in picks]
                    base_w = calc_invvol_weights(tickers, mr, m, cap=CAP)
                    tilt = np.array([p[1] for p in picks])
                    w = base_w * tilt
                    cur, cur_w = tickers, w / w.sum()
                cash = False
            held = 0
        else:
            held += 1
        rows.append({"date": m, "regime": regime, "equity": equity,
                     "ret_m": ret_m, "cash": cash, "n_picks": len(cur),
                     "spy_frac": spy_frac})
    return pd.DataFrame(rows)


def metrics(eq, mr, label):
    n = len(eq)
    r = eq["ret_m"].astype(float)
    cagr = eq["equity"].iloc[-1] ** (12 / n) - 1
    sharpe = (r.mean() / max(r.std(), 1e-9)) * np.sqrt(12)
    peak = eq["equity"].cummax()
    mdd = float(((eq["equity"] - peak) / peak).min())
    spy_ret = mr["SPY"].dropna()
    nxt = pd.DatetimeIndex(eq["date"]) + pd.offsets.MonthEnd(1)
    spy_al = pd.Series([float(spy_ret.loc[x]) if x in spy_ret.index else 0.0 for x in nxt],
                       index=eq["date"].values)
    wf = []
    for sp, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)]
        if not len(e):
            continue
        ec = (1 + e["ret_m"].astype(float)).cumprod()
        cg = ec.iloc[-1] ** (12.0 / len(ec)) - 1
        sm = spy_al[(eq["date"] >= lo).values & (eq["date"] <= hi).values]
        sc = (1 + sm).cumprod()
        scg = sc.iloc[-1] ** (12.0 / len(sc)) - 1
        wf.append({"split": sp, "cagr": cg, "spy_cagr": scg})
    wf = pd.DataFrame(wf)
    # true design/holdout cut
    des = eq[eq["date"] <= "2012-12-31"]["ret_m"].astype(float)
    hol = eq[eq["date"] >= "2013-01-01"]["ret_m"].astype(float)
    sh = lambda x: (x.mean() / max(x.std(), 1e-9)) * np.sqrt(12)
    spy_full = (1 + spy_ret.loc[eq["date"].iloc[0]:eq["date"].iloc[-1]]).prod() ** (12 / n) - 1
    return {
        "label": label, "n_months": int(n),
        "cagr_full": round(float(cagr), 4),
        "spy_cagr_full": round(float(spy_full), 4),
        "edge_pp": round(float((cagr - spy_full) * 100), 2),
        "sharpe": round(float(sharpe), 3),
        "max_dd": round(float(mdd), 4),
        "wf_mean_cagr": round(float(wf["cagr"].mean()), 4),
        "wf_min_cagr": round(float(wf["cagr"].min()), 4),
        "wf_n_beats_spy": int((wf["cagr"] > wf["spy_cagr"]).sum()),
        "wf_n_splits": int(len(wf)),
        "design_sharpe_<=2012": round(float(sh(des)), 3),
        "holdout_sharpe_>=2013": round(float(sh(hol)), 3),
        "holdout_cagr_>=2013": round(float((1 + hol).prod() ** (12 / len(hol)) - 1), 4),
    }


def dca_lens(streams: dict):
    """Reuse the committed DCA evaluator's math on each return stream."""
    import dca_investor_eval as dca
    df = dca.load_streams()
    spy = df["SPY"]
    out = {}
    for label, eq in streams.items():
        r = eq.set_index(pd.PeriodIndex(eq["date"], freq="M"))["ret_m"].reindex(spy.index).fillna(0.0)
        res = {}
        for H in (12, 36, 60, 120):
            n = len(r)
            wins, moics, worst = [], [], 1e9
            for s in range(0, n - H + 1):
                rr = r.iloc[s:s + H].to_numpy()
                ss = spy.iloc[s:s + H].to_numpy()
                tv = dca.dca_path(rr)[0][-1]
                sv = dca.dca_path(ss)[0][-1]
                moic = tv / (H)
                wins.append(tv > sv)
                moics.append(moic)
                worst = min(worst, moic)
            res[f"H{H}"] = {
                "win_vs_spy": round(float(np.mean(wins)), 4),
                "median_moic": round(float(np.median(moics)), 3),
                "min_moic": round(float(worst), 3),
            }
        full_v, full_b = dca.dca_path(r.to_numpy())
        pk = np.maximum.accumulate(full_v)
        res["full_moic"] = round(float(full_v[-1] / full_b[-1]), 2)
        res["full_max_dd"] = round(float(((full_v - pk) / pk).min()), 4)
        out[label] = res
    return out


def main():
    data = load_data()
    streams, summ = {}, []
    for mode, label in [("baseline", "deployed_K2"),
                        ("cdv", "CDV"), ("cac", "CAC"),
                        ("cdv_cac", "CDV+CAC")]:
        eq = simulate(data, mode)
        streams[label] = eq
        eq.to_csv(AUG / f"novel_v6_{mode}_equity.csv", index=False)
        summ.append(metrics(eq, data["mr"], label))

    dca = dca_lens(streams)
    result = {"lump_metrics": summ, "dca_investor_lens": dca}
    (AUG / "novel_v6_results.json").write_text(json.dumps(result, indent=2))

    print(f"{'variant':<14}{'CAGR':>8}{'edge':>8}{'Shrp':>6}{'MaxDD':>8}"
          f"{'WFmean':>8}{'beats':>7}{'holdSh':>8}")
    for s in summ:
        print(f"{s['label']:<14}{s['cagr_full']*100:>7.1f}%{s['edge_pp']:>7.1f}"
              f"{s['sharpe']:>6.2f}{s['max_dd']*100:>7.1f}%{s['wf_mean_cagr']*100:>7.1f}%"
              f"{s['wf_n_beats_spy']:>4}/{s['wf_n_splits']}{s['holdout_sharpe_>=2013']:>8.2f}")
    print("\nDCA investor lens (win vs SPY-DCA / median MOIC / min MOIC):")
    for label, r in dca.items():
        print(f"  {label:<12} 10y: win={r['H120']['win_vs_spy']*100:5.1f}%  "
              f"med={r['H120']['median_moic']:6.2f}x  min={r['H120']['min_moic']:5.2f}x  "
              f"| 1y min={r['H12']['min_moic']:.2f}x  fullDD={r['full_max_dd']*100:.0f}%  "
              f"fullMOIC={r['full_moic']:.0f}x")


if __name__ == "__main__":
    main()
