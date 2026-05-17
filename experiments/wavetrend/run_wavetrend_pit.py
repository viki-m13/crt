"""Optimise + validate WaveTrend on the PIT panels, then test it as a sleeve
against the deployed v5 stream.

Honest design (vs the repo-root `wavetrend` file which maximised in-sample
trade win-rate -- a known trap that ignores trade magnitude and drawdown):

  * Parameters are chosen ONLY on a train window (2003->2013 for S&P 500,
    2015->2020 for NDX). The holdout window is never used for selection.
  * Objective is risk-adjusted, not win-rate:
        score = sharpe + 0.5*clip(CAGR,-1,2) + 3*min(0, MDD+0.45)
    with a >=40-trade floor so degenerate "never sell at a loss" configs
    (which is what win-rate maximisation finds) are rejected.
  * A naive win-rate-maximising run is included to demonstrate the trap.
  * Output is a monthly return stream evaluated with the SAME metric code as
    the deployed v5 winner, plus a sleeve/blend study vs v5.

Usage:  python3 experiments/wavetrend/run_wavetrend_pit.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.wavetrend.wavetrend_pit import (  # noqa: E402
    EXCLUDE, OUT, PIT, Params, build_daily_membership, cagr_monthly,
    load_membership, load_prices, max_dd_monthly, metrics_block,
    monthly_returns_from_equity, sharpe_monthly, simulate, win_rate,
)

from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import (  # noqa: E402
    ConstantKernel, Matern, WhiteKernel,
)

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

PARAM_KEYS = ["wt_n1", "wt_n2", "wt_sig_len", "wt_oversold",
              "rsi_period", "rsi_overbought"]
BOUNDS = {
    "wt_n1": (15, 120),
    "wt_n2": (40, 260),
    "wt_sig_len": (2, 12),
    "wt_oversold": (-100.0, -30.0),
    "rsi_period": (40, 320),
    "rsi_overbought": (55.0, 85.0),
}


def slice_window(ret: pd.Series, lo: str, hi: str) -> pd.Series:
    return ret[(ret.index >= pd.Timestamp(lo)) & (ret.index <= pd.Timestamp(hi))]


def objective(ret_train: pd.Series, n_trades_train: int) -> float:
    if len(ret_train) < 24 or n_trades_train < 40:
        return -9.0
    sh = sharpe_monthly(ret_train)
    cg = float(np.clip(cagr_monthly(ret_train), -1.0, 2.0))
    mdd = max_dd_monthly(ret_train)
    return sh + 0.5 * cg + 3.0 * min(0.0, mdd + 0.45)


def sample_random(rng) -> Params:
    v = {}
    for k in PARAM_KEYS:
        lo, hi = BOUNDS[k]
        v[k] = rng.uniform(lo, hi)
    return Params(**v).clamp()


def to_vec(p: Params) -> np.ndarray:
    return np.array([getattr(p, k) for k in PARAM_KEYS], dtype=float)


def optimise(close, member, train_lo, train_hi, n_init=14, n_iter=26,
             n_cand=160, seed=7, win_rate_mode=False):
    """Return (best_params, history). Selection uses ONLY the train window."""
    rng = np.random.default_rng(seed)
    X, y, hist = [], [], []
    best_p, best_obj = None, -1e9

    def evaluate(p: Params):
        eq, tr = simulate(close, member, p)
        mr = monthly_returns_from_equity(eq)
        rt = slice_window(mr, train_lo, train_hi)
        ntr = 0 if tr.empty else int(
            ((tr["entry_date"] >= pd.Timestamp(train_lo)) &
             (tr["entry_date"] <= pd.Timestamp(train_hi))).sum())
        if win_rate_mode:
            obj = win_rate(tr) if (not tr.empty and len(tr) >= 5) else -9.0
        else:
            obj = objective(rt, ntr)
        return obj, mr, tr

    for i in range(n_init):
        p = sample_random(rng)
        obj, mr, tr = evaluate(p)
        X.append(to_vec(p))
        y.append(obj)
        hist.append((p, obj))
        if obj > best_obj:
            best_obj, best_p = obj, p
        print(f"  [init {i+1:2d}/{n_init}] obj={obj:7.3f} best={best_obj:7.3f} "
              f"{p}")

    X = np.vstack(X)
    y = np.array(y)
    d = X.shape[1]
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
              Matern(length_scale=np.ones(d), nu=2.5) +
              WhiteKernel(1e-4, (1e-7, 1e-1)))

    for it in range(n_iter):
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=2,
                                      random_state=seed)
        gp.fit(Xs, y)
        cands = [sample_random(rng) for _ in range(n_cand)]
        cv = np.vstack([to_vec(c) for c in cands])
        cvs = (cv - X.mean(0)) / (X.std(0) + 1e-9)
        mu, sd = gp.predict(cvs, return_std=True)
        acq = mu + 2.0 * sd  # UCB (maximise)
        p = cands[int(np.argmax(acq))]
        obj, mr, tr = evaluate(p)
        X = np.vstack([X, to_vec(p)])
        y = np.append(y, obj)
        hist.append((p, obj))
        if obj > best_obj:
            best_obj, best_p = obj, p
        print(f"  [bo  {it+1:2d}/{n_iter}] obj={obj:7.3f} best={best_obj:7.3f} "
              f"{p}")

    return best_p, hist


def wf_table(ret: pd.Series, bench: pd.Series) -> pd.DataFrame:
    rows = []
    for name, lo, hi in WF_SPLITS:
        r = slice_window(ret, lo, hi)
        b = slice_window(bench, lo, hi)
        if len(r) < 6:
            continue
        rows.append({
            "split": name, "n": len(r),
            "cagr": cagr_monthly(r), "sharpe": sharpe_monthly(r),
            "mdd": max_dd_monthly(r),
            "bench_cagr": cagr_monthly(b),
            "edge_pp": (cagr_monthly(r) - cagr_monthly(b)) * 100,
        })
    return pd.DataFrame(rows)


def benchmark_monthly(px: pd.DataFrame, tk: str, idx: pd.DatetimeIndex) -> pd.Series:
    s = px[tk].reindex(px.index).ffill()
    m = s.resample("ME").last().pct_change().dropna()
    return m.reindex(idx).dropna()


def sleeve_study(wt_ret: pd.Series, v5_ret: pd.Series) -> dict:
    """Mirror carry_sleeve_validation.json: corr, blends, WF, OOS/holdout."""
    common = wt_ret.index.intersection(v5_ret.index)
    w = wt_ret.reindex(common)
    v = v5_ret.reindex(common)
    out = {
        "overlap_months": int(len(common)),
        "overlap_start": str(common.min().date()) if len(common) else None,
        "overlap_end": str(common.max().date()) if len(common) else None,
        "wt_full": metrics_block(w),
        "v5_full": metrics_block(v),
        "corr_full": float(w.corr(v)),
    }
    split_corrs = []
    for _, lo, hi in WF_SPLITS:
        ww, vv = slice_window(w, lo, hi), slice_window(v, lo, hi)
        if len(ww) >= 6:
            split_corrs.append(abs(float(ww.corr(vv))))
    out["max_abs_corr_across_splits"] = max(split_corrs) if split_corrs else None
    out["corr_stable"] = bool(out["max_abs_corr_across_splits"] is not None
                              and out["max_abs_corr_across_splits"] < 0.5)

    blends = []
    for wv in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        b = wv * v + (1 - wv) * w
        wf = [sharpe_monthly(slice_window(b, lo, hi))
              for _, lo, hi in WF_SPLITS if len(slice_window(b, lo, hi)) >= 6]
        m = metrics_block(b)
        m["name"] = (f"{int(wv*100)}% v5 + {int((1-wv)*100)}% wt"
                     if wv < 1 else "v5 alone")
        m["wf_mean_sharpe"] = float(np.mean(wf))
        m["wf_min_sharpe"] = float(np.min(wf))
        blends.append(m)
    out["blends"] = blends
    out["oos_design_2003_2012"] = metrics_block(slice_window(w, "2003-01-01", "2012-12-31"))
    out["oos_holdout_2013_2026"] = metrics_block(slice_window(w, "2013-01-01", "2026-12-31"))
    return out


def run_universe(universe: str, start: str, train_lo: str, train_hi: str,
                 bench_tk: str) -> dict:
    print("=" * 70)
    print(f"UNIVERSE: {universe}   train {train_lo}..{train_hi}")
    print("=" * 70)
    px = load_prices()
    mem = load_membership(universe)
    uni = sorted((set(mem["ticker"].unique()) & set(px.columns)) - EXCLUDE)
    dates = px.index[px.index >= pd.Timestamp(start)]
    close = px.loc[dates, uni]
    member = build_daily_membership(mem, dates, uni)
    print(f"panel {close.shape}  tradable names {len(uni)}")

    bench = benchmark_monthly(px, bench_tk, None)

    # 1) honest risk-adjusted optimisation (train-only selection)
    print("\n[A] Honest risk-adjusted optimisation (train-only):")
    t = time.time()
    best, _ = optimise(close, member, train_lo, train_hi, win_rate_mode=False)
    print(f"  -> best {best}  ({time.time()-t:.0f}s)")

    # 2) naive win-rate optimisation (the original objective -> the trap)
    print("\n[B] Naive win-rate optimisation (original objective):")
    wr_best, _ = optimise(close, member, train_lo, train_hi,
                          n_init=10, n_iter=14, win_rate_mode=True)
    print(f"  -> best {wr_best}")

    def full_report(p: Params, tag: str) -> dict:
        eq, tr = simulate(close, member, p)
        mr = monthly_returns_from_equity(eq)
        bm = bench.reindex(mr.index).dropna()
        common = mr.index.intersection(bm.index)
        full = metrics_block(mr)
        full_tr = slice_window(mr, train_lo, train_hi)
        hold = slice_window(mr, train_hi, "2026-12-31")
        wf = wf_table(mr, bench.reindex(mr.index).ffill())
        return {
            "tag": tag,
            "params": {k: getattr(p, k) for k in PARAM_KEYS},
            "full": full,
            "train": metrics_block(full_tr),
            "holdout": metrics_block(hold),
            "n_trades": int(len(tr)),
            "win_rate": win_rate(tr),
            f"{bench_tk}_full_cagr": cagr_monthly(bm.reindex(common)),
            "wf_mean_cagr": float(wf["cagr"].mean()) if len(wf) else 0.0,
            "wf_min_cagr": float(wf["cagr"].min()) if len(wf) else 0.0,
            "wf_n_beats_bench": int((wf["edge_pp"] > 0).sum()) if len(wf) else 0,
            "wf_n_splits": int(len(wf)),
            "_mr": mr,
        }

    rep_honest = full_report(best, "honest_risk_adjusted")
    rep_naive = full_report(wr_best, "naive_win_rate")

    for r in (rep_honest, rep_naive):
        print(f"\n--- {r['tag']} ---")
        print(f"  params      {r['params']}")
        print(f"  FULL    CAGR {r['full']['cagr']*100:6.2f}%  "
              f"Sharpe {r['full']['sharpe']:.2f}  MDD {r['full']['mdd']*100:6.1f}%"
              f"  ({bench_tk} CAGR {r[bench_tk+'_full_cagr']*100:.2f}%)")
        print(f"  TRAIN   CAGR {r['train']['cagr']*100:6.2f}%  "
              f"Sharpe {r['train']['sharpe']:.2f}")
        print(f"  HOLDOUT CAGR {r['holdout']['cagr']*100:6.2f}%  "
              f"Sharpe {r['holdout']['sharpe']:.2f}  MDD {r['holdout']['mdd']*100:6.1f}%")
        print(f"  trades {r['n_trades']}  win_rate {r['win_rate']*100:.1f}%  "
              f"WF beats {bench_tk} {r['wf_n_beats_bench']}/{r['wf_n_splits']}")

    result = {
        "universe": universe,
        "panel_shape": list(close.shape),
        "tradable_names": len(uni),
        "honest": {k: v for k, v in rep_honest.items() if k != "_mr"},
        "naive_win_rate": {k: v for k, v in rep_naive.items() if k != "_mr"},
    }

    # 3) sleeve / blend study vs deployed v5 (S&P 500 only -- v5 is SPX)
    if universe == "sp500":
        v5 = pd.read_csv(PIT / "augmented" / "v5_winner_equity.csv",
                         parse_dates=["date"])
        v5_ret = v5.set_index("date")["ret_m"].astype(float)
        v5_ret.index = v5_ret.index + pd.offsets.MonthEnd(0)
        for r in (rep_honest, rep_naive):
            mr = r["_mr"].copy()
            mr.index = mr.index + pd.offsets.MonthEnd(0)
            st = sleeve_study(mr, v5_ret)
            result.setdefault("sleeve_vs_v5", {})[r["tag"]] = st
            best_blend = max(st["blends"], key=lambda b: b["wf_min_sharpe"])
            print(f"\n[sleeve {r['tag']}] corr_to_v5={st['corr_full']:.3f} "
                  f"overlap={st['overlap_months']}m  best-by-wf-min-Sharpe: "
                  f"{best_blend['name']} Sharpe={best_blend['sharpe']:.2f} "
                  f"wf_min_Sharpe={best_blend['wf_min_sharpe']:.2f} "
                  f"MDD={best_blend['mdd']*100:.1f}%")

    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    results["sp500"] = run_universe("sp500", "2003-01-01",
                                    "2003-01-01", "2013-12-31", "SPY")
    results["ndx"] = run_universe("ndx", "2015-01-01",
                                  "2015-01-01", "2020-12-31", "QQQ")
    (OUT / "wavetrend_pit_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[saved] {OUT / 'wavetrend_pit_results.json'}")


if __name__ == "__main__":
    main()
