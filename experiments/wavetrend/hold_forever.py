"""WaveTrend as a PURE ENTRY signal -- never sell -- optimised with an
explicit anti-overfit / generalisation gauntlet, then tested on PIT data.

Idea (user): never sell. WaveTrend only decides WHEN/WHAT to buy; every
unit is held forever. "Never selling" is itself the resolution of the
win-rate/return tension: winners run unbounded, so a buy-the-dip-and-hold
book in S&P 500 members has a structurally high entry hit-rate WITHOUT the
negative-skew CAGR penalty a profit-take imposes.

Anti-overfit design:
  * Only 5 parameters (n1, n2, sig_len, oversold, optional trend SMA).
    Fewer knobs = less to overfit.
  * Selection on TRAIN ONLY (S&P 500 2003-2013). Holdout 2014-2026 is
    never used for tuning.
  * Generalisation gauntlet (all reported, none used for selection):
      G1 time-OOS holdout 2014-2026
      G2 walk-forward splits vs SPY
      G3 CROSS-UNIVERSE: the S&P500-trained params run UNCHANGED on the
         PIT Nasdaq-100 (2015-2026) vs QQQ -- a true out-of-sample
         universe, the strongest generalisation test.
      G4 parameter PLATEAU check: metrics on a neighbourhood grid around
         the optimum (a spike = overfit; a plateau = robust).
      G5 optimised vs DEFAULT WT params, OOS -- did tuning actually add
         OOS value or just fit noise?
  * Sleeve study vs deployed v5 for completeness.

Usage:  python3 experiments/wavetrend/hold_forever.py
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
    EXCLUDE, OUT, PIT, HParams, build_daily_membership, cagr_monthly,
    load_membership, load_prices, max_dd_monthly, metrics_block,
    monthly_returns_from_equity, sharpe_monthly, simulate_hold_forever,
)
from experiments.wavetrend.run_wavetrend_pit import (  # noqa: E402
    WF_SPLITS, sleeve_study, slice_window,
)
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import (  # noqa: E402
    ConstantKernel, Matern, WhiteKernel,
)

KEYS = ["wt_n1", "wt_n2", "wt_sig_len", "wt_oversold", "trend_sma"]
BOUNDS = {"wt_n1": (15, 120), "wt_n2": (30, 220), "wt_sig_len": (2, 10),
          "wt_oversold": (-95, -25), "trend_sma": (0, 220)}
TRAIN_LO, TRAIN_HI = "2003-01-01", "2013-12-31"
HOLD_LO = "2014-01-01"
MIN_ENTRIES_TRAIN = 150


def windowed_entry_winrate(close, member, p, lo, hi):
    """Entry win-rate for units bought within [lo,hi], marked at hi."""
    r = simulate_hold_forever(close, member, p)  # not used; recompute below
    return r  # placeholder (kept simple: full-sample WR reported instead)


def book_metrics(close, member, p, lo=None, hi=None):
    r = simulate_hold_forever(close, member, p)
    mr = monthly_returns_from_equity(r["equity"])
    if lo or hi:
        mr = slice_window(mr, lo or "1900-01-01", hi or "2100-01-01")
    m = metrics_block(mr)
    m["n_entries"] = r["n_entries"]
    m["entry_winrate"] = r["entry_winrate"]
    return m, mr, r


def objective(close, member, p, spy_train_cagr):
    m, mr, r = book_metrics(close, member, p, TRAIN_LO, TRAIN_HI)
    if r["n_entries"] < MIN_ENTRIES_TRAIN or len(mr) < 24:
        return -9.0, m
    sh = m["sharpe"]
    cg = float(np.clip(m["cagr"], -1, 1))
    mdd = m["mdd"]
    # risk-adjusted, mild CAGR bonus, penalise DD beyond -60%
    return sh + 0.3 * cg + 2.0 * min(0.0, mdd + 0.60), m


def sample(rng) -> HParams:
    return HParams(
        wt_n1=rng.uniform(*BOUNDS["wt_n1"]),
        wt_n2=rng.uniform(*BOUNDS["wt_n2"]),
        wt_sig_len=rng.uniform(*BOUNDS["wt_sig_len"]),
        wt_oversold=rng.uniform(*BOUNDS["wt_oversold"]),
        trend_sma=0.0 if rng.random() < 0.3 else rng.uniform(40, 220),
    ).clamp()


def vec(p):
    return np.array([getattr(p, k) for k in KEYS], dtype=float)


def optimise(close, member, spy_train_cagr, n_init=20, n_iter=30,
             n_cand=200, seed=13):
    rng = np.random.default_rng(seed)
    X, y = [], []
    best = None
    for i in range(n_init):
        p = sample(rng)
        s, m = objective(close, member, p, spy_train_cagr)
        X.append(vec(p)); y.append(s)
        if best is None or s > best[1]:
            best = (p, s, m)
        print(f"  [init {i+1:2d}/{n_init}] score={s:6.3f} "
              f"Sh={m['sharpe']:.2f} CAGR={m['cagr']*100:5.1f}% "
              f"DD={m['mdd']*100:5.1f}% ent={m['n_entries']:4d} best={best[1]:.3f}")
    X = np.vstack(X); y = np.array(y)
    d = X.shape[1]
    kern = (ConstantKernel(1.0, (1e-3, 1e3)) *
            Matern(length_scale=np.ones(d), nu=2.5) +
            WhiteKernel(1e-4, (1e-7, 1e-1)))
    for it in range(n_iter):
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        gp = GaussianProcessRegressor(kernel=kern, normalize_y=True,
                                      n_restarts_optimizer=2, random_state=seed)
        gp.fit(Xs, y)
        cs = [sample(rng) for _ in range(n_cand)]
        cv = np.vstack([vec(c) for c in cs])
        mu, sd = gp.predict((cv - X.mean(0)) / (X.std(0) + 1e-9),
                            return_std=True)
        p = cs[int(np.argmax(mu + 2.0 * sd))]
        s, m = objective(close, member, p, spy_train_cagr)
        X = np.vstack([X, vec(p)]); y = np.append(y, s)
        if s > best[1]:
            best = (p, s, m)
        print(f"  [bo  {it+1:2d}/{n_iter}] score={s:6.3f} "
              f"Sh={m['sharpe']:.2f} CAGR={m['cagr']*100:5.1f}% "
              f"DD={m['mdd']*100:5.1f}% ent={m['n_entries']:4d} best={best[1]:.3f}")
    return best[0]


def wf_vs_bench(mr, bench):
    rows = []
    for name, lo, hi in WF_SPLITS:
        r = slice_window(mr, lo, hi)
        b = slice_window(bench, lo, hi)
        if len(r) < 6:
            continue
        rows.append({"split": name, "n": len(r),
                     "cagr": cagr_monthly(r), "sharpe": sharpe_monthly(r),
                     "bench_cagr": cagr_monthly(b),
                     "edge_pp": (cagr_monthly(r) - cagr_monthly(b)) * 100})
    return pd.DataFrame(rows)


def load_universe(universe, start):
    px = load_prices()
    mem = load_membership(universe)
    uni = sorted((set(mem["ticker"].unique()) & set(px.columns)) - EXCLUDE)
    dates = px.index[px.index >= pd.Timestamp(start)]
    close = px.loc[dates, uni]
    member = build_daily_membership(mem, dates, uni)
    return px, close, member


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}

    # ---- S&P 500: optimise (train-only) ----
    px, close, member = load_universe("sp500", "2003-01-01")
    spy_m = (px["SPY"].reindex(close.index).ffill()
             .resample("ME").last().pct_change().dropna())
    spy_train_cagr = cagr_monthly(slice_window(spy_m, TRAIN_LO, TRAIN_HI))
    print(f"S&P500 panel {close.shape}  SPY train CAGR "
          f"{spy_train_cagr*100:.2f}%")
    t = time.time()
    best = optimise(close, member, spy_train_cagr)
    print(f"\n[best train-selected] {best}  ({time.time()-t:.0f}s)")
    default = HParams()  # n1=60,n2=140,sig=4,os=-60, no trend gate

    full_b, mr_b, r_b = book_metrics(close, member, best)
    full_d, mr_d, r_d = book_metrics(close, member, default)
    tr_b = book_metrics(close, member, best, TRAIN_LO, TRAIN_HI)[0]
    ho_b = book_metrics(close, member, best, HOLD_LO, "2100-01-01")[0]
    ho_d = book_metrics(close, member, default, HOLD_LO, "2100-01-01")[0]

    print("\n=== S&P 500 (never-sell book) ===")
    for tag, m in [("OPTIMISED full", full_b), ("OPTIMISED train", tr_b),
                   ("OPTIMISED holdout 2014+", ho_b),
                   ("DEFAULT full", full_d),
                   ("DEFAULT holdout 2014+", ho_d)]:
        print(f"  {tag:26s} CAGR {m['cagr']*100:6.2f}%  Sharpe "
              f"{m['sharpe']:.2f}  MDD {m['mdd']*100:6.1f}%  "
              f"entry_wr {m['entry_winrate']*100:5.1f}%  ent {m['n_entries']}")
    print(f"  SPY full: CAGR {cagr_monthly(spy_m)*100:.2f}%  "
          f"Sharpe {sharpe_monthly(spy_m):.2f}  MDD {max_dd_monthly(spy_m)*100:.1f}%")

    # G2 walk-forward vs SPY
    wf = wf_vs_bench(mr_b, spy_m)
    print("\n[G2 walk-forward vs SPY]")
    print(wf.round(3).to_string(index=False))
    print(f"  splits beating SPY: {(wf['edge_pp']>0).sum()}/{len(wf)}")

    # G4 plateau: vary each param +/- around the optimum, hold others
    print("\n[G4 parameter plateau around optimum] (full-window CAGR/Sharpe)")
    plateau = {}
    bp = best
    for k, deltas in [("wt_n1", [-15, -7, 7, 15]),
                      ("wt_n2", [-40, -20, 20, 40]),
                      ("wt_sig_len", [-2, -1, 1, 2]),
                      ("wt_oversold", [-15, -7, 7, 15]),
                      ("trend_sma", [-40, -20, 20, 40])]:
        line = []
        for dlt in deltas:
            kw = {kk: getattr(bp, kk) for kk in KEYS}
            kw[k] = kw[k] + dlt
            mm = book_metrics(close, member, HParams(**kw))[0]
            line.append((dlt, round(mm["cagr"], 3), round(mm["sharpe"], 2)))
        plateau[k] = line
        print(f"  {k:12s} " + "  ".join(
            f"{d:+d}:C{c*100:.0f}/S{s}" for d, c, s in line))

    # G3 CROSS-UNIVERSE: same S&P500-trained params on PIT NDX, unchanged
    pxn, closen, membern = load_universe("ndx", "2015-01-01")
    qqq_m = (pxn["QQQ"].reindex(closen.index).ffill()
             .resample("ME").last().pct_change().dropna())
    full_n = book_metrics(closen, membern, best)[0]
    mr_n = monthly_returns_from_equity(simulate_hold_forever(
        closen, membern, best)["equity"])
    wf_n = wf_vs_bench(mr_n, qqq_m)
    print("\n[G3 CROSS-UNIVERSE: S&P500-trained params on PIT NDX 2015+]")
    print(f"  NDX book CAGR {full_n['cagr']*100:.2f}%  Sharpe "
          f"{full_n['sharpe']:.2f}  MDD {full_n['mdd']*100:.1f}%  "
          f"entry_wr {full_n['entry_winrate']*100:.1f}%")
    print(f"  QQQ      CAGR {cagr_monthly(qqq_m)*100:.2f}%  Sharpe "
          f"{sharpe_monthly(qqq_m):.2f}")
    print(f"  WF splits beating QQQ: {(wf_n['edge_pp']>0).sum()}/{len(wf_n)}")

    # Sleeve vs deployed v5
    v5 = pd.read_csv(PIT / "augmented" / "v5_winner_equity.csv",
                     parse_dates=["date"])
    v5r = v5.set_index("date")["ret_m"].astype(float)
    v5r.index = v5r.index + pd.offsets.MonthEnd(0)
    mx = mr_b.copy(); mx.index = mx.index + pd.offsets.MonthEnd(0)
    st = sleeve_study(mx, v5r)
    bb = max(st["blends"], key=lambda b: b["wf_min_sharpe"])
    print(f"\n[sleeve vs v5] corr={st['corr_full']:.3f} "
          f"max_split={st['max_abs_corr_across_splits']:.3f}  "
          f"best blend {bb['name']}: Sharpe {bb['sharpe']:.2f} "
          f"wf_min {bb['wf_min_sharpe']:.2f} MDD {bb['mdd']*100:.1f}%")

    res = {
        "selected_params": {k: getattr(best, k) for k in KEYS},
        "default_params": {k: getattr(default, k) for k in KEYS},
        "sp500": {
            "optimised_full": full_b, "optimised_train": tr_b,
            "optimised_holdout": ho_b, "default_full": full_d,
            "default_holdout": ho_d,
            "spy_full": metrics_block(spy_m),
            "wf_vs_spy": wf.to_dict("records"),
            "wf_n_beats_spy": int((wf["edge_pp"] > 0).sum()),
        },
        "G3_cross_universe_ndx": {
            "ndx_book": full_n, "qqq": metrics_block(qqq_m),
            "wf_vs_qqq": wf_n.to_dict("records"),
            "wf_n_beats_qqq": int((wf_n["edge_pp"] > 0).sum()),
        },
        "G4_plateau": plateau,
        "sleeve_vs_v5": st,
    }
    (OUT / "hold_forever_results.json").write_text(
        json.dumps(res, indent=2, default=str))
    mr_b.to_csv(OUT / "hold_forever_sp500_monthly_returns.csv",
                header=["ret_m"])
    print(f"\n[saved] {OUT / 'hold_forever_results.json'}")


if __name__ == "__main__":
    main()
