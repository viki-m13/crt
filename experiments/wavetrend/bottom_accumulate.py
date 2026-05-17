"""WT-Bottom: enhanced WaveTrend "absolute bottom" detector + depth
pyramiding, NEVER SELL. Optimised train-only, full anti-overfit gauntlet,
tested on PIT data.

Enhancements over plain WaveTrend (see wavetrend_pit.simulate_bottom_
accumulate docstring): rolling-range bottom-quartile filter, SPY-regime
uptrend gate (works when the market trends), WaveTrend curl confirmation,
depth-scaled pyramiding (more units the deeper the bottom), re-arm so each
bottom is a distinct event, and never sell.

Anti-overfit / generalisation (same gauntlet as Part 3):
  G1 time-OOS holdout 2014-2026 (never tuned on)
  G2 walk-forward vs SPY
  G3 CROSS-UNIVERSE: S&P500-trained params, unchanged, on PIT NDX vs QQQ
  G4 parameter plateau around the optimum
  G5 vs Part-3 plain never-sell  AND  vs default WT-Bottom -- did the
     enhancements + tuning add real OOS value?
Selection uses the S&P-500 train window (2003-2013) only.

Usage:  python3 experiments/wavetrend/bottom_accumulate.py
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
    EXCLUDE, OUT, PIT, BParams, HParams, build_daily_membership, cagr_monthly,
    load_membership, load_prices, max_dd_monthly, metrics_block,
    monthly_returns_from_equity, sharpe_monthly, simulate_bottom_accumulate,
    simulate_hold_forever,
)
from experiments.wavetrend.run_wavetrend_pit import (  # noqa: E402
    WF_SPLITS, sleeve_study, slice_window,
)
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import (  # noqa: E402
    ConstantKernel, Matern, WhiteKernel,
)

KEYS = ["wt_n1", "wt_n2", "wt_sig_len", "wt_oversold", "roll_window",
        "q_bottom", "trend_sma", "mkt_sma", "dd_step"]
TRAIN_LO, TRAIN_HI = "2003-01-01", "2013-12-31"
HOLD_LO = "2014-01-01"
MIN_EVENTS_TRAIN = 200


def book(close, member, spy, p, lo=None, hi=None):
    r = simulate_bottom_accumulate(close, member, p, spy)
    mr = monthly_returns_from_equity(r["equity"])
    if lo or hi:
        mr = slice_window(mr, lo or "1900-01-01", hi or "2100-01-01")
    m = metrics_block(mr)
    m["n_events"] = r["n_events"]
    m["n_units"] = r["n_units"]
    m["entry_winrate"] = r["entry_winrate"]
    return m, mr, r


def objective(close, member, spy, p):
    m, mr, r = book(close, member, spy, p, TRAIN_LO, TRAIN_HI)
    if r["n_events"] < MIN_EVENTS_TRAIN or len(mr) < 24:
        return -9.0, m
    return (m["sharpe"] + 0.3 * float(np.clip(m["cagr"], -1, 1))
            + 2.0 * min(0.0, m["mdd"] + 0.55)), m


def sample(rng) -> BParams:
    return BParams(
        wt_n1=rng.uniform(15, 120),
        wt_n2=rng.uniform(20, 200),
        wt_sig_len=rng.uniform(2, 9),
        wt_oversold=rng.uniform(-90, -15),
        roll_window=rng.uniform(40, 400),
        q_bottom=rng.uniform(0.1, 0.5),
        trend_sma=0.0 if rng.random() < 0.5 else rng.uniform(50, 220),
        mkt_sma=0.0 if rng.random() < 0.3 else rng.uniform(50, 220),
        dd_step=rng.uniform(0.04, 0.4),
    ).clamp()


def vec(p):
    return np.array([getattr(p, k) for k in KEYS], dtype=float)


def optimise(close, member, spy, n_init=22, n_iter=32, n_cand=220, seed=17):
    rng = np.random.default_rng(seed)
    X, y = [], []
    best = None
    for i in range(n_init):
        p = sample(rng)
        s, m = objective(close, member, spy, p)
        X.append(vec(p)); y.append(s)
        if best is None or s > best[1]:
            best = (p, s, m)
        print(f"  [init {i+1:2d}/{n_init}] score={s:6.3f} Sh={m['sharpe']:.2f} "
              f"CAGR={m['cagr']*100:5.1f}% DD={m['mdd']*100:5.1f}% "
              f"ev={m['n_events']:4d} best={best[1]:.3f}")
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
        s, m = objective(close, member, spy, p)
        X = np.vstack([X, vec(p)]); y = np.append(y, s)
        if s > best[1]:
            best = (p, s, m)
        print(f"  [bo  {it+1:2d}/{n_iter}] score={s:6.3f} Sh={m['sharpe']:.2f} "
              f"CAGR={m['cagr']*100:5.1f}% DD={m['mdd']*100:5.1f}% "
              f"ev={m['n_events']:4d} best={best[1]:.3f}")
    return best[0]


def wf(mr, bench):
    rows = []
    for name, lo, hi in WF_SPLITS:
        r = slice_window(mr, lo, hi)
        b = slice_window(bench, lo, hi)
        if len(r) < 6:
            continue
        rows.append({"split": name, "n": len(r), "cagr": cagr_monthly(r),
                     "sharpe": sharpe_monthly(r), "bench_cagr": cagr_monthly(b),
                     "edge_pp": (cagr_monthly(r) - cagr_monthly(b)) * 100})
    return pd.DataFrame(rows)


def load_uni(universe, start):
    px = load_prices()
    mem = load_membership(universe)
    uni = sorted((set(mem["ticker"].unique()) & set(px.columns)) - EXCLUDE)
    dates = px.index[px.index >= pd.Timestamp(start)]
    return px, px.loc[dates, uni], build_daily_membership(mem, dates, uni)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px, close, member = load_uni("sp500", "2003-01-01")
    spy = px["SPY"]
    spy_m = (spy.reindex(close.index).ffill()
             .resample("ME").last().pct_change().dropna())
    print(f"S&P500 panel {close.shape}")

    t = time.time()
    best = optimise(close, member, spy)
    print(f"\n[best train-selected] {best}  ({time.time()-t:.0f}s)")

    full_b, mr_b, r_b = book(close, member, spy, best)
    tr_b = book(close, member, spy, best, TRAIN_LO, TRAIN_HI)[0]
    ho_b = book(close, member, spy, best, HOLD_LO, "2100-01-01")[0]
    default = BParams()
    full_def = book(close, member, spy, default)[0]
    # G5 reference: Part-3 plain never-sell winner
    p3 = HParams(wt_n1=99, wt_n2=40, wt_sig_len=8,
                 wt_oversold=-29.8966707, trend_sma=210)
    r_p3 = simulate_hold_forever(close, member, p3)
    mr_p3 = monthly_returns_from_equity(r_p3["equity"])
    p3_full = metrics_block(mr_p3)
    p3_full["entry_winrate"] = r_p3["entry_winrate"]
    p3_ho = metrics_block(slice_window(mr_p3, HOLD_LO, "2100-01-01"))

    print("\n=== S&P 500 (WT-Bottom never-sell) ===")
    for tag, m in [("OPTIMISED full", full_b), ("OPTIMISED train", tr_b),
                   ("OPTIMISED holdout 14+", ho_b),
                   ("DEFAULT full", full_def)]:
        print(f"  {tag:24s} CAGR {m['cagr']*100:6.2f}%  Sh {m['sharpe']:.2f}  "
              f"DD {m['mdd']*100:6.1f}%  ewr {m['entry_winrate']*100:5.1f}%  "
              f"ev {m.get('n_events','-')} u {m.get('n_units','-')}")
    print(f"  PART3 plain never-sell    full CAGR {p3_full['cagr']*100:.2f}% "
          f"Sh {p3_full['sharpe']:.2f} DD {p3_full['mdd']*100:.1f}% "
          f"ewr {p3_full['entry_winrate']*100:.1f}% | holdout Sh "
          f"{p3_ho['sharpe']:.2f} CAGR {p3_ho['cagr']*100:.1f}%")
    print(f"  SPY full CAGR {cagr_monthly(spy_m)*100:.2f}%  "
          f"Sh {sharpe_monthly(spy_m):.2f}  DD {max_dd_monthly(spy_m)*100:.1f}%")

    wfs = wf(mr_b, spy_m)
    print("\n[G2 walk-forward vs SPY]")
    print(wfs.round(3).to_string(index=False))
    print(f"  splits beating SPY: {(wfs['edge_pp']>0).sum()}/{len(wfs)}")

    print("\n[G4 parameter plateau] (full CAGR/Sharpe)")
    plateau = {}
    for k, ds in [("wt_n1", [-20, -10, 10, 20]), ("wt_n2", [-30, -15, 15, 30]),
                  ("wt_oversold", [-15, -7, 7, 15]),
                  ("roll_window", [-60, -30, 30, 60]),
                  ("q_bottom", [-0.1, -0.05, 0.05, 0.1]),
                  ("mkt_sma", [-40, -20, 20, 40]),
                  ("dd_step", [-0.05, -0.02, 0.02, 0.05])]:
        line = []
        for dl in ds:
            kw = {kk: getattr(best, kk) for kk in KEYS}
            kw[k] = kw[k] + dl
            mm = book(close, member, spy, BParams(**kw))[0]
            line.append([dl, round(mm["cagr"], 3), round(mm["sharpe"], 2)])
        plateau[k] = line
        print(f"  {k:12s} " + "  ".join(
            f"{d:+g}:C{c*100:.0f}/S{s}" for d, c, s in line))

    # G3 cross-universe NDX
    pxn, closen, membern = load_uni("ndx", "2015-01-01")
    qqq = pxn["QQQ"]
    qqq_m = (qqq.reindex(closen.index).ffill()
             .resample("ME").last().pct_change().dropna())
    fb_n, mr_n, r_n = book(closen, membern, qqq, best)
    wfn = wf(mr_n, qqq_m)
    print("\n[G3 CROSS-UNIVERSE: S&P500-trained params on PIT NDX 2015+]")
    print(f"  NDX book CAGR {fb_n['cagr']*100:.2f}%  Sh {fb_n['sharpe']:.2f}  "
          f"DD {fb_n['mdd']*100:.1f}%  ewr {fb_n['entry_winrate']*100:.1f}%  "
          f"ev {fb_n['n_events']}")
    print(f"  QQQ CAGR {cagr_monthly(qqq_m)*100:.2f}%  "
          f"Sh {sharpe_monthly(qqq_m):.2f}")
    print(f"  WF splits beating QQQ: {(wfn['edge_pp']>0).sum()}/{len(wfn)}")

    v5 = pd.read_csv(PIT / "augmented" / "v5_winner_equity.csv",
                     parse_dates=["date"])
    v5r = v5.set_index("date")["ret_m"].astype(float)
    v5r.index = v5r.index + pd.offsets.MonthEnd(0)
    mx = mr_b.copy(); mx.index = mx.index + pd.offsets.MonthEnd(0)
    st = sleeve_study(mx, v5r)
    bb = max(st["blends"], key=lambda b: b["wf_min_sharpe"])
    print(f"\n[sleeve vs v5] corr={st['corr_full']:.3f} "
          f"max_split={st['max_abs_corr_across_splits']:.3f}  "
          f"best {bb['name']}: Sharpe {bb['sharpe']:.2f} "
          f"wf_min {bb['wf_min_sharpe']:.2f} MDD {bb['mdd']*100:.1f}%")

    res = {
        "selected_params": {k: getattr(best, k) for k in KEYS},
        "sp500": {"optimised_full": full_b, "optimised_train": tr_b,
                  "optimised_holdout": ho_b, "default_full": full_def,
                  "part3_plain_full": p3_full, "part3_plain_holdout": p3_ho,
                  "spy_full": metrics_block(spy_m),
                  "wf_vs_spy": wfs.to_dict("records"),
                  "wf_n_beats_spy": int((wfs["edge_pp"] > 0).sum())},
        "G3_cross_universe_ndx": {
            "ndx_book": fb_n, "qqq": metrics_block(qqq_m),
            "wf_vs_qqq": wfn.to_dict("records"),
            "wf_n_beats_qqq": int((wfn["edge_pp"] > 0).sum())},
        "G4_plateau": plateau,
        "sleeve_vs_v5": st,
    }
    (OUT / "bottom_accumulate_results.json").write_text(
        json.dumps(res, indent=2, default=str))
    mr_b.to_csv(OUT / "bottom_accumulate_sp500_monthly_returns.csv",
                header=["ret_m"])
    print(f"\n[saved] {OUT / 'bottom_accumulate_results.json'}")


if __name__ == "__main__":
    main()
