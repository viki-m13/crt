"""Push WaveTrend win-rate as high as possible -- HONESTLY.

A high win-rate alone is the Result-1 trap (87.5% win / 4.3% CAGR / 16
trades). Here we maximise win-rate on a TRAIN window subject to hard
floors so the result is a real strategy, not a degenerate one:

    maximise   train win-rate
    subject to train trades >= 50
               train CAGR   >= max(6%, 0.6 * SPY train CAGR)
               train MaxDD   >= -55%

The optimiser climbs a soft-penalised version of this; selection uses the
train window only. The holdout win-rate (never tuned on) is reported so we
can see whether the high win-rate is real out-of-sample. We also dump the
full win-rate / CAGR / trade-count frontier and re-run the sleeve study.

Usage:  python3 experiments/wavetrend/explore_winrate.py
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
    EXCLUDE, OUT, PIT, FParams, build_daily_membership, cagr_monthly,
    load_membership, load_prices, max_dd_monthly, metrics_block,
    monthly_returns_from_equity, sharpe_monthly, simulate_filtered, win_rate,
)
from experiments.wavetrend.run_wavetrend_pit import (  # noqa: E402
    WF_SPLITS, sleeve_study, slice_window,
)

from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import (  # noqa: E402
    ConstantKernel, Matern, WhiteKernel,
)

KEYS = ["wt_n1", "wt_n2", "wt_sig_len", "wt_oversold", "rsi_period",
        "rsi_overbought", "trend_sma", "mkt_sma", "rs_lookback", "confirm",
        "profit_take", "stop_loss", "max_hold", "no_pyramid"]

TRAIN_LO, TRAIN_HI = "2003-01-01", "2013-12-31"
MIN_TRADES = 50


def sample(rng) -> FParams:
    def opt(p_off, lo, hi):
        return 0.0 if rng.random() < p_off else rng.uniform(lo, hi)
    return FParams(
        wt_n1=rng.uniform(15, 90),
        wt_n2=rng.uniform(30, 170),
        wt_sig_len=rng.uniform(2, 8),
        wt_oversold=rng.uniform(-90, -25),
        rsi_period=rng.uniform(60, 300),
        rsi_overbought=rng.uniform(60, 90),
        trend_sma=opt(0.25, 40, 220),
        mkt_sma=opt(0.35, 50, 220),
        rs_lookback=opt(0.4, 20, 252),
        confirm=rng.integers(0, 4),
        profit_take=opt(0.3, 0.05, 0.6),
        stop_loss=opt(0.35, 0.05, 0.5),
        max_hold=opt(0.4, 20, 500),
        no_pyramid=int(rng.random() < 0.5),
    ).clamp()


def vec(p: FParams) -> np.ndarray:
    return np.array([getattr(p, k) for k in KEYS], dtype=float)


def evaluate(close, member, spy, p, spy_train_cagr):
    eq, tr = simulate_filtered(close, member, spy, p)
    mr = monthly_returns_from_equity(eq)
    rt = slice_window(mr, TRAIN_LO, TRAIN_HI)
    if tr.empty:
        return dict(score=-9, wr=0, ntr=0, mr=mr, tr=tr)
    in_tr = ((tr["entry_date"] >= pd.Timestamp(TRAIN_LO)) &
             (tr["entry_date"] <= pd.Timestamp(TRAIN_HI)))
    tr_tr = tr[in_tr]
    ntr = int(len(tr_tr))
    wr_tr = float((tr_tr["ret"] > 0).mean()) if ntr else 0.0
    cg = cagr_monthly(rt)
    mdd = max_dd_monthly(rt)
    cagr_floor = max(0.06, 0.6 * spy_train_cagr)
    pen = (3.0 * max(0.0, (MIN_TRADES - ntr) / MIN_TRADES)
           + 3.0 * max(0.0, cagr_floor - cg)
           + 2.0 * max(0.0, (-0.55 - mdd)))
    feasible = ntr >= MIN_TRADES and cg >= cagr_floor and mdd >= -0.55
    return dict(score=wr_tr - pen, wr=wr_tr, ntr=ntr, cg=cg, mdd=mdd,
                feasible=feasible, mr=mr, tr=tr)


def optimise(close, member, spy, spy_train_cagr, n_init=26, n_iter=34,
             n_cand=240, seed=11):
    rng = np.random.default_rng(seed)
    X, y, rows = [], [], []
    best = None
    for i in range(n_init):
        p = sample(rng)
        r = evaluate(close, member, spy, p, spy_train_cagr)
        X.append(vec(p)); y.append(r["score"])
        rows.append((p, r))
        if best is None or r["score"] > best[1]["score"]:
            best = (p, r)
        print(f"  [init {i+1:2d}/{n_init}] wr={r['wr']*100:5.1f}% "
              f"ntr={r['ntr']:4d} score={r['score']:6.3f} "
              f"best_wr={best[1]['wr']*100:5.1f}%")
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
        cvs = (cv - X.mean(0)) / (X.std(0) + 1e-9)
        mu, sd = gp.predict(cvs, return_std=True)
        p = cs[int(np.argmax(mu + 2.0 * sd))]
        r = evaluate(close, member, spy, p, spy_train_cagr)
        X = np.vstack([X, vec(p)]); y = np.append(y, r["score"])
        rows.append((p, r))
        if r["score"] > best[1]["score"]:
            best = (p, r)
        print(f"  [bo  {it+1:2d}/{n_iter}] wr={r['wr']*100:5.1f}% "
              f"ntr={r['ntr']:4d} score={r['score']:6.3f} "
              f"best_wr={best[1]['wr']*100:5.1f}%")
    return best, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px = load_prices()
    mem = load_membership("sp500")
    uni = sorted((set(mem["ticker"].unique()) & set(px.columns)) - EXCLUDE)
    dates = px.index[px.index >= pd.Timestamp("2003-01-01")]
    close = px.loc[dates, uni]
    member = build_daily_membership(mem, dates, uni)
    spy = px["SPY"]
    spy_m = spy.resample("ME").last().pct_change().dropna()
    spy_train_cagr = cagr_monthly(slice_window(spy_m, TRAIN_LO, TRAIN_HI))
    print(f"panel {close.shape}  SPY train CAGR {spy_train_cagr*100:.2f}%  "
          f"floor {max(0.06, 0.6*spy_train_cagr)*100:.2f}%")

    t = time.time()
    best, rows = optimise(close, member, spy, spy_train_cagr)
    bp, br = best
    print(f"\n[best feasible-constrained win-rate config] ({time.time()-t:.0f}s)")
    print(f"  params {bp}")

    # Frontier table over every evaluated config
    fr = []
    for p, r in rows:
        mr = r["mr"]; tr = r["tr"]
        ho = slice_window(mr, "2014-01-01", "2026-12-31")
        ho_tr = tr[tr["entry_date"] >= pd.Timestamp("2014-01-01")] if not tr.empty else tr
        rec = {k: getattr(p, k) for k in KEYS}
        rec.update(
            train_wr=r["wr"], train_ntr=r["ntr"],
            full_wr=win_rate(tr), full_ntr=int(len(tr)),
            holdout_wr=float((ho_tr["ret"] > 0).mean()) if len(ho_tr) else 0.0,
            holdout_ntr=int(len(ho_tr)),
            full_cagr=cagr_monthly(mr), full_sharpe=sharpe_monthly(mr),
            full_mdd=max_dd_monthly(mr),
            holdout_cagr=cagr_monthly(ho), holdout_sharpe=sharpe_monthly(ho),
            feasible=bool(r.get("feasible", False)),
        )
        fr.append(rec)
    frdf = pd.DataFrame(fr).sort_values("train_wr", ascending=False)
    frdf.to_csv(OUT / "winrate_frontier.csv", index=False)

    feas = frdf[frdf["feasible"]].copy()
    print(f"\n[frontier] {len(frdf)} configs, {len(feas)} feasible "
          "(>=50 train trades, CAGR>=floor, MaxDD>=-55%)")
    cols = ["train_wr", "holdout_wr", "full_wr", "full_ntr", "full_cagr",
            "full_sharpe", "full_mdd", "profit_take", "stop_loss",
            "trend_sma", "mkt_sma", "max_hold", "no_pyramid"]
    print("Top-10 feasible by TRAIN win-rate (holdout_wr = honest OOS):")
    print(feas.head(10)[cols].round(3).to_string(index=False))

    # Full report for the selected config
    eq, tr = simulate_filtered(close, member, spy, bp)
    mr = monthly_returns_from_equity(eq)
    full = metrics_block(mr)
    ho = slice_window(mr, "2014-01-01", "2026-12-31")
    ho_tr = tr[tr["entry_date"] >= pd.Timestamp("2014-01-01")]
    by_reason = tr["reason"].value_counts().to_dict()
    print(f"\n[selected] full: CAGR {full['cagr']*100:.2f}%  "
          f"Sharpe {full['sharpe']:.2f}  MDD {full['mdd']*100:.1f}%  "
          f"trades {len(tr)}  win {win_rate(tr)*100:.1f}%")
    print(f"           holdout (2014+): win {float((ho_tr['ret']>0).mean())*100:.1f}%"
          f"  CAGR {cagr_monthly(ho)*100:.2f}%  Sharpe {sharpe_monthly(ho):.2f}")
    print(f"           exit reasons: {by_reason}")

    # Sleeve study vs deployed v5
    v5 = pd.read_csv(PIT / "augmented" / "v5_winner_equity.csv",
                     parse_dates=["date"])
    v5r = v5.set_index("date")["ret_m"].astype(float)
    v5r.index = v5r.index + pd.offsets.MonthEnd(0)
    mrx = mr.copy(); mrx.index = mrx.index + pd.offsets.MonthEnd(0)
    st = sleeve_study(mrx, v5r)
    bb = max(st["blends"], key=lambda b: b["wf_min_sharpe"])
    print(f"\n[sleeve] corr_to_v5={st['corr_full']:.3f}  "
          f"best-by-wf-min-Sharpe {bb['name']}: Sharpe {bb['sharpe']:.2f} "
          f"wf_min {bb['wf_min_sharpe']:.2f} MDD {bb['mdd']*100:.1f}%")

    result = {
        "selected_params": {k: getattr(bp, k) for k in KEYS},
        "selected_full": full,
        "selected_full_win_rate": win_rate(tr),
        "selected_full_trades": int(len(tr)),
        "selected_holdout_win_rate": float((ho_tr["ret"] > 0).mean()) if len(ho_tr) else 0.0,
        "selected_holdout_cagr": cagr_monthly(ho),
        "selected_holdout_sharpe": sharpe_monthly(ho),
        "selected_train_win_rate": br["wr"],
        "exit_reason_counts": by_reason,
        "n_feasible": int(len(feas)),
        "sleeve_vs_v5": st,
        "top5_feasible_by_train_wr": feas.head(5)[cols].round(4).to_dict("records"),
    }
    (OUT / "winrate_explore_results.json").write_text(
        json.dumps(result, indent=2, default=str))
    tr.to_csv(OUT / "winrate_selected_trades.csv", index=False)
    mr.to_csv(OUT / "winrate_selected_monthly_returns.csv", header=["ret_m"])
    print(f"\n[saved] {OUT / 'winrate_explore_results.json'}")


if __name__ == "__main__":
    main()
