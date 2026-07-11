"""Directional long with an early-close trigger — does closing early to
lock a profit push accuracy to 99%, and does the edge survive?

Three exit styles, all on SPY daily closes (no intraday; conservative on
touches), horizon-capped at X sessions, entered every session:

  fixed(X, dp[, ds])   buy; exit +dp the first close it's reached; optional
                       stop at -ds; else exit at the horizon close.
  trailing(X, arm, tr) buy; once up +arm, exit when price falls tr below its
                       running max; else exit at the horizon close. Lets
                       winners run instead of clipping them at a fixed +dp.

We report win-rate (overall + validation>=2016), expectancy per trade, the
win/loss asymmetry, the worst trade, and — the number that actually matters —
the CAGR of a sequential one-position-at-a-time book vs SPY buy-and-hold.

The headline result (see VALIDATION.md §6): early-close DOES buy a ~99% win
rate, but only by accepting avg_loss ~= 20x avg_win, and the resulting CAGR
falls BELOW buy-and-hold. Accuracy and edge trade off along win ~ 1/(1+payoff);
you cannot have both. The best joint point is a trailing trigger at ~90% win
and ~buy-and-hold return.
"""
from __future__ import annotations
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SPY_PATH = os.environ.get(
    "SPX_SPY_PATH",
    os.path.join(HERE, "..", "credit_spread", "cache_full", "SPY.json"))
SPLIT = np.datetime64("2016-01-01")


def _load():
    b = json.load(open(SPY_PATH))
    dates = np.array(b["series"]["dates"], dtype="datetime64[D]")
    px = np.array(b["series"]["prices"], float)
    return dates, px


def fixed(px, split_i, X, dp, ds=None, start=60):
    n = len(px); pnl = []; hold = []; win = []; val = []
    for i in range(start, n - X):
        e = px[i]; tg = e * (1 + dp); st = e * (1 - ds) if ds else -1.0; done = False
        for j in range(i + 1, i + X + 1):
            if px[j] >= tg:
                pnl.append(dp); hold.append(j - i); win.append(1); done = True; break
            if ds and px[j] <= st:
                pnl.append(-ds); hold.append(j - i); win.append(0); done = True; break
        if not done:
            r = px[i + X] / e - 1; pnl.append(r); hold.append(X); win.append(int(r > 0))
        val.append(i >= split_i)
    return map(np.array, (pnl, hold, win, val))


def trailing(px, split_i, X, arm, tr, start=60):
    n = len(px); pnl = []; hold = []; win = []; val = []
    for i in range(start, n - X):
        e = px[i]; armed = False; pk = e; done = False
        for j in range(i + 1, i + X + 1):
            p = px[j]; pk = max(pk, p)
            if not armed and p >= e * (1 + arm):
                armed = True
            if armed and p <= pk * (1 - tr):
                r = p / e - 1; pnl.append(r); hold.append(j - i); win.append(int(r > 0)); done = True; break
        if not done:
            r = px[i + X] / e - 1; pnl.append(r); hold.append(X); win.append(int(r > 0))
        val.append(i >= split_i)
    return map(np.array, (pnl, hold, win, val))


def sequential(dates, px, X, dp, ds=None, arm=None, tr=None, start=60):
    """Non-overlapping one-position-at-a-time equity curve -> real CAGR."""
    n = len(px); i = start; cap = 1.0; trades = 0; wins = 0; worst = 0.0
    while i < n - 1:
        e = px[i]; jend = min(i + X, n - 1); ex = jend; r = px[jend] / e - 1
        pk = e; armed = False
        for j in range(i + 1, jend + 1):
            p = px[j]
            if arm is not None:
                pk = max(pk, p)
                if not armed and p >= e * (1 + arm):
                    armed = True
                if armed and p <= pk * (1 - tr):
                    ex = j; r = p / e - 1; break
            else:
                if p >= e * (1 + dp):
                    ex = j; r = dp; break
                if ds and p <= e * (1 - ds):
                    ex = j; r = -ds; break
        cap *= (1 + r); trades += 1; wins += int(r > 0); worst = min(worst, r); i = ex
    yrs = (dates[min(i, n - 1)] - dates[start]).astype("timedelta64[D]").astype(int) / 365.25
    return dict(trades=trades, win=wins / trades, cagr=cap ** (1 / yrs) - 1,
                final=cap, years=yrs, worst=worst)


def _row(tag, pnl, hold, win, val):
    aw = pnl[win == 1].mean(); al = pnl[win == 0].mean() if (win == 0).sum() else 0.0
    return dict(tag=tag, win=float(win.mean()),
                win_val=float(win[val].mean()) if val.sum() else None,
                mean_pnl=float(pnl.mean()), avg_win=float(aw), avg_loss=float(al),
                worst=float(pnl.min()), hold=float(hold.mean()),
                ann=float(pnl.mean() / hold.mean() * 252))


def main():
    dates, px = _load()
    split_i = int(np.searchsorted(dates, SPLIT))
    yrs = (dates[-1] - dates[0]).astype("timedelta64[D]").astype(int) / 365.25
    bh = (px[-1] / px[0]) ** (1 / yrs) - 1
    print(f"SPY buy-and-hold CAGR: {bh*100:.2f}%/yr\n")

    print("FIXED target (push win-rate up -> return falls):")
    fixed_rows = []
    for X, dp in [(21, 0.03), (63, 0.03), (252, 0.02), (252, 0.01), (504, 0.01)]:
        r = _row(f"fixed X={X} dp={dp}", *fixed(px, split_i, X, dp))
        fixed_rows.append(r)
        print(f"  {r['tag']:20s} win={r['win']*100:5.1f}% val={ (r['win_val'] or 0)*100:5.1f}%  "
              f"avgWin={r['avg_win']*100:+.1f}% avgLoss={r['avg_loss']*100:+.1f}%  "
              f"worst={r['worst']*100:.0f}%  ann~{r['ann']*100:.1f}%")

    print("\nTRAILING trigger (let winners run -> best joint point):")
    trail_rows = []
    for X, arm, tr in [(126, 0.03, 0.02), (252, 0.05, 0.03), (252, 0.03, 0.02)]:
        r = _row(f"trail X={X} arm={arm} tr={tr}", *trailing(px, split_i, X, arm, tr))
        trail_rows.append(r)
        print(f"  {r['tag']:24s} win={r['win']*100:5.1f}% val={ (r['win_val'] or 0)*100:5.1f}%  "
              f"avgWin={r['avg_win']*100:+.1f}% avgLoss={r['avg_loss']*100:+.1f}%  "
              f"worst={r['worst']*100:.0f}%  ann~{r['ann']*100:.1f}%")

    print("\nSEQUENTIAL capital (real CAGR, one position at a time):")
    seqs = {
        "fixed 252/1% (99% win)": sequential(dates, px, 252, 0.01),
        "fixed 252/2%": sequential(dates, px, 252, 0.02),
        "trail 126 arm3 tr2": sequential(dates, px, 126, None, arm=0.03, tr=0.02),
        "trail 252 arm5 tr3": sequential(dates, px, 252, None, arm=0.05, tr=0.03),
    }
    for k, s in seqs.items():
        print(f"  {k:26s} {s['trades']:>4} trades  win={s['win']*100:5.1f}%  "
              f"CAGR={s['cagr']*100:5.2f}%  worst_trade={s['worst']*100:.0f}%")
    print(f"  {'buy & hold SPY':26s} {'':>4}         win=  n/a   CAGR={bh*100:5.2f}%  worst_trade=-55%")

    os.makedirs(RESULTS, exist_ok=True)
    json.dump({"buy_hold_cagr": bh, "fixed": fixed_rows, "trailing": trail_rows,
               "sequential": {k: v for k, v in seqs.items()}},
              open(os.path.join(RESULTS, "early_close.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(RESULTS, 'early_close.json')}")


if __name__ == "__main__":
    main()
