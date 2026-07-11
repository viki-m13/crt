"""Walk-forward backtest + calibration for the SPY level predictor.

Everything here is strictly point-in-time: the physical probability at
date t uses only forward returns realized before t.  We report:

  1. Calibration  — physical_prob bucket vs realized frequency (proves the
     estimator is honest: ~99% predicted -> ~99% realized).
  2. No-breach track record — realized accuracy of the headline rule at
     several operating points, split design (<2016) / validation (>=2016),
     plus a non-overlapping subsample (defeats horizon-overlap inflation),
     and the mean market-disagreement (edge) on fired predictions.
  3. Direction track record — realized hit rate of the risk-premium
     directional calls vs the market's coin-flip implied odds.
  4. The misses — which calendar years produce the breaches (crash onsets).

Run:  python3 backtest.py   (writes results/backtest.json + prints report)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from predict import (Panel, load_panel, market_prob_above,
                     physical_prob_above, DESIGN_END, RESULTS, IV_MULT)


def _split_index(p: Panel) -> int:
    return int(np.searchsorted(p.dates, np.datetime64(DESIGN_END[:10])))


def fired(p: Panel, X: int, off: float, p_min: float, edge_min: float,
          start: int = 1070):
    """Yield fired no-breach predictions: (i, phys, mkt, outcome_safe)."""
    out = []
    split = _split_index(p)
    for i in range(start, len(p.px) - X):
        mkt = market_prob_above(p.rv[i], X, off)
        if not np.isfinite(mkt):
            continue
        phys = physical_prob_above(p, i, X, off)
        if phys is None or phys < p_min or (phys - mkt) < edge_min:
            continue
        safe = 1.0 if p.px[i + X] > p.px[i] * (1 + off) else 0.0
        out.append((i, phys, mkt, safe, i >= split))
    return out


def calibration(p: Panel) -> list[dict]:
    rows = []
    pooled = []
    for X in [21, 63]:
        for off in [-0.03, -0.05, -0.07, -0.10, -0.15]:
            for (i, phys, mkt, safe, isval) in fired(p, X, off, 0.80, -1.0):
                pooled.append((phys, safe))
    pooled = np.array(pooled)
    for lo, hi in [(0.80, 0.90), (0.90, 0.95), (0.95, 0.98),
                   (0.98, 0.99), (0.99, 0.995), (0.995, 1.0001)]:
        m = (pooled[:, 0] >= lo) & (pooled[:, 0] < hi)
        if m.sum():
            rows.append({"bucket": f"[{lo:.3f},{hi:.3f})",
                         "predicted_mid": round((lo + hi) / 2, 4),
                         "realized": round(float(pooled[m, 1].mean()), 4),
                         "n": int(m.sum())})
    return rows


def nobreach_track(p: Panel):
    rows = []
    for X, off, p_min, edge_min in [
        (21, -0.10, 0.99, 0.03),
        (21, -0.13, 0.999, 0.005),
        (63, -0.15, 0.99, 0.03),
        (126, -0.15, 0.99, 0.03),
        (63, -0.10, 0.99, 0.03),
    ]:
        f = fired(p, X, off, p_min, edge_min)
        if not f:
            continue
        a = np.array([(x[1], x[2], x[3], x[4]) for x in f])
        idxs = [x[0] for x in f]
        # non-overlapping subsample
        keep, last = [], -10 ** 9
        for k, ii in enumerate(idxs):
            if ii - last >= X:
                keep.append(k); last = ii
        keep = np.array(keep)
        des = ~a[:, 3].astype(bool)  # placeholder, replaced below
        isval = a[:, 3].astype(bool)
        # a columns: phys, mkt, safe, isval
        phys, mkt, safe, val = a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(bool)
        rows.append({
            "horizon": X, "level_off": off,
            "fires": len(f),
            "realized_all": round(float(safe.mean()), 4),
            "realized_design": round(float(safe[~val].mean()), 4) if (~val).sum() else None,
            "realized_val": round(float(safe[val].mean()), 4) if val.sum() else None,
            "n_design": int((~val).sum()), "n_val": int(val.sum()),
            "realized_nonoverlap": round(float(safe[keep].mean()), 4),
            "n_nonoverlap": int(len(keep)),
            "mean_physical": round(float(phys.mean()), 4),
            "mean_market": round(float(mkt.mean()), 4),
            "mean_edge_pp": round(float((phys - mkt).mean() * 100), 2),
        })
    return rows


def direction_track(p: Panel):
    """Realized hit rate of ATM/up directional calls vs market implied."""
    rows = []
    split = _split_index(p)
    for X in [21, 63, 126, 252]:
        for off in [0.0, 0.03, 0.05]:
            preds = []
            for i in range(1070, len(p.px) - X):
                mkt = market_prob_above(p.rv[i], X, off)
                if not np.isfinite(mkt):
                    continue
                phys = physical_prob_above(p, i, X, off)
                if phys is None:
                    continue
                side_up = phys >= 0.5
                pp = phys if side_up else 1 - phys
                mp = mkt if side_up else 1 - mkt
                if (pp - mp) * 100 < 5.0:
                    continue
                above = p.px[i + X] > p.px[i] * (1 + off)
                hit = 1.0 if (above == side_up) else 0.0
                preds.append((hit, pp, mp, i >= split))
            if len(preds) < 50:
                continue
            a = np.array(preds)
            rows.append({
                "horizon": X, "level_off": off, "n": len(preds),
                "realized_hit": round(float(a[:, 0].mean()), 4),
                "mean_our_prob": round(float(a[:, 1].mean()), 4),
                "mean_market_prob": round(float(a[:, 2].mean()), 4),
                "mean_edge_pp": round(float((a[:, 1] - a[:, 2]).mean() * 100), 2),
                "realized_val": round(float(a[a[:, 3] == 1, 0].mean()), 4)
                if (a[:, 3] == 1).sum() else None,
                "n_val": int((a[:, 3] == 1).sum()),
            })
    return rows


def misses_by_year(p: Panel, X=21, off=-0.10):
    f = fired(p, X, off, 0.99, 0.03)
    tot, miss = {}, {}
    for i, phys, mkt, safe, isval in f:
        y = str(p.dates[i])[:4]
        tot[y] = tot.get(y, 0) + 1
        if safe == 0.0:
            miss[y] = miss.get(y, 0) + 1
    return {y: {"misses": miss.get(y, 0), "fires": tot[y]}
            for y in sorted(tot) if miss.get(y, 0)}


def main():
    p = load_panel()
    cal = calibration(p)
    nb = nobreach_track(p)
    dr = direction_track(p)
    ms = misses_by_year(p)

    print(f"SPY {p.dates[0]}..{p.dates[-1]}  n={len(p.px)}  IV_MULT={IV_MULT}\n")

    print("CALIBRATION (physical_prob bucket -> realized freq):")
    for r in cal:
        print(f"  {r['bucket']}  predicted~{r['predicted_mid']*100:5.1f}%  "
              f"realized={r['realized']*100:6.2f}%  n={r['n']}")

    print("\nNO-BREACH TRACK RECORD (down-level 'won't fall below'):")
    for r in nb:
        print(f"  {r['horizon']:>3}d off={r['level_off']:+.2f}  fires={r['fires']:>4}  "
              f"realized={r['realized_all']*100:6.2f}%  "
              f"design={ (r['realized_design'] or 0)*100:6.2f}%  "
              f"nonoverlap={r['realized_nonoverlap']*100:6.2f}%({r['n_nonoverlap']})  "
              f"edge=+{r['mean_edge_pp']:.1f}pp")

    print("\nDIRECTION TRACK RECORD (ATM/up, risk-premium edge):")
    for r in dr:
        print(f"  {r['horizon']:>3}d off={r['level_off']:+.2f}  n={r['n']:>4}  "
              f"hit={r['realized_hit']*100:6.2f}%  our={r['mean_our_prob']*100:5.1f}%  "
              f"mkt={r['mean_market_prob']*100:5.1f}%  edge=+{r['mean_edge_pp']:.1f}pp")

    print("\nTHE MISSES (21d/-10% no-breach failures by year — crash onsets):")
    for y, v in ms.items():
        print(f"  {y}: {v['misses']} / {v['fires']}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "backtest.json"), "w") as fh:
        json.dump({"calibration": cal, "nobreach": nb, "direction": dr,
                   "misses_by_year": ms,
                   "span": [str(p.dates[0]), str(p.dates[-1])],
                   "iv_mult": IV_MULT}, fh, indent=2)
    print(f"\nwrote {os.path.join(RESULTS, 'backtest.json')}")


if __name__ == "__main__":
    main()
