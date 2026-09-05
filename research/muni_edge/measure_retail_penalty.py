#!/usr/bin/env python3
"""Does retail systematically overpay for municipal bonds? Measure it honestly.

Business-validation test, not a trading study. The proposed product tells a
wealth manager what their clients' muni fills actually cost against the tape.
That claim is only worth selling if the penalty is real, large, and survives
the confounds — so the confounds are attacked here rather than hoped away.

BASE COMPARISON: same bond, same day, same side. Comparing a retail buy to an
institutional buy of THE SAME security on THE SAME day removes bond quality,
coupon, maturity, credit and the day's rate move. What remains is size.
Retail <= $100k par, institutional >= $1m, with a deliberate gap between the
buckets so the boundary is not the argument. Par-weighted within each bucket.

The four things that could make a real-looking penalty fake, and what is done
about each:

  1. INTRADAY DRIFT. Timestamps here vary within a day (measured: ~3 distinct
     times per bond-day), so retail and institutions may simply trade at
     different moments while the market moves. Controlled by re-running the
     comparison restricted to retail/institutional pairs executed within a
     tight time window, where drift has no room to accumulate.

  2. DIRECTION. A price gap in one direction could be drift; a markup cannot
     be. The decisive test is SYMMETRY — a dealer spread penalises the
     customer on BOTH sides (they buy high AND sell low). Drift or noise
     would not produce a consistent penalty in opposite directions. This is
     the control that separates "markup" from "the market moved".

  3. CORRELATED OBSERVATIONS. Bond-days within the same bond are not
     independent, so a t-stat computed over bond-days is badly overstated —
     the same error that turned 38 option names into 13.7 independent bets
     elsewhere in this project. Significance is therefore computed on
     per-BOND means, clustering at the bond.

  4. LIQUIDITY SELECTION. These 3,085 bonds were downloaded because they are
     the most liquid on the tape. Liquid bonds have tighter spreads, so any
     penalty found here UNDERSTATES what a typical retail investor faces in a
     thinly traded bond. Stated in the output so the number is never
     oversold.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

MUNIS = "/home/user/viki-m13/bonds/munis"
RETAIL_MAX = 100_000
INSTIT_MIN = 1_000_000
NEAR_MINUTES = 60          # for the intraday-drift control


def _bond_table(path: str) -> pd.DataFrame | None:
    """Par-weighted price per (date, side, size bucket) for one bond."""
    try:
        d = pd.read_csv(path)
    except Exception:
        return None
    if not len(d) or "side" not in d.columns:
        return None
    ts = pd.to_datetime(d.ts, errors="coerce")
    d = d.assign(date=ts.dt.date, t=ts)
    d = d[(d.price > 20) & (d.price < 250) & (d.par > 0)]
    d = d.dropna(subset=["date", "price", "par", "side"])
    if not len(d):
        return None
    d = d[d.side.isin(["S", "P"])]
    if not len(d):
        return None
    bucket = np.where(d.par <= RETAIL_MAX, "R",
                      np.where(d.par >= INSTIT_MIN, "I", ""))
    d = d.assign(bucket=bucket)
    d = d[d.bucket != ""]
    if not len(d):
        return None
    d["pw"] = d.price * d.par
    g = d.groupby(["date", "side", "bucket"], as_index=False).agg(
        pw=("pw", "sum"), par=("par", "sum"),
        ytw=("ytw", "mean"), tmin=("t", "min"), tmax=("t", "max"))
    g["px"] = g.pw / g.par
    return g


def _penalty_frame(g: pd.DataFrame, side: str) -> pd.DataFrame:
    """Retail-minus-institutional price for one side, per bond-day."""
    s = g[g.side == side]
    r = s[s.bucket == "R"].set_index("date")
    i = s[s.bucket == "I"].set_index("date")
    common = r.index.intersection(i.index)
    if not len(common):
        return pd.DataFrame()
    out = pd.DataFrame({
        "date": common,
        "retail_px": r.loc[common, "px"].to_numpy(),
        "instit_px": i.loc[common, "px"].to_numpy(),
        "retail_ytw": r.loc[common, "ytw"].to_numpy(),
        "instit_ytw": i.loc[common, "ytw"].to_numpy(),
        "r_tmin": r.loc[common, "tmin"].to_numpy(),
        "r_tmax": r.loc[common, "tmax"].to_numpy(),
        "i_tmin": i.loc[common, "tmin"].to_numpy(),
        "i_tmax": i.loc[common, "tmax"].to_numpy(),
    })
    # A customer BUYING (side S) is hurt by paying MORE: retail - instit.
    # A customer SELLING (side P) is hurt by receiving LESS: instit - retail.
    # Both are expressed so that POSITIVE always means "retail did worse".
    out["penalty"] = (out.retail_px - out.instit_px) if side == "S" \
        else (out.instit_px - out.retail_px)
    # Yield: a buyer is hurt by a LOWER yield, a seller by a HIGHER one.
    out["ytw_penalty"] = (out.instit_ytw - out.retail_ytw) if side == "S" \
        else (out.retail_ytw - out.instit_ytw)
    gap = (out.r_tmin - out.i_tmax).abs().dt.total_seconds() / 60.0
    gap2 = (out.i_tmin - out.r_tmax).abs().dt.total_seconds() / 60.0
    out["minutes_apart"] = np.minimum(gap, gap2)
    return out


def _summary(p: pd.Series, label: str, per_bond: pd.Series | None = None):
    if not len(p):
        print(f"  {label:<34} no data")
        return
    med, mean = p.median(), p.mean()
    line = (f"  {label:<34} n={len(p):>6,}  worse {100*(p>0).mean():>5.1f}%  "
            f"median {med:+.3f} pts (${med*1000:>7,.0f}/100k)  mean {mean:+.3f}")
    if per_bond is not None and len(per_bond) > 2:
        t = per_bond.mean() / (per_bond.std() / np.sqrt(len(per_bond)))
        line += f"  t={t:+.1f} (clustered, {len(per_bond):,} bonds)"
    print(line)


def main():
    files = sorted(glob.glob(os.path.join(MUNIS, "data", "trades", "*.csv.gz")))
    if not files:
        print("no trade files found")
        return 1
    print(f"scanning {len(files):,} bonds from the MSRB EMMA tape\n")

    buys, sells = [], []
    for n, f in enumerate(files, 1):
        g = _bond_table(f)
        if g is None:
            continue
        bond = os.path.basename(f)[:12]
        for side, sink in (("S", buys), ("P", sells)):
            fr = _penalty_frame(g, side)
            if len(fr):
                fr["bond"] = bond
                sink.append(fr)
        if n % 750 == 0:
            print(f"  …{n:,}/{len(files):,}", flush=True)

    B = pd.concat(buys, ignore_index=True) if buys else pd.DataFrame()
    S = pd.concat(sells, ignore_index=True) if sells else pd.DataFrame()

    print("\n" + "=" * 96)
    print("1. HEADLINE — retail vs institutional, same bond, same day")
    print("   (positive = retail got the worse deal)")
    print("=" * 96)
    for frame, lab in ((B, "BUYING (retail pays more)"),
                       (S, "SELLING (retail receives less)")):
        if len(frame):
            _summary(frame.penalty, lab, frame.groupby("bond").penalty.mean())

    print("\n" + "=" * 96)
    print("2. CONTROL — SYMMETRY. The test that separates markup from drift.")
    print("   A dealer spread hurts the customer on BOTH sides. Market drift")
    print("   cannot: it would help retail on one side and hurt on the other.")
    print("=" * 96)
    if len(B) and len(S):
        mb, ms = B.penalty.median(), S.penalty.median()
        print(f"  median penalty buying : {mb:+.3f} points")
        print(f"  median penalty selling: {ms:+.3f} points")
        if mb > 0 and ms > 0:
            print("  -> retail is penalised on BOTH sides. Consistent with a")
            print("     dealer spread; NOT explicable by the market drifting.")
        else:
            print("  -> penalty flips sign between sides. That is what DRIFT")
            print("     looks like, not a markup. The thesis would fail here.")
        both = pd.concat([B.assign(side="buy"), S.assign(side="sell")])
        pb = both.groupby("bond").penalty.mean()
        print(f"  bonds where the average retail trade was worse on the day: "
              f"{100*(pb>0).mean():.1f}% of {len(pb):,}")

    print("\n" + "=" * 96)
    print(f"3. CONTROL — INTRADAY DRIFT. Restricted to pairs traded within")
    print(f"   {NEAR_MINUTES} minutes of each other, where drift has no room.")
    print("=" * 96)
    for frame, lab in ((B, "BUYING, close in time"), (S, "SELLING, close in time")):
        if len(frame):
            near = frame[frame.minutes_apart <= NEAR_MINUTES]
            if len(near):
                _summary(near.penalty, lab, near.groupby("bond").penalty.mean())
            else:
                print(f"  {lab:<34} no pairs within the window")

    print("\n" + "=" * 96)
    print("4. YIELD CHECK — price could mislead; yield cannot.")
    print("=" * 96)
    for frame, lab in ((B, "BUYING"), (S, "SELLING")):
        if len(frame):
            y = frame.ytw_penalty.dropna()
            if len(y):
                print(f"  {lab:<8} retail gave up {y.median()*100:+.1f} bp of yield "
                      f"(median), worse on {100*(y>0).mean():.1f}% of bond-days, "
                      f"n={len(y):,}")

    print("\n" + "=" * 96)
    print("5. WHAT IT COSTS A CLIENT")
    print("=" * 96)
    if len(B):
        med = B.penalty.median()
        print(f"  A $2,000,000 muni ladder bought in retail-size lots, at the")
        print(f"  median observed penalty, costs ${med/100*2_000_000:,.0f} more than the")
        print(f"  same bonds bought institutionally that same day.")
        if len(S):
            rt = med + S.penalty.median()
            print(f"  Buying and later selling at the median on both sides: "
                  f"${rt/100*2_000_000:,.0f}.")

    print("\n" + "=" * 96)
    print("BIAS NOTES — read before quoting any of this")
    print("=" * 96)
    print("  * These are the MOST LIQUID bonds on the tape. Liquid bonds trade")
    print("    tighter, so this UNDERSTATES the typical retail experience.")
    print("  * Requires both a retail and an institutional print on the same")
    print("    bond-day, which selects toward actively traded days.")
    print("  * Significance is clustered by bond; the bond-day count is NOT")
    print("    the sample size and must not be quoted as one.")
    print("  * EMMA prints are anonymous: this measures what retail-SIZED")
    print("    trades paid, not what any named firm's clients paid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
