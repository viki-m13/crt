"""How much does being MORE SELECTIVE buy us on downside?

Two selectivity knobs, evaluated on the cached scored frame (read-only):

  A) concentration   — take the top-K by FloorScore each month, K in {1,3,5,10,20}
  B) conviction gate — only buy names that clear strict, absolute downside
                        bars (so some months you buy fewer names, or nothing)

We report the safety/coverage frontier: as you get pickier, time-underwater
and end-below fall and the "basically/never underwater" rates rise, but the
number of buys and the share of months with any buy fall too.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from floor_lib import build, HERE

H = "3m"   # headline horizon for the sweep (also print 12m)


def z(s):
    s = s.astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 1e-12 else s * 0.0


def floor_score(g):
    return (z(g["gbm_maxdd_3m"]) - z(g["gbm_uw_frac_3m"]) - z(g["vol_3m_xs"])
            + 0.5 * z(g["trend_health_5y_xs"]) + 0.5 * z(g["chr_trough_q30_3m"]))


def realized(rows, h):
    m = rows[~rows[f"censored_{h}"]]
    if len(m) == 0:
        return None
    return dict(
        n=len(m),
        uw=m[f"uw_frac_{h}"].mean(),
        ever=m[f"ever_below_{h}"].mean(),
        endb=m[f"end_below_{h}"].mean(),
        dd=m[f"maxdd_{h}"].mean(),
        ret=m[f"end_ret_{h}"].mean(),
        medret=m[f"end_ret_{h}"].median(),
        safe=(m[f"uw_frac_{h}"] < 0.15).mean(),     # rarely underwater
        vsafe=(m[f"uw_frac_{h}"] < 0.05).mean(),    # almost never underwater
        never=(m[f"ever_below_{h}"] == 0).mean(),   # literally never dipped below buy
    )


def main():
    df = build().copy()
    df["fs"] = df.groupby("asof", group_keys=False).apply(
        lambda g: floor_score(g), include_groups=False)
    df = df.sort_values(["asof", "fs"], ascending=[True, False])
    n_months = df["asof"].nunique()

    def topk(frame, k):
        return frame.groupby("asof", group_keys=False).head(k)

    def show(tag, picks):
        cov_months = picks["asof"].nunique()
        for h in (H, "12m"):
            s = realized(picks, h)
            if s is None:
                continue
            print(f"{tag:<22}{h:>4}{s['n']:>7}{s['n']/n_months:>7.1f}"
                  f"{cov_months/n_months*100:>7.0f}%{s['uw']:>8.3f}{s['ever']:>8.3f}"
                  f"{s['endb']:>8.3f}{s['dd']:>8.3f}{s['safe']*100:>7.1f}"
                  f"{s['vsafe']*100:>7.1f}{s['never']*100:>7.1f}{s['medret']:>8.3f}")

    hdr = (f"{'selector':<22}{'hz':>4}{'nbuy':>7}{'/mo':>7}{'mo%':>7}"
           f"{'uw':>8}{'ever<':>8}{'end<':>8}{'maxdd':>8}"
           f"{'safe%':>7}{'vsf%':>7}{'nvr%':>7}{'medret':>8}")

    # ---- A) concentration: top-K by FloorScore ----
    print("=== A) concentration (top-K by FloorScore each month) ===")
    print(hdr)
    for K in (20, 10, 5, 3, 1):
        show(f"top{K}", topk(df, K))
    show("universe", df)

    # ---- B) conviction gate: absolute downside bars, take all survivors ----
    print("\n=== B) conviction gate (buy ALL names clearing strict bars) ===")
    print(hdr)
    gates = {
        "loose":  dict(uw=0.45, dd=-0.10, vol=0.7, drift=-0.02),
        "medium": dict(uw=0.42, dd=-0.07, vol=0.3, drift=0.00),
        "strict": dict(uw=0.40, dd=-0.05, vol=0.0, drift=0.00),
        "ultra":  dict(uw=0.38, dd=-0.045, vol=-0.3, drift=0.01),
    }
    for name, gt in gates.items():
        m = ((df["gbm_uw_frac_3m"] <= gt["uw"]) & (df["gbm_maxdd_3m"] >= gt["dd"])
             & (df["vol_3m_xs"] <= gt["vol"]) & (df["chr_p50_end_3m"] >= gt["drift"]))
        show(f"gate:{name}", df[m])

    # ---- C) gate + cap to best 3 per month (selective AND concentrated) ----
    print("\n=== C) strict gate, then top-3 FloorScore among survivors ===")
    print(hdr)
    for name in ("medium", "strict", "ultra"):
        gt = gates[name]
        m = ((df["gbm_uw_frac_3m"] <= gt["uw"]) & (df["gbm_maxdd_3m"] >= gt["dd"])
             & (df["vol_3m_xs"] <= gt["vol"]) & (df["chr_p50_end_3m"] >= gt["drift"]))
        show(f"gate:{name}+top3", topk(df[m], 3))


if __name__ == "__main__":
    main()
