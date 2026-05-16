"""Downside lever that is actually evidence-backed: a drawdown-conditional
rotation between the v5 picker and the already-validated market-neutral
sleeve (rho~0 to v5, WF-min Sharpe 1.00 per Phase B).

This is NOT a new alpha. It is an honest portfolio-state switch between
two streams the repo already validated. Rule is a-priori:

  - DCA normally into v5.
  - When the DCA portfolio's drawdown from its running peak breaches
    -TH, route the whole book + new contributions into the MN sleeve.
  - Switch back to v5 once the portfolio recovers to within -TH/2 of
    its peak (hysteresis -> avoids whipsaw).

TH in {0.20, 0.25, 0.30} reported as SENSITIVITY (a plateau check),
not optimized. Compared against v5-only and the static 60/40 blend on
the investor-experienced metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dca_investor_eval as dca  # noqa: E402

AUG = HERE.parents[1] / "cache" / "v2" / "sp500_pit" / "augmented"


def dca_switch(v5, mn, spy, th):
    """Returns (value_path, basis_path, months_in_mn)."""
    v = 0.0
    in_mn = False
    peak = 0.0
    vals, basis, mn_months = [], [], 0
    for t in range(len(v5)):
        v += 1.0
        r = mn[t] if in_mn else v5[t]
        v *= (1.0 + r)
        peak = max(peak, v)
        dd = v / peak - 1.0
        if not in_mn and dd <= -th:
            in_mn = True
        elif in_mn and dd >= -th / 2.0:
            in_mn = False
        mn_months += in_mn
        vals.append(v)
        basis.append(t + 1.0)
    return np.array(vals), np.array(basis), mn_months


def rolling(series_fn, spy, H):
    n = len(spy)
    wins, moic = [], []
    worst = 1e9
    for s in range(0, n - H + 1):
        tv = series_fn(s, H)
        sv = dca.dca_path(spy[s:s + H])[0][-1]
        m = tv / H
        wins.append(tv > sv)
        moic.append(m)
        worst = min(worst, m)
    return (round(float(np.mean(wins)), 4),
            round(float(np.median(moic)), 3),
            round(float(worst), 3))


def main():
    df = dca.load_streams()
    v5 = df["v5"].to_numpy()
    mn = df["mn"].to_numpy()
    spy = df["SPY"].to_numpy()
    blend = df["blend60_40"].to_numpy()

    def make(series):
        return lambda s, H: dca.dca_path(series[s:s + H])[0][-1]

    out = {"horizons": {}, "full_history": {}}
    variants = {
        "v5_only": ("series", v5),
        "static_60_40": ("series", blend),
        "mn_only": ("series", mn),
        "switch_TH20": ("switch", 0.20),
        "switch_TH25": ("switch", 0.25),
        "switch_TH30": ("switch", 0.30),
    }

    for H in (12, 36, 60, 120):
        row = {}
        for name, (kind, val) in variants.items():
            if kind == "series":
                fn = make(val)
            else:
                fn = lambda s, HH, th=val: dca_switch(
                    v5[s:s + HH], mn[s:s + HH], spy[s:s + HH], th)[0][-1]
            row[name] = dict(zip(("win_vs_spy", "median_moic", "min_moic"),
                                 rolling(fn, spy, H)))
        out["horizons"][f"H{H}"] = row

    for name, (kind, val) in variants.items():
        if kind == "series":
            vv, bb = dca.dca_path(val)
            mnm = None
        else:
            vv, bb, mnm = dca_switch(v5, mn, spy, val)
        pk = np.maximum.accumulate(vv)
        out["full_history"][name] = {
            "moic": round(float(vv[-1] / bb[-1]), 2),
            "max_dd": round(float(((vv - pk) / pk).min()), 4),
            "irr": round(float(dca.irr_from_terminal(vv[-1], len(vv))), 4),
            "months_in_mn": (int(mnm) if mnm is not None else None),
        }

    (AUG / "novel_v6_mn_overlay.json").write_text(json.dumps(out, indent=2))

    print("FULL-HISTORY DCA (2003-2026):")
    print(f"{'variant':<16}{'MOIC':>9}{'IRR':>8}{'maxDD':>9}{'mo_in_MN':>10}")
    for k, v in out["full_history"].items():
        print(f"{k:<16}{v['moic']:>8.1f}x{v['irr']*100:>7.1f}%{v['max_dd']*100:>8.1f}%"
              f"{str(v['months_in_mn']):>10}")
    print("\nROLLING win vs SPY-DCA / median MOIC / MIN MOIC:")
    for H in (12, 36, 60, 120):
        print(f" H{H}:")
        for k, v in out["horizons"][f"H{H}"].items():
            print(f"   {k:<15} win={v['win_vs_spy']*100:5.1f}%  "
                  f"med={v['median_moic']:6.2f}x  min={v['min_moic']:5.2f}x")


if __name__ == "__main__":
    main()
