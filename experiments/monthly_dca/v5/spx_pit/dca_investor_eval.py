"""DCA-investor evaluation of the deployed v5 picker.

Every other metric in this repo is lump-sum (CAGR / Sharpe / walk-forward).
None of them describe what a person who contributes a fixed amount every
month-end actually experiences. This script measures exactly that, on the
PIT-correct return streams already audited in the repo:

  - v5  : deployed K=2 rule-based picker  (augmented/v5_winner_equity.csv)
  - mn  : v5 market-neutral sleeve        (augmented/v5_mn_sleeve_returns.csv)
  - 60/40: 0.6*v5 + 0.4*SPY monthly blend (repo's flagged risk-reduction)
  - SPY : PIT benchmark                   (monthly_returns_clean.parquet)

For a DCA investor the relevant "hit rate" is NOT the monthly win rate.
It is: over a rolling H-month accumulation, did identical monthly
contributions into the strategy end up worth more than into SPY?
That is what this computes, with no parameter tuning of any kind.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

AUG = Path(__file__).resolve().parents[2] / "cache" / "v2" / "sp500_pit" / "augmented"
WEBAPP_DATA = Path(__file__).resolve().parents[3] / "docs" / "monthly-dca" / "data.json"
HORIZONS = [12, 24, 36, 60, 120]   # months: 1y, 2y, 3y, 5y, 10y


def load_streams():
    """CANONICAL stream loader.

    NOTE (2026-05-16 data-integrity fix): `v5_winner_equity.csv` is a
    STALE/CORRUPTED artifact — its returns from 2023-04 onward are
    inflated to an implausible ~85%/yr (e.g. 2023-05 shows +40% in one
    month). The trustworthy v5 stream is the live production sim
    (`run_full_sim` -> rets_log) that the website is built from; it is
    exported as `dca_investor.growth[*].r` in the deployed data.json.
    We read THAT here so research == website. SPY = same data.json `s`.
    """
    djson = json.loads((WEBAPP_DATA).read_text())
    g = pd.DataFrame(djson["dca_investor"]["growth"])
    g["p"] = pd.to_datetime(g["date"]).dt.to_period("M")
    v5 = g.set_index("p")["r"].astype(float)
    spy = g.set_index("p")["s"].astype(float)

    mn = pd.read_csv(AUG / "v5_mn_sleeve_returns.csv", index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    mn.index = mn.index.to_period("M")

    idx = v5.index
    spy = spy.reindex(idx).fillna(0.0)
    mn = mn.reindex(idx).fillna(0.0)
    blend = 0.6 * v5 + 0.4 * spy
    df = pd.DataFrame({"v5": v5, "mn": mn, "blend60_40": blend, "SPY": spy})
    # crash mask not available from data.json growth; not needed by the
    # canonical analyses (shield variant is legacy).
    df.attrs["crash_mask"] = (v5.values == 0.0)
    return df


def dca_path_shield(v5r, spyr, crash, contrib: float = 1.0):
    """Stretch variant: a contribution made in a v5 'crash'-regime month is
    parked in SPY and stays there; every other contribution goes to the v5
    picker. Uses only the regime label already in the audited stream — no
    new alpha, no look-ahead (regime is trailing-SPY based, repo-audited)."""
    pot_v5, pot_spy = 0.0, 0.0
    vals, basis = [], []
    for t in range(len(v5r)):
        if crash[t]:
            pot_spy += contrib
        else:
            pot_v5 += contrib
        pot_v5 *= (1.0 + v5r[t])
        pot_spy *= (1.0 + spyr[t])
        vals.append(pot_v5 + pot_spy)
        basis.append(contrib * (t + 1))
    return np.array(vals), np.array(basis)


def dca_path(rets: np.ndarray, contrib: float = 1.0):
    """Contribute `contrib` at the start of each month, then earn that
    month's return. Returns (value_series, cost_basis_series)."""
    v = 0.0
    vals, basis = [], []
    for t, r in enumerate(rets):
        v += contrib
        v *= (1.0 + r)
        vals.append(v)
        basis.append(contrib * (t + 1))
    return np.array(vals), np.array(basis)


def irr_from_terminal(terminal: float, H: int, contrib: float = 1.0):
    """Annualized money-weighted IRR for a DCA schedule that paid `contrib`
    at the start of months 0..H-1 and is worth `terminal` at end of H-1."""

    def npv(i):
        # outflows at t=0..H-1, inflow `terminal` at t=H-1 (start-of-month
        # contributions, last one earns month H-1's return -> received at H-1).
        out = sum(contrib / (1.0 + i) ** t for t in range(H))
        return terminal / (1.0 + i) ** (H - 1) - out

    lo, hi = -0.5, 0.5
    flo = npv(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = npv(mid)
        if abs(fm) < 1e-10:
            break
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return (1.0 + mid) ** 12 - 1.0


def money_weighted_irr(rets: np.ndarray, contrib: float = 1.0):
    term, _ = dca_path(rets, contrib)
    return irr_from_terminal(term[-1], len(rets), contrib)


def rolling_eval(df: pd.DataFrame):
    out = {}
    crash_all = df.attrs["crash_mask"]
    for H in HORIZONS:
        rows = []
        n = len(df)
        for s in range(0, n - H + 1):
            win = df.iloc[s:s + H]
            rec = {"start": str(win.index[0]), "end": str(win.index[-1])}
            spy_term = dca_path(win["SPY"].to_numpy())[0][-1]
            for col in df.columns:
                term, basis = dca_path(win[col].to_numpy())
                rec[f"{col}_moic"] = term[-1] / basis[-1]          # x money-in
                rec[f"{col}_term"] = term[-1]
                rec[f"{col}_irr"] = money_weighted_irr(win[col].to_numpy())
                if col != "SPY":
                    rec[f"{col}_vs_spy"] = term[-1] / spy_term      # >1 = beat SPY
            sh, shb = dca_path_shield(win["v5"].to_numpy(), win["SPY"].to_numpy(),
                                      crash_all[s:s + H])
            rec["v5_shield_moic"] = sh[-1] / shb[-1]
            rec["v5_shield_term"] = sh[-1]
            rec["v5_shield_vs_spy"] = sh[-1] / spy_term
            rec["v5_shield_irr"] = irr_from_terminal(sh[-1], H)
            rows.append(rec)
        r = pd.DataFrame(rows)
        summ = {"horizon_months": H, "n_windows": len(r)}
        for col in list(df.columns) + ["v5_shield"]:
            if col == "SPY":
                continue
            beat = r[f"{col}_vs_spy"] > 1.0
            summ[col] = {
                "win_rate_vs_spy": round(float(beat.mean()), 4),
                "n_windows": int(len(r)),
                "median_moic": round(float(r[f"{col}_moic"].median()), 3),
                "p05_moic": round(float(r[f"{col}_moic"].quantile(0.05)), 3),
                "min_moic": round(float(r[f"{col}_moic"].min()), 3),
                "median_irr": round(float(r[f"{col}_irr"].median()), 4),
                "p05_irr": round(float(r[f"{col}_irr"].quantile(0.05)), 4),
                "min_irr": round(float(r[f"{col}_irr"].min()), 4),
                "median_vs_spy": round(float(r[f"{col}_vs_spy"].median()), 3),
                "min_vs_spy": round(float(r[f"{col}_vs_spy"].min()), 3),
                "worst_window": {
                    "start": r.loc[r[f"{col}_vs_spy"].idxmin(), "start"],
                    "end": r.loc[r[f"{col}_vs_spy"].idxmin(), "end"],
                    f"{col}_moic": round(float(r.loc[r[f"{col}_vs_spy"].idxmin(), f"{col}_moic"]), 3),
                    "spy_moic": round(float(r.loc[r[f"{col}_vs_spy"].idxmin(), "SPY_moic"]), 3),
                },
            }
        summ["spy_median_moic"] = round(float(r["SPY_moic"].median()), 3)
        out[f"H{H}"] = summ
        r.to_csv(AUG / f"dca_rolling_H{H}.csv", index=False)
    return out


def full_history(df: pd.DataFrame):
    out = {}
    crash = df.attrs["crash_mask"]
    for col in list(df.columns) + ["v5_shield"]:
        if col == "v5_shield":
            val, basis = dca_path_shield(df["v5"].to_numpy(),
                                         df["SPY"].to_numpy(), crash)
            irr = irr_from_terminal(val[-1], len(val))
        else:
            rets = df[col].to_numpy()
            val, basis = dca_path(rets)
            irr = money_weighted_irr(rets)
        peak = np.maximum.accumulate(val)
        dd = (val - peak) / peak                       # value peak-to-trough
        underwater = (val - basis) / basis             # vs money contributed
        out[col] = {
            "months": int(len(rets)),
            "total_contributed": float(basis[-1]),
            "terminal_value": round(float(val[-1]), 1),
            "terminal_moic": round(float(val[-1] / basis[-1]), 2),
            "money_weighted_irr": round(float(irr), 4),
            "max_value_drawdown": round(float(dd.min()), 4),
            "worst_underwater_vs_contrib": round(float(underwater.min()), 4),
            "months_underwater_vs_contrib": int((underwater < 0).sum()),
        }
    return out


def main():
    df = load_streams()
    result = {
        "window": f"{df.index[0]} .. {df.index[-1]}",
        "n_months": len(df),
        "convention": "contribute 1 unit at start of each month, earn that "
                      "month's return; identical schedule for every stream; "
                      "v5 stream is net of 10bps costs (repo canonical sim)",
        "full_history_dca": full_history(df),
        "rolling": rolling_eval(df),
    }
    (AUG / "dca_investor_eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
