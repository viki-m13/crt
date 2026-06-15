"""Build forward DOWNSIDE labels for every (asof, ticker) in the PIT panel.

The Floor picker's objective is *not* total return. It is: after you buy,
how often / how deep does the position sit BELOW the purchase price.
So for each entry we compute, over forward horizons H in trading days,
path statistics relative to the entry (purchase) price:

  uw_frac_H   fraction of the next H trading days whose CLOSE < entry price
              (the headline "how often is it underwater" metric)
  maxdd_H     worst close/entry - 1 over the window (deepest dip below buy)
  ever_below_H 1 if the close was ever strictly below entry in the window
  end_below_H 1 if the close at +H is below entry
  end_ret_H   close[+H]/entry - 1
  censored_H  True if we do not yet have H trading days of market data after
              asof (right edge of the sample) -> that horizon is excluded.

Delisting is treated honestly (survivorship): once a ticker stops trading
inside a window (its own series ends before the global last date), the
remaining days are filled at price 0 -> counted as fully underwater
(-100%). This is the same philosophy as the repo's forward-return engine.

Inputs (all cached, offline):
  cache/v2/sp500_pit/augmented/sp500_pit_panel.parquet   (asof, ticker grid)
  cache/v2/sp500_pit/prices_extended_pit.parquet          (daily closes)

Output:
  experiments/downside_floor/downside_labels.parquet
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PIT = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit"
AUG = PIT / "augmented"
OUT = Path(__file__).resolve().parent / "downside_labels.parquet"

HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}


def main():
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet", columns=["asof", "ticker"])
    panel["asof"] = pd.to_datetime(panel["asof"])
    daily = pd.read_parquet(PIT / "prices_extended_pit.parquet").sort_index()
    gdates = daily.index.values  # global trading calendar (datetime64)
    n_g = len(gdates)
    global_last = gdates[-1]
    maxH = max(HORIZONS.values())

    # group panel asofs per ticker
    by_tkr = {tk: sub["asof"].values for tk, sub in panel.groupby("ticker", sort=False)}

    rows = []
    skipped = 0
    for tk, asof_arr in by_tkr.items():
        if tk not in daily.columns:
            skipped += len(asof_arr)
            continue
        s = daily[tk]
        valid = s.notna().values
        if not valid.any():
            skipped += len(asof_arr)
            continue
        px = s.values.astype(np.float64)
        last_valid_pos = np.where(valid)[0][-1]
        ticker_last = gdates[last_valid_pos]
        delisted = ticker_last < global_last  # stopped trading before sample end

        # Build an evaluation price array aligned to the global calendar:
        #   - real close where it traded
        #   - 0.0 for days AFTER the ticker stopped trading but still inside
        #     the global sample (delisting -> underwater)
        #   - NaN before the ticker's first trade
        ev = px.copy()
        if delisted:
            ev[last_valid_pos + 1:] = 0.0  # everything after delist = wiped out
        # forward-fill rare interior gaps so a NaN day doesn't break the window
        ev_s = pd.Series(ev).ffill().values

        # entry positions: last trading day on/before each asof
        epos = np.searchsorted(gdates, asof_arr, side="right") - 1
        for k, e in enumerate(epos):
            if e < 0 or not valid[e]:
                continue
            entry = px[e]
            if not np.isfinite(entry) or entry <= 0:
                continue
            rec = {"asof": asof_arr[k], "ticker": tk, "entry_px": entry}
            for name, H in HORIZONS.items():
                end = e + H
                if end >= n_g:
                    # not enough global market data yet -> right-censored
                    rec[f"censored_{name}"] = True
                    for f in ("uw_frac", "maxdd", "ever_below", "end_below", "end_ret"):
                        rec[f"{f}_{name}"] = np.nan
                    continue
                win = ev_s[e + 1: end + 1]
                below = win < entry
                rec[f"uw_frac_{name}"] = float(below.mean())
                rec[f"maxdd_{name}"] = float(win.min() / entry - 1.0)
                rec[f"ever_below_{name}"] = int(below.any())
                rec[f"end_below_{name}"] = int(win[-1] < entry)
                rec[f"end_ret_{name}"] = float(win[-1] / entry - 1.0)
                rec[f"censored_{name}"] = False
            rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["asof", "ticker"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {out.shape} -> {OUT}  (skipped {skipped} no-price rows)")
    # quick base-rate sanity print
    for name in HORIZONS:
        m = out[~out[f"censored_{name}"]]
        print(f"  {name:>3}: n={len(m):>6}  mean uw_frac={m[f'uw_frac_{name}'].mean():.3f}  "
              f"P(ever_below)={m[f'ever_below_{name}'].mean():.3f}  "
              f"P(end_below)={m[f'end_below_{name}'].mean():.3f}  "
              f"mean maxdd={m[f'maxdd_{name}'].mean():.3f}")


if __name__ == "__main__":
    main()
