"""Per-name selectivity plateaus (see explore_selectivity.py). The two levers
that actually cut how often you're below your cost basis are:

  1. DIVERSIFY the picks into an equal-weight BASKET — idiosyncratic dips
     cancel, so the basket's value sits below its cost far less often than
     its average constituent does.
  2. REGIME-TIME the entry — only buy when SPY is above its 200-day SMA, to
     avoid buying right before broad market drawdowns (the main reason a buy
     ends up underwater).

We measure underwater at the PORTFOLIO level: each month form the top-K
FloorScore basket, equal weight, buy-and-hold, and track the basket value
vs its cost (=1.0). Honest delisting: a name that stops trading -> 0.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from floor_lib import build, ROOT

PIT = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit"
HZD = {"3m": 63, "12m": 252}


def z(s):
    s = s.astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 1e-12 else s * 0.0


def floor_score(g):
    return (z(g["gbm_maxdd_3m"]) - z(g["gbm_uw_frac_3m"]) - z(g["vol_3m_xs"])
            + 0.5 * z(g["trend_health_5y_xs"]) + 0.5 * z(g["chr_trough_q30_3m"]))


def main():
    df = build().copy()
    df["fs"] = df.groupby("asof", group_keys=False).apply(
        lambda g: floor_score(g), include_groups=False)
    df = df.sort_values(["asof", "fs"], ascending=[True, False])

    daily = pd.read_parquet(PIT / "prices_extended_pit.parquet").sort_index()
    gdates = daily.index.values
    spy = daily["SPY"]
    spy_sma200 = spy.rolling(200).mean()

    # cache delisting-aware price arrays per ticker
    ev_cache, last_pos_cache = {}, {}
    def ev_of(tk):
        if tk not in ev_cache:
            s = daily[tk]
            valid = s.notna().values
            arr = s.values.astype(np.float64)
            lp = np.where(valid)[0][-1] if valid.any() else -1
            if lp >= 0 and lp < len(arr) - 1:
                arr[lp + 1:] = 0.0          # delisted -> wiped out
            ev_cache[tk] = pd.Series(arr).ffill().values
            last_pos_cache[tk] = lp
        return ev_cache[tk]

    def basket_stats(picks_df, K, regime=False):
        recs = {h: [] for h in HZD}
        months_used = set()
        for asof, g in picks_df.groupby("asof"):
            gpos = np.searchsorted(gdates, np.datetime64(asof), side="right") - 1
            if gpos < 0:
                continue
            if regime:
                sv, smav = spy.values[gpos], spy_sma200.values[gpos]
                if not (np.isfinite(smav) and sv > smav):
                    continue                 # market not in uptrend -> sit out
            names = list(g.head(K)["ticker"])
            for h, H in HZD.items():
                if gpos + H >= len(gdates):
                    continue
                paths = []
                for tk in names:
                    ev = ev_of(tk)
                    entry = ev[gpos]
                    if not np.isfinite(entry) or entry <= 0:
                        continue
                    paths.append(ev[gpos + 1: gpos + H + 1] / entry)
                if not paths:
                    continue
                val = np.mean(np.vstack(paths), axis=0)   # equal-weight basket value
                below = val < 1.0
                recs[h].append(dict(uw=below.mean(), ever=int(below.any()),
                                    endb=int(val[-1] < 1.0), dd=val.min() - 1.0,
                                    ret=val[-1] - 1.0))
            months_used.add(asof)
        out = {}
        n_all = picks_df["asof"].nunique()
        for h in HZD:
            r = pd.DataFrame(recs[h])
            if len(r) == 0:
                continue
            out[h] = dict(n=len(r), mo=len(months_used), mo_pct=len(r) / n_all * 100,
                          uw=r["uw"].mean(), ever=r["ever"].mean(), endb=r["endb"].mean(),
                          dd=r["dd"].mean(), ret=r["ret"].mean(),
                          safe=(r["uw"] < 0.15).mean(), never=(r["ever"] == 0).mean())
        return out

    hdr = (f"{'basket':<26}{'hz':>4}{'n':>6}{'mo%':>6}{'uw':>8}{'ever<':>8}"
           f"{'end<':>8}{'maxdd':>8}{'safe%':>7}{'never%':>8}{'meanret':>8}")
    print("PORTFOLIO-level underwater (basket value vs its cost basis)\n")
    print(hdr)

    def show(tag, st):
        for h in HZD:
            if h not in st:
                continue
            s = st[h]
            print(f"{tag:<26}{h:>4}{s['n']:>6}{s['mo_pct']:>5.0f}%{s['uw']:>8.3f}"
                  f"{s['ever']:>8.3f}{s['endb']:>8.3f}{s['dd']:>8.3f}"
                  f"{s['safe']*100:>6.1f}%{s['never']*100:>7.1f}%{s['ret']:>8.3f}")

    for K in (10, 20, 30):
        show(f"FloorScore top{K}", basket_stats(df, K))
    print("  --- with regime gate (buy only when SPY > 200d SMA) ---")
    for K in (10, 20, 30):
        show(f"FloorScore top{K} +regime", basket_stats(df, K, regime=True))

    # SPY benchmark (single asset = its own basket)
    print("  --- benchmark ---")
    spy_only = df.drop_duplicates("asof").assign(ticker="SPY")
    show("SPY (buy & hold)", basket_stats(spy_only, 1))
    show("SPY +regime", basket_stats(spy_only, 1, regime=True))

    import json
    from floor_lib import HERE
    results = {
        "floor_top10": basket_stats(df, 10),
        "floor_top20": basket_stats(df, 20),
        "floor_top10_regime": basket_stats(df, 10, regime=True),
        "floor_top20_regime": basket_stats(df, 20, regime=True),
        "spy": basket_stats(spy_only, 1),
        "spy_regime": basket_stats(spy_only, 1, regime=True),
    }
    (HERE / "floor_portfolio_results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {HERE/'floor_portfolio_results.json'}")


if __name__ == "__main__":
    main()
