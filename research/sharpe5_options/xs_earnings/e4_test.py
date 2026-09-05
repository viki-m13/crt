#!/usr/bin/env python3
"""THE test: does a cross-sectional earnings-vol signal predict the market's
ERROR (implied - realized), and does it survive real bid-ask?

Follows the pre-registration in
money-poly-bot/v19-btc-predictor/docs/EARNINGS_XS_TEST.md §0.

Note on the exit convention (data-driven, documented in §2.3 of that doc): the
DoltHub panel carries only 3 expiries per date and rotates which ones, so the
identical contract is quoted again after the event on only ~36% of events. The
full-sample tradable target is therefore the front straddle SOLD at the bid
before the event and SETTLED AT INTRINSIC at expiry (no exit spread at all,
which can only flatter the strategy). The quote-to-quote round trip is reported
on the subsample where it exists.

  python3 e4_test.py                 # earnings
  python3 e4_test.py --placebo       # P1: non-earnings dates
  python3 e4_test.py --second        # P2: 2nd-largest move of the quarter
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
BARS = "/home/user/bonds/data/intraday_daily"
EXTRA = os.path.join(HERE, "extra_bars")
MAXGAP = 4
WARMUP = 4
MINNAMES = 6
SQ2PI = np.sqrt(2.0 / np.pi)      # E|X| = sqrt(2/pi) * sigma for X ~ N(0, sigma)

PLACEBO = "--placebo" in sys.argv
SHIFT2ND = "--second" in sys.argv
TAG = "P1_nonearnings" if PLACEBO else ("P2_2ndmove" if SHIFT2ND else "EARNINGS")


def load_bars():
    out = {}
    src = [(BARS, f) for f in sorted(os.listdir(BARS))]
    if os.path.isdir(EXTRA):
        src += [(EXTRA, f) for f in sorted(os.listdir(EXTRA))]
    for root, f in src:
        if f.endswith(".csv"):
            d = pd.read_csv(os.path.join(root, f))
            d["date"] = pd.to_datetime(d["ts"]).dt.tz_localize(None).dt.normalize()
            out[f[:-4]] = d[["date", "close"]].dropna().drop_duplicates("date") \
                           .sort_values("date").reset_index(drop=True)
    return out


def event_strip(iv1, t1, iv2, t2):
    """Two-expiry event-vol strip: both expiries contain the event and differ
    only by diffusive time, so excess front variance is the event jump."""
    den = t2 - t1
    j2 = np.where(den > 1e-9, (iv1 ** 2 - iv2 ** 2) * t1 * t2 / np.where(den > 1e-9, den, 1), np.nan)
    return np.where(j2 > 0, np.sqrt(np.abs(j2)), np.nan)


def window_ic(df, sig, tgt):
    ics, ns = [], []
    for _, g in df.groupby("window"):
        g = g[[sig, tgt]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(g) < MINNAMES or g[sig].nunique() < 3 or g[tgt].nunique() < 3:
            continue
        r = stats.spearmanr(g[sig], g[tgt]).statistic
        if np.isfinite(r):
            ics.append(r); ns.append(len(g))
    ics = np.asarray(ics)
    if len(ics) < 5:
        return None
    t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))
    rng = np.random.default_rng(7)
    bs = np.array([rng.choice(ics, len(ics), replace=True).mean() for _ in range(2000)])
    return dict(ic=float(ics.mean()), t=float(t), nwin=int(len(ics)), nobs=int(sum(ns)),
                lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))


def build():
    ev = pd.read_parquet(os.path.join(HERE, "events.parquet"))
    atm = pd.read_parquet(os.path.join(HERE, "atm.parquet"))
    pan = pd.read_parquet(os.path.join(HERE, "panel.parquet"))
    bars = load_bars()
    cdates = np.sort(atm.date.unique())

    if PLACEBO:
        # P1: shift the WHOLE calendar by a fixed 35 days so the cross-sectional
        # clustering (and hence window density) is preserved, then drop anything
        # that lands within 10 days of a real earnings date for that name.
        real = {}
        for sy, d in zip(ev.symbol, ev.event_session):
            real.setdefault(sy, []).append(pd.Timestamp(d))
        keep = []
        for _, r in ev.iterrows():
            b = bars[r.symbol]
            d = pd.Timestamp(r.event_session) + pd.Timedelta(days=35)
            if min(abs((d - t).days) for t in real[r.symbol]) <= 10:
                keep.append(pd.NaT); continue
            i = int(np.searchsorted(b.date.values, np.datetime64(d, "ns"), "left"))
            keep.append(pd.Timestamp(b.date.values[i]) if 0 < i < len(b) else pd.NaT)
        ev["event_session"] = keep
        ev = ev.dropna(subset=["event_session"])

    if SHIFT2ND:
        keep = []
        for _, r in ev.iterrows():
            b = bars[r.symbol]
            w = b[(b.date >= r.event_session - pd.Timedelta(days=45)) &
                  (b.date <= r.event_session + pd.Timedelta(days=45))].copy()
            w["lr"] = np.log(w.close).diff()
            w = w.dropna()
            w = w[w.date != r.event_session]
            keep.append(pd.Timestamp(w.loc[w.lr.abs().idxmax(), "date"]) if len(w) else pd.NaT)
        ev["event_session"] = keep
        ev = ev.dropna(subset=["event_session"])

    # realized move for whatever session is in play
    rr = []
    for _, r in ev.iterrows():
        b = bars[r.symbol]
        i = int(np.searchsorted(b.date.values, np.datetime64(r.event_session, "ns"), "left"))
        rr.append(np.log(b.close.values[i] / b.close.values[i - 1]) if 0 < i < len(b) else np.nan)
    ev["r_event"] = rr
    ev["abs_r"] = ev.r_event.abs()
    ev = ev.dropna(subset=["abs_r"])

    # entry chain date = last snapshot strictly before the event session
    ie = np.searchsorted(cdates, ev.event_session.values.astype("datetime64[ns]"), "left")
    ev["entry"] = [cdates[i - 1] if i > 0 else np.datetime64("NaT") for i in ie]
    ev["exit"] = [cdates[i] if i < len(cdates) else np.datetime64("NaT") for i in ie]
    ev = ev.dropna(subset=["entry"])
    ev["gap_in"] = (ev.event_session - ev.entry).dt.days
    n0 = len(ev)
    ev = ev[(ev.gap_in >= 1) & (ev.gap_in <= MAXGAP)]
    print(f"[{TAG}] events with a pre-event chain snapshot within {MAXGAP}d: {len(ev)} / {n0}")

    # front + second expiry, both strictly after the event session
    a = atm.rename(columns={"date": "entry"})
    m = ev.merge(a, on=["entry", "symbol"], how="inner")
    m = m[m.expiration > m.event_session].sort_values(["symbol", "event_session", "expiration"])
    m["rk"] = m.groupby(["symbol", "event_session"]).cumcount()
    f = m[m.rk == 0].copy()
    s2 = m[m.rk == 1][["symbol", "event_session", "atm_iv", "tau", "dte"]].rename(
        columns={"atm_iv": "iv2", "tau": "tau2", "dte": "dte2"})
    f = f.merge(s2, on=["symbol", "event_session"], how="left")
    print(f"  with a front expiry after the event: {len(f)}"
          f"  (second expiry on {f.iv2.notna().mean():.0%})")

    # ---- underlying at expiry (raw), from the adjusted-bar ratio -----------
    sx, held = [], []
    for _, r in f.iterrows():
        b = bars[r.symbol]
        i0 = int(np.searchsorted(b.date.values, np.datetime64(r.entry, "ns"), "right")) - 1
        i1 = int(np.searchsorted(b.date.values, np.datetime64(r.expiration, "ns"), "right")) - 1
        if 0 <= i0 < i1 < len(b):
            sx.append(r.spot * b.close.values[i1] / b.close.values[i0])
            held.append(int(i1 - i0))
        else:
            sx.append(np.nan); held.append(np.nan)
    f["S_exp"] = sx
    f["n_hold"] = held
    f = f.dropna(subset=["S_exp"])

    # ---- T2: short the front ATM straddle at the BID, settle at intrinsic ---
    f["intrinsic"] = np.maximum(0, f.S_exp - f.strike) + np.maximum(0, f.strike - f.S_exp)
    f["pnl_exp_worst"] = (f.str_bid - f.intrinsic) / f.str_mid     # entry half-spread paid
    f["pnl_exp_mid"] = (f.str_mid - f.intrinsic) / f.str_mid       # no costs at all
    f["entry_cost"] = (f.str_ask - f.str_bid) / 2 / f.str_mid

    # ---- T2': quote-to-quote round trip, subsample -------------------------
    px = pan[["date", "symbol", "expiration", "strike", "cbid", "cask", "pbid", "pask"]] \
        .rename(columns={"date": "exit", "cbid": "xcb", "cask": "xca",
                         "pbid": "xpb", "pask": "xpa"})
    f = f.merge(px, on=["exit", "symbol", "expiration", "strike"], how="left")
    f["gap_out"] = (f["exit"] - f.event_session).dt.days
    ok = f.xcb.notna() & (f.gap_out >= 0) & (f.gap_out <= MAXGAP)
    f["pnl_rt_worst"] = np.where(ok, (f.str_bid - (f.xca + f.xpa)) / f.str_mid, np.nan)
    f["pnl_rt_mid"] = np.where(ok, (f.str_mid - (f.xcb + f.xca + f.xpb + f.xpa) / 2) / f.str_mid, np.nan)
    print(f"  quote-to-quote round trip available on {int(ok.sum())} ({ok.mean():.0%})")

    # ---- implied moves ------------------------------------------------------
    f["M1"] = f.str_mid / f.spot                        # raw front straddle / spot
    f["M2"] = event_strip(f.atm_iv.values, f.tau.values, f.iv2.values, f.tau2.values)
    f["IMP"] = SQ2PI * f.M2                             # implied EXPECTED |move|
    f["R"] = f.abs_r

    # ---- targets ------------------------------------------------------------
    f["T1"] = f.IMP - f.R                               # implied - realized (the ask)
    f["T1_raw"] = f.M1 - f.R                            # same with the unstripped straddle

    # ---- past-only signals --------------------------------------------------
    # S1 / hist_absr come from events.parquet where they were built on the FULL
    # event history (past-only), so the warm-up does not discard chain-covered
    # events. For the placebo runs the sessions are shifted, so rebuild them
    # in-sample instead.
    f = f.sort_values(["symbol", "event_session"]).reset_index(drop=True)
    if PLACEBO or SHIFT2ND:
        g = f.groupby("symbol")
        f["S1"] = g.z.transform(lambda s: np.log(s.clip(lower=1e-4)).shift(1).rolling(4).mean())
        f["hist_absr"] = g.abs_r.transform(lambda s: s.shift(1).rolling(4).mean())
        f["nprior"] = g.cumcount()
    else:
        f["S1"] = f.prior_logz4
        f["hist_absr"] = f.prior_absr4
    f["S2"] = f.IMP / f.hist_absr
    f["S3"] = f.IMP
    f["S2raw"] = f.M1 / f.hist_absr

    f["window"] = f.event_session.dt.to_period("W").astype(str)
    f = f[(f.nprior >= WARMUP) & f.M2.notna()]
    f["nw"] = f.groupby("window").symbol.transform("size")
    f = f[f.nw >= MINNAMES]
    for c in ["S1", "S2", "S3", "S2raw", "T1", "T1_raw", "R",
              "pnl_exp_worst", "pnl_exp_mid", "pnl_rt_worst", "pnl_rt_mid"]:
        f[c + "_d"] = f[c] - f.groupby("window")[c].transform("mean")
    return f


def main():
    f = build()
    yrs = (f.event_session.max() - f.event_session.min()).days / 365.25
    print(f"\n================== {TAG} ==================")
    print(f"final sample: {len(f)} events | {f.symbol.nunique()} symbols | "
          f"{f.window.nunique()} windows | {f.event_session.min().date()}..{f.event_session.max().date()}")
    print(f"span {yrs:.2f} yr -> {len(f)/yrs:.0f} events/yr | median names/window {f.nw.median():.0f} "
          f"| median DTE {f.dte.median():.0f} | median hold {f.n_hold.median():.0f} sessions")

    print("\n-- levels (median unless noted) --")
    for c, lab in [("M1", "raw straddle / spot"), ("M2", "event-vol strip sigma"),
                   ("IMP", "implied E|move|"), ("R", "realized |move|"),
                   ("entry_cost", "entry half-spread / premium"),
                   ("spread_frac", "full straddle spread / mid")]:
        print(f"  {lab:<32}{f[c].median():>9.4f}")
    print(f"  {'mean IMP - R (variance premium)':<32}{f.T1.mean():>+9.4f}")
    for c in ["pnl_exp_mid", "pnl_exp_worst", "pnl_rt_mid", "pnl_rt_worst"]:
        x = f[c].dropna()
        if len(x) > 10:
            print(f"  mean {c:<27}{x.mean():>+9.4f}  (n={len(x)}, t={x.mean()/x.sem():+.2f})")

    print("\n-- IC: per-window Spearman on cross-sectionally demeaned pairs --")
    print(f"{'signal':<8}{'target':<16}{'IC':>9}{'t':>8}{'95%CI':>21}{'win':>6}{'obs':>7}{'IR':>8}")
    rows = []
    for sg in ["S1", "S2", "S3", "S2raw"]:
        for tg in ["R", "T1", "T1_raw", "pnl_exp_mid", "pnl_exp_worst",
                   "pnl_rt_mid", "pnl_rt_worst"]:
            r = window_ic(f, sg + "_d", tg + "_d")
            if r is None:
                continue
            ir = r["ic"] * np.sqrt(len(f) / yrs)
            ci = f"[{r['lo']:+.3f},{r['hi']:+.3f}]"
            print(f"{sg:<8}{tg:<16}{r['ic']:>+9.4f}{r['t']:>+8.2f}{ci:>21}"
                  f"{r['nwin']:>6}{r['nobs']:>7}{ir:>+8.2f}")
            rows.append(dict(tag=TAG, signal=sg, target=tg, **r, ir=ir))

    print("\n-- tercile long/short straddle book (short high-signal, long low-signal) --")
    print(f"{'signal':<8}{'pnl':<16}{'mean/win':>11}{'sd':>9}{'t':>8}{'Sharpe(ann)':>13}{'win':>6}")
    prows = []
    for sg in ["S1", "S2", "S3", "S2raw"]:
        for pl in ["pnl_exp_worst", "pnl_exp_mid", "pnl_rt_worst"]:
            per = []
            for _, g in f.groupby("window"):
                g = g[[sg + "_d", pl]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(g) < MINNAMES:
                    continue
                k = max(1, len(g) // 3)
                o = g.sort_values(sg + "_d")
                per.append(o[pl].iloc[-k:].mean() - o[pl].iloc[:k].mean())
            per = np.asarray(per)
            if len(per) < 5:
                continue
            t = per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))
            wpy = len(per) / yrs
            sh = per.mean() / per.std(ddof=1) * np.sqrt(wpy)
            print(f"{sg:<8}{pl:<16}{per.mean():>+11.4f}{per.std(ddof=1):>9.4f}"
                  f"{t:>+8.2f}{sh:>+13.2f}{len(per):>6}")
            prows.append(dict(tag=TAG, signal=sg, pnl=pl, mean=per.mean(), t=t, sharpe=sh,
                              nwin=len(per)))

    pd.DataFrame(rows).to_csv(os.path.join(HERE, f"ic_{TAG}.csv"), index=False)
    pd.DataFrame(prows).to_csv(os.path.join(HERE, f"port_{TAG}.csv"), index=False)
    f.to_parquet(os.path.join(HERE, f"sample_{TAG}.parquet"), index=False)


if __name__ == "__main__":
    main()
