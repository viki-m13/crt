#!/usr/bin/env python3
"""Inventory + quote-quality audit over the fetched EOD option cache.

Streams cache/chains/*.parquet one date at a time (never loads the whole panel),
accumulates:
  - per-symbol coverage (dates, rows, first/last)
  - two-sided-quote fraction, zero bids, crossed markets, null greeks
  - bid-ask spread as % of mid, bucketed by moneyness (K/S from put-call
    parity spot) x DTE
  - stale-quote proxy: (bid,ask) identical to the same contract's previous
    observation date
Writes a JSON blob of results to cache/inventory.json for the markdown writer.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import engine  # noqa: E402

CHAINS = os.path.join(ROOT, "cache", "chains")
VOL = os.path.join(ROOT, "cache", "vol")
OUT = os.path.join(ROOT, "cache", "inventory.json")

KS_EDGES = [0.0, 0.90, 0.95, 0.98, 1.02, 1.05, 1.10, 9.9]
KS_LABELS = ["<0.90", "0.90-0.95", "0.95-0.98", "0.98-1.02", "1.02-1.05", "1.05-1.10", ">1.10"]
DTE_EDGES = [-1, 7, 14, 30, 60, 90, 10000]
DTE_LABELS = ["0-7", "8-14", "15-30", "31-60", "61-90", "90+"]


def main() -> None:
    paths = sorted(glob.glob(os.path.join(CHAINS, "*.parquet")))
    print(f"{len(paths)} chain files", flush=True)

    per_sym: dict[str, dict] = {}
    tot = dict(rows=0, bid_null=0, ask_null=0, iv_null=0, greek_null=0,
               zero_bid=0, zero_ask=0, crossed=0, locked=0, two_sided=0,
               mid_le_0=0, spread_gt_mid=0)
    keep = []          # compact per-row records for the spread tables
    stale_num = stale_den = 0
    stale_liq = [0, 0]
    prev_key = None    # previous date's contract->(bid,ask) map
    prev_day = None
    per_year: dict[str, int] = {}
    dates: list[str] = []
    exp_per_date = []
    spot_fail = 0

    for i, p in enumerate(paths):
        day = os.path.basename(p)[:-8]
        raw = pd.read_parquet(p)
        if raw.empty:
            continue
        dates.append(day)
        for c in ("bid", "ask", "vol", "delta", "gamma", "theta", "vega", "strike"):
            if c in raw.columns:
                raw[c] = pd.to_numeric(raw[c], errors="coerce")

        n = len(raw)
        tot["rows"] += n
        per_year[day[:4]] = per_year.get(day[:4], 0) + n
        tot["bid_null"] += int(raw.bid.isna().sum())
        tot["ask_null"] += int(raw.ask.isna().sum())
        tot["iv_null"] += int(raw["vol"].isna().sum())
        tot["greek_null"] += int(raw[["delta", "gamma", "theta", "vega"]].isna().any(axis=1).sum())
        b, a = raw.bid.fillna(-1), raw.ask.fillna(-1)
        tot["zero_bid"] += int((b == 0).sum())
        tot["zero_ask"] += int((a == 0).sum())
        tot["crossed"] += int((b > a).sum())
        tot["locked"] += int(((b == a) & (b > 0)).sum())
        two = (b > 0) & (a > 0) & (a >= b)
        tot["two_sided"] += int(two.sum())

        # per-symbol coverage
        for sym, g in raw.groupby("act_symbol"):
            d = per_sym.setdefault(sym, dict(rows=0, dates=0, first=day, last=day,
                                             two_sided=0, zero_bid=0, crossed=0,
                                             spread_pct_sum=0.0, spread_pct_n=0))
            d["rows"] += len(g)
            d["dates"] += 1
            d["last"] = day
            gb, ga = g.bid.fillna(-1), g.ask.fillna(-1)
            gtwo = (gb > 0) & (ga > 0) & (ga >= gb)
            d["two_sided"] += int(gtwo.sum())
            d["zero_bid"] += int((gb == 0).sum())
            d["crossed"] += int((gb > ga).sum())
            d["n_exp"] = d.get("n_exp", 0) + int(g.expiration.nunique())
            d["n_strk"] = d.get("n_strk", 0) + int(g.strike.nunique())

        exp_per_date.append(raw.groupby("act_symbol").expiration.nunique().mean())

        # ---- stale-quote proxy (same contract, consecutive observation dates)
        key = raw.set_index(["act_symbol", "expiration", "strike", "call_put"])[["bid", "ask"]]
        key = key[~key.index.duplicated()]
        gap_ok = prev_day is not None and (pd.Timestamp(day) - pd.Timestamp(prev_day)).days <= 4
        if prev_key is not None and gap_ok:
            j = key.join(prev_key, how="inner", rsuffix="_p")
            if len(j):
                same = (j.bid == j.bid_p) & (j.ask == j.ask_p)
                stale_num += int(same.sum())
                stale_den += len(j)
                liq = ((j.bid + j.ask) / 2.0) >= 0.50
                stale_liq[0] += int((same & liq).sum())
                stale_liq[1] += int(liq.sum())
        prev_key, prev_day = key, day

        # ---- spread table inputs (two-sided rows only)
        try:
            spots = engine.parity_spots(engine.load_chain(day))
        except Exception:  # noqa: BLE001
            spots = {}
        if not spots:
            spot_fail += 1
            continue
        q = raw[two].copy()
        q["spot"] = q.act_symbol.map(spots)
        q = q[q.spot.notna() & (q.spot > 0)]
        if q.empty:
            continue
        q["mid"] = (q.bid + q.ask) / 2.0
        tot["mid_le_0"] += int((q["mid"] <= 0).sum())
        q = q[q["mid"] > 0]
        q["spread_pct"] = 100.0 * (q.ask - q.bid) / q["mid"]
        q["ks"] = q.strike / q.spot
        q["dte"] = (pd.to_datetime(q.expiration) - pd.to_datetime(q.date)).dt.days
        rec = pd.DataFrame({
            "ks_b": pd.cut(q.ks, KS_EDGES, labels=KS_LABELS).cat.codes.astype("int8"),
            "dte_b": pd.cut(q.dte, DTE_EDGES, labels=DTE_LABELS).cat.codes.astype("int8"),
            "cp": (q.call_put == "Call").to_numpy(),
            "spread_pct": q.spread_pct.astype("float32").to_numpy(),
            "spread_abs": (q.ask - q.bid).astype("float32").to_numpy(),
            "mid": q["mid"].astype("float32").to_numpy(),
            "adelta": q.delta.abs().astype("float32").to_numpy(),
            "sym": q.act_symbol.to_numpy(),
        })
        rec["yr"] = np.int16(int(day[:4]))
        keep.append(rec)
        for sym, g in rec.groupby("sym"):
            d = per_sym[sym]
            d["spread_pct_sum"] += float(g.spread_pct.sum())
            d["spread_pct_n"] += len(g)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(paths)} rows={tot['rows']:,}", flush=True)

    R = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
    print(f"quote rows for spread tables: {len(R):,}", flush=True)

    def tab(idx_col, idx_labels, sub=None):
        d = R if sub is None else R[sub]
        out = {}
        for i, lab in enumerate(idx_labels):
            m = d[d[idx_col] == i]
            if len(m) == 0:
                continue
            out[lab] = dict(n=int(len(m)), med=float(m.spread_pct.median()),
                            mean=float(m.spread_pct.mean()),
                            p90=float(m.spread_pct.quantile(0.90)),
                            med_abs=float(m.spread_abs.median()),
                            med_mid=float(m.mid.median()))
        return out

    grid = {}
    for i, kl in enumerate(KS_LABELS):
        for jj, dl in enumerate(DTE_LABELS):
            m = R[(R.ks_b == i) & (R.dte_b == jj)]
            if len(m) < 50:
                continue
            grid[f"{kl}|{dl}"] = dict(n=int(len(m)), med=float(m.spread_pct.median()),
                                      p90=float(m.spread_pct.quantile(0.90)),
                                      med_abs=float(m.spread_abs.median()))

    # delta-bucketed view
    dbins = [0, .10, .25, .40, .60, .75, .90, 1.01]
    dlabs = ["0-10", "10-25", "25-40", "40-60", "60-75", "75-90", "90-100"]
    dcut = pd.cut(R.adelta, dbins, labels=dlabs)
    dtab = {}
    for lab, m in R.groupby(dcut, observed=True):
        if len(m) < 50:
            continue
        dtab[str(lab)] = dict(n=int(len(m)), med=float(m.spread_pct.median()),
                              p90=float(m.spread_pct.quantile(0.90)),
                              med_abs=float(m.spread_abs.median()),
                              med_mid=float(m.mid.median()))

    # SPY-only grid
    spy = R[R.sym == "SPY"]
    spy_grid = {}
    for i, kl in enumerate(KS_LABELS):
        for jj, dl in enumerate(DTE_LABELS):
            m = spy[(spy.ks_b == i) & (spy.dte_b == jj)]
            if len(m) < 30:
                continue
            spy_grid[f"{kl}|{dl}"] = dict(n=int(len(m)), med=float(m.spread_pct.median()),
                                          p90=float(m.spread_pct.quantile(0.90)),
                                          med_abs=float(m.spread_abs.median()))

    per_sym_med = {s: float(g.spread_pct.median()) for s, g in R.groupby("sym")}
    by_year = {int(y): dict(n=int(len(g)), med=float(g.spread_pct.median()),
                            p90=float(g.spread_pct.quantile(0.90)))
               for y, g in R.groupby("yr")}
    spy_atm = spy[(spy.ks_b == KS_LABELS.index("0.98-1.02"))]
    spy_atm_by_dte = {DTE_LABELS[j]: dict(n=int(len(m)), med=float(m.spread_pct.median()),
                                          med_abs=float(m.spread_abs.median()),
                                          med_mid=float(m.mid.median()))
                      for j, m in ((j, spy_atm[spy_atm.dte_b == j]) for j in range(len(DTE_LABELS)))
                      if len(m) >= 30}

    # ---- vol table coverage
    vpaths = sorted(glob.glob(os.path.join(VOL, "*.parquet")))
    vrows = 0
    vsyms: dict[str, int] = {}
    vcols: list[str] = []
    vnull: dict[str, int] = {}
    for p in vpaths:
        v = pd.read_parquet(p)
        vrows += len(v)
        if not vcols:
            vcols = list(v.columns)
        for s, c in v.act_symbol.value_counts().items():
            vsyms[s] = vsyms.get(s, 0) + int(c)
        for c in v.columns:
            vnull[c] = vnull.get(c, 0) + int(v[c].isna().sum())

    # ---- calendar gaps
    ds = pd.to_datetime(pd.Series(dates))
    allbd = pd.bdate_range(ds.min(), ds.max())
    missing = sorted(set(allbd.strftime("%Y-%m-%d")) - set(dates))

    res = dict(
        n_files=len(paths), n_dates=len(dates), first=dates[0] if dates else None,
        last=dates[-1] if dates else None, totals=tot, per_sym=per_sym,
        per_sym_med_spread=per_sym_med, grid=grid, spy_grid=spy_grid,
        by_year=by_year, spy_atm_by_dte=spy_atm_by_dte,
        by_ks=tab("ks_b", KS_LABELS), by_dte=tab("dte_b", DTE_LABELS),
        by_delta=dtab,
        call_med=float(R[R.cp].spread_pct.median()), put_med=float(R[~R.cp].spread_pct.median()),
        stale_frac=(stale_num / stale_den if stale_den else None), stale_den=stale_den,
        per_year=per_year,
        stale_frac_liq=(stale_liq[0]/stale_liq[1] if stale_liq[1] else None),
        stale_liq_den=stale_liq[1],
        manifest=json.load(open(os.path.join(ROOT,'cache','manifest.json'))) if os.path.exists(os.path.join(ROOT,'cache','manifest.json')) else {},
        spot_fail=spot_fail, mean_exp_per_sym_date=float(np.mean(exp_per_date)),
        vol=dict(files=len(vpaths), rows=vrows, syms=vsyms, cols=vcols, nulls=vnull),
        missing_bdays=missing, dates=dates,
    )
    json.dump(res, open(OUT, "w"))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
