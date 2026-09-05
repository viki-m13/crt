#!/usr/bin/env python3
"""Audit an advisor's municipal bond fills against the MSRB tape.

    python scripts/muni_audit.py trades.csv [--out report.html] [--offline]

Input CSV needs: cusip, date, side, par, price
  side is from the CLIENT's perspective — buy or sell
  date is the trade date, YYYY-MM-DD
  par is face value in dollars, price is per 100 of par

For every trade it finds what OTHER customers paid or received for the same
bond on the same day, and reports the difference in points and dollars. The
comparison is the one validated in research/FINDINGS.md; the arithmetic and
its guard rails live in api/_muniaudit.py.

What it will not do: estimate a benchmark when the tape has no comparable
print that day. Those trades come back "not assessable" and are listed as
such. An advisor can check every number here against emma.msrb.org, which is
the point — the report is only worth anything if it is checkable.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "api"))
sys.path.insert(0, "/home/user/viki-m13/bonds/munis/scripts")

import _muniaudit as M  # noqa: E402


def load_trades(path: str) -> list[M.Trade]:
    out = []
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            row = { (k or "").strip().lower(): (v or "").strip()
                    for k, v in row.items() }
            try:
                out.append(M.Trade(
                    cusip=row["cusip"].upper(),
                    date=row["date"][:10],
                    side=row["side"],
                    par=float(row["par"].replace(",", "").replace("$", "")),
                    price=float(row["price"].replace(",", "")),
                    description=row.get("description", "")))
            except (KeyError, ValueError) as e:
                print(f"  line {i}: skipped ({e})", file=sys.stderr)
    return out


class Tape:
    """Prints for a bond on a date. Live from EMMA unless --offline."""

    def __init__(self, offline: bool = False, tape_dir: str | None = None):
        self.offline = offline
        self.tape_dir = tape_dir
        self.client = None
        self._by_cusip: dict[str, list] = {}
        self._resolved: dict[str, str | None] = {}
        if tape_dir:
            # Local securityId-keyed tape. Used for demonstrations and for
            # testing the whole path without depending on EMMA being up;
            # identical arithmetic, same prints, just already downloaded.
            self.offline = True
            return
        if not offline:
            try:
                from emma_client import EmmaClient
                jar = os.path.join(ROOT, ".emma_cookies.txt")
                self.client = EmmaClient(cookie_jar=jar)
                self.client.ensure_session()
            except Exception as e:  # noqa: BLE001
                print(f"  EMMA unavailable ({type(e).__name__}: {e}); "
                      f"running offline", file=sys.stderr)
                self.offline = True

    def prints_for(self, cusip: str, date: str) -> list[dict]:
        if cusip not in self._by_cusip:
            self._by_cusip[cusip] = self._fetch(cusip)
        return [p for p in self._by_cusip[cusip] if p["date"] == date]

    def _fetch(self, cusip: str) -> list[dict]:
        if self.tape_dir:
            import glob as _g
            import pandas as _pd
            hit = _g.glob(os.path.join(self.tape_dir, cusip + "*.csv.gz"))
            if not hit:
                return []
            d = _pd.read_csv(hit[0])
            d["date"] = _pd.to_datetime(d.ts, errors="coerce").dt.date
            d = d.dropna(subset=["date", "price", "par"])
            return [{"date": str(r.date), "price": float(r.price),
                     "par": float(r.par), "side": r.side,
                     "ytw": getattr(r, "ytw", None)} for r in d.itertuples()]
        if self.offline or not self.client:
            return []
        try:
            v = self.client.validate_cusip(cusip)
            six = (v.get("securityId") or v.get("SecurityId")
                   or (v.get("data") or {}).get("securityId"))
            if not six:
                return []
            info = self.client.security_trade_info(six)
            rows = []
            for r in info.get("data", []):
                ms = r.get("TDT")
                if ms is None:
                    continue
                d = dt.datetime.utcfromtimestamp(int(ms) / 1000).date().isoformat()
                rows.append({"date": d, "price": float(r["PX"]),
                             "par": float(r.get("TA") or 0),
                             "side": r.get("TT"), "ytw": r.get("YX")})
            return rows
        except Exception as e:  # noqa: BLE001
            print(f"  {cusip}: lookup failed ({type(e).__name__})", file=sys.stderr)
            return []


def render(verdicts, totals, out_path: str):
    def money(v):
        return "—" if v is None else f"${v:,.0f}"
    rows = []
    for v in sorted(verdicts, key=lambda x: -(x.cost_dollars or -1e9)):
        d = M.to_dict(v)
        if not d["assessable"]:
            rows.append(
                f"<tr class='skip'><td>{html.escape(d['cusip'])}</td>"
                f"<td>{html.escape(d['date'])}</td><td>{html.escape(d['side'])}</td>"
                f"<td class='n'>{d['par']:,.0f}</td><td class='n'>{d['price']:.3f}</td>"
                f"<td colspan='3'>not assessable — {html.escape(d['reason'])}</td></tr>")
            continue
        cls = "bad" if (d["cost_dollars"] or 0) > 0 else "good"
        rows.append(
            f"<tr><td>{html.escape(d['cusip'])}</td><td>{html.escape(d['date'])}</td>"
            f"<td>{html.escape(d['side'])}</td><td class='n'>{d['par']:,.0f}</td>"
            f"<td class='n'>{d['price']:.3f}</td>"
            f"<td class='n'>{d['benchmark']:.3f}<span class='k'> "
            f"{html.escape(d['benchmark_kind'])}</span></td>"
            f"<td class='n {cls}'>{d['cost_points']:+.3f}</td>"
            f"<td class='n {cls}'>{money(d['cost_dollars'])}</td></tr>")

    pct = totals["cost_as_pct_of_par"]
    inst = totals["benchmarked_institutionally"]
    weak = totals["trades_assessed"] - inst
    if weak <= 0:
        yardstick = ("All of them were compared against an institutional-size "
                     "print on the same day — the strongest available yardstick.")
    elif inst == 0:
        yardstick = (f"None had an institutional-size print that day, so all "
                     f"{weak} were compared against other customer trades — a "
                     f"weaker yardstick that generally understates the cost.")
    else:
        yardstick = (f"{inst} were compared against an institutional-size print; "
                     f"the other {weak} against customer trades only, which is a "
                     f"weaker yardstick and generally understates the cost.")
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Municipal execution review</title><style>
body{{font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
max-width:1000px;margin:0 auto;padding:28px 18px;color:#111}}
h1{{font-size:26px;letter-spacing:-.02em;margin:0 0 6px}}
.sub{{color:#666;margin:0 0 26px}}
.big{{font-size:40px;font-weight:700;letter-spacing:-.03em;margin:6px 0}}
.card{{border:1px solid #e3e3e8;border-radius:14px;padding:18px;margin:16px 0;
background:#fafafc}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:10px}}
th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
color:#777;padding:0 8px 8px 0;border-bottom:1px solid #e3e3e8}}
td{{padding:9px 8px 9px 0;border-bottom:1px solid #f0f0f3}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
.bad{{color:#c0392b;font-weight:600}} .good{{color:#1e8449}}
tr.skip td{{color:#999}} .k{{color:#999;font-size:10.5px;margin-left:5px}}
.note{{background:#fff8e6;border:1px solid #f0dca8;border-radius:12px;
padding:14px;font-size:13px;margin-top:18px}}
</style></head><body>
<h1>Municipal execution review</h1>
<p class="sub">Every fill compared with what other customers paid for the same
bond, on the same day, on the same side — from the MSRB's own trade tape.</p>

<div class="card">
  <div style="color:#666;font-size:12px;text-transform:uppercase;
  letter-spacing:.06em">Execution cost vs the tape</div>
  <div class="big">{money(totals['total_cost_dollars'])}</div>
  <div style="color:#666">across {totals['trades_assessed']} assessable trades
  on {money(totals['par_assessed'])} of par
  {f"({pct:.2f}% of par)" if pct is not None else ""}</div>
</div>

<p><b>{totals['trades_worse_than_benchmark']}</b> of
<b>{totals['trades_assessed']}</b> assessed trades were worse than the day's
benchmark. {yardstick}
{f"<b>{totals['trades_not_assessable']}</b> trades could not be assessed because the tape held no comparable print that day." if totals['trades_not_assessable'] else ""}</p>

<table><thead><tr><th>CUSIP</th><th>Date</th><th>Side</th><th class="n">Par</th>
<th class="n">Your price</th><th class="n">Benchmark</th>
<th class="n">Points</th><th class="n">Cost</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>

<div class="note"><b>How to check this.</b> Every figure comes from
emma.msrb.org, the MSRB's public trade tape, where every dealer must report
every municipal trade. Look up any CUSIP and date above and you will see the
same prints. Where no comparable print existed we say so rather than
estimating one — no matrix prices and no interpolation.
<br><br>This measures execution against the tape. It is not investment advice
and does not assess whether a bond was suitable.</div>
</body></html>"""
    with open(out_path, "w") as f:
        f.write(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trades")
    ap.add_argument("--out", default="muni_review.html")
    ap.add_argument("--offline", action="store_true",
                    help="skip EMMA; only useful for testing the renderer")
    ap.add_argument("--tape-dir",
                    help="read prints from a local securityId-keyed tape "
                         "instead of EMMA (demos and end-to-end testing)")
    a = ap.parse_args()

    trades = load_trades(a.trades)
    if not trades:
        print("no usable trades"); return 1
    print(f"auditing {len(trades)} trades…")

    tape = Tape(offline=a.offline, tape_dir=a.tape_dir)
    verdicts = []
    for t in trades:
        verdicts.append(M.audit_trade(t, tape.prints_for(t.cusip, t.date)))
    totals = M.audit_portfolio(verdicts)

    print(f"\n  assessed {totals['trades_assessed']}/{totals['trades_submitted']}")
    if totals["trades_assessed"]:
        print(f"  execution cost vs tape: ${totals['total_cost_dollars']:,.0f}"
              f" on ${totals['par_assessed']:,.0f} of par")
        print(f"  worse than benchmark: {totals['trades_worse_than_benchmark']}"
              f"/{totals['trades_assessed']}")
        if totals["worst"]:
            w = totals["worst"]
            print(f"  worst single trade: {w.trade.cusip} {w.trade.date} "
                  f"${w.cost_dollars:,.0f}")
    render(verdicts, totals, a.out)
    print(f"\n  report -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
