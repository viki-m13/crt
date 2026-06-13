"""Early-exit ("sell before it goes ITM") and liquidity-overlay
simulation on the frozen v3 trade set.

Tests two common proposals for pushing the win rate to 100%:

  1. EARLY EXIT — close the spread when the underlying's close crosses
     a trigger near the short strike, instead of holding to expiry.
     Execution is modeled conservatively: the exit fires on a CLOSE
     (you cannot act on a close before it prints) and is filled at the
     NEXT session's close, paying Black-Scholes spread value at
     1.5x-stressed IV plus 15% slippage (min $0.05).
  2. LIQUIDITY FILTER — restrict to underlyings above an average-daily-
     dollar-volume threshold (90-day ADV, yfinance).

Result (see VALIDATION.md §8): neither — nor their combination —
reaches 100%. Stops mathematically convert near-strike trades into
realized losses (a vertical at the short strike costs ~0.3-0.5x width
to buy back vs the few-cent credit collected) and add whipsaw losses
on trades that would have expired worthless; win rate FALLS
(99.40% -> 98.94% at a strike-touch stop; 97.88% at strike+5%) and so
does total P&L. The only benefit is severity capping (worst trade
-$2,218 -> -$1,680). The liquidity filter shrinks the book faster than
it removes losses — the two survivors of a $1B/day ADV filter are
Mastercard and Broadcom, among the most liquid options on earth.

Inputs: results/replay_rows_full.csv.gz + cache_full/ + optionable.json
(see VALIDATION.md §7). Run:  python3 exit_sim.py
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from pricing import bs_call, bs_put, iv_at_strike
from replay_analysis import HERE, attach_pricing, load_rows
from research import CAP_LONG, CAP_SHORT, HIST_CLEAR, K_SIGMA
from pricing import MIN_TRADEABLE_FILL

EXIT_STRESS_IV = 1.5
EXIT_SLIP_FRAC = 0.15
EXIT_SLIP_FLOOR = 0.05
VALIDATION_START = "2019-01-01"


def frozen_trade_set() -> dict[str, np.ndarray]:
    """Rebuild the deduped v3 published trade set (same as validate_v3)."""
    starts = {}
    for p in glob.glob(os.path.join(HERE, "cache_full", "*.json")):
        t = os.path.basename(p)[:-5]
        with open(p) as fh:
            head = fh.read(200)
        m = re.search(r'"dates": \["(\d{4}-\d{2}-\d{2})"', head)
        if m:
            starts[t] = m.group(1)
    R = load_rows(os.path.join(HERE, "results", "replay_rows_full.csv.gz"))
    res = R["win"] >= 0
    h = R["horizon"]
    sig_d = R["sigma60"] / np.sqrt(252)
    with open(os.path.join(HERE, "results", "optionable.json")) as fh:
        opt = json.load(fh)["optionable"]
    optionable = np.array([opt.get(t, False) for t in R["ticker"]])
    start_arr = np.array([starts.get(t, "2026-01-01") for t in R["ticker"]])
    years10 = (R["date"].astype("datetime64[D]")
               - start_arr.astype("datetime64[D]")).astype(int) >= 3652
    caps = np.where(h >= 42, CAP_LONG, CAP_SHORT)
    b = K_SIGMA * sig_d * np.sqrt(h) + 0.01
    okb = np.isfinite(b) & (b <= caps) & (b >= HIST_CLEAR * R["hist_max"])
    Rt = dict(R)
    Rt["buffer"] = b
    attach_pricing(Rt, iv_mult=1.30)
    pub = optionable & years10 & okb & (Rt["net"] >= MIN_TRADEABLE_FILL)
    m = pub & res
    seen: set = set()
    keep = []
    for i in np.lexsort((R["date"],)):
        if not m[i]:
            continue
        key = (R["ticker"][i], R["side"][i], int(h[i]), R["expiry_date"][i])
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    keep = np.array(keep)
    return {
        "ticker": R["ticker"][keep], "side": R["side"][keep],
        "date": R["date"][keep], "expiry": R["expiry_date"][keep],
        "spot": R["spot"][keep], "b": b[keep], "net": Rt["net"][keep],
        "width": Rt["width"][keep], "sigma": R["sigma60"][keep],
        "close_exp": R["close_at_expiry"][keep],
    }


def spread_value(side: str, S: float, Ks: float, Kl: float, Trem: float,
                 sig: float) -> float:
    if Trem <= 0:
        return (min(max(Ks - S, 0.0), abs(Ks - Kl)) if side == "put"
                else min(max(S - Ks, 0.0), abs(Kl - Ks)))
    ss = iv_at_strike(S, Ks, Trem, sig, side)
    sl = iv_at_strike(S, Kl, Trem, sig, side)
    v = ((bs_put(S, Ks, Trem, ss) - bs_put(S, Kl, Trem, sl)) if side == "put"
         else (bs_call(S, Ks, Trem, ss) - bs_call(S, Kl, Trem, sl)))
    return max(v, 0.0)


def run_exit_sim(T: dict, series: dict, trigger_g: float | None) -> tuple[np.ndarray, np.ndarray]:
    """trigger_g=None -> hold to expiry. Else exit when close crosses
    strike*(1+g) (puts; symmetric for calls), filled next session."""
    n = len(T["ticker"])
    pnl = np.zeros(n)
    stopped = np.zeros(n, bool)
    for i in range(n):
        t = T["ticker"][i]
        ds, ps = series[t]
        d0 = np.datetime64(T["date"][i], "D")
        de = np.datetime64(T["expiry"][i], "D")
        j0 = int(np.searchsorted(ds, d0))
        je = int(np.searchsorted(ds, de))
        side = T["side"][i]
        b, width = float(T["b"][i]), float(T["width"][i])
        Ks = T["spot"][i] * (1 - b) if side == "put" else T["spot"][i] * (1 + b)
        Kl = Ks - width if side == "put" else Ks + width
        if trigger_g is not None:
            trig = Ks * (1 + trigger_g) if side == "put" else Ks * (1 - trigger_g)
            for j in range(j0 + 1, min(je, len(ps) - 1) + 1):
                hit = ps[j] <= trig if side == "put" else ps[j] >= trig
                if hit:
                    jx = min(j + 1, len(ps) - 1)
                    Trem = max(int((de - ds[jx]).astype(int)), 0) / 365.0
                    v = spread_value(side, ps[jx], Ks, Kl, Trem,
                                     float(T["sigma"][i]) * EXIT_STRESS_IV)
                    debit = min(v + max(EXIT_SLIP_FLOOR, EXIT_SLIP_FRAC * v), width)
                    pnl[i] = (float(T["net"][i]) - debit) * 100.0
                    stopped[i] = True
                    break
        if not stopped[i]:
            S_T = float(T["close_exp"][i])
            intr = (min(max(Ks - S_T, 0), width) if side == "put"
                    else min(max(S_T - Ks, 0), width))
            pnl[i] = (float(T["net"][i]) - intr) * 100.0
    return pnl, stopped


def adv_map(tickers: list[str]) -> dict[str, float]:
    """90-day average daily dollar volume (today's; liquidity proxy)."""
    import yfinance as yf
    df = yf.download(tickers=tickers, period="6mo", interval="1d",
                     auto_adjust=True, progress=False, group_by="column")
    out = {}
    for t in tickers:
        try:
            c = df["Close"][t].dropna()
            v = df["Volume"][t].dropna()
            out[t] = float((c * v).tail(90).mean())
        except Exception:  # noqa: BLE001
            out[t] = 0.0
    return out


def main() -> int:
    T = frozen_trade_set()
    uniq = sorted(set(T["ticker"]))
    series = {}
    for t in uniq:
        d = json.load(open(os.path.join(HERE, "cache_full", f"{t}.json")))["series"]
        series[t] = (np.array(d["dates"], dtype="datetime64[D]"),
                     np.array(d["prices"], float))
    val = T["date"] >= VALIDATION_START

    pnl0, _ = run_exit_sim(T, series, None)
    print("HOLD TO EXPIRY:    val losing "
          f"{int((pnl0[val] < 0).sum())}/{int(val.sum())} "
          f"pnl ${pnl0[val].sum():.0f} worst ${pnl0.min():.0f}")
    for g in (0.00, 0.03, 0.05):
        pnl, stopped = run_exit_sim(T, series, g)
        whip = stopped & (pnl0 >= 0)
        print(f"EXIT strike+{g:.0%}:  val losing "
              f"{int((pnl[val] < 0).sum())}/{int(val.sum())} "
              f"(win {100 * float((pnl[val] >= 0).mean()):.2f}%) "
              f"pnl ${pnl[val].sum():.0f} worst ${pnl.min():.0f} "
              f"whipsaws {int(whip.sum())} (${pnl[whip].sum():.0f})")

    adv = adv_map(uniq)
    adv_arr = np.array([adv.get(t, 0.0) for t in T["ticker"]])
    for thr in (100e6, 300e6, 1000e6):
        m = val & (adv_arr >= thr)
        lt = sorted(set(T["ticker"][m & (pnl0 < 0)]))
        print(f"LIQUID >= ${thr/1e6:.0f}M ADV: val losing "
              f"{int((m & (pnl0 < 0)).sum())}/{int(m.sum())} "
              f"pnl ${pnl0[m].sum():.0f}  losers={lt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
