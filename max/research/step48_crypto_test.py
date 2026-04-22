#!/usr/bin/env python3
"""Step 48 — Test dynamic ATR-scaled TP/SL strategy on crypto.

The user asked whether the Max strategy (monthly top-1 + dynamic ATR TP/SL,
step45-47 winner) works on crypto before we drop crypto from the Max page.

Strategy is universe-agnostic in principle:
  - Rank: each month take the "top pick" (here by simple
    trailing-12m momentum since we don't have the full CAP5 conviction
    stack for crypto).
  - Enter: next-day close.
  - TP = entry × (1 + max(0.05, min(7 × ATR14%, 0.25)))
  - SL = entry × (1 − max(0.05, min(7 × ATR14%, 0.12)))
  - Time-stop 252 bars.

Crypto has higher daily vol (often 3-6% ATR), so most trades will cap at
+25%/-12%. Curious to see if this still beats a BTC-DCA baseline.

Data: yfinance top-20 crypto by mkt cap, daily from 2018-01-01 to today.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DCA_MONTHLY = 1000.0
TRADING_DAYS_YR = 365  # crypto trades 7 days; use 365 for CAGR
WINDOW_START = "2018-01-01"
WINDOW_END = "2026-04-22"
RANK_LOOKBACK = 252  # calendar days for momentum

TP_ATR_K = 7.0
SL_ATR_K = 7.0
TP_CAP = 0.25
TP_FLOOR = 0.05
SL_CAP = 0.12
SL_FLOOR = 0.05
TS_BARS = 252

CRYPTO = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "LINK-USD", "DOGE-USD",
    "AVAX-USD", "DOT-USD", "XRP-USD", "LTC-USD", "UNI-USD", "ATOM-USD",
    "NEAR-USD", "AAVE-USD", "MATIC-USD", "ARB-USD", "OP-USD", "FIL-USD",
    "ETC-USD", "BCH-USD",
]


def month_first_indices(idx):
    out = []
    prev = None
    for i, dd in enumerate(idx):
        ym = (dd.year, dd.month)
        if ym != prev:
            out.append(i)
            prev = ym
    return out


def wilder_atr_pct(high, low, close, n=14):
    prev = close.shift(1)
    tr = pd.DataFrame({
        "hl": (high - low).abs(),
        "hp": (high - prev).abs(),
        "lp": (low - prev).abs(),
    }).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return atr / close


def simulate(close, high, low, adj, ranker, tp_k=TP_ATR_K, sl_k=SL_ATR_K,
             tp_cap=TP_CAP, sl_cap=SL_CAP):
    dates = close.index
    n = len(dates)
    mi = month_first_indices(dates)
    tk = [t for t in close.columns if t != "BTC-USD"]
    C = close[tk].to_numpy()
    H = high[tk].to_numpy()
    L = low[tk].to_numpy()
    R = ranker[tk].to_numpy()
    # ATR per ticker
    atrs = pd.DataFrame(index=close.index, columns=tk, dtype=float)
    for t in tk:
        atrs[t] = wilder_atr_pct(high[t], low[t], close[t])
    A = atrs.to_numpy()

    cash = 0.0
    equity = np.zeros(n)
    positions = []
    open_pos = None
    total_invested = 0.0

    for idx_mo, di in enumerate(mi):
        total_invested += DCA_MONTHLY
        cash += DCA_MONTHLY
        entry_idx = di + 1
        if entry_idx >= n:
            break

        if open_pos is None:
            scores = R[di]
            prices = C[di]
            best_ti, best_s = -1, -np.inf
            for i, tkname in enumerate(tk):
                s = scores[i]
                p = prices[i]
                if not (np.isfinite(s) and np.isfinite(p) and p > 0):
                    continue
                if s > best_s:
                    best_s = s
                    best_ti = i
            if best_ti >= 0:
                ent_px = C[entry_idx, best_ti]
                atr = A[di, best_ti]
                if np.isfinite(ent_px) and ent_px > 0 and np.isfinite(atr) and atr > 0:
                    tp_frac = max(TP_FLOOR, min(tp_k * atr, tp_cap))
                    sl_frac = max(SL_FLOOR, min(sl_k * atr, sl_cap))
                    deploy = cash
                    cash = 0.0
                    shares = deploy / ent_px
                    open_pos = {
                        "ti": best_ti, "tk": tk[best_ti],
                        "entry_idx": entry_idx, "entry_px": ent_px,
                        "tp_px": ent_px * (1 + tp_frac),
                        "sl_px": ent_px * (1 - sl_frac),
                        "shares": shares, "cost": deploy,
                        "stop_idx": entry_idx + TS_BARS,
                        "tp_frac": tp_frac, "sl_frac": sl_frac,
                    }

        next_di = mi[idx_mo + 1] if (idx_mo + 1) < len(mi) else n
        for d in range(di, next_di):
            if open_pos is not None and d > open_pos["entry_idx"]:
                ti = open_pos["ti"]
                hi = H[d, ti]
                lo = L[d, ti]
                exit_px = None
                reason = None
                if np.isfinite(lo) and lo <= open_pos["sl_px"]:
                    exit_px = open_pos["sl_px"]
                    reason = "sl"
                elif np.isfinite(hi) and hi >= open_pos["tp_px"]:
                    exit_px = open_pos["tp_px"]
                    reason = "tp"
                elif d >= open_pos["stop_idx"]:
                    px = C[d, ti]
                    if not (np.isfinite(px) and px > 0):
                        for back in range(d, open_pos["entry_idx"], -1):
                            if np.isfinite(C[back, ti]) and C[back, ti] > 0:
                                px = C[back, ti]
                                break
                        else:
                            px = open_pos["entry_px"]
                    exit_px = px
                    reason = "time"
                if exit_px is not None:
                    cash += open_pos["shares"] * exit_px
                    positions.append({
                        "tk": open_pos["tk"], "entry_idx": open_pos["entry_idx"],
                        "exit_idx": d, "entry_px": open_pos["entry_px"],
                        "exit_px": exit_px, "cost": open_pos["cost"],
                        "proceeds": open_pos["shares"] * exit_px,
                        "days_held": d - open_pos["entry_idx"],
                        "reason": reason,
                        "ret": exit_px / open_pos["entry_px"] - 1,
                    })
                    open_pos = None
            eq = cash
            if open_pos is not None:
                if d < open_pos["entry_idx"]:
                    eq += open_pos["cost"]
                else:
                    px = C[d, open_pos["ti"]]
                    if np.isfinite(px):
                        eq += open_pos["shares"] * px
                    else:
                        eq += open_pos["cost"]
            equity[d] = eq

    # Cleanup / forward fill zero equity
    for i in range(n):
        if equity[i] == 0 and i > 0:
            equity[i] = equity[i - 1]

    return equity, positions, total_invested, dates


def btc_dca_baseline(close):
    dates = close.index
    mi = month_first_indices(dates)
    btc = close["BTC-USD"].ffill().to_numpy()
    n = len(dates)
    equity = np.zeros(n)
    shares = 0.0
    total = 0.0
    nxt = 0
    for d in range(n):
        while nxt < len(mi) and mi[nxt] + 1 == d:
            px = btc[d]
            if np.isfinite(px) and px > 0:
                shares += DCA_MONTHLY / px
                total += DCA_MONTHLY
            nxt += 1
        if np.isfinite(btc[d]) and btc[d] > 0:
            equity[d] = shares * btc[d]
        elif d > 0:
            equity[d] = equity[d - 1]
    return equity, total


def metrics(equity, total_invested, positions, dates):
    if total_invested <= 0:
        return {}
    final = float(equity[-1]) if len(equity) else 0.0
    start_i = next((i for i, v in enumerate(equity) if v > 0), 0)
    eq = equity[start_i:]
    yrs = len(eq) / 365.0
    cagr = (final / total_invested) ** (1 / yrs) - 1 if yrs > 0 and final > 0 else -1.0
    ret = np.zeros(len(eq))
    for i in range(1, len(eq)):
        if eq[i - 1] > 0:
            ret[i] = eq[i] / eq[i - 1] - 1
    sharpe = (ret.mean() / ret.std()) * math.sqrt(365) if ret.std() > 0 else 0.0
    peak = 0.0
    mdd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    n_t = len(positions)
    tp_hits = sum(1 for p in positions if p["reason"] == "tp")
    sl_hits = sum(1 for p in positions if p["reason"] == "sl")
    tm_hits = sum(1 for p in positions if p["reason"] == "time")
    rets = [p["ret"] for p in positions]
    avg_win = float(np.mean([r for r in rets if r > 0])) if any(r > 0 for r in rets) else 0.0
    avg_loss = float(np.mean([r for r in rets if r < 0])) if any(r < 0 for r in rets) else 0.0
    gross_wr = sum(1 for r in rets if r > 0) / n_t if n_t else 0.0
    return {
        "cagr": cagr, "mdd": mdd, "sharpe": sharpe, "calmar": (cagr / mdd) if mdd > 0 else 0.0,
        "tp_hits": tp_hits, "sl_hits": sl_hits, "time_hits": tm_hits,
        "n_trades": n_t, "gross_wr": gross_wr, "avg_winner": avg_win, "avg_loser": avg_loss,
        "final_equity": final, "total_invested": total_invested,
    }


def main():
    print(f"Downloading {len(CRYPTO)} crypto tickers from {WINDOW_START}...", flush=True)
    data = yf.download(CRYPTO, start=WINDOW_START, end=WINDOW_END, interval="1d",
                       progress=False, auto_adjust=False, group_by="column")
    close = data["Close"].dropna(how="all")
    high = data["High"].reindex(close.index)
    low = data["Low"].reindex(close.index)
    adj = data["Adj Close"].reindex(close.index)

    # Forward-fill to handle occasional missing bars
    close = close.ffill()
    high = high.ffill()
    low = low.ffill()

    print(f"  {close.shape[0]} dates, {close.shape[1]} tickers", flush=True)

    # Simple ranker: 12-month momentum (trailing 252d total return)
    ranker = close.pct_change(252).rolling(21, min_periods=1).mean()

    # Run dynamic TP
    eq, positions, total, dates = simulate(close, high, low, adj, ranker)
    m = metrics(eq, total, positions, dates)
    print("\n[DYNAMIC TP on crypto, 12m momentum ranker]")
    print(f"  CAGR {m['cagr']*100:>6.2f}%  MDD {m['mdd']*100:>5.1f}%  Sharpe {m['sharpe']:>4.2f}  Calmar {m['calmar']:>5.3f}")
    print(f"  {m['n_trades']} trades | TP {m['tp_hits']} / SL {m['sl_hits']} / time {m['time_hits']} | gross WR {m['gross_wr']*100:.1f}%")
    print(f"  avg_winner +{m['avg_winner']*100:.2f}%  avg_loser {m['avg_loser']*100:.2f}%")

    # BTC DCA baseline
    eq_b, tot_b = btc_dca_baseline(close)
    m_b = metrics(eq_b, tot_b, [], dates)
    print("\n[BTC DCA baseline]")
    print(f"  CAGR {m_b['cagr']*100:>6.2f}%  MDD {m_b['mdd']*100:>5.1f}%  Sharpe {m_b['sharpe']:>4.2f}")

    # SPY DCA comparison (for vs-equities context)
    print("\n[Max stock strategy (for comparison)]: +29.26% CAGR (2006-2026) / 41% MDD / 0.71 Calmar")

    results = {"crypto_dynamic_tp": m, "btc_dca": m_b}
    Path("/home/user/crt/max/research/step48_crypto_results.json").write_text(
        json.dumps(results, indent=2, default=float))
    print("\nWrote step48_crypto_results.json")

    # Print trade log
    if positions:
        print("\nTop tickers:")
        from collections import Counter
        tk_counts = Counter(p["tk"] for p in positions)
        for tk, n in tk_counts.most_common(10):
            print(f"  {tk}: {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
