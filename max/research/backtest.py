#!/usr/bin/env python3
"""
Honest, point-in-time backtester for the "Top Picks" stocks strategy.

Loads max/docs/data/full.json (already embeds per-ticker bt_series: dates,
prices, point-in-time final_score). Simulates monthly DCA strategies
varying:
  - hold horizon (fixed, single number)
  - pick count (fixed top-N or threshold on final_score)
  - sizing (equal vs conviction-weighted)
  - regime gate (SPY 200SMA)
  - sector cap
  - trailing stop

Benchmark: DCA into SPY with the same capital schedule.

The bt_series final_score is computed historically in daily_scan_max.py via
compute_final_score_series (no look-ahead) — we only use values at or before
the rebalance date. The per-ticker "best horizon" in the JS Top Picks
backtest comes from TODAY's probabilities; we do NOT use that here.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Iterable, Optional

import numpy as np


# ------------------------------------------------------------------
# Sector map for the curated ~100-stock universe (from daily_scan_max.py).
# Used for sector-cap diversification.
# ------------------------------------------------------------------
SECTOR_MAP = {
    # Technology
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AVGO": "Tech", "ADBE": "Tech",
    "CRM": "Tech", "AMD": "Tech", "INTC": "Tech", "CSCO": "Tech", "TXN": "Tech",
    "AMAT": "Tech", "MU": "Tech", "NOW": "Tech", "PANW": "Tech", "CDNS": "Tech",
    "ARM": "Tech", "SMCI": "Tech",
    # Communication
    "GOOGL": "Comm", "META": "Comm", "NFLX": "Comm", "DIS": "Comm", "T": "Comm",
    "VZ": "Comm",
    # Consumer Disc
    "AMZN": "Disc", "TSLA": "Disc", "HD": "Disc", "MCD": "Disc", "NKE": "Disc",
    "SBUX": "Disc", "LOW": "Disc", "TJX": "Disc", "BKNG": "Disc", "GM": "Disc",
    # Staples
    "PG": "Staples", "KO": "Staples", "PEP": "Staples", "COST": "Staples",
    "WMT": "Staples", "PM": "Staples", "CL": "Staples",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "MPC": "Energy",
    # Financials
    "JPM": "Fin", "BAC": "Fin", "WFC": "Fin", "GS": "Fin", "MS": "Fin",
    "BLK": "Fin", "AXP": "Fin", "C": "Fin", "USB": "Fin", "PNC": "Fin",
    "COIN": "Fin", "SQ": "Fin",
    # Healthcare
    "UNH": "Health", "JNJ": "Health", "LLY": "Health", "PFE": "Health",
    "ABBV": "Health", "TMO": "Health", "ABT": "Health", "MRK": "Health",
    "AMGN": "Health", "GILD": "Health",
    # Industrials
    "CAT": "Indust", "HON": "Indust", "UPS": "Indust", "BA": "Indust",
    "RTX": "Indust", "DE": "Indust", "GE": "Indust", "LMT": "Indust",
    "UNP": "Indust", "FDX": "Indust",
    # Materials
    "LIN": "Mat", "APD": "Mat", "FCX": "Mat", "NEM": "Mat", "NUE": "Mat",
    # Real Estate
    "AMT": "RE", "PLD": "RE", "CCI": "RE", "EQIX": "RE", "SPG": "RE",
    # Utilities
    "NEE": "Util", "DUK": "Util", "SO": "Util", "D": "Util", "AEP": "Util",
    # Mining-ish (MARA is a BTC miner)
    "MARA": "Crypto-linked",
}

DCA_MONTHLY = 1000.0
TRADING_DAYS_YR = 252


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
@dataclass
class MarketData:
    all_dates: list            # sorted ISO date strings (fresh-spine)
    date_idx: dict             # date -> int index in all_dates
    prices: dict               # ticker -> np.ndarray aligned to all_dates (NaN if missing)
    finals: dict               # ticker -> np.ndarray aligned to all_dates (NaN if missing)
    month_first_idx: list      # indices of the first trading day each month
    bench_filled: dict         # benchmark ticker -> forward-filled price array
    stocks: list               # stock tickers (not crypto, not benchmark)
    items_by_ticker: dict      # ticker -> item dict


def load_market(full_json_path: str, bench_tickers=("SPY",)) -> MarketData:
    with open(full_json_path) as f:
        d = json.load(f)
    bt = d["bt_series"]
    items = {it["ticker"]: it for it in d["items"]}

    # 1. Determine fresh spine (drop tickers whose last_date is >365d before the max)
    max_end = ""
    for s in bt.values():
        if s["dates"]:
            last = s["dates"][-1]
            if last > max_end:
                max_end = last
    stale_cutoff = (datetime.fromisoformat(max_end) - timedelta(days=365)).date().isoformat()

    date_set = set()
    for tk, s in bt.items():
        if not s["dates"]:
            continue
        if s["dates"][-1] >= stale_cutoff:
            date_set.update(s["dates"])
    all_dates = sorted(date_set)
    date_idx = {dd: i for i, dd in enumerate(all_dates)}

    # 2. Align each ticker to the spine (NaN where missing)
    n = len(all_dates)
    prices, finals = {}, {}
    for tk, s in bt.items():
        p = np.full(n, np.nan)
        f = np.full(n, np.nan)
        for i, dd in enumerate(s["dates"]):
            if dd in date_idx:
                ix = date_idx[dd]
                p[ix] = s["prices"][i]
                f[ix] = s["final"][i]
        prices[tk] = p
        finals[tk] = f

    # 3. Benchmarks: forward-fill so stale-ending benchmarks still produce curves
    bench_filled = {}
    for tk in bench_tickers:
        if tk not in bt:
            continue
        s = bt[tk]
        # Seed from the last known price BEFORE the spine starts
        seed = 0.0
        for dd, vv in zip(s["dates"], s["prices"]):
            if dd <= all_dates[0] and math.isfinite(vv) and vv > 0:
                seed = vv
        arr = np.zeros(n)
        last = seed
        for i, dd in enumerate(all_dates):
            pm = prices[tk][i]
            if math.isfinite(pm) and pm > 0:
                last = pm
            arr[i] = last
        bench_filled[tk] = arr

    # 4. Month-first indices
    month_first_idx = []
    prev_ym = ""
    for i, dd in enumerate(all_dates):
        ym = dd[:7]
        if ym != prev_ym:
            month_first_idx.append(i)
            prev_ym = ym

    # 5. Stocks list (exclude benchmarks and crypto)
    stocks = [
        it["ticker"] for it in d["items"]
        if not it.get("is_crypto") and it["ticker"] not in set(bench_tickers)
        and it["ticker"] in bt
    ]

    return MarketData(
        all_dates=all_dates, date_idx=date_idx, prices=prices, finals=finals,
        month_first_idx=month_first_idx, bench_filled=bench_filled,
        stocks=stocks, items_by_ticker=items,
    )


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def first_valid_month_idx(md: MarketData, universe: list, min_tickers: int = 3) -> int:
    """First month-start where at least `min_tickers` have a valid price+score."""
    for m, di in enumerate(md.month_first_idx):
        n = 0
        for tk in universe:
            p = md.prices.get(tk)
            f = md.finals.get(tk)
            if p is None or f is None:
                continue
            if math.isfinite(p[di]) and p[di] > 0 and math.isfinite(f[di]) and f[di] > 0:
                n += 1
                if n >= min_tickers:
                    return m
    return 0


def spy_200sma(md: MarketData) -> np.ndarray:
    """200-day SMA of SPY aligned to the spine. NaN until 200 bars are available."""
    if "SPY" not in md.bench_filled:
        return np.full(len(md.all_dates), np.nan)
    p = md.bench_filled["SPY"]
    sma = np.full(len(p), np.nan)
    w = 200
    for i in range(len(p)):
        if i + 1 >= w:
            sma[i] = np.mean(p[i + 1 - w : i + 1])
    return sma


# ------------------------------------------------------------------
# Strategy config
# ------------------------------------------------------------------
@dataclass
class StrategyConfig:
    top_n: int = 5                        # max picks per month (used when threshold not None)
    score_threshold: Optional[float] = None  # min final_score to pick; overrides top_n as a floor
    hold_days: int = 252                  # fixed hold horizon in trading days
    weighting: str = "equal"              # "equal" | "rank" | "score"
    regime_gate: bool = False             # require SPY > 200SMA at buy
    regime_scale_down: float = 0.0        # when gate fails, scale monthly DCA by this (0 = skip)
    sector_cap: Optional[int] = None      # max picks per sector
    trail_stop: Optional[float] = None    # e.g., 0.25 for 25% trailing stop from peak
    dd_stop: Optional[float] = None       # e.g., 0.25 for 25% drawdown from cost
    min_score: float = 0.0                # absolute minimum score (filter before ranking)
    entry_delay: int = 1                  # trading days to delay entry after signal (model runs after close)
    start_month_idx: Optional[int] = None
    label: str = "strategy"


@dataclass
class SimResult:
    equity: np.ndarray
    total_invested: float
    positions: list = field(default_factory=list)  # each pos dict


def simulate(md: MarketData, universe: list, cfg: StrategyConfig) -> SimResult:
    n = len(md.all_dates)
    equity = np.zeros(n)
    positions = []
    cash = 0.0
    total_invested = 0.0

    if cfg.start_month_idx is None:
        cfg.start_month_idx = first_valid_month_idx(md, universe, min_tickers=3)

    sma = spy_200sma(md) if cfg.regime_gate else None

    for m in range(cfg.start_month_idx, len(md.month_first_idx)):
        di = md.month_first_idx[m]
        # Regime gate: if SPY below 200SMA, optionally reduce allocation.
        monthly = DCA_MONTHLY
        if cfg.regime_gate and sma is not None:
            spy_p = md.bench_filled["SPY"][di]
            if math.isfinite(sma[di]) and spy_p < sma[di]:
                monthly = DCA_MONTHLY * cfg.regime_scale_down
                if monthly <= 0:
                    continue

        # Rank universe by point-in-time final_score
        cand = []
        for tk in universe:
            f = md.finals.get(tk)
            p = md.prices.get(tk)
            if f is None or p is None:
                continue
            fv, pv = f[di], p[di]
            if not (math.isfinite(fv) and fv > cfg.min_score and math.isfinite(pv) and pv > 0):
                continue
            cand.append((tk, fv, pv))
        if not cand:
            continue
        cand.sort(key=lambda x: x[1], reverse=True)

        # Threshold OR top-N
        if cfg.score_threshold is not None:
            picks = [c for c in cand if c[1] >= cfg.score_threshold][: cfg.top_n]
            if not picks:  # fall back to best available — at least one pick
                picks = cand[:1]
        else:
            picks = cand[: cfg.top_n]

        # Sector cap
        if cfg.sector_cap is not None:
            per_sector = {}
            filtered = []
            for tk, fv, pv in picks:
                sec = SECTOR_MAP.get(tk, "Other")
                if per_sector.get(sec, 0) < cfg.sector_cap:
                    filtered.append((tk, fv, pv))
                    per_sector[sec] = per_sector.get(sec, 0) + 1
            # If cap removed picks, refill from next candidates respecting the cap
            i = len(picks)
            while len(filtered) < cfg.top_n and i < len(cand):
                tk, fv, pv = cand[i]
                i += 1
                sec = SECTOR_MAP.get(tk, "Other")
                if per_sector.get(sec, 0) < cfg.sector_cap:
                    filtered.append((tk, fv, pv))
                    per_sector[sec] = per_sector.get(sec, 0) + 1
            picks = filtered

        if not picks:
            continue

        # Weighting
        if cfg.weighting == "equal":
            weights = [1.0 / len(picks)] * len(picks)
        elif cfg.weighting == "rank":
            raw = [1.0 / (i + 1) for i in range(len(picks))]
            s = sum(raw)
            weights = [r / s for r in raw]
        elif cfg.weighting == "score":
            raw = [max(p[1], 0.0) for p in picks]
            s = sum(raw)
            weights = [r / s for r in raw] if s > 0 else [1.0 / len(picks)] * len(picks)
        else:
            raise ValueError(cfg.weighting)

        # Entry at next-day open (model runs after close). We only have daily
        # closes, so use close at di+entry_delay as the fill price — systematic
        # 1-day shift matching the live pipeline.
        entry_idx = di + cfg.entry_delay
        if entry_idx >= n:
            continue
        # Adjust each pick's fill price to the entry-day close
        adj_picks = []
        for tk, fv, _ in picks:
            ep = md.prices.get(tk)
            if ep is None:
                continue
            px = ep[entry_idx]
            if math.isfinite(px) and px > 0:
                adj_picks.append((tk, fv, px))
        if not adj_picks:
            continue
        # Re-derive weights if any were dropped
        if len(adj_picks) != len(picks):
            if cfg.weighting == "equal":
                weights = [1.0 / len(adj_picks)] * len(adj_picks)
            elif cfg.weighting == "rank":
                raw = [1.0 / (i + 1) for i in range(len(adj_picks))]
                s = sum(raw); weights = [r / s for r in raw]
            else:
                raw = [max(p[1], 0.0) for p in adj_picks]
                s = sum(raw)
                weights = [r / s for r in raw] if s > 0 else [1.0 / len(adj_picks)] * len(adj_picks)
        total_invested += monthly
        for (tk, fv, pv), w in zip(adj_picks, weights):
            alloc = monthly * w
            positions.append({
                "tk": tk, "buy_idx": entry_idx, "sell_idx": entry_idx + cfg.hold_days,
                "shares": alloc / pv, "cost": alloc, "buy_price": pv,
                "sold": False, "sell_price": 0.0, "peak": pv,
            })

    # Walk day-by-day to evolve equity
    for d in range(n):
        # trailing-stop / dd-stop / scheduled sell
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]:
                continue
            px = md.prices[pos["tk"]][d]
            if math.isfinite(px) and px > 0:
                if px > pos["peak"]:
                    pos["peak"] = px
                if cfg.trail_stop is not None and px <= pos["peak"] * (1 - cfg.trail_stop):
                    cash += pos["shares"] * px
                    pos["sold"] = True
                    pos["sell_price"] = px
                    continue
                if cfg.dd_stop is not None and px <= pos["buy_price"] * (1 - cfg.dd_stop):
                    cash += pos["shares"] * px
                    pos["sold"] = True
                    pos["sell_price"] = px
                    continue
            if d >= pos["sell_idx"] and math.isfinite(px) and px > 0:
                cash += pos["shares"] * px
                pos["sold"] = True
                pos["sell_price"] = px

        open_val = 0.0
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]:
                continue
            px = md.prices[pos["tk"]][d]
            if math.isfinite(px):
                open_val += pos["shares"] * px
        equity[d] = cash + open_val

    return SimResult(equity=equity, total_invested=total_invested, positions=positions)


def simulate_benchmark(md: MarketData, bench_tickers: list, hold_days: int,
                       start_month_idx: int) -> SimResult:
    n = len(md.all_dates)
    equity = np.zeros(n)
    positions = []
    cash = 0.0
    total_invested = 0.0
    for m in range(start_month_idx, len(md.month_first_idx)):
        di = md.month_first_idx[m]
        avail = [t for t in bench_tickers if md.bench_filled.get(t) is not None and md.bench_filled[t][di] > 0]
        if not avail:
            continue
        per = DCA_MONTHLY / len(avail)
        total_invested += DCA_MONTHLY
        for tk in avail:
            p = md.bench_filled[tk][di]
            positions.append({
                "tk": tk, "buy_idx": di, "sell_idx": di + hold_days,
                "shares": per / p, "cost": per, "buy_price": p,
                "sold": False, "sell_price": 0.0, "peak": p,
            })
    for d in range(n):
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]:
                continue
            if d >= pos["sell_idx"]:
                px = md.bench_filled[pos["tk"]][min(d, n - 1)]
                if px > 0:
                    cash += pos["shares"] * px
                    pos["sold"] = True
                    pos["sell_price"] = px
        open_val = 0.0
        for pos in positions:
            if pos["sold"] or d < pos["buy_idx"]:
                continue
            px = md.bench_filled[pos["tk"]][d]
            if math.isfinite(px):
                open_val += pos["shares"] * px
        equity[d] = cash + open_val
    return SimResult(equity=equity, total_invested=total_invested, positions=positions)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def compute_metrics(md: MarketData, eq: np.ndarray, total_invested: float) -> dict:
    if eq.size == 0 or total_invested <= 0:
        return {}
    final = float(eq[-1])
    # monthly returns
    monthly_rets = []
    for i in range(1, len(md.month_first_idx)):
        prev = eq[md.month_first_idx[i - 1]]
        cur = eq[md.month_first_idx[i]]
        if prev > 0:
            monthly_rets.append(cur / prev - 1)
    monthly_rets = np.array(monthly_rets)
    avg = monthly_rets.mean() if monthly_rets.size else 0.0
    std = monthly_rets.std() if monthly_rets.size else 0.0
    sharpe = (avg / std) * math.sqrt(12) if std > 0 else 0.0
    # max drawdown
    peak, maxdd = 0.0, 0.0
    for v in eq:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > maxdd:
                maxdd = dd
    total_return = (final - total_invested) / total_invested
    yrs = len(md.all_dates) / TRADING_DAYS_YR
    cagr = (final / total_invested) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    # Sortino
    neg = monthly_rets[monthly_rets < 0]
    dstd = neg.std() if neg.size else 0.0
    sortino = (avg / dstd) * math.sqrt(12) if dstd > 0 else 0.0
    return {
        "final": final, "invested": total_invested, "total_return": total_return,
        "cagr": cagr, "maxdd": maxdd, "sharpe": sharpe, "sortino": sortino,
        "n_months": len(monthly_rets),
    }


def fmt_pct(x):
    return f"{x*100:+.2f}%" if x is not None else "—"


def fmt_metrics(m: dict) -> str:
    return (f"CAGR {fmt_pct(m['cagr'])}  TR {fmt_pct(m['total_return'])}  "
            f"Sharpe {m['sharpe']:.2f}  Sortino {m['sortino']:.2f}  "
            f"MaxDD {fmt_pct(-m['maxdd'])}  Inv ${m['invested']:,.0f}  "
            f"Final ${m['final']:,.0f}")


# ------------------------------------------------------------------
# Main sweep
# ------------------------------------------------------------------
def run_sweep():
    md = load_market("/home/user/crt/max/docs/data/full.json")
    universe = md.stocks
    start_m = first_valid_month_idx(md, universe, min_tickers=3)

    print(f"Universe: {len(universe)} stocks | Spine: {md.all_dates[0]} → {md.all_dates[-1]}")
    print(f"Start month idx: {start_m} ({md.all_dates[md.month_first_idx[start_m]]})")
    print()

    # -------- Benchmarks (SPY DCA) ----------
    print("## Benchmarks")
    for hd, name in [(252, "SPY DCA 1Y hold"), (756, "SPY DCA 3Y hold"), (1260, "SPY DCA 5Y hold")]:
        b = simulate_benchmark(md, ["SPY"], hd, start_m)
        m = compute_metrics(md, b.equity, b.total_invested)
        print(f"  {name:25s} | {fmt_metrics(m)}")
    print()

    # -------- Baseline: top-5 equal-weight, 1Y hold (best-case naive strategy) ----------
    print("## Baseline (current-style)")
    for hd in [30, 63, 126, 252, 504, 756]:
        cfg = StrategyConfig(top_n=5, hold_days=hd, weighting="equal", start_month_idx=start_m,
                             label=f"Top-5 EW {hd}d")
        r = simulate(md, universe, cfg)
        m = compute_metrics(md, r.equity, r.total_invested)
        print(f"  Top-5 EW {hd:4d}d | {fmt_metrics(m)}")
    print()

    # -------- 1. Grid-search fixed hold with varying top-N ----------
    print("## Top-N × hold grid (equal weight)")
    best = None
    for n in [1, 3, 5, 10]:
        for hd in [30, 63, 126, 252, 504, 756]:
            cfg = StrategyConfig(top_n=n, hold_days=hd, weighting="equal", start_month_idx=start_m,
                                 label=f"Top-{n} EW {hd}d")
            r = simulate(md, universe, cfg)
            m = compute_metrics(md, r.equity, r.total_invested)
            print(f"  Top-{n:2d} {hd:4d}d | {fmt_metrics(m)}")
            if best is None or m["sharpe"] > best[0]["sharpe"]:
                best = (m, cfg)
    print(f"  >>> grid-best: {best[1].label} | {fmt_metrics(best[0])}")
    print()

    # -------- 2. Conviction weighting ----------
    print("## Weighting schemes (Top-5, 252d hold)")
    for w in ["equal", "rank", "score"]:
        cfg = StrategyConfig(top_n=5, hold_days=252, weighting=w, start_month_idx=start_m)
        r = simulate(md, universe, cfg)
        m = compute_metrics(md, r.equity, r.total_invested)
        print(f"  weight={w:5s} | {fmt_metrics(m)}")
    print()

    # -------- 3. Regime gate ----------
    print("## Regime gate (SPY 200SMA, Top-5 EW 252d)")
    for sd in [0.0, 0.25, 0.5, 1.0]:
        cfg = StrategyConfig(top_n=5, hold_days=252, weighting="equal",
                             regime_gate=True, regime_scale_down=sd,
                             start_month_idx=start_m)
        r = simulate(md, universe, cfg)
        m = compute_metrics(md, r.equity, r.total_invested)
        print(f"  gate scale-down={sd} | {fmt_metrics(m)}")
    print()

    # -------- 4. Sector cap ----------
    print("## Sector cap (Top-5 EW 252d)")
    for sc in [None, 1, 2, 3]:
        cfg = StrategyConfig(top_n=5, hold_days=252, weighting="equal",
                             sector_cap=sc, start_month_idx=start_m)
        r = simulate(md, universe, cfg)
        m = compute_metrics(md, r.equity, r.total_invested)
        print(f"  sector_cap={sc} | {fmt_metrics(m)}")
    print()

    # -------- 5. Trailing stop ----------
    print("## Trailing stop (Top-5 EW 252d)")
    for ts in [None, 0.15, 0.20, 0.25, 0.35]:
        cfg = StrategyConfig(top_n=5, hold_days=252, weighting="equal",
                             trail_stop=ts, start_month_idx=start_m)
        r = simulate(md, universe, cfg)
        m = compute_metrics(md, r.equity, r.total_invested)
        print(f"  trail={ts} | {fmt_metrics(m)}")
    print()

    # -------- 6. Threshold-based N ----------
    print("## Score threshold (dynamic N up to 10, 252d hold)")
    for thr in [None, 0.1, 0.5, 1.0, 2.0]:
        cfg = StrategyConfig(top_n=10, hold_days=252, weighting="equal",
                             score_threshold=thr, start_month_idx=start_m)
        r = simulate(md, universe, cfg)
        m = compute_metrics(md, r.equity, r.total_invested)
        print(f"  thr={thr} | {fmt_metrics(m)}")
    print()


if __name__ == "__main__":
    run_sweep()
