#!/usr/bin/env python3
"""Honest multi-leg EOD options backtest engine.

Rules (pre-registered in RESEARCH_LOG.md):
- fills at WORST side of the EOD spread: buy at ask, sell at bid
- legs with bid<=0 cannot be sold (no fill); crossed quotes dropped
- expiry settlement at intrinsic vs parity-extracted spot
- portfolio return series on the observation calendar; headline Sharpe from
  weekly-resampled equity (sqrt(52)); block-bootstrap CI; deflated Sharpe
"""
from __future__ import annotations

import dataclasses
import glob
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CHAINS_DIR = os.path.join(HERE, "cache", "chains")
VOL_DIR = os.path.join(HERE, "cache", "vol")


# ---------------------------------------------------------------- loading

CHAINS2_DIR = os.path.join(HERE, "cache", "chains2")


def available_dates() -> list[str]:
    return sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(CHAINS_DIR, "*.parquet")))


import functools


@functools.lru_cache(maxsize=64)
def load_chain(day: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(CHAINS_DIR, f"{day}.parquet"))
    p2 = os.path.join(CHAINS2_DIR, f"{day}.parquet")
    if os.path.exists(p2):
        df = pd.concat([df, pd.read_parquet(p2)], ignore_index=True)
    df["mid"] = (df.bid + df.ask) / 2.0
    df["spread"] = df.ask - df.bid
    # drop absurd quotes: crossed, negative, ask==0
    df = df[(df.ask > 0) & (df.spread >= 0)]
    df["dte"] = (pd.to_datetime(df.expiration) - pd.to_datetime(df.date)).dt.days
    return df


def load_vol(day: str) -> pd.DataFrame | None:
    ps = [os.path.join(VOL_DIR, f"{day}.parquet"),
          os.path.join(HERE, "cache", "vol2", f"{day}.parquet")]
    ps = [p for p in ps if os.path.exists(p)]
    if not ps:
        return None
    v = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True)
    for c in v.columns:
        if c.startswith(("hv", "iv")) and not c.endswith("date"):
            v[c] = pd.to_numeric(v[c], errors="coerce")
    return v


# ---------------------------------------------------------------- parity spot

def parity_spots(chain: pd.DataFrame) -> dict[str, float]:
    """Extract spot per symbol via put-call parity on near-ATM pairs."""
    out: dict[str, float] = {}
    for sym, g in chain.groupby("act_symbol"):
        g = g[(g.dte >= 5) & (g.dte <= 70) & (g.bid > 0)]
        if g.empty:
            continue
        exp = g[g.dte == g.dte.min()].expiration.iloc[0]
        ge = g[g.expiration == exp]
        c = ge[ge.call_put == "Call"].set_index("strike").mid
        p = ge[ge.call_put == "Put"].set_index("strike").mid
        ks = c.index.intersection(p.index)
        if len(ks) < 2:
            continue
        # ATM ≈ where |C-P| smallest; take median parity spot over 5 nearest
        diff = (c[ks] - p[ks]).abs().sort_values()
        near = diff.index[:5]
        spots = [c[k] - p[k] + float(k) for k in near]
        out[sym] = float(np.median(spots))
    return out


def build_spot_panel(dates: list[str]) -> pd.DataFrame:
    rows = []
    for d in dates:
        ch = load_chain(d)
        for sym, s in parity_spots(ch).items():
            rows.append((d, sym, s))
    sp = pd.DataFrame(rows, columns=["date", "act_symbol", "spot"])
    return sp.pivot(index="date", columns="act_symbol", values="spot").sort_index()


# T-bill (3m) approx annual yields by year — cash collateral earns this; the
# same table is subtracted in Sharpe. Source: FRED TB3MS yearly averages.
RF_BY_YEAR = {2019: 0.021, 2020: 0.004, 2021: 0.0004, 2022: 0.020,
              2023: 0.0503, 2024: 0.052, 2025: 0.043, 2026: 0.040}


def rf_at(ts: pd.Timestamp) -> float:
    return RF_BY_YEAR.get(ts.year, 0.03)


# ---------------------------------------------------------------- positions

@dataclasses.dataclass
class Leg:
    symbol: str
    expiration: str
    strike: float
    cp: str          # "Call"/"Put"
    qty: int         # contracts, signed (+long); 100 multiplier


@dataclasses.dataclass
class Fill:
    leg: Leg
    price: float     # per share
    day: str


class Quotes:
    """Fast quote lookup for one observation date."""

    def __init__(self, chain: pd.DataFrame):
        self.tbl = {}
        for r in chain.itertuples(index=False):
            self.tbl[(r.act_symbol, r.expiration, float(r.strike), r.call_put)] = (r.bid, r.ask)

    def get(self, leg: Leg):
        return self.tbl.get((leg.symbol, leg.expiration, float(leg.strike), leg.cp))


def exec_price(quote: tuple[float, float], side: int) -> float | None:
    """side +1 = buy (pay ask), -1 = sell (receive bid; needs bid>0)."""
    bid, ask = quote
    if side > 0:
        return float(ask) if ask > 0 else None
    return float(bid) if bid > 0 else None


class Backtester:
    """Event loop over the observation calendar.

    strategy(day, chain, quotes, spots, vol, state, engine) -> list[Leg] orders.
    Orders execute at worst side immediately. Engine marks portfolio at
    liquidation value each obs date, settles expiries at intrinsic.
    """

    def __init__(self, dates: list[str], capital: float = 1_000_000.0,
                 stock_hedge: bool = False, stock_cost_bps: float = 2.0):
        self.dates = dates
        self.capital = capital
        self.stock_hedge = stock_hedge
        self.stock_cost_bps = stock_cost_bps

    def run(self, strategy, spot_panel: pd.DataFrame, verbose: bool = False):
        cash = self.capital
        positions: dict[tuple, int] = defaultdict(int)   # leg key -> qty
        stock_pos: dict[str, float] = defaultdict(float)  # shares
        equity_rows = []
        state: dict = {}
        fills: list[Fill] = []
        last_spots: dict[str, float] = {}

        prev_ts = None
        for day in self.dates:
            ts = pd.Timestamp(day)
            if prev_ts is not None and cash > 0:
                cash *= 1.0 + rf_at(ts) * (ts - prev_ts).days / 365.0
            prev_ts = ts
            chain = load_chain(day)
            quotes = Quotes(chain)
            spots = spot_panel.loc[day].dropna().to_dict() if day in spot_panel.index else {}
            last_spots.update(spots)
            vol = load_vol(day)

            # 1) settle expired legs at intrinsic (expiration <= day)
            for key in [k for k, q0 in positions.items() if k[1] <= day and q0 != 0]:
                sym, exp, k_, cp = key
                qty = positions.pop(key)
                s = last_spots.get(sym)
                if s is None:
                    continue
                intr = max(s - k_, 0.0) if cp == "Call" else max(k_ - s, 0.0)
                cash += qty * intr * 100.0

            # 2) strategy orders
            orders = strategy(day, chain, quotes, spots, vol, state, self)
            for leg in orders or []:
                if leg.qty == 0:
                    continue
                qt = quotes.get(leg)
                if qt is None:
                    continue
                px = exec_price(qt, 1 if leg.qty > 0 else -1)
                if px is None:
                    continue
                key = (leg.symbol, leg.expiration, float(leg.strike), leg.cp)
                positions[key] += leg.qty
                cash -= leg.qty * px * 100.0
                fills.append(Fill(leg, px, day))

            # 3) optional delta hedge with stock at parity spot (costs bps)
            if self.stock_hedge and "hedge_targets" in state:
                for sym, tgt in state["hedge_targets"].items():
                    s = spots.get(sym)
                    if s is None:
                        continue
                    dlt = tgt - stock_pos[sym]
                    if abs(dlt) * s < 1.0:
                        continue
                    cash -= dlt * s
                    cash -= abs(dlt) * s * self.stock_cost_bps / 1e4
                    stock_pos[sym] += dlt

            # 4) mark portfolio at liquidation value
            mv = 0.0
            for key, qty in positions.items():
                if qty == 0:
                    continue
                sym, exp, k_, cp = key
                qt = quotes.get(Leg(sym, exp, k_, cp, 0))
                if qt is not None:
                    px = qt[0] if qty > 0 else qt[1]  # long→bid, short→ask
                    if qty < 0 and px == 0:
                        px = qt[1]
                    mv += qty * px * 100.0
                else:
                    s = last_spots.get(sym)
                    if s is not None:
                        intr = max(s - k_, 0.0) if cp == "Call" else max(k_ - s, 0.0)
                        mv += qty * intr * 100.0
            for sym, sh in stock_pos.items():
                s = last_spots.get(sym)
                if s is not None:
                    mv += sh * s
            equity_rows.append((day, cash + mv))
            if verbose:
                print(day, f"eq={cash+mv:,.0f} pos={sum(1 for v in positions.values() if v)}")

        eq = pd.Series(dict(equity_rows), name="equity")
        eq.index = pd.to_datetime(eq.index)
        return eq, fills, state


# ---------------------------------------------------------------- metrics

def weekly_sharpe(eq: pd.Series, rf_annual: float | None = None) -> dict:
    w = eq.resample("W-FRI").last().dropna()
    r = w.pct_change().dropna()
    if len(r) < 20 or r.std() == 0:
        return {"sharpe": np.nan, "n": len(r)}
    if rf_annual is None:
        rf_w = pd.Series([rf_at(t) for t in r.index], index=r.index) / 52.0
    else:
        rf_w = rf_annual / 52.0
    ex = r - rf_w
    sh = ex.mean() / r.std() * math.sqrt(52.0)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, 1e-9)) - 1
    dd = (eq / eq.cummax() - 1).min()
    return {"sharpe": float(sh), "cagr": float(cagr), "maxdd": float(dd),
            "n_weeks": len(r), "vol_ann": float(r.std() * math.sqrt(52)),
            "years": years}


def block_bootstrap_ci(eq: pd.Series, n_boot: int = 2000, block: int = 8,
                       rf_annual: float = 0.03, seed: int = 7) -> tuple[float, float]:
    w = eq.resample("W-FRI").last().dropna()
    rr = w.pct_change().dropna()
    r = (rr - pd.Series([rf_at(t) for t in rr.index], index=rr.index) / 52.0).values
    rng = np.random.default_rng(seed)
    n = len(r)
    out = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, n)
            L = rng.geometric(1.0 / block)
            idx.extend(((start + np.arange(L)) % n).tolist())
        rr = r[np.array(idx[:n])]
        if rr.std() > 0:
            out.append(rr.mean() / rr.std() * math.sqrt(52.0))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def deflated_sharpe(sh_hat: float, n_obs: int, n_trials: int,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """PSR vs the expected max Sharpe of n_trials of noise (Bailey/LdP)."""
    from scipy.stats import norm
    if n_trials < 1:
        n_trials = 1
    emc = 0.5772156649
    var_sh = 1.0
    sr0 = math.sqrt(var_sh) * ((1 - emc) * norm.ppf(1 - 1.0 / n_trials)
                               + emc * norm.ppf(1 - 1.0 / (n_trials * math.e)))
    sr0 /= math.sqrt(52.0)  # weekly units
    sh_w = sh_hat / math.sqrt(52.0)
    denom = math.sqrt(max(1e-12, (1 - skew * sh_w + (kurt - 1) / 4.0 * sh_w ** 2) / (n_obs - 1)))
    return float(norm.cdf((sh_w - sr0) / denom))
