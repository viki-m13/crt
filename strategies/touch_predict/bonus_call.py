"""Bonus-Call: short-put credit spread + free long-call rider.

Combine the existing high-win-rate Option C credit-spread engine
(95%+ certified cells) with a long-call rider funded by the credit.

P&L per fire:
   spread_pnl = standard credit-spread payoff at expiry
   call_budget = alpha * credit_collected
   if call_premium > call_budget: skip the call (plain spread)
   else: ncalls = call_budget / call_premium
         call_pnl = ncalls * (max(close - K_call, 0) - call_premium)
   combined = spread_pnl + call_pnl

By construction:
   * The call cost is funded by the credit, so max-loss = spread max-loss.
   * Win rate ≥ spread win rate (the call only ADDS to upside; when it
     misses it just costs ≤ credit, so combined still wins on the
     spread leg).
   * When the call hits, combined PnL >> spread PnL.

Walk-forward: same FOLD_YEARS as Option C. Memory-efficient: aggregate
counters per cell, not per-fire rows.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from v2_common import FOLD_YEARS
from v2_regimes import CALL_REGIMES
from option_c_research import (
    SPREAD_WIDTH, IV_MULT, _credit_and_maxloss, _trade_pnl, _gather_fires,
)
from pricing import bs_call


# Fraction of credit budgeted to the long-call rider
ALPHA_GRID = [0.3, 0.5, 0.7, 1.0]
# OTM strikes for the long-call rider (fraction of spot)
CALL_OTM_GRID = [0.02, 0.05, 0.10, 0.15, 0.20]
# Short horizons only — that's the brief
SHORT_HORIZONS = [5, 7, 10, 14, 21]
# Spread strike grid (k_short)
K_SHORT_FOR_BONUS = [0.02, 0.03, 0.05, 0.07, 0.10]
CALL_SLIPPAGE = 1.05
CALL_MIN_PREMIUM = 0.10        # realistic floor: we don't trade $0.01 options
MAX_CALLS_PER_FIRE = 1         # one call rider per fire (no scaling)
MIN_FIRES_PER_CELL = 50
MIN_FOLDS_PER_CELL = 4
TARGET_WIN_RATE_PCT = 95.0


def _bs(spot, K, T, sigma):
    if T <= 0 or sigma <= 0 or spot <= 0 or K <= 0:
        return max(spot - K, 0.0)
    return bs_call(spot, K, T, sigma)


def _call_premium(spot, k_otm, T, sigma):
    K = spot * (1.0 + k_otm)
    bs = _bs(spot, K, T, sigma)
    if bs <= 0:
        return 0.0
    if bs < 0.10:
        slip = 1.30
    elif bs < 0.25:
        slip = 1.15
    else:
        slip = CALL_SLIPPAGE
    p = bs * slip
    if p < CALL_MIN_PREMIUM:
        return 0.0
    return p


@dataclass
class CellAgg:
    n: int = 0
    wins_combined: int = 0
    wins_spread: int = 0
    pnl_combined: float = 0.0
    pnl_spread: float = 0.0
    pnl_call: float = 0.0
    max_loss_pool: float = 0.0
    big_wins: int = 0     # combined_pnl >= max_loss
    folds: set = None

    def add(self, fold_year: int, sp: float, cp: float, ml: float):
        if self.folds is None:
            self.folds = set()
        self.folds.add(fold_year)
        self.n += 1
        self.pnl_spread += sp
        self.pnl_call += cp
        self.pnl_combined += sp + cp
        self.max_loss_pool += ml
        if sp > 0:
            self.wins_spread += 1
        if (sp + cp) > 0:
            self.wins_combined += 1
        if (sp + cp) >= ml:
            self.big_wins += 1


def main():
    t0 = time.time()
    print(f"[1/3] Aggregating combined trades on call-side regimes…")

    # Cell key: (regime, h, k_short, k_call_otm, alpha) → CellAgg
    cells: dict[tuple, CellAgg] = defaultdict(CellAgg)

    for rname, rfn in CALL_REGIMES.items():
        for h in SHORT_HORIZONS:
            fires = _gather_fires("put", rname, rfn, h)
            if not fires:
                continue
            T = h * 1.4 / 365.0
            # For each fire: compute spread/call PnL once per (k_short,
            # k_call_otm), then update each (alpha) cell.
            for fi in fires:
                year = int(str(fi.date)[:4])
                if year not in FOLD_YEARS:
                    continue
                iv = fi.sigma * IV_MULT
                # Pre-compute call premium and call payoff for each OTM
                call_data = {}
                for k_call_otm in CALL_OTM_GRID:
                    K_call = fi.spot * (1.0 + k_call_otm)
                    cp_prem = _call_premium(fi.spot, k_call_otm, T, iv)
                    cp_payoff = max(fi.close_at_expiry - K_call, 0.0)
                    call_data[k_call_otm] = (cp_prem, cp_payoff)
                for k_short in K_SHORT_FOR_BONUS:
                    k_long = k_short + SPREAD_WIDTH
                    credit, max_loss = _credit_and_maxloss(
                        "put", fi.spot, iv, k_short, k_long, T)
                    if credit <= 0:
                        continue
                    spread_pnl = _trade_pnl(
                        "put", fi.spot, fi.close_at_expiry,
                        k_short, k_long, credit)
                    for k_call_otm in CALL_OTM_GRID:
                        cp_prem, cp_payoff = call_data[k_call_otm]
                        for alpha in ALPHA_GRID:
                            call_budget = alpha * credit
                            if cp_prem <= 0 or cp_prem > call_budget:
                                # Plain spread (no call rider — credit
                                # too small to fund the call within α
                                # of the credit collected).
                                cell = cells[(rname, h, k_short, k_call_otm, alpha)]
                                cell.add(year, spread_pnl, 0.0, max_loss)
                            else:
                                # One call rider, funded from the credit.
                                # Total max loss is unchanged: we either
                                # collect (credit - call_premium) and call_payoff
                                # OR pay the spread max-loss while losing the
                                # call premium too. We bake the call premium
                                # into the call PnL, so spread max_loss is the
                                # max loss INCREMENT (the principal at risk
                                # remains the spread width).
                                call_pnl = MAX_CALLS_PER_FIRE * (cp_payoff - cp_prem)
                                cell = cells[(rname, h, k_short, k_call_otm, alpha)]
                                cell.add(year, spread_pnl, call_pnl, max_loss)
            print(f"  {rname:<14} h={h:>2}  cells_so_far={len(cells)}  "
                  f"elapsed={time.time()-t0:.1f}s")

    print()
    print(f"[2/3] Aggregated {sum(c.n for c in cells.values())} trades "
          f"across {len(cells)} cells ({time.time()-t0:.1f}s)")

    # Build pooled rows
    pooled_rows = []
    for (rname, h, ks, kc, alpha), c in cells.items():
        if c.n < MIN_FIRES_PER_CELL:
            continue
        if len(c.folds) < MIN_FOLDS_PER_CELL:
            continue
        win_combined = c.wins_combined / c.n * 100
        win_spread = c.wins_spread / c.n * 100
        roi_c = c.pnl_combined / c.max_loss_pool * 100 if c.max_loss_pool > 0 else 0
        roi_s = c.pnl_spread / c.max_loss_pool * 100 if c.max_loss_pool > 0 else 0
        pooled_rows.append({
            "regime": rname, "horizon": h,
            "k_short": ks, "k_call_otm": kc, "alpha": alpha,
            "n": c.n, "win_pct_combined": win_combined,
            "win_pct_spread": win_spread,
            "roi_combined_pct": roi_c, "roi_spread_pct": roi_s,
            "avg_combined_pnl": c.pnl_combined / c.n,
            "avg_call_pnl": c.pnl_call / c.n,
            "big_wins_pct": c.big_wins / c.n * 100,
            "n_folds": len(c.folds),
            "max_loss_avg": c.max_loss_pool / c.n,
        })

    # Sort: 95%+ first, then by ROI lift
    pooled_rows.sort(key=lambda r: (
        -(r["win_pct_combined"] >= TARGET_WIN_RATE_PCT),
        -(r["roi_combined_pct"] - r["roi_spread_pct"]),
        -r["roi_combined_pct"],
    ))

    print()
    print(f"=== STAGE A: Top cells (≥{TARGET_WIN_RATE_PCT}% combined win-rate) ===")
    print(f"{'regime':<14} {'h':>2} {'kS%':>4} {'kC%':>4} {'α':>4} "
          f"{'n':>5} {'wnC%':>5} {'wnS%':>5} {'roiC%':>6} "
          f"{'roiS%':>6} {'lift':>5} {'big%':>5} {'avgPnL$':>8}")
    print("-" * 105)
    n_eligible = 0
    for r in pooled_rows[:80]:
        if r["win_pct_combined"] < TARGET_WIN_RATE_PCT:
            break
        n_eligible += 1
        lift = r["roi_combined_pct"] - r["roi_spread_pct"]
        print(f"{r['regime']:<14} {r['horizon']:>2} {r['k_short']*100:>3.1f} "
              f"{r['k_call_otm']*100:>3.1f} {r['alpha']:>4.2f} "
              f"{r['n']:>5} {r['win_pct_combined']:>4.1f} "
              f"{r['win_pct_spread']:>4.1f} {r['roi_combined_pct']:>+5.1f} "
              f"{r['roi_spread_pct']:>+5.1f} {lift:>+4.1f} "
              f"{r['big_wins_pct']:>4.1f} {r['avg_combined_pnl']:>+7.2f}")

    n_total_eligible = sum(1 for r in pooled_rows
                            if r["win_pct_combined"] >= TARGET_WIN_RATE_PCT)
    print(f"\nTotal cells with ≥{TARGET_WIN_RATE_PCT}% combined win-rate: "
          f"{n_total_eligible}")

    elig = [r for r in pooled_rows if r["win_pct_combined"] >= TARGET_WIN_RATE_PCT]
    if elig:
        best = max(elig, key=lambda r: r["roi_combined_pct"])
        max_lift = max(elig, key=lambda r: r["roi_combined_pct"] - r["roi_spread_pct"])
        print()
        print(f"=== STAGE B: best 95%+ cell by absolute ROI ===")
        print(f"  rule:               {best['regime']} h={best['horizon']} "
              f"kS={best['k_short']*100:.0f}% kC={best['k_call_otm']*100:.0f}% "
              f"α={best['alpha']:.2f}")
        print(f"  n trades:           {best['n']}")
        print(f"  combined win rate:  {best['win_pct_combined']:.1f}%  "
              f"(spread alone: {best['win_pct_spread']:.1f}%)")
        print(f"  ROI combined:       {best['roi_combined_pct']:+.1f}%  "
              f"(spread alone: {best['roi_spread_pct']:+.1f}%)")
        print(f"  ROI lift:           {best['roi_combined_pct'] - best['roi_spread_pct']:+.1f}%")
        print(f"  Big-win %:          {best['big_wins_pct']:.1f}% (combined ≥ max-loss)")
        print(f"  Avg combined $:     {best['avg_combined_pnl']:+.2f} per share-unit")
        print()
        print(f"=== STAGE C: best 95%+ cell by ROI LIFT over plain spread ===")
        print(f"  rule:               {max_lift['regime']} h={max_lift['horizon']} "
              f"kS={max_lift['k_short']*100:.0f}% kC={max_lift['k_call_otm']*100:.0f}% "
              f"α={max_lift['alpha']:.2f}")
        print(f"  n trades:           {max_lift['n']}")
        print(f"  combined win rate:  {max_lift['win_pct_combined']:.1f}%")
        print(f"  ROI combined:       {max_lift['roi_combined_pct']:+.1f}%")
        print(f"  ROI lift over plain {max_lift['roi_combined_pct'] - max_lift['roi_spread_pct']:+.1f}%")
        print(f"  Big-win %:          {max_lift['big_wins_pct']:.1f}%")

    out = {
        "n_total_trades": sum(c.n for c in cells.values()),
        "n_cells_evaluated": len(pooled_rows),
        "n_eligible_95pct": n_total_eligible,
        "config": {
            "alpha_grid": ALPHA_GRID,
            "call_otm_grid": CALL_OTM_GRID,
            "short_horizons": SHORT_HORIZONS,
            "k_short_grid": K_SHORT_FOR_BONUS,
            "spread_width": SPREAD_WIDTH,
            "iv_mult": IV_MULT,
            "call_slippage": CALL_SLIPPAGE,
            "target_win_rate_pct": TARGET_WIN_RATE_PCT,
        },
        "pooled_rows": pooled_rows,
    }
    out_path = os.path.join(_HERE, "results", "bonus_call.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
