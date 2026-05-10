"""Analyse the chosen v6 winner: per-split WF, year-by-year, drawdown ledger,
turnover, picks distribution, sub-period stability, bias overlay."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lib_engine import (
    V2, WF_SPLITS, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def yearly(eq: pd.DataFrame, spy_aligned: pd.DataFrame) -> pd.DataFrame:
    eq = eq.copy()
    eq["year"] = pd.to_datetime(eq["date"]).dt.year
    yr = eq.groupby("year")["ret_m"].apply(lambda x: (1 + x).prod() - 1).rename("year_ret")
    sp = spy_aligned.copy()
    sp["year"] = pd.to_datetime(sp["date"]).dt.year
    sy = sp.groupby("year")["spy_ret_m"].apply(lambda x: (1 + x).prod() - 1).rename("spy_year_ret")
    out = yr.to_frame().join(sy.to_frame(), how="left")
    out["edge_pp"] = (out["year_ret"] - out["spy_year_ret"]) * 100
    return out


def drawdowns(eq: pd.DataFrame, threshold: float = -0.05) -> pd.DataFrame:
    s = pd.Series(eq["equity"].values, index=pd.DatetimeIndex(eq["date"]))
    peak = s.cummax()
    dd = (s - peak) / peak
    episodes, in_dd, start, depth, trough = [], False, None, 0, None
    for d, v in dd.items():
        if not in_dd and v < threshold:
            in_dd, start, depth, trough = True, d, v, d
        elif in_dd:
            if v < depth:
                depth, trough = v, d
            if v >= -0.001:
                episodes.append({"start": start, "trough": trough, "end": d, "depth_pct": depth*100})
                in_dd = False
    if in_dd:
        episodes.append({"start": start, "trough": trough, "end": s.index[-1], "depth_pct": depth*100})
    return pd.DataFrame(episodes).sort_values("depth_pct")


def per_split(eq: pd.DataFrame, spy_aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy()
        if len(e) == 0:
            continue
        r = e["ret_m"].astype(float)
        ec = (1 + r).cumprod()
        cagr = ec.iloc[-1] ** (12.0 / len(ec)) - 1
        sh = (r.mean() / max(r.std(), 1e-9)) * np.sqrt(12)
        peak = ec.cummax()
        mdd = float(((ec - peak) / peak).min())
        spy = spy_aligned[(spy_aligned["date"] >= lo) & (spy_aligned["date"] <= hi)]
        sr = spy["spy_ret_m"].astype(float)
        sc = (1 + sr).cumprod()
        scagr = sc.iloc[-1] ** (12.0 / len(sc)) - 1
        rows.append({
            "split": split, "from": lo.date(), "to": hi.date(), "n_m": len(e),
            "cagr_pct": cagr * 100, "spy_cagr_pct": scagr * 100,
            "edge_pp": (cagr - scagr) * 100, "sharpe": sh,
            "max_dd_pct": mdd * 100,
        })
    return pd.DataFrame(rows)


def turnover(eq: pd.DataFrame) -> dict:
    last, sames = [], []
    for _, r in eq.iterrows():
        cur = str(r.get("picks", "") or "")
        cur_set = set(cur.split(",")) if cur else set()
        if last and cur_set:
            sames.append(len(set(last) & cur_set) / max(len(cur_set), 1))
        last = list(cur_set)
    s = np.array(sames)
    return {
        "n_pairs": int(len(s)),
        "mean_overlap": float(s.mean()) if len(s) else 0.0,
        "approx_annl_turnover": float((1 - s.mean()) * 12) if len(s) else 0.0,
    }


def most_picked(eq: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    counts = {}
    for picks in eq["picks"].dropna():
        for tk in str(picks).split(","):
            if tk:
                counts[tk] = counts.get(tk, 0) + 1
    return (pd.DataFrame(counts.items(), columns=["ticker", "n_months_picked"])
            .sort_values("n_months_picked", ascending=False).head(top_n))


def bias_overlay(eq: pd.DataFrame, monthly_returns: pd.DataFrame,
                 alphas=(0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20),
                 n_iters: int = 30, cost_bps: float = 10.0,
                 seed: int = 42) -> pd.DataFrame:
    rng_master = np.random.default_rng(seed)
    rows = []
    per_month = []
    for _, r in eq.iterrows():
        d = pd.Timestamp(r["date"])
        if r.get("regime", "active") == "cash" or not r.get("picks"):
            per_month.append({"date": d, "active": False, "rets": [], "weights": []})
            continue
        picks = str(r["picks"]).split(",")
        pos = monthly_returns.index.searchsorted(d)
        cands = []
        for j in (pos - 1, pos):
            if 0 <= j < len(monthly_returns.index):
                cands.append((j, abs((monthly_returns.index[j] - d).days)))
        cands.sort(key=lambda x: x[1])
        if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(monthly_returns.index):
            per_month.append({"date": d, "active": False, "rets": [], "weights": []})
            continue
        next_d = monthly_returns.index[cands[0][0] + 1]
        rets = []
        for tk in picks:
            if tk in monthly_returns.columns:
                rr = monthly_returns.at[next_d, tk]
                rets.append(-1.0 if pd.isna(rr) else float(rr))
            else:
                rets.append(-1.0)
        # Use the actual recorded weights when available
        wcsv = str(r.get("weights_csv", "") or "")
        if wcsv and len(wcsv.split(",")) == len(rets):
            weights = [float(x) for x in wcsv.split(",")]
        else:
            w = float(r.get("gross", 1.0))
            n = len(rets)
            weights = [w / n] * n
        per_month.append({"date": d, "active": True, "rets": rets, "weights": weights})

    cf = cost_bps / 10000.0
    # Identify rebalance months from the equity dataframe by detecting changes
    # in pick set (cost is charged only at rebalance, matching the simulator).
    is_reb = []
    last_picks = None
    for m in per_month:
        if not m["active"]:
            is_reb.append(False)
            last_picks = None
            continue
        cur = tuple(sorted(m.get("date", "")) ) if False else tuple()  # placeholder
        # We use date plus pick identity recovered from rets length proxy
        is_reb.append(last_picks != m.get("rets"))
        last_picks = m.get("rets")
    # Easier: replicate simulator behaviour — rebalance every hold_months.
    # Approximate by treating every month where weights differ from prior as rebalance.
    for alpha in alphas:
        p_month = 1 - (1 - alpha) ** (1/12) if alpha > 0 else 0
        finals = []
        for it in range(n_iters):
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            equity = 1.0
            prev_w = None
            for m in per_month:
                if not m["active"] or not m["rets"]:
                    prev_w = None
                    continue
                rets = np.array(m["rets"], dtype=float)
                weights = np.array(m["weights"], dtype=float)
                if p_month > 0:
                    wipe = rng.random(len(rets)) < p_month
                    rets[wipe] = -1.0
                ret_m = float((rets * weights).sum())
                # Charge cost only on rebalance (when weights or pick set change)
                rebalance = (prev_w is None or len(prev_w) != len(weights)
                             or not np.allclose(prev_w, weights, atol=1e-6))
                if rebalance:
                    equity *= (1 + ret_m) * (1 - cf * float(weights.sum()))
                else:
                    equity *= (1 + ret_m)
                prev_w = weights.copy()
            finals.append(equity)
        finals = np.array(finals)
        years = len(per_month) / 12.0
        cagrs = finals ** (1/years) - 1
        rows.append({
            "alpha_yr": alpha,
            "p10": float(np.percentile(cagrs, 10) * 100),
            "median": float(np.median(cagrs) * 100),
            "p90": float(np.percentile(cagrs, 90) * 100),
            "mean": float(np.mean(cagrs) * 100),
            "n_iters": n_iters,
        })
    return pd.DataFrame(rows)


def main(cfg: V6Config, label: str):
    panel = load_score_panel(cfg.scorer, cfg.universe, attach_pullback=True)
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()

    eq = simulate(cfg, panel, monthly_returns, spy_feats)
    spy_aligned = build_spy_aligned(eq, monthly_returns)
    metrics = evaluate(eq, spy_aligned, cfg.name)

    print(f"\n=== {label} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save artifacts
    eq.to_csv(OUT / f"{label}_equity.csv", index=False)
    (OUT / f"{label}_metrics.json").write_text(json.dumps(metrics, indent=2))

    sp = per_split(eq, spy_aligned)
    sp.to_csv(OUT / f"{label}_per_split.csv", index=False)
    print("\n[per-split walk-forward]")
    print(sp.round(3).to_string(index=False))

    yr = yearly(eq, spy_aligned)
    yr["Strategy_pct"] = (yr["year_ret"] * 100).round(1)
    yr["SPY_pct"] = (yr["spy_year_ret"] * 100).round(1)
    yr["edge_pp_r"] = yr["edge_pp"].round(1)
    yr.to_csv(OUT / f"{label}_yearly.csv")
    print("\n[year-by-year]")
    print(yr[["Strategy_pct", "SPY_pct", "edge_pp_r"]].to_string())

    dd = drawdowns(eq).head(10)
    dd.to_csv(OUT / f"{label}_drawdowns.csv", index=False)
    print("\n[top drawdowns]")
    print(dd.round(2).to_string(index=False))

    to = turnover(eq)
    print(f"\n[turnover] {to}")

    mp = most_picked(eq, 20)
    mp.to_csv(OUT / f"{label}_most_picked.csv", index=False)
    print("\n[most picked]")
    print(mp.to_string(index=False))

    print("\n[bias overlay]")
    bo = bias_overlay(eq, monthly_returns)
    bo.to_csv(OUT / f"{label}_bias.csv", index=False)
    print(bo.round(2).to_string(index=False))


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "v6_winner"
    cfg_kwargs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    cfg = V6Config(**cfg_kwargs)
    main(cfg, label)
