"""Downside protection: is the strategy better than buy-and-hold SPY in
prolonged drawdowns — including ones that HAVEN'T happened?

1. Historical bear windows: strategy vs SPY over each major drawdown.
2. Synthetic stress paths appended to real history (so the 200dma/IV
   state is warm and realistic at stress onset):
   a. JAPAN-STYLE: -60% over ~3y, then 7y flat/sideways (Nikkei 1990s).
   b. LOST DECADE: block-bootstrapped real returns re-drifted to -5%/yr
      for 10y (vol clustering preserved), several seeds.
   c. WHIPSAW GRIND: adversarial — repeated legs of -20% then +16%
      rallies that cross back above the 200dma (re-arming entries at
      local tops) for 8 years. Designed to attack the regime filter.
Strategy = the shipped put ladder (v2 pricing) run on each path.
"""
import importlib.util
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("spxsig", os.path.join(HERE, "signal.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

real = json.load(open(S.SPY_PATH))
rdates = np.array(real['series']['dates'], dtype='datetime64[D]')
rpx = np.array(real['series']['prices'], float)
rlr = np.diff(np.log(rpx))

SCRATCH = os.path.join(HERE, 'results')

def run_path(dates, px, stress_start_idx, tag):
    """Run the shipped put ladder on (dates,px); report strategy vs B&H
    over the stress window only."""
    p = os.path.join(SCRATCH, 'syn_' + ''.join(c if c.isalnum() else '_' for c in tag) + '.json')
    json.dump({"ticker": "SYN", "series": {"dates": [str(d) for d in dates],
               "prices": list(map(float, px))}}, open(p, 'w'))
    mkt = S.Market(p)
    trades, opens, curve = S.simulate_ladder(mkt, S.STRUCTURES['put'])
    # equity at/after stress start
    d0 = str(dates[stress_start_idx])
    cd = [c for c in curve if c[0] >= d0]
    eq0 = cd[0][1] if cd else 1.0
    eqv = np.array([c[1] for c in cd]) / eq0 if cd else np.array([1.0])
    dd = float((eqv/np.maximum.accumulate(eqv)-1).min())
    strat_ret = float(eqv[-1] - 1)
    bh_ret = float(px[-1]/px[stress_start_idx] - 1)
    bh_eq = px[stress_start_idx:]/px[stress_start_idx]
    bh_dd = float((bh_eq/np.maximum.accumulate(bh_eq)-1).min())
    yrs = (dates[-1]-dates[stress_start_idx]).astype('timedelta64[D]').astype(int)/365.25
    n_open_end = len(opens)
    print(f"{tag:26s} ({yrs:.1f}y): strategy {strat_ret*100:+7.1f}% (maxDD {dd*100:4.0f}%)  "
          f"vs SPY B&H {bh_ret*100:+7.1f}% (maxDD {bh_dd*100:4.0f}%)")
    return strat_ret, bh_ret

# ---------- 1. HISTORICAL bear windows (real path, real ladder) ----------
print("=== Historical drawdown windows: strategy vs SPY buy-and-hold ===")
mkt = S.Market(S.SPY_PATH)
trades, opens, curve = S.simulate_ladder(mkt, S.STRUCTURES['put'])
cdates = np.array([c[0] for c in curve]); cvals = np.array([c[1] for c in curve])
def window(tag, a, b):
    ca = cvals[cdates >= a]; da = cdates[cdates >= a]
    cw = ca[da <= b]
    if len(cw) < 2:
        print(f"  {tag:28s}: no closed trades in window"); return
    sr = cw[-1]/cw[0]-1
    ia, ib = int(np.searchsorted(rdates, np.datetime64(a))), int(np.searchsorted(rdates, np.datetime64(b)))
    br = rpx[min(ib, len(rpx)-1)]/rpx[ia]-1
    print(f"  {tag:28s}: strategy {sr*100:+7.1f}%   SPY {br*100:+7.1f}%")
for tag, a, b in [("dot-com bear 2000-09..2002-10", "2000-09-01", "2002-10-09"),
                  ("GFC 2007-10..2009-03", "2007-10-09", "2009-03-09"),
                  ("COVID 2020-02..2020-03", "2020-02-19", "2020-03-23"),
                  ("2022 bear 2022-01..2022-10", "2022-01-03", "2022-10-12")]:
    window(tag, a, b)

# ---------- 2. SYNTHETIC stress paths ----------
print("\n=== Synthetic stress (appended after real history to 2019-12-31) ===")
cut = int(np.searchsorted(rdates, np.datetime64('2020-01-01')))
base_d, base_p = rdates[:cut], rpx[:cut]

def bd_range(start, n):
    """n business days from start (approx: skip weekends)."""
    out = []; d = start
    while len(out) < n:
        d = d + np.timedelta64(1, 'D')
        if d.astype('datetime64[D]').astype(int) % 7 not in (3, 4):  # crude weekend skip
            out.append(d)
    return np.array(out, dtype='datetime64[D]')

def append_path(lrs):
    nd = bd_range(base_d[-1], len(lrs))
    npx = base_p[-1]*np.exp(np.cumsum(lrs))
    return np.concatenate([base_d, nd]), np.concatenate([base_p, npx])

rng = np.random.RandomState(7)

# (a) Japan-style: -60% over 3y then 7y sideways (vol from 2000-02 bear blocks)
bear_lr = rlr[int(np.searchsorted(rdates,np.datetime64('2000-09-01'))):
              int(np.searchsorted(rdates,np.datetime64('2002-10-09')))]
flat_lr = rlr[int(np.searchsorted(rdates,np.datetime64('2004-01-01'))):
              int(np.searchsorted(rdates,np.datetime64('2007-01-01')))]
def blocks(pool, n, drift_target, rng, block=63):
    out = []
    while len(out) < n:
        j = rng.randint(0, len(pool)-block)
        out.extend(pool[j:j+block])
    out = np.array(out[:n])
    out = out - out.mean() + drift_target/252.0
    return out
japan = np.concatenate([blocks(bear_lr, 756, math.log(0.40)/3.0, rng),
                        blocks(flat_lr, 1764, 0.0, rng)])
d1, p1 = append_path(japan)
run_path(d1, p1, cut, "JAPAN -60% then flat 7y")

# (b) Lost decade: bootstrapped from FULL history, re-drifted to -5%/yr, 10y
for seed in (1, 2, 3):
    r2 = np.random.RandomState(seed)
    lost = blocks(rlr, 2520, math.log(0.60)/10.0, r2)
    d2, p2 = append_path(lost)
    run_path(d2, p2, cut, f"LOST DECADE -5%/yr s{seed}")

# (c) Whipsaw grind: 6m leg -20%, then 3m rally +16% (recrosses 200dma), x8y
legs = []
for k in range(11):
    legs.append(blocks(bear_lr, 126, math.log(0.80)*2.0, rng))   # -20% over 6m
    legs.append(blocks(flat_lr, 63, math.log(1.16)*4.0, rng))    # +16% over 3m
whip = np.concatenate(legs)
d3, p3 = append_path(whip)
run_path(d3, p3, cut, "WHIPSAW grind ~8y")
