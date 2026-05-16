"""Investor chart: the MN drawdown-switch Pareto-dominates the static
60/40 blend at 3-5y DCA horizons (better worst case AND better median)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUG = Path(__file__).resolve().parents[2] / "cache" / "v2" / "sp500_pit" / "augmented"
o = json.load(open(AUG / "novel_v6_mn_overlay.json"))["horizons"]
yrs = [1, 3, 5, 10]
H = ["H12", "H36", "H60", "H120"]


def series(name, key):
    return [o[h][name][key] for h in H]


fig, ax = plt.subplots(1, 2, figsize=(13, 5))
cfg = [("v5_only", "v5 picker (parabolic)", "#1a7f37", "o-"),
       ("static_60_40", "static 60/40 v5+SPY", "#57606a", "s--"),
       ("switch_TH25", "MN drawdown-switch (new)", "#0969da", "D-")]

for name, lab, c, ls in cfg:
    ax[0].plot(yrs, series(name, "min_moic"), ls, color=c, lw=2.3, label=lab)
ax[0].axhline(1.0, color="grey", ls=":", lw=1)
ax[0].set_title("WORST-CASE outcome by DCA horizon\n(min terminal ÷ money in, across all windows)")
ax[0].set_xlabel("DCA horizon (years)"); ax[0].set_ylabel("worst MOIC (×)")
ax[0].set_xticks(yrs); ax[0].legend(loc="upper left"); ax[0].grid(alpha=0.3)

for name, lab, c, ls in cfg:
    ax[1].plot(yrs, series(name, "median_moic"), ls, color=c, lw=2.3, label=lab)
ax[1].set_title("MEDIAN outcome by DCA horizon\n(higher = more upside kept)")
ax[1].set_xlabel("DCA horizon (years)"); ax[1].set_ylabel("median MOIC (×)")
ax[1].set_xticks(yrs); ax[1].legend(loc="upper left"); ax[1].grid(alpha=0.3)

fig.suptitle("Downside lever that works: drawdown-conditional rotation into the "
             "validated market-neutral sleeve\n(PIT data 2003-2026, threshold a-priori, "
             "not optimized) — Pareto-beats static 60/40 at 3-5y", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = AUG / "novel_v6_overlay.png"
fig.savefig(out, dpi=130)
print("wrote", out)
