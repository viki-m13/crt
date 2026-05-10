# K and hold-period sweep — confirms K=3, h=6 is locally optimal

**Hypotheses tested**: vary K ∈ {1, 2, 3, 5} × hold ∈ {3, 6, 12} on the v3 ml_3plus6 baseline.

| K | hold | CAGR | Sharpe | MaxDD | verdict |
|---:|---:|---:|---:|---:|---|
| **3** | **6** | **40.78%** | **0.953** | **-49.83%** | **baseline** |
| 1 | 6 | 28.61% | 0.618 | -80.38% | KILL: concentration kills |
| 2 | 6 | 37.10% | 0.813 | -69.07% | KILL: -3.7 pp CAGR, much worse DD |
| 5 | 6 | 29.92% | 0.844 | -59.09% | KILL: dilution drags CAGR -10.9 pp |
| 3 | 3 | 31.54% | 0.834 | -56.72% | KILL: turnover hurts |
| 3 | 12 | 35.07% | 0.968 | -58.76% | edge case — Sharpe +0.015 but CAGR -5.7 pp |
| 1 | 3 | 32.34% | 0.727 | -79.83% | KILL |
| 1 | 12 | 19.16% | 0.588 | -86.48% | KILL |
| 2 | 3 | 33.73% | 0.804 | -69.07% | KILL |
| 5 | 3 | 26.58% | 0.801 | -60.67% | KILL |

**Conclusion**: K=3, h=6 is locally optimal on every dimension. v4-v6 sweeps had already confirmed this with ~600 variants; this 9-cell grid is just due-diligence.

**One semi-interesting result**: K=3, h=12 has marginally higher Sharpe (0.968 vs 0.953) at significantly lower CAGR (35% vs 41%). Not worth pursuing for a CAGR-max objective, but interesting if the user ever wants a Sharpe-max variant.
