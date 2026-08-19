# 07 — Equity Portfolio Optimization & Risk Allocation (`eq_port`)

Flagship-quality implementation of the classical portfolio construction
stack, built around one theme: **naive Markowitz is an estimation-error
maximizer, and everything that works in practice — shrinkage,
Black-Litterman, risk parity, constraints — is a defense against that.**

```
Return Estimation ──► Covariance Modeling ──► Mean-Variance Optimization
 (sample / EWMA /      (sample / EWMA /        (closed forms + SLSQP:
  James-Stein /         Ledoit-Wolf 2004        min-var, tangency,
  reverse-opt /         from scratch /          target-return/-risk,
  Black-Litterman)      1-factor / PSD repair)  long-only & box bounds)
        │                                             │
        ▼                                             ▼
 Efficient Frontier ◄── Risk Parity (ERC, CCD) ──► Sharpe & Risk Metrics
 (analytic two-fund      + inverse-vol,            (Lo-adjusted SE,
  + numeric constrained)   vol-target overlay)      Sortino, MDD, DR, effN)
                              │
                              ▼
              Walk-Forward Backtest (no lookahead, exact
              turnover & transaction-cost ledger, strategy race)
```

Optimization uses `scipy.optimize` (SLSQP) and closed forms only — no
cvxpy; the QPs here are small and smooth, every solver output is
cross-checked against closed forms in the tests, and the dependency
weight isn't justified (see docs/METHODOLOGY.md §4).

## Quickstart

```bash
cd python/equity/07-portfolio-optimization
pip install -e .[dev]
pytest -q                        # 150 tests, ~3 s, offline
python examples/run_pipeline.py  # full pipeline, ~15 s, seeded
```

```python
import numpy as np, eq_port as ep
from eq_port.data import generate_panel

panel = generate_panel(n_assets=8, n_periods=2400, seed=1, regimes=True)
r = panel.returns                      # (T, N) daily simple returns

lw  = ep.ledoit_wolf_cc(r)             # shrunk covariance, delta in [0,1]
mu  = ep.james_stein_mean(r).mean      # shrunk mean
w   = ep.max_sharpe_constrained(mu, ep.psd_repair(lw.cov), bounds=(0, 1))
erc = ep.erc_weights(lw.cov)           # equal risk contribution (mean-free)
res = ep.run_backtest(r, ep.make_erc_strategy(), window=252,
                      rebalance_every=21, cost_bps=10.0)
print(ep.summary_table({"ERC": res.net_returns}))
```

## Headline results (from `examples/run_pipeline.py`, seeded)

**The estimation-error problem, in numbers** — 20 disjoint 252-day
windows, portfolios evaluated under the *true* moments (achievable
tangency Sharpe 0.452, equal weight 0.326):

| Estimated portfolio | mean true Sharpe |
|---|---|
| Tangency, raw sample mean, unconstrained | **0.141** (16.2x avg gross leverage, fails 8/20 windows) |
| Tangency, raw mean, long-only | 0.300 (below EW in 11/20 windows) |
| Tangency, James-Stein mean | 0.313 |
| Min-variance (Ledoit-Wolf) | **0.339** |
| ERC (Ledoit-Wolf) | 0.333 (tightest spread: 0.328–0.339) |

**Walk-forward race** (monthly rebalance, 10bp costs, mean net Sharpe
across 6 independent 2400-day panels with a crisis regime):
MinVar **+0.03** > TanJS −0.00 > ERC −0.04 > EW −0.05 > Static 60/40
−0.06 > **Tangency-raw −0.06 with 8.3x/yr turnover and the worst
drawdowns** — the classic result: raw-mean MVO loses to naive 1/N while
mean-free allocators do fine.

**Ledoit-Wolf from scratch**: closed-form 2004 intensity, δ ∈ [0,1]
always; cond(sample) 20.8 → 16.9 on 252×8; still invertible when T < N.

**Black-Litterman**: reverse-optimization round trip exact to 3.5e-16;
one +3%/yr relative view moves the posterior sensibly (viewed asset
+0.6%, counter-asset −1.7%, correlated assets adjust) and the long-only
tangency weight 0.24 → 0.42.

**Crisis regime** (correlations 0.47 → 0.87): concentrated raw tangency
suffers most (crisis MaxDD 0.47, eff. N 1.6); ERC/EW degrade gracefully;
min-variance loses least. Details in docs/VALIDATION.md.

## Layout

```
src/eq_port/          returns_est, covariance, mvo, risk_parity,
                      backtest, metrics, data/{synthetic,live}
tests/                150 offline seeded tests (identities, solver
                      cross-checks, no-lookahead, edge cases)
examples/run_pipeline.py   reproduces every number above
docs/                 METHODOLOGY.md · VALIDATION.md · DESK_GUIDE.md
```
