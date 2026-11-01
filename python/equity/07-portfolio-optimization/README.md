# 07: Equity Portfolio Optimization and Risk Allocation (`eq_port`)

A from-scratch implementation of the classical portfolio construction
stack. The theme: mean-variance optimisation amplifies estimation error,
and shrinkage, Black-Litterman, risk parity and constraints are defences
against that.

```
Return Estimation ──► Covariance Modeling ──► Mean-Variance Optimization
 (sample / EWMA /      (sample / EWMA /        (closed forms, convex-QP
  James-Stein /         Ledoit-Wolf 2004        max-Sharpe, SLSQP for
  reverse-opt /         from scratch /          min-var, target-return/-risk,
  Black-Litterman)      1-factor / PSD repair)  long-only and box bounds)
        │                                             │
        ▼                                             ▼
 Efficient Frontier ◄── Risk Parity (ERC, CCD) ──► Sharpe & Risk Metrics
 (analytic two-fund      + inverse-vol,            (Lo-adjusted SE, Sharpe-
  + numeric constrained)   vol-target overlay)      difference test, Sortino,
                              │                     MDD, DR, effN)
                              ▼
              Walk-Forward Backtest (no lookahead, exact
              turnover and transaction-cost ledger, strategy race)
```

Optimisation uses closed forms and `scipy.optimize` (SLSQP). The
maximum-Sharpe problem is solved as a convex QP after the substitution
y = κw, with analytic gradients and a KKT check on the result. When no
feasible portfolio has a positive expected excess return, it returns the
long-only minimum-variance portfolio and reports `fallback=True`.

## Quickstart

```bash
cd python/equity/07-portfolio-optimization
pip install -e .[dev]
pytest -q                        # 170 tests, about 16 s, offline
python examples/run_pipeline.py  # full pipeline, about 90 s, seeded
```

```python
import numpy as np, eq_port as ep
from eq_port.data import generate_panel

panel = generate_panel(n_assets=8, n_periods=2400, seed=1, regimes=True)
r = panel.returns                      # (T, N) daily simple returns

lw  = ep.ledoit_wolf_cc(r)             # shrunk covariance, delta in [0,1]
mu  = ep.james_stein_mean(r).mean      # shrunk mean
res = ep.max_sharpe_qp(mu, ep.psd_repair(lw.cov), bounds=(0, 1))
print(res.weights, res.fallback, res.kkt_residual)
erc = ep.erc_weights(lw.cov)           # equal risk contribution (mean-free)
bt  = ep.run_backtest(r, ep.make_erc_strategy(), window=252,
                      rebalance_every=21, cost_bps=10.0)
print(ep.summary_table({"ERC": bt.net_returns}))
```

## Headline results

All numbers come from `python examples/run_pipeline.py` on synthetic
panels. The intervals are 95% t-intervals across independent synthetic
panels.

**Estimation error.** 20 disjoint 252-day windows on each of 10 synthetic
panels (seeds 7-16). Each estimated portfolio is scored under the true
moments of its panel. Mean achievable (true tangency) Sharpe is 0.427.

| Estimated portfolio | mean true Sharpe | 95% CI |
|---|---|---|
| Tangency, raw mean, unconstrained | 0.115 | 0.068 to 0.161 |
| Tangency, raw mean, long-only | 0.279 | 0.250 to 0.309 |
| Tangency, James-Stein mean, long-only | 0.328 | 0.286 to 0.370 |
| Min-variance, Ledoit-Wolf, long-only | 0.333 | 0.289 to 0.377 |
| ERC, Ledoit-Wolf | 0.322 | 0.304 to 0.340 |
| Equal weight | 0.314 | 0.301 to 0.327 |

Raw-mean tangency is worse than min-variance (paired difference -0.054,
CI -0.078 to -0.030). James-Stein tangency, ERC and equal weight are not
distinguishable from min-variance on these synthetic panels. James-Stein
tangency fell back to min-variance in 73 of 200 windows because no
shrunk mean was positive.

**Walk-forward race.** Monthly rebalance, 10 bp costs, 20 synthetic
2400-day panels with a crisis regime (seeds 1-20). Mean net Sharpe:

| Strategy | mean net Sharpe | 95% CI | diff vs MinVar | annual turnover |
|---|---|---|---|---|
| MinVar (LW) | -0.083 | -0.252 to 0.087 | | 1.32 |
| Tangency JS | -0.105 | -0.260 to 0.051 | -0.022 (-0.069 to 0.025) | 2.68 |
| ERC (LW) | -0.107 | -0.260 to 0.046 | -0.024 (-0.093 to 0.045) | 0.60 |
| EqualWeight | -0.114 | -0.266 to 0.038 | -0.032 (-0.108 to 0.045) | 0.57 |
| Static tilt | -0.120 | -0.267 to 0.027 | -0.037 (-0.114 to 0.040) | 0.57 |
| Tangency raw | -0.200 | -0.330 to -0.070 | -0.117 (-0.239 to 0.004) | 6.76 |

No strategy's Sharpe is statistically different from min-variance at 5%
on these synthetic panels. Raw-mean tangency has the lowest mean and
about five times the turnover of min-variance. The Jobson-Korkie/Memmel
test on the main panel gives p > 0.2 for every pair against min-variance.
"Static tilt" is a fixed 15%/10% split over the eight synthetic equity
assets. It contains no bonds.

**Ledoit-Wolf from scratch.** Closed-form 2004 intensity, δ in [0, 1];
the matrix stays invertible when T < N.

**Black-Litterman.** The reverse-optimisation round trip is exact to
3.6e-16. One +3%/yr relative view moves the long-only tangency weight on
the viewed asset from 0.24 to 0.42 (synthetic panel).

## Limitations

- Every result above is on synthetic data from a factor model with a
  planted crisis regime.
- The James-Stein estimator is the isotropic grand-mean form, not
  Jorion's Bayes-Stein.
- There are no turnover, tracking-error or leverage constraints in the
  optimisers. Costs enter only in the backtest.

## Layout

```
src/eq_port/          returns_est, covariance, mvo, risk_parity,
                      backtest, metrics, data/{synthetic,live}
tests/                170 offline seeded tests
examples/run_pipeline.py   reproduces every number above (prints only)
docs/                 METHODOLOGY.md · VALIDATION.md · DESK_GUIDE.md
```
