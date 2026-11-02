# 10. Equity regime-switching overlay

Regime detection for equities with a from-scratch Gaussian HMM, GMM and PCA
(cross-checked against `hmmlearn` and `scikit-learn`), traded through a
hysteresis-banded, vol-targeted allocation in a causal walk-forward backtest.

All results below come from synthetic data. The generator plants a bear state
with a -30% a year drift, and the strategy profits mainly from that planted
drift. No real-data result exists yet; a real-data study is Phase 4 of the
remediation plan.

```
multi-asset panel
  -> features.py    realized vol (10/21/63d), dispersion, average pairwise
                    correlation, drawdown, trend vs 200d MA, credit and term
                    proxies, all expanding-window z-scored (causal)
  -> pca.py         correlation-matrix eigendecomposition, sign-fixed loadings
  -> hmm.py/gmm.py  scaled forward-backward, Baum-Welch EM, Viterbi,
                    stationary distribution, durations, BIC
  -> detection.py   expanding-window refits, online filtered probabilities,
                    economic labels bull / transition / bear
  -> strategy.py    bull 1.0 / transition 0.5 / bear 0.0, hysteresis band
                    0.70 / 0.30, 10% vol target, optional causal
                    minimum-duration confirmation
  -> backtest.py    exact ledger, execution model, costs and financing,
                    baselines: buy-and-hold, 200d MA, random timing at equal
                    exposure, plain vol targeting
  -> risk.py        per-regime statistics on the regime known at t-1,
                    transition attribution, flip aftermath
  -> research_stats.py  Lo / Newey-West Sharpe SE, deflated Sharpe, PBO,
                    block bootstrap, multi-seed summaries
```

## Execution convention

Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost. Costs are 5 bp per unit traded.
Leverage above 1.0 pays the cash rate plus 50 bp a year; cash earns the cash
rate (0 by default).

## Quickstart

```bash
cd python/equity/10-regime-switching
pip install -e ../../core   # qp-core, the shared package (not on PyPI)
pip install -e .[dev]
pytest -q                              # 195 tests, about 5 minutes
python examples/run_pipeline.py        # about 18 minutes with 10 workers
```

The pipeline prints only; it writes nothing to disk.

## Headline results (synthetic)

Command: `python examples/run_pipeline.py`. 20 fresh seeds per arm (101 to
120), 2,520 days, 8 assets, net of costs and financing. Each cell is the mean
annualised Sharpe ratio across seeds with a 95% confidence interval for that
mean. Planted arm: default generator (bear drift -30% a year). Equal-drift
arm: the same generator with 7% a year drift in every state, so only vol and
correlation switch.

| synthetic, Sharpe | planted drifts | equal drift |
|---|---|---|
| regime strategy | 0.96 [0.73, 1.19] | 0.45 [0.27, 0.63] |
| buy and hold | 0.11 [-0.10, 0.31] | 0.28 [0.12, 0.44] |
| 200d MA rule | 0.23 [0.04, 0.42] | 0.20 [-0.00, 0.40] |
| random timing, equal exposure | 0.50 [0.24, 0.75] | 0.33 [0.14, 0.51] |
| plain vol targeting | 0.69 [0.47, 0.91] | 0.47 [0.30, 0.64] |

Where the synthetic edge comes from (paired differences across the same
seeds, mean [95% CI]):

| synthetic, Sharpe difference | planted drifts | equal drift |
|---|---|---|
| strategy minus buy and hold | 0.86 [0.66, 1.05] | 0.17 [-0.02, 0.37] |
| strategy minus plain vol targeting | 0.27 [0.16, 0.38] | -0.02 [-0.14, 0.10] |
| plain vol targeting minus buy and hold | 0.58 [0.48, 0.69] | 0.19 [0.09, 0.29] |

* Of the 0.86 Sharpe gain over buy-and-hold on synthetic data, 0.68
  [0.60, 0.76] disappears when the bear drift is removed. That part is the
  planted drift.
* With equal drifts the strategy does no better than plain vol targeting on
  synthetic data (-0.02 [-0.14, 0.10]). What remains of the edge there is vol
  timing.
* On one synthetic path (seed 101) the strategy Sharpe is 0.47 with a
  Lo / Newey-West SE of 0.41, and the deflated Sharpe at the 4 logged trials
  is 0.58, well short of 0.95.

## Layout

```
src/eq_regime/            features, pca, gmm, hmm, detection, strategy,
                          backtest, risk, research_stats, data/synthetic
tests/                    195 offline tests: causality mutation tests,
                          execution-convention tests, EM monotonicity,
                          sklearn / hmmlearn cross-checks, null guards
examples/run_pipeline.py  reproduces every number in this README and in
                          docs/VALIDATION.md
docs/                     METHODOLOGY.md, VALIDATION.md, DESK_GUIDE.md
```
