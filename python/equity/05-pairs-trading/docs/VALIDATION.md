# Validation: equity statistical pairs trading

Every number below comes from `python examples/run_pipeline.py` (30 seeds by
default) or from the test suite. All data is synthetic. The generator plants
cointegrated pairs with known half-lives, so every positive P&L figure here
measures how well the pipeline recovers a planted effect. None of it is
evidence of an edge in real markets; no real-data result exists yet
(Phase 4).

---

## 1. Cross-checks against statsmodels

| check | tolerance | test |
|---|---|---|
| ADF t-stat, fixed lag in {0, 1, 4, 8}, regression `c` | 1e-8 | `test_fixed_lag_stat_matches` |
| ADF t-stat and selected lag, `autolag="AIC"`, regressions `n`/`c`/`ct` | 1e-8, same lag | `test_autolag_aic_matches` |
| MacKinnon N=1 and N=2 critical values | 1e-10 | `test_finite_sample_surface_matches_statsmodels` |

statsmodels is a test-only dependency.

## 2. Size of the cointegration test

200 replications of Engle-Granger on independent random walks (T=400): the
N=2 critical values reject at close to the nominal 5% (asserted in
[0.5%, 12%]); plain ADF values reject more than 1.5 times as often
(`test_spurious_rejection_rate_close_to_size`). A second size check on 40
correlated random walks allows at most 6 rejections at 5%
(P(Binomial(40, 0.05) > 6) = 0.34%).

## 3. Selection funnel and parameter recovery (synthetic, seed 7)

Each Engle-Granger regression is oriented by ticker order. Ground truth only
labels the rows of the table.

| stage | count |
|---|---:|
| same-sector candidates | 40 |
| pass return-correlation screen (rho >= 0.6) | 8 |
| pass Engle-Granger at 5% | 4 |
| planted pairs among the 4 | 4 |
| correlated-random-walk traps through the correlation screen / rejected by EG | 3 / 3 |

Over 30 synthetic seeds (section 9 of the pipeline): planted pairs passing
EG 3.93 of 4 [3.84, 4.03]; trap pairs passing EG 0.20 [0.05, 0.35].

OU recovery on the seed-7 survivors (beta_true is 1/beta of the planted
regression because the orientation is reversed):

| pair | beta true | beta est | HL true (d) | HL OLS (d) | HL MLE (d) |
|---|---:|---:|---:|---:|---:|
| CO1_X/CO1_Y | 0.739 | 0.726 | 8.2 | 9.5 | 9.5 |
| CO2_X/CO2_Y | 0.582 | 0.577 | 6.9 | 7.0 | 7.0 |
| CO3_X/CO3_Y | 0.638 | 0.640 | 8.5 | 8.6 | 8.6 |
| CO0_X/CO0_Y | 0.702 | 0.697 | 5.0 | 5.4 | 5.4 |

## 4. Headline: multi-seed distribution (synthetic)

30 synthetic panels (seeds 1000 to 1029), 20 names x 1500 days, $6mm capital
(6 pair slots x $1mm gross), costs as in METHODOLOGY §3, fill at the
modelled t+1 price with f = 0.5. Mean [95% CI] across seeds, sd in the last
column. Command: `python examples/run_pipeline.py`, section 9.

| metric (synthetic) | mean [95% CI] | sd |
|---|---|---:|
| in-sample Sharpe (selection on the full sample) | 2.68 [2.54, 2.83] | 0.39 |
| walk-forward Sharpe | 1.33 [1.14, 1.53] | 0.52 |
| walk-forward Sharpe, fill at next close (f = 1) | 1.21 [1.03, 1.40] | 0.48 |
| walk-forward Sharpe, close auction (10 bp) | 0.81 [0.61, 1.00] | 0.53 |
| walk-forward annual return on capital | 1.4% [1.1%, 1.6%] | 0.7% |
| walk-forward annual vol | 1.2% [0.8%, 1.6%] | 1.0% |
| walk-forward max drawdown | $78k [$37k, $119k] | $110k |
| walk-forward cost drag per year | 1.2% [1.1%, 1.3%] | 0.2% |
| walk-forward trades | 85.6 [79.7, 91.4] | 15.6 |

The in-sample Sharpe is about twice the walk-forward Sharpe on the same
synthetic panels. Selection and fitting on the trading data inflate it.

## 5. One synthetic path in detail (seed 7)

| metric (synthetic, seed 7) | in-sample | walk-forward |
|---|---:|---:|
| net P&L | $2.11mm | $0.99mm |
| annual return on $6mm | 5.9% | 3.3% |
| annual vol | 2.5% | 3.1% |
| Sharpe | 2.41 | 1.09 |
| Sharpe SE, iid / Lo (Newey-West) | 0.41 / 0.29 | 0.45 / 0.37 |
| max drawdown | $79k | $174k |
| hit rate | 90.8% | 58.5% |
| trades / average hold | 65 / 14.6 d | 65 / 12.0 d |
| turnover / cost drag | 3.6x / 1.48% | 4.3x / 1.74% |

One planted pair, CO0, made 71% of the seed-7 walk-forward net P&L. One
trap pair and the regime-break pair entered the walk-forward book in some
windows and lost money. A circular block bootstrap (11-day blocks, 2,000
resamples) of the seed-7 synthetic walk-forward returns gives a Sharpe of
1.09 with 95% CI [0.31, 1.77].

## 6. Trial log, deflated Sharpe and PBO (synthetic, seed 7)

The pipeline evaluates 9 walk-forward configurations: hedge in {static,
rolling, rls} x entry z in {1.5, 2.0, 2.5}. All are logged.

| config | annual Sharpe (synthetic) |
|---|---:|
| static / 1.5 | 1.543 |
| static / 2.0 (headline) | 1.094 |
| static / 2.5 | 1.322 |
| rolling / 1.5 | 1.010 |
| rolling / 2.0 | 0.879 |
| rolling / 2.5 | 0.669 |
| rls / 1.5 | 1.149 |
| rls / 2.0 | 0.999 |
| rls / 2.5 | 1.153 |

| statistic | value |
|---|---:|
| trials logged | 9 |
| deflated Sharpe ratio of the headline config | 0.955 |
| PBO, CSCV with 10 splits | 0.270 |

The headline configuration (static hedge, entry 2.0) was fixed before the
grid was run; it is not the best of the nine on this seed.

## 7. Execution timing and impact (synthetic, seed 7)

| execution | Sharpe | annual return | cost drag |
|---|---:|---:|---:|
| next, f = 0.5 (default) | 1.094 | 3.33% | 1.74% |
| next, f = 1.0 (next close) | 1.116 | 3.47% | 1.76% |
| close auction, 10 bp | 0.917 | 2.89% | 2.30% |

| impact Y | gross per pair | per-leg trade as % of ADV | Sharpe | annual return |
|---:|---:|---:|---:|---:|
| 0.0 | $1mm | 2.5% | 1.555 | 4.72% |
| 0.0 | $10mm | 25% | 1.555 | 4.72% |
| 0.3 | $1mm | 2.5% | 1.360 | 4.13% |
| 0.3 | $10mm | 25% | 0.927 | 2.84% |
| 0.7 | $1mm | 2.5% | 1.094 | 3.33% |
| 0.7 | $10mm | 25% | 0.101 | 0.33% |
| 1.0 | $1mm | 2.5% | 0.893 | 2.74% |
| 1.0 | $10mm | 25% | -0.451 | -1.56% |

ADV is an assumed $20mm per name because the synthetic data has no volume.
At Y = 0.7 the synthetic book has almost no Sharpe left at $10mm per pair.

## 8. Regime break (synthetic)

A planted pair is cointegrated for 750 days, then its spread becomes a
random walk. It is fitted on the pre-break data and traded through the
break.

| net P&L (synthetic) | seed 42 | 30 seeds, mean [95% CI] |
|---|---:|---|
| before the break | $80k | $82k [$71k, $93k] |
| after the break, with stops and re-entry arming | -$25k | -$20k [-$26k, -$15k] |
| after the break, without stops | -$724k | -$472k [-$540k, -$404k] |

## 9. Execution convention tests

`TestExecutionConvention` in `tests/test_backtest.py`:

- a hand ledger shows the fill price is p_t + f (p_{t+1} - p_t);
- a perfect-foresight signal s_t = sign(r_{t+1}) earns a mean gross P&L
  within 4 SE of zero at f = 1, while the same signal filled at the close of
  t earns more than 10 SE; a mutant engine that fills at the close of t
  fails five of these tests;
- f = 0.5 captures half of the foresight P&L (within 0.05);
- the close-auction mode charges exactly `auction_cost_bp` per leg on traded
  notional.

## 10. Accounting and walk-forward tests

- `net = gross - commission - slippage - impact - borrow - financing -
  auction` to 1e-9, in both execution modes, and trade P&Ls sum to net.
- Square-root impact matches the closed form per leg and scales as size to
  the power 1.5; financing and rebate match day by day.
- A three-round-trip scenario at f = 1 matches hand-computed P&L to 1e-9.
- Walk-forward windows never overlap (constructor-enforced).
- Frozen parameters: scrambling the prices inside one trading window leaves
  the parameters recorded for that window and the P&L of every earlier
  window unchanged (`test_parameters_frozen_during_trading_window`). A
  mutant that fits on formation plus trading data fails this test.
- Rolling and RLS hedges are causal: scrambling the last 40 days leaves
  every earlier position unchanged (`test_dynamic_hedge_is_causal`).

## 11. Estimator bias of the OU half-life

Kendall (1954): the AR(1) coefficient estimated with a fitted mean is biased
down by about (1 + 3 phi)/n. kappa = -ln(phi) is therefore biased up and
estimated half-lives are too short. `test_ols_kappa_is_biased_upward_over_many_seeds`
checks this over 200 seeds at (n, kappa) = (12,000, 0.01) and (2,000, 0.03):
the mean relative bias is positive by more than 3 Monte Carlo SE and within
3 SE plus 1 point of the Kendall value. A time stop at k estimated
half-lives is therefore slightly too tight.

On a pure random walk OLS still returns b slightly below 1, so an OU fit
alone reports a spurious half-life of hundreds of days
(`test_an_ou_fit_alone_cannot_reject_a_random_walk`). The Engle-Granger
gate and the half-life cap are what reject it.

## 12. Research statistics tests

`tests/test_research_stats.py` checks the Lo/Newey-West SE against a
Monte Carlo of AR(1) returns, the expected maximum Sharpe against
simulation, that the deflated Sharpe is uniform under the null, that PBO is
near 0.5 on noise and near 0 for a dominant configuration, block-bootstrap
coverage, and finite-input validation.

## 13. Limits

- All results are synthetic and the edge is planted.
- The impact coefficient and ADV are assumptions.
- Borrow recalls, hard-to-borrow fees and delisting returns are not modelled.
- Thresholds (entry 2, exit 0, stop 4, time stop 3 half-lives, half-life
  gate [1, 126], correlation 0.6) were chosen globally, not inside the
  walk-forward.
