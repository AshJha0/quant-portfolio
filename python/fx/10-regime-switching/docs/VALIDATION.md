# Validation: FX Regime Switching

Numbers in this file come from `python examples/run_pipeline.py` on
synthetic data (3-state RORO panels, 1,500 business days, 12 currencies)
unless a test is cited. The generator plants the crash drift that the
strategy profits from. The only real-data result is a training-period
run in `research/`, not out-of-sample evidence.

## 1. Cross-checks against reference implementations

PCA, GMM and HMM are written from scratch. scikit-learn and hmmlearn are
used only in tests.

| check | tolerance | test |
|---|---|---|
| PCA components and explained variance vs `sklearn.decomposition.PCA` | 1e-8 (up to sign) | `test_pca.py::test_sklearn_cross_check` |
| PCA identities: orthonormality, reconstruction, variance partition | 1e-8 to 1e-10 | `test_pca.py` |
| GMM log-density vs `scipy.stats.multivariate_normal` | 1e-10 | `test_gmm.py::test_gaussian_logpdf_matches_scipy` |
| GMM `score_samples` vs sklearn with identical parameters | 1e-8 | `test_gmm.py::test_sklearn_cross_check_fixed_params` |
| GMM fitted optimum vs sklearn `GaussianMixture` | mean loglik within 0.01 | `test_gmm.py::test_sklearn_cross_check_fitted_loglik` |
| HMM likelihood, posteriors, Viterbi vs `hmmlearn`, identical parameters | 1e-6 / 1e-8 / exact | `test_hmm.py::test_hmmlearn_cross_check_fixed_model` |
| HMM fitted transmat and means vs fitted `hmmlearn` | 0.02 / 0.05 after permutation matching | `test_hmm.py::test_hmmlearn_cross_check_fitted` |
| scaled forward/backward vs textbook log-space recursion | 1e-11 relative | `test_hmm.py::test_scaled_recursions_match_brute_force_log_space` |
| reported likelihood equals the likelihood of the returned model at `max_iter` | 1e-9 | `test_hmm.py::test_reported_likelihood_is_that_of_returned_model_at_max_iter` |
| EM monotonicity (GMM and Baum-Welch) | non-decreasing to 1e-6 | `test_gmm.py`, `test_hmm.py` |
| stationary distribution, Viterbi vs brute force, durations 1/(1 - A_ii) | exact / 1e-10 | `test_hmm.py` |

## 2. Recovery on known truth (synthetic)

From section 1 of the pipeline, synthetic seed 11:

| quantity (synthetic) | value |
|---|---|
| 2-state HMM on returns: max abs transition-matrix error | 0.034 |
| 2-state HMM on returns: Viterbi accuracy | 99.1% |
| PC1 explained variance; carry/EM vs JPY/CHF opposite signs | 54.2%; yes |
| BIC-selected k on the five HMM input features, true k = 3 | GMM 3, HMM 4 |

HMM-BIC over-selects k on autocorrelated rolling-window features, so
k = 3 is an assumption. On iid synthetic returns BIC recovers the true k
(`test_gmm.py`). Detection accuracy on 2-state synthetic panels is at
least 75% (`test_detection.py::test_detection_recovers_regimes`). On the
8 synthetic evaluation seeds the risk-state classification accuracy
(risk_off and usd_squeeze merged) is 0.824 [0.771, 0.878].

## 3. Design and evaluation protocol

The configuration was chosen on design seeds and evaluated on fresh seeds.
The design table below lists all twelve configurations tried, including
the ones that were dropped during development (21/10/21-day windows,
`fwd_ts` in the HMM input, a risk-off basket without EM legs).

Design seeds 1001-1003, filtered Sharpe on synthetic data:

| windows / HMM columns / risk-off basket | mean Sharpe | min | max |
|---|---|---|---|
| 8/5/12, five columns, AUD/NZD+EM (chosen) | 1.002 | 0.540 | 1.287 |
| 8/5/12, five columns, AUD/NZD | 1.002 | 0.637 | 1.244 |
| 8/5/12, six with usd_str, AUD/NZD+EM | 0.836 | 0.332 | 1.107 |
| 8/5/12, six with usd_str, AUD/NZD | 0.805 | 0.336 | 1.044 |
| 21/10/21, six with usd_str, AUD/NZD | 0.763 | 0.424 | 1.386 |
| 21/10/21, six with usd_str, AUD/NZD+EM | 0.730 | 0.393 | 1.296 |
| 21/10/21, five columns, AUD/NZD | 0.611 | 0.129 | 0.931 |
| 21/10/21, five columns, AUD/NZD+EM | 0.596 | 0.076 | 0.975 |
| 8/5/12, seven with fwd_ts, AUD/NZD+EM | 0.439 | 0.089 | 0.636 |
| 21/10/21, seven with fwd_ts, AUD/NZD | 0.423 | 0.117 | 0.670 |
| 21/10/21, seven with fwd_ts, AUD/NZD+EM | 0.415 | 0.155 | 0.549 |
| 8/5/12, seven with fwd_ts, AUD/NZD | 0.385 | 0.055 | 0.588 |

* Trial count: 12. PBO by CSCV (10 blocks) across the twelve: 0.15.
* The top two differ only in the basket and tie to three decimals; the
  first by sort order was kept.
* HMM effort in the pipeline: refit every 63 days, one GMM-seeded start,
  a competing cold restart every fourth refit, 40 EM iterations. This
  keeps the run to 5-7 minutes with 6 worker processes.

## 4. Evaluation on fresh synthetic seeds

Seeds 2001-2008, mean across seeds with a 95% Student-t CI. Each seed is
also run with zero drift in every state (same hidden states, same
shocks): the vol-clustering null.

| strategy (synthetic, planted drift) | Sharpe | ann. return | max drawdown (additive) |
|---|---|---|---|
| oracle | 0.76 [0.43, 1.09] | 8.9% [4.9%, 12.9%] | -18.6% [-21.6%, -15.6%] |
| filtered | 0.47 [-0.09, 1.02] | 5.2% [-1.0%, 11.4%] | -19.8% [-24.4%, -15.2%] |
| static carry | -0.12 [-0.65, 0.40] | -1.4% [-7.3%, 4.5%] | -34.9% [-49.5%, -20.3%] |
| random timing (mean of 4 shifts) | 0.16 [-0.01, 0.34] | | |

| Sharpe difference (synthetic) | mean [95% CI] |
|---|---|
| filtered minus static, planted drift | 0.59 [-0.19, 1.37] |
| filtered minus static, zero drift | -0.39 [-1.01, 0.23] |
| part due to the planted drift (paired difference) | 0.98 [0.77, 1.19] |
| filtered minus random timing, planted drift | 0.30 [-0.15, 0.75] |
| filtered minus random timing, zero drift | -0.11 [-0.53, 0.31] |

| statistic (synthetic) | value |
|---|---|
| seed 2001 filtered Sharpe, Lo/Newey-West SE | 0.10, 0.49 |
| pooled 8-seed filtered Sharpe, block-bootstrap 95% CI (20-day blocks) | 0.47 [0.13, 0.80] |
| deflated Sharpe ratio at 12 logged trials | 0.723 |

Reading: on synthetic data the filter's advantage over static carry is
the planted crash drift. Without the drift link the filter is no better
than static carry or random timing. The deflated Sharpe ratio is below
0.95, so even the planted-drift result does not clear a multiple-testing
bar at 12 trials.

Fill models, filtered Sharpe on the 8 synthetic seeds:

| fill model | Sharpe |
|---|---|
| next, f = 0.5 (default) | 0.47 [-0.09, 1.02] |
| next, f = 1.0 | 0.40 [-0.18, 0.98] |
| close auction, 10 bp | 0.28 [-0.24, 0.80] |

## 5. Detection lag

Fresh flips into a risk state, detection counted when either crash label
is flagged (both books cut carry). 8 synthetic seeds:

| hysteresis | flips per seed | detection rate | mean lag (days) | Sharpe |
|---|---|---|---|---|
| fast (0.70/0.30, 2 days) | 12.4 [10.4, 14.4] | 0.95 [0.91, 0.99] | 2.7 [2.0, 3.5] | 0.47 [-0.09, 1.02] |
| conservative (0.90/0.10, 5 days) | 12.4 [10.4, 14.4] | 0.87 [0.81, 0.94] | 5.3 [4.0, 6.6] | 0.37 [-0.24, 0.98] |

Share of static carry's risk-state P&L earned in the first K days of each
risk spell (synthetic, 8 seeds): K=1 0.10 [-0.23, 0.43], K=3
-0.11 [-0.80, 0.58], K=5 -0.04 [-0.76, 0.69], K=10 0.43 [-0.11, 0.97].
The synthetic data does not show front-loaded carry losses.

## 6. Null tests

* `test_risk.py::test_vol_regime_null_no_timing_edge`: 6 synthetic seeds
  with vol and correlation regimes and zero drift in every state. The
  filtered Sharpe minus the mean Sharpe of four random-timing twins (same
  labels, shifted in time, so equal exposure) must have mean below 3 SE.
* `test_risk.py::test_timing_edge_statistic_has_power`: the same
  statistic, with the true states as the signal, on 12 synthetic seeds
  of 2,000 days: mean above 3 SE with the planted drift, and within 3 SE
  of zero without it. This shows the guard can fail.
* `test_gmm.py`: BIC prefers k = 1 on a single-state null panel.
* `test_edge_cases_review.py`: a forced 2-state fit on single-state data
  does not manufacture separation (fitted variance ratio below 3x).

## 7. Real-world failure modes not covered by the synthetic tests

* Pegged or managed currencies: a peg contributes zero vol and undefined
  correlations. Numerically handled (tested with a pegged column), but a
  managed float biases features until the band breaks.
* Peg breaks (SNB, January 2015) are jumps, not regimes. A filtered HMM
  reacts after the move.
* Idiosyncratic EM crises: one currency crashes while the risk block is
  calm. A systemic regime model does not flip, and a carry basket holding
  that currency still loses.
* Slow carry drawdowns without a vol spike: the features barely move and
  the filter stays risk-on.
* Cross-currency basis in squeezes: CIP-based carry is overstated.
* Spreads widen at the flip; constant pip spreads understate switching
  costs.

## 8. Test inventory

196 tests (`pytest` from the project root, about 15 minutes on a loaded
12-core machine):

| file | tests | covers |
|---|---|---|
| `test_synthetic.py` | 13 | generator recovery and validation |
| `test_features.py` | 14 | features, point-in-time mutation |
| `test_pca.py` | 12 | PCA identities, sklearn, RORO axis |
| `test_gmm.py` | 13 | GMM EM, BIC, sklearn |
| `test_hmm.py` | 19 | HMM identities, recovery, hmmlearn, likelihood bookkeeping |
| `test_detection.py` | 19 | causality, labeling, hysteresis, cold restarts |
| `test_strategy.py` | 15 | books, vol target, ACT/365 carry, pip conversion |
| `test_backtest.py` | 11 | ledger identity, no-lookahead mutation |
| `test_execution.py` | 10 | t+1 fill price, foresight detector, auction cost |
| `test_risk.py` | 19 | partitions, oracle ordering, vol-regime null, power |
| `test_research_stats.py` | 18 | Sharpe SE, deflated Sharpe, PBO, bootstrap |
| `test_edge_cases_review.py` | 33 | single-regime data, label switching, peg breaks, hysteresis, NaN/Inf |
