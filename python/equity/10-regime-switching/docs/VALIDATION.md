# Validation: equity regime-switching overlay

How the model was validated and where it fails. Every number below comes
from `python examples/run_pipeline.py` (about 18 minutes with 10 workers) or
from the test suite (`pytest -q`, 195 tests, about 5 minutes). All data are
synthetic. The bear state's negative drift is planted by the generator, and
no real-data result exists yet (Phase 4).

## 1. Cross-model consistency checks

| Check | Tolerance | Result |
|---|---|---|
| PCA vs `sklearn.decomposition.PCA` on standardized data, components and explained variances | 1e-8, up to sign | pass (`test_pca.py`) |
| GMM vs `sklearn.mixture.GaussianMixture`, per-observation log-likelihood | 1e-4 | pass (`test_gmm.py`) |
| GMM parameters evaluated by an independent scipy density | 1e-8 | pass |
| HMM parameters loaded into `hmmlearn.GaussianHMM.score` vs our log-likelihood | 1e-6 | pass (`test_hmm.py`) |
| Independent hmmlearn EM fit vs our fit, per-observation log-likelihood | 1e-3 | pass |
| Stationary distribution `pi A = pi` | 1e-12 | pass |
| Expected durations vs `1/(1 - a_ii)` | 1e-12 | pass |
| EM monotonicity (GMM and Baum-Welch), every iteration | >= -1e-7 | pass |
| `fit_hmm` stopped by `max_iter`: reported log-likelihood equals the score of the returned parameters | 1e-12 relative | pass (`test_hmm.py::test_log_likelihood_belongs_to_returned_parameters_at_max_iter`) |

## 2. Parameter recovery on synthetic ground truth

Synthetic 3-state panel, 2,520 days, seed 7. Full-sample 3-state HMM on
index returns, permutation-matched to the truth.

| state | true mu (ann) | est mu | true p_stay | est p_stay | true E[dur] | est E[dur] |
|---|---|---|---|---|---|---|
| bull | 0.150 | 0.148 | 0.985 | 0.986 | 66.7d | 72.6d |
| transition | 0.020 | -0.007 | 0.940 | 0.943 | 16.7d | 17.4d |
| bear | -0.300 | -0.456 | 0.950 | 0.945 | 20.0d | 18.3d |

Stationary distribution 0.574 / 0.275 / 0.151 against empirical occupancy
0.577 / 0.258 / 0.165 on this synthetic path. The bear drift has the widest
error: 416 bear days at 38% vol give a mean-return standard error of about
30% a year. Regime means are the least reliable parameters.

K selection on the synthetic panel:

| method | K=1 | K=2 | K=3 | K=4 | K=5 | chosen |
|---|---|---|---|---|---|---|
| GMM BIC on 1-D index returns | -16,205 | -17,569 | -17,577 | -17,557 | -17,534 | 3 |
| HMM on PCA-3 features, out-of-sample log-likelihood per day (train 70%, test 30%) | | -3.562 | -3.450 | -3.294 | | 4 |

The GMM BIC is on 1-D returns and ignores the dynamics. The out-of-sample
likelihood scores the model class that trades and prefers K=4. The overlay
keeps K=3 because labels map three states to three weights; this choice was
made before the runs reported here and is a limitation.

## 3. Filtered vs smoothed probabilities

Around the first true bull-to-bear transition of the synthetic path (index
320):

| day | true state | P_bear filtered | P_bear smoothed |
|---|---|---|---|
| t-3 | transition | 0.016 | 0.137 |
| t-2 | transition | 0.014 | 0.318 |
| t-1 | transition | 0.103 | 0.743 |
| t+0 | bear | 0.367 | 0.946 |
| t+1 | bear | 0.973 | 0.998 |

The smoothed probability is 74% bear the day before the change because it
uses later returns. Trading uses filtered probabilities only. The mutation
tests (`test_detection.py::TestCriticalCausality`,
`test_backtest.py::TestNoLookahead`) show the filtered path and every ledger
row are unchanged when future prices are altered.

## 4. Execution convention

Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost.

| Test (`tests/test_execution.py`) | What it pins |
|---|---|
| `TestFillPrice::test_hand_computed_next_fill` | fill price equals p_t + f (p_{t+1} - p_t), every ledger cell by hand |
| `TestFillPrice::test_perfect_foresight_earns_nothing_at_next_close` | a signal equal to sign(r_{t+1}) earns zero within 4 SE at f = 1; a same-close fill earns more than 20 SE, so the test fails on an engine that fills at the close of t |
| `TestCloseAuction::test_charges_exactly_the_auction_cost` | close-auction cost is cost_bps plus auction_cost_bp per unit traded |
| `TestFinancing::test_leverage_pays_and_cash_earns` | leverage pays cash rate plus spread on the time-weighted weight |

Effect on synthetic Sharpe (planted arm, 20 seeds, difference from the
default f = 0.5, mean [95% CI]):

| variant | Sharpe minus default |
|---|---|
| close auction, 10 bp | -0.040 [-0.062, -0.018] |
| next close, f = 1 | -0.047 [-0.067, -0.026] |
| causal minimum duration, 5 days | -0.083 [-0.154, -0.013] |

## 5. Null-data guards (`tests/test_null_guard.py`)

Both guards are judged on Sharpe across 12 seeds. The mean difference must be
below 3 standard errors of that mean.

| null | comparison | measured, synthetic (mean, SE) |
|---|---|---|
| GBM, no regimes | strategy minus random timing at equal exposure | +0.08 (0.10) |
| Markov-switching vol and correlation, equal 7% drift | strategy minus plain vol targeting | -0.01 (0.06) |
| planted drifts, same configuration (power check, not a test) | strategy minus random timing | +0.48 (0.11) |
| planted drifts, same configuration (power check, not a test) | strategy minus plain vol targeting | +0.22 (0.08) |

Random timing is the same regime-weight path circularly shifted by a random
offset, times the same vol scale, rescaled to the strategy's mean weight.
BIC on GBM returns selects K=1 (`test_bic_prefers_fewer_states_on_null_returns`).

## 6. Walk-forward study (synthetic)

Settings: 3 states, PCA-3, first fit at 378 days, refit every 63 days, one
EM start and 50 iterations per refit, hysteresis 0.70 / 0.30, 10% vol target,
5 bp costs, 50 bp leverage spread. 20 fresh seeds (101 to 120) per arm. These
settings were fixed before the runs. Mean Sharpe [95% CI] across seeds:

| synthetic, Sharpe | planted drifts | equal drift |
|---|---|---|
| regime strategy | 0.96 [0.73, 1.19] | 0.45 [0.27, 0.63] |
| buy and hold | 0.11 [-0.10, 0.31] | 0.28 [0.12, 0.44] |
| 200d MA rule | 0.23 [0.04, 0.42] | 0.20 [-0.00, 0.40] |
| random timing, equal exposure | 0.50 [0.24, 0.75] | 0.33 [0.14, 0.51] |
| plain vol targeting | 0.69 [0.47, 0.91] | 0.47 [0.30, 0.64] |

## 7. Edge decomposition (synthetic)

Paired differences on the same seeds, mean [95% CI]:

| synthetic, Sharpe difference | planted drifts | equal drift |
|---|---|---|
| strategy minus buy and hold | 0.86 [0.66, 1.05] | 0.17 [-0.02, 0.37] |
| strategy minus random timing, equal exposure | 0.46 [0.31, 0.62] | 0.13 [-0.02, 0.28] |
| strategy minus plain vol targeting | 0.27 [0.16, 0.38] | -0.02 [-0.14, 0.10] |
| plain vol targeting minus buy and hold | 0.58 [0.48, 0.69] | 0.19 [0.09, 0.29] |

The planted bear drift accounts for 0.68 [0.60, 0.76] of the 0.86 Sharpe gain
over buy-and-hold on synthetic data (planted arm minus equal-drift arm). With
equal drifts the HMM overlay adds nothing over plain vol targeting on
synthetic data. A real-data study has to test whether the detected high-vol
state has a negative conditional mean out of sample; without that the overlay
is a vol-timing overlay.

## 8. One synthetic path in detail (planted arm, seed 101)

| synthetic, seed 101 | strategy | buy and hold | 200d MA | random timing | vol target |
|---|---|---|---|---|---|
| CAGR | 3.6% | -7.1% | 0.3% | 0.7% | 3.1% |
| ann. vol | 8.3% | 19.9% | 10.9% | 9.2% | 11.6% |
| Sharpe | 0.47 | -0.27 | 0.08 | 0.12 | 0.32 |
| max drawdown | 25.8% | 58.9% | 20.5% | 25.4% | 30.3% |

* Strategy Sharpe 0.47, Lo / Newey-West SE 0.41 (Bartlett lag 7).
* Sharpe difference against buy-and-hold 0.74, paired circular block
  bootstrap 95% CI [0.03, 1.42].
* Trial log: 4 strategy variants evaluated on every seed (default, minimum
  duration 5, close auction, next close). Deflated Sharpe of the default at
  N = 4: 0.58. PBO by CSCV over the 4 variants (10 blocks): 0.06. The count
  excludes choices made before this study (K = 3, the 0.70 / 0.30 band, the
  10% target, the 21-day vol window), so the deflated Sharpe is an upper
  bound.
* Per-regime returns use the regime detected at t-1 (`regimes_known_at_decision`),
  never the same-day label and never a back-fill. Annualised mean net return
  in detected bear regimes: strategy 0.6%, buy-and-hold -7.8%; in detected
  bull regimes: strategy 10.8%, buy-and-hold 4.6% (synthetic, one path).
* Flip aftermath: mean 10-day strategy P&L after a flip to bear -0.15%,
  worst -3.2% over 17 flips (synthetic, one path).

## 9. Known failure modes

1. Regimes are detected late. On the seed-7 synthetic path (full-sample fit,
   filtered argmax) the mean bear-entry lag is 1.5 days and the mean exit lag
   1.0 day. Real regimes are less separated, so the lag is longer.
2. Label switching across refits. States are re-labelled at every refit by
   the posterior-weighted vol feature mean (highest vol is bear).
3. Choice of K. BIC and out-of-sample likelihood disagree on the synthetic
   panel (section 2).
4. A 2-state fit on a 3-state world merges the transition state: state vols
   6.3% / 26.6% against true 10% / 18% / 38%, log-likelihood 9,171 against
   9,302 for K=3 (synthetic, seed 7).
5. Gaussian emissions mis-assign extremes. Hysteresis and the causal
   minimum-duration confirmation absorb short flickers, at a cost of 0.08
   Sharpe for a 5-day confirmation on synthetic data (section 4).
6. Expanding z-scores adapt slowly to structural breaks.
7. The walk-forward HMM uses one EM start and 50 iterations at every refit, a
   setting chosen for run time, not accuracy.

## 10. Edge-case and numerical validation

In `tests/test_edge_cases.py` and `tests/test_degenerate_regimes.py`:

* identical observations, state collapse and a singleton cluster with a
  50-sigma outlier keep EM finite and covariances positive-definite;
* a 50-sigma day does not underflow the forward pass;
* K=1 gives a valid fit with infinite expected duration;
* short series, NaN and Inf inputs, invalid K, thresholds, windows, costs and
  execution parameters raise `ValueError`;
* probabilities exactly at the hysteresis thresholds do not flip;
* K=2 on one-regime data returns finite parameters with state means within
  3 sigma of each other;
* label switching: vol-sorted labels and the decoded economic path are
  invariant to state permutations; `match_permutation` recovers every planted
  relabelling;
* absorbing, identity and random transition matrices satisfy `pi P = pi`
  with the documented degeneracy flags;
* truncating the future leaves every filtered row unchanged and moves the
  smoothed rows.
