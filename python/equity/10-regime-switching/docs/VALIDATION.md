# Validation — Equity Regime-Switching Strategy

Documentation contract items 3–4: **how the model was validated** and
**where it fails**. Every number below is reproduced by
`python examples/run_pipeline.py` (53s) or by the test suite
(`pytest -q`, 152 tests, ~20s, offline, seeded).

## 1. Cross-model consistency checks

| Check | Tolerance | Result |
|---|---|---|
| PCA (scratch, correlation-matrix eigendecomposition) vs `sklearn.decomposition.PCA` on standardized data — components & explained variances | 1e-8, up to sign | pass (`test_pca.py`) |
| GMM (scratch EM) vs `sklearn.mixture.GaussianMixture`, per-observation log-likelihood on the same data | 1e-4 (measured delta **3.4e-14**) | pass (`test_gmm.py`) |
| GMM parameters evaluated by independent scipy density | 1e-8 | pass |
| HMM: our fitted parameters loaded into `hmmlearn.GaussianHMM.score` vs our log-likelihood | 1e-6 | pass (`test_hmm.py`) |
| HMM: independent hmmlearn EM fit vs our fit, per-observation log-likelihood | 1e-3 | pass |
| Stationary distribution `πA = π` | 1e-12 | pass |
| Expected durations vs `1/(1−a_ii)` identity | 1e-12 | pass |
| EM monotonicity (GMM and Baum–Welch), every iteration | ≥ −1e-7 | pass |

## 2. Parameter recovery on synthetic ground truth

3-state panel, 2 520 days, seed 7; full-sample 3-state HMM on index
returns, permutation-matched to the truth:

| state | true μ (ann) | est μ | true σ (ann) | est σ | true p_stay | est p_stay | true E[dur] | est E[dur] |
|---|---|---|---|---|---|---|---|---|
| bull | 0.150 | 0.148 | 0.100 | ~0.10 | 0.985 | 0.986 | 66.7d | 72.6d |
| transition | 0.020 | −0.007 | 0.180 | ~0.18 | 0.940 | 0.943 | 16.7d | 17.4d |
| bear | −0.300 | −0.456 | 0.380 | ~0.38 | 0.950 | 0.945 | 20.0d | 18.3d |

Stationary distribution 0.574/0.275/0.151 vs empirical occupancy
0.577/0.258/0.165. Drift in the bear state is estimated with the widest
error (−0.46 vs −0.30): only 416 bear days and σ=38% ⇒ a mean-return
standard error of ~30% annualised — **regime means are the least reliable
parameters**, which is why the strategy conditions on vol-ranked labels,
not on estimated means.

Recovery/accuracy tests (`test_hmm.py`): transition diagonal within 0.02,
means within 0.10, vols within 0.08 on a well-separated 2-state series;
Viterbi path accuracy > 90% (measured ≈ 99% on that series).

**BIC k-selection** (GMM on index returns): BIC = −16 205 (k=1),
−17 569 (k=2), **−17 577 (k=3, selected)**, −17 557 (k=4), −17 534 (k=5).
Recovers the true k=3. Tests also cover k=2 recovery on a 2-component
mixture and k=3 on a 3-component mixture.

## 3. The filtered-vs-smoothed honesty table

Around the first true bull→bear transition (index 320, 2016-03-25):

| day | true state | P_bear filtered | P_bear smoothed |
|---|---|---|---|
| t−3 | transition | 0.016 | 0.137 |
| t−2 | transition | 0.014 | 0.318 |
| t−1 | transition | 0.103 | 0.743 |
| t+0 | bear | 0.367 | 0.946 |
| t+1 | bear | 0.973 | 0.998 |

The smoothed probability is already 74% bear the day *before* the regime
changes — it has seen the future. The filtered probability crosses only at
t+1. **Backtesting on smoothed probabilities is lookahead bias**; the
mutation tests (`test_detection.py::TestCriticalCausality`) prove the
filtered path is bit-identical under future data mutations while the
smoothed path is not, and the full-pipeline mutation test
(`test_backtest.py::TestNoLookahead`) proves the same for every ledger row.

## 4. Null-data false-positive guard

On no-regime GBM panels (constant drift/vol/correlation) the machinery must
find nothing:

* Walk-forward regime strategy vs buy-and-hold, net CAGR excess by seed:
  **−4.4%, −7.6%, −3.1%** (seeds 1–3). The overlay *loses* money on null
  data (missed drift + costs) — no spurious alpha. Test asserts the mean
  excess < +2%.
* BIC on null returns: k=1 selected (−10 597 vs −10 575 for k=2,
  −10 553 for k=3) — **fewer states preferred**, no hallucinated regimes.

## 5. Known failure modes

1. **Regimes are detected LATE by construction.** The filter needs
   evidence to accumulate. Measured on the synthetic 3-state panel
   (full-sample fit, filtered argmax): mean bear-entry detection lag
   **1.5 days** (median 1), mean bear-exit lag **1.0 days**. On real data
   with less separation the lag is materially longer; the flip-aftermath
   report (mean 10-day P&L after a flip to bear: +0.2%, worst −3.2%)
   quantifies what the lag costs. A regime model is a *loss limiter*, not
   a crash predictor.
2. **Label switching across refits.** State index 0 in one refit can be
   state 2 in the next. Raw indices are never used downstream: states are
   re-labelled at every refit by sorting on the (posterior-weighted) vol
   feature mean — highest vol = bear (`labels_from_vol_means`,
   test-enforced). Comparisons with ground truth use explicit permutation
   matching (`match_permutation`).
3. **Overfitting k.** BIC on a finite sample can select k that fits noise;
   more states ⇒ shorter durations ⇒ more churn. Governance: cap k at 3
   and require BIC improvement AND economically distinct state vols before
   adding a state.
4. **A 2-state fit on a 3-state world** merges the transition state into
   its neighbours: measured k=2 state vols 6.3%/26.6% (ann.) against true
   10%/18%/38%, log-likelihood 9 171 vs 9 302 for k=3. Consequence: the
   merged "calm" state understates risk just before turns, and de-risking
   becomes all-or-nothing (no scaled transition book).
5. **Gaussian emissions mis-assign extremes.** A single fat-tail day can
   flip the filtered state; hysteresis (0.70/0.30) and the min-duration
   filter absorb most single-day flickers (constructed-case tests in
   `test_detection.py`).
6. **Non-stationarity of the feature map** (see assumptions register):
   expanding z-scores adapt slowly to structural breaks.

## 6. Edge-case and numerical validation

All in `tests/test_edge_cases.py` unless noted:

* all-identical observations: EM stays finite via `reg_covar` (GMM & HMM);
* state collapse (3 states on 2 clusters): finite, rows still sum to 1,
  `πA = π` still holds;
* singleton cluster with a 50σ outlier: covariance stays positive-definite;
* 50-sigma single day: forward pass does not underflow (log-emission shift);
* k=1: valid fit, expected duration = ∞;
* series too short, NaN inputs, invalid k / thresholds / windows / costs:
  `ValueError` with informative messages (per-module validation tests);
* probabilities exactly at hysteresis thresholds: no flip (strict
  inequalities, documented convention);
* ledger arithmetic: exact hand-computed scenario, entry cost included;
* vol targeting realizes the ex-ante target within 15% on constant-vol data.

### 6.1 Degenerate-regime suite (`tests/test_degenerate_regimes.py`)

**Single-regime (null) data.** Fitting K = 2 to genuinely one-regime data
must fail *visibly*, not silently: all parameters stay finite, every fitted
covariance stays positive-definite (the `reg_covar` ridge), rows still sum to
1 to 1e-12, and the two state means end up within 3σ of each other — the
honest signature of "there is no second regime here". The population-level
check is that the decoded path on a null GBM panel flip-flops strictly more
than on a true 2-regime panel, which is the diagnostic that rejects a
spurious model.

**Label switching.** EM numbers the states arbitrarily, so the economic
labelling must be permutation-invariant. Tested three ways: (i) vol-sorted
labels attach to the right states under an explicit index permutation;
(ii) the *decoded economic path* is bit-identical when the state numbering
and the vol means are permuted together; (iii) `match_permutation` recovers
every planted relabelling of a 3-component mean set exactly (all 4
permutations checked).

**Degenerate transition matrices.**

| Matrix | Expected behaviour | Status |
|---|---|---|
| absorbing state `[[1,0],[0.2,0.8]]` | `π = (1, 0)` exactly; `πP = π` to 1e-12; duration `(∞, 5)` | tested |
| identity `I₃` | reducible — *every* distribution is stationary; solver returns the min-norm (uniform) one, `πP = π` still exact to 1e-14, all durations `∞` flag the degeneracy | tested + documented in the docstring |
| K = 1 fit | single absorbing state, `π = (1,)`, duration `∞` | tested |
| random row-stochastic, K ∈ {2,3,5} | `πP = π` to 1e-12, `Σπ = 1`, `π ≥ 0` over 60 random chains | property test |
| non-stochastic rows / non-square / NaN / negative entries | informative `ValueError` | tested |

**Fitted-chain invariants (property tests over seeds).** EM must never emit a
row that fails to normalise, and never a hard zero — the 1e-12 clip is what
keeps the Viterbi log-space recursion free of `-inf` propagation. Both are
asserted across four seeds, with a Viterbi decode run afterwards to confirm.

**Probability invariants.** Filtered and smoothed probabilities are valid
distributions (rows in [0,1], summing to 1 to 1e-10); forward and
forward-backward report the same log-likelihood to 1e-12; expected transition
counts `Σξ` total exactly `T − 1`; EM log-likelihood is monotone
non-decreasing over iterations.

**Causality (the tradeability property).** Truncating the future leaves every
filtered row bit-stable (drift ≤ machine epsilon) while shifting the smoothed
rows by 4+ orders of magnitude more — the quantitative statement of why
filtered probabilities may be traded and smoothed ones may not. The boundary
identity `γ_T = α̂_T` (β_T = 1) is checked alongside.

### 6.2 NaN/Inf rejection (bug found by this test pass)

`fit_hmm` and `fit_gmm` checked `np.isnan(x)` only. `Inf` therefore passed
validation and poisoned every sufficient statistic — means, covariances and
the log-likelihood all came back `NaN`, accompanied by nothing louder than a
handful of NumPy `RuntimeWarning`s. Both fitters now require `np.isfinite`
and raise an informative `ValueError`; `stationary_distribution` and
`expected_durations` gained the same guard plus shape and
probability-domain checks. Regression-tested for both `+Inf`, `-Inf` and
`NaN`.

## 7. Walk-forward result (synthetic 3-state panel, net of 5 bps)

| | strategy | buy & hold | 200d-MA rule |
|---|---|---|---|
| CAGR | **10.6%** | 1.6% | 1.6% |
| ann. vol | 8.5% | 15.1% | 11.0% |
| Sharpe | **1.22** | 0.18 | 0.20 |
| max drawdown | **10.2%** | 44.4% | 19.2% |
| final equity (7.5y) | 2.12 | 1.12 | 1.12 |

Per-regime attribution: in detected bear regimes the strategy is flat
(+0.9% ann.) while buy-and-hold loses −15.8% ann. with a 50.4% drawdown —
the entire edge is *not losing in bears*, exactly what a regime overlay is
for. These numbers are on synthetic data whose regimes are, by
construction, detectable; they are an upper bound, not a live-trading
forecast.
