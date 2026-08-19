# Methodology — Equity Regime-Switching Strategy

This document answers, in writing: **why this model**, **what the maths is**,
and **what assumptions it stands on** (documentation contract items 1–2).

## 1. The problem

Equity risk premia are not stationary. Long stretches of low-vol drift are
punctuated by high-vol, high-correlation drawdowns in which the usual
diversification fails ("correlations go to one"). A strategy that knew the
current regime could hold the equity factor in calm markets and de-risk in
stressed ones. The regime is not observable — it must be inferred from
observable features (realized vol, dispersion, average correlation,
drawdown, trend, credit/term proxies).

Pipeline: **features → PCA → HMM (or GMM) → filtered regime probabilities →
regime-conditional allocation with hysteresis and vol targeting →
walk-forward backtest → regime-conditional risk report.**

## 2. Why a Gaussian HMM? — comparison against alternatives

| Criterion | **Gaussian HMM** (chosen) | GMM (no dynamics) | Threshold rules (e.g. VIX > 25, 200d MA) | Markov-switching regression (Hamilton) |
|---|---|---|---|---|
| Captures regime **persistence** | Yes — explicit transition matrix; expected duration `1/(1−p_ii)` | No — i.i.d. mixture, classifies each day independently | Implicitly, via the smoothness of the indicator | Yes — same Markov chain |
| Produces **probabilities** (sizable signal) | Yes — filtered `P(s_t \| x_{1..t})` | Yes, but per-day (noisy, flip-flops) | No — binary | Yes |
| Online / causal inference | Yes — forward filter is one recursion per day | Yes | Yes | Yes |
| Number of parameters (K states, D features) | `K² + KD + KD(D+1)/2` | `K−1 + KD + KD(D+1)/2` | ~1–2 (thresholds) | as HMM + regression betas |
| Interpretability | State means/vols map to bull/bear economics | Same, minus dynamics | Highest | High |
| Main failure | Gaussian emissions, label switching, late detection | Daily flip-flop churn | Arbitrary threshold, no uncertainty | Needs a specified regression; more fragile in high D |
| Verdict | **Best trade-off**: persistence + probabilities + tractable EM | Kept as k-selection tool (BIC) and cross-check | Kept as the *benchmark* (200d-MA rule in every backtest) | Overkill here: we model features, not a conditional mean equation |

The GMM is not discarded: it is the ignore-the-dynamics special case of the
HMM (transition matrix rows all equal), so it serves two roles here —
**BIC-based selection of the state count** and an **independent cross-check**
of emission parameters. The threshold rule is kept as an honest benchmark:
if the HMM cannot beat a 200d moving average net of costs, it is not
earning its complexity.

## 3. The model

Hidden state `s_t ∈ {1..K}` follows a first-order Markov chain with
transition matrix `A` (rows sum to 1). Observations `x_t ∈ R^D` (feature
vector or its leading principal components) are conditionally Gaussian:
`x_t | s_t = k ~ N(μ_k, Σ_k)` with full covariance.

* **Forward filter** (causal, tradeable): `α̂_t(k) = P(s_t = k | x_{1..t})`,
  computed with Rabiner scaling plus a per-step log-emission shift so that
  50-sigma outliers do not underflow. The log-likelihood is the sum of log
  scaling constants.
* **Smoother** (diagnostic only): `γ_t(k) = P(s_t = k | x_{1..T})` from the
  backward pass. **Filtered vs smoothed is the central honesty point of the
  project**: smoothed probabilities peek at the future and are visibly
  "early" around turns (see the t−5..t+7 table in `examples/run_pipeline.py`
  and docs/VALIDATION.md). All trading uses filtered probabilities —
  test-enforced by mutation tests.
* **Baum–Welch EM** re-estimates `A, μ_k, Σ_k`; the log-likelihood is
  monotone non-decreasing (asserted at every iteration in tests). Ridge
  `reg_covar` on covariance diagonals guards against state collapse.
* **Viterbi** gives the most likely historical path (reporting only).
* **Stationary distribution** solves `πA = π` (to 1e-12) and expected
  durations are `1/(1−a_ii)`.

PCA (from scratch, eigendecomposition of the correlation matrix, sign fixed
so the largest loading is positive) compresses the 9 features to 3
orthogonal factors; on the synthetic panel PC1 (62% of variance) is a
"market stress" factor loading on all vol/correlation/drawdown measures.

### Portfolio construction

* bull → weight 1.0 in the equity factor, bear → 0.0 (defensive/cash),
  transition → 0.5.
* **Hysteresis band** on the filtered bear probability: enter bear at
  `p > 0.70`, exit at `p < 0.30` (strict inequalities; the band is closed).
  A single 0.5 threshold flips every time noise crosses one line; the band
  requires a decisive move through 0.4 of probability mass to flip, so
  oscillations inside the band cost **zero** trades. Measured effect: on a
  noisy probability path turnover falls from 263 to 48 (−82%); on a genuine
  two-switch path from 24 to 8 while both switches are still captured
  (tests/test_strategy.py).
* **Vol targeting**: position scaled by `target_vol / trailing 21d realized
  vol`, capped at 1.5x. Trailing-only, so ex-ante.

## 4. Assumptions register

Each assumption states *what breaks if violated*.

1. **Gaussian emissions.** Daily returns/features have fat tails; a 5-sigma
   day under the true law is a ~10-sigma day under a Gaussian, so extreme
   days are **mis-assigned with overconfidence** to the high-vol state (or
   distort its Σ). Violation ⇒ spurious bear flips on single outlier days.
   Mitigations: hysteresis band, min-duration filter; extension: t-emissions.
2. **Constant transition matrix.** `A` is assumed time-invariant. In
   reality crash dynamics differ (2020 was faster than 2008). Violation ⇒
   expected durations and filtered probabilities mis-calibrated exactly when
   they matter most; the walk-forward refit (every 63d, expanding) adapts
   only slowly.
3. **The regime count K is stable and known.** We select K by BIC on
   training data. If the world adds a state (e.g. an inflation regime),
   the fit absorbs it into existing states — shown quantitatively in
   VALIDATION.md §5 (k=2 on a 3-state world merges bull+transition).
4. **First-order Markov dynamics.** Sojourn times are geometric. Real
   bear markets may have duration memory (the longer in, the likelier out),
   which geometric sojourns cannot express ⇒ duration estimates biased.
5. **Stationary feature → regime mapping.** Expanding z-scores assume the
   long-run mean/vol of each feature is meaningful. A structural break in a
   feature's level (e.g. structurally higher correlation after ETF-ization)
   ⇒ persistent z-score bias until the expanding window absorbs it.
6. **Features are observable without lag and tradeable next day.** We
   trade at the next close with linear costs (5 bps). Violation (gaps,
   liquidity holes in crises) ⇒ realized slippage exceeds modelled cost
   exactly in bear entries — quantified by the flip-aftermath report.
7. **The equity factor is investable long-only, cash earns zero.** No
   shorting in bear states; de-risking is the only defense. Violation is
   conservative (adding a short/hedge sleeve could only be layered on top).
8. **The data genuinely contain K distinguishable regimes.** EM will always
   return K states, whether or not they exist. *If violated* (a one-regime
   world fitted with K = 2): the fit does not fail — it returns two
   arbitrary, unstable partitions of the same distribution, and every
   downstream signal becomes noise trading with real transaction costs. The
   defences are diagnostic, not automatic: BIC model selection on the null
   panel prefers K = 1 (`test_null_guard.py`), the flip-flop rate on
   null data is materially higher than on true regime data, near-identical
   fitted state means signal non-separation, and a near-identity transition
   matrix (infinite expected durations) marks a degenerate chain. A desk
   must read these before trading the model, which is why they are surfaced
   on the fit object rather than buried.
9. **The chain is irreducible.** The stationary distribution is unique only
   for an irreducible chain. *If violated* (an absorbing state, or the
   identity matrix that EM approaches on single-regime data): infinitely
   many stationary distributions exist and `stationary_distribution` returns
   the minimum-norm one — mathematically valid (`πP = π` holds exactly) but
   economically arbitrary. Treat infinite `expected_durations` as the flag
   that the number reported by `stationary_distribution` is not meaningful.
10. **Inputs are finite.** Features must contain no `NaN` or `Inf`.
    *If violated*: a single non-finite observation propagates through the
    E-step into every mean and covariance, so the whole fit returns `NaN`.
    This used to happen silently for `Inf` (only `NaN` was screened); both
    are now rejected up front. `build_features` already drops the warm-up
    rows that are legitimately `NaN`, so a non-finite value reaching the
    fitter indicates a genuine data problem upstream.

## 5. Real-life scenarios and edge cases (each unit-tested)

| Scenario / edge case | Where documented | Test |
|---|---|---|
| Fast crash (COVID Mar-2020 analogue): detection lag costs P&L | DESK_GUIDE §4 | `test_backtest.py::TestNoLookahead`, flip-aftermath in `test_risk.py` |
| Slow bear (2022 analogue): regime model shines | DESK_GUIDE §4 | walk-forward beats B&H on 3-state panel (`examples/run_pipeline.py`) |
| False alarms in corrections (2015/2018 analogue) | DESK_GUIDE §4 | hysteresis turnover tests, min-duration filter tests |
| No-regime world (null GBM): no spurious alpha, BIC prefers 1 state | VALIDATION §4 | `test_null_guard.py` |
| All-identical observations / zero variance | VALIDATION §6 | `test_edge_cases.py::TestDegenerateData` |
| State collapse (more states than clusters) | VALIDATION §6 | `test_edge_cases.py` (GMM+HMM collapse tests) |
| Singleton/degenerate cluster | VALIDATION §6 | `test_gmm.py::test_degenerate_singleton_cluster_regularized` |
| 50-sigma outlier day (no underflow) | VALIDATION §6 | `test_edge_cases.py::TestNumericalExtremes` |
| Probability exactly at hysteresis thresholds | §3 above | `test_edge_cases.py::TestBoundaries`, `test_strategy.py` |
| Very short series / k=1 / invalid inputs | VALIDATION §6 | `test_edge_cases.py::TestSmallAndShort`, validation-error tests per module |
| Label switching across refits | VALIDATION §5, §6.1 | economic labelling tests in `test_detection.py`; permutation-invariance of the decoded path and `match_permutation` recovery in `test_degenerate_regimes.py` |
| Single-regime data fitted with K=2 | VALIDATION §6.1 | `test_degenerate_regimes.py::test_single_regime_*`, `test_gbm_null_panel_regimes_flip_more_than_true_regime_panel` |
| Absorbing / identity (reducible) transition matrices | VALIDATION §6.1 | `test_absorbing_state_stationary_and_duration`, `test_identity_transition_matrix_is_stationary_but_not_unique` |
| Non-finite (`NaN`/`Inf`) observations | VALIDATION §6.2 | `test_hmm_rejects_non_finite_observations`, `test_gmm_rejects_non_finite_observations` |
| Filtered-vs-smoothed causality (tradeability) | VALIDATION §3, §6.1 | `test_filtered_probabilities_are_causal` |

## 6. Conventions

Daily log-returns; vols annualised ACT/252; costs in bps of traded
notional, one-way; all randomness through explicit seeds
(`numpy.random.default_rng`); weights decided at close `t` earn the return
`t → t+1`.
