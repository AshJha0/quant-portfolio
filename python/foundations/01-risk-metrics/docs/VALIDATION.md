# Validation: Single-Asset Equity Risk Metrics

This document answers documentation-contract items 3 and 4: **how this was
validated** (analytic checks, what "good" output looks like on the bundled
data) and **where it fails** (known failure modes with reproducible
examples). Every number below is produced by the committed code,
`python examples/run_pipeline.py` regenerates the report;
`pytest -q` (217 tests, offline, seeded) enforces the analytic checks
permanently.

Bundled synthetic data: `eq_risk_metrics.data.synthetic.generate(seed=2,
n_days=2520)`, a two-regime (calm/stressed) Markov-switching model with
negatively skewed Student-t(6) shocks, ~10 years of business days ending
at a fixed anchor date of 2026-08-14 (the anchor is fixed so the sample,
and therefore every number in this document, is identical no matter when
the generator is run). It is **not real market
data**; see `docs/METHODOLOGY.md` and the README for why it is shaped the
way it is.

---

## 1. Analytic checks (what the test suite enforces)

| Check | Result | Test |
|---|---|---|
| Gaussian VaR equals the exact closed form `-(mu + sigma*z)` | matches to abs 1e-12, 4 confidence levels, seeded normal sample | `test_var_parametric_matches_closed_form_exactly` |
| Gaussian VaR on a standardised (mean 0, std 1) sample matches the raw standard-normal quantile | matches to abs 1e-9 | `test_var_parametric_zero_mean_unit_scale_matches_standard_normal_quantile` |
| Cornish-Fisher VaR collapses to Gaussian VaR when skew = kurtosis = 0 (monkeypatched) | matches to abs 1e-12, 3 confidence levels | `test_cornish_fisher_collapses_to_gaussian_when_skew_kurtosis_zero` |
| Cornish-Fisher VaR differs from Gaussian on fat-tailed (Student-t df=8) data inside the CF domain | differs by > 1e-6 at 99% | `test_cornish_fisher_differs_from_gaussian_with_real_skew_kurtosis` |
| Historical VaR is the order statistic `L(ceil(n alpha))`; ES is Acerbi-Tasche with the fractional boundary weight, including ties | hand-computed, abs 1e-15 | `test_var_historical_hand_computed_95/99`, `test_var_historical_is_order_statistic_not_interpolated`, `test_expected_shortfall_is_acerbi_tasche_with_fractional_atom`, `test_expected_shortfall_on_ties_uses_the_atom_weight` |
| Expected Shortfall >= historical VaR at the same confidence (holds for every sample by construction) | holds on 20 seed/confidence combinations, Student-t + normal mixtures | `test_expected_shortfall_greater_or_equal_to_var` |
| ES-VaR gap is larger for a fat-tailed sample than a thin-tailed one at 99%  | holds on a 5,000-point normal vs Student-t(df=3) comparison | `test_expected_shortfall_coherence_intuition_fat_tails_exceed_thin_tails` |
| `max_drawdown` on a hand-built 7-point price series with a known, non-terminal peak/trough | max drawdown -50.00% exactly, correct peak/trough dates | `test_max_drawdown_hand_built_series`, `test_max_drawdown_peak_is_local_not_global_before_trough` |
| Sharpe/Sortino on a constructed 6-point return series, cross-checked against an independently written mean/std formula | matches to rel 1e-10 | `test_sharpe_ratio_hand_computed_zero_rf`, `test_sortino_ratio_hand_computed` |
| Sortino > Sharpe when losses are tight and gains are dispersed (asymmetric risk) | holds on a 500-point constructed sample | `test_sortino_ge_sharpe_when_downside_vol_below_total_vol` |
| Jarque-Bera does not reject a large (n=5000) normal sample | p = well above 0.05 | `test_jarque_bera_does_not_reject_large_normal_sample` |
| Jarque-Bera rejects a fat-tailed (Student-t df=3) sample and a skewed (lognormal) sample | p < 0.05, both cases | `test_jarque_bera_rejects_fat_tailed_sample`, `test_jarque_bera_rejects_skewed_sample` |
| EWMA volatility is the zero-mean RiskMetrics recursion seeded with mean(r^2) over the first min(30, n) returns | hand recursion, rel 1e-14 | `test_ewma_volatility_two_points_hand_computed`, `test_ewma_volatility_is_zero_mean_riskmetrics_recursion`, `test_ewma_volatility_seed_uses_only_the_seed_window` |
| Sortino uses the downside deviation `sqrt(mean(min(excess,0)^2))` over all observations | hand-computed, rel 1e-13 | `test_sortino_uses_downside_deviation_over_all_observations` |
| Cornish-Fisher domain: `g'(z) > 0` on `[-z_a, z_a]`, closed form | 6 hand-checked (skew, kurtosis, alpha) cases | `test_cornish_fisher_valid_closed_form`, `test_var_cornish_fisher_raises_outside_domain` |
| Kupiec refuses NaN VaR or NaN return days | `ValueError` | `test_kupiec_raises_on_all_nan_var`, `test_kupiec_raises_on_nan_return_day` |
| Every public callable: non-finite input raises; finite input gives finite output or `ValueError` | hypothesis, 60 examples per callable per property | `tests/test_fuzz_public_api.py` |
| Rolling volatility, window=2 closed form `std = |a-b|/sqrt(2)` | matches to rel 1e-12 | `test_rolling_volatility_window_two_hand_computed` |
| Synthetic data generator is deterministic given a seed; reproduces excess kurtosis (fat tails) | exact reproduction; kurtosis > 0.5 on 10y sample | `test_generate_is_deterministic_given_same_seed`, `test_generate_reproduces_stylised_facts_fat_tails` |
| Kupiec LR statistic is 0 when the observed exception rate equals the nominal rate | 0 to abs 1e-6 (the residual is the `1 - 0.99` representation error), p-value 1 | `test_kupiec_statistic_is_exactly_zero_when_observed_rate_equals_nominal` |
| Kupiec LR statistic matches an independently hand-computed closed-form value (n=100, x=5, p=1%) | LR = 8.258217002871675, p = 0.004056795256739709, to rel 1e-10 | `test_kupiec_statistic_matches_a_hand_computed_value` |
| Kupiec p-value equals the chi-squared(1) upper tail at its own statistic | rel 1e-12 on an arbitrary Student-t sample | `test_kupiec_statistic_equals_the_chi2_survival_of_its_own_statistic` |
| Kupiec statistic stays finite at both degenerate corners (0 exceptions, all-days exceptions), where the `0 log 0` convention is required | finite, positive, rejects in both cases | `test_kupiec_zero_exceptions_is_finite_and_rejects_an_absurdly_high_var`, `test_kupiec_all_days_exceptions_is_finite` |
| Kupiec is invariant to exception *clustering* (unconditional coverage only, by construction) | identical statistic for 5 spread-out vs 5 consecutive exceptions, rel 1e-15 | `test_kupiec_is_blind_to_clustering_by_construction` |
| Empty input raises an informative `ValueError` (not a raw `IndexError`) from all four VaR/ES estimators | message names the function; 4 estimators | `test_var_and_es_on_empty_series_raise_informative_value_error` |
| Out-of-range confidence (0, 1, 1.5, -0.1, NaN, inf) raises `ValueError` rather than returning a silent `NaN` | 4 estimators x 6 bad values | `test_var_and_es_reject_confidence_outside_the_open_unit_interval` |
| Non-finite (`inf`/`NaN`) returns are rejected rather than silently poisoning the estimate | 4 estimators x 3 bad values | `test_var_and_es_reject_non_finite_returns` |

## 2. What "good" looks like on the bundled synthetic data

Running `python examples/run_pipeline.py` on the bundled sample
(seed=2, 2,519 daily returns, 2016-12-20 to 2026-08-14) produces the
report below (synthetic data):

```
--- Volatility ---
Annualised volatility (full sample) : 15.64%
Latest 21-day rolling volatility    : 10.26%
Latest EWMA (lambda=0.94) volatility: 11.61%

--- Value at Risk (1-day, % of position value) ---
  95% confidence:
    Historical VaR      : 1.46%
    Gaussian VaR        : 1.56%
    Cornish-Fisher VaR  : not reported (skew/kurtosis outside the CF domain)
    Expected Shortfall  : 2.53%
  99% confidence:
    Historical VaR      : 3.18%
    Gaussian VaR        : 2.23%
    Cornish-Fisher VaR  : not reported (skew/kurtosis outside the CF domain)
    Expected Shortfall  : 4.34%


--- VaR backtest (Kupiec proportion-of-failures, in-sample) ---
  95% Historical:  125 exceptions in 2519 days (expected 126.0), LR = 0.01, p = 0.931 -> not rejected
  95% Gaussian  :  116 exceptions in 2519 days (expected 126.0), LR = 0.85, p = 0.357 -> not rejected
  99% Historical:   25 exceptions in 2519 days (expected 25.2), LR = 0.00, p = 0.970 -> not rejected
  99% Gaussian  :   60 exceptions in 2519 days (expected 25.2), LR = 35.02, p = 0.000 -> REJECT

--- Drawdown ---
Maximum drawdown : -28.27%
Peak date        : 2021-06-18
Trough date      : 2021-11-05

--- Risk-adjusted performance (rf = 3% annual) ---
Sharpe ratio  : 0.85
Sortino ratio : 1.16

--- Distribution diagnostics ---
Skewness         : -0.512
Excess kurtosis  : +9.284
Jarque-Bera stat : 9157.0 (p = 0.00e+00)
Normality rejected at 5%: True
```

Cross-checking against the four observations flagged in the original
project notes as "what the results should mean" (now the acceptance
criteria for a healthy run):

1. **Historical 99% VaR (3.18%) > Gaussian 99% VaR (2.23%).** Confirmed.
   The gap is +0.95pp: on this synthetic sample the normal distribution
   understates the 99% daily loss threshold by about 30% relative to the
   empirical one. At 95% the sign flips (Gaussian 1.56% above historical
   1.46%). The normal fit is optimistic in the tail and slightly
   conservative in the body, which is the shape a fat-tailed distribution
   produces.
2. **ES noticeably above VaR at the same confidence.** Confirmed. At 99%,
   ES (4.34%) is 1.16pp above historical VaR (3.18%).
3. **JB test rejects normality.** Confirmed. p is 0 in double precision,
   with excess kurtosis +9.28 and skewness -0.51.
4. **Volatility is time-varying (clustering).** Confirmed. 21-day rolling
   vol ranges from 5.33% (2022-05-23) to 45.72% (2018-11-01) on this
   synthetic sample. The largest EWMA gap is 41.2% on 2021-03-08 against
   a full-sample figure of 15.6% (§3.4).
5. **The Gaussian VaR fails its coverage backtest at 99%; the historical
   one does not.** Confirmed. The Gaussian 99% VaR was exceeded on 60 of
   2,519 days against 25.2 expected (Kupiec LR = 35.02, p = 0.000,
   rejected). The historical 99% VaR was exceeded 25 times (LR = 0.00,
   p = 0.970, not rejected). At 95% neither is rejected.
6. **Cornish-Fisher is outside its domain on this sample.** With skewness
   -0.51 and excess kurtosis +9.28, `cornish_fisher_valid` is false at
   both 95% and 99%, so `var_cornish_fisher` raises and the pipeline
   prints "not reported". The expansion is not monotone over the
   quantiles it would use.

On real market data expect the same qualitative findings. The exact
numbers differ by ticker and window.

**Reading the backtest.** Both estimators are scored on the same
window they were fitted to, so the historical estimator's pass is close to
tautological, the empirical 1% quantile of a sample is exceeded by ~1% of
that sample by construction. What the in-sample test *can* falsify is a
model whose distributional assumption is wrong, which is exactly what
happens to the Gaussian one. A genuine control re-estimates VaR on a
trailing window and scores the *next* day's return (out-of-sample), and
adds an independence test for clustering; see `docs/DESK_GUIDE.md` §3.

## 3. Known failure modes (reproducible examples)

### 3.1 Very short samples (5-day example)

```python
import pandas as pd
from eq_risk_metrics import var_historical, expected_shortfall

dates = pd.bdate_range("2024-01-02", periods=5)
returns = pd.Series([0.01, -0.02, 0.005, 0.015, -0.008], index=dates)
var_historical(returns, 0.95)       # 0.02  (2.00%)
expected_shortfall(returns, 0.95)   # 0.02  (2.00%)
```

The Kupiec test is no help here either, and that is worth being explicit
about: with `n=5` the asymptotic chi-squared reference distribution is
meaningless, and even at `n=250` (a full year, the standard regulatory
window) a 99% VaR with **twice** the expected exception rate is not
rejected at 5%, pinned by
`tests/test_backtest.py::test_kupiec_power_is_low_on_a_short_window`.
Short samples do not just make the risk estimate noisy; they make the
test that is supposed to catch a bad risk estimate powerless.

Both return a number. With `n = 5` and 95% confidence, `k = 0.25` and
`m = 1`, so VaR and ES are both the single largest loss, 2.00%. Neither
number is a serious risk estimate
(`tests/test_edge_cases.py::test_five_day_sample_produces_a_number_but_is_statistically_meaningless`).
The functions do not enforce a minimum sample size. That is the
caller's job. A common desk convention is at least 252 observations
before quoting 99% VaR.

### 3.2 Exactly-constant returns

```python
import pandas as pd
from eq_risk_metrics import sharpe_ratio, var_cornish_fisher, normality_report

returns = pd.Series([0.002] * 30)
sharpe_ratio(returns)              # ValueError: excess returns are constant
var_cornish_fisher(returns, 0.95)  # ValueError: the sample is constant
normality_report(returns)          # ValueError: the sample is constant
```

Skewness and kurtosis of a constant sample are 0/0, and the Sharpe
denominator is zero up to rounding. All three functions raise. Pinned by
`test_constant_returns_sharpe_raises`,
`test_constant_returns_cornish_fisher_raises` and
`test_normality_report_on_exactly_constant_series_raises`. A series with
tiny noise is not constant: `test_near_constant_returns_cornish_fisher_matches_gaussian`
shows Cornish-Fisher collapses to Gaussian VaR there.

### 3.3 All-positive return series (Sortino)

```python
import numpy as np, pandas as pd
from eq_risk_metrics import sortino_ratio

returns = pd.Series(np.linspace(0.001, 0.02, 20))
sortino_ratio(returns, rf_annual=0.0)   # ValueError: downside deviation is zero
```

The downside deviation `sqrt(mean(min(excess, 0)^2))` is zero when no
excess return is negative, so the ratio is undefined and the function
raises (`test_all_positive_returns_sortino_raises`).

### 3.4 Regime divergence: EWMA vs full-sample volatility

On the bundled synthetic sample, the largest full-sample-vs-EWMA gap is
on 2021-03-08: EWMA volatility reads 41.2% against a full-sample figure
of 15.6%, a 2.6x divergence. The 21-day rolling vol on that date is
39.7%.

```python
gap = (ewma_vol - full_sample_vol).abs()
gap.idxmax()   # Timestamp('2021-03-08')
```

When EWMA or rolling vol diverges sharply from the full-sample number,
any figure computed with full-sample `sigma` is stale. That includes
`var_parametric` and `var_cornish_fisher`, which use the whole-sample
mean and standard deviation. A report should flag the divergence, for
example when `abs(ewma_vol - full_sample_vol) / full_sample_vol > 0.5`.

### 3.5 Numerical limits and input validation

Every public function raises `ValueError` on empty input, on NaN or
infinite values, and on a length that is too short for the statistic.
`tests/test_fuzz_public_api.py` checks this with hypothesis for every
callable in `eq_risk_metrics.__all__`. A new public function without a
fuzz entry fails that test.

| Input | Behaviour |
|---|---|
| Confidence not in (0, 1), NaN or inf | `ValueError` (all VaR/ES functions and `kupiec_pof_test`) |
| Confidence not in (0.5, 1) | `ValueError` in `var_cornish_fisher` |
| NaN or inf in returns, prices or VaR | `ValueError` everywhere, including `count_var_exceptions` and `kupiec_pof_test` |
| Fewer than 2 observations | `ValueError` in `annualised_volatility`, `var_parametric`, `var_cornish_fisher`, `sharpe_ratio`; `simple_returns` and `log_returns` need 2 prices |
| One observation | `var_historical` and `expected_shortfall` return that loss |
| `rolling_volatility` window | must be an int >= 2; a window longer than the series gives all-NaN warm-up values |
| `ewma_volatility` | `lam` in (0, 1), `seed_window` a positive int |
| `max_drawdown` | prices must be strictly positive |

- **Extreme confidence levels:** at `1 - 1e-9` the historical estimator
  *saturates* at the worst observed loss (it cannot report a loss the
  sample has never shown), while the Gaussian one extrapolates past it,
  the two are compared directly in
  `test_parametric_var_at_extreme_confidence_grows_without_saturating`.
  Neither number is trustworthy; they fail in opposite, instructive
  directions. At `1e-9` confidence, "VaR" is minus the largest *gain* in
  the sample, a negative loss. It stays finite and is not special-cased.
- **Confidence exactly 0.5:** `var_parametric` returns exactly `-mu`
  (`z = Phi^{-1}(0.5) = 0`), tested to abs 1e-12, VaR at 50% confidence
  has lost its usual "tail risk" meaning (it is just minus the mean) but
  the function does not special-case this away.
- **Confidence near the boundary (0.9999):** returns a finite, larger
  number than 99% VaR, as expected of a monotone quantile function, no
  numerical blow-up, but at that confidence a 2,519-observation historical
  sample has essentially no real data at that quantile; the number is
  almost entirely an extrapolation of the two smallest observations. Extreme
  value theory (`docs/METHODOLOGY.md` §3.3) would do better here.
