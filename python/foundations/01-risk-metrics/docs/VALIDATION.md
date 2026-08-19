# Validation — Single-Asset Equity Risk Metrics

This document answers documentation-contract items 3 and 4: **how this was
validated** (analytic checks, what "good" output looks like on the bundled
data) and **where it fails** (known failure modes with reproducible
examples). Every number below is produced by the committed code —
`python examples/run_pipeline.py` regenerates the report;
`pytest -q` (163 tests, offline, seeded) enforces the analytic checks
permanently.

Bundled synthetic data: `eq_risk_metrics.data.synthetic.generate(seed=2,
n_days=2520)` — a two-regime (calm/stressed) Markov-switching model with
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
| Cornish-Fisher VaR *differs* from Gaussian on genuinely fat-tailed (Student-t df=3) data | differs by > 1e-6 at 99% | `test_cornish_fisher_differs_from_gaussian_with_real_skew_kurtosis` |
| Historical VaR/ES hand-computed on a 7- and 20-point exactly-known sample (linear-interpolation percentile arithmetic worked out by hand) | matches to abs 1e-12 | `test_var_historical_hand_computed_95/99`, `test_expected_shortfall_hand_computed` |
| Expected Shortfall >= historical VaR at the same confidence (coherence property) | holds on 20 seed/confidence combinations, Student-t + normal mixtures | `test_expected_shortfall_greater_or_equal_to_var` |
| ES-VaR gap is larger for a fat-tailed sample than a thin-tailed one at 99% (coherence intuition) | holds on a 5,000-point normal vs Student-t(df=3) comparison | `test_expected_shortfall_coherence_intuition_fat_tails_exceed_thin_tails` |
| `max_drawdown` on a hand-built 7-point price series with a known, non-terminal peak/trough | max drawdown -50.00% exactly, correct peak/trough dates | `test_max_drawdown_hand_built_series`, `test_max_drawdown_peak_is_local_not_global_before_trough` |
| Sharpe/Sortino on a constructed 6-point return series, cross-checked against an independently written mean/std formula | matches to rel 1e-10 | `test_sharpe_ratio_hand_computed_zero_rf`, `test_sortino_ratio_hand_computed` |
| Sortino > Sharpe when losses are tight and gains are dispersed (asymmetric risk) | holds on a 500-point constructed sample | `test_sortino_ge_sharpe_when_downside_vol_below_total_vol` |
| Jarque-Bera does not reject a large (n=5000) genuinely normal sample | p = well above 0.05 | `test_jarque_bera_does_not_reject_large_normal_sample` |
| Jarque-Bera rejects a fat-tailed (Student-t df=3) sample and a skewed (lognormal) sample | p < 0.05, both cases | `test_jarque_bera_rejects_fat_tailed_sample`, `test_jarque_bera_rejects_skewed_sample` |
| EWMA volatility, 2-point closed form `var_1 = alpha(1-alpha)(r1-r0)^2` | matches to rel 1e-10 | `test_ewma_volatility_two_points_hand_computed` |
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
(seed=2, 2,519 daily returns, 2016-12-19 to 2026-08-14) produces:

```
--- Volatility ---
Annualised volatility (full sample) : 15.64%
Latest 21-day rolling volatility    : 10.26%
Latest EWMA (lambda=0.94) volatility: 10.50%

--- Value at Risk (1-day, % of position value) ---
  95% confidence:
    Historical VaR      : 1.45%
    Gaussian VaR        : 1.56%
    Cornish-Fisher VaR  : 1.51%
    Expected Shortfall  : 2.53%
  99% confidence:
    Historical VaR      : 3.17%
    Gaussian VaR        : 2.23%
    Cornish-Fisher VaR  : 4.64%
    Expected Shortfall  : 4.30%

--- VaR backtest (Kupiec proportion-of-failures, in-sample) ---
  95% Historical:  126 exceptions in 2519 days (expected 126.0), LR = 0.00, p = 0.996 -> not rejected
  95% Gaussian  :  116 exceptions in 2519 days (expected 126.0), LR = 0.85, p = 0.357 -> not rejected
  99% Historical:   26 exceptions in 2519 days (expected 25.2), LR = 0.03, p = 0.872 -> not rejected
  99% Gaussian  :   60 exceptions in 2519 days (expected 25.2), LR = 35.02, p = 0.000 -> REJECT

--- Drawdown ---
Maximum drawdown : -28.27%
Peak date        : 2021-06-18
Trough date      : 2021-11-05

--- Risk-adjusted performance (rf = 3% annual) ---
Sharpe ratio  : 0.85
Sortino ratio : 0.98

--- Distribution diagnostics ---
Skewness         : -0.512
Excess kurtosis  : +9.284
Jarque-Bera stat : 9157.0 (p = 0.00e+00)
Normality rejected at 5%: True
```

Cross-checking against the four observations flagged in the original
project notes as "what the results should mean" (now the acceptance
criteria for a healthy run):

1. **Historical 99% VaR (3.17%) > Gaussian 99% VaR (2.23%).** Confirmed —
   a **+0.94pp gap**, i.e. the normal distribution understates the 99%
   daily loss threshold by about 30% relative to the empirical one. This
   is the fat-tail effect from §3.2 of `docs/METHODOLOGY.md` made
   concrete. Note the sign flips at 95% (Gaussian 1.56% *above*
   historical 1.45%): the normal assumption is not uniformly optimistic,
   it is optimistic *in the tail* and slightly conservative in the body,
   which is exactly the shape a fat-tailed distribution produces.
2. **ES noticeably above VaR at the same confidence.** Confirmed — at
   99%, ES (4.30%) is **+1.14pp above** historical VaR (3.17%), a ~36%
   relative step up. The tail is a slope, not a cliff edge.
3. **JB test rejects normality decisively.** Confirmed — p ≈ 0 (reported
   as `0.00e+00` in double precision) with excess kurtosis **+9.28** and
   skewness **−0.51**, against the 0/0 a normal distribution would show.
   This is the direct justification for not trusting the Gaussian VaR
   number at 99%.
4. **Volatility is time-varying (clustering).** Confirmed — 21-day
   rolling vol ranges from **5.33%** (2022-05-23, calm regime) to
   **45.72%** (2018-11-01, stressed regime) over the sample, an ~8.6x
   range on the same asset. At the most extreme single point, EWMA vol
   hit **41.0%** on 2018-10-24 against a full-sample figure of **15.6%**
   — see the failure-mode discussion below.
5. **The Gaussian VaR fails its coverage backtest at 99%; the historical
   one does not.** Confirmed, and this is the strongest of the five: the
   Gaussian 99% VaR was exceeded on **60** of 2,519 days against **25.2**
   expected (Kupiec LR = 35.02, p ≈ 0 → rejected), while the historical
   99% VaR was exceeded 26 times against 25.2 expected (LR = 0.03,
   p = 0.87 → not rejected). At 95% neither is rejected, which locates
   the failure precisely where the theory says it should be: in the tail,
   not the body.

All five hold, which is what "the pipeline ran correctly on data with the
stylised facts it was designed to have" looks like. On real market data
you should expect the same qualitative findings, though the exact numbers
will differ by ticker and window.

**Reading the backtest honestly.** Both estimators are scored on the same
window they were fitted to, so the historical estimator's pass is close to
tautological — the empirical 1% quantile of a sample is exceeded by ~1% of
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
var_historical(returns, 0.95)       # 0.0176  (1.76%)
expected_shortfall(returns, 0.95)   # 0.02    (2.00%)
```

The Kupiec test is no help here either, and that is worth being explicit
about: with `n=5` the asymptotic chi-squared reference distribution is
meaningless, and even at `n=250` (a full year, the standard regulatory
window) a 99% VaR with **twice** the expected exception rate is not
rejected at 5% — pinned by
`tests/test_backtest.py::test_kupiec_power_is_low_on_a_short_window`.
Short samples do not just make the risk estimate noisy; they make the
test that is supposed to catch a bad risk estimate powerless.

Both return a number — nothing crashes — but with `n=5`, the "95th
percentile" is a linear interpolation between the two smallest
observations, and the ES tail average is at most 1-2 points. Neither
number is a serious risk estimate; they are included here (see
`tests/test_edge_cases.py::test_five_day_sample_produces_a_number_but_is_statistically_meaningless`)
specifically to document that **the functions do not protect you from
running them on too little data** — that judgment call is the caller's
responsibility. A production risk report should enforce a minimum sample
size (a common desk convention is >= 1 year, i.e. >= ~252 observations,
before quoting 99% VaR at all).

### 3.2 Exactly-constant returns

```python
import pandas as pd
from eq_risk_metrics import sharpe_ratio, var_cornish_fisher

returns = pd.Series([0.002] * 30)
sharpe_ratio(returns)          # 6.78e+16 -- huge, not +inf
var_cornish_fisher(returns, 0.95)  # RuntimeWarning, then NaN
```

Two distinct failure shapes on the same degenerate input:

- **Sharpe/Sortino:** `excess.std(ddof=1)` on 30 bit-identical floats
  does *not* come out to exactly `0.0` (subtracting a non-terminating
  daily risk-free rate introduces floating-point noise on the order of
  `1e-19`), so the ratio is a **huge finite number**, not `inf` or a
  clean error. A report that prints this without a sanity bound (e.g.
  "flag Sharpe ratios with `|value| > 100`") will show something
  nonsensical rather than fail loudly.
- **Cornish-Fisher VaR (and `normality_report`):** both call into
  `scipy.stats.skew`/`kurtosis`, which detect the exact zero-variance
  case as catastrophic cancellation and emit a `RuntimeWarning` +
  return `NaN`. This project's `pytest` configuration promotes
  `RuntimeWarning` to a hard test failure (`filterwarnings =
  ["error::RuntimeWarning"]`), so this behaviour is pinned down by
  `tests/test_edge_cases.py::test_constant_returns_cornish_fisher_matches_gaussian`
  and `tests/test_diagnostics.py::test_normality_report_on_exactly_constant_series_warns_and_returns_nan`
  rather than silently drifting.

Realistic near-constant data (any real price series with even
femtosecond-scale floating noise between "identical" days) does **not**
hit this path — `test_near_constant_returns_cornish_fisher_matches_gaussian`
confirms Cornish-Fisher cleanly collapses to Gaussian VaR once the input
isn't bit-for-bit identical. The exact-constant case is a synthetic
corner, not a realistic one, but it is worth knowing your code does this
before you hit it live (e.g. a data-vendor feed that repeats yesterday's
close on a holiday it thinks is a trading day).

### 3.3 All-positive return series (Sortino's empty downside array)

```python
import numpy as np, pandas as pd
from eq_risk_metrics import sortino_ratio

returns = pd.Series(np.linspace(0.001, 0.02, 20))
sortino_ratio(returns, rf_annual=0.0)   # nan
```

`downside = excess[excess < 0]` is empty, and `.std(ddof=1)` of an empty
pandas Series is `NaN` — quietly, with no warning (pandas' convention,
unlike scipy's). **`NaN` here means "no downside observations to measure
risk from", not "zero risk"** — a report that defaults missing Sortino to
zero, or drops it silently, would misleadingly suggest a riskless
position rather than "too little data in this direction to say". Sharpe
on the same series is well-defined and positive, which is itself a useful
signal: if Sharpe is fine but Sortino is `NaN`, the return stream has no
observed downside at all — worth a second look, not a data artefact to
paper over.

### 3.4 Regime divergence: EWMA vs full-sample volatility

On the bundled synthetic sample, the single largest full-sample-vs-EWMA
gap occurs at **2018-10-24**: EWMA volatility reads **41.0%** against a
full-sample figure of **15.6%** (21-day rolling vol agrees with EWMA at
**44.3%** on the same date) — a **2.6x** divergence.

```python
gap = (ewma_vol - full_sample_vol).abs()
gap.idxmax()   # Timestamp('2018-10-24')
```

This is the concrete trigger for the guidance in `docs/METHODOLOGY.md`
§9 and `docs/DESK_GUIDE.md`: **when EWMA/rolling vol diverges sharply
from the full-sample number, the unconditional figure is describing a
market that no longer exists**, and any risk number computed with
full-sample `sigma` (this includes `var_parametric` and
`var_cornish_fisher` as implemented here, which use the *whole-sample*
mean/std, not a current EWMA estimate) is stale exactly when it matters
most. A report generator should flag this divergence automatically
(e.g. `abs(ewma_vol - full_sample_vol) / full_sample_vol > 0.5`) rather
than let a reader notice it by eye.

### 3.5 Numerical limits and input validation

Previous revisions of this document listed the empty-series behaviour as a
known gap: `var_historical`/`expected_shortfall` raised a raw `IndexError`
out of `numpy.percentile`. **That gap is now closed** — all four VaR/ES
estimators share a `_validated` front door and raise a `ValueError` naming
the function and the expectation. The same front door added two checks
that were silently wrong before, both of which returned a plausible-looking
number instead of failing:

- **Out-of-range confidence.** `var_parametric(r, 1.5)` used to return
  `NaN` (`scipy.stats.norm.ppf` of an out-of-range probability), and
  `var_historical(r, 1.5)` a NumPy percentile error. Confidence must now be
  a finite number strictly inside `(0, 1)`; `0.5` remains legal (see
  below), `0.0` and `1.0` do not.
- **Non-finite returns.** A single `inf` or `NaN` anywhere in the sample
  propagated straight through both the percentile and the mean/std paths,
  producing a `NaN` VaR with only a `RuntimeWarning` to show for it. Such
  a value almost always means a missing price (`NaN`) or a zero price in a
  return denominator (`inf`), so it is now rejected with a message that
  says so.

The other validation added in the same pass, for the same reason —
silently-wrong beats loudly-wrong:

- `rolling_volatility(returns, window=1)` returned an **all-`NaN` series**,
  indistinguishable from "the window has not warmed up yet"; `window` must
  now be an integer `>= 2`. A window *larger* than the series is still
  legal and still returns all-`NaN` — that is a genuine warm-up state, not
  an error, and the returned series keeps the input index so downstream
  joins still align (`test_rolling_volatility_window_larger_than_sample_is_all_nan`).
- `ewma_volatility(returns, lam=0.0)` returned an identically-zero
  volatility (no memory at all, so the recursion never accumulates
  anything); `lam` must now be strictly inside `(0, 1)`.
- `max_drawdown` on an empty series raised pandas' cryptic "attempt to get
  argmin of an empty sequence", and on a series containing a zero or
  negative price happily reported a drawdown worse than −100%. Both are
  now explicit `ValueError`s — a non-positive price level is a data-feed
  bug, not a market event.

Remaining limits, deliberately left as they are:

- **Single-observation series:** `annualised_volatility` returns `NaN`
  (sample std with `ddof=1` is undefined for `n=1`) without warning, and so
  do `var_parametric`/`var_cornish_fisher`, which need a standard
  deviation. `NaN` here means "not enough data", and a report must not
  coerce it to `0` — that would read as "no risk". `var_historical` and
  `expected_shortfall` on a single point return that point itself at any
  confidence level (there is only one possible quantile), which is a
  defensible answer rather than an error. All four behaviours are pinned in
  `tests/test_edge_cases.py`.
- **Extreme confidence levels:** at `1 - 1e-9` the historical estimator
  *saturates* at the worst observed loss (it cannot report a loss the
  sample has never shown), while the Gaussian one extrapolates past it —
  the two are compared directly in
  `test_parametric_var_at_extreme_confidence_grows_without_saturating`.
  Neither number is trustworthy; they fail in opposite, instructive
  directions. At `1e-9` confidence, "VaR" is minus the largest *gain* in
  the sample — a negative loss. It stays finite and is not special-cased.
- **Confidence exactly 0.5:** `var_parametric` returns exactly `-mu`
  (`z = Phi^{-1}(0.5) = 0`), tested to abs 1e-12 — VaR at 50% confidence
  has lost its usual "tail risk" meaning (it is just minus the mean) but
  the function does not special-case this away.
- **Confidence near the boundary (0.9999):** returns a finite, larger
  number than 99% VaR, as expected of a monotone quantile function — no
  numerical blow-up, but at that confidence a 2,519-observation historical
  sample has essentially no real data at that quantile; the number is
  almost entirely an extrapolation of the two smallest observations. This
  is exactly where extreme value theory (`docs/METHODOLOGY.md` §3.3)
  would do materially better.
