# Validation — Single-Asset Equity Risk Metrics

This document answers documentation-contract items 3 and 4: **how this was
validated** (analytic checks, what "good" output looks like on the bundled
data) and **where it fails** (known failure modes with reproducible
examples). Every number below is produced by the committed code —
`python examples/run_pipeline.py` regenerates the report;
`pytest -q` (82 tests, offline, seeded) enforces the analytic checks
permanently.

Bundled synthetic data: `eq_risk_metrics.data.synthetic.generate(seed=2,
n_days=2520)` — a two-regime (calm/stressed) model with Student-t shocks,
~10 years of trading days ending 2026-08-14. It is **not real market
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

## 2. What "good" looks like on the bundled synthetic data

Running `python examples/run_pipeline.py` on the bundled sample
(seed=2, 2,519 daily returns, 2016-12-19 to 2026-08-14) produces:

```
--- Volatility ---
Annualised volatility (full sample) : 13.90%
Latest 21-day rolling volatility    : 11.54%
Latest EWMA (lambda=0.94) volatility: 12.71%

--- Value at Risk (1-day, % of position value) ---
  95% confidence:
    Historical VaR      : 1.30%
    Gaussian VaR        : 1.41%
    Cornish-Fisher VaR  : 1.36%
    Expected Shortfall  : 1.97%
  99% confidence:
    Historical VaR      : 2.38%
    Gaussian VaR        : 2.01%
    Cornish-Fisher VaR  : 3.12%
    Expected Shortfall  : 3.27%

--- Drawdown ---
Maximum drawdown : -27.11%
Peak date        : 2025-06-25
Trough date      : 2026-07-20

--- Risk-adjusted performance (rf = 3% annual) ---
Sharpe ratio  : 0.33
Sortino ratio : 0.47

--- Distribution diagnostics ---
Skewness         : -0.149
Excess kurtosis  : +5.022
Jarque-Bera stat : 2656.0 (p = 0.00e+00)
Normality rejected at 5%: True
```

Cross-checking against the four observations flagged in the original
project notes as "what the results should mean" (now the acceptance
criteria for a healthy run):

1. **Historical 99% VaR (2.38%) > Gaussian 99% VaR (2.01%).** Confirmed —
   a **+0.37pp gap**, i.e. the normal distribution understates the 99%
   daily loss threshold by about 18% relative to the empirical one. This
   is the fat-tail effect from §3.2 of `docs/METHODOLOGY.md` made
   concrete.
2. **ES noticeably above VaR at the same confidence.** Confirmed — at
   99%, ES (3.27%) is **+0.89pp above** historical VaR (2.38%), a ~37%
   relative step up. The tail is a slope, not a cliff edge.
3. **JB test rejects normality decisively.** Confirmed — p ≈ 0 (reported
   as `0.00e+00` in double precision) with excess kurtosis **+5.02**,
   far above the 0 a normal distribution would show. This is the
   direct justification for not trusting the Gaussian VaR number at 99%.
4. **Volatility is time-varying (clustering).** Confirmed — 21-day
   rolling vol ranges from **5.19%** (2021-12-30, calm regime) to
   **35.29%** (2019-04-12, stressed regime) over the sample, a ~7x
   range on the same asset. At the most extreme single point, EWMA vol
   hit **33.4%** on 2022-03-03 against a full-sample figure of **13.9%**
   — see the failure-mode discussion below.

All four hold, which is what "the pipeline ran correctly on data with the
stylised facts it was designed to have" looks like. On real market data
you should expect the same four qualitative findings, though the exact
numbers will differ by ticker and window.

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
gap occurs at **2022-03-03**: EWMA volatility reads **33.4%** against a
full-sample figure of **13.9%** (21-day rolling vol agrees with EWMA at
**33.0%** on the same date) — a **2.4x** divergence.

```python
gap = (ewma_vol - full_sample_vol).abs()
gap.idxmax()   # Timestamp('2022-03-03')
```

This is the concrete trigger for the guidance in `docs/METHODOLOGY.md`
§8 and `docs/DESK_GUIDE.md`: **when EWMA/rolling vol diverges sharply
from the full-sample number, the unconditional figure is describing a
market that no longer exists**, and any risk number computed with
full-sample `sigma` (this includes `var_parametric` and
`var_cornish_fisher` as implemented here, which use the *whole-sample*
mean/std, not a current EWMA estimate) is stale exactly when it matters
most. A report generator should flag this divergence automatically
(e.g. `abs(ewma_vol - full_sample_vol) / full_sample_vol > 0.5`) rather
than let a reader notice it by eye.

### 3.5 Numerical limits

- **Empty series:** `var_historical`/`expected_shortfall` raise
  `IndexError` from the underlying `numpy.percentile` call on an empty
  array — not a `ValueError` with a friendly message. This is
  intentionally left as-is (a genuine gap versus the polished-`ValueError`
  contract used elsewhere in this portfolio) and is pinned by
  `tests/test_edge_cases.py::test_var_historical_on_empty_series_raises`
  so a future refactor changing this behaviour is a deliberate choice,
  not an accident.
- **Single-observation series:** `annualised_volatility` returns `NaN`
  (sample std with `ddof=1` is undefined for `n=1`) without warning;
  `var_historical` on a single point returns that point itself at any
  confidence level (there is only one possible quantile).
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
