# Validation: Equity Volatility Modeling & Forecasting

How the library was validated and where it fails. All data are synthetic
and seeded. Every number comes from one of three commands:

| command | produces |
|---|---|
| `python examples/validation_tables.py` | sections 1 and 2 |
| `python examples/run_pipeline.py` | section 4 |
| `python -m pytest tests -q` | every test named below |

---

## 1. Parameter recovery on synthetic data (known truth)

Maximum-likelihood fits on series simulated from each model. "Robust SE" is
the Bollerslev-Wooldridge sandwich; on correctly specified Gaussian data it
equals the classic SE up to sampling noise, as it should.

**GARCH(1,1), n = 20,000, seed 1:**

| param | true | estimate | classic SE | robust SE | (est - true)/robust SE |
|---|---|---|---|---|---|
| omega | 5e-06 | 6.244e-06 | 8.3e-07 | 8.3e-07 | +1.51 |
| alpha | 0.05 | 0.05505 | 0.0047 | 0.0046 | +1.09 |
| beta | 0.9 | 0.8813 | 0.012 | 0.012 | -1.61 |

**GJR-GARCH(1,1), n = 15,000, seed 2:**

| param | true | estimate | classic SE | robust SE | (est - true)/robust SE |
|---|---|---|---|---|---|
| omega | 5e-06 | 4.73e-06 | 4.4e-07 | 4.4e-07 | -0.61 |
| alpha | 0.03 | 0.03226 | 0.0053 | 0.0052 | +0.43 |
| gamma | 0.1 | 0.1051 | 0.0083 | 0.0082 | +0.62 |
| beta | 0.88 | 0.8787 | 0.0071 | 0.007 | -0.18 |

**EGARCH(1,1), n = 8,000, seed 3:**

| param | true | estimate | classic SE | robust SE | (est - true)/robust SE |
|---|---|---|---|---|---|
| omega | -0.4 | -0.4646 | 0.057 | 0.057 | -1.14 |
| alpha | 0.1 | 0.1148 | 0.012 | 0.011 | +1.33 |
| gamma | -0.08 | -0.08818 | 0.0077 | 0.0076 | -1.08 |
| beta | 0.96 | 0.9535 | 0.0057 | 0.0057 | -1.15 |

**GARCH-t, n = 20,000, seed 4:** `nu` true 8.0, estimate 8.78 (robust SE 0.48).

Every estimate is within 1.7 robust SEs of the truth. One seed per model
cannot show that the SEs are correctly sized. That is shown by the arch
cross-check in section 2 and by the sandwich tests in
`tests/test_inference.py`. Leverage sign recovery is tested separately: on
leveraged data the GJR `gamma` t-stat exceeds +3 and the EGARCH `gamma`
t-stat is below -3; on symmetric data both estimates are within 0.02 of zero.

## 2. Cross-check against the `arch` package

`arch` (K. Sheppard) is an optional test dependency used only as a benchmark
(`tests/test_arch_crosscheck.py`, skipped if `arch` is absent). `arch` is fed
percent returns, so `omega_pct = 1e4 omega_dec` (GARCH/GJR),
`omega_pct = omega_dec + (1 - beta) ln 1e4` (EGARCH), and log-likelihoods
differ by `n ln 100`.

n = 5,000 synthetic GARCH observations, seed 42:

| quantity | eq_vol | arch | abs difference |
|---|---|---|---|
| alpha | 0.03712506649 | 0.03712663595 | 1.6e-06 |
| beta | 0.8939872293 | 0.8939825277 | 4.7e-06 |
| omega (pct^2) | 0.06794494465 | 0.06794815631 | 3.2e-06 |
| log-likelihood (rescaled) | 15981.98145 | 15981.98145 | 1.6e-08 |
| GARCH: our LL at arch's parameters | | | 3.6e-12 |
| EGARCH: our LL at arch's parameters | | | 3.6e-12 |

At identical parameters the likelihoods agree to round-off for GARCH and for
EGARCH. EGARCH starts from the same one-step-from-backcast convention as
arch (pre-sample shock at its expectation). The parameter differences are
optimiser tolerance. The test file also checks GJR parameters (1e-3) and
GARCH/GJR multi-step forecasts at identical parameters (1e-10 relative).

Standard errors, like for like. Gaussian QMLE on Student-t(5) data,
n = 5,000, seed 7:

| param | classic (ours) | classic (arch) | robust (ours) | robust (arch) |
|---|---|---|---|---|
| alpha | 0.00721 | 0.00721 | 0.01133 | 0.01133 |
| beta | 0.01232 | 0.01232 | 0.01811 | 0.01811 |

On fat-tailed data the classic SE understates the robust one by about a
third. Use `robust_std_errors` for inference.

## 3. Forecast and test validation

* GARCH/GJR k-step forecasts decay geometrically to the unconditional
  variance at rate P (checked to 1e-10); the 1-step forecast equals the
  advanced recursion to 1e-12; EWMA forecasts are flat.
* EGARCH Monte Carlo forecasts are reproducible for a fixed seed;
  independent seeds agree within 2% at 40k paths; horizon 1 matches the
  closed form to 1e-12.
* QLIKE is minimised by the true conditional variance against a
  squared-return proxy (30k obs, beats 15-40% distortions).
* Mincer-Zarnowitz slope for the true-variance forecast is consistent with
  1; MZ R^2 is about 0.02 even for the true variance because the proxy noise
  (var(z^2) = 2) attenuates it. R^2 is not a model-quality measure here.
* Diebold-Mariano size. With i.i.d. loss differentials (n = 300, 1,000
  replications) the rejection rate at 5% is within [2%, 9%]. With an AR(1)
  loss differential (phi = 0.5, n = 500, 1,000 replications) the i.i.d.
  variance rejects about 28% of the time and the default Newey-West lag
  (`floor(4 (n/100)^(2/9))` = 5 at n = 500) about 9%
  (`test_inference.py::TestDieboldMarianoHAC`).

## 4. Out-of-sample forecast race (synthetic data)

`examples/run_pipeline.py`: true model GJR (omega 3e-6, alpha 0.04,
gamma 0.12, beta 0.87; persistence 0.97), 3,000 training days, 500
out-of-sample days, 1-step forecasts, refit every 25 days, QLIKE against
squared returns. DM uses Newey-West lag 5 with the HLN correction.

| model | QLIKE | MSE x1e8 | DM vs GARCH | p |
|---|---|---|---|---|
| GJR | -7.6027 | 17.03 | -2.75 | 0.006 |
| EGARCH | -7.5785 | 17.22 | -0.86 | 0.39 |
| GARCH | -7.5638 | 16.91 | n/a | n/a |
| EWMA(0.94) | -7.4996 | 17.78 | +2.91 | 0.004 |
| Rolling 21d hist | -7.4963 | 18.42 | +2.69 | 0.007 |

The oracle QLIKE (true conditional variance) is -7.6067. The fitted GJR
closes (-7.6027 + 7.5638) / (-7.6067 + 7.5638) = 90.7% of the gap between
GARCH and the oracle. The QLIKE ranking is GJR > EGARCH > GARCH > EWMA >
rolling historical. GJR vs GARCH and the two baselines vs GARCH are
significant at 1%. MSE ranks GARCH first on the same data: MSE is also
robust to proxy noise, but it has much lower power.

Mincer-Zarnowitz on the GJR forecast gives slope 0.81 (SE 0.11) and joint
p = 0.010. On 500 days of squared-return proxy this rejection is not
evidence against the model: it uses the i.i.d. OLS covariance on a
heteroskedastic, autocorrelated proxy. The EGARCH in-sample Ljung-Box and
ARCH-LM p-values are 0.045 and 0.048 on this seed.

In-sample, GJR beats GARCH by 32 AIC points. Fitted GJR news impact: a -2
sigma shock moves next-day vol to 18.2% vs 15.2% for +2 sigma.

## 5. Failure modes

**F1. Structural breaks.** Case study `simulate_crisis` (seed 140,
`test_edge_cases.py::TestCrisisRegimeJump`): true vol jumps from 15% to 75%
annualised for 60 days, then settles at 30%.
* EWMA(0.94) moves from 13.3% before the break to 35.0% ten days in and
  65.4% thirty days in. It lags by construction (half-life 11 days), so a
  1-day 99% VaR from it is breached repeatedly in the first week.
* GARCH fitted across the break shows persistence 0.983. The break looks
  like near-IGARCH persistence and drags the long-run vol above both true
  regimes.
* Thirty days in, a 10-day window reads 75.1% and a 250-day window 35.9%.

Mitigation: monitor persistence and standardised-residual outliers; after a
confirmed break, shorten the window or refit from the break date.

**F2. IGARCH boundary.** As `alpha + beta -> 1` the unconditional variance
explodes. Estimates at the transform ceiling (0.9999) mean the data reject
stationarity. Fits there set `extra["at_boundary"] = 1`; their standard
errors are not meaningful. Unconditional quantities raise `ValueError`;
forecasts still work (P = 1 gives linear growth).

**F3. Fat tails.** Gaussian QMLE variance parameters remain consistent
(Bollerslev-Wooldridge), and the sandwich SEs are the valid ones. Gaussian
tail quantiles understate risk: for t(8) innovations the 1% quantile is
about 1.10x the Gaussian one. Use `dist="t"` for VaR. Treat estimated
`nu < 5` as a data-quality flag.

**F4. Noisy proxy.** Squared returns are unbiased but very noisy. Only
Patton-robust losses (QLIKE, MSE) rank correctly under them, and power is
low. No realised-variance proxy is offered because the package has no
intraday data.

**F5. Short samples.** The alpha SE on 3,000 daily obs is about 0.013.
Weekly data over 5 years (~260 points) gives about 0.05, and persistence
cannot be told apart from 0.8 or 1.0. Fitters refuse fewer than 100
observations.

**F6. Optimiser failures.** The EGARCH likelihood has an explosive far
field that defeats L-BFGS-B line search from some starts, so the fitter
uses SLSQP with an L-BFGS-B polish. Any optimiser failure raises
`ConvergenceError` (or returns `converged=False` on request). The rolling
harness counts failed refits in `n_failed_refits` and reuses the previous
parameters.

**F7. EGARCH long-run level.** `exp(omega/(1-beta))` is the median-type
level, below `E[sigma^2]` by Jensen's inequality.
`unconditional_vol_annual` is the exact mean for Gaussian innovations
(Nelson 1991 product formula) and `inf` for Student-t, where the mean does
not exist. `median_vol_annual` is reported alongside.

## 6. Edge cases (each unit-tested)

| Case | Behaviour | Test |
|---|---|---|
| Constant series | fitters raise `ValueError` ("zero variance"); historical/EWMA return the degenerate answer | `TestDegenerateSeries` |
| Single -25% outlier day | fits converge; variance spikes >5x next day then mean-reverts | `TestOutliers` |
| Fewer than 100 obs | `ValueError` naming the minimum | `TestShortSeries` |
| NaN / inf anywhere | `ValueError`, never dropped | `TestNaNPolicy` |
| Optimiser failure | `ConvergenceError` or `converged=False` | `TestConvergenceFailureSurfaced` |
| `alpha+beta >= 1` in unconditional formulas | `ValueError` | `test_garch.py`, `test_gjr.py` |
| Fit at persistence cap | `extra["at_boundary"] = 1` | `test_inference.py::TestBoundaryFlag` |
| Vol regime jump | documented lag and adaptation | `TestCrisisRegimeJump` |
| Zero-return days in QLIKE proxy | finite loss | `test_qlike_handles_zero_proxy` |
| Look-ahead in the OOS harness | forecasts invariant to future data | `test_forecast_alignment_no_lookahead` |
| sigma near 0 (1e-6 annualised) | GARCH converges, every `sigma2 > 0` | `TestExtremeVolRegimes` |
| sigma 800% annualised | fits converge, no overflow | `TestExtremeVolRegimes` |
| EGARCH explosive parameters | +-60 log-variance clip keeps `exp()` finite | `test_egarch_log_clip_keeps_recursion_finite_on_explosive_params` |
| lambda at 1 or near 0 | flat path at the initial level, or the last squared return | `TestBoundaryValues` |
| lambda outside (0, 1], incl. NaN | `ValueError` | `test_lambda_outside_unit_interval_raises` |
| Horizon 1 / horizon <= 0 | h = 1 valid; h <= 0 raises | `TestBoundaryValues` |
| Rolling window equal to sample length | one estimate at the last index | `test_window_equal_to_sample_length_is_allowed` |
| Parameter near zero in the Hessian | step floor keeps the diagonal out of round-off | `test_inference.py::TestHessianStep` |

### 6.1 Structural invariants

| Invariant | Tolerance | Test |
|---|---|---|
| Scale equivariance: `r -> c r` scales variances by c^2 | 1e-12 relative | `TestScaleEquivariance` |
| GARCH alpha, beta, persistence scale-free; omega scales by c^2 | alpha/beta 1e-4 abs, omega 1e-3 rel | `test_garch_fit_dynamics_invariant_to_scale` |
| EGARCH: `omega -> omega + (1-beta) ln c^2` reproduces `c^2 sigma^2` | 1e-10 relative | `test_egarch_recursion_scale_shift_in_log_space` |
| EWMA reacts more to a shock for smaller lambda | strict ordering | `TestMonotonicity` |
| GARCH term structure monotone toward the long-run level | 1e-4 relative at h = 400 | `test_garch_term_structure_monotone_toward_unconditional` |
| `avg_vol_annual` bracketed by `forward_vol_annual` | 1e-12 | `test_term_structure_avg_vol_between_spot_and_forward_limits` |
| Rolling-vol noise decreases with window, tracks sigma/sqrt(2w) | factor 3 | `test_realized_vol_window_noise_decreases_with_window` |
| GJR news impact: minimum at z = 0, convex, left above right | exact | `TestNewsImpactGeometry` |
| EGARCH news impact: positive, asymmetric for gamma < 0 | 1e-12 | `TestNewsImpactGeometry` |
| QLIKE penalises under-prediction more | strict | `test_qlike_penalises_underprediction_more_than_overprediction` |
| Loss rankings invariant to data scale | strict | `test_qlike_scale_sensitive_mse_scale_sensitive_but_ranking_agrees` |
| GJR/EGARCH collapse toward GARCH on symmetric data | abs(gamma) < 0.06 | `test_asymmetric_models_reduce_to_symmetric_on_symmetric_data` |
