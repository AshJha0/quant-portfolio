# Validation: FX Volatility Modeling & Forecasting

Documentation-contract items 3 (how validated) and 4 (where it fails). All
numbers below are produced on synthetic data by
`python examples/run_pipeline.py` (seeded) or by the test suite
(`python -m pytest tests -q`, offline).

---

## 1. Analytic identities and unit-level checks

- **Recursion identities**: GARCH/GJR/EGARCH filters verified term by term
  against the defining equations at machine precision
  (`test_garch.py::TestFilter`, `test_gjr.py::TestFilter`,
  `test_egarch.py::TestFilter`).
- **EWMA identities**: recursion, closed-form geometric expansion
  `s2_t = λ^t s2_0 + (1−λ) Σ λ^{i-1} r²`, weight-sum `1 − λ^n`, and the flat
  multi-step forecast (persistence exactly 1): all exact
  (`test_ewma.py`).
- **Inversion invariance** (FX-specific): inverting BASE/QUOTE negates log
  returns and leaves every vol estimator *and the full GARCH fit* invariant
  (likelihood depends on r² only); EGARCH leverage flips sign exactly under
  inversion, α and β unchanged (`test_returns.py`, `test_egarch.py`).
- **Vol triangle**: `σ_x² = σ_1² + σ_2² + 2 c_1 c_2 ρ σ_1 σ_2` is an exact
  in-sample identity when fed sample moments (verified to 1e−10 relative) and
  recovers the true cross vol on 40k-observation correlated simulations
  within 3%, for positive, negative and zero correlation and for both sign
  products (`test_returns.py::TestCrossVolatility`). Pipeline demo: EURJPY
  triangle 11.57% vs direct 11.57% (identical), legs 8.66%/10.29%, ρ = −0.26.
- **Forecast identities**: 1-step forecast equals the filter update; multi-step
  equals the geometric-decay closed form; convergence to unconditional
  variance at h = 800 within 1e−6 relative (`test_forecasting.py`).
- **Student-t absolute moment** (EGARCH centering): analytic formula vs
  2M-draw Monte Carlo within 0.5%, Gaussian limit recovered at ν = 1e8.

## 2. Parameter recovery on 20k-observation synthetic simulations

From `python examples/run_pipeline.py` section 4. Classic and robust
(Bollerslev-Wooldridge) standard errors; the test suite asserts
|estimate - truth| < 4 SE per parameter.

| Model | Parameter | True | Estimate | classic SE | robust SE |
|---|---|---|---|---|---|
| GARCH (Gaussian) | ω | 1.0e-6 | 9.80e-7 | 1.4e-7 | 1.5e-7 |
| | α | 0.050 | 0.0487 | 0.0039 | 0.0042 |
| | β | 0.920 | 0.9218 | 0.0071 | 0.008 |
| GARCH-t | ω | 1.5e-6 | 1.36e-6 | 1.5e-7 | 1.5e-7 |
| | α | 0.060 | 0.0603 | 0.0044 | 0.0043 |
| | β | 0.900 | 0.9025 | 0.0073 | 0.007 |
| | ν | 6.0 | 6.43 | 0.27 | 0.28 |
| GJR | ω | 1.0e-6 | 9.75e-7 | 8.6e-8 | 8.5e-8 |
| | α | 0.030 | 0.0399 | 0.0048 | 0.0047 |
| | γ | 0.100 | 0.0958 | 0.0072 | 0.0073 |
| | β | 0.880 | 0.8747 | 0.0068 | 0.0067 |
| EGARCH | ω | -0.500 | -0.526 | 0.057 | 0.058 |
| | α | 0.150 | 0.1502 | 0.010 | 0.0099 |
| | γ | -0.060 | -0.049 | 0.0054 | 0.0054 |
| | β | 0.950 | 0.9474 | 0.0057 | 0.0058 |
| GARCH-X | γ_x | 5.0e-5 | 4.89e-5 | 2.0e-6 | n/a |

The EGARCH ω standard error is mapped from the rescaled fit by the delta
method, because `ω_r = ω_y + (1-β) ln s²` depends on β
(`test_inference.py::TestRobustStandardErrors::test_egarch_omega_se_is_unit_consistent`,
`test_arch_crosscheck.py::TestEgarchCrosscheck::test_omega_se_matches_arch_in_percent_units`).

Additional recovery facts under test: Student-t ν runs to > 20 on Gaussian
data; GJR γ estimates < 0.02 on symmetric data and about -0.10 on data whose
asymmetry sits on positive returns; EGARCH recovers positive as well as
negative leverage; day-of-week factors recover an injected weekly pattern
within 8%; Parkinson/Garman-Klass recover true GBM vol within 7%.

## 3. Cross-check against the `arch` package

`arch` is an optional test dependency (`pip install .[test]`);
`tests/test_arch_crosscheck.py` is skipped without it. arch is fed percent
returns, our fitter raw decimals: ω scales by 100² (GARCH/GJR) or shifts by
`(1-β) ln 1e4` (EGARCH), the log-likelihood by `n log 100`. On 5,000-obs
synthetic series:

| check | tolerance in the test |
|---|---|
| Gaussian GARCH α, β | 1e-4 absolute |
| Gaussian GARCH ω | 0.1% relative |
| Gaussian GARCH log-likelihood | 0.01 |
| Conditional-variance path | 0.5% pointwise |
| Classic SEs vs `cov_type="classic"`, robust SEs vs `cov_type="robust"`, Gaussian QMLE on t(6) data | 2% relative |
| GARCH-t α, β; ν | 5e-4 absolute; 2% relative |
| GJR α, γ, β; log-likelihood | 1e-3 absolute; 0.05 |
| GJR 10-step forecast at identical parameters | 1e-10 relative |
| EGARCH parameters | 5e-3 absolute |
| EGARCH log-likelihood at the arch parameters | 1e-6 absolute |
| EGARCH ω and β SEs on percent data | 2% relative |

In-sample ranking on the two pipeline pairs (run_pipeline section 3):

| Pair | Best (AIC) | dAIC GARCH-t | dAIC Gaussian GARCH | Sign-bias p |
|---|---|---|---|---|
| EURUSD-like (G10) | GARCH-t | 0.00 | +102.0 | 0.51 |
| USDMXN-like (EM) | EGARCH-t (γ = +0.046) | +19.6 | +920.1 | 0.63 |

On the quoted USDMXN direction the asymmetry sits on positive returns. GJR-t
estimates γ = -0.123 (SE 0.040) there and γ = +0.123 on inverted MXNUSD,
with the same log-likelihood (METHODOLOGY section 3). The Gaussian GARCH on
USDMXN pins persistence at 1.0000 and is flagged `at_boundary`.

## 4. Out-of-sample race (synthetic data, 500 days, rolling 1-step, refit every 125 days)

QLIKE (lower is better). Diebold-Mariano vs the GARCH-t benchmark with
Newey-West lag 5 (`floor(4 (500/100)^(2/9))`) and the HLN correction;
negative DM favours the row model.

| Model | EURUSD-like QLIKE | DM (p) | USDMXN-like QLIKE | DM (p) |
|---|---|---|---|---|
| EWMA(0.94) | -9.5893 | +0.92 (0.36) | -8.5017 | +0.76 (0.45) |
| GARCH | -9.6075 | -0.09 (0.93) | -8.7357 | +2.42 (0.016) |
| GARCH-t | -9.6073 | benchmark | -8.7813 | benchmark |
| GJR | -9.6094 | -0.45 (0.65) | -8.7048 | +1.01 (0.31) |
| EGARCH | -9.6129 | -1.19 (0.23) | -8.6665 | +0.78 (0.43) |

On USDMXN, 3 of the 4 Gaussian GARCH refits and 3 of the 4 Gaussian GJR
refits are at a parameter boundary; none failed to converge.

Reading: on the G10-style pair nothing significantly beats GARCH-t. On the
EM-style pair the t-likelihood matters for variance forecasting: Gaussian
GARCH is significantly worse (the Gaussian MLE lets jump days move α).
With 500 days and an r² proxy only large gaps are detectable (A8).

Statistical tooling is itself tested: DM size on equal-quality forecasts at
the 10% level is within [7.5%, 12.5%] over 2,000 replications (measured
10.0%); with an AR(1) loss differential (φ = 0.5) the i.i.d. variance
rejects about 28% at 5% and the default HAC lag about 9%; QLIKE's minimum
over candidate scalings sits at the true variance; Mincer-Zarnowitz on the
true conditional variance returns slope within 0.1 of 1 and rejects a 2x
biased forecast at p < 1e-6; Newey-West recovers the analytic MA(1)
long-run variance within 5% (`test_evaluation.py`, `test_inference.py`).

## 5. Failure modes and edge cases (contract items 4 & 6: each unit-tested)

- **Pegged / managed currencies (HKD-style).** Daily vol of ~2 bp. Internal
  unit-variance rescaling means fits neither underflow nor blow up
  (`test_edge_cases.py::TestPeggedCurrency`); recovered unconditional
  variance has the right order of magnitude and 100-day forecasts stay below
  1% annualized. *Known behaviour*: persistence often pins near the IGARCH
  boundary: micro mean-reversion inside the band is not GARCH dynamics.
  Long-horizon forecasts for pegs are not meaningful; monitor the peg, not
  the GARCH. Such fits set `extra["at_boundary"]`, and their SEs may be NaN.
- **Depeg jumps (CHF 15-Jan-2015, −15% in a day).** Fit through the jump
  converges with finite likelihood and persistence < 1; conditional variance
  the day after the jump is > 20× the pre-jump level; forecasts from the
  jump-day state start > 10× unconditional and decay monotonically back
  (`test_edge_cases.py::TestDepegJump`). *Documented degeneracy*: if the
  sample **ends** on the jump day, α is unidentified (the jump never feeds a
  later observation) and the MLE legitimately drifts to a near-IGARCH
  constant-variance corner: still finite and positive, but the forecast
  will NOT spike. Desks refit the morning after with the jump interior to
  the sample.
- **EM structural breaks / jump regimes.** The EM-style series (t tails,
  ν ≈ 3.6, one-sided jumps) fits without failure; Gaussian fits show
  persistence pinned at ~1.0000 (run_pipeline §3): treat a Gaussian
  boundary fit on EM data as a *misspecification signal*, not a finding.
- **Intervention regimes.** Unscheduled intervention (BoJ 2022-style) is an
  innovation outlier, not a calendar dummy: GARCH-X cannot absorb it
  (assumption A7); it appears as a jump and follows the depeg logic above.
  Scheduled-event variance is handled and recovered (γ_x within 3%).
- **Weekend gaps / seasonality.** Range estimators assume continuous
  monitoring: discrete intraday sampling biases them down (~7% tolerance at
  780 steps/day in tests, worse for coarser bars) and the Friday–Monday gap
  is invisible to them. Day-of-week factors quantify the Monday effect;
  unmodelled, it mildly fattens residual tails (A10).
- **Constant series** (broken feed, hard peg at fix precision): all fitters
  raise `ValueError("returns are constant...")`: degenerate likelihood,
  refuse rather than return garbage. The degeneracy test is **relative**
  (`std <= 1e-12 · max|r|`), not `std == 0`: a series pinned at a constant
  *non-zero* value has `std ≈ 1e-19` from floating-point cancellation, not
  exactly zero (`test_edge_cases_extra.py::TestConstantAndZeroInputs`).
- **GARCH-X misuse raises.** Passing `x` without `gamma_x` (or the
  reverse), non-finite `gamma_x` and negative event dummies raise with
  actionable messages
  (`test_edge_cases_extra.py::TestGarchXMisuse`).
- **Quote direction is part of the model.** On a simulated USD/EM series
  the asymmetry sits on positive pair returns (EM selling off): realised
  next-day |r| after up-moves 0.0051 vs 0.0044 after down-moves. GJR-t
  estimates a negative γ in the quoted direction; inverting the pair flips
  the sign, maps α to α + γ and leaves the log-likelihood unchanged to 1e-3.
  EGARCH-t inversion flips γ's sign and leaves α, β, ω and the
  log-likelihood unchanged to 1e-5. On jumpy EM data the effect shows up
  under the t likelihood; the Gaussian MLE chases the jumps
  (`test_edge_cases_extra.py::TestQuoteDirectionAsymmetry`).
- **Managed-band regimes (CNY-style, 1.5 bp daily vol)**: all three fitters
  converge with positive finite variance paths; 500-day forecasts stay
  under 2% annualised; and estimates are scale-invariant across four orders
  of magnitude (α, β to 1e-4; ω by the exact 1e8 factor; log-likelihood by
  the exact `−n·log(1e4)` change-of-variables constant).
- **Vol-triangle degeneracies**: ρ = −1 with equal leg vols cancels exactly
  to 0 (the negative-variance guard returns 0, never NaN); ρ = +1 adds
  linearly; the sign product flips addition to subtraction; a zero leg vol
  returns the other leg.
- **VRP through a depeg**: a short-variance program marked at peg-regime
  implied vol (5%) is profitable day to day but the 20-day window spanning
  a −15% gap alone produces a P&L worse than −0.5 vega units and turns the
  whole program net negative: the convexity of the variance payoff working
  against the seller (`test_edge_cases_extra.py::TestVolPremiumInCrisis`).
- **Tiny/empty samples**: empty series, single observations and
  out-of-range forward-RV windows all raise; two observations is the
  documented floor for `close_to_close_vol` / `ewma_variance` and works.
- **NaN policy**: reject, never impute: any NaN/inf in returns, prices or
  regressors raises with an explicit message (tested across all fitters).
- **NaN parameters, not just NaN data** (`test_filter_param_guards.py`).
  `garch_filter`, `gjr_filter` and `egarch_filter` pass their scalars through
  `fx_vol._mle.validate_filter_params` and reject a non-finite or
  non-positive `initial_variance`; `egarch_filter` also checks
  `abs_moment`. The same guard covers `ewma_variance(init=...)`,
  `cross_volatility`, `variance_swap_pnl(vega_notional=...)` and the
  `periods_per_year` factor of the realised-vol functions.
- **Convergence.** A non-converged fit raises `ConvergenceError` by default.
  The rolling harness keeps the previous parameters after a failed refit,
  counts it in `failed_refits`, raises if the first fit fails, and counts
  boundary fits in `boundary_refits` (`test_inference.py::TestConvergence`).
- **Annualisation-convention identity**: `close_to_close_vol` at 260 vs 252
  periods per year differs by exactly √(260/252) = 1.01575: asserted to
  1e-12, so a desk comparing realized vol to an implied quote on the other
  convention knows the 1.6%-of-vol offset is convention, not signal.
- **Short series**: fewer than 100 observations raises; the floor is a
  keyword (`min_obs`) so a desk can consciously override.
- **Numerical limits**: transformed parameters are clipped at ±30 (expit/exp
  saturation); EGARCH log-variance clipped at ±60 to survive absurd
  parameter proposals during optimization; QLIKE requires strictly positive
  forecasts (raises otherwise); DM refuses identical loss series (zero
  long-run variance) with an informative error.

## 6. Reproducibility

Every simulator takes an explicit seed / `numpy.random.Generator`
(reproducibility asserted in `test_synthetic.py`). The pipeline prints all
quoted numbers; the suite runs offline with fixed seeds. `arch` appears only
in `tests/test_arch_crosscheck.py`.
