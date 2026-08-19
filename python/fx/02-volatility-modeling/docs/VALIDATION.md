# Validation — FX Volatility Modeling & Forecasting

Documentation-contract items 3 (how validated) and 4 (where it fails). All
numbers below are produced by `python examples/run_pipeline.py` (seeded,
deterministic) or by the test suite (`python -m pytest tests -q`, 322 tests,
offline, ~35 s). Nothing is hand-typed from memory.

---

## 1. Analytic identities and unit-level checks

- **Recursion identities**: GARCH/GJR/EGARCH filters verified term by term
  against the defining equations at machine precision
  (`test_garch.py::TestFilter`, `test_gjr.py::TestFilter`,
  `test_egarch.py::TestFilter`).
- **EWMA identities**: recursion, closed-form geometric expansion
  `s2_t = λ^t s2_0 + (1−λ) Σ λ^{i-1} r²`, weight-sum `1 − λ^n`, and the flat
  multi-step forecast (persistence exactly 1) — all exact
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

## 2. Parameter recovery on 20k-observation simulations

From run_pipeline §4 (Hessian standard errors in parentheses); every
estimate is within two standard errors of truth, and the test suite asserts
|estimate − truth| < 4·SE per parameter:

| Model | Parameter | True | Estimated (SE) |
|---|---|---|---|
| GARCH (Gaussian) | ω | 1.0e−6 | 9.80e−7 (1.4e−7) |
| | α | 0.050 | 0.0487 (0.0039) |
| | β | 0.920 | 0.9218 (0.0071) |
| GARCH-t | ω | 1.5e−6 | 1.36e−6 (1.5e−7) |
| | α | 0.060 | 0.0603 (0.0044) |
| | β | 0.900 | 0.9025 (0.0073) |
| | ν | 6.0 | 6.43 (0.27) |
| GJR | ω | 1.0e−6 | 9.75e−7 (8.6e−8) |
| | α | 0.030 | 0.0399 (0.0048) |
| | γ | 0.100 | 0.0958 (0.0072) |
| | β | 0.880 | 0.8747 (0.0068) |
| EGARCH | ω | −0.500 | −0.526 (0.0013)* |
| | α | 0.150 | 0.1502 (0.010) |
| | γ | −0.060 | −0.049 (0.0054) |
| | β | 0.950 | 0.9474 (0.0057) |
| GARCH-X | γ_x | 5.0e−5 | 4.89e−5 (2.0e−6), t = 23.9 |

*EGARCH ω and β are strongly correlated (ω/(1−β) is the identified level);
the reported ω SE conditions on the near-flat direction — the level itself is
tightly pinned.

Additional recovery facts under test: Student-t ν runs to > 20 on Gaussian
data (correctly detecting thin tails); GJR γ estimates < 0.02 on symmetric
data; EGARCH recovers positive as well as negative leverage; day-of-week
factors recover an injected weekly pattern within 8%; Parkinson/Garman–Klass
recover true GBM vol within 7% (discrete-monitoring bias documented below)
and are strictly less dispersed than close-to-close across replications.

## 3. Cross-validation against the `arch` package

Same series, arch fed percent returns (its recommended scaling), our fitter
fed raw decimals — conventions reconciled explicitly (ω scales by 100²,
log-likelihood by n·log 100). On 5000-obs simulations
(`test_arch_crosscheck.py`):

- Gaussian GARCH(1,1): α, β agree within 1e−4 absolute (observed ~1e−6),
  ω within 0.1% relative, log-likelihood within 0.01 (observed ~6e−8),
  conditional-variance paths within 0.5% pointwise.
- GARCH-t: α, β within 5e−4, ν within 2% relative.
- GJR: α, γ, β within 1e−3, log-likelihood within 0.05.

Scale invariance of our own implementation (decimal vs percent input) is
additionally tested to 1e−6 on α/β with the exact log-likelihood mapping.

In-sample ranking on the two pipeline pairs (run_pipeline §3):

| Pair | Best (AIC) | ΔAIC GARCH-t | Δ AIC Gaussian GARCH | Sign-bias p |
|---|---|---|---|---|
| EURUSD-like (G10) | GARCH-t | 0.00 | +102.0 | 0.51 |
| USDMXN-like (EM) | EGARCH-t (γ=+0.046) | +19.5 | +920.1 | 0.63† |

†Sign bias on the *quoted* USDMXN direction is on positive returns; GJR-t
pins γ = 0 there but recovers γ = +0.123 (SE 0.040) on inverted MXNUSD —
the quote-direction effect described in METHODOLOGY §3.1.

## 4. Out-of-sample race (500 days, rolling 1-step, refit every 125 days)

QLIKE (Patton-robust; lower is better), Diebold–Mariano vs the GARCH-t
benchmark with from-scratch Newey–West variance (negative DM favours the row
model):

| Model | EURUSD-like QLIKE | DM (p) | USDMXN-like QLIKE | DM (p) |
|---|---|---|---|---|
| EWMA(0.94) | −9.5893 | +0.85 (0.39) | −8.5017 | +0.76 (0.45) |
| GARCH | −9.6075 | −0.10 (0.92) | −8.7357 | **+2.38 (0.017)** |
| GARCH-t | −9.6073 | benchmark | −8.7813 | benchmark |
| GJR | −9.6094 | −0.44 (0.66) | −8.7357 | **+2.38 (0.017)** |
| EGARCH | −9.6130 | −1.20 (0.23) | −8.6661 | +0.84 (0.40) |

Honest reading: on the G10-style pair *nothing* significantly beats
GARCH-t — exactly the "GARCH-t is sufficient for G10" claim; the asymmetric
models' point estimates are marginally better but far from significance. On
the EM-style pair the t-likelihood matters even for pure variance
forecasting (Gaussian GARCH and Gaussian GJR are significantly worse: the
Gaussian MLE lets jump days whip α around), while EGARCH's advantage is
in-sample (density fit) more than 1-step-QLIKE. With 500 days and an r²
proxy, only large gaps are detectable — assumption A8.

Statistical tooling is itself validated: DM size on equal-quality forecasts
is within [0.04, 0.18] at the 10% level over 300 replications; QLIKE's
minimum over candidate scalings sits at the true variance (Patton
robustness); Mincer–Zarnowitz on the *true* conditional variance returns
(a, b) = (0, 1) (slope within 0.1, joint Wald p > 0.01) and rejects a 2×
biased forecast at p < 1e−6; Newey–West recovers the analytic MA(1) long-run
variance within 5% (`test_evaluation.py`).

## 5. Failure modes and edge cases (contract items 4 & 6 — each unit-tested)

- **Pegged / managed currencies (HKD-style).** Daily vol of ~2 bp. Internal
  unit-variance rescaling means fits neither underflow nor blow up
  (`test_edge_cases.py::TestPeggedCurrency`); recovered unconditional
  variance has the right order of magnitude and 100-day forecasts stay below
  1% annualized. *Known behaviour*: persistence often pins near the IGARCH
  boundary — micro mean-reversion inside the band is not GARCH dynamics.
  Long-horizon forecasts for pegs are not meaningful; monitor the peg, not
  the GARCH. Hessian SEs may be NaN at the boundary (reported as NaN, never
  invented).
- **Depeg jumps (CHF 15-Jan-2015, −15% in a day).** Fit through the jump
  converges with finite likelihood and persistence < 1; conditional variance
  the day after the jump is > 20× the pre-jump level; forecasts from the
  jump-day state start > 10× unconditional and decay monotonically back
  (`test_edge_cases.py::TestDepegJump`). *Documented degeneracy*: if the
  sample **ends** on the jump day, α is unidentified (the jump never feeds a
  later observation) and the MLE legitimately drifts to a near-IGARCH
  constant-variance corner — still finite and positive, but the forecast
  will NOT spike. Desks refit the morning after with the jump interior to
  the sample.
- **EM structural breaks / jump regimes.** The EM-style series (t tails,
  ν ≈ 3.6, one-sided jumps) fits without failure; Gaussian fits show
  persistence pinned at ~1.0000 (run_pipeline §3) — treat a Gaussian
  boundary fit on EM data as a *misspecification signal*, not a finding.
- **Intervention regimes.** Unscheduled intervention (BoJ 2022-style) is an
  innovation outlier, not a calendar dummy — GARCH-X cannot absorb it
  (assumption A7); it appears as a jump and follows the depeg logic above.
  Scheduled-event variance is handled and recovered (γ_x within 3%).
- **Weekend gaps / seasonality.** Range estimators assume continuous
  monitoring: discrete intraday sampling biases them down (~7% tolerance at
  780 steps/day in tests, worse for coarser bars) and the Friday–Monday gap
  is invisible to them. Day-of-week factors quantify the Monday effect;
  unmodelled, it mildly fattens residual tails (A10).
- **Constant series** (broken feed, hard peg at fix precision): all fitters
  raise `ValueError("returns are constant...")` — degenerate likelihood,
  refuse rather than return garbage. The degeneracy test is **relative**
  (`std <= 1e-12 · max|r|`), not `std == 0`: a series pinned at a constant
  *non-zero* value has `std ≈ 1e-19` from floating-point cancellation, not
  exactly zero. Before this was tightened, `fit_garch(np.full(500, 4e-4))`
  returned `α = 0.617, β = 0.383` — pure optimiser noise on a flat
  likelihood — with a confident-looking log-likelihood of 3202. Now it
  raises. (`test_edge_cases_extra.py::TestConstantAndZeroInputs`.)
- **GARCH-X plumbing errors fail loudly.** Passing `x` without `gamma_x`
  previously produced an **all-NaN variance path silently**
  (`np.asarray(None, dtype=float)` → `nan`); both half-specified
  combinations, non-finite `gamma_x` and negative event dummies now raise
  with actionable messages
  (`test_edge_cases_extra.py::TestGarchXMisuse`).
- **Quote direction is part of the model.** On a simulated USD/EM series
  the asymmetry sits on *positive* pair returns (EM selling off): realised
  next-day |r| after up-moves 0.0051 vs 0.0044 after down-moves. GJR-t
  (which constrains γ ≥ 0 onto *negative* returns) pins γ at the boundary
  in the quoted direction and recovers γ ≈ 0.07–0.13 on the inverted pair,
  with a higher likelihood. EGARCH-t is exactly quote-direction agnostic:
  inversion flips γ's sign and leaves α, β, ω and the log-likelihood
  unchanged to 1e-5. On jumpy EM data this only shows up under the **t**
  likelihood — the Gaussian MLE chases the jumps and obscures it.
  (`test_edge_cases_extra.py::TestQuoteDirectionAsymmetry`.)
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
  whole program net negative — the convexity of the variance payoff working
  against the seller (`test_edge_cases_extra.py::TestVolPremiumInCrisis`).
- **Tiny/empty samples**: empty series, single observations and
  out-of-range forward-RV windows all raise; two observations is the
  documented floor for `close_to_close_vol` / `ewma_variance` and works.
- **NaN policy**: reject, never impute — any NaN/inf in returns, prices or
  regressors raises with an explicit message (tested across all fitters).
- **NaN *parameters*, not just NaN data** (`test_filter_param_guards.py`).
  The filters previously guarded their coefficients with inequalities only
  (`if omega <= 0 or alpha < 0 or beta < 0 or beta >= 1: raise`). Every
  comparison against NaN is False, so a NaN ω/α/β passed the guard and
  `garch_filter` returned an all-NaN variance path — which then became a NaN
  forecast and a NaN VaR with no exception anywhere. All three filters
  (`garch_filter`, `gjr_filter`, `egarch_filter`) now route their scalars
  through `fx_vol._mle.validate_filter_params` first, and reject a
  non-finite or non-positive `initial_variance` seed; `egarch_filter` also
  checks `abs_moment`. The same class of hole is closed on
  `ewma_variance(init=…)`, `cross_volatility(vol1, vol2, corr)`,
  `variance_swap_pnl(vega_notional=…)` and the `periods_per_year`
  annualisation factor of `close_to_close_vol`, `rolling_close_vol`,
  `parkinson_vol`, `garman_klass_vol` and `realized_vol_forward` (a negative
  factor previously produced `sqrt` of a negative variance → NaN vol).
- **Annualisation-convention identity**: `close_to_close_vol` at 260 vs 252
  periods per year differs by exactly √(260/252) = 1.01575 — asserted to
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
(reproducibility asserted in `test_synthetic.py`); the pipeline prints all
quoted numbers in ~30 s; the suite runs offline in ~35 s with deterministic
seeds throughout. `arch` appears only in `tests/test_arch_crosscheck.py`.
