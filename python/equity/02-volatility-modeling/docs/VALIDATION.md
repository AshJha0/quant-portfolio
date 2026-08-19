# Validation — Equity Volatility Modeling & Forecasting

Contract items 3 and 4: **how the library was validated** and **where it
fails**. All numbers below are produced by the committed test suite
(`python -m pytest tests -q`, 185 tests, ~4 s, fully offline, seeded) and by
`examples/run_pipeline.py`; every quoted figure is reproducible.

---

## 1. Parameter recovery on simulated data (known truth)

Maximum-likelihood fits on series simulated from each model with known
parameters (seeds fixed in `tests/conftest.py`).

**GARCH(1,1), n = 20,000 (seed 1):**

| param | true | estimate | std err | |t−true|/SE |
|---|---|---|---|---|
| omega | 5.0e-6 | 6.24e-6 | 8.3e-7 | 1.5 |
| alpha | 0.05 | 0.0551 | 0.0047 | 1.1 |
| beta | 0.90 | 0.8813 | 0.0116 | 1.6 |

**GJR-GARCH(1,1), n = 15,000 (seed 2):**

| param | true | estimate | std err |
|---|---|---|---|
| omega | 5.0e-6 | 4.73e-6 | 4.4e-7 |
| alpha | 0.03 | 0.0323 | 0.0053 |
| gamma | 0.10 | 0.1051 | 0.0083 |
| beta | 0.88 | 0.8787 | 0.0071 |

**EGARCH(1,1), n = 8,000 (seed 3):**

| param | true | estimate | std err |
|---|---|---|---|
| omega | −0.40 | −0.465 | 0.057 |
| alpha | 0.10 | 0.1148 | 0.012 |
| gamma | −0.08 | −0.0882 | 0.0077 |
| beta | 0.96 | 0.9535 | 0.0057 |

**GARCH-t, n = 20,000:** `nu` true 8.0 → estimated 7.79; alpha/beta recovered
as above. Leverage **sign** recovery is tested separately: on leveraged data
the GJR `gamma` t-stat exceeds +3 and the EGARCH `gamma` t-stat is below −3;
on symmetric (plain GARCH) data both estimates are within ±0.02 of zero.
Every estimate above sits within ~1.6 standard errors of truth — consistent
with correct MLE *and* correctly sized standard errors.

## 2. Cross-validation against the `arch` package

All models are implemented from scratch; Kevin Sheppard's `arch` is used as
an independent benchmark only (`tests/test_arch_crosscheck.py`). Scaling
convention: `arch` is fed percent returns (×100), so `omega_pct = 1e4 ·
omega_dec` (GARCH/GJR), `omega_pct = omega_dec + (1−beta)·ln 1e4` (EGARCH,
log scale), and log-likelihoods differ by exactly `n·ln 100`.

On n = 5,000 simulated GARCH observations (seed 42):

| quantity | eq_vol | arch | difference |
|---|---|---|---|
| alpha | 0.0371250 | 0.0371266 | 1.6e-6 |
| beta | 0.8939866 | 0.8939825 | 4.1e-6 |
| omega (pct²) | 0.0679456 | 0.0679482 | 2.6e-6 |
| log-likelihood (rescaled) | 15981.98145309 | 15981.98145308 | 1.7e-8 |
| **our LL evaluated at arch's params** | — | — | **1.8e-12** |

The last row is the strongest statement: with arch's fitted parameters and
arch's backcast initialisation (which our recursion reproduces exactly), the
two implementations' likelihoods agree to machine precision — recursion and
density are line-for-line equivalent, and the tiny parameter differences are
purely optimiser tolerance. GJR agrees to ~1e-6 in parameters and 1e-9 in
LL; EGARCH to ~4e-5 in parameters (initialisation conventions differ
slightly, documented in the test).

## 3. Forecast validation

* **Analytic identities**: GARCH/GJR k-step forecasts decay geometrically to
  the unconditional variance at exactly rate P (checked to 1e-10), monotone
  from above and below; the 1-step forecast equals the advanced recursion to
  1e-12; EWMA term structure is exactly flat.
* **EGARCH Monte Carlo forecasts**: seeded-Generator reproducibility is
  bit-exact; independent seeds agree within 2% at 40k paths; horizon-1 is
  deterministic and matches the hand formula to 1e-12.
* **Statistical calibration** (`tests/test_evaluation.py`):
  * QLIKE is empirically minimised by the *true* conditional variance
    against a squared-return proxy (beats ±15–40% distortions and the flat
    unconditional forecast on 30k obs).
  * Mincer–Zarnowitz slope for the true-variance forecast: 0.94 ± 0.05
    (consistent with 1); a forecast biased low by 2x is detected with joint
    p < 1e-6. Note MZ **R² ≈ 0.02 even for the perfect forecast** — proxy
    noise (`var(z²) = 2`) attenuates R², which is why R² must never be used
    to judge a vol model.
  * Diebold–Mariano **size**: under an exactly-equal-accuracy null
    (two QLIKE-matched distorted forecasts), empirical rejection rate at the
    5% level is within [2%, 9%] over 1,000 Monte Carlo replications of
    n = 300 — the test is correctly sized with the Harvey correction.

## 4. Out-of-sample forecast race (headline result)

`examples/run_pipeline.py`: true model GJR with strong leverage
(omega 3e-6, alpha 0.04, gamma 0.12, beta 0.87; persistence 0.97), 3,000
training days, **500 out-of-sample days**, 1-step forecasts, refit every 25
days, QLIKE against squared returns:

| model | QLIKE | MSE ×1e8 | DM vs GARCH | p |
|---|---|---|---|---|
| **GJR** | **−7.6027** | 17.03 | **−2.91** | 0.004 |
| EGARCH | −7.5785 | 17.22 | −0.99 | 0.32 |
| GARCH | −7.5638 | 16.91 | — | — |
| EWMA(0.94) | −7.4996 | 17.78 | +3.83 | 0.0001 |
| Rolling 21d hist | −7.4963 | 18.42 | +3.25 | 0.001 |

Oracle QLIKE (the true conditional variance of the DGP): **−7.6067** — the
fitted GJR captures ~94% of the achievable QLIKE gap between GARCH and the
oracle. Ranking is exactly as theory predicts on asymmetric data:
**GJR > EGARCH > GARCH > EWMA > rolling historical**, with the asymmetric-vs-
symmetric and model-vs-EWMA gaps statistically significant. (MSE ranks far
more noisily than QLIKE on the same data — Patton's point in practice.)

In-sample on the same data: GJR beats GARCH by 32 AIC points; the news impact
curves quantify the asymmetry (fitted GJR: a −2σ shock moves next-day vol to
18.2% vs 15.2% for +2σ).

## 5. Failure modes (contract item 4)

**F1 — Structural breaks / regime jumps (most important).**
Case study (`simulate_crisis`, seed 140, tested in
`test_edge_cases.py::TestCrisisRegimeJump`): true vol jumps 15% → 75%
annualised for 60 days (COVID-March-2020 scale), then settles at 30%.
Observed behaviour:
* EWMA(0.94) repriced from 13.3% (day 740, pre-break) to 35.0% ten days into
  the crisis and 65.4% after thirty days — adaptation is fast but *lagged by
  construction* (half-life 11 days): during the first week, 1-day 99% VaR
  computed from it would have been breached repeatedly.
* GARCH fitted across the break shows **persistence 0.983** vs 0.95-ish on
  break-free data — the break masquerades as near-IGARCH persistence and
  drags the estimated long-run vol far above either true regime. Forecast
  term structures from such a fit mean-revert to a meaningless level.
* Short rolling windows adapt fastest at the cost of noise: 30 days into the
  crisis a 10-day window reads 75.1% (correct) while a 250-day window reads
  35.9% (less than half the truth).
Desk mitigation: monitor persistence estimates and standardised-residual
outliers; after a confirmed break, shorten the estimation window or refit
from the break date (see DESK_GUIDE.md).

**F2 — IGARCH boundary persistence.** As `alpha + beta → 1`, the
unconditional variance `omega/(1−alpha−beta)` explodes and half-life
diverges; estimates at the transform ceiling (0.9999) mean the data reject
stationarity (often symptom of F1). The library refuses to compute
unconditional quantities there (`ValueError`, tested) instead of returning
garbage; forecasting still works (the exact recursion handles P = 1 as linear
growth).

**F3 — Distribution misspecification (fat tails).** Gaussian QMLE variance
parameters remain consistent (our Gaussian fit on t(8) data recovers
alpha/beta correctly — tested), but Gaussian tail quantiles understate risk:
for t(8) innovations the true 1% quantile is ~1.10x the Gaussian one (~10%
VaR understatement, worse for smaller nu). Use `dist="t"`; `nu` itself is
recovered within 0.2 on 20k obs but has wide error bars on desk-sized
samples — treat estimated `nu < 5` as a data-quality flag.

**F4 — Noisy-proxy evaluation traps.** Squared returns are unbiased but very
noisy; only robust losses (QLIKE/MSE) rank correctly under them
(Patton 2011). MZ R² is structurally tiny even for perfect forecasts (see
§3). Both facts are unit-tested so nobody "fixes" them into bugs later.

**F5 — Low-frequency / short samples.** GARCH MLE needs hundreds of
observations to identify alpha (its SE on 3,000 daily obs is already ±0.013);
weekly data over 5 years is ~260 points — expect ±0.05 on alpha, and
persistence indistinguishable from both 0.8 and 1.0. Fitters refuse < 100
observations with an informative error (tested); treat 100–1,000 as
"estimates come with wide, honest error bars" (they are printed in every
summary).

**F6 — Numerical/optimisation failures.** The EGARCH likelihood has an
explosive far-field that defeats L-BFGS-B line search from some starting
points; the fitter therefore uses SLSQP with an L-BFGS-B polish (verified
8/8 convergence across seeds vs 1/8 for raw L-BFGS-B during development).
Any residual optimiser failure raises `ConvergenceError` (or returns
`converged=False` on request) — never a silent bad fit; this path is
explicitly tested by forcing a failure.

## 6. Edge cases (contract item 6 — each documented *and* unit-tested)

| Case | Behaviour | Test |
|---|---|---|
| Constant series (zero variance) | fitters raise `ValueError` ("zero variance"); historical/EWMA return the degenerate answer finitely | `TestDegenerateSeries` |
| Single −25% outlier day | fits converge; conditional variance spikes >5x next day then mean-reverts; params finite | `TestOutliers` |
| Series shorter than 100 obs | informative `ValueError` naming the minimum | `TestShortSeries` |
| NaN / inf anywhere | `ValueError` — never silently dropped (policy in `_utils.validate_returns`) | `TestNaNPolicy` |
| Optimiser failure | `ConvergenceError` raised or `converged=False` flagged | `TestConvergenceFailureSurfaced` |
| `alpha+beta ≥ 1` in unconditional formulas | `ValueError`, no negative variance returned | `test_garch.py`, `test_gjr.py` |
| Vol regime jump (crisis) | documented lag/adaptation behaviour above | `TestCrisisRegimeJump` |
| Zero-return days in QLIKE proxy | finite loss (functional form chosen for this) | `test_qlike_handles_zero_proxy` |
| Look-ahead in the OOS harness | forecasts invariant to future data | `test_forecast_alignment_no_lookahead` |
| **σ → 0** (1e-6 annualised vol, daily variance ≈ 4e-15) | GARCH converges, `omega > 0`, every `sigma2 > 0` — no underflow to zero variance; EWMA and realized vol stay positive and finite | `TestExtremeVolRegimes::test_near_zero_vol_series_fits_without_underflow` |
| **σ → ∞** (800% annualised, daily σ ≈ 0.5) | fits converge, no overflow/NaN in the recursion or the likelihood; standardised residuals still have unit variance | `TestExtremeVolRegimes::test_extreme_vol_series_fits_without_overflow` |
| EGARCH explosive parameters (ω=5, α=3, β=0.99) | the ±60 log-variance clip keeps `exp()` finite and positive instead of overflowing mid-optimisation | `test_egarch_log_clip_keeps_recursion_finite_on_explosive_params` |
| λ at the boundaries (λ=1, λ→0⁺) | λ=1 gives an exactly flat variance path at the initial level; λ→0⁺ reduces to the last squared return | `TestBoundaryValues` |
| λ outside (0, 1] — including NaN | `ValueError` naming `lambda` | `test_lambda_outside_unit_interval_raises` |
| Horizon = 1 / horizon ≤ 0 | h=1 is a valid boundary (one-row term structure); h ≤ 0 raises | `TestBoundaryValues` |
| Rolling window equal to the sample length | one finite estimate at the last index, NaN before it (no silent truncation) | `test_window_equal_to_sample_length_is_allowed` |

### 6.1 Structural (property-based) invariants

Beyond individual cases, the suite pins down the *shape* of the estimators —
these hold for any correct implementation, so they catch scale bugs and sign
errors that point checks miss:

| Invariant | Tolerance | Test |
|---|---|---|
| **Scale equivariance**: `r → c·r` ⇒ every variance × c², vol × c | 1e-12 relative (EWMA, realized vol, GARCH recursion) | `TestScaleEquivariance` |
| GARCH *dynamics* are scale-free: α, β, persistence unchanged; ω scales by c² | α/β to 1e-4 absolute, ω to 1e-3 relative | `test_garch_fit_dynamics_invariant_to_scale` |
| EGARCH scale enters log-variance additively: `ω → ω + (1−β)ln c²` reproduces `c²σ²` | 1e-10 relative | `test_egarch_recursion_scale_shift_in_log_space` |
| EWMA reacts more to a shock the smaller λ is; half-life increasing in λ | strict ordering over λ ∈ {0.80, 0.90, 0.94, 0.99} | `TestMonotonicity` |
| GARCH term structure moves **monotonically** to the long-run level and the gap to it never widens | converges to `ω/(1−α−β)` within 1e-4 relative at h=400 | `test_garch_term_structure_monotone_toward_unconditional` |
| `avg_vol_annual` is bracketed by the running min/max of `forward_vol_annual` | 1e-12 | `test_term_structure_avg_vol_between_spot_and_forward_limits` |
| Rolling-vol sampling noise decreases with window and tracks σ/√(2w) | within a factor 3 of theory | `test_realized_vol_window_noise_decreases_with_window` |
| GJR news-impact curve: minimum at z=0, convex, left branch above right (γ>0) | exact | `TestNewsImpactGeometry` |
| EGARCH news-impact curve: strictly positive, asymmetric for γ<0, exactly symmetric for γ=0 | 1e-12 | `TestNewsImpactGeometry` |
| QLIKE penalises under-prediction more than over-prediction of equal ratio | strict | `test_qlike_penalises_underprediction_more_than_overprediction` |
| Loss *rankings* are invariant to the data scale (Patton robustness) | strict at c = 1 and c = 100 | `test_qlike_scale_sensitive_mse_scale_sensitive_but_ranking_agrees` |
| On symmetric data GJR/EGARCH collapse toward GARCH (|γ| < 0.06) and the extra parameter never lowers the maximised likelihood | — | `test_asymmetric_models_reduce_to_symmetric_on_symmetric_data` |
