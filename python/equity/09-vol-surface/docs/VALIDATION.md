# Validation: Equity Volatility Surface & Heston

How the stack was validated (analytic benchmarks, cross-method
consistency, bias studies, recovery) and where it fails. All data is
synthetic. Numbers come from `examples/run_pipeline.py` (pipeline) and
`examples/validation_tables.py` (sections 3, 5.1, 5.2); both are offline
and seeded. Test count and runtime are in the README.

---

## 1. Analytic benchmarks

| Check | Result | Test |
|---|---|---|
| BS known value (S=K=100, T=1, r=q=0, σ=0.2) | 7.965567455405804, abs err < 1e-12 | `test_known_value_atm_no_carry` |
| Implied vol round trip σ→price→σ | ≤ 1e-8 across strike/expiry/vol grid, calls & puts | `test_implied_vol_round_trip_grid` |
| Heston φ(0) = 1 | < 1e-12 | `test_cf_at_zero_is_one` |
| Heston φ(−i) = forward (martingale) | rel err < 1e-10 | `test_cf_martingale_identity` |
| Heston → BS limit (ξ=0, v0=θ) | abs err ~1e-10 (tol 1e-8) | `test_degenerate_limit_matches_black_scholes` |
| ξ=0, v0≠θ → BS at integrated deterministic variance | < 1e-8 | `test_xi_small_time_dependent_variance...` |
| Put–call parity (Heston) | < 1e-8 | `test_put_call_parity` |
| Deep ITM → discounted intrinsic; deep OTM → 0 | < 1e-6 / < 1e-8 | `test_deep_itm/otm_limit` |
| SVI analytic w′, w″ vs numerical | < 1e-8 | `test_analytic_*_derivative` |
| Little-trap CF continuity (T=10, ξ=1, dense u-grid) | no branch jumps | `test_little_trap_cf_continuous_in_u` |

Property checks: price monotone in σ and in v0, monotone decreasing and
convex in strike, delta bounds, γ > 0, Durrleman g ≥ 0 for known-good SVI,
variance paths non-negative under both MC schemes.

## 2. Cross-method consistency

Three independent pricing routes agree pairwise to **< 1.5e-7** across
S=100, K ∈ {55…160}, T ∈ {1w…2y}, on five parameter sets including a
Feller-violating ξ=1/ρ=−0.9 set and the boundaries ρ=±1, ξ=0:

* Heston P₁/P₂ semi-analytic (adaptive quadrature),
* Carr–Madan damped direct integration (adaptive quadrature),
* damped Gauss–Legendre fast path (vectorised over strikes).

Tests assert ≤ 1e-6 (`test_p1p2_vs_damped_agree`, `test_gl_vs_damped_agree_wide_grid`).

A fourth route, the **variance-swap triangle**, closes independently: the
static log-contract replication strip priced *off the fitted surface* gives a
1y fair vol of **19.94 %**, vs the *analytic* Heston expected average variance
√(θ + (v0−θ)(1−e^{−κT})/κT) = **20.09 %** on the calibrated parameters
(strip truncation at [0.5F, 2F] plus wing extrapolation accounts for the
0.15 vol-point gap), against 18.29 % for flat ATM vol. Both smile-aware
methods carry the smile's convexity premium.

## 3. Monte Carlo: convergence and bias (400k paths, ATM 1y call, r = q = 0)

Command: `python examples/validation_tables.py mc`. Fourier reference;
err = MC − Fourier, in brackets in standard errors. QE includes Andersen's
martingale correction; no antithetics or control variate here, so the SE
is the plain-MC one.

**Mild set (Feller satisfied): v0=θ=0.04, κ=2, ρ=−0.5, ξ=0.3; ref 7.6666**

| steps/yr | Euler (full trunc.) | QE |
|---|---|---|
| 4 | +0.0824 (+4.5 SE) | +0.0166 (+0.9 SE) |
| 8 | +0.0460 (+2.5 SE) | +0.0110 (+0.6 SE) |
| 16 | +0.0282 (+1.6 SE) | −0.0025 (−0.1 SE) |
| 32 | +0.0164 (+0.9 SE) | −0.0053 (−0.3 SE) |
| 64 | +0.0117 (+0.6 SE) | −0.0031 (−0.2 SE) |

**Extreme set (Feller ratio 0.08): v0=θ=0.04, κ=1, ρ=−0.9, ξ=1; ref 4.9801**

| steps/yr | Euler (full trunc.) | QE |
|---|---|---|
| 4 | +2.7289 (+184.5 SE) | −0.0023 (−0.3 SE) |
| 8 | +1.5906 (+144.2 SE) | −0.0094 (−1.3 SE) |
| 16 | +0.7928 (+88.9 SE) | +0.0027 (+0.4 SE) |
| 32 | +0.3475 (+43.8 SE) | +0.0071 (+1.0 SE) |
| 64 | +0.1656 (+21.9 SE) | +0.0089 (+1.2 SE) |

Euler's truncation bias is O(dt) and large under Feller violation: more
than half the ATM price at 4 steps and still 22 SE at 64 steps. QE is
within Monte Carlo noise from 4 steps on the same set. The suite asserts
this pattern (`test_qe_within_3se_at_coarse_steps_where_euler_is_biased`),
Euler agreement at fine steps on the mild set, seeded reproducibility,
and E[S_T] = F within 3 SE. The martingale correction is pinned by
`test_qe_martingale_correction_removes_forward_bias` (xi = 1.5, rho = −0.9,
T = 5, 2 and 5 steps: uncorrected forward off by more than 5 SE,
corrected within 4 SE). Antithetics plus the control variate cut the ATM
standard error to 0.65 of plain MC at the same path count
(`test_variance_reduction_keeps_price_and_cuts_stderr`).

The pipeline's section 8 QE rows (seed 42) all sit 1.0 to 1.8 SE below
Fourier. They share one seed, so they move together. Over seeds 0 to 9
the ATM 6m QE z-score (32 steps, antithetic and control variate)
averages +0.40 with a standard error of 0.24, so there is no detectable
bias.

## 4. Calibration recovery (known ground truth)

Chain generated from known parameters and priced with the adaptive P1/P2
quadrature, inverted to vols, and calibrated back with the Gauss-Legendre
pricer (3 starts). The two pricers are independent routes through the
characteristic function, so a pricer bug cannot cancel between generation
and calibration (`test_heston_chain_is_priced_independently_of_the_calibrator`).
Recovery is to 6 decimals on clean data (pipeline section 5):

| param | true | calibrated | abs error |
|---|---|---|---|
| v0 | 0.035 | 0.035000 | < 1e-6 |
| κ | 1.8 | 1.800000 | < 1e-6 |
| θ | 0.045 | 0.045000 | < 1e-6 |
| ρ | −0.65 | −0.650000 | < 1e-6 |
| ξ | 0.45 | 0.450000 | < 1e-6 |

Overall RMSE 0.0000 vol points; Jacobian condition number 1.2e3 (reported,
asserted > 1e2: the ridge is present even when recovery is exact); Feller
ratio 0.80 → warning emitted (asserted). Documented test tolerances encode
the identifiability ordering: v0 to 1e-3 and ρ to 0.02 (tight), θ to 10 %,
κ and ξ only to 35 % (ridge). With 0.3 vp of seeded quote noise the fit
degrades gracefully to ~0.3 vp RMSE and all parameters stay in-tolerance
(`test_noisy_data_still_converges`).

## 5. Failure modes (reproducible)

### 5.1 Short-dated skew underfit: Heston vs a non-Heston (SVI) surface

Calibrating Heston to the SVI-generated chain (`mode="svi"`, seed 7, a
surface with realistic 1/√T skew decay that *no* diffusive Heston can match)
leaves a structured residual pattern concentrated at the short end:

| expiry | RMSE (vp) | min resid | ATM resid | max resid |
|---|---|---|---|---|
| 1 w | 2.79 | −4.95 | +0.76 | +2.45 |
| 1 m | 1.69 | −3.48 | +0.57 | +1.42 |
| 3 m | 1.09 | −2.14 | +0.20 | +1.54 |
| 6 m | 0.84 | −1.22 | −0.06 | +1.67 |
| 1 y | 0.33 | −0.44 | −0.15 | +0.83 |
| 2 y | 0.12 | −0.22 | +0.13 | +0.17 |

The signature, model too flat in the short-dated wings (large negative
residual on the put wing), near-perfect at 2 y, is the classic "equity
short-dated skew needs jumps" failure. A desk marking weeklies off pure
Heston would be off by ~3–5 vol points in the wings. Mitigation: Bates
(jumps), rough vol, or simply *not* using the calibrated model inside the
first pillar (see DESK_GUIDE.md).

### 5.2 Parameter ridge instability day-over-day

Command: `python examples/validation_tables.py ridge`. Recalibrating to the
same true surface under fresh N(0, 0.3 vol point) noise ("three days",
seeds 11/22/33; truth v0=0.035, κ=1.8, θ=0.045, ρ=−0.65, ξ=0.45):

| day | v0 | κ | θ | ρ | ξ | RMSE (vp) | cond(J) |
|---|---|---|---|---|---|---|---|
| 1 | 0.0348 | 1.751 | 0.0458 | −0.675 | 0.429 | 0.29 | 1.2e3 |
| 2 | 0.0345 | 1.785 | 0.0466 | −0.666 | 0.465 | 0.35 | 1.2e3 |
| 3 | 0.0355 | 1.696 | 0.0455 | −0.646 | 0.449 | 0.30 | 1.1e3 |

With the surface statistically unchanged, κ ranges over 1.70 to 1.79 and
ξ over 0.43 to 0.46, while v0 stays within 1.5 % of truth. Any exotic
whose value loads on κ or ξ individually (forward-skew products) inherits
this noise. The desk answer is parameter smoothing and a model reserve.

### 5.3 Wing extrapolation risk

Beyond the quoted strike range, SVI wings are linear in total variance -
inside Lee's bound by construction of the fit but not validated by quotes. The variance-strip
test quantifies the exposure: truncating the replication strip at [0.5F, 2F]
vs the analytic value leaves 0.15 vp on a 20-vol underlying, i.e. wing
assumptions move variance-swap-style marks by tenths of a vol point.
Deep-wing quotes are refused (`nan`), so the wings are pure model.
`test_query_outside_range` behaviour is pinned: short-end flat vol, long-end
slope-floored linear total variance.

### 5.4 Calendar issues around discrete events

The calendar check/enforcement operates on smooth ACT/365F total variance.
Earnings and discrete dividends concentrate variance at points, so a
perfectly legitimate pre/post-earnings pillar pair can *look* like calendar
arbitrage in smooth time (or worse, the running-max "fix" flattens real
event variance). Enforcement is therefore **opt-in** (`enforce_calendar=True`
warns before adjusting), and the planted-violation tests confirm detection
rather than silent repair. Event-time handling is out of scope and flagged in
DESK_GUIDE.md.

### 5.5 Numerical limits

* Fixed-cutoff Fourier integration loses tail mass at high ξ (measured
  1.5e-4 at ξ=1 with u_max=200), hence the CF-envelope probe for the cutoff;
  the cross-method 1e-6 agreement test would catch any regression.
* T → 0: prices → intrinsic (tested at T=1e-4); implied vol of a T=0 quote
  is refused with a warning.
* ρ = ±1 and ξ = 0 boundaries: priced, simulated and differentiated without
  special-casing by the caller (tested end-to-end).
* Sub-intrinsic quotes, zero-time-value ITM quotes and numerically-zero OTM
  quotes: `nan` + warning, asserted to never return a number.
* Short-end variance floor: `VolSurface.total_variance` floors the
  variance, `w >= 1e-12 · T`, a uniform 0.0001-vol-point floor on implied
  vol that leaves the T → 0 limit intact. An absolute floor on total
  variance would report an ATM vol of 1.0 at T = 1e-12 instead of the flat
  first-pillar vol. Tested from `T = 1e-14` upward
  (`test_short_end_vol_is_flat_as_T_goes_to_zero`).
* Fourier prices more negative than 1e-7·max(S, K) (ten times the
  quadrature tolerance) return NaN with a RuntimeWarning instead of being
  floored at zero (`test_negative_price_beyond_roundoff_is_nan_not_zero`).

## 6. No-arbitrage and wing invariants (property tests)

`tests/test_arbitrage_and_wings.py`, `tests/test_svi_constrained.py` and
`tests/test_surface.py` turn the no-arbitrage rules into executable
invariants:

| Invariant | Assertion | Test |
|---|---|---|
| Butterfly checker fires | the unconstrained fit of a V-shaped smile has `min g < 0`, `arb_free=False`, and warns | `test_butterfly_checker_fires_on_arbitrage_violating_quotes` |
| Constrained fit is butterfly-free | same V-shaped quotes: constrained fit has `g >= -1e-10` on a 4,600-point grid, the butterfly constraint is reported active | `test_v_shaped_quotes_fit_arbitrage_free` |
| Constrained fit respects Lee | quotes from an SVI with wing slope 3: fitted `b(1+\|ρ\|) <= 2` | `test_steep_wings_respect_lee_bound` |
| Property: arbitrage-free SVI recovered | hypothesis over (a, b, ρ, m, σ): fit RMSE < 1e-4 vol, Lee bound holds, g >= 0 | `test_fit_recovers_arbitrage_free_svi` |
| Butterfly re-check on the surface | g at pillars and midpoints; Vogt's slice flagged at its pillar and at an interpolated expiry; after calendar enforcement the PCHIP surface is C1 in k and re-checked | `test_butterfly_report_*`, `test_enforced_surface_is_c1_in_k_and_rechecked` |
| Flat smile is the lognormal reference | `g(k) ≡ 1` to 1e-12 for `b = 0` | `test_durrleman_g_is_one_for_a_flat_smile` |
| Density non-negativity in *price* space | `C(K)` reconstructed from the fitted surface is convex (`Δ²C ≥ −1e-10`) and strictly decreasing in K over 200 strikes on [0.5F, 1.8F] | `test_call_price_convex_in_strike_for_arb_free_slice` |
| Calendar checker fires | pillar pair with decreasing total variance ⇒ `is_free=False`, negative worst violation, warning raised | `test_calendar_checker_detects_decreasing_total_variance` |
| Calendar enforcement is real | after `enforce_calendar=True`, `w` is monotone in T on the grid **and** at query level across 120 maturities × 5 log-moneyness points, including both extrapolation regions | `test_calendar_enforcement_makes_total_variance_monotone_in_T` |
| Healthy surface is calendar-free everywhere | `w(k,T)` non-decreasing in T over 25 × 100 grid spanning `T ∈ [1e-6, 6]` | `test_arb_free_surface_total_variance_monotone_in_T_everywhere` |
| Long-end extrapolation cannot manufacture arbitrage | slope floored at 0, so `w(T) ≥ w(T_max)` even when the last two pillars decrease | `test_extrapolated_slope_never_decreases_total_variance` |
| SVI wings are asymptotically linear | `w'(k) → b(ρ ± 1)` to 1e-6 at `k = ±200`; `w''` decays exactly as `|k|^-3` (8× per doubling) | `test_svi_wings_are_asymptotically_linear_in_k` |
| Lee slope bound (hand-built slice) | `\|w'(k)\| ≤ 2` over `k ∈ [−50, 50]` | `test_lee_slope_bound_respected_in_wings` |
| Wings stay finite | `w > 0` and finite at `k = ±500`; surface vols finite for strikes from 1 to 10,000 | `test_deep_wing_total_variance_stays_positive_and_finite`, `test_surface_wing_vols_finite_far_outside_quoted_strikes` |
| Deep-wing IV refuses to guess | vega < 1e-12 ⇒ `nan`, never a bracket endpoint | `test_deep_wing_implied_vol_returns_nan_not_garbage` |
| Equity skew sign | vols strictly decreasing in K over the put wing (`ρ < 0`) | `test_vol_monotone_decreasing_in_strike_on_the_downside_skew` |
| Degenerate pillar sets rejected | duplicate, out-of-order, zero, empty expiries; slice/expiry count mismatch; non-positive spot and strikes | `test_surface_rejects_degenerate_pillar_sets`, `test_surface_rejects_non_positive_strikes` |
| Single-pillar surface | flat vol at every T from 1e-6 to 30 | `test_single_pillar_surface_is_flat_in_T` |
| SVI parameter domain | negative `b`, `\|ρ\| ≥ 1`, `σ ≤ 0` and `a + bσ√(1−ρ²) ≤ 0` all raise | `test_svi_params_reject_negative_total_variance` |

## 7. The November 2026 release oracle tests (`tests/test_round3_volsurface.py`)

| Model | Oracle | Tolerance |
|---|---|---|
| SSVI | slice equals raw SVI under the Gatheral-Jacquier mapping | 1e-15 |
| SSVI | Theorem 4.2 conditions hold, so Durrleman g >= 0 on [-4, 4] | 1e-12 |
| SSVI/eSSVI | fit recovers known parameters from noiseless data | 1e-5 / 1e-3 |
| Dupire | flat vol returns sigma; term-structure-only surface returns dw/dT | 1e-6 rel |
| Dupire | the route via w and the route via prices agree on an SSVI surface | 2e-4 rel |
| Dupire | local-vol MC reprices the input surface (40k paths, 100 steps) | 4 SE + 0.02 |
| Bates | lam = 0 gives Heston | 1e-7 |
| Bates | xi = 0, v0 = theta gives the Merton (1976) series | 1e-8 |
| Bates | put-call parity; K -> 0 gives the discounted forward | 1e-12 / 1e-7 rel |
| Bates | Euler MC of the dynamics (100k paths, 200 steps) | 4 SE + 0.03 |
