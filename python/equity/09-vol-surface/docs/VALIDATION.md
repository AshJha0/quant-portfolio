# Validation — Equity Volatility Surface & Heston

Contract items 3 and 4: **how the stack was validated** (analytic benchmarks,
cross-method consistency, convergence/bias studies, recovery tests) and
**where it fails** (with reproducible numbers). All numbers below are
produced by `examples/run_pipeline.py` and the scripts referenced in each
section; everything is offline and seeded. Suite: 141 tests, ~35 s.

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

A fourth route — the **variance-swap triangle** — closes independently: the
static log-contract replication strip priced *off the fitted surface* gives a
1y fair vol of **19.94 %**, vs the *analytic* Heston expected average variance
√(θ + (v0−θ)(1−e^{−κT})/κT) = **20.09 %** on the calibrated parameters
(strip truncation at [0.5F, 2F] plus wing extrapolation accounts for the
0.15 vp gap), vs 18.30 % for flat ATM vol — the smile convexity premium is
real and both smile-aware methods capture it.

## 3. Monte Carlo: convergence and bias (400k paths, ATM 1y call)

Fourier reference; err = MC − Fourier, (·) = err in standard errors.

**Mild set (Feller satisfied): v0=θ=0.04, κ=2, ρ=−0.5, ξ=0.3; ref 8.0961**

| steps/yr | Euler (full trunc.) | QE |
|---|---|---|
| 4 | +0.0803 (+4.3 SE) | +0.0324 (+1.8 SE) |
| 8 | +0.0457 (+2.5 SE) | +0.0168 (+0.9 SE) |
| 16 | +0.0285 (+1.5 SE) | −0.0008 (−0.0 SE) |
| 64 | +0.0124 (+0.7 SE) | −0.0026 (−0.1 SE) |

**Extreme set (Feller violated, ratio 0.08): v0=θ=0.04, κ=1, ρ=−0.9, ξ=1; ref 5.6493**

| steps/yr | Euler (full trunc.) | QE |
|---|---|---|
| 4 | +2.6251 (+173.8 SE) | +0.0130 (+1.7 SE) |
| 8 | +1.5409 (+135.4 SE) | −0.0084 (−1.1 SE) |
| 16 | +0.7713 (+83.0 SE) | +0.0040 (+0.5 SE) |
| 32 | +0.3383 (+40.6 SE) | +0.0083 (+1.1 SE) |
| 64 | +0.1612 (+20.2 SE) | +0.0090 (+1.2 SE) |

Reading: Euler's truncation bias is O(dt) and catastrophic under Feller
violation — nearly 50 % of the ATM price at 4 steps, still 20 SE at 64
steps — while QE is within MC noise at **8 steps** on the same set. The suite
asserts exactly this comparative pattern
(`test_qe_within_3se_at_coarse_steps_where_euler_is_biased`) plus Euler
agreement at fine steps on the mild set, seeded reproducibility, and the
martingale property E[S_T] = F within 3 SE.

## 4. Calibration recovery (known ground truth)

Chain generated from known parameters, priced by Fourier, inverted to vols,
calibrated back (3 starts). Recovery is **exact to ≥ 6 decimals** on clean
data (pipeline §5):

| param | true | calibrated | abs error |
|---|---|---|---|
| v0 | 0.035 | 0.035000 | < 1e-6 |
| κ | 1.8 | 1.800000 | < 1e-6 |
| θ | 0.045 | 0.045000 | < 1e-6 |
| ρ | −0.65 | −0.650000 | < 1e-6 |
| ξ | 0.45 | 0.450000 | < 1e-6 |

Overall RMSE 0.0000 vol points; Jacobian condition number **1.2e3** (reported,
asserted > 1e2 — the ridge is present even when recovery is exact); Feller
ratio 0.80 → warning emitted (asserted). Documented test tolerances encode
the identifiability ordering: v0 to 1e-3 and ρ to 0.02 (tight), θ to 10 %,
κ and ξ only to 35 % (ridge). With 0.3 vp of seeded quote noise the fit
degrades gracefully to ~0.3 vp RMSE and all parameters stay in-tolerance
(`test_noisy_data_still_converges`).

## 5. Failure modes (reproducible)

### 5.1 Short-dated skew underfit — Heston vs a non-Heston (SVI) surface

Calibrating Heston to the SVI-generated chain (`mode="svi"`, seed 7 — a
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

The signature — model too flat in the short-dated wings (large negative
residual on the put wing), near-perfect at 2 y — is the classic "equity
short-dated skew needs jumps" failure. A desk marking weeklies off pure
Heston would be off by ~3–5 vol points in the wings. Mitigation: Bates
(jumps), rough vol, or simply *not* using the calibrated model inside the
first pillar (see DESK_GUIDE.md).

### 5.2 Parameter ridge instability day-over-day

Recalibrating to the same true surface under fresh 0.3 vp noise draws
("three days", seeds 11/22/33; truth v0=0.035, κ=1.8, θ=0.045, ρ=−0.65,
ξ=0.45):

| day | v0 | κ | θ | ρ | ξ | RMSE (vp) | cond(J) |
|---|---|---|---|---|---|---|---|
| 1 | 0.0346 | 1.991 | 0.0447 | −0.644 | 0.476 | 0.27 | 1.1e3 |
| 2 | 0.0356 | 1.822 | 0.0449 | −0.663 | 0.432 | 0.35 | 9.7e2 |
| 3 | 0.0347 | 1.957 | 0.0447 | −0.648 | 0.475 | 0.29 | 1.0e3 |

κ swings ±10 % and ξ ±5 % while v0/θ/ρ move < 2 % — with the surface itself
statistically unchanged. Any exotic whose value loads on κ or ξ individually
(forward-skew products) inherits this P&L noise; the desk answer is parameter
smoothing and a model reserve, not a better optimiser.

### 5.3 Wing extrapolation risk

Beyond the quoted strike range, SVI wings are linear in total variance —
consistent with Lee bounds but *not* validated by quotes. The variance-strip
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
  1.5e-4 at ξ=1 with u_max=200) — hence the CF-envelope probe for the cutoff;
  the cross-method 1e-6 agreement test would catch any regression.
* T → 0: prices → intrinsic (tested at T=1e-4); implied vol of a T=0 quote
  is refused with a warning.
* ρ = ±1 and ξ = 0 boundaries: priced, simulated and differentiated without
  special-casing by the caller (tested end-to-end).
* Sub-intrinsic quotes, zero-time-value ITM quotes and numerically-zero OTM
  quotes: `nan` + warning, asserted to never return a number.
* **Short-end variance floor (bug found and fixed by this test pass).**
  `VolSurface.total_variance` floored the *total* variance at an absolute
  `1e-12`. That floor is harmless at normal maturities but destroys the
  T → 0 limit: at `T = 1e-12` a perfectly legitimate `w = 3e-13` was clamped
  up to `1e-12`, and `sqrt(w/T)` then reported an ATM implied vol of **1.0
  (100 vol points)** instead of the flat first-pillar 0.548. Overnight and
  same-day expiries sit exactly in this regime. The floor is now applied to
  the *variance*, `w >= 1e-12 · T`, which is a uniform 0.0001-vol-point floor
  on implied vol and leaves the T → 0 limit intact. Regression-tested from
  `T = 1e-14` upward (`test_short_end_vol_is_flat_as_T_goes_to_zero`), along
  with the linear-in-T scaling of `w` at the short end.

## 6. No-arbitrage and wing invariants (property tests)

`tests/test_arbitrage_and_wings.py` turns the documented no-arbitrage story
into executable invariants rather than spot checks:

| Invariant | Assertion | Test |
|---|---|---|
| Butterfly checker actually fires | a kinked (V-shaped) total-variance smile fits with `min g < 0`, `arb_free=False`, and the fitter emits the Durrleman warning | `test_butterfly_checker_fires_on_arbitrage_violating_quotes` |
| Flat smile is the lognormal reference | `g(k) ≡ 1` to 1e-12 for `b = 0` | `test_durrleman_g_is_one_for_a_flat_smile` |
| Density non-negativity in *price* space | `C(K)` reconstructed from the fitted surface is convex (`Δ²C ≥ −1e-10`) and strictly decreasing in K over 200 strikes on [0.5F, 1.8F] | `test_call_price_convex_in_strike_for_arb_free_slice` |
| Calendar checker fires | pillar pair with decreasing total variance ⇒ `is_free=False`, negative worst violation, warning raised | `test_calendar_checker_detects_decreasing_total_variance` |
| Calendar enforcement is real | after `enforce_calendar=True`, `w` is monotone in T on the grid **and** at query level across 120 maturities × 5 log-moneyness points, including both extrapolation regions | `test_calendar_enforcement_makes_total_variance_monotone_in_T` |
| Healthy surface is calendar-free everywhere | `w(k,T)` non-decreasing in T over 25 × 100 grid spanning `T ∈ [1e-6, 6]` | `test_arb_free_surface_total_variance_monotone_in_T_everywhere` |
| Long-end extrapolation cannot manufacture arbitrage | slope floored at 0, so `w(T) ≥ w(T_max)` even when the last two pillars decrease | `test_extrapolated_slope_never_decreases_total_variance` |
| SVI wings are asymptotically linear | `w'(k) → b(ρ ± 1)` to 1e-6 at `k = ±200`; `w''` decays exactly as `|k|^-3` (8× per doubling) | `test_svi_wings_are_asymptotically_linear_in_k` |
| Lee slope bound | `\|w'(k)\| ≤ 2` over `k ∈ [−50, 50]`, and `b(1+\|ρ\|) ≤ 2` for the fitted slice | `test_lee_slope_bound_respected_in_wings` |
| Wings stay finite | `w > 0` and finite at `k = ±500`; surface vols finite for strikes from 1 to 10,000 | `test_deep_wing_total_variance_stays_positive_and_finite`, `test_surface_wing_vols_finite_far_outside_quoted_strikes` |
| Deep-wing IV refuses to guess | vega < 1e-12 ⇒ `nan`, never a bracket endpoint | `test_deep_wing_implied_vol_returns_nan_not_garbage` |
| Equity skew sign | vols strictly decreasing in K over the put wing (`ρ < 0`) | `test_vol_monotone_decreasing_in_strike_on_the_downside_skew` |
| Degenerate pillar sets rejected | duplicate, out-of-order, zero, empty expiries; slice/expiry count mismatch; non-positive spot and strikes | `test_surface_rejects_degenerate_pillar_sets`, `test_surface_rejects_non_positive_strikes` |
| Single-pillar surface | flat vol at every T from 1e-6 to 30 | `test_single_pillar_surface_is_flat_in_T` |
| SVI parameter domain | negative `b`, `\|ρ\| ≥ 1`, `σ ≤ 0` and `a + bσ√(1−ρ²) ≤ 0` all raise | `test_svi_params_reject_negative_total_variance` |
