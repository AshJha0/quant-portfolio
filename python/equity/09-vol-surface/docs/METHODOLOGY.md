# Methodology — Equity Volatility Surface & Heston Stochastic Volatility

This document answers contract items 1 and 2: **why these models** (against
named alternatives, with trade-offs) and **what assumptions are made** (with
what breaks when each is violated).

---

## 1. The pipeline and the maths

```
option chain ──► implied vols ──► SVI smiles ──► total-variance surface
                                      │                  │
                                      ▼                  ▼
                             butterfly check      calendar check
                                      └──────┬───────────┘
                                             ▼
                                   Heston calibration
                                             │
                        ┌────────────────────┼────────────────────┐
                        ▼                    ▼                    ▼
                 Fourier pricing      Monte Carlo (QE)         Greeks
```

### Implied volatility

Black–Scholes inversion with Brent bracketing on σ ∈ [1e-6, 5] plus Newton
polish. Robustness rules (each unit-tested): sub-intrinsic or above-bound
prices, numerically zero prices, and deep-ITM quotes with no measurable time
value all return `nan` **with a warning** — never a garbage number. A wing
quote whose vega has underflowed contains no volatility information; inverting
it anyway would silently poison every downstream fit.

### Smiles: raw SVI

Total implied variance per expiry, `w(k) = σ²(k)·T`, `k = ln(K/F)`:

    w(k) = a + b·( ρ(k−m) + √((k−m)² + σ²) )

Butterfly (strike) arbitrage is checked via the Durrleman condition — the
risk-neutral density is non-negative iff

    g(k) = (1 − k·w′/2w)² − (w′²/4)(1/w + 1/4) + w″/2 ≥ 0  for all k,

with the SVI derivatives computed **analytically**:
`w′ = b(ρ + (k−m)/√((k−m)²+σ²))`, `w″ = bσ²/((k−m)²+σ²)^{3/2}`
(both validated against numerical differentiation to 1e-8).

### Surface: interpolate total variance, never vol

At fixed forward moneyness, absence of calendar arbitrage ⇔ total variance
non-decreasing in T. Linear interpolation of a monotone quantity stays
monotone, so calendar-free pillars give a calendar-free surface, and the
implied forward variance between pillars is piecewise-constant and
non-negative. Linear interpolation **of vol** has no such property: a
high-vol short pillar next to a low-vol long pillar can interpolate to
decreasing total variance — negative forward variance — between two
individually arbitrage-free slices. Extrapolation policy (documented in
`surface.py`, unit-tested): flat vol below the first pillar
(w ∝ T), linear-in-T continuation of the last slope beyond the last pillar
with the slope floored at zero.

### Heston

    dS = (r−q)S dt + √v S dW₁,   dv = κ(θ−v) dt + ξ√v dW₂,   ⟨dW₁,dW₂⟩ = ρ dt

Characteristic function in the **Gatheral / "little Heston trap"** form
(Albrecher et al. 2007): the original Heston formulation contains
`ln((1−g₁e^{dT})/(1−g₁))` with a growing exponential whose argument winds
across the negative real axis, so the principal-branch complex log jumps by
2πi — discontinuous prices for long maturities/large u. Rewriting with the
conjugate root g₂ = (b−d)/(b+d) makes the exponential decaying
(`e^{−dT} → 0`), the log argument never circles the origin, and the principal
branch is continuous everywhere relevant. Tested by scanning φ(u) on a dense
grid at T=10, ξ=1 (no jumps).

Pricing goes through **two independent Fourier routes**, cross-validated to
1e-6 in the suite: (i) Heston's semi-analytic P₁/P₂ probabilities under
adaptive quadrature; (ii) Carr–Madan-style damped direct integration
(α = 1.5), plus a Gauss–Legendre vectorised version of (ii) as the fast path
for calibration. The integration cutoff is chosen by probing the CF envelope,
because the Heston CF decays only like `exp(−u(v₀+κθT)√(1−ρ²)/ξ)` — slowly at
high vol-of-vol, where a fixed cutoff silently truncates tail mass (this was
measurable: a fixed u_max = 200 lost 1.5e-4 of price at ξ = 1).

**Feller condition** 2κθ ≥ ξ²: checked, **warned, never raised**. Equity
index calibrations routinely violate it — short-dated skew demands high ξ
relative to κθ — and the model remains perfectly well-defined (variance
touches zero and reflects; the CF is still valid). The fitted surface in this
project has Feller ratio 0.80. What violation *does* affect is simulation
scheme choice (see below) and the credibility of far-forward smile dynamics.

### Monte Carlo

Full-truncation Euler (uses v⁺ in drift and diffusion) and Andersen's QE
scheme (moment-matched squared-Gaussian / exponential mixture for the CIR
step, correlation-preserving log-spot update with central weights). Euler
carries O(dt) truncation bias that explodes when Feller is violated — at 8
steps/year on the ξ=1 set the bias is ~135 standard errors of a 400k-path
run — while QE stays within MC noise at the same step count (numbers in
VALIDATION.md).

### Calibration

Weighted least squares in implied-vol space with vega weights (approximates
price-space error while keeping expiries comparable), `scipy.least_squares`
(TRF, box bounds), multi-start with deterministic seeds, Jacobian condition
number reported at the optimum. See §3 on identifiability.

---

## 2. Why these models — alternatives and trade-offs

### Why SVI for smiles (vs polynomial / kernel fits)

| Criterion | Raw SVI | Polynomial in k or delta | Spline / kernel |
|---|---|---|---|
| Parameters per expiry | 5, each interpretable | 3–6, uninterpretable | many |
| Wings | Linear in total variance — consistent with Lee's moment bounds | Explode (odd powers) or collapse | Uncontrolled beyond data |
| No-arbitrage machinery | Durrleman g(k) analytic; known sufficient conditions (Gatheral–Jacquier) | None | None |
| Fit quality on equity smiles | Near-exact (0.005–0.04 vp here) | 0.5–0.75 vp here | Exact in-sample, wild out |
| Extrapolation | Meaningful by construction | Meaningless | Meaningless |

The quadratic-in-delta baseline is implemented (`fit_quadratic_delta`)
precisely to make this concrete: on the same slices it fits 10–100× worse
(0.50–0.75 vol points vs 0.0006–0.04 for SVI — pipeline §3 table) and offers
no density-positivity diagnostics.

### Why Heston (vs local vol vs SABR)

| Criterion | Heston | Local vol (Dupire) | SABR (per expiry) |
|---|---|---|---|
| Reprices today's vanillas | Approximately (2–3 vp short-dated residual) | **Exactly** by construction | Per-slice only, no term structure |
| Forward smile dynamics | Realistic: smile moves with spot, vol clusters, mean-reverts | Known-wrong: forward smiles flatten (Hagan critique) | None across expiries |
| Path-dependent / forward-start exotics | Credible | Systematically mispriced | Not applicable |
| Semi-analytic vanillas | Yes (Fourier) | PDE/MC only | Yes (Hagan expansion, biased in wings) |
| Parameters | 5 global, economically meaningful | A full surface (over-parameterised) | 4 per expiry |
| Calibration risk | κ/ξ ridge (documented, measured) | Interpolation artefacts differentiate badly | Slice-to-slice inconsistency |

The trade-off is explicit: **local vol buys exact repricing of today's
vanillas at the cost of wrong smile dynamics; Heston buys credible dynamics
at the cost of imperfect repricing** (the short-dated skew residual is
measured and documented in VALIDATION.md). For pricing forward-skew-sensitive
exotics (cliquets, forward-starts, autocalls) dynamics matter more than an
exact static fit, which is why the desk-standard compromise is stochastic
(-local) vol. SABR is the market standard for *quoting* single expiries
(rates especially) but has no arbitrage-free term-structure story, which this
project needs.

### Why Fourier + quadrature (vs FFT grid, vs tree/PDE)

Carr–Madan FFT prices a whole log-strike *grid* at once but forces uniform
grid spacing tied to the damping/aliasing trade-off; we need arbitrary
strikes for calibration, and one CF evaluation per quadrature node serves a
whole strike column anyway (the CF is strike-independent). Adaptive
quadrature gives rigorous error control for the reference pricers; fixed
Gauss–Legendre gives ~100× speed for calibration and is validated against the
reference to 1e-6.

---

## 3. Parameter identifiability — the κ/ξ ridge

Vanilla smiles inform Heston parameters only through smile level, skew,
curvature and their decay in T. `v0` (short ATM level) and `ρ` (skew sign and
size) map almost one-to-one onto observables and calibrate tightly. `κ` and
`ξ` are identified *jointly* through the rate at which skew/curvature decay
with maturity: raising mean reversion and raising vol-of-vol nearly cancel in
vanilla prices, leaving a curved, flat-bottomed valley in the objective. The
suite asserts the measured symptom — Jacobian condition number ≳ 1e3 at the
optimum (1.2e3 on the clean fit) — and VALIDATION.md shows the day-over-day
consequence: under 0.3 vp of quote noise, recalibrated κ and ξ wander by
±5 % while v0, θ, ρ move by < 2 %. Desk handling (parameter smoothing,
reserves) is in DESK_GUIDE.md.

---

## 4. Assumptions register

Each assumption states what breaks if violated. Items marked (T) have a
dedicated unit test; edge cases are in `tests/test_edge_cases.py`.

1. **Frictionless continuous hedging** (no transaction costs, continuous
   rebalancing). *If violated*: implied vol ≠ replication cost; the
   difference is hedging slippage, handled on a desk as a reserve, not in
   the model.
2. **No jumps in spot or variance.** *If violated*: short-dated smile
   curvature and skew are underfit — diffusive Heston cannot generate enough
   skew at 1 week (skew ∝ √T dies too fast). This is not hypothetical: the
   calibration-to-SVI failure mode in VALIDATION.md shows residuals of 2.8 vp
   at 1 w decaying to 0.1 vp at 2 y. Realistic short-dated equity marking
   needs jumps (Bates) or rough vol. (T: residual pattern reproduced.)
3. **Single volatility factor.** *If violated*: cannot decorrelate short-
   and long-dated vol moves; term-structure twists (e.g. inversion in a
   crash) force parameter instability instead of factor reallocation.
4. **Deterministic rates and proportional continuous dividend yield q,
   ACT/365F.** *If violated*: equity-rate hybrid effects and discrete
   dividend jumps are mispriced; near ex-div dates the forward (and hence
   moneyness) is wrong — see DESK_GUIDE.md on dividend risk. (T: q > 0
   handled throughout; zero and high q tested.)
5. **European exercise.** *If violated*: American early-exercise premium
   (puts especially, high r or q) makes listed-quote inversion biased;
   real marking uses de-Americanisation first.
6. **Quotes are simultaneous, mid = value.** *If violated*: stale or wide
   quotes distort the fit; vega weighting and (optional) bid/ask noise in the
   generator quantify sensitivity. (T: noisy calibration converges.)
7. **CF branch handling: little-trap form is continuous for the parameter
   ranges used** (κ > 0, ξ ≥ 0, |ρ| ≤ 1, T ≤ ~10y). *If violated* (exotic
   parameter corners): would show as integrand discontinuities; the
   continuity test and dual-method cross-check would catch it. (T)
8. **Feller may be violated by design** — variance touches zero. *What
   breaks*: naive Euler simulation (bias measured at +135 SE), NOT the
   pricing. Full truncation/QE handle it. (T)
9. **Surface extrapolation policy is a modelling choice** (flat vol short
   end, slope-floored linear w long end, SVI wings in k). *If violated by
   reality* (e.g. earnings before the first pillar): short-end marks are
   wrong — see DESK_GUIDE.md earnings scenario. (T: policy pinned by tests.)
10. **Deep wings carry no invertible information** below double-precision
    time value; they are dropped (`nan`) rather than fitted. *If instead
    inverted*: garbage vols in the wings would corrupt SVI wings and hence
    density tails. (T)
11. **No-arbitrage is diagnosed, not silently repaired.** Butterfly
    (Durrleman `g ≥ 0`) and calendar (`∂w/∂T ≥ 0`) violations raise warnings
    and are exposed on the result objects; the running-max calendar fix is
    opt-in. *If violated* — i.e. if a desk consumed the surface without
    reading the diagnostics — a negative risk-neutral density produces
    negative butterfly prices and nonsensical variance-swap and digital
    marks. The checkers are themselves tested against planted violations, so
    a silent regression in the checker would fail the suite rather than pass
    unnoticed. (T)
12. **The surface's vol floor must be uniform in T.** Implied vol is floored
    at 1e-6, imposed as `w ≥ 1e-12·T`. *If violated* — as it was, with an
    absolute floor on total variance — the floor binds only at tiny T and
    silently rewrites the short end: overnight and same-day expiries reported
    100 vol points instead of the correct flat short-end vol. Any clamp
    applied to total variance rather than variance has this failure mode.
    (T: pinned from `T = 1e-14` upward.)
