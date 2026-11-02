# Methodology: Equity Volatility Surface & Heston Stochastic Volatility

Why these models (against named alternatives) and what assumptions are made
(with what breaks when each is violated). Numbers are from synthetic data.

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
value all return `nan` with a warning. A wing quote whose vega has
underflowed contains no volatility information, and inverting it would
feed a meaningless number into every downstream fit.

### Smiles: raw SVI

Total implied variance per expiry, `w(k) = σ²(k)·T`, `k = ln(K/F)`:

    w(k) = a + b·( ρ(k−m) + √((k−m)² + σ²) )

Butterfly (strike) arbitrage is absent iff Durrleman's

    g(k) = (1 − k·w′/2w)² − (w′²/4)(1/w + 1/4) + w″/2 ≥ 0  for all k,

with the SVI derivatives computed analytically:
`w′ = b(ρ + (k−m)/√((k−m)²+σ²))`, `w″ = bσ²/((k−m)²+σ²)^{3/2}`
(validated against numerical differentiation to 1e-8).

`fit_svi` is a constrained fit. It imposes the raw-SVI domain, Lee's wing
bound `b(1+|ρ|) ≤ 2` on total variance (the moment formula; in variance
units it is `2/T`, inside the Rogers-Tehranchi necessary bound `4/T`), and
`g(k) ≥ 0` on the quotes' k range plus a margin of 1 on each side
(Gatheral and Jacquier 2014, Thm 2.1). Under the Lee bound the limit of
g in the wings is `1/4 − slope²/16 ≥ 0`, so the far wings are covered too.
The butterfly constraint is enforced on a grid, and after each SLSQP solve
the local minima of g are located on a fine grid, polished with a scalar
minimiser, and added as constraint points until g ≥ 0 everywhere checked.
Starting points come from the Zeliade quasi-explicit split: for fixed
(m, σ) the slice is linear in (a, d = ρbσ, c = bσ), solved in closed form
on a grid. Default residual weights `1/(2√(wT))` make the objective a vol
error; vega or bid-ask weights can be passed. Active constraints and
bounds are reported on the result. `arbitrage_free=False` keeps the
unconstrained least-squares fit.

### Surface: interpolate total variance, never vol

At fixed forward moneyness, absence of calendar arbitrage ⇔ total variance
non-decreasing in T. Linear interpolation of a monotone quantity stays
monotone, so calendar-free pillars give a calendar-free surface, and the
implied forward variance between pillars is piecewise-constant and
non-negative. Linear interpolation **of vol** has no such property: a
high-vol short pillar next to a low-vol long pillar can interpolate to
decreasing total variance, negative forward variance, between two
individually arbitrage-free slices. Extrapolation policy (documented in
`surface.py`, unit-tested): flat vol below the first pillar
(w ∝ T), linear-in-T continuation of the last slope beyond the last pillar
with the slope floored at zero. Linear-in-T interpolation keeps calendar
arbitrage out but not butterfly arbitrage: a blend of two butterfly-free
slices can have g < 0. `VolSurface.butterfly_report` checks g on the
surface itself at pillars and midpoints, and runs automatically after
calendar enforcement (which interpolates the adjusted grid with PCHIP in k
so the density stays finite).

### Heston

    dS = (r−q)S dt + √v S dW₁,   dv = κ(θ−v) dt + ξ√v dW₂,   ⟨dW₁,dW₂⟩ = ρ dt

Characteristic function in the **Gatheral / "little Heston trap"** form
(Albrecher et al. 2007): the original Heston formulation contains
`ln((1−g₁e^{dT})/(1−g₁))` with a growing exponential whose argument winds
across the negative real axis, so the principal-branch complex log jumps by
2πi, discontinuous prices for long maturities/large u. Rewriting with the
conjugate root g₂ = (b−d)/(b+d) makes the exponential decaying
(`e^{−dT} → 0`), the log argument never circles the origin, and the principal
branch is continuous everywhere relevant. Tested by scanning φ(u) on a dense
grid at T=10, ξ=1 (no jumps).

Pricing goes through two independent Fourier routes, cross-validated to
1e-6 in the suite: (i) Heston's semi-analytic P₁/P₂ probabilities under
adaptive quadrature; (ii) Carr–Madan-style damped direct integration
(α = 1.5), plus a Gauss–Legendre vectorised version of (ii) as the fast path
for calibration. The integration cutoff is chosen by probing the CF envelope,
because the Heston CF decays only like `exp(−u(v₀+κθT)√(1−ρ²)/ξ)`, slowly at
high vol-of-vol, where a fixed cutoff silently truncates tail mass (this was
measurable: a fixed u_max = 200 lost 1.5e-4 of price at ξ = 1).

**Feller condition** 2κθ ≥ ξ²: checked and warned, never raised. Equity
index calibrations routinely violate it (short-dated skew needs high ξ
relative to κθ), and the model remains well-defined (variance
touches zero and reflects; the CF is still valid). The fitted surface in this
project has Feller ratio 0.80. What violation *does* affect is simulation
scheme choice (see below) and the credibility of far-forward smile dynamics.

### Monte Carlo

Full-truncation Euler (v⁺ in drift and diffusion) and Andersen's QE scheme
(moment-matched squared-Gaussian / exponential mixture for the CIR step,
log-spot update with central weights and Andersen's martingale correction,
so the simulated discounted forward has mean S e^{−qT} at any step size).
Euler carries an O(dt) truncation bias that grows sharply when Feller is
violated; QE stays near Monte Carlo noise at coarse steps (VALIDATION.md
section 3). `heston_mc_price` offers antithetic variates and a
terminal-spot control variate.

### Calibration

Weighted least squares in implied-vol space. Residuals are
`(model_iv − market_iv) · vega_mkt`, which to first order is a price
residual: vega grows like √T, so long expiries and near-ATM strikes carry
more weight than short expiries and far wings. The out-of-the-money
option is inverted at each strike. Where a trial parameter set gives a
price with no implied vol, the residual falls back to the price error,
keeping the objective continuous. `scipy.least_squares` (TRF, box
bounds), multi-start with deterministic seeds, Jacobian condition number
reported at the optimum. See section 3 on identifiability.

---

## 2. Why these models: alternatives and trade-offs

### Why SVI for smiles (vs polynomial / kernel fits)

| Criterion | Raw SVI | Polynomial in k or delta | Spline / kernel |
|---|---|---|---|
| Parameters per expiry | 5, each interpretable | 3–6, uninterpretable | many |
| Wings | Linear in total variance; Lee bound imposed in the fit | Explode (odd powers) or collapse | Uncontrolled beyond data |
| No-arbitrage machinery | Durrleman g(k) analytic and imposed in the fit | None | None |
| Fit quality on the synthetic equity smiles | 0.0006 to 0.041 vol points | 0.50 to 0.75 vol points | Exact in-sample, unstable outside |
| Extrapolation | Meaningful by construction | Meaningless | Meaningless |

The quadratic-in-delta baseline (`fit_quadratic_delta`) makes this
concrete: on the same slices it fits 10 to 100 times worse (0.50 to 0.75
vol points against 0.0006 to 0.041 for SVI, pipeline section 3) and has no
density-positivity diagnostics.

### Why Heston (vs local vol vs SABR)

| Criterion | Heston | Local vol (Dupire) | SABR (per expiry) |
|---|---|---|---|
| Reprices today's vanillas | Approximately (2–3 vp short-dated residual) | **Exactly** by construction | Per-slice only, no term structure |
| Smile dynamics | Sticky-moneyness at fixed v0 (homogeneous of degree one in S, K); vol clusters and mean-reverts | Smile moves against spot; forward smiles flatten (Hagan et al. 2002) | None across expiries |
| Path-dependent / forward-start exotics | Credible | Systematically mispriced | Not applicable |
| Semi-analytic vanillas | Yes (Fourier) | PDE/MC only | Yes (Hagan expansion, biased in wings) |
| Parameters | 5 global, economically meaningful | A full surface (over-parameterised) | 4 per expiry |
| Calibration risk | κ/ξ ridge (documented, measured) | Interpolation artefacts differentiate badly | Slice-to-slice inconsistency |

The trade-off: local vol reprices today's vanillas exactly but implies
forward smiles that flatten; Heston gives stochastic-vol dynamics but
reprices imperfectly (the short-dated skew residual is measured in
VALIDATION.md). Neither regime matches equity index behaviour in a
sell-off exactly, which is closer to local vol than to sticky-moneyness
(DESK_GUIDE.md section 4). For pricing forward-skew-sensitive
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

## 3. Parameter identifiability: the κ/ξ ridge

Vanilla smiles inform Heston parameters only through smile level, skew,
curvature and their decay in T. `v0` (short ATM level) and `ρ` (skew sign and
size) map almost one-to-one onto observables and calibrate tightly. `κ` and
`ξ` are identified *jointly* through the rate at which skew/curvature decay
with maturity: raising mean reversion and raising vol-of-vol nearly cancel in
vanilla prices, leaving a curved, flat-bottomed valley in the objective. The
suite asserts the measured symptom, a Jacobian condition number of about
1e3 at the optimum (1.2e3 on the clean fit), and VALIDATION.md section 5.2
shows the day-over-day consequence under 0.3 vol points of quote noise. Desk handling (parameter smoothing,
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
   curvature and skew are underfit, diffusive Heston cannot generate enough
   skew at 1 week. The calibration-to-SVI failure mode in VALIDATION.md
   section 5.1 shows the residual concentrated at the short end. Realistic short-dated equity marking
   needs jumps (Bates) or rough vol. (T: residual pattern reproduced.)
3. **Single volatility factor.** *If violated*: cannot decorrelate short-
   and long-dated vol moves; term-structure twists (e.g. inversion in a
   crash) force parameter instability instead of factor reallocation.
4. **Deterministic rates and proportional continuous dividend yield q,
   ACT/365F.** *If violated*: equity-rate hybrid effects and discrete
   dividend jumps are mispriced; near ex-div dates the forward (and hence
   moneyness) is wrong, see DESK_GUIDE.md on dividend risk. (T: q > 0
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
8. **Feller may be violated**: variance touches zero. *What breaks*:
   naive Euler simulation (bias of tens to hundreds of SE, VALIDATION.md
   section 3), not the pricing. Full truncation/QE handle it. (T)
9. **Surface extrapolation policy is a modelling choice** (flat vol short
   end, slope-floored linear w long end, SVI wings in k). *If violated by
   reality* (e.g. earnings before the first pillar): short-end marks are
   wrong, see DESK_GUIDE.md earnings scenario. (T: policy pinned by tests.)
10. **Deep wings carry no invertible information** below double-precision
    time value; they are dropped (`nan`) rather than fitted. *If instead
    inverted*: garbage vols in the wings would corrupt SVI wings and hence
    density tails. (T)
11. **No-arbitrage**: butterfly arbitrage is excluded in each SVI fit by
    construction (Lee bound and Durrleman constraint), and re-checked on
    the surface at interpolated expiries. Calendar violations warn and are
    exposed on the result; the running-max repair is opt-in. *If a
    surface with arbitrage were consumed*, a negative risk-neutral density
    produces negative butterfly prices and wrong variance-swap and digital
    marks. The checkers are tested against planted violations. (T)
12. **The surface's vol floor is uniform in T.** Implied vol is floored
    at 1e-6, imposed as `w ≥ 1e-12·T`. An absolute floor on total variance
    would bind only at tiny T and rewrite the short end: overnight expiries
    would report 100 vol points instead of the flat short-end vol. Any clamp
    applied to total variance rather than variance has this failure mode.
    (T: pinned from `T = 1e-14` upward.)

## 5. November 2026 additions: SSVI/eSSVI, Dupire, Bates

**SSVI.** One formula for the whole surface: total variance at log-moneyness
k is `theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))`, where
theta is the ATM total variance of the expiry and `phi(theta)` is a power law
(Gatheral and Jacquier 2014, eq. 4.1, Remark 4.5). Each slice is a raw SVI
slice, so the existing SVI tools apply. If `eta (1 + |rho|) <= 2` the surface
has no static arbitrage (Theorem 4.1, Remark 4.4). eSSVI lets rho move with
theta (Hendriks and Martini 2019); here rho is linear in theta between two
anchors, and calendar arbitrage is checked on a grid rather than proved.

**Dupire.** Local variance is the time slope of total variance divided by a
convexity term in k (Gatheral 2006, eq. 1.10), or, equivalently, from call
prices (Dupire 1994; Gatheral 2006, eq. 1.4). Both use central finite
differences. If the numerator or the denominator goes negative, the input
surface has arbitrage there, and the code raises `LocalVolArbitrageError`
instead of clipping.

**Bates.** Heston plus compound-Poisson lognormal jumps in the log price.
The characteristic function is the Heston one times the jump factor, with a
compensator that keeps the forward a martingale (Bates 1996; Gatheral 2006,
eq. 5.6). Pricing uses the Lewis (2001) single integral.

Assumptions: jumps independent of the diffusion; constant jump parameters;
eSSVI calendar no-arbitrage is checked numerically, not guaranteed.
