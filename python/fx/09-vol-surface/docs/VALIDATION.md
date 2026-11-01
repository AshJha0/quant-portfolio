# Validation: FX Volatility Surface & Stochastic Volatility

All numbers below come from the offline, seeded test suite (`pytest -q`)
and `examples/run_pipeline.py` on synthetic data. Test count and runtimes
are in the README.

## 1. Analytic identities and round trips

| Check | Tolerance | Result |
|---|---|---|
| Quotes ↔ five smile vols (linear RR/BF relations) | 1e-14 | exact round trip, all presets |
| GK put–call parity | 1e-14 | pass |
| GK implied vol round trip (incl. σ = 0.35, deep ITM) | 1e-10 | pass |
| Δ→K→Δ round trip, all 4 conventions × call/put × {10Δ, 25Δ} | 1e-8 | pass (typ. < 1e-12) |
| ATM DNS strike: `F·e^{+σ²T/2}` (unadj), `F·e^{−σ²T/2}` (pa); straddle delta-neutrality | 1e-12 | pass, both variants |
| pa call delta two-candidate solve: both roots reproduce Δ; market (high-strike) branch selected | 1e-10 | pass |
| Premium-adjustment identity Δ_pa = Δ − V/S | 1e-14 | pass |
| Strike ordering K10P < K25P < KATM < K25C < K10C | strict | pass, all preset slices |
| GK Greeks vs central FD (delta, gamma, vega, vanna, volga, both rhos, theta) | 1e-6 rel | pass |
| VV weights solve 3×3 replication system | 1e-12 | pass |
| VV reproduces pillar vols (by construction) | 1e-10 | pass |
| Surface returns every quoted pillar (native convention): all presets, SVI and VV, all four conventions on EURUSD | 1e-10 in vol, plus the smile's own miss where SVI cannot fit | pass (`test_surface_conventions.py`) |
| Broker one-vol strangle to smile BF and back | 1e-10 (EURUSD), 5x the SVI miss (USDJPY) | pass |
| Heston φ(0) = 1, φ(−i) = F (martingale), conjugate symmetry | 1e-13 | pass |
| Heston parity (both Fourier methods) | 1e-8 | pass |
| Digital: Heston `e^{−r_d T}P2` vs −dC/dK finite difference | 5e-5 | pass |

## 2. Cross-model / cross-method consistency

* **Two Fourier methods** (Gil-Pelaez quadrature vs COS N=1024, L=24):
  below 5e-9 in price at 1y for K in {0.9, ..., 1.3}
  (`test_cos_default_matches_gil_pelaez_quadrature`). At the old
  L = 14 the COS put carried a 9.7e-7 bias at every strike because the
  c2-only truncation interval was too narrow. COS prices more negative
  than 1e-12 K are returned as NaN with a warning instead of floored.
* **ξ→0 degeneracy**: Heston with ξ = 1e-4, ρ = 0 vs GK at the
  deterministic effective vol √(w̄/T): |diff| < 1e-6 (measured ~6e-10).
* **Monte Carlo vs Fourier** (3-SE contract): full-truncation Euler,
  120k paths × 250 steps: |z| ≤ 1.6; QE, 120k paths × **24** steps:
  |z| ≤ 2.4; QE under Feller violation (ξ = 0.8, 2κθ/ξ² = 0.05): pass;
  E[S_T] = F martingale check: pass. QE uses Andersen's martingale
  correction: at xi = 1.5, rho = -0.9, T = 5 on 2 and 5 steps the
  uncorrected forward is off by 7 to 13 SE and the corrected one is
  within 1 SE (`test_qe_martingale_correction_removes_forward_bias`).
  Pipeline run (200k paths, 1y ATMF, one seed): Euler 0.030845 ± 0.000094
  (z = 1.60), QE 0.031212 ± 0.000093 (z = 2.34) vs COS 0.030995. Over
  seeds 0 to 7 the QE z-scores average 0.49 (standard error 0.35), so
  the pipeline seed's 2.34 is sampling noise.
* **SVI vs vanna–volga** (EURUSD 3m, two independent constructions
  through the same pillars):

  | point | K | SVI vol | VV vol | diff |
  |---|---|---|---|---|
  | 35ΔP | 1.0890 | 7.391% | 7.378% | +1.3 bp |
  | 15ΔP | 1.0630 | 7.890% | 7.934% | −4.4 bp |
  | 15ΔC | 1.1450 | 7.310% | 7.318% | −0.8 bp |
  | 5ΔC  | 1.1710 | 7.645% | 7.751% | −10.5 bp |

  Body agreement to a few bp; wing divergence ~10 bp at 5Δ is the
  quantified interpolation model risk (nothing pins the smile there).
* **Digital three ways** (EURUSD 6m, K = 1.15): flat GK 0.2054, VV
  smile-consistent 0.1895, calibrated Heston 0.1899. The 1.6-point gap
  between flat and smile-aware prices is the skew correction
  −vega_dig·∂σ/∂K; VV (static) and Heston (dynamic) agree here to
  ~4 bp of discounted probability, that residual is the true model
  spread a desk reserves against on touch products.

## 3. Calibration evidence

**Ground-truth recovery** (quotes generated from Heston with
v0=0.0064, κ=1.8, θ=0.008, ξ=0.45, ρ=−0.35 and priced with the Gil-Pelaez
quadrature; calibration prices with COS, so the two do not share a
pricer; smile-consistent pillar strikes solved by fixed point;
calibration started from market-implied heuristics):

| param | true | recovered | rel err |
|---|---|---|---|
| v0 | 0.00640 | 0.00640 | < 0.01% |
| κ | 1.800 | 1.800 | < 0.01% |
| θ | 0.00800 | 0.00800 | < 0.01% |
| ξ | 0.450 | 0.450 | < 0.01% |
| ρ | −0.350 | −0.350 | < 0.01% |

RMSE 0.0000 vol points over 30 quotes. Test tolerances are looser (ρ ±0.02, v0 2%, θ 10%,
**κ 50%, ξ 30%, but ξ²/κ 15%**), documenting the **κ–ξ ridge**:
vanillas identify the convexity combination ξ²/κ (recovered 0.1125 vs
true 0.1125), not κ and ξ separately. With noisy quotes the optimiser
walks the ridge; day-on-day κ/ξ drift is expected and must not be
interpreted as a market signal.

**Preset calibrations** (clean quotes, RMSE contract < 0.25 vol pts):

| market | v0 | κ | θ | ξ | ρ | RMSE (vol pts) | max err |
|---|---|---|---|---|---|---|---|
| EURUSD | 0.00558 | 1.09 | 0.00768 | 0.19 | **−0.18** | 0.227 | 0.63 |
| USDJPY | 0.01107 | 0.86 | 0.01728 | 0.25 | **−0.47** | 0.181 | 0.50 |

The **economic pattern test**: EURUSD (small RR) calibrates to small
|ρ|; USDJPY (large negative RR, pa-quoted) to large negative ρ; the EM
preset's positive RR would flip ρ positive. Both Feller ratios ≈ 0.46 -
calibrated FX smiles routinely violate Feller, which is why QE is the
MC scheme of record.

## 4. Failure modes (each reproducible)

**F1, Short-dated jump risk (CB events) underpriced by pure
diffusion.** A diffusive Heston skew decays like √T as T→0; market 1w
skews around CB meetings do not. Visible in the EURUSD calibration
residuals: max error 0.63 vol pts sits at the 1w wing pillars while
3m–2y fit to < 0.1. Remedies: Bates jumps, or (desk practice) event
weights on the short-dated interpolation. Do not use this surface to
mark sub-1w event risk.

**F2: pa-delta convention mismatch corrupts the surface without an
error** (`run_pipeline.py` section 2). Take the USDJPY preset (pa-quoted)
and solve strikes with unadjusted deltas: the same quote sheet puts the
1y ATM strike at 145.34 instead of 143.76 (1.58 JPY too high; wing strikes
off 40 to 94 pips), and the marked surface is wrong at the true pillar
strikes by:

| tenor | 1w | 1m | 3m | 6m | 1y | 2y |
|---|---|---|---|---|---|---|
| vol error (pts) | 0.007 | 0.026 | 0.067 | 0.123 | **0.216** | **0.318** |

Every number stays plausible. The surface avoids this by reading each
slice in its quote convention (`delta_convention="native"`, the default).
Forcing one common convention on a pa-quoted market brings the error back.

**F3: broker one-vol strangle consumed as a smile BF.** The difference is
of order RR^2/ATM: on the USDJPY preset at 1y it is 0.16 vol points at
25 delta and 0.05 at 10 delta. Convert broker quotes with
`smile_quotes_from_broker` before building `SmileQuotes`.

**F4: extreme skew is not raw-SVI-interpolable.** The USDJPY call wing
keeps falling past 10 delta, which raw SVI cannot produce. The fit lands
on the rho = -0.9999 bound and misses the pillars by up to 0.0415 vol
points at 1y. `build_slice` warns, `SmileSlice.pillar_fit_error` records
the miss and `SVISmile.active_bounds` names the bound. EURUSD fits
exactly. Wing-sensitive uses should prefer the VV smile at pillars or an
SVI-JW / eSSVI extension.

**F5, 10Δ wing data sparsity.** 10Δ quotes are wide/illiquid; beyond
them the surface is pure extrapolation and the SVI-vs-VV divergence
(section 2) is the error bar. Vega-weighting already floors wing
weights at 0.05 so wing noise cannot hijack the calibration.

**F6, Pegged / managed pairs.** For USDHKD-, EURCHF-floor-style
pairs, the lognormal-diffusion premise is wrong: realised vol ≈ 0 with
a huge devaluation tail. RR/BF quotes become jump-risk prices; Heston
calibration will chase ξ→bound and the surface should not be used -
the 2015 CHF scenario in DESK_GUIDE shows what happens when a peg
breaks. (The EM preset with 35% vol / +6% RR covers the *floating*
devaluation-risk case, which does work.)

**F7: numerical limits.** Implied-vol inversion refuses prices outside
the static no-arbitrage bounds and prices at or within round-off of
intrinsic (raises, or NaN on request); it never returns the bracket edge.
Calibration replaces a missing model vol with a price-over-vega residual.
`FXVolSurface.vol` raises `SurfaceConvergenceError` when the fixed point
does not settle. pa call deltas above the attainable maximum raise with
the maximum reported. Planted calendar arbitrage (1m ATM bumped to 20%)
is detected at fixed forward moneyness, and planted butterfly arbitrage
(SVI b = 0.35, rho = -0.9) by the Durrleman check.

**F8: non-finite inputs.** The pricer, Greeks, forward and strike
solvers reject NaN and infinite spot, strike, expiry, rates and vol with
a `ValueError`. A parametrised sweep checks each of the six inputs
against NaN, +Inf and -Inf (`test_edge_cases_review.py`).

**F9, Deep-wing premium underflow (documented, not a defect).** At
microscopic expiries the OTM premium underflows to *exactly* 0.0 in
double precision (`N(d2)` with `d2 ~ -1e4`), e.g. a 1.20 strike at
T = 1e-6. This is correct IEEE behaviour, not an error: the value is
below the smallest representable positive double relative to spot.
The tests therefore assert `price >= 0` and `isfinite(price)` rather
than strict positivity, and the deep-wing bound test confirms prices
stay inside the static no-arbitrage envelope with put-call parity
holding to 1e-12 out to a 6.00 strike
(`test_deep_wing_option_prices_stay_within_no_arbitrage_bounds`).
Consequence for the desk: never invert an implied vol from a
deep-wing premium at very short expiry, there is no information left
in the number.

## 5. Convergence / stability studies

* COS truncation and terms against Gil-Pelaez (1y, EURUSD-like
  parameters): L=14 leaves 9.7e-7 at any N; L=24 gives 4e-8 at N=256 and
  7e-10 at N=512 and N=1024. Defaults: L=24 with N=1024 for pricing and
  N=512 for calibration.
* Heston FD Greeks: halving all bump sizes moves every Greek by
  < 0.5% (test `test_fd_step_stability`); vanna/volga FD vs BS
  analytic: signs agree at 25Δ/10Δ wings in both models; with ξ→0 and
  flat variance, all nine FD Greeks converge to analytic GK.
* Little-trap CF: continuous and |φ| ≤ 1 on u ∈ (0, 100] at T = 15y
  (the regime where the original Heston form loses the branch).
* QE vs Euler step economics: QE at 24 steps/yr matches Fourier within
  MC noise where Euler needs 250, which is why QE is used under Feller
  violation.
