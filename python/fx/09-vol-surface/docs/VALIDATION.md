# Validation — FX Volatility Surface & Stochastic Volatility

All numbers below are produced by the offline, seeded test suite
(`pytest -q`: **195 tests**, ~11 s) and `examples/run_pipeline.py`
(~6 s). Every figure quoted here is reproducible from those two
commands.

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
| Heston φ(0) = 1, φ(−i) = F (martingale), conjugate symmetry | 1e-13 | pass |
| Heston parity (both Fourier methods) | 1e-8 | pass |
| Digital: Heston `e^{−r_d T}P2` vs −dC/dK finite difference | 5e-5 | pass |

## 2. Cross-model / cross-method consistency

* **Two Fourier methods** (Gil-Pelaez quadrature vs COS N=1024, L=14):
  max |diff| < 1e-6 over T ∈ {1w, 3m, 1y, 2y} × K ∈ {0.90…1.40}·S
  (measured max ≈ 9.7e-7 at the short-dated far wing; ≈ 3e-8 in the
  body). COS noise floor at >8-sigma strikes is ~1e-6 absolute —
  prices there are floored at intrinsic/zero.
* **ξ→0 degeneracy**: Heston with ξ = 1e-4, ρ = 0 vs GK at the
  deterministic effective vol √(w̄/T): |diff| < 1e-6 (measured ~6e-10).
* **Monte Carlo vs Fourier** (3-SE contract): full-truncation Euler,
  120k paths × 250 steps: |z| ≤ 1.6; QE, 120k paths × **24** steps:
  |z| ≤ 2.4; QE under Feller violation (ξ = 0.8, 2κθ/ξ² = 0.05): pass;
  E[S_T] = F martingale check: pass. Pipeline run (200k paths, 1y
  ATMF): Euler 0.030845 ± 0.000094 (z = 1.60), QE 0.031212 ± 0.000093
  (z = 2.34) vs COS 0.030995.
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
  ~4 bp of discounted probability — that residual is the true model
  spread a desk reserves against on touch products.

## 3. Calibration evidence

**Ground-truth recovery** (quotes generated *from* Heston with
v0=0.0064, κ=1.8, θ=0.008, ξ=0.45, ρ=−0.35; smile-consistent pillar
strikes solved by fixed point; calibration started from generic
market-implied heuristics):

| param | true | recovered | rel err |
|---|---|---|---|
| v0 | 0.00640 | 0.00640 | < 0.01% |
| κ | 1.800 | 1.800 | < 0.01% |
| θ | 0.00800 | 0.00800 | < 0.01% |
| ξ | 0.450 | 0.450 | < 0.01% |
| ρ | −0.350 | −0.350 | < 0.01% |

RMSE 0.0000 vol pts over 30 quotes; 42 objective evaluations, ~0.1 s.
Test tolerances are intentionally looser (ρ ±0.02, v0 2%, θ 10%,
**κ 50%, ξ 30%, but ξ²/κ 15%**) — documenting the **κ–ξ ridge**:
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
preset's positive RR would flip ρ positive. Both Feller ratios ≈ 0.46 —
calibrated FX smiles routinely violate Feller, which is why QE is the
MC scheme of record.

## 4. Failure modes (each reproducible)

**F1 — Short-dated jump risk (CB events) underpriced by pure
diffusion.** A diffusive Heston skew decays like √T as T→0; market 1w
skews around CB meetings do not. Visible in the EURUSD calibration
residuals: max error 0.63 vol pts sits at the 1w wing pillars while
3m–2y fit to < 0.1. Remedies: Bates jumps, or (desk practice) event
weights on the short-dated interpolation. Do not use this surface to
mark sub-1w event risk.

**F2 — pa-delta convention mismatch silently corrupts the surface**
(the desk-realistic bug demo, `run_pipeline.py` §2). Take the USDJPY
preset (pa-quoted) and *wrongly* solve strikes with unadjusted deltas:
the same quote sheet puts the 1y ATM strike at 145.34 instead of 143.76
(1.58 JPY / 158 pips too high; wing strikes off 40–94 pips), and the
marked surface is wrong at the true pillar strikes by:

| tenor | 1w | 1m | 3m | 6m | 1y | 2y |
|---|---|---|---|---|---|---|
| vol error (pts) | 0.007 | 0.026 | 0.067 | 0.123 | **0.216** | **0.318** |

0.2–0.3 vol pts on a 10-vol surface is many times a market-maker's
edge, and *nothing fails loudly* — every number is plausible. This is
why convention governance (DESK_GUIDE §5) exists.

**F3 — Broker one-vol strangle consumed as smile BF.** See
METHODOLOGY §1: O(RR²/σ) wing error, up to ~1 vol pt at 10Δ for skewed
pairs. This package's quotes are smile-BF by construction; ingesting
real broker strangles requires the nonlinear conversion first.

**F4 — Extreme skew is not raw-SVI-interpolable.** The USDJPY smile
would need SVI ρ < −1 for an exact 5-point fit; the fit lands on the
ρ = −0.9999 bound with pillar residual ≤ 0.05 vol pts (measured
0.041 at 1y; EURUSD fits exactly, < 1e-7). Tested explicitly. Wing-
sensitive uses should prefer the VV smile at pillars or an SVI-JW /
eSSVI extension.

**F5 — 10Δ wing data sparsity.** 10Δ quotes are wide/illiquid; beyond
them the surface is pure extrapolation and the SVI-vs-VV divergence
(§2) is the honest error bar. Vega-weighting already floors wing
weights at 0.05 so wing noise cannot hijack the calibration.

**F6 — Pegged / managed pairs.** For USDHKD-, EURCHF-floor-style
pairs, the lognormal-diffusion premise is wrong: realised vol ≈ 0 with
a huge devaluation tail. RR/BF quotes become jump-risk prices; Heston
calibration will chase ξ→bound and the surface should not be used —
the 2015 CHF scenario in DESK_GUIDE shows what happens when a peg
breaks. (The EM preset with 35% vol / +6% RR covers the *floating*
devaluation-risk case, which does work.)

**F7 — Numerical limits.** COS absolute noise ~1e-6 beyond 8σ strikes
(prices floored); implied-vol inversion refuses prices outside static
no-arb bounds (raises, or NaN on request — calibration maps these to
bracket edges); pa call deltas above the attainable maximum raise with
the maximum reported; planted calendar arbitrage (1m ATM bumped to
20%) and planted butterfly arbitrage (SVI b = 0.35, ρ = −0.9) are both
detected by the checks.

## 5. Convergence / stability studies

* COS N-refinement: N=256/L=12 → max 4.1e-6 vs Gil-Pelaez; N=1024/L=14
  → 9.7e-7 (default). Calibration uses N=256 (0.0004 vol-pt effect,
  ~40× faster than quadrature).
* Heston FD Greeks: halving all bump sizes moves every Greek by
  < 0.5% (test `test_fd_step_stability`); vanna/volga FD vs BS
  analytic: signs agree at 25Δ/10Δ wings in both models; with ξ→0 and
  flat variance, all nine FD Greeks converge to analytic GK.
* Little-trap CF: continuous and |φ| ≤ 1 on u ∈ (0, 100] at T = 15y
  (the regime where the original Heston form loses the branch).
* QE vs Euler step economics: QE at 24 steps/yr matches Fourier within
  MC noise where Euler needs 250 — the reason QE is the production
  scheme under Feller violation.
