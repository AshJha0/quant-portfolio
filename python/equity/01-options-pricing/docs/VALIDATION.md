# Validation: Equity Options Pricing and Greeks Engine

This document covers how the engine was validated (analytic benchmarks,
an independent oracle, convergence studies, cross-model and cross-language
checks) and where it fails. Two commands produce every number below:

- `PYTHONPATH=src python examples/run_pipeline.py` (sections 2, 3, 4, 5).
- `PYTHONPATH=src python examples/validation_tables.py` (variance
  reduction, hedging noise, sections 6.1 and 6.3).

`python -m pytest tests -q` runs 2039 offline, seeded tests (most of them
parametrised grid points; see the README for the breakdown).

Reference contract unless stated otherwise:
`S=100, K=100, T=1y, r=5%, q=1%, sigma=20%`.

---

## 1. Benchmarks and oracles

| Check | Result | Test |
|---|---|---|
| Golden vectors: 456 records from mpmath at 50 digits (BSM price and all Greeks incl. vanna/volga, Black-76, implied vol, CRR European and American, edge limits) | every output within its own tolerance; worst deviation 0.18 of tolerance | `test_golden.py` |
| Put-call parity over a 1080-point grid | max abs error < 1e-10 | `test_put_call_parity_full_grid_1e10` |
| Hull: S=42, K=40, T=0.5, r=10%, sigma=20% | call 4.7594, put 0.8086 | `test_textbook_values` |
| Haug generalised-BSM put: S=75, K=70, T=0.5, r=10%, b=5%, sigma=35% | 4.0870 | `test_textbook_values` |
| Black-76 with `F = S e^{(r-q)T}` equals BSM | abs diff < 1e-10 | `test_black76_equals_bsm_on_model_forward` |
| Analytic Greeks vs central finite differences | rel err < 1e-4 | `test_analytic_vs_finite_difference_all_greeks` |
| Implied vol against mpmath over ln(K/S) in [-5, 5], T in [1e-7, 50], sigma in [1e-4, 5] | within 4 max(eps max(S,K)/vega, 1e-12 sigma), at most 10 evaluations | `test_iv_accuracy.py` |
| Property tests (hypothesis): parity, monotone in vol and strike, strike convexity, delta bounds, IV round trip | 300 examples each | `test_property_based.py` |
| Cross-language: Python vs Rust vs C++ on 640 seeded random requests | max abs deviation 1e-13 (prices), 2e-13 (implied vol) | `test_differential.py` |

The golden tolerances are per output: `64 eps (1 + d1^2)(|v| + max(S, K))`
for closed forms, `4 max(eps max(S,K)/vega, 1e-12 sigma)` for implied vol
and `4 n eps max(S, K)` for an n-step tree. The JSON states each rule.

## 2. Convergence: CRR tree to Black-Scholes

European ATM call, reference contract (BS price **9.826298**):

| n_steps | tree price | abs error | error x n |
|---:|---:|---:|---:|
| 10 | 9.632077 | 0.194221 | 1.94 |
| 25 | 9.898850 | 0.072552 | 1.81 |
| 50 | 9.787003 | 0.039295 | 1.96 |
| 100 | 9.806625 | 0.019673 | 1.97 |
| 250 | 9.818423 | 0.007875 | 1.97 |
| 500 | 9.822359 | 0.003939 | 1.97 |
| 1000 | 9.824328 | 0.001970 | 1.97 |
| 2000 | 9.825313 | 0.000985 | 1.97 |

`error x n` is about 1.97 at every n, so convergence is O(1/n). The
odd/even oscillation shows at low n (n=25 overshoots).

## 3. Convergence: Monte Carlo to Black-Scholes

Antithetic plus control variate, independent seeds per row:

| n_paths | MC price | abs error | std error | SE x sqrt(n) |
|---:|---:|---:|---:|---:|
| 1,000 | 9.811312 | 0.014986 | 0.080063 | 2.53 |
| 4,000 | 9.812672 | 0.013626 | 0.042859 | 2.71 |
| 16,000 | 9.831026 | 0.004728 | 0.022084 | 2.79 |
| 64,000 | 9.826594 | 0.000296 | 0.011232 | 2.84 |
| 256,000 | 9.824166 | 0.002132 | 0.005625 | 2.85 |

`SE x sqrt(n)` settles near 2.85, the O(n^{-1/2}) law.

Variance reduction at 100k paths (seed 7):

| estimator | SE |
|---|---:|
| plain | 0.0449 |
| antithetic | 0.0326 |
| control variate | 0.0179 |
| antithetic + control variate | 0.0089 |

The control-variate coefficient is fitted on the antithetic pair averages,
which are the independent samples. Fitting it on the raw draws and pairing
afterwards mis-sizes the coefficient; that combination gave 0.0247 here,
worse than the control variate alone. Tests check, over seven seeds, that
the combined SE is no larger than the control-variate SE.

MC Greeks: pathwise and likelihood-ratio delta and vega match analytic
values within 3 SE; pathwise has lower variance than LR for vanilla delta.

## 4. American exercise (CRR)

- American put >= European put on every tested contract.
- American call with q=0 equals the European call to 1e-10 (Merton).
- Reference 1y ATM put (q=1%): American **6.366622**, early-exercise
  premium **0.423350** at 2000 steps.
- S=K=100, T=1, r=5%, sigma=20%, q=0: American put **6.0896 +/- 0.005**
  at 2000 steps.
- S=0 with r<0: the holder waits, value `K e^{-rT}` (> K).
- Exercise values are precomputed per node level (O(n) exps), so a
  2000-step American put takes 26 ms instead of 64 ms on the test
  machine (timing loop around `crr_price`, single run).

## 5. Hedging validation

Short 3M ATM call (S=100, r=2%, sigma=20%), 4000 paths, seeded:

| n_rebalance | P&L std | sqrt(pi/4) vega sigma / sqrt(N) |
|---:|---:|---:|
| 4 | 1.5993 | 1.7590 |
| 16 | 0.8300 | 0.8795 |
| 64 | 0.4263 | 0.4397 |
| 256 | 0.2195 | 0.2199 |

The simulated std follows the Derman-Kamal estimate
`sqrt(pi/4) vega sigma / sqrt(N)`, the 1/sqrt(N) hedging-error law.

- Hedged at true vol, N=128: mean P&L **+0.0014** (SE 0.0035).
- Sold at 25 vol, realised 15 vol: mean P&L **+1.996** against the
  gamma-weighted vol-spread value **+1.987**.
- 5bp proportional costs at N=128: mean P&L **-0.2341**.

## 6. Failure modes

### 6.1 Vol smile contradiction
BS uses one sigma for all strikes. Pricing a 3M put chain at the synthetic
skew against a single ATM vol (synthetic data, `validation_tables.py`):

| K | skew IV | put @ skew IV | put @ ATM IV | mispricing |
|---:|---:|---:|---:|---:|
| 80 | 25.6% | 0.1703 | 0.0345 | -80% |
| 90 | 22.5% | 0.9129 | 0.6442 | -29% |
| 110 (ITM) | 18.4% | 10.2756 | 10.4953 | +2% |

A flat-vol user underprices the 80-strike put by about 5x on this
synthetic skew. Vol is a per-quote input for this reason.

### 6.2 Discrete hedging error
Hedging 4 times over a 3M option leaves a P&L std of about 1.6 on a 4.42
premium even at the correct vol (section 5). This is the risk the
replication argument assumes away (assumption A2).

### 6.3 Dividend modelling error
A continuous yield spreads a discrete dividend over time. Stock pays D=3
at t=0.1y; ATM calls, sigma=20%, r=3% (`validation_tables.py`, escrowed
model as the reference):

| Expiry | escrowed dividend | continuous q=3% | error |
|---|---:|---:|---:|
| T=0.08 (before ex-date) | 2.3756 | 2.2510 | -0.125 |
| T=0.12 (after ex-date) | 1.6015 | 2.7535 | +1.152 |

Continuous q is acceptable for indices and long-dated single names. Short
single-name options around ex-dates need an escrowed or discrete-dividend
model, which this package does not implement.

### 6.4 CRR limitations
- O(1/n) convergence with odd/even oscillation: n=10 misprices the
  reference call by 0.19. Use n >= 500.
- Tree Greeks are noisy; `fd_greeks` on the tree needs bumps >= 1e-3.
- The tree raises when `p` leaves (0, 1), that is when
  `|r - q| sqrt(dt) > sigma`.
- No discrete dividends or term structures on the tree.

### 6.5 Numerical limits
- **T=0, sigma=0:** defined limits. The sigma=0 price is computed as
  `S e^{-qT} - K e^{-rT}`, so it stays finite when the forward
  overflows ((r-q)T > 709).
- **Overflow:** a discount factor or forward that overflows raises
  `ValueError`, never `OverflowError` or `inf`.
- **Implied vol near the bounds:** a price at or below the discounted
  forward intrinsic, or at or above the upper bound, raises `ValueError`.
- **Implied vol on the in-the-money side:** the solver works on the
  out-of-the-money time value, but an ITM quote carries less information.
  With S=100, K=90, T=0.02, sigma=10%, one ulp of the call price
  (1.8e-15) moves the vol by about 5e-4 (vega 3.5e-12). The solver then
  returns a vol that reprices the quote to a few ulps; the OTM put at the
  same strike recovers the vol to 1e-12.
- **Implied vol accuracy elsewhere:** within
  `max(eps price / vega, 1e-12 sigma)`; the 1e-12 floor comes from
  rounding of the normal-CDF argument in the far tail.
- **NaN/Inf inputs:** rejected with `ValueError` at every public entry
  point, including r and q.
- **Monte Carlo:** fewer than two independent samples (three with the
  control variate) raises; one antithetic pair has no standard error.

## 7. Oracle tests for the newer models (November 2026)

`tests/test_phase5_models.py`, 51 tests. Every model has at least one oracle:

| Model | Oracle | Tolerance |
|---|---|---|
| LR tree | Black-Scholes at 401 steps; error ratio per doubling in (3, 5) | 2e-5 abs |
| CN PDE | Black-Scholes (European); LR 2001/4001 Richardson (American); error ratio > 3 per doubling | 5e-4, 3e-4 abs |
| ALO | LR 2001/4001 Richardson, four cases (put, call, q > r, ITM) | 2e-5 abs |
| ALO, negative rates | LR 2001/4001 Richardson: four single-boundary cases (r > 0 > q, r = 0 > q, negative-rate call) and four double-boundary cases (q < r < 0 put incl. T past the band's close, r < q < 0 call); grid convergence; band shape (starts at [K r/q, K], monotone, closes); no-exercise regions equal Black-Scholes | 2e-5 abs |
| ALO double boundary, collocation (round 4) | Five `q < r < 0` cases against the 400-step march and the LR 2001/4001 Richardson tree; T = 30 (band closed at tau* ~ 11) against a 1600-step march Richardson value; convergence in `n_cheb`; closing time and boundary values against the march; fallback to the march; best-of-seven time under 50 ms; the march at 1600 steps (3200 checked by hand) on `S=80, K=100, T=0.5, r=-1 %, q=-3 %, vol 30 %` runs to T with a monotone band and matches collocation (regression for the first-step failure fixed in round 4) (`tests/test_round4_alo_collocation.py`) | 1e-5 abs (tree), 2e-5 (march) |
| RGW | Haug (2007) section 9.2.1 published value 4.3860 | 5e-5 abs |
| Dividend tree | RGW at 4001 steps; escrowed closed form for Europeans | 1e-3, 1e-5 abs |
| Term structures | Hand-computed Merton averages; flat curves equal BS | 1e-14 rel |
| Bachelier | mpmath quadrature of the payoff; mpmath derivatives; implied-vol round trip | 1e-15 abs, 1e-13 abs, 1e-10 rel |
| SABR | Hagan ATM formula; beta = 1, nu = 0 limit; smoothness through ATM; density sign | 1e-14 rel |
| SABR PDE | Mass and forward conservation (both schemes, four parameter sets); non-negative masses and convex, decreasing calls where the expansion's density is negative; exact put-call parity; flat vol in the lognormal limit; Hagan's expansion at T = 0.5; grid convergence; TR-BDF2 against fine-step implicit Euler | 1e-12, 0, 1e-15, 1e-4, 2e-5 / 5e-4 vol, 2e-5 vol |
| Higher Greeks | `mpmath.diff` at 30 digits, eight cases | 1e-9 rel |
| Ladders, grids | Direct repricing; bucket sums | 1e-14 rel |

A precision defect found while testing: the naive `log` form of SABR's
`x(z)` loses `eps / z` relative accuracy just above the series cut-over
(`|z| = 1e-6`), which showed up as a 1e-3 relative jump in the smile slope.
The `log1p` form fixes it and the smoothness test guards it.

Not validated: SABR (expansion or PDE) against Monte Carlo of the full two-factor model; the
CN pricer beyond vanilla payoffs.
