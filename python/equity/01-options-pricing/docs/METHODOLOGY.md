# Methodology: Equity Options Pricing and Greeks Engine

This document answers documentation-contract items 1 and 2: **why these
models** (against alternatives, with trade-offs) and **what assumptions we
make** (numbered register, each with "what breaks if violated").

Conventions used throughout: rates `r` and dividend yields `q` are
continuously compounded, annualised, ACT/365F; `T` is in years; `sigma` is
the annualised volatility of log-returns.

---

## 1. The models and why we chose them

The engine contains *four* pricers for the same European
contract plus an American extension. That redundancy is the point: on a
desk, independent implementations that must agree are the first line of
model validation, and each engine is the right tool in a different regime.

### 1.1 Black-Scholes-Merton (closed form), the baseline

The price of a European call with continuous dividend yield is

    C = S e^{-qT} N(d1) − K e^{-rT} N(d2)
    d1 = [ln(S/K) + (r − q + σ²/2) T] / (σ√T),   d2 = d1 − σ√T

**Why baseline.** It is exact under the model assumptions, evaluates in
about 4 µs per call in this scalar Python implementation (timeit loop on
`bs_price`, test machine), has closed-form
Greeks to arbitrary order (we ship delta, gamma, vega, theta, rho, vanna,
volga), and is the *quoting convention* of the listed-options market:
prices are communicated as implied BS vols. Even desks that price with
local/stochastic vol still mark, hedge and report risk in BS-implied terms.

**Alternatives considered.**

| Alternative | Why not the baseline here |
|---|---|
| Local vol (Dupire) | Fits the whole smile, but needs a calibrated vol surface as *input*; wrong forward-smile dynamics; overkill for vanillas. This project is the reference layer such a model is validated against. |
| Stochastic vol (Heston/SABR) | Captures smile dynamics and vol-of-vol, but pricing needs numerical integration/FFT, calibration is ill-posed on sparse chains, and Greeks are model-dependent. Right tool for exotics and smile *dynamics*, not for a vanilla reference engine. |
| Jump-diffusion (Merton) | Explains short-dated smile, but adds 3 poorly identified parameters; hedging is incomplete under jumps so "the" delta is ambiguous. |

**Trade-off accepted:** BS assumes a single flat vol. We treat the smile as
an *input marking* (one vol per strike/expiry, exactly how desks use it)
rather than a model output, and we demonstrate the resulting inconsistency
explicitly in VALIDATION.md ("vol smile contradiction").

### 1.2 Cox-Ross-Rubinstein binomial tree, for early exercise

`u = e^{σ√Δt}`, `d = 1/u`, `p = (e^{(r−q)Δt} − d)/(u − d)`, vectorised
backward induction.

**Why a tree.** The only model in this project that prices *American*
exercise, which single-name equity options actually carry. It converges to
BS at O(1/n) for Europeans, which is simultaneously our validation of the
tree and a demonstration of when you would *not* use it (a tree is ~10⁴×
slower than the closed form for the same European price; see runtimes).

**Alternatives considered.**

| Alternative | Trade-off |
|---|---|
| PDE / Crank-Nicolson finite differences | Better convergence (O(Δt²)+O(ΔS²)) and smooth Greeks, but more machinery (grids, boundary conditions, ψ-shifts for dividends). A tree is the transparent minimum for early exercise; a PDE engine is the natural next project. |
| Longstaff-Schwartz MC | Needed for high-dimensional/path-dependent American products; for a 1-D vanilla it is strictly dominated by the tree (regression bias, noise). |
| Barone-Adesi-Whaley / Bjerksund-Stensland approximations | Fast analytic approximations with fixed, non-reducible error (~1e-2–1e-3); fine for screening, not for a reference engine. |

### 1.3 Black-76, options on forwards/futures

`C = e^{-rT} [F N(d1) − K N(d2)]` with `d1 = [ln(F/K) + σ²T/2]/(σ√T)`.

**Why include it.** Equity *index futures* options (and index options
marked off the forward) are quoted this way: the observable is the futures
price `F`, which already embeds financing and the dividend stream -
including discrete and uncertain dividends that the continuous-`q` BSM
form models only crudely. Mathematically Black-76 with
`F = S e^{(r−q)T}` reproduces BSM exactly (verified to 1e-10 in tests),
so it also serves as a change-of-variables consistency check.

### 1.4 Monte Carlo (exact GBM scheme), the extensible engine

`S_T = S exp((r − q − σ²/2)T + σ√T Z)`, sampled *exactly*, so the only
error is statistical, reported as a standard error and 95% CI
(`MCResult`). Variance reduction: antithetic variates and a
discounted-terminal-stock control variate (martingale mean `S e^{-qT}`),
together cutting the SE by about 5x at equal paths (0.0449 to 0.0089
at 100k paths, VALIDATION.md section 3). The control-variate coefficient
is fitted on the antithetic pair averages, the independent samples. Greeks by pathwise and
likelihood-ratio estimators with finite-difference (common-random-numbers)
fallback.

**Why include it.** For a vanilla under GBM, MC is the *worst* tool
(O(n^{-1/2}) vs closed form), and the comparison harness quantifies that. It earns its place because it is the only engine that
generalises to path-dependence, baskets, and any dynamics you can
simulate; and because an MC vs closed-form agreement test (within 3 SE) is
a powerful bug detector for both engines.

### 1.5 Implied volatility

The quote is mapped to normalised Black form `b = price / (e^{-rT} sqrt(F K))`
with `x = -|ln(F/K)|`. An in-the-money quote is converted to the
out-of-the-money option by parity, so the solver works on time value. The
equation `b(x, s) = beta` in the total vol `s = sigma sqrt(T)` is solved by
safeguarded Halley iteration on `1/ln b` below the inflection point
`s_c = sqrt(2|x|)`, on `ln b` above it, and on `ln(b_max - b)` near the
upper bound, from an asymptotic initial guess, with a relative stopping
rule. It takes at most 10 evaluations (2.6 on average over the test grid).
The same algorithm, constants and branch thresholds run in the C++ and
Rust engines. It is not Jackel's "Let's Be Rational": there is no rational
cubic initial guess, so it needs a few more iterations, and its accuracy
floor is 1e-12 relative rather than machine precision in the far tails.

### 1.6 Discrete delta-hedging simulator, model risk made visible

Pricing theory says a short option hedged continuously at the true vol has
zero P&L. The simulator shows what survives discretisation and
misspecification: P&L std ∝ 1/√N, and hedging at the wrong vol earns/pays
the gamma-weighted vol spread

    E[P&L] ≈ ∫₀ᵀ ½ (σ_hedge² − σ_realized²) S_t² Γ_hedge(S_t, t) dt

which the simulator reproduces path-by-path (simulated +1.996 vs theory
+1.987 on the reference setup). This is the quantitative content of
"volatility trading": the option is a position in realized-vs-implied vol.

---

## 2. Assumptions register

Each assumption states **what breaks if violated** and where the breakage
is demonstrated or tested.

| # | Assumption | What breaks if violated |
|---|---|---|
| A1 | Underlying follows GBM: lognormal returns, continuous paths, constant σ. | Real returns have fat tails, jumps and stochastic vol → a single BS vol cannot price all strikes: the market shows a skew/smile. Deep-OTM puts are systematically *underpriced* by ATM vol. Demonstrated in VALIDATION.md §"vol smile contradiction" using the synthetic skewed chain; short-dated gamma/jump risk appears in DESK_GUIDE.md earnings scenario. |
| A2 | Continuous, frictionless hedging is possible. | Discrete rebalancing leaves irreducible P&L noise: std ≈ √(π/4)·vega·σ/√N (Derman-Kamal; at N=4 per 3M ATM option this gives 1.76 against a simulated 1.60, on a premium of about 4.4; VALIDATION.md section 5). Transaction costs turn continuous hedging into a divergent-cost limit, with 5bp costs the mean P&L drops by ~$0.24 at N=128 (tested). |
| A3 | Constant risk-free rate r; borrowing = lending; no funding/collateral spread. | Rho hedging is mis-stated when the funding curve is not flat; discounting error grows with T. Negative r is *supported* (tested), but a stochastic-rates world (equity-rate correlation) breaks the deterministic-discounting factorisation for long-dated options. |
| A4 | Dividends are a continuous yield q. | Single stocks pay discrete dividends: the spot drops by the dividend on the ex-date. Continuous-q misprices short-dated options around ex-dates and distorts the American-call early-exercise decision (exercise clusters just before ex-date, which a continuous yield smears out). Quantified in VALIDATION.md §"dividend modeling error". Index options: acceptable; use Black-76 on the futures to absorb realized dividends entirely. |
| A5 | European exercise for BS/Black-76/MC engines. | Single-name listed options are American. Ignoring the exercise premium under-marks puts (e.g. 0.423 on the 1y ATM reference put, ~7% of value) and calls on high-dividend names. Handled by the CRR tree; premium ≥ 0 and call-parity with q=0 are tested. |
| A6 | No counterparty/liquidity/market-impact effects; mid-market prices. | Real quotes have bid/ask ~ vega·(vol spread); implied vol from a stale or crossed quote is garbage, `implied_vol` refuses prices outside no-arbitrage bounds (raises `ValueError`, tested) rather than returning a number. |
| A7 | Constant σ over the option's life (term structure flat). | Calendar-spread risk is invisible to a single-vol model; theta at a fixed vol misattributes P&L when the term structure rolls. Desks mark a full surface (DESK_GUIDE.md §vol-surface dependency). |
| A8 | Tree/MC discretisation parameters are "large enough". | CRR error oscillates odd/even and decays O(1/n): at n=10 the error on the reference call is 0.194 (~2% of premium), a naive coarse tree is a real mispricing. MC with few paths has SE larger than typical bid/ask. Both quantified with convergence tables (VALIDATION.md) and enforced by tests. |

## 3. When each tool is the right one (decision rule)

- **Quoting/risking European index options:** Black-76 on the forward.
- **Anything European under flat vol:** closed-form BS (speed + exact Greeks).
- **American single-name options:** CRR tree (500+ steps; premium from the
  same tree to cancel discretisation error).
- **Path-dependent/multi-asset/exotic dynamics:** Monte Carlo (with the
  variance-reduction toolkit and CI reporting shipped here).
- **Sizing model risk of any of the above:** the hedging simulator.

## 4. Phase 5 models (Round 3)

Each module's docstring gives the plain idea, the formal statement with its
reference (paper, section, equation) and a worked example. In short:

* **Leisen-Reimer** (1996): probabilities from a Peizer-Pratt inversion of
  `d1` and `d2`, so the tree centres on the strike. European error O(1/n^2)
  without the CRR odd/even wobble. Use it as the default tree.
* **Crank-Nicolson** in log-spot with two implicit half steps first
  (Rannacher 1984) and Brennan-Schwartz projection for American exercise.
  Second order in both grids; slower than ALO, but it extends to payoffs a
  boundary method cannot price.
* **Andersen-Lake-Offengeim** (2016): exercise boundary by the FP-B fixed
  point on Chebyshev nodes, Gauss-Legendre quadrature after removing the
  `1/sqrt(tau - u)` singularity. The fastest accurate American pricer
  here. Scope: put with `r > 0, q >= 0` (calls by put-call symmetry); no
  negative-rate double boundary.
* **Discrete dividends**: escrowed model (spot less the dividends' present
  value is lognormal). Roll-Geske-Whaley for one dividend on a call; an LR
  tree on the escrowed spot for several dividends or puts.
* **Term structures**: Merton (1973) averages; rates linear in `r(t) t`,
  vols linear in total variance, which rejects calendar arbitrage.
* **Bachelier**: normal model for quotes in normal vol.
* **SABR**: Hagan et al. (2002) and Obloj (2008) variants; West (2005)
  stable evaluation of `z/x(z)` near the money (series plus `log1p`). A
  Breeden-Litzenberger density check flags the expansion's known negative
  density at low strikes. The arbitrage-free SABR PDE is not implemented.
* **Higher-order Greeks**: Haug (2007) formulas; time derivatives are per
  year in calendar time, the same sign convention as `theta`.
* **Ladders and grids**: full revaluation, no Taylor terms; vega split
  linearly between neighbouring expiry buckets (a convention).
