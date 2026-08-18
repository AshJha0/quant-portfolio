# Methodology — why these models

This project has two layers with different time scales and different model
choices. This document answers, for each: **why this model**, compared with
at least two alternatives, and lists the **assumptions register**.

---

## 1. Alpha layer: cross-sectional factor long-short

### Why cross-sectional L/S (and not the alternatives)?

The alpha layer ranks a universe of stocks each day on point-in-time
features (momentum 12-1 and 6-1, 1-month reversal, realised vol, MA
crossover, RSI, abnormal volume), combines them into a composite z-score,
and holds the top decile long / bottom decile short, dollar neutral.

**Alternative 1 — time-series (directional) models per stock.** Forecasting
each stock's *level* of return requires beating the market's own drift and
exposes the book to beta. Cross-sectional ranking differences out the market
and all common factors that affect the ranking symmetrically: the L/S book
is hedged by construction (net exposure = 0, tested to 1e-12), and the
statistical object — the daily rank IC — has ~n_stocks independent-ish
observations per day rather than one. A daily IC of 0.03 with 100 names is
detectable in months; a directional edge of the same size takes decades.

**Alternative 2 — full covariance mean-variance optimisation (Markowitz /
Black-Litterman).** MVO converts forecasts into weights optimally *if* the
covariance matrix and expected returns are known; in practice it amplifies
estimation error and produces unstable, concentrated books that need heavy
regularisation. Decile L/S with per-name caps is the robust special case:
equal weights within buckets are the minimax choice under ranking
uncertainty, turnover control is explicit (signal-band freezing), and the
P&L attribution is transparent. Project 07 (portfolio-optimization) covers
the MVO machinery; here the focus is the signal -> execution chain.

**Alternative 3 — ML rank models (gradient boosting / neural rankers).**
Strictly more flexible, but they multiply the researcher-degrees-of-freedom
problem this project explicitly guards against (see deflated Sharpe below),
and they obscure the PIT audit trail. The pipeline is deliberately linear
and auditable; an ML signal could be dropped into `signals.py` and would
inherit the same PIT tests, cost model and evaluation stack.

### Signal evaluation choices

- **Spearman rank IC** rather than Pearson: invariant to monotone feature
  transforms and robust to outliers; matches `scipy.stats.spearmanr`
  (test-verified, including ties).
- **Newey-West t-stats** on IC series: daily ICs are autocorrelated (and
  multi-day-horizon ICs mechanically so, from overlap); the naive i.i.d. SE
  understates uncertainty. Bartlett-kernel NW is the standard correction
  (tested: NW SE > naive by >30% on an AR(1) IC series with rho = 0.6).
- **Deflated Sharpe ratio (Bailey & Lopez de Prado 2014)** against
  multiple testing: if N signal variants were tried, the best backtest
  Sharpe must beat the *expected maximum* of N zero-skill Sharpes,
  `SR* = sqrt(V)[(1-gamma)z_{1-1/N} + gamma z_{1-1/(Ne)}]`, not zero.
  DSR = PSR evaluated at SR*; at N=1, SR* = 0 and DSR = PSR (identity,
  tested). The multiple-testing problem is the dominant failure mode of
  backtests: with 45 tried variants, pure noise is *expected* to deliver an
  annualised Sharpe of ~1.15 on this sample length — which is why the demo
  strategy's SR of 1.35 deflates from PSR 0.995 to DSR 0.65.

### Transaction cost model (daily layer)

`cost = sum_i |dw_i| (linear_bps + k * sigma_i * sqrt(AUM |dw_i| / ADV$_i))`

Linear term = spread + fees. The impact term follows the **empirical
square-root law** — measured meta-analyses (Almgren et al. 2005; Toth et
al. 2011) consistently find cost ≈ Y·sigma·sqrt(Q/V) with Y of order 0.5-1,
across venues, horizons and asset classes. Alternatives: a purely linear
impact overstates cost for large trades and understates it for small ones;
a 3/5-power law fits marginally better in some studies but adds a free
parameter with no qualitative change. The same law drives the capacity
curve: drag grows like sqrt(AUM), so net Sharpe declines monotonically in
AUM (tested).

---

## 2. Execution layer: Almgren-Chriss with linear/sqrt impact

### The model

Sell/buy X shares over N slices, holdings x_0 = X, ..., x_N = 0:

- E[cost] = 0.5·gamma·X² + epsilon·Σ|n_k| + (eta_tilde/tau)·Σ n_k²,
  eta_tilde = eta − gamma·tau/2
- V[cost] = sigma²·tau·Σ x_k²

Minimising E + lambda·V yields the discrete recursion
x_{j-1} + x_{j+1} = 2cosh(kappa·tau)·x_j with closed form
**x_j = X·sinh(kappa(T−t_j))/sinh(kappa·T)**, where
cosh(kappa·tau) = 1 + kappa_tilde²tau²/2, kappa_tilde² = lambda·sigma²/eta_tilde.
lambda→0 gives TWAP exactly; lambda→∞ front-loads into the first slice
(both tested).

### Why AC (and not the alternatives)?

**Alternative 1 — Obizhaeva-Wang (2013) resilience models.** OW models the
limit-order book as a block that is consumed and *refills at a finite rate*
(exponential resilience). Its optimal strategy is characteristic: large
discrete trades at the open and close plus a constant trickle in between.
It is the right model when trading at horizons comparable to the book's
resilience time (minutes) and when queue dynamics matter. Trade-off: it
needs the resilience rate — a parameter that is hard to estimate and
unstable — and its bucket-level predictions at our half-hour granularity
are nearly indistinguishable from AC's. AC's two parameters (eta, gamma)
map directly onto measurable regression coefficients from TCA data.

**Alternative 2 — proprietary/empirical schedule optimisers** (dynamic
programming on fitted nonlinear impact, e.g. power-law temporary impact
with p ≈ 0.6, signal-driven adaptive schedules). These beat AC in
production at the margin but are opaque, data-hungry, and lose the
closed-form efficient frontier that makes the urgency conversation with a
PM quantitative ("+3.7 bps expected buys a 35% cut in cost volatility").
AC is the industry baseline precisely because the frontier is analytic.

**Alternative 3 — pure static benchmarks (VWAP/TWAP/POV).** These are
special cases, not competitors: TWAP is AC at lambda = 0; VWAP reshapes
TWAP by the volume profile; POV caps participation. AC subsumes them and
prices the risk dimension they ignore.

**Why linear permanent + square-root temporary?** Huberman-Stanzl (2004):
permanent impact must be *linear* in size or the model admits price
manipulation (round-trip profits). Temporary impact is where the empirical
square-root law lives. The simulator therefore implements
`perm = perm_coef·sigma·(q/V)·mid` (linear) and
`temp = temp_coef·sigma·sqrt(q/V)·mid` (sqrt); the AC optimiser uses the
linear-temporary approximation of the classic paper. This deliberate
mismatch (optimise in a tractable model, evaluate in a richer one) mirrors
desk reality and is quantified in VALIDATION.md.

### Intraday simulator choices

- **U-shaped volume profile** `p_j ∝ 1 + c(2u_j−1)²`: equity volume is
  reliably U-shaped (open/close auctions and index flow); the profile sums
  to 1 exactly and ends > middle (tested).
- **LOB-lite fill model**: market orders always fill at mid + side·(half
  spread + temporary impact). No queue position, no partial fills.
- Per-bucket noise sigma_b = sigma_daily/sqrt(n_buckets); permanent impact
  is applied *after* the bucket's fill (the trade moves the price for
  everyone after you).

---

## 3. Assumptions register

Each assumption states *what breaks if violated*.

1. **Impact stationarity.** eta, gamma, and the sqrt-law coefficient are
   constants. Violated in stress (costs spike exactly when you must trade):
   realised costs exceed modelled, AC schedules are too aggressive, and the
   capacity curve is optimistic. Mitigation: re-fit from TCA weekly, stress
   multipliers (DESK_GUIDE).
2. **No adverse selection / no alpha in the market during execution.** The
   simulator's mid is a martingale apart from our own impact. Violated when
   the parent order's signal is correlated with short-term drift (usual for
   momentum entries): arrival-slippage is understated; delay cost in the
   Perold decomposition partially captures it.
3. **Fill certainty.** Market orders always fill at the modelled price; no
   halts, no queues, no odd lots. Violated in fast markets: opportunity
   cost appears (the decomposition's third term exists for exactly this).
4. **Daily-close alpha vs intraday execution separation.** The alpha layer
   assumes trades happen at the close at which the signal is formed; the
   execution layer schedules them across the *next* day's buckets. The
   cross-term (intraday alpha decay while executing) is delay cost in TCA
   and is not fed back into the daily backtest.
5. **Linear permanent impact, sqrt temporary.** If true temporary impact is
   more concave (p < 0.5), large child orders are cheaper than modelled and
   optimal schedules concentrate more; if permanent impact is nonlinear the
   no-manipulation property fails and E[cost] is trajectory-dependent
   beyond 0.5·gamma·X².
6. **Universe integrity of the panel**: no survivorship handling needed
   (synthetic panel has no delistings unless injected); real data requires
   PIT universe membership. The backtester supports names leaving the
   universe (NaN prices; tested) but the demo panel does not exercise
   corporate actions.
7. **Gaussian-ish daily returns for DSR/PSR moments.** The PSR formula uses
   sample skew/kurtosis to third order; extreme fat tails make DSR
   optimistic. Non-normality is partially handled (skew/kurt enter the
   formula, tested), not fully.
8. **Costs do not feed back into signals or prices in the daily backtest**
   (our own trading does not move the close). Reasonable below ~5% ADV
   participation; the capacity curve is exactly the tool that flags when
   this assumption dies.
