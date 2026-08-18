# Methodology — Equity Portfolio Optimization & Risk Allocation

Pipeline: **Return estimation → Covariance modeling → Mean-variance
optimization → Efficient frontier → Risk parity → Sharpe/metrics →
Walk-forward backtesting** (package `eq_port`).

Everything in this project is organised around one central theme:

> **Markowitz optimization is an estimation-error amplifier.** The maths of
> mean-variance optimization (MVO) is trivial; the practice is dominated by
> the fact that its inputs — especially expected returns — cannot be
> estimated to useful precision. Every component here (James-Stein
> shrinkage, reverse optimization, Black-Litterman, Ledoit-Wolf, risk
> parity, long-only constraints) is a different answer to that single
> problem.

---

## 1. The estimation-error problem ("error maximization")

**Merton (1980): means are the hard part.** The standard error of a mean
return estimated over a horizon of `Y` years is `sigma / sqrt(Y)` —
*independent of sampling frequency*. Sampling daily instead of monthly
sharpens variance estimates (which improve with the number of
observations) but does nothing for the mean (which improves only with the
calendar span). With `sigma ≈ 20%` and one year of data the SE of the
annual mean is ≈ 20% — an order of magnitude larger than the cross-
sectional spread of true expected returns (our synthetic truth spans
3%–8%/yr; typical real risk premia differ by less).

**Why MVO makes it worse (Michaud 1989).** Unconstrained MVO weights are
`w ∝ Σ⁻¹(μ − rf)`: the optimizer takes *differences* of noisy means and
multiplies them by an inverse covariance that amplifies its smallest,
worst-estimated eigenvalues. Assets that got lucky in-sample (high
estimated mean, low estimated vol/correlation) receive extreme weights —
the optimizer loads up on exactly the estimation errors. Quantified in
`examples/run_pipeline.py` §2: across 20 independent 252-day estimation
windows, the raw-mean unconstrained tangency portfolio averaged **16.2x
gross leverage** (max 58.7x), failed outright in 8/20 windows
(`1'Σ⁻¹μ ≤ 0`), and delivered a mean *true* Sharpe of **0.14** versus
**0.33 for naive equal weight** and **0.45 achievable** with the true
moments. That is the whole story of this project in three numbers.

## 2. Return estimators (`returns_est.py`)

| Estimator | Idea | Trade-off |
|---|---|---|
| Sample mean | MLE under iid | Unbiased but hopelessly noisy (above) |
| EWMA mean | Recency weighting | Reacts to drifting means; even noisier effective sample |
| **James-Stein** | Shrink toward the grand mean with data-driven intensity `phi = clip((N−3)·σ̄²/T / Σᵢ(μᵢ−m̄)², 0, 1)` | Biased toward "all assets are equal", but dominates the sample mean in total squared error for N ≥ 4 (Stein 1956); kills exactly the cross-sectional noise MVO trades on |
| **Reverse optimization** | Discard sample means entirely: `π = δ Σ w_mkt` are the returns that make the market portfolio optimal | No time-series noise at all; you inherit the market's view (CAPM equilibrium) |
| **Black-Litterman** | Bayesian blend of the equilibrium prior with explicit views: `μ_BL = π + τΣPᵀ(PτΣPᵀ+Ω)⁻¹(Q−Pπ)`, posterior covariance `Σ + M` with `M = τΣ − τΣPᵀ(PτΣPᵀ+Ω)⁻¹PτΣ` | Weights change only where you *have* a view; requires choosing `τ` and `Ω` (default `Ω = diag(PτΣPᵀ)`, He-Litterman) |

Why this stack rather than alternatives: (a) *factor-model expected
returns* (Fama-French style) need factor premia estimates that suffer the
same Merton problem; (b) *full Bayesian predictive* (e.g. Jorion's
Bayes-Stein with predictive covariance inflation) adds machinery for a
second-order effect. James-Stein + BL capture 90% of the practical
benefit with closed forms we can test to machine precision (no-view
posterior = prior exactly; `Ω → 0` view honoured exactly — our
implementation never inverts `Ω`, so perfect confidence is handled
without limits).

## 3. Covariance estimators (`covariance.py`)

The sample covariance has `N(N+1)/2` parameters. With `T` not ≫ `N` it is
ill-conditioned; for `T ≤ N` it is singular. MVO uses `Σ⁻¹`, which blows
up the smallest (worst-estimated) eigenvalues.

**Chosen: Ledoit-Wolf (2004) shrinkage to constant correlation, from
scratch.** `Σ_LW = δF + (1−δ)S` where `F` keeps the sample variances and
replaces every correlation by the average correlation, and the intensity
`δ = clip((π̂−ρ̂)/γ̂/T, 0, 1)` is the closed-form asymptotic minimiser of
Frobenius loss (paper's estimators `π̂`, `ρ̂`, `γ̂` implemented verbatim;
see docstring). Properties we verify: `δ ∈ [0,1]` always; conditioning
improves (cond 20.8 → 16.9 on a 252×8 window; finite instead of ∞ when
`T < N`); shrinkage is a no-op when the sample *is* the target (single
asset ⇒ `δ = 0`).

Alternatives considered:

- **EWMA / RiskMetrics** (implemented, `λ = 0.94`): reacts fast to vol
  regimes, but the effective sample (~33 days) makes correlations noisy
  and the matrix rank-deficient for N > effective T; kept as a
  diagnostic, not the optimizer input.
- **Single-factor (market) model** (implemented): guaranteed PSD, very
  low variance, but biased — it forces all comovement through one factor
  and misses sector blocks (our synthetic truth has three).
- **Ledoit-Wolf to identity / diagonal**: simpler targets, but constant
  correlation is the natural target for a one-market equity universe and
  keeps the (well-estimated) variances untouched.
- **PSD repair** (eigenvalue clipping at a relative floor) is applied
  after any estimator before inversion; it is a projection, not an
  estimator — used for numerical hygiene only.

## 4. Optimizers (`mvo.py`, `risk_parity.py`)

**Closed forms wherever they exist**: unconstrained min-variance
`Σ⁻¹1/(1'Σ⁻¹1)`, tangency `Σ⁻¹μₑ/(1'Σ⁻¹μₑ)`, and the analytic frontier
`w(m) = λΣ⁻¹1 + γΣ⁻¹μ` (two-fund theorem: frontier weights are affine in
the target return — tested as an exact identity).

**Constrained problems: `scipy.optimize` SLSQP, deliberately no cvxpy.**
The QPs here are small (N ~ 10–100), smooth, and — after rescaling the
covariance to O(1), which the module does internally — well-conditioned;
SLSQP reproduces the closed forms to ~1e-7 in weights (tested) and solves
each rebalance in milliseconds. A conic-modeling stack (cvxpy + ECOS/OSQP)
would add a heavyweight dependency, its own solver-tolerance quirks, and
no capability we need: long-only boxes, budget, target-return equality
and variance-cap inequality are all native SLSQP constraints. The trade-
off documented honestly: SLSQP gives local KKT points, not certificates —
acceptable because our objectives are convex QPs (global by convexity)
and every solver output is cross-checked against closed forms or
perturbation tests in the suite.

**Risk parity**: equal risk contribution (ERC) portfolio solved by
cyclical coordinate descent on `½y'Σy − Σᵢ bᵢ ln yᵢ` (Griveau-Billion,
Richard & Roncalli 2013; existence/uniqueness by Spinu 2013), from
scratch, with the exact coordinate update (positive root of a scalar
quadratic). ERC needs *no mean estimates at all* and no matrix inversion
— that is its point: it is the maximally estimation-robust allocation
that still uses correlation structure (unlike naive inverse-vol, to which
it provably collapses when all correlations are equal — tested).
Because unlevered ERC of a low-vol book runs below equity-like risk,
real risk-parity funds lever to a vol target: `vol_target_overlay`
implements `L = σ_target/σ(w)` with an optional leverage cap (the desk
control; see DESK_GUIDE on the March-2020 deleveraging spiral).

## 5. Backtesting (`backtest.py`) and metrics (`metrics.py`)

Walk-forward engine: estimation window `[t−W, t−1]`, trade at `t`
(strictly no lookahead — enforced by construction and by a cheat-
detection test where a return spike on the rebalance day must be
uncapturable), weights drift between rebalances, two-sided turnover
`Σ|w_new − w_drift|` charged at `cost_bps`. Exactness of the ledger is
tested against a hand-computed two-rebalance scenario.

Metrics: geometric annualised return, annualised vol, Sharpe with **Lo
(2002)** autocorrelation-adjusted annualisation and standard error
(smoothed/autocorrelated returns overstate sqrt-time Sharpe), Sortino,
max drawdown, Calmar, Choueifaty-Coignard diversification ratio (≥ 1,
tested), effective N `1/Σw²`, realized risk contributions.

---

## 6. Assumptions register

Each assumption states *what breaks if violated*.

1. **Returns are iid draws from a fixed (μ, Σ) within an estimation
   window.** Breaks: under regime shifts the window mixes regimes and
   both moments are biased; the crisis subperiod in
   `examples/run_pipeline.py` §7 shows realized correlations jumping
   0.47 → 0.87, at which point every calm-window covariance understates
   portfolio risk (all strategies' crisis vol is 2–3x their calm vol).
2. **Covariance is stable over the holding period (estimate at t, hold
   for a month).** Breaks: vol/correlation spikes make ex-ante vol
   targets and risk budgets wrong intramonth — exactly when it matters.
   Mitigant: EWMA diagnostics, shorter rebalance, vol-target overlay.
3. **Quadratic utility / mean-variance sufficiency (no higher
   moments).** Breaks: MVO is indifferent to skew/kurtosis; a short-vol
   asset with negative skew looks ideal to MVO. Crisis fat tails show up
   in MaxDD (0.31–0.47 in the race) far beyond what a normal with the
   same vol implies.
4. **Frictionless trading except proportional costs; unlimited
   liquidity.** Breaks: linear `bps × turnover` understates cost of
   trading in stress (impact is convex, spreads widen exactly in
   crises); the raw tangency's 8.3x annual turnover would be far more
   expensive in practice than the 71bp/period ledger suggests.
5. **Leverage is available at the risk-free rate and never recalled.**
   Breaks: the RP vol-target overlay assumes financing at rf; in a
   funding squeeze (Mar-2020) leverage is cut at the worst prices — see
   DESK_GUIDE scenario.
6. **Long-only box constraints reflect the true mandate.** Breaks:
   constraints act as implicit shrinkage (Jagannathan-Ma 2003) — helpful
   against noise (raw tangency true-Sharpe improves 0.14 → 0.30 when
   long-only is imposed) but they also cap the achievable frontier
   (long-only frontier vol ≥ unconstrained at every return level, §3
   table) and concentrate portfolios at the top end (max weight 0.75 at
   the frontier tip).
7. **The market portfolio is mean-variance efficient (for reverse
   optimization / BL prior).** Breaks: if the market is not efficient
   the implied returns π inherit its inefficiency; BL posteriors are
   anchored to a wrong prior and views must be strong to move it.
8. **Estimation windows contain enough data for second moments
   (T > N preferred).** Breaks: `T ≤ N` makes the sample covariance
   singular; Ledoit-Wolf still returns an invertible matrix (tested with
   T=5, N=8) but the estimate leans almost entirely on the target.

## References

Markowitz (1952); Merton (1980); Michaud (1989); Jorion (1986);
Black & Litterman (1992); He & Litterman (1999); Ledoit & Wolf (2004)
"Honey, I Shrunk the Sample Covariance Matrix"; Jagannathan & Ma (2003);
Lo (2002) "The Statistics of Sharpe Ratios"; Maillard, Roncalli &
Teïletche (2010); Spinu (2013); Griveau-Billion, Richard & Roncalli
(2013); DeMiguel, Garlappi & Uppal (2009) "1/N".
