# Methodology: Equity Portfolio Optimization and Risk Allocation

Pipeline: **Return estimation → Covariance modeling → Mean-variance
optimization → Efficient frontier → Risk parity → Sharpe/metrics →
Walk-forward backtesting** (package `eq_port`).

Everything in this project is organised around one central theme:

> **Markowitz optimization is an estimation-error amplifier.** The maths of
> mean-variance optimization (MVO) is trivial; the practice is dominated by
> the fact that its inputs, especially expected returns, cannot be
> estimated to useful precision. Every component here (James-Stein
> shrinkage, reverse optimization, Black-Litterman, Ledoit-Wolf, risk
> parity, long-only constraints) is a different answer to that single
> problem.

---

## 1. The estimation-error problem ("error maximization")

**Merton (1980): means are the hard part.** The standard error of a mean
return estimated over a horizon of `Y` years is `sigma / sqrt(Y)`:
*independent of sampling frequency*. Sampling daily instead of monthly
sharpens variance estimates (which improve with the number of
observations) but does nothing for the mean (which improves only with the
calendar span). With `sigma ≈ 20%` and one year of data the SE of the
annual mean is ≈ 20%: an order of magnitude larger than the cross-
sectional spread of true expected returns (our synthetic truth spans
3%–8%/yr; typical real risk premia differ by less).

**Why MVO makes it worse (Michaud 1989).** Unconstrained MVO weights are
`w ∝ Σ⁻¹(μ − rf)`: the optimizer takes *differences* of noisy means and
multiplies them by an inverse covariance that amplifies its smallest,
worst-estimated eigenvalues. Assets that got lucky in-sample (high
estimated mean, low estimated vol/correlation) receive extreme weights:
the optimizer loads up on exactly the estimation errors. Quantified in
`examples/run_pipeline.py` §2 on 200 synthetic 252-day windows (10
panels): the raw-mean unconstrained tangency portfolio is undefined
(`1'Σ⁻¹μ ≤ 0`) in 69 windows, averages 26.2x gross leverage where it
exists, and has a mean true Sharpe of 0.115 (95% CI 0.068 to 0.161),
against 0.314 for equal weight and 0.427 achievable with the true
moments.

## 2. Return estimators (`returns_est.py`)

| Estimator | Idea | Trade-off |
|---|---|---|
| Sample mean | MLE under iid | Unbiased but hopelessly noisy (above) |
| EWMA mean | Recency weighting | Reacts to drifting means; even noisier effective sample |
| **James-Stein** (isotropic, not Jorion's Bayes-Stein) | Shrink toward the grand mean with data-driven intensity `phi = clip((N−3)·σ̄²/T / Σᵢ(μᵢ−m̄)², 0, 1)` | Biased toward "all assets are equal", but dominates the sample mean in total squared error for N ≥ 4 (Stein 1956); kills exactly the cross-sectional noise MVO trades on |
| **Reverse optimization** | Discard sample means entirely: `π = δ Σ w_mkt` are the returns that make the market portfolio optimal | No time-series noise at all; you inherit the market's view (CAPM equilibrium) |
| **Black-Litterman** | Bayesian blend of the equilibrium prior with explicit views: `μ_BL = π + τΣPᵀ(PτΣPᵀ+Ω)⁻¹(Q−Pπ)`, posterior covariance `Σ + M` with `M = τΣ − τΣPᵀ(PτΣPᵀ+Ω)⁻¹PτΣ` | Weights change only where you *have* a view; requires choosing `τ` and `Ω` (default `Ω = diag(PτΣPᵀ)`, He-Litterman) |

Why this stack rather than alternatives: (a) *factor-model expected
returns* (Fama-French style) need factor premia estimates that suffer the
same Merton problem; (b) *full Bayesian predictive* (e.g. Jorion's
Bayes-Stein with predictive covariance inflation) adds machinery for a
second-order effect. James-Stein + BL capture 90% of the practical
benefit with closed forms we can test to machine precision (no-view
posterior = prior exactly; `Ω → 0` view honoured exactly: our
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
improves (cond 16.2 to 14.0 on one synthetic 252x8 window; finite instead of ∞ when
`T < N`); shrinkage is a no-op when the sample *is* the target (single
asset ⇒ `δ = 0`).

Alternatives considered:

- **EWMA / RiskMetrics** (implemented, `λ = 0.94`): reacts fast to vol
  regimes, but the effective sample (~33 days) makes correlations noisy
  and the matrix rank-deficient for N > effective T; kept as a
  diagnostic, not the optimizer input.
- **Single-factor (market) model** (implemented): guaranteed PSD, very
  low variance, but biased: it forces all comovement through one factor
  and misses sector blocks (our synthetic truth has three).
- **Ledoit-Wolf to identity / diagonal**: simpler targets, but constant
  correlation is the natural target for a one-market equity universe and
  keeps the (well-estimated) variances untouched.
- **PSD repair** (eigenvalue clipping at a relative floor) is applied
  after any estimator before inversion; it is a projection, not an
  estimator: used for numerical hygiene only.

## 4. Optimizers (`mvo.py`, `risk_parity.py`)

**Closed forms wherever they exist**: unconstrained min-variance
`Σ⁻¹1/(1'Σ⁻¹1)`, tangency `Σ⁻¹μₑ/(1'Σ⁻¹μₑ)`, and the analytic frontier
`w(m) = λΣ⁻¹1 + γΣ⁻¹μ` (two-fund theorem: frontier weights are affine in
the target return: tested as an exact identity).

**Constrained problems: `scipy.optimize` SLSQP, no cvxpy.**
The problems here are small (N ~ 10-100) and smooth. The module rescales
the covariance to O(1) internally. SLSQP reproduces the closed forms to
about 1e-7 in weights (tested). Min-variance, target-return and
target-risk problems are convex, so a KKT point is the global optimum.

**Maximum Sharpe as a convex QP.** The ratio (w'μ - rf)/√(w'Σw) is not
convex in w. With excess returns e and the substitution y = κw, κ > 0
(Cornuejols and Tutuncu 2007), the long-only or box-constrained problem
becomes `min y'Σy` subject to `e'y = 1`, `Σy = κ`,
`loᵢκ ≤ yᵢ ≤ hiᵢκ`, `κ ≥ 0`. That is a convex QP, solved with analytic
gradients. `max_sharpe_qp` then checks the first-order conditions of
the original ratio problem and raises if the relative KKT residual
exceeds 1e-6. The substitution needs a feasible portfolio with e'w > 0.
When none exists (long-only: max μ ≤ rf), every feasible Sharpe ratio
is non-positive and maximising it would select the most volatile
corner. The function returns the long-only minimum-variance portfolio
instead and sets `fallback=True`. Backtest strategies count these
fallbacks.

**Risk parity**: equal risk contribution (ERC) portfolio solved by
cyclical coordinate descent on `½y'Σy − Σᵢ bᵢ ln yᵢ` (Griveau-Billion,
Richard & Roncalli 2013; existence/uniqueness by Spinu 2013), from
scratch, with the exact coordinate update (positive root of a scalar
quadratic). ERC needs *no mean estimates at all* and no matrix inversion:
that is its point: it is the maximally estimation-robust allocation
that still uses correlation structure (unlike naive inverse-vol, to which
it provably collapses when all correlations are equal: tested).
Because unlevered ERC of a low-vol book runs below equity-like risk,
real risk-parity funds lever to a vol target: `vol_target_overlay`
implements `L = σ_target/σ(w)` with an optional leverage cap (the desk
control; see DESK_GUIDE on the March-2020 deleveraging spiral).

## 5. Backtesting (`backtest.py`) and metrics (`metrics.py`)

Walk-forward engine: estimation window `[t−W, t−1]`, trade at `t`
(strictly no lookahead: enforced by construction and by a cheat-
detection test where a return spike on the rebalance day must be
uncapturable), weights drift between rebalances, two-sided turnover
`Σ|w_new − w_drift|` charged at `cost_bps`. Exactness of the ledger is
tested against a hand-computed two-rebalance scenario.

Metrics: geometric annualised return, annualised vol, Sharpe with **Lo
(2002)** autocorrelation-adjusted annualisation and standard error
(autocorrelated returns overstate sqrt-time Sharpe), Sortino,
max drawdown, Calmar, Choueifaty-Coignard diversification ratio (≥ 1,
tested), effective N `1/Σw²`, realized risk contributions.

The Lo adjustment sums sample autocorrelations up to lag L = ⌊T^(1/3)⌋
with Bartlett weights 1 - k/(L+1) (Newey-West). Summing all 251 lags at
q = 252 is mostly noise: on iid data the adjusted-to-raw ratio then
ranges from about 0.76 to 1.54. `sharpe_difference_test` gives the
Jobson-Korkie statistic with Memmel's (2003) correction for comparing
two strategies on the same dates. It assumes iid normal returns.

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
   targets and risk budgets wrong intramonth: exactly when it matters.
   Mitigant: EWMA diagnostics, shorter rebalance, vol-target overlay.
3. **Quadratic utility / mean-variance sufficiency (no higher
   moments).** Breaks: MVO is indifferent to skew/kurtosis; a short-vol
   asset with negative skew looks ideal to MVO. Crisis fat tails show up
   in MaxDD (0.31 to 0.45 in the seed-1 synthetic crisis) far beyond what a normal with the
   same vol implies.
4. **Frictionless trading except proportional costs; unlimited
   liquidity.** Breaks: linear `bps × turnover` understates cost of
   trading in stress (impact is convex, spreads widen exactly in
   crises). Raw tangency's 6.8x mean annual turnover in the synthetic
   race would cost more in practice than the linear ledger shows.
5. **Leverage is available at the risk-free rate and never recalled.**
   Breaks: the RP vol-target overlay assumes financing at rf; in a
   funding squeeze (Mar-2020) leverage is cut at the worst prices: see
   DESK_GUIDE scenario.
6. **Long-only box constraints reflect the true mandate.** Breaks:
   constraints act as implicit shrinkage (Jagannathan-Ma 2003): helpful
   against noise (raw tangency true Sharpe improves from 0.115 to
   0.279 when long-only is imposed, synthetic study) but they also cap
   the achievable frontier. No long-only portfolio returns more than
   max(μ), and the long-only frontier vol is at least the unconstrained
   vol at every attainable return.
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
Lo (2002) "The Statistics of Sharpe Ratios"; Newey & West (1987);
Jobson & Korkie (1981); Memmel (2003); Cornuejols & Tutuncu (2007); Maillard, Roncalli &
Teïletche (2010); Spinu (2013); Griveau-Billion, Richard & Roncalli
(2013); DeMiguel, Garlappi & Uppal (2009) "1/N".

## Round 3 additions

- **HRP** (`hrp.py`) follows Lopez de Prado (2016). It uses single linkage on sqrt((1 - rho)/2), quasi-diagonal ordering and recursive bisection with inverse-variance clusters. It needs no matrix inversion.
- **Resampled MVO** (`robust_mvo.resampled_weights`) follows Michaud (1998). It averages the mean-variance solutions over inputs re-estimated from simulated samples.
- **Robust MVO** (`robust_mvo.robust_weights`) uses an ellipsoidal uncertainty set on the mean (Ceria and Stubbs 2006). When Omega = Sigma/T, the result is a classic portfolio at a higher risk aversion.
- **Cost and turnover QP** (`cost_qp.rebalance`) has proportional and quadratic costs, an optional turnover cap, an optional budget and box bounds (Boyd et al. 2017). `no_trade_band_solution` is the closed form for a diagonal Sigma.
- **Factor risk model** (`factor_model`) runs OLS time-series regressions, giving Sigma = B Sigma_f B' + D. The Euler factor and asset contributions add up exactly to the totals (Grinold and Kahn ch. 3).
- **Tracking error** (`tracking_error`) has the closed-form maximum-IR active weights at a target TE (Grinold and Kahn ch. 14, Roll 1992). A long-only or box-bounded SLSQP version is also provided.
- **Sharpe-difference tests** (`sharpe_tests`) follow Ledoit and Wolf (2008): a HAC delta method, with their prewhitened quadratic-spectral kernel (`kernel="qs_pw"`, Andrews and Monahan 1992) or a Newey-West Bartlett kernel at fixed lags (the default, kept for compatibility), and a studentised circular block bootstrap whose draws use the block-sum covariance (section 3.2). `studentize=False` gives the basic centred bootstrap.
