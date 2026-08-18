# Methodology — FX Portfolio Optimization & Currency Risk Allocation

## 1. The problem

Allocate risk across currencies and across the three canonical FX styles
(carry, momentum, value), size the book against carry's crash tail, and set
currency hedge ratios for an international portfolio — all reported in the
investor's base currency (GBP option for a London desk).

Conventions (repo-wide): pairs quoted BASE/QUOTE with USD as quote
(``AUD`` column = AUDUSD = USD per 1 AUD); deposit rates continuously
compounded, annualised; daily accrual ACT/252; daily log returns.

## 2. Currency total returns

Holding currency *i* financed in base currency *b* earns, per day,

```
total_t = log(S_t/S_{t-1}) + (i_{t-1} − i_b,{t-1}) · dt
```

The carry leg uses **previous-close rates** so no same-day rate information
enters the return (no lookahead), and the decomposition
``total = spot + carry`` is exact by construction (tested to 0 error).
Uncovered interest parity (UIP) says the spot leg should on average offset
the carry leg; empirically it does not — the **forward premium puzzle**
(Fama 1984): high-yield currencies do not depreciate enough, so carry earns
a premium. We take that premium as given, with the standard honesty caveat:
part of the measured premium may be a **peso problem** — compensation for
rare crashes that a finite sample under-represents. Our synthetic generator
builds exactly that structure: no offsetting drift, plus a crash mechanism
proportional to the rate differential.

## 3. Style portfolios — why style-based allocation

Direct MVO over 12 currencies needs 12 mean estimates — the dominant source
of error in any optimization (Merton 1980: means are an order of magnitude
harder to estimate than variances; Michaud 1989: MVO is an
"estimation-error maximizer"). Compressing the cross-section into three
long-standing, economically-motivated styles:

- **CARRY**: rank currencies by deposit-rate differential vs the funding
  currency. Long high-yielders, short funders.
- **MOMENTUM**: 12-1 spot momentum, ``log(S_{t-21}/S_{t-252})``, skipping
  the last month to avoid short-horizon reversal.
- **VALUE**: real-exchange-rate deviation from a supplied PPP anchor,
  ``log(PPP/S)`` — long undervalued, short overvalued.

Each style is a **dollar-neutral** rank portfolio: demeaned cross-sectional
ranks scaled to a gross-leverage budget (Σw = 0, Σ|w| = gross). Rank
weights, not signal-proportional weights, because ranks are robust to
signal-scale outliers (a 15% TRY differential would otherwise dominate the
book). Signals formed at *t* trade at *t+1* (one-day lag, enforced in code).

Alternatives considered:
- *Direct currency-level MVO* — rejected as primary (estimation error;
  kept as a supported mode with James-Stein mean shrinkage, intensity 0.67
  on the 12-currency panel in the pipeline run).
- *Signal-proportional weights* — rejected for outlier sensitivity;
  rank weights cost a little efficiency for a lot of robustness.
- *Optimized style construction (max-Sharpe within style)* — rejected:
  compounds estimation error inside and across styles.

## 4. Covariance menu — why these four

| Estimator | Use case | Trade-off |
|---|---|---|
| Sample | T ≫ N diagnostics | Noisy, ill-conditioned for T ≲ 5N |
| EWMA (λ=0.94) | Fast regime tracking, risk monitors | No cross-sectional structure; effective sample ~33 days |
| **Ledoit-Wolf (2004)**, from scratch | Default for optimization | Biased toward identity, but minimax-safe; intensity δ ∈ [0,1] analytic |
| **Risk-on/off one-factor** | Stress logic, interpretation | Misses idiosyncratic correlation (e.g. EUR-CHF); imposes the FX-specific structure |

LW shrinks the sample covariance toward a scaled identity with analytically
optimal intensity δ = min(b²,d²)/d² — implemented from first principles and
tested for δ ∈ [0,1] and improved conditioning. The one-factor model
estimates the first principal component, orients it so it correlates
positively with the average currency return vs USD (dollar-down = risk-on),
and regresses each currency on it: recovered loadings have the classic
signs — AUD/NZD/EM positive, JPY/CHF negative (tested). PSD repair by
eigenvalue clipping (Higham projection; optional strictly-positive floor for
pegged currencies) guards every optimizer input.

## 5. Allocation

**Mean-variance.** Closed forms (min-var, tangency, two-fund frontier, and
the sum-to-zero dollar-neutral maximizer natural for long-short FX) are
implemented and validated to 1e-12; SLSQP counterparts add the FX-native
constraint set — net budget (0 for long-short, 1 for a fully-collateralised
long-only basket vs USD), gross-leverage cap Σ|w| ≤ G (solved smoothly via
the w = p − q split), and per-currency boxes. SLSQP is cross-validated
against the closed forms.

**Risk parity (ERC).** Equalising Euler risk contributions
RC_i = w_i(Σw)_i / w'Σw avoids mean estimates entirely — the right default
when you trust your covariance more than your means. Solved by cyclical
coordinate descent on the log-barrier formulation (Griveau-Billion et al.
2013); each coordinate update is an exact scalar quadratic, convergence to
1e-12. Applied both across styles and across currencies, with exact ex-ante
vol targeting.

**CVaR-constrained sizing (the FX-specific step).** Variance is symmetric;
carry's risk is not. Mean-variance sees carry's premium and modest vol and
loads up — precisely because its skew lives outside the variance. We
implement the **Rockafellar-Uryasev (2000)** linear programming formulation
over historical scenarios:

```
CVaR_α(w) = min_z  z + E[(−r'w − z)⁺]/(1−α)
```

Both "min CVaR" and "max μ'w s.t. CVaR_α ≤ c" are LPs in (w⁺, w⁻, z, u)
solved with `scipy.optimize.linprog` (HiGHS). At the optimum the RU
objective equals the empirical CVaR of the solution (tested to 1e-9), and
on toy data the LP matches exhaustive grid search. Positive homogeneity
gives the one-line closed form for single-sleeve sizing:
s\* = min(leverage cap, CVaR budget / CVaR(sleeve)).

Alternatives: variance-only sizing (rejected — blind to skew, this is the
point of the module); parametric Cornish-Fisher CVaR (rejected — fragile
beyond small skew/kurtosis; historical simulation is transparent and
LP-exact); full mean-CVaR frontier (available via `return_floor`).

## 6. Optimal currency hedging

For a base investor with unhedged portfolio return r_u and currency
exposures x, hedging fraction h_i of each exposure gives
r(h) = r_u − Σ h_i x_i r_fx,i. Minimising variance over hedge notionals
H = h∘x is an OLS projection:

```
H* = Cov(r_fx)⁻¹ Cov(r_fx, r_u),   h*_i = H*_i / x_i
```

Verified against brute-force numerical minimisation to 1e-6. If local
returns are uncorrelated with FX, h\* = 1 (full hedge) — recovered exactly
in tests. With safe havens negatively correlated to risk assets
(JPY, CHF), h\* < 1 and often < 0: **full hedging is not variance-optimal**
(Campbell-Serfaty-de Medeiros-Viceira 2010). Forward carry cost shifts the
hedged *mean*, not the variance, so it enters as a separate mean adjustment
in the decision, not in h\*.

## 7. Backtesting and base currency

Walk-forward: the allocator sees history up to and including the rebalance
date; weights earn P&L from the next day. Ledger splits spot P&L, carry
accrual, and costs (quoted in pips, converted via ``pips × pip_size / spot``
to bps of notional, charged on turnover). Weights are treated as reset to
target at each rebalance; intra-month drift of long-short FX weights is
second-order and documented as a simplification.

Base-currency conversion adds the old base's total return (in the new base)
to every asset's log total return. Because the added term is common across
assets, a **dollar-neutral book's log-return series is exactly invariant**
to the base choice (Σw = 0 kills the common term) — tested to 1e-14.
For arithmetic returns the invariance is only approximate (cross terms
r_i·c of order vol²); this is the documented, exact-conditions version of
the folklore claim "long-short FX P&L doesn't care about your base ccy".
Net-long books shift by the base currency's own return — the pipeline shows
the EW book's Sharpe moving 0.87 (USD) → 0.74 (GBP).

## 8. Assumptions register

| # | Assumption | What breaks if violated |
|---|---|---|
| 1 | UIP fails on average (carry premium exists) | If UIP holds, carry's mean is 0 and the carry sleeve is pure crash risk with no compensation; sizing should be 0. The premium may partly be a peso problem — treat backtested carry Sharpe as an upper bound. |
| 2 | Carry crash risk is *understated by vol* (negative skew, fat left tail) | If returns were Gaussian, vol targeting would suffice and the CVaR machinery is redundant. All sizing here assumes tails matter; CVaR estimates themselves need enough tail scenarios (≥ ~2 years daily). |
| 3 | Deposit-rate differentials are persistent (monthly rebalance is enough for carry) | Sudden policy moves (2015 SNB) change the signal intraday; monthly rebalance holds a stale book through the event. |
| 4 | One-day implementation lag is achievable at quoted pip costs | In crisis liquidity (2016 GBP flash crash) neither the lag nor the cost assumption holds; slippage models needed for EM. |
| 5 | Covariance is estimable from a trailing window (504d) and roughly stationary within it | Correlation regime flips (risk-on/off inversions) make trailing Σ wrong exactly when it matters; the one-factor model's stress logic is the mitigant. |
| 6 | PPP anchor is observable and slowly varying | Value signal inherits any anchor mis-measurement; PPP misvaluations can persist for a decade (documented failure mode). |
| 7 | Log-return arithmetic (portfolio return = Σ w·log-returns) | Exact only to first order; the base-invariance identity is exact in this convention, approximate in arithmetic returns. Error is O(vol²/2) per day. |
| 8 | Costs scale linearly with turnover at fixed bps | Ignores market impact and spread widening in stress — precisely when carry unwinds force turnover. Treat crisis-period backtest costs as understated. |
| 9 | Synthetic calibration (crash ~1.3/yr tied to rate differential, factor structure) approximates FX stylised facts | Conclusions about *relative* sizing (CVaR vs vol) transfer; absolute Sharpes do not. Never quote the synthetic Sharpes as achievable. |

## 9. References

Fama (1984), *Forward and spot exchange rates*; Rockafellar & Uryasev
(2000), *Optimization of conditional value-at-risk*; Ledoit & Wolf (2004),
*A well-conditioned estimator for large-dimensional covariance matrices*;
Maillard, Roncalli & Teïletche (2010), *The properties of equally weighted
risk contribution portfolios*; Griveau-Billion, Richard & Roncalli (2013),
*A fast algorithm for computing high-dimensional risk parity portfolios*;
Campbell, Serfaty-de Medeiros & Viceira (2010), *Global currency hedging*;
Menkhoff, Sarno, Schmeling & Schrimpf (2012), *Carry trades and global
foreign exchange volatility*; Lo (2002), *The statistics of Sharpe ratios*.
