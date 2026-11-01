# Methodology: Equity VaR and Expected Shortfall Engine

This document says which models the engine uses, why, and what each one
assumes. Section 2 states the conventions every function follows. They are
shared with the FX engine and the compiled ports (spec S1-S13 of the ws-var
workstream).

## 1. The problem

Estimate, each day, how much a book of equities, index futures and options
can lose over 1 day (and 10 days) at 95 % or 99 %, and the average loss
beyond that point (Expected Shortfall).

## 2. Conventions

| # | Convention | Statement |
|---|---|---|
| S1 | alpha | `alpha` is the tail probability everywhere (0.01 is 99 % VaR). Any public function refuses alpha outside (0, 0.5) with `ValueError` ("alpha is the tail probability"). |
| S1 | Sign | VaR and ES are positive numbers for losses. P&L is in currency, losses negative. |
| S2 | Quantile | Losses `L = -pnl` sorted descending, `k = n alpha`, `m = ceil(k - 1e-9)` clamped to `[1, n]`. `VaR = L(m)`, an order statistic. No interpolation. |
| S2 | ES | `ES = [sum_{i<m} L(i) + (k - (m - 1)) L(m)] / k`, the exact integral of the empirical quantile function over the worst `alpha` of mass (Acerbi-Tasche). `ES >= VaR` always. |
| S2 | Weighted | BRW: weights normalised, sorted with the losses (stable), `m` = first `j` with cumulative weight `>= alpha - 1e-12`, `VaR = L(m)`, ES with the fractional weight `alpha - W_{m-1}` on `L(m)`. |
| S3 | Kupiec | `kupiec_pof(n_obs=, n_exceptions=, alpha=)`. `LR = -2 ln[(1-a)^(n-x) a^x] + 2 ln[(1-p)^(n-x) p^x]`, `p = x/n`, `0 ln 0 = 0`, chi2(1) p-value. |
| S4 | Basel | `F(x) = P(X <= x)`, `X ~ Binomial(n_obs, alpha)`, by pmf recursion. Green if `F < 0.95`, yellow if `F < 0.9999`, red otherwise. Yellow plus factor read at the smallest 250-day count `x'` in 5..9 with `F250(x') >= F(x)`, else 9. Any window, any alpha. |
| S5 | ES backtest | Acerbi-Szekely `Z2 = sum_t pnl_t I_t / (T alpha ES_t) + 1`. Negative means ES understated. One-sided p-value `P(Z2_sim <= Z2_obs)` simulated with the same VaR and ES forecasts. |
| S6 | Cornish-Fisher | Valid iff the CF map is increasing on `[-z_a, z_a]`, `z_a = Phi^{-1}(1 - alpha)`. Checked in closed form. Non-finite moments raise. |
| S7 | Cholesky | Relative symmetry check (1e-10). Jitter from `1e-12` to at most `1e-6` times the mean variance, with a `RuntimeWarning`. Above the cap it raises unless `repair="clip"` or `repair="higham"` is requested. |
| S8 | EWMA | Zero-mean RiskMetrics recursion `s2_t = lam s2_{t-1} + (1 - lam) r_{t-1}^2`, seed `s2_0 = mean(r^2)` over the first `min(30, n)` observations. |
| S9 | FHS | Filter each factor with its own EWMA vol, rescale to the one-step-ahead vol, fully revalue the book. |
| S10 | Monte Carlo | Price factors move by log returns (`S exp(r)`), vol factors by an absolute change floored at zero. No path raises. One simulation gives VaR, ES and the bootstrap SE. |
| S11 | Empty book | A `Portfolio` with no positions raises "book has no positions". |
| S12 | Inputs | NaN, +inf, -inf, empty input and out-of-domain values raise `ValueError` at every public entry point. |

Other conventions: price-factor scenarios given to `Portfolio.pnl` are simple
returns, implied-vol scenarios are absolute changes in decimal vol. Day
count ACT/365F, option rates and dividends continuously compounded, vol on
log returns, annualised.

## 3. Why three VaR families coexist

No single estimator dominates, so the engine has three and the disagreement
between them is a diagnostic (`examples/run_pipeline.py`, VALIDATION.md §3).

| | Historical simulation | Parametric (var-covar) | Monte Carlo |
|---|---|---|---|
| Distributional assumption | none (empirical) | normal, t or Cornish-Fisher | any simulable factor model |
| Non-linear positions | full revaluation of scenarios | linear (delta) only | full revaluation per path |
| Fat tails | captured if they are in the window | only via t or CF | captured if the factor model has them |
| Responds to current vol | plain HS no, FHS yes | yes with EWMA covariance | yes with EWMA covariance |
| Estimation noise at 99 % | high (2-3 tail points per year) | low | falls with path count |
| Main failure | window myopia | wrong tail shape | wrong model |

Defaults and reasons:

- **Filtered historical simulation** (Barone-Adesi, Giannopoulos and
  Vosper; Hull and White). Each factor return is standardised by its
  one-step-ahead EWMA vol and rescaled to tomorrow's forecast. The book is
  then fully revalued under the rescaled factor moves
  (`factor_filtered_historical_var`). Filtering factors rather than P&L
  keeps option convexity and does not mix position changes with market
  vol. The older P&L-level `filtered_historical_var` is deprecated.
  Age-weighted BRW is a middle ground: it reweights old days but does not
  rescale them.
- **Parametric VaR** uses `sigma_p = sqrt(w' Sigma w)` with dollar exposures
  `w` and sample or EWMA covariance. Normal quantiles are the baseline.
  Variance-matched Student-t and Cornish-Fisher are tail corrections. CF is
  refused outside its validity region (§6).
- **Monte Carlo VaR** simulates log factor returns (multivariate normal, or
  multivariate t as `Z / sqrt(W / nu)` with one common mixing variable, so
  every factor pair has the same df and tail dependence) and fully revalues
  the book per path. Full revaluation is vectorised over scenarios.

## 4. Portfolio representation and P&L

`Portfolio` maps positions onto named `RiskFactor`s (stock prices, index
level, implied vol). Linear positions are revalued exactly. Options are
revalued two ways:

- full revaluation: Black-Scholes at the shocked spot and vol, expiry fixed;
- delta-gamma-vega: `dV = Delta dS + 0.5 Gamma dS^2 + vega dsigma`, with
  diagonal gamma only (no vanna, volga or cross-gamma).

`Portfolio.approximation_error` measures the gap. On the demo book it is
$93k (37 % of the scenario P&L) in the 2020 replay (synthetic demo book).

## 5. Expected Shortfall

ES measures tail severity and is subadditive, which VaR is not. The classic
two-bond counterexample is in the tests: two independent positions each
with 95 % VaR of -5 (a gain) combine to a VaR of 95, while ES stays
subadditive on the same book. FRTB uses 97.5 % ES. For normal P&L,
`ES_97.5 = 2.338 sigma` against `VaR_99 = 2.326 sigma` (unit-tested). ES at
a given alpha averages few tail points, so its estimation error is larger
than the VaR's (`es_standard_error_bootstrap`).

The ES backtest is Acerbi-Szekely Z2 with a simulated p-value. The default
null draws `pnl_t ~ Normal(0, sigma_t)` with `sigma_t = ES_t alpha / phi(z)`.
If the model is not normal, pass `null_sampler` so the null is the model's
own distribution.

## 6. Cornish-Fisher and its validity region

`z_cf = z + (z^2 - 1) S/6 + (z^3 - 3z) K/24 - (2z^3 - 5z) S^2/36`, with skew
`S` and excess kurtosis `K`. The map is a quantile function only where it is
increasing. Its derivative is the quadratic
`g'(z) = A z^2 + B z + C`, `A = K/8 - S^2/6`, `B = S/3`,
`C = 1 - K/8 + 5 S^2/36`. `cornish_fisher_valid(S, K, alpha)` takes the exact
minimum of `g'` on `[-z_a, z_a]` (both endpoints, plus the vertex when
`A > 0` and it lies inside) and requires it to be positive.

The check depends on alpha because only the part of the curve up to the
requested quantile matters. Example: `(S, K) = (0.122, -0.427)` has its
minimum derivative, -9.15e-4, at `z = -4`. It is valid at alpha 0.01 and
0.001. The old fixed interval `|z| <= 4` rejected it.
`(-0.0105, 8.0001)` has its minimum, about -2.5e-7, at `z = 0.00175`, so it
is invalid at every alpha.

## 7. Covariance factorisation

`cholesky(cov)` tries plain Cholesky, then diagonal jitter up to
`1e-6 x mean(diag)` with a `RuntimeWarning`. A matrix that needs more is
materially indefinite and the call raises, naming the most negative
eigenvalue. Explicit repairs:

- `repair="clip"`: clip eigenvalues at `1e-10 x mean(diag)`, rebuild, rescale
  to the original variances.
- `repair="higham"`: Higham (2002) nearest correlation matrix by alternating
  projections with Dykstra's correction (tolerance 1e-12 on successive
  iterates, at most 1000 iterations), rescaled by the original vols.

Both warn. For `A = [[1, .9, .9], [.9, 1, -.9], [.9, -.9, 1]]` (eigenvalues
-0.8, 1.9, 1.9) both give 0.5 off the diagonal with the signs of `A`.

`ewma_covariance` still seeds with the sample covariance and iterates over
the whole sample. It has not moved to the S8 seed window because the
compiled ports pin its current output.

## 8. Multi-day horizons

- **sqrt(t) scaling** is exact for i.i.d. zero-drift returns. Under
  volatility clustering the error has no fixed sign. Scaling a conditional
  VaR measured in a calm state understates risk because variance
  mean-reverts upward (tested on a simulated GARCH path started at a quarter
  of the long-run variance). Scaling an unconditional VaR from a fat-tailed
  sample overstates the direct 10-day quantile, because aggregation thins
  the tail: on a 20,000-day synthetic GARCH sample the excess kurtosis falls
  from 4.76 (1 day) to 1.94 (10 days). Drift adds bias in either direction.
- **Overlapping 10-day windows** are model-free but serially dependent: the
  effective sample is about n/10. The demo pipeline gives $172.9k (sqrt(t))
  against $144.5k (overlapping) on synthetic data.

Monte Carlo has no multi-day path model and options keep expiry fixed, so
10-day option VaR ignores theta.

## 9. Backtesting stack

- Kupiec POF (S3) for the exception rate. At 250 days and 99 % its
  asymptotic test rejects a correct model 9.5 % of the time at the nominal
  5 % (VALIDATION.md).
- Christoffersen independence and conditional coverage
  (`LR_cc = LR_uc + LR_ind`, chi2(2)). `LR_uc` uses T observations and
  `LR_ind` T - 1 transitions, the usual approximation.
- Basel traffic light (S4) for any window and alpha.
- Acerbi-Szekely Z2 (S5).

## 10. Assumptions register

| # | Assumption | Where | What breaks if violated |
|---|---|---|---|
| A1 | Factor returns capture all P&L drivers; futures basis carry ignored | portfolio.py | Single-name gaps hit P&L but not VaR |
| A2 | Option expiry fixed over the horizon (no theta) | portfolio.py | 10-day option VaR is overstated for long gamma and understated for short gamma |
| A3 | One parallel implied-vol factor per underlier | portfolio.py | Skew and term-structure moves are not measured |
| A4 | Daily P&L mean is about 0 | all VaR modules | With drift, VaR is biased by `-mu h`; pass `mean` |
| A5 | i.i.d. returns in the window (plain HS, sample covariance) | historical, parametric | Exceptions cluster in vol regimes; use FHS or EWMA |
| A6 | EWMA lam = 0.94 and a 30-day seed window | historical | Slow decay lags regime shifts; the first 30 forecasts use the seed window |
| A7 | sqrt(t) scaling for 10-day VaR | historical, parametric | No fixed sign of error under clustering (§8) |
| A8 | Student-t df (default 6), never estimated | parametric, MC | Wrong df gives the wrong tail; df <= 2 raises |
| A9 | Covariance roughly stationary | parametric, MC | Correlations rise in crashes; stress tests cover this |
| A10 | MC sampling error acceptable | monte_carlo_var | Reported: 100k paths give a bootstrap SE of about 0.8 % of the 99 % VaR on the synthetic demo book |
| A11 | Historical stress shocks approximate the named episodes | stress_testing | Scenario P&L is indicative; multi-day episodes are applied as one-day shocks |
| A12 | Small samples | historical | At `n alpha < 1` the rank is clamped to 1 and the VaR is the sample maximum loss |

## 11. Not implemented

Risk decomposition (component or incremental VaR), FRTB (liquidity
horizons, stressed calibration, P&L attribution), EVT tails, variance
reduction in Monte Carlo, covariance shrinkage and exact or simulated
Kupiec and Christoffersen p-values.

## 12. References

- Kupiec (1995), *Techniques for Verifying the Accuracy of Risk Measurement Models*.
- Christoffersen (1998), *Evaluating Interval Forecasts*.
- Boudoukh, Richardson, Whitelaw (1998), *The Best of Both Worlds*.
- Barone-Adesi, Giannopoulos, Vosper (1999); Hull and White (1998).
- Artzner, Delbaen, Eber, Heath (1999), *Coherent Measures of Risk*.
- Acerbi and Tasche (2002), *On the coherence of Expected Shortfall*.
- Acerbi and Szekely (2014), *Backtesting Expected Shortfall*.
- Higham (2002), *Computing the nearest correlation matrix*.
- BCBS (1996), *Supervisory framework for the use of backtesting*; BCBS
  (2019), *Minimum capital requirements for market risk*.
