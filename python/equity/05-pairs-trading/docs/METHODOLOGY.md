# Methodology: equity statistical pairs trading

Pipeline: pair candidates, correlation or SSD screen, Engle-Granger
cointegration, Ornstein-Uhlenbeck (OU) spread model, z-score signals,
backtest with costs, walk-forward validation, research statistics.

Conventions: daily closes in dollars, 252 trading days a year, OU `kappa`
per trading day (half-life `ln 2 / kappa` in trading days), costs in basis
points of per-leg traded notional, borrow and financing annualised ACT/252.

Status: every result in this project comes from a synthetic generator that
plants cointegrated pairs. The edge is planted. No real-data result exists
yet; that belongs to the research platform in `research/`.

---

## 1. Model choice

The core model is Engle-Granger cointegration with an OU spread.

**Distance method (Gatev, Goetzmann, Rouwenhorst SSD).** It ranks pairs by
the sum of squared differences of normalised prices. It has no null
hypothesis, so it cannot separate a common factor from luck, and it gives no
half-life or stationary variance. It is available as a pre-screen
(`walk_forward_portfolio(screen="ssd")`) and the pipeline reports it beside
the correlation screen.

**Johansen.** Johansen estimates the cointegration rank of an N-variable
VAR. For two legs the rank is 0 or 1 and Engle-Granger answers the same
question with a hedge ratio a trader can use. Engle-Granger depends on which
leg is the regressand. This project fixes the orientation by ticker order
(the candidate list is sorted), never by the result. Johansen is not
implemented.

**Learned selection (clustering, embeddings).** Such methods can propose
candidates. A proposed pair must still pass a stationarity test, which is
the part this project implements.

**OU against a rolling z-score.** A rolling z-score (`zscore_rolling`)
cannot say how fast the spread reverts. The OU fit gives kappa, hence a
half-life, a time stop at k half-lives and a stationary variance for the
z-score. OU is fitted by OLS on the exact AR(1) discretisation and by
numerical MLE, and the two must agree.

**Critical values.** The step-2 ADF statistic on Engle-Granger residuals is
compared with MacKinnon's Engle-Granger (N=2) response surface, not the
plain ADF (N=1) surface. OLS picks the hedge ratio that minimises residual
variance, which pushes the residuals towards stationarity and shifts the
null distribution left. At 5% and T=500 the plain ADF value is -2.87 and the
N=2 value is -3.35. Using the N=1 values on N=2 statistics raises the
false-discovery rate (tested in `test_spurious_rejection_rate_close_to_size`).

**Correlation screen on returns.** Price levels of independent random walks
have spurious sample correlation (Granger and Newbold). The screen uses
returns and is only a cheap filter. Correlated random walks with return
correlation near 0.92 still fail the ADF stage.

**Hedge ratio.** The cointegrating regression has an intercept by default
because the long-run spread level between two stocks need not be zero. A
split changes the scale of one series, which changes beta on unadjusted
prices; use split-adjusted prices. With an intercept in step 1, the step-2
ADF regression runs without a constant against the N=2 constant surface.

Three hedge-ratio methods are wired into the walk-forward
(`walk_forward_portfolio(hedge=...)`):

| hedge | spread at t | z-score at t |
|---|---|---|
| `static` | formation OLS alpha and beta, frozen | `(s_t - mu) / (sigma / sqrt(2 kappa))` with frozen OU parameters |
| `rolling` | rolling OLS over `hedge_window` days ending at t | rolling z-score over `hedge_window` days |
| `rls` | recursive least squares, forgetting factor `1 - 1/hedge_window`, no intercept | rolling z-score over `hedge_window` days |

RLS with a forgetting factor is the Kalman filter for a random-walk
coefficient with the noise ratio tied to the forgetting factor rather than
estimated, so it is called RLS here, not Kalman. The full filter, with the
observation and state noise estimated by maximum likelihood, is
`eq_pairs.kalman.kalman_hedge_ratio`; it is not one of the three pipeline
hedges. With a short memory an intercept is nearly collinear with the
price level, so RLS runs without one and the rolling z-score removes the
level. All three use the same Engle-Granger and half-life gates on the
formation window. The pipeline logs all three as trials (VALIDATION §6).

---

## 2. Execution convention

Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost.

The data has closes only, so the t+1 fill is a point on the path from the
close of t to the close of t+1. f = 1 is the next close. A small f stands in
for a next-open fill. `ExecutionConfig(mode="next", fill_fraction=f)`
selects it. f = 0 is rejected: a fill at the close the signal was computed
on must use `ExecutionConfig(mode="close_auction", auction_cost_bp=a)`,
which charges `a` bp per leg on traded notional (default 10 bp).

The same convention, with the same parameter names, is used by the other
alpha and execution projects in this repository.

---

## 3. Cost model

| cost | formula | pipeline value |
|---|---|---|
| commission | `cost_bps` x traded notional, per leg | 5 bp |
| fixed slippage | `slippage_bps` x traded notional, per leg | 2 bp |
| market impact | `impact_coef x sigma_leg x sqrt(notional / adv_dollars)` x traded notional, per leg | Y = 0.7, ADV $20mm per name |
| borrow | `borrow_bps` / 252 per night on short market value | 50 bp a year |
| financing | `financing_bps` / 252 per night on long market value | 50 bp a year |
| short rebate | `short_rebate_bps` / 252 per night on short proceeds, earned | 0 |
| close auction | `auction_cost_bp` x traded notional, per leg, close-auction mode only | 10 bp |

`sigma_leg` is the leg's daily return volatility known before the fill: the
formation-window estimate in the walk-forward, a trailing 60-day estimate
otherwise. The synthetic data has no volume, so ADV is an assumption, and
the pipeline reports the walk-forward Sharpe at Y = 0, 0.3, 0.7 and 1.0 and
at two position sizes. The synthetic prices are total-return prices, so a
dividend on the short leg is already inside the price P&L. Hard-to-borrow
fees, recalls and locate failures are not modelled.

---

## 4. Assumptions register

| # | Assumption | What breaks if it is violated |
|---|---|---|
| A1 | The spread stays stationary out of sample. | The book shorts a diverging spread. The regime-break case quantifies it (VALIDATION §8). Mitigants: walk-forward re-testing, stop-loss, time stop, re-entry arming. |
| A2 | The hedge ratio is stable over formation and trading. | The book carries residual directional exposure. Mitigants: re-fit every window; the rolling and RLS hedges. |
| A3 | The spread is OU with Gaussian shocks. | Half-life and z-scores are wrong; fat tails make stops fire more often than modelled. |
| A4 | Borrow is available at a stable fee. | Fee spikes or recalls force the short leg closed at a bad price. Not modelled. |
| A5 | Fills happen at the modelled t+1 price with the cost model of §3. | The impact law and ADV are assumptions on synthetic data; capacity numbers depend on them. |
| A6 | Daily closes are synchronous and clean; gaps of up to 5 days may be forward-filled. | Longer gaps raise `ValueError` in `align_pair`. On real data `align_pair` trims trailing rows after a leg stops trading, which drops a delisted leg's final return instead of booking it. |
| A7 | A decision at the close of t is filled no earlier than the modelled t+1 price, unless the close-auction cost is paid. | Same-close fills without that cost overstate short-horizon P&L. Tested in `TestExecutionConvention`. |
| A8 | Pair P&Ls are roughly independent. | Crowded books unwind together (August 2007). Portfolio Sharpe then overstates capacity. |
| A9 | The estimated half-life is a fair guide to convergence time. | OLS on the AR(1) with a fitted mean biases phi down (Kendall), so kappa is biased up and estimated half-lives are too short. The bias grows as reversion slows and as samples shrink. A time stop at k estimated half-lives is therefore too tight, not too loose. Tested over 200 seeds in `test_ols_kappa_is_biased_upward_over_many_seeds`. |
| A10 | A mean-reverting OU fit implies a tradeable spread. | It does not. On a pure random walk OLS returns b slightly below 1, so `mean_reverting` is True with a spurious half-life of hundreds of days. The funnel gates on Engle-Granger first and caps the half-life (tested in `test_an_ou_fit_alone_cannot_reject_a_random_walk`). |

---

## 5. Mathematics

**Engle-Granger step 1.** OLS of `y_t = alpha + beta x_t + u_t`. Under
cointegration beta-hat converges at rate T.

**Step 2, ADF.** On the residuals:
`d u_t = rho u_{t-1} + sum_i phi_i d u_{t-i} + e_t` without a constant. The
statistic is the t-value of rho-hat. The lag order is chosen by AIC over
p = 0..p_max (Schwert rule) with all candidates on a common sample, which
reproduces `statsmodels.adfuller(autolag="AIC")`. Critical values:
`crit(T) = b0 + b1/T + b2/T^2 + b3/T^3` with MacKinnon (2010) coefficients.

**OU.** `ds = kappa (mu - s) dt + sigma dW`, exact discretisation
`s_{t+1} = mu (1 - b) + b s_t + e_t`, `b = exp(-kappa)`,
`Var e = sigma^2 (1 - b^2) / (2 kappa)`. Half-life `ln 2 / kappa`,
stationary sd `sigma / sqrt(2 kappa)`. b >= 1 is flagged as not mean
reverting (half-life infinite, do not trade). b <= 0 raises.

**Signals.** Enter at |z| >= 2, exit through the exit band (z = 0), hard
stop at |z| >= 4, time stop at ceil(3 x half-life) bars. After any exit the
pair re-arms only once |z| is back inside the entry band.

**Sizing.** Dollar-neutral (gross/2 per leg at the fill price) or
beta-neutral (shares in ratio 1 : beta).

**Accounting.** On day t with fill price p_fill, holdings carried in earn
`q_old (p_fill - p_{t-1})` and holdings after the fill earn
`q_new (p_t - p_fill)`. The identity
`net = gross - commission - slippage - impact - borrow - financing - auction`
holds to the penny (tested). Walk-forward: formation `[i, i+F)` fits
everything, trading `[i+F, i+F+T)` uses frozen parameters, windows never
overlap, positions are force-closed at window ends, and the trailing days
that do not fill a whole window form one shorter last window
(`allow_partial=True` in the pipeline).

**Capital.** Returns are net dollar P&L divided by a fixed capital base of
gross per pair times the number of pair slots (6 x $1mm = $6mm in the
pipeline). The same base is used in sample and walk-forward.

---

## 6. Research statistics

`eq_pairs.research_stats` holds the statistics every reported number goes
through. It re-exports `qp_core.stats`, which the other alpha projects use
too.

- **Sharpe SE.** Lo (2002) GMM delta method. The per-period Sharpe
  `s = mu / sigma` has influence function
  `psi_t = (x_t - mu)/sigma - (s / (2 sigma^2)) ((x_t - mu)^2 - sigma^2)`.
  Its variance is the Newey-West (Bartlett) long-run variance of psi over n,
  truncated at `floor(4 (n/100)^(2/9))` lags. With no lags and Gaussian
  returns it is `(1 + s^2/2) / n`.
- **Deflated Sharpe ratio** (Bailey and Lopez de Prado, 2014). The
  probability that the true Sharpe exceeds the expected maximum of N null
  trials, `sqrt(V) ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))`, with V
  the variance of the logged trial Sharpe ratios and skew and kurtosis
  corrections. N comes from a `TrialLog` that records every configuration
  evaluated.
- **PBO by CSCV** (Bailey, Borwein, Lopez de Prado, Zhu, 2017). Split the
  sample into S blocks, take every half as in-sample, find the in-sample
  best configuration and its out-of-sample rank; PBO is the share of splits
  where that rank is at or below the median.
- **Block bootstrap.** Circular blocks of length `ceil(n^(1/3))` for a
  confidence interval on one path.
- **Seed distributions.** Mean, sd and a Student-t 95% interval of a
  statistic over independent synthetic seeds.

---

## 7. References

- Engle and Granger (1987), Co-integration and error correction.
- MacKinnon (1996, 2010), critical-value response surfaces.
- Kendall (1954), Note on bias in the estimation of autocorrelation.
- Gatev, Goetzmann and Rouwenhorst (2006), Pairs trading.
- Lo (2002), The statistics of Sharpe ratios.
- Bailey and Lopez de Prado (2014), The deflated Sharpe ratio.
- Bailey, Borwein, Lopez de Prado and Zhu (2017), The probability of
  backtest overfitting.
- Almgren, Thum, Hauptmann and Li (2005), Direct estimation of equity
  market impact.
- Avellaneda and Lee (2010), Statistical arbitrage in the US equities market.
- Khandani and Lo (2007), What happened to the quants in August 2007?
