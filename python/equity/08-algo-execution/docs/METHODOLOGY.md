# Methodology: why these models

The project has two layers with different time scales. For each layer this
note gives the model, at least two alternatives, and an assumptions
register. All results in this project are on synthetic data; no real-data
result exists yet.

---

## 1. Alpha layer: cross-sectional factor long-short

### Model

Each day the universe is ranked on point-in-time features (momentum 12-1
and 6-1, 1-month reversal, realised vol, MA crossover, RSI, abnormal
volume). Features are combined into a composite z-score with expanding,
lagged IC weights. The book holds the top decile long and the bottom decile
short, dollar neutral, with a per-name cap.

### Alternatives

1. Time-series (directional) forecasts per stock. These carry market beta
   and have one observation per day instead of one per name. A daily rank
   IC of 0.03 on 100 names is detectable in months; a directional edge of
   the same size takes decades.
2. Full mean-variance optimisation (Markowitz, Black-Litterman). Optimal if
   the inputs are known, but it amplifies estimation error and gives
   concentrated books. Decile long-short with caps is the robust special
   case. Project 07 covers the optimiser.
3. Machine-learned rankers. More flexible, with more researcher degrees of
   freedom and a harder point-in-time audit. A model could be dropped into
   `signals.py` and would inherit the same tests and costs.

### Synthetic data

`generate_daily_panel` draws
`r_{i,t} = 0.0008 z(mom 12-1)_{i,t-1} - 0.0006 z(ret 21d)_{i,t-1} + sigma_{i,t} eps`,
with AR(1) log-vol. The features the book combines are the generator's own
regressors. Every alpha statistic is therefore a recovery test, not a
discovery. The planted daily IC of about 0.03 is far above real 12-1
momentum, whose monthly IC is of order 0.02-0.05.

### Execution convention

Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost.

With only closes available, f stands for how much of the next day's move
has happened before the fill. f = 0.5 is a stand-in for a next-open or
early-session fill; f = 1 fills at the next close. In return terms the old
weights earn f r_{t+1} and the new weights earn (1 + r_{t+1}) / (1 + f
r_{t+1}) - 1. The trade and its cost are booked on day t+1. The
close-auction mode fills at p_t and charges `auction_cost_bp` (default 10
bp) per unit of turnover on top of the normal costs. On synthetic data the
auction variant has the lower net Sharpe (section 4 of VALIDATION).

### Signal evaluation

- Spearman rank IC, matching `scipy.stats.spearmanr`.
- Newey-West t-stats on IC series, which are autocorrelated.
- `research_stats.py`, a re-export of `qp_core.stats` (shared by every
  strategy project):
  - Lo (2002) standard error of the Sharpe ratio by the GMM delta method,
    with a Newey-West (Bartlett) long-run variance truncated at
    floor(4 (n/100)^(2/9)) lags.
  - Deflated Sharpe ratio (Bailey and Lopez de Prado 2014) at the trial
    count in a `TrialLog`. The pipeline logs every configuration it
    backtests: seven single features, the combined signal and two execution
    variants, ten trials.
  - Probability of backtest overfitting by CSCV (Bailey, Borwein, Lopez de
    Prado, Zhu 2017) across the signal configurations.
  - Circular block-bootstrap CIs, block length ceil(n^(1/3)).
  - Mean, sd and Student-t CI across seeds. Every headline in the README is
    a 20-seed distribution.

### Transaction costs (daily layer)

`cost = sum_i |dw_i| (linear_bps + Y sigma_i sqrt(AUM |dw_i| / ADV$_i))`
plus a borrow fee `borrow_bps_annual / 252` per day on the short gross.

The linear term covers spread and fees. The impact term is the empirical
square-root law: Almgren et al. (2005) and Toth et al. (2011) find cost of
about Y sigma sqrt(Q/V) with Y of order 0.5-1. The default is Y = 0.7.
`sigma_i` and `ADV$_i` are trailing 63-day and 21-day estimates known at
the decision close. Capacity is measured by re-running the full backtest at
each AUM for Y = 0.3, 0.7 and 1.0 (`capacity_backtest`). The closed-form
`capacity_curve` remains as a planning approximation.

---

## 2. Execution layer: Almgren-Chriss with linear and square-root impact

### Model

Trade X shares over N slices with holdings x_0 = X, ..., x_N = 0:

- E[cost] = 0.5 gamma X^2 + epsilon sum |n_k| + (eta_tilde / tau) sum n_k^2,
  eta_tilde = eta - gamma tau / 2
- V[cost] = sigma^2 tau sum x_k^2

Minimising E + lambda V gives x_{j-1} + x_{j+1} = 2 cosh(kappa tau) x_j and
x_j = X sinh(kappa (T - t_j)) / sinh(kappa T). lambda = 0 is TWAP; large
lambda front-loads into the first slice. The ratio of sinh terms is
evaluated in exponential form so it stays finite for kappa T > 710.

### Alternatives

1. Obizhaeva-Wang (2013) resilience models. Right when the horizon is
   comparable to the book's refill time. They need a resilience rate that
   is hard to estimate, and at half-hour buckets they differ little from AC.
2. Empirical schedule optimisers on fitted nonlinear impact. Better at the
   margin, opaque, and without an analytic frontier.
3. Static schedules (TWAP, VWAP, POV). Special cases: TWAP is AC at
   lambda = 0.

### Simulator

- Volume profile `p_j proportional to 1 + c (2 u_j - 1)^2`, U-shaped.
- Permanent impact `perm = perm_coef sigma (q / V_day) mid`, linear, as
  Huberman-Stanzl (2004) require for no price manipulation.
- Temporary impact `temp = temp_coef sigma sqrt(q_j / (n_buckets v_j)) mid`
  where v_j is the bucket's market volume. With a flat profile this is
  sqrt(q / V_day). A child order in a thin bucket costs more than the same
  order in a heavy bucket.
- Market orders always fill at mid plus side times (half spread +
  temporary impact). No queues, no partial fills.
- Per-bucket noise sigma_daily / sqrt(n_buckets).

The AC optimiser uses linear temporary impact and is evaluated in the
square-root simulator. Schedules are ranked by the expected IS on the
zero-noise simulator and by the Monte Carlo std (`evaluate_schedules`).

---

## 3. Assumptions register

Each entry states what breaks if it is false.

1. Planted alpha. The generator contains the signal. On real data the
   alpha numbers do not carry over.
2. Impact stationarity. Y, eta and gamma are constants. In stress they rise
   and the capacity table is optimistic.
3. No adverse selection during execution. The simulated mid is a
   martingale apart from our own impact. If the parent's signal predicts
   intraday drift, slippage against arrival is understated.
4. Fill certainty. Market orders always fill. In fast markets opportunity
   cost appears.
5. Fill timing. Daily fills are at a price interpolated between two closes.
   Real fills have an open auction, intraday drift and a schedule; the
   execution layer models the schedule but does not feed back into the
   daily backtest.
6. Square-root temporary and linear permanent impact. If temporary impact
   is more concave, large child orders cost less than modelled.
7. Universe integrity. The synthetic panel has no delistings or corporate
   actions. Names can leave the universe (NaN prices; tested).
8. Moment-based deflated Sharpe. The PSR formula uses sample skew and
   kurtosis; heavy tails make it optimistic.
9. Costs do not move the close. Reasonable below a few percent of ADV;
   the capacity table reports the 95th-percentile participation to show
   where this fails.
10. Inputs are finite and in domain. Every public entry point raises
    `ValueError` on NaN or Inf rather than returning a number.
