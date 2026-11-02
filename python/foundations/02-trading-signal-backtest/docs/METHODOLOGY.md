# Methodology: MA-Crossover Signal Backtest

Pipeline: close prices, moving-average crossover signal, modelled t+1
fill and transaction costs, performance statistics, in-sample parameter
selection, out-of-sample and walk-forward validation, parameter-sensitivity
map, deflated Sharpe, PBO and a multi-seed distribution.

Every number quoted in these docs comes from synthetic data. No real-data
result exists yet; that is Phase 4 of the repository plan.

Conventions: daily closes in price-index units (synthetic data starts at
100), 252 trading days a year, simple daily returns, costs in basis points
of traded value, exposure in {0.0, 1.0} (never short, never leveraged).

---

## 1. Why this model

The signal is a binary long/flat simple-moving-average crossover: long when
the fast average of closes is above the slow average, otherwise cash. It is
the simplest trend-following rule. Three alternatives were considered.

**Time-series momentum** ("long if the trailing 12-month return is
positive"; Moskowitz, Ooi and Pedersen, 2012) has one parameter instead of
two. Its signal can flip on the day the oldest observation leaves the
window even when nothing happened that day. The crossover changes only when
the two averages cross. Neither rule has a model of why trends persist.

**Mean reversion** (a z-scored deviation from a rolling mean) is the
opposite bet. Long-only daily mean reversion fights the equity drift and
needs shorting (excluded, A3) or high turnover. `docs/VALIDATION.md` §8
shows the crossover losing on a synthetic mean-reverting path, which is the
failure mode this signal class is expected to have.

**A machine-learning classifier** can capture nonlinear, multi-factor
structure. It is not used because flexibility raises overfitting risk, one
daily single-asset history is a small sample, and a crossover's position
can be explained in one sentence. The walk-forward and selection-bias
machinery here is the prerequisite for testing such a model.

**Simple rather than exponential averages** keep a clean warm-up: the first
`slow - 1` signal values are exactly 0.0. The evaluation machinery is the
same for either average.

---

## 2. Assumptions register

| # | Assumption | What breaks if violated |
|---|------------|--------------------------|
| A1 | **Fill timing.** Signals use data up to the close of day t. Orders fill at a modelled t+1 price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction variant fills at p_t and pays an auction cost (10 bp by default). | Filling a day-t decision at the close of day t for free uses the closing print the decision was computed from. `tests/test_execution.py` builds a perfect-foresight signal: under a next-close fill it earns nothing, and under a same-close fill it earns about 37 standard errors on the test path. With only closes available, f = 0.5 is a stand-in for a next-open or early-session fill. It is not calibrated to any market. |
| A2 | **Costs are a flat 5 bp of traded value per position change**, charged on the fill day. | 5 bp is an all-in figure for a liquid large-cap ETF at small size. It does not scale with trade size or spread, so it understates cost at any size that moves the price. A cost-free grid search favours over-trading pairs (`docs/VALIDATION.md` §7). |
| A3 | **Long/flat only.** | Shorting needs borrow, margin and recall modelling that this engine does not have. In a falling market the strategy can only step aside. |
| A4 | **Cash earns 0%.** | Understates returns when rates are positive, roughly by the flat share of days times the cash rate. |
| A5 | **Grid-search selection is a source of selection bias.** | The best of 49 noisy Sharpe estimates is biased upward. The pipeline logs every grid cell as a trial (`TrialLog`), reports the deflated Sharpe ratio (Bailey and Lopez de Prado, 2014) against that count, reports PBO by CSCV over the grid, and measures the in-sample to out-of-sample gap over 100 synthetic seeds (0.35 Sharpe on average). |
| A6 | **One train/test split is one draw.** | A lucky split can flatter or damn the strategy. Walk-forward repeats the selection over rolling windows, and the multi-seed run repeats everything on independent synthetic paths. |
| A7 | **Single asset.** | Every number is about one synthetic asset. Generalisation across assets is untested. |
| A8 | **Prices are clean, positive daily closes.** | A split or unadjusted dividend shows up as a fake return. `run_backtest` rejects non-finite, zero and negative prices (`docs/VALIDATION.md` §10). A wrong but plausible price cannot be detected. |
| A9 | **A seed is part of a single-path result.** | Ten-year synthetic paths range from about -14% to +28% CAGR by seed. Headline numbers are therefore reported as a 100-seed distribution; seed 32 is shown only as a worked example. |

---

## 3. The mathematics

**Signal.** `fast_ma_t = mean(price[t-fast+1 : t+1])`,
`slow_ma_t = mean(price[t-slow+1 : t+1])`,
`signal_t = 1 if fast_ma_t > slow_ma_t else 0`. During warm-up the
comparison against `NaN` is `False`, so the signal is 0.

**Fill and P&L** (`strategy_returns`). Let `new_d = signal_{d-1}` be the
position after the fill on day d and `old_d = signal_{d-2}` the position
before it (both 0 at the start). Let `r_d = p_d / p_{d-1} - 1`.

- Mode `"next"` with fill fraction f: `p_fill = p_{d-1} + f (p_d - p_{d-1})`,
  so `p_fill / p_{d-1} - 1 = f r_d` and
  `p_d / p_fill - 1 = (1 + r_d) / (1 + f r_d) - 1`. The gross return is
  `(1 + old_d f r_d) (1 + new_d ((1 + r_d)/(1 + f r_d) - 1)) - 1`.
- Mode `"close_auction"`: gross return `new_d r_d`, with the auction cost
  added to the cost rate.
- Cost: `|new_d - old_d| (cost_bps [+ auction_cost_bp]) / 10_000`.
- Net return `max(gross - cost, -1)`; equity is the running product of
  `1 + net`.

Worked example (from `tests/test_execution.py`): closes 100, 100, 110, 121,
signal 0, 1, 1, 0, f = 0.5. The decision to buy is taken at the close of
day 1 (p = 100). It fills on day 2 at 100 + 0.5 x 10 = 105, so day 2 earns
110/105 - 1 = 4.76%, and day 3 earns 10%.

**Performance statistics** over `n` daily observations, `n_years = n/252`:
CAGR `equity_final^(1/n_years) - 1`; volatility `std(ddof=1) sqrt(252)`;
Sharpe `mean/std(ddof=1) sqrt(252)`, `NaN` when the standard deviation is
zero; max drawdown `min(equity / running max - 1)`; exposure
`mean(returns != 0)`.

**Research statistics** (`research_stats.py`, a re-export of
`qp_core.stats`):

- Sharpe SE: Lo (2002) delta method with a Newey-West (Bartlett) long-run
  variance, lag `floor(4 (n/100)^(2/9))`.
- Deflated Sharpe: `Phi((s - SR0) sqrt(n-1) / sqrt(1 - g3 s + (g4-1)/4 s^2))`
  with `SR0` the expected maximum of N null Sharpe estimates, whose spread
  is the variance of the logged trial Sharpe ratios.
- PBO: CSCV over 16 contiguous blocks; the share of splits where the
  in-sample best configuration ranks at or below the out-of-sample median.
- Block bootstrap: circular blocks of length `ceil(n^(1/3))`, percentile
  interval.
- Seed distribution: mean, sd and a Student-t 95% interval for the mean.

**Parameter selection** (`select_best_params`): argmax of the grid's
Sharpe, on the training window only.

**Walk-forward** (`walk_forward_backtest`): non-overlapping formation and
trading windows, contiguous trading windows by default. Parameters are
selected on each formation window and frozen for the trading window. The
signal is computed over formation plus trading so the averages are warm;
the signal at day t still reads only `price[<= t]`. The trading-window
returns are stitched into one out-of-sample curve.

---

## 4. Not implemented

- A cost model that scales with size (market impact).
- Multiple assets or asset classes.
- Volatility-scaled position sizing.
- A turnover constraint inside the parameter search.
- Real data. The loader in `data/live.py` exists and no result uses it.

## 5. References

- Moskowitz, Ooi and Pedersen (2012), "Time Series Momentum".
- Lo (2002), "The Statistics of Sharpe Ratios".
- Newey and West (1994), "Automatic Lag Selection in Covariance Matrix
  Estimation".
- Bailey and Lopez de Prado (2014), "The Deflated Sharpe Ratio".
- Bailey, Borwein, Lopez de Prado and Zhu (2017), "The Probability of
  Backtest Overfitting".
