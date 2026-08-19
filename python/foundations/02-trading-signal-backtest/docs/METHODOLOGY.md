# Methodology — MA-Crossover Signal Backtest

Pipeline: **close prices → moving-average crossover signal → one-day
execution lag + transaction costs → performance statistics → in-sample
parameter selection → out-of-sample / walk-forward validation →
parameter-sensitivity map**.

Conventions: daily closes in price-index units (synthetic data starts at
100), trading-day time (252/yr), returns are simple (not log) daily
returns, transaction costs in basis points of traded value, long/flat
exposure in {0.0, 1.0} (never short, never leveraged).

---

## 1. Why this model? (contract item 1)

The core modelling decision is a **binary long/flat simple-moving-average
crossover**: go long when a short-window average of closes is above a
longer-window average, otherwise hold cash. This is deliberately the
simplest possible trend-following rule, chosen against three alternatives:

### vs. a momentum / time-series-momentum signal

Momentum signals (e.g. "long if the trailing 12-month return is positive")
measure the sign of a return over a fixed lookback rather than the
relative position of two smoothed averages. They are simpler to reason
about statistically (one parameter — the lookback — instead of two) and
are the basis of a large academic literature (Moskowitz, Ooi & Pedersen,
2012). The trade-off: a single-lookback momentum signal reacts to price
*level* changes only at its two lookback endpoints, so it can whipsaw
sharply the day the oldest observation drops out of the window even
though nothing happened that day (a "phantom" signal flip). A moving-average
crossover's *two* windows smooth this: the signal changes only when the
fast/slow relationship actually crosses, which happens gradually as new
information enters both averages. The crossover is also more naturally a
*state* variable (it's already 0/1) than a raw momentum score, which
needs an extra thresholding decision. Both are trend-following and share
the same fundamental weakness: neither has a model of *why* trends exist,
so neither can distinguish a persistent trend from a temporary one before
the fact.

### vs. a mean-reversion signal (e.g. z-scored deviation from a rolling mean)

A mean-reversion signal bets on prices *reverting* to a rolling average
rather than *continuing* to move away from it — the opposite economic bet.
Single-asset equity mean reversion at daily frequency is a much harder
trade in practice: outright long-only equities trend upward over long
horizons (equity risk premium), and a mean-reversion rule fighting that
drift needs either shorting (excluded here — see Assumption A3 below) or
very short holding periods with correspondingly higher turnover and cost
sensitivity. Section 5 of `docs/VALIDATION.md` demonstrates the two
regimes directly: this project's crossover signal *is* effectively a
mean-reversion trade when applied to a genuinely mean-reverting
(range-bound) price path, and it loses money there — which is exactly the
predicted failure mode, not a surprise. The right tool depends on which
regime you believe you're in; a project that only implements the
trend-following side must say so plainly (it does, here and in
`docs/VALIDATION.md`).

### vs. a machine-learning classifier (e.g. gradient-boosted trees or logistic regression on engineered features)

An ML classifier trained to predict next-day (or next-week) direction from
a feature set (returns, volatility, volume, macro series, ...) can in
principle capture nonlinear, multi-factor relationships a two-parameter
crossover cannot. The trade-offs are the reasons this project does not use
one: (1) **overfitting risk scales with model flexibility** — a
crossover has exactly two integer hyperparameters and a bounded,
interpretable hypothesis space (the `parameter_grid` sensitivity map in
this project can be read end-to-end by eye; a hundred-tree ensemble's
decision boundary cannot); (2) **sample size** — daily-frequency single-
asset history rarely exceeds a few thousand independent-ish observations,
which is thin for a high-capacity learner without careful cross-validation
and regularisation, and the walk-forward machinery in this project is a
prerequisite for doing that honestly *before* an ML model is worth trying;
(3) **auditability** — a systematic desk (see `docs/DESK_GUIDE.md`) can
explain a crossover's every position to a risk committee in one sentence;
explaining why a trained classifier is long today is a live research
problem (feature attribution, SHAP, etc.) with its own failure modes. This
project is a deliberately transparent baseline that an ML overlay would
need to beat *after* accounting for its extra degrees of freedom, not a
claim that MA crossovers are the best possible signal.

### Why simple (not exponential) moving averages

An exponentially-weighted moving average (EWMA) weights recent
observations more heavily and needs no fixed lookback, which can react
faster to genuine regime changes. It was not used here because (a) it
loses the clean, auditable "warm-up then flat" property of a simple
rolling window (`ma_crossover_signal`'s first `slow - 1` observations are
provably 0.0 rather than defined-but-noisy from day one), which matters
for the no-look-ahead tests in `tests/test_engine.py`, and (b) the
question this project is answering is about *evaluation discipline*
(no-look-ahead, costs, in/out-of-sample, walk-forward, sensitivity), which
is identical regardless of which moving-average variant is used — adding
EWMA would add a parameter without adding to the point being demonstrated.

---

## 2. Assumptions register (contract item 2)

Every assumption below is enforced in code (not just stated in prose);
where it is enforced is noted, and every row is tested.

| # | Assumption | What breaks if violated |
|---|------------|--------------------------|
| A1 | **No look-ahead: the signal computed on day t's close is executed on day t+1** (`position = signal.shift(1)`). | Trading the same close used to compute the signal is the single most common backtest bug. It silently inflates every downstream number because the "prediction" is made with same-day information the strategy could not actually have had at the moment of the trade. `tests/test_engine.py::TestNoLookAhead` proves this structurally: a detector test constructs a 30% overnight jump that a same-day ("cheat") execution captures and the honest t-1 engine does not. |
| A2 | **Transaction costs are a fixed 5 bps of traded value per position change**, applied on the day of the change. | 5 bps is a reasonable all-in figure (commission + half spread + slippage) for a liquid large-cap ETF; it is optimistic for anything less liquid, and it does not scale with trade size or the current bid-ask spread regime (it is flat, not impact-based). The bias a zero-cost backtest introduces is systematic and always in the same direction — it flatters high-turnover parameter pairs — and it is quantified in `docs/VALIDATION.md` §7: on the bundled data, moving from 0 to 20 bps costs the fastest grid pair 1.21pp of CAGR against 0.24pp for the slowest, a 5x difference. Whether that bias is large enough to *reverse* the ranking depends on the sample (on this one it does not), which is precisely why costs are never omitted: their absence is a known bias of unknown size. The cost model is also flat rather than impact-based — it does not scale with trade size or the prevailing spread regime — so it understates the true cost of any size that moves the market. |
| A3 | **Long/flat only — no shorting.** Position is strictly in {0.0, 1.0}. | Shorting would add borrow fees, margin/haircut mechanics, and the possibility of a forced buy-in — none of which this engine models, so it does not pretend to be able to trade a "sell" signal. In a genuine mean-reverting or bearish regime the strategy can only step aside (flat), not profit from the decline; this caps both the downside (protective) and the opportunity captured on the short side. |
| A4 | **Idle cash earns exactly 0%** while flat (`position == 0` contributes exactly `0` to that day's return). | Understates strategy returns in any period with a positive risk-free rate — the un-modelled refinement is a T-bill accrual on the flat days. The effect compounds with `exposure` (the fraction of days flat, reported in `performance_stats`): a strategy that is flat 40% of the time in a 5%-rate environment is understating its real-world return by roughly `0.4 * 5% = 2%` annualised, a number the current stats do not surface. |
| A5 | **Grid-search parameter selection is itself a source of selection bias**, mitigated but not eliminated by evaluating only out-of-sample. | Picking the `argmax` Sharpe of a grid of noisy Sharpe estimates is, in expectation, biased upward relative to any individual cell's true Sharpe (the "winner's curse" of multiple testing) — see `select_best_params`'s docstring. `docs/VALIDATION.md` §6 quantifies the actual in-sample/out-of-sample Sharpe gap on the bundled data; a wider gap than that is the signature of a search that explored too many candidates relative to the training sample length. A deflated-Sharpe correction (Bailey & López de Prado, 2014) for the number of grid cells tried is the natural next step and is not implemented (see §7). |
| A6 | **A single train/test split is only one draw from history.** | A lucky (or unlucky) split boundary can flatter or damn the strategy; one 70/30 split answers "did it work *this once*", not "does it work in general". `walk_forward_backtest` is the direct mitigation — it repeats the same discipline across several rolling formation/trading windows and stitches together only the out-of-sample segments — but even walk-forward windows on one asset's one historical path are correlated with each other (overlapping regimes), so it is evidence, not proof. |
| A7 | **Single-asset, single-history evaluation.** | Every number in this project is a statement about one simulated (or, with `data/live.py`, one real) price path. A signal validated on one instrument's one realised history may be a story about that decade/instrument rather than about the signal class in general — survivorship is a real risk if the instrument is a well-known index/ETF that is already known to have "worked". The mitigation this project does *not* implement (documented as a limitation, §7) is testing across multiple, ideally independent, assets and asset classes. |
| A8 | **Prices are clean, strictly positive daily closes with no gaps, splits, or corporate actions.** | The engine trusts `prices.pct_change()` to be a real economic return every day. A stock split or unadjusted dividend would appear as a fictitious return spike, corrupting both the signal (a fake crossover) and the cost accounting (a fake "trade" if the spike straddles a rolling window). `data/live.py` uses `yfinance`'s `auto_adjust=True`, which handles this for real data; the bundled synthetic generator has no such events by construction. The *detectable* violations of this assumption are now enforced rather than assumed: `run_backtest` rejects non-finite, zero and negative prices, because `pct_change().fillna(0.0)` would otherwise record a missing close as a genuine flat day and a zero close would send every subsequent equity value to `NaN` — both silently. What remains un-enforceable is a price that is wrong but plausible (an unadjusted split of exactly the right size looks like a real return), which is why the assumption stays on this list. See `docs/VALIDATION.md` §10. |
| A9 | **The bundled sample is one draw, and the seed is part of the result.** | Ten-year paths from the bundled generator range from roughly -14% to +28% CAGR depending on the seed, so any statement of the form "the strategy returned X on the bundled data" is a statement about one path. The default seed (32) was chosen because its path is close to the model's central case, not because it is favourable — it produces a *negative* out-of-sample result. Reporting a number from a seed selected for its outcome would be the backtest-overfitting failure this project is about, committed one level up in the data-generating process. |

---

## 3. The mathematics

**Signal.** `fast_ma_t = mean(price[t-fast+1 : t+1])`,
`slow_ma_t = mean(price[t-slow+1 : t+1])`,
`signal_t = 1 if fast_ma_t > slow_ma_t else 0` (including when either
average is undefined during warm-up, where the comparison against `NaN`
is `False` by construction — no special-casing needed).

**Execution & costs.** `position_t = signal_{t-1}` (`position_0 = 0`,
since there is no `t=-1`). `trade_t = |position_t - position_{t-1}|`
(`trade_0 = 0`). `cost_t = trade_t * cost_bps / 10_000`.
`strategy_return_t = position_t * simple_return_t - cost_t`.
`equity_t = prod_{s<=t} (1 + strategy_return_s)`.

**Performance statistics** (`performance_stats`, over `n` daily
observations spanning `n_years = n / 252`):

- `CAGR = equity_final ** (1 / n_years) - 1`.
- `volatility = std(returns, ddof=1) * sqrt(252)`.
- `Sharpe = mean(returns) / std(returns, ddof=1) * sqrt(252)`, defined as
  `NaN` (not 0, not ±inf) when `std == 0` or the sample has fewer than 2
  observations — a Sharpe ratio requires a defined, nonzero return
  variance, and reporting a spurious finite number for a degenerate input
  is worse than reporting "undefined".
- `max_drawdown = min(equity_t / running_max(equity)_t - 1)`, a
  non-positive number (0 for a monotonically non-decreasing equity curve).
- `exposure = mean(returns != 0)`, a rough proxy for time spent in the
  market (an approximation: a trade whose daily return happens to be
  exactly zero is not counted as "in the market" that day).

**Parameter selection** (`select_best_params`): argmax over
`parameter_grid`'s Sharpe surface, restricted to the training window.

**Walk-forward** (`walk_forward_backtest`): tile the sample into
non-overlapping `(formation, trading)` windows (`walk_forward_windows`,
default `step = trading`, so trading windows are contiguous and gapless).
For each window, `select_best_params` runs on the formation slice only;
the resulting `(fast, slow)` is *frozen* and applied to a signal computed
over `formation + trading` combined (so the moving averages are properly
warmed up going into the trading window — this uses no information from
inside the trading window itself, since the signal at day `t` still only
reads `price[<=t]`). Only the trading-window portion of the resulting
daily returns is kept; these out-of-sample-only segments are concatenated
across all windows into one continuous stitched equity curve. Parameters
selected for window *k* are never re-fit using any day inside trading
window *k* — this is the property that makes the stitched curve an honest
out-of-sample track record.

---

## 4. What this project deliberately does not do

This is a backtest of a *signal's historical statistical behaviour under
stated assumptions*, not a trading strategy ready for capital. Missing,
by design, and left for `docs/DESK_GUIDE.md`'s pre-launch checklist to
require before any of the mitigations below matter:

- Execution modelling that scales with trade size (fixed-bps costs
  understate impact at any size that moves the market).
- A deflated Sharpe ratio correcting for the number of `(fast, slow)`
  combinations tried in the grid search (Bailey & López de Prado, 2014).
- Multiple assets / asset classes, to test whether the effect generalises
  beyond one simulated (or one real) price history.
- Volatility-scaled position sizing instead of binary 0/1 exposure.
- Statistical significance testing of the strategy-vs-benchmark Sharpe gap
  (e.g. a stationary bootstrap on daily returns) instead of eyeballing
  point estimates. This is the most consequential omission on the list:
  with an out-of-sample Sharpe of -0.12 over ~3 years, the confidence
  interval comfortably contains zero *and* the in-sample 0.87, so the
  honest statement is "no evidence of an edge", which is weaker (and more
  accurate) than "evidence of no edge".
- A minimum-turnover or trade-frequency filter, which would let a desk
  express "do not take this signal if it implies more than N trades a
  year" as a constraint on the search rather than a post-hoc observation.

## 5. References

- Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum".
- Lo (2002), "The Statistics of Sharpe Ratios".
- Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting, and Non-Normality".
