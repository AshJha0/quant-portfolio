# Project 2 — Build and Backtest a Simple Trading Signal

A long/flat moving-average crossover strategy, backtested with next-day execution, transaction costs, an in-sample/out-of-sample split, and a parameter-sensitivity map.

The strategy itself is deliberately simple. The point of the project is the *evaluation discipline*: showing where it worked, where it failed, what was assumed, and what would break it.

## How to run

```
pip install -r requirements.txt
python generate_sample_data.py      # only needed if data/ is empty
python analysis.py
```

For real market data: `pip install yfinance`, then `python download_data.py SPY 2010-01-01` and `python analysis.py data/SPY.csv`.

Outputs: `output/report.txt` and `output/figures/backtest_overview.png`.

## Strategy

Go long when the fast moving average is above the slow moving average; otherwise hold cash. The (fast, slow) windows are chosen by grid search **on the first 70% of history only**, then the strategy is judged on the untouched final 30%.

## What the backtest does correctly (assumptions made explicit)

- **No look-ahead.** The signal computed on day *t*'s close is executed on day *t+1* (`position = signal.shift(1)`). Trading the same close you used to compute the signal is the single most common backtest bug and silently inflates results.
- **Transaction costs.** 5 bps one-way on every position change — a reasonable all-in estimate (commission + half spread + slippage) for a liquid large-cap ETF, optimistic for anything else. A zero-cost variant is reported alongside, so the cost drag is visible rather than hidden.
- **Long/flat only.** Shorting involves borrow fees and margin mechanics this engine does not model, so it doesn't pretend to.
- **Cash earns nothing** while flat. This slightly understates strategy returns in high-rate periods — a T-bill return on idle cash would be the refinement.

## Where it worked, where it failed

Run the report and look at the four numbers that matter:

1. **In-sample vs out-of-sample Sharpe.** The in-sample number is contaminated by the grid search (picking the argmax of a grid *is* mild overfitting — the code says so in a comment). The out-of-sample row is the honest one, and some decay is the expected, realistic result.
2. **Strategy vs buy & hold.** Trend-following crossovers typically earn their keep by *avoiding deep drawdowns*, not by out-compounding a bull market. Compare max drawdowns, not just CAGRs.
3. **Cost drag.** The gap between the costed and cost-free CAGR shows what trading frequency costs. Faster parameter pairs look better cost-free and worse after costs — that reversal is the lesson.
4. **The sensitivity heatmap.** If performance is a lone green cell in a red sea, the parameters are curve-fit. A broad plateau of similar Sharpe values is what robustness looks like.

**Regime dependence, stated plainly:** this strategy class does well in sustained trends and prolonged bear markets (it steps aside), and loses in choppy, range-bound markets where every crossover is a whipsaw that pays costs and captures nothing.

## What could cause overfitting here

- Grid-searching parameters, even on the training set only (selection bias — mitigated but not eliminated by the holdout).
- Testing on a single asset and a single historical path; the result may be a story about that decade, not about the signal.
- Survivorship in asset choice: backtesting on an index/ETF that we already know did well.
- One train/test split. A lucky split boundary can flatter or damn the strategy.

## Why this is a backtest, not a trading strategy

A real strategy needs execution modelling (slippage that scales with size and volatility, partial fills), capacity analysis, borrowing/financing terms, tax treatment, and a live monitoring process for regime change. This project measures a *signal's historical statistical behaviour under stated assumptions* — nothing more, and it tries to be precise about exactly which assumptions those are.

## What I would improve

- Walk-forward analysis: re-select parameters on a rolling window and stitch together only out-of-sample segments.
- Multiple assets and asset classes to test whether the effect generalises.
- Volatility-scaled position sizing instead of binary 0/1 exposure.
- Statistical significance of the Sharpe difference (e.g. bootstrap on daily returns) rather than eyeballing point estimates.
- Deflated Sharpe ratio to account for the number of parameter combinations tried.
