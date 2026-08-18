"""
Vectorised backtest engine for a long/flat moving-average crossover.

Design decisions that matter (all discussed in the README):

1. NO LOOK-AHEAD. The signal computed on day t's close is traded at
   day t+1 (position = signal.shift(1)). Trading on the same close
   you used to compute the signal is the most common backtest bug.

2. TRANSACTION COSTS. A fixed cost in basis points is charged on
   every change in position. Cost-free backtests of fast signals are
   fiction.

3. LONG/FLAT ONLY. Shorting adds borrow costs and margin mechanics
   this simple engine does not model, so it does not pretend to.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: pd.Series          # strategy equity curve (starts at 1.0)
    benchmark: pd.Series       # buy & hold equity curve
    position: pd.Series        # 0 or 1, after the execution lag
    n_trades: int
    stats: dict


def ma_crossover_signal(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    """1 when fast MA > slow MA, else 0. Computed on close prices."""
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float)


def run_backtest(prices: pd.Series,
                 signal: pd.Series,
                 cost_bps: float = 5.0) -> BacktestResult:
    """Backtest a 0/1 signal with next-day execution and costs.

    cost_bps: one-way transaction cost in basis points of traded value.
    5 bps is a reasonable all-in figure (commission + half spread +
    slippage) for a liquid large-cap ETF; it is optimistic for anything
    less liquid.
    """
    rets = prices.pct_change().fillna(0.0)

    # Execution lag: today's position is yesterday's signal.
    position = signal.shift(1).fillna(0.0)

    trades = position.diff().abs().fillna(0.0)
    costs = trades * cost_bps / 10_000

    strat_rets = position * rets - costs
    equity = (1 + strat_rets).cumprod()
    benchmark = (1 + rets).cumprod()

    stats = performance_stats(strat_rets, equity)
    stats["benchmark"] = performance_stats(rets, benchmark)
    return BacktestResult(
        equity=equity,
        benchmark=benchmark,
        position=position,
        n_trades=int(trades.sum()),
        stats=stats,
    )


def performance_stats(returns: pd.Series, equity: pd.Series) -> dict:
    n_years = len(returns) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    vol = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
              if returns.std(ddof=1) > 0 else np.nan)
    dd = (equity / equity.cummax() - 1).min()
    # Time in market: share of days with nonzero exposure
    exposure = (returns != 0).mean()
    return {
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "exposure": exposure,
    }


def parameter_grid(prices: pd.Series,
                   fast_range: range,
                   slow_range: range,
                   cost_bps: float = 5.0) -> pd.DataFrame:
    """Sharpe ratio for every (fast, slow) pair.

    Used to visualise how sensitive the result is to parameter choice.
    A strategy whose Sharpe collapses one cell away from the chosen
    parameters is curve-fit, not robust.
    """
    rows = []
    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue
            sig = ma_crossover_signal(prices, fast, slow)
            res = run_backtest(prices, sig, cost_bps)
            rows.append({"fast": fast, "slow": slow,
                         "sharpe": res.stats["sharpe"]})
    return pd.DataFrame(rows).pivot(index="fast", columns="slow",
                                    values="sharpe")
