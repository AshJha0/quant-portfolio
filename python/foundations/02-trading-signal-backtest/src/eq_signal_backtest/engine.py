"""Vectorised backtest engine for a long/flat moving-average crossover.

Design decisions that matter (expanded in ``docs/METHODOLOGY.md``):

1. NO LOOK-AHEAD. The signal computed on day t's close is traded at day
   t+1 (``position = signal.shift(1)``). Trading on the same close used
   to compute the signal is the most common backtest bug and silently
   inflates every downstream number -- ``tests/test_engine.py`` proves
   this structurally, not just by inspection.

2. TRANSACTION COSTS. A fixed cost in basis points is charged on every
   *change* in position (0->1 or 1->0), applied to that day's return.
   Cost-free backtests of fast signals are fiction; a zero-cost variant
   is always available (``cost_bps=0.0``) so the drag is visible rather
   than hidden.

3. LONG/FLAT ONLY. Shorting adds borrow costs and margin mechanics this
   simple engine does not model, so it does not pretend to. Positions
   are strictly in {0.0, 1.0}.

4. CASH EARNS NOTHING while flat (position 0.0 contributes exactly 0 to
   the daily return). This slightly understates strategy returns in
   high-rate periods -- see the assumptions register in
   ``docs/METHODOLOGY.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252

__all__ = ["TRADING_DAYS", "BacktestResult", "run_backtest", "performance_stats"]


@dataclass
class BacktestResult:
    """Output of :func:`run_backtest`.

    Attributes
    ----------
    equity : pandas.Series
        Strategy equity curve, starts at 1.0, net of transaction costs.
    benchmark : pandas.Series
        Buy & hold equity curve over the same prices, starts at 1.0.
    position : pandas.Series
        Executed position (0.0 or 1.0) after the one-day execution lag.
    n_trades : int
        Number of position changes (0->1 or 1->0) over the sample.
    stats : dict
        Output of :func:`performance_stats` for the strategy, plus a
        nested ``"benchmark"`` key holding the same stats for buy & hold.
    """

    equity: pd.Series
    benchmark: pd.Series
    position: pd.Series
    n_trades: int
    stats: dict[str, Any]


def run_backtest(
    prices: pd.Series, signal: pd.Series, cost_bps: float = 5.0
) -> BacktestResult:
    """Backtest a 0/1 signal with next-day execution and transaction costs.

    Parameters
    ----------
    prices : pandas.Series
        Close prices, float, ascending date index, same index as
        ``signal``.
    signal : pandas.Series
        Output of :func:`eq_signal_backtest.signals.ma_crossover_signal`
        (or any other 0.0/1.0-valued series on the same index).
    cost_bps : float, default 5.0
        One-way transaction cost in basis points of traded value, charged
        on every day the position changes. 5 bps is a reasonable all-in
        figure (commission + half spread + slippage) for a liquid
        large-cap ETF; it is optimistic for anything less liquid.

    Returns
    -------
    BacktestResult
    """
    rets = prices.pct_change().fillna(0.0)

    # Execution lag: today's position is yesterday's signal. This is the
    # single line that prevents look-ahead bias.
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


def performance_stats(returns: pd.Series, equity: pd.Series) -> dict[str, float]:
    """Summary statistics for a daily return series and its equity curve.

    Parameters
    ----------
    returns : pandas.Series
        Daily simple returns (e.g. strategy returns net of costs, or
        buy & hold returns).
    equity : pandas.Series
        Cumulative equity curve implied by ``returns``, starting at 1.0
        (i.e. ``(1 + returns).cumprod()``).

    Returns
    -------
    dict
        ``cagr`` : float
            Compound annual growth rate, annualised over
            ``len(returns) / TRADING_DAYS`` years. ``NaN`` if the sample
            spans zero years (empty series).
        ``volatility`` : float
            Annualised standard deviation of daily returns
            (``ddof=1``, scaled by ``sqrt(TRADING_DAYS)``).
        ``sharpe`` : float
            Annualised Sharpe ratio (mean / std * sqrt(TRADING_DAYS),
            zero risk-free rate). ``NaN`` when the return series has zero
            (or undefined, e.g. single-observation) standard deviation --
            a Sharpe ratio is not defined for a strategy with no return
            variance, and reporting 0 or +/-inf would be misleading.
        ``max_drawdown`` : float
            Minimum of ``equity / equity.cummax() - 1`` (a non-positive
            number; 0.0 if equity is non-decreasing).
        ``exposure`` : float
            Share of days with a nonzero return, used as a rough proxy
            for time spent in the market (an approximation: a trade
            whose return happens to be exactly zero is not counted).
    """
    n_years = len(returns) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    vol = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    std = returns.std(ddof=1)
    sharpe = (
        returns.mean() / std * np.sqrt(TRADING_DAYS)
        if pd.notna(std) and std > 0
        else np.nan
    )
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
