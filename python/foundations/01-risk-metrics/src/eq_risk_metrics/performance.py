"""Drawdown and risk-adjusted performance measures."""
from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd

from .volatility import TRADING_DAYS

__all__ = ["DrawdownResult", "max_drawdown", "sharpe_ratio", "sortino_ratio"]


class DrawdownResult(TypedDict):
    """Return type of :func:`max_drawdown`."""

    max_drawdown: float
    peak_date: object
    trough_date: object
    drawdown_series: pd.Series


def max_drawdown(prices: pd.Series) -> DrawdownResult:
    """Maximum peak-to-trough decline of a price series, with dates.

    ``drawdown_t = P_t / running_max(P)_t - 1``, always <= 0. The
    reported peak is the running maximum *at or before* the trough
    date, i.e. the specific high from which the worst decline occurred
    (not necessarily the series' global maximum, which may occur after
    the trough and be irrelevant to that particular drawdown).

    Parameters
    ----------
    prices : pandas.Series
        Price level series, indexed by date, strictly positive, at
        least 1 observation.

    Returns
    -------
    DrawdownResult
        ``max_drawdown`` (float, <= 0, the worst fractional decline),
        ``peak_date``, ``trough_date`` (index labels), and
        ``drawdown_series`` (the full drawdown path, same index as
        ``prices``).

    Raises
    ------
    ValueError
        If ``prices`` is empty, or contains a non-finite (NaN/inf) or a
        non-positive value. A zero
        or negative "price" makes ``P_t / running_max - 1`` meaningless
        (a drawdown below -100%, or a sign flip), so it is rejected at
        the door rather than propagated into a report -- in practice it
        means a bad data feed, not a real market event.
    """
    prices = pd.Series(prices)
    if len(prices) == 0:
        raise ValueError("max_drawdown: prices is empty; need at least 1 observation")
    if not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError(
            "max_drawdown: prices contains non-finite values (NaN/inf); "
            "forward-fill or drop the gaps before measuring drawdown"
        )
    if (prices <= 0).any():
        raise ValueError(
            "max_drawdown: prices must be strictly positive "
            "(a non-positive price level is a data error, not a drawdown)"
        )
    running_max = prices.cummax()
    drawdown = prices / running_max - 1
    trough_date = drawdown.idxmin()
    peak_date = prices.loc[:trough_date].idxmax()
    return {
        "max_drawdown": drawdown.min(),
        "peak_date": peak_date,
        "trough_date": trough_date,
        "drawdown_series": drawdown,
    }


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.03) -> float:
    """Annualised Sharpe ratio.

    ``Sharpe = mean(excess) / std(excess) * sqrt(252)`` where
    ``excess = returns - rf_daily`` and ``rf_daily`` is the annual
    risk-free rate compounded down to a daily rate,
    ``(1 + rf_annual)^(1/252) - 1``.

    Assumptions: constant risk-free rate over the sample; ``sqrt(252)``
    annualisation is only valid if returns are i.i.d. (positive serial
    correlation inflates the annualised ratio relative to the true
    risk-adjusted return, negative correlation deflates it).

    Parameters
    ----------
    returns : pandas.Series
        Simple daily returns, unitless.
    rf_annual : float
        Annualised risk-free rate, continuously-compounded-equivalent
        not required -- treated as an annual compounding rate.

    Returns
    -------
    float
        Annualised Sharpe ratio (unitless). ``+/-inf`` or ``NaN`` if
        excess returns have zero standard deviation (e.g. a constant
        or single-observation series) -- this is the correct limiting
        behaviour of the formula, not a bug; callers presenting this
        number to a report should guard against it explicitly.
    """
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    return excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.03) -> float:
    """Annualised Sortino ratio: like Sharpe, but penalises only downside risk.

    ``Sortino = mean(excess) / std(excess[excess < 0]) * sqrt(252)``.
    Upside volatility is not penalised, which better matches how most
    investors actually perceive risk (an asymmetric utility over gains
    vs losses).

    Assumptions: same annualisation caveat as :func:`sharpe_ratio`,
    plus the downside deviation is estimated from a smaller sub-sample
    (only the negative-excess-return days), so it is noisier than the
    full-sample Sharpe denominator, especially in short or strongly
    trending-up samples.

    Parameters
    ----------
    returns : pandas.Series
        Simple daily returns, unitless.
    rf_annual : float
        Annualised risk-free rate (see :func:`sharpe_ratio`).

    Returns
    -------
    float
        Annualised Sortino ratio (unitless). ``NaN`` if there are no
        excess-return observations below zero (e.g. an all-positive
        return series) -- the downside sample is empty so its standard
        deviation is undefined; callers should treat ``NaN`` here as
        "downside risk could not be estimated", not "zero risk".
    """
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    downside = excess[excess < 0]
    return excess.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS)
