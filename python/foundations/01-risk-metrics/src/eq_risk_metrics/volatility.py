"""Return construction and volatility estimators.

Conventions used throughout this package:

- 252 trading days per year (``TRADING_DAYS``).
- Volatility is annualised (scaled by ``sqrt(TRADING_DAYS)``) unless stated
  otherwise, and is computed on **simple daily returns**
  (``P_t / P_{t-1} - 1``) unless the function name says ``log``.
- Log returns aggregate additively over time (useful for compounding);
  simple returns aggregate across a portfolio at a point in time. For
  single-asset daily risk measurement the numerical difference between the
  two is small, but the choice matters and is documented per function.
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from ._validate import finite_series

TRADING_DAYS = 252

__all__ = [
    "TRADING_DAYS",
    "annualised_volatility",
    "ewma_volatility",
    "log_returns",
    "rolling_volatility",
    "simple_returns",
]


# ----------------------------------------------------------------------
# Return construction
# ----------------------------------------------------------------------
def _positive_prices(prices, func: str) -> pd.Series:
    prices = finite_series(prices, func, name="prices", min_len=2)
    if (prices <= 0).any():
        raise ValueError(f"{func}: prices must be strictly positive")
    return prices


def simple_returns(prices: pd.Series) -> pd.Series:
    """Simple daily returns ``P_t / P_{t-1} - 1``.

    Parameters
    ----------
    prices : pandas.Series
        Price level series, indexed by date, strictly positive.

    Returns
    -------
    pandas.Series
        Simple returns (unitless, fraction not percent), one element
        shorter than ``prices`` (the first observation has no prior
        price to compare against).
    """
    prices = _positive_prices(prices, "simple_returns")
    return prices.pct_change().iloc[1:]


def log_returns(prices: pd.Series) -> pd.Series:
    """Log (continuously compounded) daily returns ``ln(P_t / P_{t-1})``.

    Parameters
    ----------
    prices : pandas.Series
        Price level series, indexed by date, strictly positive.

    Returns
    -------
    pandas.Series
        Log returns (unitless, fraction not percent), one element
        shorter than ``prices``.
    """
    prices = _positive_prices(prices, "log_returns")
    return cast(pd.Series, np.log(prices / prices.shift(1))).iloc[1:]


# ----------------------------------------------------------------------
# Volatility
# ----------------------------------------------------------------------
def annualised_volatility(returns: pd.Series) -> float:
    """Unconditional (full-sample) volatility, annualised by ``sqrt(252)``.

    Uses the sample standard deviation with Bessel's correction
    (``ddof=1``).

    Assumption: returns are i.i.d., so variance scales linearly with
    time (the "square-root-of-time" rule). Real returns show volatility
    clustering (autocorrelated squared returns), which makes this an
    approximation of the *current* risk level -- see
    :func:`rolling_volatility` and :func:`ewma_volatility` for
    time-varying alternatives.

    Parameters
    ----------
    returns : pandas.Series
        Simple (or log) daily returns, unitless.

    Returns
    -------
    float
        Annualised volatility, unitless (e.g. 0.18 = 18%/year).

    Raises
    ------
    ValueError
        If ``returns`` has fewer than 2 observations or a non-finite value.
    """
    returns = finite_series(returns, "annualised_volatility", min_len=2)
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Rolling annualised volatility over a trailing window.

    ``window=21`` trading days is roughly one calendar month, so this
    reflects "the current regime" rather than the full-sample average.

    Parameters
    ----------
    returns : pandas.Series
        Simple (or log) daily returns, unitless.
    window : int
        Trailing window length in trading days, >= 2.

    Returns
    -------
    pandas.Series
        Annualised rolling volatility, same index as ``returns``; the
        first ``window - 1`` entries are ``NaN`` (insufficient history).
        If ``window`` exceeds ``len(returns)`` the result is **all**
        ``NaN`` -- no window is ever full, which is the correct answer
        rather than an error (a rolling series that has not warmed up
        yet is a normal state at the start of a backtest).

    Raises
    ------
    ValueError
        If ``window`` is not an integer >= 2. ``window=1`` is rejected
        rather than silently returning all-``NaN``: the sample standard
        deviation of a single observation with ``ddof=1`` is ``0/0``, so
        a 1-day "rolling volatility" is undefined, not zero.
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise ValueError(f"rolling_volatility: window must be an int, got {window!r}")
    if window < 2:
        raise ValueError(
            f"rolling_volatility: window must be >= 2, got {window} "
            "(std with ddof=1 needs at least two observations)"
        )
    returns = finite_series(returns, "rolling_volatility")
    return returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def ewma_volatility(
    returns: pd.Series, lam: float = 0.94, seed_window: int = 30
) -> pd.Series:
    """RiskMetrics-style exponentially weighted moving average volatility.

    ``sigma_t^2 = (1 - lambda) r_t^2 + lambda * sigma_{t-1}^2``, seeded
    at the first observation. ``lambda = 0.94`` is the RiskMetrics
    convention for daily data (0.97 is the monthly convention).

    Reacts faster to regime changes than an equal-weighted rolling
    window because recent observations receive exponentially more
    weight, at the cost of more sampling noise in the estimate itself.

    Parameters
    ----------
    returns : pandas.Series
        Simple (or log) daily returns, unitless.
    lam : float
        Decay factor in (0, 1); higher = slower-reacting, smoother
        estimate. RiskMetrics daily convention is 0.94.
    seed_window : int
        Number of leading returns whose mean square seeds the recursion.

    Returns
    -------
    pandas.Series
        Annualised EWMA volatility, same index and length as
        ``returns``. Entry ``t`` is the one-step-ahead forecast
        ``sqrt(252 * s2_t)`` with the zero-mean RiskMetrics recursion
        ``s2_t = lam * s2_{t-1} + (1 - lam) * r_{t-1}^2``. The seed
        ``s2_0`` is ``mean(r^2)`` over the first ``min(seed_window, n)``
        returns, so the seed looks ahead by at most ``seed_window`` days.

    Raises
    ------
    ValueError
        If ``lam`` is not strictly inside ``(0, 1)``. ``lam=0`` is
        rejected because it means "no memory at all", which makes the
        recursion return an identically-zero variance rather than a
        maximally reactive estimate -- a silently useless answer;
        ``lam>=1`` means the estimate never updates.
    """
    if not np.isfinite(lam) or not 0.0 < lam < 1.0:
        raise ValueError(
            f"ewma_volatility: lam must be a finite number strictly between "
            f"0 and 1 (RiskMetrics daily convention is 0.94), got {lam!r}"
        )
    returns = pd.Series(returns, dtype=float)
    r = returns.to_numpy()
    if r.size == 0:
        raise ValueError("ewma_volatility: returns is empty")
    if not np.isfinite(r).all():
        raise ValueError("ewma_volatility: returns contains a non-finite value")
    if not isinstance(seed_window, (int, np.integer)) or seed_window < 1:
        raise ValueError(f"ewma_volatility: seed_window must be a positive int, got {seed_window!r}")
    w = min(int(seed_window), r.size)
    s2 = np.empty(r.size)
    s2[0] = float(np.mean(r[:w] ** 2))
    for t in range(1, r.size):
        s2[t] = lam * s2[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return pd.Series(np.sqrt(s2 * TRADING_DAYS), index=returns.index)
