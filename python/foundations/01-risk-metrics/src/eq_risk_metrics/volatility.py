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

import numpy as np
import pandas as pd

TRADING_DAYS = 252

__all__ = [
    "TRADING_DAYS",
    "simple_returns",
    "log_returns",
    "annualised_volatility",
    "rolling_volatility",
    "ewma_volatility",
]


# ----------------------------------------------------------------------
# Return construction
# ----------------------------------------------------------------------
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
    return prices.pct_change().dropna()


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
    return np.log(prices / prices.shift(1)).dropna()


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
        Annualised volatility, unitless (e.g. 0.18 = 18%/year). ``NaN``
        if ``returns`` has fewer than 2 observations.
    """
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


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
    """
    return returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
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

    Returns
    -------
    pandas.Series
        Annualised EWMA volatility, same index and length as
        ``returns`` (no warm-up ``NaN``s, since the recursion is
        seeded at ``t=0``).
    """
    var = returns.ewm(alpha=1 - lam, adjust=False).var(bias=True)
    return np.sqrt(var * TRADING_DAYS)
