"""Historical (realized) volatility estimators for FX, plus weekly seasonality.

Annualization convention
------------------------
FX spot trades ~24h a day, 5 days a week. Two daily-count conventions coexist
on desks:

* ``252`` -- equity-style trading days, the most common default and the one
  used throughout this package unless stated otherwise;
* ``260`` -- 52 weeks x 5 FX trading days, sometimes used for FX because the
  market takes almost no mid-week holidays (it is open somewhere in the world
  whenever it is a weekday anywhere).

The difference is a constant factor ``sqrt(260/252) ~ 1.0157`` (~1.6% of vol),
which matters when comparing realized vol to implied vol quotes -- be
consistent with the counterparty's convention. Every function here takes
``periods_per_year`` explicitly.

All estimators consume *log* returns / log price ranges and return
**annualized volatility in the same decimal unit as the inputs** (e.g. daily
decimal log returns in, annualized decimal vol out).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .returns import _as_1d_array

__all__ = [
    "close_to_close_vol",
    "rolling_close_vol",
    "parkinson_vol",
    "garman_klass_vol",
    "day_of_week_vol_factors",
]

_MIN_OBS = 2


def close_to_close_vol(
    returns: Sequence[float] | np.ndarray | pd.Series,
    periods_per_year: int = 252,
    ddof: int = 1,
) -> float:
    """Annualized close-to-close volatility: ``sqrt(ppy) * std(returns)``.

    Parameters
    ----------
    returns : array-like
        Per-period log returns (no NaNs -- ``ValueError`` otherwise).
    periods_per_year : int
        Annualization factor; 252 (default) or 260 for the FX 24h5d convention.
    ddof : int
        Delta degrees of freedom for the sample standard deviation.
    """
    arr = _as_1d_array(returns, "returns")
    if arr.size < _MIN_OBS:
        raise ValueError(f"need at least {_MIN_OBS} returns, got {arr.size}")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    return float(np.sqrt(periods_per_year) * arr.std(ddof=ddof))


def rolling_close_vol(
    returns: pd.Series,
    window: int,
    periods_per_year: int = 252,
    ddof: int = 1,
) -> pd.Series:
    """Rolling annualized close-to-close volatility (window of log returns)."""
    if not isinstance(returns, pd.Series):
        returns = pd.Series(np.asarray(returns, dtype=float))
    _as_1d_array(returns.to_numpy(), "returns")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if window > returns.size:
        raise ValueError(f"window={window} exceeds series length {returns.size}")
    return returns.rolling(window).std(ddof=ddof) * np.sqrt(periods_per_year)


def parkinson_vol(
    high: Sequence[float] | np.ndarray | pd.Series,
    low: Sequence[float] | np.ndarray | pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Parkinson (1980) range estimator from daily highs and lows.

    ``sigma^2 = ppy / (4 ln 2) * mean( ln(H/L)^2 )``

    ~5x more efficient than close-to-close under driftless GBM, but biased
    *down* when the market is closed part of the day. For FX the 24h weekday
    session makes the range estimator unusually well-suited (no overnight gap
    Mon-Fri); the weekend gap is the residual bias (documented in
    docs/VALIDATION.md).
    """
    h = _as_1d_array(high, "high")
    l = _as_1d_array(low, "low")
    if h.shape != l.shape:
        raise ValueError("high and low must have equal length")
    if h.size < _MIN_OBS:
        raise ValueError(f"need at least {_MIN_OBS} observations, got {h.size}")
    if (l <= 0).any():
        raise ValueError("low prices must be strictly positive")
    if (h < l).any():
        raise ValueError("high < low encountered -- corrupt bar data")
    hl = np.log(h / l)
    var = np.mean(hl ** 2) / (4.0 * np.log(2.0)) * periods_per_year
    return float(np.sqrt(var))


def garman_klass_vol(
    open_: Sequence[float] | np.ndarray | pd.Series,
    high: Sequence[float] | np.ndarray | pd.Series,
    low: Sequence[float] | np.ndarray | pd.Series,
    close: Sequence[float] | np.ndarray | pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Garman-Klass (1980) OHLC estimator.

    ``sigma^2 = ppy * mean( 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2 )``

    ~7.4x efficiency vs close-to-close under driftless GBM; same weekend-gap
    caveat as Parkinson for FX.
    """
    o = _as_1d_array(open_, "open")
    h = _as_1d_array(high, "high")
    l = _as_1d_array(low, "low")
    c = _as_1d_array(close, "close")
    if not (o.shape == h.shape == l.shape == c.shape):
        raise ValueError("open/high/low/close must have equal length")
    if o.size < _MIN_OBS:
        raise ValueError(f"need at least {_MIN_OBS} observations, got {o.size}")
    if (l <= 0).any() or (o <= 0).any():
        raise ValueError("prices must be strictly positive")
    if (h < l).any():
        raise ValueError("high < low encountered -- corrupt bar data")
    hl = np.log(h / l)
    co = np.log(c / o)
    per_day = 0.5 * hl ** 2 - (2.0 * np.log(2.0) - 1.0) * co ** 2
    var = np.mean(per_day) * periods_per_year
    if var < 0:
        raise ValueError("Garman-Klass variance estimate is negative -- check bar data")
    return float(np.sqrt(var))


def day_of_week_vol_factors(returns: pd.Series, min_obs_per_day: int = 5) -> pd.Series:
    """Estimate multiplicative day-of-week volatility factors for FX.

    FX volatility has a pronounced weekly pattern: Mondays open with the
    weekend gap plus Wellington/Sydney illiquidity, mid-week days carry the
    bulk of US data releases (FOMC Wednesdays, CPI), Fridays include NFP.
    This estimator computes, per weekday d,

        factor_d = std(returns on weekday d) / rms of the per-day stds

    normalized so that ``mean(factor_d^2) = 1`` across the weekdays present
    (i.e. the factors redistribute variance without changing the average
    variance level). Use it to de-seasonalize returns before GARCH fitting
    (``r_t / factor_{d(t)}``) or to scale forecasts to a target day.

    Parameters
    ----------
    returns : pandas.Series
        Log returns with a DatetimeIndex (weekdays only; weekend stamps raise).
    min_obs_per_day : int
        Minimum observations required for each weekday present.

    Returns
    -------
    pandas.Series
        Factors indexed by weekday name (Monday..Friday), unit r.m.s.
    """
    if not isinstance(returns, pd.Series) or not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns must be a pandas Series with a DatetimeIndex")
    _as_1d_array(returns.to_numpy(), "returns")
    dow = returns.index.dayofweek
    if (dow >= 5).any():
        raise ValueError("weekend timestamps found -- FX daily series should be Mon-Fri only")
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    stds: dict[str, float] = {}
    for d in sorted(np.unique(dow)):
        sub = returns[dow == d]
        if sub.size < min_obs_per_day:
            raise ValueError(
                f"only {sub.size} observations for {names[d]}, need >= {min_obs_per_day}"
            )
        stds[names[d]] = float(sub.std(ddof=1))
    vals = np.array(list(stds.values()))
    if (vals == 0).all():
        raise ValueError("returns are constant -- day-of-week factors undefined")
    rms = np.sqrt(np.mean(vals ** 2))
    return pd.Series({k: v / rms for k, v in stds.items()}, name="dow_vol_factor")
