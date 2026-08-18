"""Historical (realized) volatility estimators.

Close-to-close rolling volatility plus three range-based estimators. All
volatilities are annualised on log-returns with ``sqrt(252)`` (package
convention; see ``_utils``).

Efficiency discussion
---------------------
For a driftless GBM observed continuously, the relative sampling variances of
one-day variance estimates are approximately:

* close-to-close (squared return): efficiency 1.0 (baseline);
* Parkinson (high-low): ~4.9x more efficient — the daily range contains far
  more information about diffusion variance than the single close;
* Garman-Klass (OHLC): ~7.4x more efficient — adds open/close information;
* Rogers-Satchell (OHLC): ~6x more efficient *and* drift-robust — Parkinson
  and Garman-Klass are biased upward under non-zero drift, Rogers-Satchell is
  unbiased for any drift.

Caveats: all range estimators assume continuous monitoring (discrete trading
biases the observed range down), no overnight gaps (Garman-Klass has a
gap-adjusted variant), and no microstructure noise (bid-ask bounce inflates
the observed high-low range). See docs/METHODOLOGY.md.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ._utils import TRADING_DAYS, validate_returns

__all__ = [
    "realized_vol",
    "close_to_close_var",
    "parkinson_var",
    "garman_klass_var",
    "rogers_satchell_var",
    "range_vol",
    "window_sensitivity",
]


def _ann_vol(daily_var: np.ndarray, annualization: int) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.sqrt(daily_var * annualization)


def close_to_close_var(returns: np.ndarray, demean: bool = False) -> np.ndarray:
    """Per-day variance proxy from close-to-close log-returns.

    ``r_t^2`` (or ``(r_t - rbar)^2`` if ``demean``). For daily equity data the
    mean is tiny relative to the vol (|mu| dt << sigma sqrt(dt)) so the
    zero-mean convention is standard and is the default.
    """
    r = validate_returns(returns, min_obs=1)
    if demean:
        r = r - r.mean()
    return r**2


def realized_vol(
    returns: Sequence[float] | np.ndarray,
    window: int = 21,
    annualization: int = TRADING_DAYS,
    demean: bool = False,
) -> np.ndarray:
    """Rolling close-to-close realized volatility, annualised.

    vol_t = sqrt( annualization * mean(r_{t-window+1..t}^2) ).

    Parameters
    ----------
    returns : array-like
        Daily log-returns, decimal units.
    window : int
        Rolling window length in days (21 ~ one trading month).
    annualization : int
        Periods per year (252 for daily data).
    demean : bool
        Subtract the *rolling* mean before squaring. Default False (zero-mean
        convention, standard for daily data).

    Returns
    -------
    numpy.ndarray
        Same length as input; first ``window - 1`` entries are NaN.
    """
    r = validate_returns(returns, min_obs=window, name="returns")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    s = pd.Series(r)
    if demean:
        var = s.rolling(window).apply(lambda x: np.mean((x - x.mean()) ** 2), raw=True)
    else:
        var = (s**2).rolling(window).mean()
    return _ann_vol(var.to_numpy(), annualization)


def _validate_ohlc(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    o, h, l, c = (np.asarray(x, dtype=float).ravel() for x in (open_, high, low, close))
    if not (o.size == h.size == l.size == c.size):
        raise ValueError("open/high/low/close must have equal length")
    if o.size == 0:
        raise ValueError("empty OHLC input")
    for name, x in (("open", o), ("high", h), ("low", l), ("close", c)):
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{name} contains NaN/inf values; clean the data first")
        if np.any(x <= 0):
            raise ValueError(f"{name} contains non-positive prices")
    if np.any(h < np.maximum.reduce([o, c, l])) or np.any(l > np.minimum.reduce([o, c, h])):
        raise ValueError("inconsistent OHLC bars: need low <= open,close <= high")
    return o, h, l, c


def parkinson_var(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Parkinson (1980) per-day variance from the high-low range.

    var_t = ln(H_t/L_t)^2 / (4 ln 2).

    ~4.9x more efficient than the squared close-to-close return under
    driftless GBM; biased upward by drift and downward by discrete monitoring.
    """
    h = np.asarray(high, dtype=float).ravel()
    l = np.asarray(low, dtype=float).ravel()
    _validate_ohlc(l, h, l, h)  # reuse checks: low<=high, positive, finite
    return np.log(h / l) ** 2 / (4.0 * np.log(2.0))


def garman_klass_var(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Garman-Klass (1980) per-day variance from OHLC.

    var_t = 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2.

    ~7.4x more efficient than close-to-close under driftless GBM with no
    opening gaps; assumes zero drift and continuous trading.
    """
    o, h, l, c = _validate_ohlc(open_, high, low, close)
    return 0.5 * np.log(h / l) ** 2 - (2.0 * np.log(2.0) - 1.0) * np.log(c / o) ** 2


def rogers_satchell_var(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Rogers-Satchell (1991) per-day variance from OHLC.

    var_t = ln(H/C) ln(H/O) + ln(L/C) ln(L/O).

    Drift-independent (unbiased for any constant drift), unlike Parkinson and
    Garman-Klass; ~6x more efficient than close-to-close.
    """
    o, h, l, c = _validate_ohlc(open_, high, low, close)
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


def range_vol(
    ohlc: pd.DataFrame,
    estimator: str = "parkinson",
    window: int = 21,
    annualization: int = TRADING_DAYS,
) -> np.ndarray:
    """Rolling annualised volatility from a range estimator.

    Parameters
    ----------
    ohlc : pandas.DataFrame
        Columns ``open, high, low, close`` (case-insensitive).
    estimator : {"parkinson", "garman_klass", "rogers_satchell"}
    window : int
        Rolling mean window for the per-day variances.
    """
    cols = {c.lower(): c for c in ohlc.columns}
    try:
        o, h, l, c = (ohlc[cols[k]].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    except KeyError as exc:
        raise ValueError("ohlc must contain open/high/low/close columns") from exc
    if estimator == "parkinson":
        dv = parkinson_var(h, l)
    elif estimator == "garman_klass":
        dv = garman_klass_var(o, h, l, c)
    elif estimator == "rogers_satchell":
        dv = rogers_satchell_var(o, h, l, c)
    else:
        raise ValueError(
            f"unknown estimator {estimator!r}; use 'parkinson', 'garman_klass' "
            f"or 'rogers_satchell'"
        )
    if window < 1 or window > dv.size:
        raise ValueError(f"window must be in [1, {dv.size}], got {window}")
    var = pd.Series(dv).rolling(window).mean().to_numpy()
    return _ann_vol(var, annualization)


def window_sensitivity(
    returns: Sequence[float] | np.ndarray,
    windows: Sequence[int] = (10, 21, 63, 126, 252),
    annualization: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Window-sensitivity study for rolling realized vol.

    Short windows react fast but are noisy (sampling std of the vol estimate
    ~ sigma / sqrt(2 * window)); long windows are smooth but lag regime
    changes. This utility quantifies the trade-off on a given series.

    Returns
    -------
    pandas.DataFrame
        One row per window: latest vol estimate, full-sample mean/std of the
        rolling estimate, and the approximate theoretical sampling std.
    """
    r = validate_returns(returns, min_obs=max(windows))
    rows = []
    for w in windows:
        vol = realized_vol(r, window=w, annualization=annualization)
        valid = vol[~np.isnan(vol)]
        sigma_hat = float(np.sqrt(np.mean(r**2) * annualization))
        rows.append(
            {
                "window": w,
                "latest_vol": float(vol[-1]),
                "mean_vol": float(valid.mean()),
                "std_of_estimate": float(valid.std()),
                "approx_sampling_std": sigma_hat / np.sqrt(2.0 * w),
            }
        )
    return pd.DataFrame(rows).set_index("window")
