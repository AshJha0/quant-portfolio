"""Shared non-finite input rejection for :mod:`fx_pairs`.

Why this module exists
----------------------
Two failure patterns recur across this package and both end in a *silent*
wrong answer rather than an exception:

1. **Inequality-only guards.**  ``if sigma <= 0: raise``,
   ``if stop <= entry: raise``, ``if notional <= 0: raise`` — every
   comparison against NaN is ``False``, so NaN passes the guard.  A NaN
   ``sigma`` gives an all-NaN z-score and the strategy trades nothing; a NaN
   ``stop`` disables the hard stop without saying so — precisely the control
   that exists to survive a regime break (the SNB floor case study).

2. **``isnan``-only guards.**  Several series validators tested
   ``np.isnan(x).any()`` and therefore accepted **±Inf**.  Inf is the
   realistic corruption here, not NaN: a zero or missing price becomes
   ``-inf`` the moment it is passed through ``log``.  Inf then flows into the
   OLS design matrix, ``lstsq`` returns NaN coefficients, the ADF statistic
   is NaN, and ``bool(nan < critical_value)`` is ``False`` — so
   :func:`fx_pairs.engle_granger` reports "not cointegrated" on data it could
   not test at all.

Use :func:`require_finite` for scalars and :func:`finite_series` for arrays.
"""

from __future__ import annotations

import numpy as np

__all__ = ["require_finite", "finite_series"]


def require_finite(**values) -> None:
    """Raise ``ValueError`` if any named scalar/array value is NaN or Inf.

    ``None`` values are skipped so optional arguments pass straight through.

    Raises
    ------
    ValueError
        Naming the first non-finite argument found.
    """
    for name, value in values.items():
        if value is None:
            continue
        if not np.all(np.isfinite(np.asarray(value, dtype=float))):
            raise ValueError(f"{name} must be finite, got {value!r}")


def finite_series(values, name: str, *, positive: bool = False) -> np.ndarray:
    """Coerce to a finite 1-D float array, raising on NaN/Inf.

    Parameters
    ----------
    values : array-like
        Input series.
    name : str
        Name used in error messages.
    positive : bool
        Also require every element to be strictly positive (price levels,
        which are logged downstream — a non-positive price would become
        ``-inf`` or ``nan`` inside ``log``).

    Returns
    -------
    numpy.ndarray
        1-D float array.

    Raises
    ------
    ValueError
        On non-1-D input, on NaN/Inf, or on a non-positive value when
        ``positive`` is set.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got ndim={arr.ndim}")
    if not np.isfinite(arr).all():
        raise ValueError(
            f"{name} contains NaN or infinite values; clean the series first "
            "(fx_pairs rejects rather than imputing)"
        )
    if positive and np.any(arr <= 0.0):
        raise ValueError(
            f"{name} must be strictly positive (FX price levels are logged "
            f"downstream), found min={arr.min()}"
        )
    return arr
