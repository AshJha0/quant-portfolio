"""Shared non-finite input rejection for :mod:`fx_pairs`.

Why this module exists
----------------------
Two failure patterns recur across this package and both end in a *silent*
wrong answer rather than an exception:

1. **Inequality-only guards.**  ``if sigma <= 0: raise``,
   ``if stop <= entry: raise``, ``if notional <= 0: raise``: every
   comparison against NaN is ``False``, so NaN passes the guard.  A NaN
   ``sigma`` gives an all-NaN z-score and the strategy trades nothing; a NaN
   ``stop`` disables the hard stop without saying so: precisely the control
   that exists to survive a regime break (the SNB floor case study).

2. **``isnan``-only guards.**  Several series validators tested
   ``np.isnan(x).any()`` and therefore accepted **±Inf**.  Inf is the
   realistic corruption here, not NaN: a zero or missing price becomes
   ``-inf`` the moment it is passed through ``log``.  Inf then flows into the
   OLS design matrix, ``lstsq`` returns NaN coefficients, the ADF statistic
   is NaN, and ``bool(nan < critical_value)`` is ``False``: so
   :func:`fx_pairs.engle_granger` reports "not cointegrated" on data it could
   not test at all.

Use :func:`require_finite` for scalars and :func:`finite_series` for arrays.
Both are built on :mod:`qp_core.validation`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qp_core.validation import finite_series as _finite_series
from qp_core.validation import require_finite

__all__ = ["finite_series", "require_finite"]


def finite_series(values: ArrayLike, name: str, *, positive: bool = False) -> NDArray[np.float64]:
    """Coerce to a finite 1-D float array, raising on NaN/Inf.

    Parameters
    ----------
    values : array-like
        Input series.
    name : str
        Name used in error messages.
    positive : bool
        Also require every element to be strictly positive (price levels,
        which are logged downstream: a non-positive price would become
        ``-inf`` or ``nan`` inside ``log``).

    Raises
    ------
    ValueError
        On non-1-D input, on NaN/Inf, or on a non-positive value when
        ``positive`` is set.
    """
    return _finite_series(
        values,
        name,
        positive=positive,
        nonfinite_hint="; clean the series first (fx_pairs rejects rather than imputing)",
        positive_reason=" (FX price levels are logged downstream)",
    )
