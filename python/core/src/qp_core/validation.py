"""Finite-input guards.

An inequality guard such as ``if sigma <= 0: raise`` lets NaN through,
because every comparison with NaN is False. An ``isnan`` guard lets +-Inf
through, and Inf is the usual corruption (``log`` of a zero price). Both end
in a plausible-looking wrong number instead of an exception.

Call :func:`require_finite` (scalars or arrays) or :func:`finite_series`
(1-D arrays) before the inequality checks, so bad input raises a named
``ValueError``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["finite_scalar", "finite_series", "require_finite"]


def require_finite(**values: object) -> None:
    """Raise ``ValueError`` if any named scalar or array value is NaN or Inf.

    ``None`` values are skipped, so optional arguments pass straight through.

    Examples
    --------
    >>> require_finite(spot=1.085, tau=0.25)
    >>> require_finite(spot=float("nan"))
    Traceback (most recent call last):
        ...
    ValueError: spot must be finite, got nan
    """
    for name, value in values.items():
        if value is None:
            continue
        if not np.all(np.isfinite(np.asarray(value, dtype=float))):
            raise ValueError(f"{name} must be finite, got {value!r}")


def finite_scalar(name: str, value: float) -> float:
    """Return ``float(value)``; raise ``ValueError`` if it is NaN or Inf."""
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def finite_series(
    values: ArrayLike,
    name: str,
    *,
    positive: bool = False,
    nonfinite_hint: str = "",
    positive_reason: str = "",
) -> NDArray[np.float64]:
    """Coerce to a finite 1-D float array, raising on NaN or Inf.

    Parameters
    ----------
    values : array-like
        Input series.
    name : str
        Name used in error messages.
    positive : bool
        Also require every element to be strictly positive (for example price
        levels that are logged downstream).
    nonfinite_hint, positive_reason : str
        Text appended to the two error messages, so a caller can say what to
        do or why the rule exists.

    Raises
    ------
    ValueError
        On non-1-D input, on NaN or Inf, or on a non-positive value when
        ``positive`` is set.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got ndim={arr.ndim}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values{nonfinite_hint}")
    if positive and np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive{positive_reason}, found min={arr.min()}")
    return arr
