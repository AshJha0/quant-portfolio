"""Shared input checks (spec S1 and S12).

Every public entry point refuses NaN, +inf, -inf, empty input and
out-of-domain values with ``ValueError``. It never returns a number from bad
input.
"""

from __future__ import annotations

import numbers

import numpy as np

ALPHA_HINT = "alpha is the tail probability (e.g. 0.01 for 99 %)"


def validate_alpha(alpha: object) -> float:
    """Return ``alpha`` as a float, or raise unless it lies in (0, 0.5)."""
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
        raise ValueError(f"alpha must be a real number in (0, 0.5); {ALPHA_HINT}; got {alpha!r}")
    a = float(alpha)
    if not (np.isfinite(a) and 0.0 < a < 0.5):
        raise ValueError(f"alpha must be in (0, 0.5); {ALPHA_HINT}; got {alpha!r}")
    return a


def finite_scalar(name: str, value: object, *, min_value: float | None = None,
                  strict: bool = False) -> float:
    """Finite real scalar, optionally ``>= min_value`` (``> min_value`` if strict)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if min_value is not None:
        if strict and not v > min_value:
            raise ValueError(f"{name} must be > {min_value}, got {value!r}")
        if not strict and not v >= min_value:
            raise ValueError(f"{name} must be >= {min_value}, got {value!r}")
    return v


def integer(name: str, value: object, *, min_value: int = 1) -> int:
    """Integer (not bool, not a float) that is ``>= min_value``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    v = int(value)
    if v < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value!r}")
    return v


def finite_array(name: str, x: object, *, min_size: int = 1, ndim: int | None = None) -> np.ndarray:
    """Float array with every entry finite and at least ``min_size`` entries."""
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric: {exc}") from None
    if ndim == 1:
        arr = arr.ravel()
    elif ndim == 2:
        arr = np.atleast_2d(arr)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    if arr.size < min_size:
        raise ValueError(f"{name} needs at least {min_size} entries, got {arr.size}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr
