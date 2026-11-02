"""Input checks shared by the Round 3 risk modules.

These modules are asset-agnostic and ship as identical files in ``eq_var``
and ``fx_var``, so they carry their own small checks instead of importing a
package-specific validation module. Same contract as the rest of the
package: NaN, infinities, empty input and out-of-domain values raise
``ValueError``; nothing returns a number from bad input.
"""

from __future__ import annotations

import numbers

import numpy as np


def alpha_(alpha: object, *, upper: float = 0.5) -> float:
    """Tail probability in (0, upper)."""
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
        raise ValueError(f"alpha must be a real number in (0, {upper}), got {alpha!r}")
    a = float(alpha)
    if not (np.isfinite(a) and 0.0 < a < upper):
        raise ValueError(f"alpha must be in (0, {upper}), got {alpha!r}")
    return a


def scalar(name: str, value: object, *, lo: float | None = None, strict: bool = False) -> float:
    """Finite real scalar, optionally ``>= lo`` (``> lo`` when strict)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if lo is not None and (v <= lo if strict else v < lo):
        raise ValueError(f"{name} must be {'>' if strict else '>='} {lo}, got {value!r}")
    return v


def count(name: str, value: object, *, lo: int = 1) -> int:
    """Integer (not bool) that is ``>= lo``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if int(value) < lo:
        raise ValueError(f"{name} must be >= {lo}, got {value!r}")
    return int(value)


def array(name: str, x: object, *, ndim: int = 1, min_size: int = 1) -> np.ndarray:
    """Finite float array of the given rank with at least ``min_size`` entries."""
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric: {exc}") from None
    if ndim == 1 and arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {arr.shape}")
    if arr.size < min_size:
        raise ValueError(f"{name} needs at least {min_size} entries, got {arr.size}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def correlation(name: str, c: object) -> np.ndarray:
    """Symmetric positive-definite correlation matrix with unit diagonal."""
    m = array(name, c, ndim=2)
    if m.shape[0] != m.shape[1]:
        raise ValueError(f"{name} must be square, got {m.shape}")
    if not np.allclose(m, m.T, atol=1e-12) or not np.allclose(np.diag(m), 1.0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric with unit diagonal")
    if np.linalg.eigvalsh(m).min() <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    return m


def loss_quantile(pnl: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical (VaR, ES) of a P&L vector, losses positive.

    Same estimator as ``expected_shortfall`` in this package: with losses
    sorted descending and ``k = alpha n``, VaR is the ``ceil(k)``-th worst
    loss and ES integrates the empirical quantile over ``(0, alpha]`` with a
    fractional weight on the boundary order statistic.
    """
    losses = np.sort(-pnl)[::-1]
    k = alpha * losses.size
    m = max(1, int(np.ceil(k - 1e-9)))
    var = float(losses[m - 1])
    es = float((losses[: m - 1].sum() + (k - (m - 1)) * losses[m - 1]) / k)
    return var, es
