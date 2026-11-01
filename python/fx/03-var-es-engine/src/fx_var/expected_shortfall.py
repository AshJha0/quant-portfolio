"""Expected Shortfall (ES / CVaR) and the shared tail-quantile machinery.

Conventions
-----------
* ``alpha`` is the tail probability: 0.01 means 99 % VaR.  Every function
  rejects ``alpha`` outside (0, 0.5).  The keyword ``confidence=`` (0.99)
  still works for one release with a ``DeprecationWarning``.
* P&L arrays are profit (+) / loss (-).  VaR and ES are reported as
  positive loss amounts in the book's base currency.
* Empirical VaR (uniform weights): sort losses ``L = -pnl`` descending,
  ``k = n alpha``, ``m = ceil(k - 1e-9)`` clamped to ``[1, n]``,
  ``VaR = L(m)``.  This is the order statistic, with no interpolation.
* Empirical ES (Acerbi-Tasche, exact fractional atom):
  ``ES = [sum_{i<m} L(i) + (k - (m-1)) L(m)] / k``.  When ``k`` is an
  integer this is the mean of the ``k`` worst losses.
* Weighted samples (age-weighted HS): weights are normalised to sum 1 and
  sorted with the losses (stable sort).  ``W_j`` is the cumulative weight,
  ``m`` the first ``j`` with ``W_j >= alpha - 1e-12``, ``VaR = L(m)`` and
  ``ES = [sum_{i<m} w_i L(i) + (alpha - W_{m-1}) L(m)] / alpha``.
* ``ES >= VaR`` holds for every sample.  The estimator integrates the
  empirical quantile function over exactly ``alpha`` of probability mass.
  On the empirical distribution of a fixed set of scenarios it is a
  coherent risk measure, including when the sample has atoms.
* Closed forms: for Normal(mu, sigma) P&L, ``VaR = -mu + sigma z`` and
  ``ES = -mu + sigma phi(z)/alpha`` with ``z = Phi^{-1}(1 - alpha)``.  For
  Student-t the unit-variance t is used, so ``sigma`` is the actual P&L
  standard deviation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as student_t

from .common import resolve_alpha, validate_alpha, validate_finite

__all__ = [
    "empirical_var",
    "empirical_es",
    "empirical_var_es",
    "normal_var",
    "normal_es",
    "t_var",
    "t_es",
]

_W_TOL = 1e-12
_K_TOL = 1e-9


def _tail(pnl, alpha: float, weights):
    """Return (losses sorted descending, sorted weights or None, index m-1).

    ``weights`` comes back as None in the uniform case, so the ES can use
    the exact ``k = n alpha`` formula instead of a floating cumulative sum.
    """
    alpha = validate_alpha(alpha)
    pnl = np.asarray(pnl, dtype=float).ravel()
    if pnl.size == 0:
        raise ValueError("pnl sample is empty")
    if not np.isfinite(pnl).all():
        raise ValueError(
            "pnl sample contains NaN or infinite values (NaN policy: refuse)"
        )
    losses = -pnl
    n = losses.size
    if weights is None:
        losses_d = np.sort(losses)[::-1]
        return losses_d, None, tail_rank(n, alpha) - 1
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != n:
        raise ValueError("weights must match pnl length")
    if not np.isfinite(w).all():
        raise ValueError("weights contain NaN or infinite values")
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative and sum > 0")
    w = w / w.sum()
    order = np.argsort(-losses, kind="stable")
    losses_d = losses[order]
    w_d = w[order]
    cum = np.cumsum(w_d)
    idx = int(np.searchsorted(cum, alpha - _W_TOL, side="left"))
    return losses_d, w_d, min(idx, n - 1)


def tail_rank(n: int, alpha: float) -> int:
    """1-based rank ``m = ceil(n alpha - 1e-9)`` of the VaR order statistic."""
    m = int(np.ceil(n * alpha - _K_TOL))
    return min(max(m, 1), n)


def _es_from_tail(losses_d: np.ndarray, w_d, idx: int, alpha: float) -> float:
    """Acerbi-Tasche ES over exactly ``alpha`` of probability mass."""
    if w_d is None:
        k = losses_d.size * alpha
        full = float(np.sum(losses_d[:idx]))
        return (full + (k - idx) * float(losses_d[idx])) / k
    full = float(np.sum(losses_d[:idx] * w_d[:idx]))
    cum_before = float(np.sum(w_d[:idx]))
    frac = max(alpha - cum_before, 0.0)
    return (full + frac * float(losses_d[idx])) / alpha


def empirical_var(pnl, alpha: float | None = None, weights=None, *,
                  confidence: float | None = None) -> float:
    """Empirical VaR (positive loss) at tail probability ``alpha``.

    Parameters
    ----------
    pnl : array_like
        Scenario P&L (profit positive).
    alpha : float
        Tail probability in (0, 0.5).  Default 0.01 (99 % VaR).
    weights : array_like, optional
        Non-negative scenario weights (age-weighted HS), normalised
        internally.  Uniform if omitted.
    confidence : float, optional
        Deprecated.  ``alpha = 1 - confidence``.
    """
    alpha = resolve_alpha(alpha, confidence)
    losses_d, _, idx = _tail(pnl, alpha, weights)
    return float(losses_d[idx])


def empirical_es(pnl, alpha: float | None = None, weights=None, *,
                 confidence: float | None = None) -> float:
    """Empirical ES (positive loss), Acerbi-Tasche estimator."""
    alpha = resolve_alpha(alpha, confidence)
    losses_d, w_d, idx = _tail(pnl, alpha, weights)
    return float(_es_from_tail(losses_d, w_d, idx, alpha))


def empirical_var_es(pnl, alpha: float | None = None, weights=None, *,
                     confidence: float | None = None) -> tuple[float, float]:
    """Return ``(VaR, ES)`` from one pass over the sample."""
    alpha = resolve_alpha(alpha, confidence)
    losses_d, w_d, idx = _tail(pnl, alpha, weights)
    return float(losses_d[idx]), float(_es_from_tail(losses_d, w_d, idx, alpha))


def _check_sigma(sigma: float, mean: float) -> None:
    validate_finite(sigma=sigma, mean=mean)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")


def normal_var(sigma: float, alpha: float | None = None, mean: float = 0.0, *,
               confidence: float | None = None) -> float:
    """Closed-form Normal VaR: ``-mean + sigma * Phi^{-1}(1 - alpha)``."""
    alpha = resolve_alpha(alpha, confidence)
    _check_sigma(sigma, mean)
    return float(-mean + sigma * norm.isf(alpha))


def normal_es(sigma: float, alpha: float | None = None, mean: float = 0.0, *,
              confidence: float | None = None) -> float:
    """Closed-form Normal ES: ``-mean + sigma phi(z) / alpha``."""
    alpha = resolve_alpha(alpha, confidence)
    _check_sigma(sigma, mean)
    z = norm.isf(alpha)
    return float(-mean + sigma * norm.pdf(z) / alpha)


def _t_scale(df: float) -> float:
    validate_finite(df=df)
    if df <= 2:
        raise ValueError("Student-t df must be > 2 for finite variance")
    return np.sqrt((df - 2.0) / df)


def t_var(sigma: float, alpha: float | None = None, df: float = 6.0,
          mean: float = 0.0, *, confidence: float | None = None) -> float:
    """Unit-variance Student-t VaR with P&L standard deviation ``sigma``.

    The scaling ``sqrt((df-2)/df)`` makes ``sigma`` the actual standard
    deviation, so normal and t VaR are comparable at equal risk.
    """
    alpha = resolve_alpha(alpha, confidence)
    _check_sigma(sigma, mean)
    scale = _t_scale(df)
    q = student_t.isf(alpha, df)
    return float(-mean + sigma * scale * q)


def t_es(sigma: float, alpha: float | None = None, df: float = 6.0,
         mean: float = 0.0, *, confidence: float | None = None) -> float:
    """Unit-variance Student-t ES (closed form).

    For the standard t, ``E[X | X > q] = f(q) (df + q^2) / (alpha (df - 1))``
    with ``q`` the upper ``alpha`` quantile.
    """
    alpha = resolve_alpha(alpha, confidence)
    _check_sigma(sigma, mean)
    scale = _t_scale(df)
    q = student_t.isf(alpha, df)
    es_std = student_t.pdf(q, df) * (df + q**2) / (alpha * (df - 1.0))
    return float(-mean + sigma * scale * es_std)
