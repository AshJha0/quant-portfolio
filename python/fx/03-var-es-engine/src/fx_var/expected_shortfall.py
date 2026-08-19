"""Expected Shortfall (ES / CVaR) and the shared tail-quantile machinery.

Conventions
-----------
* P&L arrays are profit (+) / loss (-); VaR and ES are reported as
  **positive loss amounts** in the book's base currency.
* Empirical VaR at level ``alpha`` on ``n`` scenarios with weights ``w``
  (uniform by default) is the order-statistic / inverse-ECDF quantile:
  sort losses descending, accumulate weights, VaR = the loss at which the
  cumulative tail weight first reaches ``1 - alpha``.  With uniform weights
  this is the ``m``-th worst loss, ``m = ceil(n * (1 - alpha))``.
* Empirical ES uses the Acerbi-Tasche tail-splitting estimator: the worst
  losses are averaged over *exactly* ``1 - alpha`` of probability mass,
  taking only a fractional share of the atom at the VaR level:
  ``ES = [sum_{i<m} w_i L_i + (1-alpha - W_{m-1}) L_m] / (1-alpha)``.
  With uniform weights and integer ``n (1-alpha)`` this reduces to the
  mean of the ``m`` worst losses.  This is the *coherent* (subadditive)
  estimator - the naive "mean of observations beyond VaR" over-weights
  probability atoms and can spuriously break subadditivity exactly in the
  pegged-currency jump case this project cares about.  ``ES >= VaR`` holds
  for every sample.
* Closed forms: for Normal(mu, sigma) P&L,
  ``VaR = -mu + sigma z_a`` and ``ES = -mu + sigma phi(z_a)/(1-a)``.
  For Student-t the *standardised* (unit-variance) t is used so ``sigma``
  is always the true P&L standard deviation.

ES is the Basel FRTB headline measure precisely because, unlike VaR, it is
subadditive (coherent) and sees tail *sizes* - both properties matter for
FX books holding pegged currencies (see tests/test_expected_shortfall.py
for a VaR subadditivity counterexample built from peg-jump assets).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as student_t

from .common import validate_alpha, validate_finite

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


def _tail(pnl: np.ndarray, alpha: float, weights: np.ndarray | None):
    """Return (sorted losses desc, weights desc, tail index m) for level alpha."""
    validate_alpha(alpha)
    pnl = np.asarray(pnl, dtype=float).ravel()
    if pnl.size == 0:
        raise ValueError("pnl sample is empty")
    if np.isnan(pnl).any():
        raise ValueError("pnl sample contains NaNs (NaN policy: refuse)")
    losses = -pnl
    if weights is None:
        w = np.full(losses.size, 1.0 / losses.size)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != losses.size:
            raise ValueError("weights must match pnl length")
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative and sum > 0")
        w = w / w.sum()
    order = np.argsort(-losses, kind="stable")
    losses_d = losses[order]
    w_d = w[order]
    cum = np.cumsum(w_d)
    target = 1.0 - alpha
    idx = int(np.searchsorted(cum, target - _W_TOL, side="left"))
    idx = min(idx, losses_d.size - 1)
    return losses_d, w_d, idx


def empirical_var(pnl, alpha: float = 0.99, weights=None) -> float:
    """Empirical VaR (positive loss) at level ``alpha``.

    Parameters
    ----------
    pnl : array_like
        Scenario P&L (profit positive).
    alpha : float
        Confidence level in (0, 1), e.g. 0.99.
    weights : array_like, optional
        Non-negative scenario weights (age-weighted HS); normalised
        internally.  Uniform if omitted.
    """
    losses_d, _, idx = _tail(pnl, alpha, weights)
    return float(losses_d[idx])


def _es_from_tail(losses_d: np.ndarray, w_d: np.ndarray, idx: int, alpha: float) -> float:
    """Acerbi-Tasche ES: average the worst losses over exactly 1-alpha mass."""
    target = 1.0 - alpha
    full = float(np.sum(losses_d[:idx] * w_d[:idx]))  # fully included obs
    cum_before = float(np.sum(w_d[:idx]))
    frac = max(target - cum_before, 0.0)  # partial share of the VaR atom
    return (full + frac * float(losses_d[idx])) / target


def empirical_es(pnl, alpha: float = 0.99, weights=None) -> float:
    """Empirical ES (positive loss): coherent tail-splitting estimator."""
    losses_d, w_d, idx = _tail(pnl, alpha, weights)
    return float(_es_from_tail(losses_d, w_d, idx, alpha))


def empirical_var_es(pnl, alpha: float = 0.99, weights=None) -> tuple[float, float]:
    """Return ``(VaR, ES)`` from one pass over the sample."""
    losses_d, w_d, idx = _tail(pnl, alpha, weights)
    var = float(losses_d[idx])
    es = float(_es_from_tail(losses_d, w_d, idx, alpha))
    return var, es


def normal_var(sigma: float, alpha: float = 0.99, mean: float = 0.0) -> float:
    """Closed-form Normal VaR: ``-mean + sigma * z_alpha`` (positive loss)."""
    validate_alpha(alpha)
    validate_finite(sigma=sigma, mean=mean)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    return float(-mean + sigma * norm.ppf(alpha))


def normal_es(sigma: float, alpha: float = 0.99, mean: float = 0.0) -> float:
    """Closed-form Normal ES: ``-mean + sigma * phi(z_alpha)/(1-alpha)``."""
    validate_alpha(alpha)
    validate_finite(sigma=sigma, mean=mean)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    z = norm.ppf(alpha)
    return float(-mean + sigma * norm.pdf(z) / (1.0 - alpha))


def _t_scale(df: float) -> float:
    validate_finite(df=df)
    if df <= 2:
        raise ValueError("Student-t df must be > 2 for finite variance")
    return np.sqrt((df - 2.0) / df)


def t_var(sigma: float, alpha: float = 0.99, df: float = 6.0, mean: float = 0.0) -> float:
    """Standardised Student-t VaR with true P&L std ``sigma``.

    Uses the unit-variance scaling ``sqrt((df-2)/df)`` so that ``sigma`` is
    the actual standard deviation, making normal and t VaR directly
    comparable at equal risk.
    """
    validate_alpha(alpha)
    validate_finite(sigma=sigma, mean=mean)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    scale = _t_scale(df)  # validates df before any t special function
    q = student_t.ppf(alpha, df)
    return float(-mean + sigma * scale * q)


def t_es(sigma: float, alpha: float = 0.99, df: float = 6.0, mean: float = 0.0) -> float:
    """Standardised Student-t ES (closed form).

    For standard t, ``E[X | X > q_a] = f(q_a) (df + q_a^2) / ((1-a)(df-1))``.
    """
    validate_alpha(alpha)
    validate_finite(sigma=sigma, mean=mean)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    scale = _t_scale(df)  # validates df before any t special function
    q = student_t.ppf(alpha, df)
    es_std = student_t.pdf(q, df) * (df + q**2) / ((1.0 - alpha) * (df - 1.0))
    return float(-mean + sigma * scale * es_std)
