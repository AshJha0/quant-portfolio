"""Covariance estimators for the research portfolio layer.

* ``sample_cov``: the unbiased sample covariance (ddof = 1).
* ``ledoit_wolf``: shrink towards ``mu * I`` with the optimal intensity of
  Ledoit and Wolf (2004), "A well-conditioned estimator for large-dimensional
  covariance matrices", J. Multivariate Anal. 88, 365-411, eqs. in Sec. 3.3
  (the same estimator as ``sklearn.covariance.ledoit_wolf``, which uses the
  biased 1/T sample covariance; we follow that convention so the two can be
  compared to machine precision).
* ``pca_cov``: a k-factor model from the top principal components of the
  sample covariance plus a diagonal of residual variances, so the diagonal
  matches the sample exactly (statistical factor model; Connor and Korajczyk
  1986).

Worked example: two assets with sample variances 4 and 1 and correlation
0.5 shrink towards ``2.5 I``; the off-diagonal moves towards 0 by the
intensity ``delta`` and the diagonal towards 2.5.
"""

from __future__ import annotations

import numpy as np
from qp_core.model import model

__all__ = ["ledoit_wolf", "pca_cov", "sample_cov"]


def _panel(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 1:
        raise ValueError("returns must be a T x N array with T >= 2")
    if not np.isfinite(a).all():
        raise ValueError("returns contain NaN or infinite values")
    return a


@model(id="BT-COV-PCA")
def sample_cov(returns: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance of a T x N return panel."""
    return np.atleast_2d(np.cov(_panel(returns), rowvar=False, ddof=1))


@model(id="BT-COV-LW")
def ledoit_wolf(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage to a scaled identity.

    Returns ``(sigma, delta)`` with ``sigma = delta * mu I + (1 - delta) S``,
    ``S`` the 1/T sample covariance of demeaned returns, ``mu = tr(S)/N``.
    """
    x = _panel(returns)
    t, n = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / t
    mu = np.trace(s) / n
    d2 = float(np.sum((s - mu * np.eye(n)) ** 2))
    b2_bar = float(sum(np.sum((np.outer(r, r) - s) ** 2) for r in xc)) / t**2
    b2 = min(b2_bar, d2)
    delta = 0.0 if d2 == 0 else b2 / d2
    return delta * mu * np.eye(n) + (1 - delta) * s, delta


@model(id="BT-COV-PCA")
def pca_cov(returns: np.ndarray, k: int) -> np.ndarray:
    """k-factor PCA covariance: ``V_k L_k V_k' + diag(residual)``."""
    s = sample_cov(returns)
    n = s.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    vals, vecs = np.linalg.eigh(s)
    order = np.argsort(vals)[::-1][:k]
    v, lam = vecs[:, order], vals[order]
    common = (v * lam) @ v.T
    resid = np.clip(np.diag(s) - np.diag(common), 0.0, None)
    return common + np.diag(resid)
