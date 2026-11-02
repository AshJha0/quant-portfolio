"""Hierarchical risk parity (HRP).

Lopez de Prado (2016), "Building diversified portfolios that outperform out
of sample", Journal of Portfolio Management 42(4), section 3 and appendix:

1. Tree clustering: single linkage on the correlation distance
   d_ij = sqrt((1 - rho_ij) / 2).
2. Quasi-diagonalisation: order the assets by the leaves of the dendrogram.
3. Recursive bisection: split the ordered list in two halves; give each half
   the inverse-variance portfolio inside it, with cluster variance
   V = w' Sigma w; split the parent weight with alpha = 1 - V_1 / (V_1 + V_2)
   to the first half.

No matrix inversion is needed, so HRP works on singular covariances.
Oracle: with a diagonal covariance every split reproduces the
inverse-variance portfolio, so HRP equals w_i proportional to 1 / sigma_i^2.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def _check_cov(cov: np.ndarray) -> np.ndarray:
    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] == 0:
        raise ValueError("cov must be a non-empty square matrix")
    if not np.allclose(c, c.T, atol=1e-12):
        raise ValueError("cov must be symmetric")
    if np.any(np.diag(c) <= 0.0):
        raise ValueError("cov must have a positive diagonal")
    return c


def correlation_distance(cov: np.ndarray) -> np.ndarray:
    c = _check_cov(cov)
    sd = np.sqrt(np.diag(c))
    rho = np.clip(c / np.outer(sd, sd), -1.0, 1.0)
    d = np.sqrt(np.clip(0.5 * (1.0 - rho), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return d


def quasi_diagonal_order(cov: np.ndarray, method: str = "single") -> np.ndarray:
    d = correlation_distance(cov)
    if d.shape[0] == 1:
        return np.array([0])
    link = linkage(squareform(d, checks=False), method=method)
    return leaves_list(link)


def _cluster_var(c: np.ndarray, idx: np.ndarray) -> float:
    sub = c[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(cov: np.ndarray, method: str = "single") -> np.ndarray:
    """HRP weights (long-only, summing to one) in the input asset order."""
    c = _check_cov(cov)
    order = quasi_diagonal_order(c, method)
    w = np.ones(c.shape[0])
    stack = [order]
    while stack:
        items = stack.pop()
        if items.size < 2:
            continue
        half = items.size // 2
        left, right = items[:half], items[half:]
        v_l, v_r = _cluster_var(c, left), _cluster_var(c, right)
        alpha = 1.0 - v_l / (v_l + v_r)
        w[left] *= alpha
        w[right] *= 1.0 - alpha
        stack.extend([left, right])
    return w / w.sum()
