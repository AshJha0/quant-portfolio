"""Covariance estimators for the panel studies: linear and nonlinear shrinkage, and PCA (W23).

Plain idea: with 30 currencies and 120 months the sample covariance matrix is
noisy in a specific way. Its largest eigenvalues come out too large and its
smallest too small, so a mean-variance optimiser loads up on the directions
that only look riskless. Shrinkage pulls the eigenvalues back towards each
other while keeping the sample eigenvectors. Linear shrinkage pulls every
eigenvalue towards the average by one common fraction. Nonlinear shrinkage
picks a separate pull for each eigenvalue. A PCA (statistical factor) model
keeps the top ``k`` directions and treats what is left as independent noise.

Formal statements.

* ``linear_shrinkage`` (Ledoit and Wolf 2004, J. Multivariate Analysis 88,
  section 3.3): ``S* = (1 - d) S + d m I`` with ``m = tr(S)/p`` and
  ``d = min(b^2, d^2) / d^2``, ``d^2 = ||S - m I||_F^2 / p``,
  ``b^2 = (1/n^2) sum_t ||x_t x_t' - S||_F^2 / p``.
* ``nonlinear_shrinkage`` (Ledoit and Wolf 2020, Annals of Statistics 48(5),
  "Analytical nonlinear shrinkage of large-dimensional covariance
  matrices", section 4 and their published Matlab code): with sample
  eigenvalues ``l_i``, bandwidth ``h = n^(-1/3)``, local bandwidths
  ``h l_j``, the Epanechnikov kernel density ``f~`` and its Hilbert transform
  ``Hf~`` evaluated at each ``l_i``, the shrunk eigenvalue is

      d_i = l_i / ((pi c l_i f~(l_i))^2 + (1 - c - pi c l_i Hf~(l_i))^2),  c = p/n <= 1,

  and for ``p > n`` the ``p - n`` null eigenvalues get ``d_0`` from the
  paper's equation (C.9). ``n`` is the number of observations minus one
  when the mean is removed.
* ``pca_covariance``: ``S_k = V_k L_k V_k' + D`` where ``V_k, L_k`` are the
  top ``k`` sample eigenpairs and ``D`` is diagonal, chosen so the diagonal
  of ``S_k`` equals the sample variances (Connor and Korajczyk 1986 style
  statistical factors; the residual is floored at zero).

Worked example (``research/panel/tests/test_covariance.py``): with a true
covariance of the identity, ``p = 100`` and ``n = 200`` the sample
eigenvalues spread from about 0.1 to 2.9 (the Marchenko-Pastur law,
``(1 +/- sqrt(1/2))^2 = 0.09`` to ``2.91``). Nonlinear shrinkage returns
eigenvalues between about 0.83 and 1.30. The sample's worst eigenvalue
error is above 1.5; the shrunk one is about 0.3. The shrunk eigenvalues are
not forced to keep the sample order (the paper does not, either).

These estimators are research code, not pricing or risk, so the D1 Rust and
C++ parity rule does not apply.
"""

from __future__ import annotations

import math

import numpy as np


def _prep(x: np.ndarray, demean: bool) -> tuple[np.ndarray, int]:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be observations x variables")
    if np.isnan(x).any():
        raise ValueError("x has NaN; align or drop rows first")
    if demean:
        x = x - x.mean(axis=0)
        return x, x.shape[0] - 1
    return x, x.shape[0]


def sample_cov(x: np.ndarray, demean: bool = True) -> np.ndarray:
    xc, n = _prep(x, demean)
    return xc.T @ xc / n


def linear_shrinkage(x: np.ndarray, demean: bool = True) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage to the scaled identity; returns (matrix, intensity).

    As in the paper (and ``sklearn.covariance.ledoit_wolf``) the sample
    covariance here divides by the row count ``t``, not ``t - 1``.
    """
    xc, _ = _prep(x, demean)
    t, p = xc.shape
    s = xc.T @ xc / t
    m = float(np.trace(s)) / p
    d2 = float(((s - m * np.eye(p)) ** 2).sum()) / p
    # (1/t^2) sum_k ||x_k x_k' - S||_F^2 = (sum_k (x_k'x_k)^2 / t - ||S||_F^2) / t
    rows = (xc**2).sum(axis=1)
    b2bar = (float((rows**2).sum()) / t - float((s**2).sum())) / t / p
    b2 = min(max(b2bar, 0.0), d2)
    delta = b2 / d2 if d2 > 0 else 0.0
    return (1 - delta) * s + delta * m * np.eye(p), delta


def nonlinear_shrinkage(x: np.ndarray, demean: bool = True) -> np.ndarray:
    """Ledoit-Wolf (2020) analytical nonlinear shrinkage; see the module docstring."""
    xc, n = _prep(x, demean)
    p = xc.shape[1]
    if n < 12:
        raise ValueError("need at least 12 effective observations (sqrt(5) n^(-1/3) < 1)")
    s = xc.T @ xc / n
    lam, u = np.linalg.eigh(s)  # ascending
    lam = np.maximum(lam, 0.0)
    keep = lam[max(0, p - n):]  # the min(p, n) non-null eigenvalues
    c = p / n
    h = n ** (-1.0 / 3.0)
    big_h = h * keep[None, :]  # bandwidth of kernel j, broadcast over rows i
    xx = (keep[:, None] - keep[None, :]) / big_h
    r5 = math.sqrt(5.0)
    f_t = np.mean(np.maximum(1 - xx**2 / 5, 0.0) / big_h, axis=1) * (3 / 4 / r5)
    with np.errstate(divide="ignore", invalid="ignore"):
        hf = (-3 / 10 / math.pi) * xx + (3 / 4 / r5 / math.pi) * (1 - xx**2 / 5) * np.log(
            np.abs((r5 - xx) / (r5 + xx)))
    edge = np.isclose(np.abs(xx), r5)
    hf[edge] = (-3 / 10 / math.pi) * xx[edge]
    hf_t = np.mean(hf / big_h, axis=1)
    if p <= n:
        d = keep / ((math.pi * c * keep * f_t) ** 2 + (1 - c - math.pi * c * keep * hf_t) ** 2)
    else:
        hf0 = (1 / math.pi) * (3 / 10 / h**2 + 3 / 4 / r5 / h * (1 - 1 / 5 / h**2)
                               * math.log((1 + r5 * h) / (1 - r5 * h))) * float(np.mean(1 / keep))
        d0 = 1 / (math.pi * (p - n) / n * hf0)
        d1 = keep / (math.pi**2 * keep**2 * (f_t**2 + hf_t**2))
        d = np.r_[np.full(p - n, d0), d1]
    out = (u * d) @ u.T
    return 0.5 * (out + out.T)


def pca_covariance(x: np.ndarray, k: int, demean: bool = True) -> np.ndarray:
    """Top-``k`` statistical factors plus a diagonal residual matching the sample variances."""
    s = sample_cov(x, demean)
    p = s.shape[0]
    if not 0 <= k <= p:
        raise ValueError(f"k must be in [0, {p}]")
    lam, v = np.linalg.eigh(s)
    lam, v = lam[::-1][:k], v[:, ::-1][:, :k]
    common = (v * lam) @ v.T
    resid = np.maximum(np.diag(s) - np.diag(common), 0.0)
    return common + np.diag(resid)


def frobenius_loss(est: np.ndarray, true: np.ndarray) -> float:
    """``||est - true||_F^2 / p``, the loss used in the Ledoit-Wolf papers' tables."""
    return float(((est - true) ** 2).sum()) / true.shape[0]
