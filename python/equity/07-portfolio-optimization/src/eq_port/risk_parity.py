"""Risk parity: equal-risk-contribution (ERC) weights via cyclical
coordinate descent, risk-contribution diagnostics, naive inverse-vol
parity, and a volatility-targeting leverage overlay.

Conventions: ``cov`` is the (N, N) per-period covariance; long-only
weights; risk contributions are stated in VARIANCE terms,
RC_i = w_i (Sigma w)_i, which satisfy the Euler identity
sum_i RC_i = w' Sigma w exactly.

Leverage note: an unlevered ERC portfolio of low-vol assets typically
runs well below an equity-like risk level, so real risk-parity funds
LEVER the ERC weights to a volatility target (see
:func:`vol_target_overlay`). That leverage is exactly what forced the
March-2020 deleveraging spiral discussed in docs/DESK_GUIDE.md.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "risk_contributions",
    "erc_weights",
    "inverse_vol_weights",
    "vol_target_overlay",
]


def _validate_cov(cov: np.ndarray) -> np.ndarray:
    sigma = np.asarray(cov, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"cov must be square, got shape {sigma.shape}")
    if not np.all(np.isfinite(sigma)):
        raise ValueError("cov contains NaN or infinite values")
    if not np.allclose(sigma, sigma.T, atol=1e-8):
        raise ValueError("cov must be symmetric")
    return 0.5 * (sigma + sigma.T)


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Variance risk contributions RC_i = w_i (Sigma w)_i.

    By Euler's theorem for the homogeneous function w'Sigma w, the
    contributions sum exactly to the portfolio variance.

    Parameters
    ----------
    weights : (N,) array-like
        Portfolio weights (any sign, any leverage).
    cov : (N, N) array-like
        Per-period covariance.

    Returns
    -------
    np.ndarray
        (N,) variance contributions; ``sum == w' Sigma w`` to machine precision.
    """
    sigma = _validate_cov(cov)
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape[0] != sigma.shape[0]:
        raise ValueError(
            f"dimension mismatch: weights have {w.shape[0]} entries, cov is {sigma.shape}"
        )
    return w * (sigma @ w)


def erc_weights(
    cov: np.ndarray,
    budget: np.ndarray | None = None,
    tol: float = 1e-14,
    max_iter: int = 10_000,
) -> np.ndarray:
    """Long-only equal-risk-contribution weights via cyclical coordinate
    descent (Griveau-Billion, Richard & Roncalli 2013), from scratch.

    ERC solves RC_i = b_i * w'Sigma w for risk budgets b (default equal).
    Equivalently, minimise the strictly convex

        F(y) = 1/2 y' Sigma y - sum_i b_i log(y_i),   y > 0,

    whose first-order condition is y_i (Sigma y)_i = b_i; the ERC weights
    are y / sum(y). Coordinate-wise, holding m_i = (Sigma y)_i -
    Sigma_ii y_i fixed, the update is the positive root of
    Sigma_ii y_i^2 + m_i y_i - b_i = 0:

        y_i <- ( -m_i + sqrt(m_i^2 + 4 Sigma_ii b_i) ) / (2 Sigma_ii).

    Each update strictly decreases F, and F is strictly convex on the
    positive orthant, so the iteration converges to the unique solution
    (Spinu 2013 proves existence/uniqueness for PD Sigma).

    Parameters
    ----------
    cov : (N, N) array-like
        Per-period covariance with strictly positive diagonal. PD is
        required for uniqueness; PSD with positive diagonal usually works.
    budget : (N,) array-like, optional
        Strictly positive risk budgets, normalised internally (default
        equal budgets 1/N — true ERC).
    tol : float
        Convergence tolerance on the max absolute weight change per sweep.
    max_iter : int
        Maximum number of full coordinate sweeps.

    Returns
    -------
    np.ndarray
        (N,) strictly positive weights summing to 1 with
        RC_i / portfolio variance == b_i (to ~1e-10 relative).

    Raises
    ------
    ValueError
        Zero/negative diagonal entries (zero-vol asset has no defined risk
        contribution), non-positive budgets, or non-convergence.
    """
    sigma = _validate_cov(cov)
    n = sigma.shape[0]
    diag = np.diag(sigma)
    if np.any(diag <= 0.0):
        raise ValueError(
            "cov has non-positive diagonal entries (zero-volatility asset): "
            "risk contributions are undefined — drop the asset or use "
            "psd_repair with a positive eigenvalue floor."
        )
    if budget is None:
        b = np.full(n, 1.0 / n)
    else:
        b = np.asarray(budget, dtype=float).ravel()
        if b.shape[0] != n:
            raise ValueError(f"budget must have {n} entries, got {b.shape[0]}")
        if np.any(b <= 0.0):
            raise ValueError("risk budgets must be strictly positive")
        b = b / b.sum()
    if n == 1:
        return np.ones(1)

    # start at inverse-vol (good warm start, strictly positive)
    y = (1.0 / np.sqrt(diag))
    y /= y.sum()
    sy = sigma @ y
    for _ in range(max_iter):
        y_prev = y.copy()
        for i in range(n):
            m_i = sy[i] - sigma[i, i] * y[i]
            yi_new = (-m_i + np.sqrt(m_i * m_i + 4.0 * sigma[i, i] * b[i])) / (
                2.0 * sigma[i, i]
            )
            delta = yi_new - y[i]
            if delta != 0.0:
                sy = sy + delta * sigma[:, i]
                y[i] = yi_new
        if np.max(np.abs(y - y_prev)) < tol * max(1.0, np.max(np.abs(y))):
            break
    else:
        raise ValueError(
            f"ERC coordinate descent failed to converge in {max_iter} sweeps; "
            "check that cov is positive definite (repair with psd_repair)."
        )
    return y / y.sum()


def inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    """Naive risk parity: weights proportional to 1/vol_i.

    Coincides with true ERC when all pairwise correlations are equal
    (e.g. uncorrelated assets, or a constant-correlation matrix);
    otherwise it ignores correlation structure.

    Parameters
    ----------
    cov : (N, N) array-like
        Per-period covariance with strictly positive diagonal.

    Returns
    -------
    np.ndarray
        (N,) positive weights summing to 1.
    """
    sigma = _validate_cov(cov)
    diag = np.diag(sigma)
    if np.any(diag <= 0.0):
        raise ValueError("cov has non-positive diagonal entries (zero-vol asset)")
    w = 1.0 / np.sqrt(diag)
    return w / w.sum()


def vol_target_overlay(
    weights: np.ndarray,
    cov: np.ndarray,
    target_vol: float,
    periods_per_year: float = 252.0,
    max_leverage: float | None = None,
) -> np.ndarray:
    """Scale weights so the ex-ante ANNUALISED vol equals ``target_vol``.

    The scaled position is ``L * w`` with
    ``L = target_vol / (sqrt(w'Sigma w) * sqrt(periods_per_year))``;
    the remainder ``1 - L * sum(w)`` is implicitly cash (rf ignored).
    This is the risk-parity leverage overlay: low-vol ERC books are
    levered up to an equity-like risk target.

    Parameters
    ----------
    weights : (N,) array-like
        Unlevered weights.
    cov : (N, N) array-like
        PER-PERIOD covariance (e.g. daily); annualisation uses
        ``periods_per_year``.
    target_vol : float
        Annualised target volatility (> 0), e.g. 0.10 for 10%.
    periods_per_year : float
        Annualisation factor for the covariance frequency.
    max_leverage : float, optional
        Cap on L (gross scaling); breaching it clips L (documented desk
        control — see DESK_GUIDE on deleveraging spirals).

    Returns
    -------
    np.ndarray
        (N,) levered weights with exact ex-ante annual vol = target_vol
        (unless the leverage cap binds).
    """
    if target_vol <= 0:
        raise ValueError(f"target_vol must be > 0, got {target_vol}")
    sigma = _validate_cov(cov)
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape[0] != sigma.shape[0]:
        raise ValueError(
            f"dimension mismatch: weights have {w.shape[0]} entries, cov is {sigma.shape}"
        )
    vol_ann = float(np.sqrt(max(w @ sigma @ w, 0.0) * periods_per_year))
    if vol_ann == 0.0:
        raise ValueError("portfolio has zero ex-ante volatility; cannot vol-target")
    lev = target_vol / vol_ann
    if max_leverage is not None:
        lev = min(lev, max_leverage)
    return lev * w
