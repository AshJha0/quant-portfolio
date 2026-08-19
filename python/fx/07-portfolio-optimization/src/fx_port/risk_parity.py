"""Equal risk contribution (ERC) allocation and ex-ante vol targeting.

Risk contributions come from the Euler decomposition of portfolio volatility:
with ``sigma_p = sqrt(w'Sigma w)``, the marginal contribution of asset *i* is
``(Sigma w)_i / sigma_p`` and the (normalised) risk contribution is

    RC_i = w_i (Sigma w)_i / (w'Sigma w),      sum_i RC_i = 1  (Euler identity).

The ERC portfolio equalises RC_i (or matches a supplied risk budget).  We
solve it with the cyclical coordinate-descent algorithm of
Griveau-Billion/Richard/Roncalli (2013) on the log-barrier formulation

    min_x  0.5 x'Sigma x - sum_i b_i log(x_i),   x > 0,

whose first-order condition is ``(Sigma x)_i = b_i / x_i`` (risk
contributions proportional to budgets); the ERC weights are ``w = x/sum(x)``.
Each coordinate update solves a scalar quadratic exactly, so convergence to
1e-12 in the contributions takes a handful of sweeps.

Applies both across CURRENCIES (risk-balanced currency basket) and across
STYLES (carry/momentum/value multi-style book) — the caller just supplies the
relevant covariance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252


def risk_contributions(
    weights: pd.Series | np.ndarray,
    sigma: pd.DataFrame | np.ndarray,
    normalize: bool = True,
) -> pd.Series:
    """Euler risk contributions of a portfolio.

    Parameters
    ----------
    weights : array-like
        Portfolio weights.
    sigma : array-like
        Covariance matrix (per period).
    normalize : bool
        If True (default) return RC_i / sum RC = w_i(Sigma w)_i / (w'Sigma w);
        if False return the unnormalised ``w_i (Sigma w)_i`` (which sum to the
        portfolio VARIANCE — the Euler identity, tested).

    Returns
    -------
    pd.Series
        One contribution per asset.
    """
    w = np.asarray(weights, dtype=float).ravel()
    s = np.asarray(sigma, dtype=float)
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(s)):
        raise ValueError("weights and sigma must be finite (no NaN/Inf)")
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] != w.size:
        raise ValueError(
            f"sigma must be square and conformable with weights, got "
            f"{s.shape} vs {w.size}"
        )
    labels = (
        list(weights.index)
        if isinstance(weights, pd.Series)
        else (list(sigma.index) if isinstance(sigma, pd.DataFrame) else range(len(w)))
    )
    contrib = w * (s @ w)
    total = float(contrib.sum())
    if normalize:
        if total <= 0:
            raise ValueError("portfolio variance is zero; contributions undefined")
        contrib = contrib / total
    return pd.Series(contrib, index=labels, name="risk_contribution")


def erc_weights(
    sigma: pd.DataFrame,
    budget: pd.Series | np.ndarray | None = None,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> pd.Series:
    """Risk-budgeted (default: equal-risk-contribution) long-only weights.

    Parameters
    ----------
    sigma : pd.DataFrame
        Covariance matrix; diagonal must be strictly positive (a zero-vol
        pegged currency cannot receive a defined risk budget — drop it or
        floor the covariance with ``psd_repair(min_eig>0)`` first).
    budget : array-like, optional
        Target risk budgets b_i > 0 (normalised internally); default equal.
    tol : float
        Convergence tolerance on the largest absolute deviation of the
        normalised risk contributions from their budgets.
    max_iter : int
        Maximum coordinate-descent sweeps.

    Returns
    -------
    pd.Series
        Long-only weights summing to 1 with ``RC_i ≈ b_i``.

    Raises
    ------
    ValueError
        On non-positive diagonal, invalid budgets, or non-convergence.
    """
    s = np.asarray(sigma, dtype=float)
    n = len(s)
    labels = list(sigma.index) if isinstance(sigma, pd.DataFrame) else list(range(n))
    if not np.all(np.isfinite(s)):
        raise ValueError("sigma contains NaN/Inf; clean the covariance estimate first")
    if np.any(np.diag(s) <= 0):
        raise ValueError(
            "sigma diagonal must be strictly positive for ERC "
            "(zero-vol asset in universe? drop it or floor eigenvalues)"
        )
    if budget is None:
        b = np.full(n, 1.0 / n)
    else:
        b = np.asarray(budget, dtype=float).ravel()
        if len(b) != n or np.any(b <= 0):
            raise ValueError("budget must be positive and match sigma's size")
        b = b / b.sum()

    x = b / np.sqrt(np.diag(s))  # sensible start: inverse-vol scaled budgets
    for _ in range(max_iter):
        for i in range(n):
            c_i = float(s[i] @ x - s[i, i] * x[i])
            x[i] = (-c_i + np.sqrt(c_i * c_i + 4.0 * s[i, i] * b[i])) / (
                2.0 * s[i, i]
            )
        rc = x * (s @ x)
        dev = float(np.max(np.abs(rc / rc.sum() - b)))
        if dev < tol:
            break
    else:
        raise ValueError(f"ERC failed to converge (last deviation {dev:.2e})")
    w = x / x.sum()
    return pd.Series(w, index=labels, name="erc")


def portfolio_vol(
    weights: pd.Series | np.ndarray,
    sigma: pd.DataFrame | np.ndarray,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised ex-ante volatility ``sqrt(w'Sigma w * periods_per_year)``."""
    w = np.asarray(weights, dtype=float).ravel()
    s = np.asarray(sigma, dtype=float)
    return float(np.sqrt(max(w @ s @ w, 0.0) * periods_per_year))


def vol_target(
    weights: pd.Series,
    sigma: pd.DataFrame,
    target_vol: float,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """Scale weights so the ex-ante annualised vol equals ``target_vol`` exactly.

    Parameters
    ----------
    weights : pd.Series
        Unscaled weights (any net/gross).
    sigma : pd.DataFrame
        Per-period covariance.
    target_vol : float
        Desired annualised volatility, must be > 0.
    periods_per_year : int
        Annualisation factor (default 252).

    Returns
    -------
    pd.Series
        ``weights * target_vol / current_vol``; the scaled portfolio's
        ex-ante vol equals ``target_vol`` to float precision (tested).

    Raises
    ------
    ValueError
        If the portfolio has zero ex-ante vol.
    """
    if target_vol <= 0:
        raise ValueError(f"target_vol must be > 0, got {target_vol}")
    cur = portfolio_vol(weights, sigma, periods_per_year)
    if cur <= 0:
        raise ValueError("cannot vol-target a zero-variance portfolio")
    return weights * (target_vol / cur)
