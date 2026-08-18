"""RiskMetrics EWMA (exponentially weighted moving average) variance.

Model
-----
sigma2_t = lambda * sigma2_{t-1} + (1 - lambda) * r_{t-1}^2,   0 < lambda <= 1.

This is the J.P. Morgan RiskMetrics (1996) filter with the classic daily decay
``lambda = 0.94``. It is an IGARCH(1,1) with omega = 0 and alpha + beta = 1:
shocks never decay toward a long-run level, which is exactly why the EWMA
*forecast term structure is flat* (see :func:`ewma_forecast`).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from ._utils import validate_returns

__all__ = [
    "ewma_variance",
    "ewma_variance_recursive",
    "ewma_weights",
    "lambda_to_halflife",
    "halflife_to_lambda",
    "ewma_forecast",
]


def _check_lambda(lam: float) -> None:
    if not 0.0 < lam <= 1.0:
        raise ValueError(f"lambda must be in (0, 1], got {lam}")


def ewma_variance(
    returns: np.ndarray,
    lam: float = 0.94,
    init_var: float | None = None,
) -> np.ndarray:
    """Vectorised EWMA conditional variance path (scipy ``lfilter``).

    The recursion is a first-order linear filter in sigma2, so the whole path
    is computed in one ``lfilter`` call — O(n) with no Python loop. Bitwise
    equivalence with the explicit recursion is unit-tested.

    Parameters
    ----------
    returns : array-like
        Daily log-returns, decimal units.
    lam : float
        Decay factor in (0, 1]. RiskMetrics daily standard is 0.94.
        ``lam = 1`` degenerates to the constant initial variance.
    init_var : float, optional
        sigma2_0. Defaults to the sample mean of squared returns.

    Returns
    -------
    numpy.ndarray
        sigma2_t for t = 0..n-1, where sigma2_t conditions on returns up to
        t-1 (a genuine one-step-ahead filtered variance at each t).
    """
    r = validate_returns(returns, min_obs=2)
    _check_lambda(lam)
    v0 = float(np.mean(r**2)) if init_var is None else float(init_var)
    if v0 < 0 or not np.isfinite(v0):
        raise ValueError(f"init_var must be finite and >= 0, got {init_var}")
    n = r.size
    sigma2 = np.empty(n)
    sigma2[0] = v0
    if lam == 1.0:
        sigma2[1:] = v0
        return sigma2
    x = (1.0 - lam) * r[:-1] ** 2
    sigma2[1:] = lfilter([1.0], [1.0, -lam], x, zi=np.array([lam * v0]))[0]
    return sigma2


def ewma_variance_recursive(
    returns: np.ndarray,
    lam: float = 0.94,
    init_var: float | None = None,
) -> np.ndarray:
    """Explicit-loop reference implementation of :func:`ewma_variance`.

    Kept as the readable specification of the recursion; the vectorised
    version is tested to match it exactly.
    """
    r = validate_returns(returns, min_obs=2)
    _check_lambda(lam)
    v0 = float(np.mean(r**2)) if init_var is None else float(init_var)
    if v0 < 0 or not np.isfinite(v0):
        raise ValueError(f"init_var must be finite and >= 0, got {init_var}")
    sigma2 = np.empty(r.size)
    sigma2[0] = v0
    for t in range(1, r.size):
        sigma2[t] = lam * sigma2[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return sigma2


def ewma_weights(n: int, lam: float = 0.94) -> np.ndarray:
    """Weights on the last ``n`` squared returns implied by the recursion.

    w_i = (1 - lambda) lambda^i for lag i = 0..n-1 (most recent first); the
    remaining mass lambda^n sits on the initial variance. Useful for the
    brute-force cross-check of the recursion.
    """
    _check_lambda(lam)
    if lam == 1.0:
        return np.zeros(n)
    return (1.0 - lam) * lam ** np.arange(n)


def lambda_to_halflife(lam: float) -> float:
    """Half-life (days) of the EWMA weight decay: h = -ln 2 / ln lambda.

    lambda = 0.94 -> ~11.2 days; lambda -> 1 gives infinite half-life.
    """
    _check_lambda(lam)
    if lam == 1.0:
        return np.inf
    return float(-np.log(2.0) / np.log(lam))


def halflife_to_lambda(halflife: float) -> float:
    """Decay factor with the given half-life in days: lambda = 2^(-1/h)."""
    if halflife <= 0:
        raise ValueError(f"halflife must be > 0, got {halflife}")
    return float(2.0 ** (-1.0 / halflife))


def ewma_forecast(
    returns: np.ndarray,
    horizon: int = 1,
    lam: float = 0.94,
    init_var: float | None = None,
) -> np.ndarray:
    """Multi-step EWMA variance forecast — flat at sigma2_{T+1}.

    Why flat: EWMA is IGARCH(1,1) with zero intercept, so
    E_T[sigma2_{T+k}] = E_T[lambda sigma2_{T+k-1} + (1-lambda) r_{T+k-1}^2]
                      = lambda E_T[sigma2_{T+k-1}] + (1-lambda) E_T[sigma2_{T+k-1}]
                      = E_T[sigma2_{T+k-1}]  for k >= 2,
    because E_T[r^2] = E_T[sigma2] for future dates. The forecast at every
    horizon therefore equals the one-step forecast: there is **no mean
    reversion** and no unconditional variance to revert to. This is the key
    structural difference from stationary GARCH, whose term structure decays
    toward the long-run variance.

    Returns
    -------
    numpy.ndarray
        Length-``horizon`` array, all entries equal to sigma2_{T+1}.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    r = validate_returns(returns, min_obs=2)
    sigma2 = ewma_variance(r, lam=lam, init_var=init_var)
    v_next = lam * sigma2[-1] + (1.0 - lam) * r[-1] ** 2
    return np.full(horizon, v_next)
