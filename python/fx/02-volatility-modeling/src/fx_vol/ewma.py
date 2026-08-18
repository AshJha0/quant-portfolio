"""RiskMetrics EWMA (exponentially weighted moving average) variance.

The RiskMetrics (1996) filter is an IGARCH(1,1) special case with
``omega = 0`` and ``alpha + beta = 1``:

    sigma2_t = lambda * sigma2_{t-1} + (1 - lambda) * r_{t-1}^2

``sigma2_t`` is the *conditional* variance for period t formed with
information up to t-1 (one-step-ahead convention, matching the GARCH modules).
The classic daily decay is ``lambda = 0.94``; RiskMetrics monthly uses 0.97.

Because persistence is exactly 1, the EWMA multi-step forecast is *flat*:
``E[sigma2_{T+h}] = sigma2_{T+1}`` for every h -- there is no mean reversion,
which is the main modelling difference vs GARCH(1,1) and is unit-tested.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .returns import _as_1d_array

__all__ = ["ewma_variance", "ewma_forecast", "ewma_weights"]


def ewma_variance(
    returns: Sequence[float] | np.ndarray | pd.Series,
    lam: float = 0.94,
    init: float | None = None,
) -> np.ndarray:
    """Run the RiskMetrics recursion over a return series.

    Parameters
    ----------
    returns : array-like
        Per-period log returns (decimal). NaNs raise ``ValueError``.
    lam : float
        Decay factor in (0, 1); 0.94 is the RiskMetrics daily standard.
    init : float, optional
        ``sigma2_1`` seed. Defaults to the full-sample variance (ddof=0),
        the standard pragmatic choice; the seed's influence decays like
        ``lam^t``.

    Returns
    -------
    numpy.ndarray
        ``sigma2`` of the same length as ``returns``; ``sigma2[t]`` is the
        conditional variance of ``returns[t]`` given returns[0..t-1].
    """
    arr = _as_1d_array(returns, "returns")
    if arr.size < 2:
        raise ValueError(f"need at least 2 returns, got {arr.size}")
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lambda must be in (0, 1), got {lam}")
    if init is None:
        init = float(np.mean(arr ** 2))
    if init < 0:
        raise ValueError("init variance must be non-negative")
    n = arr.size
    sigma2 = np.empty(n)
    sigma2[0] = init
    one_minus = 1.0 - lam
    r2 = arr ** 2
    for t in range(1, n):
        sigma2[t] = lam * sigma2[t - 1] + one_minus * r2[t - 1]
    return sigma2


def ewma_forecast(
    returns: Sequence[float] | np.ndarray | pd.Series,
    horizon: int,
    lam: float = 0.94,
    init: float | None = None,
) -> np.ndarray:
    """Multi-step EWMA variance forecast -- flat at the next-step value.

    ``sigma2_{T+1} = lam * sigma2_T + (1 - lam) * r_T^2`` repeated for every
    horizon 1..H, because the EWMA/IGARCH persistence is exactly 1.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    sigma2 = ewma_variance(returns, lam=lam, init=init)
    arr = _as_1d_array(returns, "returns")
    next_var = lam * sigma2[-1] + (1.0 - lam) * arr[-1] ** 2
    return np.full(horizon, next_var)


def ewma_weights(lam: float, n: int) -> np.ndarray:
    """Effective weights on the last ``n`` squared returns.

    ``w_i = (1 - lam) * lam^i`` for lag i = 0..n-1 (most recent first).
    Sums to ``1 - lam^n`` -> 1 as n grows; the identity is unit-tested.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lambda must be in (0, 1), got {lam}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return (1.0 - lam) * lam ** np.arange(n)
