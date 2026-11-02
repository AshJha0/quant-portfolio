"""Historical-simulation VaR: plain, age-weighted (BRW) and filtered (FHS).

Conventions
-----------
* ``pnl`` arrays are P&L in currency units, loss < 0.
* ``alpha`` is the tail probability: ``alpha=0.01`` gives 99 % VaR.
* VaR is a positive number for a loss.
* Quantile: the order statistic ``L(m)`` of the losses with
  ``m = ceil(n * alpha)`` (spec S2, ``eq_var._tail``). No interpolation.
  Weighted (BRW) VaR inverts the weighted step CDF with the same rule.
* EWMA volatility (spec S8): zero-mean RiskMetrics recursion seeded with the
  mean of ``x**2`` over the first ``seed_window = min(30, n)`` observations.
  The seed uses those first observations and nothing later.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import _tail
from ._validation import finite_array, finite_scalar, integer, validate_alpha

__all__ = [
    "age_weighted_es",
    "age_weighted_var",
    "brw_weights",
    "ewma_forecast",
    "ewma_volatility",
    "factor_filtered_historical_var",
    "filtered_historical_scenarios",
    "filtered_historical_var",
    "historical_var",
    "overlapping_horizon_pnl",
    "scale_var_sqrt_time",
]

MIN_OBS = 50  # fewer observations than this cannot resolve a 1-5 % tail sensibly
SEED_WINDOW = 30
_UNSET = object()


def _validate_lam(lam: object) -> float:
    lam = finite_scalar("lam", lam)
    if not 0.0 < lam < 1.0:
        raise ValueError(f"decay lam must be in (0, 1), got {lam}")
    return lam


def _validate_pnl(pnl: object, alpha: float, min_obs: int = MIN_OBS) -> np.ndarray:
    arr = finite_array("pnl", pnl, ndim=1, min_size=0)
    if arr.size < min_obs:
        raise ValueError(
            f"need at least {min_obs} P&L observations for historical VaR, got {arr.size}; "
            "empirical tail quantiles are meaningless on shorter samples"
        )
    return arr


def historical_var(pnl: np.ndarray, alpha: float = 0.01) -> float:
    """Plain historical-simulation VaR (equal weights).

    Returns ``L(m)``, the ``m``-th largest loss with ``m = ceil(n * alpha)``
    (spec S2). Raises when ``n < 50``. If ``n * alpha < 1`` the rank is
    clamped to 1 and the VaR is the largest loss in the sample.
    """
    alpha = validate_alpha(alpha)
    arr = _validate_pnl(pnl, alpha)
    return _tail.var_es(arr, alpha)[0]


def brw_weights(n: int, lam: float = 0.98) -> np.ndarray:
    """Boudoukh-Richardson-Whitelaw exponential age weights.

    Observation ``i`` (0 = oldest, n-1 = most recent) gets weight proportional
    to ``lam**(n-1-i)``. The weights sum to 1.
    """
    lam = _validate_lam(lam)
    n = integer("n", n, min_value=1)
    ages = np.arange(n - 1, -1, -1, dtype=float)  # age of each obs, oldest first
    return (1.0 - lam) * lam**ages / (1.0 - lam**n)


def _age_weighted(pnl: object, alpha: object, lam: object) -> tuple[float, float]:
    alpha = validate_alpha(alpha)
    lam = _validate_lam(lam)
    arr = finite_array("pnl", pnl, ndim=1, min_size=0)
    if arr.size < MIN_OBS:
        raise ValueError(f"need at least {MIN_OBS} P&L observations, got {arr.size}")
    return _tail.weighted_var_es(arr, brw_weights(arr.size, lam), alpha)


def age_weighted_var(pnl: np.ndarray, alpha: float = 0.01, lam: float = 0.98) -> float:
    """Age-weighted (BRW) historical VaR: the weighted S2 order statistic.

    ``lam -> 1`` recovers plain historical simulation up to the 1e-12
    tolerance in the rank rule.
    """
    return _age_weighted(pnl, alpha, lam)[0]


def age_weighted_es(pnl: np.ndarray, alpha: float = 0.01, lam: float = 0.98) -> float:
    """Age-weighted (BRW) Expected Shortfall with the fractional boundary atom (S2)."""
    return _age_weighted(pnl, alpha, lam)[1]


def _ewma_var_path(arr: np.ndarray, lam: float, seed: float) -> np.ndarray:
    """Variance forecasts ``s2[0..n]``; ``s2[n]`` is the forecast after the sample."""
    s2 = np.empty(arr.size + 1)
    s2[0] = max(seed, 1e-16)  # zero-variance series: avoid division by zero downstream
    for t in range(1, arr.size + 1):
        s2[t] = lam * s2[t - 1] + (1.0 - lam) * arr[t - 1] ** 2
    return s2


def _window_seed(arr: np.ndarray, seed_window: int) -> float:
    w = min(seed_window, arr.size)
    return float(np.mean(arr[:w] ** 2))


def ewma_volatility(
    x: np.ndarray, lam: float = 0.94, init: object = _UNSET, *, seed_window: int = SEED_WINDOW
) -> np.ndarray:
    """One-step-ahead EWMA (RiskMetrics) volatility forecasts.

    ``sigma2[t] = lam * sigma2[t-1] + (1 - lam) * x[t-1]**2`` with
    ``sigma2[0] = mean(x[:w]**2)``, ``w = min(seed_window, n)`` (spec S8).
    ``sigma[t]`` is the forecast for day ``t``. Apart from the seed window it
    uses information up to ``t-1`` only.

    ``init`` is deprecated: ``"sample"`` (full-sample variance, a look-ahead)
    and ``"first"`` (``x[0]**2``) keep their old meaning for one release.

    Returns an array of the same length as ``x``.
    """
    lam = _validate_lam(lam)
    arr = finite_array("x", x, ndim=1, min_size=0)
    if arr.size < 2:
        raise ValueError("need at least 2 observations for EWMA volatility")
    seed_window = integer("seed_window", seed_window, min_value=1)
    if init is _UNSET:
        seed = _window_seed(arr, seed_window)
    else:
        warnings.warn(
            "ewma_volatility(init=...) is deprecated; the seed is now the mean of x**2 "
            "over the first seed_window observations",
            DeprecationWarning,
            stacklevel=2,
        )
        if init == "sample":
            seed = float(np.var(arr))
        elif init == "first":
            seed = float(arr[0] ** 2)
        else:
            raise ValueError(f"init must be 'sample' or 'first', got {init!r}")
    return np.sqrt(np.maximum(_ewma_var_path(arr, lam, seed)[:-1], 1e-32))


def ewma_forecast(x: np.ndarray, lam: float = 0.94, *, seed_window: int = SEED_WINDOW) -> float:
    """EWMA volatility forecast for the day after the sample, ``sigma_{n}`` (S8)."""
    lam = _validate_lam(lam)
    arr = finite_array("x", x, ndim=1, min_size=0)
    if arr.size < 2:
        raise ValueError("need at least 2 observations for EWMA volatility")
    seed_window = integer("seed_window", seed_window, min_value=1)
    return float(np.sqrt(max(_ewma_var_path(arr, lam, _window_seed(arr, seed_window))[-1], 1e-32)))


def filtered_historical_scenarios(
    factor_returns: np.ndarray, lam: float = 0.94, *, seed_window: int = SEED_WINDOW
) -> np.ndarray:
    """Per-factor filtered scenarios (spec S9).

    For every factor ``j``: EWMA vol ``sigma_{t,j}`` (S8), standardised
    ``z_{t,j} = r_{t,j} / sigma_{t,j}``, rescaled ``r*_{t,j} = z_{t,j} *
    sigma_{T+1,j}`` with ``sigma_{T+1,j}`` the forecast for the day after the
    sample. Returns an array of the same shape as ``factor_returns`` (T, n).
    """
    lam = _validate_lam(lam)
    seed_window = integer("seed_window", seed_window, min_value=1)
    r = finite_array("factor_returns", factor_returns, ndim=2, min_size=1)
    if r.shape[0] < 2:
        raise ValueError(f"need at least 2 days of factor returns, got {r.shape[0]}")
    out = np.empty_like(r)
    for j in range(r.shape[1]):
        col = r[:, j]
        sig = np.sqrt(np.maximum(_ewma_var_path(col, lam, _window_seed(col, seed_window)), 1e-32))
        out[:, j] = col / sig[:-1] * sig[-1]
    return out


def factor_filtered_historical_var(
    portfolio,
    factor_returns: np.ndarray,
    alpha: float = 0.01,
    lam: float = 0.94,
    *,
    seed_window: int = SEED_WINDOW,
    method: str = "full",
) -> float:
    """Filtered historical simulation VaR with per-factor filtering (spec S9).

    Builds ``filtered_historical_scenarios`` and fully revalues the book under
    each one (``method="full"``), then takes the S2 order statistic. Option
    convexity is kept because the book is revalued, not the P&L rescaled.
    """
    alpha = validate_alpha(alpha)
    scen = filtered_historical_scenarios(factor_returns, lam, seed_window=seed_window)
    pnl = portfolio.pnl(scen, method=method)
    return historical_var(pnl, alpha)


def filtered_historical_var(pnl: np.ndarray, alpha: float = 0.01, lam: float = 0.94) -> float:
    """Deprecated P&L-level FHS. Use ``factor_filtered_historical_var``.

    Kept for one release with its full-sample EWMA seed. The tail is read
    with the same order statistic as every other VaR path
    (``eq_var._tail.var_es``, ``m = ceil(n alpha)``). It used numpy type-7
    interpolation before, so FHS and HS numbers were not like for like.
    Filtering portfolio P&L ignores option convexity and mixes position
    changes with market volatility.
    """
    warnings.warn(
        "filtered_historical_var(pnl, ...) filters portfolio P&L and is deprecated; "
        "use factor_filtered_historical_var(portfolio, factor_returns, ...)",
        DeprecationWarning,
        stacklevel=2,
    )
    alpha = validate_alpha(alpha)
    lam = _validate_lam(lam)
    arr = _validate_pnl(pnl, alpha)
    sigma = np.sqrt(np.maximum(_ewma_var_path(arr, lam, float(np.var(arr)))[:-1], 1e-32))
    z = arr / sigma
    sigma_next = np.sqrt(lam * sigma[-1] ** 2 + (1.0 - lam) * arr[-1] ** 2)
    return _tail.var_es(z * sigma_next, alpha)[0]


def scale_var_sqrt_time(var_1d: float, horizon_days: int) -> float:
    """Square-root-of-time scaling: ``VaR_h = VaR_1 * sqrt(h)``.

    Valid only for i.i.d. returns with zero drift. It understates multi-day
    risk under volatility clustering (docs/VALIDATION.md). ``var_1d`` must be
    finite and ``>= 0``.
    """
    var_1d = finite_scalar("var_1d", var_1d, min_value=0.0)
    horizon_days = integer("horizon_days", horizon_days, min_value=1)
    return var_1d * float(np.sqrt(horizon_days))


def overlapping_horizon_pnl(pnl: np.ndarray, horizon_days: int) -> np.ndarray:
    """Overlapping h-day P&L sums ``sum(pnl[t:t+h])`` for ``t = 0..n-h``.

    Overlapping sums are serially dependent, so the effective sample size is
    about ``n/h``.
    """
    horizon_days = integer("horizon_days", horizon_days, min_value=1)
    arr = finite_array("pnl", pnl, ndim=1, min_size=0)
    if arr.size < horizon_days + MIN_OBS:
        raise ValueError(
            f"need at least {horizon_days + MIN_OBS} observations for {horizon_days}-day "
            f"overlapping windows, got {arr.size}"
        )
    csum = np.concatenate([[0.0], np.cumsum(arr)])
    return csum[horizon_days:] - csum[:-horizon_days]
