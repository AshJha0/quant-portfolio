"""Historical-simulation VaR: plain, age-weighted (BRW) and filtered (FHS).

Conventions
-----------
* ``pnl`` arrays are P&L in currency units, loss < 0.
* ``alpha`` is the tail probability: ``alpha=0.01`` -> 99 % VaR.
* VaR is reported as a **positive** number for a loss:
  ``VaR_alpha = -Q_alpha(pnl)``.
* Quantile interpolation: plain historical VaR uses NumPy's ``"linear"``
  (type-7) interpolation between order statistics.  This is the most common
  desk choice; alternatives (lower/higher order statistic) differ by O(1/n)
  and are documented in docs/METHODOLOGY.md.  Weighted quantiles (BRW)
  use the standard step-function inversion of the weighted empirical CDF.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = [
    "historical_var",
    "age_weighted_var",
    "brw_weights",
    "ewma_volatility",
    "filtered_historical_var",
    "scale_var_sqrt_time",
    "overlapping_horizon_pnl",
]

MIN_OBS = 50  # fewer observations than this cannot resolve a 1-5 % tail sensibly


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5) (tail probability), got {alpha}")


def _validate_pnl(pnl: np.ndarray, min_obs: int = MIN_OBS) -> np.ndarray:
    arr = np.asarray(pnl, dtype=float).ravel()
    if arr.size < min_obs:
        raise ValueError(
            f"need at least {min_obs} P&L observations for historical VaR, got {arr.size}; "
            "empirical tail quantiles are meaningless on shorter samples"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("pnl contains NaN or infinite values")
    return arr


def historical_var(pnl: np.ndarray, alpha: float = 0.01) -> float:
    """Plain historical-simulation VaR (equal weights).

    Parameters
    ----------
    pnl : array
        Historical scenario P&L (currency units, loss < 0).
    alpha : float
        Tail probability (0.01 -> 99 % VaR).

    Returns
    -------
    float
        VaR as a positive loss: ``-quantile_alpha(pnl)`` with linear (type-7)
        interpolation between order statistics.
    """
    _validate_alpha(alpha)
    arr = _validate_pnl(pnl)
    return float(-np.quantile(arr, alpha, method="linear"))


def brw_weights(n: int, lam: float = 0.98) -> np.ndarray:
    """Boudoukh-Richardson-Whitelaw exponential age weights.

    Observation ``i`` (0 = oldest, n-1 = most recent) gets weight proportional
    to ``lam**(n-1-i)``; weights sum to 1 exactly.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"decay lam must be in (0, 1), got {lam}")
    if n < 1:
        raise ValueError("n must be >= 1")
    ages = np.arange(n - 1, -1, -1, dtype=float)  # age of each obs, oldest first
    w = (1.0 - lam) * lam**ages / (1.0 - lam**n)
    return w


def age_weighted_var(pnl: np.ndarray, alpha: float = 0.01, lam: float = 0.98) -> float:
    """Age-weighted (BRW) historical VaR.

    Recent observations receive exponentially larger weight
    (``w_t ~ lam**age``).  The weighted empirical CDF is inverted at
    ``alpha``: VaR is minus the smallest P&L whose cumulative weight
    (ascending P&L order) reaches ``alpha``.

    Notes
    -----
    ``lam -> 1`` recovers plain historical simulation (up to the
    interpolation-scheme difference: BRW uses the step-CDF inversion).
    """
    _validate_alpha(alpha)
    arr = _validate_pnl(pnl)
    w = brw_weights(arr.size, lam)
    order = np.argsort(arr, kind="stable")
    sorted_pnl = arr[order]
    cum_w = np.cumsum(w[order])
    idx = int(np.searchsorted(cum_w, alpha, side="left"))
    idx = min(idx, arr.size - 1)
    return float(-sorted_pnl[idx])


def ewma_volatility(x: np.ndarray, lam: float = 0.94, init: Literal["sample", "first"] = "sample") -> np.ndarray:
    """One-step-ahead EWMA (RiskMetrics) volatility forecasts.

    ``sigma2[t] = lam * sigma2[t-1] + (1 - lam) * x[t-1]**2`` — i.e.
    ``sigma[t]`` is the forecast for day ``t`` made with information up to
    ``t-1``, so standardising ``x[t] / sigma[t]`` uses no look-ahead.

    Parameters
    ----------
    x : array
        Return or P&L series (mean assumed ~0 at daily horizon).
    lam : float
        Decay, RiskMetrics default 0.94 for daily data.
    init : {"sample", "first"}
        Seed variance: full-sample variance (default, stabler) or ``x[0]**2``.

    Returns
    -------
    ndarray, same length as ``x``, of vol forecasts ``sigma[t] > 0``.

    Raises
    ------
    ValueError
        If ``lam`` is outside (0, 1), fewer than 2 observations are supplied,
        ``init`` is unknown, or ``x`` contains NaN/infinite values.  The
        recursion is multiplicative in its own past, so a single non-finite
        input would poison every subsequent forecast; it is rejected up front
        rather than propagated.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"decay lam must be in (0, 1), got {lam}")
    if init not in ("sample", "first"):
        raise ValueError(f"init must be 'sample' or 'first', got {init!r}")
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError("need at least 2 observations for EWMA volatility")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x contains NaN or infinite values")
    sig2 = np.empty(arr.size)
    seed = float(np.var(arr)) if init == "sample" else float(arr[0] ** 2)
    if seed <= 0.0:
        seed = 1e-16  # zero-variance series: avoid division by zero downstream
    sig2[0] = seed
    for t in range(1, arr.size):
        sig2[t] = lam * sig2[t - 1] + (1.0 - lam) * arr[t - 1] ** 2
    return np.sqrt(np.maximum(sig2, 1e-32))


def filtered_historical_var(
    pnl: np.ndarray,
    alpha: float = 0.01,
    lam: float = 0.94,
) -> float:
    """Filtered historical simulation (FHS) VaR — the industry workhorse.

    Procedure (Barone-Adesi et al. / Hull-White devolatilisation):

    1. compute one-step-ahead EWMA vol forecasts ``sigma_t`` for each day;
    2. standardise: ``z_t = pnl_t / sigma_t`` (i.i.d.-ish innovations);
    3. rescale every innovation to *tomorrow's* vol forecast
       ``sigma_{T+1}^2 = lam * sigma_T^2 + (1-lam) * pnl_T^2``;
    4. take the empirical ``alpha`` quantile of the rescaled scenarios.

    This makes VaR respond to the current vol regime while keeping the
    empirical (fat-tailed, skewed) shape of the standardised innovations.
    """
    _validate_alpha(alpha)
    arr = _validate_pnl(pnl)
    sigma = ewma_volatility(arr, lam)
    z = arr / sigma
    sigma_next = np.sqrt(lam * sigma[-1] ** 2 + (1.0 - lam) * arr[-1] ** 2)
    scenarios = z * sigma_next
    return float(-np.quantile(scenarios, alpha, method="linear"))


def scale_var_sqrt_time(var_1d: float, horizon_days: int) -> float:
    """Square-root-of-time scaling: ``VaR_h = VaR_1 * sqrt(h)``.

    Valid only for i.i.d. returns with zero drift; understates multi-day risk
    under volatility clustering / autocorrelation (see docs/VALIDATION.md).
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    return float(var_1d) * float(np.sqrt(horizon_days))


def overlapping_horizon_pnl(pnl: np.ndarray, horizon_days: int) -> np.ndarray:
    """Overlapping h-day P&L sums for direct multi-day historical VaR.

    Returns the rolling sums ``sum(pnl[t:t+h])`` for
    ``t = 0..n-h``.  Caveat: overlapping sums are serially dependent, so the
    effective sample size is ~``n/h`` and quantile standard errors are much
    larger than the nominal count suggests (documented in METHODOLOGY.md).

    Raises
    ------
    ValueError
        If ``horizon_days < 1``, the series is too short, or ``pnl`` contains
        NaN/infinite values (one bad day would contaminate ``horizon_days``
        overlapping windows, so it is rejected rather than propagated).
    """
    arr = np.asarray(pnl, dtype=float).ravel()
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("pnl contains NaN or infinite values")
    if arr.size < horizon_days + MIN_OBS:
        raise ValueError(
            f"need at least {horizon_days + MIN_OBS} observations for {horizon_days}-day "
            f"overlapping windows, got {arr.size}"
        )
    csum = np.concatenate([[0.0], np.cumsum(arr)])
    return csum[horizon_days:] - csum[:-horizon_days]
