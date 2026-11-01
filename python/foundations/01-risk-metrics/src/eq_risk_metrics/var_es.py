"""Value at Risk (three methods) and Expected Shortfall.

Convention: VaR and Expected Shortfall are reported as **positive
numbers representing a loss** (i.e. ``VaR 95% = 0.021`` means "on the
worst 5% of days you lose at least 2.1% of position value"). This is
the opposite sign convention to raw returns, where a loss is negative --
every function here negates internally so the caller never has to.

All three VaR estimators answer the same question -- "what daily loss is
exceeded with probability ``1 - confidence``?" -- under different
assumptions about the return distribution; see
``docs/METHODOLOGY.md`` for the full comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "cornish_fisher_valid",
    "var_historical",
    "var_parametric",
    "var_cornish_fisher",
    "expected_shortfall",
]


def _validated(returns: pd.Series, confidence: float, func: str) -> pd.Series:
    """Coerce ``returns`` to a Series and validate it against ``confidence``.

    Shared front door for all four estimators, so every one of them fails
    the same way with the same message rather than surfacing whatever the
    underlying NumPy/SciPy call happens to raise (or, worse, silently
    returning ``NaN`` -- which ``scipy.stats.norm.ppf`` does for an
    out-of-range confidence).

    Parameters
    ----------
    returns : pandas.Series or array_like
        Candidate return sample.
    confidence : float
        Candidate confidence level; must be strictly inside ``(0, 1)``.
    func : str
        Caller name, used in the error message.

    Returns
    -------
    pandas.Series
        ``returns`` as a Series (a no-op copy-free conversion when it
        already is one).

    Raises
    ------
    ValueError
        If ``returns`` is empty or contains a non-finite value, or if
        ``confidence`` is not a finite number strictly between 0 and 1.
    """
    returns = pd.Series(returns)
    if len(returns) == 0:
        raise ValueError(
            f"{func}: cannot compute a quantile from an empty return series "
            "(need at least 1 observation; a desk-grade estimate needs "
            "hundreds -- see docs/VALIDATION.md 3.1)"
        )
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError(
            f"{func}: confidence must be a finite number strictly between 0 "
            f"and 1 (e.g. 0.95, 0.99), got {confidence!r}"
        )
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        n_bad = int((~np.isfinite(returns.to_numpy(dtype=float))).sum())
        raise ValueError(
            f"{func}: returns contains {n_bad} non-finite value(s) (NaN/inf). "
            "A single inf or NaN silently poisons the whole estimate -- "
            "numpy.percentile and the mean/std path both return NaN -- so it "
            "is rejected here instead. Clean the series first (a NaN usually "
            "means a missing price, an inf a zero price in the denominator)."
        )
    return returns


def _tail_order_stat(returns: pd.Series, confidence: float) -> tuple[np.ndarray, int, float]:
    """Losses sorted descending, the VaR rank m and k = n * alpha.

    ``alpha = 1 - confidence`` is the tail probability. ``k = n * alpha``,
    ``m = ceil(k - 1e-9)`` clamped to ``[1, n]``. VaR is the m-th largest loss.
    """
    losses = np.sort(-returns.to_numpy(dtype=float))[::-1]
    n = losses.size
    k = n * (1.0 - confidence)
    m = int(min(max(np.ceil(k - 1e-9), 1), n))
    return losses, m, k


def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical (empirical) Value at Risk.

    No distributional assumption -- just the empirical quantile of the
    realised return sample.

    Assumption: the past sample window is representative of the
    future. A calm sample understates future risk; a stressed sample
    overstates it. Also unstable in small samples, since it is a
    single order statistic.

    Estimator: with losses ``L = -r`` sorted descending,
    ``k = n(1 - confidence)`` and ``m = ceil(k)`` clamped to ``[1, n]``,
    VaR is ``L(m)``. There is no interpolation.

    Parameters
    ----------
    returns : pandas.Series or array_like
        Simple daily returns, unitless.
    confidence : float
        Confidence level in (0, 1), e.g. 0.95 or 0.99.

    Returns
    -------
    float
        VaR as a positive loss fraction (unitless).

    Raises
    ------
    ValueError
        If ``returns`` is empty or ``confidence`` is outside (0, 1).
    """
    returns = _validated(returns, confidence, "var_historical")
    losses, m, _ = _tail_order_stat(returns, confidence)
    return float(losses[m - 1])


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """Gaussian (variance-covariance / delta-normal) Value at Risk.

    Assumes ``returns ~ Normal(mu, sigma)`` and reports the
    corresponding quantile: ``VaR = -(mu + sigma * z)`` where
    ``z = Phi^{-1}(1 - confidence)`` is negative for confidence > 50%,
    so ``VaR`` is positive whenever the loss quantile is below the mean.

    Assumption: normality. Daily equity returns have excess kurtosis
    (fat tails), so this systematically *understates* tail risk at high
    confidence levels (typically visible above ~97.5%). Included
    precisely to demonstrate that gap against
    :func:`var_historical` -- see ``docs/VALIDATION.md``.

    Parameters
    ----------
    returns : pandas.Series or array_like
        Simple daily returns, unitless.
    confidence : float
        Confidence level in (0, 1).

    Returns
    -------
    float
        VaR as a positive loss fraction (unitless).

    Raises
    ------
    ValueError
        If ``returns`` has fewer than 2 observations, contains a non-finite
        value, or ``confidence`` is outside (0, 1).
    """
    returns = _validated(returns, confidence, "var_parametric")
    if len(returns) < 2:
        raise ValueError(
            "var_parametric: need at least 2 observations to estimate a "
            f"standard deviation, got {len(returns)}"
        )
    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = stats.norm.ppf(1 - confidence)
    return -(mu + sigma * z)


def cornish_fisher_valid(skew: float, ex_kurt: float, alpha: float) -> bool:
    """Whether the Cornish-Fisher map is monotone over the quantiles used at ``alpha``.

    ``alpha`` is the tail probability (0.01 for 99 %). With
    ``z_a = Phi^{-1}(1 - alpha)``, the map
    ``g(z) = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36`` is valid when
    ``g'(z) > 0`` on the closed interval ``[-z_a, z_a]``. ``g'`` is the
    quadratic ``A z^2 + B z + C`` with ``A = K/8 - S^2/6``, ``B = S/3`` and
    ``C = 1 - K/8 + 5 S^2/36``. Its minimum on the interval is at an endpoint,
    or at the vertex when the vertex lies inside.

    Raises
    ------
    ValueError
        If any input is non-finite or ``alpha`` is outside (0, 0.5).
    """
    for name, v in (("skew", skew), ("ex_kurt", ex_kurt), ("alpha", alpha)):
        if not np.isfinite(v):
            raise ValueError(f"cornish_fisher_valid: {name} must be finite, got {v!r}")
    if not 0.0 < alpha < 0.5:
        raise ValueError(
            "cornish_fisher_valid: alpha is the tail probability "
            f"(e.g. 0.01 for 99 %), must be in (0, 0.5), got {alpha!r}"
        )
    s, k = float(skew), float(ex_kurt)
    za = float(stats.norm.ppf(1.0 - alpha))
    a = k / 8.0 - s * s / 6.0
    b = s / 3.0
    c = 1.0 - k / 8.0 + 5.0 * s * s / 36.0

    def d(z: float) -> float:
        return a * z * z + b * z + c

    candidates = [d(-za), d(za)]
    if a != 0.0:
        zv = -b / (2.0 * a)
        if -za <= zv <= za:
            candidates.append(d(zv))
    return bool(min(candidates) > 0.0)


def var_cornish_fisher(returns: pd.Series, confidence: float = 0.95) -> float:
    """Modified (Cornish-Fisher) Value at Risk.

    Adjusts the Gaussian quantile ``z`` for the sample's skewness ``s``
    and excess kurtosis ``k`` via the Cornish-Fisher expansion:

    ``z_cf = z + (z^2 - 1) s / 6 + (z^3 - 3z) k / 24 - (2z^3 - 5z) s^2 / 36``

    then applies it the same way as the Gaussian formula:
    ``VaR = -(mu + sigma * z_cf)``. When ``s = k = 0`` this collapses
    exactly to :func:`var_parametric` (``z_cf = z``).

    A pragmatic middle ground between fully parametric and fully
    empirical: still assumes a smooth quantile function, but
    acknowledges non-normality via the first four moments.

    Assumption: the expansion is a local (Edgeworth-style) correction
    around the normal and is only reliable for mild skew/kurtosis; at
    extreme confidence levels or with heavily fat-tailed samples it can
    become non-monotonic in ``z`` and produce a *worse* estimate than
    the plain Gaussian VaR. It is also unstable when skew/kurtosis are
    estimated from small samples (high sampling variance of the third
    and fourth moments).

    Parameters
    ----------
    returns : pandas.Series or array_like
        Simple daily returns, unitless.
    confidence : float
        Confidence level in (0, 1).

    Returns
    -------
    float
        VaR as a positive loss fraction (unitless).

    Raises
    ------
    ValueError
        If ``returns`` has fewer than 2 observations, is constant, contains a
        non-finite value, ``confidence`` is outside (0.5, 1), or the sample
        skew and kurtosis fail :func:`cornish_fisher_valid` at
        ``alpha = 1 - confidence``.
    """
    returns = _validated(returns, confidence, "var_cornish_fisher")
    alpha = 1.0 - confidence
    if not 0.0 < alpha < 0.5:
        raise ValueError(
            "var_cornish_fisher: confidence must be in (0.5, 1) so the tail "
            f"probability is in (0, 0.5), got {confidence!r}"
        )
    if len(returns) < 2:
        raise ValueError(
            f"var_cornish_fisher: need at least 2 observations, got {len(returns)}"
        )
    mu, sigma = returns.mean(), returns.std(ddof=1)
    if not sigma > 0.0 or np.ptp(returns.to_numpy(dtype=float)) == 0.0:
        raise ValueError(
            "var_cornish_fisher: the sample is constant, so skewness and "
            "kurtosis are undefined"
        )
    s = stats.skew(returns)
    k = stats.kurtosis(returns)  # excess kurtosis (Fisher definition)
    if not (np.isfinite(s) and np.isfinite(k)):
        raise ValueError("var_cornish_fisher: sample skewness or kurtosis is not finite")
    if not cornish_fisher_valid(s, k, alpha):
        raise ValueError(
            "var_cornish_fisher: the Cornish-Fisher map is not monotone on "
            f"[-z, z] for skew={s:.4g}, excess kurtosis={k:.4g} at "
            f"confidence {confidence}, so the expansion is outside its domain"
        )
    z = stats.norm.ppf(1 - confidence)
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * k / 24
        - (2 * z**3 - 5 * z) * s**2 / 36
    )
    return -(mu + sigma * z_cf)


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Expected Shortfall (Acerbi-Tasche), positive loss convention.

    With losses ``L = -r`` sorted descending, ``alpha = 1 - confidence``,
    ``k = n * alpha`` and ``m = ceil(k)`` clamped to ``[1, n]``:

    ``ES = [ sum_{i<m} L(i) + (k - (m-1)) * L(m) ] / k``

    This is the average of the worst ``k`` losses, with a fractional weight
    on the boundary loss when ``k`` is not an integer (Acerbi and Tasche,
    2002). It is the integral of the empirical quantile function over the
    tail, so ``ES >= var_historical`` at the same confidence always holds.

    ES as a functional of the loss distribution is coherent: monotone,
    translation invariant, positively homogeneous and subadditive. This
    estimator is the ES of the empirical distribution, so it is subadditive
    for positions evaluated on the same scenarios. It is still an estimate,
    and with few tail observations its sampling error is large.

    Parameters
    ----------
    returns : pandas.Series or array_like
        Simple daily returns, unitless.
    confidence : float
        Confidence level in (0, 1).

    Returns
    -------
    float
        Expected Shortfall as a positive loss fraction (unitless).

    Raises
    ------
    ValueError
        If ``returns`` is empty, contains a non-finite value, or
        ``confidence`` is outside (0, 1).
    """
    returns = _validated(returns, confidence, "expected_shortfall")
    losses, m, k = _tail_order_stat(returns, confidence)
    return float((losses[: m - 1].sum() + (k - (m - 1)) * losses[m - 1]) / k)
