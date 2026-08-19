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


def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical (empirical) Value at Risk.

    No distributional assumption -- just the empirical quantile of the
    realised return sample.

    Assumption: the past sample window is representative of the
    future. A calm sample understates future risk; a stressed sample
    overstates it. Also unstable in small samples, since it is a
    single order statistic (or an interpolation between two).

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
    return -float(np.percentile(returns, 100 * (1 - confidence)))


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
        VaR as a positive loss fraction (unitless). ``NaN`` when the
        sample has fewer than 2 observations (``std(ddof=1)`` is
        undefined), which is a genuine "not enough data" answer rather
        than a misleadingly precise number.

    Raises
    ------
    ValueError
        If ``returns`` is empty or ``confidence`` is outside (0, 1).
    """
    returns = _validated(returns, confidence, "var_parametric")
    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = stats.norm.ppf(1 - confidence)
    return -(mu + sigma * z)


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
        VaR as a positive loss fraction (unitless). ``NaN`` for a sample
        with fewer than 2 observations, or an exactly-constant one (see
        ``docs/VALIDATION.md`` 3.2).

    Raises
    ------
    ValueError
        If ``returns`` is empty or ``confidence`` is outside (0, 1).
    """
    returns = _validated(returns, confidence, "var_cornish_fisher")
    mu, sigma = returns.mean(), returns.std(ddof=1)
    s = stats.skew(returns)
    k = stats.kurtosis(returns)  # excess kurtosis (Fisher definition)
    z = stats.norm.ppf(1 - confidence)
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * k / 24
        - (2 * z**3 - 5 * z) * s**2 / 36
    )
    return -(mu + sigma * z_cf)


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Expected Shortfall (CVaR), positive loss convention.

    The average loss *given* that the loss exceeds (historical) VaR at
    the same confidence level:
    ``ES = -E[r | r <= -VaR_historical(confidence)]``.

    Unlike VaR, Expected Shortfall is a *coherent* risk measure
    (sub-additive: the ES of a combined position never exceeds the sum
    of the parts), and it answers "how bad is the tail on average", not
    just "where does the tail start". ``ES >= VaR`` always holds at the
    same confidence level, by construction.

    Assumption: same as :func:`var_historical` (representative sample
    window; unstable in small samples -- ES additionally averages over
    the tail, which can be very few observations at 99% confidence in
    a short sample).

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
        ``NaN`` if no observations fall in the tail (e.g. a degenerate
        or too-short sample).

    Raises
    ------
    ValueError
        If ``returns`` is empty or ``confidence`` is outside (0, 1).
    """
    returns = _validated(returns, confidence, "expected_shortfall")
    var = var_historical(returns, confidence)
    tail = returns[returns <= -var]
    return -tail.mean()
