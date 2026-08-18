"""Seeded synthetic FX data generators -- the test-suite data source.

Every generator takes an explicit ``seed`` (or ``numpy.random.Generator``)
and is fully deterministic given it; no network access anywhere. Burn-in
periods are discarded so simulated series start near the stationary
distribution.

Generators cover the FX-specific scenarios the tests and docs exercise:
G10-style symmetric fat-tailed series, EM-style asymmetric jumpy series,
correlated pair legs for triangle tests, scheduled-event (GARCH-X) series,
pegged currencies (HKD-style near-zero vol) and depegs (CHF-2015-style
single-day jump).
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from .._mle import student_t_abs_moment

__all__ = [
    "as_generator",
    "simulate_constant_vol",
    "simulate_garch",
    "simulate_gjr",
    "simulate_egarch",
    "simulate_garch_x",
    "simulate_correlated_pairs",
    "simulate_em_series",
    "simulate_pegged",
    "simulate_depeg",
    "simulate_seasonal_returns",
]


def as_generator(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Normalize a seed / Generator / None into a numpy Generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _innovations(rng: np.random.Generator, n: int, dist: str, nu: float | None) -> np.ndarray:
    if dist == "gaussian":
        return rng.standard_normal(n)
    if dist == "t":
        if nu is None or nu <= 2.0:
            raise ValueError("Student-t innovations need nu > 2")
        return rng.standard_t(nu, n) * sqrt((nu - 2.0) / nu)  # unit variance
    raise ValueError(f"dist must be 'gaussian' or 't', got {dist!r}")


def simulate_constant_vol(
    n: int,
    vol: float,
    seed: int | np.random.Generator | None = 0,
    dist: str = "gaussian",
    nu: float | None = None,
) -> np.ndarray:
    """i.i.d. zero-mean returns with per-period volatility ``vol``."""
    if n < 1 or vol < 0:
        raise ValueError("need n >= 1 and vol >= 0")
    rng = as_generator(seed)
    return vol * _innovations(rng, n, dist, nu)


def simulate_garch(
    n: int,
    omega: float,
    alpha: float,
    beta: float,
    dist: str = "gaussian",
    nu: float | None = None,
    seed: int | np.random.Generator | None = 0,
    burn: int = 500,
    return_sigma2: bool = False,
):
    """Simulate a GARCH(1,1) return series with known parameters.

    Returns per-period log returns (and optionally the true conditional
    variance path) after discarding ``burn`` observations.
    """
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        raise ValueError("require omega > 0, alpha, beta >= 0, alpha + beta < 1")
    rng = as_generator(seed)
    total = n + burn
    z = _innovations(rng, total, dist, nu)
    sigma2 = np.empty(total)
    r = np.empty(total)
    s2 = omega / (1.0 - alpha - beta)
    for t in range(total):
        sigma2[t] = s2
        r[t] = sqrt(s2) * z[t]
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    if return_sigma2:
        return r[burn:], sigma2[burn:]
    return r[burn:]


def simulate_gjr(
    n: int,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    dist: str = "gaussian",
    nu: float | None = None,
    seed: int | np.random.Generator | None = 0,
    burn: int = 500,
    return_sigma2: bool = False,
):
    """Simulate GJR-GARCH(1,1,1); requires ``alpha + gamma/2 + beta < 1``."""
    if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0 or alpha + 0.5 * gamma + beta >= 1:
        raise ValueError("require omega > 0, alpha, gamma, beta >= 0, alpha + gamma/2 + beta < 1")
    rng = as_generator(seed)
    total = n + burn
    z = _innovations(rng, total, dist, nu)
    sigma2 = np.empty(total)
    r = np.empty(total)
    s2 = omega / (1.0 - alpha - 0.5 * gamma - beta)
    for t in range(total):
        sigma2[t] = s2
        r[t] = sqrt(s2) * z[t]
        s2 = omega + (alpha + gamma * (r[t] < 0.0)) * r[t] ** 2 + beta * s2
    if return_sigma2:
        return r[burn:], sigma2[burn:]
    return r[burn:]


def simulate_egarch(
    n: int,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    dist: str = "gaussian",
    nu: float | None = None,
    seed: int | np.random.Generator | None = 0,
    burn: int = 500,
    return_sigma2: bool = False,
):
    """Simulate EGARCH(1,1); requires ``|beta| < 1``. ``gamma`` may be any sign."""
    if not abs(beta) < 1.0:
        raise ValueError("require |beta| < 1")
    rng = as_generator(seed)
    total = n + burn
    z = _innovations(rng, total, dist, nu)
    am = sqrt(2.0 / np.pi) if dist == "gaussian" else student_t_abs_moment(nu)
    sigma2 = np.empty(total)
    r = np.empty(total)
    ls2 = omega / (1.0 - beta)
    for t in range(total):
        s2 = np.exp(ls2)
        sigma2[t] = s2
        r[t] = sqrt(s2) * z[t]
        ls2 = omega + beta * ls2 + alpha * (abs(z[t]) - am) + gamma * z[t]
    if return_sigma2:
        return r[burn:], sigma2[burn:]
    return r[burn:]


def simulate_garch_x(
    n: int,
    omega: float,
    alpha: float,
    beta: float,
    gamma_x: float,
    event_prob: float = 0.05,
    dist: str = "gaussian",
    nu: float | None = None,
    seed: int | np.random.Generator | None = 0,
    burn: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate GARCH-X with scheduled-event variance dummies.

    Event days (probability ``event_prob``, e.g. ~8 FOMC + ~8 ECB meetings a
    year out of 252 days ~ 0.06) receive an extra ``gamma_x`` of variance:
    ``sigma2_t = omega + gamma_x * x_t + alpha r_{t-1}^2 + beta sigma2_{t-1}``.
    The calendar is known in advance -- x is drawn once and treated as
    deterministic.

    Returns
    -------
    (returns, x) : tuple of arrays, both length n.
    """
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        raise ValueError("require omega > 0, alpha, beta >= 0, alpha + beta < 1")
    if gamma_x < 0:
        raise ValueError("gamma_x must be non-negative")
    if not 0.0 <= event_prob < 1.0:
        raise ValueError("event_prob must be in [0, 1)")
    rng = as_generator(seed)
    total = n + burn
    x = (rng.random(total) < event_prob).astype(float)
    z = _innovations(rng, total, dist, nu)
    r = np.empty(total)
    s2 = (omega + gamma_x * event_prob) / (1.0 - alpha - beta)
    for t in range(total):
        s2_t = s2 if t == 0 else omega + gamma_x * x[t] + alpha * r[t - 1] ** 2 + beta * s2_prev
        r[t] = sqrt(s2_t) * z[t]
        s2_prev = s2_t
    return r[burn:], x[burn:]


def simulate_correlated_pairs(
    n: int,
    vol1: float,
    vol2: float,
    rho: float,
    s1_0: float = 1.10,
    s2_0: float = 150.0,
    seed: int | np.random.Generator | None = 0,
) -> dict:
    """Simulate two correlated constant-vol FX legs (e.g. EURUSD & USDJPY).

    Gaussian log returns with per-period vols ``vol1, vol2`` and correlation
    ``rho``; price paths start at ``s1_0, s2_0``. Used by the triangle tests:
    the cross built from the legs must have variance
    ``vol1^2 + vol2^2 + 2 rho vol1 vol2`` (signs per quote direction).

    Returns
    -------
    dict with ``returns1``, ``returns2``, ``prices1``, ``prices2``.
    """
    if vol1 < 0 or vol2 < 0:
        raise ValueError("vols must be non-negative")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    rng = as_generator(seed)
    z = rng.standard_normal((n, 2))
    r1 = vol1 * z[:, 0]
    r2 = vol2 * (rho * z[:, 0] + sqrt(max(1.0 - rho ** 2, 0.0)) * z[:, 1])
    p1 = s1_0 * np.exp(np.concatenate([[0.0], np.cumsum(r1)]))
    p2 = s2_0 * np.exp(np.concatenate([[0.0], np.cumsum(r2)]))
    return {"returns1": r1, "returns2": r2, "prices1": p1, "prices2": p2}


def simulate_em_series(
    n: int,
    omega: float = 2e-6,
    alpha: float = 0.06,
    gamma: float = 0.14,
    beta: float = 0.82,
    nu: float = 4.5,
    jump_prob: float = 0.004,
    jump_scale: float = 0.03,
    seed: int | np.random.Generator | None = 0,
    burn: int = 500,
) -> np.ndarray:
    """EM-style pair (think USDMXN/USDTRY): asymmetric GJR + fat t tails + jumps.

    The asymmetry loads on *positive* pair returns in USD/EM quote direction
    -- EM depreciation (pair up) raises vol -- implemented by simulating GJR
    on the *negated* innovation sign and flipping back, plus rare one-sided
    depreciation jumps of typical size ``jump_scale``.
    """
    rng = as_generator(seed)
    # GJR with asymmetry on negative returns, then negate the series so the
    # asymmetry sits on positive (EM-depreciation) returns of the quoted pair.
    r = simulate_gjr(n, omega, alpha, gamma, beta, dist="t", nu=nu, seed=rng, burn=burn)
    r = -r
    jumps = (rng.random(n) < jump_prob) * rng.exponential(jump_scale, n)
    return r + jumps  # jumps are positive: EM currency sells off


def simulate_pegged(
    n: int,
    band_vol: float = 2e-5,
    seed: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """Hard-pegged / tightly banded pair (HKD-style): near-zero, non-zero vol.

    Mean-reverting micro-noise inside a band -- daily vol of order 1-3 bp.
    Models must fit without blowing up (internal rescaling handles the tiny
    scale); persistence often pins near the IGARCH boundary, which is a
    documented behaviour, not a bug (docs/VALIDATION.md).
    """
    rng = as_generator(seed)
    return band_vol * rng.standard_normal(n)


def simulate_depeg(
    n: int,
    jump_return: float = -0.15,
    jump_index: int | None = None,
    base_vol: float = 0.002,
    alpha: float = 0.05,
    beta: float = 0.90,
    seed: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """CHF-2015-style depeg: GARCH background with a single injected jump.

    A GARCH(1,1) series with unconditional daily vol ``base_vol`` in which
    the return on ``jump_index`` is overwritten with ``jump_return`` (default
    -15%, the EURCHF move of 15 Jan 2015 when the SNB floor was removed).
    Crucially, the injected jump feeds through the GARCH recursion, so
    post-jump volatility genuinely spikes and then decays -- the structure
    the depeg tests verify fitted models can capture.
    """
    if not 0 < abs(jump_return) < 1:
        raise ValueError("jump_return must be a non-zero return with |r| < 1")
    if alpha < 0 or beta < 0 or alpha + beta >= 1:
        raise ValueError("require alpha, beta >= 0 and alpha + beta < 1")
    rng = as_generator(seed)
    if jump_index is None:
        jump_index = int(0.75 * n)
    if not 0 < jump_index < n - 1:
        raise ValueError(f"jump_index must be inside (0, {n - 1}), got {jump_index}")
    omega = base_vol ** 2 * (1.0 - alpha - beta)
    z = rng.standard_normal(n)
    r = np.empty(n)
    s2 = base_vol ** 2
    for t in range(n):
        r[t] = jump_return if t == jump_index else sqrt(s2) * z[t]
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return r


def simulate_seasonal_returns(
    n: int,
    weekday_factors: dict[int, float] | None = None,
    base_vol: float = 0.006,
    start: str = "2018-01-01",
    seed: int | np.random.Generator | None = 0,
) -> pd.Series:
    """Business-day return series with injected day-of-week vol seasonality.

    ``weekday_factors`` maps weekday number (0=Monday .. 4=Friday) to a
    multiplicative vol factor; defaults to a stylized FX week (quiet Monday,
    busy Wednesday/Friday -- FOMC and NFP days).
    """
    if weekday_factors is None:
        weekday_factors = {0: 0.85, 1: 0.95, 2: 1.15, 3: 1.0, 4: 1.10}
    rng = as_generator(seed)
    idx = pd.bdate_range(start=start, periods=n)
    factors = np.array([weekday_factors.get(d, 1.0) for d in idx.dayofweek])
    r = base_vol * factors * rng.standard_normal(n)
    return pd.Series(r, index=idx, name="returns")
