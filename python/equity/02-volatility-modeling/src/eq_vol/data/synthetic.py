"""Seeded synthetic data generators with known true parameters.

All tests run offline against these generators (CONVENTIONS.md: no network in
test code paths). Every generator takes an explicit ``seed`` /
:class:`numpy.random.Generator` and returns daily log-returns in decimal units.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from .._utils import TRADING_DAYS, as_generator


class SimulatedSeries(NamedTuple):
    """Simulated returns together with the true conditional variance path."""

    returns: np.ndarray
    sigma2: np.ndarray


def _innovations(
    rng: np.random.Generator, n: int, dist: str, nu: float
) -> np.ndarray:
    """Unit-variance innovations: standard normal or standardised Student-t."""
    if dist == "normal":
        return rng.standard_normal(n)
    if dist == "t":
        if nu <= 2:
            raise ValueError("Student-t innovations require nu > 2 for finite variance.")
        raw = rng.standard_t(nu, n)
        return raw * np.sqrt((nu - 2.0) / nu)  # rescale to unit variance
    raise ValueError(f"unknown dist {dist!r}; use 'normal' or 't'")


def simulate_gbm_returns(
    n: int,
    sigma_annual: float = 0.20,
    mu_annual: float = 0.0,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Daily log-returns of a GBM with constant annualised volatility.

    r_t = (mu - sigma^2/2) dt + sigma sqrt(dt) z_t, dt = 1/252.
    """
    rng = as_generator(seed)
    dt = 1.0 / TRADING_DAYS
    drift = (mu_annual - 0.5 * sigma_annual**2) * dt
    return drift + sigma_annual * np.sqrt(dt) * rng.standard_normal(n)


def simulate_gbm_ohlc(
    n_days: int,
    sigma_annual: float = 0.20,
    mu_annual: float = 0.0,
    steps_per_day: int = 390,
    s0: float = 100.0,
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Daily OHLC bars from a finely discretised GBM path.

    Used to test range-based estimators (Parkinson, Garman-Klass,
    Rogers-Satchell). Note the discrete-monitoring bias: with ``m`` intraday
    steps the observed high/low understate the continuous extremes by
    ``O(1/sqrt(m))``, so range estimators are biased slightly *downward* in
    tests; tolerances account for this.

    Returns
    -------
    pandas.DataFrame
        Columns ``open, high, low, close`` (no overnight gap: open equals the
        previous close, so Garman-Klass and Rogers-Satchell apply cleanly).
    """
    rng = as_generator(seed)
    dt = 1.0 / (TRADING_DAYS * steps_per_day)
    drift = (mu_annual - 0.5 * sigma_annual**2) * dt
    steps = drift + sigma_annual * np.sqrt(dt) * rng.standard_normal((n_days, steps_per_day))
    within_day = np.cumsum(steps, axis=1)          # cumulative within each day
    day_total = within_day[:, -1]                  # each day's total log-return
    log_open = np.log(s0) + np.concatenate(([0.0], np.cumsum(day_total)[:-1]))
    # prepend the open (0 offset) so highs/lows include the opening print
    rel = np.concatenate((np.zeros((n_days, 1)), within_day), axis=1)
    log_px = log_open[:, None] + rel
    return pd.DataFrame(
        {
            "open": np.exp(log_px[:, 0]),
            "high": np.exp(log_px.max(axis=1)),
            "low": np.exp(log_px.min(axis=1)),
            "close": np.exp(log_px[:, -1]),
        }
    )


def simulate_garch(
    n: int,
    omega: float = 5e-6,
    alpha: float = 0.05,
    beta: float = 0.90,
    dist: str = "normal",
    nu: float = 8.0,
    burn: int = 500,
    seed: int | np.random.Generator | None = None,
) -> SimulatedSeries:
    """Simulate a GARCH(1,1) process with known true parameters.

    sigma2_t = omega + alpha r_{t-1}^2 + beta sigma2_{t-1};  r_t = sigma_t z_t.

    Defaults give unconditional daily variance 1e-4 (about 16% annualised vol)
    with persistence 0.95. A ``burn``-in draws the recursion into its
    stationary distribution before returning ``n`` observations.
    """
    if not (omega > 0 and alpha >= 0 and beta >= 0 and alpha + beta < 1):
        raise ValueError("require omega>0, alpha,beta>=0, alpha+beta<1 for simulation")
    rng = as_generator(seed)
    z = _innovations(rng, n + burn, dist, nu)
    total = n + burn
    sigma2 = np.empty(total)
    r = np.empty(total)
    prev_r = 0.0
    for t in range(total):
        if t == 0:
            sigma2[t] = omega / (1.0 - alpha - beta)
        else:
            sigma2[t] = omega + alpha * prev_r**2 + beta * sigma2[t - 1]
        r[t] = prev_r = np.sqrt(sigma2[t]) * z[t]
    return SimulatedSeries(r[burn:], sigma2[burn:])


def simulate_gjr(
    n: int,
    omega: float = 5e-6,
    alpha: float = 0.03,
    gamma: float = 0.10,
    beta: float = 0.88,
    dist: str = "normal",
    nu: float = 8.0,
    burn: int = 500,
    seed: int | np.random.Generator | None = None,
) -> SimulatedSeries:
    """Simulate a GJR-GARCH(1,1) process (leverage via indicator term).

    sigma2_t = omega + (alpha + gamma * 1[r_{t-1} < 0]) r_{t-1}^2
               + beta sigma2_{t-1}.

    ``gamma > 0`` means negative shocks raise next-period variance more than
    positive shocks of equal size (the equity leverage effect). Stationarity
    (symmetric innovations): alpha + gamma/2 + beta < 1.
    """
    persistence = alpha + 0.5 * gamma + beta
    if not (omega > 0 and alpha >= 0 and alpha + gamma >= 0 and beta >= 0 and persistence < 1):
        raise ValueError(
            "require omega>0, alpha>=0, alpha+gamma>=0, beta>=0 and "
            "alpha + gamma/2 + beta < 1 for simulation"
        )
    rng = as_generator(seed)
    z = _innovations(rng, n + burn, dist, nu)
    total = n + burn
    sigma2 = np.empty(total)
    r = np.empty(total)
    v = omega / (1.0 - persistence)
    prev_r = 0.0
    for t in range(total):
        if t == 0:
            sigma2[t] = v
        else:
            a_eff = alpha + (gamma if prev_r < 0 else 0.0)
            sigma2[t] = omega + a_eff * prev_r**2 + beta * sigma2[t - 1]
        r[t] = prev_r = np.sqrt(sigma2[t]) * z[t]
    return SimulatedSeries(r[burn:], sigma2[burn:])


def simulate_egarch(
    n: int,
    omega: float = -0.40,
    alpha: float = 0.10,
    gamma: float = -0.08,
    beta: float = 0.96,
    dist: str = "normal",
    nu: float = 8.0,
    burn: int = 500,
    seed: int | np.random.Generator | None = None,
) -> SimulatedSeries:
    """Simulate an EGARCH(1,1) process.

    ln sigma2_t = omega + beta ln sigma2_{t-1}
                  + alpha (|z_{t-1}| - E|z|) + gamma z_{t-1}.

    Sign convention: **gamma < 0** produces the leverage effect (negative
    shocks raise volatility more). Stationarity requires |beta| < 1; no
    positivity constraints are needed because variance is exponentiated.
    Unconditional log-variance is omega / (1 - beta).
    """
    if not abs(beta) < 1:
        raise ValueError("require |beta| < 1 for stationarity of EGARCH simulation")
    rng = as_generator(seed)
    z = _innovations(rng, n + burn, dist, nu)
    if dist == "normal":
        e_abs_z = np.sqrt(2.0 / np.pi)
    else:
        # E|t_nu standardized| = 2 sqrt(nu-2) Gamma((nu+1)/2) / (sqrt(pi) (nu-1) Gamma(nu/2))
        from scipy.special import gammaln

        e_abs_z = (
            2.0
            * np.sqrt(nu - 2.0)
            * np.exp(gammaln((nu + 1) / 2.0) - gammaln(nu / 2.0))
            / (np.sqrt(np.pi) * (nu - 1.0))
        )
    total = n + burn
    log_s2 = np.empty(total)
    r = np.empty(total)
    ls = omega / (1.0 - beta)
    prev_z = 0.0
    for t in range(total):
        if t == 0:
            log_s2[t] = ls
        else:
            log_s2[t] = omega + beta * log_s2[t - 1] + alpha * (abs(prev_z) - e_abs_z) + gamma * prev_z
        r[t] = np.sqrt(np.exp(log_s2[t])) * z[t]
        prev_z = z[t]
    return SimulatedSeries(r[burn:], np.exp(log_s2[burn:]))


def simulate_crisis(
    n_pre: int = 750,
    n_crisis: int = 60,
    n_post: int = 250,
    sigma_pre_annual: float = 0.15,
    sigma_crisis_annual: float = 0.75,
    sigma_post_annual: float = 0.30,
    seed: int | np.random.Generator | None = None,
) -> SimulatedSeries:
    """COVID-Mar-2020-style volatility regime jump (structural break).

    Piecewise-constant true volatility: a calm regime, a sudden crisis regime
    (default 15% -> 75% annualised, comparable to S&P realised vol in
    March 2020), then an elevated after-shock regime. Used to study how each
    model adapts to a structural break it does not know about
    (docs/VALIDATION.md case study).
    """
    rng = as_generator(seed)
    sig = np.concatenate(
        [
            np.full(n_pre, sigma_pre_annual),
            np.full(n_crisis, sigma_crisis_annual),
            np.full(n_post, sigma_post_annual),
        ]
    )
    daily_var = sig**2 / TRADING_DAYS
    r = np.sqrt(daily_var) * rng.standard_normal(sig.size)
    return SimulatedSeries(r, daily_var)
