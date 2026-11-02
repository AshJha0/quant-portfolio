"""Robust tests for the difference of two Sharpe ratios.

The Jobson-Korkie / Memmel test (``metrics.sharpe_difference_test`` in the
equity package) assumes iid normal returns. Ledoit and Wolf (2008),
"Robust performance hypothesis testing with the Sharpe ratio", Journal of
Empirical Finance 15(5), give two replacements:

HAC delta method (section 3.1)
------------------------------
With v = (mu_1, mu_2, gamma_1, gamma_2), gamma_i = E[r_i^2], the
difference is f(v) = mu_1 / sqrt(gamma_1 - mu_1^2) - mu_2 / sqrt(gamma_2 - mu_2^2)
and sqrt(T) (f(v_hat) - f(v)) -> N(0, grad' Psi grad), where Psi is the
long-run covariance of y_t = (r_1t, r_2t, r_1t^2, r_2t^2). Psi is estimated
with the Newey-West (1987) Bartlett kernel at ``lags`` lags (Ledoit-Wolf use
a prewhitened quadratic-spectral kernel; Bartlett is the simpler choice and
is documented as such). Gradient:

    df/dmu_i    = +/- gamma_i / (gamma_i - mu_i^2)^{3/2}
    df/dgamma_i = -/+ mu_i / (2 (gamma_i - mu_i^2)^{3/2})

Under iid normal returns and lags = 0 the variance converges to Memmel's
(2003) [2 - 2 rho + (SR_1^2 + SR_2^2 - 2 SR_1 SR_2 rho^2) / 2] / T.

Circular block bootstrap (section 3.2, simplified)
--------------------------------------------------
Resample blocks of pairs (Politis and Romano 1992) and report the two-sided
p-value of the centred statistic, P*(|Delta* - Delta_hat| >= |Delta_hat|).
Ledoit-Wolf studentise each bootstrap draw; this basic version does not,
which is less accurate in small samples and is documented as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class SharpeHacResult:
    diff: float
    se: float
    z: float
    p_value: float


def _pair(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    a = np.asarray(r1, dtype=float).ravel()
    b = np.asarray(r2, dtype=float).ravel()
    if a.shape != b.shape or a.size < 10:
        raise ValueError("need two equal-length series of at least 10 observations")
    return np.column_stack([a, b])


def _sr_diff(x: np.ndarray) -> float:
    m = x.mean(axis=0)
    g = (x**2).mean(axis=0)
    s = np.sqrt(g - m**2)
    if np.any(s <= 0.0):
        raise ValueError("a series has zero variance")
    return float(m[0] / s[0] - m[1] / s[1])


def sharpe_diff_gradient(mu: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    v = (gamma - mu**2) ** 1.5
    return np.array([gamma[0] / v[0], -gamma[1] / v[1], -0.5 * mu[0] / v[0], 0.5 * mu[1] / v[1]])


def hac_cov(y: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West long-run covariance of the rows of y (demeaned inside)."""
    if lags < 0:
        raise ValueError("lags must be non-negative")
    z = y - y.mean(axis=0)
    t = z.shape[0]
    psi = z.T @ z / t
    for j in range(1, lags + 1):
        g = z[j:].T @ z[:-j] / t
        psi += (1.0 - j / (lags + 1.0)) * (g + g.T)
    return psi


def sharpe_diff_hac(r1: np.ndarray, r2: np.ndarray, lags: int = 0) -> SharpeHacResult:
    x = _pair(r1, r2)
    t = x.shape[0]
    mu, gamma = x.mean(axis=0), (x**2).mean(axis=0)
    grad = sharpe_diff_gradient(mu, gamma)
    psi = hac_cov(np.column_stack([x, x**2]), lags)
    se = float(np.sqrt(max(grad @ psi @ grad, 0.0) / t))
    d = _sr_diff(x)
    z = d / se if se > 0.0 else 0.0
    return SharpeHacResult(d, se, z, float(2.0 * norm.sf(abs(z))))


def sharpe_diff_bootstrap(
    r1: np.ndarray, r2: np.ndarray, block: int = 5, n_boot: int = 2000, seed: int = 0
) -> float:
    """Two-sided circular-block-bootstrap p-value for H0: SR_1 = SR_2."""
    x = _pair(r1, r2)
    t = x.shape[0]
    if not 1 <= block <= t or n_boot < 1:
        raise ValueError("block must lie in [1, T] and n_boot be positive")
    d = _sr_diff(x)
    rng = np.random.default_rng(seed)
    n_blocks = -(-t // block)
    offs = np.arange(block)
    hits = 0
    for _ in range(n_boot):
        starts = rng.integers(0, t, n_blocks)
        idx = ((starts[:, None] + offs[None, :]) % t).ravel()[:t]
        hits += abs(_sr_diff(x[idx]) - d) >= abs(d)
    return (hits + 1.0) / (n_boot + 1.0)
