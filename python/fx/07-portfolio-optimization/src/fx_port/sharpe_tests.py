"""Robust tests for the difference of two Sharpe ratios.

The Jobson-Korkie / Memmel test (``eq_port.metrics.sharpe_difference_test``)
assumes iid normal returns. Ledoit and Wolf (2008),
"Robust performance hypothesis testing with the Sharpe ratio", Journal of
Empirical Finance 15(5), 850-859, give two replacements, both here.

HAC delta method (section 3.1)
------------------------------
With v = (mu_1, mu_2, gamma_1, gamma_2), gamma_i = E[r_i^2], the
difference is f(v) = mu_1 / sqrt(gamma_1 - mu_1^2) - mu_2 / sqrt(gamma_2 - mu_2^2)
and sqrt(T) (f(v_hat) - f(v)) -> N(0, grad' Psi grad), where Psi is the
long-run covariance of y_t = (r_1t, r_2t, r_1t^2, r_2t^2). Gradient:

    df/dmu_i    = +/- gamma_i / (gamma_i - mu_i^2)^{3/2}
    df/dgamma_i = -/+ mu_i / (2 (gamma_i - mu_i^2)^{3/2})

Two estimators of Psi:

* ``kernel="qs_pw"`` (what Ledoit-Wolf use): the prewhitened
  quadratic-spectral estimator of Andrews and Monahan (1992), "An improved
  heteroskedasticity and autocorrelation consistent covariance matrix
  estimator", Econometrica 60(4), 953-966. Fit a VAR(1)
  ``z_t = A z_{t-1} + e_t`` to the demeaned y, estimate the long-run
  covariance of the residuals with the QS kernel
  ``k(x) = 25 / (12 pi^2 x^2) [sin(6 pi x / 5) / (6 pi x / 5) - cos(6 pi x / 5)]``
  at Andrews' (1991, Econometrica 59(3), 817-858, section 6) automatic
  bandwidth ``S_T = 1.3221 (alpha(2) T)^{1/5}``, with
  ``alpha(2) = sum 4 rho_a^2 s_a^4 / (1 - rho_a)^8 / sum s_a^4 / (1 - rho_a)^4``
  from univariate AR(1) fits to each residual series (equal weights), then
  recolour: ``Psi = (I - A)^{-1} Psi_e (I - A)^{-T}``. Andrews-Monahan
  shrink A when a singular value exceeds 0.97 so that ``I - A`` stays
  well conditioned; we do the same.
* ``kernel="bartlett"`` (the default of :func:`sharpe_diff_hac`, kept for
  compatibility): Newey-West (1987) at a fixed ``lags``.

Under iid normal returns and lags = 0 the variance converges to Memmel's
(2003) [2 - 2 rho + (SR_1^2 + SR_2^2 - 2 SR_1 SR_2 rho^2) / 2] / T.

Studentised circular block bootstrap (section 3.2)
--------------------------------------------------
Resample blocks of pairs (Politis and Romano 1992). Each draw is
studentised: ``|Delta* - Delta_hat| / se*`` is compared with
``|Delta_hat| / se``, where ``se`` is the HAC standard error of the sample
and ``se*`` uses the block-sum covariance of the draw,
``Psi* = (1/l) sum_j zeta_j zeta_j'`` with ``zeta_j = b^{-1/2} sum (y*_t - ybar*)``
over block j (Ledoit-Wolf 2008, section 3.2, after Goetze and Kuensch
1996). The p-value is ``(#{studentised draws >= observed} + 1) / (B + 1)``.
Studentising makes the bootstrap second-order accurate, which matters in
the short samples where the test gets used. ``studentize=False`` gives the
basic centred bootstrap, ``P*(|Delta* - Delta_hat| >= |Delta_hat|)``.
"""

from __future__ import annotations

import math
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


def qs_kernel(x: np.ndarray) -> np.ndarray:
    """Quadratic-spectral kernel (Andrews 1991, section 2); ``k(0) = 1``."""
    x = np.asarray(x, dtype=float)
    a = 6.0 * math.pi * x / 5.0
    a2 = a * a
    # Near 0 the closed form cancels catastrophically; use its Taylor series
    # 1 - a^2/10 + a^4/280 - a^6/15120 (next term ~a^8/1e6, below 1e-14 here).
    out = 1.0 - a2 / 10.0 + a2 * a2 / 280.0 - a2**3 / 15120.0
    big = np.abs(a) >= 0.05
    ab = a[big]
    out[big] = 3.0 / ab**2 * (np.sin(ab) / ab - np.cos(ab))
    return out


def qs_bandwidth(e: np.ndarray) -> float:
    """Andrews (1991) AR(1) plug-in bandwidth for the QS kernel, equal weights."""
    t = e.shape[0]
    num = den = 0.0
    for col in np.asarray(e, dtype=float).T:
        x0, x1 = col[:-1], col[1:]
        ss = float(x0 @ x0)
        if ss == 0.0:
            continue
        rho = float(np.clip(x0 @ x1 / ss, -0.97, 0.97))
        s2 = float(np.mean((x1 - rho * x0) ** 2))
        num += 4.0 * rho**2 * s2**2 / (1.0 - rho) ** 8
        den += s2**2 / (1.0 - rho) ** 4
    alpha2 = num / den if den > 0.0 else 0.0
    return 1.3221 * (alpha2 * t) ** 0.2


def hac_cov_qs_prewhitened(y: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    """Andrews-Monahan prewhitened QS long-run covariance of the rows of y.

    ``bandwidth=None`` picks Andrews' automatic bandwidth on the VAR(1)
    residuals; pass a number to fix it (0 means no kernel lags).
    """
    z = np.asarray(y, dtype=float)
    z = z - z.mean(axis=0)
    t, k = z.shape
    if t < 3 * k + 2:
        raise ValueError("too few observations to prewhiten")
    x0, x1 = z[:-1], z[1:]
    a = np.linalg.lstsq(x0, x1, rcond=None)[0].T  # z_t = A z_{t-1} + e_t
    u, s, vt = np.linalg.svd(a)
    if s.max() > 0.97:  # Andrews-Monahan (1992): keep I - A invertible
        a = u @ np.diag(np.minimum(s, 0.97)) @ vt
    e = x1 - x0 @ a.T
    n = e.shape[0]
    st = qs_bandwidth(e) if bandwidth is None else float(bandwidth)
    if not st >= 0.0:
        raise ValueError("bandwidth must be non-negative")
    psi_e = e.T @ e / n
    if st > 0.0:
        # QS has unbounded support; weights decay like x^-2, so stop once the
        # remaining lags cannot move the sum by more than rounding.
        j_max = min(n - 1, int(math.ceil(200.0 * st)))
        w = qs_kernel(np.arange(1, j_max + 1) / st)
        for j in range(1, j_max + 1):
            g = e[j:].T @ e[:-j] / n
            psi_e += w[j - 1] * (g + g.T)
    inv = np.linalg.inv(np.eye(k) - a)
    return inv @ psi_e @ inv.T


def _psi(y: np.ndarray, kernel: str, lags: int) -> np.ndarray:
    if kernel == "bartlett":
        return hac_cov(y, lags)
    if kernel == "qs_pw":
        return hac_cov_qs_prewhitened(y)
    raise ValueError(f"kernel must be 'bartlett' or 'qs_pw', got {kernel!r}")


def _se(x: np.ndarray, psi: np.ndarray) -> float:
    mu, gamma = x.mean(axis=0), (x**2).mean(axis=0)
    grad = sharpe_diff_gradient(mu, gamma)
    return float(np.sqrt(max(grad @ psi @ grad, 0.0) / x.shape[0]))


def sharpe_diff_hac(
    r1: np.ndarray, r2: np.ndarray, lags: int = 0, kernel: str = "bartlett"
) -> SharpeHacResult:
    """HAC delta-method test of H0: SR_1 = SR_2.

    ``kernel="qs_pw"`` is the Ledoit-Wolf choice (``lags`` is then ignored);
    ``kernel="bartlett"`` uses Newey-West at ``lags``.
    """
    x = _pair(r1, r2)
    se = _se(x, _psi(np.column_stack([x, x**2]), kernel, lags))
    d = _sr_diff(x)
    z = d / se if se > 0.0 else 0.0
    return SharpeHacResult(d, se, z, float(2.0 * norm.sf(abs(z))))


def sharpe_diff_bootstrap(
    r1: np.ndarray,
    r2: np.ndarray,
    block: int = 5,
    n_boot: int = 2000,
    seed: int = 0,
    studentize: bool = True,
    kernel: str = "qs_pw",
    lags: int = 0,
) -> float:
    """Two-sided circular-block-bootstrap p-value for H0: SR_1 = SR_2.

    ``studentize=True`` (the default) is the Ledoit-Wolf studentised
    bootstrap; ``kernel`` and ``lags`` choose the HAC estimator for the
    sample standard error. Each draw's standard error uses the block-sum
    covariance over the ``T // block`` full blocks.
    """
    x = _pair(r1, r2)
    t = x.shape[0]
    if not 1 <= block <= t or n_boot < 1:
        raise ValueError("block must lie in [1, T] and n_boot be positive")
    d = _sr_diff(x)
    rng = np.random.default_rng(seed)
    n_blocks = -(-t // block)
    offs = np.arange(block)
    observed = 0.0
    l_full = t // block
    if studentize:
        se = _se(x, _psi(np.column_stack([x, x**2]), kernel, lags))
        if not se > 0.0:
            raise ValueError("zero standard error: cannot studentise")
        observed = abs(d) / se
    hits = 0
    for _ in range(n_boot):
        starts = rng.integers(0, t, n_blocks)
        idx = ((starts[:, None] + offs[None, :]) % t).ravel()[:t]
        xs = x[idx]
        ds = _sr_diff(xs)
        if not studentize:
            hits += abs(ds - d) >= abs(d)
            continue
        ys = np.column_stack([xs, xs**2])
        ys = ys - ys.mean(axis=0)
        zeta = ys[: l_full * block].reshape(l_full, block, -1).sum(axis=1) / math.sqrt(block)
        se_s = _se(xs, zeta.T @ zeta / l_full)
        hits += bool(se_s > 0.0 and abs(ds - d) / se_s >= observed)
    return (hits + 1.0) / (n_boot + 1.0)
