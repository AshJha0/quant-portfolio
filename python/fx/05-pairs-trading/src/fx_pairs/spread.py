"""Spread construction and dynamics: log-spread, OU fitting (OLS & MLE), RLS hedge.

Conventions
-----------
* Spreads are in **log units**: ``s_t = log p1_t - alpha - beta * log p2_t``.
  A move of 0.01 in the spread is ~1% relative-value move of the pair-of-pairs
  position (long 1 unit of pair 1, short beta units of pair 2).
* The Ornstein-Uhlenbeck process ``ds = kappa (theta - s) dt + sigma dW`` is
  fitted on the exact discretisation

  ``s_{t+1} = c + phi s_t + eps_t``,  ``phi = exp(-kappa dt)``,
  ``c = theta (1 - phi)``,  ``Var(eps) = sigma^2 (1 - phi^2) / (2 kappa)``.

  ``kappa`` is **annualised** (per year); ``dt`` defaults to 1/252 (business
  daily).  Half-life is reported in business days:
  ``half_life_days = ln 2 / (kappa * dt)``.
* MLE is the exact Gaussian transition-density likelihood; for an AR(1) with
  intercept the conditional MLE of (c, phi) coincides with OLS, so the two
  estimators agree closely — the test suite asserts this as a cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from ._validation import finite_series, require_finite

__all__ = [
    "log_spread",
    "OUFit",
    "fit_ou_ols",
    "fit_ou_mle",
    "half_life_days",
    "RLSHedge",
]


def log_spread(
    p1: pd.Series | np.ndarray,
    p2: pd.Series | np.ndarray,
    beta: float,
    alpha: float = 0.0,
) -> pd.Series | np.ndarray:
    """Log spread ``log p1 - alpha - beta log p2`` (log units).

    Parameters
    ----------
    p1, p2 : array-like
        Spot price series of the two currency pairs (levels, not logs).
    beta : float
        Hedge ratio from the cointegrating regression.
    alpha : float
        Intercept of the cointegrating regression.
    """
    require_finite(alpha=alpha, beta=beta)
    finite_series(np.asarray(p1, dtype=float).ravel(), "p1", positive=True)
    finite_series(np.asarray(p2, dtype=float).ravel(), "p2", positive=True)
    return np.log(p1) - alpha - beta * np.log(p2)


def half_life_days(kappa: float, dt: float = 1.0 / 252.0) -> float:
    """Half-life of an OU process in steps of size ``dt`` (business days).

    ``inf`` when ``kappa <= 0`` (no mean reversion).
    """
    require_finite(dt=dt)
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if np.isnan(kappa):
        raise ValueError("kappa must not be NaN")
    if kappa <= 0:
        return float("inf")
    return float(np.log(2.0) / (kappa * dt))


@dataclass
class OUFit:
    """Fitted OU parameters.

    Attributes
    ----------
    kappa : float
        Mean-reversion speed, annualised (per year).
    theta : float
        Long-run mean (log units).
    sigma : float
        Diffusion volatility, annualised (log units per sqrt(year)).
    half_life : float
        Half-life in business days (``ln 2 / (kappa dt)``); ``inf`` if
        ``kappa <= 0``.
    phi, c : float
        Discrete AR(1) parameters ``s_{t+1} = c + phi s_t + eps``.
    eps_std : float
        Standard deviation of the one-step innovation.
    dt : float
        Step size in years.
    method : str
        ``"ols"`` or ``"mle"``.
    """

    kappa: float
    theta: float
    sigma: float
    half_life: float
    phi: float
    c: float
    eps_std: float
    dt: float
    method: str


def _ou_from_ar1(c: float, phi: float, eps_std: float, dt: float, method: str) -> OUFit:
    if phi <= 0.0 or phi >= 1.0:
        # No (or explosive) mean reversion at this sample: report kappa<=0
        kappa = -np.log(phi) / dt if phi > 0 else float("inf")
        theta = c / (1.0 - phi) if phi != 1.0 else float("nan")
        sigma = eps_std / np.sqrt(dt)  # RW-limit scaling
        return OUFit(kappa=float(kappa), theta=float(theta), sigma=float(sigma),
                     half_life=half_life_days(kappa, dt) if np.isfinite(kappa) else 0.0,
                     phi=float(phi), c=float(c), eps_std=float(eps_std),
                     dt=dt, method=method)
    kappa = -np.log(phi) / dt
    theta = c / (1.0 - phi)
    sigma = eps_std * np.sqrt(2.0 * kappa / (1.0 - phi**2))
    return OUFit(
        kappa=float(kappa), theta=float(theta), sigma=float(sigma),
        half_life=half_life_days(kappa, dt), phi=float(phi), c=float(c),
        eps_std=float(eps_std), dt=dt, method=method,
    )


def _validate_spread(spread: pd.Series | np.ndarray) -> np.ndarray:
    s = np.asarray(spread, dtype=float)
    if s.ndim != 1:
        raise ValueError("spread must be one-dimensional")
    if len(s) < 30:
        raise ValueError(f"spread too short to fit OU: n={len(s)}")
    # isfinite, not isnan: +/-Inf (a logged zero/missing price) would pass an
    # isnan check, poison the AR(1) least squares and yield NaN OU parameters.
    if not np.isfinite(s).all():
        raise ValueError("spread contains NaN or infinite values; clean the series first")
    if np.std(s) < 1e-12:
        raise ValueError("spread has zero variance (degenerate); nothing to fit")
    return s


def fit_ou_ols(spread: pd.Series | np.ndarray, dt: float = 1.0 / 252.0) -> OUFit:
    """Fit OU by OLS on the exact AR(1) discretisation.

    Regress ``s_{t+1}`` on ``[1, s_t]``; map ``(c, phi, eps_std)`` to
    ``(kappa, theta, sigma)`` (see module docstring for the mapping and units).
    """
    s = _validate_spread(spread)
    y, x = s[1:], s[:-1]
    X = np.column_stack([np.ones(len(x)), x])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    eps_std = float(np.sqrt(resid @ resid / (len(y) - 2)))
    return _ou_from_ar1(float(coef[0]), float(coef[1]), eps_std, dt, "ols")


def fit_ou_mle(spread: pd.Series | np.ndarray, dt: float = 1.0 / 252.0) -> OUFit:
    """Fit OU by exact Gaussian transition-density maximum likelihood.

    Maximises ``sum_t log N(s_{t+1}; theta + (s_t - theta) phi, v)`` with
    ``phi = exp(-kappa dt)``, ``v = sigma^2 (1 - phi^2)/(2 kappa)`` over
    ``(kappa, theta, sigma)``, initialised at the OLS fit.  For AR(1) the
    conditional MLE of the mean parameters coincides with OLS, so this is a
    consistency cross-check more than a different estimator.
    """
    s = _validate_spread(spread)
    ols = fit_ou_ols(s, dt)
    k0 = ols.kappa if 0 < ols.kappa < 1e4 and np.isfinite(ols.kappa) else 1.0
    x0 = np.array([np.log(k0), ols.theta if np.isfinite(ols.theta) else float(np.mean(s)),
                   np.log(max(ols.sigma, 1e-12))])

    y, x = s[1:], s[:-1]

    def negll(params: np.ndarray) -> float:
        kappa = np.exp(params[0])
        theta = params[1]
        sigma = np.exp(params[2])
        phi = np.exp(-kappa * dt)
        v = sigma**2 * (1.0 - phi**2) / (2.0 * kappa)
        if v <= 0 or not np.isfinite(v):
            return 1e12
        mu = theta + (x - theta) * phi
        return float(0.5 * np.sum(np.log(2.0 * np.pi * v) + (y - mu) ** 2 / v))

    res = optimize.minimize(negll, x0, method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000})
    kappa = float(np.exp(res.x[0]))
    theta = float(res.x[1])
    sigma = float(np.exp(res.x[2]))
    phi = float(np.exp(-kappa * dt))
    eps_std = float(sigma * np.sqrt((1.0 - phi**2) / (2.0 * kappa)))
    c = theta * (1.0 - phi)
    return OUFit(kappa=kappa, theta=theta, sigma=sigma,
                 half_life=half_life_days(kappa, dt), phi=phi, c=c,
                 eps_std=eps_std, dt=dt, method="mle")


class RLSHedge:
    """Recursive least squares hedge-ratio tracker with forgetting factor.

    Model ``log p1_t = alpha_t + beta_t log p2_t + e_t`` updated recursively:

    with regressor ``x_t = [1, log p2_t]``, gain
    ``k_t = P x_t / (lam + x_t' P x_t)``, update
    ``theta_t = theta_{t-1} + k_t (y_t - x_t' theta_{t-1})`` and
    ``P_t = (P_{t-1} - k_t x_t' P_{t-1}) / lam``.

    ``lam = 1`` is recursive OLS (converges to the batch OLS estimate);
    ``lam < 1`` discounts old data with an effective memory of about
    ``1 / (1 - lam)`` observations, letting the hedge ratio adapt when the
    cointegrating relation drifts (e.g. slow policy divergence).

    Parameters
    ----------
    lam : float
        Forgetting factor in (0, 1].
    delta : float
        Initial covariance scale ``P_0 = delta * I`` (large = diffuse prior).
    """

    def __init__(self, lam: float = 0.995, delta: float = 1e4) -> None:
        require_finite(lam=lam, delta=delta)
        if not 0.0 < lam <= 1.0:
            raise ValueError(f"lam must be in (0, 1], got {lam}")
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.lam = float(lam)
        self.theta = np.zeros(2)
        self.P = np.eye(2) * float(delta)
        self.n_obs = 0

    @property
    def alpha(self) -> float:
        """Current intercept estimate."""
        return float(self.theta[0])

    @property
    def beta(self) -> float:
        """Current hedge-ratio estimate."""
        return float(self.theta[1])

    def update(self, log_p2: float, log_p1: float) -> tuple[float, float]:
        """One RLS step; returns the updated ``(alpha, beta)``."""
        # A single non-finite observation permanently poisons theta and P:
        # the filter has no mechanism to recover, so every later hedge ratio
        # would be NaN with no error raised.
        require_finite(log_p2=log_p2, log_p1=log_p1)
        x = np.array([1.0, float(log_p2)])
        Px = self.P @ x
        k = Px / (self.lam + x @ Px)
        err = float(log_p1) - x @ self.theta
        self.theta = self.theta + k * err
        self.P = (self.P - np.outer(k, Px)) / self.lam
        # enforce symmetry for numerical hygiene
        self.P = 0.5 * (self.P + self.P.T)
        self.n_obs += 1
        return self.alpha, self.beta

    def fit_path(
        self, p2: pd.Series | np.ndarray, p1: pd.Series | np.ndarray
    ) -> pd.DataFrame:
        """Run the filter over price series; returns per-date alpha/beta path.

        Parameters
        ----------
        p2, p1 : array-like
            Price levels (logs are taken internally); ``p1`` is the dependent
            pair, ``p2`` the hedge pair, mirroring ``engle_granger(y, x)``.
        """
        lp1 = np.log(finite_series(np.asarray(p1, dtype=float).ravel(),
                                   "p1", positive=True))
        lp2 = np.log(finite_series(np.asarray(p2, dtype=float).ravel(),
                                   "p2", positive=True))
        if lp1.shape != lp2.shape:
            raise ValueError("p1 and p2 must have equal length")
        alphas = np.empty(len(lp1))
        betas = np.empty(len(lp1))
        for t in range(len(lp1)):
            a, b = self.update(lp2[t], lp1[t])
            alphas[t] = a
            betas[t] = b
        index = p1.index if isinstance(p1, pd.Series) else pd.RangeIndex(len(lp1))
        return pd.DataFrame({"alpha": alphas, "beta": betas}, index=index)
