"""SABR implied volatility: Hagan's expansion with the West and Obloj fixes.

Shared home of SABR for every asset class (plan W15): equity options,
rates caps/floors (``fi_rates.caps``) and the swaption cube
(``fi_rates.swaptions``). Moved from ``eq_options.sabr`` unchanged in
arithmetic; :func:`sabr_calibrate` and :func:`black76_price` are new.

Plain idea: SABR lets the forward's volatility itself be random and
correlated with the forward. Hagan and co-authors found an approximate
formula for the Black vol the model implies at each strike. As printed, the
formula has two weak spots. It divides zero by zero at the money, so a
naive implementation loses digits near ``K = F``; and its leading term is
wrong for long expiries and far strikes. This module fixes both.

Formal statement.

* Hagan, Kumar, Lesniewski and Woodward (2002), "Managing smile risk",
  Wilmott Magazine, 84-108, eq. (2.17a): ``variant="hagan"``.
* West (2005), "Calibration of the SABR model in illiquid markets", Appl.
  Math. Finance 12(4), 371-385, section 2: evaluate ``z / x(z)`` by its
  series ``1 - rho z/2 + (2 - 3 rho^2) z^2 / 12`` near ``z = 0`` and take
  the ``beta = 1`` limit with logs, so the formula is smooth through the
  money. Always on.
* Obloj (2008), "Fine-tune your smile: correction to Hagan et al.",
  arXiv:0708.0998, eq. (3.4): leading term
  ``nu ln(F/K) / x(z)`` with ``z = nu (F^{1-beta} - K^{1-beta}) / (alpha (1 - beta))``
  (``z = nu ln(F/K) / alpha`` at ``beta = 1``): ``variant="obloj"``
  (the default).
* Shifted SABR for negative forwards: ``F + shift``, ``K + shift``.
* :func:`sabr_density_check` returns the risk-neutral density implied by
  the formula (Breeden-Litzenberger, second difference of call prices) and
  flags strikes where it is negative: the known arbitrage of the expansion
  at low strikes (Hagan, Kumar, Lesniewski and Woodward 2014, "Arbitrage-
  free SABR", Wilmott 69, 60-75, section 1).

The fix for that arbitrage, the arbitrage-free PDE of Hagan et al. (2014)
on the grid and time stepping of Le Floc'h and Kennedy (2014, "Finite
difference techniques for arbitrage-free SABR", J. Comput. Finance
20(3)), lives in :mod:`qp_core.vol.sabr_pde`; this module's density check
detects the problem, that one removes it.

Worked example: ``F = K = 0.03, T = 5, alpha = 0.02, beta = 0.5,
rho = -0.3, nu = 0.4``: both variants give 0.121209 at the money; at
K = 0.01 Hagan gives 0.28182 and Obloj 0.28346.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from qp_core.normal import norm_cdf

__all__ = ["SabrFit", "black76_price", "sabr_calibrate", "sabr_density_check", "sabr_vol"]


def black76_price(F: float, K: float, T: float, r: float, sigma: float,
                  option_type: str = "call") -> float:
    """Black-76 value ``e^{-rT} w (F N(w d1) - K N(w d2))``, ``w = +1`` call, ``-1`` put.

    Same expression as ``eq_options.black76.black76_price`` (Black 1976),
    kept here so ``qp_core`` does not import an asset-class package.
    ``T == 0`` is intrinsic; ``sigma == 0``, ``K == 0`` or ``F == 0`` the
    discounted intrinsic.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if not (F >= 0.0 and K >= 0.0 and T >= 0.0 and sigma >= 0.0):
        raise ValueError("need F, K, T, sigma >= 0")
    w = 1.0 if option_type == "call" else -1.0
    if T == 0.0:
        return max(w * (F - K), 0.0)
    df = math.exp(-r * T)
    if sigma == 0.0 or K == 0.0 or F == 0.0:
        return df * max(w * (F - K), 0.0)
    s = sigma * math.sqrt(T)
    d1 = math.log(F / K) / s + 0.5 * s
    d2 = d1 - s
    return df * w * (F * norm_cdf(w * d1) - K * norm_cdf(w * d2))


def _z_over_x(z: float, rho: float) -> float:
    if abs(z) < 1e-6:
        return 1.0 - 0.5 * rho * z + (2.0 - 3.0 * rho * rho) * z * z / 12.0
    # x(z) = ln((sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho)), written with
    # log1p so small z keeps full precision (the naive log loses eps/z).
    root = math.sqrt(1.0 - 2.0 * rho * z + z * z)
    x = math.log1p((z + (z * z - 2.0 * rho * z) / (root + 1.0)) / (1.0 - rho))
    return z / x


def _check(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float,
           shift: float) -> None:
    for name, v in (("F", F), ("K", K), ("T", T), ("alpha", alpha), ("beta", beta),
                    ("rho", rho), ("nu", nu), ("shift", shift)):
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v!r}")
    if F + shift <= 0.0 or K + shift <= 0.0:
        raise ValueError("F + shift and K + shift must be > 0")
    if T < 0.0 or alpha <= 0.0 or nu < 0.0 or not 0.0 <= beta <= 1.0 or not -1.0 < rho < 1.0:
        raise ValueError("need T >= 0, alpha > 0, nu >= 0, 0 <= beta <= 1, -1 < rho < 1")


def sabr_vol(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float,
             shift: float = 0.0, variant: str = "obloj") -> float:
    """Black (lognormal) implied vol of the SABR model at strike ``K``."""
    _check(F, K, T, alpha, beta, rho, nu, shift)
    if variant not in ("hagan", "obloj"):
        raise ValueError(f"variant must be 'hagan' or 'obloj', got {variant!r}")
    f, k = F + shift, K + shift
    omb = 1.0 - beta
    log_fk = math.log(f / k)
    fk_mid = (f * k) ** (0.5 * omb)
    correction = 1.0 + T * (
        omb * omb * alpha * alpha / (24.0 * fk_mid * fk_mid)
        + rho * beta * nu * alpha / (4.0 * fk_mid)
        + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0)

    if variant == "hagan":
        z = nu / alpha * fk_mid * log_fk
        denom = 1.0 + omb**2 / 24.0 * log_fk**2 + omb**4 / 1920.0 * log_fk**4
        return float(alpha / (fk_mid * denom) * _z_over_x(z, rho) * correction)

    # Obloj: leading term alpha (1-beta) ln(F/K) / (F^{1-beta} - K^{1-beta}) * z/x(z).
    if omb == 0.0:
        z = nu * log_fk / alpha
        lead = alpha
    else:
        diff = k**omb * math.expm1(omb * log_fk)  # f^{1-beta} - k^{1-beta}, no cancellation
        z = nu * diff / (alpha * omb)
        lead = alpha / k**omb if log_fk == 0.0 else alpha * omb * log_fk / diff
    return float(lead * _z_over_x(z, rho) * correction)


def sabr_density_check(F: float, T: float, alpha: float, beta: float, rho: float, nu: float,
                       strikes: ArrayLike, shift: float = 0.0,
                       variant: str = "obloj") -> tuple[np.ndarray, np.ndarray]:
    """Return ``(strikes, density)``; the density is negative where the formula admits arbitrage.

    ``strikes`` must be an evenly spaced increasing grid; the density is
    ``(C(K-h) - 2 C(K) + C(K+h)) / h^2`` on undiscounted Black-76 calls at
    the interior strikes.
    """
    ks = np.asarray(strikes, dtype=float)
    if ks.ndim != 1 or ks.size < 3:
        raise ValueError("need at least three strikes")
    h = ks[1] - ks[0]
    if not h > 0.0 or not np.allclose(np.diff(ks), h, rtol=1e-9, atol=0.0):
        raise ValueError("strikes must be evenly spaced and increasing")
    calls = np.array([
        black76_price(F + shift, k + shift, T, 0.0,
                      sabr_vol(F, k, T, alpha, beta, rho, nu, shift, variant), "call")
        for k in ks])
    return ks[1:-1], (calls[:-2] - 2.0 * calls[1:-1] + calls[2:]) / (h * h)


@dataclass(frozen=True)
class SabrFit:
    """Result of :func:`sabr_calibrate`. ``rmse`` is in vol units."""

    alpha: float
    beta: float
    rho: float
    nu: float
    shift: float
    rmse: float


def sabr_calibrate(F: float, T: float, strikes: ArrayLike, vols: ArrayLike, beta: float,
                   shift: float = 0.0, variant: str = "obloj",
                   weights: ArrayLike | None = None) -> SabrFit:
    """Fit ``(alpha, rho, nu)`` with ``beta`` fixed to a smile of Black vols.

    Plain idea: market practice fixes ``beta`` (from a regression of ATM
    vol on the forward, or by convention: 0.5 for rates) and fits the
    other three to one expiry's smile. Formal statement: weighted least
    squares in vol, ``min sum w_i (sigma_SABR(K_i) - sigma_i)^2`` over
    ``alpha > 0, -1 < rho < 1, nu >= 0`` (Hagan et al. 2002, section 3.3;
    West 2005, section 3). ``alpha`` starts from the ATM-vol relation
    ``sigma_ATM (F + shift)^{1-beta}``. Worked example: the test fits a
    smile generated from known parameters and recovers them to 1e-6.
    """
    ks = np.asarray(strikes, dtype=float)
    vs = np.asarray(vols, dtype=float)
    if ks.ndim != 1 or ks.shape != vs.shape or ks.size < 3:
        raise ValueError("need matching 1-D strikes and vols with at least three points")
    if not np.all(np.isfinite(vs)) or np.any(vs <= 0.0):
        raise ValueError("vols must be finite and > 0")
    w = np.ones_like(ks) if weights is None else np.sqrt(np.asarray(weights, dtype=float))
    atm = float(np.interp(F, ks, vs)) if ks[0] <= F <= ks[-1] else float(vs[np.argmin(abs(ks - F))])
    a0 = atm * (F + shift) ** (1.0 - beta)

    def resid(p: np.ndarray) -> np.ndarray:
        a, r, n = (float(x) for x in p)
        out: np.ndarray = w * (np.array([sabr_vol(F, k, T, a, beta, r, n, shift, variant) for k in ks]) - vs)
        return out

    sol = least_squares(resid, x0=[a0, 0.0, 0.3], bounds=([1e-8, -0.9999, 0.0], [np.inf, 0.9999, 10.0]),
                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
    a, r, n = (float(x) for x in sol.x)
    rmse = float(np.sqrt(np.mean((resid(sol.x) / w) ** 2)))
    return SabrFit(a, beta, r, n, shift, rmse)
