"""Dupire local volatility.

Two routes, which the tests check against each other:

1. From total implied variance ``w(k, T)``, ``k = ln(K/F_T)``
   (Gatheral 2006, "The Volatility Surface", Wiley, eq. 1.10):

       sigma_loc^2 = (dw/dT) / (1 - (k/w) w_k + 1/4 (-1/4 - 1/w + k^2/w^2) w_k^2 + 1/2 w_kk)

2. From call prices ``C(K, T)`` (Dupire 1994, "Pricing with a smile",
   Risk 7(1), with rates and dividends as in Gatheral 2006, eq. 1.4):

       sigma_loc^2 = (C_T + (r - q) K C_K + q C) / (1/2 K^2 C_KK)

Derivatives are central finite differences. A negative numerator or
denominator means the input surface has calendar or butterfly arbitrage at
that point; we raise rather than return a made-up number.

``local_vol_mc_price`` simulates the local-vol SDE (log-Euler) so a
calibrated surface can be checked for repricing (Gatheral 2006, ch. 1).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

TotalVariance = Callable[[np.ndarray, float], np.ndarray]


class LocalVolArbitrageError(ValueError):
    """The input surface implies negative local variance at some point."""


def local_vol_from_total_variance(
    w: TotalVariance,
    k: np.ndarray | float,
    T: float,
    dk: float = 1e-4,
    dT: float = 1e-5,
) -> np.ndarray | float:
    """Local vol at log-moneyness ``k`` and time ``T`` from ``w(k, T)`` (route 1)."""
    if T <= dT:
        raise ValueError(f"T must exceed dT={dT}, got {T}")
    kk = np.atleast_1d(np.asarray(k, dtype=float))
    w0 = np.asarray(w(kk, T), dtype=float)
    wp = np.asarray(w(kk + dk, T), dtype=float)
    wm = np.asarray(w(kk - dk, T), dtype=float)
    w_k = (wp - wm) / (2.0 * dk)
    w_kk = (wp - 2.0 * w0 + wm) / (dk * dk)
    w_T = (np.asarray(w(kk, T + dT)) - np.asarray(w(kk, T - dT))) / (2.0 * dT)
    den = (1.0 - kk / w0 * w_k + 0.25 * (-0.25 - 1.0 / w0 + kk * kk / (w0 * w0)) * w_k * w_k
           + 0.5 * w_kk)
    if np.any(w_T < 0.0) or np.any(den <= 0.0):
        raise LocalVolArbitrageError("negative local variance: surface has arbitrage")
    out = np.sqrt(w_T / den)
    return out if np.ndim(k) else float(out[0])


def local_vol_from_prices(
    call: Callable[[float, float], float],
    K: float,
    T: float,
    r: float,
    q: float,
    dK: float | None = None,
    dT: float = 1e-4,
) -> float:
    """Local vol at strike ``K`` and time ``T`` from a call-price function (route 2)."""
    h = 1e-3 * K if dK is None else dK
    c0 = call(K, T)
    cp, cm = call(K + h, T), call(K - h, T)
    c_K = (cp - cm) / (2.0 * h)
    c_KK = (cp - 2.0 * c0 + cm) / (h * h)
    c_T = (call(K, T + dT) - call(K, T - dT)) / (2.0 * dT)
    num = c_T + (r - q) * K * c_K + q * c0
    den = 0.5 * K * K * c_KK
    if num < 0.0 or den <= 0.0:
        raise LocalVolArbitrageError("negative local variance: prices have arbitrage")
    return float(np.sqrt(num / den))


def local_vol_mc_price(
    w: TotalVariance,
    S0: float,
    K: np.ndarray,
    T: float,
    r: float,
    q: float,
    n_paths: int = 20_000,
    n_steps: int = 100,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo call prices under the local-vol model built from ``w``.

    Returns ``(price, standard_error)`` per strike. Antithetic normals.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = n_paths // 2
    x = np.zeros(2 * half)  # ln(S_t / F_t)
    for i in range(n_steps):
        t = max((i + 0.5) * dt, 2e-5)  # mid-step; stay clear of t = 0
        sig = np.clip(local_vol_from_total_variance(w, x, t), 1e-4, 5.0)
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
        x = x - 0.5 * sig * sig * dt + sig * np.sqrt(dt) * z
    F = S0 * np.exp((r - q) * T)
    ST = F * np.exp(x)
    K = np.atleast_1d(np.asarray(K, dtype=float))
    pay = np.maximum(ST[None, :] - K[:, None], 0.0)
    # pair antithetic samples so the SE reflects the estimator actually used
    paired = 0.5 * (pay[:, :half] + pay[:, half:])
    df = np.exp(-r * T)
    return df * paired.mean(axis=1), df * paired.std(axis=1, ddof=1) / np.sqrt(half)
