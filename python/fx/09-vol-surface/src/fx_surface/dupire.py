"""Dupire local volatility for an FX pair: a shim over ``qp_core.vol.dupire``.

The maths lives once in :mod:`qp_core.vol.dupire` (plan W20): Garman-Kohlhagen
is Black-Scholes with ``q = r_f``, so this module only renames the rates to
``(r_d, r_f)``, keeps the FX spot/rate checks, and adds
:func:`local_vol_at_delta`, which finds the smile-consistent strike for a
quoted delta under any of the four conventions (Clark 2011, "Foreign
Exchange Option Pricing", Wiley, s.4.2) before reading the local vol there.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from qp_core.vol import dupire as _core
from qp_core.vol.dupire import (
    LocalVolArbitrageError,
    TotalVariance,
    local_vol_from_total_variance,
)

from ._smile_delta import strike_at_smile_delta
from .garman_kohlhagen import gk_forward

__all__ = [
    "LocalVolArbitrageError",
    "local_vol_at_delta",
    "local_vol_from_prices",
    "local_vol_from_total_variance",
    "local_vol_mc_price",
]


def local_vol_from_prices(
    call: Callable[[float, float], float], K: float, T: float, r_d: float, r_f: float,
    dK: float | None = None, dT: float = 1e-4,
) -> float:
    """Local vol at strike ``K`` from a domestic-currency call-price function ``call(K, T)``."""
    return _core.local_vol_from_prices(call, K, T, r_d, r_f, dK, dT)


def local_vol_at_delta(
    w: TotalVariance, delta: float, T: float, S: float, r_d: float, r_f: float,
    cp: int = 1, convention: str = "spot",
) -> tuple[float, float]:
    """Local vol at a quoted delta: returns ``(strike, local_vol)``."""
    def vol_of_k(k: float) -> float:
        return float(np.sqrt(np.asarray(w(np.array([k]), T))[0] / T))

    K = strike_at_smile_delta(vol_of_k, delta, cp, S, T, r_d, r_f, convention)
    k = math.log(K / gk_forward(S, T, r_d, r_f))
    return K, float(local_vol_from_total_variance(w, k, T))


def local_vol_mc_price(
    w: TotalVariance, S0: float, K: np.ndarray, T: float, r_d: float, r_f: float,
    n_paths: int = 20_000, n_steps: int = 100, seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo GK call prices (domestic per unit foreign) under local vol from ``w``."""
    gk_forward(S0, T, r_d, r_f)  # FX spot and rate checks
    return _core.local_vol_mc_price(w, S0, K, T, r_d, r_f, n_paths, n_steps, seed)
