"""SSVI and eSSVI total-variance surfaces for FX smiles: a shim over ``qp_core.vol.ssvi``.

The surface maths (Gatheral and Jacquier 2014; Hendriks and Martini 2019)
lives once in :mod:`qp_core.vol.ssvi`, shared with ``eq_surface.ssvi`` (plan
W20). The FX additions stay here. The quoted DNS ATM strike sits at
``k = +-sigma^2 T / 2``, not at the forward, so :func:`fit_ssvi_from_pillars`
reads ``theta`` off the pillar total variances interpolated at ``k = 0``;
:func:`ssvi_strike_at_delta` returns the smile-consistent strike for a quoted
delta under any of the four conventions, premium-adjusted included.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qp_core.vol.ssvi import (
    ESSVIParams,
    SSVIFitResult,
    SSVIParams,
    calendar_arbitrage_free,
    fit_ssvi,
    ssvi_slice_raw_svi,
    ssvi_total_variance,
)

from ._smile_delta import strike_at_smile_delta
from .smile import SVIParams

__all__ = [
    "ESSVIParams",
    "SSVIFitResult",
    "SSVIParams",
    "calendar_arbitrage_free",
    "fit_ssvi",
    "fit_ssvi_from_pillars",
    "ssvi_slice_as_svi",
    "ssvi_strike_at_delta",
    "ssvi_total_variance",
]


def ssvi_slice_as_svi(theta: float, rho: float, phi: float) -> SVIParams:
    """The raw SVI parameters of one SSVI slice (``fx_surface.smile.SVIParams``)."""
    a, b, r, m, s = ssvi_slice_raw_svi(theta, rho, phi)
    return SVIParams(a=a, b=b, rho=r, m=m, s=s)


def fit_ssvi_from_pillars(
    strikes: Sequence[Sequence[float]],
    vols: Sequence[Sequence[float]],
    forwards: Sequence[float],
    expiries: Sequence[float],
    extended: bool = False,
) -> SSVIFitResult:
    """Fit SSVI to FX pillar quotes given as solved strikes and vols per expiry.

    ``theta_i`` is the pillar total variance linearly interpolated at
    ``k = 0`` (the forward), because the quoted DNS ATM strike is not the
    forward. Each expiry needs at least three pillars bracketing the forward.
    """
    if not (len(strikes) == len(vols) == len(forwards) == len(expiries)):
        raise ValueError("strikes, vols, forwards and expiries need one entry per expiry")
    ks, ws, thetas = [], [], []
    for K_in, v_in, F, T in zip(strikes, vols, forwards, expiries, strict=True):
        K = np.asarray(K_in, dtype=float)
        v = np.asarray(v_in, dtype=float)
        if K.shape != v.shape or K.size < 3:
            raise ValueError("each expiry needs at least three (strike, vol) pillars")
        if not (F > 0.0 and T > 0.0) or np.any(~(K > 0.0)) or np.any(~(v > 0.0)):
            raise ValueError("forwards, expiries, strikes and vols must be positive")
        k = np.log(K / F)
        order = np.argsort(k)
        k, w = k[order], (v[order] ** 2) * T
        if not k[0] < 0.0 < k[-1]:
            raise ValueError("pillars must bracket the forward (k = 0)")
        ks.append(k)
        ws.append(w)
        thetas.append(float(np.interp(0.0, k, w)))
    return fit_ssvi(ks, ws, np.array(thetas), extended=extended)


def ssvi_strike_at_delta(
    params: SSVIParams | ESSVIParams,
    theta: float,
    delta: float,
    cp: int,
    S: float,
    T: float,
    r_d: float,
    r_f: float,
    convention: str = "spot",
) -> tuple[float, float]:
    """Smile-consistent ``(strike, vol)`` of a quoted delta on one SSVI slice."""
    if not T > 0.0:
        raise ValueError("T must be positive")

    def vol_of_k(k: float) -> float:
        return float(np.sqrt(params.total_variance(k, theta) / T))

    K = strike_at_smile_delta(vol_of_k, delta, cp, S, T, r_d, r_f, convention)
    F = S * np.exp((r_d - r_f) * T)
    return K, vol_of_k(float(np.log(K / F)))
