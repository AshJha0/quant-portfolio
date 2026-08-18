"""Heston finite-difference Greeks with FX-desk risk buckets.

FX desks quote smile risk in the *vega / vanna / volga* buckets because
those are the sensitivities hedged with the traded instruments: vega
with ATM straddles, vanna with risk reversals, volga with butterflies.
This module computes Heston Greeks by finite differences on the COS
price, including BOTH interest-rate rhos (domestic and foreign - an FX
option is a position in two yield curves), and first-class vanna and
volga defined against a *parallel shift of the model vol level*
(sqrt(v0) and sqrt(theta) bumped together by the same additive vol
amount), which is the Heston analogue of a BS sigma bump.

Sticky-delta note: FX smiles are quoted and re-marked in delta space,
so when spot moves the surface floats with it (sticky delta).  A pure
FD spot bump at *fixed Heston parameters* is a sticky-parameter delta;
Heston's dynamics generate their own smile move, which is closer to
sticky-delta behaviour than a frozen local-vol surface.  The comparison
of Heston FD vanna/volga with BS-world analytic values (tests, pipeline
table) shows both the agreement in sign/magnitude near ATM and the
model-dependent divergence in the wings.
"""

from __future__ import annotations

import math

import numpy as np

from .garman_kohlhagen import (
    gk_delta,
    gk_gamma,
    gk_price,
    gk_rho_domestic,
    gk_rho_foreign,
    gk_theta,
    gk_vanna,
    gk_vega,
    gk_volga,
)
from .heston import HestonParams, price_cos

__all__ = ["gk_greeks", "heston_greeks_fd"]


def gk_greeks(
    S: float, K: float, T: float, r_d: float, r_f: float, sigma: float, cp: int = 1
) -> dict[str, float]:
    """All analytic Garman-Kohlhagen Greeks in one dict (BS-world
    reference values for the Heston FD comparison)."""
    return {
        "price": gk_price(S, K, T, r_d, r_f, sigma, cp),
        "delta": gk_delta(S, K, T, r_d, r_f, sigma, cp, "spot"),
        "gamma": gk_gamma(S, K, T, r_d, r_f, sigma),
        "vega": gk_vega(S, K, T, r_d, r_f, sigma),
        "vanna": gk_vanna(S, K, T, r_d, r_f, sigma),
        "volga": gk_volga(S, K, T, r_d, r_f, sigma),
        "rho_d": gk_rho_domestic(S, K, T, r_d, r_f, sigma, cp),
        "rho_f": gk_rho_foreign(S, K, T, r_d, r_f, sigma, cp),
        "theta": gk_theta(S, K, T, r_d, r_f, sigma, cp),
    }


def _bump_vol_level(params: HestonParams, h: float) -> HestonParams:
    """Parallel additive shift of the model vol level: sqrt(v0) and
    sqrt(theta) each move by h (the Heston analogue of a sigma bump)."""
    return HestonParams(
        v0=(math.sqrt(params.v0) + h) ** 2,
        kappa=params.kappa,
        theta=(math.sqrt(params.theta) + h) ** 2,
        xi=params.xi,
        rho=params.rho,
    )


def heston_greeks_fd(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    params: HestonParams,
    cp: int = 1,
    dS_rel: float = 1e-3,
    dvol: float = 1e-3,
    dr: float = 1e-5,
    dT: float = 1e-4,
    N: int = 1024,
) -> dict[str, float]:
    """Central finite-difference Heston Greeks on the COS price.

    Returns
    -------
    dict
        ``price, delta, gamma, vega, vanna, volga, rho_d, rho_f, theta``.
        delta/gamma: spot bumps at fixed parameters (sticky-parameter);
        vega/vanna/volga: parallel model-vol-level bumps (see
        :func:`_bump_vol_level`); rho_d / rho_f: independent bumps of
        the domestic and foreign zero rates (spot held fixed, so the
        forward moves - the market convention for FX rho risk);
        theta: calendar theta ``dV/dt = -dV/dT``.

    Notes
    -----
    A large fixed COS grid (N=1024, L=14) keeps the FD differences well
    above the pricing noise floor; the step sizes are validated for
    stability in tests (halving steps changes Greeks by < 0.5%).
    """

    def price(S_=None, prm=None, rd_=None, rf_=None, T_=None) -> float:
        return float(
            price_cos(
                S if S_ is None else S_,
                K,
                T if T_ is None else T_,
                r_d if rd_ is None else rd_,
                r_f if rf_ is None else rf_,
                params if prm is None else prm,
                cp,
                N=N,
            )
        )

    p0 = price()
    dS = dS_rel * S

    p_up, p_dn = price(S_=S + dS), price(S_=S - dS)
    delta = (p_up - p_dn) / (2.0 * dS)
    gamma = (p_up - 2.0 * p0 + p_dn) / (dS * dS)

    prm_u, prm_d = _bump_vol_level(params, dvol), _bump_vol_level(params, -dvol)
    pv_u, pv_d = price(prm=prm_u), price(prm=prm_d)
    vega = (pv_u - pv_d) / (2.0 * dvol)
    volga = (pv_u - 2.0 * p0 + pv_d) / (dvol * dvol)

    vanna = (
        price(S_=S + dS, prm=prm_u)
        - price(S_=S + dS, prm=prm_d)
        - price(S_=S - dS, prm=prm_u)
        + price(S_=S - dS, prm=prm_d)
    ) / (4.0 * dS * dvol)

    rho_d = (price(rd_=r_d + dr) - price(rd_=r_d - dr)) / (2.0 * dr)
    rho_f = (price(rf_=r_f + dr) - price(rf_=r_f - dr)) / (2.0 * dr)
    theta = -(price(T_=T + dT) - price(T_=max(T - dT, 1e-8))) / (T + dT - max(T - dT, 1e-8))

    return {
        "price": p0,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "vanna": vanna,
        "volga": volga,
        "rho_d": rho_d,
        "rho_f": rho_f,
        "theta": theta,
    }
