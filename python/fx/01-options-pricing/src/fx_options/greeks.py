"""Analytic Garman-Kohlhagen Greeks, including both rhos and vanna/volga.

FX specifics:

* **Two rhos.**  An FX option has rate sensitivity to *both* legs:
  ``rho_d = dV/dr_d`` (positive for calls — higher domestic rate lifts
  the forward) and ``rho_f = dV/dr_f`` (negative for calls — higher
  foreign rate is a larger 'dividend' on the base currency).
* **Vanna and volga.**  FX desks mark smiles with risk reversals and
  butterflies, whose P&L maps directly onto vanna (dDelta/dVol) and
  volga (dVega/dVol).  A vanilla book's smile risk is quoted in these
  buckets, so they are first-class here.

All Greeks are per unit foreign notional, prices in domestic currency.
Theta is per year (divide by 365 for a daily theta); vega is per unit of
vol (divide by 100 for 'per vol point').
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from scipy.stats import norm

from ._common import validate_inputs, validate_option_type
from .garman_kohlhagen import d1 as _d1_fn, gk_price

__all__ = ["GreeksResult", "analytic_greeks", "finite_difference_greeks",
           "vega", "gamma", "vanna", "volga"]


@dataclass(frozen=True)
class GreeksResult:
    """Full GK Greek set for one option.

    ``delta_spot``/``delta_forward`` are unadjusted deltas (see
    :mod:`fx_options.deltas` for premium-adjusted variants).
    """

    price: float
    delta_spot: float
    delta_forward: float
    gamma: float
    vega: float
    theta: float
    rho_domestic: float
    rho_foreign: float
    vanna: float
    volga: float

    def as_dict(self) -> dict[str, float]:
        """Return the Greeks as a plain dict."""
        return asdict(self)


def _core(S: float, K: float, T: float, r_d: float, r_f: float,
          sigma: float) -> tuple[float, float, float, float]:
    _d1 = _d1_fn(S, K, T, r_d, r_f, sigma)
    _d2 = _d1 - sigma * math.sqrt(T)
    return _d1, _d2, math.exp(-r_f * T), math.exp(-r_d * T)


def gamma(S: float, K: float, T: float, r_d: float, r_f: float,
          sigma: float) -> float:
    """Spot gamma ``d2V/dS2 = e^{-r_f T} n(d1) / (S sigma sqrt(T))``."""
    _d1, _, df_f, _ = _core(S, K, T, r_d, r_f, sigma)
    return df_f * norm.pdf(_d1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, T: float, r_d: float, r_f: float,
         sigma: float) -> float:
    """Vega ``dV/dsigma = S e^{-r_f T} n(d1) sqrt(T)`` (call = put)."""
    _d1, _, df_f, _ = _core(S, K, T, r_d, r_f, sigma)
    return S * df_f * norm.pdf(_d1) * math.sqrt(T)


def vanna(S: float, K: float, T: float, r_d: float, r_f: float,
          sigma: float) -> float:
    """Vanna ``d2V/(dS dsigma) = -e^{-r_f T} n(d1) d2 / sigma``.

    The sensitivity a 25-delta risk reversal position monetises.
    """
    _d1, _d2, df_f, _ = _core(S, K, T, r_d, r_f, sigma)
    return -df_f * norm.pdf(_d1) * _d2 / sigma


def volga(S: float, K: float, T: float, r_d: float, r_f: float,
          sigma: float) -> float:
    """Volga ``d2V/dsigma2 = vega * d1 * d2 / sigma``.

    The sensitivity a 25-delta butterfly position monetises.
    """
    _d1, _d2, _, _ = _core(S, K, T, r_d, r_f, sigma)
    return vega(S, K, T, r_d, r_f, sigma) * _d1 * _d2 / sigma


def analytic_greeks(S: float, K: float, T: float, r_d: float, r_f: float,
                    sigma: float, option_type: str) -> GreeksResult:
    """Closed-form GK Greeks.

    Parameters
    ----------
    S, K, T, r_d, r_f, sigma : float
        As in :func:`fx_options.garman_kohlhagen.gk_price`; requires
        T > 0 and sigma > 0.
    option_type : str
        ``"call"`` or ``"put"``.

    Returns
    -------
    GreeksResult
        price, delta_spot, delta_forward, gamma, vega, theta (per year),
        rho_domestic, rho_foreign, vanna, volga.
    """
    phi = validate_option_type(option_type)
    validate_inputs(S, K, T, r_d, r_f, sigma)
    if T <= 0.0 or sigma <= 0.0:
        raise ValueError("analytic_greeks requires T > 0 and sigma > 0")
    _d1, _d2, df_f, df_d = _core(S, K, T, r_d, r_f, sigma)
    sqrt_t = math.sqrt(T)
    n_d1 = norm.pdf(_d1)
    N_pd1 = norm.cdf(phi * _d1)
    N_pd2 = norm.cdf(phi * _d2)

    price = phi * (S * df_f * N_pd1 - K * df_d * N_pd2)
    delta_spot = phi * df_f * N_pd1
    theta = (-S * df_f * n_d1 * sigma / (2.0 * sqrt_t)
             + phi * (r_f * S * df_f * N_pd1 - r_d * K * df_d * N_pd2))
    return GreeksResult(
        price=price,
        delta_spot=delta_spot,
        delta_forward=phi * N_pd1,
        gamma=df_f * n_d1 / (S * sigma * sqrt_t),
        vega=S * df_f * n_d1 * sqrt_t,
        theta=theta,
        rho_domestic=phi * K * T * df_d * N_pd2,
        rho_foreign=-phi * S * T * df_f * N_pd1,
        vanna=-df_f * n_d1 * _d2 / sigma,
        volga=S * df_f * n_d1 * sqrt_t * _d1 * _d2 / sigma,
    )


def finite_difference_greeks(S: float, K: float, T: float, r_d: float,
                             r_f: float, sigma: float, option_type: str,
                             rel_bump: float = 1e-5) -> dict[str, float]:
    """Central finite-difference Greeks for validating the analytic set.

    Uses relative bumps of size ``rel_bump`` on S and sigma, absolute
    bumps on rates, and a forward difference in calendar time for theta
    (``theta = -dV/dT``).  Second-order Greeks (gamma, vanna, volga) use
    the standard central stencils.

    Returns
    -------
    dict
        Keys matching :class:`GreeksResult` fields (except price and
        delta_forward).
    """
    validate_option_type(option_type)
    validate_inputs(S, K, T, r_d, r_f, sigma)
    h_s = S * rel_bump
    h_v = max(sigma * rel_bump, 1e-7)
    h_r = 1e-6
    h_t = min(1e-6, T / 4.0)

    def p(s=S, sig=sigma, rd=r_d, rf=r_f, t=T) -> float:
        return gk_price(s, K, t, rd, rf, sig, option_type)

    # Larger bump for the sigma second difference: with h ~ sigma*1e-5 the
    # O(eps/h^2) round-off term dominates; sigma*1e-3 balances round-off
    # against the O(h^2) truncation error.
    h_v2 = max(sigma * 1e-3, 1e-5)
    base = p()
    up_s, dn_s = p(s=S + h_s), p(s=S - h_s)
    up_v, dn_v = p(sig=sigma + h_v), p(sig=sigma - h_v)
    up_v2, dn_v2 = p(sig=sigma + h_v2), p(sig=sigma - h_v2)
    return {
        "delta_spot": (up_s - dn_s) / (2 * h_s),
        "gamma": (up_s - 2 * base + dn_s) / (h_s * h_s),
        "vega": (up_v - dn_v) / (2 * h_v),
        "theta": -(p(t=T + h_t) - p(t=T - h_t)) / (2 * h_t),
        "rho_domestic": (p(rd=r_d + h_r) - p(rd=r_d - h_r)) / (2 * h_r),
        "rho_foreign": (p(rf=r_f + h_r) - p(rf=r_f - h_r)) / (2 * h_r),
        "vanna": (p(s=S + h_s, sig=sigma + h_v) - p(s=S + h_s, sig=sigma - h_v)
                  - p(s=S - h_s, sig=sigma + h_v)
                  + p(s=S - h_s, sig=sigma - h_v)) / (4 * h_s * h_v),
        "volga": (up_v2 - 2 * base + dn_v2) / (h_v2 * h_v2),
    }
