"""Heston Greeks by finite differences on the Fourier pricer.

Finite differences with optional Richardson extrapolation
---------------------------------------------------------
First-order Greeks use central differences (O(h^2) error); with
``richardson=True`` the h and h/2 estimates are combined as
``(4 D(h/2) - D(h)) / 3``, cancelling the h^2 term (O(h^4)).  Gamma uses the
second central difference with the analogous ``(16 G(h/2) - G(h)) / 15``
extrapolation.  The Fourier pricer is smooth in all inputs, so FD Greeks are
stable across a wide range of bump sizes (tested).

Vega convention: under Heston the natural first-order vol sensitivity is
``dV/dv0`` (sensitivity to today's *variance*), reported here as
``vega_v0``.  The BS-equivalent vega ``dV/dsigma_imp`` relates via
``dV/dv0 = vega_BS * d(sigma_imp)/d(v0)``; ATM and short-dated,
``sigma_imp ~ sqrt(v0)`` so ``d sigma/d v0 ~ 1/(2 sqrt(v0))``.

Sticky-strike vs sticky-delta smile dynamics (and the delta correction)
-----------------------------------------------------------------------
When spot moves, what happens to the smile?

* **Sticky-strike**: the vol at each *fixed strike K* is unchanged;
  the hedger's delta is the plain BS delta at sigma(K).  Typical of
  short-horizon, range-bound markets.
* **Sticky-delta / sticky-moneyness**: the smile is a function of moneyness
  ``k = ln(K/F)`` and *rides along with the forward*; a spot move re-prices
  each strike at a new point on the same moneyness smile.  Then

      dV/dS = delta_BS + vega_BS * d(sigma)/dS,
      d(sigma)/dS = (d sigma/d k) * (dk/dS) = -(1/S) * d(sigma)/dk,

  so with the usual negative equity skew (d sigma/dk < 0) the sticky-
  moneyness delta is *higher* than sticky-strike delta for calls.  Local-vol
  models imply smile dynamics of roughly this type (Derman's "sticky local
  vol" regimes); Heston's own dynamics sit in between.
  :func:`smile_adjusted_delta` computes both deltas from a smile slope so
  the desk can see the size of the correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import bs_delta, bs_gamma, bs_vega, implied_vol
from .heston import HestonParams, heston_call_gl

__all__ = ["HestonGreeks", "heston_greeks", "bs_equivalent_greeks", "smile_adjusted_delta"]


@dataclass(frozen=True)
class HestonGreeks:
    """Finite-difference Greeks of a Heston European call.

    Units: ``delta`` per unit spot; ``gamma`` per unit spot^2; ``vega_v0``
    per unit of initial *variance* v0; ``rho_rate`` per unit rate
    (1.00 = 100 percentage points).
    """

    price: float
    delta: float
    gamma: float
    vega_v0: float
    rho_rate: float


def _price(S: float, K: float, T: float, r: float, q: float, p: HestonParams) -> float:
    return float(heston_call_gl(S, K, T, r, q, p))


def _central(f, x0: float, h: float) -> float:
    return (f(x0 + h) - f(x0 - h)) / (2.0 * h)


def _second(f, x0: float, h: float) -> float:
    return (f(x0 + h) - 2.0 * f(x0) + f(x0 - h)) / (h * h)


def _richardson1(f, x0: float, h: float) -> float:
    return (4.0 * _central(f, x0, h / 2.0) - _central(f, x0, h)) / 3.0


def _richardson2(f, x0: float, h: float) -> float:
    return (16.0 * _second(f, x0, h / 2.0) - _second(f, x0, h)) / 15.0


def heston_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    rel_bump: float = 1e-3,
    richardson: bool = False,
) -> HestonGreeks:
    """Delta, gamma, vega (dV/dv0) and rate rho of a Heston call by FD.

    Parameters
    ----------
    rel_bump : float
        Relative bump for spot (h = rel_bump * S); v0 and r use bumps scaled
        to their own magnitudes.
    richardson : bool
        Apply Richardson extrapolation (h and h/2) to each difference.

    Returns
    -------
    HestonGreeks
    """
    if T <= 0.0:
        raise ValueError(f"T must be positive for Greeks, got {T}")
    if rel_bump <= 0.0 or rel_bump > 0.2:
        raise ValueError(f"rel_bump must be in (0, 0.2], got {rel_bump}")

    price = _price(S, K, T, r, q, p)
    hS = rel_bump * S
    hv = max(rel_bump * p.v0, 1e-6)
    hr = max(rel_bump * max(abs(r), 0.01), 1e-6)

    f_S = lambda s: _price(s, K, T, r, q, p)

    def f_v(v0: float) -> float:
        return _price(S, K, T, r, q, HestonParams(v0, p.kappa, p.theta, p.rho, p.xi))

    f_r = lambda rr: _price(S, K, T, rr, q, p)

    d1 = _richardson1 if richardson else _central
    d2 = _richardson2 if richardson else _second

    return HestonGreeks(
        price=price,
        delta=float(d1(f_S, S, hS)),
        gamma=float(d2(f_S, S, hS)),
        vega_v0=float(d1(f_v, p.v0, hv)),
        rho_rate=float(d1(f_r, r, hr)),
    )


def bs_equivalent_greeks(
    S: float, K: float, T: float, r: float, q: float, p: HestonParams
) -> dict:
    """BS Greeks evaluated at the Heston-implied vol of the same option.

    Computes the Heston price, inverts it to an implied vol, and returns the
    BS delta/gamma/vega at that vol -- the numbers a BS-based risk system
    would show for the same market price.  Returns a dict with keys
    ``implied_vol``, ``delta``, ``gamma``, ``vega`` (vega is dV/dsigma).
    """
    price = _price(S, K, T, r, q, p)
    iv = implied_vol(price, S, K, T, r, q, "call")
    if not np.isfinite(iv):
        return {"implied_vol": np.nan, "delta": np.nan, "gamma": np.nan, "vega": np.nan}
    return {
        "implied_vol": iv,
        "delta": bs_delta(S, K, T, r, q, iv, "call"),
        "gamma": bs_gamma(S, K, T, r, q, iv),
        "vega": bs_vega(S, K, T, r, q, iv),
    }


def smile_adjusted_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    dsigma_dk: float,
) -> dict:
    """Sticky-strike vs sticky-moneyness call delta from a smile slope.

    Parameters
    ----------
    sigma : float
        Implied vol at (K, T).
    dsigma_dk : float
        Smile slope d(sigma)/dk at that point, k = ln(K/F) (e.g. from an SVI
        slice: ``d sigma/dk = w'(k) / (2 sigma T)``).

    Returns
    -------
    dict
        ``delta_sticky_strike`` (plain BS delta -- smile fixed in K),
        ``delta_sticky_moneyness`` (BS delta + vega * dsigma/dS with
        ``dsigma/dS = -dsigma_dk / S`` -- smile fixed in moneyness, i.e. the
        local-vol-style regime), and ``adjustment`` (their difference).
        With negative skew the adjustment is positive for calls.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    delta_ss = bs_delta(S, K, T, r, q, sigma, "call")
    vega = bs_vega(S, K, T, r, q, sigma)
    dsig_dS = -dsigma_dk / S
    delta_sm = delta_ss + vega * dsig_dS
    return {
        "delta_sticky_strike": float(delta_ss),
        "delta_sticky_moneyness": float(delta_sm),
        "adjustment": float(delta_sm - delta_ss),
    }
