"""Analytic Black-Scholes Greeks and finite-difference Greeks for any pricer.

Units and conventions
---------------------
* delta : dV/dS, dimensionless (per unit of spot).
* gamma : d2V/dS2, per currency unit.
* vega  : dV/dsigma, currency units per unit of annualised vol
  (divide by 100 for the market's 'per vol point').
* theta : dV/dt, currency units per *year* (divide by 365 for per-day).
* rho   : dV/dr, currency units per unit of rate (divide by 100 for per bp*100).
* vanna : d2V/(dS dsigma).
* volga : d2V/dsigma2 (a.k.a. vomma).

All rates continuously compounded, annualised; ``T`` in years (ACT/365F).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Callable

from scipy.stats import norm

from .black_scholes import OptionType, bs_price, d1_d2, validate_inputs

__all__ = [
    "BSGreeks",
    "bs_greeks",
    "fd_greeks",
    "compare_greeks",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "vanna",
    "volga",
]

_SQRT_2PI = math.sqrt(2.0 * math.pi)

Pricer = Callable[..., float]


@dataclass(frozen=True)
class BSGreeks:
    """Container for the full Greek set of a European option.

    Attributes use the units documented in the module docstring.
    """

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    volga: float

    def as_dict(self) -> dict[str, float]:
        """Return the Greeks as a plain ``dict`` (field name -> value)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> BSGreeks:
    """Analytic Black-Scholes-Merton Greeks (with continuous dividend yield).

    Parameters
    ----------
    S, K : float
        Spot and strike (currency units), strictly positive.
    T : float
        Time to expiry in years (ACT/365F), strictly positive.
    r : float
        Continuously compounded annualised risk-free rate.
    sigma : float
        Annualised volatility, strictly positive.
    q : float
        Continuously compounded annualised dividend yield.
    option_type : {"call", "put"}
        Option payoff direction.

    Returns
    -------
    BSGreeks
        Price plus delta, gamma, vega, theta, rho, vanna, volga.

    Raises
    ------
    ValueError
        If inputs are invalid or ``S``, ``K``, ``T``, ``sigma`` are not
        strictly positive (Greeks are singular at the boundary).
    """
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    sqrt_t = math.sqrt(T)
    df_q = math.exp(-q * T)
    df_r = math.exp(-r * T)
    pdf_d1 = math.exp(-0.5 * d1 * d1) / _SQRT_2PI

    gamma_ = df_q * pdf_d1 / (S * sigma * sqrt_t)
    vega_ = S * df_q * pdf_d1 * sqrt_t
    vanna_ = -df_q * pdf_d1 * d2 / sigma
    volga_ = vega_ * d1 * d2 / sigma
    common_theta = -S * df_q * pdf_d1 * sigma / (2.0 * sqrt_t)

    if option_type == "call":
        price = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
        delta_ = df_q * norm.cdf(d1)
        theta_ = common_theta + q * S * df_q * norm.cdf(d1) - r * K * df_r * norm.cdf(d2)
        rho_ = K * T * df_r * norm.cdf(d2)
    elif option_type == "put":
        price = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
        delta_ = -df_q * norm.cdf(-d1)
        theta_ = common_theta - q * S * df_q * norm.cdf(-d1) + r * K * df_r * norm.cdf(-d2)
        rho_ = -K * T * df_r * norm.cdf(-d2)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    return BSGreeks(
        price=price,
        delta=delta_,
        gamma=gamma_,
        vega=vega_,
        theta=theta_,
        rho=rho_,
        vanna=vanna_,
        volga=volga_,
    )


def delta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
          option_type: OptionType = "call") -> float:
    """Analytic delta dV/dS (dimensionless). See :func:`bs_greeks`."""
    return bs_greeks(S, K, T, r, sigma, q, option_type).delta


def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
          option_type: OptionType = "call") -> float:
    """Analytic gamma d2V/dS2 (per currency unit). See :func:`bs_greeks`."""
    return bs_greeks(S, K, T, r, sigma, q, option_type).gamma


def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
         option_type: OptionType = "call") -> float:
    """Analytic vega dV/dsigma (per unit of vol). See :func:`bs_greeks`."""
    return bs_greeks(S, K, T, r, sigma, q, option_type).vega


def theta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
          option_type: OptionType = "call") -> float:
    """Analytic theta dV/dt (per year). See :func:`bs_greeks`."""
    return bs_greeks(S, K, T, r, sigma, q, option_type).theta


def rho(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
        option_type: OptionType = "call") -> float:
    """Analytic rho dV/dr (per unit of rate). See :func:`bs_greeks`."""
    return bs_greeks(S, K, T, r, sigma, q, option_type).rho


def vanna(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
          option_type: OptionType = "call") -> float:
    """Analytic vanna d2V/(dS dsigma). See :func:`bs_greeks`."""
    return bs_greeks(S, K, T, r, sigma, q, option_type).vanna


def volga(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
          option_type: OptionType = "call") -> float:
    """Analytic volga d2V/dsigma2. See :func:`bs_greeks`."""
    return bs_greeks(S, K, T, r, sigma, q, option_type).volga


def fd_greeks(
    pricer: Pricer,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    rel_bump: float = 1e-5,
    rel_bump2: float = 2e-4,
    **pricer_kwargs: object,
) -> BSGreeks:
    """Central finite-difference Greeks for *any* pricer with the BS signature.

    The pricer must accept ``(S, K, T, r, sigma, q, option_type, **kwargs)``
    positionally/by keyword and return a float price. Central differences
    are used everywhere; second derivatives use the standard three-point
    stencil; vanna uses the four-point cross stencil. Theta is reported as
    ``dV/dt = -dV/dT`` (per year).

    Parameters
    ----------
    pricer : callable
        Pricing function, e.g. :func:`eq_options.black_scholes.bs_price`
        or a lambda around :func:`eq_options.binomial.crr_price`.
    S, K, T, r, sigma, q, option_type
        Contract and market inputs, as in :func:`bs_greeks`.
    rel_bump : float
        Relative bump size for first derivatives; absolute bumps are
        ``rel_bump * max(|x|, 1)``. The default ``1e-5`` balances
        truncation vs round-off error for analytic pricers; noisy pricers
        (MC) need larger bumps.
    rel_bump2 : float
        Relative bump for second derivatives (gamma, vanna, volga), where
        round-off scales like ``eps / h^2`` and needs a larger ``h``
        (optimal ``h ~ eps^0.25``).
    **pricer_kwargs
        Extra keyword arguments forwarded to ``pricer`` (e.g. ``n_steps``).

    Returns
    -------
    BSGreeks
        Finite-difference price and Greeks (vanna/volga included).

    Raises
    ------
    ValueError
        If inputs are invalid, or ``T`` is too small to bump centrally.
    """
    validate_inputs(S, K, T, sigma, option_type)

    def f(s: float = S, sig: float = sigma, t: float = T, rr: float = r) -> float:
        return float(pricer(s, K, t, rr, sig, q, option_type, **pricer_kwargs))

    h_s = rel_bump * max(abs(S), 1.0)
    h_v = rel_bump * max(abs(sigma), 1.0)
    h_t = rel_bump * max(abs(T), 1.0)
    h_r = rel_bump * max(abs(r), 1.0)
    h_s2 = rel_bump2 * max(abs(S), 1.0)
    h_v2 = rel_bump2 * max(abs(sigma), 1.0)
    if T - h_t <= 0.0:
        raise ValueError(f"T={T} too small for a central theta bump of {h_t}")

    price = f()
    delta_ = (f(s=S + h_s) - f(s=S - h_s)) / (2.0 * h_s)
    gamma_ = (f(s=S + h_s2) - 2.0 * price + f(s=S - h_s2)) / (h_s2 * h_s2)
    vega_ = (f(sig=sigma + h_v) - f(sig=sigma - h_v)) / (2.0 * h_v)
    theta_ = -(f(t=T + h_t) - f(t=T - h_t)) / (2.0 * h_t)
    rho_ = (f(rr=r + h_r) - f(rr=r - h_r)) / (2.0 * h_r)
    vanna_ = (
        f(s=S + h_s2, sig=sigma + h_v2)
        - f(s=S + h_s2, sig=sigma - h_v2)
        - f(s=S - h_s2, sig=sigma + h_v2)
        + f(s=S - h_s2, sig=sigma - h_v2)
    ) / (4.0 * h_s2 * h_v2)
    volga_ = (f(sig=sigma + h_v2) - 2.0 * price + f(sig=sigma - h_v2)) / (h_v2 * h_v2)

    return BSGreeks(
        price=price,
        delta=delta_,
        gamma=gamma_,
        vega=vega_,
        theta=theta_,
        rho=rho_,
        vanna=vanna_,
        volga=volga_,
    )


def compare_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    rel_bump: float = 1e-5,
) -> dict[str, dict[str, float]]:
    """Tabulate analytic vs central finite-difference Black-Scholes Greeks.

    Parameters
    ----------
    S, K, T, r, sigma, q, option_type
        As in :func:`bs_greeks`.
    rel_bump : float
        Relative bump used for the finite-difference leg.

    Returns
    -------
    dict
        ``{greek: {"analytic": x, "finite_diff": y, "abs_err": |x-y|}}``
        for each Greek including price.
    """
    ana = bs_greeks(S, K, T, r, sigma, q, option_type).as_dict()
    num = fd_greeks(bs_price, S, K, T, r, sigma, q, option_type, rel_bump).as_dict()
    return {
        name: {
            "analytic": ana[name],
            "finite_diff": num[name],
            "abs_err": abs(ana[name] - num[name]),
        }
        for name in ana
    }
