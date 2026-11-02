"""Black-Scholes in cost-of-carry form: the kernel behind the equity and FX
option packages.

The plain idea
--------------
A European option on a stock that pays a continuous dividend yield ``q`` and
a European option on a currency whose foreign interest rate is ``r_f`` are
the same contract to a pricer. In both cases the underlying grows in the
risk-neutral measure at the cost of carry ``b`` (``b = r - q`` for a stock,
``b = r_d - r_f`` for a currency) and the payoff is discounted at ``r``
(``r_d`` for a currency). Garman and Kohlhagen (1983) made exactly this
observation. So one kernel prices both, and the packages
``eq_options`` (Black-Scholes-Merton) and ``fx_options`` (Garman-Kohlhagen)
are thin wrappers that keep their own names, argument orders, validation and
boundary rules.

The formal statement
--------------------
With ``phi = +1`` for a call and ``-1`` for a put, ``sqrt_t = sqrt(T)`` and
the carry yield ``q = r - b``::

    d1 = [ln(S/K) + (b + sigma^2/2) T] / (sigma sqrt_t),  d2 = d1 - sigma sqrt_t
    V  = phi [S e^{-qT} N(phi d1) - K e^{-rT} N(phi d2)]

and the analytic Greeks (theta per year, vega per unit of vol)::

    delta = phi e^{-qT} N(phi d1)          gamma = e^{-qT} n(d1) / (S sigma sqrt_t)
    vega  = S e^{-qT} n(d1) sqrt_t         vanna = -e^{-qT} n(d1) d2 / sigma
    volga = vega d1 d2 / sigma             rho   = phi K T e^{-rT} N(phi d2)
    rho_q = -phi S T e^{-qT} N(phi d1)     (dV/dq; the FX package calls it rho_foreign)
    theta = -S e^{-qT} n(d1) sigma / (2 sqrt_t) + phi [q S e^{-qT} N(phi d1) - r K e^{-rT} N(phi d2)]

The functions take the discount rate ``r`` and the carry yield ``q`` rather
than ``b``. That is the cost-of-carry form written so the two discount
factors ``e^{-qT}`` and ``e^{-rT}`` are computed from the caller's own
numbers: recovering ``q`` as ``r - b`` would round, and the packages' golden
values would move in the last bit. ``d1`` uses ``r - q``, which is ``b``.

Every function here is unchecked: the caller has validated its inputs
(``S, K, T, sigma > 0`` and finite rates for the general formula). The
packages own validation and the boundary branches (``T == 0``, ``sigma == 0``
and, for equity, ``S == 0`` or ``K == 0``), because their domains differ.

Worked example
--------------
S = 100, K = 100, T = 1, r = 5 %, q = 2 %, sigma = 20 %: ``b = 3 %``,
``d1 = (0 + 0.03 + 0.02) / 0.2 = 0.25``, ``d2 = 0.05``, so the call is
``100 e^{-0.02} N(0.25) - 100 e^{-0.05} N(0.05) = 9.2270`` to four decimals.
The same number is a EURUSD-style call with S = K = 100, r_d = 5 %,
r_f = 2 %.

>>> round(carry_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.2, 1.0), 4)
9.227
>>> d1, d2 = carry_d1_d2(100.0, 100.0, 1.0, 0.05, 0.02, 0.2)
>>> round(d1, 12), round(d2, 12)
(0.25, 0.05)

Floating point
--------------
The expressions are evaluated in a fixed order so both packages return the
values they returned before they shared this module, bit for bit (the
equity wrappers add ``+ 0.0`` to the phi form, which turns the ``-0.0`` of a
put worth exactly zero back into the ``+0.0`` they always returned)
(``tests/test_carry_twin.py`` at the repository root pins the agreement).
``N`` is :func:`qp_core.normal.norm_cdf` (``0.5 erfc(-x / sqrt 2)``, full
relative precision in the lower tail). The scalar Greeks use
``n(x) = exp(-x^2/2) / sqrt(2 pi)``; the array Greeks multiply by the rounded
``1 / sqrt(2 pi)`` instead, which is what the array entry points always did.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc

from qp_core.normal import norm_cdf

__all__ = [
    "CarryGreeks",
    "CarryTerms",
    "carry_d1_d2",
    "carry_discounts",
    "carry_greeks",
    "carry_greeks_array",
    "carry_price",
    "carry_price_array",
    "carry_terms_array",
    "carry_value",
    "finite",
    "safe_exp",
]

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_RSQRT2 = 1.0 / np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)

Array = NDArray[np.float64]


def safe_exp(x: float, what: str) -> float:
    """``math.exp`` that raises ``ValueError`` (not ``OverflowError``) on overflow."""
    try:
        return math.exp(x)
    except OverflowError:
        raise ValueError(f"{what}: exp({x!r}) overflows a double") from None


def finite(value: float, what: str) -> float:
    """Return ``value``, or raise ``ValueError`` if it is not finite."""
    if not math.isfinite(value):
        raise ValueError(f"{what} is not representable as a finite double")
    return value


def carry_discounts(S: float, K: float, T: float, r: float, q: float, what: str) -> tuple[float, float]:
    """``(S e^{-qT}, K e^{-rT})``; ``ValueError`` naming ``what`` on overflow."""
    return S * safe_exp(-q * T, what), K * safe_exp(-r * T, what)


def carry_d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    """``d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T)`` and ``d2 = d1 - sigma sqrt T``."""
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    return d1, d1 - sigma * sqrt_t


def carry_value(disc_s: float, disc_k: float, d1: float, d2: float, phi: float) -> float:
    """``phi [disc_s N(phi d1) - disc_k N(phi d2)]`` from precomputed terms."""
    return phi * (disc_s * norm_cdf(phi * d1) - disc_k * norm_cdf(phi * d2))


def carry_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float, phi: float, what: str = "carry_price"
) -> float:
    """Price for ``S, K, T, sigma > 0``; ``phi`` is +1 (call) or -1 (put).

    Raises ``ValueError`` naming ``what`` if a discount factor overflows or
    the price is not finite.
    """
    disc_s, disc_k = carry_discounts(S, K, T, r, q, what)
    d1, d2 = carry_d1_d2(S, K, T, r, q, sigma)
    return finite(carry_value(disc_s, disc_k, d1, d2, phi), what)


class CarryGreeks(NamedTuple):
    """Price and analytic Greeks; units as in the module docstring."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    rho_q: float
    vanna: float
    volga: float
    d1: float
    d2: float


def carry_greeks(
    S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool
) -> CarryGreeks:
    """Analytic Greeks for ``S, K, T, sigma > 0``.

    ``price``, ``delta``, ``rho`` and ``theta`` are the call or put
    expressions written out separately (not through ``phi``), the order the
    equity package has always used.
    """
    d1, d2 = carry_d1_d2(S, K, T, r, q, sigma)
    sqrt_t = math.sqrt(T)
    df_q = math.exp(-q * T)
    df_r = math.exp(-r * T)
    pdf_d1 = math.exp(-0.5 * d1 * d1) / _SQRT_2PI

    gamma = df_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * df_q * pdf_d1 * sqrt_t
    vanna = -df_q * pdf_d1 * d2 / sigma
    volga = vega * d1 * d2 / sigma
    common_theta = -S * df_q * pdf_d1 * sigma / (2.0 * sqrt_t)

    if is_call:
        n1, n2 = norm_cdf(d1), norm_cdf(d2)
        price = S * df_q * n1 - K * df_r * n2
        delta = df_q * n1
        theta = common_theta + q * S * df_q * n1 - r * K * df_r * n2
        rho = K * T * df_r * n2
        rho_q = -S * T * df_q * n1
    else:
        m1, m2 = norm_cdf(-d1), norm_cdf(-d2)
        price = K * df_r * m2 - S * df_q * m1
        delta = -df_q * m1
        theta = common_theta - q * S * df_q * m1 + r * K * df_r * m2
        rho = -K * T * df_r * m2
        rho_q = S * T * df_q * m1
    return CarryGreeks(price, delta, gamma, vega, theta, rho, rho_q, vanna, volga, d1, d2)


# --------------------------------------------------------------------------
# NumPy arrays. Same formulas element by element; inputs are float64 arrays
# of one broadcast shape, already validated, with harmless stand-ins (1.0)
# wherever the caller will take a boundary branch instead.
# --------------------------------------------------------------------------


def _ncdf_array(x: Array) -> Array:
    out: Array = 0.5 * erfc(-x * _RSQRT2)
    return out


def carry_price_array(
    S: Array,
    K: Array,
    T: Array,
    r: Array,
    q: Array,
    sigma: Array,
    phi: Array,
    disc_s: Array | None = None,
    disc_k: Array | None = None,
) -> Array:
    """General-formula prices, ``phi [S e^{-qT} N(phi d1) - K e^{-rT} N(phi d2)]``.

    ``disc_s`` and ``disc_k`` (``S e^{-qT}``, ``K e^{-rT}``) may be passed in
    when the caller already has them for its boundary branches. Call inside
    ``np.errstate``: stand-in elements may overflow harmlessly.
    """
    if disc_s is None:
        disc_s = S * np.exp(-q * T)
    if disc_k is None:
        disc_k = K * np.exp(-r * T)
    vst = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vst
    d2 = d1 - vst
    out: Array = phi * (disc_s * _ncdf_array(phi * d1) - disc_k * _ncdf_array(phi * d2))
    return out


class CarryTerms(NamedTuple):
    """The shared pieces of the array Greeks (``n1 = N(phi d1)``, ``n2 = N(phi d2)``)."""

    sqrt_t: Array
    d1: Array
    d2: Array
    df_q: Array
    df_r: Array
    pdf: Array
    n1: Array
    n2: Array


def carry_terms_array(
    S: Array, K: Array, T: Array, r: Array, q: Array, sigma: Array, phi: Array
) -> CarryTerms:
    """``d1``, ``d2``, discount factors, ``n(d1)``, ``N(phi d1)`` and ``N(phi d2)``."""
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return CarryTerms(
        sqrt_t,
        d1,
        d2,
        np.exp(-q * T),
        np.exp(-r * T),
        np.exp(-0.5 * d1 * d1) * _INV_SQRT_2PI,
        _ncdf_array(phi * d1),
        _ncdf_array(phi * d2),
    )


def carry_greeks_array(
    S: Array, K: Array, T: Array, r: Array, q: Array, sigma: Array, phi: Array
) -> dict[str, Array]:
    """Price and analytic Greeks for arrays with ``S, K, T, sigma > 0``.

    Keys: price, delta, gamma, vega, theta, rho, rho_q, vanna, volga. Theta
    adds its terms in the call/put order of :func:`carry_greeks`
    (``common + a - b`` for a call, ``common - a + b`` for a put).
    """
    t = carry_terms_array(S, K, T, r, q, sigma, phi)
    vega = S * t.df_q * t.pdf * t.sqrt_t
    common_theta = -S * t.df_q * t.pdf * sigma / (2.0 * t.sqrt_t)
    a = q * S * t.df_q * t.n1
    b = r * K * t.df_r * t.n2
    is_call = phi > 0.0
    return {
        "price": phi * (S * t.df_q * t.n1 - K * t.df_r * t.n2),
        "delta": phi * t.df_q * t.n1,
        "gamma": t.df_q * t.pdf / (S * sigma * t.sqrt_t),
        "vega": vega,
        "theta": np.where(is_call, common_theta + a - b, common_theta - a + b),
        "rho": phi * K * T * t.df_r * t.n2,
        "rho_q": -phi * S * T * t.df_q * t.n1,
        "vanna": -t.df_q * t.pdf * t.d2 / sigma,
        "volga": vega * t.d1 * t.d2 / sigma,
    }
