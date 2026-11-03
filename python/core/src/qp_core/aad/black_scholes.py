"""Black-Scholes price with adjoint Greeks.

Plain idea: the price is written once, generically, so it runs on floats
(the price) or on tape variables (the price plus every first-order Greek
from one backward sweep). Formal statement: with carry ``b = r - q``,
``d1 = (ln(S/K) + (b + sigma^2/2) T) / (sigma sqrt T)``, ``d2 = d1 - sigma sqrt T``,
the price is ``phi (S e^{-qT} N(phi d1) - K e^{-rT} N(phi d2))`` (Hull 2022,
*Options, Futures and Other Derivatives*, ch. 19; the same expression as
``qp_core.black_scholes.carry_price``). The adjoint returns ``dV/dS``
(delta), ``dV/dsigma`` (vega), ``dV/dr`` (rho), ``dV/dq`` (rho_q), ``dV/dK``
and ``dV/dT``; theta in the calendar sense is ``-dV/dT``.

Worked example: S = K = 100, T = 1, r = 5 %, q = 2 %, sigma = 20 % gives a
call of 9.2270 with delta 0.5869 and vega 37.90 (per unit sigma), the
figures of ``carry_greeks`` to machine precision
(``tests/test_aad_python.py``).
"""

from __future__ import annotations

from typing import NamedTuple

from .tape import Real, Tape, exp, log, ncdf, sqrt


def bs_price(S: Real, K: Real, T: Real, r: Real, q: Real, sigma: Real, phi: float = 1.0) -> Real:
    """Generic price: floats in, float out; tape variables in, tape variable out."""
    sd = sigma * sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sd
    d2 = d1 - sd
    return phi * (S * exp(-q * T) * ncdf(phi * d1) - K * exp(-r * T) * ncdf(phi * d2))


class AadGreeks(NamedTuple):
    price: float
    delta: float
    vega: float
    rho: float
    rho_q: float
    dK: float
    dT: float


def bs_greeks_aad(S: float, K: float, T: float, r: float, q: float, sigma: float,
                  is_call: bool = True) -> AadGreeks:
    """Price and all first-order Greeks from one forward and one backward pass."""
    tape = Tape()
    xs = [tape.var(v) for v in (S, K, T, r, q, sigma)]
    vS, vK, vT, vr, vq, vs = xs
    v = bs_price(vS, vK, vT, vr, vq, vs, 1.0 if is_call else -1.0)
    assert not isinstance(v, float)
    g = tape.gradient(v, xs)
    return AadGreeks(v.value, g[0], g[5], g[3], g[4], g[1], g[2])
