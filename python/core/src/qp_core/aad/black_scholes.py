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

import threading
from typing import NamedTuple

from qp_core.model import model

from .tape import Real, Tape, exp_neg_mul, half_square, ln_div, ncdf_scaled, sqrt


def bs_price(S: Real, K: Real, T: Real, r: Real, q: Real, sigma: Real, phi: float = 1.0) -> Real:
    """Generic price: floats in, float out; tape variables in, tape variable out."""
    sd = sigma * sqrt(T)
    d1 = (ln_div(S, K) + (r - q + half_square(sigma)) * T) / sd
    d2 = d1 - sd
    return phi * (S * exp_neg_mul(q, T) * ncdf_scaled(d1, phi)
                  - K * exp_neg_mul(r, T) * ncdf_scaled(d2, phi))


class AadGreeks(NamedTuple):
    price: float
    delta: float
    vega: float
    rho: float
    rho_q: float
    dK: float
    dT: float


@model(id="CORE-AAD-BS")
def bs_greeks_aad(S: float, K: float, T: float, r: float, q: float, sigma: float,
                  is_call: bool = True) -> AadGreeks:
    """Price and all first-order Greeks from one forward and one backward pass.

    Records on a per-thread tape that is cleared, not rebuilt, between
    calls; ``bs_greeks_aad_on`` takes a caller's tape instead.
    """
    tape = getattr(_LOCAL, "tape", None)
    if tape is None:
        tape = _LOCAL.tape = Tape()
    return bs_greeks_aad_on(tape, S, K, T, r, q, sigma, is_call)


_LOCAL = threading.local()


@model(id="CORE-AAD-BS")
def bs_greeks_aad_on(tape: Tape, S: float, K: float, T: float, r: float, q: float,
                     sigma: float, is_call: bool = True) -> AadGreeks:
    """:func:`bs_greeks_aad` recorded on ``tape``, which is cleared first."""
    tape.clear()
    xs = tape.vars((S, K, T, r, q, sigma))
    vS, vK, vT, vr, vq, vs = xs
    v = bs_price(vS, vK, vT, vr, vq, vs, 1.0 if is_call else -1.0)
    assert not isinstance(v, float)
    tape.zero_adjoints()
    tape.backward(v, 1.0, len(xs))  # inputs have no parents: stop above them
    g = tape.adjoints(xs)
    return AadGreeks(v.value, g[0], g[5], g[3], g[4], g[1], g[2])
