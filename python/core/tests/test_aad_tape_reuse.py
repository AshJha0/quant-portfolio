"""Round 18: tape reuse and fused nodes in ``qp_core.aad`` (Black-Scholes).

The float pricer must keep its old bits (it is the cost-ratio denominator
and the price users see), and a reused tape must give exactly the Greeks of
a fresh one.
"""

from __future__ import annotations

import math
import threading

import pytest
from qp_core.aad import Tape, bs_greeks_aad, bs_price
from qp_core.aad.black_scholes import bs_greeks_aad_on
from qp_core.aad.tape import exp, exp_neg_mul, half_square, ln_div, log, ncdf, ncdf_scaled

CASES = [
    (100.0, 100.0, 1.0, 0.05, 0.02, 0.20),
    (80.0, 120.0, 2.0, 0.01, 0.0, 0.40),
    (1.3, 1.25, 0.25, -0.005, 0.03, 0.11),
]


def _old_float_price(S, K, T, r, q, sigma, phi):
    """The pre-Round-18 expression, verbatim."""
    sd = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sd
    d2 = d1 - sd
    n = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))  # noqa: E731
    return phi * (S * math.exp(-q * T) * n(phi * d1) - K * math.exp(-r * T) * n(phi * d2))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("phi", [1.0, -1.0])
def test_float_price_bits_unchanged(case, phi):
    assert bs_price(*case, phi) == _old_float_price(*case, phi)


@pytest.mark.parametrize("c", [1.0, -1.0])
@pytest.mark.parametrize("a,b", [(0.05, 1.3), (1.7, 0.4)])
def test_fused_nodes_match_unfused(a, b, c):
    fused = [lambda x, y: exp_neg_mul(x, y), lambda x, y: ln_div(x, y),
             lambda x, y: ncdf_scaled(x, c) * y, lambda x, y: half_square(x) * y,
             lambda x, y: x * 1.0 * y]
    plain = [lambda x, y: exp(-(x * y)), lambda x, y: log(x / y),
             lambda x, y: ncdf(c * x) * y, lambda x, y: x * x * 0.5 * y,
             lambda x, y: x * y]
    for f, p in zip(fused, plain, strict=True):
        assert f(a, b) == p(a, b)
        t = Tape()
        x, y = t.var(a), t.var(b)
        vf, vp = f(x, y), p(x, y)
        assert vf.value == vp.value
        gf, gp = t.gradient(vf, [x, y]), t.gradient(vp, [x, y])
        assert gf == pytest.approx(gp, rel=1e-14, abs=1e-15)


def test_multiply_by_one_records_nothing():
    t = Tape()
    x = t.var(2.0)
    assert x * 1.0 is x
    assert len(t) == 1


def test_reused_tape_gives_identical_greeks():
    tape = Tape()
    a = bs_greeks_aad_on(tape, *CASES[0])
    n = len(tape)
    bs_greeks_aad_on(tape, *CASES[1], is_call=False)
    assert bs_greeks_aad_on(tape, *CASES[0]) == a
    assert len(tape) == n
    assert bs_greeks_aad(*CASES[0]) == a
    assert bs_greeks_aad(*CASES[0]) == a


def test_thread_local_tape_per_thread():
    out: list[object] = []
    th = threading.Thread(target=lambda: out.append(bs_greeks_aad(*CASES[1])))
    th.start()
    th.join()
    assert out == [bs_greeks_aad(*CASES[1])]
