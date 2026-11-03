"""qp_core.black_scholes: the cost-of-carry kernel on its own."""

import math

import numpy as np
import pytest
from qp_core.black_scholes import (
    carry_d1_d2,
    carry_greeks,
    carry_greeks_array,
    carry_price,
    carry_price_array,
    safe_exp,
)


def test_worked_example():
    # S = K = 100, T = 1, r = 5 %, q = 2 %, sigma = 20 %: d1 = 0.25, d2 = 0.05.
    d1, d2 = carry_d1_d2(100.0, 100.0, 1.0, 0.05, 0.02, 0.2)
    assert d1 == pytest.approx(0.25, abs=1e-15)
    assert d2 == pytest.approx(0.05, abs=1e-15)
    n = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))  # noqa: E731
    want = 100.0 * math.exp(-0.02) * n(0.25) - 100.0 * math.exp(-0.05) * n(0.05)
    assert carry_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.2, 1.0) == pytest.approx(want, rel=1e-15)


@pytest.mark.parametrize("q", [-0.01, 0.0, 0.03])
def test_put_call_parity(q):
    S, K, T, r, v = 103.0, 97.0, 0.7, 0.04, 0.3
    c = carry_price(S, K, T, r, q, v, 1.0)
    p = carry_price(S, K, T, r, q, v, -1.0)
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), rel=1e-13)


def test_zero_carry_is_black76_on_the_spot():
    # b = 0 (q = r): the forward equals the spot.
    S, K, T, r, v = 100.0, 110.0, 2.0, 0.03, 0.25
    d1, _ = carry_d1_d2(S, K, T, r, r, v)
    assert d1 == pytest.approx((math.log(S / K) + 0.5 * v * v * T) / (v * math.sqrt(T)), rel=1e-14)


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_match_finite_differences(is_call):
    S, K, T, r, q, v = 100.0, 95.0, 0.5, 0.03, 0.01, 0.25
    phi = 1.0 if is_call else -1.0
    g = carry_greeks(S, K, T, r, q, v, is_call)

    def p(s=S, t=T, rr=r, qq=q, vv=v):
        return carry_price(s, K, t, rr, qq, vv, phi)

    h = 1e-4
    assert g.price == pytest.approx(p(), rel=1e-14)
    assert g.delta == pytest.approx((p(s=S + h) - p(s=S - h)) / (2 * h), rel=1e-7)
    assert g.gamma == pytest.approx((p(s=S + 1e-2) - 2 * p() + p(s=S - 1e-2)) / 1e-4, rel=1e-5)
    assert g.vega == pytest.approx((p(vv=v + h) - p(vv=v - h)) / (2 * h), rel=1e-7)
    assert g.theta == pytest.approx(-(p(t=T + h) - p(t=T - h)) / (2 * h), rel=1e-6)
    assert g.rho == pytest.approx((p(rr=r + h) - p(rr=r - h)) / (2 * h), rel=1e-7)
    assert g.rho_q == pytest.approx((p(qq=q + h) - p(qq=q - h)) / (2 * h), rel=1e-7)


def test_arrays_match_scalars():
    rng = np.random.default_rng(3)
    n = 500
    S, K = rng.uniform(50, 150, n), rng.uniform(50, 150, n)
    T, v = rng.uniform(0.01, 5, n), rng.uniform(0.05, 1.0, n)
    r, q = rng.uniform(-0.02, 0.1, n), rng.uniform(-0.02, 0.1, n)
    phi = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    prices = carry_price_array(S, K, T, r, q, v, phi)
    greeks = carry_greeks_array(S, K, T, r, q, v, phi)
    for i in range(n):
        args = (S[i], K[i], T[i], r[i], q[i], v[i])
        # scipy's erfc and math.erfc may differ in the last bit.
        assert abs(prices[i] - carry_price(*args, phi[i])) <= 1e-13 * max(1.0, S[i], K[i])
        g = carry_greeks(*args, phi[i] > 0)
        for name in ("price", "delta", "gamma", "vega", "theta", "rho", "rho_q", "vanna", "volga"):
            scale = max(1.0, S[i], K[i], abs(getattr(g, name)))
            assert abs(greeks[name][i] - getattr(g, name)) <= 1e-13 * scale, name


def test_overflow_is_a_value_error():
    with pytest.raises(ValueError, match="overflows"):
        safe_exp(1000.0, "x")
    with pytest.raises(ValueError, match="carry_price"):
        carry_price(100.0, 100.0, 10.0, -100.0, 0.0, 0.2, 1.0)
