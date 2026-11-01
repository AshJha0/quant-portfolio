"""Property-based tests (hypothesis) on the closed form, Greeks and IV."""

import math

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from eq_options import bs_greeks, bs_price, implied_vol

spot = st.floats(1.0, 1000.0)
moneyness = st.floats(0.5, 2.0)
expiry = st.floats(0.01, 5.0)
rate = st.floats(-0.02, 0.10)
div = st.floats(0.0, 0.08)
vol = st.floats(0.05, 1.0)
EPS = 2.220446049250313e-16


@settings(max_examples=300, deadline=None)
@given(spot, moneyness, expiry, rate, div, vol)
def test_put_call_parity(S, m, T, r, q, sigma) -> None:
    K = S * m
    lhs = bs_price(S, K, T, r, sigma, q, "call") - bs_price(S, K, T, r, sigma, q, "put")
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(lhs - rhs) <= 64 * EPS * max(S, K)


@settings(max_examples=300, deadline=None)
@given(spot, moneyness, expiry, rate, div, vol, st.floats(1.001, 1.5))
def test_price_increases_with_vol(S, m, T, r, q, sigma, bump) -> None:
    K = S * m
    for otype in ("call", "put"):
        lo = bs_price(S, K, T, r, sigma, q, otype)
        hi = bs_price(S, K, T, r, sigma * bump, q, otype)
        assert hi >= lo - 16 * EPS * max(S, K)


@settings(max_examples=300, deadline=None)
@given(spot, moneyness, expiry, rate, div, vol, st.floats(1.001, 1.3))
def test_monotone_and_convex_in_strike(S, m, T, r, q, sigma, step) -> None:
    K1 = S * m
    K2 = K1 * step
    K3 = K2 + (K2 - K1)
    noise = 64 * EPS * max(S, K3)
    for otype in ("call", "put"):
        p1, p2, p3 = (bs_price(S, k, T, r, sigma, q, otype) for k in (K1, K2, K3))
        if otype == "call":
            assert p2 <= p1 + noise
        else:
            assert p2 >= p1 - noise
        assert p1 + p3 - 2.0 * p2 >= -4 * noise  # equally spaced strikes


@settings(max_examples=300, deadline=None)
@given(spot, moneyness, expiry, rate, div, vol)
def test_delta_bounds(S, m, T, r, q, sigma) -> None:
    K = S * m
    dq = math.exp(-q * T)
    call = bs_greeks(S, K, T, r, sigma, q, "call").delta
    put = bs_greeks(S, K, T, r, sigma, q, "put").delta
    assert 0.0 <= call <= dq * (1 + 4 * EPS)
    assert -dq * (1 + 4 * EPS) <= put <= 0.0
    assert abs((call - put) - dq) <= 4 * EPS


@settings(max_examples=300, deadline=None)
@given(spot, moneyness, expiry, rate, div, vol, st.sampled_from(["call", "put"]))
def test_iv_round_trip(S, m, T, r, q, sigma, otype) -> None:
    K = S * m
    price = bs_price(S, K, T, r, sigma, q, otype)
    vega = bs_greeks(S, K, T, r, sigma, q, otype).vega
    lower = bs_price(S, K, T, r, 0.0, q, otype)
    assume(price - lower > 1e-12 * max(S, K))
    got = implied_vol(price, S, K, T, r, q, otype)
    # A few ulps of the quote (magnitude max(price, S, K) after the parity
    # shift) mapped through vega, plus the solver's 1e-12 relative floor.
    assert abs(got - sigma) <= 8 * EPS * max(S, K) / vega + 2e-12 * sigma
