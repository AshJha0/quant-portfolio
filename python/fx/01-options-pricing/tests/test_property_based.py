"""Property-based tests (hypothesis): parity, monotonicity, convexity, deltas, IV."""

import math

from fx_options import analytic_greeks, delta, gk_price, implied_vol, strike_from_delta
from hypothesis import assume, given, settings
from hypothesis import strategies as st

spot = st.floats(0.5, 200.0)
moneyness = st.floats(0.6, 1.6)
expiry = st.floats(0.02, 5.0)
rate = st.floats(-0.01, 0.08)
vol = st.floats(0.03, 0.8)
EPS = 2.220446049250313e-16


@settings(max_examples=300, deadline=None)
@given(spot, moneyness, expiry, rate, rate, vol)
def test_put_call_parity(S, m, T, rd, rf, sigma) -> None:
    K = S * m
    lhs = gk_price(S, K, T, rd, rf, sigma, "call") - gk_price(S, K, T, rd, rf, sigma, "put")
    rhs = S * math.exp(-rf * T) - K * math.exp(-rd * T)
    assert abs(lhs - rhs) <= 64 * EPS * max(S, K)


@settings(max_examples=300, deadline=None)
@given(spot, moneyness, expiry, rate, rate, vol, st.floats(1.001, 1.5))
def test_price_increases_with_vol(S, m, T, rd, rf, sigma, bump) -> None:
    K = S * m
    for otype in ("call", "put"):
        lo = gk_price(S, K, T, rd, rf, sigma, otype)
        hi = gk_price(S, K, T, rd, rf, sigma * bump, otype)
        assert hi >= lo - 16 * EPS * max(S, K)


@settings(max_examples=300, deadline=None)
@given(spot, moneyness, expiry, rate, rate, vol, st.floats(1.001, 1.3))
def test_monotone_and_convex_in_strike(S, m, T, rd, rf, sigma, step) -> None:
    K1 = S * m
    K2 = K1 * step
    K3 = 2.0 * K2 - K1
    noise = 64 * EPS * max(S, K3)
    for otype in ("call", "put"):
        p1, p2, p3 = (gk_price(S, k, T, rd, rf, sigma, otype) for k in (K1, K2, K3))
        assert (p2 <= p1 + noise) if otype == "call" else (p2 >= p1 - noise)
        assert p1 + p3 - 2.0 * p2 >= -4 * noise


@settings(max_examples=300, deadline=None)
@given(spot, moneyness, expiry, rate, rate, vol)
def test_delta_bounds_all_conventions(S, m, T, rd, rf, sigma) -> None:
    K = S * m
    dff = math.exp(-rf * T)
    tol = 8 * EPS
    for conv, cap in (("spot", dff), ("forward", 1.0)):
        c = delta(S, K, T, rd, rf, sigma, "call", conv)
        p = delta(S, K, T, rd, rf, sigma, "put", conv)
        assert -tol <= c <= cap + tol and -cap - tol <= p <= tol
        assert abs((c - p) - cap) <= tol
    # Premium-adjusted call delta never exceeds the unadjusted one.
    assert delta(S, K, T, rd, rf, sigma, "call", "spot_pa") <= \
        delta(S, K, T, rd, rf, sigma, "call", "spot") + tol


@settings(max_examples=200, deadline=None)
@given(spot, expiry, rate, rate, vol, st.floats(0.05, 0.45),
       st.sampled_from(["spot", "forward", "forward_pa"]))
def test_strike_from_delta_round_trip(S, T, rd, rf, sigma, d, conv) -> None:
    K = strike_from_delta(-d, S, T, rd, rf, sigma, "put", conv)
    assert abs(delta(S, K, T, rd, rf, sigma, "put", conv) + d) <= 1e-10


@settings(max_examples=300, deadline=None)
@given(spot, moneyness, expiry, rate, rate, vol, st.sampled_from(["call", "put"]))
def test_iv_round_trip(S, m, T, rd, rf, sigma, otype) -> None:
    K = S * m
    price = gk_price(S, K, T, rd, rf, sigma, otype)
    vega = analytic_greeks(S, K, T, rd, rf, sigma, otype).vega
    lower = gk_price(S, K, T, rd, rf, 0.0, otype)
    assume(price - lower > 1e-12 * max(S, K))
    got = implied_vol(price, S, K, T, rd, rf, otype)
    assert abs(got - sigma) <= 8 * EPS * max(S, K) / vega + 2e-12 * sigma
