"""
Tests derived from theory rather than from a reference implementation:
put-call parity, no-arbitrage bounds, limiting behaviour, Greeks vs
finite differences, and implied-vol round trips.

Run with either:
    python test_black_scholes.py
    pytest test_black_scholes.py
"""
import math

import black_scholes as bs

S, K, r, sigma, T = 100.0, 105.0, 0.03, 0.25, 0.75


def test_put_call_parity():
    # C - P = S - K e^{-rT} must hold exactly for European options.
    c = bs.call_price(S, K, r, sigma, T)
    p = bs.put_price(S, K, r, sigma, T)
    assert abs((c - p) - (S - K * math.exp(-r * T))) < 1e-10


def test_no_arbitrage_bounds():
    c = bs.call_price(S, K, r, sigma, T)
    assert max(S - K * math.exp(-r * T), 0) <= c <= S


def test_monotonic_in_vol():
    # A call is strictly increasing in volatility (vega > 0).
    prices = [bs.call_price(S, K, r, v, T) for v in (0.1, 0.2, 0.3, 0.4)]
    assert all(a < b for a, b in zip(prices, prices[1:]))


def test_limits():
    # Deep ITM with negligible vol -> forward intrinsic value.
    c = bs.call_price(200.0, 100.0, r, 1e-6, T)
    assert abs(c - (200.0 - 100.0 * math.exp(-r * T))) < 1e-6
    # Deep OTM -> worthless.
    assert bs.call_price(10.0, 100.0, r, 0.2, T) < 1e-8


def test_greeks_match_finite_differences():
    g = bs.call_greeks(S, K, r, sigma, T)
    h = 1e-4
    delta_fd = (bs.call_price(S + h, K, r, sigma, T)
                - bs.call_price(S - h, K, r, sigma, T)) / (2 * h)
    vega_fd = (bs.call_price(S, K, r, sigma + h, T)
               - bs.call_price(S, K, r, sigma - h, T)) / (2 * h)
    gamma_fd = (bs.call_price(S + h, K, r, sigma, T)
                - 2 * bs.call_price(S, K, r, sigma, T)
                + bs.call_price(S - h, K, r, sigma, T)) / h**2
    assert abs(g.delta - delta_fd) < 1e-6
    assert abs(g.vega - vega_fd) < 1e-4
    assert abs(g.gamma - gamma_fd) < 1e-4


def test_implied_vol_round_trip():
    for true_vol in (0.08, 0.2, 0.55, 1.2):
        price = bs.call_price(S, K, r, true_vol, T)
        iv = bs.implied_volatility(price, S, K, r, T)
        assert abs(iv - true_vol) < 1e-6, (true_vol, iv)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
