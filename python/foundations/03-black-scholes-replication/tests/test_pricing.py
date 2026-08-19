"""Model-free pricing identities: put-call parity, no-arbitrage bounds,
and monotonicity in volatility.

These are checks derived from theory, not from a reference
implementation -- if they fail, the code is wrong regardless of whether
the Black-Scholes model itself is a good description of reality.
"""
import math

from eq_bs_replication import call_price, put_price

S, K, r, sigma, T = 100.0, 105.0, 0.03, 0.25, 0.75


def test_put_call_parity():
    # C - P = S - K e^{-rT} must hold exactly for European options
    # (replicate a call minus a put with a forward; this is a static,
    # model-free arbitrage relation, not something specific to BS).
    c = call_price(S, K, r, sigma, T)
    p = put_price(S, K, r, sigma, T)
    assert abs((c - p) - (S - K * math.exp(-r * T))) < 1e-10


def test_put_call_parity_grid():
    # The same identity across a broader grid of contracts, including
    # zero and negative rates.
    for s in (50.0, 100.0, 150.0):
        for k in (80.0, 100.0, 120.0):
            for rr in (-0.02, 0.0, 0.05):
                for t in (0.05, 1.0, 5.0):
                    for vol in (0.1, 0.5):
                        c = call_price(s, k, rr, vol, t)
                        p = put_price(s, k, rr, vol, t)
                        assert abs((c - p) - (s - k * math.exp(-rr * t))) < 1e-9


def test_no_arbitrage_bounds():
    c = call_price(S, K, r, sigma, T)
    assert max(S - K * math.exp(-r * T), 0) <= c <= S


def test_no_arbitrage_bounds_put():
    p = put_price(S, K, r, sigma, T)
    disc_k = K * math.exp(-r * T)
    assert max(disc_k - S, 0) <= p <= disc_k


def test_monotonic_in_vol():
    # A call is strictly increasing in volatility (vega > 0).
    prices = [call_price(S, K, r, v, T) for v in (0.1, 0.2, 0.3, 0.4)]
    assert all(a < b for a, b in zip(prices, prices[1:]))


def test_monotonic_in_vol_put():
    prices = [put_price(S, K, r, v, T) for v in (0.1, 0.2, 0.3, 0.4)]
    assert all(a < b for a, b in zip(prices, prices[1:]))


def test_call_decreasing_in_strike():
    # A call is monotone non-increasing in strike (deeper OTM is
    # weakly cheaper); the derivative is -K e^{-rT} N(d2) <= 0.
    prices = [call_price(S, k, r, sigma, T) for k in (80.0, 100.0, 120.0, 140.0)]
    assert all(a > b for a, b in zip(prices, prices[1:]))


def test_put_increasing_in_strike():
    prices = [put_price(S, k, r, sigma, T) for k in (80.0, 100.0, 120.0, 140.0)]
    assert all(a < b for a, b in zip(prices, prices[1:]))
