"""Black-Scholes analytic identities, textbook benchmarks, put-call parity."""

import math

import pytest

from eq_options import bs_price, d1_d2, forward_price, intrinsic_value

S_GRID = [50.0, 80.0, 100.0, 120.0, 200.0]
K_GRID = [60.0, 100.0, 140.0]
T_GRID = [0.02, 0.25, 1.0, 3.0]
R_GRID = [-0.01, 0.0, 0.05]
Q_GRID = [0.0, 0.03]


def test_put_call_parity_full_grid_1e10() -> None:
    """C - P == S e^{-qT} - K e^{-rT} to 1e-10 across the whole grid."""
    checked = 0
    for S in S_GRID:
        for K in K_GRID:
            for T in T_GRID:
                for r in R_GRID:
                    for q in Q_GRID:
                        for sigma in (0.05, 0.2, 0.6):
                            c = bs_price(S, K, T, r, sigma, q, "call")
                            p = bs_price(S, K, T, r, sigma, q, "put")
                            lhs = c - p
                            rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
                            assert lhs == pytest.approx(rhs, abs=1e-10)
                            checked += 1
    assert checked == len(S_GRID) * len(K_GRID) * len(T_GRID) * len(R_GRID) * len(Q_GRID) * 3


@pytest.mark.parametrize(
    ("S", "K", "T", "r", "sigma", "q", "otype", "expected"),
    [
        # Hull, Options Futures and Other Derivatives (classic example):
        # S=42, K=40, r=10%, sigma=20%, T=0.5
        (42.0, 40.0, 0.5, 0.10, 0.20, 0.0, "call", 4.7594),
        (42.0, 40.0, 0.5, 0.10, 0.20, 0.0, "put", 0.8086),
        # ATM 1y benchmark widely tabulated
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "call", 10.4506),
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "put", 5.5735),
        # Haug, Complete Guide to Option Pricing Formulas: generalized BSM
        # put with cost-of-carry b=5% (q = r - b = 5%)
        (75.0, 70.0, 0.5, 0.10, 0.35, 0.05, "put", 4.0870),
    ],
)
def test_textbook_values(S: float, K: float, T: float, r: float, sigma: float,
                         q: float, otype: str, expected: float) -> None:
    assert bs_price(S, K, T, r, sigma, q, otype) == pytest.approx(expected, abs=1e-4)


def test_d1_d2_relationship() -> None:
    d1, d2 = d1_d2(100.0, 110.0, 2.0, 0.03, 0.25, 0.01)
    assert d2 == pytest.approx(d1 - 0.25 * math.sqrt(2.0), abs=1e-14)


def test_d1_d2_atm_forward() -> None:
    """At K = forward, d1 = sigma*sqrt(T)/2 exactly."""
    S, T, r, q, sigma = 100.0, 1.0, 0.05, 0.02, 0.3
    K = forward_price(S, T, r, q)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    assert d1 == pytest.approx(0.5 * sigma * math.sqrt(T), abs=1e-12)
    assert d2 == pytest.approx(-0.5 * sigma * math.sqrt(T), abs=1e-12)


def test_call_monotone_increasing_in_vol() -> None:
    vols = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    prices = [bs_price(100, 100, 1, 0.02, v, 0.01, "call") for v in vols]
    assert all(b > a for a, b in zip(prices, prices[1:]))


def test_price_convex_in_strike() -> None:
    """Butterfly K-h, K, K+h must have non-negative value (convexity)."""
    for K in (80.0, 100.0, 120.0):
        h = 5.0
        fly = (
            bs_price(100, K - h, 1, 0.03, 0.25, 0.0, "call")
            - 2 * bs_price(100, K, 1, 0.03, 0.25, 0.0, "call")
            + bs_price(100, K + h, 1, 0.03, 0.25, 0.0, "call")
        )
        assert fly >= 0.0


def test_call_decreasing_in_strike_put_increasing() -> None:
    strikes = [60.0, 80.0, 100.0, 120.0, 140.0]
    calls = [bs_price(100, k, 1, 0.03, 0.2, 0.0, "call") for k in strikes]
    puts = [bs_price(100, k, 1, 0.03, 0.2, 0.0, "put") for k in strikes]
    assert all(b < a for a, b in zip(calls, calls[1:]))
    assert all(b > a for a, b in zip(puts, puts[1:]))


def test_price_bounds() -> None:
    """max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}; analogous for puts."""
    for sigma in (0.05, 0.3, 1.0):
        S, K, T, r, q = 100.0, 90.0, 2.0, 0.04, 0.02
        c = bs_price(S, K, T, r, sigma, q, "call")
        p = bs_price(S, K, T, r, sigma, q, "put")
        assert max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0) - 1e-12 <= c
        assert c <= S * math.exp(-q * T) + 1e-12
        assert max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0) - 1e-12 <= p
        assert p <= K * math.exp(-r * T) + 1e-12


def test_deep_itm_call_approaches_discounted_forward_intrinsic() -> None:
    S, K, T, r, q, sigma = 1000.0, 10.0, 1.0, 0.05, 0.01, 0.2
    c = bs_price(S, K, T, r, sigma, q, "call")
    expected = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert c == pytest.approx(expected, rel=1e-10)


def test_deep_otm_call_near_zero_but_positive() -> None:
    c = bs_price(100.0, 1000.0, 0.25, 0.02, 0.15, 0.0, "call")
    assert 0.0 <= c < 1e-10


def test_intrinsic_value_helper() -> None:
    assert intrinsic_value(105.0, 100.0, "call") == 5.0
    assert intrinsic_value(95.0, 100.0, "call") == 0.0
    assert intrinsic_value(95.0, 100.0, "put") == 5.0
    assert intrinsic_value(105.0, 100.0, "put") == 0.0


def test_forward_price_identity() -> None:
    assert forward_price(100.0, 2.0, 0.05, 0.02) == pytest.approx(
        100.0 * math.exp(0.06), rel=1e-14
    )


def test_negative_rates_supported_and_parity_holds() -> None:
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, -0.02, 0.01, 0.2
    c = bs_price(S, K, T, r, sigma, q, "call")
    p = bs_price(S, K, T, r, sigma, q, "put")
    assert c > 0 and p > 0
    assert c - p == pytest.approx(
        S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10
    )
