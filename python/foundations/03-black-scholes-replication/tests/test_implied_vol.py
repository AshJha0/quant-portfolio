"""Implied-vol round trips: price -> implied vol -> price, across low
and high vol regimes (which exercises both the Newton-Raphson stage
and the bisection fallback), plus the no-arbitrage-bound guard.
"""
import pytest

from eq_bs_replication import call_price, implied_volatility

S, K, r, sigma, T = 100.0, 105.0, 0.03, 0.25, 0.75


def test_implied_vol_round_trip():
    for true_vol in (0.08, 0.2, 0.55, 1.2):
        price = call_price(S, K, r, true_vol, T)
        iv = implied_volatility(price, S, K, r, T)
        assert abs(iv - true_vol) < 1e-6, (true_vol, iv)


def test_implied_vol_round_trip_across_moneyness():
    # Deep ITM combined with low vol (e.g. K=60, sigma=10% at S=100) is
    # deliberately excluded here: vega is tiny in that corner, so the
    # price carries almost no information about sigma and the round
    # trip is imprecise by design, not by bug -- see
    # test_edge_cases.py::test_deep_itm_low_vol_round_trip_is_imprecise
    # for a dedicated, documented test of that limitation.
    for k in (80.0, 100.0, 120.0):
        for true_vol in (0.1, 0.3, 0.8):
            price = call_price(S, k, r, true_vol, T)
            iv = implied_volatility(price, S, k, r, T)
            assert abs(iv - true_vol) < 1e-5, (k, true_vol, iv)
    # 160 (moderately OTM) at low vol is still fine but slightly
    # coarser than ATM; check it with a looser (still tight) tolerance.
    price = call_price(S, 160.0, r, 0.1, T)
    iv = implied_volatility(price, S, 160.0, r, T)
    assert abs(iv - 0.1) < 1e-4


def test_implied_vol_round_trip_short_and_long_dated():
    for t in (7 / 365, 0.5, 3.0):
        for true_vol in (0.15, 0.6):
            price = call_price(S, K, r, true_vol, t)
            iv = implied_volatility(price, S, K, r, t)
            assert abs(iv - true_vol) < 1e-5, (t, true_vol, iv)


def test_implied_vol_rejects_price_below_intrinsic():
    with pytest.raises(ValueError):
        implied_volatility(-1.0, S, K, r, T)


def test_implied_vol_rejects_price_above_spot():
    with pytest.raises(ValueError):
        implied_volatility(S + 1.0, S, K, r, T)
