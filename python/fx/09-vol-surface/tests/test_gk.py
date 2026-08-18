"""Garman-Kohlhagen: identities, Greeks vs FD, robust implied vol."""

import math

import numpy as np
import pytest

from fx_surface import (
    gk_delta,
    gk_digital,
    gk_forward,
    gk_gamma,
    gk_price,
    gk_rho_domestic,
    gk_rho_foreign,
    gk_theta,
    gk_vanna,
    gk_vega,
    gk_volga,
    implied_vol,
)

S, T, RD, RF, SIG = 1.10, 0.75, 0.045, 0.033, 0.085


@pytest.mark.parametrize("K", [0.95, 1.05, 1.10, 1.15, 1.30])
def test_put_call_parity(K):
    c = gk_price(S, K, T, RD, RF, SIG, +1)
    p = gk_price(S, K, T, RD, RF, SIG, -1)
    F = gk_forward(S, T, RD, RF)
    assert c - p == pytest.approx(math.exp(-RD * T) * (F - K), abs=1e-14)


def test_price_bounds_and_monotonicity_in_vol():
    K = 1.12
    lower = max(S * math.exp(-RF * T) - K * math.exp(-RD * T), 0.0)
    prices = [gk_price(S, K, T, RD, RF, s, 1) for s in (0.02, 0.05, 0.10, 0.30, 0.80)]
    assert all(lower < p < S * math.exp(-RF * T) for p in prices)
    assert all(a < b for a, b in zip(prices, prices[1:]))  # vega > 0


def test_convexity_in_strike():
    Ks = np.linspace(0.9, 1.35, 41)
    c = np.array([gk_price(S, k, T, RD, RF, SIG, 1) for k in Ks])
    assert np.all(np.diff(c, 2) > -1e-12)


@pytest.mark.parametrize("cp", [+1, -1])
@pytest.mark.parametrize("K,sig", [(0.98, 0.06), (1.10, 0.085), (1.25, 0.20), (1.5, 0.35)])
def test_implied_vol_round_trip(cp, K, sig):
    price = gk_price(S, K, T, RD, RF, sig, cp)
    assert implied_vol(price, S, K, T, RD, RF, cp) == pytest.approx(sig, abs=1e-10)


def test_implied_vol_violating_bounds():
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_vol(-0.01, S, 1.1, T, RD, RF, 1)
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_vol(S * math.exp(-RF * T) + 0.01, S, 1.1, T, RD, RF, 1)
    assert math.isnan(implied_vol(-0.01, S, 1.1, T, RD, RF, 1, on_fail="nan"))


def test_spot_vs_forward_delta_relation():
    for cp in (+1, -1):
        d_s = gk_delta(S, 1.12, T, RD, RF, SIG, cp, "spot")
        d_f = gk_delta(S, 1.12, T, RD, RF, SIG, cp, "forward")
        assert d_s == pytest.approx(math.exp(-RF * T) * d_f, abs=1e-15)


def test_premium_adjustment_identity():
    """delta_pa(spot) == delta(spot) - V/S (premium in base ccy)."""
    for cp in (+1, -1):
        for K in (1.02, 1.10, 1.2):
            v = gk_price(S, K, T, RD, RF, SIG, cp)
            d = gk_delta(S, K, T, RD, RF, SIG, cp, "spot")
            d_pa = gk_delta(S, K, T, RD, RF, SIG, cp, "spot_pa")
            assert d_pa == pytest.approx(d - v / S, abs=1e-14)


def _fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def test_greeks_vs_finite_differences():
    K = 1.13
    h = 1e-5
    assert gk_delta(S, K, T, RD, RF, SIG, 1, "spot") == pytest.approx(
        _fd(lambda s: gk_price(s, K, T, RD, RF, SIG, 1), S, h), abs=1e-7
    )
    assert gk_vega(S, K, T, RD, RF, SIG) == pytest.approx(
        _fd(lambda v: gk_price(S, K, T, RD, RF, v, 1), SIG, h), abs=1e-7
    )
    assert gk_gamma(S, K, T, RD, RF, SIG) == pytest.approx(
        (gk_price(S + h, K, T, RD, RF, SIG, 1) - 2 * gk_price(S, K, T, RD, RF, SIG, 1)
         + gk_price(S - h, K, T, RD, RF, SIG, 1)) / h**2, rel=1e-5
    )
    assert gk_vanna(S, K, T, RD, RF, SIG) == pytest.approx(
        _fd(lambda s: gk_vega(s, K, T, RD, RF, SIG), S, h), rel=1e-6
    )
    assert gk_volga(S, K, T, RD, RF, SIG) == pytest.approx(
        _fd(lambda v: gk_vega(S, K, T, RD, RF, v), SIG, h), rel=1e-6
    )
    assert gk_rho_domestic(S, K, T, RD, RF, SIG, 1) == pytest.approx(
        _fd(lambda r: gk_price(S, K, T, r, RF, SIG, 1), RD, h), abs=1e-8
    )
    assert gk_rho_foreign(S, K, T, RD, RF, SIG, 1) == pytest.approx(
        _fd(lambda r: gk_price(S, K, T, RD, r, SIG, 1), RF, h), abs=1e-8
    )
    assert gk_theta(S, K, T, RD, RF, SIG, 1) == pytest.approx(
        -_fd(lambda t: gk_price(S, K, t, RD, RF, SIG, 1), T, h), abs=1e-7
    )


def test_digital_equals_strike_derivative():
    K = 1.08
    h = 1e-6
    fd = -(gk_price(S, K + h, T, RD, RF, SIG, 1) - gk_price(S, K - h, T, RD, RF, SIG, 1)) / (2 * h)
    # flat vol: -dC/dK = e^{-rdT} N(d2) exactly
    assert gk_digital(S, K, T, RD, RF, SIG, 1) == pytest.approx(fd, abs=1e-8)
    # call + put digitals = discounted 1
    total = gk_digital(S, K, T, RD, RF, SIG, 1) + gk_digital(S, K, T, RD, RF, SIG, -1)
    assert total == pytest.approx(math.exp(-RD * T), abs=1e-15)


def test_negative_rates_both_legs():
    rd, rf = -0.005, -0.0075
    F = gk_forward(S, T, rd, rf)
    assert F > S  # rd > rf still lifts the forward
    c = gk_price(S, 1.1, T, rd, rf, SIG, 1)
    p = gk_price(S, 1.1, T, rd, rf, SIG, -1)
    assert c - p == pytest.approx(math.exp(-rd * T) * (F - 1.1), abs=1e-14)
    assert implied_vol(c, S, 1.1, T, rd, rf, 1) == pytest.approx(SIG, abs=1e-10)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="spot"):
        gk_price(-1.0, 1.1, T, RD, RF, SIG, 1)
    with pytest.raises(ValueError, match="strike"):
        gk_price(S, 0.0, T, RD, RF, SIG, 1)
    with pytest.raises(ValueError, match="expiry"):
        gk_price(S, 1.1, 0.0, RD, RF, SIG, 1)
    with pytest.raises(ValueError, match="volatility"):
        gk_price(S, 1.1, T, RD, RF, -0.1, 1)
    with pytest.raises(ValueError, match="cp"):
        gk_price(S, 1.1, T, RD, RF, SIG, 0)
    with pytest.raises(ValueError, match="convention"):
        gk_delta(S, 1.1, T, RD, RF, SIG, 1, "delta50")
