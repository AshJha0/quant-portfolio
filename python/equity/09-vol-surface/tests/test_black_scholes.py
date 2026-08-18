"""Black-Scholes pricer and robust implied-vol solver."""

from __future__ import annotations

import numpy as np
import pytest

from eq_surface.black_scholes import (
    ImpliedVolWarning,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_vega,
    implied_vol,
    implied_vol_vector,
)

S, R, Q = 100.0, 0.02, 0.01


def test_known_value_atm_no_carry():
    # S=K=100, T=1, r=q=0, sigma=0.2:  C = S*(2*N(0.1) - 1) = 7.96556745540...
    c = bs_price(100, 100, 1.0, 0.0, 0.0, 0.2, "call")
    assert c == pytest.approx(7.965567455405804, abs=1e-12)


def test_put_call_parity():
    for K in [70.0, 100.0, 130.0]:
        for T in [0.1, 1.0, 3.0]:
            c = bs_price(S, K, T, R, Q, 0.25, "call")
            p = bs_price(S, K, T, R, Q, 0.25, "put")
            parity = S * np.exp(-Q * T) - K * np.exp(-R * T)
            assert c - p == pytest.approx(parity, abs=1e-10)


def test_zero_vol_returns_discounted_intrinsic():
    F = S * np.exp((R - Q) * 0.5)
    assert bs_price(S, 90, 0.5, R, Q, 0.0, "call") == pytest.approx(
        np.exp(-R * 0.5) * (F - 90), abs=1e-12
    )
    assert bs_price(S, 130, 0.5, R, Q, 0.0, "call") == 0.0


def test_expiry_returns_intrinsic():
    assert bs_price(S, 90, 0.0, R, Q, 0.3, "call") == 10.0
    assert bs_price(S, 130, 0.0, R, Q, 0.3, "put") == 30.0


def test_price_monotone_in_vol():
    sigmas = np.linspace(0.01, 2.0, 50)
    prices = [bs_price(S, 110, 0.5, R, Q, s, "call") for s in sigmas]
    assert np.all(np.diff(prices) > 0.0)


def test_vega_matches_finite_difference():
    h = 1e-6
    for K in [80.0, 100.0, 125.0]:
        fd = (bs_price(S, K, 0.7, R, Q, 0.25 + h) - bs_price(S, K, 0.7, R, Q, 0.25 - h)) / (2 * h)
        assert bs_vega(S, K, 0.7, R, Q, 0.25) == pytest.approx(fd, abs=1e-6)


def test_delta_gamma_match_finite_difference():
    h = 1e-4
    for K, kind in [(90.0, "call"), (110.0, "put")]:
        fd_d = (bs_price(S + h, K, 0.7, R, Q, 0.25, kind) - bs_price(S - h, K, 0.7, R, Q, 0.25, kind)) / (2 * h)
        assert bs_delta(S, K, 0.7, R, Q, 0.25, kind) == pytest.approx(fd_d, abs=1e-7)
    h = 1e-3  # larger bump for the second difference (roundoff control)
    fd_g = (
        bs_price(S + h, 100, 0.7, R, Q, 0.25) - 2 * bs_price(S, 100, 0.7, R, Q, 0.25) + bs_price(S - h, 100, 0.7, R, Q, 0.25)
    ) / (h * h)
    assert bs_gamma(S, 100, 0.7, R, Q, 0.25) == pytest.approx(fd_g, abs=1e-6)
    assert bs_gamma(S, 100, 0.7, R, Q, 0.25) > 0.0


def test_delta_call_put_relation():
    dc = bs_delta(S, 105, 0.5, R, Q, 0.2, "call")
    dp = bs_delta(S, 105, 0.5, R, Q, 0.2, "put")
    assert dc - dp == pytest.approx(np.exp(-Q * 0.5), abs=1e-12)


def test_implied_vol_round_trip_grid():
    """sigma -> price -> sigma to 1e-8 across strikes/expiries/vols, both kinds.

    Quotes with no measurable time value (numerically worthless wings or deep
    ITM at the zero-vol limit) must return nan -- that IS the documented
    behaviour, so the test asserts it rather than skipping.
    """
    import warnings as w

    from eq_surface.black_scholes import _price_bounds

    for kind in ("call", "put"):
        for K in [60.0, 85.0, 100.0, 120.0, 160.0]:
            for T in [0.05, 0.5, 2.0]:
                for sigma in [0.08, 0.2, 0.6, 1.2]:
                    price = bs_price(S, K, T, R, Q, sigma, kind)
                    lower, _ = _price_bounds(S, K, T, R, Q, kind)
                    with w.catch_warnings():
                        w.simplefilter("ignore", ImpliedVolWarning)
                        iv = implied_vol(price, S, K, T, R, Q, kind)
                    if price < 1e-9 or price - lower < 1e-9:
                        assert np.isnan(iv), (kind, K, T, sigma)  # no vol info
                    else:
                        assert iv == pytest.approx(sigma, abs=1e-8), (kind, K, T, sigma)


def test_implied_vol_negative_rates_round_trip():
    price = bs_price(S, 95, 1.0, -0.01, 0.0, 0.3, "call")
    assert implied_vol(price, S, 95, 1.0, -0.01, 0.0, "call") == pytest.approx(0.3, abs=1e-8)


def test_sub_intrinsic_price_rejected_with_warning():
    lower = S * np.exp(-Q * 0.5) - 90 * np.exp(-R * 0.5)
    with pytest.warns(ImpliedVolWarning, match="sub-intrinsic"):
        iv = implied_vol(lower - 0.01, S, 90, 0.5, R, Q, "call")
    assert np.isnan(iv)


def test_price_above_upper_bound_rejected():
    with pytest.warns(ImpliedVolWarning, match="upper bound"):
        iv = implied_vol(S * np.exp(-Q * 0.5) + 0.01, S, 90, 0.5, R, Q, "call")
    assert np.isnan(iv)


def test_deep_wing_zero_vega_returns_nan_not_garbage():
    """Deep ITM at exactly the zero-vol limit: vol unidentifiable -> nan."""
    lower = S * np.exp(-Q * 0.1) - 20 * np.exp(-R * 0.1)  # K=20, deep ITM
    with pytest.warns(ImpliedVolWarning):
        iv = implied_vol(lower, S, 20, 0.1, R, Q, "call")
    assert np.isnan(iv)


def test_deep_otm_tiny_price_nan_with_warning():
    with pytest.warns(ImpliedVolWarning):
        iv = implied_vol(1e-300, S, 300, 0.05, R, Q, "call")
    assert np.isnan(iv)


def test_t_zero_implied_vol_nan():
    with pytest.warns(ImpliedVolWarning, match="T=0"):
        assert np.isnan(implied_vol(5.0, S, 100, 0.0, R, Q))


def test_non_finite_price_nan():
    with pytest.warns(ImpliedVolWarning):
        assert np.isnan(implied_vol(np.nan, S, 100, 1.0, R, Q))


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="spot"):
        bs_price(-1.0, 100, 1.0, R, Q, 0.2)
    with pytest.raises(ValueError, match="strike"):
        bs_price(S, 0.0, 1.0, R, Q, 0.2)
    with pytest.raises(ValueError, match="non-negative"):
        bs_price(S, 100, -0.5, R, Q, 0.2)
    with pytest.raises(ValueError, match="volatility"):
        bs_price(S, 100, 1.0, R, Q, -0.2)
    with pytest.raises(ValueError, match="kind"):
        bs_price(S, 100, 1.0, R, Q, 0.2, "straddle")


def test_implied_vol_vector_handles_failures_silently():
    Ks = np.array([90.0, 100.0, 400.0])
    prices = np.array([bs_price(S, 90, 0.5, R, Q, 0.2), bs_price(S, 100, 0.5, R, Q, 0.2), 1e-300])
    ivs = implied_vol_vector(prices, S, Ks, 0.5, R, Q)
    assert ivs[0] == pytest.approx(0.2, abs=1e-8)
    assert ivs[1] == pytest.approx(0.2, abs=1e-8)
    assert np.isnan(ivs[2])


def test_implied_vol_vector_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        implied_vol_vector(np.array([1.0, 2.0]), S, np.array([100.0]), 0.5, R, Q)


def test_high_vol_round_trip():
    price = bs_price(S, 100, 1.0, R, Q, 4.0)
    assert implied_vol(price, S, 100, 1.0, R, Q) == pytest.approx(4.0, abs=1e-8)
