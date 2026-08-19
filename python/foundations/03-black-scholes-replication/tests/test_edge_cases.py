"""Edge cases per CONVENTIONS.md item 6: T->0, sigma->0, deep ITM/OTM,
zero/negative rates, and invalid-input handling.

Two of these ("what happens at T=0" and "what happens at sigma=0") are
NOT literal limits the closed form evaluates -- ``_d1_d2`` divides by
``sigma*sqrt(T)``, so T=0 or sigma=0 raise ``ValueError`` rather than
silently returning NaN or an intrinsic value. That is itself a
documented, tested design choice (see ``docs/VALIDATION.md``): the
economically correct behaviour at those boundaries is intrinsic value,
but this module does not special-case it, so callers must take the
limit explicitly. The tests below check (a) that approaching the limit
numerically converges to the expected intrinsic value, and (b) that the
exact boundary raises.
"""
import math

import pytest

from eq_bs_replication import (
    call_greeks,
    call_price,
    implied_volatility,
    put_greeks,
    put_price,
)

S, K, r, sigma, T = 100.0, 105.0, 0.03, 0.25, 0.75


# ---------------------------------------------------------------------
# T -> 0 (approaching expiry)
# ---------------------------------------------------------------------

def test_call_price_approaches_intrinsic_as_t_shrinks():
    tiny_T = 1e-6
    c_itm = call_price(120.0, 100.0, r, sigma, tiny_T)
    assert abs(c_itm - (120.0 - 100.0 * math.exp(-r * tiny_T))) < 1e-4
    c_otm = call_price(80.0, 100.0, r, sigma, tiny_T)
    assert c_otm < 1e-6


def test_put_price_approaches_intrinsic_as_t_shrinks():
    tiny_T = 1e-6
    p_itm = put_price(80.0, 100.0, r, sigma, tiny_T)
    assert abs(p_itm - (100.0 * math.exp(-r * tiny_T) - 80.0)) < 1e-4
    p_otm = put_price(120.0, 100.0, r, sigma, tiny_T)
    assert p_otm < 1e-6


def test_zero_or_negative_T_raises():
    with pytest.raises(ValueError):
        call_price(S, K, r, sigma, 0.0)
    with pytest.raises(ValueError):
        put_price(S, K, r, sigma, -0.1)


# ---------------------------------------------------------------------
# sigma -> 0 (deterministic underlying)
# ---------------------------------------------------------------------

def test_call_price_approaches_forward_intrinsic_as_vol_shrinks():
    # Deep ITM with negligible vol -> discounted forward intrinsic value.
    c = call_price(200.0, 100.0, r, 1e-6, T)
    assert abs(c - (200.0 - 100.0 * math.exp(-r * T))) < 1e-6


def test_put_price_approaches_zero_when_otm_and_vol_shrinks():
    p = put_price(200.0, 100.0, r, 1e-6, T)
    assert p < 1e-6


def test_zero_or_negative_sigma_raises():
    with pytest.raises(ValueError):
        call_price(S, K, r, 0.0, T)
    with pytest.raises(ValueError):
        put_price(S, K, r, -0.2, T)


# ---------------------------------------------------------------------
# Deep ITM / deep OTM at extreme moneyness
# ---------------------------------------------------------------------

def test_deep_otm_call_is_worthless():
    assert call_price(10.0, 100.0, r, 0.2, T) < 1e-8


def test_deep_itm_call_has_delta_near_one():
    g = call_greeks(1000.0, 100.0, r, sigma, T)
    assert g.delta > 0.999999


def test_deep_otm_call_has_delta_near_zero():
    g = call_greeks(1.0, 1000.0, r, sigma, T)
    assert g.delta < 1e-6


def test_deep_itm_put_has_delta_near_minus_one():
    g = put_greeks(1.0, 1000.0, r, sigma, T)
    assert g.delta < -0.999999


def test_deep_otm_put_has_delta_near_zero():
    g = put_greeks(1000.0, 100.0, r, sigma, T)
    assert g.delta > -1e-6


def test_deep_itm_low_vol_round_trip_is_imprecise():
    # Known, documented limitation (docs/VALIDATION.md numerical-limits
    # section): deep ITM combined with low vol makes vega tiny, so the
    # Newton step (diff / vega) is ill-conditioned and the round trip
    # loses precision -- here to ~1e-2 rather than the ~1e-8 typical
    # elsewhere. This is expected numerical behaviour, not a bug: the
    # price in this corner is close to intrinsic value and simply does
    # not carry much information about sigma.
    price = call_price(100.0, 60.0, r, 0.1, T)
    iv = implied_volatility(price, 100.0, 60.0, r, T)
    assert abs(iv - 0.1) < 2e-2


def test_implied_vol_refuses_deep_itm_information_free_quote():
    # At/near intrinsic, vega is ~0 and the price carries no vol
    # information beyond "somewhere below no-arbitrage bound"; a price
    # strictly outside the bounds is refused rather than inverted.
    S_deep, K_deep = 500.0, 10.0
    intrinsic = S_deep - K_deep * math.exp(-r * T)
    with pytest.raises(ValueError):
        implied_volatility(intrinsic - 1.0, S_deep, K_deep, r, T)


# ---------------------------------------------------------------------
# Zero and negative interest rates
# ---------------------------------------------------------------------

def test_zero_rate_is_supported():
    c = call_price(S, K, 0.0, sigma, T)
    p = put_price(S, K, 0.0, sigma, T)
    assert abs((c - p) - (S - K)) < 1e-10
    assert c > 0


def test_negative_rate_is_supported():
    # The formula has no positivity constraint on r: exp(-rT) is well
    # defined and put-call parity still holds exactly.
    c = call_price(S, K, -0.01, sigma, T)
    p = put_price(S, K, -0.01, sigma, T)
    assert abs((c - p) - (S - K * math.exp(0.01 * T))) < 1e-10
    assert c > 0 and p > 0


def test_negative_rate_implied_vol_round_trips():
    price = call_price(S, K, -0.01, 0.3, T)
    iv = implied_volatility(price, S, K, -0.01, T)
    assert abs(iv - 0.3) < 1e-6


# ---------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------

def test_negative_spot_raises():
    with pytest.raises(ValueError):
        call_price(-1.0, K, r, sigma, T)


def test_negative_strike_raises():
    with pytest.raises(ValueError):
        call_price(S, -1.0, r, sigma, T)


def test_error_message_is_informative():
    with pytest.raises(ValueError, match="sigma and T must be strictly positive"):
        call_price(S, K, r, 0.0, T)
