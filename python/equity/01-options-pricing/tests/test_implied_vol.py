"""Implied-vol round trips across moneyness/expiry and arbitrage-bound behaviour."""

import math

import pytest

from eq_options import bs_price, implied_vol

MONEYNESS = [0.5, 0.8, 1.0, 1.25, 2.0]  # K / S
EXPIRIES = [1.0 / 52.0, 0.25, 1.0, 5.0]  # 1 week to 5 years


@pytest.mark.parametrize("moneyness", MONEYNESS)
@pytest.mark.parametrize("T", EXPIRIES)
@pytest.mark.parametrize("otype", ["call", "put"])
def test_round_trip_sigma_price_sigma(moneyness: float, T: float, otype: str) -> None:
    """sigma -> price -> sigma to 1e-8 across moneyness 0.5-2.0, T 1w-5y.

    The test vol is floored so the strike stays within ~3 standard
    deviations of the forward: further out the option's *time value*
    underflows double precision and no solver can recover vol (this
    regime is itself covered by test_price_at_intrinsic_raises).
    """
    S, r, q = 100.0, 0.03, 0.01
    K = S * moneyness
    sigma = max(0.22, abs(math.log(K / S)) / (3.0 * math.sqrt(T)))
    price = bs_price(S, K, T, r, sigma, q, otype)
    recovered = implied_vol(price, S, K, T, r, q, otype)
    assert recovered == pytest.approx(sigma, abs=1e-8)


@pytest.mark.parametrize("sigma", [0.01, 0.08, 0.5, 1.5, 3.0])
def test_round_trip_extreme_vols(sigma: float) -> None:
    """ATM-forward strike keeps |d| small even at 1% vol."""
    S, T, r, q = 100.0, 0.5, 0.02, 0.0
    K = S * math.exp((r - q) * T)
    price = bs_price(S, K, T, r, sigma, q, "call")
    assert implied_vol(price, S, K, T, r, q, "call") == pytest.approx(sigma, abs=1e-8)


def test_round_trip_deep_itm_low_vega() -> None:
    """Deep ITM (K = 0.35 S) short-dated: vega is tiny — Newton must fall
    back to bisection/Brent gracefully."""
    S, K, T, r, q, sigma = 100.0, 35.0, 0.1, 0.03, 0.0, 1.2
    price = bs_price(S, K, T, r, sigma, q, "call")
    assert implied_vol(price, S, K, T, r, q, "call") == pytest.approx(sigma, abs=1e-6)


def test_round_trip_deep_otm_short_dated() -> None:
    S, K, T, r, q, sigma = 100.0, 150.0, 1.0 / 52.0, 0.01, 0.0, 0.8
    price = bs_price(S, K, T, r, sigma, q, "call")
    assert implied_vol(price, S, K, T, r, q, "call") == pytest.approx(sigma, abs=1e-8)


def test_round_trip_negative_rates() -> None:
    S, K, T, r, q, sigma = 100.0, 95.0, 2.0, -0.015, 0.02, 0.25
    price = bs_price(S, K, T, r, sigma, q, "put")
    assert implied_vol(price, S, K, T, r, q, "put") == pytest.approx(sigma, abs=1e-8)


def test_price_at_intrinsic_raises() -> None:
    """Price at or below the sigma->0 bound has no implied vol."""
    S, K, T, r, q = 100.0, 80.0, 1.0, 0.05, 0.0
    lower = bs_price(S, K, T, r, 0.0, q, "call")  # discounted fwd intrinsic
    with pytest.raises(ValueError, match="arbitrage bound"):
        implied_vol(lower, S, K, T, r, q, "call")
    with pytest.raises(ValueError, match="arbitrage bound"):
        implied_vol(lower * 0.9, S, K, T, r, q, "call")


def test_price_above_upper_bound_raises() -> None:
    with pytest.raises(ValueError, match="sigma->inf"):
        implied_vol(101.0, 100.0, 100.0, 1.0, 0.05, 0.0, "call")


def test_zero_or_negative_price_raises() -> None:
    with pytest.raises(ValueError):
        implied_vol(0.0, 100.0, 120.0, 1.0, 0.05, 0.0, "call")
    with pytest.raises(ValueError):
        implied_vol(-1.0, 100.0, 120.0, 1.0, 0.05, 0.0, "call")


def test_expired_option_raises() -> None:
    with pytest.raises(ValueError, match="T > 0"):
        implied_vol(5.0, 100.0, 95.0, 0.0, 0.05, 0.0, "call")


def test_put_upper_bound_raises() -> None:
    K, T, r = 100.0, 1.0, 0.03
    upper = K * math.exp(-r * T)
    with pytest.raises(ValueError):
        implied_vol(upper + 0.01, 100.0, K, T, r, 0.0, "put")


def test_round_trip_long_dated_high_vol_flat_vega() -> None:
    """S=K, T=25y, sigma=300%: |d1| ~ 7.7, so vega ~ exp(-d1^2/2) underflows
    towards zero and the price sits within double-precision noise of the
    sigma->inf arbitrage bound (K exp(-rT) for the put). This is the
    solver's hardest legitimate corner: a price-residual-only stopping rule
    can declare convergence while sigma is still off by whole vol points,
    because the tiny price residual maps through a near-zero vega to a
    large sigma residual. The solver must fall through to a bracket-based
    (Brent) refinement rather than trusting the price tolerance alone.
    """
    S, K, T, r, q, sigma = 100.0, 100.0, 25.0, 0.10, 0.0, 3.0
    price = bs_price(S, K, T, r, sigma, q, "put")
    recovered = implied_vol(price, S, K, T, r, q, "put")
    assert recovered == pytest.approx(sigma, abs=2e-4)
