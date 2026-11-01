"""Regression tests for wrong numbers at the edges of the input domain."""

import math

import pytest

from eq_options import (
    black76_greeks,
    bs_price,
    crr_price,
    implied_vol,
    mc_price,
)


def test_sigma_zero_large_drift_is_finite_and_exact() -> None:
    """(r - q) T = 800 overflows the forward; the price must not be NaN.

    With q = 0 the zero-vol call is S - K e^{-rT}, which rounds to S.
    """
    assert bs_price(100.0, 100.0, 1000.0, 0.8, 0.0, 0.0, "call") == 100.0
    assert bs_price(100.0, 100.0, 1000.0, 0.8, 0.0, 0.0, "put") == 0.0


def test_overflow_raises_value_error_not_overflow_error() -> None:
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 1000.0, -0.8, 0.2, 0.0, "put")  # e^{-rT} overflows
    with pytest.raises(ValueError):
        implied_vol(5.0, 100.0, 100.0, 1000.0, -0.8, 0.0, "call")


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_rates_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 1.0, bad, 0.2, 0.0, "call")
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 1.0, 0.02, 0.2, bad, "call")
    with pytest.raises(ValueError):
        crr_price(100.0, 100.0, 1.0, bad, 0.2, 0.0, "put", "american", 50)


@pytest.mark.parametrize("bad_type", ["Put", "xyz", ""])
def test_black76_greeks_rejects_unknown_option_type(bad_type: str) -> None:
    with pytest.raises(ValueError):
        black76_greeks(100.0, 100.0, 1.0, 0.02, 0.2, bad_type)  # type: ignore[arg-type]


@pytest.mark.parametrize("r", [-0.02, -0.005])
def test_american_put_at_zero_spot_with_negative_rate_waits(r: float) -> None:
    """S = 0, r < 0: exercising now pays K, waiting pays K e^{-rT} > K."""
    K, T = 100.0, 1.0
    value = crr_price(0.0, K, T, r, 0.2, 0.0, "put", "american", 100)
    assert value == pytest.approx(K * math.exp(-r * T), rel=4e-16)
    assert value > K


def test_american_put_at_zero_spot_with_positive_rate_exercises() -> None:
    assert crr_price(0.0, 100.0, 1.0, 0.03, 0.2, 0.0, "put", "american", 100) == 100.0


def test_mc_needs_two_independent_samples() -> None:
    """Two antithetic paths are one pair: no standard error exists.

    With the control variate, two samples fit the regression exactly and
    leave a zero residual, so three are needed.
    """
    args = (100, 100, 1, 0.05, 0.2, 0.0, "call")
    with pytest.raises(ValueError):
        mc_price(*args, n_paths=2, antithetic=True, control_variate=False, seed=1)
    with pytest.raises(ValueError):
        mc_price(*args, n_paths=1, antithetic=False, control_variate=False, seed=1)
    with pytest.raises(ValueError):
        mc_price(*args, n_paths=4, antithetic=True, control_variate=True, seed=1)
    res = mc_price(*args, n_paths=3, antithetic=True, control_variate=False, seed=1)
    assert res.std_error > 0.0 and res.n_paths == 4
    res = mc_price(*args, n_paths=6, antithetic=True, control_variate=True, seed=1)
    assert res.std_error > 0.0


def test_mc_overflowing_drift_raises() -> None:
    with pytest.raises(ValueError):
        mc_price(100, 100, 1000.0, 0.8, 0.2, 0.0, "call", n_paths=1000, seed=1)
