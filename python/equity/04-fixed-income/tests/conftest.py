"""Shared fixtures: settlement date, bootstrapped curves, sample bonds."""

from __future__ import annotations

import datetime as dt

import pytest

import fi_rates as fr
from fi_rates.data import market_quotes, sample_portfolio

SETTLEMENT = dt.date(2026, 8, 18)


@pytest.fixture(scope="session")
def settlement() -> dt.date:
    return SETTLEMENT


@pytest.fixture(scope="session")
def quotes():
    """Noise-free upward-sloping deposit+swap quote set."""
    return market_quotes("upward", seed=42, noise_bp=0.0)


@pytest.fixture(scope="session")
def curve(quotes):
    """Default log-linear-DF bootstrapped curve."""
    return fr.bootstrap_curve(quotes, interpolation="loglinear_df")


@pytest.fixture(scope="session")
def flat_curve():
    """Flat 4% (continuous zero) curve on standard pillars."""
    times = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    return fr.DiscountCurve.from_zero_rates(times, [0.04] * len(times))


@pytest.fixture(scope="session")
def portfolio(settlement):
    return sample_portfolio(settlement, seed=42)


@pytest.fixture(scope="session")
def bond_5y(settlement) -> fr.FixedRateBond:
    """Semiannual 4% 5y bond, regular schedule, settlement on a coupon date."""
    return fr.FixedRateBond(
        effective=settlement,
        maturity=dt.date(2031, 8, 18),
        coupon=0.04,
        frequency=2,
        daycount="30/360US",
    )
