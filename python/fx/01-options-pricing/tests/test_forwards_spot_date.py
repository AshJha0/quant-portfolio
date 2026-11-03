"""W2: continuous-rate CIP forwards measured from the spot date.

``cip_forward(S, T, r_d, r_f)`` treats the quoted spot as a t = 0 price.
A quoted spot rate is for delivery on the spot date ``t_s`` (T+2, T+1 for
USDCAD), so the carry runs over ``T - t_s``, not ``T`` (Clark 2011, ch. 1;
Wystup 2017, ch. 1):

    F = S * exp((r_d - r_f) * (T - t_s)).
"""

import datetime as dt
import math

import pytest
from fx_options.forwards import cip_forward, forward_from_dates, forward_points

D = dt.date
S, RD, RF = 1.0850, 0.045, 0.025


def test_t_spot_shortens_the_carry():
    ts = 4 / 365
    assert cip_forward(S, 1.0, RD, RF, t_spot=ts) == pytest.approx(
        S * math.exp((RD - RF) * (1.0 - ts)), rel=1e-14)
    assert cip_forward(S, ts, RD, RF, t_spot=ts) == pytest.approx(S, rel=1e-15)
    assert forward_points(S, 1.0, RD, RF, t_spot=ts) == pytest.approx(
        (S * math.exp((RD - RF) * (1.0 - ts)) - S) * 1e4, rel=1e-12)


def test_default_is_unchanged():
    assert cip_forward(S, 1.0, RD, RF) == pytest.approx(S * math.exp(RD - RF), rel=1e-15)


def test_forward_from_dates_uses_the_spot_date():
    # Thu 5 Jun 2025, EURUSD spot Mon 9 Jun; value date 9 Jun 2026 is one
    # year after spot. The old t = 0 reading carried over 369 days and gave
    # S * exp(0.02 * 369/365) = 1.107161 instead of 1.106918.
    f = forward_from_dates(S, D(2025, 6, 5), D(2026, 6, 9), RD, RF, "EURUSD")
    assert f == pytest.approx(S * math.exp((RD - RF) * 365 / 365), rel=1e-14)


def test_forward_from_dates_t_plus_1_pair():
    # USDCAD Fri 6 Jun 2025 settles Mon 9 Jun (T+1)
    f = forward_from_dates(1.37, D(2025, 6, 6), D(2025, 9, 9), 0.0275, 0.045, "USDCAD")
    assert f == pytest.approx(1.37 * math.exp((0.0275 - 0.045) * 92 / 365), rel=1e-14)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        cip_forward(S, 1.0, RD, RF, t_spot=-0.01)
    with pytest.raises(ValueError):
        forward_from_dates(S, D(2025, 6, 5), D(2025, 6, 4), RD, RF, "EURUSD")
