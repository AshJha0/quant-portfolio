"""Carry: hand-computed accruals, forward points, sign conventions, identities."""

import numpy as np
import pandas as pd
import pytest

from fx_pairs.carry import (
    carry_accrual,
    carry_adjusted_log_price,
    carry_ledger,
    daily_roll_yield,
    day_count_fractions,
    forward_outright,
    swap_points,
)


@pytest.fixture
def bdays():
    return pd.bdate_range("2020-01-06", periods=10)  # Mon..Fri, Mon..Fri


class TestDayCount:
    def test_weekday_gap_is_one_day(self, bdays):
        dt = day_count_fractions(bdays)
        assert dt[0] == 0.0
        assert dt[1] == pytest.approx(1.0 / 365.0, abs=1e-15)

    def test_weekend_gap_is_three_days(self, bdays):
        """Fri->Mon accrues 3/365 — our simplification of the real-world
        Wednesday 3-day swap (T+2 settlement shifts it; total is identical)."""
        dt = day_count_fractions(bdays)
        # index 5 is the Monday after the first Friday
        assert dt[5] == pytest.approx(3.0 / 365.0, abs=1e-15)

    def test_total_accrued_days_match_calendar(self, bdays):
        dt = day_count_fractions(bdays)
        total_days = (bdays[-1] - bdays[0]).days
        assert dt.sum() == pytest.approx(total_days / 365.0, abs=1e-15)


class TestForwardPoints:
    def test_forward_outright_hand_computed(self):
        # S=1.20, r_base=3%, r_quote=1%, tau=0.25y
        F = forward_outright(1.20, 0.03, 0.01, 0.25)
        assert F == pytest.approx(1.20 * (1 + 0.01 * 0.25) / (1 + 0.03 * 0.25),
                                  abs=1e-15)

    def test_high_yield_base_trades_at_discount(self):
        assert forward_outright(1.20, 0.05, 0.01, 0.5) < 1.20
        assert swap_points(1.20, 0.05, 0.01, 0.5) < 0.0
        # and at a premium when the base yields less
        assert swap_points(1.20, 0.00, 0.02, 0.5) > 0.0

    def test_daily_roll_yield_exact_vs_linear(self):
        tau = 1.0 / 365.0
        exact = daily_roll_yield(0.08, 0.01, tau, method="swap")
        linear = daily_roll_yield(0.08, 0.01, tau, method="linear")
        assert exact == pytest.approx((0.08 - 0.01) * tau / (1 + 0.08 * tau),
                                      abs=1e-18)
        assert linear == pytest.approx((0.08 - 0.01) * tau, abs=1e-18)
        assert abs(exact - linear) < 5e-8  # O(tau^2) agreement

    def test_bad_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            daily_roll_yield(0.05, 0.01, 1 / 365, method="magic")


class TestAccrual:
    def test_accrual_matches_hand_computed_exactly(self, bdays):
        """Linear accrual on a known differential, checked day by day."""
        accr = carry_accrual(0.05, 0.01, bdays, method="linear")
        dt = day_count_fractions(bdays)
        expected = 0.04 * dt
        assert np.allclose(accr, expected, atol=1e-18, rtol=0)

    def test_accrual_uses_lagged_rates(self, bdays):
        """The accrual over (t-1, t] must use rates known at t-1 (no lookahead)."""
        rb = pd.Series(0.05, index=bdays)
        rb.iloc[-1] = 99.0  # a rate spike on the last day must not affect accruals
        accr = carry_accrual(rb, 0.01, bdays, method="linear")
        base = carry_accrual(0.05, 0.01, bdays, method="linear")
        assert np.allclose(accr, base, atol=1e-18)

    def test_long_high_yielder_earns_positive_carry(self, bdays):
        pos = pd.Series(1.0, index=bdays)
        led = carry_ledger(pos, r_base=0.08, r_quote=0.01)
        assert led.iloc[1:].min() > 0.0
        assert led.iloc[0] == 0.0

    def test_short_high_yielder_pays_carry(self, bdays):
        pos = pd.Series(-1.0, index=bdays)
        led = carry_ledger(pos, r_base=0.08, r_quote=0.01)
        assert led.iloc[1:].max() < 0.0

    def test_zero_differential_zero_carry(self, bdays):
        pos = pd.Series(1.0, index=bdays)
        led = carry_ledger(pos, r_base=0.03, r_quote=0.03)
        assert np.allclose(led.values, 0.0, atol=1e-18)

    def test_ledger_scales_with_notional_and_position(self, bdays):
        pos = pd.Series(1.0, index=bdays)
        base = carry_ledger(pos, 0.05, 0.01, notional=1.0)
        double_not = carry_ledger(pos, 0.05, 0.01, notional=2.0)
        half_pos = carry_ledger(0.5 * pos, 0.05, 0.01, notional=1.0)
        assert np.allclose(double_not.values, 2 * base.values, atol=1e-18)
        assert np.allclose(half_pos.values, 0.5 * base.values, atol=1e-18)

    def test_gapped_index_total_accrual(self):
        """Missing days: the ledger accrues over the actual calendar gap."""
        idx = pd.DatetimeIndex(["2020-01-06", "2020-01-07", "2020-01-17"])
        pos = pd.Series(1.0, index=idx)
        led = carry_ledger(pos, 0.05, 0.01, method="linear")
        assert led.iloc[2] == pytest.approx(0.04 * 10 / 365.0, abs=1e-15)
        assert led.sum() == pytest.approx(0.04 * 11 / 365.0, abs=1e-15)


class TestCarryAdjustedPrice:
    def test_identity_vs_cumulative_accrual(self, bdays):
        rng = np.random.default_rng(0)
        price = pd.Series(1.2 * np.exp(np.cumsum(0.005 * rng.standard_normal(10))),
                          index=bdays)
        tr = carry_adjusted_log_price(price, 0.06, 0.01)
        accr = carry_accrual(0.06, 0.01, bdays)
        assert np.allclose(tr.values - np.log(price.values), np.cumsum(accr),
                           atol=1e-15)

    def test_requires_series(self):
        with pytest.raises(ValueError, match="Series"):
            carry_adjusted_log_price(np.ones(10), 0.05, 0.01)


class TestValidation:
    def test_rate_length_mismatch_raises(self, bdays):
        with pytest.raises(ValueError, match="length"):
            carry_accrual(np.ones(5) * 0.05, 0.01, bdays)

    def test_ledger_without_index_raises(self):
        with pytest.raises(ValueError, match="index"):
            carry_ledger(np.ones(10), 0.05, 0.01)
