"""Day count hand-checks and settlement date logic."""

import datetime as dt

import pytest

from fx_rates.daycount import (
    VALID_CONVENTIONS,
    add_calendar_days,
    forward_settlement_date,
    spot_date,
    tenor_to_years,
    year_fraction,
)


class TestYearFraction:
    def test_act360_hand_check(self):
        # 2024-01-15 -> 2024-07-15 is 182 days
        assert year_fraction(
            dt.date(2024, 1, 15), dt.date(2024, 7, 15), "ACT/360"
        ) == pytest.approx(182 / 360, abs=1e-15)

    def test_act365f_hand_check(self):
        assert year_fraction(
            dt.date(2024, 1, 15), dt.date(2024, 7, 15), "ACT/365F"
        ) == pytest.approx(182 / 365, abs=1e-15)

    def test_act365f_leap_year_still_365(self):
        # ACT/365 Fixed always divides by 365, even across 29 Feb
        assert year_fraction(
            dt.date(2024, 1, 1), dt.date(2025, 1, 1), "ACT/365F"
        ) == pytest.approx(366 / 365, abs=1e-15)

    def test_30_360_full_year(self):
        assert year_fraction(
            dt.date(2024, 1, 15), dt.date(2025, 1, 15), "30/360"
        ) == pytest.approx(1.0, abs=1e-15)

    def test_30_360_month_ends(self):
        # Jan 31 -> Jul 31: d1=31->30, then d2=31->30 => exactly 6 months
        assert year_fraction(
            dt.date(2024, 1, 31), dt.date(2024, 7, 31), "30/360"
        ) == pytest.approx(0.5, abs=1e-15)

    def test_30_360_d2_31_d1_below_30(self):
        # d1=15 (<30) so d2=31 stays: (30*6 + 16)/360
        assert year_fraction(
            dt.date(2024, 1, 15), dt.date(2024, 7, 31), "30/360"
        ) == pytest.approx((30 * 6 + 16) / 360, abs=1e-15)

    def test_zero_when_same_date(self):
        d = dt.date(2024, 3, 1)
        for conv in VALID_CONVENTIONS:
            assert year_fraction(d, d, conv) == 0.0

    def test_reversed_dates_raise(self):
        with pytest.raises(ValueError, match="precedes"):
            year_fraction(dt.date(2024, 2, 1), dt.date(2024, 1, 1), "ACT/360")

    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError, match="Unknown day-count"):
            year_fraction(dt.date(2024, 1, 1), dt.date(2024, 2, 1), "ACT/ACT")


class TestSettlement:
    def test_spot_t_plus_2_calendar(self):
        assert spot_date(dt.date(2024, 5, 10)) == dt.date(2024, 5, 12)

    def test_spot_lag_zero(self):
        d = dt.date(2024, 5, 10)
        assert spot_date(d, 0) == d

    def test_negative_lag_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            spot_date(dt.date(2024, 5, 10), -1)

    def test_forward_settlement_after_spot(self):
        trade = dt.date(2024, 5, 10)
        settle = forward_settlement_date(trade, "3M")
        assert settle > spot_date(trade)
        # ~91 days after spot
        assert abs((settle - spot_date(trade)).days - 91) <= 1

    def test_add_calendar_days(self):
        assert add_calendar_days(dt.date(2024, 12, 30), 3) == dt.date(2025, 1, 2)


class TestTenorParsing:
    @pytest.mark.parametrize(
        "tenor,years",
        [("3M", 0.25), ("6M", 0.5), ("1Y", 1.0), ("5Y", 5.0),
         ("1W", 7 / 365), ("30D", 30 / 365), ("ON", 1 / 365)],
    )
    def test_known_tenors(self, tenor, years):
        assert tenor_to_years(tenor) == pytest.approx(years, abs=1e-15)

    @pytest.mark.parametrize("bad", ["", "M", "XX3", "3Q", "-3M", "0Y"])
    def test_bad_tenors_raise(self, bad):
        with pytest.raises(ValueError):
            tenor_to_years(bad)
