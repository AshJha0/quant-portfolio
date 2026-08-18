"""Day counts against hand-computed known-date examples; schedules."""

from __future__ import annotations

import datetime as dt

import pytest

from fi_rates.daycount import (
    add_months,
    adjust_modified_following,
    generate_schedule,
    year_fraction,
)

D = dt.date


class TestAct365F:
    def test_thirty_days(self):
        assert year_fraction(D(2024, 1, 1), D(2024, 1, 31), "ACT/365F") == 30 / 365

    def test_non_leap_year(self):
        assert year_fraction(D(2023, 1, 1), D(2024, 1, 1), "ACT/365F") == 1.0

    def test_leap_year_exceeds_one(self):
        assert year_fraction(D(2024, 1, 1), D(2025, 1, 1), "ACT/365F") == 366 / 365

    def test_across_feb_29(self):
        assert year_fraction(D(2024, 2, 28), D(2024, 3, 1), "ACT/365F") == 2 / 365

    def test_negative_when_reversed(self):
        assert year_fraction(D(2024, 1, 31), D(2024, 1, 1), "ACT/365F") == -30 / 365


class TestAct360:
    def test_half_year(self):
        # 2024-01-01 -> 2024-07-01 is 182 actual days (leap Feb)
        assert year_fraction(D(2024, 1, 1), D(2024, 7, 1), "ACT/360") == 182 / 360

    def test_three_month_deposit(self):
        # 2026-08-18 -> 2026-11-18: 13+30+31+18 = 92 days
        assert year_fraction(D(2026, 8, 18), D(2026, 11, 18), "ACT/360") == 92 / 360

    def test_full_year_exceeds_one(self):
        assert year_fraction(D(2023, 1, 1), D(2024, 1, 1), "ACT/360") == 365 / 360


class Test30360US:
    def test_regular_semiannual_is_exactly_half(self):
        assert year_fraction(D(2024, 1, 15), D(2024, 7, 15), "30/360US") == 0.5

    def test_month_end_31_to_31(self):
        # d1=31->30, then d2=31->30: exactly 6 * 30/360
        assert year_fraction(D(2024, 1, 31), D(2024, 7, 31), "30/360US") == 0.5

    def test_d1_30_d2_31(self):
        # d1=30 so d2=31->30: Jan 30 -> Mar 31 = 60/360
        assert year_fraction(D(2024, 1, 30), D(2024, 3, 31), "30/360US") == 60 / 360

    def test_d1_not_30_d2_31_keeps_31(self):
        # d1=15, d2=31 not adjusted: Jan 15 -> Jan 31 = 16/360
        assert year_fraction(D(2024, 1, 15), D(2024, 1, 31), "30/360US") == 16 / 360

    def test_feb_end_no_eom_rule(self):
        # No end-of-Feb rule (documented): Feb 28 -> Mar 1 = 3/360
        assert year_fraction(D(2023, 2, 28), D(2023, 3, 1), "30/360US") == 3 / 360

    def test_leap_feb_29(self):
        # 2024-02-29 -> 2024-03-31: d1=29, d2=31 kept: 30 + (31-29) = 32/360
        assert year_fraction(D(2024, 2, 29), D(2024, 3, 31), "30/360US") == 32 / 360


class TestActActISDA:
    def test_same_non_leap_year(self):
        # Mar 1 -> Sep 1 2023: 184 actual days / 365
        assert year_fraction(D(2023, 3, 1), D(2023, 9, 1), "ACT/ACT-ISDA") == 184 / 365

    def test_same_leap_year(self):
        assert (
            year_fraction(D(2024, 2, 28), D(2024, 3, 1), "ACT/ACT-ISDA") == 2 / 366
        )

    def test_year_boundary_split(self):
        # 1 day in 2023 (/365) + 1 day in 2024 (/366)
        got = year_fraction(D(2023, 12, 31), D(2024, 1, 2), "ACT/ACT-ISDA")
        assert got == pytest.approx(1 / 365 + 1 / 366, abs=1e-15)

    def test_exact_multi_year_anniversary(self):
        # Jun 15 2023 -> Jun 15 2026: 200/365 + 2 + 165/365 = exactly 3.0
        got = year_fraction(D(2023, 6, 15), D(2026, 6, 15), "ACT/ACT-ISDA")
        assert got == pytest.approx(3.0, abs=1e-15)

    def test_full_leap_year_is_one(self):
        got = year_fraction(D(2024, 1, 1), D(2025, 1, 1), "ACT/ACT-ISDA")
        assert got == pytest.approx(1.0, abs=1e-15)


class TestValidationAndUtils:
    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError, match="Unknown day count"):
            year_fraction(D(2024, 1, 1), D(2024, 2, 1), "ACT/252")

    def test_add_months_leap_clamp(self):
        assert add_months(D(2024, 1, 31), 1) == D(2024, 2, 29)

    def test_add_months_non_leap_clamp(self):
        assert add_months(D(2023, 1, 31), 1) == D(2023, 2, 28)

    def test_add_months_negative(self):
        assert add_months(D(2024, 3, 31), -1) == D(2024, 2, 29)

    def test_modified_following_rolls_forward(self):
        # Sat 2026-08-01 -> Mon 2026-08-03 (same month)
        assert adjust_modified_following(D(2026, 8, 1)) == D(2026, 8, 3)

    def test_modified_following_rolls_back_at_month_end(self):
        # Sun 2026-05-31 -> Mon would cross into June -> back to Fri 2026-05-29
        assert adjust_modified_following(D(2026, 5, 31)) == D(2026, 5, 29)

    def test_weekday_unchanged(self):
        assert adjust_modified_following(D(2026, 8, 18)) == D(2026, 8, 18)


class TestSchedule:
    def test_regular_semiannual(self):
        dates = generate_schedule(D(2024, 1, 15), D(2026, 1, 15), 2)
        assert dates == [
            D(2024, 7, 15),
            D(2025, 1, 15),
            D(2025, 7, 15),
            D(2026, 1, 15),
        ]

    def test_short_front_stub(self):
        dates = generate_schedule(D(2024, 2, 1), D(2026, 1, 15), 2)
        assert dates[0] == D(2024, 7, 15)  # short first period Feb 1 -> Jul 15
        assert dates[-1] == D(2026, 1, 15)
        assert len(dates) == 4

    def test_quarterly_count(self):
        dates = generate_schedule(D(2024, 1, 15), D(2027, 1, 15), 4)
        assert len(dates) == 12
        assert dates[0] == D(2024, 4, 15)

    def test_annual_month_end_clamp(self):
        dates = generate_schedule(D(2023, 2, 28), D(2026, 2, 28), 1)
        assert dates == [D(2024, 2, 28), D(2025, 2, 28), D(2026, 2, 28)]

    def test_adjusted_payment_dates_are_weekdays(self):
        dates = generate_schedule(D(2024, 1, 15), D(2030, 1, 15), 2, adjust=True)
        assert all(d.weekday() < 5 for d in dates)

    def test_invalid_frequency_raises(self):
        with pytest.raises(ValueError, match="frequency"):
            generate_schedule(D(2024, 1, 15), D(2026, 1, 15), 3)

    def test_maturity_before_effective_raises(self):
        with pytest.raises(ValueError, match="must be after"):
            generate_schedule(D(2026, 1, 15), D(2024, 1, 15), 2)
