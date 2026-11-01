"""Business-day FX value dates, ON/TN before spot, money-market day counts."""

import datetime as dt

import pytest

from fx_rates.daycount import (
    MONEY_MARKET_DAYCOUNT,
    forward_settlement_date,
    money_market_year_fraction,
    spot_date,
    tenor_dates,
)

D = dt.date


class TestSpot:
    def test_thursday_trade_settles_monday(self):
        assert spot_date(D(2025, 6, 5), pair=("EUR", "USD")) == D(2025, 6, 9)

    def test_non_usd_holiday_extends_spot(self):
        # 1 May is a TARGET holiday: Wed 30-Apr-2025 trade, T+1 = Fri 2 May, spot Mon 5 May
        assert spot_date(D(2025, 4, 30), pair=("EUR", "USD")) == D(2025, 5, 5)

    def test_usd_holiday_on_spot_rolls_forward(self):
        # Wed 2-Jul-2025 +2 EUR days = Fri 4 Jul, a USD holiday -> Mon 7 Jul
        assert spot_date(D(2025, 7, 2), pair=("EUR", "USD")) == D(2025, 7, 7)

    def test_usd_holiday_on_t_plus_1_does_not_count_for_eurusd(self):
        # Thu 3-Jul-2025: T+1 is 4 Jul (USD holiday, EUR open) -> spot Mon 7 Jul
        assert spot_date(D(2025, 7, 3), pair=("EUR", "USD")) == D(2025, 7, 7)

    def test_usdcad_t_plus_1(self):
        assert spot_date(D(2025, 6, 6), 1, pair=("USD", "CAD")) == D(2025, 6, 9)


class TestTenors:
    def test_on_and_tn_settle_before_spot(self):
        trade = D(2025, 6, 5)  # Thursday
        on, tn = tenor_dates(trade, "ON", ("EUR", "USD")), tenor_dates(trade, "TN", ("EUR", "USD"))
        spot = spot_date(trade, pair=("EUR", "USD"))
        assert on == (D(2025, 6, 5), D(2025, 6, 6))
        assert tn == (D(2025, 6, 6), spot)
        assert on[1] < spot

    def test_sn_starts_at_spot(self):
        assert tenor_dates(D(2025, 6, 5), "SN") == (D(2025, 6, 9), D(2025, 6, 10))

    def test_month_tenor_modified_following(self):
        # spot Mon 30-Jun-2025 is the last business day of June -> end-end:
        # 1M = last business day of July (Thu 31 Jul)
        assert forward_settlement_date(D(2025, 6, 26), "1M") == D(2025, 7, 31)

    def test_month_tenor_does_not_cross_month_end(self):
        # spot Thu 29-May-2025 (not month-end bd: 30 May is Friday)
        # +3M = 29 Aug 2025 (Friday) -> stays
        assert forward_settlement_date(D(2025, 5, 27), "3M") == D(2025, 8, 29)
        # spot Mon 28-Apr-2025 + 1M = 28 May (Wed)
        assert forward_settlement_date(D(2025, 4, 24), "1M") == D(2025, 5, 28)

    def test_weekend_end_date_rolls(self):
        # spot Fri 31-Jan-2025 (last bd of Jan) +1M -> last bd Feb = Fri 28 Feb
        assert forward_settlement_date(D(2025, 1, 29), "1M") == D(2025, 2, 28)

    @pytest.mark.parametrize("bad", ["", "M", "XM", "0M", "3Q"])
    def test_bad_tenor(self, bad):
        with pytest.raises(ValueError):
            tenor_dates(D(2025, 6, 5), bad)


class TestMoneyMarket:
    def test_usd_eur_jpy_act360_gbp_act365(self):
        a, b = D(2025, 1, 2), D(2025, 4, 2)  # 90 days
        assert money_market_year_fraction(a, b, "USD") == 90 / 360
        assert money_market_year_fraction(a, b, "eur") == 90 / 360
        assert money_market_year_fraction(a, b, "GBP") == 90 / 365
        assert MONEY_MARKET_DAYCOUNT["JPY"] == "ACT/360"

    def test_unknown_currency(self):
        with pytest.raises(ValueError, match="money-market"):
            money_market_year_fraction(D(2025, 1, 2), D(2025, 4, 2), "XYZ")
