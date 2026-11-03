"""The 40-case settlement-date table (W29 acceptance). Each expected date was
worked by hand from the 2026 NYSE, Federal Reserve, TARGET2 and UK calendars;
the comment says which rule the case exercises."""

from __future__ import annotations

from datetime import date

import pytest
from qp_posttrade.settlement import fx_spot_date, nyse_holidays, us_equity_settle

D = date
EQ = [
    (D(2026, 11, 2), D(2026, 11, 3), "plain T+1"),
    (D(2026, 11, 6), D(2026, 11, 9), "over weekend"),
    (D(2026, 11, 10), D(2026, 11, 12), "Veterans Day: NYSE open, Fed shut"),
    (D(2026, 10, 9), D(2026, 10, 13), "weekend then Columbus Day"),
    (D(2026, 11, 25), D(2026, 11, 27), "Thanksgiving"),
    (D(2026, 12, 24), D(2026, 12, 28), "Christmas Friday"),
    (D(2026, 7, 2), D(2026, 7, 6), "NYSE observes 3 July"),
    (D(2026, 4, 2), D(2026, 4, 6), "Good Friday NYSE"),
    (D(2026, 1, 16), D(2026, 1, 20), "MLK"),
    (D(2026, 2, 13), D(2026, 2, 17), "Presidents Day"),
    (D(2026, 5, 22), D(2026, 5, 26), "Memorial Day"),
    (D(2026, 6, 18), D(2026, 6, 22), "Juneteenth Friday"),
    (D(2026, 9, 4), D(2026, 9, 8), "Labor Day"),
    (D(2026, 12, 30), D(2026, 12, 31), "year end"),
    (D(2026, 1, 2), D(2026, 1, 5), "first Friday"),
    (D(2026, 10, 12), D(2026, 10, 13), "trade on Columbus Day"),
    (D(2026, 11, 11), D(2026, 11, 12), "trade on Veterans Day"),
]
FX = [
    ("EURUSD", D(2026, 11, 2), D(2026, 11, 4), "plain T+2"),
    ("EURUSD", D(2026, 11, 5), D(2026, 11, 9), "over weekend"),
    ("EURUSD", D(2026, 11, 6), D(2026, 11, 10), "Friday"),
    ("EURUSD", D(2026, 11, 9), D(2026, 11, 12), "USD holiday on spot rolls"),
    ("EURUSD", D(2026, 10, 9), D(2026, 10, 13), "USD holiday on T+1 still counts"),
    ("EURUSD", D(2026, 10, 8), D(2026, 10, 13), "USD holiday on spot rolls"),
    ("EURUSD", D(2026, 4, 1), D(2026, 4, 7), "TARGET Easter"),
    ("EURUSD", D(2026, 4, 30), D(2026, 5, 5), "TARGET 1 May"),
    ("EURUSD", D(2026, 12, 23), D(2026, 12, 28), "TARGET Christmas"),
    ("EURUSD", D(2026, 11, 24), D(2026, 11, 27), "Thanksgiving on spot"),
    ("EURUSD", D(2026, 7, 3), D(2026, 7, 7), "Fed open on 3 July"),
    ("EURUSD", D(2026, 7, 2), D(2026, 7, 6), "no Fed holiday 3 July"),
    ("GBPUSD", D(2026, 8, 27), D(2026, 9, 1), "UK August bank holiday"),
    ("GBPUSD", D(2026, 8, 28), D(2026, 9, 2), "UK holiday on T+1"),
    ("GBPUSD", D(2026, 12, 23), D(2026, 12, 29), "UK 28 Dec substitute"),
    ("GBPUSD", D(2026, 9, 3), D(2026, 9, 8), "Labor Day on spot"),
    ("GBPUSD", D(2026, 4, 30), D(2026, 5, 5), "UK early May"),
    ("EURGBP", D(2026, 4, 30), D(2026, 5, 6), "both crosses' holidays"),
    ("EURGBP", D(2026, 11, 6), D(2026, 11, 10), "cross plain"),
    ("EURGBP", D(2026, 11, 9), D(2026, 11, 12), "cross avoids USD holiday"),
    ("EURGBP", D(2026, 5, 21), D(2026, 5, 26), "UK spring holiday"),
    ("GBPUSD", D(2026, 12, 18), D(2026, 12, 22), "December plain"),
]


def test_table_has_40_cases() -> None:
    assert len(EQ) + len(FX) + 1 == 40  # +1 for the out-of-table case below


@pytest.mark.parametrize(("td", "want", "why"), EQ)
def test_us_equity(td: date, want: date, why: str) -> None:
    assert us_equity_settle(td) == want, why


@pytest.mark.parametrize(("pair", "td", "want", "why"), FX)
def test_fx_spot(pair: str, td: date, want: date, why: str) -> None:
    assert fx_spot_date(pair, td) == want, why


def test_year_end_rolls_into_next_year() -> None:
    # The calendars are rules-based (qp_core.holidays and nyse_holidays), so a
    # trade on 31 Dec 2026 settles over New Year's Day (Fri 1 Jan 2027).
    assert us_equity_settle(D(2026, 12, 31)) == D(2027, 1, 4)


def test_outside_the_rules_range_raises() -> None:
    with pytest.raises(ValueError, match="2100"):
        us_equity_settle(D(2099, 12, 31))
    with pytest.raises(ValueError, match="1999"):
        fx_spot_date("EURUSD", D(1999, 12, 30))


# The 2026 tables the 40 cases were worked from by hand (NYSE and Fed
# published calendars). The rule generators must reproduce them exactly.
NYSE_2026 = {D(2026, m, d) for m, d in [(1, 1), (1, 19), (2, 16), (4, 3), (5, 25), (6, 19), (7, 3),
                                         (9, 7), (11, 26), (12, 25)]}
FED_2026 = {D(2026, m, d) for m, d in [(1, 1), (1, 19), (2, 16), (5, 25), (6, 19), (9, 7), (10, 12),
                                        (11, 11), (11, 26), (12, 25)]}


def test_nyse_rules_reproduce_the_2026_table() -> None:
    assert set(nyse_holidays(2026)) == NYSE_2026


def test_core_fed_calendar_reproduces_the_2026_table() -> None:
    from qp_core.holidays import holidays

    assert set(holidays("USD", 2026)) == FED_2026


@pytest.mark.parametrize(("year", "day"), [
    (2022, D(2021, 12, 31)),  # Sat New Year: NYSE does not close Fri 31 Dec
    (2021, D(2021, 12, 24)),  # Sat Christmas observed Friday
    (2023, D(2023, 1, 2)),    # Sun New Year observed Monday
    (2012, D(2012, 10, 29)),  # Hurricane Sandy
])
def test_nyse_observance_rules(year: int, day: date) -> None:
    if year == 2022:
        assert day not in nyse_holidays(2021) and D(2022, 1, 1) not in nyse_holidays(2022)
    else:
        assert day in nyse_holidays(year)


def test_usdcad_is_t_plus_1() -> None:
    # The old 2026-only tables had no CAD calendar; qp_core ships one.
    assert fx_spot_date("USDCAD", D(2026, 11, 2)) == D(2026, 11, 3)


@pytest.mark.parametrize("bad", [D(2026, 11, 7), D(2026, 11, 26), D(2026, 4, 3)])
def test_non_trading_day_raises(bad: date) -> None:
    with pytest.raises(ValueError):
        us_equity_settle(bad)


def test_unknown_currency_raises() -> None:
    # qp_core counts weekends only for a currency it has no calendar for;
    # post-trade refuses instead of returning a possibly wrong value date.
    with pytest.raises(ValueError, match="MXN"):
        fx_spot_date("USDMXN", D(2026, 11, 2))
