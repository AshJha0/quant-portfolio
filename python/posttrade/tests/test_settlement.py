"""The 40-case settlement-date table (W29 acceptance). Each expected date was
worked by hand from the 2026 NYSE, Federal Reserve, TARGET2 and UK calendars;
the comment says which rule the case exercises."""

from __future__ import annotations

from datetime import date

import pytest
from qp_posttrade.settlement import fx_spot_date, us_equity_settle

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


def test_year_end_runs_off_table_raises() -> None:
    with pytest.raises(ValueError, match="2027"):
        us_equity_settle(D(2026, 12, 31))


@pytest.mark.parametrize("bad", [D(2026, 11, 7), D(2026, 11, 26), D(2026, 4, 3)])
def test_non_trading_day_raises(bad: date) -> None:
    with pytest.raises(ValueError):
        us_equity_settle(bad)


def test_unknown_currency_raises() -> None:
    with pytest.raises(ValueError, match="JPY"):
        fx_spot_date("USDJPY", D(2026, 11, 2))
