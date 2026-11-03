"""qp_core.holidays: business-day helpers and the one-off closure hook.

The rule-by-rule calendar checks stay in the FX options tests
(test_fx_strangle_holidays.py), which import through the re-export.
"""

from __future__ import annotations

import datetime as dt

import pytest
from qp_core import holidays as H

D = dt.date


@pytest.fixture(autouse=True)
def _no_leftover_closures():
    H.clear_closures()
    yield
    H.clear_closures()


def test_fx_reexport_is_the_same_object():
    fx = pytest.importorskip("fx_options.holidays")
    assert fx.holidays is H.holidays and fx.pair_holidays is H.pair_holidays


def test_is_business_day_weekend_and_holiday():
    assert not H.is_business_day(D(2026, 11, 7))            # Saturday
    assert H.is_business_day(D(2026, 11, 11))               # weekends only
    assert not H.is_business_day(D(2026, 11, 11), "USD")    # Veterans Day
    assert H.is_business_day(D(2026, 11, 11), "EUR")


def test_joint_calendar_forms_agree():
    d = D(2026, 4, 6)  # Easter Monday: TARGET and London closed, New York open
    assert H.is_business_day(d, "USD")
    for cal in ("EURUSD", "EUR+USD", "eur/usd", ("EUR", "USD")):
        assert not H.is_business_day(d, cal)


def test_bad_calendar_codes():
    with pytest.raises(ValueError):
        H.is_business_day(D(2026, 1, 5), "XYZ")
    with pytest.raises(ValueError):
        H.is_business_day(D(2026, 1, 5), "EURU")
    with pytest.raises(ValueError):
        H.is_business_day(D(2026, 1, 5), ())
    with pytest.raises(ValueError):
        H.is_business_day(D(1999, 1, 5), "USD")  # outside MIN_YEAR


def test_adjust_following_crosses_holiday_run():
    # GBP Christmas 2027: Sat 25, Sun 26, substitutes Mon 27 and Tue 28.
    assert H.adjust_following(D(2027, 12, 25), "GBP") == D(2027, 12, 29)
    assert H.adjust_following(D(2027, 12, 25)) == D(2027, 12, 27)
    assert H.adjust_following(D(2027, 12, 29), "GBP") == D(2027, 12, 29)  # already open


def test_add_business_days():
    assert H.add_business_days(D(2026, 11, 6), 3) == D(2026, 11, 11)
    assert H.add_business_days(D(2026, 11, 6), 3, "USD") == D(2026, 11, 12)
    assert H.add_business_days(D(2026, 11, 7), 0, "USD") == D(2026, 11, 7)
    with pytest.raises(ValueError):
        H.add_business_days(D(2026, 11, 6), -1)


def test_add_closure_and_clear():
    d = D(2031, 5, 6)
    assert d not in H.holidays("GBP", 2031)
    H.add_closure("gbp", d)
    assert d in H.holidays("GBP", 2031)
    assert d not in H.holidays("GBP", 2030) and d not in H.holidays("EUR", 2031)
    assert not H.is_business_day(d, "GBP")
    assert d in H.pair_holidays("EURGBP", 2031, 2031)
    H.clear_closures("GBP")
    assert d not in H.holidays("GBP", 2031)


def test_add_closure_on_existing_holiday_is_noop():
    before = H.holidays("USD", 2026)
    H.add_closure("USD", D(2026, 11, 11))
    assert H.holidays("USD", 2026) == before


def test_add_closure_rejects_bad_input():
    with pytest.raises(ValueError):
        H.add_closure("GBP", D(2031, 5, 3))           # Saturday
    with pytest.raises(ValueError):
        H.add_closure("GBP", D(2100, 5, 4))           # past MAX_YEAR
    with pytest.raises(ValueError):
        H.add_closure("XXX", D(2031, 5, 6))
    with pytest.raises(TypeError):
        H.add_closure("GBP", dt.datetime(2031, 5, 6))  # type: ignore[arg-type]


def test_extra_closures_restores_even_on_error():
    keep = D(2030, 3, 5)
    H.add_closure("EUR", keep)
    tmp = D(2030, 3, 6)
    with pytest.raises(RuntimeError), H.extra_closures({"EUR": [tmp], "JPY": [tmp]}):
        assert tmp in H.holidays("EUR", 2030) and tmp in H.holidays("JPY", 2030)
        raise RuntimeError
    assert tmp not in H.holidays("EUR", 2030) and tmp not in H.holidays("JPY", 2030)
    assert keep in H.holidays("EUR", 2030)  # pre-existing closure survives


def test_rules_cache_not_polluted_by_closures():
    d = D(2032, 6, 8)
    with H.extra_closures({"CHF": [d]}):
        assert d in H.holidays("CHF", 2032)
    assert H.holidays("CHF", 2032) is H._rule_holidays("CHF", 2032)


def test_year_validation_rejects_bool():
    with pytest.raises(ValueError):
        H.holidays("USD", True)  # type: ignore[arg-type]
