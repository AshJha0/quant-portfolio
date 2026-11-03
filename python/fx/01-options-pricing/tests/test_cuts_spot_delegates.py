"""``fx_options.cuts.spot_date`` uses the one FX spot rule in
:func:`qp_core.holidays.fx_spot_date` (Clark 2011, ch. 1; Wystup 2017, ch. 1).

Before round 7 ``cuts`` had its own rule: T+2 on a caller-supplied holiday
set (or the joint pair calendar with ``"auto"``), so a USD holiday on T+1
stopped the count, USDPHP was T+2, and with the default empty set no
holiday was seen at all.
"""

from __future__ import annotations

import datetime as dt

import fx_options
import pytest
from fx_options.cuts import expiry_and_delivery, spot_date, spot_lag
from qp_core.holidays import fx_spot_date

D = dt.date


@pytest.mark.parametrize("trade, pair, want", [
    # Thanksgiving Thu 26-Nov-2026 is T+1; only EUR days count, spot Fri 27.
    # The old "auto" calendar counted it as closed and gave Mon 30-Nov.
    (D(2026, 11, 25), "EURUSD", D(2026, 11, 27)),
    # Fri 4-Jul-2025 is T+2 for EURJPY; a cross cannot settle on a USD
    # holiday. The old default (weekends only) gave Fri 4-Jul.
    (D(2025, 7, 2), "EURJPY", D(2025, 7, 7)),
    # Good Friday 18-Apr-2025 (TARGET and London closed): old default gave 18-Apr.
    (D(2025, 4, 16), "GBPUSD", D(2025, 4, 22)),
    # USDPHP is T+1 (old table: T+2).
    (D(2025, 6, 17), "USDPHP", D(2025, 6, 18)),
])
def test_default_spot_date_is_the_shared_rule(trade, pair, want):
    assert spot_date(trade, pair) == want
    assert spot_date(trade, pair, "auto") == want
    assert want == fx_spot_date(trade, pair)


def test_spot_lag_is_the_shared_table():
    assert spot_lag("USDPHP") == 1 and spot_lag("PHPUSD") == 1
    assert spot_lag("USD/CAD") == 1 and spot_lag("EURUSD") == 2


def test_explicit_holiday_set_still_honoured():
    # A caller-supplied set keeps the old counting rule on that set only.
    assert spot_date(D(2026, 11, 25), "EURUSD", ()) == D(2026, 11, 27)
    assert spot_date(D(2026, 11, 25), "EURUSD", {D(2026, 11, 27)}) == D(2026, 11, 30)


def test_auto_delivery_uses_the_shared_spot_rule():
    # 1W from Wed 25-Nov-2026: expiry Wed 2-Dec, delivery its spot Fri 4-Dec.
    exp, dlv = expiry_and_delivery(D(2026, 11, 25), "1W", "EURUSD", "auto")
    assert exp == D(2026, 12, 2) and dlv == fx_spot_date(exp, "EURUSD")
    # 1M from Wed 25-Nov-2026: spot Fri 27-Nov (was Mon 30-Nov), delivery
    # Mon 28-Dec (27-Dec is a Sunday), expiry the latest day whose spot <= delivery.
    exp, dlv = expiry_and_delivery(D(2026, 11, 25), "1M", "EURUSD", "auto")
    assert dlv == D(2026, 12, 28)
    assert fx_spot_date(exp, "EURUSD") <= dlv < fx_spot_date(exp + dt.timedelta(days=1),
                                                             "EURUSD")


def test_forward_from_dates_exported():
    assert "forward_from_dates" in fx_options.__all__
    from fx_options.forwards import forward_from_dates
    assert fx_options.forward_from_dates is forward_from_dates
