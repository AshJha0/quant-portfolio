"""W14b: ``smile_dynamics=`` on ``fx_var.book.Book`` risk calls.

``Market`` carries one implied vol per pair (no smile), so spot shocks leave
the vol unchanged under every dynamic and all three give identical P&L; the
default (sticky strike) keeps the numbers from before the keyword.
"""

from __future__ import annotations

import numpy as np
import pytest
from fx_var.book import Book, Market, Option, Spot


def _setup():
    market = Market(spot_usd={"EUR": 1.10, "GBP": 1.27}, rates={"USD": 0.04, "EUR": 0.02, "GBP": 0.035},
                    vols={"EURUSD": 0.08, "GBPUSD": 0.09})
    book = Book([Spot("EURUSD", 1e6), Option("EURUSD", 2e6, 1.12, 0.5, "call"),
                 Option("GBPUSD", -1e6, 1.25, 1.0, "put")])
    shocks = {"FX:EUR": np.array([-0.02, 0.0, 0.015]), "VOL:EURUSD": np.array([0.01, 0.0, -0.005])}
    return market, book, shocks


@pytest.mark.parametrize("d", ["sticky_strike", "sticky_moneyness", "sticky_delta"])
def test_default_preserves_numbers_and_dynamics_agree_on_flat_vols(d):
    market, book, shocks = _setup()
    assert book.value_usd(market, smile_dynamics=d) == book.value_usd(market)
    np.testing.assert_array_equal(book.pnl(market, shocks, smile_dynamics=d), book.pnl(market, shocks))
    assert book.linear_exposures(market, smile_dynamics=d).equals(book.linear_exposures(market))


def test_unknown_dynamic_is_refused():
    market, book, shocks = _setup()
    for call in (lambda: book.value_usd(market, smile_dynamics="x"),
                 lambda: book.pnl(market, shocks, smile_dynamics="x"),
                 lambda: book.linear_exposures(market, smile_dynamics="x")):
        with pytest.raises(ValueError, match="smile_dynamics"):
            call()
