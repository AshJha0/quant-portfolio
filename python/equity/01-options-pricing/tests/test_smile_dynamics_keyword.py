"""W14b: the ``smile_dynamics=`` keyword on the existing equity option risk calls.

These calls price with one flat vol per option, so the smile has no slope
and ``d sigma / dS = 0`` under every dynamic: the default (sticky strike)
must reproduce the numbers from before the keyword existed, and the other
two dynamics must give the same numbers (the keyword records the assumption
and rejects typos). Smile-consistent Greeks are ``qp_core.risk.smile_delta``.
"""

from __future__ import annotations

import pytest
from eq_options.black_scholes import bs_price
from eq_options.greeks import bs_greeks, fd_greeks
from eq_options.risk_grids import scenario_grid, spot_ladder, vega_ladder

DYNAMICS = ["sticky_strike", "sticky_moneyness", "sticky_delta"]
BOOK = [(10.0, 95.0, 0.5, "call"), (-5.0, 105.0, 1.5, "put")]


def test_default_preserves_the_numbers_and_every_dynamic_agrees_on_a_flat_smile():
    base = bs_greeks(100.0, 95.0, 0.5, 0.03, 0.2, 0.01, "put")
    # Pinned from before the keyword existed (same kernel, same inputs).
    assert base.delta == pytest.approx(-0.3055570144421483, rel=1e-14)
    for d in DYNAMICS:
        assert bs_greeks(100.0, 95.0, 0.5, 0.03, 0.2, 0.01, "put", smile_dynamics=d) == base
        assert fd_greeks(bs_price, 100.0, 95.0, 0.5, 0.03, 0.2, smile_dynamics=d) == fd_greeks(
            bs_price, 100.0, 95.0, 0.5, 0.03, 0.2)
        a, b = spot_ladder(BOOK, 100.0, 0.03, 0.2, smile_dynamics=d), spot_ladder(BOOK, 100.0, 0.03, 0.2)
        assert all((a[k] == b[k]).all() for k in a)
        assert vega_ladder(BOOK, 100.0, 0.03, 0.2, smile_dynamics=d) == vega_ladder(BOOK, 100.0, 0.03, 0.2)
        assert (scenario_grid(BOOK, 100.0, 0.03, 0.2, smile_dynamics=d)
                == scenario_grid(BOOK, 100.0, 0.03, 0.2)).all()


@pytest.mark.parametrize(
    "call",
    [
        lambda d: bs_greeks(100.0, 95.0, 0.5, 0.03, 0.2, smile_dynamics=d),
        lambda d: fd_greeks(bs_price, 100.0, 95.0, 0.5, 0.03, 0.2, smile_dynamics=d),
        lambda d: spot_ladder(BOOK, 100.0, 0.03, 0.2, smile_dynamics=d),
        lambda d: vega_ladder(BOOK, 100.0, 0.03, 0.2, smile_dynamics=d),
        lambda d: scenario_grid(BOOK, 100.0, 0.03, 0.2, smile_dynamics=d),
    ],
)
def test_unknown_dynamic_is_refused(call):
    with pytest.raises(ValueError, match="smile_dynamics"):
        call("sticky_vol")
