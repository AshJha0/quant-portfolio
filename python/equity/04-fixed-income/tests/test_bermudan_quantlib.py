"""Bermudan swaption against QuantLib 1.44 ``TreeSwaptionEngine`` (required CI job).

Same Hull-White model, same trinomial construction. QuantLib's dates use
``SimpleDayCounter`` and ``NullCalendar`` so its times are whole years and
equal ours; the index has no fixing lag, so the float leg is ``1 - P`` on the
single curve, as in ``bermudan``. Two checks: with the same number of tree
steps the two prices agree to 1e-12 (same lattice), and our default grid is
within 0.5 bp of notional of QuantLib at 1000 steps (plan W15 acceptance).
"""

from __future__ import annotations

import numpy as np
import pytest
from fi_rates.bermudan import bermudan_swaption
from fi_rates.curve import DiscountCurve

ql = pytest.importorskip("QuantLib")

CASES = [  # flat rate, a, sigma, strike, payer
    (0.03, 0.03, 0.01, 0.03, True),
    (0.03, 0.03, 0.01, 0.035, False),
    (0.045, 0.1, 0.015, 0.04, True),
    (-0.005, 0.5, 0.008, 0.0, False),
]


def _ql_price(r, a, sigma, K, payer, steps):
    d0 = ql.Date(15, 1, 2026)
    ql.Settings.instance().evaluationDate = d0
    dc = ql.SimpleDayCounter()
    yc = ql.YieldTermStructureHandle(ql.FlatForward(d0, r, dc, ql.Continuous))
    idx = ql.IborIndex("X1Y", ql.Period(1, ql.Years), 0, ql.USDCurrency(), ql.NullCalendar(),
                       ql.Unadjusted, False, dc, yc)
    sch = ql.Schedule(ql.Date(15, 1, 2027), ql.Date(15, 1, 2031), ql.Period(1, ql.Years),
                      ql.NullCalendar(), ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
    kind = ql.Swap.Payer if payer else ql.Swap.Receiver
    swap = ql.VanillaSwap(kind, 1.0, sch, K, dc, sch, idx, 0.0, dc)
    opt = ql.Swaption(swap, ql.BermudanExercise([ql.Date(15, 1, 2027 + i) for i in range(4)]))
    opt.setPricingEngine(ql.TreeSwaptionEngine(ql.HullWhite(yc, a, sigma), steps))
    return opt.NPV()


def _ours(r, a, sigma, K, payer, spy):
    t = np.arange(1, 81) * 0.25
    curve = DiscountCurve(t, np.exp(-r * t))
    return bermudan_swaption(a, sigma, curve, [1.0, 2.0, 3.0, 4.0, 5.0], [1.0] * 4, K,
                             [1.0, 2.0, 3.0, 4.0], payer, steps_per_year=spy)


@pytest.mark.parametrize(("r", "a", "sigma", "K", "payer"), CASES)
def test_same_lattice_same_price(r, a, sigma, K, payer):
    assert _ours(r, a, sigma, K, payer, 40) == pytest.approx(_ql_price(r, a, sigma, K, payer, 200),
                                                            abs=1e-12)


@pytest.mark.parametrize(("r", "a", "sigma", "K", "payer"), CASES)
def test_default_grid_within_half_bp_of_converged_quantlib(r, a, sigma, K, payer):
    assert abs(_ours(r, a, sigma, K, payer, 48) - _ql_price(r, a, sigma, K, payer, 1000)) < 0.5e-4
