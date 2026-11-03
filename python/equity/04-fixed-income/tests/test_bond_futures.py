"""Bond futures (plan W18): conversion factors against 20 published values,
basis and implied repo against an independent implementation, CTD and the
delivery option.

Published conversion factors (20):
* CME Group, "Calculating U.S. Treasury Futures Conversion Factors"
  (IR232), examples 1-5 (TU, 3-year, FV, TY, US).
* CME TY factors for eight deliverables 2002-2003, as printed in the
  MathWorks ``convfactor`` documentation (Financial Instruments Toolbox,
  example 1).
* Hull, *Options, Futures, and Other Derivatives* (11th ed., s. 6.2): the
  10% bond with 20y 2m left (1.4623) and the 8% bond with 18y 4m (1.2199).
* ICE Long Gilt June 2000 basket (notional 7%), four gilts, factors as
  printed in the rateslib ``BondFuture`` documentation.
* Eurex: 4% Bund 4 Jan 2018 into the March 2009 contract (0.8659 to 4 dp,
  MathWorks example 2).

Implied repo and net basis: the same Long Gilt basket at 16 Mar 2000,
futures 112.98, delivery 30 Jun 2000, ACT/365, repo 6.24% -- rateslib
prints the values to 16 digits, an implementation written independently of
this one.
"""

from __future__ import annotations

from datetime import date as D

import pytest
from fi_rates.bond import FixedRateBond
from fi_rates.bond_futures import (
    Deliverable,
    cheapest_to_deliver,
    cme_cf_kernel,
    cme_conversion_factor,
    delivery_option,
    delivery_option_kernel,
    gross_basis,
    implied_repo,
    net_basis,
    yield_conversion_factor,
)

CME_DOC = [  # coupon, maturity, delivery month, contract, published CF
    (0.015, D(2010, 10, 31), D(2008, 12, 1), "TU", 0.9229),
    (0.01125, D(2012, 1, 15), D(2009, 3, 1), "Z3N", 0.8747),
    (0.0275, D(2013, 10, 31), D(2008, 12, 1), "FV", 0.8653),
    (0.0375, D(2018, 11, 15), D(2008, 12, 1), "TY", 0.8357),
    (0.045, D(2038, 5, 15), D(2008, 12, 1), "US", 0.7943),
]
CME_TY_TABLE = [  # delivery month, maturity, coupon, CF
    (D(2002, 12, 1), D(2012, 11, 15), 0.04, 0.8539),
    (D(2003, 3, 1), D(2012, 8, 15), 0.04375, 0.8858),
    (D(2003, 6, 1), D(2012, 2, 15), 0.04875, 0.9259),
    (D(2003, 9, 1), D(2011, 2, 15), 0.05, 0.9418),
    (D(2003, 12, 1), D(2011, 8, 15), 0.05, 0.9403),
    (D(2003, 9, 1), D(2010, 8, 15), 0.0575, 0.9862),
    (D(2002, 12, 1), D(2009, 8, 15), 0.06, 1.0000),
    (D(2003, 6, 1), D(2010, 2, 15), 0.065, 1.0266),
]
GILTS = [  # coupon, maturity, clean price 16 Mar 2000, ICE CF (7 dp)
    (0.0575, D(2009, 12, 7), 102.732, 0.9142255),
    (0.09, D(2011, 7, 12), 131.461, 1.1525705),
    (0.0625, D(2010, 11, 25), 107.877, 0.9449312),
    (0.09, D(2012, 8, 6), 134.455, 1.1619558),
]
RATESLIB_CF = [0.9142254519590043, 1.152570524483184, 0.9449311626312816, 1.1619558330528528]
RATESLIB_REPO = [7.381345196635331, 3.5646853481343777, 2.1997545597624004, -1.4146703289488534]
RATESLIB_NET = [-0.3436542561467064, 1.0336684511125043, 1.275866135927373, 3.0103709249974315]
RATESLIB_GROSS = [-0.5571915623283132, 1.2435821438898813, 1.1186772459177945, 3.177229981688697]


def _gilt(c: float, m: D) -> FixedRateBond:
    return FixedRateBond(D(1995, m.month, m.day), m, c, 2, "ACT/ACT-ICMA", eom=False)


@pytest.mark.parametrize(("c", "m", "dm", "k", "want"), CME_DOC)
def test_cme_brochure_examples(c, m, dm, k, want):
    assert cme_conversion_factor(c, m, dm, k) == want


@pytest.mark.parametrize(("dm", "m", "c", "want"), CME_TY_TABLE)
def test_cme_ty_table(dm, m, c, want):
    assert cme_conversion_factor(c, m, dm, "TY") == want


def test_hull_examples():
    assert round(cme_cf_kernel(0.10, 20, 2, True), 4) == 1.4623
    assert round(cme_cf_kernel(0.08, 18, 4, True), 4) == 1.2199


@pytest.mark.parametrize(("c", "m", "price", "want"), GILTS)
def test_long_gilt_factors(c, m, price, want):
    assert yield_conversion_factor(_gilt(c, m), D(2000, 6, 1), 0.07, 7, 7, "GBP") == want


def test_ex_dividend_is_what_fixes_the_first_gilt():
    b = _gilt(0.0575, D(2009, 12, 7))
    assert yield_conversion_factor(b, D(2000, 6, 1), 0.07, 7) == 0.914193
    assert yield_conversion_factor(b, D(2000, 6, 1), 0.07, 7, 7, "GBP") == 0.9142255


def test_eurex_bund():
    b = FixedRateBond(D(2008, 1, 4), D(2018, 1, 4), 0.04, 1, "ACT/ACT-ICMA", eom=False)
    cf = yield_conversion_factor(b, D(2009, 3, 10), 0.06, 6)
    assert round(cf, 4) == 0.8659


def test_cme_coupon_rounding_and_quarter_rule():
    # 3.07% rounds to 3 1/8%; 4.0625% (a tie) rounds up to 4 1/8%
    assert cme_conversion_factor(0.0307, D(2018, 11, 15), D(2008, 12, 1)) == \
        cme_conversion_factor(0.03125, D(2018, 11, 15), D(2008, 12, 1))
    assert cme_conversion_factor(0.040625, D(2018, 11, 15), D(2008, 12, 1)) == \
        cme_conversion_factor(0.04125, D(2018, 11, 15), D(2008, 12, 1))
    # TY truncates z=11 to 9; FV keeps it
    assert cme_cf_kernel(0.04, 9, 11, True) == cme_cf_kernel(0.04, 9, 9, True)
    assert cme_cf_kernel(0.04, 4, 11, False) != cme_cf_kernel(0.04, 4, 10, False)
    # a 6% bond on a whole half-year is exactly par; off it, the linear accrual in b
    # leaves 1.03**0.5 - 0.015 = 0.99989 (CME rounds to 0.9999)
    assert cme_cf_kernel(0.06, 9, 0, True) == pytest.approx(1.0, abs=1e-15)
    assert round(cme_cf_kernel(0.06, 9, 3, True), 4) == 0.9999
    with pytest.raises(ValueError):
        cme_conversion_factor(0.04, D(2008, 11, 15), D(2008, 12, 1))
    with pytest.raises(ValueError):
        cme_conversion_factor(0.04, D(2018, 11, 15), D(2008, 12, 1), "XX")


@pytest.mark.parametrize("i", range(4))
def test_gilt_basis_repo_against_rateslib(i):
    c, m, p, _ = GILTS[i]
    b, cf = _gilt(c, m), RATESLIB_CF[i]
    s, d = D(2000, 3, 16), D(2000, 6, 30)
    assert gross_basis(p, 112.98, cf) == pytest.approx(RATESLIB_GROSS[i], abs=1e-12)
    assert implied_repo(b, p, s, d, 112.98, cf, 365.0) * 100 == pytest.approx(RATESLIB_REPO[i], abs=1e-10)
    assert net_basis(b, p, s, d, 112.98, cf, 0.0624, 365.0) == pytest.approx(RATESLIB_NET[i], abs=1e-10)


def test_net_basis_is_zero_at_the_implied_repo():
    c, m, p, _ = GILTS[2]
    b, s, d = _gilt(c, m), D(2000, 3, 16), D(2000, 6, 30)
    r = implied_repo(b, p, s, d, 112.98, RATESLIB_CF[2], 365.0)
    assert net_basis(b, p, s, d, 112.98, RATESLIB_CF[2], r, 365.0) == pytest.approx(0.0, abs=1e-12)


def test_ctd_rateslib_order():
    basket = [Deliverable(_gilt(c, m), p, cf) for (c, m, p, _), cf in zip(GILTS, RATESLIB_CF, strict=True)]
    order = cheapest_to_deliver(basket, D(2000, 3, 16), D(2000, 6, 30), 112.98, 365.0)
    assert [i for i, _ in order] == [0, 1, 2, 3]


def test_hull_cheapest_to_deliver_by_cost():
    # Hull s. 6.2: futures 93.25; cost = quoted price - F * CF
    cost = [gross_basis(p, 93.25, cf) for p, cf in [(99.50, 1.0382), (143.50, 1.5188), (119.75, 1.2615)]]
    assert [round(x, 2) for x in cost] == [2.69, 1.87, 2.12]
    assert min(range(3), key=cost.__getitem__) == 1


def _treasury(c: float, m: D) -> FixedRateBond:
    return FixedRateBond(D(2015, m.month, 15), m, c, 2, "ACT/ACT-ICMA")


def test_delivery_option_properties():
    s, d = D(2026, 11, 3), D(2026, 12, 31)
    dm = D(2026, 12, 1)
    raw = [(0.0275, D(2033, 8, 15), 96.0), (0.04, D(2034, 2, 15), 102.5), (0.045, D(2035, 11, 15), 104.0)]
    basket = [Deliverable(_treasury(c, m), p, cme_conversion_factor(c, m, dm)) for c, m, p in raw]
    zero = delivery_option(basket, s, d, 0.04, 0.0)
    assert zero["option_value"] == 0.0
    vals = [delivery_option(basket, s, d, 0.04, v)["option_value"] for v in (0.005, 0.01, 0.02)]
    assert 0.0 < vals[0] < vals[1] < vals[2]
    one = delivery_option(basket[:1], s, d, 0.04, 0.02)
    assert one["option_value"] == pytest.approx(0.0, abs=1e-12)
    # The fair futures price sits below today's forward of the CTD.
    assert delivery_option(basket, s, d, 0.04, 0.01)["fair_futures"] < zero["fair_futures"]


def test_delivery_option_kernel_two_identical_bonds_has_no_option():
    e, f = [[1.0, 2.0]], [[3.0, 103.0]]
    _fair, opt = delivery_option_kernel(e * 2, f * 2, [0.0, 0.0], [2, 2], [0.05, 0.05], [1.0, 1.0], 0.01)
    assert opt == pytest.approx(0.0, abs=1e-13)
    with pytest.raises(ValueError):
        delivery_option_kernel(e, f, [0.0], [2], [0.05], [1.0], -1.0)
