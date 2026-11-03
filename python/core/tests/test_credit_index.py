"""Credit index and ISDA standard model (plan W18).

Outside oracle: the twenty Markit upfronts for trade date 21 May 2009 that
QuantLib carries in its ``testIsdaEngine`` (test-suite/creditdefaultswap.cpp,
"data comes from Markit published rates"). The USD discount curve those
values used (ISDA deposits and swaps of that day) is frozen below as the
knot times and zero rates of QuantLib 1.44's ``PiecewiseLogLinearDiscount``
on the same quotes; ``test_frozen_curve_matches_quantlib`` rebuilds it when
QuantLib is installed.
"""

from __future__ import annotations

import math
from datetime import date

import pytest
from qp_core.credit_index import (
    CDX_HY,
    CDX_IG,
    ITRAXX_MAIN,
    CdsContract,
    CreditIndex,
    FlatForwardCurve,
    HazardCurve,
    adjust_to_index,
    bootstrap_hazard,
    cds_legs,
    index_basis,
    index_maturity,
    intrinsic,
    par_spread,
    quoted_spread_from_upfront,
    roll_date,
    series_on,
    upfront_from_quoted_spread,
)

T0 = date(2009, 5, 21)
KNOT_T = [0.0958904109589041, 0.18356164383561643, 0.26301369863013696, 0.5150684931506849,
          0.7671232876712328, 1.010958904109589, 2.010958904109589, 3.0136986301369864,
          4.019178082191781, 5.016438356164383, 6.013698630136986, 7.016438356164383,
          8.016438356164384, 9.016438356164384, 10.021917808219179, 12.01917808219178,
          15.027397260273972, 20.024657534246575, 25.027397260273972, 30.03013698630137]
KNOT_Z = [0.0031233773559633366, 0.005451229621997696, 0.007083660822323275,
          0.012345173456788223, 0.013961730744337162, 0.01544604999250054,
          0.011806299150152936, 0.016921936067010365, 0.021223872585157812,
          0.02458221884142598, 0.02720157310738926, 0.02934788168356177,
          0.03099700759675116, 0.03231202673572397, 0.03348840999666881,
          0.03543566375028994, 0.037343756162235314, 0.03803042081938171,
          0.03815673485487269, 0.038564319399313964]
DISC09 = FlatForwardCurve(KNOT_T, KNOT_Z)

MARKIT = [  # upfront per 10mm, coupon 100bp; loops: maturity, quoted spread, recovery
    -97798.29358, -97776.11889, 914971.5977, 894985.6298,
    -186921.3594, -186839.8148, 1646623.672, 1579803.626,
    -274298.9203, -274122.4725, 2279730.93, 2147972.527,
    -592420.2297, -591571.2294, 3993550.206, 3545843.418,
    -797501.1422, -795915.9787, 4702034.688, 4042340.999,
]
CASES = [(m, s, r) for m in (date(2010, 6, 20), date(2011, 6, 20), date(2012, 6, 20),
                             date(2016, 6, 20), date(2019, 6, 20))
         for s in (0.001, 0.1) for r in (0.2, 0.4)]


@pytest.mark.parametrize(("case", "want"), list(zip(CASES, MARKIT, strict=True)))
def test_markit_published_upfronts(case, want):
    m, s, r = case
    got = upfront_from_quoted_spread(T0, m, s, 0.01, DISC09, r)["clean_upfront"] * 1e7
    # Markit prints 10 significant figures; 1e-3 is QuantLib's own tolerance.
    assert got == pytest.approx(want, abs=1e-3)


def test_frozen_curve_matches_quantlib():
    ql = pytest.importorskip("QuantLib")
    td = ql.Date(21, 5, 2009)
    ql.Settings.instance().evaluationDate = td
    hs = [ql.DepositRateHelper(q, ql.Period(t, ql.Months), 2, ql.WeekendsOnly(),
                               ql.ModifiedFollowing, False, ql.Actual360())
          for t, q in zip([1, 2, 3, 6, 9, 12],
                          [0.003081, 0.005525, 0.007163, 0.012413, 0.014, 0.015488], strict=True)]
    ibor = ql.IborIndex("IsdaIbor", ql.Period(3, ql.Months), 2, ql.USDCurrency(), ql.WeekendsOnly(),
                        ql.ModifiedFollowing, False, ql.Actual360())
    for t, q in zip([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30],
                    [0.011907, 0.01699, 0.021198, 0.02444, 0.026937, 0.028967, 0.030504,
                     0.031719, 0.03279, 0.034535, 0.036217, 0.036981, 0.037246, 0.037605],
                    strict=True):
        hs.append(ql.SwapRateHelper(q, ql.Period(t, ql.Years), ql.WeekendsOnly(), ql.Semiannual,
                                    ql.ModifiedFollowing, ql.Thirty360(ql.Thirty360.BondBasis), ibor))
    c = ql.PiecewiseLogLinearDiscount(0, ql.WeekendsOnly(), hs, ql.Actual365Fixed())
    for d, t, z in zip(list(c.dates())[1:], KNOT_T, KNOT_Z, strict=True):
        assert (date(d.year(), d.month(), d.dayOfMonth()) - T0).days / 365 == t
        assert -math.log(c.discount(d)) / t == pytest.approx(z, rel=1e-12)


def test_without_half_day_bias_the_markit_values_are_missed(monkeypatch):
    """The bias is what closes the 1000bp cases (150-440 per 10mm without it)."""
    import qp_core.credit_index.isda as isda

    m, s, r = CASES[6]
    monkeypatch.setattr(isda, "_BIAS", 0.0)
    off = upfront_from_quoted_spread(T0, m, s, 0.01, DISC09, r)["clean_upfront"] * 1e7
    assert abs(off - MARKIT[6]) > 100.0


@pytest.mark.parametrize("spread", [0.0005, 0.01, 0.035, 0.2])
def test_spread_upfront_round_trip(spread):
    m = date(2014, 6, 20)
    u = upfront_from_quoted_spread(T0, m, spread, 0.05, DISC09)["clean_upfront"]
    assert quoted_spread_from_upfront(T0, m, u, 0.05, DISC09) == pytest.approx(spread, rel=1e-10)


def test_bootstrap_reprices_par_spreads():
    mats = [date(2010, 6, 20), date(2012, 6, 20), date(2014, 6, 20), date(2019, 6, 20)]
    spr = [0.004, 0.007, 0.009, 0.012]
    hz = bootstrap_hazard(T0, mats, spr, DISC09, 0.4)
    for m, s in zip(mats, spr, strict=True):
        cds = CdsContract(T0, m, s, 0.4)
        assert cds_legs(cds, hz, DISC09)["clean_upfront"] == pytest.approx(0.0, abs=1e-13)
        assert par_spread(cds, hz, DISC09) == pytest.approx(s, rel=1e-11)


def test_contract_dates_follow_the_standard():
    cds = CdsContract(T0, date(2014, 6, 20), 0.01)
    assert cds.accrual_start == date(2009, 3, 20)
    assert cds.cash_settle == date(2009, 5, 26)  # Thu 21 May + 3 business days
    assert cds.accrued() == pytest.approx(0.01 * 63 / 360)
    assert cds.periods()[-1][1] == date(2014, 6, 21)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        CdsContract(T0, T0, 0.01)
    with pytest.raises(ValueError):
        CdsContract(T0, date(2014, 6, 20), 0.01, recovery=1.0)
    with pytest.raises(ValueError):
        HazardCurve((1.0, 0.5), (0.01, 0.01))
    with pytest.raises(ValueError):
        HazardCurve((1.0,), (-0.01,))
    with pytest.raises(ValueError):
        FlatForwardCurve((1.0,), (math.nan,))
    with pytest.raises(ValueError):
        bootstrap_hazard(T0, [date(2014, 6, 20)], [0.0], DISC09)


# ------------------------------------------------------------ series/roll
def test_series_anchors():
    assert series_on(CDX_IG, date(2019, 9, 20)) == 33
    assert series_on(CDX_IG, date(2019, 9, 19)) == 32
    assert series_on(CDX_IG, date(2020, 3, 20)) == 34
    assert series_on(ITRAXX_MAIN, date(2025, 9, 22)) == 44
    assert series_on(ITRAXX_MAIN, date(2026, 11, 3)) == 46
    assert roll_date(CDX_IG, 34) == date(2020, 3, 20)
    assert roll_date(CDX_IG, 32) == date(2019, 3, 20)
    assert index_maturity(CDX_IG, 33) == date(2024, 12, 20)
    assert index_maturity(CDX_IG, 34) == date(2025, 6, 20)
    assert index_maturity(ITRAXX_MAIN, 40) == date(2028, 12, 20)
    assert index_maturity(ITRAXX_MAIN, 40, 10) == date(2033, 12, 20)
    for s in range(20, 60):
        assert series_on(CDX_IG, roll_date(CDX_IG, s)) == s


# ------------------------------------------------------- index mechanics
NAMES = tuple(f"N{i:03d}" for i in range(125))


def _index(**kw) -> CreditIndex:
    return CreditIndex(CDX_IG, 12, date(2014, 6, 20), NAMES, **kw)


def test_factor_and_defaults():
    ix = _index()
    assert ix.factor == 1.0 and ix.version == 1
    d = ix.with_default("N007", 0.3)
    assert d.factor == pytest.approx(0.992, abs=1e-15)
    assert d.version == 2
    assert d.default_settlement() == pytest.approx(0.008 * 0.7, abs=1e-15)
    u0 = ix.upfront_from_spread(T0, 0.02, DISC09)
    assert d.upfront_from_spread(T0, 0.02, DISC09) == pytest.approx(0.992 * u0, rel=1e-14)
    assert d.spread_from_upfront(T0, 0.992 * u0, DISC09) == pytest.approx(0.02, rel=1e-10)
    with pytest.raises(ValueError):
        d.with_default("N007", 0.3)
    with pytest.raises(ValueError):
        _index(defaults={"XXX": 0.4})
    with pytest.raises(ValueError):
        CreditIndex(CDX_IG, 12, date(2014, 6, 20), ("A", "B"), (0.6, 0.6))


def test_homogeneous_intrinsic_equals_index():
    ix = _index()
    flat = bootstrap_hazard(T0, [ix.maturity], [0.01], DISC09, 0.4)
    curves = {n: flat for n in NAMES}
    assert index_basis(ix, T0, 0.01, curves, DISC09) == pytest.approx(0.0, abs=1e-12)
    assert intrinsic(ix, T0, curves, DISC09)["spread"] == pytest.approx(
        par_spread(CdsContract(T0, ix.maturity, 0.01), flat, DISC09), rel=1e-12)


def test_heterogeneous_basis_and_adjustment():
    ix = _index().with_default("N000", 0.25)
    curves = {n: bootstrap_hazard(T0, [ix.maturity], [0.004 + 0.0002 * i], DISC09, 0.4)
              for i, n in enumerate(NAMES)}
    intr = intrinsic(ix, T0, curves, DISC09)
    # Quoting the index at the intrinsic spread leaves only a small convexity basis.
    basis = index_basis(ix, T0, intr["spread"], curves, DISC09)
    assert abs(basis) < 2e-3
    beta, adj = adjust_to_index(ix, T0, 0.02, curves, DISC09)
    assert beta > 1.0
    assert intrinsic(ix, T0, adj, DISC09)["upfront"] == pytest.approx(
        ix.upfront_from_spread(T0, 0.02, DISC09), abs=1e-12)


def test_price_quote_hy():
    assert CreditIndex.price_from_upfront(0.0325) == pytest.approx(96.75)
    assert CreditIndex.upfront_from_price(103.5) == pytest.approx(-0.035)
    assert CDX_HY.quote == "price" and CDX_HY.coupon == 0.05
