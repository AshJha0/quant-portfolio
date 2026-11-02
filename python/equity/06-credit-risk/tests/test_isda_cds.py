"""ISDA standard-model CDS: dates, exact leg integrals, bootstrap, upfront."""

from datetime import date, timedelta

import numpy as np
import pytest
from eq_credit.cds import HazardCurve
from eq_credit.isda_cds import (
    IsdaCds,
    LogLinearCurve,
    add_business_days,
    adjust_following,
    bootstrap_isda,
    legs,
    par_spread,
    previous_coupon_date,
    upfront_from_quoted_spread,
)
from scipy.integrate import quad

TRADE = date(2026, 11, 2)  # a Monday
DISC = LogLinearCurve(np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]),
                      np.array([0.036, 0.0355, 0.034, 0.0325, 0.0322, 0.0328, 0.0337, 0.035, 0.036]))
MATS = [date(2027, 12, 20), date(2029, 12, 20), date(2031, 12, 20), date(2033, 12, 20),
        date(2036, 12, 20)]
SPREADS = [0.0060, 0.0085, 0.0110, 0.0125, 0.0135]


# ---------------------------------------------------------------- dates
def test_standard_dates():
    cds = IsdaCds(TRADE, date(2031, 12, 20), 0.01)
    assert cds.step_in == date(2026, 11, 3)
    assert cds.cash_settle == date(2026, 11, 5)
    # 20 Sep 2026 is a Sunday: the accrual start rolls to Monday 21 Sep
    assert cds.accrual_start == date(2026, 9, 21)
    assert cds.accrued() == pytest.approx(0.01 * 43 / 360, abs=1e-16)
    p = cds.periods()
    assert p[0][0] == date(2026, 9, 21) and p[0][1] == date(2026, 12, 21)
    assert p[-1][1] == date(2031, 12, 21)  # maturity + 1 day
    assert p[-1][2] == date(2031, 12, 22)  # 20 Dec 2031 is a Saturday
    assert len(p) == 21
    for (_, e0, _), (s1, _, _) in zip(p[:-1], p[1:], strict=True):
        assert e0 == s1


def test_date_helpers():
    assert previous_coupon_date(date(2026, 3, 19)) == date(2025, 12, 20)
    assert previous_coupon_date(date(2026, 3, 20)) == date(2026, 3, 20)
    assert previous_coupon_date(date(2026, 1, 5)) == date(2025, 12, 20)
    assert adjust_following(date(2026, 6, 20)) == date(2026, 6, 22)
    assert add_business_days(date(2026, 11, 6), 3) == date(2026, 11, 11)


# ----------------------------------------------------- exact integrals
def test_leg_integrals_match_quadrature():
    hz = HazardCurve(np.array([1.2, 3.1, 5.15]), np.array([0.01, 0.018, 0.025]))
    cds = IsdaCds(TRADE, date(2031, 12, 20), 0.01, 0.4)
    out = legs(cds, hz, DISC)
    t_end = cds.t(date(2031, 12, 21))
    pts = sorted({*hz.times.tolist(), *DISC.times.tolist()})

    def lam(u):
        return float(hz.hazards[min(np.searchsorted(hz.times, u), 2)])

    def dens(u):
        return lam(u) * float(hz.survival(u)[0]) * DISC.df(u)

    prot = 0.6 * quad(dens, 0.0, t_end, points=pts, limit=400, epsabs=1e-14)[0]
    assert out["protection"] == pytest.approx(prot, rel=1e-10)
    rpv = 0.0
    for s_d, e_d, pay in cds.periods():
        s, e = cds.t(s_d), cds.t(e_d)
        rpv += (e_d - s_d).days / 360 * float(hz.survival(e)[0]) * DISC.df(cds.t(pay) + 1 / 365)
        rpv += 365 / 360 * quad(lambda u, s=s: (u - s) * dens(u), max(s, 0.0), e,
                                points=[p for p in pts if max(s, 0) < p < e] or None,
                                epsabs=1e-15)[0]
    assert out["rpv01"] == pytest.approx(rpv, rel=1e-10)


def test_credit_triangle_with_act360_accrual():
    """Flat hazard, flat rate: par spread ~ lambda (1 - R) * 360/365, because
    premium accrues ACT/360 while time runs ACT/365F, divided by
    (1 - r tau / 2) because survivors pay quarterly in arrears rather than
    continuously (first order in r tau)."""
    flat = LogLinearCurve(np.array([1.0, 30.0]), np.array([0.03, 0.03]))
    lam = 0.02
    hz = HazardCurve(np.array([20.0]), np.array([lam]))
    s = par_spread(IsdaCds(TRADE, date(2031, 12, 20), 0.01, 0.4), hz, flat)
    assert s == pytest.approx(lam * 0.6 * 360 / 365 / (1 - 0.03 * 0.25 / 2), rel=5e-4)


# ------------------------------------------------------------ bootstrap
def test_bootstrap_reprices_every_par_spread():
    hz = bootstrap_isda(TRADE, MATS, SPREADS, DISC, 0.4)
    assert np.all(hz.hazards > 0.0)
    for m, s in zip(MATS, SPREADS, strict=True):
        cds = IsdaCds(TRADE, m, s, 0.4)
        assert legs(cds, hz, DISC)["clean_upfront"] == pytest.approx(0.0, abs=1e-13)
        assert par_spread(cds, hz, DISC) == pytest.approx(s, abs=1e-13)


def test_bootstrap_flat_spreads_give_near_flat_hazard():
    hz = bootstrap_isda(TRADE, MATS, [0.01] * 5, DISC, 0.4)
    assert np.ptp(hz.hazards) < 2e-4
    assert hz.hazards[0] == pytest.approx(0.01 / 0.6 * 365 / 360, rel=5e-3)


def test_bootstrap_validation():
    with pytest.raises(ValueError):
        bootstrap_isda(TRADE, MATS[::-1], SPREADS, DISC)
    with pytest.raises(ValueError):
        bootstrap_isda(TRADE, MATS, [0.01, -0.01, 0.01, 0.01, 0.01], DISC)
    with pytest.raises(ValueError):
        bootstrap_isda(TRADE, MATS[:2], [0.03, 0.001], DISC)  # inverted: negative hazard
    with pytest.raises(ValueError):
        IsdaCds(TRADE, TRADE + timedelta(days=1), 0.01)
    with pytest.raises(ValueError):
        IsdaCds(TRADE, MATS[0], 0.01, recovery=1.0)


# -------------------------------------------------------------- upfront
def test_upfront_conversion():
    m = date(2031, 12, 20)
    at = upfront_from_quoted_spread(TRADE, m, 0.01, 0.01, DISC)
    assert at["clean_upfront"] == pytest.approx(0.0, abs=1e-13)
    assert at["dirty_upfront"] == pytest.approx(-at["accrued"], abs=1e-13)
    wide = upfront_from_quoted_spread(TRADE, m, 0.03, 0.01, DISC)
    assert wide["clean_upfront"] > 0.0
    # first order: upfront ~ (S - C) * RPV01 (risky duration at the quoted spread)
    approx = 0.02 * wide["rpv01"]
    assert wide["clean_upfront"] == pytest.approx(approx, rel=0.03)
    tight = upfront_from_quoted_spread(TRADE, m, 0.005, 0.01, DISC)
    assert tight["clean_upfront"] < 0.0 and tight["price"] > 1.0


def test_loglinear_curve():
    c = LogLinearCurve(np.array([1.0, 2.0]), np.array([0.02, 0.03]))
    assert c.df(1.0) == pytest.approx(np.exp(-0.02), abs=1e-15)
    assert c.df(1.5) == pytest.approx(np.exp(-0.04), abs=1e-15)  # r t linear
    assert c.df(3.0) == pytest.approx(np.exp(-0.06 - 0.04), abs=1e-15)  # flat last forward
    with pytest.raises(ValueError):
        LogLinearCurve(np.array([1.0, 0.5]), np.array([0.02, 0.03]))


# ------------------------------------------------- holiday calendars (round 4)
def test_calendar_moves_cash_settle_over_holiday():
    m = date(2031, 12, 20)
    plain = IsdaCds(date(2026, 11, 6), m, 0.01)
    usd = IsdaCds(date(2026, 11, 6), m, 0.01, calendar="USD")
    assert plain.cash_settle == date(2026, 11, 11)
    assert usd.cash_settle == date(2026, 11, 12)  # 11 Nov Veterans Day
    assert IsdaCds(date(2026, 11, 6), m, 0.01, calendar="EUR").cash_settle == date(2026, 11, 11)


def test_calendar_rolls_coupon_dates():
    # 20 Dec 2027 is a Monday (open); 20 Mar 2028 Monday; GBP Easter 2028:
    # Good Friday 14 Apr, Easter Monday 17 Apr. Use JPY: 20 Mar 2028 is the
    # vernal equinox (holiday), so the payment rolls to 21 Mar.
    cds = IsdaCds(date(2027, 12, 1), date(2028, 6, 20), 0.01, calendar="JPY")
    pay = [p for _, _, p in cds.periods()]
    assert date(2028, 3, 21) in pay
    plain = IsdaCds(date(2027, 12, 1), date(2028, 6, 20), 0.01)
    assert date(2028, 3, 20) in [p for _, _, p in plain.periods()]


def test_calendar_accrual_start_rolls():
    # 20 Mar 2028 JPY holiday: a trade stepping in on 21 Mar accrues from 21 Mar.
    cds = IsdaCds(date(2028, 3, 20), date(2033, 3, 20), 0.01, calendar="JPY")
    assert cds.accrual_start == date(2028, 3, 21)
    assert cds.accrued() == pytest.approx(0.0)


def test_calendar_none_matches_weekend_only_pricing():
    disc = DISC
    a = bootstrap_isda(TRADE, MATS, SPREADS, disc)
    b = bootstrap_isda(TRADE, MATS, SPREADS, disc, calendar=None)
    np.testing.assert_array_equal(a.hazards, b.hazards)


def test_calendar_bootstrap_reprices_par():
    disc = DISC
    hz = bootstrap_isda(TRADE, MATS, SPREADS, disc, calendar="USD")
    for m, s in zip(MATS, SPREADS, strict=True):
        cds = IsdaCds(TRADE, m, s, 0.4, "USD")
        assert abs(legs(cds, hz, disc)["clean_upfront"]) < 1e-12


def test_calendar_runtime_closure_hook():
    from qp_core.holidays import extra_closures
    m = date(2031, 12, 20)
    with extra_closures({"USD": [date(2026, 11, 12)]}):
        assert IsdaCds(date(2026, 11, 6), m, 0.01, calendar="USD").cash_settle == date(2026, 11, 13)
    assert IsdaCds(date(2026, 11, 6), m, 0.01, calendar="USD").cash_settle == date(2026, 11, 12)


def test_calendar_bad_code_raises():
    with pytest.raises(ValueError):
        IsdaCds(TRADE, MATS[0], 0.01, calendar="XYZ")
