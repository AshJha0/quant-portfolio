"""W14: par-rate risk (Jacobian against rebuild), cross-gamma and curve P&L explain."""

from __future__ import annotations

import math

import numpy as np
import pytest
from fi_rates.bootstrap import FRA, Deposit, ParSwap, bootstrap_curve, reprice_instruments
from fi_rates.curve import DiscountCurve
from fi_rates.risk import (
    CashflowBook,
    explain_curve_day,
    par_rate_gamma,
    par_rate_risk,
    par_rate_risk_by_rebuild,
    quote_jacobian,
)
from fi_rates.risk.par import model_quotes, node_gamma, with_rates
from qp_core.risk import cross_gamma


def _quotes(shift: float = 0.0):
    return [
        Deposit(0.5, 0.040 + shift),
        FRA(0.5, 1.0, 0.041 + shift),
        ParSwap(2.0, 0.042 + shift),
        ParSwap(3.0, 0.0415 + shift),
        ParSwap(5.0, 0.040 + shift),
        ParSwap(7.0, 0.0405 + shift),
        ParSwap(10.0, 0.041 + shift),
    ]


def _book() -> CashflowBook:
    # A long 6y 4.5% annual bond, a short 3.5y amortising strip and a
    # 9.25y bullet: flows sit between pillars on purpose.
    t = [1, 2, 3, 4, 5, 6, 0.75, 1.75, 2.75, 3.5, 9.25]
    a = [4.5] * 5 + [104.5] + [-30.0, -30.0, -30.0, -30.0] + [80.0]
    return CashflowBook(tuple(float(x) for x in t), tuple(a))


INTERPS_LOCAL = ["loglinear_df", "linear_zero"]


@pytest.mark.parametrize("interp", INTERPS_LOCAL)
def test_jacobian_matches_finite_differences_to_1e6(interp):
    # Acceptance (plan W14): the Jacobian route equals bumping each quote
    # and rebuilding the curve, to 1e-6 relative per instrument.
    q = _quotes()
    curve = bootstrap_curve(q, interp)
    book = _book()

    def pv(c: DiscountCurve) -> float:
        return book.pv(c, 0.0)

    via_jac = par_rate_risk(pv, q, curve)
    rebuild = par_rate_risk_by_rebuild(pv, q, interp)
    scale = np.max(np.abs(rebuild))
    np.testing.assert_allclose(via_jac, rebuild, rtol=0, atol=1e-6 * scale)
    # The par-rate DV01s sum to a parallel quote shift (rebuild route).
    up = pv(bootstrap_curve(_quotes(1e-6), interp))
    dn = pv(bootstrap_curve(_quotes(-1e-6), interp))
    assert via_jac.sum() == pytest.approx((up - dn) / 2e-6, rel=1e-6)


@pytest.mark.parametrize("interp", ["pchip_zero", "monotone_convex"])
def test_non_local_sequential_bootstrap_is_refused(interp):
    # These curves miss their own quotes by ~7e-6 (sequential bootstrap,
    # non-local interpolation), so the Jacobian route would be 0.7-1.8% off.
    q = _quotes()
    curve = bootstrap_curve(q, interp)
    with pytest.raises(ValueError, match="does not reprice"):
        par_rate_risk(lambda c: _book().pv(c, 0.0), q, curve)


def test_jacobian_is_lower_triangular_for_a_local_sequential_bootstrap():
    q = _quotes()
    jac = quote_jacobian(q, bootstrap_curve(q))
    assert np.all(np.abs(np.triu(jac, 1)) < 1e-9)
    assert np.all(np.diag(jac) > 0)  # quotes rise with their own pillar zero


def test_par_rate_risk_of_the_hedge_instrument_itself():
    q = [Deposit(1.0, 0.04), ParSwap(2.0, 0.045)]
    curve = bootstrap_curve(q)
    swap = CashflowBook((1.0, 2.0), (4.5, 104.5))  # receive 4.5% fixed + par

    def pv(c):
        return swap.pv(c, 0.0)

    g = par_rate_risk(pv, q, curve) * 1e-4
    assert g[1] == pytest.approx(-0.0188, abs=5e-5)
    assert abs(g[0]) < 1e-9


def test_quote_gamma_mapping_against_rebuild_hessian():
    q = _quotes()
    curve = bootstrap_curve(q)
    book = _book()

    def pv(c):
        return book.pv(c, 0.0)

    mapped = par_rate_gamma(pv, q, curve)
    rates = np.array([i.rate for i in q])
    rebuilt = cross_gamma(lambda r: pv(bootstrap_curve(with_rates(q, r))), rates, h=1e-4)
    scale = np.max(np.abs(rebuilt))
    assert np.max(np.abs(mapped - rebuilt)) < 1e-5 * scale  # measured 1.7e-7
    # The shortcut J^-T Gamma_z J^-1 (no bootstrap curvature) is wrong by
    # about twice the largest entry on this book: pinned so it stays fixed.
    jac = quote_jacobian(q, curve)
    inv = np.linalg.inv(jac)
    shortcut = inv.T @ node_gamma(pv, curve) @ inv
    assert np.max(np.abs(shortcut - rebuilt)) > 1.0 * scale


def test_model_quotes_reprice_inputs_and_pillar_mismatch_refused():
    q = _quotes()
    curve = bootstrap_curve(q)
    np.testing.assert_allclose(model_quotes(q, curve), [i.rate for i in q], atol=1e-12)
    bumped = curve.bumped_pillars(np.linspace(-1e-3, 2e-3, curve.times.size))
    ref = [e + i.rate for i, e in reprice_instruments(q, bumped)]
    np.testing.assert_allclose(model_quotes(q, bumped), ref, rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="pillar"):
        quote_jacobian(q[:-1], curve)


# ------------------------------------------------------------------- explain


def test_flat_curve_carry_is_the_overnight_rate():
    q = [Deposit(1.0, math.expm1(0.04)), ParSwap(5.0, math.expm1(0.04))]
    book = CashflowBook((5.0,), (100.0,))
    dt = 1.0 / 365
    e = explain_curve_day(book, q, q, 0.0, dt)
    assert e.time["carry"] == pytest.approx(100 * (math.exp(-0.04 * (5 - dt)) - math.exp(-0.2)), rel=1e-9)
    assert abs(e.time["rolldown"]) < 1e-9
    assert abs(sum(e.risk.values())) < 1e-12
    assert abs(e.unexplained) < 1e-9


def test_explain_components_add_up_and_gamma_helps():
    q0 = _quotes()
    rng = np.random.default_rng(5)
    rates1 = np.array([i.rate for i in q0]) + 0.0015 * rng.standard_normal(len(q0))
    q1 = with_rates(q0, rates1)
    book = _book()
    with_g = explain_curve_day(book, q0, q1, 0.0, 1 / 365)
    without = explain_curve_day(book, q0, q1, 0.0, 1 / 365, with_gamma=False)
    assert with_g.hpl == pytest.approx(with_g.rtpl + with_g.unexplained)
    assert with_g.hpl == pytest.approx(without.hpl)
    assert abs(with_g.unexplained) < 0.2 * abs(without.unexplained)
    assert abs(with_g.unexplained) < 1e-4 * abs(with_g.hpl - with_g.theta)


def test_cashflow_paid_in_period_counts_once():
    q = _quotes()
    book = CashflowBook((0.5, 2.0), (10.0, 100.0))
    e = explain_curve_day(book, q, q, 0.499, 0.501)
    # HPL includes the coupon paid; carry contains it too, so no jump in
    # the unexplained term.
    assert abs(e.unexplained) < 1e-9
    assert e.hpl == pytest.approx(e.theta, abs=1e-9)


def test_explain_input_validation():
    q = _quotes()
    with pytest.raises(ValueError, match="t1 > t0"):
        explain_curve_day(_book(), q, q, 1.0, 1.0)
    with pytest.raises(ValueError, match="same instruments"):
        explain_curve_day(_book(), q, q[:-1], 0.0, 0.01)
    with pytest.raises(ValueError):
        CashflowBook((1.0,), (1.0, 2.0))
