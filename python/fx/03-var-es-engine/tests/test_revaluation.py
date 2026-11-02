"""Option revaluation and P&L explain: closed-form Greeks vs finite
differences, linear exactness, third-order residual, delta-gamma vs full."""

import numpy as np
import pytest
from fx_var.revaluation import (
    LinearLeg,
    OptionLeg,
    bsm_greeks,
    bsm_price,
    full_revaluation_pnl,
    pnl_explain,
    taylor_pnl,
)

CALL = OptionLeg(10.0, "call", 100.0, 105.0, 0.5, 0.03, 0.01, 0.25, 0)
PUT = OptionLeg(-7.0, "put", 50.0, 45.0, 1.25, 0.02, 0.04, 0.35, 1)


def _px(leg, S=None, v=None, t=None):
    return float(bsm_price(leg.spot if S is None else S, leg.strike, leg.tau if t is None else t,
                           leg.rate, leg.carry, leg.vol if v is None else v, leg.kind)[0])


@pytest.mark.parametrize("leg", [CALL, PUT])
def test_greeks_match_finite_differences(leg):
    g = bsm_greeks(leg)
    hs, hv, ht = 1e-3 * leg.spot, 1e-4, 1e-5
    S, v, t = leg.spot, leg.vol, leg.tau
    assert g["delta"] == pytest.approx((_px(leg, S + hs) - _px(leg, S - hs)) / (2 * hs), rel=1e-6)
    assert g["gamma"] == pytest.approx(
        (_px(leg, S + hs) - 2 * _px(leg) + _px(leg, S - hs)) / hs**2, rel=1e-5)
    assert g["vega"] == pytest.approx((_px(leg, v=v + hv) - _px(leg, v=v - hv)) / (2 * hv), rel=1e-7)
    assert g["theta"] == pytest.approx(-(_px(leg, t=t + ht) - _px(leg, t=t - ht)) / (2 * ht), rel=1e-6)
    vanna = (_px(leg, S + hs, v + hv) - _px(leg, S + hs, v - hv)
             - _px(leg, S - hs, v + hv) + _px(leg, S - hs, v - hv)) / (4 * hs * hv)
    assert g["vanna"] == pytest.approx(vanna, rel=1e-3)  # cross difference: O(h_S^2) error
    volga = (_px(leg, v=v + hv) - 2 * _px(leg) + _px(leg, v=v - hv)) / hv**2
    assert g["volga"] == pytest.approx(volga, rel=1e-4)


def test_put_call_parity():
    c = _px(CALL)
    p = float(bsm_price(100.0, 105.0, 0.5, 0.03, 0.01, 0.25, "put")[0])
    assert c - p == pytest.approx(100 * np.exp(-0.01 * 0.5) - 105 * np.exp(-0.03 * 0.5), rel=1e-13)


def test_linear_book_is_exact():
    legs = [LinearLeg(3.0, 100.0, 0), LinearLeg(-2.0, 50.0, 1)]
    x = np.random.default_rng(1).normal(0, 0.02, size=(100, 2))
    full = full_revaluation_pnl(legs, x)
    np.testing.assert_allclose(taylor_pnl(legs, x, order="delta"), full, rtol=1e-13, atol=1e-12)
    e = pnl_explain(legs, x[0])
    assert e["unexplained"] == pytest.approx(0.0, abs=1e-12)


def test_explain_residual_is_third_order():
    legs = [CALL, PUT, LinearLeg(-4.0, 100.0, 0)]
    base = np.array([0.02, -0.03])
    dv = np.array([0.02, -0.01])
    r1 = abs(pnl_explain(legs, base, dv, 0.0)["unexplained"])
    r2 = abs(pnl_explain(legs, base / 2, dv / 2, 0.0)["unexplained"])
    assert 5.0 < r1 / r2 < 11.0  # ~8 for a third-order residual


def test_explain_terms_add_up_and_theta():
    e = pnl_explain([CALL], [0.0], [0.0], 1 / 365)
    assert e["delta"] == e["gamma"] == e["vega"] == 0.0
    assert e["theta"] == pytest.approx(10 * bsm_greeks(CALL)["theta"] / 365)
    assert e["explained"] == pytest.approx(sum(e[k] for k in
                                           ("delta", "gamma", "vega", "vanna", "volga", "theta")))
    assert e["actual"] == pytest.approx(e["theta"], rel=2e-3)


def test_delta_gamma_beats_delta_on_large_moves():
    x = np.random.default_rng(2).normal(0, 0.03, size=(2000, 2))
    full = full_revaluation_pnl([CALL, PUT], x)
    err_d = np.abs(taylor_pnl([CALL, PUT], x, order="delta") - full).mean()
    err_dg = np.abs(taylor_pnl([CALL, PUT], x, order="delta_gamma") - full).mean()
    assert err_dg < 0.2 * err_d


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        OptionLeg(1.0, "straddle", 100, 100, 1, 0, 0, 0.2)
    with pytest.raises(ValueError):
        OptionLeg(1.0, "call", 100, 100, 0.0, 0, 0, 0.2)
    with pytest.raises(ValueError):
        full_revaluation_pnl([PUT], np.zeros((3, 1)))  # factor 1 missing
    with pytest.raises(ValueError):
        taylor_pnl([CALL], np.zeros((3, 1)), order="cubic")


def test_matches_package_garman_kohlhagen():
    # BSM with carry = r_f is Garman-Kohlhagen; cross-check against fx_var.gk.
    from fx_var import gk_delta, gk_gamma, gk_price, gk_vega

    for leg in (CALL, PUT):
        args = (leg.spot, leg.strike, leg.tau, leg.rate, leg.carry, leg.vol)
        assert _px(leg) == pytest.approx(gk_price(*args, kind=leg.kind), rel=1e-13)
        g = bsm_greeks(leg)
        assert g["delta"] == pytest.approx(gk_delta(*args, kind=leg.kind), rel=1e-12)
        assert g["gamma"] == pytest.approx(gk_gamma(*args), rel=1e-12)
        assert g["vega"] == pytest.approx(gk_vega(*args), rel=1e-12)
