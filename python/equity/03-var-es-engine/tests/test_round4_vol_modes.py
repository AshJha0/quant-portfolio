"""Round 4: option VaR vol scenario modes (sticky strike, sticky delta,
historical ATM shift). The golden numbers are pasted into the four compiled
engines' tests (C++ and Rust, equity and FX)."""

import numpy as np
import pytest
from eq_var import revaluation as rv

CALL = rv.OptionLeg(10.0, "call", 100.0, 105.0, 0.5, 0.03, 0.01, 0.25, 0, skew=-0.1)
PUT = rv.OptionLeg(-7.0, "put", 50.0, 45.0, 1.25, 0.02, 0.04, 0.35, 1, skew=0.15)
X = np.array([[0.01, -0.02], [-0.05, 0.03], [0.0, 0.0]])
DV = np.array([[0.01, 0.0], [0.05, -0.02], [0.0, 0.0]])
DT = 1.0 / 365.0

GOLDEN = {
    "sticky_strike": [4.92183106078723, -0.6845739357538516, -0.1540808210950111],
    "sticky_delta": [4.793674377411101, -1.3383700871131143, -0.1540808210950111],
    "historical_atm": [2.8099164421405085, -11.777009547269026, -0.1540808210950111],
}


@pytest.mark.parametrize("mode", rv.VOL_MODES)
def test_golden(mode):
    pnl = rv.full_revaluation_pnl([CALL, PUT], X, DV, DT, vol_mode=mode)
    np.testing.assert_allclose(pnl, GOLDEN[mode], rtol=1e-13, atol=1e-13)


def test_default_is_sticky_strike_bit_for_bit():
    a = rv.full_revaluation_pnl([CALL, PUT], X, DV, DT)
    b = rv.full_revaluation_pnl([CALL, PUT], X, DV, DT, vol_mode="sticky_strike")
    np.testing.assert_array_equal(a, b)


def test_sticky_delta_with_flat_smile_equals_sticky_strike():
    flat = [rv.OptionLeg(10.0, "call", 100.0, 105.0, 0.5, 0.03, 0.01, 0.25, 0),
            rv.OptionLeg(-7.0, "put", 50.0, 45.0, 1.25, 0.02, 0.04, 0.35, 1)]
    np.testing.assert_array_equal(
        rv.full_revaluation_pnl(flat, X, DV, DT, vol_mode="sticky_delta"),
        rv.full_revaluation_pnl(flat, X, DV, DT, vol_mode="sticky_strike"))


def test_sticky_delta_by_hand():
    # one call, spot down 5 %, skew -0.1: the strike's vol moves by -skew*x = -0.005
    x, dv = -0.05, 0.01
    p0 = rv.bsm_price(100.0, 105.0, 0.5, 0.03, 0.01, 0.25, "call")[0]
    p1 = rv.bsm_price(100.0 * np.exp(x), 105.0, 0.5 - DT, 0.03, 0.01, 0.25 - 0.005 + dv, "call")[0]
    got = rv.full_revaluation_pnl([CALL], [[x]], [[dv]], DT, vol_mode="sticky_delta")[0]
    assert got == pytest.approx(10.0 * (p1 - p0), rel=1e-14)


def test_sticky_delta_sign_on_equity_skew():
    """Downward skew, spot falls: the fixed strike is now further out of the
    money on the call side of the smile, so its vol falls (Derman 1999)."""
    v = rv.shocked_vol(CALL, -0.05, 0.0, "sticky_delta")
    assert float(v) == pytest.approx(0.25 - 0.005, abs=1e-15)
    assert float(rv.shocked_vol(CALL, -0.05, 0.0, "sticky_strike")) == 0.25


def test_historical_atm_is_proportional():
    assert float(rv.shocked_vol(PUT, 0.3, np.log(1.2), "historical_atm")) == pytest.approx(0.42)
    # zero shock: same as sticky strike with zero shock, whatever the spot move
    z = np.zeros_like(DV)
    np.testing.assert_array_equal(
        rv.full_revaluation_pnl([CALL, PUT], X, z, DT, vol_mode="historical_atm"),
        rv.full_revaluation_pnl([CALL, PUT], X, z, DT, vol_mode="sticky_strike"))


def test_vol_floor_in_every_mode():
    for mode, dv in (("sticky_strike", -1.0), ("sticky_delta", -1.0), ("historical_atm", -800.0)):
        assert float(rv.shocked_vol(CALL, 0.0, dv, mode)) == 1e-12


def test_linear_legs_ignore_vol_mode():
    book = [rv.LinearLeg(3.0, 100.0, 0)]
    for mode in rv.VOL_MODES:
        np.testing.assert_array_equal(
            rv.full_revaluation_pnl(book, X, DV, vol_mode=mode),
            rv.full_revaluation_pnl(book, X, DV))


def test_bad_mode_and_skew_rejected():
    with pytest.raises(ValueError, match="vol_mode"):
        rv.full_revaluation_pnl([CALL], X, DV, vol_mode="sticky_moneyness")
    with pytest.raises(ValueError, match="vol_mode"):
        rv.shocked_vol(CALL, 0.0, 0.0, "x")
    with pytest.raises(ValueError):
        rv.OptionLeg(1.0, "call", 100.0, 100.0, 1.0, 0.0, 0.0, 0.2, skew=float("nan"))
