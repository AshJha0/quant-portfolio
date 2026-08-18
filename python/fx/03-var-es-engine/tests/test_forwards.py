"""FX forward revaluation: CIP consistency, deposit-leg decomposition, rate risk."""

import numpy as np
import pytest

from fx_var import Book, Forward, Market, Spot


@pytest.fixture()
def market():
    return Market(
        spot_usd={"EUR": 1.08, "JPY": 1.0 / 149.0, "GBP": 1.27},
        rates={"USD": 0.053, "EUR": 0.039, "JPY": 0.001, "GBP": 0.052},
    )


def test_atm_forward_zero_value(market):
    """A forward struck at the CIP forward has exactly zero initial value."""
    fwd = Book([Forward("EURUSD", 25e6, 0.5)])
    assert fwd.value_usd(market) == pytest.approx(0.0, abs=1e-6)


def test_exact_revaluation_equals_deposit_legs(market):
    """Engine forward value == hand-built spot + two deposit legs, exactly,
    under joint FX and IR shocks (CIP-consistent revaluation)."""
    n, t, k = 10e6, 0.75, 1.10
    book = Book([Forward("EURUSD", n, t, strike=k)])
    rng = np.random.default_rng(3)
    for _ in range(20):
        de = rng.normal(0, 0.02)
        dr_us, dr_eu = rng.normal(0, 0.005, 2)
        shocks = {"FX:EUR": de, "IR:USD": dr_us, "IR:EUR": dr_eu}
        v1 = book.value_usd(market, shocks)
        s_e = 1.08 * np.exp(de)
        r_d, r_f = 0.053 + dr_us, 0.039 + dr_eu
        legs = n * np.exp(-r_f * t) * s_e - n * k * np.exp(-r_d * t) * 1.0
        assert v1 == pytest.approx(legs, abs=1e-6)
        # equivalently the discounted-forward formula e^{-r_d T} N (F' - K)
        f1 = s_e * np.exp((r_d - r_f) * t)
        assert v1 == pytest.approx(np.exp(-r_d * t) * n * (f1 - k), abs=1e-6)


def test_forward_pnl_approx_spot_pnl_plus_carry(market):
    """For small FX shocks, forward P&L ~ spot P&L discounted by the foreign
    leg - equal up to O(r T) carry terms (CIP consistency)."""
    n, t = 10e6, 1.0 / 12.0  # 1-month forward
    fwd = Book([Forward("EURUSD", n, t)])
    spot = Book([Spot("EURUSD", n)])
    for shock in (0.005, -0.005, 0.01, -0.01):
        p_f = fwd.pnl(market, {"FX:EUR": shock})
        p_s = spot.pnl(market, {"FX:EUR": shock})
        # difference is pure carry/discounting: bounded by max(r)*T + eps
        assert abs(p_f - p_s) <= abs(p_s) * (0.053 * t * 2 + 1e-9)
        assert np.sign(p_f) == np.sign(p_s)


def test_rate_shock_enters_forward_pnl(market):
    """Interest-rate factors move forward P&L (forward-point risk): small
    relative to FX delta, but present, with the right sign and size."""
    n, t = 10e6, 0.5
    book = Book([Forward("EURUSD", n, t)])
    k = market.forward("EURUSD", t)
    dr = 0.005  # +50bp USD
    pnl = book.pnl(market, {"IR:USD": dr})
    # short USD deposit leg: value rises when r_d rises
    expected = -n * k * (np.exp(-(0.053 + dr) * t) - np.exp(-0.053 * t))
    assert pnl == pytest.approx(expected, rel=1e-10)
    assert pnl > 0
    # foreign leg: long EUR deposit loses when r_f rises
    pnl_f = book.pnl(market, {"IR:EUR": dr})
    assert pnl_f < 0
    # rate risk is second-order vs an FX move of usual daily size
    fx_pnl = abs(book.pnl(market, {"FX:EUR": 0.005}))
    assert abs(pnl) < fx_pnl  # 50bp rate move < 0.5% FX move in P&L terms


def test_forward_dv01_magnitude(market):
    """DV01 of the quote leg ~ N * K * T * e^{-r T} * 1e-4."""
    n, t = 10e6, 0.5
    book = Book([Forward("EURUSD", n, t)])
    k = market.forward("EURUSD", t)
    w = book.linear_exposures(market)
    dv01 = w["IR:USD"] * 1e-4
    assert dv01 == pytest.approx(n * k * t * np.exp(-0.053 * t) * 1e-4, rel=1e-4)


def test_expiring_forward_behaves_like_spot(market):
    """T=0 forward P&L equals spot P&L for pure FX shocks (no carry left)."""
    n = 5e6
    fwd = Book([Forward("EURUSD", n, 0.0)])
    spot = Book([Spot("EURUSD", n)])
    for shock in (0.01, -0.03):
        assert fwd.pnl(market, {"FX:EUR": shock}) == pytest.approx(
            spot.pnl(market, {"FX:EUR": shock}), rel=1e-12
        )


def test_cross_forward_triangulates(market):
    """A EURJPY forward decomposes into EUR and JPY deposit legs vs USD."""
    n, t = 10e6, 0.25
    book = Book([Forward("EURJPY", n, t)])
    k = market.forward("EURJPY", t)
    de, dj = 0.015, -0.02
    v1 = book.value_usd(market, {"FX:EUR": de, "FX:JPY": dj})
    s_e, s_j = 1.08 * np.exp(de), (1.0 / 149.0) * np.exp(dj)
    legs = n * np.exp(-0.039 * t) * s_e - n * k * np.exp(-0.001 * t) * s_j
    assert v1 == pytest.approx(legs, abs=1e-6)
