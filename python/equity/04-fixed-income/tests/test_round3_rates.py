"""Round 3 oracle tests: OIS multi-curve, caps/swaptions, Hull-White."""

import numpy as np
import pytest
from fi_rates.curve import DiscountCurve
from fi_rates.hull_white import HullWhite
from fi_rates.multicurve import (
    bootstrap_ois,
    bootstrap_projection,
    ois_ibor_basis,
    ois_par_rate,
    swap_par_rate,
)
from fi_rates.rate_options import (
    bachelier,
    black76,
    cap_price,
    forward_swap_rate,
    implied_vol,
    swap_annuity,
    swaption_price,
)

MATS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
OIS = [0.030, 0.031, 0.032, 0.033, 0.034, 0.035]
IRS = [0.033, 0.0345, 0.0355, 0.0365, 0.0372, 0.0380]


def _smooth_curve():
    t = np.linspace(0.05, 15.0, 300)
    z = 0.03 + 0.01 * (1 - np.exp(-t / 3.0))
    return DiscountCurve(t, np.exp(-z * t))


# ------------------------------------------------------------ multi-curve
def test_ois_bootstrap_reprices_quotes():
    d = bootstrap_ois(MATS, OIS)
    for T, R in zip(MATS, OIS, strict=False):
        assert ois_par_rate(d, T) == pytest.approx(R, abs=1e-13)


def test_projection_bootstrap_reprices_quotes():
    d = bootstrap_ois(MATS, OIS)
    p = bootstrap_projection(d, MATS, IRS)
    for T, R in zip(MATS, IRS, strict=False):
        assert swap_par_rate(d, p, T) == pytest.approx(R, abs=1e-13)
    assert ois_ibor_basis(d, p, 1.0, 1.25) > 0.0


def test_projection_collapses_to_ois_when_no_basis():
    """IRS quotes equal to OIS par rates -> projection curve equals OIS curve."""
    d = bootstrap_ois(MATS, OIS)
    p = bootstrap_projection(d, MATS, OIS)
    t = np.linspace(0.1, 10.0, 50)
    np.testing.assert_allclose(p.df(t), d.df(t), rtol=0, atol=1e-12)


# ---------------------------------------------------- Black / Bachelier
def test_black_put_call_parity_and_atm():
    F, K, T, s = 0.03, 0.035, 2.0, 0.25
    assert black76(F, K, T, s) - black76(F, K, T, s, call=False) == pytest.approx(F - K, abs=1e-15)
    # ATM: F (2 N(s sqrt(T)/2) - 1)
    from scipy.stats import norm
    assert black76(F, F, T, s) == pytest.approx(F * (2 * norm.cdf(0.5 * s * np.sqrt(T)) - 1), rel=1e-14)


def test_bachelier_atm_closed_form_and_parity():
    F, T, sn = 0.01, 3.0, 0.008
    assert bachelier(F, F, T, sn) == pytest.approx(sn * np.sqrt(T / (2 * np.pi)), rel=1e-14)
    assert bachelier(-0.005, 0.0, T, sn) - bachelier(-0.005, 0.0, T, sn, call=False) == pytest.approx(-0.005, abs=1e-15)


def test_implied_vol_round_trip():
    for model, vol in (("black", 0.3), ("normal", 0.007)):
        fn = black76 if model == "black" else bachelier
        px = fn(0.03, 0.028, 1.5, vol)
        assert implied_vol(px, 0.03, 0.028, 1.5, model=model) == pytest.approx(vol, rel=1e-10)


def test_payer_minus_receiver_is_forward_swap():
    d = bootstrap_ois(MATS, OIS)
    p = bootstrap_projection(d, MATS, IRS)
    K = 0.036
    pay = swaption_price(d, p, 2.0, 5.0, K, 0.2)
    rec = swaption_price(d, p, 2.0, 5.0, K, 0.2, payer=False)
    _, A = swap_annuity(d, 2.0, 5.0)
    F = forward_swap_rate(d, p, 2.0, 5.0)
    assert pay - rec == pytest.approx(A * (F - K), abs=1e-14)


def test_cap_minus_floor_is_swap():
    d = bootstrap_ois(MATS, OIS)
    p = bootstrap_projection(d, MATS, IRS)
    K = 0.035
    cap = cap_price(d, p, K, 5.0, 0.2)
    flr = cap_price(d, p, K, 5.0, 0.2, floor=True)
    t = np.arange(1, 21) / 4.0
    swap = sum(0.25 * d.df(t1) * (p.simple_forward_rate(t0, t1) - K) for t0, t1 in zip(t[:-1], t[1:], strict=False))
    assert cap - flr == pytest.approx(swap, abs=1e-14)


# ---------------------------------------------------------- Hull-White
def test_hw_zcb_reprices_initial_curve():
    c = _smooth_curve()
    hw = HullWhite(0.1, 0.01, c)
    T = np.array([0.5, 2.0, 7.0])
    np.testing.assert_allclose(hw.zcb(0.0, T, hw.inst_forward(0.0)), c.df(T), rtol=1e-7)


def test_hw_bond_option_parity():
    hw = HullWhite(0.1, 0.012, _smooth_curve())
    c = hw.curve
    call = hw.bond_option(1.0, 5.0, 0.88)
    put = hw.bond_option(1.0, 5.0, 0.88, call=False)
    assert call - put == pytest.approx(c.df(5.0) - 0.88 * c.df(1.0), abs=1e-14)


def test_hw_jamshidian_payer_receiver_parity():
    hw = HullWhite(0.08, 0.01, _smooth_curve())
    c = hw.curve
    K = 0.04
    pay = hw.swaption(2.0, 5.0, K)
    rec = hw.swaption(2.0, 5.0, K, payer=False)
    ti = 2.0 + np.arange(1, 6)
    swap = c.df(2.0) - c.df(7.0) - K * np.sum(c.df(ti))
    assert pay - rec == pytest.approx(swap, abs=1e-13)


def test_hw_monte_carlo_oracle():
    """Exact-OU Monte Carlo: zero bond, caplet and swaption within 4 SE."""
    hw = HullWhite(0.1, 0.015, _smooth_curve())
    c = hw.curve
    T0 = 2.0
    rT, I = hw.simulate(T0, 400, 200_000, seed=9)
    disc = np.exp(-I)
    half = disc.size // 2

    def check(payoff, ref):
        x = disc * payoff
        x = 0.5 * (x[:half] + x[half:])
        se = x.std(ddof=1) / np.sqrt(half)
        assert abs(x.mean() - ref) < 4 * se + 2e-5, (x.mean(), ref, se)

    check(np.ones_like(rT), c.df(T0))
    # caplet on [2, 2.5]: payoff at 2 = P(2,2.5) * tau * (L - K)^+
    K = 0.04
    P = hw.zcb(T0, 2.5, rT)
    L = (1 / P - 1) / 0.5
    check(P * 0.5 * np.maximum(L - K, 0), hw.caplet(T0, 2.5, K))
    ti = T0 + np.arange(1, 6)
    bond = sum((K + (i == 4)) * hw.zcb(T0, t, rT) for i, t in enumerate(ti))
    check(np.maximum(1.0 - bond, 0.0), hw.swaption(T0, 5.0, K))


def test_hw_small_sigma_caplet_to_intrinsic():
    c = _smooth_curve()
    hw = HullWhite(0.1, 1e-7, c)
    F = c.simple_forward_rate(1.0, 1.5)
    K = F - 0.005
    assert hw.caplet(1.0, 1.5, K) == pytest.approx(0.5 * c.df(1.5) * (F - K), rel=1e-6)


def test_hw_caplet_matches_bachelier_for_small_a():
    """HW forward rates are near-Gaussian: implied normal vol ~ sigma for small a, tau."""
    c = _smooth_curve()
    hw = HullWhite(1e-4, 0.01, c)
    F = c.simple_forward_rate(1.0, 1.25)
    px = hw.caplet(1.0, 1.25, F)
    sn = implied_vol(px / (0.25 * c.df(1.25)), F, F, 1.0, model="normal")
    assert sn == pytest.approx(0.01 * (1 + 0.25 * F), rel=2e-3)
