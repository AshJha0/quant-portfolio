"""Round 3 fixes: Hagan-West monotone-convex curve, two-curve Hull-White,
Hull-White swaption-grid calibration."""

import numpy as np
import pytest
from fi_rates.curve import DiscountCurve, _MonotoneConvex
from fi_rates.hull_white import HullWhite, SwaptionQuote, calibrate_hull_white
from fi_rates.multicurve import bootstrap_ois, bootstrap_projection
from fi_rates.rate_options import forward_swap_rate, implied_vol, swap_annuity

MATS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
OIS = [0.030, 0.031, 0.032, 0.033, 0.034, 0.035]
IRS = [0.033, 0.0345, 0.0355, 0.0365, 0.0372, 0.0380]

# Hagan and West (2006) use a curve of this shape (rising, then flattening)
HW_T = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0]
HW_Z = [0.040, 0.041, 0.043, 0.046, 0.048, 0.050, 0.051, 0.0515, 0.050]


def _mc_curve():
    return DiscountCurve.from_zero_rates(HW_T, HW_Z, "monotone_convex")


# ------------------------------------------------------ monotone convex
def test_mc_reprices_every_pillar():
    c = _mc_curve()
    np.testing.assert_allclose(c.df(np.array(HW_T)), np.exp(-np.array(HW_Z) * HW_T),
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(c.zero_rate(np.array(HW_T)), HW_Z, atol=1e-15)


def test_mc_forward_is_continuous_at_pillars():
    c = _mc_curve()
    for t in HW_T[:-1]:
        left = c.instantaneous_forward(t - 1e-10)
        right = c.instantaneous_forward(t + 1e-10)
        assert left == pytest.approx(right, abs=1e-8)


def test_mc_forward_is_derivative_of_minus_log_df():
    c = _mc_curve()
    t = np.linspace(0.01, 19.9, 777)
    h = 1e-6
    fd = -(np.log(c.df(t + h)) - np.log(c.df(t - h))) / (2 * h)
    np.testing.assert_allclose(c.instantaneous_forward(t), fd, atol=1e-7)


def test_mc_monotone_discrete_forwards_give_monotone_positive_forward():
    """Increasing discrete forwards -> the interpolated forward is increasing
    and stays inside the range of the node forwards (no overshoot)."""
    t = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
    fwd = np.array([0.01, 0.015, 0.02, 0.03, 0.035])  # discrete forwards
    y = np.cumsum(fwd * np.diff(np.concatenate(([0.0], t))))
    c = DiscountCurve(t, np.exp(-y), "monotone_convex")
    grid = np.linspace(1e-4, 10.0, 4001)
    f = np.asarray(c.instantaneous_forward(grid))
    assert np.all(np.diff(f) >= -1e-13)
    assert f.min() > 0.0
    # piecewise-constant forwards (loglinear) jump; monotone convex does not
    assert np.max(np.abs(np.diff(f))) < 1e-4


def test_mc_is_local():
    """Bumping pillar k moves the curve only on (t_{k-2}, t_{k+2})."""
    c0 = _mc_curve()
    k = 4  # 3y pillar
    z = np.array(HW_Z)
    z[k] += 1e-4
    c1 = DiscountCurve.from_zero_rates(HW_T, z, "monotone_convex")
    far = np.array([0.1, 0.4, 0.9, 9.0, 15.0])  # outside (t_{k-2}, t_{k+2}) = (1, 7)
    np.testing.assert_allclose(c1.instantaneous_forward(far), c0.instantaneous_forward(far),
                               atol=1e-15)
    assert abs(c1.instantaneous_forward(2.5) - c0.instantaneous_forward(2.5)) > 1e-6


@pytest.mark.parametrize("g0,g1", [(-1.0, 1.0), (1.0, -1.0),        # region (i)
                                   (-1.0, 3.0), (1.0, -3.0),        # region (ii)
                                   (1.0, -0.2), (-1.0, 0.2),        # region (iii)
                                   (1.0, 2.0), (-1.0, -0.5),        # region (iv)
                                   (0.0, 0.0)])
def test_mc_g_regions_integrate_to_zero_and_hit_end_values(g0, g1):
    """Every Hagan-West region: G(0) = g0, G(1) = g1, int_0^1 G = 0, and the
    closed-form integral matches quadrature."""
    G0, _ = _MonotoneConvex._g(g0, g1, 0.0)
    G1, I1 = _MonotoneConvex._g(g0, g1, 1.0)
    assert G0 == pytest.approx(g0, abs=1e-14)
    assert G1 == pytest.approx(g1, abs=1e-14)
    assert I1 == pytest.approx(0.0, abs=1e-14)
    xs = np.linspace(0.0, 0.7, 70001)
    gs = np.array([_MonotoneConvex._g(g0, g1, x)[0] for x in xs])
    trap = float(np.sum(0.5 * (gs[1:] + gs[:-1]) * np.diff(xs)))
    assert _MonotoneConvex._g(g0, g1, 0.7)[1] == pytest.approx(trap, abs=1e-8)


def test_mc_single_pillar_and_extrapolation():
    c = DiscountCurve([5.0], [0.8], "monotone_convex")
    assert c.df(5.0) == pytest.approx(0.8, abs=1e-15)
    assert c.zero_rate(2.0) == pytest.approx(-np.log(0.8) / 5.0, abs=1e-14)
    with pytest.warns(Warning):
        assert c.zero_rate(8.0) == pytest.approx(-np.log(0.8) / 5.0, abs=1e-14)


# ------------------------------------------------- two-curve Hull-White
def _two_curves():
    d = bootstrap_ois(MATS, OIS)
    p = bootstrap_projection(d, MATS, IRS)
    return d, p


def test_hw_two_curve_reduces_to_single_curve():
    d, _ = _two_curves()
    hw = HullWhite(0.05, 0.01, d)
    for K in (0.025, 0.033, 0.045):
        a = hw.swaption(2.0, 5.0, K)
        b = hw.swaption(2.0, 5.0, K, forward_curve=d, float_frequency=4)
        assert a == pytest.approx(b, abs=1e-14)


def test_hw_two_curve_parity_matches_annuity_formula():
    """payer - receiver = A (F - K) with F the multi-curve forward swap rate
    of rate_options (an independent code path)."""
    d, p = _two_curves()
    hw = HullWhite(0.05, 0.01, d)
    for exp, ten in ((1.0, 5.0), (2.0, 3.0), (3.0, 7.0)):
        F = forward_swap_rate(d, p, exp, ten, 1, 4)
        _, A = swap_annuity(d, exp, ten, 1)
        for K in (F - 0.01, F, F + 0.01):
            pay = hw.swaption(exp, ten, K, 1, True, p, 4)
            rec = hw.swaption(exp, ten, K, 1, False, p, 4)
            assert pay - rec == pytest.approx(A * (F - K), abs=1e-13)


def test_hw_two_curve_monte_carlo_oracle():
    """Exact-OU Monte Carlo of the two-curve payer payoff at expiry."""
    d, p = _two_curves()
    hw = HullWhite(0.05, 0.012, d)
    T0, ten = 2.0, 5.0
    F = forward_swap_rate(d, p, T0, ten, 1, 4)
    K = F + 0.002
    rT, integral = hw.simulate(T0, 400, 200_000, seed=11)
    s = T0 + np.arange(0, 21) / 4.0
    beta = (p.df(s[:-1]) / p.df(s[1:])) * (d.df(s[1:]) / d.df(s[:-1]))
    flt = sum(beta[j] * hw.zcb(T0, s[j], rT) - hw.zcb(T0, s[j + 1], rT) for j in range(20))
    fix = K * sum(hw.zcb(T0, T0 + i, rT) for i in range(1, 6))
    x = np.exp(-integral) * np.maximum(flt - fix, 0.0)
    half = x.size // 2
    x = 0.5 * (x[:half] + x[half:])
    se = x.std(ddof=1) / np.sqrt(half)
    ref = hw.swaption(T0, ten, K, 1, True, p, 4)
    assert abs(x.mean() - ref) < 4 * se + 2e-5, (x.mean(), ref, se)


def test_hw_two_curve_differs_from_single_curve():
    d, p = _two_curves()
    hw = HullWhite(0.05, 0.01, d)
    F = forward_swap_rate(d, p, 2.0, 5.0, 1, 4)
    assert hw.swaption(2.0, 5.0, F, 1, True, p, 4) > hw.swaption(2.0, 5.0, F) + 1e-4


# ----------------------------------------------- swaption calibration
def _grid(hw, d, p, model="normal"):
    quotes = []
    for exp in (1.0, 2.0, 5.0):
        for ten in (2.0, 5.0, 10.0):
            F = forward_swap_rate(d, p, exp, ten, 1, 4)
            _, A = swap_annuity(d, exp, ten, 1)
            px = hw.swaption(exp, ten, F, 1, True, p, 4)
            vol = implied_vol(px / A, F, F, exp, model=model)
            quotes.append(SwaptionQuote(exp, ten, vol, None, True, model, 1, 4))
    return quotes


@pytest.mark.parametrize("model", ["normal", "black"])
def test_calibration_recovers_known_parameters(model):
    d, p = _two_curves()
    true = HullWhite(0.07, 0.0095, d)
    res = calibrate_hull_white(d, _grid(true, d, p, model), forward_curve=p)
    assert res.success
    assert res.model.a == pytest.approx(0.07, rel=1e-6)
    assert res.model.sigma == pytest.approx(0.0095, rel=1e-6)
    assert res.max_abs_error < 1e-10


def test_calibration_with_fixed_mean_reversion():
    d, p = _two_curves()
    true = HullWhite(0.03, 0.011, d)
    res = calibrate_hull_white(d, _grid(true, d, p), forward_curve=p, a=0.03)
    assert res.model.a == 0.03
    assert res.model.sigma == pytest.approx(0.011, rel=1e-8)
    # wrong a: sigma alone cannot fit the whole grid exactly
    off = calibrate_hull_white(d, _grid(true, d, p), forward_curve=p, a=0.3)
    assert off.max_abs_error > 1e-5


def test_calibration_single_curve_and_validation():
    d, _ = _two_curves()
    true = HullWhite(0.05, 0.01, d)
    q = []
    for exp, ten in ((1.0, 5.0), (5.0, 5.0)):
        F = forward_swap_rate(d, d, exp, ten, 1, 1)
        _, A = swap_annuity(d, exp, ten, 1)
        vol = implied_vol(true.swaption(exp, ten, F) / A, F, F, exp, model="normal")
        q.append(SwaptionQuote(exp, ten, vol, None, True, "normal", 1, 1))
    res = calibrate_hull_white(d, q, a=0.05)
    assert res.model.sigma == pytest.approx(0.01, rel=1e-8)
    with pytest.raises(ValueError):
        calibrate_hull_white(d, [])
    with pytest.raises(ValueError):
        calibrate_hull_white(d, [SwaptionQuote(1.0, 5.0, 0.01, model="sabr")])
