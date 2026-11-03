"""Tests for qp_core.risk (W14, W14b): Jacobian, vega grid, smile dynamics, explain."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qp_core.black_scholes import carry_greeks
from qp_core.normal import norm_cdf
from qp_core.risk import (
    Smile,
    SmileDynamics,
    VolGrid,
    cross_gamma,
    explain,
    fd_gradient,
    fd_jacobian,
    node_to_quote_risk,
    premium_adjusted_delta,
    quote_to_node_risk,
    rank_smile_dynamics,
    smile_delta,
    smile_gamma,
    smile_price,
    taylor_attribution,
    vega_grid,
)

# ------------------------------------------------------------------ jacobian


def test_jacobian_of_linear_map_is_exact():
    a = np.array([[1.0, 2.0, 0.0], [0.5, -1.0, 3.0]])
    jac = fd_jacobian(lambda x: a @ x, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(jac, a, atol=1e-9)


def test_node_to_quote_worked_example():
    jac = np.diag([-0.5, -0.5])
    g_q = node_to_quote_risk(np.array([-2.0, -4.0]), jac)
    np.testing.assert_allclose(g_q, [4.0, 8.0])
    np.testing.assert_allclose(quote_to_node_risk(g_q, jac), [-2.0, -4.0])


def test_node_to_quote_matches_reparametrised_gradient():
    # q = Q(z) nonlinear and invertible; P(z) arbitrary. dP/dq by the
    # Jacobian route must equal the gradient of P(Q^{-1}(q)).
    def quotes(z):
        return np.array([z[0] + 0.1 * z[0] ** 2, z[1] + 0.3 * z[0] * z[1]])

    def inverse(q):
        z0 = (-1.0 + math.sqrt(1.0 + 0.4 * q[0])) / 0.2
        return np.array([z0, q[1] / (1.0 + 0.3 * z0)])

    def pv(z):
        return math.exp(-z[0]) + 2.0 * math.exp(-2.0 * z[1])

    z = np.array([0.03, 0.04])
    g_z = fd_gradient(pv, z)
    jac = fd_jacobian(quotes, z)
    via_jac = node_to_quote_risk(g_z, jac)
    direct = fd_gradient(lambda q: pv(inverse(q)), quotes(z))
    np.testing.assert_allclose(via_jac, direct, rtol=1e-6)


def test_singular_jacobian_refused():
    with pytest.raises(ValueError, match="singular"):
        node_to_quote_risk(np.ones(2), np.array([[1.0, 1.0], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="square"):
        node_to_quote_risk(np.ones(2), np.ones((3, 2)))


def test_cross_gamma_of_quadratic_is_exact_and_symmetric():
    h = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, -0.3], [0.0, -0.3, 4.0]])
    got = cross_gamma(lambda x: 0.5 * x @ h @ x + x.sum(), np.array([0.1, -0.2, 0.3]))
    np.testing.assert_allclose(got, h, atol=1e-6)
    np.testing.assert_array_equal(got, got.T)


@pytest.mark.parametrize("bad", [[np.nan, 1.0], []])
def test_fd_inputs_refused(bad):
    with pytest.raises(ValueError):
        fd_jacobian(lambda x: x, bad)


# ------------------------------------------------------------------ vega grid


def _bs_book(grid: VolGrid, options) -> float:
    total = 0.0
    for t, k in options:
        sigma = grid.vol(t, k)
        total += carry_greeks(100.0, k, t, 0.01, 0.0, sigma, True).price
    return total


def test_vega_grid_splits_between_expiry_buckets():
    grid = VolGrid((np.array([1.0, 2.0]), np.array([90.0, 100.0, 110.0])),
                   np.full((2, 3), 0.2), ("expiry", "strike"))
    v = vega_grid(lambda g: _bs_book(g, [(1.5, 100.0)]), grid)
    analytic = carry_greeks(100.0, 100.0, 1.5, 0.01, 0.0, 0.2, True).vega
    np.testing.assert_allclose(v[:, 1], [analytic / 2, analytic / 2], rtol=1e-8)
    assert np.all(v[:, [0, 2]] == 0.0)


def test_vega_grid_sums_to_parallel_vega_2d_and_3d():
    rng = np.random.default_rng(7)
    grid2 = VolGrid((np.array([0.5, 1.0, 2.0]), np.array([80.0, 100.0, 120.0])),
                    0.15 + 0.1 * rng.random((3, 3)))
    book = [(0.7, 95.0), (1.4, 112.0), (3.0, 70.0)]
    v = vega_grid(lambda g: _bs_book(g, book), grid2)
    par = (_bs_book(grid2.shifted(1e-4), book) - _bs_book(grid2.shifted(-1e-4), book)) / 2e-4
    assert v.sum() == pytest.approx(par, rel=1e-8)

    cube = VolGrid((np.array([1.0, 5.0]), np.array([2.0, 10.0]), np.array([-0.01, 0.0, 0.01])),
                   0.005 + 0.002 * rng.random((2, 2, 3)), ("expiry", "tenor", "strike"))

    def swaption_like(g: VolGrid) -> float:  # Bachelier straddle-ish, value smooth in vol
        return sum(math.sqrt(t) * g.vol(t, n, m) * a for t, n, m, a in
                   [(2.0, 5.0, 0.002, 1.0), (4.0, 7.0, -0.004, 2.0)])

    vc = vega_grid(swaption_like, cube)
    parc = (swaption_like(cube.shifted(1e-4)) - swaption_like(cube.shifted(-1e-4))) / 2e-4
    assert vc.shape == (2, 2, 3)
    assert vc.sum() == pytest.approx(parc, rel=1e-10)


def test_grid_weights_partition_of_unity_and_validation():
    grid = VolGrid((np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])), np.full((2, 3), 0.1))
    for pt in [(0.5, 0.0), (1.3, 2.7), (9.0, 9.0)]:
        assert grid.weights(*pt).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        VolGrid((np.array([1.0, 1.0]), np.array([1.0])), np.zeros((2, 1)))
    with pytest.raises(ValueError):
        VolGrid((np.array([1.0]),), np.zeros(1))
    with pytest.raises(ValueError):
        VolGrid((np.array([1.0]), np.array([1.0])), np.array([[np.nan]]))


# ------------------------------------------------------------- smile dynamics


def _linear_smile(slope=-0.001, r=0.0, q=0.0, t=1.0):
    return Smile(100.0, t, r, q, lambda k: 0.20 + slope * (k - 100.0))


def test_worked_example_numbers():
    sm = _linear_smile()
    ss = smile_delta(sm, 100.0, 100.0, True, smile_dynamics="sticky_strike")
    sm_m = smile_delta(sm, 100.0, 100.0, True, smile_dynamics=SmileDynamics.STICKY_MONEYNESS)
    assert ss == pytest.approx(0.5398, abs=5e-5)
    assert sm_m - ss == pytest.approx(0.0397, abs=5e-5)
    assert sm_m == pytest.approx(0.5795, abs=5e-5)


@pytest.mark.parametrize("dyn", list(SmileDynamics))
@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
@pytest.mark.parametrize("is_call", [True, False])
def test_delta_matches_fd_of_surface_shifted_the_matching_way(dyn, strike, is_call):
    # Quadratic smile, non-zero carry. Independent FD: reprice at S +/- h
    # with the vol read off a surface moved explicitly by the dynamic's rule.
    s0, r, q, t = 100.0, 0.03, 0.01, 0.75

    def vol0(k):
        x = math.log(k / 100.0)
        return 0.18 - 0.10 * x + 0.30 * x * x

    sm = Smile(s0, t, r, q, vol0)
    f0 = s0 * math.exp((r - q) * t)

    def shifted_vol(spot):
        if dyn is SmileDynamics.STICKY_STRIKE:
            return vol0(strike)
        if dyn is SmileDynamics.STICKY_MONEYNESS:
            return vol0(strike / spot * s0)
        return vol0(strike / (spot * math.exp((r - q) * t)) * f0)

    def value(spot):
        return carry_greeks(spot, strike, t, r, q, shifted_vol(spot), is_call).price

    h = 1e-4
    fd = (value(s0 * (1 + h)) - value(s0 * (1 - h))) / (2 * s0 * h)
    got = smile_delta(sm, strike, s0, is_call, smile_dynamics=dyn)
    assert got == pytest.approx(fd, abs=1e-6)
    fd_gamma = (value(s0 * (1 + h)) - 2 * value(s0) + value(s0 * (1 - h))) / (s0 * h) ** 2
    assert smile_gamma(sm, strike, s0, is_call, smile_dynamics=dyn) == pytest.approx(fd_gamma, rel=1e-4)
    assert smile_price(sm, strike, s0, is_call, smile_dynamics=dyn) == pytest.approx(value(s0))


def test_sticky_delta_equals_moneyness_without_carry_shock_and_differs_with_one():
    sm = _linear_smile(r=0.02, q=0.01)
    a = sm.vol(105.0, 103.0, smile_dynamics="sticky_moneyness")
    b = sm.vol(105.0, 103.0, smile_dynamics="sticky_delta")
    assert a == pytest.approx(b, abs=1e-15)
    c = sm.vol(105.0, 103.0, smile_dynamics="sticky_delta", r=0.05)
    assert abs(c - a) > 1e-5


def test_unknown_dynamic_refused():
    with pytest.raises(ValueError, match="smile_dynamics"):
        smile_delta(_linear_smile(), 100.0, 100.0, True, smile_dynamics="sticky_mood")


def test_premium_adjusted_delta_closed_form():
    s, k, t, rd, rf, sig = 1.10, 1.15, 0.5, 0.04, 0.02, 0.09
    d1 = (math.log(s / k) + (rd - rf + 0.5 * sig * sig) * t) / (sig * math.sqrt(t))
    d2 = d1 - sig * math.sqrt(t)
    call = premium_adjusted_delta(s, k, t, rd, rf, sig, True)
    put = premium_adjusted_delta(s, k, t, rd, rf, sig, False)
    assert call == pytest.approx(k / s * math.exp(-rd * t) * norm_cdf(d2), rel=1e-12)
    assert put == pytest.approx(-k / s * math.exp(-rd * t) * norm_cdf(-d2), rel=1e-12)


@pytest.mark.parametrize("truth", [SmileDynamics.STICKY_STRIKE, SmileDynamics.STICKY_MONEYNESS])
def test_explain_identifies_the_dynamic_of_a_realised_path(truth):
    # Synthetic path (labelled): prices generated under ``truth``; the
    # explain under the true dynamic leaves the smallest unexplained P&L.
    rng = np.random.default_rng(11)
    sm = _linear_smile(slope=-0.004, t=0.5)
    n, dt = 60, 1.0 / 252
    spots = 100.0 * np.exp(np.cumsum(np.r_[0.0, 0.012 * rng.standard_normal(n - 1)]))
    times = np.arange(n) * dt
    prices = [
        carry_greeks(s, 100.0, sm.expiry - t, 0.0, 0.0,
                     sm.vol(100.0, s, smile_dynamics=truth), True).price
        for s, t in zip(spots, times, strict=True)
    ]
    ranked = rank_smile_dynamics(spots, prices, times, sm, 100.0, True)
    best = min(ranked, key=ranked.get)
    if truth is SmileDynamics.STICKY_STRIKE:
        assert best is truth
    else:  # moneyness and delta coincide with no carry
        assert best in (SmileDynamics.STICKY_MONEYNESS, SmileDynamics.STICKY_DELTA)
        assert ranked[SmileDynamics.STICKY_STRIKE] > 5 * ranked[best]


# -------------------------------------------------------------------- explain


def test_explain_worked_example():
    e = explain(hpl=9.25, time={"theta": -1.0}, delta=[100.0], move=[0.1],
                gamma=np.array([[40.0]]), labels=["x"])
    assert e.rtpl == pytest.approx(9.2)
    assert e.unexplained == pytest.approx(0.05)
    assert e.total == pytest.approx(9.25)
    assert e.as_dict()["risk:x"] == pytest.approx(10.2)


def test_taylor_attribution_sums_to_quadratic_form():
    rng = np.random.default_rng(3)
    d, dx = rng.normal(size=4), rng.normal(size=4)
    g = rng.normal(size=(4, 4))
    attrib = taylor_attribution(d, dx, g)
    sym = 0.5 * (g + g.T)
    assert attrib.sum() == pytest.approx(d @ dx + 0.5 * dx @ sym @ dx)


def test_explain_refuses_nan_and_label_mismatch():
    with pytest.raises(ValueError):
        explain(hpl=float("nan"), time={}, delta=[1.0], move=[1.0], labels=["a"])
    with pytest.raises(ValueError):
        explain(hpl=1.0, time={}, delta=[1.0], move=[1.0], labels=["a", "b"])
