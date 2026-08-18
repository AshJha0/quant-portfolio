"""Piecewise Almgren-Chriss: closed-form anchor, limits, optimality."""

import numpy as np
import pytest

from fx_algo import (
    EURUSD,
    MarketSimulator,
    ac_closed_form_schedule,
    ac_expected_cost,
    eta_from_depth,
    liquidity_weighted_schedule,
    piecewise_ac_schedule,
    twap_schedule,
)


def test_piecewise_reduces_to_closed_form_when_liquidity_constant():
    X, N, eta, sigma, lam, tau = 100.0, 40, 0.5, 2.0, 0.01, 1.0
    cf = ac_closed_form_schedule(X, N, eta, sigma, lam, tau)
    pw = piecewise_ac_schedule(X, np.full(N, eta), np.full(N, sigma), lam, tau)
    assert np.allclose(cf, pw, atol=1e-9)
    assert cf.sum() == pytest.approx(X, abs=1e-9)


def test_closed_form_lambda_zero_is_twap():
    cf = ac_closed_form_schedule(60.0, 12, eta=1.0, sigma=3.0, risk_aversion=0.0)
    assert np.allclose(cf, 5.0)


def test_closed_form_zero_sigma_is_twap():
    cf = ac_closed_form_schedule(60.0, 12, eta=1.0, sigma=0.0, risk_aversion=5.0)
    assert np.allclose(cf, 5.0)


def test_lambda_zero_gives_liquidity_weighted_twap_analog():
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    eta = eta_from_depth(sim.depth_bucket)
    pw = piecewise_ac_schedule(500.0, eta, sim.sigma_bucket_pips, 0.0)
    lw = liquidity_weighted_schedule(500.0, sim.depth_bucket)
    assert np.allclose(pw, lw, atol=1e-8)


def test_dp_solution_sums_exactly_to_parent():
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    eta = eta_from_depth(sim.depth_bucket, k_eta=0.02)
    for lam in (0.0, 1e-6, 1e-5, 1e-4):
        pw = piecewise_ac_schedule(500.0, eta, sim.sigma_bucket_pips, lam)
        assert pw.sum() == pytest.approx(500.0, abs=1e-10)
        assert (pw >= -1e-12).all()  # one-sided by default


def test_risk_aversion_front_loads():
    N = 50
    eta = np.full(N, 0.5)
    sigma = np.full(N, 2.0)
    pw = piecewise_ac_schedule(100.0, eta, sigma, 0.01)
    x = 100.0 - np.cumsum(pw)
    assert np.all(np.diff(x) < 0)  # holdings strictly decreasing
    assert pw[0] > pw[-1]
    tw = piecewise_ac_schedule(100.0, eta, sigma, 0.0)
    assert pw[0] > tw[0]


def test_liquidity_aware_cost_leq_naive_ac_under_time_varying_liquidity():
    sim = MarketSimulator(EURUSD, dt_minutes=15.0)
    eta = eta_from_depth(sim.depth_bucket, k_eta=0.02)
    sigma = sim.sigma_bucket_pips
    for lam in (1e-6, 1e-5, 1e-4):
        aware = piecewise_ac_schedule(500.0, eta, sigma, lam)
        naive = ac_closed_form_schedule(
            500.0, sim.n_buckets, float(eta.mean()), float(sigma.mean()), lam
        )
        c_aware = ac_expected_cost(aware, eta, sigma, lam)
        c_naive = ac_expected_cost(naive, eta, sigma, lam)
        assert c_aware <= c_naive + 1e-9
    # and strictly better for a meaningful lambda
    aware = piecewise_ac_schedule(500.0, eta, sigma, 1e-5)
    naive = ac_closed_form_schedule(500.0, sim.n_buckets, float(eta.mean()), float(sigma.mean()), 1e-5)
    assert ac_expected_cost(aware, eta, sigma, 1e-5) < 0.99 * ac_expected_cost(naive, eta, sigma, 1e-5)


def test_optimality_against_perturbations():
    N = 30
    rng = np.random.default_rng(0)
    eta = 0.2 + rng.uniform(0, 1, N)
    sigma = 1.0 + rng.uniform(0, 2, N)
    lam = 0.005
    star = piecewise_ac_schedule(50.0, eta, sigma, lam, allow_sells=True)
    c_star = ac_expected_cost(star, eta, sigma, lam)
    for _ in range(20):
        d = rng.standard_normal(N)
        d -= d.mean()  # stay on the constraint surface
        pert = star + 0.1 * d
        assert ac_expected_cost(pert, eta, sigma, lam) >= c_star - 1e-9


def test_expected_cost_manual():
    n = np.array([3.0, 2.0, 1.0])
    # X=6, x = [3, 1, 0]
    c = ac_expected_cost(n, eta=2.0, sigma=1.5, risk_aversion=0.1, tau=1.0)
    manual = 2.0 * (9 + 4 + 1) + 0.1 * 1.5**2 * (9 + 1 + 0)
    assert c == pytest.approx(manual)


def test_single_bucket_and_sell_side():
    assert piecewise_ac_schedule(5.0, np.array([1.0]), np.array([1.0]), 0.1).tolist() == [5.0]
    buy = piecewise_ac_schedule(100.0, np.full(10, 0.5), np.full(10, 2.0), 0.01)
    sell = piecewise_ac_schedule(-100.0, np.full(10, 0.5), np.full(10, 2.0), 0.01)
    assert np.allclose(buy, -sell)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        ac_closed_form_schedule(1.0, 10, eta=0.0, sigma=1.0, risk_aversion=0.1)
    with pytest.raises(ValueError):
        piecewise_ac_schedule(1.0, np.array([1.0, -1.0]), np.array([1.0, 1.0]), 0.1)
    with pytest.raises(ValueError):
        piecewise_ac_schedule(1.0, np.array([1.0]), np.array([-1.0]), 0.1)
    with pytest.raises(ValueError):
        piecewise_ac_schedule(1.0, np.array([1.0]), np.array([1.0]), -0.1)
    with pytest.raises(ValueError):
        ac_expected_cost(np.array([1.0]), 1.0, 1.0, 0.1, tau=0.0)


def test_active_set_clamps_thin_buckets_at_high_lambda():
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    eta = eta_from_depth(sim.depth_bucket, k_eta=0.02)
    pw = piecewise_ac_schedule(500.0, eta, sim.sigma_bucket_pips, 1e-4)
    assert (pw >= 0).all()
    assert (pw < 1e-12).sum() > 0  # some late buckets unused
    assert pw.sum() == pytest.approx(500.0, abs=1e-10)
