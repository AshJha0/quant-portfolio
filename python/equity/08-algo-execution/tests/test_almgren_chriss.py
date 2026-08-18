"""Almgren-Chriss tests: optimality recursion, TWAP limit, front-loading,
frontier monotonicity, exact share conservation, simulator horse race."""

import numpy as np
import pytest
from scipy import stats

from eq_algo import (ACParams, IntradayConfig, IntradayMarket, ac_cost_moments,
                     ac_kappa, ac_trades, ac_trajectory, efficient_frontier,
                     evaluate_schedules, twap_schedule)


P = ACParams(total_shares=100_000.0, n_slices=20, total_time=1.0,
             sigma=2.0, eta=2.5e-6, gamma=1e-6, epsilon=0.0025)


def test_trajectory_satisfies_discrete_optimality_recursion():
    """x_{j-1} + x_{j+1} = 2*cosh(kappa*tau)*x_j for all interior j —
    the discrete Euler-Lagrange condition of the E + lam*V objective."""
    for lam in (1e-7, 5e-6, 1e-4):
        x = ac_trajectory(P, lam)
        k = ac_kappa(P, lam)
        c = 2.0 * np.cosh(k * P.tau)
        lhs = x[:-2] + x[2:]
        rhs = c * x[1:-1]
        np.testing.assert_allclose(lhs, rhs, rtol=1e-8)
        # equivalent second-difference form with kappa_tilde^2 = lam*sigma^2/eta~
        kt2 = lam * P.sigma**2 / P.eta_tilde
        np.testing.assert_allclose(x[:-2] - 2 * x[1:-1] + x[2:],
                                   kt2 * P.tau**2 * x[1:-1], rtol=1e-8)


def test_lambda_zero_gives_exact_twap():
    n = ac_trades(P, 0.0)
    np.testing.assert_allclose(n, P.total_shares / P.n_slices, rtol=1e-10)
    x = ac_trajectory(P, 0.0)
    np.testing.assert_allclose(
        x, P.total_shares * (1 - np.arange(P.n_slices + 1) / P.n_slices),
        atol=1e-10 * P.total_shares)


def test_lambda_to_zero_limit_converges_to_twap():
    n = ac_trades(P, 1e-16)
    np.testing.assert_allclose(n, P.total_shares / P.n_slices, rtol=1e-6)


def test_trajectory_monotone_decreasing():
    for lam in (0.0, 1e-6, 1e-4):
        x = ac_trajectory(P, lam)
        assert np.all(np.diff(x) < 0)
        assert x[0] == pytest.approx(P.total_shares)
        assert x[-1] == 0.0


def test_higher_risk_aversion_front_loads():
    lams = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    first = [ac_trades(P, lam)[0] for lam in lams]
    assert all(b > a for a, b in zip(first, first[1:]))
    # lambda -> inf: essentially everything in the first slice
    n_inf = ac_trades(P, 10.0)
    assert n_inf[0] > 0.999 * P.total_shares


def test_total_executed_equals_parent_exactly():
    for lam in (0.0, 1e-6, 1e-3, 10.0):
        n = ac_trades(P, lam)
        assert n.sum() == pytest.approx(P.total_shares, rel=1e-12)


def test_kappa_zero_at_lambda_zero_and_increasing():
    assert ac_kappa(P, 0.0) == 0.0
    ks = [ac_kappa(P, lam) for lam in (1e-7, 1e-6, 1e-5, 1e-4)]
    assert all(b > a for a, b in zip(ks, ks[1:]))


def test_efficient_frontier_monotone_tradeoff():
    front = efficient_frontier(P, [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    e = front["expected_cost"].to_numpy()
    v = front["variance"].to_numpy()
    assert np.all(np.diff(e) > 0)      # cost rises with urgency
    assert np.all(np.diff(v) < 0)      # variance falls with urgency
    np.testing.assert_allclose(front["std"], np.sqrt(front["variance"]), rtol=1e-12)


def test_cost_moments_hand_computed():
    p = ACParams(total_shares=10.0, n_slices=2, total_time=1.0, sigma=3.0,
                 eta=0.1, gamma=0.04, epsilon=0.01)
    # trajectory [10, 4, 0]: trades [6, 4]; tau=0.5; eta~ = 0.1 - 0.04*0.5/2 = 0.09
    x = np.array([10.0, 4.0, 0.0])
    e, v = ac_cost_moments(p, x)
    e_hand = 0.5 * 0.04 * 100.0 + 0.01 * 10.0 + (0.09 / 0.5) * (36.0 + 16.0)
    v_hand = 9.0 * 0.5 * (16.0 + 0.0)
    assert e == pytest.approx(e_hand, rel=1e-12)
    assert v == pytest.approx(v_hand, rel=1e-12)


def test_ac_beats_twap_on_cost_variance_on_simulator():
    """Statistical test over seeded replications: for lam > 0 the AC schedule
    has materially lower IS variance than TWAP (front-loading cuts the time
    spent exposed to price noise)."""
    cfg = IntradayConfig(mid0=100.0, day_volume=1e6, n_buckets=26,
                         sigma_daily=0.02, spread_bps=5.0, temp_coef=1.0,
                         perm_coef=0.5, vol_noise=0.2)
    mkt = IntradayMarket(cfg)
    X = 50_000.0
    acp = ACParams(total_shares=X, n_slices=26, sigma=2.0, eta=2.0e-6,
                   gamma=1e-6, epsilon=0.025)
    schedules = {"TWAP": twap_schedule(X, 26), "AC": ac_trades(acp, 5e-6)}
    tab = evaluate_schedules(mkt, schedules, side=1, n_reps=300, seed=1234)
    assert tab.loc["AC", "std_is_bps"] < 0.85 * tab.loc["TWAP", "std_is_bps"]

    # formal variance test on the two IS samples (recomputed to get raw draws)
    from eq_algo import benchmark_slippage
    is_tw, is_ac = [], []
    for r in range(300):
        is_tw.append(benchmark_slippage(
            mkt.execute(schedules["TWAP"], side=1, seed=1234 + r))["vs_arrival_bps"])
        is_ac.append(benchmark_slippage(
            mkt.execute(schedules["AC"], side=1, seed=1234 + r))["vs_arrival_bps"])
    _, pval = stats.levene(is_tw, is_ac)
    assert pval < 0.01


def test_ac_variance_reduction_matches_theory_direction():
    """Model-implied variance ratio AC/TWAP < 1 and the simulator agrees in
    ranking with the analytic frontier."""
    lam = 5e-6
    x_ac = ac_trajectory(P, lam)
    x_tw = P.total_shares * (1 - np.arange(P.n_slices + 1) / P.n_slices)
    e_ac, v_ac = ac_cost_moments(P, x_ac)
    e_tw, v_tw = ac_cost_moments(P, x_tw)
    assert v_ac < v_tw          # less variance...
    assert e_ac > e_tw          # ...bought with more expected cost


def test_params_validation():
    with pytest.raises(ValueError):
        ACParams(total_shares=0.0, n_slices=10)
    with pytest.raises(ValueError):
        ACParams(total_shares=100.0, n_slices=0)
    with pytest.raises(ValueError):
        ACParams(total_shares=100.0, n_slices=10, eta=0.0)
    with pytest.raises(ValueError, match="eta_tilde"):
        # gamma*tau/2 >= eta -> ill-posed
        ACParams(total_shares=100.0, n_slices=1, total_time=1.0,
                 eta=1e-6, gamma=3e-6)
    with pytest.raises(ValueError):
        ac_kappa(P, -1.0)


def test_cost_moments_validation():
    with pytest.raises(ValueError):
        ac_cost_moments(P, np.ones(3))                      # wrong length
    bad = np.linspace(P.total_shares, 5.0, P.n_slices + 1)  # doesn't end at 0
    with pytest.raises(ValueError):
        ac_cost_moments(P, bad)
