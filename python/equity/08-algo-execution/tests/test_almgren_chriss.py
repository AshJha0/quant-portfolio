"""Almgren-Chriss tests: optimality recursion, TWAP limit, front-loading,
frontier monotonicity, exact share conservation, simulator horse race."""

import numpy as np
import pytest
from eq_algo import (
    ACParams,
    IntradayConfig,
    IntradayMarket,
    ac_cost_moments,
    ac_kappa,
    ac_trades,
    ac_trajectory,
    efficient_frontier,
    evaluate_schedules,
    twap_schedule,
)
from scipy import stats

P = ACParams(total_shares=100_000.0, n_slices=20, total_time=1.0,
             sigma=2.0, eta=2.5e-6, gamma=1e-6, epsilon=0.0025)


def test_trajectory_satisfies_discrete_optimality_recursion():
    """x_{j-1} + x_{j+1} = 2*cosh(kappa*tau)*x_j for all interior j:
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
    assert all(b > a for a, b in zip(first, first[1:], strict=False))
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
    assert all(b > a for a, b in zip(ks, ks[1:], strict=False))


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


class TestClosedFormIsTheDiscreteMinimiser:
    """TEST-10: ``ac_trajectory`` minimises ``E + lambda V`` over admissible
    trajectories (``x_0 = X``, ``x_N = 0``).

    Almgren & Chriss (2001) eq. (16)-(18): with ``kappa`` from
    ``cosh(kappa tau) = 1 + kappa_tilde^2 tau^2 / 2`` the sinh trajectory is
    the *exact* discrete-time minimiser, not a continuous-time approximation.
    Random perturbations of the interior holdings must therefore never lower
    the objective, and the objective's gradient must vanish at the optimum.
    """

    PARAMS = ACParams(total_shares=250_000.0, n_slices=25, total_time=2.0,
                      sigma=0.8, eta=2.5e-6, gamma=2.0e-7, epsilon=0.0025)

    @staticmethod
    def _objective(params, x, lam):
        e, v = ac_cost_moments(params, x)
        return e + lam * v

    @pytest.mark.parametrize("lam", [0.0, 1e-9, 1e-7, 1e-6, 1e-5, 1e-3, 1.0])
    def test_random_perturbations_never_improve(self, lam):
        params = self.PARAMS
        x_opt = ac_trajectory(params, lam)
        j_opt = self._objective(params, x_opt, lam)
        rng = np.random.default_rng(int(1e6 * lam) + 11)
        for scale in (1e-4, 1e-2, 0.1, 0.5):
            for _ in range(20):
                delta = rng.normal(size=x_opt.size) * scale * params.total_shares
                delta[0] = delta[-1] = 0.0  # keep the boundary conditions
                x = x_opt + delta
                j = self._objective(params, x, lam)
                assert j >= j_opt - 1e-9 * abs(j_opt), (lam, scale, j, j_opt)

    @pytest.mark.parametrize("lam", [1e-8, 1e-6, 1e-4])
    def test_first_order_condition_holds(self, lam):
        # Central-difference gradient w.r.t. each interior holding is ~0
        # relative to the curvature scale.
        params = self.PARAMS
        x_opt = ac_trajectory(params, lam)
        for k in range(1, x_opt.size - 1):
            # Keep the perturbed path monotone so the epsilon*|n_k| term
            # stays on its smooth branch (the kink is *at* the optimum only
            # when h exceeds the local step).
            h = 0.1 * min(x_opt[k - 1] - x_opt[k], x_opt[k] - x_opt[k + 1])
            if h < 1e-6 * params.total_shares:
                continue
            up, dn = x_opt.copy(), x_opt.copy()
            up[k] += h
            dn[k] -= h
            grad = (self._objective(params, up, lam) - self._objective(params, dn, lam)) / (2 * h)
            curv = (self._objective(params, up, lam) - 2 * self._objective(params, x_opt, lam)
                    + self._objective(params, dn, lam)) / h**2
            # |grad| should be O(rounding), i.e. far below the curvature times h.
            assert abs(grad) < 1e-6 * curv * h + 1e-9 * abs(self._objective(params, x_opt, lam)) / h

    def test_front_loading_increases_with_risk_aversion(self):
        params = self.PARAMS
        prev = ac_trajectory(params, 0.0)
        for lam in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
            cur = ac_trajectory(params, lam)
            # every interior holding is lower (sold sooner) at higher lambda
            assert np.all(cur[1:-1] <= prev[1:-1] + 1e-9)
            prev = cur
        # the large-lambda limit is "sell everything in the first slice"
        extreme = ac_trajectory(params, 1e3)
        assert extreme[1] < 1e-6 * params.total_shares

    def test_twap_is_the_zero_risk_aversion_optimum(self):
        params = self.PARAMS
        x = ac_trajectory(params, 0.0)
        np.testing.assert_allclose(-np.diff(x), params.total_shares / params.n_slices, rtol=1e-12)
