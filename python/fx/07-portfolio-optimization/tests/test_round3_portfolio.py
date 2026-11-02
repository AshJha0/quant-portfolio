"""Round 3 portfolio models: HRP, resampled and robust MVO, cost QP,
factor risk model, tracking-error optimisation, HAC Sharpe tests.
Each test is an oracle (closed form, identity or known limit)."""

import numpy as np
import pytest
from fx_port.cost_qp import no_trade_band_solution, rebalance
from fx_port.factor_model import decompose_risk, fit_factor_model
from fx_port.hrp import hrp_weights
from fx_port.robust_mvo import mean_variance_weights, resampled_weights, robust_weights
from fx_port.sharpe_tests import sharpe_diff_bootstrap, sharpe_diff_gradient, sharpe_diff_hac
from fx_port.tracking_error import (
    information_ratio_bound,
    max_ir_active_weights,
    te_constrained_weights,
    tracking_error,
)


def _cov(n: int, seed: int) -> np.ndarray:
    a = np.random.default_rng(seed).normal(size=(n, n))
    return 0.04 * (a @ a.T / n + np.eye(n))


MU = np.array([0.05, 0.07, 0.06, 0.09, 0.04])
COV = _cov(5, 0)

# --- HRP -------------------------------------------------------------------


def test_hrp_diagonal_equals_inverse_variance():
    var = np.array([0.04, 0.01, 0.09, 0.0225, 0.0625])
    w = hrp_weights(np.diag(var))
    np.testing.assert_allclose(w, (1 / var) / (1 / var).sum(), rtol=1e-13)


def test_hrp_two_assets_closed_form():
    c = np.array([[0.04, 0.006], [0.006, 0.01]])
    w = hrp_weights(c)
    np.testing.assert_allclose(w, [0.01 / 0.05, 0.04 / 0.05], rtol=1e-13)


def test_hrp_long_only_and_handles_singular_cov():
    b = np.random.default_rng(1).normal(size=(6, 2))
    c = b @ b.T + 1e-12 * np.eye(6)
    w = hrp_weights(c)
    assert np.all(w > 0) and w.sum() == pytest.approx(1.0)


# --- resampled and robust MVO ----------------------------------------------


def test_mv_closed_form_satisfies_foc():
    w = mean_variance_weights(MU, COV, 3.0)
    g = MU - 3.0 * COV @ w
    assert w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(g, g[0], atol=1e-14)


def test_resampled_converges_to_classic():
    w = resampled_weights(MU, COV, 3.0, n_obs=200_000, n_resamples=20, seed=2)
    np.testing.assert_allclose(w, mean_variance_weights(MU, COV, 3.0), atol=0.03)
    assert w.sum() == pytest.approx(1.0)


def test_resampled_long_only_sums_to_one():
    w = resampled_weights(MU, COV, 3.0, n_obs=60, n_resamples=20, long_only=True, seed=3)
    assert np.all(w >= 0) and w.sum() == pytest.approx(1.0)


def test_robust_kappa_zero_is_classic():
    np.testing.assert_allclose(
        robust_weights(MU, COV, 3.0, 0.0, n_obs=60), mean_variance_weights(MU, COV, 3.0)
    )


def test_robust_equals_classic_at_effective_risk_aversion():
    n_obs, kappa, lam = 60, 1.5, 3.0
    w = robust_weights(MU, COV, lam, kappa, n_obs=n_obs)
    lam_eff = lam + kappa / (n_obs * np.sqrt(w @ (COV / n_obs) @ w))
    np.testing.assert_allclose(w, mean_variance_weights(MU, COV, lam_eff), atol=1e-6)


# --- cost / turnover QP ----------------------------------------------------


def test_no_trade_band_matches_qp():
    var = np.array([0.04, 0.02, 0.09, 0.03])
    mu = np.array([0.03, 0.02, 0.10, 0.001])
    w0 = np.array([0.2, 0.3, 0.1, 0.4])
    cost = np.array([0.002, 0.05, 0.01, 0.003])
    for impact in (0.0, 0.5):
        res = rebalance(mu, np.diag(var), w0, 2.0, cost, impact, budget=False)
        np.testing.assert_allclose(
            res.weights, no_trade_band_solution(mu, var, w0, 2.0, cost, impact), atol=1e-7
        )
    # Asset 1 sits inside its band and must not trade.
    assert res.weights[1] == pytest.approx(w0[1], abs=1e-8)


def test_zero_cost_budget_qp_is_classic_mv():
    w0 = np.full(5, 0.2)
    res = rebalance(MU, COV, w0, 3.0)
    np.testing.assert_allclose(res.weights, mean_variance_weights(MU, COV, 3.0), atol=1e-6)


def test_turnover_cap_binds():
    w0 = np.full(5, 0.2)
    free = rebalance(MU, COV, w0, 3.0, cost=0.001)
    capped = rebalance(MU, COV, w0, 3.0, cost=0.001, max_turnover=0.5 * free.turnover)
    assert capped.turnover == pytest.approx(0.5 * free.turnover, abs=1e-7)
    assert capped.weights.sum() == pytest.approx(1.0)
    assert capped.objective <= free.objective + 1e-12


def test_box_bounds_respected():
    res = rebalance(MU, COV, np.full(5, 0.2), 1.0, cost=0.0005, bounds=(0.0, 0.3))
    assert res.weights.min() >= -1e-9 and res.weights.max() <= 0.3 + 1e-9


# --- factor model ----------------------------------------------------------


def test_factor_model_recovers_loadings_and_euler_identities():
    rng = np.random.default_rng(4)
    t, n, k = 20_000, 6, 2
    f = rng.normal(0, 0.01, size=(t, k))
    b_true = rng.normal(1.0, 0.3, size=(n, k))
    r = 0.0002 + f @ b_true.T + rng.normal(0, 0.005, size=(t, n))
    m = fit_factor_model(r, f)
    np.testing.assert_allclose(m.loadings, b_true, atol=0.03)
    np.testing.assert_allclose(m.specific_var, 0.005**2, rtol=0.05)
    w = rng.dirichlet(np.ones(n))
    d = decompose_risk(m, w)
    assert d.factor_contrib.sum() == pytest.approx(d.factor_var, rel=1e-13)
    assert d.asset_contrib.sum() == pytest.approx(d.total_vol, rel=1e-13)
    assert d.total_vol**2 == pytest.approx(w @ m.covariance @ w, rel=1e-13)


def test_factor_model_matches_lstsq_exactly():
    rng = np.random.default_rng(5)
    f = rng.normal(size=(50, 3))
    r = rng.normal(size=(50, 4))
    m = fit_factor_model(r, f)
    x = np.column_stack([np.ones(50), f])
    coef = np.linalg.solve(x.T @ x, x.T @ r)
    np.testing.assert_allclose(m.loadings, coef[1:].T, atol=1e-12)


# --- tracking error --------------------------------------------------------


def test_closed_form_active_weights():
    alpha = np.array([0.01, -0.005, 0.02, 0.0, 0.004])
    a = max_ir_active_weights(alpha, COV, 0.03)
    assert a.sum() == pytest.approx(0.0, abs=1e-14)
    assert np.sqrt(a @ COV @ a) == pytest.approx(0.03, rel=1e-13)
    assert alpha @ a == pytest.approx(0.03 * information_ratio_bound(alpha, COV), rel=1e-12)


def test_constrained_te_matches_closed_form_when_bounds_slack():
    b = np.full(5, 0.2)
    alpha = np.array([0.01, -0.005, 0.02, 0.0, 0.004])
    w = te_constrained_weights(alpha, COV, b, 0.01, bounds=None)
    np.testing.assert_allclose(w - b, max_ir_active_weights(alpha, COV, 0.01), atol=1e-6)
    wl = te_constrained_weights(alpha, COV, b, 0.5)
    assert wl.min() >= -1e-9 and tracking_error(wl, b, COV) <= 0.5 + 1e-9


# --- Sharpe-difference tests -----------------------------------------------


def test_hac_lag0_matches_memmel_under_normality():
    rng = np.random.default_rng(6)
    t, rho = 200_000, 0.6
    z = rng.multivariate_normal([0.0, 0.0], [[1, rho], [rho, 1]], size=t)
    r1, r2 = 0.05 + 1.0 * z[:, 0], 0.02 + 0.5 * z[:, 1]
    s1, s2 = 0.05, 0.04
    memmel = np.sqrt((2 - 2 * rho + 0.5 * (s1**2 + s2**2 - 2 * s1 * s2 * rho**2)) / t)
    assert sharpe_diff_hac(r1, r2).se == pytest.approx(memmel, rel=0.02)


def test_gradient_matches_finite_difference():
    mu, gamma = np.array([0.03, -0.01]), np.array([0.02, 0.05])

    def f(v):
        return v[0] / np.sqrt(v[2] - v[0] ** 2) - v[1] / np.sqrt(v[3] - v[1] ** 2)

    v0 = np.concatenate([mu, gamma])
    h = 1e-7
    num = np.array([(f(v0 + h * e) - f(v0 - h * e)) / (2 * h) for e in np.eye(4)])
    np.testing.assert_allclose(sharpe_diff_gradient(mu, gamma), num, rtol=1e-6)


def test_hac_widens_under_autocorrelation_and_bootstrap_detects_difference():
    rng = np.random.default_rng(7)
    e = rng.normal(size=(3000, 2))
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, len(e)):
        x[i] = 0.6 * x[i - 1] + e[i]
    r1, r2 = 0.02 + x[:, 0], 0.02 + x[:, 1]
    assert sharpe_diff_hac(r1, r2, lags=20).se > 1.3 * sharpe_diff_hac(r1, r2, lags=0).se
    big = rng.normal(size=(2000, 2))
    assert sharpe_diff_bootstrap(0.3 + big[:, 0], big[:, 1], n_boot=300) < 0.01
    assert sharpe_diff_bootstrap(big[:, 0], big[:, 1], n_boot=300) > 0.05
