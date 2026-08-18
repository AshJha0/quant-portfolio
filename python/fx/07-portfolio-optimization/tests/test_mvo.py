"""Tests for closed-form and SLSQP mean-variance optimization."""

import numpy as np
import pandas as pd
import pytest

from fx_port import (
    dollar_neutral_weights,
    efficient_frontier,
    frontier_weights,
    lw_shrinkage,
    max_utility,
    min_variance_slsqp,
    min_variance_weights,
    tangency_weights,
    total_log_returns,
)
from fx_port.data import make_panel


@pytest.fixture(scope="module")
def market():
    panel = make_panel(seed=2, n_days=1000)
    ret = total_log_returns(panel.spots, panel.rates).total
    sigma, _ = lw_shrinkage(ret)
    mu = ret.mean()
    return mu, sigma


def test_min_variance_closed_form_identity(market):
    _, sigma = market
    w = min_variance_weights(sigma)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    # KKT: Sigma w must be proportional to ones (equal marginal variance)
    g = sigma.to_numpy() @ w.to_numpy()
    assert np.max(np.abs(g - g.mean())) < 1e-12 * max(1.0, np.abs(g.mean()))
    # direct formula check
    inv1 = np.linalg.solve(sigma.to_numpy(), np.ones(len(sigma)))
    assert np.allclose(w, inv1 / inv1.sum(), atol=1e-12)


def test_tangency_closed_form_identity(market):
    _, sigma = market
    # positive risk-premium vector: guarantees a long tangency portfolio
    mu = pd.Series(
        0.05 * np.sqrt(np.diag(sigma.to_numpy())) + 1e-4, index=sigma.index
    )
    w = tangency_weights(mu, sigma)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    # w proportional to Sigma^-1 mu
    z = np.linalg.solve(sigma.to_numpy(), mu.to_numpy())
    assert np.allclose(w, z / z.sum(), atol=1e-12)
    # tangency maximises Sharpe among frontier portfolios: compare a few
    sr_tan = (mu @ w) / np.sqrt(w @ sigma.to_numpy() @ w)
    for tgt in np.linspace(mu.min(), mu.max(), 7):
        wf = frontier_weights(mu, sigma, float(tgt)).to_numpy()
        sr = (mu @ wf) / np.sqrt(wf @ sigma.to_numpy() @ wf)
        assert sr <= sr_tan + 1e-10


def test_frontier_weights_hit_target(market):
    mu, sigma = market
    tgt = float(mu.mean())
    w = frontier_weights(mu, sigma, tgt)
    assert float(mu @ w) == pytest.approx(tgt, abs=1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_frontier_at_minvar_mean_recovers_minvar(market):
    mu, sigma = market
    wmv = min_variance_weights(sigma)
    tgt = float(mu @ wmv)
    w = frontier_weights(mu, sigma, tgt)
    assert np.allclose(w, wmv, atol=1e-10)


def test_slsqp_matches_closed_form_min_variance(market):
    _, sigma = market
    closed = min_variance_weights(sigma)
    num = min_variance_slsqp(sigma)
    assert num.success
    assert np.max(np.abs(num.weights - closed)) < 1e-5


def test_slsqp_matches_closed_form_frontier(market):
    mu, sigma = market
    tgt = float(mu.mean())
    closed = frontier_weights(mu, sigma, tgt)
    num = min_variance_slsqp(sigma, target_return=tgt, mu=mu)
    assert num.success
    assert np.max(np.abs(num.weights - closed)) < 1e-5


def test_dollar_neutral_closed_form(market):
    mu, sigma = market
    gamma = 5.0
    w = dollar_neutral_weights(mu, sigma, gamma=gamma)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    # KKT: gamma*Sigma*w - mu = -lambda * ones (constant vector)
    g = gamma * sigma.to_numpy() @ w.to_numpy() - mu.to_numpy()
    assert np.max(np.abs(g - g.mean())) < 1e-12


def test_dollar_neutral_gamma_scaling(market):
    mu, sigma = market
    w1 = dollar_neutral_weights(mu, sigma, gamma=1.0)
    w4 = dollar_neutral_weights(mu, sigma, gamma=4.0)
    assert np.allclose(w1 / 4.0, w4, atol=1e-14)


def test_max_utility_slsqp_matches_dollar_neutral(market):
    mu, sigma = market
    gamma = 20.0
    closed = dollar_neutral_weights(mu, sigma, gamma=gamma)
    num = max_utility(mu, sigma, gamma=gamma, sum_to=0.0)
    assert num.success
    assert np.max(np.abs(num.weights - closed)) < 1e-6


def test_gross_leverage_respected(market):
    mu, sigma = market
    res = max_utility(mu, sigma, gamma=1.0, sum_to=0.0, gross_limit=2.0)
    assert res.success
    assert res.weights.abs().sum() <= 2.0 + 1e-8
    assert res.weights.sum() == pytest.approx(0.0, abs=1e-8)
    # unconstrained solution has larger gross (i.e. the constraint binds)
    free = dollar_neutral_weights(mu, sigma, gamma=1.0)
    assert free.abs().sum() > 2.0


def test_gross_zero_returns_flat_book(market):
    mu, sigma = market
    res = max_utility(mu, sigma, sum_to=0.0, gross_limit=0.0)
    assert (res.weights == 0).all()
    assert res.volatility == 0.0


def test_infeasible_budget_raises(market):
    mu, sigma = market
    with pytest.raises(ValueError, match="infeasible"):
        max_utility(mu, sigma, sum_to=1.0, gross_limit=0.5)


def test_frontier_monotonic_vol(market):
    mu, sigma = market
    front = efficient_frontier(mu, sigma, n_points=15)
    vols = front["volatility"].to_numpy()
    assert np.all(np.diff(vols) >= -1e-12)  # vol non-decreasing above min-var
    assert np.all(np.diff(front["target_return"].to_numpy()) > 0)


def test_frontier_constrained_monotonic(market):
    mu, sigma = market
    front = efficient_frontier(
        mu, sigma, n_points=8, sum_to=0.0, gross_limit=2.0
    )
    assert len(front) >= 4
    vols = front["volatility"].to_numpy()
    assert np.all(np.diff(vols) >= -1e-8)
    # every point respects the gross budget
    w = front[[c for c in front.columns if c not in ("target_return", "volatility")]]
    assert (w.abs().sum(axis=1) <= 2.0 + 1e-6).all()


def test_singular_sigma_raises():
    sigma = pd.DataFrame(np.zeros((2, 2)), index=["A", "B"], columns=["A", "B"])
    with pytest.raises(ValueError, match="singular"):
        min_variance_weights(sigma)


def test_dimension_mismatch_raises(market):
    mu, sigma = market
    with pytest.raises(ValueError, match="dimensions"):
        tangency_weights(mu.iloc[:-1], sigma)


def test_invalid_gamma_raises(market):
    mu, sigma = market
    with pytest.raises(ValueError, match="gamma"):
        dollar_neutral_weights(mu, sigma, gamma=0.0)
    with pytest.raises(ValueError, match="gamma"):
        max_utility(mu, sigma, gamma=-1.0)


def test_tangency_negative_denominator_raises(market):
    _, sigma = market
    mu_bad = pd.Series(-0.01, index=sigma.index)
    with pytest.raises(ValueError, match="tangency"):
        tangency_weights(mu_bad, sigma)


def test_single_asset_min_variance():
    sigma = pd.DataFrame([[0.01]], index=["EUR"], columns=["EUR"])
    w = min_variance_weights(sigma)
    assert w["EUR"] == pytest.approx(1.0, abs=1e-14)
