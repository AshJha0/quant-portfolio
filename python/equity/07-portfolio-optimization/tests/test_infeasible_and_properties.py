"""Infeasible-constraint behaviour and portfolio-level invariants
(weight-sum-to-one, frontier convexity) added in the hardening pass."""

import numpy as np
import pytest

from eq_port.mvo import (
    efficient_frontier,
    max_sharpe_constrained,
    min_variance_constrained,
    portfolio_vol,
    target_return_portfolio,
    target_risk_portfolio,
)
from eq_port.risk_parity import erc_weights, risk_contributions


def _spd(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n + 10, n)) * 0.01
    cov = a.T @ a / (n + 10)
    mu = rng.uniform(0.0002, 0.001, n)
    return mu, cov


def test_infeasible_target_return_above_box_max_raises() -> None:
    mu, cov = _spd(4)
    # Long-only, weights in [0, 1]: max achievable return is max(mu) at most
    # sum(hi * mu+) = sum(mu). Ask for far more.
    with pytest.raises(ValueError, match="infeasible target return"):
        target_return_portfolio(mu, cov, target=float(mu.sum()) + 1.0, bounds=(0.0, 1.0))


def test_infeasible_target_return_below_box_min_raises() -> None:
    mu, cov = _spd(4, seed=1)
    with pytest.raises(ValueError, match="infeasible target return"):
        target_return_portfolio(mu, cov, target=-1.0, bounds=(0.0, 1.0))


def test_infeasible_budget_vs_bounds_raises() -> None:
    # 4 assets capped at 10% each cannot sum to a budget of 1.
    mu, cov = _spd(4, seed=2)
    with pytest.raises(ValueError):
        min_variance_constrained(cov, bounds=(0.0, 0.1), budget=1.0)


def test_infeasible_vol_cap_below_min_variance_raises() -> None:
    mu, cov = _spd(5, seed=3)
    w_mv = min_variance_constrained(cov, bounds=(0.0, 1.0))
    min_vol = portfolio_vol(w_mv, cov)
    with pytest.raises(ValueError):
        target_risk_portfolio(mu, cov, target_vol=min_vol / 100.0, bounds=(0.0, 1.0))


def test_weights_sum_to_one_property_across_solvers() -> None:
    for seed in range(5):
        mu, cov = _spd(6, seed=seed)
        for w in (
            min_variance_constrained(cov, bounds=(0.0, 1.0)),
            max_sharpe_constrained(mu, cov, bounds=(0.0, 1.0)),
            erc_weights(cov),
        ):
            assert w.sum() == pytest.approx(1.0, abs=1e-8)
            assert np.all(w >= -1e-9)  # long-only respected


def test_custom_budget_respected() -> None:
    mu, cov = _spd(5, seed=7)
    w = min_variance_constrained(cov, bounds=(0.0, 1.0), budget=0.5)
    assert w.sum() == pytest.approx(0.5, abs=1e-8)


def test_analytic_frontier_variance_is_quadratic_in_target() -> None:
    # Merton: sigma^2(m) = (A m^2 - 2 B m + C)/D — second differences of the
    # variance on an equally spaced target grid must be constant.
    mu, cov = _spd(5, seed=11)
    fr = efficient_frontier(mu, cov, n_points=30)
    var = fr.vols**2
    d2 = np.diff(var, 2)
    assert np.allclose(d2, d2[0], rtol=1e-6, atol=1e-16)
    assert np.all(d2 > 0)  # strictly convex


def test_frontier_vols_convex_in_return() -> None:
    # sqrt of a positive quadratic is convex: check discrete convexity.
    mu, cov = _spd(6, seed=13)
    fr = efficient_frontier(mu, cov, n_points=25)
    d2 = np.diff(fr.vols, 2)
    assert np.all(d2 >= -1e-12)


def test_erc_risk_contributions_property_random_matrices() -> None:
    for seed in range(4):
        _, cov = _spd(5, seed=seed + 20)
        w = erc_weights(cov)
        rc = risk_contributions(w, cov)
        assert np.allclose(rc, rc.mean(), rtol=1e-6)
        # Euler identity holds exactly.
        assert rc.sum() == pytest.approx(float(w @ cov @ w), rel=1e-12)
