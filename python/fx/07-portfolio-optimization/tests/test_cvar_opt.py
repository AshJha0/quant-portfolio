"""Tests for the Rockafellar-Uryasev CVaR LP and skew-aware sizing."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar

from fx_port import (
    carry_sizing,
    empirical_cvar,
    empirical_var,
    max_return_cvar_constrained,
    min_cvar,
)


def _toy_scenarios(seed=0, n=500):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    a = 0.004 * rng.standard_normal(n) + 0.0004          # benign asset
    b = 0.003 * rng.standard_normal(n) + 0.0016          # carry-like: best mean...
    crash = rng.random(n) < 0.02
    b[crash] -= np.abs(rng.normal(0.03, 0.01, crash.sum()))  # ...crash tail
    return pd.DataFrame({"benign": a, "carry": b}, index=idx)


def test_empirical_cvar_hand_computed():
    # 10 scenarios, alpha=0.8 -> tail mass 2 scenarios: mean of 2 worst losses
    r = np.array([0.01, 0.02, -0.05, 0.005, -0.01, 0.03, -0.04, 0.0, 0.015, 0.02])
    cvar = empirical_cvar(r, alpha=0.8)
    assert cvar == pytest.approx((0.05 + 0.04) / 2.0, abs=1e-12)


def test_empirical_var_hand_computed():
    r = np.array([0.01, -0.02, -0.05, 0.03, 0.0])
    assert empirical_var(r, alpha=0.75) == pytest.approx(0.02, abs=1e-15)


def test_empirical_cvar_equals_ru_minimisation():
    rng = np.random.default_rng(7)
    r = 0.01 * rng.standard_normal(400)
    alpha = 0.95
    losses = -r

    def ru(z):
        return z + np.maximum(losses - z, 0.0).mean() / (1 - alpha)

    res = minimize_scalar(ru, bounds=(losses.min(), losses.max()), method="bounded",
                          options={"xatol": 1e-12})
    assert empirical_cvar(r, alpha) == pytest.approx(ru(res.x), abs=1e-9)


def test_cvar_positive_homogeneity():
    rng = np.random.default_rng(1)
    r = 0.01 * rng.standard_normal(300) - 0.001
    assert empirical_cvar(3.0 * r, 0.9) == pytest.approx(
        3.0 * empirical_cvar(r, 0.9), rel=1e-12
    )


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        empirical_cvar(np.array([0.01]), alpha=1.0)
    with pytest.raises(ValueError, match="alpha"):
        min_cvar(_toy_scenarios(), alpha=0.0)


def test_min_cvar_matches_exhaustive_grid():
    scen = _toy_scenarios()
    res = min_cvar(scen, alpha=0.95, sum_to=1.0, max_weight=1.0)
    grid = np.linspace(0.0, 1.0, 2001)
    r = scen.to_numpy()
    best = min(
        (empirical_cvar(r @ np.array([g, 1 - g]), 0.95), g) for g in grid
    )
    assert res.cvar <= best[0] + 1e-9  # LP at least as good as the fine grid
    assert abs(res.weights.iloc[0] - best[1]) < 2e-3  # and at the same point
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_ru_objective_equals_empirical_cvar_at_optimum():
    scen = _toy_scenarios(seed=3)
    res = min_cvar(scen, alpha=0.95, sum_to=1.0, max_weight=1.0)
    emp = empirical_cvar(scen.to_numpy() @ res.weights.to_numpy(), 0.95)
    assert res.cvar == pytest.approx(emp, abs=1e-9)


def test_cvar_constraint_binds_and_cuts_tail():
    scen = _toy_scenarios(seed=5)
    alpha = 0.95
    # unconstrained mean-chaser under the same budget: piles into carry
    uncon = max_return_cvar_constrained(
        scen, alpha=alpha, cvar_limit=10.0, sum_to=None, gross_limit=2.0
    )
    tail_uncon = empirical_cvar(scen.to_numpy() @ uncon.weights.to_numpy(), alpha)
    limit = 0.5 * tail_uncon
    con = max_return_cvar_constrained(
        scen, alpha=alpha, cvar_limit=limit, sum_to=None, gross_limit=2.0
    )
    tail_con = empirical_cvar(scen.to_numpy() @ con.weights.to_numpy(), alpha)
    assert tail_con <= limit + 1e-9          # constraint respected...
    assert con.cvar == pytest.approx(limit, abs=1e-8)  # ...and binding
    assert tail_con < tail_uncon             # tail actually cut
    assert con.expected_return <= uncon.expected_return + 1e-12
    # binding RU objective equals the empirical CVaR of the solution
    assert con.cvar == pytest.approx(tail_con, abs=1e-9)


def test_cvar_constrained_cuts_carry_weight():
    scen = _toy_scenarios(seed=5)
    uncon = max_return_cvar_constrained(
        scen, alpha=0.95, cvar_limit=10.0, sum_to=None, gross_limit=2.0
    )
    con = max_return_cvar_constrained(
        scen, alpha=0.95, cvar_limit=0.004, sum_to=None, gross_limit=2.0
    )
    # mean-variance-blind sizing loves the carry asset; the CVaR budget cuts it
    assert con.weights["carry"] < uncon.weights["carry"]


def test_gross_limit_respected_in_lp():
    scen = _toy_scenarios(seed=9)
    res = max_return_cvar_constrained(
        scen, alpha=0.95, cvar_limit=0.01, sum_to=0.0, gross_limit=1.5
    )
    assert res.weights.abs().sum() <= 1.5 + 1e-8
    assert res.weights.sum() == pytest.approx(0.0, abs=1e-9)


def test_min_cvar_return_floor():
    scen = _toy_scenarios(seed=2)
    mu = scen.mean()
    floor = float(mu.max()) * 0.9
    res = min_cvar(scen, alpha=0.95, sum_to=1.0, max_weight=1.0, return_floor=floor)
    assert float(mu @ res.weights) >= floor - 1e-10


def test_infeasible_lp_raises():
    idx = pd.bdate_range("2020-01-01", periods=50)
    scen = pd.DataFrame({"loser": np.full(50, -0.05)}, index=idx)
    with pytest.raises(ValueError, match="infeasible|failed"):
        max_return_cvar_constrained(
            scen, alpha=0.9, cvar_limit=0.001, sum_to=1.0, gross_limit=1.0
        )


def test_negative_gross_limit_raises():
    with pytest.raises(ValueError, match="gross"):
        min_cvar(_toy_scenarios(), gross_limit=-1.0)


def test_carry_sizing_homogeneity():
    rng = np.random.default_rng(11)
    r = pd.Series(0.005 * rng.standard_normal(500) + 0.0005)
    r.iloc[::97] -= 0.03
    base = empirical_cvar(r, 0.95)
    assert base > 0
    limit = 0.4 * base
    s, cvar_scaled = carry_sizing(r, alpha=0.95, cvar_limit=limit, max_leverage=10.0)
    assert s == pytest.approx(limit / base, rel=1e-12)
    assert cvar_scaled == pytest.approx(limit, rel=1e-12)
    # leverage cap kicks in when the limit is generous
    s2, _ = carry_sizing(r, alpha=0.95, cvar_limit=100.0, max_leverage=3.0)
    assert s2 == 3.0
    with pytest.raises(ValueError, match="max_leverage"):
        carry_sizing(r, max_leverage=-1.0)
