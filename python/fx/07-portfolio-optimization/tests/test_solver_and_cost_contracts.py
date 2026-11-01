"""Solver-failure, PD-input, cost, CVaR and hedging contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import OptimizeResult

import fx_port as fp
import fx_port.mvo as mvo_mod

INDEFINITE = pd.DataFrame(np.array([[1.0, 0.9, 0.7], [0.9, 1.0, 0.9], [0.7, 0.9, 0.6]]),
                          index=list("abc"), columns=list("abc"))
MU = pd.Series([0.01, 0.02, 0.03], index=list("abc"))


# ------------------------------------------------ closed forms need PD input

@pytest.mark.parametrize("fn", [
    lambda: fp.min_variance_weights(INDEFINITE),
    lambda: fp.tangency_weights(MU, INDEFINITE),
    lambda: fp.frontier_weights(MU, INDEFINITE, 0.02),
    lambda: fp.dollar_neutral_weights(MU, INDEFINITE),
])
def test_closed_forms_reject_indefinite_covariance(fn):
    # eigenvalue -0.123: the old np.linalg.solve path returned {0.2, 1.4, -0.6}
    assert np.linalg.eigvalsh(INDEFINITE.to_numpy()).min() < 0
    with pytest.raises(ValueError, match="positive definite"):
        fn()


# -------------------------------------------- solver failure raises, never drops

def _failing_minimize(*args, **kwargs):
    x0 = np.asarray(args[1] if len(args) > 1 else kwargs["x0"], dtype=float)
    return OptimizeResult(x=x0, success=False, message="forced failure", fun=0.0)


def test_max_utility_raises_on_solver_failure(monkeypatch):
    monkeypatch.setattr(mvo_mod, "minimize", _failing_minimize)
    cov = pd.DataFrame(np.eye(3) * 1e-4, index=list("abc"), columns=list("abc"))
    with pytest.raises(ValueError, match="SLSQP failed"):
        fp.max_utility(MU * 1e-2, cov, gamma=1.0, sum_to=1.0, bounds=(0.0, 0.6))


def test_min_variance_slsqp_raises_on_solver_failure(monkeypatch):
    monkeypatch.setattr(mvo_mod, "minimize", _failing_minimize)
    cov = pd.DataFrame(np.eye(3) * 1e-4, index=list("abc"), columns=list("abc"))
    with pytest.raises(ValueError, match="SLSQP failed"):
        fp.min_variance_slsqp(cov, sum_to=1.0, gross_limit=1.5)


def test_efficient_frontier_does_not_silently_drop_points(monkeypatch):
    cov = pd.DataFrame(np.diag([1.0, 2.0, 3.0]) * 1e-4, index=list("abc"), columns=list("abc"))
    real = fp.min_variance_slsqp
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if k.get("target_return") is not None and calls["n"] == 4:
            raise ValueError("SLSQP failed in min_variance_slsqp: forced")
        return real(*a, **k)

    monkeypatch.setattr(mvo_mod, "min_variance_slsqp", flaky)
    with pytest.raises(ValueError, match="SLSQP failed"):
        fp.efficient_frontier(MU * 1e-3, cov, n_points=6, sum_to=1.0, gross_limit=1.5)


# ------------------------------------------------------------- carry sizing

def test_carry_sizing_negative_mean_gives_zero_size():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(-0.0003, 0.005, 2000))  # about -7.6%/yr
    assert fp.carry_sizing(r, cvar_limit=0.004, max_leverage=2.0) == (0.0, 0.0)


def test_carry_sizing_positive_mean_unchanged():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0004, 0.005, 2000))
    s, c = fp.carry_sizing(r, cvar_limit=0.004, max_leverage=10.0)
    assert s == pytest.approx(0.004 / fp.empirical_cvar(r), rel=1e-12)
    assert c == pytest.approx(0.004, rel=1e-12)


# --------------------------------------------------------- sort-based CVaR

def _ru_bruteforce(returns, alpha):
    losses = -np.asarray(returns, float)
    cand = np.unique(losses)
    vals = cand + np.maximum(losses[None, :] - cand[:, None], 0.0).mean(axis=1) / (1 - alpha)
    return float(vals.min())


@settings(max_examples=80, deadline=None)
@given(seed=st.integers(0, 10_000), n=st.integers(1, 400),
       alpha=st.sampled_from([0.5, 0.9, 0.95, 0.975, 0.99]),
       ties=st.booleans())
def test_sort_cvar_equals_rockafellar_uryasev_minimum(seed, n, alpha, ties):
    rng = np.random.default_rng(seed)
    x = rng.standard_t(3, n) * 0.01
    if ties:
        x = np.round(x, 3)
    got = fp.empirical_cvar(x, alpha)
    ref = _ru_bruteforce(x, alpha)
    # both are sums of at most n terms of size ~0.1: a few ulps of n * 0.1
    assert got == pytest.approx(ref, rel=1e-12, abs=1e-14)


def test_cvar_scales_to_large_scenario_sets():
    # the old S x S matrix needed ~8 * 2e5^2 = 320 GB here
    x = np.random.default_rng(2).normal(size=200_000)
    v = fp.empirical_cvar(x, 0.975)
    assert v == pytest.approx(2.338, abs=0.03)  # N(0,1) ES at 97.5%


# ------------------------------------------------------- sleeve-level costs

def test_style_returns_charges_currency_level_turnover():
    idx = pd.bdate_range("2020-01-01", periods=4)
    cols = ["A", "B", "C", "D"]
    tot = pd.DataFrame(0.0, index=idx, columns=cols)
    sig = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1], [4, 3, 2, 1]],
                       index=idx, columns=cols, dtype=float)
    gross_r, w = fp.style_returns(tot, sig, gross=2.0)
    net_r, _ = fp.style_returns(tot, sig, gross=2.0, cost_bps=10.0)
    turn = fp.sleeve_turnover(w)
    # day 1 builds the book (gross 2), day 2 holds, day 3 flips (4 units)
    np.testing.assert_allclose(turn.to_numpy(), [0.0, 2.0, 0.0, 4.0], atol=1e-15)
    np.testing.assert_allclose((gross_r - net_r).to_numpy(), turn.to_numpy() * 1e-3,
                               atol=1e-18)
    with pytest.raises(ValueError):
        fp.style_returns(tot, sig, cost_bps=-1.0)


# ------------------------------------------------------------------ hedging

def _hedge_market(seed=0, t=2000):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=t)
    fx = pd.DataFrame(rng.normal(0, 0.005, (t, 2)), index=idx, columns=["EUR", "JPY"])
    x = pd.Series([0.3, 0.2], index=["EUR", "JPY"])
    true_h = np.array([1.0, 0.4])
    ru = pd.Series(rng.normal(0, 0.008, t), index=idx) + fx @ (true_h * x.to_numpy())
    return ru, fx, x


def test_hedged_returns_include_forward_carry():
    ru, fx, x = _hedge_market()
    carry = pd.Series([0.0001, -0.0002], index=["EUR", "JPY"])
    spot_only = fp.hedged_returns(ru, fx, x, 1.0)
    with_carry = fp.hedged_returns(ru, fx, x, 1.0, forward_carry=carry)
    expected_shift = 0.3 * 0.0001 + 0.2 * -0.0002
    np.testing.assert_allclose((with_carry - spot_only).to_numpy(), expected_shift, atol=1e-18)
    assert with_carry.var() == pytest.approx(spot_only.var(), rel=1e-12)


def test_hedge_ratio_standard_errors_match_monte_carlo():
    est, ses = [], []
    for seed in range(300):
        ru, fx, x = _hedge_market(seed, t=500)
        inf = fp.hedge_ratio_inference(ru, fx, x)
        est.append(inf.hedge_ratios.to_numpy())
        ses.append(inf.std_errors.to_numpy())
    mc_sd = np.std(est, axis=0, ddof=1)
    mean_se = np.mean(ses, axis=0)
    # sd of a 300-draw sample sd is about sd / sqrt(600) ~ 4%: 15% is ~3.5 sigma
    np.testing.assert_allclose(mean_se, mc_sd, rtol=0.15)
    np.testing.assert_allclose(np.mean(est, axis=0), [1.0, 0.4], atol=0.05)


def test_out_of_sample_hedge_uses_only_training_block():
    ru, fx, x = _hedge_market(3)
    a = fp.out_of_sample_hedge(ru, fx, x, train_frac=0.5)
    ru2 = ru.copy()
    ru2.iloc[1000:] += np.random.default_rng(9).normal(0, 0.05, len(ru) - 1000)
    b = fp.out_of_sample_hedge(ru2, fx, x, train_frac=0.5)
    pd.testing.assert_series_equal(a.hedge_ratios_train, b.hedge_ratios_train)
    assert a.reduction_optimal > 0
