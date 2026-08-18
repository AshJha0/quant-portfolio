"""Tests for ERC allocation, Euler identity and vol targeting."""

import numpy as np
import pandas as pd
import pytest

from fx_port import (
    carry_signal,
    erc_weights,
    lw_shrinkage,
    momentum_signal,
    portfolio_vol,
    risk_contributions,
    style_returns,
    total_log_returns,
    value_signal,
    vol_target,
)
from fx_port.data import make_panel


@pytest.fixture(scope="module")
def sigma():
    panel = make_panel(seed=9, n_days=900)
    ret = total_log_returns(panel.spots, panel.rates).total
    return lw_shrinkage(ret)[0]


def test_erc_equal_contributions(sigma):
    w = erc_weights(sigma)
    rc = risk_contributions(w, sigma)
    n = len(sigma)
    assert np.max(np.abs(rc - 1.0 / n)) < 1e-8
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert (w > 0).all()


def test_erc_two_asset_diagonal_analytic():
    sigma = pd.DataFrame(np.diag([0.04, 0.01]), index=["A", "B"], columns=["A", "B"])
    w = erc_weights(sigma)
    # diagonal case: w proportional to 1/vol -> (1/0.2, 1/0.1) ~ (1/3, 2/3)
    assert w["A"] == pytest.approx(1.0 / 3.0, abs=1e-10)
    assert w["B"] == pytest.approx(2.0 / 3.0, abs=1e-10)


def test_erc_risk_budgets_respected(sigma):
    n = len(sigma)
    budget = np.linspace(1, 2, n)
    budget = budget / budget.sum()
    w = erc_weights(sigma, budget=budget)
    rc = risk_contributions(w, sigma)
    assert np.max(np.abs(rc.to_numpy() - budget)) < 1e-8


def test_euler_identity(sigma):
    rng = np.random.default_rng(4)
    w = pd.Series(rng.standard_normal(len(sigma)), index=sigma.index)
    contrib = risk_contributions(w, sigma, normalize=False)
    var = float(w @ sigma.to_numpy() @ w)
    assert contrib.sum() == pytest.approx(var, rel=1e-12)


def test_normalized_contributions_sum_to_one(sigma):
    w = erc_weights(sigma)
    assert risk_contributions(w, sigma).sum() == pytest.approx(1.0, rel=1e-12)


def test_erc_zero_diag_raises():
    sigma = pd.DataFrame(np.diag([0.01, 0.0]), index=["A", "PEG"], columns=["A", "PEG"])
    with pytest.raises(ValueError, match="diagonal"):
        erc_weights(sigma)


def test_erc_invalid_budget_raises(sigma):
    with pytest.raises(ValueError, match="budget"):
        erc_weights(sigma, budget=np.zeros(len(sigma)))
    with pytest.raises(ValueError, match="budget"):
        erc_weights(sigma, budget=np.ones(3))


def test_vol_target_exact_ex_ante(sigma):
    w = erc_weights(sigma)
    scaled = vol_target(w, sigma, target_vol=0.10)
    assert portfolio_vol(scaled, sigma) == pytest.approx(0.10, rel=1e-12)


def test_vol_target_invalid(sigma):
    w = erc_weights(sigma)
    with pytest.raises(ValueError, match="target_vol"):
        vol_target(w, sigma, target_vol=0.0)
    zero_w = pd.Series(0.0, index=sigma.index)
    with pytest.raises(ValueError, match="zero-variance"):
        vol_target(zero_w, sigma, target_vol=0.1)


def test_risk_contributions_zero_variance_raises(sigma):
    w = pd.Series(0.0, index=sigma.index)
    with pytest.raises(ValueError, match="variance"):
        risk_contributions(w, sigma)


def test_erc_across_styles():
    panel = make_panel(seed=9, n_days=900)
    dec = total_log_returns(panel.spots, panel.rates)
    ccys = list(panel.spots.columns)
    idx = dec.total.index
    styles = pd.DataFrame(
        {
            "carry": style_returns(dec.total, carry_signal(panel.rates, ccys).reindex(idx))[0],
            "momentum": style_returns(dec.total, momentum_signal(panel.spots).reindex(idx))[0],
            "value": style_returns(dec.total, value_signal(panel.spots, panel.ppp).reindex(idx))[0],
        }
    ).iloc[260:]
    style_cov = styles.cov()
    w = erc_weights(style_cov)
    rc = risk_contributions(w, style_cov)
    assert np.max(np.abs(rc - 1.0 / 3.0)) < 1e-8
    assert set(w.index) == {"carry", "momentum", "value"}
