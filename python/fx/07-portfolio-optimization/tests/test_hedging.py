"""Tests for optimal currency hedging: closed form, brute force, safe havens."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from fx_port import hedged_returns, optimal_hedge_ratios, variance_decomposition
from fx_port.data import make_equity_portfolio


@pytest.fixture(scope="module")
def market():
    return make_equity_portfolio(seed=1, n_days=1500)


def test_closed_form_matches_brute_force(market):
    h_closed = optimal_hedge_ratios(
        market.unhedged_returns, market.fx_returns, market.exposures
    )

    def var_of(h):
        r = hedged_returns(
            market.unhedged_returns,
            market.fx_returns,
            market.exposures,
            pd.Series(h, index=market.exposures.index),
        )
        return float(r.var(ddof=1))

    res = minimize(var_of, np.ones(len(market.exposures)), method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-16, "maxiter": 20000})
    assert res.success
    assert np.max(np.abs(res.x - h_closed.to_numpy())) < 1e-6


def test_hedged_variance_leq_unhedged(market):
    rep = variance_decomposition(
        market.unhedged_returns, market.fx_returns, market.exposures
    )
    assert rep.var_optimal <= rep.var_unhedged + 1e-16
    assert rep.var_optimal <= rep.var_full + 1e-16
    assert rep.reduction_optimal >= rep.reduction_full - 1e-12
    assert rep.reduction_optimal == pytest.approx(
        1.0 - rep.var_optimal / rep.var_unhedged, rel=1e-12
    )


def test_safe_haven_underhedged(market):
    rep = variance_decomposition(
        market.unhedged_returns, market.fx_returns, market.exposures
    )
    # JPY and CHF are negatively correlated with the equity factor: their
    # unhedged exposure is itself a hedge, so the optimal ratio is < 1.
    assert rep.hedge_ratios["JPY"] < 1.0
    assert rep.hedge_ratios["CHF"] < 1.0
    # and materially below the risk-on currency's ratio
    assert rep.hedge_ratios["JPY"] < rep.hedge_ratios["AUD"]


def test_full_hedge_optimal_when_no_local_fx_correlation():
    # r_u = x' r_fx exactly (no local return): optimal hedge removes ALL fx
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=400)
    fx = pd.DataFrame(0.006 * rng.standard_normal((400, 3)), index=idx,
                      columns=["EUR", "JPY", "AUD"])
    x = pd.Series([0.3, 0.4, 0.3], index=fx.columns)
    r_u = (fx * x).sum(axis=1)
    h = optimal_hedge_ratios(r_u, fx, x)
    assert np.allclose(h, 1.0, atol=1e-10)
    hedged = hedged_returns(r_u, fx, x, h)
    assert float(hedged.var(ddof=1)) < 1e-24


def test_full_hedge_removes_fx_leg():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2020-01-01", periods=300)
    fx = pd.DataFrame(0.005 * rng.standard_normal((300, 2)), index=idx,
                      columns=["EUR", "JPY"])
    local = pd.Series(0.01 * rng.standard_normal(300), index=idx)
    x = pd.Series([0.5, 0.5], index=fx.columns)
    r_u = local + (fx * x).sum(axis=1)
    hedged = hedged_returns(r_u, fx, x, 1.0)  # scalar broadcast
    assert np.allclose(hedged, local, atol=1e-15)


def test_scalar_and_series_hedge_ratios_agree(market):
    h_scalar = hedged_returns(
        market.unhedged_returns, market.fx_returns, market.exposures, 0.5
    )
    h_series = hedged_returns(
        market.unhedged_returns,
        market.fx_returns,
        market.exposures,
        pd.Series(0.5, index=market.exposures.index),
    )
    pd.testing.assert_series_equal(h_scalar, h_series, check_names=False)


def test_zero_exposure_raises(market):
    bad = market.exposures.copy()
    bad.iloc[0] = 0.0
    with pytest.raises(ValueError, match="exposure"):
        optimal_hedge_ratios(market.unhedged_returns, market.fx_returns, bad)


def test_misaligned_inputs_raise(market):
    with pytest.raises(ValueError, match="match"):
        optimal_hedge_ratios(
            market.unhedged_returns,
            market.fx_returns[market.fx_returns.columns[::-1]],
            market.exposures,
        )
    with pytest.raises(ValueError, match="index"):
        optimal_hedge_ratios(
            market.unhedged_returns.iloc[:-5], market.fx_returns, market.exposures
        )


def test_missing_hedge_ratio_currency_raises(market):
    partial = pd.Series(1.0, index=market.exposures.index[:-1])
    with pytest.raises(ValueError, match="missing"):
        hedged_returns(
            market.unhedged_returns, market.fx_returns, market.exposures, partial
        )


def test_singular_fx_covariance_raises():
    idx = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(0)
    a = 0.005 * rng.standard_normal(100)
    fx = pd.DataFrame({"EUR": a, "DUP": a}, index=idx)  # perfectly collinear
    x = pd.Series([0.5, 0.5], index=fx.columns)
    r_u = pd.Series(0.01 * rng.standard_normal(100), index=idx)
    with pytest.raises(ValueError, match="singular"):
        optimal_hedge_ratios(r_u, fx, x)
