"""Spread modelling: OU recovery (OLS & MLE), half-life, RLS hedge tracking."""

import numpy as np
import pandas as pd
import pytest

from fx_pairs.data import synthetic as syn
from fx_pairs.spread import (
    RLSHedge,
    fit_ou_mle,
    fit_ou_ols,
    half_life_days,
    log_spread,
)

DT = 1.0 / 252.0


class TestLogSpread:
    def test_formula_exact(self):
        p1 = np.array([1.10, 1.12])
        p2 = np.array([0.75, 0.76])
        s = log_spread(p1, p2, beta=1.3, alpha=0.05)
        expected = np.log(p1) - 0.05 - 1.3 * np.log(p2)
        assert np.allclose(s, expected, atol=1e-15)


class TestOURecovery:
    def test_ols_recovers_known_parameters(self):
        kappa, sigma = 20.0, 0.05
        s = syn.simulate_ou(8000, kappa, 0.01, sigma, seed=11)
        fit = fit_ou_ols(s)
        assert fit.kappa == pytest.approx(kappa, rel=0.25)
        assert fit.theta == pytest.approx(0.01, abs=0.005)
        assert fit.sigma == pytest.approx(sigma, rel=0.05)

    def test_half_life_consistent_with_kappa(self):
        s = syn.simulate_ou(5000, 25.0, 0.0, 0.05, seed=12)
        fit = fit_ou_ols(s)
        assert fit.half_life == pytest.approx(np.log(2) / (fit.kappa * DT), rel=1e-12)
        # true half-life ln2/(25/252) ~ 7 business days
        assert fit.half_life == pytest.approx(np.log(2) / (25.0 * DT), rel=0.3)

    def test_half_life_days_function(self):
        assert half_life_days(np.log(2.0) / DT) == pytest.approx(1.0)
        assert half_life_days(0.0) == np.inf
        assert half_life_days(-1.0) == np.inf

    def test_mle_matches_ols(self):
        """Conditional Gaussian MLE and OLS coincide for AR(1): cross-check."""
        s = syn.simulate_ou(3000, 15.0, 0.0, 0.04, seed=13)
        ols = fit_ou_ols(s)
        mle = fit_ou_mle(s)
        assert mle.kappa == pytest.approx(ols.kappa, rel=0.02)
        assert mle.theta == pytest.approx(ols.theta, abs=1e-3)
        assert mle.sigma == pytest.approx(ols.sigma, rel=0.02)

    def test_random_walk_has_no_mean_reversion(self):
        rng = np.random.default_rng(14)
        rw = np.cumsum(0.003 * rng.standard_normal(2000))
        fit = fit_ou_ols(rw)
        # phi ~ 1: half-life much longer than the sample or infinite
        assert fit.half_life > 250

    def test_white_noise_reverts_almost_instantly(self):
        rng = np.random.default_rng(15)
        wn = 0.01 * rng.standard_normal(2000)
        fit = fit_ou_ols(wn)
        assert fit.half_life < 1.0


class TestOUValidation:
    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            fit_ou_ols(np.zeros(10))

    def test_constant_spread_raises(self):
        with pytest.raises(ValueError, match="zero variance"):
            fit_ou_ols(np.full(100, 0.5))

    def test_nan_raises(self):
        s = syn.simulate_ou(100, 10.0, 0.0, 0.05, seed=1)
        s[5] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            fit_ou_ols(s)


class TestRLS:
    def test_converges_to_ols_on_stable_relation(self):
        p1, p2, truth = syn.make_cointegrated_pair(n=1500, beta=1.4, alpha=0.2,
                                                   seed=21)
        rls = RLSHedge(lam=1.0, delta=1e8)  # no forgetting = recursive OLS
        path = rls.fit_path(p2, p1)
        lp1, lp2 = np.log(p1.values), np.log(p2.values)
        X = np.column_stack([np.ones(len(lp2)), lp2])
        beta_ols = np.linalg.lstsq(X, lp1, rcond=None)[0]
        assert path["beta"].iloc[-1] == pytest.approx(beta_ols[1], abs=1e-5)
        assert path["alpha"].iloc[-1] == pytest.approx(beta_ols[0], abs=1e-5)

    def test_tracks_hedge_ratio_shift(self):
        """Structural break in beta mid-sample: forgetting-factor RLS follows."""
        n = 1200
        rng = np.random.default_rng(22)
        lp2 = np.cumsum(0.006 * rng.standard_normal(n)) + 0.5
        beta_path = np.where(np.arange(n) < n // 2, 1.0, 1.6)
        lp1 = beta_path * lp2 + 0.002 * rng.standard_normal(n)
        rls = RLSHedge(lam=0.98)
        path = rls.fit_path(pd.Series(np.exp(lp2)), pd.Series(np.exp(lp1)))
        assert path["beta"].iloc[n // 2 - 1] == pytest.approx(1.0, abs=0.15)
        assert path["beta"].iloc[-1] == pytest.approx(1.6, abs=0.15)

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError):
            RLSHedge(lam=0.0)
        with pytest.raises(ValueError):
            RLSHedge(lam=1.1)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            RLSHedge().fit_path(np.ones(10), np.ones(9))
