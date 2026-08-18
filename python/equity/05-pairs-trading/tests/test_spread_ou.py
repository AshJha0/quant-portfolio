"""Spread construction, OU parameter recovery, RLS hedge-ratio tracking."""

import numpy as np
import pandas as pd
import pytest

from eq_pairs.cointegration import hedge_ratio
from eq_pairs.data import business_index, simulate_ou
from eq_pairs.spread import (
    compute_spread,
    fit_ou_mle,
    fit_ou_ols,
    half_life_from_kappa,
    rls_hedge_ratio,
    rolling_ols_hedge_ratio,
)


class TestComputeSpread:
    def test_array_spread(self):
        y = np.array([10.0, 12.0, 14.0])
        x = np.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(
            compute_spread(y, x, beta=2.0, alpha=1.0), [1.0, 1.0, 1.0]
        )

    def test_series_preserves_index(self):
        idx = business_index(3)
        y = pd.Series([10.0, 12.0, 14.0], index=idx)
        x = pd.Series([4.0, 5.0, 6.0], index=idx)
        s = compute_spread(y, x, 2.0)
        assert isinstance(s, pd.Series) and s.index.equals(idx)

    def test_mismatched_index_raises(self):
        y = pd.Series([1.0, 2.0], index=business_index(2))
        x = pd.Series([1.0, 2.0], index=business_index(2, start="2020-01-01"))
        with pytest.raises(ValueError, match="indices differ"):
            compute_spread(y, x, 1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            compute_spread(np.arange(3.0), np.arange(4.0), 1.0)


class TestOURecovery:
    KAPPA, SIGMA, MU = 0.05, 1.2, 0.7

    def _path(self, n=20000, seed=40):
        return simulate_ou(n, kappa=self.KAPPA, sigma=self.SIGMA, mu=self.MU, seed=seed)

    def test_ols_parameter_recovery(self):
        fit = fit_ou_ols(self._path())
        assert fit.mean_reverting
        assert fit.kappa == pytest.approx(self.KAPPA, rel=0.15)
        assert fit.sigma == pytest.approx(self.SIGMA, rel=0.05)
        assert fit.mu == pytest.approx(self.MU, abs=0.5)

    def test_mle_parameter_recovery(self):
        fit = fit_ou_mle(self._path())
        assert fit.kappa == pytest.approx(self.KAPPA, rel=0.15)
        assert fit.sigma == pytest.approx(self.SIGMA, rel=0.05)

    def test_ols_and_mle_agree(self):
        """Conditional MLE point estimates coincide with OLS for the AR(1);
        only variance normalisation differs (O(1/n))."""
        path = self._path(n=5000, seed=41)
        a = fit_ou_ols(path)
        b = fit_ou_mle(path)
        assert b.kappa == pytest.approx(a.kappa, rel=1e-3)
        assert b.mu == pytest.approx(a.mu, abs=1e-3)
        assert b.sigma == pytest.approx(a.sigma, rel=5e-3)

    def test_half_life_formula(self):
        assert half_life_from_kappa(np.log(2.0) / 10.0) == pytest.approx(10.0, abs=1e-12)
        fit = fit_ou_ols(self._path())
        assert fit.half_life == pytest.approx(np.log(2.0) / fit.kappa, abs=1e-12)

    def test_half_life_nonpositive_kappa_inf(self):
        assert half_life_from_kappa(0.0) == np.inf
        assert half_life_from_kappa(-0.1) == np.inf

    def test_random_walk_gives_huge_or_infinite_half_life(self):
        rw = simulate_ou(2000, kappa=0.0, sigma=1.0, seed=42)
        fit = fit_ou_ols(rw)
        # OLS on a random walk: b_hat ~ 1 - O(1/n) -> half-life explodes
        assert (not fit.mean_reverting) or fit.half_life > 150

    def test_explosive_series_flagged_not_mean_reverting(self):
        s = 1.02 ** np.arange(300) + simulate_ou(300, 0.5, 0.01, seed=43)
        fit = fit_ou_ols(s)
        assert not fit.mean_reverting
        assert fit.kappa == 0.0
        assert fit.half_life == np.inf
        assert fit.stationary_std == np.inf

    def test_mle_on_non_mean_reverting_falls_back(self):
        s = 1.02 ** np.arange(300)
        fit = fit_ou_mle(s)
        assert not fit.mean_reverting
        assert fit.method == "mle"

    def test_stationary_std(self):
        fit = fit_ou_ols(self._path())
        expected = fit.sigma / np.sqrt(2.0 * fit.kappa)
        assert fit.stationary_std == pytest.approx(expected, abs=1e-12)

    def test_validations(self):
        with pytest.raises(ValueError, match="n >= 10"):
            fit_ou_ols(np.arange(5.0))
        with pytest.raises(ValueError, match="NaN"):
            fit_ou_ols(np.array([np.nan] + list(np.arange(20.0))))
        with pytest.raises(ValueError, match="zero variance"):
            fit_ou_ols(np.full(50, 3.0))
        with pytest.raises(ValueError, match="dt"):
            fit_ou_ols(simulate_ou(100, 0.1, 1.0, seed=1), dt=0.0)

    def test_oscillatory_series_raises(self):
        s = np.tile([1.0, -1.0], 50) + 0.01 * np.random.default_rng(2).standard_normal(100)
        with pytest.raises(ValueError, match="oscillatory"):
            fit_ou_ols(s)


class TestRLSHedgeRatio:
    def test_converges_to_static_beta(self):
        rng = np.random.default_rng(50)
        x = 100 + np.cumsum(rng.normal(0, 1, 2000))
        y = 5.0 + 1.5 * x + rng.normal(0, 0.5, 2000)
        est = rls_hedge_ratio(y, x, lam=0.999)
        assert est["beta"].iloc[-1] == pytest.approx(1.5, abs=0.02)
        assert est["alpha"].iloc[-1] == pytest.approx(5.0, abs=2.0)

    def test_tracks_drifting_beta_bounded_error(self):
        """True beta drifts 1.0 -> 2.0; RLS with forgetting must track it
        with bounded error after burn-in, where static OLS cannot."""
        rng = np.random.default_rng(51)
        n = 3000
        x = 100 + np.cumsum(rng.normal(0, 1, n))
        beta_true = np.linspace(1.0, 2.0, n)
        y = beta_true * x + rng.normal(0, 0.5, n)
        est = rls_hedge_ratio(y, x, lam=0.98, intercept=False)["beta"].to_numpy()
        burn = 300
        track_err = np.abs(est[burn:] - beta_true[burn:])
        assert track_err.mean() < 0.05
        assert track_err.max() < 0.10
        # static full-sample OLS is off by construction at the endpoints
        beta_static, _, _ = hedge_ratio(y, x, intercept=False)
        assert abs(beta_static - beta_true[-1]) > 5 * np.abs(
            est[-1] - beta_true[-1]
        )

    def test_intercept_plus_short_memory_degrades_identification(self):
        """Documented caveat: with a short memory the constant and the slope
        are nearly collinear (prices vary little vs their level), so the
        with-intercept RLS tracks far worse than the no-intercept fit."""
        rng = np.random.default_rng(51)
        n = 3000
        x = 100 + np.cumsum(rng.normal(0, 1, n))
        beta_true = np.linspace(1.0, 2.0, n)
        y = beta_true * x + rng.normal(0, 0.5, n)
        err_int = np.abs(
            rls_hedge_ratio(y, x, lam=0.98)["beta"].to_numpy()[300:]
            - beta_true[300:]
        ).mean()
        err_noint = np.abs(
            rls_hedge_ratio(y, x, lam=0.98, intercept=False)["beta"].to_numpy()[300:]
            - beta_true[300:]
        ).mean()
        assert err_noint < 0.05 < err_int

    def test_lam_one_matches_expanding_ols(self):
        rng = np.random.default_rng(52)
        x = 100 + np.cumsum(rng.normal(0, 1, 500))
        y = 2.0 + 0.8 * x + rng.normal(0, 0.3, 500)
        est = rls_hedge_ratio(y, x, lam=1.0, delta=1e8)
        beta_ols, alpha_ols, _ = hedge_ratio(y, x)
        assert est["beta"].iloc[-1] == pytest.approx(beta_ols, abs=1e-4)
        assert est["alpha"].iloc[-1] == pytest.approx(alpha_ols, abs=1e-2)

    def test_no_intercept_mode(self):
        rng = np.random.default_rng(53)
        x = 100 + np.cumsum(rng.normal(0, 1, 800))
        y = 1.3 * x + rng.normal(0, 0.3, 800)
        est = rls_hedge_ratio(y, x, lam=0.999, intercept=False)
        assert "alpha" not in est.columns
        assert est["beta"].iloc[-1] == pytest.approx(1.3, abs=0.01)

    def test_validations(self):
        y = np.arange(10.0)
        with pytest.raises(ValueError, match="lam"):
            rls_hedge_ratio(y, y, lam=0.0)
        with pytest.raises(ValueError, match="lam"):
            rls_hedge_ratio(y, y, lam=1.5)
        with pytest.raises(ValueError, match="delta"):
            rls_hedge_ratio(y, y, delta=-1.0)
        with pytest.raises(ValueError, match="length mismatch"):
            rls_hedge_ratio(y, np.arange(9.0))


class TestRollingOLS:
    def test_warmup_nan_then_matches_windowed_ols(self):
        rng = np.random.default_rng(54)
        idx = business_index(300)
        x = pd.Series(100 + np.cumsum(rng.normal(0, 1, 300)), index=idx)
        y = pd.Series(1.0 + 1.7 * x.to_numpy() + rng.normal(0, 0.4, 300), index=idx)
        est = rolling_ols_hedge_ratio(y, x, window=60)
        assert est["beta"].iloc[:59].isna().all()
        beta_last, _, _ = hedge_ratio(y.iloc[-60:], x.iloc[-60:])
        assert est["beta"].iloc[-1] == pytest.approx(beta_last, abs=1e-10)

    def test_validations(self):
        idx = business_index(10)
        y = pd.Series(np.arange(10.0), index=idx)
        with pytest.raises(ValueError, match="window"):
            rolling_ols_hedge_ratio(y, y, window=2)
        x = pd.Series(np.arange(10.0), index=business_index(10, start="2021-01-01"))
        with pytest.raises(ValueError, match="indices differ"):
            rolling_ols_hedge_ratio(y, x, window=5)
