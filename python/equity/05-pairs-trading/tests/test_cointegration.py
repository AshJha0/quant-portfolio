"""Engle-Granger, from-scratch ADF vs statsmodels, MacKinnon critical values."""

import numpy as np
import pytest
from statsmodels.tsa.adfvalues import mackinnoncrit
from statsmodels.tsa.stattools import adfuller

from eq_pairs.cointegration import (
    adf_test,
    engle_granger,
    hedge_ratio,
    mackinnon_crit,
)
from eq_pairs.data import cointegrated_pair, correlated_random_walks, simulate_ou


class TestHedgeRatio:
    def test_exact_linear_recovery(self):
        x = np.linspace(50, 150, 200)
        y = 1.0 + 2.0 * x
        beta, alpha, resid = hedge_ratio(y, x)
        assert beta == pytest.approx(2.0, abs=1e-10)
        assert alpha == pytest.approx(1.0, abs=1e-8)
        assert np.max(np.abs(resid)) < 1e-8

    def test_no_intercept(self):
        x = np.linspace(50, 150, 200)
        y = 3.0 * x
        beta, alpha, _ = hedge_ratio(y, x, intercept=False)
        assert beta == pytest.approx(3.0, abs=1e-10)
        assert alpha == 0.0

    def test_zero_variance_raises(self):
        with pytest.raises(ValueError, match="zero variance"):
            hedge_ratio(np.arange(50.0), np.full(50, 7.0))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            hedge_ratio(np.arange(10.0), np.arange(9.0))


class TestADFvsStatsmodels:
    """The from-scratch ADF must reproduce statsmodels.adfuller exactly."""

    @pytest.mark.parametrize("lag", [0, 1, 4, 8])
    def test_fixed_lag_stat_matches(self, lag):
        rng = np.random.default_rng(11)
        y = 100 + np.cumsum(rng.standard_normal(600))
        mine = adf_test(y, regression="c", lags=lag)
        stat_sm, _, _, nobs_sm, _ = adfuller(
            y, maxlag=lag, regression="c", autolag=None
        )
        assert mine.stat == pytest.approx(stat_sm, abs=1e-8)
        assert mine.nobs == nobs_sm

    @pytest.mark.parametrize("regression", ["n", "c", "ct"])
    def test_autolag_aic_matches(self, regression):
        rng = np.random.default_rng(12)
        # ARMA-ish series so AIC picks a non-trivial lag
        e = rng.standard_normal(500)
        y = np.empty(500)
        y[0] = e[0]
        for t in range(1, 500):
            y[t] = 0.5 * y[t - 1] + e[t] + 0.4 * e[t - 1]
        mine = adf_test(y, regression=regression)
        stat_sm, _, lags_sm, _, _, _ = adfuller(
            y, regression=regression, autolag="AIC"
        )[:6]
        assert mine.lags == lags_sm
        assert mine.stat == pytest.approx(stat_sm, abs=1e-8)

    def test_critical_values_match_statsmodels(self):
        rng = np.random.default_rng(13)
        y = np.cumsum(rng.standard_normal(400))
        mine = adf_test(y, regression="c", lags=2)
        res = adfuller(y, maxlag=2, regression="c", autolag=None)
        for lvl in ("1%", "5%", "10%"):
            assert mine.crit[lvl] == pytest.approx(res[4][lvl], abs=1e-10)

    def test_rejects_stationary_series(self):
        s = simulate_ou(1500, kappa=0.2, sigma=1.0, seed=14)
        res = adf_test(s, regression="c")
        assert res.reject("1%")

    def test_fails_to_reject_random_walk(self):
        rng = np.random.default_rng(15)
        y = np.cumsum(rng.standard_normal(1500))
        res = adf_test(y, regression="c")
        assert not res.reject("10%")

    def test_validations(self):
        with pytest.raises(ValueError, match="zero variance"):
            adf_test(np.full(100, 5.0))
        with pytest.raises(ValueError, match="NaN"):
            adf_test(np.array([1.0, np.nan] + [2.0] * 60))
        with pytest.raises(ValueError, match="regression"):
            adf_test(np.random.default_rng(0).standard_normal(100), regression="x")
        with pytest.raises(ValueError, match="lags"):
            adf_test(np.random.default_rng(0).standard_normal(100), lags=-1)
        with pytest.raises(ValueError, match="too short"):
            adf_test(np.random.default_rng(0).standard_normal(12), lags=8)


class TestMacKinnonCriticalValues:
    """EG (N=2) values are NOT plain ADF (N=1) values — the classic mistake."""

    def test_asymptotic_values_are_published_ones(self):
        plain = mackinnon_crit(1, "c", np.inf)
        eg = mackinnon_crit(2, "c", np.inf)
        assert plain["5%"] == pytest.approx(-2.86154, abs=1e-5)
        assert eg["5%"] == pytest.approx(-3.33613, abs=1e-5)
        assert eg["1%"] == pytest.approx(-3.89644, abs=1e-5)

    @pytest.mark.parametrize("nobs", [100, 250, 1000])
    @pytest.mark.parametrize("n_series", [1, 2])
    def test_finite_sample_surface_matches_statsmodels(self, nobs, n_series):
        mine = mackinnon_crit(n_series, "c", nobs)
        sm = mackinnoncrit(N=n_series, regression="c", nobs=nobs)
        for i, lvl in enumerate(("1%", "5%", "10%")):
            assert mine[lvl] == pytest.approx(sm[i], abs=1e-10)

    def test_eg_values_materially_stricter_than_plain_adf(self):
        plain = mackinnon_crit(1, "c", 500)
        eg = mackinnon_crit(2, "c", 500)
        for lvl in ("1%", "5%", "10%"):
            assert eg[lvl] < plain[lvl] - 0.4  # EG bar is much higher

    def test_engle_granger_result_uses_n2_surface(self):
        df, _ = cointegrated_pair(n=800, seed=21)
        eg = engle_granger(df["Y"].to_numpy(), df["X"].to_numpy())
        expected = mackinnon_crit(2, "c", eg.adf.nobs)
        assert eg.crit == expected
        wrong = mackinnon_crit(1, "c", eg.adf.nobs)
        assert eg.crit["5%"] != pytest.approx(wrong["5%"], abs=0.1)

    def test_unavailable_surface_raises(self):
        with pytest.raises(ValueError, match="no MacKinnon surface"):
            mackinnon_crit(2, "n")
        with pytest.raises(ValueError, match="no MacKinnon surface"):
            mackinnon_crit(3, "c")
        with pytest.raises(ValueError, match="nobs"):
            mackinnon_crit(1, "c", nobs=-5)


class TestEngleGranger:
    def test_recovers_hedge_ratio(self):
        df, truth = cointegrated_pair(n=3000, beta=1.5, kappa=0.05, sigma=1.0, seed=22)
        eg = engle_granger(df["Y"].to_numpy(), df["X"].to_numpy())
        assert eg.beta == pytest.approx(truth.beta, abs=0.05)

    @pytest.mark.parametrize("seed", [30, 31, 32])
    def test_rejects_unit_root_on_cointegrated_large_sample(self, seed):
        df, _ = cointegrated_pair(n=3000, beta=1.2, kappa=0.06, sigma=1.0, seed=seed)
        eg = engle_granger(df["Y"].to_numpy(), df["X"].to_numpy())
        assert eg.cointegrated("5%")

    def test_trap_pair_not_cointegrated(self):
        df, _ = correlated_random_walks(n=2000, rho=0.92, seed=23)
        eg = engle_granger(df["A"].to_numpy(), df["B"].to_numpy())
        assert not eg.cointegrated("5%")

    def test_spurious_rejection_rate_close_to_size(self):
        """On independent random walks the EG test should reject at ~ the
        nominal 5% size. This is the guard against spurious regression: with
        PLAIN ADF critical values the rate would be several times nominal."""
        n_reps, n = 200, 400
        rng = np.random.default_rng(24)
        rej_eg = 0
        rej_wrong = 0
        for _ in range(n_reps):
            y = 100 + np.cumsum(rng.standard_normal(n))
            x = 100 + np.cumsum(rng.standard_normal(n))
            eg = engle_granger(y, x, lags=1)
            rej_eg += eg.cointegrated("5%")
            wrong_crit = mackinnon_crit(1, "c", eg.adf.nobs)  # the classic mistake
            rej_wrong += eg.stat < wrong_crit["5%"]
        rate_eg = rej_eg / n_reps
        rate_wrong = rej_wrong / n_reps
        assert 0.005 <= rate_eg <= 0.12  # ~ nominal size, loose tolerance
        assert rate_wrong > 1.5 * rate_eg  # plain-ADF values are oversized

    def test_no_intercept_flagged_as_approximation(self):
        df, _ = cointegrated_pair(n=500, seed=25)
        eg = engle_granger(df["Y"].to_numpy(), df["X"].to_numpy(), intercept=False)
        assert eg.crit_approx is True
        assert eg.alpha == 0.0

    def test_residuals_are_spread(self):
        df, _ = cointegrated_pair(n=500, seed=26)
        eg = engle_granger(df["Y"].to_numpy(), df["X"].to_numpy())
        manual = df["Y"].to_numpy() - eg.beta * df["X"].to_numpy() - eg.alpha
        np.testing.assert_allclose(eg.resid, manual, atol=1e-8)
