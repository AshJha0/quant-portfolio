"""ADF and Engle-Granger: cross-validation vs statsmodels, size control, degeneracy."""

import numpy as np
import pytest
from statsmodels.tsa.stattools import adfuller, coint

from fx_pairs.cointegration import (
    adf_test,
    engle_granger,
    is_degenerate_spread,
    mackinnon_crit,
)
from fx_pairs.data import synthetic as syn
from fx_pairs.universe import make_cross, triangular_spread


@pytest.fixture(scope="module")
def random_walk():
    rng = np.random.default_rng(0)
    return np.cumsum(0.01 * rng.standard_normal(400))


@pytest.fixture(scope="module")
def stationary_ar1():
    return syn.simulate_ou(600, kappa=30.0, theta=0.0, sigma=0.05, seed=1)


class TestADFvsStatsmodels:
    def test_stat_matches_fixed_lag0(self, random_walk):
        mine = adf_test(random_walk, regression="c", lags=0)
        sm = adfuller(random_walk, maxlag=0, autolag=None, regression="c")
        assert mine.stat == pytest.approx(sm[0], abs=1e-10)

    def test_stat_matches_fixed_lag4(self, random_walk):
        mine = adf_test(random_walk, regression="c", lags=4)
        sm = adfuller(random_walk, maxlag=4, autolag=None, regression="c")
        assert mine.stat == pytest.approx(sm[0], abs=1e-10)

    def test_stat_matches_no_constant(self, stationary_ar1):
        mine = adf_test(stationary_ar1, regression="n", lags=2)
        sm = adfuller(stationary_ar1, maxlag=2, autolag=None, regression="n")
        assert mine.stat == pytest.approx(sm[0], abs=1e-10)

    def test_autolag_aic_matches_stat_and_lag(self, stationary_ar1):
        mine = adf_test(stationary_ar1, regression="c")
        sm = adfuller(stationary_ar1, autolag="AIC", regression="c")
        assert mine.usedlag == sm[2]
        assert mine.stat == pytest.approx(sm[0], abs=1e-8)

    def test_crit_values_match_statsmodels(self, random_walk):
        mine = adf_test(random_walk, regression="c", lags=0)
        sm = adfuller(random_walk, maxlag=0, autolag=None, regression="c")
        for lvl in ("1%", "5%", "10%"):
            assert mine.crit_values[lvl] == pytest.approx(sm[4][lvl], abs=1e-10)


class TestADFDecisions:
    def test_random_walk_not_rejected(self, random_walk):
        res = adf_test(random_walk, regression="c")
        assert not res.reject("5%")
        assert res.pvalue > 0.05

    def test_stationary_ar1_rejected(self, stationary_ar1):
        res = adf_test(stationary_ar1, regression="c")
        assert res.reject("1%")
        assert res.pvalue < 0.05


class TestCriticalValues:
    def test_eg_crit_more_negative_than_adf(self):
        """Residual-based EG testing needs the N=2 table: plain ADF critical
        values on estimated residuals would over-reject."""
        adf_cv = mackinnon_crit(1, 250)
        eg_cv = mackinnon_crit(2, 250)
        for lvl in ("1%", "5%", "10%"):
            assert eg_cv[lvl] < adf_cv[lvl] - 0.3

    def test_finite_sample_more_negative_than_asymptotic(self):
        asym = mackinnon_crit(1, np.inf)
        finite = mackinnon_crit(1, 100)
        for lvl in ("1%", "5%", "10%"):
            assert finite[lvl] < asym[lvl]

    def test_invalid_nvars_raises(self):
        with pytest.raises(ValueError):
            mackinnon_crit(7, 100)


class TestEngleGranger:
    def test_hedge_ratio_recovery(self):
        p1, p2, truth = syn.make_cointegrated_pair(n=2000, beta=1.4, alpha=0.2,
                                                   kappa=25.0, sigma_ou=0.04, seed=7)
        eg = engle_granger(np.log(p1.values), np.log(p2.values))
        assert eg.beta == pytest.approx(truth["beta"], abs=0.05)
        assert eg.alpha == pytest.approx(truth["alpha"], abs=0.05)
        assert eg.cointegrated

    def test_stat_matches_statsmodels_coint(self):
        p1, p2, _ = syn.make_cointegrated_pair(n=1000, beta=1.4, alpha=0.2, seed=7)
        lp1, lp2 = np.log(p1.values), np.log(p2.values)
        eg = engle_granger(lp1, lp2)
        sm_stat, _, sm_crit = coint(lp1, lp2, trend="c", autolag="aic")
        assert eg.stat == pytest.approx(sm_stat, abs=1e-8)
        assert eg.crit_values["5%"] == pytest.approx(sm_crit[1], abs=1e-8)

    def test_spurious_regression_size_control(self):
        """Independent random walks: EG at 5% must reject ~5% of the time,
        not the near-certain 'significance' that levels OLS t-stats suggest."""
        n_sims, rejections = 200, 0
        for i in range(n_sims):
            p1, p2 = syn.make_correlated_walks(n=250, rho=0.0, seed=1000 + i)
            eg = engle_granger(np.log(p1.values), np.log(p2.values), lags=0)
            rejections += eg.cointegrated
        rate = rejections / n_sims
        assert rate < 0.10  # 5% nominal + Monte Carlo noise
        assert rate > 0.0   # sanity: the test isn't degenerate/never-rejecting

    def test_correlated_walks_not_cointegrated(self):
        """High return correlation is NOT cointegration."""
        p1, p2 = syn.make_correlated_walks(n=1000, rho=0.9, seed=42)
        rets = np.diff(np.log(np.column_stack([p1.values, p2.values]), ), axis=0)
        assert np.corrcoef(rets.T)[0, 1] > 0.85
        eg = engle_granger(np.log(p1.values), np.log(p2.values))
        assert not eg.cointegrated

    def test_triangular_spread_flagged_degenerate(self):
        """The no-arbitrage null case: cointegration machinery must declare
        the triangular relation degenerate, not tradable."""
        legs, _ = syn.make_two_block_panel(n=400, seed=2)
        lp_a = np.log(make_cross(legs, "AUD", "JPY").values)
        # synthetic leg: AUDUSD * USDJPY, identical to AUDJPY by construction
        lp_b = np.log(make_cross(legs, "AUD", "USD").values) + \
            np.log(make_cross(legs, "USD", "JPY").values)
        eg = engle_granger(lp_a, lp_b)
        assert eg.degenerate
        assert not eg.cointegrated
        assert np.isnan(eg.stat)
        assert eg.beta == pytest.approx(1.0, abs=1e-10)

    def test_is_degenerate_spread(self):
        legs, _ = syn.make_two_block_panel(n=400, seed=2)
        tri = triangular_spread(legs, "AUD", "USD", "JPY")
        assert is_degenerate_spread(tri.values)
        s = syn.simulate_ou(400, 20.0, 0.0, 0.05, seed=3)
        assert not is_degenerate_spread(s)


class TestValidation:
    def test_short_series_raises(self):
        with pytest.raises(ValueError, match="too short"):
            adf_test(np.arange(10.0))
        with pytest.raises(ValueError, match="too short"):
            engle_granger(np.arange(20.0), np.arange(20.0))

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="equal length"):
            engle_granger(np.arange(100.0), np.arange(90.0))

    def test_nan_raises(self):
        y = np.cumsum(np.random.default_rng(0).standard_normal(100))
        y[3] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            adf_test(y)

    def test_bad_regression_raises(self, random_walk=None):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="regression"):
            adf_test(np.cumsum(rng.standard_normal(100)), regression="ct")
