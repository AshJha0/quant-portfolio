"""Inference and evaluation: sandwich SEs, Hessian steps, HAC Diebold-Mariano,
EGARCH unconditional variance, Yang-Zhang, rolling harness options.

Every test here failed on the code before the fix it pins.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from eq_vol import (
    default_hac_lags,
    diebold_mariano,
    egarch_unconditional_variance,
    fit_egarch,
    fit_garch,
    fit_gjr,
    forecast_race_table,
    range_vol,
    rolling_one_step_forecasts,
    yang_zhang_var,
)
from eq_vol._utils import numerical_hessian
from eq_vol.data import synthetic as syn
from eq_vol.egarch import egarch_unconditional_logvar


class TestSandwichStandardErrors:
    def test_robust_close_to_classic_under_correct_density(self):
        # Gaussian data, Gaussian likelihood: information-matrix equality
        # holds, so A^-1 B A^-1 -> A^-1. On n=5000 the two agree to ~10%
        # (sampling noise of B); a 25% band is ~3 sampling SDs of the ratio.
        f = fit_garch(syn.simulate_garch(5000, seed=11).returns)
        for k in ("alpha", "beta"):
            assert f.robust_std_errors[k] == pytest.approx(f.std_errors[k], rel=0.25)

    def test_robust_larger_under_fat_tails(self):
        f = fit_garch(syn.simulate_garch(5000, dist="t", nu=5.0, seed=7).returns)
        for k in ("alpha", "beta"):
            assert f.robust_std_errors[k] > 1.3 * f.std_errors[k]

    @pytest.mark.parametrize("fitter", [fit_gjr, fit_egarch])
    def test_every_model_reports_robust_errors(self, fitter, gjr_sim):
        f = fitter(gjr_sim.returns[:4000])
        assert set(f.robust_std_errors) == set(f.params)
        assert all(np.isfinite(v) and v > 0 for v in f.robust_std_errors.values())

    def test_summary_prints_both(self, garch_fit):
        s = garch_fit.summary()
        assert "robust se" in s


class TestHessianStep:
    def test_parameter_near_zero_gets_usable_step(self):
        # f = 0.5 x'Ax, exact Hessian A. With x_2 = 1e-10 the old step
        # 1e-4 * max(|x|, 1e-8) = 1e-12 makes the diagonal round-off.
        a = np.array([[3.0, 0.5], [0.5, 2.0]])
        x = np.array([0.9, 1e-10])
        f = lambda v: 0.5 * v @ a @ v  # noqa: E731
        # Central differences are exact on a quadratic up to round-off
        # eps |f| / h^2 ~ 2e-16 / 1e-12 = 2e-4 for the h = 1e-6 step used here.
        h = numerical_hessian(f, x, floor=np.array([0.9, 1e-2]))
        np.testing.assert_allclose(h, a, rtol=1e-3)

    def test_gjr_gamma_se_on_symmetric_data(self):
        # True gamma = 0: the fitted gamma is ~1e-3 and the classic and robust
        # SEs must both be finite and agree (Gaussian data), not round-off.
        r = syn.simulate_gjr(8000, gamma=0.0, alpha=0.06, seed=21).returns
        f = fit_gjr(r)
        se, rse = f.std_errors["gamma"], f.robust_std_errors["gamma"]
        assert 1e-3 < se < 0.05
        assert rse == pytest.approx(se, rel=0.3)


class TestDieboldMarianoHAC:
    def test_default_lag_rule(self):
        assert default_hac_lags(500) == math.floor(4 * 5 ** (2 / 9))
        assert default_hac_lags(500, h=10) == 9
        assert default_hac_lags(100) == 4

    def test_default_reported_lag_is_used(self):
        rng = np.random.default_rng(3)
        d = rng.standard_normal(500) + 0.1
        assert diebold_mariano(d, np.zeros(500)).stat == pytest.approx(
            diebold_mariano(d, np.zeros(500), hac_lags=default_hac_lags(500)).stat
        )

    def test_size_under_autocorrelated_differential(self):
        # AR(1) loss differential (phi = 0.5) with zero mean. The i.i.d.
        # variance (lag 0) rejects ~28% at 5%; the default HAC lag ~9%.
        # 1000 replications: MC SE of a 5% rate is 0.7 pp, so the bands
        # separate the two by many SEs.
        rng = np.random.default_rng(0)
        d = lfilter([1.0], [1.0, -0.5], rng.standard_normal((1000, 500)), axis=1)
        z = np.zeros(500)
        rej0 = np.mean([diebold_mariano(x, z, hac_lags=0).pvalue < 0.05 for x in d])
        rej = np.mean([diebold_mariano(x, z).pvalue < 0.05 for x in d])
        assert rej0 > 0.2
        assert rej < 0.12

    def test_bad_lag_rejected(self):
        with pytest.raises(ValueError, match="hac_lags"):
            diebold_mariano(np.ones(20) + np.arange(20), np.zeros(20), hac_lags=20)

    def test_race_table_passes_lags(self):
        rng = np.random.default_rng(5)
        proxy = rng.chisquare(1, 400) * 1e-4
        fc = {"a": np.full(400, 1e-4), "b": np.full(400, 1.2e-4)}
        t0 = forecast_race_table(fc, proxy, hac_lags=0)
        t1 = forecast_race_table(fc, proxy)
        assert t0.loc["b", "dm_stat_vs_benchmark"] != t1.loc["b", "dm_stat_vs_benchmark"]


class TestEgarchUnconditionalVariance:
    def test_exceeds_median_level(self):
        p = dict(omega=-0.4, alpha=0.1, gamma=-0.08, beta=0.96)
        median = math.exp(egarch_unconditional_logvar(p["omega"], p["beta"]))
        assert egarch_unconditional_variance(**p) > 1.05 * median

    def test_matches_long_simulation(self):
        # Mean of sigma2 over a long Gaussian simulation. The sigma2 series is
        # highly persistent; with 400k draws the batch-means SE is ~1.5% of
        # the mean, so 5% is ~3 SE.
        p = dict(omega=-0.4, alpha=0.1, gamma=-0.08, beta=0.96)
        sim = syn.simulate_egarch(400_000, seed=99, **p)
        assert egarch_unconditional_variance(**p) == pytest.approx(np.mean(sim.sigma2), rel=0.05)

    def test_t_innovations_have_no_finite_mean(self):
        assert math.isinf(egarch_unconditional_variance(-0.4, 0.1, -0.08, 0.96, dist="t"))

    def test_no_shocks_reduces_to_level(self):
        assert egarch_unconditional_variance(-0.4, 0.0, 0.0, 0.96) == pytest.approx(math.exp(-10.0))

    def test_fit_reports_mean_and_median(self, egarch_fit):
        e = egarch_fit.extra
        assert e["unconditional_vol_annual"] > e["median_vol_annual"]


class TestYangZhang:
    def test_matches_definition_on_one_window(self):
        rng = np.random.default_rng(8)
        n, w = 60, 20
        c = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
        o = np.r_[100.0, c[:-1]] * np.exp(0.003 * rng.standard_normal(n))
        h = np.maximum(o, c) * 1.004
        lo = np.minimum(o, c) * 0.996
        v = yang_zhang_var(o, h, lo, c, window=w)
        t = n - 1
        on = np.log(o[t - w + 1 : t + 1] / c[t - w : t])
        oc = np.log(c[t - w + 1 : t + 1] / o[t - w + 1 : t + 1])
        hh, ll, oo, cc = (x[t - w + 1 : t + 1] for x in (h, lo, o, c))
        rs = np.log(hh / cc) * np.log(hh / oo) + np.log(ll / cc) * np.log(ll / oo)
        k = 0.34 / (1.34 + (w + 1) / (w - 1))
        expected = on.var(ddof=1) + k * oc.var(ddof=1) + (1 - k) * rs.mean()
        assert v[t] == pytest.approx(expected, rel=1e-12)
        assert np.all(np.isnan(v[:w]))

    def test_range_vol_dispatch(self):
        rng = np.random.default_rng(9)
        c = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(80)))
        o = np.r_[100.0, c[:-1]]
        df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.005,
                           "low": np.minimum(o, c) * 0.995, "close": c})
        vol = range_vol(df, estimator="yang_zhang", window=21)
        assert np.isfinite(vol[-1]) and 0.05 < vol[-1] < 0.5


class TestRollingHarness:
    def test_student_t_dist_is_used(self):
        r = syn.simulate_garch(1300, dist="t", nu=5.0, seed=4).returns
        a = rolling_one_step_forecasts(r, "garch", min_train=1200, refit_every=50)
        b = rolling_one_step_forecasts(r, "garch", min_train=1200, refit_every=50, dist="t")
        assert not np.allclose(a.forecasts, b.forecasts)

    def test_bad_dist_rejected(self):
        r = syn.simulate_garch(1300, seed=4).returns
        with pytest.raises(ValueError, match="dist"):
            rolling_one_step_forecasts(r, "garch", min_train=1200, dist="skewt")


class TestBoundaryFlag:
    def test_random_walk_log_vol_pins_persistence_and_is_flagged(self):
        # Log-vol random walk: no finite unconditional variance, so the fit
        # runs to the persistence cap and the SEs there are not meaningful.
        rng = np.random.default_rng(1)
        sig = np.exp(np.cumsum(0.05 * rng.standard_normal(4000))) * 0.01
        f = fit_garch(sig * rng.standard_normal(4000))
        assert f.extra["persistence"] > 0.999
        assert f.extra["at_boundary"] == 1.0

    def test_interior_fit_not_flagged(self, garch_fit):
        assert garch_fit.extra["at_boundary"] == 0.0
