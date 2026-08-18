"""Forecast evaluation: losses, MZ regression, DM test, diagnostics."""

import numpy as np
import pytest

from eq_vol.data import synthetic as syn
from eq_vol.evaluation import (
    arch_lm_test,
    diebold_mariano,
    forecast_race_table,
    ljung_box_squared,
    mincer_zarnowitz,
    mse_loss,
    qlike_loss,
    sign_bias_test,
)
from eq_vol.garch import fit_garch
from eq_vol.gjr import fit_gjr


@pytest.fixture(scope="module")
def het_data():
    """True conditional variances + squared-return proxy from GARCH sim."""
    sim = syn.simulate_garch(30_000, seed=100)
    return sim.sigma2, sim.returns**2


class TestLosses:
    def test_qlike_minimised_by_true_variance(self, het_data):
        sigma2, proxy = het_data
        q_true = qlike_loss(sigma2, proxy).mean()
        for c in (0.7, 0.85, 1.15, 1.4):
            assert q_true < qlike_loss(c * sigma2, proxy).mean()
        # unconditional (flat) forecast also loses
        q_flat = qlike_loss(np.full_like(sigma2, sigma2.mean()), proxy).mean()
        assert q_true < q_flat

    def test_mse_minimised_by_true_variance(self, het_data):
        sigma2, proxy = het_data
        m_true = mse_loss(sigma2, proxy).mean()
        for c in (0.7, 1.4):
            assert m_true < mse_loss(c * sigma2, proxy).mean()

    def test_qlike_handles_zero_proxy(self):
        # a zero return day gives proxy = 0; ln f + p/f is still finite
        out = qlike_loss(np.array([1e-4, 2e-4]), np.array([0.0, 1e-4]))
        assert np.all(np.isfinite(out))

    def test_qlike_rejects_nonpositive_forecast(self):
        with pytest.raises(ValueError, match="positive"):
            qlike_loss(np.array([0.0, 1e-4]), np.array([1e-4, 1e-4]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="lengths differ"):
            mse_loss(np.ones(3), np.ones(4))


class TestMincerZarnowitz:
    def test_correct_model_has_unit_slope(self, het_data):
        sigma2, proxy = het_data
        mz = mincer_zarnowitz(sigma2, proxy)
        assert mz.slope == pytest.approx(1.0, abs=0.1)
        assert abs(mz.intercept) < 3.0 * mz.intercept_se + 1e-6
        assert mz.joint_pvalue > 0.01  # cannot reject (0, 1)

    def test_biased_forecast_detected(self, het_data):
        sigma2, proxy = het_data
        mz = mincer_zarnowitz(0.5 * sigma2, proxy)  # under-forecast by half
        assert mz.slope == pytest.approx(2.0, abs=0.2)
        assert mz.joint_pvalue < 1e-6

    def test_r2_positive_for_informative_forecast(self, het_data):
        # note: R^2 is heavily attenuated by proxy noise (var(z^2) = 2 for
        # Gaussian z) even for the TRUE conditional variance — this is why
        # low MZ R^2 does not mean a bad vol forecast (see docs)
        sigma2, proxy = het_data
        assert mincer_zarnowitz(sigma2, proxy).r2 > 0.01


class TestDieboldMariano:
    def test_size_close_to_nominal_under_null(self):
        # two forecasts constructed to have EXACTLY equal expected QLIKE:
        # f1 = a*sigma2, f2 = b*sigma2 with ln a + 1/a = ln b + 1/b, a != b.
        from scipy.optimize import brentq

        a = 1.5
        target = np.log(a) + 1.0 / a
        b = brentq(lambda x: np.log(x) + 1.0 / x - target, 0.2, 0.999)
        rng = np.random.default_rng(200)
        n, reps = 300, 1000
        rejections = 0
        for _ in range(reps):
            z2 = rng.standard_normal(n) ** 2  # proxy = sigma2 * z^2, sigma2 = 1
            l1 = np.log(a) + z2 / a
            l2 = np.log(b) + z2 / b
            dm = diebold_mariano(l1, l2, h=1)
            rejections += dm.pvalue < 0.05
        rate = rejections / reps
        assert 0.02 < rate < 0.09  # near nominal 5%, loose tolerance

    def test_detects_genuinely_better_forecast(self, het_data):
        sigma2, proxy = het_data
        good = qlike_loss(sigma2, proxy)
        bad = qlike_loss(np.full_like(sigma2, sigma2.mean()), proxy)
        dm = diebold_mariano(good, bad)
        assert dm.stat < -3.0 and dm.pvalue < 0.01

    def test_sign_convention(self, het_data):
        sigma2, proxy = het_data
        good = qlike_loss(sigma2[:2000], proxy[:2000])
        bad = qlike_loss(1.5 * sigma2[:2000], proxy[:2000])
        assert diebold_mariano(good, bad).stat < 0  # model 1 better => negative
        assert diebold_mariano(bad, good).stat > 0

    def test_degenerate_losses_raise(self):
        l = np.ones(100)
        with pytest.raises(ValueError, match="degenerate|non-positive"):
            diebold_mariano(l, l)

    def test_short_series_raises(self):
        with pytest.raises(ValueError, match="at least"):
            diebold_mariano(np.ones(5), np.zeros(5))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            diebold_mariano(np.ones(50), np.ones(60))


@pytest.fixture(scope="module")
def garch_data_fit():
    sim = syn.simulate_garch(5000, seed=101)
    return sim, fit_garch(sim.returns)


class TestInSampleDiagnostics:
    def test_ljung_box_passes_for_correct_model(self, garch_data_fit):
        _, res = garch_data_fit
        lb = ljung_box_squared(res.std_residuals, lags=10)
        assert float(lb["lb_pvalue"].iloc[0]) > 0.01

    def test_ljung_box_detects_clustering_in_raw_returns(self, garch_data_fit):
        sim, _ = garch_data_fit
        z_raw = sim.returns / sim.returns.std()  # constant-vol standardisation
        lb = ljung_box_squared(z_raw, lags=10)
        assert float(lb["lb_pvalue"].iloc[0]) < 0.01

    def test_arch_lm_detects_and_clears(self, garch_data_fit):
        sim, res = garch_data_fit
        assert arch_lm_test(sim.returns)["lm_pvalue"] < 0.01       # raw: ARCH present
        assert arch_lm_test(res.std_residuals)["lm_pvalue"] > 0.01  # filtered: gone

    def test_sign_bias_detects_missed_leverage(self, gjr_sim):
        r = gjr_sim.returns[:10_000]
        garch_res = fit_garch(r)      # symmetric model on leveraged data
        gjr_res = fit_gjr(r)          # correct asymmetric model
        tbl_garch = sign_bias_test(garch_res)
        tbl_gjr = sign_bias_test(gjr_res)
        assert tbl_garch.loc["joint_F", "pvalue"] < 0.05    # asymmetry missed
        assert tbl_gjr.loc["joint_F", "pvalue"] > 0.05      # asymmetry captured

    def test_ljung_box_short_series_raises(self):
        with pytest.raises(ValueError, match="Ljung-Box"):
            ljung_box_squared(np.ones(5), lags=10)


class TestForecastRaceTable:
    def test_ordering_and_dm_columns(self, het_data):
        sigma2, proxy = het_data
        sigma2, proxy = sigma2[:3000], proxy[:3000]
        rng = np.random.default_rng(300)
        table = forecast_race_table(
            {
                "flat": np.full_like(sigma2, sigma2.mean()),
                "true": sigma2,
                "noisy": sigma2 * np.exp(rng.normal(0, 0.5, sigma2.size)),
            },
            proxy,
            benchmark="flat",
        )
        assert table.index[0] == "true"  # lowest QLIKE first
        assert np.isnan(table.loc["flat", "dm_stat_vs_benchmark"])
        assert table.loc["true", "dm_stat_vs_benchmark"] < 0
        assert set(table.columns) == {"qlike", "mse", "dm_stat_vs_benchmark", "dm_pvalue"}

    def test_bad_benchmark_raises(self, het_data):
        sigma2, proxy = het_data
        with pytest.raises(ValueError, match="benchmark"):
            forecast_race_table({"a": sigma2}, proxy, benchmark="zzz")
