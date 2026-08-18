"""Evaluation: QLIKE consistency, MZ calibration, Diebold-Mariano size and
power, Newey-West, Ljung-Box / ARCH-LM / sign-bias diagnostics."""

import numpy as np
import pytest

from fx_vol import (
    arch_lm,
    diebold_mariano,
    ljung_box,
    mincer_zarnowitz,
    mse_loss,
    newey_west_variance,
    qlike_loss,
    sign_bias_test,
)
from fx_vol.data import synthetic as syn


class TestLosses:
    def test_qlike_minimized_at_true_variance(self):
        """Patton robustness: E[QLIKE(f, r^2)] is minimized at f = sigma2_true,
        even though r^2 is a noisy proxy. Checked over a large simulation."""
        rng = np.random.default_rng(91)
        sigma2 = 1e-4
        r2 = sigma2 * rng.standard_normal(200_000) ** 2
        losses = {
            c: qlike_loss(np.full(r2.size, c * sigma2), r2).mean()
            for c in (0.7, 0.85, 1.0, 1.2, 1.4)
        }
        assert min(losses, key=losses.get) == 1.0

    def test_qlike_asymmetry_penalizes_underprediction(self):
        """Under-forecasting variance by 50% costs more than over-forecasting
        by 50% -- the risk-management-friendly asymmetry."""
        rng = np.random.default_rng(92)
        sigma2 = 1e-4
        r2 = sigma2 * rng.standard_normal(100_000) ** 2
        under = qlike_loss(np.full(r2.size, 0.5 * sigma2), r2).mean()
        over = qlike_loss(np.full(r2.size, 1.5 * sigma2), r2).mean()
        true = qlike_loss(np.full(r2.size, sigma2), r2).mean()
        assert (under - true) > (over - true) > 0

    def test_mse_zero_iff_equal(self):
        f = np.array([1.0, 2.0, 3.0])
        assert np.all(mse_loss(f, f) == 0)
        assert mse_loss(f, f + 1).sum() == pytest.approx(3.0)

    def test_qlike_validation(self):
        with pytest.raises(ValueError, match="positive"):
            qlike_loss([0.0, 1.0], [1.0, 1.0])
        with pytest.raises(ValueError, match="aligned"):
            qlike_loss([1.0, 1.0], [1.0])


class TestMincerZarnowitz:
    def test_calibrated_forecast_passes(self, garch_sim):
        """Regressing r_t^2 on the TRUE conditional variance gives (a,b)=(0,1)."""
        r, sigma2 = garch_sim
        out = mincer_zarnowitz(sigma2, r ** 2)
        assert out["slope"] == pytest.approx(1.0, abs=0.1)
        assert out["intercept"] == pytest.approx(0.0, abs=2e-6)
        assert out["p_joint"] > 0.01
        assert out["n"] == r.size

    def test_biased_forecast_rejected(self, garch_sim):
        r, sigma2 = garch_sim
        out = mincer_zarnowitz(2.0 * sigma2, r ** 2)  # doubled forecast
        assert out["p_joint"] < 1e-6


class TestNeweyWest:
    def test_iid_matches_sample_variance(self):
        x = np.random.default_rng(93).standard_normal(50_000)
        assert newey_west_variance(x, 0) == pytest.approx(np.var(x), rel=1e-10)

    def test_ma1_long_run_variance(self):
        """x_t = e_t + theta e_{t-1}: LRV = (1 + theta)^2 * var_e."""
        rng = np.random.default_rng(94)
        theta = 0.6
        e = rng.standard_normal(400_000)
        x = e[1:] + theta * e[:-1]
        lrv = newey_west_variance(x, lags=30)
        assert lrv == pytest.approx((1 + theta) ** 2, rel=0.05)

    def test_validation(self):
        with pytest.raises(ValueError, match="lags"):
            newey_west_variance([1.0, 2.0, 3.0], 5)


class TestDieboldMariano:
    def test_detects_dominant_model(self, garch_sim):
        """True conditional variance vs constant variance: DM must strongly
        favour the true model under QLIKE."""
        r, sigma2 = garch_sim
        r2 = r ** 2
        l_true = qlike_loss(sigma2, r2)
        l_const = qlike_loss(np.full(r2.size, r2.mean()), r2)
        out = diebold_mariano(l_true, l_const)
        assert out["stat"] < -3.0
        assert out["pvalue"] < 0.01

    def test_size_under_null(self):
        """Two equally good (noisy) forecasts: rejection rate at the 10% level
        should be near 10% across replications (loose statistical tolerance)."""
        rng = np.random.default_rng(95)
        rejections = 0
        n_reps = 300
        for _ in range(n_reps):
            sigma2 = 1e-4
            r2 = sigma2 * rng.standard_normal(300) ** 2
            f1 = sigma2 * np.exp(0.2 * rng.standard_normal(300))
            f2 = sigma2 * np.exp(0.2 * rng.standard_normal(300))
            out = diebold_mariano(qlike_loss(f1, r2), qlike_loss(f2, r2))
            rejections += out["pvalue"] < 0.10
        rate = rejections / n_reps
        assert 0.04 <= rate <= 0.18  # ~3 sigma band around 0.10 plus asymptotic slack

    def test_symmetry(self, garch_sim):
        r, sigma2 = garch_sim
        r2 = r ** 2
        l1 = qlike_loss(sigma2, r2)
        l2 = qlike_loss(1.5 * sigma2, r2)
        a = diebold_mariano(l1, l2)
        b = diebold_mariano(l2, l1)
        assert a["stat"] == pytest.approx(-b["stat"], rel=1e-12)

    def test_validation(self):
        with pytest.raises(ValueError, match="at least 10"):
            diebold_mariano([1.0] * 5, [2.0] * 5)
        with pytest.raises(ValueError, match="identical"):
            diebold_mariano(np.ones(100), np.ones(100))


class TestDiagnostics:
    def test_ljung_box_detects_vol_clustering(self, garch_sim):
        r, _ = garch_sim
        out = ljung_box(r ** 2, lags=10)
        assert out["pvalue"] < 1e-6

    def test_ljung_box_clean_on_iid(self):
        r = syn.simulate_constant_vol(5000, 0.006, seed=96)
        out = ljung_box(r ** 2, lags=10)
        assert out["pvalue"] > 0.05

    def test_arch_lm_detects_garch_effects(self, garch_sim):
        r, _ = garch_sim
        assert arch_lm(r, lags=10)["pvalue"] < 1e-6

    def test_arch_lm_clean_after_garch_fit(self, garch_fit):
        """Standardized residuals of a correctly specified fit show no
        remaining ARCH."""
        assert arch_lm(garch_fit.std_resid, lags=10)["pvalue"] > 0.05

    def test_sign_bias_detects_asymmetry(self, gjr_sim, symmetric_garch_fit_on_gjr):
        """Symmetric GARCH on GJR data leaves sign/size bias in residuals."""
        fit = symmetric_garch_fit_on_gjr
        out = sign_bias_test(fit.returns, fit.sigma2)
        assert out["joint_f_p"] < 0.01

    def test_sign_bias_clean_on_symmetric_fit(self, garch_fit):
        out = sign_bias_test(garch_fit.returns, garch_fit.sigma2)
        assert out["joint_f_p"] > 0.01

    def test_sign_bias_validation(self):
        with pytest.raises(ValueError, match="positive"):
            sign_bias_test([0.01, -0.01, 0.02], [1e-4, 0.0, 1e-4])
