"""Inference, convergence handling and evaluation fixes.

Every test here failed on the code before the fix it pins.
"""

import math

import numpy as np
import pytest
from scipy.signal import lfilter

import fx_vol.forecasting as fc
import fx_vol.garch as garch_mod
from fx_vol import (
    ConvergenceError,
    default_hac_lags,
    diebold_mariano,
    egarch_unconditional_variance,
    fit_egarch,
    fit_garch,
    fit_gjr,
    gjr_filter,
    rolling_one_step,
)
from fx_vol.data import synthetic as syn


class TestRobustStandardErrors:
    def test_robust_larger_than_classic_on_fat_tails(self):
        r = syn.simulate_garch(5000, omega=1.5e-6, alpha=0.06, beta=0.90, dist="t", nu=5.0, seed=81)
        f = fit_garch(r, dist="gaussian")
        assert f.robust_std_errors["alpha"] > 1.2 * f.std_errors["alpha"]
        assert f.robust_std_errors["beta"] > 1.2 * f.std_errors["beta"]

    @pytest.mark.parametrize("fitter", [fit_garch, fit_gjr, fit_egarch])
    def test_all_models_report_robust(self, fitter):
        r = syn.simulate_gjr(4000, omega=1e-6, alpha=0.03, gamma=0.08, beta=0.88, seed=82)
        f = fitter(r)
        assert set(f.robust_std_errors) == set(f.params)
        assert all(np.isfinite(v) and v > 0 for v in f.robust_std_errors.values())

    def test_egarch_omega_se_is_unit_consistent(self):
        # omega_r = omega_y + (1 - beta) ln s^2. Fitting r and 100 r gives
        # omega's that differ by (1 - beta) ln(1e4); their SEs differ through
        # Var(beta) and Cov(omega, beta) only, and both must be of the size
        # the delta method implies, not the scaled-data SE (old code: equal
        # SEs in both units, ~0.0024 here).
        r = syn.simulate_egarch(5000, omega=-0.5, alpha=0.15, gamma=-0.06, beta=0.95, seed=74)
        dec = fit_egarch(r)
        pct = fit_egarch(100.0 * r)
        assert dec.std_errors["omega"] > 10.0 * pct.std_errors["omega"]


class TestGjrSignConstraint:
    def test_filter_accepts_negative_gamma(self):
        y = np.array([0.01, -0.02, 0.015, -0.005])
        s2 = gjr_filter(y, 1e-6, 0.10, -0.05, 0.85, initial_variance=1e-4)
        assert np.all(s2 > 0)

    def test_filter_rejects_alpha_plus_gamma_negative(self):
        with pytest.raises(ValueError, match="alpha \\+ gamma"):
            gjr_filter(np.zeros(5) + 0.01, 1e-6, 0.05, -0.06, 0.85, initial_variance=1e-4)

    def test_negative_asymmetry_recovered(self):
        # Asymmetry on positive returns = negative gamma on this quote.
        r = -syn.simulate_gjr(8000, omega=1e-6, alpha=0.03, gamma=0.10, beta=0.88, seed=83)
        f = fit_gjr(r)
        assert f.params["gamma"] == pytest.approx(-0.10, abs=0.03)
        assert f.params["alpha"] == pytest.approx(0.13, abs=0.03)


class TestConvergence:
    def _failing_minimize(self, *a, **k):
        from scipy.optimize import OptimizeResult

        x0 = np.asarray(a[1], dtype=float)
        return OptimizeResult(x=x0, fun=1e10, success=False, message="forced", nit=0)

    def test_fit_raises_by_default(self, monkeypatch):
        monkeypatch.setattr(garch_mod, "minimize", self._failing_minimize)
        r = syn.simulate_garch(500, 1e-6, 0.05, 0.92, seed=84)
        with pytest.raises(ConvergenceError):
            fit_garch(r)
        assert fit_garch(r, raise_on_failure=False).converged is False

    def test_rolling_counts_failed_refits_and_keeps_params(self, monkeypatch):
        r = syn.simulate_garch(1400, 1e-6, 0.05, 0.92, seed=85)
        real = fc.fit_garch
        calls = {"n": 0}

        def flaky(seg, **kw):
            calls["n"] += 1
            res = real(seg, **kw)
            if calls["n"] == 2:
                res.converged = False
                res.params = {"omega": 1.0, "alpha": 0.5, "beta": 0.4}
            return res

        monkeypatch.setattr(fc, "fit_garch", flaky)
        out = rolling_one_step(r, "garch", window=1000, refit_every=100)
        assert out["failed_refits"] == 1 and out["refits"] == 4
        # the bogus omega = 1.0 was never used
        assert np.all(out["forecast"] < 1e-2)

    def test_rolling_first_failure_raises(self, monkeypatch):
        r = syn.simulate_garch(1200, 1e-6, 0.05, 0.92, seed=86)
        real = fc.fit_garch

        def broken(seg, **kw):
            res = real(seg, **kw)
            res.converged = False
            return res

        monkeypatch.setattr(fc, "fit_garch", broken)
        with pytest.raises(ConvergenceError):
            rolling_one_step(r, "garch", window=1000, refit_every=100)


class TestDieboldMariano:
    def test_default_lag_rule(self):
        assert default_hac_lags(500) == math.floor(4 * 5 ** (2 / 9))
        assert diebold_mariano(np.arange(500.0) % 7, np.zeros(500))["lags"] == default_hac_lags(500)

    def test_hln_matches_closed_form(self):
        rng = np.random.default_rng(2)
        d = rng.standard_normal(60) + 0.3
        raw = diebold_mariano(d, np.zeros(60), lags=0, harvey_correction=False)
        hln = diebold_mariano(d, np.zeros(60), lags=0)
        assert hln["stat"] == pytest.approx(raw["stat"] * math.sqrt((60 + 1 - 2) / 60))
        assert hln["pvalue"] > raw["pvalue"]

    def test_size_under_autocorrelated_differential(self):
        # AR(1) differential, phi = 0.5, zero mean, n = 500, 1000 reps.
        # Lag 0 rejects ~28% at 5%; the default HAC lag ~9%. MC SE ~0.7 pp.
        rng = np.random.default_rng(0)
        d = lfilter([1.0], [1.0, -0.5], rng.standard_normal((1000, 500)), axis=1)
        z = np.zeros(500)
        rej0 = np.mean([diebold_mariano(x, z, lags=0)["pvalue"] < 0.05 for x in d])
        rej = np.mean([diebold_mariano(x, z)["pvalue"] < 0.05 for x in d])
        assert rej0 > 0.2
        assert rej < 0.12


class TestEgarchUnconditional:
    def test_mean_exceeds_median_and_matches_simulation(self):
        p = dict(omega=-0.5, alpha=0.15, gamma=-0.06, beta=0.95)
        ev = egarch_unconditional_variance(**p)
        assert ev > 1.05 * math.exp(p["omega"] / (1 - p["beta"]))
        # Long Gaussian simulation; the batch-means SE of mean(r^2) at 400k
        # draws is ~2% here, so 6% is ~3 SE.
        r = syn.simulate_egarch(400_000, seed=91, **p)
        assert ev == pytest.approx(np.mean(r**2), rel=0.06)

    def test_t_has_no_finite_mean(self):
        assert math.isinf(egarch_unconditional_variance(-0.5, 0.15, -0.06, 0.95, dist="t"))

    def test_fit_field_is_the_mean(self):
        r = syn.simulate_egarch(4000, omega=-0.5, alpha=0.15, gamma=-0.06, beta=0.95, seed=92)
        f = fit_egarch(r)
        p = f.params
        assert f.unconditional_variance == pytest.approx(
            egarch_unconditional_variance(p["omega"], p["alpha"], p["gamma"], p["beta"])
        )
        assert f.unconditional_variance > f.extra["median_variance"]
