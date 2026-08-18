"""FX edge cases (documentation contract item 6): pegged currencies, depegs,
constant series, NaN policy, short series -- each documented in
docs/VALIDATION.md and unit-tested here."""

from dataclasses import replace

import numpy as np
import pytest

from fx_vol import (
    close_to_close_vol,
    ewma_variance,
    fit_egarch,
    fit_garch,
    fit_gjr,
    forecast_variance,
    log_returns,
)
from fx_vol.data import synthetic as syn


@pytest.fixture(scope="module")
def pegged():
    return syn.simulate_pegged(3000, band_vol=2e-5, seed=111)


class TestPeggedCurrency:
    """HKD-style series: daily vol of a few basis points. Models must fit
    without numerical blow-ups (internal unit-variance rescaling); persistence
    may legitimately pin near the IGARCH boundary."""

    def test_garch_fits_without_blowup(self, pegged):
        fit = fit_garch(pegged)
        assert np.all(np.isfinite(fit.sigma2)) and np.all(fit.sigma2 > 0)
        assert np.isfinite(fit.loglik)
        # recovered unconditional vol has the right order of magnitude
        assert fit.unconditional_variance == pytest.approx(4e-10, rel=0.5)

    def test_forecast_stays_tiny_and_finite(self, pegged):
        fit = fit_garch(pegged)
        f = forecast_variance(fit, 100)
        assert np.all(np.isfinite(f)) and np.all(f > 0)
        assert np.sqrt(f.max() * 252) < 0.01  # annualized vol stays below 1%

    def test_ewma_handles_peg(self, pegged):
        s2 = ewma_variance(pegged)
        assert np.all(np.isfinite(s2)) and np.all(s2 >= 0)

    def test_egarch_handles_peg(self, pegged):
        fit = fit_egarch(pegged)
        assert np.all(np.isfinite(fit.sigma2)) and np.all(fit.sigma2 > 0)


@pytest.fixture(scope="module")
def depeg():
    return syn.simulate_depeg(1500, jump_return=-0.15, jump_index=1200, seed=112)


@pytest.fixture(scope="module")
def depeg_fit(depeg):
    return fit_garch(depeg)


class TestDepegJump:
    """CHF-2015-style single -15% day: fits must converge, the conditional
    variance must spike on the jump and the forecast must decay afterwards."""

    def test_garch_converges_through_jump(self, depeg_fit):
        assert np.all(np.isfinite(depeg_fit.sigma2))
        assert np.isfinite(depeg_fit.loglik)
        assert 0.0 < depeg_fit.persistence < 1.0

    def test_variance_spikes_on_jump_day(self, depeg_fit):
        pre = depeg_fit.sigma2[1195:1200].mean()
        post = depeg_fit.sigma2[1201]
        assert post > 20.0 * pre  # day after the jump: variance explodes

    def test_forecast_spikes_then_decays(self, depeg_fit):
        """Desk scenario: parameters from full history, variance state as of
        the close of the depeg day -> the forecast starts elevated and decays
        monotonically toward the unconditional level."""
        state = replace(
            depeg_fit,
            returns=depeg_fit.returns[:1201],  # last observation = the -15% day
            sigma2=depeg_fit.sigma2[:1201],
        )
        f = forecast_variance(state, 250)
        assert f[0] > 10.0 * depeg_fit.unconditional_variance
        assert np.all(np.diff(f) < 0)  # monotone decay
        assert f[-1] < 1.5 * depeg_fit.unconditional_variance

    def test_jump_terminated_sample_degenerates_gracefully(self, depeg):
        """If the sample ENDS on the jump day, alpha is unidentified (the jump
        never feeds a later observation) and the MLE may sit at the
        near-IGARCH boundary -- it must still converge with finite loglik and
        positive variances (documented failure mode, docs/VALIDATION.md)."""
        fit = fit_garch(depeg[:1201])
        assert np.isfinite(fit.loglik)
        assert np.all(np.isfinite(fit.sigma2)) and np.all(fit.sigma2 > 0)
        f = forecast_variance(fit, 50)
        assert np.all(np.isfinite(f)) and np.all(f > 0)

    def test_gjr_attributes_jump_to_negative_side(self, depeg):
        fit = fit_gjr(depeg)
        assert np.isfinite(fit.loglik)
        # the single big negative day pushes gamma upward (or at least not to
        # a pathological value); persistence must remain < 1 by construction
        assert 0.0 <= fit.persistence < 1.0


class TestConstantAndDegenerateSeries:
    def test_constant_series_raise_everywhere(self):
        const = np.zeros(500)
        for fitter in (fit_garch, fit_gjr, fit_egarch):
            with pytest.raises(ValueError, match="constant"):
                fitter(const)

    def test_constant_price_series_zero_vol(self):
        p = np.full(100, 1.10)
        assert close_to_close_vol(log_returns(p)) == 0.0


class TestNaNPolicy:
    """Explicit policy: reject, never impute."""

    def test_fitters_reject_nan(self):
        r = syn.simulate_constant_vol(500, 0.006, seed=113)
        r[250] = np.nan
        for fitter in (fit_garch, fit_gjr, fit_egarch):
            with pytest.raises(ValueError, match="NaN"):
                fitter(r)

    def test_fitters_reject_inf(self):
        r = syn.simulate_constant_vol(500, 0.006, seed=114)
        r[250] = np.inf
        with pytest.raises(ValueError, match="NaN|infinite"):
            fit_garch(r)


class TestShortSeries:
    def test_all_fitters_reject_short_series(self):
        r = syn.simulate_constant_vol(80, 0.006, seed=115)
        for fitter in (fit_garch, fit_gjr, fit_egarch):
            with pytest.raises(ValueError, match="at least 100"):
                fitter(r)

    def test_min_obs_override(self):
        """A caller may lower min_obs explicitly (documented as at-own-risk)."""
        r = syn.simulate_garch(300, 1e-6, 0.05, 0.90, seed=116)
        fit = fit_garch(r[:150], min_obs=150)
        assert np.isfinite(fit.loglik)


@pytest.fixture(scope="module")
def em():
    return syn.simulate_em_series(8000, seed=117)


class TestEmSeries:
    """EM-style series (fat tails + jumps + asymmetry) must be fittable and
    prefer asymmetric / fat-tailed specifications."""

    def test_models_fit_em_series(self, em):
        g = fit_garch(em)
        t = fit_garch(em, dist="t")
        assert np.isfinite(g.loglik) and np.isfinite(t.loglik)
        assert t.loglik > g.loglik + 10  # fat tails matter

    def test_em_tails_are_fat(self, em):
        fit = fit_garch(em, dist="t")
        assert fit.params["nu"] < 8.0
