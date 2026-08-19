"""Property-based invariants and extreme-regime edge cases.

Complements test_edge_cases.py (which covers degenerate/NaN/short/crisis
inputs) with checks that constrain the *shape* of the estimators:

* scale equivariance — multiplying returns by c must multiply every variance
  estimate by c^2 and leave alpha/beta/persistence untouched;
* monotonicity — EWMA responsiveness in lambda, realized vol in the scale of
  the data, the GARCH forecast term structure in the horizon;
* news-impact-curve geometry (asymmetry direction, minimum location);
* vol -> 0 and vol -> infinity regimes (contract item 6);
* boundary lambda / horizon / window values.
"""

import math

import numpy as np
import pytest

from eq_vol.data import synthetic as syn
from eq_vol.egarch import egarch_recursion, fit_egarch
from eq_vol.egarch import news_impact_curve as egarch_nic
from eq_vol.evaluation import mse_loss, qlike_loss
from eq_vol.ewma import ewma_forecast, ewma_variance, lambda_to_halflife
from eq_vol.forecasting import forecast_garch, term_structure
from eq_vol.garch import fit_garch, garch_recursion, unconditional_variance
from eq_vol.gjr import fit_gjr
from eq_vol.gjr import news_impact_curve as gjr_nic
from eq_vol.historical import realized_vol, window_sensitivity


# ---------------------------------------------------------------------------
# scale equivariance
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def returns():
    """Shared GARCH sample for the scale-equivariance checks."""
    return syn.simulate_garch(3000, seed=707).returns


class TestScaleEquivariance:
    """r -> c*r must scale variances by c^2 and leave the dynamics unchanged.

    This is the single strongest structural check on a variance model: it
    catches any place where an absolute (rather than relative) threshold or a
    hard-coded scale has crept into the recursion or the likelihood.
    """

    @pytest.mark.parametrize("c", [0.5, 2.0, 10.0])
    def test_ewma_variance_scales_quadratically(self, returns, c):
        base = ewma_variance(returns)
        scaled = ewma_variance(c * returns)
        np.testing.assert_allclose(scaled, c**2 * base, rtol=1e-12)

    @pytest.mark.parametrize("c", [0.5, 2.0, 10.0])
    def test_realized_vol_scales_linearly(self, returns, c):
        base = realized_vol(returns, window=21)
        scaled = realized_vol(c * returns, window=21)
        np.testing.assert_allclose(scaled[20:], c * base[20:], rtol=1e-12)

    @pytest.mark.parametrize("c", [0.5, 2.0])
    def test_garch_recursion_scales_quadratically(self, returns, c):
        om, al, be, b = 5e-6, 0.05, 0.90, 1e-4
        base = garch_recursion(returns, om, al, be, b)
        scaled = garch_recursion(c * returns, c**2 * om, al, be, c**2 * b)
        np.testing.assert_allclose(scaled, c**2 * base, rtol=1e-12)

    def test_garch_fit_dynamics_invariant_to_scale(self, returns):
        """alpha, beta and persistence are scale-free; omega scales by c^2."""
        c = 2.0
        base = fit_garch(returns)
        scaled = fit_garch(c * returns)
        assert scaled.params["alpha"] == pytest.approx(base.params["alpha"], abs=1e-4)
        assert scaled.params["beta"] == pytest.approx(base.params["beta"], abs=1e-4)
        assert scaled.params["omega"] == pytest.approx(
            c**2 * base.params["omega"], rel=1e-3
        )
        assert scaled.extra["persistence"] == pytest.approx(
            base.extra["persistence"], abs=1e-4
        )

    def test_egarch_recursion_scale_shift_in_log_space(self, returns):
        """In log-variance the scale enters additively: omega -> omega +
        (1-beta) ln c^2 and init_var -> c^2 init_var reproduce c^2 sigma2."""
        c = 3.0
        om, al, ga, be, b = -0.40, 0.10, -0.08, 0.96, 1e-4
        base = egarch_recursion(returns, om, al, ga, be, b)
        shifted = egarch_recursion(
            c * returns, om + (1.0 - be) * math.log(c**2), al, ga, be, c**2 * b
        )
        np.testing.assert_allclose(shifted, c**2 * base, rtol=1e-10)


# ---------------------------------------------------------------------------
# monotonicity properties
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_ewma_lower_lambda_reacts_more_to_a_shock(self):
        """Smaller lambda = shorter memory = bigger jump after a shock."""
        r = np.concatenate([np.full(100, 0.001), [0.10], np.zeros(5)])
        jumps = []
        for lam in (0.80, 0.90, 0.94, 0.99):
            s2 = ewma_variance(r, lam=lam)
            jumps.append(s2[101] - s2[100])
        assert all(b < a for a, b in zip(jumps, jumps[1:]))

    def test_ewma_halflife_increasing_in_lambda(self):
        """Half-life is increasing in lambda (higher lambda = longer memory)."""
        hl = [lambda_to_halflife(lam) for lam in (0.80, 0.90, 0.94, 0.99)]
        assert all(b > a for a, b in zip(hl, hl[1:]))
        assert lambda_to_halflife(1.0) == np.inf

    def test_garch_term_structure_monotone_toward_unconditional(self):
        """From above or below, E_T[sigma2_{T+k}] moves monotonically to the
        long-run level and converges to it."""
        res = fit_garch(syn.simulate_garch(3000, seed=709).returns)
        f = forecast_garch(res, horizon=400)
        uv = unconditional_variance(
            res.params["omega"], res.params["alpha"], res.params["beta"]
        )
        d = np.diff(f)
        assert np.all(d > 0) or np.all(d < 0)
        assert f[-1] == pytest.approx(uv, rel=1e-4)
        # every step must reduce the distance to the long-run level
        gaps = np.abs(f - uv)
        assert np.all(np.diff(gaps) <= 1e-18)

    def test_term_structure_avg_vol_between_spot_and_forward_limits(self):
        """avg_vol is a cumulative average, so it is bracketed by the
        forward-vol path and moves monotonically in the same direction."""
        res = fit_garch(syn.simulate_garch(3000, seed=710).returns)
        ts = term_structure(res, horizon=250)
        fwd = ts["forward_vol_annual"].to_numpy()
        avg = ts["avg_vol_annual"].to_numpy()
        assert avg[0] == pytest.approx(fwd[0], rel=1e-12)
        assert np.all(avg >= np.minimum.accumulate(fwd) - 1e-12)
        assert np.all(avg <= np.maximum.accumulate(fwd) + 1e-12)
        assert np.all(np.diff(avg) * np.sign(fwd[-1] - fwd[0]) >= -1e-12)

    def test_realized_vol_window_noise_decreases_with_window(self):
        """Sampling std of the rolling estimate shrinks like 1/sqrt(window)."""
        r = syn.simulate_gbm_returns(3000, sigma_annual=0.20, seed=711)
        tbl = window_sensitivity(r, windows=(10, 21, 63, 252))
        stds = tbl["std_of_estimate"].to_numpy()
        assert all(b < a for a, b in zip(stds, stds[1:]))
        # and it tracks the theoretical sigma / sqrt(2 w) within a factor of 3
        ratio = stds / tbl["approx_sampling_std"].to_numpy()
        assert np.all((ratio > 0.3) & (ratio < 3.0))


# ---------------------------------------------------------------------------
# news impact curve geometry
# ---------------------------------------------------------------------------

class TestNewsImpactGeometry:
    def test_gjr_curve_minimum_at_zero_and_left_branch_higher(self):
        z, s2 = gjr_nic(5e-6, 0.03, 0.10, 0.88)
        assert z[int(np.argmin(s2))] == pytest.approx(0.0, abs=1e-12)
        # leverage: equal-size negative shock gives more variance
        for x in (0.5, 1.0, 2.0, 4.0):
            left = np.interp(-x, z, s2)
            right = np.interp(x, z, s2)
            assert left > right

    def test_gjr_curve_convex_in_z(self):
        _, s2 = gjr_nic(5e-6, 0.03, 0.10, 0.88, z_grid=np.linspace(0.0, 5.0, 201))
        second = np.diff(s2, 2)
        assert np.all(second >= -1e-18)

    def test_egarch_curve_positive_and_asymmetric_with_negative_gamma(self):
        z, s2 = egarch_nic(-0.40, 0.10, -0.08, 0.96)
        assert np.all(s2 > 0)
        for x in (0.5, 1.0, 2.0, 4.0):
            assert np.interp(-x, z, s2) > np.interp(x, z, s2)

    def test_egarch_curve_symmetric_and_min_at_zero_when_gamma_zero(self):
        z, s2 = egarch_nic(-0.40, 0.10, 0.0, 0.96)
        np.testing.assert_allclose(s2, s2[::-1], rtol=1e-12)
        assert z[int(np.argmin(s2))] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# extreme volatility regimes (contract item 6: vol -> 0, vol -> infinity)
# ---------------------------------------------------------------------------

class TestExtremeVolRegimes:
    def test_near_zero_vol_series_fits_without_underflow(self):
        """1e-6 annualised vol: daily variance ~4e-15. Everything must stay
        positive and finite — no underflow to zero variance."""
        r = syn.simulate_gbm_returns(1500, sigma_annual=1e-6, seed=712)
        res = fit_garch(r)
        assert res.converged
        assert np.all(res.sigma2 > 0) and np.all(np.isfinite(res.sigma2))
        assert res.params["omega"] > 0
        assert np.all(ewma_variance(r) > 0)
        assert np.all(np.isfinite(realized_vol(r, window=21)[20:]))

    def test_extreme_vol_series_fits_without_overflow(self):
        """800% annualised vol: daily sigma ~0.5. No overflow, no NaN."""
        r = syn.simulate_gbm_returns(1500, sigma_annual=8.0, seed=713)
        res = fit_garch(r)
        assert res.converged
        assert np.all(np.isfinite(res.sigma2)) and np.all(res.sigma2 > 0)
        assert np.all(np.isfinite(ewma_variance(r)))

    def test_egarch_log_clip_keeps_recursion_finite_on_explosive_params(self):
        """The log-variance clip is what stops exp() overflowing during a bad
        optimiser excursion; the recursion must stay finite and positive."""
        r = syn.simulate_garch(500, seed=714).returns
        s2 = egarch_recursion(r, omega=5.0, alpha=3.0, gamma=-2.0, beta=0.99,
                              init_var=1e-4)
        assert np.all(np.isfinite(s2)) and np.all(s2 > 0)

    def test_standardised_residuals_have_unit_variance_in_both_regimes(self):
        for sigma_annual, seed in ((1e-6, 715), (8.0, 716)):
            r = syn.simulate_gbm_returns(1500, sigma_annual=sigma_annual, seed=seed)
            z = fit_garch(r).std_residuals
            assert np.std(z) == pytest.approx(1.0, abs=0.15)


# ---------------------------------------------------------------------------
# boundary parameter values
# ---------------------------------------------------------------------------

class TestBoundaryValues:
    def test_lambda_at_upper_boundary_is_flat(self):
        r = syn.simulate_garch(300, seed=717).returns
        s2 = ewma_variance(r, lam=1.0)
        assert np.ptp(s2) == 0.0
        np.testing.assert_allclose(ewma_forecast(r, horizon=5, lam=1.0), s2[0])

    def test_tiny_lambda_is_almost_pure_last_squared_return(self):
        r = syn.simulate_garch(300, seed=718).returns
        s2 = ewma_variance(r, lam=1e-8)
        np.testing.assert_allclose(s2[1:], r[:-1] ** 2, atol=1e-9)

    @pytest.mark.parametrize("lam", [0.0, -0.1, 1.5, np.nan])
    def test_lambda_outside_unit_interval_raises(self, lam):
        r = syn.simulate_garch(300, seed=719).returns
        with pytest.raises(ValueError, match="lambda"):
            ewma_variance(r, lam=lam)

    def test_horizon_one_is_a_valid_boundary(self):
        res = fit_garch(syn.simulate_garch(1500, seed=720).returns)
        f = forecast_garch(res, horizon=1)
        assert f.size == 1 and f[0] > 0
        assert term_structure(res, horizon=1).shape == (1, 2)

    @pytest.mark.parametrize("horizon", [0, -5])
    def test_nonpositive_horizon_raises(self, horizon):
        res = fit_garch(syn.simulate_garch(1500, seed=721).returns)
        with pytest.raises(ValueError, match="horizon"):
            forecast_garch(res, horizon=horizon)

    def test_window_equal_to_sample_length_is_allowed(self):
        r = syn.simulate_garch(300, seed=722).returns
        vol = realized_vol(r, window=r.size)
        assert np.isnan(vol[:-1]).all() and np.isfinite(vol[-1])


# ---------------------------------------------------------------------------
# loss-function properties
# ---------------------------------------------------------------------------

class TestLossProperties:
    def test_qlike_scale_sensitive_mse_scale_sensitive_but_ranking_agrees(self):
        """Both losses must prefer the truth over a 2x-biased forecast at any
        data scale (Patton robustness on the ranking, not the level)."""
        rng = np.random.default_rng(723)
        truth = rng.uniform(0.5e-4, 2e-4, 500)
        proxy = truth * rng.chisquare(1, 500)  # unbiased noisy proxy
        for c in (1.0, 100.0):
            t = truth * c**2
            p = proxy * c**2
            assert qlike_loss(t, p).mean() < qlike_loss(2.0 * t, p).mean()
            assert qlike_loss(t, p).mean() < qlike_loss(0.5 * t, p).mean()
            assert mse_loss(t, p).mean() < mse_loss(2.0 * t, p).mean()

    def test_qlike_penalises_underprediction_more_than_overprediction(self):
        """The asymmetry a risk desk relies on: halving the forecast hurts
        more than doubling it."""
        f = np.full(200, 1e-4)
        p = np.full(200, 1e-4)
        under = qlike_loss(0.5 * f, p).mean()
        over = qlike_loss(2.0 * f, p).mean()
        base = qlike_loss(f, p).mean()
        assert under > base and over > base
        assert under - base > over - base


# ---------------------------------------------------------------------------
# leverage models vs symmetric data (specification sanity)
# ---------------------------------------------------------------------------

def test_asymmetric_models_reduce_to_symmetric_on_symmetric_data():
    """Fitted on data with no leverage, GJR's gamma and EGARCH's gamma must be
    small, and the implied persistence must stay close to the GARCH fit."""
    sim = syn.simulate_garch(6000, seed=724)
    g = fit_garch(sim.returns)
    j = fit_gjr(sim.returns)
    e = fit_egarch(sim.returns)
    assert abs(j.params["gamma"]) < 0.06
    assert abs(e.params["gamma"]) < 0.06
    assert j.extra["persistence"] == pytest.approx(g.extra["persistence"], abs=0.05)
    # and the extra parameter cannot lower the maximised likelihood
    assert j.loglik >= g.loglik - 1e-6
