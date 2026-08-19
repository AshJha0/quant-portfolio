"""Further FX edge cases: tiny/empty samples, quote-direction asymmetry,
managed-band regimes, GARCH-X misuse, vol-triangle degeneracies, VRP in
crisis, and forecast-horizon limits.

Complements tests/test_edge_cases.py; each case is documented in
docs/VALIDATION.md §5.
"""

import numpy as np
import pandas as pd
import pytest

from fx_vol import (
    close_to_close_vol,
    cross_volatility,
    ewma_forecast,
    ewma_variance,
    ewma_weights,
    fit_egarch,
    fit_garch,
    fit_gjr,
    forecast_variance,
    garman_klass_vol,
    invert_returns,
    log_returns,
    parkinson_vol,
    premium_summary,
    qlike_loss,
    realized_vol_forward,
    variance_swap_pnl,
    vol_risk_premium,
)
from fx_vol.data import synthetic as syn
from fx_vol.garch import garch_filter


class TestTinyAndEmptySamples:
    """Every entry point must refuse degenerate sample sizes explicitly."""

    def test_empty_series_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            log_returns([])
        with pytest.raises(ValueError, match="empty"):
            close_to_close_vol([])

    def test_single_observation_rejected(self):
        with pytest.raises(ValueError):
            close_to_close_vol([0.001])
        with pytest.raises(ValueError, match="at least 2"):
            ewma_variance([0.001])

    def test_two_observations_is_the_floor_not_an_error(self):
        v = close_to_close_vol([0.01, -0.01])
        assert np.isfinite(v) and v > 0
        s2 = ewma_variance([0.01, -0.01])
        assert s2.size == 2 and np.all(np.isfinite(s2))

    def test_range_estimators_reject_length_one(self):
        with pytest.raises(ValueError):
            parkinson_vol([1.10], [1.09])
        with pytest.raises(ValueError):
            garman_klass_vol([1.10], [1.11], [1.09], [1.10])

    def test_premium_summary_rejects_all_nan(self):
        with pytest.raises(ValueError, match="no finite"):
            premium_summary(np.full(10, np.nan))

    def test_realized_vol_forward_window_bounds(self):
        r = syn.simulate_constant_vol(50, 0.006, seed=1)
        with pytest.raises(ValueError, match="window"):
            realized_vol_forward(r, window=1)
        with pytest.raises(ValueError, match="window"):
            realized_vol_forward(r, window=51)


class TestConstantAndZeroInputs:
    def test_constant_nonzero_returns_are_still_constant(self):
        # A series pinned at a single non-zero value has zero variance:
        # the MLE is degenerate and must be refused, not fitted.
        const = np.full(500, 0.0004)
        for fitter in (fit_garch, fit_gjr, fit_egarch):
            with pytest.raises(ValueError, match="constant"):
                fitter(const)

    def test_ewma_on_constant_series_is_flat_and_positive(self):
        s2 = ewma_variance(np.full(200, 0.0004))
        assert np.allclose(s2, 0.0004 ** 2, atol=1e-18)

    def test_zero_realized_vol_gives_full_premium(self):
        # A frozen (fixed) market realises zero vol: the whole implied vol
        # is premium, and a short variance swap earns K/2 in vega terms.
        iv = np.full(5, 0.08)
        rv = np.zeros(5)
        assert np.allclose(vol_risk_premium(iv, rv), 0.08)
        assert np.allclose(variance_swap_pnl(iv, rv), 0.04)

    def test_qlike_allows_zero_realized_but_not_zero_forecast(self):
        loss = qlike_loss([1e-4, 2e-4], [0.0, 0.0])
        assert np.allclose(loss, np.log([1e-4, 2e-4]))
        with pytest.raises(ValueError, match="strictly positive"):
            qlike_loss([0.0, 1e-4], [1e-4, 1e-4])


class TestQuoteDirectionAsymmetry:
    """Asymmetry is a property of the quote direction, not of the market."""

    @staticmethod
    @pytest.fixture(scope="class")
    def em():
        # USD/EM direction: vol rises when the pair RISES (EM sells off).
        return syn.simulate_em_series(6000, seed=311)

    def test_next_day_vol_is_higher_after_em_depreciation(self, em):
        # Ground truth for the quote direction: on USD/EM, vol follows
        # POSITIVE pair returns (the EM currency selling off).
        after_up = np.abs(em[1:])[em[:-1] > 0].mean()
        after_down = np.abs(em[1:])[em[:-1] < 0].mean()
        assert after_up > after_down

    def test_gjr_gamma_flips_side_under_inversion(self, em):
        # GJR constrains gamma >= 0 and loads it on NEGATIVE returns.  On
        # the USD/EM quote the asymmetry sits on positive returns, so gamma
        # pins at the 0 boundary and the fit is strictly worse; inverting
        # the pair moves the asymmetry onto the side GJR can represent.
        # Fitted with the t likelihood: on jumpy EM data the Gaussian MLE
        # chases outliers and obscures the effect (docs/VALIDATION.md §4).
        quoted = fit_gjr(em, dist="t")
        inverted = fit_gjr(invert_returns(em), dist="t")
        assert quoted.params["gamma"] == pytest.approx(0.0, abs=1e-6)
        assert inverted.params["gamma"] > 0.02
        assert inverted.loglik > quoted.loglik

    def test_egarch_leverage_sign_flips_exactly_under_inversion(self, em):
        # EGARCH's gamma is sign-unconstrained, so inversion flips its sign
        # exactly and leaves everything else -- including the likelihood --
        # untouched.  This is why EGARCH, not GJR, is the quote-direction-
        # agnostic choice for FX.
        a = fit_egarch(em, dist="t")
        b = fit_egarch(invert_returns(em), dist="t")
        assert a.params["gamma"] == pytest.approx(-b.params["gamma"], abs=1e-5)
        assert a.params["alpha"] == pytest.approx(b.params["alpha"], abs=1e-5)
        assert a.params["beta"] == pytest.approx(b.params["beta"], abs=1e-5)
        assert a.params["omega"] == pytest.approx(b.params["omega"], abs=1e-4)
        assert a.loglik == pytest.approx(b.loglik, abs=1e-4)

    def test_symmetric_g10_series_shows_no_asymmetry_either_direction(self):
        r = syn.simulate_garch(6000, 1e-6, 0.05, 0.92, seed=312)
        for series in (r, invert_returns(r)):
            fit = fit_gjr(series)
            assert fit.params["gamma"] < 0.05


class TestManagedBandRegime:
    """A tightly managed pair (CNY-style band) has vol but no GARCH story."""

    @staticmethod
    @pytest.fixture(scope="class")
    def band():
        return syn.simulate_pegged(2500, band_vol=1.5e-4, seed=313)

    def test_all_fitters_survive_the_band(self, band):
        for fitter in (fit_garch, fit_gjr, fit_egarch):
            fit = fitter(band)
            assert np.isfinite(fit.loglik)
            assert np.all(np.isfinite(fit.sigma2)) and np.all(fit.sigma2 > 0)

    def test_band_vol_recovered_to_right_order_of_magnitude(self, band):
        assert close_to_close_vol(band, periods_per_year=252) == pytest.approx(
            1.5e-4 * np.sqrt(252), rel=0.1)

    def test_scale_invariance_across_six_orders_of_magnitude(self, band):
        # Internal unit-variance rescaling must make alpha/beta identical
        # whether the caller passes decimals or basis points.
        a = fit_garch(band)
        b = fit_garch(band * 1e4)
        assert a.params["alpha"] == pytest.approx(b.params["alpha"], abs=1e-4)
        assert a.params["beta"] == pytest.approx(b.params["beta"], abs=1e-4)
        assert b.params["omega"] == pytest.approx(a.params["omega"] * 1e8,
                                                  rel=1e-2)
        # Log-likelihood maps by the exact change-of-variables constant.
        assert b.loglik == pytest.approx(a.loglik - band.size * np.log(1e4),
                                         rel=1e-6)

    def test_band_forecast_never_explodes(self, band):
        fit = fit_garch(band)
        f = forecast_variance(fit, 500)
        assert np.all(np.isfinite(f)) and np.all(f > 0)
        assert np.sqrt(252 * f.max()) < 0.02


class TestGarchXMisuse:
    """GARCH-X regressor/coefficient plumbing must fail loudly, not silently."""

    def test_x_without_gamma_x_raises_instead_of_returning_nan(self):
        r = syn.simulate_constant_vol(300, 0.006, seed=314)
        x = (np.arange(300) % 20 == 0).astype(float)
        with pytest.raises(ValueError, match="without gamma_x"):
            garch_filter(r, 1e-6, 0.05, 0.90, x=x)

    def test_gamma_x_without_x_raises(self):
        r = syn.simulate_constant_vol(300, 0.006, seed=315)
        with pytest.raises(ValueError, match="without x"):
            garch_filter(r, 1e-6, 0.05, 0.90, gamma_x=[5e-5])

    def test_nan_gamma_x_rejected(self):
        r = syn.simulate_constant_vol(300, 0.006, seed=316)
        x = (np.arange(300) % 20 == 0).astype(float)
        with pytest.raises(ValueError, match="gamma_x contains NaN"):
            garch_filter(r, 1e-6, 0.05, 0.90, gamma_x=[np.nan], x=x)

    def test_negative_event_dummy_rejected(self):
        r = syn.simulate_constant_vol(300, 0.006, seed=317)
        x = -np.ones(300)
        with pytest.raises(ValueError, match="non-negative"):
            garch_filter(r, 1e-6, 0.05, 0.90, gamma_x=[5e-5], x=x)

    def test_variance_targeting_with_x_rejected(self):
        r, x = syn.simulate_garch_x(400, 1e-6, 0.05, 0.90, 5e-5, seed=318)
        with pytest.raises(ValueError, match="variance targeting"):
            fit_garch(r, x=x, variance_targeting=True)

    def test_x_future_on_plain_garch_rejected(self):
        r = syn.simulate_garch(400, 1e-6, 0.05, 0.90, seed=319)
        fit = fit_garch(r)
        with pytest.raises(ValueError):
            forecast_variance(fit, 5, x_future=np.ones((5, 1)))

    def test_event_calendar_raises_the_forecast_on_event_days(self):
        r, x = syn.simulate_garch_x(4000, 1e-6, 0.05, 0.88, 8e-5,
                                    event_prob=0.06, seed=320)
        fit = fit_garch(r, x=x)
        quiet = forecast_variance(fit, 5, x_future=np.zeros((5, 1)))
        cal = np.zeros((5, 1))
        cal[2, 0] = 1.0  # FOMC on day 3
        eventful = forecast_variance(fit, 5, x_future=cal)
        assert eventful[2] > quiet[2]
        assert eventful[0] == pytest.approx(quiet[0])
        # The bump propagates into later horizons through beta.
        assert eventful[4] > quiet[4]


class TestVolTriangleDegeneracies:
    def test_perfect_negative_correlation_equal_vols_gives_zero(self):
        # Exact cancellation: the cross is a constant, vol 0 (and the
        # numerical guard must not return NaN from a negative variance).
        v = cross_volatility(0.10, 0.10, -1.0, sign_product=1)
        assert v == pytest.approx(0.0, abs=1e-12)
        assert np.isfinite(v)

    def test_perfect_positive_correlation_adds_linearly(self):
        assert cross_volatility(0.08, 0.11, 1.0,
                                sign_product=1) == pytest.approx(0.19)

    def test_sign_product_minus_one_subtracts(self):
        # Same legs, opposite quote direction: correlation enters negatively.
        assert cross_volatility(0.08, 0.11, 1.0,
                                sign_product=-1) == pytest.approx(0.03)

    def test_zero_leg_vol_gives_other_leg(self):
        assert cross_volatility(0.0, 0.09, 0.5,
                                sign_product=1) == pytest.approx(0.09)

    def test_out_of_range_correlation_rejected(self):
        with pytest.raises(ValueError, match="correlation"):
            cross_volatility(0.08, 0.11, 1.5, sign_product=1)

    def test_missing_sign_information_rejected(self):
        with pytest.raises(ValueError, match="sign_product"):
            cross_volatility(0.08, 0.11, 0.3)


class TestVolPremiumInCrisis:
    def test_negative_premium_when_realized_exceeds_implied(self):
        iv = np.array([0.08, 0.08, 0.08])
        rv = np.array([0.05, 0.30, 0.09])
        vrp = vol_risk_premium(iv, rv)
        assert vrp[1] < 0
        s = premium_summary(vrp)
        assert s["frac_positive"] == pytest.approx(1 / 3)
        assert s["min"] == pytest.approx(-0.22)

    def test_short_variance_swap_convexity_bites(self):
        # Selling 8 vol and realising 30: the loss is far more than the
        # 22-vol-point gap because the payoff is in VARIANCE.
        pnl = variance_swap_pnl(np.array([0.08]), np.array([0.30]))
        assert pnl[0] < -0.5
        assert pnl[0] == pytest.approx((0.08 ** 2 - 0.30 ** 2) / 0.16)

    def test_depeg_flips_a_profitable_program_negative(self):
        r = syn.simulate_depeg(600, jump_return=-0.15, jump_index=500,
                               base_vol=0.002, seed=321)
        rv = realized_vol_forward(r, window=20)
        valid = np.isfinite(rv)
        iv = np.full(r.size, 0.05)  # implied marked at peg-regime vol
        pnl = variance_swap_pnl(iv[valid], rv[valid])
        assert pnl.max() > 0            # harvesting in the quiet regime
        assert pnl.min() < -0.5         # the depeg window wipes it out
        assert pnl.sum() < 0            # program is net loss-making

    def test_implied_nan_rejected_realized_nan_tolerated(self):
        with pytest.raises(ValueError, match="implied vol contains NaN"):
            vol_risk_premium(np.array([0.08, np.nan]), np.array([0.05, 0.05]))
        out = vol_risk_premium(np.array([0.08, 0.08]),
                               np.array([0.05, np.nan]))
        assert np.isfinite(out[0]) and np.isnan(out[1])

    def test_negative_vols_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            vol_risk_premium(np.array([0.08]), np.array([-0.01]))


class TestForecastHorizonLimits:
    @staticmethod
    @pytest.fixture(scope="class")
    def fit():
        return fit_garch(syn.simulate_garch(5000, 1e-6, 0.05, 0.92, seed=322))

    def test_horizon_zero_rejected(self, fit):
        with pytest.raises(ValueError, match="horizon"):
            forecast_variance(fit, 0)

    def test_very_long_horizon_converges_to_unconditional(self, fit):
        f = forecast_variance(fit, 5000)
        assert f[-1] == pytest.approx(fit.unconditional_variance, rel=1e-6)
        assert np.all(np.isfinite(f))

    def test_ewma_forecast_is_flat_at_every_horizon(self):
        r = syn.simulate_garch(2000, 1e-6, 0.05, 0.92, seed=323)
        f = ewma_forecast(r, horizon=500)
        assert np.allclose(f, f[0])
        # ... unlike GARCH, which mean-reverts: the key modelling contrast.
        g = forecast_variance(fit_garch(r), 500)
        assert abs(g[-1] - g[0]) > 0.05 * g[0]

    def test_ewma_weights_sum_identity(self):
        for lam in (0.90, 0.94, 0.97):
            for n in (1, 10, 250):
                assert ewma_weights(lam, n).sum() == pytest.approx(
                    1.0 - lam ** n, abs=1e-15)

    def test_ewma_rejects_bad_lambda(self):
        for bad in (0.0, 1.0, -0.5, 1.5):
            with pytest.raises(ValueError, match="lambda"):
                ewma_variance([0.01, -0.01, 0.02], lam=bad)


class TestWeekendAndSeasonality:
    def test_weekend_stamps_rejected(self):
        idx = pd.date_range("2020-01-01", periods=30, freq="D")  # incl. w/e
        s = pd.Series(np.random.default_rng(0).standard_normal(30) * 0.006,
                      index=idx)
        with pytest.raises(ValueError, match="weekend"):
            from fx_vol import day_of_week_vol_factors
            day_of_week_vol_factors(s)

    def test_annualisation_convention_gap_is_material(self):
        r = syn.simulate_constant_vol(2000, 0.006, seed=324)
        v252 = close_to_close_vol(r, periods_per_year=252)
        v260 = close_to_close_vol(r, periods_per_year=260)
        assert v260 / v252 == pytest.approx(np.sqrt(260 / 252), rel=1e-12)
        assert v260 - v252 > 0.0014  # > 0.14 vol points on a ~9.5 vol series
