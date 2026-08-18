"""Synthetic generators: determinism, moments, structural properties."""

import numpy as np
import pandas as pd
import pytest

from fx_vol.data import synthetic as syn


class TestDeterminism:
    @pytest.mark.parametrize(
        "gen,kwargs",
        [
            (syn.simulate_garch, dict(n=500, omega=1e-6, alpha=0.05, beta=0.9)),
            (syn.simulate_gjr, dict(n=500, omega=1e-6, alpha=0.03, gamma=0.1, beta=0.85)),
            (syn.simulate_egarch, dict(n=500, omega=-0.5, alpha=0.15, gamma=-0.05, beta=0.95)),
            (syn.simulate_constant_vol, dict(n=500, vol=0.006)),
            (syn.simulate_pegged, dict(n=500)),
            (syn.simulate_em_series, dict(n=500)),
        ],
    )
    def test_same_seed_same_series(self, gen, kwargs):
        a = gen(seed=7, **kwargs)
        b = gen(seed=7, **kwargs)
        c = gen(seed=8, **kwargs)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)


class TestMoments:
    def test_garch_sample_variance_matches_unconditional(self):
        omega, alpha, beta = 1e-6, 0.05, 0.90
        r = syn.simulate_garch(100_000, omega, alpha, beta, seed=121)
        assert np.var(r) == pytest.approx(omega / (1 - alpha - beta), rel=0.05)

    def test_t_innovations_are_unit_variance(self):
        r = syn.simulate_constant_vol(200_000, 1.0, dist="t", nu=5.0, seed=122)
        assert np.var(r) == pytest.approx(1.0, rel=0.03)

    def test_gjr_has_negative_skew_dynamics(self):
        """Negative returns beget higher vol: corr(r_t, |r_{t+1}|) < 0."""
        r = syn.simulate_gjr(100_000, 1e-6, 0.03, 0.15, 0.82, seed=123)
        c = np.corrcoef(r[:-1], np.abs(r[1:]))[0, 1]
        assert c < -0.01

    def test_em_series_positive_jump_skew(self):
        """USD/EM quote: depreciation jumps are positive returns -> skew > 0."""
        r = syn.simulate_em_series(50_000, seed=124)
        skew = np.mean((r - r.mean()) ** 3) / np.std(r) ** 3
        assert skew > 0.2


class TestStructure:
    def test_depeg_jump_location_and_size(self):
        r = syn.simulate_depeg(1000, jump_return=-0.15, jump_index=700, seed=125)
        assert r[700] == -0.15
        # the jump is the largest move; post-jump vol is elevated but decays
        assert np.abs(np.delete(r, 700)).max() < 0.15
        assert np.std(r[701:721]) > 3.0 * np.std(r[600:700])

    def test_correlated_pairs_hit_target_correlation(self):
        sim = syn.simulate_correlated_pairs(100_000, 0.006, 0.008, rho=-0.55, seed=126)
        rho_hat = np.corrcoef(sim["returns1"], sim["returns2"])[0, 1]
        assert rho_hat == pytest.approx(-0.55, abs=0.01)
        assert len(sim["prices1"]) == len(sim["returns1"]) + 1

    def test_garch_x_events_have_higher_variance(self):
        r, x = syn.simulate_garch_x(
            50_000, omega=1e-6, alpha=0.05, beta=0.90, gamma_x=5e-5, event_prob=0.05, seed=127
        )
        mask = x.astype(bool)
        assert 0.03 < mask.mean() < 0.07
        assert np.var(r[mask]) > 1.5 * np.var(r[~mask])

    def test_seasonal_returns_business_days_only(self):
        r = syn.simulate_seasonal_returns(500, seed=128)
        assert isinstance(r.index, pd.DatetimeIndex)
        assert (r.index.dayofweek < 5).all()

    def test_validation(self):
        with pytest.raises(ValueError):
            syn.simulate_garch(100, omega=1e-6, alpha=0.5, beta=0.6)  # nonstationary
        with pytest.raises(ValueError):
            syn.simulate_depeg(100, jump_return=0.0)
        with pytest.raises(ValueError):
            syn.simulate_constant_vol(100, 0.01, dist="t")  # nu missing
