"""Deterministic synthetic generators: reproducibility and ground truth."""

import numpy as np
import pandas as pd
import pytest

from eq_pairs.cointegration import adf_test, engle_granger
from eq_pairs.data import (
    business_index,
    cointegrated_pair,
    correlated_random_walks,
    mixed_panel,
    regime_break_pair,
    simulate_ou,
)
from eq_pairs.universe import pair_correlations


class TestReproducibility:
    def test_same_seed_same_data(self):
        for gen in (cointegrated_pair, correlated_random_walks, regime_break_pair):
            a, _ = gen(n=200, seed=5)
            b, _ = gen(n=200, seed=5)
            pd.testing.assert_frame_equal(a, b)

    def test_different_seed_different_data(self):
        a, _ = cointegrated_pair(n=200, seed=1)
        b, _ = cointegrated_pair(n=200, seed=2)
        assert not a.equals(b)

    def test_mixed_panel_reproducible(self):
        a, _ = mixed_panel(n=150, seed=8)
        b, _ = mixed_panel(n=150, seed=8)
        pd.testing.assert_frame_equal(a, b)


class TestSimulateOU:
    def test_stationary_moments(self):
        kappa, sigma, mu = 0.1, 1.0, 3.0
        x = simulate_ou(60000, kappa=kappa, sigma=sigma, mu=mu, seed=6)
        stat_sd = sigma / np.sqrt(2 * kappa)
        assert x.mean() == pytest.approx(mu, abs=4 * stat_sd / np.sqrt(60000 * kappa))
        assert x.std() == pytest.approx(stat_sd, rel=0.05)

    def test_kappa_zero_is_random_walk(self):
        x = simulate_ou(1000, kappa=0.0, sigma=1.0, mu=0.0, x0=0.0, seed=7)
        increments = np.diff(x)
        assert increments.std() == pytest.approx(1.0, rel=0.1)
        assert not adf_test(x, regression="c").reject("10%")

    def test_x0_respected(self):
        x = simulate_ou(10, kappa=0.5, sigma=1.0, x0=42.0, seed=8)
        assert x[0] == 42.0

    def test_validations(self):
        with pytest.raises(ValueError, match="n must be positive"):
            simulate_ou(0, 0.1, 1.0)
        with pytest.raises(ValueError, match="kappa"):
            simulate_ou(10, -0.1, 1.0)
        with pytest.raises(ValueError, match="sigma"):
            simulate_ou(10, 0.1, -1.0)


class TestCointegratedPair:
    def test_truth_and_shapes(self):
        df, truth = cointegrated_pair(n=500, beta=1.5, kappa=0.05, sigma=1.0, seed=9)
        assert list(df.columns) == ["Y", "X"]
        assert len(df) == 500
        assert (df > 0).all().all()
        assert isinstance(df.index, pd.DatetimeIndex)
        assert truth.kind == "cointegrated"
        assert truth.half_life == pytest.approx(np.log(2) / 0.05)

    def test_spread_is_the_planted_ou(self):
        df, truth = cointegrated_pair(n=2000, beta=1.5, alpha=2.0, kappa=0.08,
                                      sigma=1.0, seed=10)
        s = df["Y"] - truth.beta * df["X"] - truth.alpha
        assert adf_test(s.to_numpy(), regression="c").reject("1%")


class TestTrapPair:
    def test_high_return_corr_but_not_cointegrated(self):
        df, truth = correlated_random_walks(n=1500, rho=0.92, seed=11)
        corr = pair_correlations(df, [("A", "B")]).iloc[0, 0]
        assert corr > 0.8  # passes any correlation screen...
        eg = engle_granger(df["A"].to_numpy(), df["B"].to_numpy())
        assert not eg.cointegrated("10%")  # ...but is NOT cointegrated
        assert truth.kind == "correlated_rw"
        assert np.isnan(truth.beta)

    def test_invalid_rho_raises(self):
        with pytest.raises(ValueError, match="rho"):
            correlated_random_walks(n=100, rho=1.0)


class TestRegimeBreakPair:
    def test_break_structure(self):
        df, truth = regime_break_pair(n=1200, break_frac=0.5, beta=1.2,
                                      kappa=0.06, sigma=0.8, seed=12)
        assert truth.break_index == 600
        s = (df["Y"] - truth.beta * df["X"]).to_numpy()
        pre, post = s[:600], s[600:]
        # pre-break spread is stationary; post-break it walks away
        assert adf_test(pre, regression="c").reject("5%")
        assert not adf_test(post, regression="c", lags=1).reject("10%")
        assert abs(post[-1] - post[0]) > 10 * pre.std()

    def test_invalid_break_frac_raises(self):
        with pytest.raises(ValueError, match="break_frac"):
            regime_break_pair(n=100, break_frac=1.5)


class TestMixedPanel:
    def test_structure_and_truth(self):
        prices, truth = mixed_panel(
            n=300, n_cointegrated=4, n_trap=3, n_break=1, n_idiosyncratic=4, seed=13
        )
        assert prices.shape == (300, 2 * (4 + 3 + 1) + 4)
        assert set(truth.sectors) == set(prices.columns)
        assert len(truth.cointegrated_pairs()) == 4
        assert len(truth.trap_pairs()) == 3
        assert len(truth.break_pairs()) == 1
        for (a, b) in truth.pairs:
            assert truth.sectors[a] == truth.sectors[b]  # pairs share a sector

    def test_prices_positive_no_nan(self):
        prices, _ = mixed_panel(n=300, seed=14)
        assert prices.notna().all().all()
        assert (prices > 0).all().all()


class TestBusinessIndex:
    def test_length_and_freq(self):
        idx = business_index(10)
        assert len(idx) == 10
        assert idx.dayofweek.max() <= 4  # weekdays only

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="n must be positive"):
            business_index(0)
