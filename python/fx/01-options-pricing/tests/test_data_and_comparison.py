"""Synthetic data generators and the cross-model comparison harness."""

import math

import numpy as np
import pytest

from fx_options import (binomial_convergence_table, compare_models,
                        gk_price, mc_convergence_table)
from fx_options.data.synthetic import (gbm_fx_paths, synthetic_vol_quotes)

MKT = dict(S=1.10, K=1.1075, T=0.5, r_d=0.0425, r_f=0.0290, sigma=0.0825)


class TestGBMPaths:
    def test_shape_and_initial_value(self):
        paths = gbm_fx_paths(1.10, 1.0, 0.03, 0.01, 0.10, n_steps=50,
                             n_paths=200, rng=0)
        assert paths.shape == (200, 51)
        assert np.all(paths[:, 0] == 1.10)
        assert np.all(paths > 0)

    def test_deterministic_given_seed(self):
        a = gbm_fx_paths(1.10, 1.0, 0.03, 0.01, 0.10, 20, 50, rng=42)
        b = gbm_fx_paths(1.10, 1.0, 0.03, 0.01, 0.10, 20, 50, rng=42)
        assert np.array_equal(a, b)

    def test_martingale_property_under_domestic_measure(self):
        # e^{-r_d T} E[S_T] = S e^{-r_f T} within 3 SE.
        S, T, rd, rf, sig = 1.10, 1.0, 0.04, 0.01, 0.10
        paths = gbm_fx_paths(S, T, rd, rf, sig, n_steps=10, n_paths=200_000,
                             rng=1)
        disc = math.exp(-rd * T) * paths[:, -1]
        se = disc.std(ddof=1) / math.sqrt(len(disc))
        assert abs(disc.mean() - S * math.exp(-rf * T)) < 3 * se

    def test_zero_vol_deterministic_forward_path(self):
        paths = gbm_fx_paths(1.10, 1.0, 0.04, 0.01, 0.0, 4, 3, rng=0)
        assert paths[0, -1] == pytest.approx(1.10 * math.exp(0.03), abs=1e-12)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            gbm_fx_paths(1.1, 0.0, 0.03, 0.01, 0.1, 10, 10)
        with pytest.raises(ValueError, match="n_steps"):
            gbm_fx_paths(1.1, 1.0, 0.03, 0.01, 0.1, 0, 10)


class TestVolQuotes:
    def test_deterministic_and_shaped(self):
        a = synthetic_vol_quotes(rng=0)
        b = synthetic_vol_quotes(rng=0)
        assert a == b
        assert len(a) == 4
        for q in a:
            assert q.atm > 0
            assert q.bf25 >= 0  # butterflies non-negative by construction

    def test_skew_sign_propagates_to_rr(self):
        quotes = synthetic_vol_quotes(skew=-0.02, noise=0.0, rng=0)
        assert all(q.rr25 < 0 for q in quotes)
        assert all(abs(q.rr10) > abs(q.rr25) for q in quotes)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            synthetic_vol_quotes(base_atm=-0.1)
        with pytest.raises(ValueError):
            synthetic_vol_quotes(noise=-1.0)
        with pytest.raises(ValueError, match="tenors"):
            synthetic_vol_quotes(tenors=(0.0,))


class TestComparisonHarness:
    def test_all_models_agree(self):
        df = compare_models(**MKT, option_type="call", binomial_steps=800,
                            mc_paths=100_000, mc_seed=42)
        assert set(df.index) == {"garman_kohlhagen", "black76_on_forward",
                                 "binomial_800", "monte_carlo_100000"}
        gk = df.loc["garman_kohlhagen", "price"]
        assert df.loc["black76_on_forward", "abs_diff_vs_gk"] < 1e-10
        assert df.loc["binomial_800", "abs_diff_vs_gk"] < 5e-5
        mc_diff = df.loc["monte_carlo_100000", "abs_diff_vs_gk"]
        mc_se = df.loc["monte_carlo_100000", "std_error"]
        assert mc_diff < 3 * mc_se
        assert gk == pytest.approx(gk_price(**MKT, option_type="call"))

    def test_binomial_convergence_table_monotone_tail(self):
        df = binomial_convergence_table(**MKT, option_type="call",
                                        step_grid=(10, 100, 1000))
        assert df.loc[1000, "abs_error"] < df.loc[10, "abs_error"]

    def test_mc_convergence_table_se_shrinks(self):
        df = mc_convergence_table(**MKT, option_type="call",
                                  path_grid=(1_000, 100_000), seed=7)
        assert df.loc[100_000, "std_error"] < df.loc[1_000, "std_error"]

    def test_live_loader_importable_but_guarded(self):
        # Import must not trigger any network call; the guard is the
        # LiveDataUnavailable exception path (never exercised offline).
        from fx_options.data import live
        assert hasattr(live, "fetch_ecb_rates")
        assert issubclass(live.LiveDataUnavailable, RuntimeError)
