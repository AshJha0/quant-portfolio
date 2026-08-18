"""FX return construction, inversion invariance, triangulation and cross vol."""

import numpy as np
import pandas as pd
import pytest

from fx_vol import (
    close_to_close_vol,
    cross_pair_signs,
    cross_volatility,
    fit_garch,
    invert_prices,
    invert_returns,
    log_returns,
    pair_currencies,
    triangulate_prices,
    triangulate_returns,
)
from fx_vol.data import synthetic as syn


class TestLogReturns:
    def test_matches_manual_computation(self):
        p = np.array([1.10, 1.12, 1.08, 1.15])
        r = log_returns(p)
        assert np.allclose(r, np.diff(np.log(p)), atol=1e-15)

    def test_series_preserves_index(self):
        idx = pd.bdate_range("2024-01-01", periods=4)
        p = pd.Series([1.1, 1.2, 1.15, 1.18], index=idx)
        r = log_returns(p)
        assert isinstance(r, pd.Series)
        assert (r.index == idx[1:]).all()

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            log_returns([1.1, np.nan, 1.2])

    def test_rejects_nonpositive_prices(self):
        with pytest.raises(ValueError, match="positive"):
            log_returns([1.1, -0.5, 1.2])

    def test_rejects_too_short(self):
        with pytest.raises(ValueError, match="at least 2"):
            log_returns([1.1])


class TestInversionInvariance:
    """Inverting BASE/QUOTE flips log-return signs and leaves vol invariant."""

    def test_inverted_prices_give_negated_returns(self):
        p = 1.10 * np.exp(np.cumsum(syn.simulate_constant_vol(500, 0.006, seed=1)))
        r = log_returns(p)
        r_inv = log_returns(invert_prices(p))
        assert np.allclose(r_inv, -r, atol=1e-12)

    def test_invert_returns_is_sign_flip(self):
        r = syn.simulate_constant_vol(100, 0.006, seed=2)
        assert np.array_equal(invert_returns(r), -r)

    def test_vol_invariant_under_inversion(self):
        p = 150.0 * np.exp(np.cumsum(syn.simulate_constant_vol(2000, 0.007, seed=3)))
        v = close_to_close_vol(log_returns(p))
        v_inv = close_to_close_vol(log_returns(invert_prices(p)))
        assert v == pytest.approx(v_inv, abs=1e-14)

    def test_garch_fit_invariant_under_inversion(self, garch_sim):
        """Gaussian GARCH likelihood depends on r^2 only -> identical fit."""
        r = garch_sim[0][:3000]
        f1 = fit_garch(r)
        f2 = fit_garch(invert_returns(r))
        for k in ("omega", "alpha", "beta"):
            assert f1.params[k] == pytest.approx(f2.params[k], rel=1e-8)
        assert f1.loglik == pytest.approx(f2.loglik, abs=1e-6)


class TestPairAlgebra:
    def test_pair_currencies(self):
        assert pair_currencies("eurusd") == ("EUR", "USD")

    @pytest.mark.parametrize("bad", ["EUR", "EURUSDX", "EU1USD", "EUREUR", 123])
    def test_pair_currencies_rejects(self, bad):
        with pytest.raises(ValueError):
            pair_currencies(bad)

    @pytest.mark.parametrize(
        "p1,p2,cross,expected",
        [
            ("EURUSD", "USDJPY", "EURJPY", (1, 1)),
            ("EURUSD", "JPYUSD", "EURJPY", (1, -1)),
            ("USDCHF", "USDJPY", "CHFJPY", (-1, 1)),
            ("USDEUR", "USDJPY", "EURJPY", (-1, 1)),
            ("EURUSD", "GBPUSD", "EURGBP", (1, -1)),
        ],
    )
    def test_cross_pair_signs(self, p1, p2, cross, expected):
        assert cross_pair_signs(p1, p2, cross) == expected

    def test_cross_pair_signs_impossible(self):
        with pytest.raises(ValueError, match="cannot triangulate"):
            cross_pair_signs("EURUSD", "GBPJPY", "EURJPY")

    def test_triangulated_prices_identity(self):
        sim = syn.simulate_correlated_pairs(300, 0.006, 0.007, 0.3, seed=4)
        cross = triangulate_prices(sim["prices1"], "EURUSD", sim["prices2"], "USDJPY", "EURJPY")
        assert np.allclose(cross, sim["prices1"] * sim["prices2"], rtol=1e-12)

    def test_triangulated_returns_match_price_route(self):
        sim = syn.simulate_correlated_pairs(300, 0.006, 0.007, -0.4, seed=5)
        r_direct = triangulate_returns(sim["returns1"], "EURUSD", sim["returns2"], "USDJPY", "EURJPY")
        cross_p = triangulate_prices(sim["prices1"], "EURUSD", sim["prices2"], "USDJPY", "EURJPY")
        assert np.allclose(r_direct, log_returns(cross_p), atol=1e-12)


class TestCrossVolatility:
    """sigma_x^2 = s1^2 + s2^2 + 2 c1 c2 rho s1 s2 recovered on simulations."""

    @pytest.mark.parametrize("rho", [0.5, -0.5, 0.0, 0.9])
    def test_triangle_recovery_positive_sign(self, rho):
        # EURJPY = EURUSD * USDJPY  (sign product +1)
        sim = syn.simulate_correlated_pairs(40_000, 0.006, 0.008, rho, seed=int(10 * (rho + 2)))
        r_cross = triangulate_returns(sim["returns1"], "EURUSD", sim["returns2"], "USDJPY", "EURJPY")
        realized = r_cross.std(ddof=1)
        s1, s2 = sim["returns1"].std(ddof=1), sim["returns2"].std(ddof=1)
        rho_hat = np.corrcoef(sim["returns1"], sim["returns2"])[0, 1]
        predicted = cross_volatility(s1, s2, rho_hat, "EURUSD", "USDJPY", "EURJPY")
        assert predicted == pytest.approx(realized, rel=1e-10)  # exact identity in-sample
        # and against the *true* inputs, to sampling error
        true_cross = cross_volatility(0.006, 0.008, rho, sign_product=1)
        assert realized == pytest.approx(true_cross, rel=0.03)

    def test_triangle_recovery_negative_sign(self):
        # EURGBP = EURUSD / GBPUSD  (sign product -1); rho(EURUSD, GBPUSD) > 0 in reality
        rho = 0.6
        sim = syn.simulate_correlated_pairs(40_000, 0.006, 0.0055, rho, seed=6)
        r_cross = triangulate_returns(sim["returns1"], "EURUSD", sim["returns2"], "GBPUSD", "EURGBP")
        realized = r_cross.std(ddof=1)
        true_cross = cross_volatility(0.006, 0.0055, rho, sign_product=-1)
        assert realized == pytest.approx(true_cross, rel=0.03)
        # the negative sign product *reduces* cross vol for positively correlated legs
        assert true_cross < np.hypot(0.006, 0.0055)

    def test_perfect_negative_correlation_cancels(self):
        v = cross_volatility(0.01, 0.01, -1.0, sign_product=1)
        assert v == pytest.approx(0.0, abs=1e-12)

    def test_input_validation(self):
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            cross_volatility(0.01, 0.01, 1.5, sign_product=1)
        with pytest.raises(ValueError, match="non-negative"):
            cross_volatility(-0.01, 0.01, 0.0, sign_product=1)
        with pytest.raises(ValueError, match="sign_product"):
            cross_volatility(0.01, 0.01, 0.0, sign_product=2)
        with pytest.raises(ValueError, match="supply either"):
            cross_volatility(0.01, 0.01, 0.0)
