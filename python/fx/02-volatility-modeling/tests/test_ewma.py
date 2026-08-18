"""EWMA (RiskMetrics) recursion identities and forecasts."""

import numpy as np
import pytest

from fx_vol import ewma_forecast, ewma_variance, ewma_weights
from fx_vol.data import synthetic as syn


class TestRecursion:
    def test_recursion_identity(self):
        r = syn.simulate_constant_vol(200, 0.006, seed=31)
        lam = 0.94
        s2 = ewma_variance(r, lam=lam)
        # sigma2_t = lam sigma2_{t-1} + (1-lam) r_{t-1}^2, exactly
        lhs = s2[1:]
        rhs = lam * s2[:-1] + (1 - lam) * r[:-1] ** 2
        assert np.allclose(lhs, rhs, atol=1e-18)

    def test_closed_form_expansion(self):
        """sigma2_t = lam^t init + (1-lam) sum_{i=1..t} lam^{i-1} r_{t-i}^2."""
        r = syn.simulate_constant_vol(50, 0.006, seed=32)
        lam, init = 0.9, 4e-5
        s2 = ewma_variance(r, lam=lam, init=init)
        t = 30
        expansion = lam ** t * init + (1 - lam) * sum(
            lam ** (i - 1) * r[t - i] ** 2 for i in range(1, t + 1)
        )
        assert s2[t] == pytest.approx(expansion, rel=1e-12)

    def test_default_init_is_mean_square(self):
        r = syn.simulate_constant_vol(100, 0.006, seed=33)
        s2 = ewma_variance(r)
        assert s2[0] == pytest.approx(np.mean(r ** 2), rel=1e-12)

    def test_constant_variance_on_constant_squared_returns(self):
        r = np.array([0.01, -0.01] * 50)  # r^2 constant
        s2 = ewma_variance(r, lam=0.94, init=0.01 ** 2)
        assert np.allclose(s2, 0.01 ** 2, atol=1e-18)

    def test_validation(self):
        r = syn.simulate_constant_vol(100, 0.006, seed=34)
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            ewma_variance(r, lam=1.0)
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            ewma_variance(r, lam=0.0)
        with pytest.raises(ValueError, match="NaN"):
            ewma_variance(np.array([0.01, np.nan]))
        with pytest.raises(ValueError, match="at least 2"):
            ewma_variance(np.array([0.01]))


class TestForecast:
    def test_forecast_is_flat(self):
        """IGARCH persistence = 1: no mean reversion at any horizon."""
        r = syn.simulate_garch(2000, 1e-6, 0.05, 0.92, seed=35)
        f = ewma_forecast(r, horizon=100)
        assert f.shape == (100,)
        assert np.all(f == f[0])

    def test_forecast_matches_recursion_step(self):
        r = syn.simulate_constant_vol(500, 0.006, seed=36)
        lam = 0.94
        s2 = ewma_variance(r, lam=lam)
        f = ewma_forecast(r, horizon=1, lam=lam)
        assert f[0] == pytest.approx(lam * s2[-1] + (1 - lam) * r[-1] ** 2, rel=1e-12)

    def test_horizon_validation(self):
        r = syn.simulate_constant_vol(100, 0.006, seed=37)
        with pytest.raises(ValueError, match="horizon"):
            ewma_forecast(r, horizon=0)


class TestWeights:
    def test_weights_sum_identity(self):
        lam, n = 0.94, 200
        w = ewma_weights(lam, n)
        assert w.sum() == pytest.approx(1 - lam ** n, rel=1e-12)

    def test_weights_geometric_decay(self):
        w = ewma_weights(0.9, 10)
        assert np.allclose(w[1:] / w[:-1], 0.9, atol=1e-12)
        with pytest.raises(ValueError):
            ewma_weights(1.2, 10)
